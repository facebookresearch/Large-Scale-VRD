# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Detectron model construction functions.

Detectron supports a large number of model types. The configuration space is
large. To get a sense, a given model is in element in the cartesian product of:

  - backbone (e.g., VGG16, ResNet, ResNeXt)
  - FPN (on or off)
  - RPN only (just proposals)
  - Fixed proposals for Fast R-CNN, RFCN, Mask R-CNN (with or without keypoints)
  - End-to-end model with RPN + Fast R-CNN (i.e., Faster R-CNN), Mask R-CNN, ...
  - Different "head" choices for the model
  - ... many configuration options ...

A given model is made by combining many basic components. The result is flexible
though somewhat complex to understand at first.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import copy
import importlib
import logging
import numpy as np
import math

from caffe2.python import core
from caffe2.python import workspace

from modeling import (
    VGG16_rel_softmaxed_triplet
)

from core.config_rel import cfg
from core.get_gt_perturbed_proposals import get_gt_perturbed_proposals
from modeling.detector_rel import DetectionModelHelper
from roi_data.loader_rel import RoIDataLoader
from datasets.roidb_rel import combined_roidb_for_training
from datasets.roidb_rel import combined_roidb_for_val_test
from datasets.factory import get_landb
import modeling.optimizer_rel as optim
import roi_data.minibatch_rel
import utils.c2 as c2_utils
import utils.net_rel as nu
from utils import checkpoints_rel

logger = logging.getLogger(__name__)


model_creator_map = {
    'VGG16_rel_softmaxed_triplet': VGG16_rel_softmaxed_triplet}

# ---------------------------------------------------------------------------- #
# Helper functions for building various re-usable network bits
# ---------------------------------------------------------------------------- #

def create(model_type_func, train=True, split='train', gpu_id=0):
    """Generic model creation function that dispatches to specific model
    building functions.

    By default, this function will generate a data parallel model configured to
    run on cfg.NUM_GPUS devices. However, you can restrict it to build a model
    targeted to a specific GPU by specifying gpu_id. This is used by
    optimizer.build_data_parallel_model() during test time.
    """
    model = DetectionModelHelper(
        name=model_type_func + '_' + split,
        train=train,
        split=split,
        init_params=True
        # init_params=train
    )
    model.only_build_forward_pass = False
    model.target_gpu_id = gpu_id
    return build_model(model, split)


def build_model(model, split):
    def _single_gpu_build_func(model):
        return model_creator_map[cfg.MODEL.MODEL_NAME].create_model(model=model)

    if split == 'train':
        roidb = combined_roidb_for_training(cfg.DATASET + '_train', None)
        proposals = get_gt_perturbed_proposals(roidb)
        logger.info('Training proposals length: {}'.format(len(proposals)))
    elif split == 'val':
        roidb = combined_roidb_for_val_test(cfg.DATASET + '_val')
        proposals = get_gt_val_test_proposals('val', roidb)
        logger.info('Validation proposals length: {}'.format(len(proposals)))
    else:
        roidb = combined_roidb_for_val_test(cfg.DATASET + '_' + cfg.TEST.DATA_TYPE)
        proposals = get_gt_val_test_proposals(cfg.TEST.DATA_TYPE, roidb)
    logger.info('{:d} roidb entries'.format(len(roidb)))

    landb = get_landb(cfg.DATASET + '_lan')

    add_inputs(model, roidb=roidb, landb=landb, proposals=proposals, split=split)

    feed_all_word_vecs(model)

    optim.build_data_parallel_model(model, _single_gpu_build_func)
    workspace.RunNetOnce(model.param_init_net)

    odir, cdir = get_dirs(model, split)

    if split != 'test':
        setup_model(model, cfg.TRAIN.PARAMS_FILE, split)
    else:
        setup_model(model, None, split)

    return model, odir, cdir


# ---------------------------------------------------------------------------- #
# Network inputs
# ---------------------------------------------------------------------------- #

def get_dirs(model, split):

    output_dir = os.path.join(
        cfg.OUTPUT_DIR, cfg.DATASET,
        cfg.MODEL.TYPE, cfg.MODEL.SUBTYPE, cfg.MODEL.SPECS)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checkpoint_dir = os.path.join(output_dir, split)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    logger.info('Outputs saved to: {:s}'.format(os.path.abspath(output_dir)))
    dump_proto_files(model, output_dir)

    return output_dir, checkpoint_dir


def add_inputs(model, roidb=None, landb=None, proposals=None, split='train'):
    """Create network input ops and blobs used for training. To be called
    *after* model_builder.create().
    """
    # Implementation notes:
    #   Typically, one would create the input ops and then the rest of the net.
    #   However, creating the input ops depends on loading the dataset, which
    #   can take a few minutes for COCO.
    #   We prefer to avoid waiting so debugging can fail fast.
    #   Thus, we create the net *without input ops* prior to loading the
    #   dataset, and then add the input ops after loading the dataset.
    #   Since we defer input op creation, we need to do a little bit of surgery
    #   to place the input ops at the start of the network op list.

    if roidb is not None:
        # To make debugging easier you can set cfg.DATA_LOADER.NUM_THREADS = 1
        model.roi_data_loader = RoIDataLoader(
            split=split, roidb=roidb, landb=landb, proposals=proposals,
            num_loaders=cfg.DATA_LOADER.NUM_THREADS
        )
    orig_num_op = len(model.net._net.op)
    blob_names = roi_data.minibatch_rel.get_minibatch_blob_names(split)
    for gpu_id in range(cfg.NUM_DEVICES):
        with c2_utils.NamedCudaScope(gpu_id):
            for blob_name in blob_names:
                workspace.CreateBlob(core.ScopedName(blob_name))
            model.net.DequeueBlobs(
                model.roi_data_loader._blobs_queue_name, blob_names
            )
            workspace.CreateBlob(core.ScopedName('all_obj_word_vecs'))
            workspace.CreateBlob(core.ScopedName('all_prd_word_vecs'))
    # A little op surgery to move input ops to the start of the net
    diff = len(model.net._net.op) - orig_num_op
    new_op = model.net._net.op[-diff:] + model.net._net.op[:-diff]
    del model.net._net.op[:]
    model.net._net.op.extend(new_op)


def setup_model(model, weights_file, split):
    """Loaded saved weights and create the network in the C2 workspace."""

    if weights_file:
        # Override random weight initialization with weights from a saved model
        nu.initialize_gpu_from_weights_file(model, weights_file, gpu_id=0)
    # Even if we're randomly initializing we still need to synchronize
    # parameters across GPUs
    nu.broadcast_parameters(model)
    workspace.CreateNet(model.net)

    # Start loading mini-batches and enqueuing blobs
    model.roi_data_loader.register_sigint_handler()
    model.roi_data_loader.start(prefill=True)


def dump_proto_files(model, output_dir):
    """Save prototxt descriptions of the training network and parameter
    initialization network."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'net.pbtxt'), 'w') as fid:
        fid.write(str(model.net.Proto()))
    with open(os.path.join(output_dir, 'param_init_net.pbtxt'), 'w') as fid:
        fid.write(str(model.param_init_net.Proto()))


def feed_all_word_vecs(model):
    landb = model.roi_data_loader._landb
    inputs = {}
    all_obj_word_vecs = landb.obj_vecs
    all_prd_word_vecs = landb.prd_vecs
    inputs['all_obj_word_vecs'] = all_obj_word_vecs
    inputs['all_prd_word_vecs'] = all_prd_word_vecs
    logger.info('feeding all_word_vecs...')
    for gpu_id in range(cfg.ROOT_DEVICE_ID, cfg.ROOT_DEVICE_ID + cfg.NUM_DEVICES):
        logger.info('feeding on GPU {}'.format(gpu_id))
        with c2_utils.NamedCudaScope(gpu_id):
            for k, v in inputs.items():
                workspace.FeedBlob(
                    core.ScopedName(k), v.astype(np.float32, copy=True))


def get_lr_steps():
    # example input: [150150, 150150, 150150]
    split_lr_stepsizes = cfg.SOLVER.STEP_SIZES
    lr_steps = [int(step) for step in split_lr_stepsizes]
    lr_iters = []
    for idx in range(len(lr_steps)):
        if idx > 0:
            lr_iters.append(lr_steps[idx] + lr_iters[idx - 1])
        else:
            lr_iters.append(lr_steps[idx])
    # now we have [150150, 300300, 450450]
    return lr_iters


def check_and_apply_warmup(curr_iter, learning_rate):
    """
    This function takes care of applying warm-up to the training. The warmup is
    of three types:
    1. Constant warm-up LR for few iterations and jump to Base LR afterwards
    2. Constant warm-up LR for few iterations and gradually increase by a constant
       pre-specified step at every LR (Priya's warmup)
    3. Gradually increase LR until it reaches BASE_LR (Piotr's warmup)
    """

    # case 1 or 2
    if curr_iter <= cfg.SOLVER.WARM_UP_ITERS:
        learning_rate = cfg.SOLVER.WARM_UP_LR
    # case 2: if we want to increase the LR gradually, then based on the
    elif cfg.SOLVER.WARM_UP_ITERS > 0 and cfg.SOLVER.GRADUAL_INCREASE_LR:
        gradual_step = cfg.SOLVER.GRADUAL_LR_STEP
        num_steps = int(
            math.ceil((cfg.SOLVER.BASE_LR - cfg.SOLVER.WARM_UP_LR) / gradual_step)
        )
        if curr_iter <= (cfg.SOLVER.WARM_UP_ITERS + num_steps):
            step_lr = (
                cfg.SOLVER.WARM_UP_LR +
                (gradual_step * (curr_iter - cfg.SOLVER.WARM_UP_ITERS)))
            learning_rate = min(cfg.SOLVER.BASE_LR, step_lr)
    # case 3: gradually increase LR
    elif cfg.SOLVER.GRADUAL_INCREASE_LR:
        # in this case, the LR will be gradually increased from the warm-up LR
        # taking gradual_step every time
        gradual_step = cfg.SOLVER.GRADUAL_LR_STEP
        num_steps = int(math.ceil((
            cfg.SOLVER.BASE_LR - cfg.SOLVER.GRADUAL_LR_START) / gradual_step)
        )
        if curr_iter <= num_steps:
            step_lr = cfg.SOLVER.GRADUAL_LR_START + (gradual_step * (curr_iter - 1))
            learning_rate = min(cfg.SOLVER.BASE_LR, step_lr)
    return learning_rate


def add_variable_stepsize_lr(
    curr_iter, num_devices, lr_iters, start_model_iter, model=None,
    prev_checkpointed_lr=None,
):
    global CURRENT_LR
    # if the model is resumed from some checkpoint state, then we load the
    # checkpoint LR into the CURRENT_LR at the start of training only
    if prev_checkpointed_lr is not None and (curr_iter == start_model_iter):
        CURRENT_LR = prev_checkpointed_lr
    if curr_iter <= lr_iters[0]:
        gamma_pow = 0
    else:
        idx = 0
        while idx < len(lr_iters) and lr_iters[idx] < curr_iter:
            idx += 1
        gamma_pow = idx

    learning_rate = (cfg.SOLVER.BASE_LR * math.pow(cfg.SOLVER.GAMMA, gamma_pow))
    learning_rate = check_and_apply_warmup(curr_iter, learning_rate)
    root_device_id = cfg.ROOT_DEVICE_ID
    new_lr = learning_rate
    if curr_iter == 1:
        prev_lr = new_lr
    else:
        prev_lr = CURRENT_LR
    if cfg.SOLVER.SCALE_MOMENTUM and (not new_lr == prev_lr):
        scale = new_lr / float(prev_lr)
        scale_momentum(scale, model)

    CURRENT_LR = new_lr
    for idx in range(root_device_id, root_device_id + num_devices):
        with c2_utils.CudaScope(idx):
            workspace.FeedBlob(
                'gpu_{}/lr'.format(idx),
                np.array(learning_rate, dtype=np.float32)
            )
            workspace.FeedBlob(
                'gpu_{}/lr_x'.format(idx),
                np.array(learning_rate * cfg.SOLVER.LR_FACTOR, dtype=np.float32)
            )

    return CURRENT_LR


def scale_momentum(scale, model):
    # for the LR warm-up in distributed training, when we change the LR after
    # warm-up, then we need to update the momentum accordingly
    logger.info('Scaling momentum: {}'.format(scale))
    root_device_id = cfg.ROOT_DEVICE_ID
    num_devices = cfg.NUM_DEVICES
    for idx in range(root_device_id, root_device_id + num_devices):
        with c2_utils.CudaScope(idx):
            params = model.GetParams()
            for param in params:
                op = core.CreateOperator(
                    'Scale', [param + '_momentum'], [param + '_momentum'],
                    scale=scale)
                workspace.RunOperatorOnce(op)


def add_inference_inputs(model):
    """Create network input blobs used for inference."""

    def create_input_blobs_for_net(net_def):
        for op in net_def.op:
            for blob_in in op.input:
                if not workspace.HasBlob(blob_in):
                    workspace.CreateBlob(blob_in)

    create_input_blobs_for_net(model.net.Proto())
    if cfg.MODEL.MASK_ON:
        create_input_blobs_for_net(model.mask_net.Proto())
    if cfg.MODEL.KEYPOINTS_ON:
        create_input_blobs_for_net(model.keypoint_net.Proto())
