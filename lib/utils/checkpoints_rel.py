from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import numpy as np
import os
import logging
from collections import OrderedDict
import cPickle as pickle

from utils import helpers_rel
from core.config_rel import cfg
from caffe2.python import workspace, core
from caffe2.proto import caffe2_pb2

logger = logging.getLogger(__name__)


# This function looks at all the iters checkpointed and returns latest iter file
def get_checkpoint_resume_file(checkpoint_dir):
    all_files = os.listdir(checkpoint_dir)
    all_iters = []
    for f in all_files:
        if 'c2_model_iter' in f:
            iter_num = int(f.replace('.pkl', '').replace('c2_model_iter', ''))
            all_iters.append(iter_num)
    if len(all_iters) > 0:
        all_iters.sort(reverse=True)
        last_iter = int(all_iters[0])
        filepath = os.path.join(
            checkpoint_dir, 'c2_model_iter{}.pkl'.format(last_iter)
        )
        return filepath
    else:
        return None


def load_model_from_params_file(model, params_file, checkpoint_dir=None):
    # in case of cluster failure, we should resume from the last checkpoint rather
    # than the params_file if specified
    checkpoint_exists = False
    if checkpoint_dir is not None:
        checkpointed_files = os.listdir(checkpoint_dir)
        for f in checkpointed_files:
            # if f.endswith('.pkl'):
            if f.find('c2_model_iter') >= 0:
                checkpoint_exists = True
                break
    prev_lr = None
    # import pdb
    # pdb.set_trace()
    if params_file and os.path.exists(params_file) and not checkpoint_exists:
        logger.info('Initializing model parameters from {}'.format(params_file))
        start_model_iter, prev_lr = initialize_params_from_file(
            model=model, weights_file=params_file, num_devices=cfg.NUM_DEVICES,
        )
        # import pdb
        # pdb.set_trace()
    elif cfg.CHECKPOINT.RESUME and cfg.CHECKPOINT.CHECKPOINT_MODEL:
        start_model_iter = 0
        params_file = get_checkpoint_resume_file(checkpoint_dir)
        if params_file is not None and os.path.exists(params_file):
            start_model_iter, prev_lr = initialize_params_from_file(
                model=model, weights_file=params_file, num_devices=cfg.NUM_DEVICES,
            )
        else:
            logger.info('Params file does not exist: {}'.format(params_file))
    # import pdb
    # pdb.set_trace()
    return start_model_iter, prev_lr, checkpoint_exists


def get_checkpoint_directory():
    # if cfg.CHECKPOINT.DIR:
    #     odir = os.path.abspath(os.path.join(
    #         cfg.CHECKPOINT.DIR, cfg.DATASET, cfg.MODEL.MODEL_NAME
    #     ))
    # else:
    #     odir = os.path.join(
    #         helpers_rel.get_output_directory(), cfg.DATASET, cfg.MODEL.MODEL_NAME
    #     )
    if cfg.CHECKPOINT.DIR:
        odir = os.path.abspath(os.path.join(
            cfg.CHECKPOINT.DIR, cfg.DATASET,
            cfg.MODEL.TYPE, cfg.MODEL.SUBTYPE, cfg.MODEL.SPECS
        ))
    else:
        odir = os.path.join(
            helpers_rel.get_output_directory(), cfg.DATASET,
            cfg.MODEL.TYPE, cfg.MODEL.SUBTYPE, cfg.MODEL.SPECS
        )
    if cfg.DISTRIBUTED.DISTR_ON:
        odir = os.path.join(odir, cfg.DISTRIBUTED.RUN_ID)
    if not os.path.exists(odir):
        os.makedirs(odir)
    return odir


def initialize_master_device_model_params(model, weights_file):
    ws_blobs = workspace.Blobs()
    logger.info("Initializing model params from file: {}".format(weights_file))
    with open(weights_file, 'r') as fopen:
        blobs = pickle.load(fopen)
    if 'blobs' in blobs:
        blobs = blobs['blobs']
    unscoped_blob_names = OrderedDict()

    # Return the model iter from which training should start
    model_iter = 0
    if 'model_iter' in blobs:
        model_iter = blobs['model_iter']
    prev_lr = None
    if 'lr' in blobs:
        prev_lr = round(blobs['lr'], 6)

    # initialize params, params momentum, computed params
    if 'test' not in model.net.Name():
        for param in model.params:
            # By Ji on 10/21/2017
            # Layers that are frozen during finetuning have no momentums
            scoped_blob_name = str(param) + '_momentum'
            if workspace.HasBlob(scoped_blob_name):
                unscoped_blob_names[helpers_rel.unscope_name(scoped_blob_name)] = True
    # NOTE: Currently GetAllParams() and GetAllParams('') both work. Use neat version
    for blob in model.GetAllParams():
        unscoped_blob_names[helpers_rel.unscope_name(str(blob))] = True

    root_device_id = cfg.ROOT_DEVICE_ID
    device = caffe2_pb2.CUDA if cfg.DEVICE == 'GPU' else caffe2_pb2.CPU
    with core.NameScope('gpu_{}'.format(root_device_id)):
        with core.DeviceScope(core.DeviceOption(device, root_device_id)):
            for unscoped_blob_name in unscoped_blob_names.keys():
                scoped_blob_name = helpers_rel.scoped_name(unscoped_blob_name)
                if unscoped_blob_name not in blobs:
                    logger.info('{:s} not found'.format(unscoped_blob_name))
                    continue
                # By Ji on 12/02/2017: print only when training
                if model.train:
                    logger.info('{:s} loaded from weights file into: {:s}'.format(
                        unscoped_blob_name, scoped_blob_name))
                if scoped_blob_name in ws_blobs:
                    ws_blob = workspace.FetchBlob(scoped_blob_name)
                    assert ws_blob.shape == blobs[unscoped_blob_name].shape, \
                        ('Workspace blob {} with shape {} does not match '
                         'weights file shape {}').format(
                            unscoped_blob_name, ws_blob.shape,
                            blobs[unscoped_blob_name].shape)
                data = blobs[unscoped_blob_name].astype(np.float32, copy=False)
                workspace.FeedBlob(scoped_blob_name, data)
    return model_iter, prev_lr


def broadcast_parameters(model, num_devices):
    if num_devices == 1:
        return
    root_device_id = cfg.ROOT_DEVICE_ID
    device = caffe2_pb2.CUDA if cfg.DEVICE == 'GPU' else caffe2_pb2.CPU
    prefix = 'gpu_' if cfg.DEVICE == 'GPU' else 'cpu_'
    all_model_params = model.GetAllParams(prefix + str(root_device_id))
    all_params_momentum = []
    if 'test' not in model.net.Name():
        for param in model.GetParams(prefix + str(root_device_id)):
            scoped_blob_name = str(param) + '_momentum'
            if workspace.HasBlob(scoped_blob_name):
                all_params_momentum.append(scoped_blob_name)
    all_params = all_model_params + all_params_momentum
    for param in all_params:
        data = workspace.FetchBlob(str(param))
        unscoped_param_name = helpers_rel.unscope_name(str(param))
        # By Ji on 12/02/2017: print only when training
        if model.train:
            logger.info('Broadcasting {} to'.format(str(param)))
        for idx in range(root_device_id + 1, root_device_id + num_devices):
            with core.NameScope(prefix + str(idx)):
                with core.DeviceScope(core.DeviceOption(device, idx)):
                    device_scoped_name = helpers_rel.scoped_name(unscoped_param_name)
                    # By Ji on 12/02/2017: print only when training
                    if model.train:
                        logger.info(' |-> {}'.format(device_scoped_name))
                    workspace.FeedBlob(device_scoped_name, data)


# initialize the model from a file and broadcast the parameters to all_gpus
# if num_devices > 1
def initialize_params_from_file(model, weights_file, num_devices):
    logger.info('Initializing model params from {}'.format(weights_file))
    model_iter, prev_lr = initialize_master_device_model_params(
        model, weights_file
    )
    broadcast_parameters(model, num_devices)
    return model_iter, prev_lr


def save_model_params(model, params_file, checkpoint_dir, model_iter):
    logger.info("Saving model params to weights file {}".format(params_file))
    root_device_id = cfg.ROOT_DEVICE_ID
    prefix = 'gpu_' if cfg.DEVICE == 'GPU' else 'cpu_'
    save_params = [str(param) for param in model.GetParams(
        prefix + str(root_device_id))]
    save_computed_params = [str(param) for param in model.GetComputedParams(
        prefix + str(root_device_id))]
    save_blobs = {}
    # also save total model iterations so far
    save_blobs['model_iter'] = model_iter + 1
    save_blobs['lr'] = workspace.FetchBlob(prefix + '{}/lr'.format(root_device_id))
    # save param momentum as well
    for param in save_params:
        scoped_blob_name = str(param) + '_momentum'
        # By Ji on 10/21/2017
        # Layers that are frozen during finetuning have no momentums
        if workspace.HasBlob(scoped_blob_name):
            unscoped_blob_name = helpers_rel.unscope_name(scoped_blob_name)
            if unscoped_blob_name not in save_blobs:
                data = workspace.FetchBlob(scoped_blob_name)
                save_blobs[unscoped_blob_name] = data

    for param in save_params + save_computed_params:
        scoped_blob_name = str(param)
        unscoped_blob_name = helpers_rel.unscope_name(scoped_blob_name)
        if unscoped_blob_name not in save_blobs:
            data = workspace.FetchBlob(
                scoped_blob_name
            )
            save_blobs[unscoped_blob_name] = data
            # logger.info(
            #     '{:s} -> {:s}'.format(scoped_blob_name, unscoped_blob_name)
            # )
    with open(params_file, 'w') as fwrite:
        pickle.dump(dict(blobs=save_blobs), fwrite, pickle.HIGHEST_PROTOCOL)
