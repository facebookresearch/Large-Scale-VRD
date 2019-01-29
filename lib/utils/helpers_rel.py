# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import logging
import json
import math
import numpy as np
import subprocess
from caffe2.python import workspace, scope
from core.config_rel import cfg
from caffe2.python.predictor import mobile_exporter
# from everstore import Everstore
# @lint-avoid-servicerouter-import-warning
# from ServiceRouter import ServiceRouter

logger = logging.getLogger(__name__)

EVERPASTE_PROFILE = 'https://our.intern.facebook.com/intern/fblearner/\
tensorboardgraph/everpaste/'


def sum_multi_gpu_blob(blob_name):
    """Return the sum of a scalar blob held on multiple GPUs."""
    prefix = 'gpu_' if cfg.DEVICE == 'GPU' else 'cpu_'
    val = 0
    for gpu_id in range(cfg.ROOT_DEVICE_ID, cfg.ROOT_DEVICE_ID + cfg.NUM_DEVICES):
        val += float(workspace.FetchBlob(prefix + '{}/{}'.format(gpu_id, blob_name)))
    return val


def average_multi_gpu_blob(blob_name):
    """Return the average of a scalar blob held on multiple GPUs."""
    return sum_multi_gpu_blob(blob_name) / cfg.NUM_DEVICES


def check_nan_losses(model, num_devices):
    # if any of the losses is NaN, raise exception
    iter_values = {}
    for k in model.losses:
        key = str(k)
        if key.find('/') >= 0:
            key = key.split('/')[1]
        # logger.info('loss name: {}'.format(key))
        iter_values[key] = sum_multi_gpu_blob(key)
    loss = np.sum(np.array(iter_values.values()))
    if math.isnan(loss):
        logger.error("ERROR: NaN losses detected")
        logger.info(iter_values)
        os._exit(0)


def save_json_stats(json_stats, node_id):
    if cfg.CHECKPOINT.DIR:
        odir = os.path.abspath(os.path.join(
            cfg.CHECKPOINT.DIR, cfg.DATASET, cfg.MODEL.MODEL_NAME,
            'depth_' + str(cfg.MODEL.DEPTH), cfg.DISTRIBUTED.RUN_ID
        ))
    else:
        odir = os.path.join(
            get_output_directory(), cfg.DATASET, cfg.MODEL.MODEL_NAME,
            'depth_' + str(cfg.MODEL.DEPTH), cfg.DISTRIBUTED.RUN_ID
        )
    if not os.path.exists(odir):
        os.makedirs(odir)
    if cfg.TRAIN.PARAMS_FILE:
        filename = os.path.join(
            odir, 'resume_node_{}_json_stats.log'.format(node_id)
        )
    else:
        filename = os.path.join(odir, 'node_{}_json_stats.log'.format(node_id))
    logger.info('WRITING JSON STATS: {}'.format(filename))
    with open(filename, 'a') as fopen:
        fopen.write(json.dumps(json_stats, sort_keys=False))
        fopen.write("\n")


def save_layer_json_stats(layer_stats, node_id):
    if cfg.CHECKPOINT.DIR:
        odir = os.path.abspath(os.path.join(
            cfg.CHECKPOINT.DIR, cfg.DATASET, cfg.MODEL.MODEL_NAME,
            'depth_' + str(cfg.MODEL.DEPTH), cfg.DISTRIBUTED.RUN_ID
        ))
    else:
        odir = os.path.join(
            get_output_directory(), cfg.DATASET, cfg.MODEL.MODEL_NAME,
            'depth_' + str(cfg.MODEL.DEPTH), cfg.DISTRIBUTED.RUN_ID
        )
    if not os.path.exists(odir):
        os.makedirs(odir)
    if cfg.TRAIN.PARAMS_FILE:
        filename = os.path.join(
            odir, 'resume_node_{}_layer_stats.log'.format(node_id)
        )
    else:
        filename = os.path.join(odir, 'node_{}_layer_stats.log'.format(node_id))
    logger.info('WRITING LAYER STATS: {}'.format(filename))
    with open(filename, 'w') as fid:
        json.dump(layer_stats, fid)


def check_inp_arguments():
    # this function should check for the validity of arguments
    assert cfg.TRAIN.IMS_PER_BATCH > 0, \
        "Batch size should be larger than 0"
    if cfg.TEST.TEN_CROP:
        assert cfg.TEST.IMS_PER_BATCH % 10 == 0, \
            'test minibatch size is not multiple of (num_devices * 10)'
    else:
        assert cfg.TEST.IMS_PER_BATCH > 0, \
            'test minibatch size is not larger than 0'


def print_options():
    import pprint
    logger.info('Training with config:')
    logger.info(pprint.pformat(cfg))


def get_model_proto_directory():
    odir = os.path.abspath(os.path.join(
        cfg.CHECKPOINT.DIR, cfg.DATASET, cfg.MODEL.MODEL_NAME))
    if not os.path.exists(odir):
        os.makedirs(odir)
    return odir


def get_num_test_iter(db):
    if cfg.TEST.TEN_CROP:
        test_epoch_iter = int(math.ceil(
            db.get_db_size() /
            ((float(cfg.TEST.IMS_PER_BATCH) / 10) / float(cfg.MODEL.GRAD_ACCUM_FREQUENCY))
        ))
    else:
        test_epoch_iter = int(math.ceil(
            db.get_db_size() /
            (float(cfg.TEST.IMS_PER_BATCH) / float(cfg.MODEL.GRAD_ACCUM_FREQUENCY))
        ))
        # import pdb
        # pdb.set_trace()
    return test_epoch_iter


def get_batch_size(split):
    if split in ['test', 'val']:
        if cfg.TEST.TEN_CROP:
            batch_size = int(
                cfg.TEST.IMS_PER_BATCH / 10 /
                cfg.MODEL.GRAD_ACCUM_FREQUENCY
            )
        else:
            batch_size = int(
                cfg.TEST.IMS_PER_BATCH /
                cfg.MODEL.GRAD_ACCUM_FREQUENCY
            )
        return batch_size
    elif split == 'train':
        batch_size = int(
            cfg.TRAIN.IMS_PER_BATCH / cfg.MODEL.GRAD_ACCUM_FREQUENCY
        )
        return batch_size


def get_output_directory():
    root_folder = ""
    # either user specifies an output directory or we create a directory
    if cfg.OUTPUT_DIR:
        root_folder = cfg.OUTPUT_DIR
    else:
        root_folder = '/mnt/vol/gfsai-east/ai-group/users/zhangjixyz/tmp_reldn'
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    return root_folder


def set_random_seed(seed):
    logger.info("Setting random seed")
    np.random.seed(seed)
    return None


def log_json_stats(stats):
    print('\njson_stats: {:s}\n'.format(json.dumps(stats, sort_keys=False)))
    return None


def save_mobile_exporter_net(model_workspace, model):
    init_net, predict_net = mobile_exporter.Export(
        model_workspace, model.net, model.GetAllParams()
    )
    proto_path = get_model_proto_directory()
    net_name = model.net.Proto().name
    init_net_path = os.path.join(proto_path, net_name + "_mobile_init.pbtxt")
    predict_net_path = os.path.join(proto_path, net_name + "_mobile_predict.pbtxt")
    with open(init_net_path, "wb") as fopen:
        fopen.write(init_net.SerializeToString())
    with open(predict_net_path, "wb") as fopen:
        fopen.write(predict_net.SerializeToString())
    logger.info("{}: Mobile init_net proto saved to: {}".format(
        net_name, init_net_path))
    logger.info("{}: Mobile predict_net proto saved to: {}".format(
        net_name, predict_net_path))


def save_net_proto(net):
    net_proto = str(net.Proto())
    net_name = net.Proto().name
    proto_path = get_model_proto_directory()
    net_proto_path = os.path.join(proto_path, net_name + ".pbtxt")
    with open(net_proto_path, 'w') as wfile:
        wfile.write(net_proto)
    logger.info("{}: Net proto saved to: {}".format(net_name, net_proto_path))
    return None


# save model protobuf for inspection later
def save_model_proto(model):
    net_proto = str(model.net.Proto())
    net_name = model.net.Proto().name
    init_net_proto = str(model.param_init_net.Proto())
    proto_path = get_model_proto_directory()
    net_proto_path = os.path.join(proto_path, net_name + "_net.pbtxt")
    init_net_proto_path = os.path.join(proto_path, net_name + "_init_net.pbtxt")
    with open(net_proto_path, 'w') as wfile:
        wfile.write(net_proto)
    with open(init_net_proto_path, 'w') as wfile:
        wfile.write(init_net_proto)
    logger.info("{}: Net proto saved to: {}".format(net_name, net_proto_path))
    logger.info("{}: Init net proto saved to: {}".format(
        net_name, init_net_proto_path))
    return None


def CHW2HWC(image):
    if len(image.shape) >= 3:
        return image.swapaxes(0, 1).swapaxes(1, 2)
    else:
        logger.error('Your image does not have correct dimensions')
        return None


def HWC2CHW(image):
    if len(image.shape) >= 3:
        return image.swapaxes(1, 2).swapaxes(0, 1)
    else:
        logger.error('Your image does not have correct dimensions')
        return None


def display_net_proto(net):
    print("Net Protobuf:\n{}".format(net.Proto()))
    return None


def unscope_name(blob_name):
    return blob_name[blob_name.rfind(scope._NAMESCOPE_SEPARATOR) + 1:]


def scoped_name(blob_name):
    return scope.CurrentNameScope() + blob_name


# NOTE: this method was written for debug_net.py only
def layer_name(blob_name):
    return blob_name[:blob_name.rfind('_')]


def compute_layer_stats(
    layer_stats, metrics_calculator, model, curr_iter, node_id=0,
):
    prefix = 'gpu_' if cfg.DEVICE == 'GPU' else 'cpu_'
    layers_stats = metrics_calculator.get_layer_stats(model)
    layers_stats['curr_iter'] = float(curr_iter + 1)
    layers_stats['lr'] = float(workspace.FetchBlob(prefix + '0/lr'))
    layer_stats.extend([layers_stats])
    return layer_stats


def print_model_param_shape(model):
    prefix = 'gpu_' if cfg.DEVICE == 'GPU' else 'cpu_'
    all_blobs = model.GetParams(prefix + str(cfg.ROOT_DEVICE_ID))
    logger.info('All blobs in workspace:\n{}'.format(all_blobs))
    for blob_name in all_blobs:
        blob = workspace.FetchBlob(blob_name)
        logger.info("{} -> {}".format(blob_name, blob.shape))


def print_buf_net(net):
    logger.info("\n\nPrinting Model: {}\n\n".format(net.Name()))
    prefix = 'gpu_' if cfg.DEVICE == 'GPU' else 'cpu_'
    master_device = prefix + str(cfg.ROOT_DEVICE_ID)
    op_output = net.Proto().op
    for idx in range(len(op_output)):
        input_b = net.Proto().op[idx].input
        # for simplicity: only print the first output;
        output_b = str(net.Proto().op[idx].output[0])
        type_b = net.Proto().op[idx].type
        if output_b.find(master_device) >= 0:
            # Only print the forward pass network
            output_shape = np.array(workspace.FetchBlob(str(output_b))).shape
            first_blob = True
            suffix = ' ------- (op: {:s})'.format(type_b)
            for j in range(len(input_b)):
                input_shape = np.array(
                    workspace.FetchBlob(str(input_b[j]))).shape
                if input_shape != ():
                    logger.info(
                        '{:36s}: {:28s} => {:36s}: {:28s}{}'.format(
                            unscope_name(str(input_b[j])),
                            '{}'.format(input_shape),  # suppress warning
                            unscope_name(str(output_b)),
                            '{}'.format(output_shape),
                            suffix
                        ))
                    if first_blob:
                        first_blob = False
                        suffix = ' ------|'
    logger.info("End of net: {}\n\n".format(net.Name()))


def print_net(model):
    logger.info("Printing Model: {}".format(model.net.Name()))
    prefix = 'gpu_' if cfg.DEVICE == 'GPU' else 'cpu_'
    master_device = prefix + str(cfg.ROOT_DEVICE_ID)
    op_output = model.net.Proto().op
    model_params = model.GetAllParams(master_device)
    for idx in range(len(op_output)):
        input_b = model.net.Proto().op[idx].input
        # for simplicity: only print the first output;
        # not recommended if there are split layers.
        output_b = str(model.net.Proto().op[idx].output[0])
        type_b = model.net.Proto().op[idx].type
        if output_b.find(master_device) >= 0:
            # Only print the forward pass network
            if output_b.find('grad') >= 0:
                break
            output_shape = np.array(workspace.FetchBlob(str(output_b))).shape
            first_blob = True
            suffix = ' ------- (op: {:s})'.format(type_b)
            for j in range(len(input_b)):
                if input_b[j] in model_params:
                        continue
                input_shape = np.array(
                    workspace.FetchBlob(str(input_b[j]))).shape
                if input_shape != ():
                    logger.info(
                        '{:28s}: {:20s} => {:36s}: {:20s}{}'.format(
                            unscope_name(str(input_b[j])),
                            '{}'.format(input_shape),  # suppress warning
                            unscope_name(str(output_b)),
                            '{}'.format(output_shape),
                            suffix
                        ))
                    if first_blob:
                        first_blob = False
                        suffix = ' ------|'
    logger.info("End of model: {}".format(model.net.Name()))


def get_gpu_stats():
    sp = subprocess.Popen(
        ['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_list = out_str[0].split('\n')
    out_dict = {}
    for item in out_list:
        try:
            key, val = item.split(':')
            key, val = key.strip(), val.strip()
            out_dict[key] = val
        except Exception:
            pass
    used_gpu_memory = out_dict['Used GPU Memory']
    return used_gpu_memory


def get_flops_params(model, device_id):
    model_ops = model.net.Proto().op
    prefix = 'gpu_' if cfg.DEVICE == 'GPU' else 'cpu_'
    master_device = prefix + str(device_id)
    param_ops = []
    for idx in range(len(model_ops)):
        op_type = model.net.Proto().op[idx].type
        op_output = model.net.Proto().op[idx].output[0]
        # some op might have no input
        # op_input = model.net.Proto().op[idx].input[0]
        if op_type in ['Conv', 'FC'] and op_output.find(master_device) >= 0:
            param_ops.append(model.net.Proto().op[idx])

    num_flops = 0
    num_params = 0
    for idx in range(len(param_ops)):
        op = param_ops[idx]
        op_type = op.type
        op_inputs = param_ops[idx].input
        op_output = param_ops[idx].output[0]
        layer_flops = 0
        layer_params = 0
        if op_type == 'Conv':
            for op_input in op_inputs:
                if '_w' in op_input:
                    param_blob = op_input
                    param_shape = np.array(
                        workspace.FetchBlob(str(param_blob))).shape
                    layer_params = (
                        param_shape[0] * param_shape[1] *
                        param_shape[2] * param_shape[3]
                    )
                    output_shape = np.array(
                        workspace.FetchBlob(str(op_output))).shape
                    layer_flops = layer_params * output_shape[2] * output_shape[3]
        elif op_type == 'FC':
            for op_input in op_inputs:
                if '_w' in op_input:
                    param_blob = op_input
                    param_shape = np.array(
                        workspace.FetchBlob(str(param_blob))).shape
                    layer_params = param_shape[0] * param_shape[1]
                    layer_flops = layer_params
        layer_params /= 1000000
        layer_flops /= 1000000000
        num_flops += layer_flops
        num_params += layer_params
    return num_flops, num_params
