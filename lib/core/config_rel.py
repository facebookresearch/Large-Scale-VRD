# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Detectron config system.

This file specifies default config options for Detectron. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use merge_cfg_from_file(yaml_file) to load it and override the default
options.

Most tools in the tools directory take a --cfg option to specify an override
file and an optional list of override (key, value) pairs:
 - See tools/{train,test}_net.py for example code that uses merge_cfg_from_file
 - See configs/*/*.yaml for example config files

Detectron supports a lot of different model types, each of which has a lot of
different options. The result is a HUGE set of configuration options.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval
from past.builtins import basestring
from utils.collections import AttrDict
import copy
import logging
import numpy as np
import os
import os.path as osp
import yaml

from utils.io import cache_url

logger = logging.getLogger(__name__)

__C = AttrDict()
# Consumers can get config by:
#   from core.config import cfg
cfg = __C

__C.DATA_DIR = b'datasets/large_scale_VRD'
__C.OUTPUT_DIR = b'checkpoints'

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Number of GPUs to use
# __C.NUM_GPUS = 1

# Use NCCL for all reduce, otherwise use muji
# NCCL seems to work ok for 2 GPUs, but become prone to deadlocks when using
# 4 or 8
__C.USE_NCCL = False

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.DEDUP_BOXES = 1 / 16.

__C.BBOX_XFORM_CLIP = np.log(1000. / 16.)

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.SCALES = (600,)
# Max pixel size of the longest side of a scaled input image
__C.MAX_SIZE = 1000

# Training options
__C.TRAIN = AttrDict()
__C.TRAIN.PARAMS_FILE = b''
__C.TRAIN.DATA_TYPE = b'train'
# Initialize network with weights from this pickle file
__C.TRAIN.WEIGHTS = b''
# # Scales to use during training (can list multiple scales)
# # Each scale is the pixel size of an image's shortest side
# __C.TRAIN.SCALES = (600,)
# # Max pixel size of the longest side of a scaled input image
# __C.TRAIN.MAX_SIZE = 1000
# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 1
# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE_PER_IM = 128
# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25
# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5
# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.0
# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True
# Train bounding-box regressors
# This only refers to the Fast R-CNN bbox reg used to transform proposals
# This does not refer to the bbox reg that happens in RPN
__C.TRAIN.BBOX_REG = True
# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5
# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = False
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

__C.TRAIN.DROPOUT = 0.0

# Train using these proposals
__C.TRAIN.PROPOSAL_FILE = b''

# resNet or googleNet
__C.TRAIN.SCALE_JITTER_TYPE = b'resNet'
__C.TRAIN.JITTER_SCALES = [256, 480]
__C.TRAIN.CROP_SIZE = 224
# Number of iterations after which model should be tested on test/val data
__C.TRAIN.EVALUATION_FREQUENCY = 5005
__C.TRAIN.COMPUTE_MEAN_STD = False

__C.TRAIN.MARGIN_SO = 0.2
__C.TRAIN.MARGIN_P = 0.2

__C.TRAIN.NORM_SCALAR = 5.0

__C.TRAIN.OVERSAMPLE = False
__C.TRAIN.OVERSAMPLE_SO = False
__C.TRAIN.LOW_SHOT_THRESHOLD = -1

__C.TRAIN.OVERSAMPLE2 = False
__C.TRAIN.OVERSAMPLE_SO2 = False

__C.TRAIN.GENERATE = False
__C.TRAIN.GENERATE2 = False

__C.TRAIN.ADD_LOSS_WEIGHTS = False
__C.TRAIN.ADD_LOSS_WEIGHTS_SO = False

# ---------------------------------------------------------------------------- #
# Data loader options
# ---------------------------------------------------------------------------- #
__C.DATA_LOADER = AttrDict()

# Number of Python threads to use for the data loader (warning: using too many
# threads can cause GIL-based interference with Python Ops leading to *slower*
# training; 4 seems to be the sweet spot in our experience)
__C.DATA_LOADER.NUM_THREADS = 4

# Distributed training options
__C.DISTRIBUTED = AttrDict()
# below is valid in case of warm-up distributed training only
__C.DISTRIBUTED.START_ITER = 0
__C.DISTRIBUTED.DISTR_ON = False
__C.DISTRIBUTED.CLUSTER = False
# in case of distributed training, we should do random sampling without replacement
__C.DISTRIBUTED.RANDOM_SAMPLE = False
# number of machines to distribute training on
__C.DISTRIBUTED.NUM_NODES = 1
__C.DISTRIBUTED.RUN_ID = b''
__C.DISTRIBUTED.ALL_REDUCE_ENGINE = b'GLOO'
# use global shuffling for distributed training
__C.DISTRIBUTED.GLOBAL_SHUFFLE = False
__C.DISTRIBUTED.NUM_CONCURRENT_OPS = 4
# options to use Redis in distributed training
__C.DISTRIBUTED.REDIS = AttrDict()
__C.DISTRIBUTED.REDIS.REDIS_ON = False
__C.DISTRIBUTED.REDIS.REDIS_HOST = b'devgpu204.prn2.facebook.com'
__C.DISTRIBUTED.REDIS.REDIS_PORT = 6379

# Embedding Parameters
__C.OUTPUT_EMBEDDING_DIM = 300  # should be deprecated
__C.INPUT_LANG_EMBEDDING_DIM = 300  # should be deprecated
# For one hot case, should be for all cases as well
__C.INPUT_LANG_EMBEDDING_DIM_SBJ_OBJ = 300
__C.INPUT_LANG_EMBEDDING_DIM_PRD = 300
__C.OUTPUT_EMBEDDING_DIM_SBJ_OBJ = 300
__C.OUTPUT_EMBEDDING_DIM_PRD = 300

# Train model options
__C.MODEL = AttrDict()
__C.MODEL.STRIDE_1 = False
__C.MODEL.NUM_CLASSES = -1
__C.MODEL.MODEL_NAME = b''
__C.MODEL.DEPTH = 50
__C.MODEL.BN_MOMENTUM = 0.9
__C.MODEL.BN_EPSILON = 1.0000001e-5
__C.MODEL.FC_INIT_STD = 0.01
__C.MODEL.FC_BN = False
__C.MODEL.ADD_2FC = False
__C.MODEL.DIM_2FC = 1024
__C.MODEL.RESIDUAL_SCALE = 0.0
# options to optimize memory usage
__C.MODEL.ALLOW_INPLACE_SUM = True
# disable the inplace relu for collecting stats
__C.MODEL.ALLOW_INPLACE_RELU = True
__C.MODEL.MEMONGER = True
__C.MODEL.CUSTOM_BN_INIT = False
__C.MODEL.BN_INIT_GAMMA = 1.0
__C.MODEL.GRAD_ACCUM_FREQUENCY = 1
__C.MODEL.STRIDE_1 = False

# for ResNe(X)t only
__C.MODEL.RESNEXT = AttrDict()
__C.MODEL.RESNEXT.WIDTH_PER_GROUP = 4
__C.MODEL.RESNEXT.NUM_GROUPS = 32
# For pyramid net
__C.MODEL.PYRAMIDNET = AttrDict()
__C.MODEL.PYRAMIDNET.ALPHA = 462

# for ShuffleNet only
__C.MODEL.SHUFFLENET = AttrDict()
__C.MODEL.SHUFFLENET.ARCH_TYPE = b'arch1'
__C.MODEL.SHUFFLENET.NUM_GROUPS = 1
__C.MODEL.SHUFFLENET.WIDTH_PER_GROUP = 144

__C.MODEL.TYPE = b''
__C.MODEL.SUBTYPE = b''
__C.MODEL.SPECS = b''
__C.MODEL.LOSS_TYPE = b''  # TRIPLET, EUC
__C.MODEL.NUM_CLASSES_SBJ_OBJ = -1
__C.MODEL.NUM_CLASSES_PRD = -1
__C.MODEL.NUM_CLUSTERS_SBJ_OBJ = -1
__C.MODEL.NUM_CLUSTERS_PRD = -1
__C.MODEL.NUM_NEG_NAMES_SBJ_OBJ = -1
__C.MODEL.NUM_NEG_NAMES_PRD = -1

__C.MODEL.RPN_ONLY = False
__C.MODEL.FASTER_RCNN = False

# for trplet_and_softmax only
__C.MODEL.TRIPLET_RATIO = 1.0
__C.MODEL.CLUSTER_RATIO = 0.0
__C.MODEL.SOFTMAX_RATIO = 0.0

# Caffe2 net execution type
# Use 'prof_dag' to get profiling statistics
__C.MODEL.EXECUTION_TYPE = b'dag'

__C.VISUAL_EMBEDDING = AttrDict()
__C.VISUAL_EMBEDDING.WEIGHT_SHARING = b''  # SBJ_OBJ_SHARED, ALL_SHARED, ALL_UNSHARED
__C.VISUAL_EMBEDDING.L2_NORMALIZE = False
__C.TEXT_EMBEDDING = AttrDict()
__C.TEXT_EMBEDDING.HIDDEN_LAYERS = 0  # 1, 0
__C.TEXT_EMBEDDING.L2_NORMALIZE = False

# ---------------------------------------------------------------------------- #
# Fast R-CNN options
# ---------------------------------------------------------------------------- #
__C.FAST_RCNN = AttrDict()
__C.FAST_RCNN.MLP_HEAD_DIM = 1024
__C.FAST_RCNN.ROI_XFORM_METHOD = b'RoIPoolF'
# Only applies to RoIWarp, RoIWarpMax, and RoIAlign
__C.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO = 0
# Models may ignore this and use fixed values
__C.FAST_RCNN.ROI_XFORM_RESOLUTION = 14

__C.RPN = AttrDict()
__C.RPN.ON = False
# Note: these options are *not* used by FPN RPN; see FPN.RPN* options
# RPN anchor sizes
__C.RPN.SIZES = (64, 128, 256, 512)
# Stride of the feature map that RPN is attached to
__C.RPN.STRIDE = 16
# RPN anchor aspect ratios
__C.RPN.ASPECT_RATIOS = (0.5, 1, 2)

# For FPN only
__C.FPN = AttrDict()
__C.FPN.FPN_ON = False
__C.FPN.DIM = 256
__C.FPN.MAX_LEVEL = 5
__C.FPN.MIN_LEVEL = 2
# not sure how the below option will be useful, for now, let's have non-zero init
__C.FPN.ZERO_INIT_LATERAL = False
# Right now, only 2 layers can fit in memory for ResNet50-FPN model
__C.FPN.NUM_GLASS_LAYERS = 2


# Validation options
__C.VAL = AttrDict()
__C.VAL.PARAMS_FILE = b''
__C.VAL.DATA_TYPE = b'val'
# __C.VAL.IMS_PER_BATCH = 256
__C.VAL.TEN_CROP = False
__C.VAL.SCALE = 256
__C.VAL.CROP_SIZE = 224
# Initialize network with weights from this pickle file
__C.VAL.WEIGHTS = b''
# Scales to use during VALing (can list multiple scales)
# Each scale is the pixel size of an image's shorVAL side
# __C.VAL.SCALES = (600,)
# Max pixel size of the longest side of a scaled input image
# __C.VAL.MAX_SIZE = 1000
# using bounding-box regressors
__C.VAL.BBOX_REG = True
__C.VAL.PROPOSAL_FILES = ()
# using these proposals
__C.VAL.PROPOSAL_FILE = b''
__C.VAL.IMS_PER_BATCH = 1
__C.VAL.GROUP_SIZE = 4096


# Test options
__C.TEST = AttrDict()
__C.TEST.PARAMS_FILE = b''
__C.TEST.DATA_TYPE = b'test'
# __C.TEST.IMS_PER_BATCH = 256
__C.TEST.TEN_CROP = False
__C.TEST.SCALE = 256
__C.TEST.CROP_SIZE = 224

# Initialize network with weights from this pickle file
__C.TEST.WEIGHTS = b''
# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
# __C.TEST.SCALES = (600,)
# Max pixel size of the longest side of a scaled input image
# __C.TEST.MAX_SIZE = 1000
# Test using bounding-box regressors
__C.TEST.BBOX_REG = True
__C.TEST.PROPOSAL_FILES = ()
# Test using these proposals
__C.TEST.PROPOSAL_FILE = b''

__C.TEST.IMS_PER_BATCH = 1

__C.TEST.GROUP_SIZE = 4096

__C.TEST.GET_ALL_LAN_EMBEDDINGS = False
__C.TEST.GET_ALL_VIS_EMBEDDINGS = False

# SOLVER
__C.SOLVER = AttrDict()
__C.SOLVER.NESTEROV = True
__C.SOLVER.BASE_LR = 0.1
__C.SOLVER.LR_FACTOR = 1.0
# For imagenet1k, 150150 = 30 epochs for batch size of 256 images over 8 gpus
__C.SOLVER.STEP_SIZES = [150150, 150150, 150150]
# For above batch size, running 120 epochs = 600600 iterations
__C.SOLVER.NUM_ITERATIONS = 600600
__C.SOLVER.WEIGHT_DECAY = 0.0001
__C.SOLVER.WEIGHT_DECAY_BN = 0.0001
__C.SOLVER.MOMENTUM = 0.9
__C.SOLVER.LR_POLICY = b'step'
# The LR step sizes can be 'constant' or 'variable'
__C.SOLVER.LR_TYPE = b'variable'
__C.SOLVER.WARM_UP_ITERS = -1
__C.SOLVER.WARM_UP_LR = 1.0
__C.SOLVER.GRADUAL_INCREASE_LR = False
__C.SOLVER.GRADUAL_LR_STEP = 0.005
__C.SOLVER.GRADUAL_LR_START = 0.1
__C.SOLVER.SCALE_MOMENTUM = False
__C.SOLVER.GAMMA = 0.1
__C.SOLVER.WARM_UP_TYPE = b'NA'

# Checkpoint options
__C.CHECKPOINT = AttrDict()
__C.CHECKPOINT.CHECKPOINT_MODEL = True
__C.CHECKPOINT.CHECKPOINT_PERIOD = -1
__C.CHECKPOINT.RESUME = True
__C.CHECKPOINT.DIR = b'.'

# Metrics option
__C.METRICS = AttrDict()
# __C.METRICS.EVALUATE_ALL_CLASSES = True
__C.METRICS.EVALUATE_ALL_CLASSES = False
__C.METRICS.EVALUATE_FIRST_N_WAYS = False
__C.METRICS.FIRST_N_WAYS = 1000
__C.METRICS.NUM_MEDIAN_EPOCHS = 5

# GPU or CPU
__C.DEVICE = b'GPU'
# for example 8 gpus
__C.NUM_DEVICES = 8

__C.DATADIR = b''
__C.DATASET = b''
__C.DATA_SHUFFLE_K = 1
# The sources for imagenet dataset are: gfsai | laser
__C.DATA_SOURCE = b'gfsai'
__C.ROOT_DEVICE_ID = 0
__C.CUDNN_WORKSPACE_LIMIT = 256
__C.RNG_SEED = 2
__C.COMPUTE_LAYER_STATS = False
__C.LAYER_STATS_FREQ = 10
__C.STATS_LAYERS = [b'conv1', b'pred', b'res5_2_branch2', b'res2_0_branch2']
# use the following option to save the model proto for mobile predictions
__C.SAVE_MOBILE_PROTO = False

# Turn on the minibatch stats to debug whether data loader is slow
# NOTE that other CPU processes might interfere with the data loader
__C.PRINT_MB_STATS = False
__C.LOGGER_FREQUENCY = 10

__C.DATA = AttrDict()
__C.DATA.IMAGENET = AttrDict()
# Mean and Std are in BGR order for imagenet dataset
__C.DATA.IMAGENET.DATA_MEAN = [0.406, 0.456, 0.485]
__C.DATA.IMAGENET.DATA_STD = [0.225, 0.224, 0.229]
# PCA is in RGB order
__C.DATA.IMAGENET.PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203]
    ]
}
__C.DATA.CIFAR10 = AttrDict()
# Mean and std are in RGB order for cifar10 dataset
__C.DATA.CIFAR10.DATA_MEAN = [125.3, 123.0, 113.9]
__C.DATA.CIFAR10.DATA_STD = [63.0, 62.1, 66.7]


def merge_dicts(dict_a, dict_b):
    from ast import literal_eval
    for key, value in dict_a.items():
        if key not in dict_b:
            raise KeyError('Invalid key in config file: {}'.format(key))
        if type(value) is dict:
            dict_a[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        # the types must match, too
        old_type = type(dict_b[key])
        if old_type is not type(value) and value is not None:
                raise ValueError(
                    'Type mismatch ({} vs. {}) for config key: {}'.format(
                        type(dict_b[key]), type(value), key)
                )
        # recursively merge dicts
        if isinstance(value, AttrDict):
            try:
                merge_dicts(dict_a[key], dict_b[key])
            except BaseException:
                logger.critical('Error under config key: {}'.format(key))
                raise
        else:
            dict_b[key] = value


def load_params_from_file(filename):
    import yaml
    with open(filename, 'r') as fopen:
        yaml_config = AttrDict(yaml.load(fopen))
    merge_dicts(yaml_config, __C)


def load_params_from_list(args_list):
    from ast import literal_eval
    assert len(args_list) % 2 == 0, \
        'Looks like you forgot to specify values or keys for some args'
    for key, value in zip(args_list[0::2], args_list[1::2]):
        key_list = key.split('.')
        cfg = __C
        for subkey in key_list[:-1]:
            assert subkey in cfg, 'Config key {} not found'.format(subkey)
            cfg = cfg[subkey]
        subkey = key_list[-1]
        assert subkey in cfg, 'Config key {} not found'.format(subkey)
        try:
            # handle the case when v is a string literal
            val = literal_eval(value)
        except BaseException:
            val = value
        assert isinstance(val, type(cfg[subkey])) or cfg[subkey] is None, \
            'type {} does not match original type {}'.format(
                type(val), type(cfg[subkey]))
        cfg[subkey] = val


def get_output_dir(datasets, training=True):
    """Get the output directory determined by the current global config."""
    assert isinstance(datasets, (tuple, list, basestring)), \
        'datasets argument must be of type tuple, list or string'
    is_string = isinstance(datasets, basestring)
    dataset_name = datasets if is_string else ':'.join(datasets)
    tag = 'train' if training else 'test'
    # <output-dir>/<train|test>/<dataset-name>/<model-type>/
    outdir = osp.join(__C.OUTPUT_DIR, tag, dataset_name, __C.MODEL.TYPE)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    return outdir
