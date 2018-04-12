from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import numpy as np
import math
import argparse
import pprint

import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)

from core.config_rel import (cfg, load_params_from_file, load_params_from_list)
from modeling import model_builder_rel
import utils.c2
import utils.train
from utils.timer import Timer
from utils.training_stats_rel import TrainingStats
import utils.env as envu
import utils.net_rel as nu
import utils.metrics_rel as metrics
from utils import helpers_rel
from utils import checkpoints_rel
from utils import evaluator_rel

from caffe2.python import workspace

import logging

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

utils.c2.import_contrib_ops()
utils.c2.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a network with Detectron'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--multi-gpu-testing',
        dest='multi_gpu_testing',
        help='Use cfg.NUM_GPUS GPUs for inference',
        action='store_true'
    )
    parser.add_argument(
        '--skip-test',
        dest='skip_test',
        help='Do not test the final model',
        action='store_true'
    )
    parser.add_argument(
        'opts',
        help='See lib/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def test():

    test_model, output_dir, checkpoint_dir = \
        model_builder_rel.create(cfg.MODEL.MODEL_NAME, train=False, split='test')
    logger.info('Test model built.')
    total_test_iters = int(math.ceil(
        float(len(test_model.roi_data_loader._roidb)) / float(cfg.NUM_DEVICES))) + 5
    test_evaluator = evaluator_rel.Evaluator(
        split=cfg.TEST.DATA_TYPE,
        roidb_size=len(test_model.roi_data_loader._roidb))
    test_timer = Timer()
    logger.info('Test epoch iters: {}'.format(total_test_iters))

    accumulated_accs = {}
    for key in test_evaluator.__dict__.keys():
        if key.find('acc') >= 0:
            accumulated_accs[key] = []
    # wins are for showing different plots
    wins = {}
    for key in test_evaluator.__dict__.keys():
        if key.find('acc') >= 0:
            wins[key] = None

    # params_file = os.path.join(checkpoint_dir, 'latest.pkl')
    params_file = cfg.TEST.WEIGHTS
    checkpoints_rel.initialize_params_from_file(
        model=test_model, weights_file=params_file,
        num_devices=cfg.NUM_DEVICES,
    )
    test_evaluator.reset()
    print("=> Testing model")
    for test_iter in range(0, total_test_iters):
        test_timer.tic()
        workspace.RunNet(test_model.net.Proto().name)
        test_timer.toc()
        test_evaluator.eval_im_dets_triplet_topk()
        logger.info('Tested {}/{} time: {}'.format(
            test_iter, total_test_iters, round(test_timer.average_time, 3)))
    iter_accs = test_evaluator.calculate_and_plot_accuracy()
    for key in iter_accs.keys():
        accumulated_accs[key].append(iter_accs[key])
    test_evaluator.save_all_dets()

    test_model.roi_data_loader.shutdown()

    logger.info('Testing has successfully finished...exiting!')


if __name__ == '__main__':

    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        load_params_from_file(args.cfg_file)
    if args.opts is not None:
        load_params_from_list(args.opts)
    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    test()
