from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import numpy as np

from core.config_rel import cfg
from datasets.roidb_rel import combined_roidb_for_training

import logging

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    roidb = combined_roidb_for_training('visual_genome_spo_jn_mg_train', None)
    logger.info('roidb length: {}'.format(len(roidb)))
