from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import numpy as np

from core.config_rel import cfg
from datasets.factory import get_imdb

import logging

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    ds = get_imdb('visual_genome_spo_jn_mg_test')
    roidb = ds.gt_roidb()
    logger.info('roidb length: {}'.format(len(roidb)))
