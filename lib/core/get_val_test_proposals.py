from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from core.config_rel import cfg
import os
import cPickle as pickle

import logging
logger = logging.getLogger(__name__)


def get_relpn_test_proposals():

    proposal_file = cfg.TEST.PROPOSAL_FILE
    with open(proposal_file, 'rb') as fid:
        relpn_proposals = pickle.load(fid)
    # format conversion
    proposals = [dict(boxes_sbj=relpn_proposals['boxes_sbj'][ind],
                      boxes_obj=relpn_proposals['boxes_obj'][ind],
                      boxes_rel=relpn_proposals['boxes_rel'][ind])
                      for ind in range(len(relpn_proposals['boxes_sbj']))]
    return proposals


def get_gt_val_test_proposals(split, gt_roidb):

    if cfg.DATASET.find('vrd') >= 0:
        data_name = 'vrd'
    else:
        data_name = 'vg'
    proposal_file = os.path.join(
        cfg.DATA_DIR, 'proposals', data_name, 'gt_proposals_' + split + '.pkl')
    if os.path.exists(proposal_file):
        logger.info('Loading existing proposals from {}'.format(proposal_file))
        with open(proposal_file, 'rb') as fid:
            gt_proposals = pickle.load(fid)
        # format conversion
        proposals = [dict(boxes_sbj=gt_proposals['boxes_sbj'][ind],
                          boxes_obj=gt_proposals['boxes_obj'][ind],
                          boxes_rel=gt_proposals['boxes_rel'][ind])
                          for ind in range(len(gt_proposals['boxes_sbj']))]
    else:
        logger.info('generating proposals for {}'.format(data_name + '_' + split))
        num_images = len(gt_roidb)
        im_list_boxes_sbj = [[]] * num_images
        im_list_boxes_obj = [[]] * num_images
        im_list_boxes_rel = [[]] * num_images
        for im_i, _ in enumerate(gt_roidb):
            boxes_sbj = gt_roidb[im_i]['sbj_boxes']
            boxes_obj = gt_roidb[im_i]['obj_boxes']
            boxes_rel = gt_roidb[im_i]['rel_boxes']
            im_list_boxes_sbj[im_i] = boxes_sbj
            im_list_boxes_obj[im_i] = boxes_obj
            im_list_boxes_rel[im_i] = boxes_rel

        gt_proposals = dict(boxes_sbj=im_list_boxes_sbj,
                            boxes_obj=im_list_boxes_obj,
                            boxes_rel=im_list_boxes_rel)
        with open(proposal_file, 'wb') as fid:
            pickle.dump(gt_proposals, fid, pickle.HIGHEST_PROTOCOL)
        logger.info('wrote gt proposals to {}'.format(proposal_file))

        # format conversion
        proposals = [dict(boxes_sbj=gt_proposals['boxes_sbj'][ind],
                          boxes_obj=gt_proposals['boxes_obj'][ind],
                          boxes_rel=gt_proposals['boxes_rel'][ind])
                          for ind in range(len(gt_proposals['boxes_sbj']))]

    return proposals
