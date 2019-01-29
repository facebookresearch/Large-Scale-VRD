# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from core.config_rel import cfg

import os
import numpy as np
import cPickle as pickle

import logging
logger = logging.getLogger(__name__)


def _augment_gt_boxes_by_perturbation(unique_gt_boxes, im_width, im_height):

    num_gt = unique_gt_boxes.shape[0]
    num_rois = 1000
    rois = np.zeros((num_rois, 4), dtype=np.float32)
    cnt = 0
    for i in range(num_gt):
        box = unique_gt_boxes[i]
        box_width = box[2] - box[0] + 1
        box_height = box[3] - box[1] + 1
        x_offset_max = (box_width - 1) // 2
        y_offset_max = (box_height - 1) // 2
        for _ in range(num_rois // num_gt):
            x_min_offset = np.random.uniform(low=-x_offset_max, high=x_offset_max)
            y_min_offset = np.random.uniform(low=-y_offset_max, high=y_offset_max)
            x_max_offset = np.random.uniform(low=-x_offset_max, high=x_offset_max)
            y_max_offset = np.random.uniform(low=-y_offset_max, high=y_offset_max)

            new_x_min = min(max(np.round(box[0] + x_min_offset), 0), im_width - 1)
            new_y_min = min(max(np.round(box[1] + y_min_offset), 0), im_height - 1)
            new_x_max = min(max(np.round(box[2] + x_max_offset), 0), im_width - 1)
            new_y_max = min(max(np.round(box[3] + y_max_offset), 0), im_height - 1)

            new_box = np.array(
                [new_x_min, new_y_min, new_x_max, new_y_max]).astype(np.float32)
            rois[cnt] = new_box
            cnt += 1

    return rois


def get_gt_perturbed_proposals(gt_roidb):

    data_dir = os.path.join(cfg.DATA_DIR, 'proposals')
    proposal_file_path = os.path.join(data_dir, 'vg')
    proposal_name = 'gt_perturbed_proposals_flipped.pkl'
    proposal_file = os.path.join(proposal_file_path, proposal_name)
    logger.info('proposal file: {}'.format(proposal_file))
    if os.path.exists(proposal_file):
        logger.info('Loading existing proposals...')
        with open(proposal_file, 'rb') as fid:
            proposals = pickle.load(fid)
        return proposals
    else:
        logger.info('Generating gt perturbed proposals...')
        num_images = len(gt_roidb)

        blob_names = ['unique_all_rois_sbj', 'unique_all_rois_obj',
                      'unique_sbj_gt_inds', 'unique_obj_gt_inds']
        all_blobs = [{}] * num_images

        for im_i, entry in enumerate(gt_roidb):

            logger.info('Preparing roidb {}/{}'.format(im_i, num_images))

            all_blobs[im_i] = {k: [] for k in blob_names}

            sbj_gt_inds = np.where((entry['gt_sbj_classes'] > 0))[0]
            obj_gt_inds = np.where((entry['gt_obj_classes'] > 0))[0]

            scale = 1.0
            sbj_gt_rois = entry['sbj_boxes'][sbj_gt_inds, :] * scale
            obj_gt_rois = entry['obj_boxes'][obj_gt_inds, :] * scale

            sbj_gt_rois = sbj_gt_rois.astype(np.float32)
            obj_gt_rois = obj_gt_rois.astype(np.float32)

            sbj_gt_boxes = np.zeros((len(sbj_gt_inds), 6), dtype=np.float32)
            sbj_gt_boxes[:, 0] = 0  # batch inds
            sbj_gt_boxes[:, 1:5] = sbj_gt_rois
            sbj_gt_boxes[:, 5] = entry['gt_sbj_classes'][sbj_gt_inds]

            obj_gt_boxes = np.zeros((len(obj_gt_inds), 6), dtype=np.float32)
            obj_gt_boxes[:, 0] = 0  # batch inds
            obj_gt_boxes[:, 1:5] = obj_gt_rois
            obj_gt_boxes[:, 5] = entry['gt_obj_classes'][obj_gt_inds]

            # Get unique boxes
            rows = set()
            unique_sbj_gt_inds = []
            for idx, row in enumerate(sbj_gt_boxes):
                if tuple(row) not in rows:
                    rows.add(tuple(row))
                    unique_sbj_gt_inds.append(idx)
            unique_sbj_gt_boxes = sbj_gt_boxes[unique_sbj_gt_inds, :]

            rows = set()
            unique_obj_gt_inds = []
            for idx, row in enumerate(obj_gt_boxes):
                if tuple(row) not in rows:
                    rows.add(tuple(row))
                    unique_obj_gt_inds.append(idx)
            unique_obj_gt_boxes = obj_gt_boxes[unique_obj_gt_inds, :]

            # use better sampling by default
            im_width = entry['width'] * scale
            im_height = entry['height'] * scale

            _rois_sbj = _augment_gt_boxes_by_perturbation(
                unique_sbj_gt_boxes[:, 1:5], im_width, im_height)
            rois_sbj = np.zeros((_rois_sbj.shape[0], 5), dtype=np.float32)
            rois_sbj[:, 0] = 0
            rois_sbj[:, 1:5] = _rois_sbj

            _rois_obj = _augment_gt_boxes_by_perturbation(
                unique_obj_gt_boxes[:, 1:5], im_width, im_height)
            rois_obj = np.zeros((_rois_obj.shape[0], 5), dtype=np.float32)
            rois_obj[:, 0] = 0
            rois_obj[:, 1:5] = _rois_obj

            rows = set()
            unique_sbj_rois_inds = []
            for idx, row in enumerate(rois_sbj):
                if tuple(row) not in rows:
                    rows.add(tuple(row))
                    unique_sbj_rois_inds.append(idx)
            unique_rois_sbj = rois_sbj[unique_sbj_rois_inds, :]

            rows = set()
            unique_obj_rois_inds = []
            for idx, row in enumerate(rois_obj):
                if tuple(row) not in rows:
                    rows.add(tuple(row))
                    unique_obj_rois_inds.append(idx)
            unique_rois_obj = rois_obj[unique_obj_rois_inds, :]

            unique_all_rois_sbj = \
                np.vstack((unique_rois_sbj, unique_sbj_gt_boxes[:, :-1]))
            unique_all_rois_obj = \
                np.vstack((unique_rois_obj, unique_obj_gt_boxes[:, :-1]))

            for k in all_blobs[im_i]:
                all_blobs[im_i][k] = locals()[k]

        proposals = all_blobs

        import pdb
        pdb.set_trace()

        with open(proposal_file, 'wb') as fid:
            pickle.dump(proposals, fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote shdet gt perturbed proposals to {}'.format(
            os.path.abspath(proposal_file)))

        return proposals
