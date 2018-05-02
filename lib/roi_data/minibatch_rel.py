# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training/validating/testing a Fast R-CNN network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2

from core.config_rel import cfg
import utils.blob as blob_utils
from utils.timer import Timer
from roi_data.fast_rcnn_rel import add_fast_rcnn_blobs

import logging
logger = logging.getLogger(__name__)


def get_minibatch_blob_names(split):
    """Returns blob names in the order in which they are read by the data
    loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    # Fast R-CNN like models trained on precomputed proposals
    # rois blob: holds R regions of interest, each is a 5-tuple
    # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
    # rectangle (x1, y1, x2, y2)

    # if split == 'train' or split.find('tail_P') >= 0:
    if split == 'train':
        blob_names = ['data']
        blob_names += ['sbj_rois']
        blob_names += ['obj_rois']
        # if cfg.MODEL.TYPE.find('spo') < 0 and \
        #    cfg.MODEL.TYPE.find('v2') < 0:
        #     blob_names += ['rel_rois']
        # else:
        blob_names += ['rel_rois_sbj']
        blob_names += ['rel_rois_obj']
        blob_names += ['rel_rois_prd']
        if cfg.MODEL.LOSS_TYPE.find('TRIPLET') >= 0:
            # By Ji on 08/12/2017
            blob_names += ['sbj_pos_vecs']
            blob_names += ['obj_pos_vecs']
            blob_names += ['rel_pos_vecs']
            if cfg.DATASET.find('visual_genome') >= 0 and \
                    cfg.MODEL.SUBTYPE.find('yall') < 0:
                blob_names += ['sbj_neg_vecs']
                blob_names += ['obj_neg_vecs']
                blob_names += ['rel_neg_vecs']
            blob_names += ['sbj_neg_affinity_mask']
            blob_names += ['obj_neg_affinity_mask']
            blob_names += ['rel_neg_affinity_mask']
            # By Ji on 12/04/2017
            # if cfg.DATASET.find('visual_relationship_detection') >= 0 or \
            #         cfg.MODEL.SUBTYPE.find('yall') >= 0:
            #     blob_names += ['all_obj_word_vecs']
            #     blob_names += ['all_prd_word_vecs']
            # BY Ji on 09/17/2017
            if cfg.MODEL.LOSS_TYPE.find('CLUSTER') >= 0:
                blob_names += ['sbj_pos_affinity_mask']
                blob_names += ['obj_pos_affinity_mask']
                blob_names += ['rel_pos_affinity_mask']
        # elif cfg.MODEL.LOSS_TYPE == 'SOFTMAX':
        #     blob_names += ['sbj_labels_int32']
        #     blob_names += ['obj_labels_int32']
        #     blob_names += ['rel_labels_int32']
        blob_names += ['sbj_pos_labels_int32']
        blob_names += ['obj_pos_labels_int32']
        blob_names += ['rel_pos_labels_int32']
        blob_names += ['sbj_pos_starts']
        blob_names += ['obj_pos_starts']
        blob_names += ['rel_pos_starts']
        blob_names += ['sbj_pos_ends']
        blob_names += ['obj_pos_ends']
        blob_names += ['rel_pos_ends']
        blob_names += ['sbj_neg_starts']
        blob_names += ['obj_neg_starts']
        blob_names += ['rel_neg_starts']
        blob_names += ['sbj_neg_ends']
        blob_names += ['obj_neg_ends']
        blob_names += ['rel_neg_ends']
        # if cfg.TRAIN.OVERSAMPLE_SO:
        #     blob_names += ['sbj_low_shot_ends']
        #     blob_names += ['sbj_regular_starts']
        #     # blob_names += ['sbj_real_starts']
        #     blob_names += ['obj_low_shot_ends']
        #     blob_names += ['obj_regular_starts']
        #     # blob_names += ['obj_real_starts']
        # if cfg.TRAIN.OVERSAMPLE:
        #     blob_names += ['rel_low_shot_ends']
        #     blob_names += ['rel_regular_starts']
        #     # blob_names += ['rel_real_starts']
        #     # if cfg.TRAIN.GENERATE or cfg.TRAIN.GENERATE2:
        #     #     blob_names += ['rel_z_sbj']
        #     #     blob_names += ['rel_z_obj']
        #     #     blob_names += ['rel_z_prd']
        if cfg.TRAIN.ADD_LOSS_WEIGHTS:
            blob_names += ['rel_pos_weights']
        if cfg.TRAIN.ADD_LOSS_WEIGHTS_SO:
            blob_names += ['sbj_pos_weights']
            blob_names += ['obj_pos_weights']
    else:  # val/test
        blob_names = ['data']
        blob_names += ['sbj_rois']
        blob_names += ['obj_rois']
        # if cfg.MODEL.TYPE.find('spo') < 0 and \
        #    cfg.MODEL.TYPE.find('v2') < 0:
        #     blob_names += ['rel_rois']
        # else:
        blob_names += ['rel_rois_sbj']
        blob_names += ['rel_rois_obj']
        blob_names += ['rel_rois_prd']
        blob_names += ['sbj_pos_labels_int32']
        blob_names += ['obj_pos_labels_int32']
        blob_names += ['rel_pos_labels_int32']
        blob_names += ['sbj_gt_boxes']
        blob_names += ['obj_gt_boxes']
        blob_names += ['rel_gt_boxes']
        # By Ji on 12/13/2017
        blob_names += ['image_idx']
        if cfg.DATASET.find('vrd') < 0:
            blob_names += ['image_id']
        # By Ji on 12/20/2017
        blob_names += ['image_scale']
        # By Ji on 12/20/2017
        blob_names += ['subbatch_id']
        # By Ji on 12/27/2017
        blob_names += ['num_proposals']

    return blob_names


def get_minibatch(split, landb, roidb, roidb_inds, proposals, low_shot_helper):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names(split)}

    # logger.info('len(roidb): {}'.format(len(roidb)))
    # logger.info('len(proposals): {}'.format(len(proposals)))

    # Get the input image blob, formatted for caffe2
    im_blob, im_scales = _get_image_blob(roidb)
    blobs['data'] = im_blob
    # add_fast_rcnn_blobs_timer = Timer()
    # add_fast_rcnn_blobs_timer.tic()

    valid = add_fast_rcnn_blobs(
        blobs, im_scales, landb, roidb, roidb_inds, proposals, split,
        low_shot_helper)

    # add_fast_rcnn_blobs_timer.toc()
    # logger.info(
    #     'add_fast_rcnn_blobs_time: {}'.format(add_fast_rcnn_blobs_timer.total_time))

    return blobs, valid


def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.SCALES), size=num_images)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        cnt = 1
        while im is None:
            cnt += 1
            logger.info(
                'NoneType image found. Trying to read for {:d} times'.format(cnt))
            im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.SCALES[scale_inds[i]]
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, target_size, cfg.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales
