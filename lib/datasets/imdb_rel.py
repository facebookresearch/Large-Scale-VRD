from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import os.path as osp
import PIL
from utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse
from core.config_rel import cfg


class imdb_rel(object):
    """Image database."""

    def __init__(self, name):
        self._name = name
        self._num_object_classes = -1
        self._num_predicate_classes = -1
        self._object_classes = []
        self._predicate_classes = []
        self._image_index = []
        self._obj_proposer = 'gt'
        self._roidb = None
        self._roidb_handler = self.default_roidb
        # Use this dict for storing dataset specific config options
        self.config = {}

    @property
    def name(self):
        return self._name

    @property
    def num_object_classes(self):
        return len(self._object_classes)

    @property
    def object_classes(self):
        return self._object_classes

    @property
    def num_predicate_classes(self):
        return len(self._predicate_classes)

    @property
    def predicate_classes(self):
        return self._predicate_classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    def set_proposal_method(self, method):
        method = eval('self.' + method + '_roidb')
        self.roidb_handler = method

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def num_images(self):
        return len(self.image_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def _get_widths(self):
        return [PIL.Image.open(self.image_path_at(i)).size[0]
                for i in range(self.num_images)]

    def append_flipped_images(self):
        num_images = self.num_images
        widths = self._get_widths()
        for i in range(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes': boxes,
                     'gt_overlaps': self.roidb[i]['gt_overlaps'],
                     'gt_classes': self.roidb[i]['gt_classes'],
                     'flipped': True}
            self.roidb.append(entry)
        self._image_index = self._image_index * 2

    def evaluate_recall(self, candidate_boxes=None, thresholds=None,
                        area='all', limit=None):
        """Evaluate detection proposal recall metrics.
                Returns:
                    results: dictionary of results with keys
                        'ar': average recall
                        'recalls': vector recalls at each IoU overlap threshold
                        'thresholds': vector of IoU overlap thresholds
                        'gt_overlaps': vector of all ground-truth overlaps
                """
        # Record max overlap value for each gt box
        # Return vector of overlap values

        assert candidate_boxes is not None

        areas = {'all': 0, 'small': 1, 'medium': 2, 'large': 3,
                 '96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}
        area_ranges = [[0 ** 2, 1e5 ** 2],  # all
                       [0 ** 2, 32 ** 2],  # small
                       [32 ** 2, 96 ** 2],  # medium
                       [96 ** 2, 1e5 ** 2],  # large
                       [96 ** 2, 128 ** 2],  # 96-128
                       [128 ** 2, 256 ** 2],  # 128-256
                       [256 ** 2, 512 ** 2],  # 256-512
                       [512 ** 2, 1e5 ** 2],  # 512-inf
                       ]
        assert area in areas, 'unknown area range: {}'.format(area)
        area_range = area_ranges[areas[area]]
        gt_overlaps = np.zeros(0)
        # By Ji
        gt_sbj_overlaps = np.zeros(0)
        gt_obj_overlaps = np.zeros(0)
        # By Ji
        gt_all_inds = []
        gt_all_ovrs = []
        bx_sbj_inds = []
        bx_sbj_ovrs = []
        bx_obj_inds = []
        bx_obj_ovrs = []
        sbj_num_pos = 0
        obj_num_pos = 0
        # rel_num_pos = 0

        # By Ji on 09/11/2016: only consider top N proposals
        # N = 5000

        for i in range(self.num_images):

            print('Calculating image %d/%d...' % (i + 1, self.num_images))

            boxes = candidate_boxes
            # if boxes.shape[0] == 0:
            #     continue
            # if limit is not None and boxes.shape[0] > limit:
            #     boxes = boxes[:limit, :]

            sbj_boxes = boxes['boxes_sbj'][i]
            obj_boxes = boxes['boxes_obj'][i]

            scores = boxes['scores'][i]
            order = scores.argsort(kind='mergesort')[::-1]
            if limit is not None:
                order = order[:limit]
            sbj_boxes = sbj_boxes[order, :]
            obj_boxes = obj_boxes[order, :]

            print('num_proposals: ', sbj_boxes.shape[0])

            # Checking for max_overlaps == 1 avoids including crowd annotations
            # (...pretty hacking :/)

            # calculate overlaps for subjects
            max_gt_sbj_overlaps = self.roidb[i][
                'gt_sbj_overlaps'].toarray().max(axis=1)
            gt_sbj_inds = np.where((self.roidb[i]['gt_sbj_classes'] > 0) &
                                   (max_gt_sbj_overlaps == 1))[0]
            gt_sbj_boxes = self.roidb[i]['sbj_boxes'][gt_sbj_inds, :]
            gt_sbj_areas = self.roidb[i]['sbj_seg_areas'][gt_sbj_inds]
            valid_gt_sbj_inds = np.where((gt_sbj_areas >= area_range[0]) &
                                         (gt_sbj_areas <= area_range[1]))[0]
            gt_sbj_boxes = gt_sbj_boxes[valid_gt_sbj_inds, :]
            sbj_num_pos += len(valid_gt_sbj_inds)
            sbj_overlaps = bbox_overlaps(sbj_boxes.astype(np.float32),
                                         gt_sbj_boxes.astype(np.float32))
            # calculate overlaps for objects
            max_gt_obj_overlaps = self.roidb[i][
                'gt_obj_overlaps'].toarray().max(axis=1)
            gt_obj_inds = np.where((self.roidb[i]['gt_obj_classes'] > 0) &
                                   (max_gt_obj_overlaps == 1))[0]
            gt_obj_boxes = self.roidb[i]['obj_boxes'][gt_obj_inds, :]
            gt_obj_areas = self.roidb[i]['obj_seg_areas'][gt_obj_inds]
            valid_gt_obj_inds = np.where((gt_obj_areas >= area_range[0]) &
                                         (gt_obj_areas <= area_range[1]))[0]
            gt_obj_boxes = gt_obj_boxes[valid_gt_obj_inds, :]
            obj_num_pos += len(valid_gt_obj_inds)
            obj_overlaps = bbox_overlaps(obj_boxes.astype(np.float32),
                                         gt_obj_boxes.astype(np.float32))
            # # calculate overlaps for relations
            # max_gt_rel_overlaps = self.roidb[i][
            #     'gt_rel_overlaps'].toarray().max(axis=1)
            # gt_rel_inds = np.where((self.roidb[i]['gt_rel_classes'] > 0) &
            #                        (max_gt_rel_overlaps == 1))[0]
            # gt_rel_boxes = self.roidb[i]['rel_boxes'][gt_rel_inds, :]
            # gt_rel_areas = self.roidb[i]['rel_seg_areas'][gt_rel_inds]
            # valid_gt_rel_inds = np.where((gt_rel_areas >= area_range[0]) &
            #                              (gt_rel_areas <= area_range[1]))[0]
            # gt_rel_boxes = gt_rel_boxes[valid_gt_rel_inds, :]
            # # rel_num_pos += len(valid_gt_rel_inds)
            # rel_overlaps = bbox_overlaps(rel_boxes.astype(np.float),
            #                              gt_rel_boxes.astype(np.float))

            assert sbj_overlaps.shape == obj_overlaps.shape

            # add up the sbj and obj overlaps to get total overlaps
            all_overlaps = (sbj_overlaps + obj_overlaps) / 2.0
            # all_overlaps = rel_overlaps
            # print 'sbj_overlaps.shape: ', sbj_overlaps.shape
            # print 'obj_overlaps.shape: ', obj_overlaps.shape
            # print 'all_overlaps.shape: ', all_overlaps.shape
            # print 'sbj_overlaps[0]: ', sbj_overlaps[0]
            # print 'obj_overlaps[0]: ', obj_overlaps[0]
            # print 'all_overlaps[0]: ', all_overlaps[0]
            # print all_overlaps

            _gt_overlaps = np.zeros((gt_sbj_boxes.shape[0]))
            _gt_sbj_overlaps = np.zeros((gt_sbj_boxes.shape[0]))
            _gt_obj_overlaps = np.zeros((gt_sbj_boxes.shape[0]))
            # By Ji
            _gt_all_inds = []
            _gt_all_ovrs = []
            _bx_sbj_inds = []
            _bx_sbj_ovrs = []
            _bx_obj_inds = []
            _bx_obj_ovrs = []
            num_iters = min(gt_sbj_boxes.shape[0], sbj_boxes.shape[0])
            for j in range(num_iters):
                # find which proposal box maximally covers each gt box
                argmax_all_overlaps = all_overlaps.argmax(axis=0)
                # and get the iou amount of coverage for each gt pair
                max_all_overlaps = all_overlaps.max(axis=0)

                # find which gt pair is 'best' covered (i.e. 'best' = most iou)
                gt_ind = max_all_overlaps.argmax()
                gt_ovr = max_all_overlaps.max()
                assert(gt_ovr >= 0)
                # find the proposal pair that covers the best covered gt box
                pair_ind = argmax_all_overlaps[gt_ind]
                # By Ji
                _gt_all_inds.append(gt_ind)
                _gt_all_ovrs.append(gt_ovr)
                _gt_all_inds.append(gt_ind)
                _gt_all_ovrs.append(gt_ovr)
                _bx_sbj_inds.append(pair_ind)
                _bx_sbj_ovrs.append(sbj_overlaps[pair_ind, gt_ind])
                _bx_obj_inds.append(pair_ind)
                _bx_obj_ovrs.append(obj_overlaps[pair_ind, gt_ind])
                # record the iou coverage of this gt box
                _gt_overlaps[j] = all_overlaps[pair_ind, gt_ind]
                _gt_sbj_overlaps[j] = sbj_overlaps[pair_ind, gt_ind]
                _gt_obj_overlaps[j] = obj_overlaps[pair_ind, gt_ind]
                assert(_gt_overlaps[j] == gt_ovr)
                # mark the proposal box and the gt box as used
                all_overlaps[pair_ind, :] = -1
                all_overlaps[:, gt_ind] = -1

            # append recorded iou coverage level
            gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))
            gt_sbj_overlaps = np.hstack((gt_sbj_overlaps, _gt_sbj_overlaps))
            gt_obj_overlaps = np.hstack((gt_obj_overlaps, _gt_obj_overlaps))
            # By Ji: _gt_overlaps are already sorted in descending order
            gt_all_inds.append(_gt_all_inds)
            gt_all_ovrs.append(_gt_all_ovrs)
            bx_sbj_inds.append(_bx_sbj_inds)
            bx_sbj_ovrs.append(_bx_sbj_ovrs)
            bx_obj_inds.append(_bx_obj_inds)
            bx_obj_ovrs.append(_bx_obj_ovrs)

        gt_overlaps = np.sort(gt_overlaps)
        if thresholds is None:
            step = 0.05
            thresholds = np.arange(0.5, 0.95 + 1e-5, step)
        recalls = np.zeros_like(thresholds)
        # compute recall for each iou threshold
        num_pos = (sbj_num_pos + obj_num_pos) / 2
        # for i, t in enumerate(thresholds):
        #     recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
        for idx, t in enumerate(thresholds):
            # cnt = 0
            # for i in xrange(gt_sbj_overlaps.shape[0]):
            #     for j in xrange(gt_obj_overlaps.shape[0]):
            #         if gt_sbj_overlaps[i] >= t and gt_obj_overlaps[j] >= t:
            #             cnt += 1
            # recalls[idx] = cnt / float(num_pos)
            inds = np.where((gt_sbj_overlaps >= t) & (gt_obj_overlaps >= t))[0]
            print('inds.size: ', inds.size)
            print('num_pos: ', num_pos)
            recalls[idx] = inds.size / float(num_pos)
        # ar = 2 * np.trapz(recalls, thresholds)
        ar = np.mean(recalls)
        # return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
        #         'gt_overlaps': gt_overlaps}
        # By Ji
        return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
                'gt_all_inds': gt_all_inds, 'gt_all_ovrs': gt_all_ovrs,
                'bx_sbj_inds': bx_sbj_inds, 'bx_sbj_ovrs': bx_sbj_ovrs,
                'bx_obj_inds': bx_obj_inds, 'bx_obj_ovrs': bx_obj_ovrs}

    def create_roidb_from_box_list(self, box_list, gt_roidb):
        assert len(box_list) == self.num_images, \
            'Number of boxes must match number of ground-truth images'
        roidb = []
        for i in range(self.num_images):
            boxes = box_list[i]
            num_boxes = boxes.shape[0]
            overlaps = np.zeros(
                (num_boxes, self.num_classes), dtype=np.float32)

            if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
                gt_boxes = gt_roidb[i]['boxes']
                gt_classes = gt_roidb[i]['gt_classes']
                gt_overlaps = bbox_overlaps(boxes.astype(np.float32),
                                            gt_boxes.astype(np.float32))
                argmaxes = gt_overlaps.argmax(axis=1)
                maxes = gt_overlaps.max(axis=1)
                I = np.where(maxes > 0)[0]
                overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

            overlaps = scipy.sparse.csr_matrix(overlaps)
            roidb.append({
                'boxes': boxes,
                'gt_classes': np.zeros((num_boxes,), dtype=np.int32),
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': np.zeros((num_boxes,), dtype=np.float32),
            })
        return roidb

    @staticmethod
    def merge_roidbs(a, b):
        assert len(a) == len(b)
        for i in range(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                            b[i]['gt_classes']))
            a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                       b[i]['gt_overlaps']])
            a[i]['seg_areas'] = np.hstack((a[i]['seg_areas'],
                                           b[i]['seg_areas']))
        return a

    def competition_mode(self, on):
        """Turn competition mode on or off."""
        pass
