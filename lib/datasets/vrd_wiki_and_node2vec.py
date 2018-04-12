# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from datasets.imdb_rel import imdb_rel
import utils.boxes as box_utils
import numpy as np
import scipy.sparse
import json
import cPickle
from core.config_rel import cfg

import gensim
# from autocorrect import spell
from numpy import linalg as la
import PIL

import logging
logger = logging.getLogger(__name__)


class vrd_wiki_and_node2vec(imdb_rel):
    def __init__(self, image_set):
        imdb_rel.__init__(
            self, 'vrd_wiki_and_node2vec_' + image_set)
        self._image_set = image_set
        self._data_path = os.path.join(cfg.DATA_DIR, 'Visual_Relation_Detection')
        self._object_classes = ('__background__',  # always index 0
                                'person',
                                'sky',
                                'building',
                                'truck',
                                'bus',
                                'table',
                                'shirt',
                                'chair',
                                'car',
                                'train',
                                'glasses',
                                'tree',
                                'boat',
                                'hat',
                                'trees',
                                'grass',
                                'pants',
                                'road',
                                'motorcycle',
                                'jacket',
                                'monitor',
                                'wheel',
                                'umbrella',
                                'plate',
                                'bike',
                                'clock',
                                'bag',
                                'shoe',
                                'laptop',
                                'desk',
                                'cabinet',
                                'counter',
                                'bench',
                                'shoes',
                                'tower',
                                'bottle',
                                'helmet',
                                'stove',
                                'lamp',
                                'coat',
                                'bed',
                                'dog',
                                'mountain',
                                'horse',
                                'plane',
                                'roof',
                                'skateboard',
                                'traffic light',
                                'bush',
                                'phone',
                                'airplane',
                                'sofa',
                                'cup',
                                'sink',
                                'shelf',
                                'box',
                                'van',
                                'hand',
                                'shorts',
                                'post',
                                'jeans',
                                'cat',
                                'sunglasses',
                                'bowl',
                                'computer',
                                'pillow',
                                'pizza',
                                'basket',
                                'elephant',
                                'kite',
                                'sand',
                                'keyboard',
                                'plant',
                                'can',
                                'vase',
                                'refrigerator',
                                'cart',
                                'skis',
                                'pot',
                                'surfboard',
                                'paper',
                                'mouse',
                                'trash can',
                                'cone',
                                'camera',
                                'ball',
                                'bear',
                                'giraffe',
                                'tie',
                                'luggage',
                                'faucet',
                                'hydrant',
                                'snowboard',
                                'oven',
                                'engine',
                                'watch',
                                'face',
                                'street',
                                'ramp',
                                'suitcase')
        self._predicate_classes = ('__background__',  # always index 0
                                   'on',
                                   'wear',
                                   'has',
                                   'next to',
                                   'sleep next to',
                                   'sit next to',
                                   'stand next to',
                                   'park next',
                                   'walk next to',
                                   'above',
                                   'behind',
                                   'stand behind',
                                   'sit behind',
                                   'park behind',
                                   'in the front of',
                                   'under',
                                   'stand under',
                                   'sit under',
                                   'near',
                                   'walk to',
                                   'walk',
                                   'walk past',
                                   'in',
                                   'below',
                                   'beside',
                                   'walk beside',
                                   'over',
                                   'hold',
                                   'by',
                                   'beneath',
                                   'with',
                                   'on the top of',
                                   'on the left of',
                                   'on the right of',
                                   'sit on',
                                   'ride',
                                   'carry',
                                   'look',
                                   'stand on',
                                   'use',
                                   'at',
                                   'attach to',
                                   'cover',
                                   'touch',
                                   'watch',
                                   'against',
                                   'inside',
                                   'adjacent to',
                                   'across',
                                   'contain',
                                   'drive',
                                   'drive on',
                                   'taller than',
                                   'eat',
                                   'park on',
                                   'lying on',
                                   'pull',
                                   'talk',
                                   'lean on',
                                   'fly',
                                   'face',
                                   'play with',
                                   'sleep on',
                                   'outside of',
                                   'rest on',
                                   'follow',
                                   'hit',
                                   'feed',
                                   'kick',
                                   'skate on')
        self._num_object_classes = len(self._object_classes)
        self._object_class_to_ind = \
            dict(zip(self._object_classes, range(self._num_object_classes)))
        print(len(self._object_class_to_ind))
        self._num_predicate_classes = len(self._predicate_classes)
        self._predicate_class_to_ind = \
            dict(zip(self._predicate_classes, range(self._num_predicate_classes)))
        print(len(self._predicate_class_to_ind))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        self.node2vec_model = None
        self.node2vec_vec_mean = None
        self.stop_words = ['a', 'an', 'the']

        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, self._image_set + '_images',
                                  str(index) + '.jpg')
        if not os.path.exists(image_path):
            image_path = os.path.join(self._data_path, self._image_set + '_images',
                                      str(index) + '.png')
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def get_widths_and_heights(self):
        cache_file = \
            os.path.join(self._data_path, self.name + '_image_sizes.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                sizes = cPickle.load(fid)
            print('{} image sizes loaded from {}'.format(self.name, cache_file))
            return sizes[:, 0], sizes[:, 1]

        sizes_list = [None] * self.num_images
        for i in range(self.num_images):
            sizes_list[i] = PIL.Image.open(self.image_path_at(i)).size
            print('getting size for image ', i)

        sizes = np.array(sizes_list)
        print('sizes.shape: ', sizes.shape)

        with open(cache_file, 'wb') as fid:
            cPickle.dump(sizes, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote image sizes to {}'.format(cache_file))

        return sizes[:, 0], sizes[:, 1]

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._data_path,
                                      'vrd_' + self._image_set + '_index.json')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            data = json.load(f)
        # remove ".jpg"
        data = [datum[:-4] for datum in data]
        return data

    def load_node2vec_vocab(self):
        base_fn = \
            os.path.join(cfg.DATA_DIR, 'label_embeddings', 'vg_300d_node2vec_unweighted_directed')
        node_fn = base_fn + '.nodes.txt'
        v = np.load(base_fn + '.syn0.npy')
        w = np.load(base_fn + '.ind2word.npy')
        nodes = {}
        with open(node_fn, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 1:
                    name = ''
                    nid = parts[0]
                else:
                    name, nid = parts
                nodes[np.int64(nid)] = name
        voc = {nodes[k].lower(): v for k, v in zip(w, v)}
        return voc, v.mean(axis=0)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            logger.info('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        # Load gt data from scratch
        # Load Google's pre-trained Word2Vec model.
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            cfg.DATA_DIR + '/models/GoogleNews-vectors-negative300.bin', binary=True)
        print('Model loaded.')
        # change everything into lowercase
        for key in self.model.vocab.keys():
            new_key = key.lower()
            self.model.vocab[new_key] = self.model.vocab.pop(key)
        print('Wiki words converted to lowercase.')

        # Load gt data from scratch
        # Load Yannis' Node2Vec model.
        self.node2vec_model, self.node2vec_vec_mean = self.load_node2vec_vocab()
        print('Node2vec model loaded.')
        print('Node2vec words converted to lowercase.')

        rel_data_path = os.path.join(self._data_path,
                                     'vrd_' + self._image_set + '_data.json')
        with open(rel_data_path) as f:
            all_rels = json.load(f)
        gt_roidb = \
            [self._load_vrd_annotation(img_rels, cnt, len(self.image_index))
             for cnt, img_rels in enumerate(all_rels)]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_vrd_annotation(self, img_rels, cnt, length):
        """
        Load image and bounding boxes info.
        """

        print("Loading image %d/%d..." % (cnt + 1, length))

        num_rels = len(img_rels)

        sbj_boxes = np.zeros((num_rels, 4), dtype=np.uint16)
        obj_boxes = np.zeros((num_rels, 4), dtype=np.uint16)
        rel_boxes = np.zeros((num_rels, 4), dtype=np.uint16)
        sbj_names = np.zeros((num_rels), dtype='U100')
        obj_names = np.zeros((num_rels), dtype='U100')
        prd_names = np.zeros((num_rels), dtype='U100')
        gt_sbj_classes = np.zeros((num_rels), dtype=np.int32)
        gt_obj_classes = np.zeros((num_rels), dtype=np.int32)
        gt_rel_classes = np.zeros((num_rels), dtype=np.int32)
        sbj_overlaps = np.zeros(
            (num_rels, self.num_object_classes), dtype=np.float32)
        obj_overlaps = np.zeros(
            (num_rels, self.num_object_classes), dtype=np.float32)
        rel_overlaps = np.zeros(
            (num_rels, self.num_predicate_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        sbj_seg_areas = np.zeros((num_rels), dtype=np.float32)
        obj_seg_areas = np.zeros((num_rels), dtype=np.float32)
        rel_seg_areas = np.zeros((num_rels), dtype=np.float32)

        # variables for word vectors
        half_dim = int(cfg.INPUT_LANG_EMBEDDING_DIM / 2)
        sbj_vecs = np.zeros(
            (num_rels, cfg.INPUT_LANG_EMBEDDING_DIM), dtype=np.float32)
        obj_vecs = np.zeros(
            (num_rels, cfg.INPUT_LANG_EMBEDDING_DIM), dtype=np.float32)
        prd_vecs = np.zeros(
            (num_rels, cfg.INPUT_LANG_EMBEDDING_DIM), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, rel in enumerate(img_rels):
            sbj_box = rel['sbj_box']
            obj_box = rel['obj_box']
            # from [ymin, ymax, xmin, xmax] to [xmin, ymin, xmax, ymax]
            sbj_box = [sbj_box[2], sbj_box[0], sbj_box[3], sbj_box[1]]
            obj_box = [obj_box[2], obj_box[0], obj_box[3], obj_box[1]]
            rel_box = box_utils.box_union(sbj_box, obj_box)
            sbj_boxes[ix, :] = sbj_box
            obj_boxes[ix, :] = obj_box
            rel_boxes[ix, :] = rel_box
            sbj_names[ix] = rel['sbj_name']
            obj_names[ix] = rel['obj_name']
            prd_names[ix] = rel['prd_name']
            sbj_cls = self._object_class_to_ind[str(sbj_names[ix])]
            obj_cls = self._object_class_to_ind[str(obj_names[ix])]
            prd_cls = self._predicate_class_to_ind[str(prd_names[ix])]
            gt_sbj_classes[ix] = sbj_cls
            gt_obj_classes[ix] = obj_cls
            gt_rel_classes[ix] = prd_cls
            sbj_overlaps[ix, sbj_cls] = 1.0
            obj_overlaps[ix, obj_cls] = 1.0
            rel_overlaps[ix, prd_cls] = 1.0
            sbj_seg_areas[ix] = (sbj_box[2] - sbj_box[0] + 1) * \
                                (sbj_box[3] - sbj_box[1] + 1)
            obj_seg_areas[ix] = (obj_box[2] - obj_box[0] + 1) * \
                                (obj_box[3] - obj_box[1] + 1)
            rel_seg_areas[ix] = (rel_box[2] - rel_box[0] + 1) * \
                                (rel_box[3] - rel_box[1] + 1)

            # add vectors for sbjs, objs and rels
            # sbj vec
            sbj_vecs_wiki = np.zeros(half_dim, dtype=np.float32)
            sbj_words = sbj_names[ix].split()
            for word in sbj_words:
                if word in self.model.vocab:
                    raw_word = self.model[word]
                    sbj_vecs_wiki += (raw_word / la.norm(raw_word))
                else:
                    print('Singular word found: ', word)
                    raise NameError('Terminated.')
            sbj_vecs_wiki /= len(sbj_words)
            sbj_vecs_wiki /= la.norm(sbj_vecs_wiki)

            sbj_vecs_node2vec = np.zeros(half_dim, dtype=np.float32)
            sbj_words = sbj_names[ix].split()
            for word in sbj_words:
                if word in self.node2vec_model:
                    raw_word = self.node2vec_model[word] - self.node2vec_vec_mean
                    sbj_vecs_node2vec += (raw_word / la.norm(raw_word))
                else:
                    # print('Singular sbj word found: ', word)
                    # sbj_vecs_node2vec += \
                    #     (self.node2vec_vec_mean / la.norm(self.node2vec_vec_mean))
                    sbj_vecs_node2vec += 1e-08
                    # raise NameError('Terminated.')
            # logger.info('sbj_vecs_norm: {}'.format(la.norm(sbj_vecs_node2vec)))
            sbj_vecs_node2vec /= len(sbj_words)
            sbj_vecs_node2vec /= la.norm(sbj_vecs_node2vec)

            sbj_vecs[ix][:half_dim] = sbj_vecs_wiki
            sbj_vecs[ix][half_dim:] = sbj_vecs_node2vec

            # obj vec
            obj_vecs_wiki = np.zeros(half_dim, dtype=np.float32)
            obj_words = obj_names[ix].split()
            for word in obj_words:
                if word in self.model.vocab:
                    raw_word = self.model[word]
                    obj_vecs_wiki += (raw_word / la.norm(raw_word))
                else:
                    print('Singular word found: ', word)
                    raise NameError('Terminated.')
            obj_vecs_wiki /= len(obj_words)
            obj_vecs_wiki /= la.norm(obj_vecs_wiki)

            obj_vecs_node2vec = np.zeros(half_dim, dtype=np.float32)
            obj_words = obj_names[ix].split()
            for word in obj_words:
                if word in self.node2vec_model:
                    raw_word = self.node2vec_model[word] - self.node2vec_vec_mean
                    obj_vecs_node2vec += (raw_word / la.norm(raw_word))
                else:
                    # print('Singular obj word found: ', word)
                    # obj_vecs_node2vec += \
                    #     (self.node2vec_vec_mean / la.norm(self.node2vec_vec_mean))
                    obj_vecs_node2vec += 1e-08
                    # raise NameError('Terminated.')
            # logger.info('obj_vecs_norm: {}'.format(la.norm(obj_vecs_node2vec)))
            obj_vecs_node2vec /= len(obj_words)
            obj_vecs_node2vec /= la.norm(obj_vecs_node2vec)

            obj_vecs[ix][:half_dim] = obj_vecs_wiki
            obj_vecs[ix][half_dim:] = obj_vecs_node2vec

            # prd vec
            prd_vecs_wiki = np.zeros(half_dim, dtype=np.float32)
            prd_words = prd_names[ix].split()
            for word in prd_words:
                if word in self.model.vocab:
                    raw_word = self.model[word]
                    prd_vecs_wiki += (raw_word / la.norm(raw_word))
                else:
                    print('Singular word found: ', word)
                    raise NameError('Terminated.')
            prd_vecs_wiki /= len(prd_words)
            prd_vecs_wiki /= la.norm(prd_vecs_wiki)

            prd_vecs_node2vec = np.zeros(half_dim, dtype=np.float32)
            prd_words = prd_names[ix].split()
            for word in prd_words:
                if word in self.node2vec_model:
                    raw_word = self.node2vec_model[word] - self.node2vec_vec_mean
                    prd_vecs_node2vec += (raw_word / la.norm(raw_word))
                else:
                    # print('Singular prd word found: ', word)
                    # prd_vecs_node2vec += \
                    #     (self.node2vec_vec_mean / la.norm(self.node2vec_vec_mean))
                    prd_vecs_node2vec += 1e-08
                    # raise NameError('Terminated.')
            # logger.info('prd_vecs_norm: {}'.format(la.norm(prd_vecs_node2vec)))
            prd_vecs_node2vec /= len(prd_words)
            prd_vecs_node2vec /= la.norm(prd_vecs_node2vec)

            prd_vecs[ix][:half_dim] = prd_vecs_wiki
            prd_vecs[ix][half_dim:] = prd_vecs_node2vec

        sbj_overlaps = scipy.sparse.csr_matrix(sbj_overlaps)
        obj_overlaps = scipy.sparse.csr_matrix(obj_overlaps)
        rel_overlaps = scipy.sparse.csr_matrix(rel_overlaps)

        return {'sbj_boxes': sbj_boxes,
                'obj_boxes': obj_boxes,
                'rel_boxes': rel_boxes,
                'sbj_names': sbj_names,
                'obj_names': obj_names,
                'prd_names': prd_names,
                'gt_sbj_classes': gt_sbj_classes,
                'gt_obj_classes': gt_obj_classes,
                'gt_rel_classes': gt_rel_classes,
                'gt_sbj_overlaps': sbj_overlaps,
                'gt_obj_overlaps': obj_overlaps,
                'gt_rel_overlaps': rel_overlaps,
                'sbj_seg_areas': sbj_seg_areas,
                'obj_seg_areas': obj_seg_areas,
                'rel_seg_areas': rel_seg_areas,
                'sbj_vecs': sbj_vecs,
                'obj_vecs': obj_vecs,
                'prd_vecs': prd_vecs,
                'flipped': False}
