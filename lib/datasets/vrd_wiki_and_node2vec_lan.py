# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import os.path as osp
import numpy as np
import cPickle
from core.config_rel import cfg

import gensim
from numpy import linalg as la

import logging
logger = logging.getLogger(__name__)


class vrd_wiki_and_node2vec_lan():
    def __init__(self):
        self._data_path = os.path.join(cfg.DATA_DIR, 'Visual_Relation_Detection')
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)
        self._object_categories = ('person',
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
        self._predicate_categories = ('on',
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

        cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
        cache_file = os.path.join(cache_path, 'vrd_wiki_and_node2vec_gt_landb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                landb = cPickle.load(fid)
            logger.info(
                'vrd wiki and node2vec gt landb loaded from {}'.format(cache_file))
            self.obj_vecs = landb['obj_vecs']
            self.prd_vecs = landb['prd_vecs']
            return

        self.node2vec_model = None
        self.node2vec_vec_mean = None
        self.stop_words = ['a', 'an', 'the']

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

        half_dim = int(cfg.INPUT_LANG_EMBEDDING_DIM / 2)

        obj_singular_cnt = 0
        prd_singular_cnt = 0
        obj_cnt = 0
        prd_cnt = 0
        # variables for word vectors
        self.obj_vecs = np.zeros((len(self._object_categories),
                                 cfg.INPUT_LANG_EMBEDDING_DIM), dtype=np.float32)
        for ix, name in enumerate(self._object_categories):
            obj_vecs_wiki = np.zeros(half_dim, dtype=np.float32)
            words = name.split()
            for word in words:
                if word in self.model.vocab:
                    raw_word = self.model[word]
                    obj_vecs_wiki += (raw_word / la.norm(raw_word))
                else:
                    print('Singular word found: ', word)
                    raise NameError('Terminated.')
            obj_vecs_wiki /= len(words)
            obj_vecs_wiki /= la.norm(obj_vecs_wiki)

            obj_vecs_node2vec = np.zeros(half_dim, dtype=np.float32)
            words = name.split()
            for word in words:
                obj_cnt += 1
                if word in self.node2vec_model:
                    raw_word = self.node2vec_model[word] - self.node2vec_vec_mean
                    obj_vecs_node2vec += (raw_word / la.norm(raw_word))
                else:
                    # logger.info('Singular word found: {}'.format(word))
                    obj_singular_cnt += 1
                    obj_vecs_node2vec += 1e-08
                    # obj_vecs_node2vec += \
                    #     (self.node2vec_vec_mean / la.norm(self.node2vec_vec_mean))
                    # raise NameError('Terminated.')
            obj_vecs_node2vec /= len(words)
            obj_vecs_node2vec /= la.norm(obj_vecs_node2vec)

            self.obj_vecs[ix][:half_dim] = obj_vecs_wiki
            self.obj_vecs[ix][half_dim:] = obj_vecs_node2vec

        self.prd_vecs = np.zeros((len(self._predicate_categories),
                                 cfg.INPUT_LANG_EMBEDDING_DIM), dtype=np.float32)
        for ix, name in enumerate(self._predicate_categories):
            prd_vecs_wiki = np.zeros(half_dim, dtype=np.float32)
            words = name.split()
            for word in words:
                if word in self.model.vocab:
                    raw_word = self.model[word]
                    prd_vecs_wiki += (raw_word / la.norm(raw_word))
                else:
                    print('Singular word found: ', word)
                    raise NameError('Terminated.')
            prd_vecs_wiki /= len(words)
            prd_vecs_wiki /= la.norm(prd_vecs_wiki)

            prd_vecs_node2vec = np.zeros(half_dim, dtype=np.float32)
            words = name.split()
            for word in words:
                prd_cnt += 1
                if word in self.node2vec_model:
                    raw_word = self.node2vec_model[word] - self.node2vec_vec_mean
                    prd_vecs_node2vec += (raw_word / la.norm(raw_word))
                else:
                    # logger.info('Singular word found: {}'.format(word))
                    prd_singular_cnt += 1
                    prd_vecs_node2vec += 1e-08
                    # prd_vecs_node2vec += \
                    #     (self.node2vec_vec_mean / la.norm(self.node2vec_vec_mean))
                    # raise NameError('Terminated.')
            prd_vecs_node2vec /= len(words)
            prd_vecs_node2vec /= la.norm(prd_vecs_node2vec)

            self.prd_vecs[ix][:half_dim] = prd_vecs_wiki
            self.prd_vecs[ix][half_dim:] = prd_vecs_node2vec

        logger.info('prd_singular_cnt/prd_cnt: {}/{}'.format(prd_singular_cnt, prd_cnt))

        landb = dict(obj_vecs=self.obj_vecs, prd_vecs=self.prd_vecs)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(landb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt landb to {}'.format(cache_file))

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
