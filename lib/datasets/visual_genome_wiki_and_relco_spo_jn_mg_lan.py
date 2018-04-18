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


class visual_genome_wiki_and_relco_spo_jn_mg_lan():
    def __init__(self):
        self._data_path = os.path.join(cfg.DATA_DIR, 'Visual_Genome')
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

        _object_categories = []
        with open(self._data_path + '/object_categories_spo_joined_and_merged.txt') as obj_categories:
            for line in obj_categories:
                _object_categories.append(line[:-1])
        self._object_categories = list(set(_object_categories))
        print(len(self._object_categories))

        _predicate_categories = []
        with open(self._data_path + '/predicate_categories_spo_joined_and_merged.txt') as prd_categories:
            for line in prd_categories:
                _predicate_categories.append(line[:-1])
        self._predicate_categories = list(set(_predicate_categories))
        print(len(self._predicate_categories))

        cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
        cache_file = os.path.join(cache_path, 'vg_wiki_and_relco_spo_jn_mg_gt_landb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                landb = cPickle.load(fid)
            logger.info('vg wiki and relco spo joined and merged gt landb loaded from {}'.format(cache_file))
            # self.obj_name_to_neg_vecs = landb['obj_name_to_neg_vecs']
            # self.prd_name_to_neg_vecs = landb['prd_name_to_neg_vecs']
            self.obj_vecs = landb['obj_vecs']
            self.prd_vecs = landb['prd_vecs']
            return

        self.model = None
        self.relco_model = None
        self.relco_vec_mean = None

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
        # Load Yannis' rel_cooccur_300d model.
        self.relco_model = gensim.models.Word2Vec.load(
            cfg.DATA_DIR + '/label_embeddings/vg_300d_skipgram_rel')
        print('Model loaded.')
        self.relco_vec_mean = self.relco_model.syn0.mean(axis=0)
        # change everything into lowercase
        # for key in self.model.wv.vocab.keys():
        for key in self.relco_model.vocab.keys():
            new_key = key.lower()
            self.relco_model.vocab[new_key] = self.relco_model.vocab.pop(key)
        print('Relco words converted to lowercase.')

        half_dim = int(cfg.INPUT_LANG_EMBEDDING_DIM / 2)

        # get word vectors for all categories
        self.obj_vecs = np.zeros((len(self._object_categories),
                                 cfg.INPUT_LANG_EMBEDDING_DIM), dtype=np.float32)
        for ix, name in enumerate(_object_categories):
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

            obj_vecs_relco = np.zeros(half_dim, dtype=np.float32)
            words = name.split()
            for word in words:
                if word in self.relco_model.vocab:
                    raw_word = self.relco_model[word]
                    obj_vecs_relco += (raw_word / la.norm(raw_word))
                else:
                    obj_vecs_relco += \
                        (self.relco_vec_mean / la.norm(self.relco_vec_mean))
            obj_vecs_relco /= len(words)
            obj_vecs_relco /= la.norm(obj_vecs_relco)

            self.obj_vecs[ix][:half_dim] = obj_vecs_wiki
            self.obj_vecs[ix][half_dim:] = obj_vecs_relco

        self.prd_vecs = np.zeros((len(self._predicate_categories),
                                 cfg.INPUT_LANG_EMBEDDING_DIM), dtype=np.float32)
        for ix, name in enumerate(_predicate_categories):
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

            prd_vecs_relco = np.zeros(half_dim, dtype=np.float32)
            words = name.split()
            for word in words:
                if word in self.relco_model.vocab:
                    raw_word = self.relco_model[word]
                    prd_vecs_relco += (raw_word / la.norm(raw_word))
                else:
                    prd_vecs_relco += \
                        (self.relco_vec_mean / la.norm(self.relco_vec_mean))
            prd_vecs_relco /= len(words)
            prd_vecs_relco /= la.norm(prd_vecs_relco)

            self.prd_vecs[ix][:half_dim] = prd_vecs_wiki
            self.prd_vecs[ix][half_dim:] = prd_vecs_relco

        landb = dict(obj_vecs=self.obj_vecs, prd_vecs=self.prd_vecs)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(landb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
