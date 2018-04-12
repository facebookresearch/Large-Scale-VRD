"""Factory method for easily getting imdbs by name."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from datasets.visual_genome_spo_jn_mg import visual_genome_spo_jn_mg
from datasets.visual_genome_spo_jn_mg_lan import visual_genome_spo_jn_mg_lan

from datasets.vrd_wiki_and_node2vec import vrd_wiki_and_node2vec
from datasets.vrd_wiki_and_node2vec_lan import vrd_wiki_and_node2vec_lan

__sets = {}
__sets_lan = {}

for split in ['train', 'val', 'test']:
    name = 'visual_genome_spo_jn_mg_{}'.format(split)
    __sets[name] = (lambda split=split: visual_genome_spo_jn_mg(split))

name = 'visual_genome_spo_jn_mg_lan'
__sets_lan[name] = (lambda: visual_genome_spo_jn_mg_lan())

for split in ['train', 'val', 'test']:
    name = 'vrd_wiki_and_node2vec_{}'.format(split)
    __sets[name] = (lambda split=split: vrd_wiki_and_node2vec(split))

name = 'vrd_wiki_and_node2vec_lan'
__sets_lan[name] = (lambda: vrd_wiki_and_node2vec_lan())


# By Ji on 07/06/2017
def get_landb(name):
    if name not in __sets_lan.keys():
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets_lan[name]()


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets.keys():
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()
