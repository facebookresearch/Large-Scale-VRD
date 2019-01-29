# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Factory method for easily getting imdbs by name."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from datasets.vg_wiki_and_relco import vg_wiki_and_relco
from datasets.vg_wiki_and_relco_lan import vg_wiki_and_relco_lan


__sets = {}
__sets_lan = {}

for split in ['train', 'val', 'test']:
    name = 'vg_wiki_and_relco_{}'.format(split)
    __sets[name] = (lambda split=split: vg_wiki_and_relco(split))

name = 'vg_wiki_and_relco_lan'
__sets_lan[name] = (lambda: vg_wiki_and_relco_lan())


def get_landb(name):
    if name not in __sets_lan.keys():
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets_lan[name]()


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets.keys():
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()
