# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PantanalDataset(CustomDataset):
    """Pascal VOC based dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('background', 'tree')

    PALETTE = [[0, 0, 0], [128, 0, 0]]

    def __init__(self, split, **kwargs):
        super(PantanalDataset, self).__init__(
            img_suffix='.npy', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
