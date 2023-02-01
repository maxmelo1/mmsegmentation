# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RiosDataset(CustomDataset):
    """Pascal VOC based dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('background', 'river')

    PALETTE = [[0, 0, 0], [128, 0, 0]]

    def __init__(self, split, **kwargs):
        super(RiosDataset, self).__init__(
            img_suffix='.tif', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
