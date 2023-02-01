# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES

# @register_handler('npy')
# class NpyHandler(BaseFileHandler):
#     str_like = False

#     def load_from_fileobj(self, file, **kwargs):
#         return np.load(file)

#     # Mainly the default ones are providedrb模式
#     def load_from_path(self, filepath, **kwargs):
#         return super(NpyHandler, self).load_from_path(
#             filepath, mode='rb', **kwargs)

#     def dump_to_fileobj(self, obj, file, **kwargs):
#         np.save(file, obj)

#     # Mainly the default ones are providedwb模式
#     def dump_to_path(self, obj, filepath, **kwargs):
#         super(NpyHandler, self).dump_to_path(
#             obj, filepath, mode='wb', **kwargs)

#     def dump_to_str(self, obj, **kwargs):
#         return obj.tobytes()


@PIPELINES.register_module()
class LoadMultiResolutionImageFromFile(object):
    """Load an image from file. Required keys are "img_prefix" and "img_info" (a dict that must contain the key "filename"). Added or updated keys are "filename", "img", "img_shape", "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`), "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1). Args: to_float32 (bool): Whether to convert the loaded image to a float32 numpy array. If set to False, the loaded image is an uint8 array. Defaults to False. color_type (str): The flag argument for :func:`mmcv.imfrombytes`. Defaults to 'color'. file_client_args (dict): Arguments to instantiate a FileClient. See :class:`mmcv.fileio.FileClient` for details. Defaults to ``dict(backend='disk')``. imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default: 'cv2' """

    def __init__(self,
                 to_float32=False,
                 file_client_args=dict(backend='disk'),):
        self.to_float32 = to_float32
        self.file_client_args = file_client_args.copy()
        self.buffer = {
    }

    def __call__(self, results):
        """Call functions to load image and get image meta information. Args: results (dict): Result dict from :obj:`mmseg.CustomDataset`. Returns: dict: The dict contains loaded image and meta information. """

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        # 缓存
        if self.file_client_args['backend'] == 'mem':
            if filename not in self.buffer:
                self.buffer[filename] = mmcv.load(filename)
            img = self.buffer[filename].copy()
        else:
            img = mmcv.load(filename)

        if self.to_float32:
            img = img.astype(np.float32)
        # replace nan with 0
        img[np.isnan(img)] = 0

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        return repr_str


@PIPELINES.register_module()
class LoadMultiResolutionAnnotations(object):
    """Load annotations for semantic segmentation. Args: reduce_zero_label (bool): Whether reduce all label value by 1. Usually used for datasets where 0 is background label. Default: False. file_client_args (dict): Arguments to instantiate a FileClient. See :class:`mmcv.fileio.FileClient` for details. Defaults to ``dict(backend='disk')``. imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default: 'pillow' """

    def __init__(self,
                 map_labels=None,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow',):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

        self.map_labels = map_labels
        self.use_men_buffer = False
        if self.file_client_args['backend'] == 'mem':
            self.use_men_buffer = True
            self.buffer = {
    }
            self.file_client_args['backend'] = 'disk'
        if self.map_labels:
            # avoid using underflow conversion
            self.valid_label = [k for k, v in self.map_labels.items() if len(v) == 1]
            self.invalid_label = [k for k in self.map_labels.keys() if k not in self.valid_label]

    def __call__(self, results):
        """Call function to load multiple types annotations. Args: results (dict): Result dict from :obj:`mmseg.CustomDataset`. Returns: dict: The dict contains loaded semantic segmentation annotations. """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']

        # 缓存
        if self.use_men_buffer:
            if filename not in self.buffer:
                img_bytes = self.file_client.get(filename)
                self.buffer[filename] = mmcv.imfrombytes(
                    img_bytes, flag='unchanged',
                    backend=self.imdecode_backend).squeeze().astype(np.uint8)
            gt_semantic_seg = self.buffer[filename].copy()
        else:
            img_bytes = self.file_client.get(filename)
            gt_semantic_seg = mmcv.imfrombytes(
                img_bytes, flag='unchanged',
                backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        # reduce zero_label
        if self.map_labels:
            # avoid using underflow conversion
            for i in self.invalid_label:
                gt_semantic_seg[gt_semantic_seg == i] = 255
            for i, j in enumerate(self.valid_label):
                gt_semantic_seg[gt_semantic_seg == j] = i
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str