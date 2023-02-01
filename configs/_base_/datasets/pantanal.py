# dataset settings
dataset_type = 'PantanalDataset'
data_root = 'data/pantanal'
img_norm_cfg = dict(
    # mean=[123.675, 116.28, 103.53, 103.53], std=[58.395, 57.12, 57.375, 57.375], to_rgb=False)
    mean=[5.903, 5.706, 3.433, 30.287], std=[5.744, 4.138, 2.721, 18.475], to_rgb=False)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='npy'),
    dict(type='LoadAnnotations'),
    # dict(type='LoadMultiResolutionImageFromFile', file_client_args=dict(backend='mem')),
    # dict(type='LoadMultiResolutionAnnotations', map_labels=map_labels, file_client_args=dict(backend='mem')),
    dict(type='Resize', img_scale=(256, 256), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='npy'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',
        split='ImageSets/Segmentation/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',
        split='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',
        split='ImageSets/Segmentation/test.txt',
        pipeline=test_pipeline))
