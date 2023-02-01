_base_ = [
    '../_base_/models/segformer_mit-b0_2.py',
    '../_base_/datasets/burned.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]


norm_cfg = dict(type='BN', requires_grad=True)

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa


model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        in_channels=4,
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        norm_cfg=norm_cfg,
        # loss_decode=dict(
        #     type='ImbalancedCrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, sigma=1)

    )
)

img_norm_cfg = dict(
    # mean=[123.675, 116.28, 103.53, 103.53], std=[58.395, 57.12, 57.375, 57.375], to_rgb=False)
    mean=[0.041, 0.035, 0.023, 0.126], std=[0.019, 0.013, 0.01, 0.056], to_rgb=False)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='npy'),
    dict(type='LoadAnnotations'),
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
    workers_per_gpu=2,
    train=dict(
        pipeline=train_pipeline,
        ann_dir='pseudolabels_train',
        ),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)
    )


# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# data = dict(samples_per_gpu=8, workers_per_gpu=2)