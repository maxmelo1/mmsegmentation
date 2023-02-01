_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_2.py',
    '../_base_/datasets/rios_corrigido.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://resnet101_v1c', 
    backbone=dict(
        depth=101,
        in_channels=3,
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        norm_cfg=norm_cfg,
        # loss_decode=
        # [
        #     dict(type='CrossEntropyLoss', use_sigmoid=True, loss_name='loss_ce', loss_weight=1.0),
        #     dict(type='DiceLoss', use_sigmoid=True, loss_name='loss_dice', loss_weight=3.0)
        # ]
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                class_weight=[0.58711665, 3.36971543]
        ),
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)
    ),
)


img_norm_cfg = dict(
    # mean=[123.675, 116.28, 103.53, 103.53], std=[58.395, 57.12, 57.375, 57.375], to_rgb=False)
    mean=[132.077, 131.69,  113.286], std=[42.312, 38.674, 40.324], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 3.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
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
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))



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

data = dict(samples_per_gpu=2, workers_per_gpu=2)