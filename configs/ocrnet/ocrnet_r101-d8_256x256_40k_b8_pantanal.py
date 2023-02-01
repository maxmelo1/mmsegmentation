_base_ = [
    '../_base_/models/ocrnet_r50-d8_2.py', '../_base_/datasets/pantanal2.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
	pretrained='open-mmlab://resnet101_v1c', 
	backbone=dict(depth=101,
		norm_cfg=norm_cfg
	)
)

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