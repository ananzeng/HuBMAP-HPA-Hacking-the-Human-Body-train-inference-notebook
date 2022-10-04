norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrain/beit_large_patch16_224_pt22k_ft22k.pth',
    backbone=dict(
        type='BEiT',
        img_size=(640, 640),
        patch_size=16,
        in_channels=3,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        out_indices=[7, 11, 15, 23],
        qv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_cfg=dict(type='LN', eps=1e-06),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        init_values=1e-06),
    neck=dict(type='Feature2Pyramid', embed_dim=1024, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=1024,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426)))
dataset_type = 'BinaryDataset'
data_root = 'HuBMAP_dataset'
img_norm_cfg = dict(
    mean=[208.82, 204.3, 210.8], std=[41.65, 45.95, 39.92], to_rgb=True)
crop_size = (640, 640)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2560, 640), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(640, 640), cat_max_ratio=0.75),
    dict(type='RandomRotate', prob=0.5, degree=20.0),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomCutOut',
        prob=0.5,
        n_holes=5,
        fill_in=(255, 255, 255),
        cutout_ratio=[(0.2, 0.2), (0.1, 0.1)]),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[208.82, 204.3, 210.8],
        std=[41.65, 45.95, 39.92],
        to_rgb=True),
    dict(type='Pad', size=(640, 640), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2560, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[208.82, 204.3, 210.8],
                std=[41.65, 45.95, 39.92],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type='BinaryDataset',
        data_root='HuBMAP_dataset',
        img_dir='images',
        ann_dir='labels',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(2560, 640), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(640, 640), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='RandomRotate', prob=0.5, degree=20.0),
            dict(
                type='RandomCutOut',
                prob=0.5,
                n_holes=5,
                fill_in=(255, 255, 255),
                cutout_ratio=[(0.2, 0.2), (0.1, 0.1)]),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[208.82, 204.3, 210.8],
                std=[41.65, 45.95, 39.92],
                to_rgb=True),
            dict(type='Pad', size=(640, 640), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        split='splits/train.txt'),
    val=dict(
        type='BinaryDataset',
        data_root='HuBMAP_dataset',
        img_dir='images',
        ann_dir='labels',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2560, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[208.82, 204.3, 210.8],
                        std=[41.65, 45.95, 39.92],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split='splits/val.txt'),
    test=dict(
        type='BinaryDataset',
        data_root='HuBMAP_dataset',
        img_dir='images',
        ann_dir='labels',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2560, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[208.82, 204.3, 210.8],
                        std=[41.65, 45.95, 39.92],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split='splits/val.txt'))
log_config = dict(
    interval=200, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = './work_dirs_HuBMAP/upernet_beit-large_fp16_8x1_640x640_160k_ade20k/iter_24000.pth'
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=4e-05,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.95))
optimizer_config = dict(
    type='GradientCumulativeFp16OptimizerHook', cumulative_iters=2)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=45000)
checkpoint_config = dict(by_epoch=False, interval=1500)
evaluation = dict(interval=1500, metric='mDice')
fp16 = dict()
work_dir = './work_dirs_HuBMAP/upernet_beit-large_fp16_8x1_640x640_160k_ade20k'
gpu_ids = [0]
auto_resume = False
