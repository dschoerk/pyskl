clip_len = 32

model = dict(
    type='SkeletonMAE',
    backbone=dict(
        type='SkeletonTransformer',
        graph_cfg=dict(layout='nturgb+d', mode='spatial'),
        in_channels=3,
        embed_dim=64,
        head_dim=16,
        depth=8,
        # Same temporal pyramid as supervised — all weights transfer exactly
        # T=32,D=64 -> [pool] T=16,D=128 -> [pool] T=8,D=256 -> [pool] T=4
        down_stages=[2, 4, 6],
        expand_stages=[2, 4],
        dropout=0.1,
        drop_path=0.15,
        use_graph_bias=True,
        use_cross_person=False),
    decoder_head=dict(
        type='MAEHead',
        encoder_dim=256,     # backbone output dim after 2x expansion
        decoder_dim=128,
        decoder_depth=2,
        decoder_heads=4,
        in_channels=3,
        target_T=clip_len,   # upsample from T=4 back to T=32
        use_graph_bias=True,
        graph_cfg=dict(layout='nturgb+d', mode='spatial')),
    mask_ratio=0.75,
    norm_target=True)

dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_3danno.pkl'
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='RandomScale', scale=0.1),
    dict(type='RandomRot'),
    dict(type='RandomShear'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=clip_len),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file,
                     pipeline=train_pipeline, split='xview_train')))

paramwise_cfg = dict(custom_keys={
    'norm': dict(decay_mult=0.0),
    'bias': dict(decay_mult=0.0),
})

# optimizer
optimizer = dict(type='AdamW', lr=1.5e-3, weight_decay=0.05, paramwise_cfg=paramwise_cfg)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.1)
total_epochs = 100
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/skel_transformer_mae/pretrain'
