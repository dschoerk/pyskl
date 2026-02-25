clip_len = 32

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='SkeletonTransformer',
        graph_cfg=dict(layout='nturgb+d', mode='spatial'),
        in_channels=3,
        temporal_patch_size=4,
        embed_dim=64,
        head_dim=16,  # num_heads = dim // head_dim, scales with channel expansion (4->8->16)
        depth=8,
        # Temporal pyramid: halve T and double channels at layers 2 and 4
        # T=32,D=64,h=4 -> [pool] T=16,D=128,h=8 -> [pool] T=8,D=256,h=16
        down_stages=[2, 4, 6],
        expand_stages=[2, 4],
        dropout=0.1,
        drop_path=0.15,
        use_graph_bias=True,
        use_cross_person=False),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256,
                  loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0, label_smoothing=0.1)))

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
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=clip_len, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=clip_len, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=128,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xview_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xview_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xview_val'))

paramwise_cfg = dict(custom_keys={
    'norm': dict(decay_mult=0.0),
    'bias': dict(decay_mult=0.0),
})

# optimizer
optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0.05, paramwise_cfg=paramwise_cfg)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
# learning policy - linear warmup then cosine decay
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.1)
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
# runtime settings

log_level = 'INFO'
work_dir = './work_dirs/skel_transformer/skel_transformer_ntu60_xview_3dkp/j'
load_from = 'work_dirs/skel_transformer_mae/pretrain/latest.pth'
