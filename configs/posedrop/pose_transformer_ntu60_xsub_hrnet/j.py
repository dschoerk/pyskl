clip_len = 100

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='PoseTransformerBackbone',
        num_keypoints=17,
        in_channels=12,              # 4 feats (j, b, jm, bm) × 3 channels (x, y, score)
        num_frames=clip_len,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        strategy='zero_fill',        # options: zero_fill, interpolation, token_drop, mask_token
        use_factored_attention=True,
        temporal_pool_every=0),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=128))

dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_hrnet.pkl'
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j', 'b', 'jm', 'bm']),
    dict(type='UniformSample', clip_len=clip_len),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j', 'b', 'jm', 'bm']),
    dict(type='UniformSample', clip_len=clip_len, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j', 'b', 'jm', 'bm']),
    dict(type='UniformSample', clip_len=clip_len, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xsub_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xsub_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xsub_val'))

optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
    by_epoch=False,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5,
    warmup_ratio=1e-3)
total_epochs = 24
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
# To enable wandb logging, either pass --wandb to train.py, or uncomment below:
# log_config = dict(interval=100, hooks=[
#     dict(type='TextLoggerHook'),
#     dict(type='WandbLoggerHook',
#          init_kwargs=dict(project='pyskl', name='pose_transformer_ntu60_xsub_j'),
#          define_metric_cfg={'top1_acc': 'max', 'top5_acc': 'max'}),
# ])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

log_level = 'INFO'
work_dir = './work_dirs/posedrop/pose_transformer_ntu60_xsub_hrnet/j'
