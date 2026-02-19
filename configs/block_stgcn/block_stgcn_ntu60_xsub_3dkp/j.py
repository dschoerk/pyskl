# Block-wise STGCN++ + inter-block transformer, NTU60 xsub, joint modality.
#
# The backbone splits the 100-frame clip into non-overlapping 8-frame blocks
# (→ 13 blocks after zero-padding to 104 frames), runs STGCN++ stages on every
# block in parallel, mean-pools each block to a single 256-d vector, then lets
# a 2-layer transformer attend across the 13 block tokens before classification.

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='BlockSTGCN',
        # Temporal blocking
        block_size=8,
        # STGCN++ settings (identical to standard STGCN++ j config)
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='unit_tcn',
        graph_cfg=dict(layout='nturgb+d', mode='spatial'),
        num_stages =10,
        # in_channels=3, base_channels=64, ch_ratio=2, num_stages=10
        # inflate_stages=[5,8], down_stages=[5,8]  ← defaults kept
        # Inter-block transformer
        num_heads=8,               # 8 heads × 32 = 256 d_model
        num_transformer_layers=0,
        transformer_dropout=0.1,
        transformer_drop_path=0.1,
        max_blocks=64,
        use_sic=True,              # Statistical Invariance Coding (BlockGCN, CVPR 2024)
        #pretrained='work_dirs/stgcn++/stgcn++_ntu60_xview_3dkp/j/latest.pth'
        pretrained=None
        ),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_3danno.pkl'
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=128,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file,
                     pipeline=train_pipeline, split='xsub_train')),
    val=dict(type=dataset_type, ann_file=ann_file,
             pipeline=val_pipeline, split='xsub_val'),
    test=dict(type=dataset_type, ann_file=ann_file,
              pipeline=test_pipeline, split='xsub_val'))

# optimizer
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9,
#                  weight_decay=0.0005, nesterov=True)
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)

# optimizer - AdamW with lower lr works better for transformers
optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
# learning policy - linear warmup then cosine decay
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
    by_epoch=False,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=3,
    warmup_ratio=1e-3)


total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/block_stgcn/block_stgcn_ntu60_xsub_3dkp/j'
