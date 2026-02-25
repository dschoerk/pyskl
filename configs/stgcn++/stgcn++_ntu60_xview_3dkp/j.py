
clip_len = 32

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        gcn_type='unit_transformer',
        num_stages=5,
        inflate_stages=[1, 3],
        down_stages=[1, 3],
        #base_channels=32,
        pool_stages=[2],

        #gcn_norm=dict(type='GN', num_groups=4),
        #tcn_norm=dict(type='GN', num_groups=4),
        gcn_use_graph_bias=True,
        #tcn_type='tcn_transformer',
        #tcn_qkv_kernel=9,
        #tcn_ffn_ratio=0,
        gcn_ffn_ratio=0,
        #tcn_kernel_size=9,
        # base_channels=128,
        #tcn_type='mstcn',
        tcn_type='unit_tcn',
        in_channels=3,  # 4 feats (j, b, jm, bm) × 3 channels (x, y, score)
        graph_cfg=dict(layout='nturgb+d', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=128))

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
    persistent_workers=False,
    test_dataloader=dict(videos_per_gpu=1),

    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xview_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xview_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xview_val'))

checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

paramwise_cfg=dict(custom_keys={
    'norm': dict(decay_mult=0.0),
    'bias': dict(decay_mult=0.0),
})

# optimizer
if True:
    # fine-tune
    # load_from = 'work_dirs\stgcn++\stgcn++_ntu60_xview_3dkp\j\epoch_16.pth'
    # optimizer = dict(
    #     type='SGD',
    #     lr=0.1,
    #     momentum=0.9,
    #     weight_decay=0.0001,
    #     nesterov=True,
    #     paramwise_cfg=paramwise_cfg
    #     )
    
    # optimizer_config = dict(grad_clip=dict(max_norm=10.0))
    # lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False,
    #                  warmup='linear', warmup_iters=100, warmup_ratio=0.1)

    optimizer = dict(type='AdamW', lr=5e-3, weight_decay=0.005, paramwise_cfg=paramwise_cfg)
    optimizer_config = dict(grad_clip=dict(max_norm=1.0))
    # learning policy
    lr_config = dict(policy='CosineAnnealing', min_lr=5e-5, by_epoch=False, #min_lr -> 5e-5
                     warmup='linear', warmup_iters=1500, warmup_ratio=0.1)
    
    total_epochs = 16
    #data['videos_per_gpu'] = 128

else:
    # pre-train
    optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0.0005)
    optimizer_config = dict(grad_clip=dict(max_norm=1.0))
    # learning policy
    lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False,
                     warmup='linear', warmup_iters=500, warmup_ratio=0.1)
    total_epochs = 12



# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/stgcn++/stgcn++_ntu60_xview_3dkp/j'

cudnn_benchmark = True



#'http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xview_3dkp/j.pth'