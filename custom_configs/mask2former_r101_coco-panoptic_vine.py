_base_ = '../configs/mask2former/mask2former_r101_8xb2-lsj-50e_coco-panoptic.py'

metainfo = {
    'classes': ('Node', 'Shoot', 'Trunk', 'Wire', 'Post'),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), 
    ]
}

data = dict(
    train=dict(
        classes=('Node', 'Shoot', 'Trunk', 'Wire', 'Post')),
    val=dict(
        classes=('Node', 'Shoot', 'Trunk', 'Wire', 'Post')),
    test=dict(
        classes=('Node', 'Shoot', 'Trunk', 'Wire', 'Post')))

# Modify dataset type and path
data_root = './data/panoptic/TrainingData'
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='train/train_panoptic.json',
        data_prefix=dict(
            img='train/images/', seg='train/panopticMask/'),
        metainfo = metainfo))


val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='val/val_panoptic.json',
        data_prefix=dict(
            img='val/images/', seg='val/panopticMask/'),
        metainfo = metainfo))


test_dataloader = val_dataloader

# Modify metric config
val_evaluator = [
    dict(
        type='CocoPanopticMetric',
        ann_file=data_root + '/val/val_panoptic.json',
        seg_prefix=data_root + '/val/panopticMask/',
        backend_args={{_base_.backend_args}}),
    # dict(
    #     type='CocoMetric',
    #     ann_file=data_root + '/val/val_Instance.json',
    #     metric=['bbox', 'segm'],
    #     backend_args={{_base_.backend_args}})
]
test_evaluator = val_evaluator

# Modify num classes of the model in box head and mask head
# model = dict(panoptic_head = dict(num_things_classes = 1, panoptic_fusion_head = dict(num_classes = 1)))
# things : node
# stuff : wire, shoot, trunk, post 
num_things_classes = 1
num_stuff_classes = 4
num_classes = num_things_classes + num_stuff_classes

model = dict(
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes))

auto_scale_lr = dict(base_batch_size=2)

# We can still the pre-trained Mask RCNN model to obtain a higher performance
# load_from = '../checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_r101_8xb2-lsj-50e_coco-panoptic/mask2former_r101_8xb2-lsj-50e_coco-panoptic_20220329_225104-c74d4d71.pth'  # noqa
# Set up working dir to save files and logs.
work_dir = './output/vine/mask2former_r101_coco-panoptic'


vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'mmdetection',
            'group': 'mask2former_r101-coco-panoptic_vine'
         })
]

visualizer = dict(vis_backends=vis_backends)

# We can set the evaluation interval to reduce the evaluation times
# train_val_interval = 3
# We can set the checkpoint saving interval to reduce the storage cost
# default_hooks = dict(checkpoint = dict(interval = 3))

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
# optim_wrapper = dict(optimizer = dict(lr = 0.02 / 8))
# default_hooks = dict(logger = dict(interval = 10))

train_cfg = dict(max_iters=20000, val_interval=50)