_base_ = '../configs/misc/d2_mask-rcnn_r50-caffe_fpn_ms-90k_coco.py'

# Modify dataset classes and color
metainfo = {
    'classes': ('fruitlet'),
    'palette': [
        (220, 20, 60), 
    ]
}
data = dict(
    train=dict(
        classes=('fruitlet')),
    val=dict(
        classes=('fruitlet')),
    test=dict(
        classes=('fruitlet')))

# Modify dataset type and path
data_root = './data/Fruitlets'
train_dataloader = dict(dataset=dict(
    ann_file = 'train/ann/train.json',
    data_root = data_root,
    data_prefix = dict(img = 'train/img/'),
    metainfo = metainfo
))
val_dataloader = dict(dataset=dict(
    ann_file = 'val/ann/val.json',
    data_root = data_root,
    data_prefix = dict(img = 'val/img/'),
    metainfo = metainfo
))

test_dataloader = val_dataloader

# Modify metric config
val_evaluator = dict(ann_file = data_root+'/'+'val/ann/val.json')
test_evaluator = val_evaluator

# Modify num classes of the model in box head and mask head
model = dict(detector=dict(roi_heads=dict(num_classes=1)))

# We can still the pre-trained Mask RCNN model to obtain a higher performance
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'  # noqa

# Set up working dir to save files and logs.
work_dir = './output/de_mask_rcnn_fruitlets'


vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'mmdetection',
            'group': 'd2_mask-rcnn_r50-caffe_fpn_ms-90k_coco'
         })
]

visualizer = dict(vis_backends=vis_backends)

model = dict(
    detector=dict(
        roi_heads=dict(num_classes=1),
        weights='detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl'))

# We can set the evaluation interval to reduce the evaluation times
train_val_interval = 3
# We can set the checkpoint saving interval to reduce the storage cost
default_hooks = dict(checkpoint = dict(interval = 3))

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
optim_wrapper = dict(optimizer = dict(lr = 0.02 / 8))
default_hooks = dict(logger = dict(interval = 10))

train_cfg = dict(max_iters=5000, val_interval=50)