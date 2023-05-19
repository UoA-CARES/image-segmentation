# from mmengine import Config
# from mmengine.runner import set_random_seed

# cfg = Config.fromfile('configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py')

# # Modify dataset classes and color
# cfg.metainfo = {
#     'classes': ('fruitlet'),
#     'palette': [
#         (220, 20, 60), 
#     ]
# }

# # Modify dataset type and path
# cfg.data_root = '../data/Fruitlets'
# cfg.train_dataloader.dataset.ann_file = 'train/ann/train.json'
# cfg.train_dataloader.dataset.data_root = cfg.data_root
# cfg.train_dataloader.dataset.data_prefix.img = 'train/img/'
# cfg.train_dataloader.dataset.metainfo = cfg.metainfo

# cfg.val_dataloader.dataset.ann_file = 'val/ann/val.json'
# cfg.val_dataloader.dataset.data_root = cfg.data_root
# cfg.val_dataloader.dataset.data_prefix.img = 'val/img/'
# cfg.val_dataloader.dataset.metainfo = cfg.metainfo

# cfg.test_dataloader = cfg.val_dataloader

# # Modify metric config
# cfg.val_evaluator.ann_file = cfg.data_root+'/'+'val/ann/val.json'
# cfg.test_evaluator = cfg.val_evaluator

# # Modify num classes of the model in box head and mask head
# cfg.model.roi_head.bbox_head.num_classes = 1
# cfg.model.roi_head.mask_head.num_classes = 1

# # We can still the pre-trained Mask RCNN model to obtain a higher performance
# cfg.load_from = '../checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# # Set up working dir to save files and logs.
# cfg.work_dir = '../output/fruitlets'


# # We can set the evaluation interval to reduce the evaluation times
# cfg.train_cfg.val_interval = 3
# # We can set the checkpoint saving interval to reduce the storage cost
# cfg.default_hooks.checkpoint.interval = 3

# # The original learning rate (LR) is set for 8-GPU training.
# # We divide it by 8 since we only use one GPU.
# cfg.optim_wrapper.optimizer.lr = 0.02 / 8
# cfg.default_hooks.logger.interval = 10


# # Set seed thus the results are more reproducible
# # cfg.seed = 0
# set_random_seed(0, deterministic=False)

# # We can also use tensorboard to log the training process
# cfg.visualizer.vis_backends.append({"type":'TensorboardVisBackend'})

_base_ = '../configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py'

# Modify dataset classes and color
metainfo = {
    'classes': ('fruitlet'),
    'palette': [
        (220, 20, 60), 
    ]
}

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
# train_dataloader.dataset.ann_file = 'train/ann/train.json'
# train_dataloader.dataset.data_root = data_root
# train_dataloader.dataset.data_prefix.img = 'train/img/'
# train_dataloader.dataset.metainfo = metainfo

# val_dataloader.dataset.ann_file = 'val/ann/val.json'
# val_dataloader.dataset.data_root = data_root
# val_dataloader.dataset.data_prefix.img = 'val/img/'
# val_dataloader.dataset.metainfo = metainfo

test_dataloader = val_dataloader

# Modify metric config
val_evaluator = dict(ann_file = data_root+'/'+'val/ann/val.json')
test_evaluator = val_evaluator

# Modify num classes of the model in box head and mask head
model = dict(roi_head = dict(bbox_head = dict(num_classes = 1), mask_head = dict(num_classes = 1)))

# We can still the pre-trained Mask RCNN model to obtain a higher performance
# load_from = '../checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'  # noqa
# Set up working dir to save files and logs.
work_dir = './output/fruitlets'


# We can set the evaluation interval to reduce the evaluation times
train_val_interval = 3
# We can set the checkpoint saving interval to reduce the storage cost
default_hooks = dict(checkpoint = dict(interval = 3))

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
optim_wrapper = dict(optimizer = dict(lr = 0.02 / 8))
default_hooks = dict(logger = dict(interval = 10))

train_cfg = dict(max_epochs=2)