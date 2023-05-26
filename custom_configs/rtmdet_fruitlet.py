_base_ = '../configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py'

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


load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_s_8xb32-300e_coco/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth'  # noqa
# Set up working dir to save files and logs.
work_dir = './output/fruitlets/rtmdet_ins_s_coco'


vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'mmdetection',
            'group': 'rtmdet_ins_s_coco'
         })
]

visualizer = dict(vis_backends=vis_backends)

model = dict(bbox_head=dict(num_classes=1))

train_cfg = dict(max_epochs=300)