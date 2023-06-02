from mmengine.runner import Runner
from mmengine import Config

cfg = Config.fromfile('./custom_configs/mask2former_r101_coco-panoptic_vine.py')
# cfg.visualizer.vis_backends.append({"type":'TensorboardVisBackend'})

# build the runner from config
runner = Runner.from_cfg(cfg)
# start training
runner.train()