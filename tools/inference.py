

import os
import argparse
import mmcv
import numpy as np

from mmdet.apis import init_detector, inference_detector
from mmengine import Config
from mmengine.visualization import Visualizer

from mmdet.registry import VISUALIZERS
# init visualizer(run the block only once in jupyter notebook)


parser = argparse.ArgumentParser(description='Inference images')
parser.add_argument('config', metavar='config_path', type=str,
                    help='Path to model configuration.')
parser.add_argument('checkpoint', metavar='checkpoint_path', type=str,
                    help='Path to model checkpoint file.')
parser.add_argument('images', metavar='dir_images', type=str,
                    help='Path to the directory for original images.')
parser.add_argument('output', metavar='output_path', type=str,
                    help='Path to the directory for the inference result images.')


args = parser.parse_args()

def main(args):
    if not os.path.exists(args.config):
        print(f'{args.config} does not exist. Provide a valid path to config file')
        return
        
    if not os.path.exists(args.checkpoint):
        print(f'{args.checkpoint} does not exist. Provide a valid path to checkpoint file')
        return
    
    if not os.path.exists(args.images):
        print(f'{args.configimages} does not exist. Provide a valid path to the directory for images')
        return
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    cfg = Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint, device='cuda')

    model.cfg.visualizer.save_dir = args.output

    # get built visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # visualizer_now = Visualizer.get_current_instance()
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta
    

    for path in os.listdir(args.images):        
        p = os.path.join(args.images, path)
        if os.path.isfile(p):
            img = mmcv.imread(p,channel_order='rgb')
            result = inference_detector(model, img)
            # print(result)
            # show the results
            visualizer.add_datasample(
                f'{path}_results',
                img,
                data_sample=result,
                draw_gt=False,
                wait_time=0,
                out_file=None,
                pred_score_thr=0.5
            )

if __name__ == "__main__":
    main(args)