import detectron2

# import some common libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, default_argument_parser, hooks, launch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import random
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader, DatasetMapper
from detectron2.data.datasets import register_coco_panoptic_separated
from detectron2.evaluation import COCOPanopticEvaluator, inference_on_dataset
import detectron2.data.transforms as T

import json
import sys
import os
import torch

def Load_json(path):
    #REgister data-set
    for DSet in ["train", "val","eval"]:
        #! https://detectron2.readthedocs.io/en/latest/modules/data.html?highlight=panoptic#detectron2.data.datasets.register_coco_panoptic_separated
        #? panopticMask includes both Instance and Semantic masks
        #? SemanticMask includes only semantic segments masks
        register_coco_panoptic_separated(f"dataset_{DSet}", {}, f"{path}/{DSet}/images",
                                            f"{path}/{DSet}/panopticMask", f"{path}/{DSet}/{DSet}_panoptic.json",
                                            f"{path}/{DSet}/semanticMask", f"{path}/{DSet}/{DSet}_Instance.json")


    #Read categories
    json_file = f'{path}/panoptic_coco_categories.json'
    with open(json_file) as json_file: # A json file describing all the categories (including the background) according to the
        categories = json.load(json_file) #coco panoptic guidelines

    ting_names =  [f["name"] for f in categories if f["isthing"] == 1] #load nodes
    ting_ids = [f["id"] for f in categories if f["isthing"] == 1]
    ting_dataset_id_to_contiguous_id = dict(zip(ting_ids,list(range(0,(len(ting_ids))))))

    stuff_names =  [f["name"] for f in categories if f["isthing"] == 0]
    stuff_names.insert(0,'All_things')
    stuff_ids = [f["id"] for f in categories if f["isthing"] == 0]
    
    stuff_dataset_id_to_contiguous_id = dict(zip(stuff_ids,list(range(len(ting_names),(len(stuff_ids) + len(ting_names))))))
    
    #REgister data-set
    dataset_dicts = DatasetCatalog.get("dataset_train_separated")                                    
    dataset_metadata = MetadataCatalog.get("dataset_train_separated").set( thing_classes=ting_names, stuff_classes=stuff_names, stuff_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id)
    MetadataCatalog.get("dataset_eval_separated").set( thing_classes=ting_names, thing_dataset_id_to_contiguous_id = ting_dataset_id_to_contiguous_id, stuff_classes=stuff_names, stuff_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id)

    return dataset_dicts,dataset_metadata


def init_cfg(path , dataset_metadata):

    cfg = get_cfg()
    # config_file = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
    config_file = "COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml"

    cfg.merge_from_file(model_zoo.get_config_file(config_file))

    cfg.DATASETS.TRAIN = ("dataset_train_separated",)
    #cfg.DATASETS.TEST = ("dataset_val_separated", ) 
    cfg.DATASETS.TEST = ()

    # Number of data loading threads
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)

    # Number of images per batch across all machines.
    #cfg.SOLVER.STEPS = (50, 80)
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.0025     # pick a good LearningRate
    # cfg.SOLVER.MAX_ITER = 100    #No. of iterations 20000
    #cfg.TEST.EVAL_PERIOD = 2000 # No. of iterations after which the Validation Set is evaluated. 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(dataset_metadata.get("thing_classes"))# No. of Instance classes 
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(dataset_metadata.get("stuff_classes"))  #No. of Segment classes 
    cfg.OUTPUT_DIR = f'{path}/WeightOutput'
    cfg.INPUT.MIN_SIZE_TRAIN = (1536, 1568, 1600, 1632, 1664)
    cfg.INPUT.MAX_SIZE_TRAIN = 4000
    cfg.INPUT.MIN_SIZE_TEST = 1664
    cfg.INPUT.MAX_SIZE_TEST = 4000

    #cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.DEVICE = 'cuda:1'
    #CUDA_LAUNCH_BLOCKING=4
    
    return cfg



class MyTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        #Default shortest edges 640, 672, 704, 736, 768, 800
        #Shortest edges used 1536 , 1568 ,1600, 1632, 1664
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[T.ResizeShortestEdge(short_edge_length=(1536 , 1568 ,1600, 1632, 1664), max_size=4000, sample_style='choice'),
                                                                    T.RandomBrightness(0.75, 1.25),
                                                                    T.RandomContrast(0.75, 1.25),
                                                                    T.RandomSaturation(0.8, 1.2),
                                                                    T.RandomFlip(prob=0.4, horizontal=True, vertical=False)])
        return build_detection_train_loader(cfg, mapper=mapper)

    # transform_list = [
    #     T.Resize((1920,1200)),
    #     T.RandomBrightness(0.8, 1.8),
    #     T.RandomContrast(0.6, 1.3),
    #     T.RandomSaturation(0.8, 1.4),
    #     #T.RandomRotation(angle=[90, 90]),
    #     T.RandomLighting(0.7),
    #     T.RandomFlip(prob=0.4, horizontal=True, vertical=False),
    # ] #! Augmentation list  https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.Augmentation



def lunch_training(resume = False):
    
    path = sys.argv[1:][0] #'./data/completeSegment_NoEdit'
    dataset_dicts,dataset_metadata = Load_json(path)
    cfg = init_cfg(path , dataset_metadata)

    #Save Config
    # f = open(cfg.OUTPUT_DIR+'/config.yml', 'w')
    # f.write(cfg.dump())
    # f.close()
    #visualize(cfg, dataset_metadata)
    #evaluation(cfg, 'eval')

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg) #DefaultTrainer(cfg) #MyTrainer(cfg) #
    trainer.resume_or_load(resume=resume)
    torch.cuda.empty_cache()
    trainer.train()

    #Save Config
    # f = open(cfg.OUTPUT_DIR+'/config.yml', 'w')
    # f.write(cfg.dump())
    # f.close()
    # visualize(cfg, dataset_metadata)
    evaluation(cfg, 'eval')



def visualize(cfg, dataset_metadata):

    #Use the final weights generated after successful training for inference  
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8     # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import ColorMode
    dataset_dicts = DatasetCatalog.get("dataset_eval_separated")
    for d in random.sample(dataset_dicts, 5):    
        im = cv2.imread(d["file_name"])
        panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
        v = Visualizer(im[:, :, ::-1],
                    metadata= dataset_metadata, 
                    scale=0.8, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        plt.figure(figsize = (14, 10))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()


def evaluation(cfg, DSet = 'eval'):
        
    #Use the final weights generated after successful training for inference  
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8     # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)
    evaluator = COCOPanopticEvaluator("dataset_eval_separated", cfg.OUTPUT_DIR ) #COCOEvaluator
    val_loader = build_detection_test_loader(cfg, "dataset_eval_separated")
    eval_results=inference_on_dataset(predictor.model, val_loader, evaluator)


if __name__ == '__main__':
 
    launch(lunch_training, 1, dist_url = "auto")



    




