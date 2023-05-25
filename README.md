# image-segmentation
This repository provides tools for image segmentation and label data conversion. It includes scripts for image segmentation using popular algorithms and techniques, as well as utilities for converting label data between different formats.

# Dataset Preparation
Public dataset
- https://conservancy.umn.edu/handle/11299/206575


## Supervisely
Export data as coco format. Please refer to [this](https://ecosystem.supervisely.com/apps/export-to-coco).

## Labelbox
Use tools/export_coco_labelbox.py to export coco dataset from a labelbox project

## Roboflow
Export data as coco format. Please refer to [this](https://docs.roboflow.com/exporting-data)

# Dataset Split
Use tools/split.py to split coco dataset as train/val. 

## Usage

The same as the original repo, with adding an argument (``--multi-class``) to preserve class distributions
The argument is optional to ensure backward compatibility

```
$ python cocosplit.py -h
usage: cocosplit.py [-h] -s SPLIT [--having-annotations]
                    coco_annotations train test

Splits COCO annotations file into training and test sets.

positional arguments:
  coco_annotations      Path to COCO annotations file.
  train                 Where to store COCO training annotations
  test                  Where to store COCO test annotations

optional arguments:
  -h, --help            show this help message and exit
  -s SPLIT              A percentage of a split; a number in (0, 1)
  --having-annotations  Ignore all images without annotations. Keep only these
                        with at least one annotation
  --multi-class         Split a multi-class dataset while preserving class
                        distributions in train and test sets
```

## Example
```
$ python cocosplit.py --having-annotations --multi-class -s 0.8 /path/to/your/coco_annotations.json train.json test.json
```

will split ``coco_annotation.json`` into ``train.json`` and ``test.json`` with ratio 80%/20% respectively. It will skip all
images (``--having-annotations``) without annotations.

# COCO annotation visualization
Use tools/coco_vis.py to overlay the COCO annotations on the original image to provide a visual representation of the annotations.


## Usage
In the following command, you need to replace path/to/coco_annotations.json with the actual path to your COCO annotations file, and path/to/original_images_directory with the actual path to the directory containing your original images.

```
python visualize_annotations.py path/to/coco_annotations.json path/to/original_images_directory

positional arguments:
  coco_annotations      Path to COCO annotations file.
  dir_images            Path to the directory for original images
```

## Example
```
python tools/coco_vis.py data/Fruitlets/train/ann/train.json data/Fruitlets/train/img
```

# Instance Segmentation
mmdetection model zoo
- Mask R-CNN (ICCV'2017)
- Cascade Mask R-CNN (CVPR'2018)
- Mask Scoring R-CNN (CVPR'2019)
- Hybrid Task Cascade (CVPR'2019)
- YOLACT (ICCV'2019)
- InstaBoost (ICCV'2019)
- SOLO (ECCV'2020)
- PointRend (CVPR'2020)
- DetectoRS (ArXiv'2020)
- SOLOv2 (NeurIPS'2020)
- SCNet (AAAI'2021)
- QueryInst (ICCV'2021)
- Mask2Former (ArXiv'2021)
- CondInst (ECCV'2020)
- SparseInst (CVPR'2022)
- RTMDet (ArXiv'2022)
- BoxInst (CVPR'2021)

## Configuration
TBD

## Logging
TBD

## Train
TBD

