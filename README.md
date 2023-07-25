# image-segmentation
This repository provides tools for image segmentation and label data conversion. It includes scripts for image segmentation using popular algorithms and techniques, as well as utilities for converting label data between different formats.

# Prerequisite
- Python >= 3.8 (Strongly recommend to use [conda](https://docs.conda.io/en/main/miniconda.html))
  ```
  conda create -n imgseg python=3.8
  conda activate imgseg
  ```
- [Pytorch 1.13.1](https://pytorch.org/get-started/previous-versions/)
  ```
  pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
  ```
- [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
  ```
  python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
  # (add --user if you don't have permission)
  ```
- [MMDetection](https://mmdetection.readthedocs.io/en/latest/get_started.html)
  ```
  pip install -U openmim
  mim install mmengine
  mim install "mmcv>=2.0.0"
  mim install mmdet
  ```

# Installation
Install requirements after cloning
```
conda activate imgseg
cd image-segmentation
pip install -r requirements.txt
```

# Dataset
We at CARES group employ a carefully selected set of labeling tools([Supervisely](https://supervisely.com/), [Roboflow](https://roboflow.com/)), exclusively chosen to optimize the entire pipeline for training image detection models. To gain access to download the dataset, kindly reach out to your supervisor. We highly recommend migrating to one of these tools if you are currently using a different one.


## Export data from Supervisely

To enable coco detection and segmentation, follow these steps to export your data in coco format. For the most up-to-date tutorial, please visit [the official website](https://ecosystem.supervisely.com/apps/export-to-coco).

1. Navigate to the Supervisely project you wish to export.
2. Choose the option to export it in coco format.
3. Download the {project_name}.tar file containing the exported data.
4. Unpack the {project_name}.tar file using the following command:
   ```
   tar xopf {project_name}.tar
   ```

To perform pannoptic coco segmentation, follow these steps to export your data in Supervisely format and then convert it to panoptic coco format using the python script (tools/supervisely_to_panoptic_coco.py). For the most up-to-date tutorial, please visit [the official website](https://ecosystem.supervisely.com/apps/export-to-supervisely-format).

1. Navigate to the Supervisely project you wish to export.
2. Choose the option to export it in Supervisely format.
3. Download the {project_name}.tar file containing the exported data.
4. Unpack the {project_name}.tar file using the following command:
   ```
   tar xopf {project_name}.tar
   ```
5. Use the python script (tools/supervisely_to_panoptic_coco.py) to convert the data to panoptic coco format.

## Export data from Roboflow
Export data as coco format. Please refer to [this](https://docs.roboflow.com/exporting-data)

# Dataset Split
If your dataset doesn't have train/val splits, you can use tools/split.py to split coco dataset as train/val. 

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

## Train
TBD

