
import imghdr
import os
import os.path as osp
import re
import json
import xml.etree.ElementTree as ET
import json
import PIL.ImageDraw
import PIL.Image  
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import cv2
import random
import sys
import multiprocessing as mp
import time
import functools
import argparse
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
# from regex import F        # (pip install Pillow)
import shutil

from PIL import ImageOps, Image, ImageEnhance,ImageColor
from xml.dom import minidom
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
from datetime import datetime
from panopticapi.utils import rgb2id, id2rgb
# from fcntl import F_SEAL_SEAL

# Input 
#     |-- meta.json
#     |-> train
#          |-> ann
#               |-- json files
#          |-> img
#               |-- RGB images
#     
#     |-> val
#     |    |-> ann
#               |-- Annotations
#          |-> img
#               |-- RGB images
#     
#     |-> eval
#     |    |-> ann
#               |-- Annotations
#          |-> img
#               |-- RGB images
#
# TrainingData #!(Created by script)
#     |-- panoptic_coco_categories.json
#     |-> train
#            |-- train_Instance.json
#            |-- train_panoptic.json
#            |-> images
#                    |-- RGB images
#            |-> panopticMask
#                    |-- all category mask
#            |-> semanticMask
#                    |-- all semantic mask
#        
#     |->   val
#            |-- val_Instance.json
#            |-- val_panoptic.json
#            |-> images
#                    |-- RGB images
#            |-> panopticMask
#                    |-- all category mask
#            |-> semanticMask
#                    |-- all semantic mask
#
#     |->   eval
#            |-- val_Instance.json
#            |-- val_panoptic.json
#            |-> images
#                    |-- RGB images
#            |-> panopticMask
#                    |-- all category mask
#            |-> semanticMask
#                    |-- all semantic mask

# Main FolderPath, Resize, BrithnessChanges, Contrast changes

parser = argparse.ArgumentParser(description='Convert supervisly format to COCO panoptic format')
parser.add_argument('src', metavar='dir_src', type=str,
                    help='Path to Supervisely format directory.')
parser.add_argument('name', metavar='project_name', type=str,
                    help='Name of the project on Supervisely')
parser.add_argument('--brightness', action='store_true', help='')
parser.add_argument('--contrast', action='store_true', help='')

args = parser.parse_args()


isthing_list = ('Node')

category_ids = {
    'Node' : 1,
    'Shoot': 2,
    'Trunk': 3,
    'Wire' : 4,
    'Post' : 5,
}

coco_format_panoptic = {
    "info": [
        {
            "description": "Vine 2022 COCO Dataset",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2022,
            "contributor": "Maaratech",
            "date_created": datetime.today().strftime('%Y/%m/%d')
            }
    ],
    "licenses": [
        {
            "url": "https://cares.blogs.auckland.ac.nz/research/robots-in-agriculture/data-informed-decision-making-and-automation-in-orchards-and-vineyards/",
            "name": "Maaratech License"
            }
            ],
    "categories": [
        {
        }
    ],
    "images": [
        {
        }
    ],
    "annotations": [
        {
        }
    ]
}

coco_format_Instance = {
    "info": [
        {
            "description": "Vine 2022 Panoptic COCO Dataset",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2022,
            "contributor": "Maaratech",
            "date_created": datetime.today().strftime('%Y/%m/%d')
            }
    ],
    "licenses": [
        {
            "url": "https://cares.blogs.auckland.ac.nz/research/robots-in-agriculture/data-informed-decision-making-and-automation-in-orchards-and-vineyards/",
            "name": "Maaratech License"
            }
            ],
    "categories": [
        {
            "id": 1,
            "name": 'Node'
        }
    ],
    "images": [
        {
        }
    ],

    "annotations": [
        {
        }
    ]
}

def display_rgb(rgb,title):
  plt.figure()
  plt.imshow(rgb)
  plt.title(title)
  plt.show()
  plt.pause(0.001)


def create_annotation_format_Panoptic(image_id, imageName, segment_infos):
    annotation = {
        'image_id': image_id,
        'file_name':imageName,
        'segments_info':segment_infos
    }
    return annotation

def create_annotation_format_Instance(multi_poly , area , bbox , image_id, category_id, annotation_id):
    annotation = {
        'id': annotation_id,
        'category_id': category_id,
        'segmentation': multi_poly,
        'image_id': image_id,
        'bbox': bbox,
        'area': area,
        'iscrowd': 0,
    }
    return annotation

def create_segment_infos_Panoptic(maskid, bbox, area, category_id):
    segments_info = {
        'id': int(maskid),
        'category_id': category_id,
        'area': area,
        'bbox': bbox,
        'iscrowd': 0,
    }
    return segments_info

def create_image_annotation(file_name, width, height, image_id):
    images = {
        'id': image_id,
        'file_name': file_name,
        'height': height,
        'width': width,
        'license' : None,
        'flickr_url' : None,
        'coco_url' : None,
        'date_captured' : None
    }
    return images


def convert2JPG(images, output_image):
    for root, dirs, files in os.walk(os.path.abspath(images)):
        i = 0
        for file in files:
            i += 1
            fileName,fileType = file.split('.')
            print(i, '-  Save ', fileName , " to images directory" )
            #open image
            image = Image.open(root +'/' + file)
            #! Save it as JPG
            outName = output_image + '/' + fileName + '.jpg'
            rgb_im = image.convert('RGB')
            rgb_im.save(outName)
                
            

def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation='low')

    polygons = []
    segmentations = []
    j = 0
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        
        if(poly.is_empty):
            # Go to next iteration, dont save empty values in list
            continue

        polygons.append(poly)

        #segmentation = np.array(poly.exterior.coords).ravel().tolist()
        #segmentation = [int(x) for x in segmentation]
        
        segmentation = np.array(poly.exterior.coords).tolist()
        segmentation = [[int(x),int(y)] for x,y in segmentation]
        segmentations.append(segmentation)
    
    return polygons, segmentations


def absolute_file_paths_image(dataPath):
    images = []
    json_images= []
    #find all jpg images avilable 
    for root, dirs, files in os.walk(os.path.abspath(dataPath+'/img')):
        for file in files:
            # Filter only for images in folder         
            if '.png' or '.jpg' in file: 
                images.append(os.path.join(root, file))

    #find mask and xml files and create mask info
    for root, dirs, files in os.walk(os.path.abspath(dataPath+'/ann' )):
        for file in files:
            # Filter only for images in folder         
            if '.png' or '.jpg' in file: 
                json_images.append(os.path.join(root, file))

    return images , json_images

def get_new_color (color, usedColor):
    while (1):
        color[0] += 2
        if color[0] / 255 > 1 :
            color[0] = color[0] % 255
            color[1] += 2
            if color[1] / 255 > 1 :
                color[1] = color[1] % 255
                color[2] += 2
                if color[2] / 255 > 1 :
                    color[2] = color[2] % 255

        if not color in usedColor:
            usedColor.append(color)
            break
    
    return color, usedColor


def creat_Mask_From_polygon (imageMask, polygon,  color):
    if len(polygon) == 1:
        area = np.array(polygon)
        area = np.round(area).astype(int)
        filled = cv2.fillPoly(imageMask, pts = [area], color =color)
    else:    
        for poly in enumerate(polygon):
            area = np.array(poly[1])
            area = np.round(area).astype(int)
            filled = cv2.fillPoly(imageMask, pts = [area], color =color)

    return filled

def calc_mask_BBox_area (image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3), 0)
    th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(th)
    xmin,ymin,w,h = cv2.boundingRect(coords)
    
    area = len(coords)
    
    return (xmin,ymin,w,h), area

def polygon_Convert(polys):
    outPoly_Complete = []
    areas = 0
    first = True
    for poly in polys:
        outPoly = []
        min_x_temp, min_y_temp, max_x_temp, max_y_temp = Polygon(poly).bounds
        if first:
           min_x, min_y, max_x, max_y =  min_x_temp, min_y_temp, max_x_temp, max_y_temp
           first = False
        else:
            if min_x > min_x_temp:
                min_x = min_x_temp
            if min_y > min_y_temp:
                min_y = min_y_temp
            if max_x < max_x_temp:
                max_x = max_x_temp
            if max_y < max_y_temp:
                max_y = max_y_temp

        #Calculate the area
        areas += Polygon(poly).area   
        for point in poly:
            outPoly.append(point[0])
            outPoly.append(point[1])

        outPoly_Complete.append(outPoly)

    #Find the bounding Box
    width = max_x - min_x
    height = max_y - min_y
    bbox = (min_x, min_y, width, height)

    return outPoly_Complete , areas, bbox
 

def brightnees_Contrast(input_path ,brightness, contrast):
    for root, dirs, files in os.walk(os.path.abspath(input_path+'/img/')):
        print ('Editing images')
    for file in files:
        # Filter only for images in folder         
        if '.png' or '.jpg' in file:
            image_path = os.path.join(root, file)
            jsn_path = os.path.join(root, 'ann')
            im = Image.open(image_path)

            jsonpath = os.path.abspath(input_path+'/ann/' + file + '.json' )
            with open(jsonpath) as json_file:
                data = json.load(json_file)

                if brightness:
                    newFile= 'br_'+ file
                    new_jsonpath = os.path.abspath(input_path+'/ann/' + newFile + '.json' )
                    with open(new_jsonpath,'w') as outfile:
                        json.dump(data, outfile)

                    #save file if it can find json file and save it
                    enhancer = ImageEnhance.Brightness(im)

                    factor = random.choice([0.5, 1.5])
                    im_output = enhancer.enhance(factor)
                    im_output.save(os.path.join(root,newFile))
                    print('save file: ' , newFile)
                
                if contrast:
                    newFile= 'cnt_'+ file
                    new_jsonpath = os.path.abspath(input_path+'/ann/' + newFile + '.json' )
                    with open(new_jsonpath,'w') as outfile:
                        json.dump(data, outfile)

                    #save file if it can find json file and save it
                    enhancer = ImageEnhance.Contrast(im)

                    factor = random.choice([0.7, 1.3])
                    im_output = enhancer.enhance(factor)
                    im_output.save(os.path.join(root,newFile))
                    print('save file: ' , newFile)


def images_annotations (categories, Panoptic_jsons_files, maskFolder_Ponaptic):
    annotations_Panoptic = []
    annotations_Instance = []
    images_Panoptic = []
    images_Instance = []

    image_id = 0
    annotation_id = 1

    for json_file_name in Panoptic_jsons_files :
        
        segment_infos = [] 
        ResizeFlag = False
        stuff_categories = []
        used_color = [c['color'] for c in categories]
        node_Color = [c['color'] for c in categories if c['isthing']][0]
        
        with open(json_file_name) as json_file:
            
            data = json.load(json_file)
            #extract the image file
            filename = os.path.basename(json_file_name).replace('.json','')
            #Change PNG formated name to jpg in Json
            newName,fileType = filename.split('.')
            img_name = newName + '.jpg'
            ann_name = newName + '.png'
            
            image_id = image_id + 1
            print(image_id, ' Converting file:  ' , img_name)
            image = create_image_annotation(img_name,  data['size']['width'], data['size']['height'], image_id)
            images_Panoptic.append(image)
            images_Instance.append(image)

            imageMask = np.zeros(( data['size']['height'],data['size']['width'],3), np.uint8)
            imageMask_Instance = np.zeros(( data['size']['height'],data['size']['width'],3), np.uint8)
    
            #Go through each object 
            for i in range(0,len(data['objects'])):
                multi_ploygon = []
                instance = False  #To check if has instance
                _object_ = data['objects'][i]


                #Find the category 
                category_id = category_ids[_object_['classTitle']] 
                category = next((item for item in categories if item["name"] == _object_['classTitle']), None)
                color = category['color']
                maskid = rgb2id(color)

                polygon = _object_['points']['exterior']           
                polygon.append(polygon[0])
                multi_ploygon.append(polygon)
                
                if category['isthing']: #it is instance
                    
                    multi_poly , area , bBox = polygon_Convert(multi_ploygon)
                    #! Add Instance Annotation
                    annotation = create_annotation_format_Instance(multi_poly , area , bBox , image_id, category_id, annotation_id)
                    annotations_Instance.append(annotation)
                    annotation_id += 1 #Object ID
                    
                    #Semantic annotation
                    maskid = rgb2id(node_Color)
                    segment_infos.append(create_segment_infos_Panoptic(maskid, bBox, area, category_id))
                    imageMask_Instance = creat_Mask_From_polygon(imageMask_Instance, multi_ploygon, node_Color)
                    node_Color, used_color = get_new_color (node_Color.copy(), used_color.copy())
                    
                else:
                    imageMask = creat_Mask_From_polygon(imageMask, multi_ploygon, color)
             
            
            imageMask_all = imageMask.copy()
            #?imageMask_all[imageMask_Instance != [0,0,0]] = 0
            os.makedirs(maskFolder_Ponaptic+'seg/', exist_ok = True)
            im = Image.fromarray(imageMask_all, 'RGB')
            im.save(maskFolder_Ponaptic+'seg/'+ ann_name)

            imageMask[imageMask_Instance != [0,0,0]] = imageMask_Instance[imageMask_Instance != [0,0,0]]
            
            
            for _cat in categories:
                if not _cat['isthing']:
                    temp_image = np.zeros(( data['size']['height'],data['size']['width'],3), np.uint8)
                    temp_image[imageMask_all == _cat['color']] = imageMask_all[imageMask_all == _cat['color']]
                    if temp_image.any():
                        category_id = _cat['id']
                        maskid = rgb2id(_cat['color'])
                        bBox, area = calc_mask_BBox_area (temp_image)
                        segment_infos.append(create_segment_infos_Panoptic(maskid, bBox, area, category_id))

            #! Add Panoptic Annotation
            annotation_panoptic = create_annotation_format_Panoptic(image_id, ann_name, segment_infos)
            annotations_Panoptic.append(annotation_panoptic)
            
            im = Image.fromarray(imageMask, 'RGB')
            im.save(maskFolder_Ponaptic+ ann_name)
            

    return images_Panoptic, annotations_Panoptic, images_Instance, annotations_Instance


def create_ponaptic_categories_json(input, output):
    meta_json = input + '/meta.json'
    with open(meta_json) as meta_json: # A json file describing all the categories (including the background) according to the
        meta_categories = json.load(meta_json)#coco panoptic guidelines
    
    #Create Ponaptic json file
    categories = []
    for _class in meta_categories['classes']:
        rgb_color = list(ImageColor.getcolor(_class['color'], "RGB"))

        isting = 1 if _class['title'] in isthing_list else 0
        categories.append({"id": category_ids[_class['title']],
                            "name":_class['title'],
                            "color":rgb_color,
                            "isthing": isting})

    categories_sorted = sorted(categories, key=lambda d: d['id'], reverse = False)
    with open(output + '/panoptic_coco_categories.json','w') as outfile:
                json.dump(categories_sorted, outfile)
    
    return categories_sorted


def _process_panoptic_to_semantic(input_panoptic, output_semantic, segments, id_map):
    input_panoptic = input_panoptic.replace('jpg','png')
    output_semantic = output_semantic.replace('jpg','png')

    panoptic = np.asarray(Image.open(input_panoptic), dtype=np.uint32)

    panoptic = rgb2id(panoptic)
    output = np.zeros_like(panoptic, dtype=np.uint8)
    #output[:]=255
    for seg in segments:
        cat_id = seg["category_id"]
        new_cat_id = id_map[cat_id]
        output[panoptic == seg["id"]] = new_cat_id
    Image.fromarray(output).save(output_semantic)


def separate_coco_semantic_from_panoptic(panoptic_json, panoptic_root, sem_seg_root, categories):
    """
    Create semantic segmentation annotations from panoptic segmentation
    annotations, to be used by PanopticFPN.

    It maps all thing categories to class 0, and maps all unlabeled pixels to class 255.
    It maps all stuff categories to contiguous ids starting from 1.

    Args:
        panoptic_json (str): path to the panoptic json file, in COCO's format.
        panoptic_root (str): a directory with panoptic annotation files, in COCO's format.
        sem_seg_root (str): a directory to output semantic annotation files
        categories (list[dict]): category metadata. Each dict needs to have:
            "id": corresponds to the "category_id" in the json annotations
            "isthing": 0 or 1
    """
    os.makedirs(sem_seg_root, exist_ok=True)

    stuff_ids = [k["id"] for k in categories if k["isthing"] == 0]
    thing_ids = [k["id"] for k in categories if k["isthing"] == 1]
    id_map = {}  # map from category id to id in the output semantic annotation
    assert len(stuff_ids) <= 254
    for i, stuff_id in enumerate(stuff_ids):
        id_map[stuff_id] = i + 1
    for thing_id in thing_ids:
        id_map[thing_id] = 0
    id_map[0] = 255


    pool = mp.Pool(processes=4) #max(mp.cpu_count() // 2, 4))

    def iter_annotations():
        for anno in panoptic_json["annotations"]:
            file_name = anno["file_name"]
            segments = anno["segments_info"]
            input = os.path.join(panoptic_root, file_name)
            output = os.path.join(sem_seg_root, file_name)
            yield input, output, segments

    print("Start writing to {} ...".format(sem_seg_root))
    
    pool.starmap(
        functools.partial(_process_panoptic_to_semantic, id_map=id_map),
        iter_annotations(),
        chunksize=100,
    )
    


def coco_from_supervisly(input ,output , brightness, contrast):
    #Create output folder
    os.makedirs(output, exist_ok = True)

    categories = create_ponaptic_categories_json(input,output)        #Categorise
    coco_format_panoptic['categories'] = categories

    instance_categories = []
    for _cat in categories:
        if _cat['isthing']:
            instance_categories.append(_cat)
    coco_format_Instance['categories'] = instance_categories

    for keyword in ['train', 'val', 'eval']:
        input_path = input+'/{}'.format(keyword)
        print('Loading ' + keyword + ' dataSet...')
        #! create random brightness and contast
        if brightness or contrast:
            brightnees_Contrast(input_path , brightness, contrast)
        
        images, jsons = absolute_file_paths_image(input_path)

        #? Create  Images outputfolder and make resize and convert all to jpg format
        #! images
        out_imagesFolder = output+'/{}/images'.format(keyword)
        os.makedirs(out_imagesFolder, exist_ok = True)
        convert2JPG(input_path +"/img" , out_imagesFolder)

        #? Create  Mask outputfolder
        #! Panoptic
        _maskFolder_Semantic = output+'/{}/semanticMask'.format(keyword)
        os.makedirs(_maskFolder_Semantic, exist_ok = True)
        maskFolder_Semantic = _maskFolder_Semantic + "/"

        #! Instance
        _maskFolder_Ponaptic = output+'/{}/panopticMask'.format(keyword)
        os.makedirs(_maskFolder_Ponaptic, exist_ok = True)
        maskFolder_Ponaptic = _maskFolder_Ponaptic + "/"


        #extract annotation info
        print('Creating ' + keyword + ' dataSet...')
        coco_format_panoptic['images'], coco_format_panoptic['annotations'], coco_format_Instance['images'], coco_format_Instance['annotations']  = images_annotations(categories, jsons, maskFolder_Ponaptic)
        
        #save as jason file
        print('Saving ' + keyword + '  dataSet...')
        #! Saving Panoptic
        with open(output+'/{}'.format(keyword) + '/{}_panoptic.json'.format(keyword),'w') as outfile:
                json.dump(coco_format_panoptic, outfile)
        #!Saving instance - For Nodes
        with open(output+'/{}'.format(keyword) + '/{}_Instance.json'.format(keyword),'w') as outfile:
                json.dump(coco_format_Instance, outfile)


        #Create semantic Mask
        separate_coco_semantic_from_panoptic(coco_format_panoptic, _maskFolder_Ponaptic+'/seg',
                                                _maskFolder_Semantic, categories)
    
def main(agrs):      
    input_path = args.src + '/input' 
    outputPath = args.src + '/TrainingData'
    print('Load images from :' + input_path)    
    coco_from_supervisly(input_path, outputPath, args.brightness, args.contrast)

def split(root_dir, project_name):
    ann_dir = osp.join(root_dir, project_name, 'ann')
    img_dir = osp.join(root_dir, project_name, 'img')

    data = {}
    df = pd.DataFrame()
    df['img_path'] = os.listdir(img_dir)
    df['ann_path'] = df['img_path'] + '.json'

    assert(set(df['ann_path'].tolist()) == set(os.listdir(ann_dir)))
    
    width = []
    height = []
    num_wire = []
    num_shoot = []
    num_node = []
    num_post = []
    num_trunk = []

    for idx, row in df.iterrows():
        ann_json = osp.join(ann_dir, row['ann_path'])
        with open(ann_json) as f:
            data = json.load(f)
            width.append(data['size']['width'])
            height.append(data['size']['height'])

            classes = defaultdict(int)
            for o in data['objects']:
                classes[o['classTitle']] += 1

            num_wire.append(classes['Wire'])
            num_shoot.append(classes['Shoot'])
            num_node.append(classes['Node'])
            num_post.append(classes['Post'])
            num_trunk.append(classes['Trunk'])
    
    df['width'] = width
    df['height'] = height
    df['n_wire'] = num_wire
    df['n_shoot'] = num_shoot
    df['n_node'] = num_node
    df['n_post'] = num_post
    df['n_trunk'] = num_trunk

    print(df.head())

    train_df, val_test = train_test_split(df, test_size=0.2)

    val_df, test_df = train_test_split(val_test, test_size=0.2)

    print(len(train_df), len(val_df), len(test_df))

    splits = [('train', train_df), ('val', val_df), ('test', test_df)]

    for split in splits:
        for idx, row in split[1].iterrows():
            src_ann_path = osp.join(ann_dir, row['ann_path'])
            dst_ann_dir = osp.join(root_dir, f'input/{split[0]}/ann')
            if not os.path.exists(dst_ann_dir):
                os.makedirs(dst_ann_dir)
            
            shutil.copy(src_ann_path, dst_ann_dir)

            src_img_path = osp.join(img_dir, row['img_path'])
            dst_img_dir = osp.join(root_dir, f'input/{split[0]}/img')
            if not os.path.exists(dst_img_dir):
                os.makedirs(dst_img_dir)
            
            shutil.copy(src_img_path, dst_img_dir)



if __name__ == '__main__':
    main(args)
    # split(args.src, args.name)

