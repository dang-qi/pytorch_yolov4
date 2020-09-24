# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/08 11:45
@Author        : Tianxiaomo
@File          : coco_annotatin.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import json
from collections import defaultdict
from tqdm import tqdm
import os
from pycocotools.coco import COCO

def transfer(json_path, out_path, images_dir_path):
    coco = COCO(json_path)

    catIds = coco.getCatIds(catNms=['person'])

    imgIds = coco.getImgIds(catIds=catIds )

    images = []
    # get annotation for single image
    for imgId in imgIds:
        img = coco.loadImgs(ids=[imgId])[0] #['file_name', 'flickr_url', 'date_captured', 'id', 'license', 'coco_url', 'width', 'height']
        annoIds=coco.getAnnIds(imgIds=[imgId], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annoIds) # dict_keys(['iscrowd', 'area', 'category_id', 'id', 'bbox', 'image_id', 'segmentation'])
        img['objects']=anns
        images.append(img)

    """write to txt"""
    with open(out_path, 'w') as f:
        for img in tqdm(images):
            file_path = os.path.join(images_dir_path, img['file_name'])
            f.write(file_path)
            for obj in img['objects']:
                cat_id = obj['category_id'] - 1
                box = obj['bbox']
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[0]+box[2])
                y2 = int(box[1]+box[3])
                box_info = ' {},{},{},{},{}'.format(x1, y1, x2, y2, cat_id)
                f.write(box_info)
            f.write('\n')



if __name__=='__main__':
    """hyper parameters"""
    train_json_path = os.path.expanduser('~/data/datasets/COCO/annotations/instances_train2014.json')
    train_images_dir_path = 'train2014/'
    train_output_path = '../data/coco_person_train.txt'

    val_json_path = os.path.expanduser('~/data/datasets/COCO/annotations/instances_val2014.json')
    val_images_dir_path = 'val2014/'
    val_output_path = '../data/coco_person_val.txt'

    transfer(train_json_path, train_output_path, train_images_dir_path)
    transfer(val_json_path, val_output_path, val_images_dir_path)