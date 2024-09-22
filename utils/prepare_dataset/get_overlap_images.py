import os
import cv2
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
import json


datasets = [
    dict(
        variant = 'ic15',
        data_dir = '/data/zixiao_wang/text_det/ic15/train_images',
        anno_dir = '/data/zixiao_wang/text_det/ic15/annotation.json',
    ),
    dict(
        variant = 'textocr',
        data_dir = '/data/zixiao_wang/text_det/textocr/train_images',
        anno_dir = '/data/zixiao_wang/text_det/textocr/TextOCR_0.1_train.json',
    ),
    dict(
        variant = 'textocr',
        data_dir = '/data/zixiao_wang/text_det/textocr/train_images',
        anno_dir = '/data/zixiao_wang/text_det/textocr/TextOCR_0.1_val.json',
    ),
    dict(
        variant = 'totaltext',
        data_dir = '/data/zixiao_wang/text_det/totaltext/train_images',
        anno_dir = '/data/zixiao_wang/text_det/totaltext/annotation.json',
    ),
    dict(
        variant = 'cocotext',
        data_dir = '/data/zixiao_wang/text_det/cocotext/train2014',
        anno_dir = '/data/zixiao_wang/text_det/cocotext/cocotext.v2.json',
    ),
    dict(
        variant = 'mlt',
        data_dir = '/data/zixiao_wang/text_det/mlt/train_images',
        anno_dir = '/data/zixiao_wang/text_det/mlt/annotation.json',
    ),
    dict(
        variant = 'art',
        data_dir = '/data/zixiao_wang/text_det/art/train_images',
        anno_dir = '/data/zixiao_wang/text_det/art/annotation.json',
    ),
    dict(
        variant = 'lsvt',
        data_dir = '/data/zixiao_wang/text_det/lsvt/train_images',
        anno_dir = '/data/zixiao_wang/text_det/lsvt/annotation.json',
    ),
    dict(
        variant = 'rects',
        data_dir = '/data/zixiao_wang/text_det/rects/img',
        anno_dir = '/data/zixiao_wang/text_det/rects/annotation.json',
    ),
]

target_dir = '/data/zixiao_wang/text_seg/ic13/test/all_images'

for dataset in datasets:
    target_img_list = os.listdir(target_dir)
    print('start test {}'.format(dataset['variant']))
    coco = COCO(dataset['anno_dir'])
    if dataset['variant'] == 'cocotext' or dataset['variant'] == 'textocr':
        if dataset['variant'] == 'cocotext':
            for img in coco.dataset['imgs']:
                if 'train' == coco.dataset['imgs'][img]['set']:
                    coco.imgs[img] = coco.dataset['imgs'][img]
        elif dataset['variant'] == 'textocr':
            coco.imgs = coco.dataset['imgs']
        for img in coco.dataset['imgToAnns']:
            ann_ids = coco.dataset['imgToAnns'][img]
            anns = [
                coco.dataset['anns'][str(ann_id)] for ann_id in ann_ids
            ]
            coco.dataset['imgToAnns'][img] = anns
            coco.imgToAnns = coco.dataset['imgToAnns']
            coco.anns = coco.dataset['anns']

    img_ids = coco.getImgIds()
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs([img_id])[0]
        img_info['img_id'] = img_id
        img_path = img_info['file_name']
        if img_path.endswith('.gif'):
            continue
        ann_ids = coco.getAnnIds([img_id])
        if len(ann_ids) == 0:
            continue

        img1_path = os.path.join(dataset['data_dir'], os.path.basename(img_path))
        image1 = cv2.imread(img1_path)
        image1 = cv2.resize(image1,(512,512))/255.0
        for img2_path in target_img_list:
            image2 = cv2.imread(os.path.join(target_dir, img2_path))
            image2 = cv2.resize(image2,(512,512))/255.0
            dist = (np.abs(image1-image2)).mean()
            if dist<0.05:
                with open('overlap.txt', 'a') as f:
                    f.write(dataset['variant']+' '+img_path+ ' '+img2_path+' '+str(dist)+'\n')
                target_img_list.remove(img2_path)
                print(dataset['variant']+' '+img_path+ ' '+img2_path+' '+str(dist)+'\n')
                break


overlap_dict = {'totaltext':[], 'ic15':[], 'cocotext':[], 'mlt':[], 'art':[], 'lsvt':[], 'rects':[], 'textocr':[]}

with open('overlap.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    dataset_name, img_name = line.strip().split(' ')[:2]
    overlap_dict[dataset_name].append(img_name)

json.dump(overlap_dict, open('overlap.json', 'w'), ensure_ascii=False)