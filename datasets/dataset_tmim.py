import os
import logging
import json
import random

from torch.utils.data import Dataset, ConcatDataset
import cv2
import torch
import numpy as np
from pycocotools.coco import COCO

from utils.pyt_utils import get_main_flag
from datasets.augment import rmv_transform


class StandardDataset(Dataset):
    def __init__(self, cfg):
        self.data_dir = cfg.get('data_dir')
        self.anno_dir = cfg.get('anno_dir')
        self.variant = cfg.get('variant', 'standard')
        self.transform = rmv_transform(cfg.get('img_size', None), cfg.get('augment', False), cfg.get('crop', False))

        self.charset = cfg.get('charset', [])
        self.char_index = {}
        for i in range(len(self.charset)):
            self.char_index[self.charset[i]] = i

        self.num_samples = cfg.get('num_samples', None)
        with open('datasets/overlap.json', 'r') as f:
            data = f.read()
            self.overlap = json.loads(data)
        self.get_all_samples()

        if get_main_flag():
            logging.info('dataset root:\t{}'.format(self.data_dir))
            logging.info('\tnum samples: {}'.format(self.__len__()))
            
    def __len__(self):
        return self.num_samples

    def get_all_samples(self):
        self.image_paths = []
        self.targets = []
        coco = COCO(self.anno_dir)
        if self.variant == 'cocotext' or self.variant == 'textocr':
            if self.variant == 'cocotext':
                for img in coco.dataset['imgs']:
                    if 'train' == coco.dataset['imgs'][img]['set']:
                        coco.imgs[img] = coco.dataset['imgs'][img]
            elif self.variant == 'textocr':
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
        for img_id in img_ids:
            img_info = coco.loadImgs([img_id])[0]
            img_info['img_id'] = img_id
            img_path = img_info['file_name']
            if img_path.endswith('.gif'):
                continue
            if img_path in self.overlap[self.variant]:
                continue
            ann_ids = coco.getAnnIds([img_id])
            if len(ann_ids) == 0:
                continue
            ann_ids = [str(ann_id) for ann_id in ann_ids]
            ann_info = coco.loadAnns(ann_ids)
            instances = list()
            for ann in ann_info:
                if self.variant == 'cocotext':
                    poly = np.asarray(ann['mask']).reshape(-1,2)
                elif self.variant == 'textocr':
                    poly = np.array(list(map(float, ann['points']))).reshape(-1,2)
                else:
                    poly = np.array(list(map(float, ann['segmentation'][0]))).reshape(-1,2)
                instances.append(dict(poly=poly))
            self.image_paths.append(os.path.join(self.data_dir, os.path.basename(img_path))) 
            self.targets.append(instances)

        if self.num_samples is not None:
            self.num_samples = min(self.num_samples, len(self.image_paths))
        else:
            self.num_samples = len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        target = self.targets[index]
        mask = get_mask(img, target)

        path = self.image_paths[index].split('/')[-1]

        if self.transform is not None:
            img, mask = self.transform([img], [mask])
            img = img[0]
            mask = mask[0]
        return img, mask, path


class TwoMaskDataset(StandardDataset):
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        target = self.targets[index]
        mask = get_mask(img, target)

        mask2 = np.zeros_like(mask)
        for i in range(5):
            index2 = random.randint(0, self.__len__()-1)
            image_path2 = self.image_paths[index2]
            img2 = cv2.imread(image_path2, cv2.IMREAD_COLOR)
            target2 = self.targets[index2]
            mask2_ = get_mask(img2, target2)
            mask2_ = cv2.resize(mask2_, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask2[mask2_==255] = 255
            if (mask2==255).sum()>=img.size*0.01:
                break

        if self.transform is not None:
            img, mask = self.transform([img], [mask, mask2])
            img = img[0]
            mask, mask2 = mask[0], mask[1]
        return img, mask, mask2


def get_mask(image, target):
    h, w = image.shape[:2]
    mask = np.zeros((h, w))

    for instance in target:
        polygon = instance['poly']
        polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)
        cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 1)
    return mask*255


def build_dataset(cfg):
    if isinstance(cfg['data_class'], list):
        datasets = []
        for data_cfg in cfg['data_class']:
            if data_cfg['data_class']=='standard':
                datasets.append(StandardDataset(data_cfg))
            elif data_cfg['data_class']=='2mask':
                datasets.append(TwoMaskDataset(data_cfg))
        dataset = ConcatDataset(datasets)
    else:
        if cfg['data_class']=='standard':
            dataset = StandardDataset(cfg)
        elif cfg['data_class']=='2mask':
            dataset = TwoMaskDataset(cfg)
    return dataset
