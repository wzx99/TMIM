import os
import logging
import json

from torch.utils.data import Dataset, ConcatDataset
import cv2
import torch
import numpy as np
from pycocotools.coco import COCO

from utils.pyt_utils import get_main_flag
from datasets.augment import rmv_transform


class StandardDataset(Dataset):
    def __init__(self, cfg):
        super(StandardDataset, self).__init__()
        root = cfg.get('data_dir')

        dataRoot = os.path.join(root, 'all_images')
        self.imageFiles = [os.path.join(dataRoot, filename) for filename in os.listdir(dataRoot) if CheckImageFile(filename)]
        gtRoot = os.path.join(root, 'all_labels')
        self.gtFiles = [os.path.join(gtRoot, filename) for filename in os.listdir(dataRoot) if CheckImageFile(filename)]
     
        self.transform = rmv_transform(cfg.get('img_size', None), cfg.get('augment', False), cfg.get('crop', False))

        if get_main_flag():
            logging.info('dataset root:\t{}'.format(root))
            logging.info('\tnum samples: {}'.format(self.__len__()))
    
    def __getitem__(self, index):
        img = cv2.imread(self.imageFiles[index], cv2.IMREAD_COLOR)
        gt = cv2.imread(self.gtFiles[index], cv2.IMREAD_COLOR)
        path = self.imageFiles[index].split('/')[-1]

        if self.transform is not None:
            img_list, _ = self.transform([img, gt])
            img, gt = img_list[0], img_list[1]

        return img, gt, path
    
    def __len__(self):
        return len(self.imageFiles)


class MaskDataset(Dataset):
    def __init__(self, cfg):
        super(MaskDataset, self).__init__()
        root = cfg.get('data_dir')

        dataRoot = os.path.join(root, 'all_images')
        self.imageFiles = [os.path.join(dataRoot, filename) for filename in os.listdir(dataRoot) if CheckImageFile(filename)]
        gtRoot = os.path.join(root, 'all_labels')
        self.gtFiles = [os.path.join(gtRoot, filename) for filename in os.listdir(dataRoot) if CheckImageFile(filename)]
        maskRoot = os.path.join(root, 'mask')
        self.maskFiles = [os.path.join(maskRoot, filename) for filename in os.listdir(dataRoot) if CheckImageFile(filename)]
     
        self.transform = rmv_transform(cfg.get('img_size', None), cfg.get('augment', False), cfg.get('crop', False))

        num_samples = int(len(self.imageFiles)*cfg.get('ratio',1.0))
        self.imageFiles = self.imageFiles[:num_samples]

        if get_main_flag():
            logging.info('dataset root:\t{}'.format(root))
            logging.info('\tnum samples: {}'.format(self.__len__()))
    
    def __getitem__(self, index):
        img = cv2.imread(self.imageFiles[index], cv2.IMREAD_COLOR).astype('float32')
        gt = cv2.imread(self.gtFiles[index], cv2.IMREAD_COLOR).astype('float32')
        mask = cv2.imread(self.maskFiles[index], cv2.IMREAD_GRAYSCALE).astype('float32')
        path = self.imageFiles[index].split('/')[-1]

        if self.transform is not None:
            img_list, mask_list = self.transform([img, gt], [mask])
            img, gt = img_list[0], img_list[1]
            mask = mask_list[0]
        return img, gt, mask, path
    
    def __len__(self):
        return len(self.imageFiles)


class InferDataset(Dataset):
    def __init__(self, cfg):
        self.data_dir = cfg.get('data_dir')
        self.transform = rmv_transform(cfg.get('img_size', None), cfg.get('augment', False), cfg.get('crop', False))

        self.image_paths = [os.path.join(self.data_dir, filename) for filename in os.listdir(self.data_dir) if CheckImageFile(filename)]
        self.num_samples = len(self.image_paths)

        if get_main_flag():
            logging.info('dataset root:\t{}'.format(self.data_dir))
            logging.info('\tnum samples: {}'.format(self.__len__()))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        path = self.image_paths[index].split('/')[-1]

        if self.transform is not None:
            img, _ = self.transform([img], [])
            img = img[0]
        return img, path


class InferMaskDataset(Dataset):
    def __init__(self, cfg):
        self.data_dir = cfg.get('data_dir')
        self.anno_dir = cfg.get('anno_dir')
        self.variant = cfg.get('variant', 'standard')
        self.transform = rmv_transform(cfg.get('img_size', None), cfg.get('augment', False), cfg.get('crop', False))
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
            ann_ids = coco.getAnnIds([img_id])
            if len(ann_ids) == 0:
                continue
            ann_ids = [str(ann_id) for ann_id in ann_ids]
            ann_info = coco.loadAnns(ann_ids)
            instances = list()
            for ann in ann_info:
                if self.variant == 'cocotext':
                    poly = np.asarray(ann['mask']).reshape(-1, 2)
                elif self.variant == 'textocr':
                    poly = np.array(list(map(float, ann['points']))).reshape(-1, 2)
                else:
                    poly = np.array(list(map(float, ann['segmentation'][0]))).reshape(-1, 2)
                instances.append(dict(poly=poly))
            self.image_paths.append(os.path.join(self.data_dir, os.path.basename(img_path)))
            self.targets.append(instances)
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


def get_mask(image, target):
    h, w = image.shape[:2]
    mask = np.zeros((h, w))

    for instance in target:
        polygon = instance['poly']
        polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)
        cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 1)
    return mask*255


def CheckImageFile(filename):
    return any(filename.endswith(extention) for extention in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP'])


def build_dataset(cfg):
    if isinstance(cfg['data_class'], list):
        datasets = []
        for data_cfg in cfg['data_class']:
            if data_cfg['data_class']=='standard':
                datasets.append(StandardDataset(data_cfg))
            elif data_cfg['data_class']=='mask':
                datasets.append(MaskDataset(data_cfg))
        dataset = ConcatDataset(datasets)
    else:
        if cfg['data_class']=='standard':
            dataset = StandardDataset(cfg)
        elif cfg['data_class']=='mask':
            dataset = MaskDataset(cfg)
        elif cfg['data_class']=='infermask':
            dataset = InferMaskDataset(cfg)
        elif cfg['data_class']=='infer':
            dataset = InferDataset(cfg)
    return dataset
