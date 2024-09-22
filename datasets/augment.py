import numpy as np
import random
import cv2
from PIL import Image
from torchvision import transforms as T


class rmv_transform():
    def __init__(self, img_size, augment=False, crop=False):
        self.img_size = img_size
        self.augment = augment
        if self.augment:
            self.augment_transform = augmentation(crop=crop, crop_size=img_size)
        self.final_transform = []
        if img_size is not None:
            self.final_transform.append(T.Resize(img_size, T.InterpolationMode.BICUBIC))
        self.final_transform.append(T.ToTensor())
        self.final_transform = T.Compose(self.final_transform)

        self.final_transform_mask = []
        if img_size is not None:
            self.final_transform_mask.append(T.Resize(img_size, T.InterpolationMode.NEAREST))
        self.final_transform_mask.append(T.ToTensor())
        self.final_transform_mask = T.Compose(self.final_transform_mask)

    def __call__(self, imgs, masks=[]):
        if self.augment:
            imgs, masks = self.augment_transform(imgs, masks)
        imgs = [self.final_transform(Image.fromarray(cv2.cvtColor(i.astype(np.uint8), cv2.COLOR_BGR2RGB))) for i in
                imgs]
        masks = [self.final_transform_mask(Image.fromarray(i.astype(np.uint8))) for i in masks]
        return imgs, masks


class augmentation():
    def __init__(self, brightness=True, flip=True, crop=False, crop_size=None):
        self.brightness = brightness
        self.flip = flip
        self.crop = crop
        self.crop_size = crop_size

        self.contrast_lower = 0.75
        self.contrast_upper = 1.25
        if self.crop_size is not None:
            self.crop_h = self.crop_size[0]
            self.crop_w = self.crop_size[1]


    def random_contrast(self, imgs):
        if random.random() < 0.5:
            return imgs
        alpha = random.uniform(self.contrast_lower, self.contrast_upper)
        new_imgs = []
        for img in imgs:
            img = img.astype(np.float32)
            img = img * alpha
            img = np.around(img)
            img = np.clip(img, 0, 255).astype(np.uint8)
            new_imgs.append(img)
        return new_imgs

    def random_flip(self, imgs, masks):
        if random.random() < 0.75:
            return imgs, masks
        new_imgs = []
        for img in imgs:
            img = cv2.flip(img, 1)
            new_imgs.append(img.astype(np.uint8))
        imgs = new_imgs
        new_masks = []
        for mask in masks:
            mask = cv2.flip(mask, 1)
            new_masks.append(mask.astype(np.uint8))
        masks = new_masks
        return imgs, masks

    def random_crop(self, imgs, masks):
        if random.random() < 0.25:
            return imgs, masks

        fx_scale = 0.5 + random.randint(0, 15) / 10.0
        fy_scale = 0.5 + random.randint(0, 15) / 10.0
        new_imgs = []
        new_masks = []
        for img in imgs:
            new_imgs.append(cv2.resize(img, None, fx=fx_scale, fy=fy_scale, interpolation=cv2.INTER_LINEAR))
        for mask in masks:
            new_masks.append(cv2.resize(mask, None, fx=fx_scale, fy=fy_scale, interpolation=cv2.INTER_NEAREST))
        imgs = new_imgs
        masks = new_masks

        img_h, img_w, _ = imgs[0].shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            new_imgs = []
            for img in imgs:
                img = cv2.copyMakeBorder(img, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))
                new_imgs.append(img.astype(np.uint8))
            imgs = new_imgs
            new_masks = []
            for mask in masks:
                mask = cv2.copyMakeBorder(mask, 0, pad_h, 0,
                                          pad_w, cv2.BORDER_CONSTANT,
                                          value=(0,))
                new_masks.append(mask.astype(np.uint8))
            masks = new_masks

        img_h, img_w, _ = imgs[0].shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        new_imgs = []
        for img in imgs:
            img = img[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]
            new_imgs.append(img.astype(np.uint8))
        imgs = new_imgs
        new_masks = []
        for mask in masks:
            mask = mask[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]
            new_masks.append(mask.astype(np.uint8))
        masks = new_masks
        return imgs, masks

    def __call__(self, imgs, masks=[]):
        if self.crop:
            imgs, masks = self.random_crop(imgs, masks)

        if self.brightness:
            imgs = self.random_contrast(imgs)

        if self.flip:
            imgs, masks = self.random_flip(imgs, masks)

        return imgs, masks
