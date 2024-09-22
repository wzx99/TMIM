import json
import os
import numpy as np
from PIL import Image
import cv2


class txt2CoCo:
    def __init__(self, image_dir, txt_dir):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.image_dir = image_dir
        self.txt_dir = txt_dir

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False)

    def to_coco(self):
        image_list = os.listdir(self.image_dir)
        print(len(image_list))
        for image in image_list:
            print(image, end='\r')
            self.images.append(self._image(image))

            reader = open(os.path.join(self.txt_dir, image+'.txt'), 'r', encoding='utf8').readlines()
            for line in reader:
                annotation = self._annotation(line)
                if len(annotation['segmentation'][0])<=4:
                        print('image {} only has points {} in segmentation'.format(image,annotation['segmentation'][0]))
                        continue
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = {}
        instance['license'] = []
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self._init_categories()
        return instance

    def _init_categories(self):
        category = {}
        category['supercategory'] = 'beverage'
        category['id'] = 1
        category['name'] = 'text'
        return [category]

    def _image(self, path):
        image = {}
        flag = 1
        if os.path.exists(os.path.join(self.image_dir, path.replace('.png', '.jpg'))):
            img_path = os.path.join(self.image_dir, path.replace('.png', '.jpg'))
        else:
            flag = 0
            img_path = os.path.join(self.image_dir, path.replace('.png', '.png'))
        # img_path = os.path.join(self.image_dir, imgpath)
        img = Image.open(img_path).convert('RGB')

        img = np.array(img)
        
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['data_captured'] = ""
        image['license'] = 0
        image['flickr_url']= ""
        image['coco_url'] = ""
        image['id'] = self.img_id
        if flag==1:
            image['file_name'] =path.replace('.png', '.jpg')
        else:
            image['file_name'] =path.replace('.png', '.png')
        # image['file_name'] = imgpath

        return image

    def _annotation(self, line):
        parts = line.strip().split(',')
        parts = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
        points = parts[:-1]
        text = parts[-1]

        annotation = {}
        annotation['image_id'] = self.img_id
        annotation['segmentation'] = [points]
        annotation['category_id'] = 1
        annotation['id'] = str(self.ann_id)
        annotation['text'] = text

        return annotation


if __name__ == '__main__':
    image_dir = '/data/zixiao_wang/text_det/totaltext/train_images'
    txt_dir = '/data/zixiao_wang/text_det/totaltext/train_gts'
    saved_coco_path = '/data/zixiao_wang/text_det/totaltext/annotation.json'
    
    l2c_train = txt2CoCo(image_dir=image_dir, txt_dir=txt_dir)
    train_instance = l2c_train.to_coco()
    l2c_train.save_coco_json(train_instance, saved_coco_path)
