import os
import argparse
from tqdm import tqdm
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from engine import Engine
from models import build_model
from datasets import build_dataset
from utils.pyt_utils import load_model, get_main_flag
from utils.logger import get_logger
from mmengine.config import Config


def pad(image, mask=None):
    h, w = image.size()[-2:]
    stride = 8
    pad_h = (stride + 1 - h % stride) % stride
    pad_w = (stride + 1 - w % stride) % stride
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0.)
        if mask is not None:
            mask = F.pad(mask, (0, pad_w, 0, pad_h), mode='constant', value=0)
            return image, mask
    return image


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--test-dir", type=str, default=None)
    parser.add_argument("--mask-dir", type=str, default=None)
    parser.add_argument("--variant", type=str, default='standard')
    parser.add_argument("--visualize-dir", type=str, default=None)

    with Engine(custom_parser=parser) as engine:
        main_flag = get_main_flag()

        args = parser.parse_args()
        cfg = Config.fromfile(args.cfg)

        visualize_dir = args.visualize_dir
        if not os.path.exists(visualize_dir):
            if main_flag:
                os.makedirs(visualize_dir)
        time.sleep(1)

        logger = get_logger()
        if main_flag:
            logger.info('Start Test')
            logger.info('Running with argument: \n{}'.format(
                '\n'.join('{}:{}'.format(k,v) for k,v in vars(args).items())))
            logger.info('Running with config: \n{}'.format(cfg._text))

        if args.mask_dir is not None:
            data_class = 'infermask'
        else:
            data_class = 'infer'
        cfg.test_data_cfg['data_class'] = data_class
        cfg.test_data_cfg['data_dir'] = args.test_dir
        cfg.test_data_cfg['anno_dir'] = args.mask_dir
        cfg.test_data_cfg['variant'] = args.variant
        cfg.test_data_cfg['batch_size'] = engine.world_size
        cfg.test_data_cfg['num_workers'] = 1
        dataset = build_dataset(cfg.test_data_cfg)
        test_loader, test_sampler = engine.get_test_loader(dataset, cfg.test_data_cfg)
        if main_flag:
            logger.info('Total {} samples'.format(dataset.__len__()))

        model = build_model(cfg.model_cfg)
        load_model(model, args.resume, ignore_prefix='module.')
        if main_flag:
            logger.info('resume from {}'.format(args.resume))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model = engine.data_parallel(model)
        model.eval()

        for data in tqdm(test_loader, disable= (not main_flag)):
            if len(data)==2:
                image, name =data
            elif len(data)==3:
                image, mask, name =data
                mask = mask.repeat(1,3,1,1)

            with torch.no_grad():
                h,w = image.shape[2:]
                # image = pad(image)
                # pred = model(image.to(device))[0,:,:h,:w]
                image_input = F.interpolate(image, size=cfg.img_size, mode='bilinear', align_corners=False)
                pred = model(image_input.to(device))
                pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)[0]
            pred = torch.clamp(pred, 0.0, 1.0).cpu()
            if len(data)==3:
                pred = pred*mask[0]+image[0]*(1-mask[0])

            pred = pred.numpy()

            save_img = pred.transpose(1,2,0)[:,:,::-1]
            save_img = np.around(save_img*255).astype(np.uint8)
            name = name[0].split('.')[0]+'.png'
            cv2.imwrite(os.path.join(visualize_dir,name), save_img)

        if main_flag:
            logger.info('End Test')

if __name__ == '__main__':
    main()
