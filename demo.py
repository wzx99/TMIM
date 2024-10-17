import os
import argparse
from tqdm import tqdm
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from math import ceil

from engine import Engine
from models import build_model
from datasets import build_dataset
from utils.pyt_utils import load_model, get_main_flag
from utils.logger import get_logger
from mmengine.config import Config


def pad(image, mask=None):
    h, w = image.size()[-2:]
    stride = 128
    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0.)
        if mask is not None:
            mask = F.pad(mask, (0, pad_w, 0, pad_h), mode='constant', value=0)
            return image, mask
    return image


def predict_sliding(net, image, tile_size):
    image_size = image.shape
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = max(int(ceil((image_size[2] - tile_size[0]) / stride) + 1),1)  # strided convolution formula
    tile_cols = max(int(ceil((image_size[3] - tile_size[1]) / stride) + 1),1)
    full_probs = torch.zeros((image_size[0],3, image_size[2], image_size[3]))
    count_predictions = torch.zeros((1, 3, image_size[2], image_size[3]))

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], image_size[3])
            y2 = min(y1 + tile_size[0], image_size[2])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = image[:, :, y1:y2, x1:x2]
            padded_img = pad(img)
            padded_prediction = net(padded_img.cuda(non_blocking=True))
            if isinstance(padded_prediction, list):
                padded_prediction = padded_prediction[0]
            elif isinstance(padded_prediction, dict):
                padded_prediction = padded_prediction['pred']
            prediction = padded_prediction.cpu()[:, :, 0:img.shape[2], 0:img.shape[3]]
            count_predictions[0, :, y1:y2, x1:x2] += 1
            full_probs[:, :, y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

    full_probs /= count_predictions
    return full_probs


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--test-dir", type=str, default=None)
    parser.add_argument("--mask-dir", type=str, default=None)
    parser.add_argument("--variant", type=str, default='standard')
    parser.add_argument("--visualize-dir", type=str, default=None)
    parser.add_argument("--infer-type", type=str, default='resize', choices=['resize', 'slide', 'whole'])

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
        # model.half()
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
                h, w = image.shape[2:]
                if args.infer_type=='resize':
                    image_input = F.interpolate(image, size=cfg.img_size, mode='bilinear', align_corners=False)
                    pred = model(image_input.to(device))
                    pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)[0]
                elif args.infer_type=='slide':
                    pred = predict_sliding(model, image.to(device), cfg.img_size)[0]
                elif args.infer_type == 'whole':
                    image_input = pad(image)
                    print(image_input.shape)
                    pred = model(image_input.to(device))[0, :, :h, :w]  #image_input.half().to(device)
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
