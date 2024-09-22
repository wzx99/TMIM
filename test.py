#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import math
import argparse
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
from scipy import signal, ndimage
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms.functional import resize

from models import build_model
from datasets import build_dataset
from utils.pyt_utils import load_model
from utils.logger import get_logger
from utils.gauss import fspecial_gauss
from mmengine.config import Config


def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2 (images are assumed to be uint8)
    
    This function attempts to mimic precisely the functionality of ssim.m a 
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    size = min(img1.shape[0], 11)
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
  #  import pdb;pdb.set_trace()
    mu1 = signal.fftconvolve(img1, window, mode = 'valid')
    mu2 = signal.fftconvolve(img2, window, mode = 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(img1 * img1, window, mode = 'valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2 * img2, window, mode = 'valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1 * img2, window, mode = 'valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))


def msssim(img1, img2):
    """This function implements Multi-Scale Structural Similarity (MSSSIM) Image 
    Quality Assessment according to Z. Wang's "Multi-scale structural similarity 
    for image quality assessment" Invited Paper, IEEE Asilomar Conference on 
    Signals, Systems and Computers, Nov. 2003 
    
    Author's MATLAB implementation:-
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    """
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2)) / 4.0
    # im1 = img1.astype(np.float64)
    # im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map = ssim(img1, img2, cs_map = True)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(img1, downsample_filter, 
                                                mode = 'reflect')
        filtered_im2 = ndimage.filters.convolve(img2, downsample_filter, 
                                                mode = 'reflect')
        im1 = filtered_im1[: : 2, : : 2]
        im2 = filtered_im2[: : 2, : : 2]

    # Note: Remove the negative and add it later to avoid NaN in exponential.
    sign_mcs = np.sign(mcs[0 : level - 1])
    sign_mssim = np.sign(mssim[level - 1])
    mcs_power = np.power(np.abs(mcs[0 : level - 1]), weight[0 : level - 1])
    mssim_power = np.power(np.abs(mssim[level - 1]), weight[level - 1])
    return np.prod(sign_mcs * mcs_power) * sign_mssim * mssim_power

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--ckpt-name", type=str, default=None)
    parser.add_argument("--test-dir", type=str, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save-log", action="store_true")
    args = parser.parse_args()
    cfg = Config.fromfile(args.cfg)
    
    cfg.snapshot_dir = os.path.join(cfg.snapshot_dir, args.ckpt_name)
    if args.save_log:
        if os.path.isdir(cfg.snapshot_dir):
            log_dir = cfg.snapshot_dir
        else:
            log_dir = os.path.dirname(cfg.snapshot_dir)
        logger = get_logger(log_file=os.path.join(log_dir, 'log.txt'))
    else:
        logger = get_logger()
    logger.info('Start Test')
    
    model_paths = []
    if os.path.isfile(cfg.snapshot_dir):
        model_paths.append(cfg.snapshot_dir)
    else:
        for model_path in os.listdir(cfg.snapshot_dir):
            if model_path.endswith('.pth'):
                model_paths.append(os.path.join(cfg.snapshot_dir,model_path))

    if args.visualize:
        visualize_dir = os.path.join(os.path.dirname(model_paths[0]), 'visualize')
        if not os.path.exists(visualize_dir):
            os.makedirs(visualize_dir)
    else:
        visualize_dir = None
    
    if args.test_dir is not None:
        cfg.test_data_cfg['data_dir'] = args.test_dir
    dataset = build_dataset(cfg.test_data_cfg)

    for model_path in model_paths:
        model = build_model(cfg.model_cfg)
        load_model(model, model_path, ignore_prefix='module.')
        logger.info('resume from {}'.format(model_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)

        sum_psnr = 0
        sum_ssim = 0
        sum_AGE = 0 
        sum_pCEPS = 0
        sum_pEPS = 0
        sum_mse = 0
        count = 0

        for idx in tqdm(range(len(dataset))):
            batch = dataset.__getitem__(idx)
            if len(batch)==3:
                image, background, name =batch
                mask = None
            elif len(batch)==4:
                image, background, mask, name =batch
                mask = mask.repeat(3,1,1)
                # background[mask==0] = image[mask==0]

            with torch.no_grad():
                pred = model.forward_test(image.unsqueeze(0).to(device), mask)[0]
            pred = torch.clamp(pred, 0.0, 1.0).cpu()
            if len(batch)==4:
                pred = pred*mask+image*(1-mask)

            image = image.numpy()
            pred = pred.numpy()

            if args.visualize:
                save_img = pred.transpose(1,2,0)[:,:,::-1]
                save_img = np.around(save_img*255).astype(np.uint8)
                name = name.split('.')[0]+'.png'
                cv2.imwrite(os.path.join(visualize_dir,name), save_img) 

            if args.visualize:
                pred = cv2.imread(os.path.join(visualize_dir,name))[:,:,::-1].transpose(2,0,1)/255.0
            else:
                pred = np.around(pred*255).astype(np.uint8).astype(float)/255.0

            mse = ((background - pred)**2).mean()
            sum_mse += mse
            if mse == 0:
                continue
            count += 1

            psnr = 10 * math.log10(1/mse)
            sum_psnr += psnr

            R = background[0,:, :]
            G = background[1,:, :]
            B = background[2,:, :]

            YGT = .299 * R + .587 * G + .114 * B

            R = pred[0,:, :]
            G = pred[1,:, :]
            B = pred[2,:, :]

            YBC = .299 * R + .587 * G + .114 * B
            Diff = abs(np.array(YBC*255) - np.array(YGT*255)).round().astype(np.uint8)
            AGE = np.mean(Diff)

            mssim = msssim(np.array(YGT*255), np.array(YBC*255))
            sum_ssim += mssim

            threshold = 20
            Errors = Diff > threshold
            EPs = sum(sum(Errors)).astype(float)
            pEPs = EPs / float(background.shape[1]*background.shape[2])
            sum_pEPS += pEPs

            structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            sum_AGE+=AGE
            erodedErrors = ndimage.binary_erosion(Errors, structure).astype(Errors.dtype)
            CEPs = sum(sum(erodedErrors))
            pCEPs = CEPs / float(background.shape[1]*background.shape[2])
            sum_pCEPS += pCEPs

        logger.info('average mse: {}'.format(sum_mse / count))
        logger.info('average psnr: {}'.format(sum_psnr / count))
        logger.info('average ssim: {}'.format(sum_ssim / count))
        logger.info('average AGE: {}'.format(sum_AGE / count))
        logger.info('average pEPS: {}'.format(sum_pEPS / count))
        logger.info('average pCEPS: {}'.format(sum_pCEPS / count))


if __name__ == '__main__':
    main()
