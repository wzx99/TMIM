import os
import logging
import cv2
import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
from skimage.metrics import peak_signal_noise_ratio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.loss import RecoveryLoss
from utils.pyt_utils import all_reduce_tensor


@dataclass
class BatchResult:
    num_samples: Tensor
    psnr: Tensor
    mse: Tensor


class BaseSystem(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch, batch_idx=0, mode='test'):
        if mode=='train':
            return self._train_step(batch, batch_idx)
        elif mode=='val':
            return self._eval_step(batch, True)
        elif mode=='test':
            return self.forward_test(batch)

    def _eval_step(self, batch, validation):
        images, backgrounds = batch[0], batch[1]
        bs = images.shape[0]

        preds = self.forward_test(images)

        backgrounds = backgrounds.cpu().numpy().astype(np.float32)
        preds = preds.cpu().numpy().astype(np.float32)
        psnr = 0
        mse = 0
        for i in range(bs):
            background = backgrounds[i]
            pred = preds[i]
            psnr += peak_signal_noise_ratio(background, pred, data_range=1.)
            mse += ((background-pred)**2).mean()
        return dict(output=BatchResult(bs, psnr, mse))

    @staticmethod
    def _aggregate_results(outputs) -> Tuple[float, float, float]:
        if not outputs:
            return 0., {}
        total_psnr = torch.tensor(0.0).cuda()
        total_mse = torch.tensor(0.0).cuda()
        total_size = torch.tensor(0.0).cuda()
        for result in outputs:
            result = result['output']
            total_psnr += result.psnr
            total_mse += result.mse
            total_size += result.num_samples
            
        total_psnr = all_reduce_tensor(total_psnr, norm=False)
        total_size = all_reduce_tensor(total_size, norm=False)
        total_mse = all_reduce_tensor(total_mse, norm=False)
        
        psnr = (total_psnr / total_size).cpu().numpy()
        mse = (total_mse / total_size).cpu().numpy()
        return -mse*10000, {'psnr':psnr, 'mse':mse*100}

    def validation_epoch_end(self, outputs):
        return self._aggregate_results(outputs)


class BaseTMIM(BaseSystem):
    def __init__(self):
        super().__init__()
        self.model = None
        self.loss = RecoveryLoss()

    def forward_test(self, x, masks=None, *args, **kwargs):
        prompt = torch.ones(x.shape[0]).long().cuda(non_blocking=True)
        return self.model(x, prompt)
        # masks = 1-masks.cuda(non_blocking=True)
        # images_mask = x*masks
        # images_mask[images_mask<0.04] = 0
        # prompt = torch.zeros(x.shape[0]).long().cuda(non_blocking=True)
        # return self.model(images_mask, prompt)*(1-masks)+x*masks

    def _train_step(self, batch, batch_idx):
        images, masks, masks2 = batch[0], batch[1], batch[2]
        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)
        masks = 1-masks  # bg=1,text=0
        masks2 = masks2.cuda(non_blocking=True)
        masks2 = 1-masks2  # bg=1,text=0

        mask_inpainting = masks2
        images_mask = images*mask_inpainting*masks

        totoal_input = torch.cat((images_mask, images), dim=0)
        prompt = torch.cat((torch.zeros(images_mask.shape[0]), torch.ones(images.shape[0]))).long().cuda(non_blocking=True)
        totoal_output = self.model(totoal_input, prompt)
        pred_img_masked = totoal_output[:images_mask.shape[0]]
        pred_img = totoal_output[images_mask.shape[0]:]
        
        pred_img_masked_bg = pred_img_masked*masks+images*(1-masks)
        loss_mask = self.loss(pred_img_masked_bg, images, mask_inpainting*masks) 

        ss_label = torch.clamp(pred_img_masked.detach()*(1-masks)+images*masks, 0.0, 1.0)
        loss_ss = self.loss(pred_img, ss_label, masks)*min(1,batch_idx/20000)

        # if np.random.randint(1600)==1:
        #     logging.info('loss_ss:{}, loss_mask:{}'.format(loss_ss.data, loss_mask.data))
        #
        #     save_img = images_mask.detach().cpu().numpy()[0].transpose((1,2,0))
        #     save_img = (save_img*255).astype(np.uint8)[:,:,::-1]
        #     cv2.imwrite('./inp.jpg', save_img)
        #
        #     save_img = pred_img.detach().cpu().numpy()[0].transpose((1,2,0))
        #     save_img = (save_img*255).astype(np.uint8)[:,:,::-1]
        #     cv2.imwrite('./pred.jpg', save_img)
        #
        #     save_img = pred_img_masked.detach().cpu().numpy()[0].transpose((1,2,0))
        #     save_img = (save_img*255).astype(np.uint8)[:,:,::-1]
        #     cv2.imwrite('./pred_masked.jpg', save_img)
        #
        #     save_img = ss_label.detach().cpu().numpy()[0].transpose((1,2,0))
        #     save_img = (save_img*255).astype(np.uint8)[:,:,::-1]
        #     cv2.imwrite('./label.jpg', save_img)

        return loss_ss+loss_mask


class BaseSTR(BaseSystem):
    def __init__(self):
        super().__init__()
        self.model = None
        self.loss = RecoveryLoss(ft=True)

    def forward_test(self, x, *args, **kwargs):
        if self.model.tmim == True:
            prompt = torch.ones(x.shape[0]).long().cuda(non_blocking=True)
            return self.model(x, prompt)
        else:
            return self.model(x)

    def _train_step(self, batch, batch_idx):
        images, backgrounds, masks = batch[0], batch[1], batch[2]
        images = images.cuda(non_blocking=True)
        backgrounds = backgrounds.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)
        masks = 1-masks  #bg=1, text=0

        if self.model.tmim == True:
            prompt = torch.ones(images.shape[0]).long().cuda(non_blocking=True)
            pred_bg = self.model(images, prompt)
        else:
            pred_bg = self.model(images)
        loss_bg = self.loss(pred_bg, backgrounds, masks)

        # if np.random.randint(800)==1:
        #     save_img = images.detach().cpu().numpy()[0].transpose((1,2,0))
        #     save_img = (save_img*255).astype(np.uint8)[:,:,::-1]
        #     cv2.imwrite('./inp.jpg', save_img)
        #
        #     save_img = pred_bg.detach().cpu().numpy()[0].transpose((1,2,0))
        #     save_img = (save_img*255).astype(np.uint8)[:,:,::-1]
        #     cv2.imwrite('./pred.jpg', save_img)
        #
        #     save_img = backgrounds.detach().cpu().numpy()[0].transpose((1,2,0))
        #     save_img = (save_img*255).astype(np.uint8)[:,:,::-1]
        #     cv2.imwrite('./label.jpg', save_img)

        return loss_bg