import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from torch import nn
from torchvision import models
import logging

from .focal_frequency_loss import FocalFrequencyLoss as FFL


#VGG16 feature extract
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        # vgg16 = models.vgg16(pretrained=False)
        # vgg16.load_state_dict(torch.load('pretrained_models/vgg16-397923af.pth'))
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        self.mean_ = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
        self.std_ = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def do_normalize_inputs(self, x):
        return (x - self.mean_.to(x.device)) / self.std_.to(x.device)

    def forward(self, image):
        image = self.do_normalize_inputs(image)
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w) / (ch * h * w)**0.5
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t)
    return gram

def dice_loss(input, target):
    # input = torch.sigmoid(input)

    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)

    input = input
    target = target

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = torch.mean(d)
    return 1 - dice_loss

def L2(f_):
    return (((f_ ** 2).sum(dim=1)) ** 0.5).reshape(f_.shape[0], 1, f_.shape[2], f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat / tmp
    feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
    return torch.einsum('icm,icn->imn', [feat, feat])  # non-local

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S)) ** 2) / ((f_T.shape[-1] * f_T.shape[-2]) ** 2) / f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

class RecoveryLoss(torch.nn.Module):
    def __init__(self, window_size=5, size_average=True, ft=False):
        super(RecoveryLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.l1 = nn.L1Loss()
        self.extractor = VGG16FeatureExtractor()
        self.scale = [0.5, 0.25]
        self.criterion = sim_dis_compute
        self.ffl = FFL(ft=ft, loss_weight=1.0, alpha=1.0)

    def forward(self, img, tgt, mask_bg=None):
        (_, channel, _, _) = img.size()

        if channel == self.channel and self.window.data.type() == img.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img.is_cuda:
                window = window.cuda(img.get_device())
            window = window.type_as(img)
            
            self.window = window
            self.channel = channel

        # rs loss
        if mask_bg is not None:
            holeLoss = 13 * self.l1((1 - mask_bg) * tgt, (1 - mask_bg) * img)
            validAreaLoss = 2 * self.l1(mask_bg * tgt, mask_bg * img)
            rs_loss = holeLoss + validAreaLoss
        else:
            rs_loss = 15*self.l1(img, tgt)

        f_loss = 15*self.ffl(img, tgt)

        feat_output = self.extractor(tgt)
        feat_gt = self.extractor(img)
        prcLoss = 0.0
        for i in range(3):
            prcLoss += 0.01 * self.l1(feat_output[i], feat_gt[i])
        styleLoss = 0.0
        for i in range(3):
            styleLoss += 120 * self.l1(gram_matrix(feat_output[i]),
                                       gram_matrix(feat_gt[i]))
        loss_vgg = styleLoss + prcLoss
        
        # gs loss
        gs_loss = 0
        for i in range(len(self.scale)):
            feat_S = feat_output[i]
            feat_T = feat_gt[i]
            feat_T.detach()
            total_w, total_h = feat_T.shape[2], feat_T.shape[3]
            patch_w, patch_h = int(total_w * self.scale[i]), int(total_h * self.scale[i])
            maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0,
                                   ceil_mode=True)  # change
            gs_loss_temp = self.criterion(maxpool(feat_S), maxpool(feat_T))
            gs_loss += gs_loss_temp
        rg_loss = gs_loss + rs_loss

        ssim_loss = 1 - _ssim(img, tgt, window, self.window_size, channel, self.size_average)

        loss_total_all = ssim_loss + rg_loss + loss_vgg + f_loss
        return loss_total_all

