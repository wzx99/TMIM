import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import logging
import cv2

from models.base_system import BaseTMIM,BaseSTR
from utils.pyt_utils import get_main_flag


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
    )


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, same_shape=True, **kwargs):
        super(Residual,self).__init__()
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=strides)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                 stride=strides)
        self.batch_norm2d = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if not self.same_shape:
            x = self.conv3(x)
        out = self.batch_norm2d(out + x)
        return F.relu(out)


class ConvWithActivation(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(ConvWithActivation, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x


class DeConvWithActivation(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(DeConvWithActivation, self).__init__()
        self.conv2d = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, norm_layer) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            norm_layer(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class AlignedModule(nn.Module):

    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature = x  # size: 1 > 2
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output


class UperNetAlignHead(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
                 conv3x3_type="conv", fpn_dsn=False):
        super(UperNetAlignHead, self).__init__()

        self.ppm = PSPModule(inplane, norm_layer=norm_layer, out_features=fpn_dim)
        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.fpn_out_align =[]
        self.dsn = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))

            if conv3x3_type == 'conv':
                self.fpn_out_align.append(
                AlignedModule(inplane=fpn_dim, outplane=fpn_dim//2)
                )

            if self.fpn_dsn:
                self.dsn.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
                        norm_layer(fpn_dim),
                        nn.ReLU(),
                        nn.Dropout2d(0.1),
                        nn.Conv2d(fpn_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True)
                    )
                )

        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        if self.fpn_dsn:
            self.dsn = nn.ModuleList(self.dsn)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, conv_out):
        psp_out = self.ppm(conv_out[-1])
        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = self.fpn_out_align[i]([conv_x, f])
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
            if self.fpn_dsn:
                out.append(self.dsn[i](f))
        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        return x


class Pert(nn.Module):

    def __init__(self, iteration=3, tmim=False, **kwargs) -> None:
        super().__init__()

        self.iteration = iteration
        self.tmim=tmim

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 4, 2, 1),
            nn.ReLU()
            )
        self.convb = ConvWithActivation(32, 64, kernel_size=4, stride=2, padding=1)
        self.res1 = Residual(64, 64)
        self.res2 = Residual(64, 128, same_shape=False)
        self.res3 = Residual(128, 256, same_shape=False)
        self.res4 = Residual(256, 256)
        self.res5 = Residual(256, 512, same_shape=False)
        self.res6 = Residual(512, 512)
        self.conv2 = ConvWithActivation(512,512,kernel_size=1)
        self.deconv1 = DeConvWithActivation(512, 256, kernel_size=3, padding=1, stride=2)
        self.deconv2 = DeConvWithActivation(256 * 2, 128, kernel_size=3, padding=1, stride=2)
        self.deconv3 = DeConvWithActivation(128 * 2, 64, kernel_size=3, padding=1, stride=2)
        self.deconv4 = DeConvWithActivation(64 * 2, 32, kernel_size=3, padding=1, stride=2)
        self.deconv5 = DeConvWithActivation(64, 3, kernel_size=3, padding=1, stride=2)
        self.lateral_connection1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1), )
        self.lateral_connection2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(256, 128, kernel_size=1, padding=0, stride=1), )
        self.lateral_connection3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(128, 64, kernel_size=1, padding=0, stride=1), )
        self.lateral_connection4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(64, 32, kernel_size=1, padding=0, stride=1), )

        self.sf_net = UperNetAlignHead(512, num_class=1, norm_layer=nn.BatchNorm2d,
                                         fpn_inplanes=[64, 128, 256, 512], fpn_dim=128, fpn_dsn=False)
        
        if self.tmim:
            self.task_prompt = nn.Embedding(2, 512)

        if get_main_flag():
            logging.info('# model parameters: %.2f M'%(sum(param.numel() for param in self.parameters())/ 1e6))

    def forward(self, images, prompt=None):
        batch_size, row, col = images.size(0), images.size(2), images.size(3)
        if self.tmim:
            prompt = (self.task_prompt(prompt)*(1-prompt).unsqueeze(1)).unsqueeze(2).unsqueeze(3)

        x = images
        h = torch.zeros(batch_size, 32, row, col).cuda()
        c = torch.zeros(batch_size, 32, row, col).cuda()

        for i in range(self.iteration):
            x = torch.cat((images, x), 1)

            # down sample
            x = self.conv0(x) # 32 256
            mask_in = []
            con_x1 = x
            x = self.convb(x) # 64  // 128
            mask_in.append(x)
            x = self.res1(x) # 64
            con_x2 = x
            x = self.res2(x) # 128 // 64
            con_x3 = x
            mask_in.append(x)
            x = self.res3(x) # 256 // 32
            con_x4 = x
            mask_in.append(x)
            x = self.res4(x) #
            x = self.res5(x) # 512 // 16
            mask_in.append(x)

            x = self.res6(x) # 512
            x = self.conv2(x)

            if self.tmim:
                x = x + prompt

            # BRN
            x = self.deconv1(x)
            x = F.interpolate(x,size = con_x4. shape[2:], mode = 'bilinear')
            x = torch.cat([self.lateral_connection1(con_x4), x], dim=1)
            x = self.deconv2(x)

            x = F.interpolate(x,size = con_x3.shape[2:], mode = 'bilinear')
            x = torch.cat([self.lateral_connection2(con_x3), x], dim=1)
            x = self.deconv3(x)
            xo1 = x

            x = F.interpolate(x,size = con_x2.shape[2:], mode = 'bilinear')
            x = torch.cat([self.lateral_connection3(con_x2), x], dim=1)
            x = self.deconv4(x)
            xo2 = x

            x = F.interpolate(x,size = con_x1.shape[2:], mode = 'bilinear')
            x = torch.cat([self.lateral_connection4(con_x1), x], dim=1)
            x = self.deconv5(x)

            # GGN
            mm_temo = self.sf_net(mask_in)
            mm = F.interpolate(mm_temo, size=(row, col), mode='bilinear',
                                    align_corners=True)
            x = F.interpolate(x, size=(row, col), mode='bilinear',
                                    align_corners=True)

            x = mm * x + (1 - mm) * images

        return x


class PertTMIM(BaseTMIM):
    def __init__(self, *args, **kwargs):
        super(PertTMIM, self).__init__()
        self.model = Pert(*args, **kwargs)


class PertSTR(BaseSTR):
    def __init__(self, *args, **kwargs):
        super(PertSTR, self).__init__()
        self.model = Pert(*args, **kwargs)
