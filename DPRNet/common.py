import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift_48(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        # super(MeanShift, self).__init__(3, 3, kernel_size=1)
        super(MeanShift_48, self).__init__(48, 48, kernel_size=1)

        rgb_mean_48 = (0.4488, 0.4371, 0.4040)
        rgb_std_48 = (1.0, 1.0, 1.0)
        for i in range(15):
            rgb_mean_48 = rgb_mean_48 + rgb_mean
            rgb_std_48 = rgb_std_48 + rgb_std

        rgb_mean = rgb_mean_48
        rgb_std = rgb_std_48


        std = torch.Tensor(rgb_std)

        # self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.weight.data = torch.eye(48).view(48, 48, 1, 1) / std.view(48, 1, 1, 1)

        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)

        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)

        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, in_feats, out_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(in_feats, out_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(out_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        # res += x

        return nn.ReLU(inplace=True)(res+x)


# 定义SE模块
class SENet(nn.Module):
    def __init__(self, conv, in_feats, out_feats, kernel_size, reduce=16,
                 bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(SENet, self).__init__()
        self.rb1 = ResBlock(conv, in_feats, out_feats,kernel_size,bias=bias, bn=bn, act=act, res_scale=res_scale)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_feats, out_feats, kernel_size=1),
            nn.BatchNorm2d(out_feats)
        )

        # self.gp = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(nn.Linear(out_feats, out_feats // reduce),
                                nn.ReLU(inplace=True),
                                nn.Linear(out_feats // reduce, out_feats),
                                nn.Sigmoid())
    def forward(self, input):

        x = self.rb1(input)
        b, c, _, _ = x.size()
        y = nn.AvgPool2d(x.size()[2])(x)
        y = y.view(y.shape[0], -1)

        # y = self.gp(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        out = y + input
        #out = y + self.shortcut(y)
        return out



class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

