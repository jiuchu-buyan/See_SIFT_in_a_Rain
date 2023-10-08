#!/usr/bin/env python3
# import megengine as mge
# import megengine.module as nn
# import megengine.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import cv2
import copy
import numpy as np
# def conv3x3(in_chn, out_chn, bias=True):
#     layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
#     return layer


# def conv_down(in_chn, out_chn, bias=False):
#     layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
#     return layer


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        n_inch = 3
        n_outch = 3
        kernel_size = 5

        self.sub_mean = MeanShift(1, sign=-1)
        self.add_mean = MeanShift(1, sign=1)

        # define head module
        m_head = [nn.Conv2d(n_inch, 32, kernel_size=kernel_size, padding=(kernel_size//2), bias=True)]

        # define body module
        m_body = [Atten(32, 32, kernel_size=kernel_size)]
        m_body.append(Atten(32, 32, kernel_size=kernel_size))

        m_body.append(Atten(32, 64, kernel_size=kernel_size))
        m_body.append(Atten(64, 64, kernel_size=kernel_size))
        m_body.append(Atten(64, 64, kernel_size=kernel_size))
        m_body.append(Atten(64, 64, kernel_size=kernel_size))
        m_body.append(Atten(64, 64, kernel_size=kernel_size))
        m_body.append(Atten(64, 64, kernel_size=kernel_size))

        m_body.append(Atten(64, 128, kernel_size=kernel_size))
        m_body.append(Atten(128, 128, kernel_size=kernel_size))
        m_body.append(Atten(128, 128, kernel_size=kernel_size))
        m_body.append(Atten(128, 128, kernel_size=kernel_size))

        m_body.append(Atten(128, 256, kernel_size=kernel_size))
        m_body.append(Atten(256, 256, kernel_size=kernel_size))
        m_body.append(Atten(256, 256, kernel_size=kernel_size))
        m_body.append(Atten(256, 256, kernel_size=kernel_size))
        #
        # m_body.append(Atten(256, 512, kernel_size=kernel_size))
        # m_body.append(Atten(512, 512, kernel_size=kernel_size))
        #
        # m_body.append(Atten(512, 256, kernel_size=kernel_size))
        # m_body.append(Atten(256, 256, kernel_size=kernel_size))
        # m_body.append(Atten(256, 256, kernel_size=kernel_size))
        # m_body.append(Atten(256, 256, kernel_size=kernel_size))

        m_body.append(Atten(256, 128, kernel_size=kernel_size))
        m_body.append(Atten(128, 128, kernel_size=kernel_size))
        m_body.append(Atten(128, 128, kernel_size=kernel_size))
        m_body.append(Atten(128, 128, kernel_size=kernel_size))

        m_body.append(Atten(128, 64, kernel_size=kernel_size))
        m_body.append(Atten(64, 64, kernel_size=kernel_size))
        m_body.append(Atten(64, 64, kernel_size=kernel_size))
        m_body.append(Atten(64, 64, kernel_size=kernel_size))
        m_body.append(Atten(64, 64, kernel_size=kernel_size))
        m_body.append(Atten(64, 64, kernel_size=kernel_size))

        m_body.append(Atten(64, 32, kernel_size=kernel_size))
        m_body.append(Atten(32, 32, kernel_size=kernel_size))
        m_body.append(nn.Conv2d(32, 32, kernel_size=kernel_size, padding=(kernel_size//2), bias=True))

        # define tail module
        m_tail = [
           # common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(32, n_outch, kernel_size=kernel_size, padding=(kernel_size//2), bias=True)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):


        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x

    # 定义注意力模块
class Atten(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size):
        super(Atten, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_feats, out_feats, kernel_size=kernel_size, padding=(kernel_size//2), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_feats, out_feats, kernel_size=kernel_size, padding=(kernel_size//2), bias=True),
            nn.ReLU(inplace=True))

        self.shortcut = nn.Conv2d(in_feats, out_feats, kernel_size=1, bias=True)
        self.spatialattention = SpatialAttention(kernel_size=5, in_size=out_feats)
        self.channelattention = ChannelAttention(in_size=out_feats)
        self.gridientattention = GradientAttention(in_size=out_feats)

    def forward(self, input):
        out = self.block(input)
        out = self.channelattention(out)
        # out = self.spatialattention(out)
        out = self.gridientattention(out)
        sc = self.shortcut(input)
        out += sc
        return out

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

class UNetConvBlockconv(nn.Module):

    # def __init__(self, in_size, out_size, downsample, relu_slope):
    def __init__(self, in_size, out_size, downsample, relu_slope, kernel):
        super(UNetConvBlockconv, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope))
        self.block2 = nn.Sequential(
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope))
        self.downsample = downsample
        if downsample:
            # self.downsample = conv_down(out_size, out_size, bias=False)
            self.downsample = nn.Conv2d(out_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)
        self.spatialattention = SpatialAttention(kernel_size=kernel, in_size=out_size)
        self.channelattention = ChannelAttention(in_size=out_size)

    def forward(self, x):
        # global i
        out = self.block1(x)
        out = self.block2(out)
        out = self.channelattention(out)
        out = self.spatialattention(out)
        sc = self.shortcut(x)
        out = sc + out
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out

class ChannelAttention(nn.Module):
    def __init__(self, in_size):
        super(ChannelAttention, self).__init__()
        self.ca = nn.Sequential(
            nn.Conv1d(in_size, in_size, kernel_size=5, padding="same"),
            # nn.LeakyReLU(0.2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_size, in_size, kernel_size=5, padding="same"),
            nn.Sigmoid())

    def forward(self, x):
        pooled = global_spectral_pool(x)
        x_ca = self.ca(pooled)
        x_ca = x_ca.unsqueeze(-1)
        x = x * x_ca
        return x

def global_spectral_pool(x):
    global_pool = torch.mean(x, dim=(2, 3))
    global_pool = global_pool.unsqueeze(-1)
    return global_pool

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size, in_size):
        super(SpatialAttention, self).__init__()
        self.in_size = in_size
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_size, 1, kernel_size=(1, 1)),
            # nn.LeakyReLU(0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=int((kernel_size - 1) / 2)),
            # nn.LeakyReLU(0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=int((kernel_size - 1) / 2)))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sa = self.conv(x)
        sa = self.sigmoid(sa)
        out = x * sa
        return out

def conv_grid(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    # n_channels, kw, kh = kernel.shape
    img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return F.conv2d(img, kernel, groups=n_channels)


class GradientAttention(nn.Module):
    def __init__(self, in_size,kernel_size=5):
        super(GradientAttention, self).__init__()
        self.in_size = in_size
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, padding=2, bias=True)
        self.conv2 = nn.Sequential(
        nn.Conv2d(1, 1, kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=int((kernel_size - 1) / 2)),
        nn.ReLU(inplace=True),
        nn.Conv2d(1, 1, kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=int((kernel_size - 1) / 2)),
        nn.ReLU(inplace=True))
        self.conv3 = nn.Conv2d(in_channels=self.in_size,out_channels=1,kernel_size=5,padding=2, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        #print(input.shape)
        n_channels = input.shape[1]

        kernel_x = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]], dtype=np.float)
        kernel_y = np.array([[+1, +2, +1], [0, 0, 0], [-1, -2, -1]], dtype=np.float)

        kernel_x = np.tile(kernel_x, (n_channels, 1, 1))
        kernel_y = np.tile(kernel_y, (n_channels, 1, 1))

        kernel_x = torch.FloatTensor(kernel_x[:, None, :, :])
        kernel_y = torch.FloatTensor(kernel_y[:, None, :, :])

        kernel_x = kernel_x.cuda()
        kernel_y = kernel_y.cuda()
        grid_x = conv_grid(input, kernel_x)
        grid_y = conv_grid(input, kernel_y)

        # grid_x_conv1 = nn.Conv2d(n_channels, 1, kernel_size=(1, 1))(grid_x)
        grid_x_conv1 = self.conv3(grid_x)
        gx = self.conv2(grid_x_conv1)


        # grid_y_conv1 = nn.Conv2d(n_channels, 1, kernel_size=(1, 1))(grid_y)
        grid_y_conv1 = self.conv3(grid_y)
        gy = self.conv2(grid_y_conv1)

        out = torch.cat([gx, gy], dim=1)
        out = self.conv1(out)
        out = torch.mul(input, self.sigmoid(out))

        return out

class UNetUpBlockconv(nn.Module):

    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlockconv, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlockconv(in_size, out_size, False, relu_slope, kernel=5)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


if __name__ == "__main__":
    import numpy as np
    a = Net()

    print(a)
    im = torch.tensor(np.random.randn(1, 3, 128, 128).astype(np.float32))
    print(a(im))

