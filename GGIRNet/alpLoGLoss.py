import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as fnn
from torch.autograd import Variable
import cv2


# from sympy import *




def build_laplace_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size, 0:size].T)
    laplace = lambda x: ((x - size // 2) ** 2 - sigma**2) * np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
    # 要*sigma**2吗？
    kernel = np.sum(laplace(grid), axis=2) * sigma**2
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w),
    # and since we have depth-separable convolution we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    # print(cuda)
    if cuda:
        kernel = kernel.cuda(2)
    return Variable(kernel, requires_grad=False)


def conv_lap(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = fnn.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return fnn.conv2d(img, kernel, groups=n_channels)







class ALPLoGLoss(nn.Module):
    def __init__(self, k_size=5, sigma=1.6):
        super(ALPLoGLoss, self).__init__()
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None

    def forward(self, input, target):
        # input shape :[B, N, C, H, W]
        if len(input.shape) == 5:
            B, N, C, H, W = input.size()
            input = input.view(-1, C, H, W)
            target = target.view(-1, C, H, W)
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_laplace_kernel(
                size=self.k_size, sigma=self.sigma,
                n_channels=input.shape[1], cuda=input.is_cuda
            )

        sigma01 = 1.6
        sigma02 = pow(2, 1 / 2) * 1.6
        sigma03 = pow(2, 1) * 1.6
        sigma04 = pow(2, 3 / 2) * 1.6

        a0 = -0.2464
        a1 = 0.4934
        a2 = -0.2717
        a3 = 0.0140

        b0 = 2.5021
        b1 = -4.5636
        b2 = 2.0108
        b3 = 0.1549

        c0 = -8.2007
        c1 = 12.9824
        c2 = -4.0449
        c3 = -1.0565

        d0 = 8.6432
        d1 = -10.8424
        d2 = 2.1204
        d3 = 1.3886

        # dst = cv2.Laplacian(gaussian, cv2.CV_16S, ksize=3)
        # LOG = cv2.convertScaleAbs(dst)
        # 构造laplace图像
        lapX01 = conv_lap(input,build_laplace_kernel(size=self.k_size, sigma=sigma01, n_channels=input.shape[1], cuda=input.is_cuda))
        lapX02 = conv_lap(input, build_laplace_kernel(size=self.k_size, sigma=sigma02,
                                                                       n_channels=input.shape[1], cuda=input.is_cuda))
        lapX03 = conv_lap(input, build_laplace_kernel(size=self.k_size, sigma=sigma03,
                                                                       n_channels=input.shape[1], cuda=input.is_cuda))

        lapX04 = conv_lap(input, build_laplace_kernel(size=self.k_size, sigma=sigma04,
                                                                       n_channels=input.shape[1], cuda=input.is_cuda))

        lapY01 = conv_lap(target, build_laplace_kernel(size=self.k_size, sigma=sigma01,
                                                                        n_channels=input.shape[1],
                                                                        cuda=input.is_cuda))
        lapY02 = conv_lap(target, build_laplace_kernel(size=self.k_size, sigma=sigma02,
                                                                        n_channels=input.shape[1],
                                                                        cuda=input.is_cuda))
        lapY03 = conv_lap(target, build_laplace_kernel(size=self.k_size, sigma=sigma03,
                                                                        n_channels=input.shape[1],
                                                                        cuda=input.is_cuda))

        lapY04 = conv_lap(target, build_laplace_kernel(size=self.k_size, sigma=sigma04,
                                                                       n_channels=input.shape[1],
                                                                       cuda=input.is_cuda))

        # ALP多项式系数
        imgX03 = a0 * lapX01 + a1 * lapX02 + a2 * lapX03 + a3 * lapX04
        imgY03 = a0 * lapY01 + a1 * lapY02 + a2 * lapY03 + a3 * lapY04

        imgX02 = b0 * lapX01 + b1 * lapX02 + b2 * lapX03 + b3 * lapX04
        imgY02 = b0 * lapY01 + b1 * lapY02 + b2 * lapY03 + b3 * lapY04

        imgX01 = c0 * lapX01 + c1 * lapX02 + c2 * lapX03 + c3 * lapX04
        imgY01 = c0 * lapY01 + c1 * lapY02 + c2 * lapY03 + c3 * lapY04

        pyr_input = [imgX03, imgX02, imgX01]
        pyr_target = [imgY03, imgY02, imgY01]

        # 系数相差尽可能小
        return sum(fnn.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
