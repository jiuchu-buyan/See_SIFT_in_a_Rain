import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as fnn
from torch.autograd import Variable
import cv2


# from sympy import *

def conv_grid(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    # n_channels, kw, kh = kernel.shape
    img = fnn.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return fnn.conv2d(img, kernel, groups=n_channels)

class GridLoss(nn.Module):
    def __init__(self):
        super(GridLoss, self).__init__()
        # self.sub_mean = common.MeanShift(args.rgb_range)
        # self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        
    def forward(self, input, target):
        # input shape :[B, N, C, H, W]
        cuda = True
        if len(input.shape) == 5:
            B,N,C,H,W = input.size()
            input = input.view(-1, C, H , W)
            target = target.view(-1, C, H, W)
            n_channels = C
        else:
            n_channels = 3

        kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype = np.float)
        kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype = np.float) 

        kernel_x = np.tile(kernel_x, (n_channels, 1, 1))
        kernel_y = np.tile(kernel_y, (n_channels, 1, 1))

        kernel_x = torch.FloatTensor(kernel_x[:, None, :, :])
        kernel_y = torch.FloatTensor(kernel_y[:, None, :, :])
        if cuda:
            kernel_x = kernel_x.cuda(0)
            kernel_y = kernel_y.cuda(0)



        gx_input = conv_grid(input, kernel_x)
        gy_input = conv_grid(input, kernel_y)

        gx_target = conv_grid(target, kernel_x)
        gy_target = conv_grid(target, kernel_y)

        # L1Loss = fnn.l1_loss(input, target)


        grid_loss = fnn.l1_loss(gx_input, gx_target)+fnn.l1_loss(gy_input, gy_target)
        # print("gridLoss/L1Loss={}".format(grid_loss/L1Loss))

      
        return grid_loss
