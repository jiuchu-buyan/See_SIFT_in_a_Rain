import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math


def gaussian(window_size, sigma):
	gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
	return gauss/gauss.sum()


def create_window(window_size, channel, sigma):
	_1D_window = gaussian(window_size, sigma).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
	return window


def gaussBlur(img1, img2, sigma):
	(_, channel, _, _) = img1.size()
	window_size = 3
	# pad = int(window_size/2)
	window = create_window(window_size, channel, sigma).to(img1.device)
	out_img1 = F.conv2d(img1, window, padding = 1, groups = channel)
	out_img2 = F.conv2d(img2, window, padding = 1, groups = channel)

	return out_img1, out_img2