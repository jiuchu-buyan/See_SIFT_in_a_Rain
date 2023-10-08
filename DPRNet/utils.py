

# --- Imports --- #
import os
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from PIL import Image
from networkx.drawing.tests.test_pylab import plt
from skimage import measure
from sklearn.decomposition import PCA
import numpy as np
from torchvision.transforms import  ToTensor,Normalize,Compose
import gc
from PIL import Image
from scipy.io import savemat

def to_psnr(pred_image, gt):
    mse = F.mse_loss(pred_image, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(pred_image, gt):
    pred_image_list = torch.split(pred_image, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    pred_image_list_np = [pred_image_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    ssim_list = [measure.compare_ssim(pred_image_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(pred_image_list))]

    return ssim_list


def validation(net, val_data_loader, device, category, exp_name, save_tag=False):
    """
    :param net: Gatepred_imageNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: derain or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            input_im, gt, image_name = val_data
            input_im = input_im.to(device)
            gt = gt.to(device)
            pred_image = net(input_im)

        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(pred_image, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(to_ssim_skimage(pred_image, gt))
        #print(image_name,psnr_list[-1],ssim_list[-1])

        # --- Save image --- #
        if save_tag:
            save_image(pred_image, image_name, category, exp_name)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim





def validation_real(net, val_data_loader, device, category, exp_name, save_tag=False):
    """
      :param net: Gatepred_imageNet
      :param val_data_loader: validation loader
      :param device: The GPU that loads the network
      :param category: derain or outdoor test dataset
      :param save_tag: tag of saving image or not
      :return: average PSNR value
      """
   # kk=np.zeros((20,32,42*62))
    # 不计算psnr,ssim，使用真实数据集
    for batch_id, val_data in enumerate(val_data_loader):
        with torch.no_grad():
            input_im, image_name = val_data
            input_im = input_im.to(device)
            pred_image = net(input_im)


        # --- Save image --- #
        if save_tag:
            save_image(pred_image, image_name, category, exp_name)

def to_mse_gradient(pred_image, gt):
    mse = F.mse_loss(pred_image, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    return mse_list


def save_image(pred_image, image_name, category, exp_name):
    pred_image_images = torch.split(pred_image, 1, dim=0)
    batch_num = len(pred_image_images)

    for ind in range(batch_num):
        image_name_1 = image_name[ind].split('/')[-1]
        utils.save_image(pred_image_images[ind], './{}_results/{}/{}'.format(category, exp_name, image_name_1[:-3] + 'jpg'))


def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim))

    # --- Write the training log --- #
    with open('./training_log/{}_log.txt'.format(category), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)



def adjust_learning_rate(optimizer, epoch, category, lr_decay=0.5):

    # --- Decay learning rate --- #
    step = [80, 100, 120, 140]

    # if not epoch % step and epoch > 0:
    if epoch in step:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))
    # step = 50 if category == 'dehazy' else 2
    #
    # if not epoch % step and epoch > 0:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] *= lr_decay
    #         print('Learning rate sets to {}.'.format(param_group['lr']))
    # else:
    #     for param_group in optimizer.param_groups:
    #         print('Learning rate sets to {}.'.format(param_group['lr']))





def validation_gradient_real(model, val_data_loader, device, category, exp_name, save_tag=True):
    # """
    # :param net: Gatepred_imageNet
    # :param val_data_loader: validation loader
    # :param device: The GPU that loads the network
    # :param category: derain or outdoor test dataset
    # :param save_tag: tag of saving image or not
    # :return: average PSNR value
    # """

    for batch_id, val_data in enumerate(val_data_loader):
        input_im, image_name = val_data
        input_im = input_im.to(device)
        # ---  the gradient1 network --- #
        model1 = model[0]
        pred_image1 = model1(input_im)
        # ---  the gradient2 network --- #
        model2 = model[1]
        pred_image2 = model2(input_im)
        # ---  the gradient3 network --- #
        model3 = model[2]
        pred_image3 = model3(input_im)
        # ---  the gradient4 network --- #
        model4 = model[3]
        pred_image4 = model4(input_im)
        # ---  the gradient5 network --- #
        model5 = model[4]
        pred_image5 = model5(input_im)

    # --- Save image --- #
        if save_tag:
            save_image_gradient(pred_image5, image_name, category, exp_name, 'GAUSS/05')
            save_image_gradient(pred_image4, image_name, category, exp_name, 'GAUSS/04')
            save_image_gradient(pred_image3, image_name, category, exp_name, 'GAUSS/03')
            save_image_gradient(pred_image2, image_name, category, exp_name, 'GAUSS/02')
            save_image_gradient(pred_image1, image_name, category, exp_name, 'GAUSS/01')

        torch.cuda.empty_cache()

def save_image_gradient(pred_image, image_name, category, exp_name, file_name):
    pred_image_images = torch.split(pred_image, 1, dim=0)
    batch_num = len(pred_image_images)
    if os.path.exists('./{}_results/{}_0609/{}/'.format(category, exp_name, file_name)) == False:
        os.makedirs('./{}_results/{}_0609/{}/'.format(category, exp_name, file_name))

    for ind in range(batch_num):
        image_name_1 = image_name[ind].split('/')[-1]
        save_image_(pred_image_images[ind],
                          './{}_results/{}_0609/{}/{}'.format(category, exp_name,file_name, image_name_1[:-3] + 'jpg'))

    #     img=pred_image_images[ind].squeeze()
    #     # img=img.permute(1,2,0)
    #     img = img.cpu().numpy()
    #     img = np.array(img)* 255
    #     # img = (img * 255).round().astype(np.uint8)
    #     img = Image.fromarray(img.astype(np.uint8))
    #     img.save('./{}_results/{}/{}/{}'.format(category, exp_name,file_name, image_name_1[:-3] + 'png'),quality=100)
    #     # gc.collect()
    # # torch.cuda.empty_cache()



def save_image_(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename,quality=100)

def adjust_learning_rate_1(optimizer, epoch, category, lr_decay=0.5):

    # --- Decay learning rate --- #
    step1 = 30 if category == 'derain' else 2



    if not epoch % step1 and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))


def validation_0413(net,val_data_loader, device, category, exp_name, save_tag=False):
    """
    :param net: Gatepred_imageNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: derain or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    psnr_list = []
    ssim_list = []
    gxpsnr_list = []
    gypsnr_list = []
    gxssim_list = []
    gyssim_list = []
    gradient_mse_list = []


    for batch_id, val_data in enumerate(val_data_loader):
        with torch.no_grad():
            input_im, gt, image_name = val_data
            input_im = input_im.to(device)
            gt = gt.to(device)

            pred_image= net(input_im)

        sobelx, sobely,gradient = get_gradient(pred_image)
        gtsobelx, gtsobely ,gtgradient= get_gradient(gt)

        gradient_mse_list.extend(to_mse_gradient(gradient, gtgradient))

        gxpsnr_list.extend(to_psnr(sobelx, gtsobelx))
        gypsnr_list.extend(to_psnr(sobely, gtsobely))
        gxssim_list.extend(to_ssim_skimage(sobelx, gtsobelx))
        gyssim_list.extend(to_ssim_skimage(sobely, gtsobely))

#
        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(pred_image,gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(to_ssim_skimage(pred_image, gt))

        # --- Save image --- #
        if save_tag:
            save_image(pred_image, image_name, category, exp_name)


    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    avr_gxpsnr = sum(gxpsnr_list) / len(gxpsnr_list)
    avr_gypsnr = sum(gypsnr_list) / len(gypsnr_list)
    avr_gxssim = sum(gxssim_list) / len(gxssim_list)
    avr_gyssim = sum(gyssim_list) / len(gyssim_list)
    avr_gradient_mse = sum(gradient_mse_list) / len(gradient_mse_list)
    return avr_psnr, avr_ssim,avr_gxpsnr,avr_gypsnr,avr_gxssim,avr_gyssim,avr_gradient_mse


def get_gradient(image):
    pic = image.cpu().detach().numpy()
    gradientx = []
    gradienty = []
    gradient = []


    for i in range(pic.shape[0]):
        # img = Image.fromarray(pic[i])
        img = np.squeeze(pic[i])*255.
        scharrx1 = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        scharry1 = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        scharr = 0.5*scharrx1+0.5*scharry1

        scharrx = cv2.convertScaleAbs(scharrx1)
        scharry = cv2.convertScaleAbs(scharry1)

        # gx = cv2.Sobel(img, -1, dx=1, dy=0)  # x方向的
        # gy= cv2.Sobel(img, -1, dx=0, dy=1)  # y方向的
        gx = np.array(scharrx)
        gy = np.array(scharry)
        scharr = np.array(scharr)
        transform = Compose([ToTensor()])
        gx = transform(gx).unsqueeze(0).cuda()
        gy = transform(gy).unsqueeze(0).cuda()
        scharr = transform(scharr).unsqueeze(0).cuda()
        gradientx.append(gx)
        gradienty.append(gy)
        gradient.append(scharr)
    gradientx = torch.cat(gradientx,0)
    gradienty = torch.cat(gradienty, 0)
    gradient = torch.cat(gradient,0)


    #
    # gradientx = torch.from_numpy(gradientx).type(torch.FloatTensor).cuda()
    # gradienty = torch.from_numpy(gradienty).type(torch.FloatTensor).cuda()
    return gradientx,gradienty,gradient



def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim))

    # --- Write the training log --- #
    with open('./training_log/{}_log.txt'.format(category), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)

def print_log_gradient(epoch, num_epochs, one_epoch_time, train_psnr,val_psnr, val_ssim, category, gxtrain_psnr,gytrain_psnrval_psnr,avr_gxpsnr,avr_gypsnr,avr_gxssim,avr_gyssim):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}, GX_Train_PSNR:{6:.2f},GY_Train:{7:.2f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim,gxtrain_psnr,gytrain_psnrval_psnr))
    print('val_gxpsnr:{0:.2f},val_gypsnr:{1:.2f},val_gxssim:{2:.4f},val_gyssim:{3:.4f}'
          .format(avr_gxpsnr,avr_gypsnr,avr_gxssim,avr_gyssim))
    # --- Write the training log --- #
    with open('./training_log/{}_log.txt'.format(category), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f},GX_Train_PSNR:{7:.2f},GY_Train:{8:.2f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim,gxtrain_psnr,gytrain_psnrval_psnr), file=f)
        print('val_gxpsnr:{0:.2f},val_gypsnr:{1:.2f},val_gxssim:{2:.4f},val_gyssim:{3:.4f}'
              .format(avr_gxpsnr, avr_gypsnr, avr_gxssim, avr_gyssim), file=f)
