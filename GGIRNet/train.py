

# python -m tensorboard.main --logdir='runs/Final_final'
# python -m tensorboard.main --logdir='Result' --load_fast true
# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_data import TrainData
from val_data import ValData
from model import Net
from alpLoGLoss import ALPLoGLoss
from gridLoss import GridLoss
from gaussKernel import gaussBlur

from utils import to_psnr, print_log, validation, adjust_learning_rate, to_ssim_skimage
from torchvision.models import vgg16
from perceptual import LossNetwork
import os
import numpy as np
import random
from tensorboardX import SummaryWriter
plt.switch_backend('agg')

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=2.5e-5, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[128, 128], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=16, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
#parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=2, type=int)
parser.add_argument('-category', help='Set image category (derain or dehaze?)', default='derain', type=str)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', default='Rain1200_grid',type=str)
#parser.add_argument('-lambda_GP', help='Set the lambda_GP for gploss in loss function', default=0.0015, type=float)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
#lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
category = args.category
exp_name = args.exp_name
#lambgp = args.lambda_GP
#use_GP_inlblphase = False # indication whether or not to use GP during labeled phase

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\ncategory: {}'.format(learning_rate, crop_size,
      train_batch_size, val_batch_size, category))

# --- Set category-specific hyper-parameters  --- #
if category == 'derain':
    num_epochs = 60
    train_data_dir = 'Rain_dataset/Rain1200/train/'
    val_data_dir = 'Rain_dataset/Rain1200/train/'

else:
    raise Exception('Wrong image category. Set it to derain or dehaze dateset.')


# --- Gpu device --- #
# device_ids = [Id for Id in range(torch.cuda.device_count())]
device_ids = [0,1,2,3]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
net = Net()


# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)


# --- Define the perceptual loss network --- #
# vgg_model = vgg16(pretrained=True).features[:16]
# vgg_model = vgg_model.to(device)
# # vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
# for param in vgg_model.parameters():
#     param.requires_grad = False
# loss_network = LossNetwork(vgg_model)
# loss_network.eval()


# --- Load the network weight --- #
if os.path.exists('./{}/'.format(exp_name))==False:     
    os.mkdir('./{}/'.format(exp_name))  
try:
    net.load_state_dict(torch.load('./{}/{}'.format(exp_name,category)))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')


# --- Calculate all trainable parameters in network --- #
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))


# --- Load training data and validation/test data --- #
labeled_name = 'train_Rain1200.txt'
#unlabeled_name = 'real_input_split1.txt'
val_filename = 'val_Rain1200.txt'
# --- Load training data and validation/test data --- #
#unlbl_train_data_loader = DataLoader(TrainData(crop_size, train_data_dir,unlabeled_name), batch_size=train_batch_size, shuffle=True, num_workers=8)
lbl_train_data_loader = DataLoader(TrainData(crop_size, train_data_dir,labeled_name), batch_size=train_batch_size, shuffle=True, num_workers=8)
val_data_loader = DataLoader(ValData(val_data_dir,val_filename), batch_size=val_batch_size, shuffle=False, num_workers=8)

num_labeled = train_batch_size*len(lbl_train_data_loader) # number of labeled images
#num_unlabeled = train_batch_size*len(unlbl_train_data_loader) # number of unlabeled images






# --- Previous PSNR and SSIM in testing --- #
net.eval()
old_val_psnr, old_val_ssim = validation(net, val_data_loader, device, category, exp_name)
print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))
# old_val_psnr = 0
# old_val_ssim = 0
net.train()
#intializing GPStruct
#gp_struct = GPStruct(num_labeled, train_batch_size)
writer = SummaryWriter('tensorboard/Rain1200_grid')
for epoch in range(epoch_start, num_epochs):
    psnr_list = []
    loss_list = []
    ssim_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch, category=category)
#-------------------------------------------------------------------------------------------------------------
    #Labeled phase
    # if lambgp != 0 and use_GP_inlblphase == True:
    #     gp_struct.gen_featmaps(lbl_train_data_loader,net,device)
    for batch_id, train_data in enumerate(lbl_train_data_loader):

        input_image, gt, imgid = train_data
        input_image = input_image.to(device)
        gt = gt.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        net.train()
        pred_image = net(input_image)

        pred_image_gauss01, gt_gauss01 = gaussBlur(pred_image, gt, 1.600000)
        pred_image_gauss02, gt_gauss02 = gaussBlur(pred_image, gt, 2.262742)
        pred_image_gauss03, gt_gauss03 = gaussBlur(pred_image, gt, 3.200000)
        pred_image_gauss04, gt_gauss04 = gaussBlur(pred_image, gt, 4.525483)
        pred_image_gauss05, gt_gauss05 = gaussBlur(pred_image, gt, 6.400000)
        # print(pred_image_gauss01.shape)
        # print(gt_gauss01.shape)
        L1Loss = torch.nn.L1Loss()

        # loss01 = GridLoss()(pred_image_gauss01, gt_gauss01) + L1Loss(pred_image_gauss01, gt_gauss01)
        # loss02 = GridLoss()(pred_image_gauss02, gt_gauss02) + L1Loss(pred_image_gauss02, gt_gauss02)
        # loss03 = GridLoss()(pred_image_gauss03, gt_gauss03) + L1Loss(pred_image_gauss03, gt_gauss03)
        # loss04 = GridLoss()(pred_image_gauss04, gt_gauss04) + L1Loss(pred_image_gauss04, gt_gauss04)
        # loss05 = GridLoss()(pred_image_gauss05, gt_gauss05) + L1Loss(pred_image_gauss05, gt_gauss05)
        loss01 = GridLoss()(pred_image_gauss01, gt_gauss01)
        loss02 = GridLoss()(pred_image_gauss02, gt_gauss02)
        loss03 = GridLoss()(pred_image_gauss03, gt_gauss03)
        loss04 = GridLoss()(pred_image_gauss04, gt_gauss04)
        loss05 = GridLoss()(pred_image_gauss05, gt_gauss05)

        loss = loss01 + loss02 + loss03 + loss04 + loss05

        loss.backward()
        loss_list.append(float(loss))
        optimizer.step()

        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(pred_image, gt))
        ssim_list.extend(to_ssim_skimage(pred_image, gt))
        if not (batch_id % 100):
            print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))
            iteration_time = time.time()-start_time
            print('({0:.0f}s)'.format(iteration_time))

        if not (batch_id % 1000):
            train_loss1 = sum(loss_list) / len(loss_list)
            train_psnr1 = sum(psnr_list) / len(psnr_list)
            train_ssim1 = sum(ssim_list) / len(ssim_list)
            writer.add_scalar('loss1', train_loss1, global_step=(epoch*20)+(batch_id//1000))
            writer.add_scalar('train_psnr1', train_psnr1, global_step=(epoch*20)+(batch_id//1000))
            writer.add_scalar('train_ssim1', train_ssim1, global_step=(epoch*20)+(batch_id//1000))
            # print_log(epoch+1, num_epochs, 0, train_psnr1, 0, 0, category)

    # --- Calculate the average training PSNR in one epoch --- #

    train_loss = sum(loss_list) / len(loss_list)
    train_psnr = sum(psnr_list) / len(psnr_list)
    train_ssim = sum(ssim_list) / len(ssim_list)
    # --- Save the network parameters --- #
    torch.save(net.state_dict(), './{}/{}'.format(exp_name,category))

    # --- Use the evaluation model in testing --- #
    net.eval()

    val_psnr, val_ssim = validation(net, val_data_loader, device, category, exp_name)
    one_epoch_time = time.time() - start_time
    print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category)
    #tensorboard
    writer.add_scalar('loss', train_loss, global_step=epoch)
    writer.add_scalar('train_psnr', train_psnr, global_step=epoch)
    writer.add_scalar('train_ssim', train_ssim, global_step=epoch)
    writer.add_scalar('val_psnr', val_psnr, global_step=epoch)
    writer.add_scalar('val_ssim', val_ssim, global_step=epoch)

    # --- update the network weight --- #
    if val_psnr >= old_val_psnr:
        torch.save(net.state_dict(), './{}/{}_best'.format(exp_name,category))
        print('model saved')
        old_val_psnr = val_psnr
writer.close()
    # if lambgp!=0:
    #     gp_struct.gen_featmaps(lbl_train_data_loader,net,device)
#-------------------------------------------------------------------------------------------------------------
    # Unlabeled Phase
    
    # if lambgp!=0 and epoch>10:
    #     gp_struct.gen_featmaps_unlbl(unlbl_train_data_loader,net,device)
    #     for batch_id, train_data in enumerate(unlbl_train_data_loader):
    #
    #         input_image, gt, imgid = train_data
    #         input_image = input_image.to(device)
    #         gt = gt.to(device)
    #
    #         # --- Zero the parameter gradients --- #
    #         optimizer.zero_grad()
    #
    #         # --- Forward + Backward + Optimize --- #
    #         net.train()
    #         pred_image,zy_in = net(input_image)
    #         gp_loss = 0
    #         if lambgp!=0:
    #             gp_loss = gp_struct.compute_gploss(zy_in, imgid,batch_id,0)
    #
    #         loss = lambgp*gp_loss
    #         if loss!=0:
    #             loss.backward()
    #             optimizer.step()
    #
    #         # --- To calculate average PSNR --- #
    #         psnr_list.extend(to_psnr(pred_image, gt))
    #
    #         if not (batch_id % 100):
    #             print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))
    #
    #     # --- Calculate the average training PSNR in one epoch --- #
    #     train_psnr = sum(psnr_list) / len(psnr_list)
    #
    #     # --- Save the network parameters --- #
    #     torch.save(net.state_dict(), './{}/{}'.format(exp_name,category))
    #
    #     # --- Use the evaluation model in testing --- #
    #     net.eval()
    #
    #     val_psnr, val_ssim = validation(net, val_data_loader, device, category, exp_name)
    #     one_epoch_time = time.time() - start_time
    #     print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category)
    #
    #     # --- update the network weight --- #
    #     if val_psnr >= old_val_psnr:
    #         torch.save(net.state_dict(), './{}/{}_best'.format(exp_name,category))
    #         print('model saved')
    #         old_val_psnr = val_psnr
#-------------------------------------------------------------------------------------------------------------


