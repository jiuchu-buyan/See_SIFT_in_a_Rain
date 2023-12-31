

# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import re
import cv2
import numpy as np

class TrainData_1(data.Dataset):
    def __init__(self, crop_size, train_data_dir,train_filename, gt_name,num,sigma):
        super().__init__()
        train_list = train_data_dir + train_filename
        with open(train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            # if train_filename=='sub_DDN-
            if num == 1:
                # gt_names = [i[:i.find('_')].strip().replace('rain','norain')+'.jpg' for i in input_names]
                # gt_names = [i[:i.find('_')].strip().replace('rain', 'DOG/02') + '.jpg' for i in input_names]
                gt_names = [i.strip().replace('rain', gt_name) for i in input_names]
            else:
                gt_names = [i.strip().replace('rain', 'norain') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir
        self.sigma =sigma

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        sigma = self.sigma
        img_id = re.split('/',input_name)[-1][:-4]

        # input_img = Image.open(self.train_data_dir + input_name)
        input_img = Image.open(self.train_data_dir + input_name).convert('YCbCr')
        gt_img = Image.open(self.train_data_dir + gt_name).convert('L')


        width, height = input_img.size

        if width < crop_width and height < crop_height :
            input_img = input_img.resize((crop_width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, crop_height), Image.ANTIALIAS)
        elif width < crop_width :
            input_img = input_img.resize((crop_width,height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width,height), Image.ANTIALIAS)
        elif height < crop_height :
            input_img = input_img.resize((width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((width, crop_height), Image.ANTIALIAS)


        width, height = input_img.size
        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))


        gt_crop_img = np.array(gt_crop_img)
        # input_crop_img = np.array(input_crop_img)

        gt_crop_img = cv2.GaussianBlur(gt_crop_img, (0, 0), sigmaX=sigma, sigmaY=sigma)



        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # transform_input = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_crop_img)

        gt = transform_gt(gt_crop_img)

        return input_im, gt, img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir, train_filename):
        super().__init__()
        train_list = train_data_dir + train_filename
        with open(train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]

            gt_names = [i.strip().replace('hazy','gt') for i in input_names]
            # gt_names = [i.strip()[:i.find('_')].replace('rain', 'norain') + '.jpg' for i in input_names]
            #gt_names = [i.strip().replace('rain', 'norain') for i in input_names]
        self.input_names = input_names
        self.gt_names = gt_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = re.split('/', input_name)[-1][:-4]

        input_img = Image.open(self.train_data_dir + input_name)

        try:
            gt_img = Image.open(self.train_data_dir + gt_name)
        except:
            gt_img = Image.open(self.train_data_dir + gt_name).convert('RGB')

        width, height = input_img.size

        if width < crop_width and height < crop_height :
            input_img = input_img.resize((crop_width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, crop_height), Image.ANTIALIAS)
        elif width < crop_width :
            input_img = input_img.resize((crop_width,height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width,height), Image.ANTIALIAS)
        elif height < crop_height :
            input_img = input_img.resize((width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((width, crop_height), Image.ANTIALIAS)

        width, height = input_img.size
        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_crop_img)
        gt = transform_gt(gt_crop_img)

        # --- Check the channel is 3 or not --- #
        if list(input_im.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return input_im, gt, img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)

