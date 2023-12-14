import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import scipy
import data.utils as data_utils
import os
import pickle
# Modify the following
NYU_PATH = './datasets/nyu/'
SCANNET_PATH = './scan/' # ScanNet related functionality are added
class ScanLoader(object):
    def __init__(self, args, mode,small=False):       
        self.t_samples = ScanPre(args, mode,small)
        
        # train
        if 'train' in mode:
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.t_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.t_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=False,
                                   drop_last=True,
                                   sampler=self.train_sampler)

        else:
            self.data = DataLoader(self.t_samples, 8,
                                   shuffle=False,
                                   num_workers=8,
                                   pin_memory=False)

class ScanPre(Dataset):
    def __init__(self, args, mode,small=False):
        self.args = args
        splitdir = os.path.join(SCANNET_PATH, 'train_test_split.pkl')
        with open(splitdir, 'rb') as f:
            self.filenames = pickle.load(f)[mode][0]
        if small:
            self.filenames = self.filenames[::50]
        dir = os.path.join(SCANNET_PATH, 'scannet-frames')
        # self.filenames = [f for f in os.listdir(dir) if f.endswith('-color.png')]
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225]) # same as NYU
        self.dataset_path = dir 
        self.input_height = args.input_height
        self.input_width = args.input_width

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.dataset_path, self.filenames[idx][27:])
        norm_name = img_name.replace('-color', '-normal')
        mask_name = img_name.replace('-color', '-orient-mask')
        img = Image.open(img_name).convert("RGB")
        norm_gt = Image.open(norm_name).convert("RGB")
        norm_valid_mask = Image.open(mask_name)
        if 'train' in self.mode:
            # horizontal flip (default: True)
            DA_hflip = False
            if self.args.data_augmentation_hflip:
                DA_hflip = random.random() > 0.5
                if DA_hflip:
                    # img = img[:,::-1,:].copy()
                    # norm_gt = norm_gt[:,::-1,:].copy()
                    img = TF.hflip(img)
                    norm_gt = TF.hflip(norm_gt)
                    norm_valid_mask = TF.hflip(norm_valid_mask)
            # to array
            # img = img.astype(np.float32) / 255.0
            img = np.array(img).astype(np.float32) / 255.0

            norm_gt = np.array(norm_gt).astype(np.uint8)
            

            # norm_valid_mask = np.logical_not(
            #     np.logical_and(
            #         np.logical_and(
            #             norm_gt[:, :, 0] == 0, norm_gt[:, :, 1] == 0),
            #         norm_gt[:, :, 2] == 0))
            norm_valid_mask = np.array(norm_valid_mask).astype(bool)

            norm_valid_mask = norm_valid_mask[:, :, np.newaxis]

            norm_gt = ((norm_gt.astype(np.float32) / 255.0) * 2.0) - 1.0

            if DA_hflip:
                norm_gt[:, :, 0] = - norm_gt[:, :, 0]

            # random crop (default: False)
            if self.args.data_augmentation_random_crop:
                img, norm_gt, norm_valid_mask = data_utils.random_crop(img, norm_gt, norm_valid_mask, 
                                                                     height=416, width=544)

            # color augmentation (default: True)
            if self.args.data_augmentation_color:
                if random.random() > 0.5:
                    img = data_utils.color_augmentation(img, indoors=True)
        else:
            # img = img.astype(np.float32) / 255.0
            img = np.array(img).astype(np.float32) / 255.0

            norm_gt = np.array(norm_gt).astype(np.uint8)

            norm_valid_mask = np.array(norm_valid_mask).astype(bool)

            # norm_valid_mask = np.logical_not(
            #     np.logical_and(
            #         np.logical_and(
            #             norm_gt[:, :, 0] == 0, norm_gt[:, :, 1] == 0),
            #         norm_gt[:, :, 2] == 0))
            norm_valid_mask = norm_valid_mask[:, :, np.newaxis]

            norm_gt = ((norm_gt.astype(np.float32) / 255.0) * 2.0) - 1.0

        # to tensors
        img = self.normalize(torch.from_numpy(img).permute(2, 0, 1))            # (3, H, W)
        norm_gt = torch.from_numpy(norm_gt).permute(2, 0, 1)                    # (3, H, W)
        norm_valid_mask = torch.from_numpy(norm_valid_mask).permute(2, 0, 1)    # (1, H, W)

        sample = {'img': img,
                  'norm': norm_gt,
                  'norm_valid_mask': norm_valid_mask,
                  'scene_name': self.mode,
                  'img_name': img_name}

        return sample

class NyuLoader(object):
    def __init__(self, args, mode):
        """mode: {'train_big',  # training set used by GeoNet (CVPR18, 30907 images)
                  'train',      # official train set (795 images) 
                  'test'}       # official test set (654 images)
        """
        # actually support for large version than original code
        if mode == 'train_big':
            self.t_samples = LargePreprocess(args, mode)
        else:
            self.t_samples = NyuLoadPreprocess(args, mode)

        # train, train_big
        if 'train' in mode:
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.t_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.t_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   drop_last=True,
                                   sampler=self.train_sampler)

        else:
            self.data = DataLoader(self.t_samples, 10,
                                   shuffle=False,
                                   num_workers=8,
                                   pin_memory=False)

# Added and tested by us
class LargePreprocess(Dataset):
    def __init__(self, args, mode):
        self.args = args
        # train, train_big, test, test_new
        dir = os.path.join(NYU_PATH,"large")
        self.filenames = os.listdir(dir)
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.dataset_path = dir
        self.input_height = args.input_height
        self.input_width = args.input_width

    def __len__(self):
        return len(self.filenames)
    def myfunc(self,x):
        try:
            data_dic = scipy.io.loadmat(x)
            data_img = data_dic['img'][:-1,:-1,:].copy()
            data_norm = data_dic['norm'][:-1,:-1,:].copy()
            data_mask = data_dic['mask'][:-1,:-1].copy()
            return data_img,data_norm,data_mask
        except:
            return None,None,None


    def __getitem__(self, idx):
        sample_path = self.filenames[idx]

        scene_name = self.mode
        

        img, norm_gt,norm_valid_mask = self.myfunc(os.path.join(self.dataset_path,sample_path))
        if img is None or norm_gt is None:
            return None
        img[:,:,0] = img[:,:,0] + 2* 122.175
        img[:,:,1] = img[:,:,1] + 2* 116.169
        img[:,:,2] = img[:,:,2] + 2* 103.508
        img  = img.astype(np.uint8)
        norm_gt = (((norm_gt+1)/2)*255).astype(np.uint8)
        img = Image.fromarray(img).convert('RGB').resize(size=(self.input_width, self.input_height), 
                                                            resample=Image.BILINEAR)
        norm_gt = Image.fromarray(norm_gt).convert('RGB').resize(size=(self.input_width, self.input_height),
                                                             resample=Image.NEAREST)
        if 'train' in self.mode:
            # horizontal flip (default: True)
            DA_hflip = False
            if self.args.data_augmentation_hflip:
                DA_hflip = random.random() > 0.5
                if DA_hflip:
                    # img = img[:,::-1,:].copy()
                    # norm_gt = norm_gt[:,::-1,:].copy()
                    img = TF.hflip(img)
                    norm_gt = TF.hflip(norm_gt)
                    norm_valid_mask = norm_valid_mask[:,::-1].copy()

            # to array
            # img = img.astype(np.float32) / 255.0
            img = np.array(img).astype(np.float32) / 255.0

            norm_gt = np.array(norm_gt).astype(np.uint8)
            

            # norm_valid_mask = np.logical_not(
            #     np.logical_and(
            #         np.logical_and(
            #             norm_gt[:, :, 0] == 0, norm_gt[:, :, 1] == 0),
            #         norm_gt[:, :, 2] == 0))
            norm_valid_mask = norm_valid_mask[:, :, np.newaxis].astype(bool)

            norm_gt = ((norm_gt.astype(np.float32) / 255.0) * 2.0) - 1.0

            if DA_hflip:
                norm_gt[:, :, 0] = - norm_gt[:, :, 0]

            # random crop (default: False)
            if self.args.data_augmentation_random_crop:
                img, norm_gt, norm_valid_mask = data_utils.random_crop(img, norm_gt, norm_valid_mask, 
                                                                     height=416, width=544)

            # color augmentation (default: True)
            if self.args.data_augmentation_color:
                if random.random() > 0.5:
                    img = data_utils.color_augmentation(img, indoors=True)
        else:
            # img = img.astype(np.float32) / 255.0
            img = np.array(img).astype(np.float32) / 255.0

            norm_gt = np.array(norm_gt).astype(np.uint8)


            # norm_valid_mask = np.logical_not(
            #     np.logical_and(
            #         np.logical_and(
            #             norm_gt[:, :, 0] == 0, norm_gt[:, :, 1] == 0),
            #         norm_gt[:, :, 2] == 0))
            norm_valid_mask = norm_valid_mask[:, :, np.newaxis].astype(bool)

            norm_gt = ((norm_gt.astype(np.float32) / 255.0) * 2.0) - 1.0

        # to tensors
        img = self.normalize(torch.from_numpy(img).permute(2, 0, 1))            # (3, H, W)
        norm_gt = torch.from_numpy(norm_gt).permute(2, 0, 1)                    # (3, H, W)
        norm_valid_mask = torch.from_numpy(norm_valid_mask).permute(2, 0, 1)    # (1, H, W)

        sample = {'img': img,
                  'norm': norm_gt,
                  'norm_valid_mask': norm_valid_mask,
                  'scene_name': scene_name,
                  'img_name': sample_path}

        return sample

class NyuLoadPreprocess(Dataset):
    def __init__(self, args, mode):
        self.args = args
        # train, train_big, test, test_new
        with open("./data_split/nyu_%s.txt" % mode, 'r') as f:
            self.filenames = f.readlines()
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.dataset_path = NYU_PATH
        self.input_height = args.input_height
        self.input_width = args.input_width

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]

        # img path and norm path
        img_path = self.dataset_path + '/' + sample_path.split()[0]
        norm_path = self.dataset_path + '/' + sample_path.split()[1]
        scene_name = self.mode
        img_name = img_path.split('/')[-1].split('.png')[0]

        # read img / normal
        img = Image.open(img_path).convert("RGB").resize(size=(self.input_width, self.input_height), 
                                                            resample=Image.BILINEAR)
        norm_gt = Image.open(norm_path).convert("RGB").resize(size=(self.input_width, self.input_height), 
                                                            resample=Image.NEAREST)

        if 'train' in self.mode:
            # horizontal flip (default: True)
            DA_hflip = False
            if self.args.data_augmentation_hflip:
                DA_hflip = random.random() > 0.5
                if DA_hflip:
                    img = TF.hflip(img)
                    norm_gt = TF.hflip(norm_gt)

            # to array
            img = np.array(img).astype(np.float32) / 255.0

            norm_gt = np.array(norm_gt).astype(np.uint8)

            norm_valid_mask = np.logical_not(
                np.logical_and(
                    np.logical_and(
                        norm_gt[:, :, 0] == 0, norm_gt[:, :, 1] == 0),
                    norm_gt[:, :, 2] == 0))
            norm_valid_mask = norm_valid_mask[:, :, np.newaxis]

            norm_gt = ((norm_gt.astype(np.float32) / 255.0) * 2.0) - 1.0

            if DA_hflip:
                norm_gt[:, :, 0] = - norm_gt[:, :, 0]

            # random crop (default: False)
            if self.args.data_augmentation_random_crop:
                img, norm_gt, norm_valid_mask = data_utils.random_crop(img, norm_gt, norm_valid_mask, 
                                                                     height=416, width=544)

            # color augmentation (default: True)
            if self.args.data_augmentation_color:
                if random.random() > 0.5:
                    img = data_utils.color_augmentation(img, indoors=True)
        else:
            img = np.array(img).astype(np.float32) / 255.0

            norm_gt = np.array(norm_gt).astype(np.uint8)

            norm_valid_mask = np.logical_not(
                np.logical_and(
                    np.logical_and(
                        norm_gt[:, :, 0] == 0, norm_gt[:, :, 1] == 0),
                    norm_gt[:, :, 2] == 0))
            norm_valid_mask = norm_valid_mask[:, :, np.newaxis]

            norm_gt = ((norm_gt.astype(np.float32) / 255.0) * 2.0) - 1.0

        # to tensors
        img = self.normalize(torch.from_numpy(img).permute(2, 0, 1))            # (3, H, W)
        norm_gt = torch.from_numpy(norm_gt).permute(2, 0, 1)                    # (3, H, W)
        norm_valid_mask = torch.from_numpy(norm_valid_mask).permute(2, 0, 1)    # (1, H, W)

        sample = {'img': img,
                  'norm': norm_gt,
                  'norm_valid_mask': norm_valid_mask,
                  'scene_name': scene_name,
                  'img_name': img_name}

        return sample
