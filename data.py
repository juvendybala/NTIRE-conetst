import os
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import numpy as np
import math
import random

from starter_kit.degradations import simple_deg_simulation
from starter_kit.imutils import postprocess_raw, plot_all, demosaic, convert_to_tensor, crop_center
from utils import random_crop, worker_init_fn, data_augment

class Mydataset(Dataset):
    def __init__(self, filepaths, upscale, num_patches, gt_patch_size, kernels, training, lr_paths_root=None):
        super().__init__()
        self.filepaths = filepaths
        self.upscale = upscale
        self.gt_patch_size = gt_patch_size
        self.kernels = kernels
        self.num_patches = num_patches
        self.training = training
        if self.training == False:
            self.lr_paths_root = lr_paths_root


    def __getitem__(self, index):
        filepath = self.filepaths[index]
        raw = np.load(filepath)
        raw_img = raw["raw"]
        raw_max = raw["max_val"]
        gt_img = (raw_img / raw_max).astype(np.float32)
        if self.training:
            lr_patches = []
            gt_patches = []
            for _ in range(self.num_patches):
                gt_patch = random_crop(gt_img=gt_img, gt_patch_size=self.gt_patch_size)
                gain = 0.95 + 0.1 * random.random()
                gt_patch = gt_patch * gain
                lr_patch = simple_deg_simulation(img=gt_patch, kernels=self.kernels)
                lr_patch = lr_patch.permute(2, 0, 1)
                gt_patch = convert_to_tensor(gt_patch)
                lr_patch, gt_patch = data_augment(lr_patch=lr_patch, hr_patch=gt_patch)
                lr_patches.append(lr_patch)
                gt_patches.append(gt_patch)
            lr_patches = torch.stack(lr_patches, dim=0)
            gt_patches = torch.stack(gt_patches, dim=0)
            # gt_patch = random_crop(gt_img=gt_img, gt_patch_size=self.gt_patch_size)
            # gain = 0.9 + 0.2 * random.random()
            # gt_patch = gt_patch * gain
            # lr_patch = simple_deg_simulation(img=gt_patch, kernels=self.kernels)
            # lr_patch = lr_patch.permute(2, 0, 1)
            # gt_patch = convert_to_tensor(gt_patch)
            # lr_patch, gt_patch = data_augment(lr_patch=lr_patch, hr_patch=gt_patch)
            return lr_patches, gt_patches
        else:
            lr_path = os.path.join(self.lr_paths_root, os.path.basename(filepath))
            raw = np.load(lr_path)
            raw_img = raw["raw"]
            raw_max = raw["max_val"]
            lr_img = convert_to_tensor((raw_img / raw_max).astype(np.float32))
            lr_patches = []
            lr_patches_info = []
            
            h, w, _ = gt_img.shape
            patch_size_h, patch_size_w = self.gt_patch_size, self.gt_patch_size
            scale = self.upscale
            num_row = (h - 1) // patch_size_h + 1
            num_col = (w - 1) // patch_size_w + 1
            
            step_h = patch_size_h if num_row == 1 else math.ceil((h - patch_size_h) / (num_row - 1) - 1e-8)
            step_w = patch_size_w if num_col == 1 else math.ceil((w - patch_size_w) / (num_col - 1) - 1e-8)
            
            step_h = step_h //scale * scale
            step_w = step_w //scale * scale
            
            i = 0
            while i < h:
                if i + patch_size_h >= h:
                    i = h - patch_size_h
                j = 0
                while j < w:
                    if j + patch_size_w >= w:
                        j = w - patch_size_w
                    # print(j//scale, (j + patch_size_w)//scale)
                    lr_patch = lr_img[:,i//scale:(i + patch_size_h)//scale, 
                                    j//scale:(j + patch_size_w)//scale].clone()
                    # print(lr_patch.shape)
                    lr_patches.append(lr_patch)
                    lr_patches_info.append((i, j))
                    
                    if j + patch_size_w >= w:
                        break
                    j = j + step_w
                
                if i + patch_size_h >= h:
                    break
                i = i + step_h
            
            return torch.stack(lr_patches), lr_patches_info, convert_to_tensor(gt_img)


    def __len__(self):
        return len(self.filepaths)


def prepare_validation_set(val_gt_paths, val_lr_paths_root, kernels):
    for val_gt_path in val_gt_paths:
        raw = np.load(val_gt_path)
        raw_img = raw["raw"]
        raw_max = raw["max_val"]
        gt_img = (raw_img / raw_max).astype(np.float32)
        lr_img = simple_deg_simulation(img=gt_img, kernels=kernels)
        # print(gt_img.shape, lr_img.shape)
        lr_img = lr_img.numpy()
        lr_img = (lr_img * raw_max).astype(np.float16)
        new_path = os.path.join(val_lr_paths_root, os.path.basename(val_gt_path))
        np.savez(new_path, raw=lr_img, max_val=raw_max)

def prepare_test_set(test_gt_paths, test_lr_paths_root, kernels):
    for test_gt_path in test_gt_paths:
        raw = np.load(test_gt_path)
        raw_img = raw["raw"]
        raw_max = raw["max_val"]
        gt_img = (raw_img / raw_max).astype(np.float32)
        lr_img = simple_deg_simulation(img=gt_img, kernels=kernels)
        # print(gt_img.shape, lr_img.shape)
        lr_img = lr_img.numpy()
        lr_img = (lr_img * raw_max).astype(np.float16)
        new_path = os.path.join(test_lr_paths_root, os.path.basename(test_gt_path))
        np.savez(new_path, raw=lr_img, max_val=raw_max)

def load_training_dataset(filename, kernels, patch_size):

    filepaths = sorted(glob(filename+"*.npz"))
    val_size = 40
    train_size = len(filepaths) - val_size
    train_paths = filepaths[:train_size]
    val_paths = filepaths[train_size:]
    # print(len(val_paths))

    print("start preparing validation dataset")
    folder_name = "val_lr"
    current_path = os.getcwd()
    fullpath = os.path.join(current_path, folder_name)
    if os.path.exists(fullpath) and os.path.isdir(fullpath):
        print("already prepared validation dataset")
    else:
        os.makedirs(name=folder_name)
        prepare_validation_set(val_gt_paths=val_paths, val_lr_paths_root=fullpath, kernels=kernels)
        print("preparing validation dataset ends")
    
    print("start loading training dataset")
    train_dataset = Mydataset(filepaths=train_paths, upscale=2, gt_patch_size=patch_size, kernels=kernels, num_patches=40, training=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=8, prefetch_factor=2, persistent_workers=True, worker_init_fn=worker_init_fn)
    print("loading training dataset ends")

    print("start loading validation dataset")
    val_dataset = Mydataset(filepaths=val_paths, upscale=2, gt_patch_size=patch_size, kernels=kernels, num_patches=1, training=False, lr_paths_root=fullpath)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, prefetch_factor=2, persistent_workers=True, worker_init_fn=worker_init_fn)
    print("loading validation dataset ends")
    
    return train_dataloader, val_dataloader


def load_kernels():
    kernels = np.load("./starter_kit/kernels.npy",allow_pickle=True)
    # plot_all([k for k in kernels])
    return kernels

def load_validation_dataset(filename,kernels):
    for rawf in sorted(glob(filename+"1.npz")):
        raw = np.load(rawf)
        raw_img = raw["raw"]
        raw_max = raw["max_val"]
        raw_img = (raw_img / raw_max).astype(np.float32)
        # print(raw_img.shape)  