import random
import numpy as np
import torch
import torch.nn.functional as F
import math
import cv2

def set_manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def random_crop(gt_img, gt_patch_size):
    h_gt, w_gt, _ = gt_img.shape

    top_gt = random.randint(0, h_gt - gt_patch_size)
    left_gt = random.randint(0, w_gt - gt_patch_size)
    gt_img = gt_img[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]

    return gt_img

def calculate_PSNR(hr_img, gt_img):
    assert hr_img.shape == gt_img.shape, print(f"Invalid shape {hr_img.shape}, {gt_img.shape}")
    loss = F.mse_loss(input=hr_img, target=gt_img)
    if loss == 0.0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(loss))

def gaussian(window_size, sigma):
    """生成一维高斯核"""
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                         for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    """生成二维高斯核"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def calculate_SSIM(hr_img, gt_img):
    assert hr_img.shape == gt_img.shape, print(f"Invalid shape {hr_img.shape}, {gt_img.shape}")
    # print(hr_img.device, gt_img.device)
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    window = create_window(11, hr_img.size(1)).to(hr_img.device)
    mu1 = F.conv2d(hr_img, window, padding=0, groups=hr_img.size(1))
    mu2 = F.conv2d(gt_img, window, padding=0, groups=gt_img.size(1))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(hr_img * hr_img, window, padding=0, groups=hr_img.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(gt_img * gt_img, window, padding=0, groups=gt_img.size(1)) - mu2_sq
    sigma12 = F.conv2d(hr_img * gt_img, window, padding=0, groups=hr_img.size(1)) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(ssim_map.mean())

def data_augment(lr_patch, hr_patch):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    if hflip:
        lr_patch = torch.flip(lr_patch, [2])
        hr_patch = torch.flip(hr_patch, [2])
    if vflip:
        lr_patch = torch.flip(lr_patch, [1])
        hr_patch = torch.flip(hr_patch, [1])
    return lr_patch, hr_patch