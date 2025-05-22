import os
import math
from datetime import datetime
import numpy as np
import torch
from glob import glob
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import argparse

from data import load_training_dataset, load_kernels, load_validation_dataset, prepare_test_set
from model import NAFSR
from utils import set_manual_seed, worker_init_fn, calculate_PSNR, calculate_SSIM
from starter_kit.imutils import convert_to_tensor

def generate():
    set_manual_seed(0)
    if(torch.cuda.is_available):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    kernels = load_kernels()
    patch_size = 64
    test_lr_paths = sorted(glob("./test_in/*.npz"))
    folder_name = "test_out"
    current_path = os.getcwd()
    fullpath = os.path.join(current_path, folder_name)
    os.makedirs(name=folder_name, exist_ok=True)
    best_model_path = "checkpoints/"
    checkpoint = torch.load(best_model_path + "nafssr_x2_2025-03-25 12:26:26_150best.pth")
    model = NAFSR(up_scale=2, width=48, num_blks=12, img_channel=4, drop_out_rate=0., drop_path_rate=0.).to(device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        for test_lr_path in test_lr_paths:
            raw = np.load(test_lr_path)
            raw_img = raw["raw"]
            raw_max = raw["max_val"]
            lr_img = (raw_img / raw_max).astype(np.float32)
            new_path = os.path.join(fullpath, os.path.basename(test_lr_path))
            lr_img = convert_to_tensor(lr_img).unsqueeze(0).to(device=device)
            print(lr_img.shape)
            hr_img = model(lr_img).squeeze(0).permute(1, 2, 0).to(device=torch.device("cpu")).numpy()
            print(hr_img.shape)
            hr_img = (hr_img * raw_max).astype(np.float16)
            new_path = os.path.join(fullpath, os.path.basename(test_lr_path))
            np.savez(new_path, raw=hr_img, max_val=raw_max)



def test():
    set_manual_seed(0)
    if(torch.cuda.is_available):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    kernels = load_kernels()
    patch_size = 64
    num_blks = 12
    print("start preparing testing dataset")
    test_gt_paths = sorted(glob("./test_in/*.npz"))
    folder_name = "test_lr"
    current_path = os.getcwd()
    fullpath = os.path.join(current_path, folder_name)
    if os.path.exists(fullpath) and os.path.isdir(fullpath):
        print("already prepared testing dataset")
    else:
        os.makedirs(name=folder_name)
        prepare_test_set(test_gt_paths=test_gt_paths, test_lr_paths_root=fullpath, kernels=kernels)
        print("preparing testing dataset ends")
    
    best_model_path = "checkpoints/"
    checkpoint = torch.load(best_model_path + "nafssr_x2_2025-03-25 12:26:26_150best.pth")
    model = NAFSR(up_scale=2, width=48, num_blks=num_blks, img_channel=4, drop_out_rate=0., drop_path_rate=0).to(device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    metric = []
    inference_time = []
    model.eval()
    with torch.no_grad():
        for test_gt_path in test_gt_paths:
            raw = np.load(test_gt_path)
            raw_img = raw["raw"]
            raw_max = raw["max_val"]
            gt_img = (raw_img / raw_max).astype(np.float32)
            new_path = os.path.join(fullpath, os.path.basename(test_gt_path))
            raw = np.load(new_path)
            raw_img = raw["raw"]
            raw_max = raw["max_val"]
            lr_img = (raw_img / raw_max).astype(np.float32)
            gt_img = convert_to_tensor(image=gt_img).unsqueeze(0).to(device=device)
            lr_img = convert_to_tensor(image=lr_img).unsqueeze(0).to(device=device)
            start_time = time.time()
            hr_img = model(lr_img)
            end_time = time.time()
            psnr = calculate_PSNR(hr_img=hr_img, gt_img=gt_img)
            ssim = calculate_SSIM(hr_img=hr_img, gt_img=gt_img)
            inference_time.append(end_time - start_time)
            metric.append((psnr, ssim))
        metric_psnr = sum([m[0] for m in metric]) / len(metric)
        metric_ssim = sum([m[1] for m in metric]) / len(metric)
        average_inference_time = sum(inference_time) / len(inference_time)
        print(f"test completes, PSNR: {metric_psnr:.2f}, SSIM: {metric_ssim:.4f}, average inference time: {average_inference_time:.4f}")

def main():
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    set_manual_seed(0)
    if(torch.cuda.is_available):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    kernels = load_kernels()
    patch_size = 64
    eval_freq = 5
    num_blks = 12
    max_iter = 5e4
    train_dataloader, val_dataloader = load_training_dataset(filename="./train_raws/", kernels=kernels, patch_size=patch_size)
    model = NAFSR(up_scale=2, width=48, num_blks=num_blks, img_channel=4, drop_out_rate=0., drop_path_rate=0.).to(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.9), weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=1e-7)
    criterion = nn.L1Loss()
    writer = SummaryWriter()
    # print(len(train_dataloader))
    num_epoches = math.ceil(max_iter / len(train_dataloader))
    best_psnr = 0.
    best_ssim = 0.
    best_epoch = -1
    print("start training")
    for epoch in range(num_epoches):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for i, (lr_batch, gt_batch) in enumerate(train_dataloader):
            # print(lr_batch.shape, gt_batch.shape)
            batch_size, num_patches, c, h, w = lr_batch.shape
            lr_batch, gt_batch = lr_batch.view(batch_size * num_patches, c, h, w).to(device=device), gt_batch.view(batch_size * num_patches, c, 2 * h, 2 * w).to(device=device)
            # lr_batch, gt_batch = lr_batch.to(device=device), gt_batch.to(device=device)
            outputs = model(lr_batch)
            loss = criterion(outputs, gt_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            if i % 30 == 29:
                epoch_average_loss = running_loss / 30
                writer.add_scalar('training loss', epoch_average_loss, len(train_dataloader) * epoch + i)
                running_loss = 0.0
        # epoch_average_loss = running_loss / len(train_dataloader)
        # writer.add_scalar('training loss', epoch_average_loss, len(train_dataloader) * epoch)
        # running_loss = 0.0
        end_time = time.time()
        print(f'Epoch [{epoch}/{num_epoches}], Loss: {epoch_average_loss:.4f}, time: {end_time - start_time}')
        if epoch % eval_freq == 0:
            model.eval()
            metric = []
            inference_time = []
            with torch.no_grad():
                for lr_batch, lr_batch_info, gt_batch in val_dataloader:
                    lr_batch, gt_batch = lr_batch.squeeze(0).to(device=device), gt_batch.to(device=device)
                    start_time = time.time()
                    b, c, h, w = gt_batch.shape
                    hr_batch = model(lr_batch)
                    preds = torch.zeros((1, c, h, w)).to(device)
                    count_mt = torch.zeros((1, 1, h, w)).to(device)
                    for patch_info, hr_patch in zip(lr_batch_info, hr_batch):
                        i, j = patch_info
                        patch_h, patch_w = hr_patch.shape[-2:]
                        # print(preds[:, :, i:i + patch_h, j:j + patch_w].shape)
                        # print(hr_patch.shape)
                        preds[:, :, i:i + patch_h, j:j + patch_w] += hr_patch
                        count_mt[:,:, i:i + patch_h, j:j + patch_w] += 1
                    hr_img = preds / count_mt
                    end_time = time.time()
                    psnr = calculate_PSNR(hr_img=hr_img, gt_img=gt_batch)
                    ssim = calculate_SSIM(hr_img=hr_img, gt_img=gt_batch)
                    inference_time.append(end_time - start_time)
                    metric.append((psnr, ssim))
                metric_psnr = sum([m[0] for m in metric]) / len(metric)
                metric_ssim = sum([m[1] for m in metric]) / len(metric)
                average_inference_time = sum(inference_time) / len(inference_time)
                writer.add_scalar("PSNR", metric_psnr, epoch)
                writer.add_scalar("SSIM", metric_ssim, epoch)
                print(f"eval completes, PSNR: {metric_psnr:.2f}, SSIM: {metric_ssim:.4f}, average inference time: {average_inference_time:.4f}")
                if metric_psnr > best_psnr:
                    best_psnr = metric_psnr
                    best_ssim = metric_ssim
                    best_epoch = epoch
                    save_path = f'checkpoints/nafssr_x2_{current_time}_{epoch}best.pth'
                    os.makedirs('checkpoints', exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'psnr': best_psnr,
                        'ssim': best_ssim,
                    }, save_path)
                if metric_psnr == torch.nan or metric_ssim == torch.nan:
                    save_path = f'checkpoints/nafssr_x2_{current_time}_{epoch}_NAN.pth'
                    os.makedirs('checkpoints', exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'psnr': best_psnr,
                        'ssim': best_ssim,
                    }, save_path)
    print(f"training ends, best epoch is {best_epoch}, best PSNR is {best_psnr:.2f}, best SSIM is {best_ssim:.4f}")
    # load_validation_dataset("./val_pred/", kernels)
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()
    if args.mode == "test":
        test()
    elif args.mode == "train":
        main()
    elif args.mode == "generate":
        generate()
    else:
        print("Invalid arguments")
