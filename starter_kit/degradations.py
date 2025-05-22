import rawpy
import numpy as np
import glob, os
import imageio
import argparse
from PIL import Image as PILImage
import scipy.io as scio
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2

from scipy.io import loadmat
from scipy import ndimage
from scipy.signal import convolve2d
import hdf5storage

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blur import apply_psf, add_blur
from .noise import add_natural_noise, add_gnoise, add_heteroscedastic_gnoise
from .imutils import downsample_raw, convert_to_tensor


def simple_deg_simulation(img, kernels):
    """
    Pipeline to add synthetic degradations to a (RAW/RGB) image.
    y = down(x * k) + n
    """

    img = convert_to_tensor(img)

    # Apply psf blur: x * k
    img = add_blur(img, kernels)

    # Apply downsampling down(x*k)
    img = downsample_raw(img)
    
    # Add noise down(x*k) + n
    p_noise = np.random.rand()
    if p_noise > 0.3:
        img = add_natural_noise(img)
    else:
        img = add_heteroscedastic_gnoise(img)
    
    return img