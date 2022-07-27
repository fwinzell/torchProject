
import random
import math
import torch
from torch import nn
import torchvision.transforms.functional as F
from torchvision import transforms
import numpy as np


def random_crop_and_resize(img, tar=None, p=0.8):
    random_crop = transforms.RandomResizedCrop(size=img.shape[-2:])
    params = random_crop.get_params(img, scale=[0.05, 1.0], ratio=[0.75, 1.33])

    apply = random.choices([True, False], weights=[p, 1-p], k=1)
    if apply[0]:
        cropped_img = F.crop(img, top=params[0], left=params[1], height=params[2], width=params[3])
        img = F.resize(cropped_img, img.shape[-2:])
        if tar is not None:
            cropped_mask = F.crop(tar, top=params[0], left=params[1], height=params[2], width=params[3])
            tar = F.resize(cropped_mask, tar.shape[-2:])

    return img, tar


def color_distortion(img, s=1.0, prob=0.8):
    color_jitter = transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)
    # random_crop = transforms.RandomResizedCrop(size=image.shape[-2:], scale=(0.05, 1.0))
    applier = transforms.RandomApply(nn.ModuleList([color_jitter]), p=prob)

    return applier(img)


def gaussian_blur(img, sigma=(0.5, 2.0), prob=0.8):
    blur = transforms.GaussianBlur(kernel_size=15, sigma=sigma)
    applier = transforms.RandomApply(nn.ModuleList([blur]), p=prob)

    return applier(img)



