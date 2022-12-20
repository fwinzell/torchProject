import cv2
import os
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

if __name__ == "__main__":
    dir = '/home/fi5666wi/Documents/Prostate images/train_data_with_gt'
    patches = os.path.join(dir, 'gt600_256/Patches')
    masks = os.path.join(dir, 'gt600_256/Labels')

    for im_path in os.listdir(patches)[:10]:
        image = cv2.imread(os.path.join(patches, im_path))
        mask_path = 'labels_' + im_path[6:]
        mask = cv2.imread(os.path.join(masks, mask_path))
        torch_image = torch.tensor(np.moveaxis(image, source=-1, destination=0))
        torch_mask = torch.tensor(np.moveaxis(mask, source=-1, destination=0))
        #distorted, new_mask = random_crop_and_resize(torch_image, torch_mask)

        distorted = color_distortion(torch_image) # gaussian_blur(distorted, prob=1.0)
        distorted = distorted.numpy().astype('uint8')
        distorted = np.moveaxis(distorted, source=0, destination=-1)

        #new_mask = new_mask.numpy().astype('uint8')
        #new_mask = np.moveaxis(new_mask, source=0, destination=-1)

        cv2.imshow('original', image)
        cv2.imshow('distorted', distorted)
        #cv2.imshow('new mask', new_mask/4)
        cv2.waitKey(2000)