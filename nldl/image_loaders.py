import os
import fnmatch
import random
from scipy import ndimage
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from skimage.io import imread
import math
import cv2
from torchvision import transforms
from hu_clr.datasets.transformations import *


def rotate(image, angle):
    [c, w, h] = image.size()
    dim = math.ceil(2 * math.sqrt(h ** 2 / 2))
    exp = dim - h
    # exp has to be divisible by 2
    exp = exp + (exp % 2)

    exp_img = torch.zeros(size=(c, h + exp, h + exp), dtype=image.dtype)
    s = int(exp / 2)
    exp_img[:, s:s + w, s:s + h] = image

    # Mirroring
    left = image[:, 0:s, :]
    left = left.flip(1)  #left[:, ::-1, :]
    exp_img[:, 0:s, s:s + h] = left

    right = image[:, w - s:, :]
    right = right.flip(1)  #right[:, ::-1, :]
    exp_img[:, w + s:, s:s + h] = right

    top = image[:, :, 0:s]
    top = top.flip(2)  #top[:, :, ::-1]
    exp_img[:, s:s + w, 0:s] = top

    bot = image[:, :, h - s:]
    bot = bot.flip(2)  #bot[:, :, ::-1]
    exp_img[:, s:s + w, h + s:] = bot

    # rotated = ndimage.rotate(exp_img, angle, reshape=False)
    # rotated = cv2_rotate_image(exp_img,angle)
    rotated = F.rotate(exp_img, angle, expand=False)
    cropped = rotated[:, s:s + w, s:s + h]

    return cropped


def random_crop_and_resize(img, s=[0.01, 1.0]):
    random_crop = transforms.RandomResizedCrop(size=img.shape[-2:])
    params = random_crop.get_params(torch.tensor(img), scale=s, ratio=[0.95, 1.05])

    #apply = random.choices([True, False], weights=[p, 1-p], k=1)

    cropped_img = F.crop(img, top=params[0], left=params[1], height=params[2], width=params[3])
    img = F.resize(cropped_img, img.shape[-2:])

    return img


def random_rotation_and_flip(img, p=0.5):
    do_flip = random.choices([True, False], weights=[p, 1 - p], k=1)[0]
    if do_flip:
        img = torch.flip(img, dims=[2])  # CHECK DIMENSIONS!!

    rotation_angle = random.choices(np.arange(0, 360, 45), k=1)[0]
    img = rotate(img, int(rotation_angle))

    return img


def hs_distortion(img, sat=(0.6, 1.8), hue=0.05, prob=1.0):
    hs = transforms.ColorJitter(saturation=sat, hue=hue)
    applier = transforms.RandomApply(nn.ModuleList([hs]), p=prob)
    return applier(img)


def color_distortion(img, s=1.0):
    color_jitter = transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)
    # random_crop = transforms.RandomResizedCrop(size=image.shape[-2:], scale=(0.05, 1.0))
    #applier = transforms.RandomApply(nn.ModuleList([color_jitter]), p=prob)

    return color_jitter(img)


def gaussian_blur(img, sigma=(0.1, 2.0), prob=0.5):
    blur = transforms.GaussianBlur(kernel_size=25, sigma=sigma)
    applier = transforms.RandomApply(nn.ModuleList([blur]), p=prob)

    return applier(img)


def load_img_paths(base_dir):
    input_img_paths = []

    for k, file in enumerate(os.listdir(os.path.join(base_dir, 'patches'))):
        img_dir = os.path.join(base_dir, 'patches', file)
        if not os.path.isdir(img_dir):
            continue

        this_input_paths = sorted([
            os.path.join(img_dir, fname)
            for fname in os.listdir(img_dir)
            if fname.endswith(".png")
        ])
        input_img_paths = input_img_paths + this_input_paths

    return input_img_paths


class SimCLRDataset(Dataset):
    def __init__(self,
                 image_paths=None,
                 dataset_paths=None,
                 mode='global'
                 ):
        if image_paths is not None:
            self.image_paths = image_paths
        else:
            self.image_paths = []
            for dataset in dataset_paths:
                self.image_paths = self.image_paths + load_img_paths(dataset)

        self.mode = mode
        if self.mode == 'global':
            self._augmentation_1 = random_crop_and_resize
            self._augmentation_2 = color_distortion
        else:
            self._augmentation_1 = gaussian_blur
            self._augmentation_2 = color_distortion
        self.input_dtype = torch.float32

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Generate one sample of data
        image = imread(self.image_paths[index])
        image = np.moveaxis(image.squeeze(), source=-1, destination=0)
        image = torch.from_numpy(image)

        sample_1, _ = self._augmentation_1(image)
        sample_2 = self._augmentation_2(image)

        return sample_1.type(self.input_dtype), sample_2.type(self.input_dtype)


class HCLDataset(Dataset):
    def __init__(self,
                 image_paths=None,
                 dataset_paths=None
                 ):
        if image_paths is not None:
            self.image_paths = image_paths
        else:
            self.image_paths = []
            for dataset in dataset_paths:
                self.image_paths = self.image_paths + load_img_paths(dataset)

        self.input_dtype = torch.float32

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = imread(self.image_paths[index])
        image = np.moveaxis(image.squeeze(), source=-1, destination=0)
        image = torch.from_numpy(image)

        sample_1 = self._augmentation(image)
        sample_2 = self._augmentation(image)

        return sample_1.type(self.input_dtype), sample_2.type(self.input_dtype)

    def _augmentation(self, image):
        x = random_rotation_and_flip(image)
        x = random_crop_and_resize(x)
        x = color_distortion(x)
        x = gaussian_blur(x)

        return x


def test_augmentations():
    base_dir = "/home/fi5666wi/Documents/Prostate_images/Patches_299/2023-02-02" # "C:\\Users\\fi5666wi\\Documents\\Prostate images\\Datasets\\small_set"
    img_dir = os.path.join(base_dir, 'patches', '11PL 08367-8_10x')
    this_input_paths = sorted([
        os.path.join(img_dir, fname)
        for fname in os.listdir(img_dir)
        if fname.endswith(".png")
    ])
    input_img_paths = this_input_paths[50:60]

    dataset = HCLDataset(image_paths=input_img_paths)
    loader = DataLoader(dataset, batch_size=1)

    for f1, f2 in loader:
        f1 = f1.cpu().numpy()
        f2 = f2.cpu().numpy()
        f1 = np.moveaxis(f1.squeeze(), source=0, destination=-1)
        f2 = np.moveaxis(f2.squeeze(), source=0, destination=-1)
        f1 = cast(f1)
        f2 = cast(f2)
        cv2.imshow("image_1", f1)
        cv2.imshow("image_2", f2)
        cv2.waitKey(1000)


def cast(data):
    img = data/np.amax(data)
    img = img*255
    return np.uint8(img)


class GleasonDataset(Dataset):
    def __init__(self,
                 datasets,
                 labels=['benign', 'malignant'],  # or [gleason grades]
                 mode='pca',
                 patch_size=299,
                 shuffle=True):  # pca for cancer detection, gleason for gleason grading

        self.mode = mode
        self.labels = labels
        self.patch_size = patch_size
        self.image_paths, self.targets = self._load_img_paths(datasets)
        if shuffle:
            rand = random.Random(0)
            rand.shuffle(self.image_paths)
            rand.shuffle(self.targets)

        self.input_dtype = torch.float32

    def _load_img_paths(self, datasets):
        input_img_paths = []
        targets = []

        for base_dir in datasets:
            for gg in ['benign', 'G3', 'G4', 'G5']:
                img_dir = os.path.join(base_dir, gg)
                for fname in os.listdir(img_dir):
                    if fname.endswith(".jpg"):
                        input_img_paths.append(os.path.join(img_dir, fname))
                        targets.append(gg)

        return input_img_paths, targets

    def __len__(self):
        return len(self.image_paths)


    def get_weights(self):
        instances = torch.zeros(len(self.labels))
        for y in self.targets:
            instances[self.labels.index(y)] += 1
        weights = torch.max(instances)/instances
        return weights

    def __getitem__(self, index):
        # Generate one sample of data
        image = imread(self.image_paths[index])  # Rows, Columns, Channels
        if self.patch_size != 299:
            s = int((299 - self.patch_size) / 2)
            image = image[s:s+self.patch_size, s:s+self.patch_size, :]

        if self.mode == 'pca':
            label = self.targets[index] != 'benign'
        elif self.mode == 'gleason':
            label = self.labels.index(self.targets[index])
        else:
            NotImplementedError()
        image = np.moveaxis(image.squeeze(), source=-1, destination=0)
        image = torch.from_numpy(image)

        return image.type(self.input_dtype), label


def test_gleason():
    datasets = ['/home/fi5666wi/Documents/Gleason/Data_1_SUS/20181120_SUS/train_299']
    dataset = GleasonDataset(datasets)
    loader = DataLoader(dataset, batch_size=1)
    weights = dataset.get_weights()
    print(weights)



if __name__ == "__main__":
    print("try to run me")
    #test_augmentations()
    test_gleason()
