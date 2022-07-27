import os
import fnmatch
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.io import imread
from .distortion import random_crop_and_resize, color_distortion, gaussian_blur


def load_img_paths(base_dir, val_split=0.1):
    input_img_paths = []

    for k, folder in enumerate(os.listdir(base_dir)):
        img_dir = os.path.join(base_dir, folder, 'Patches')

        this_input_paths = sorted([
            os.path.join(img_dir, fname)
            for fname in os.listdir(img_dir)
            if fname.endswith(".png")
        ])
        input_img_paths = input_img_paths + this_input_paths

    if val_split == 0.0:
        return input_img_paths
    else:
        return _split(input_img_paths, val_split=val_split)


def _split(input_img_paths, target_img_paths=None, val_split=0.1):
    val_samples = int(len(input_img_paths) * val_split)
    random.Random(1337).shuffle(input_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]

    if target_img_paths is not None:
        random.Random(1337).shuffle(target_img_paths)
        train_target_img_paths = target_img_paths[:-val_samples]
        val_target_img_paths = target_img_paths[-val_samples:]

        return train_input_img_paths, train_target_img_paths, val_input_img_paths, val_target_img_paths
    else:
        return train_input_img_paths, val_input_img_paths


def load_img_and_target_paths(base_dir, val_split=0.1):
    input_img_paths = []
    target_img_paths = []

    for k, folder in enumerate(os.listdir(base_dir)):
        img_dir = os.path.join(base_dir, folder, 'Patches')
        target_dir = os.path.join(base_dir, folder, 'Labels')

        this_input_paths = sorted([
            os.path.join(img_dir, fname)
            for fname in os.listdir(img_dir)
            if fname.endswith(".png")
        ])
        input_img_paths = input_img_paths + this_input_paths

        this_target_paths = sorted([
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ])
        target_img_paths = target_img_paths + this_target_paths

    if val_split == 0.0:
        return input_img_paths, target_img_paths
    else:
        return _split(input_img_paths, target_img_paths, val_split=val_split)


class ProstateDataset(Dataset):
    def __init__(self,
                 image_paths,
                 target_paths=None,
                 mode='global',
                 transform=None):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transform = transform
        self.mode = mode
        if self.mode == 'global':
            self._augmentation_1 = random_crop_and_resize
            self._augmentation_2 = color_distortion
        else:
            self._augmentation_1 = gaussian_blur
            self._augmentation_2 = color_distortion
        self.input_dtype = torch.float32
        self.target_dtype = torch.long

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Generate one sample of data
        image = imread(self.image_paths[index])
        if self.mode == 'global':
            if self.transform is not None:
                image = self.transform(image)

            # Typecasting
            image = torch.from_numpy(image).type(self.input_dtype)

            sample_1, _ = self._augmentation_1(image)
            sample_2 = self._augmentation_2(image)

            return sample_1, sample_2
        elif self.mode == 'local':
            if self.transform is not None:
                image = self.transform(image)

            # Typecasting
            image = torch.from_numpy(image).type(self.input_dtype)

            sample = self._augmentation_1(image)
            sample = self._augmentation_2(sample)

            return sample
        elif self.mode == 'sup':
            target = imread(self.target_paths[index])

            if self.transform is not None:
                image, target = self.transform(image, target)

            # Typecasting
            image, target = torch.from_numpy(image).type(self.input_dtype), torch.from_numpy(target).type(
                self.target_dtype)

            sample = self._augmentation_1(image)
            sample = self._augmentation_2(sample)

            return sample, target
        elif self.mode == 'seg':
            target = imread(self.target_paths[index])

            if self.transform is not None:
                image, target = self.transform(image, target)

            # Typecasting
            image, target = torch.from_numpy(image).type(self.input_dtype), torch.from_numpy(target).type(
                self.target_dtype)

            return image, target
        else:
            raise NotImplementedError('invalid mode')


