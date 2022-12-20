import os
import fnmatch
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
from hu_clr.datasets.two_dim.distortion import random_crop_and_resize, color_distortion, gaussian_blur

import cv2
from torchvision import transforms
from hu_clr.datasets.transformations import *


def load_img_paths(base_dir, val_split=0.1):
    input_img_paths = []

    for k, file in enumerate(os.listdir(base_dir)):
        img_dir = os.path.join(base_dir, file, 'Patches')
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(base_dir, file)

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

            image = torch.from_numpy(image)

            sample_1, _ = self._augmentation_1(image)
            sample_2 = self._augmentation_2(image)

            return sample_1.type(self.input_dtype), sample_2.type(self.input_dtype)
        elif self.mode == 'local':
            if self.transform is not None:
                image = self.transform(image)

            # Typecasting
            image = torch.from_numpy(image)

            sample = self._augmentation_1(image)
            sample = self._augmentation_2(sample)

            return sample.type(self.input_dtype)
        elif self.mode == 'sup':
            target = imread(self.target_paths[index])

            if self.transform is not None:
                image, target = self.transform(image, target)

            # Typecasting
            image, target = torch.from_numpy(image), torch.from_numpy(target)

            sample = self._augmentation_1(image)
            sample = self._augmentation_2(sample)

            return sample.type(self.input_dtype), target.type(self.target_dtype)
        elif self.mode == 'seg':
            target = imread(self.target_paths[index])

            if self.transform is not None:
                image, target = self.transform(image, target)

            # Typecasting
            image, target = torch.from_numpy(image), torch.from_numpy(target)

            return image.type(self.input_dtype), target.type(self.target_dtype)
        else:
            raise NotImplementedError('invalid mode')


def test_augmentations():
    base_dir = "/home/fi5666wi/Documents/Prostate images/train_data_with_gt/gt2_256" # "C:\\Users\\fi5666wi\\Documents\\Prostate images\\Datasets\\small_set"
    img_dir = os.path.join(base_dir, 'Patches')
    this_input_paths = sorted([
        os.path.join(img_dir, fname)
        for fname in os.listdir(img_dir)
        if fname.endswith(".png")
    ])
    input_img_paths = this_input_paths[50:60]

    transform = ComposeSingle([
        FunctionWrapperSingle(np.moveaxis, source=-1, destination=0)
    ])

    dataset = ProstateDataset(input_img_paths, transform=transform)
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


if __name__ == "__main__":
    print("try to run me")
    test_augmentations()
