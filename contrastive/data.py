import torch
from torch.utils.data import Dataset
import numpy as np
from skimage.io import imread
from torchvision.io import read_image
import random


def class_weights(target_paths, labels):
    weights = np.zeros(len(labels))
    for path in target_paths:
        mask = imread(path)
        mask = np.array(mask)
        for label in labels:
            area = np.sum(np.asarray(mask == label, np.int8))
            weights[label] = weights[label] + area

    weights = 1 / weights
    weights = weights / np.mean(weights)
    return weights


class ProstateImageDataset(Dataset):
    def __init__(self,
                 image_paths,
                 target_paths,
                 transform=None):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Generate one sample of data
        image = imread(self.image_paths[index])
        target = imread(self.target_paths[index])

        j = index
        while j == index:
            j = random.randint(0, (self.__len__()-1))

        random_image = imread(self.image_paths[j])
        random_target = imread(self.target_paths[j])


        # Pre-processing
        target = np.squeeze(target)
        random_target = np.squeeze(random_target)

        if self.transform is not None:
            image, target = self.transform(image, target)
            random_image, random_target = self.transform(random_image, random_target)

        # Typecasting
        image, target = torch.from_numpy(image).type(self.inputs_dtype), torch.from_numpy(target).type(
            self.targets_dtype)
        random_image, random_target = torch.from_numpy(random_image).type(self.inputs_dtype), torch.from_numpy(random_target).type(
            self.targets_dtype)

        return torch.stack((image, random_image)), torch.stack((target, random_target))

