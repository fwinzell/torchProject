import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import tkinter.messagebox

import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from skimage.io import imread
from torchvision import transforms
from hu_clr.datasets.transformations import *
from matplotlib import pyplot as plt
from augmentation.color_transformation import hsv_transformation
import random


class LocalImageDataset(Dataset):
    def __init__(self,
                 image_paths,
                 s=1.0,
                 transform=None):
        self.image_paths = image_paths
        self.transform = transform

        self.augmentation = transforms.ColorJitter(brightness=0.8 * s, contrast=0.8 * s, saturation=0.8 * s,
                                                   hue=0.2 * s)
        self.gaussian = transforms.GaussianBlur(kernel_size=15)

        self.input_dtype = torch.float32
        self.target_dtype = torch.long

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Generate one sample of data
        image = imread(self.image_paths[index])
        if self.transform is not None:
            image = self.transform(image)

        image = torch.from_numpy(image)  # .type(self.input_dtype)

        sample_1 = self.augmentation(image)
        sample_1 = self.gaussian(sample_1)
        sample_2 = self.augmentation(image)
        sample_2 = self.gaussian(sample_2)

        return sample_1.type(self.input_dtype), sample_2.type(self.input_dtype)


class LocalSupDataset(Dataset):
    def __init__(self,
                 image_paths,
                 target_paths=None,
                 s=1.0,
                 single_transform=None,
                 double_transform=None):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.single_transform = single_transform
        self.double_transform = double_transform

        self.augmentation = transforms.ColorJitter(brightness=0.1 * s, contrast=0.1 * s, saturation=0.8 * s,
                                                   hue=0.05 * s)

        self.input_dtype = torch.float32
        self.target_dtype = torch.long

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Generate one sample of data
        image = imread(self.image_paths[index])
        if len(self.target_paths) > index:
            target = imread(self.target_paths[index])
            if self.double_transform is not None:
                image, target = self.double_transform(image, target)
            target = torch.from_numpy(target).type(self.target_dtype)
        else:
            target = None
            if self.single_transform is not None:
                image = self.single_transform(image)

        image = torch.from_numpy(image)  # .type(self.input_dtype)
        aug = self.augmentation(image)

        if target is not None:
            return aug.type(self.input_dtype), target
        else:
            return aug.type(self.input_dtype), []


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


def color_augmentation(rgb_img, h=1.05, s=1.8, v=1.2):
    hsv_img = cv2.cvtColor(np.uint8(rgb_img), cv2.COLOR_RGB2HSV)
    hsv_new = np.zeros_like(hsv_img)
    hsv_new[:, :, 0] = hsv_img[:, :, 0] * random.uniform(1 / h, h)
    hsv_new[:, :, 1] = hsv_img[:, :, 1] * random.uniform(1 / s, s)
    hsv_new[:, :, 2] = hsv_img[:, :, 2] * random.uniform(1 / v, v)

    # hsv_new = np.concatenate((hue, sat, val))
    rgb_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2RGB)
    return rgb_new


def hsv_augmentation(im, h=1.08, s=1.5, v=1.5):
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    # Color augmentation
    hsv = rgb_to_hsv(im / 255)
    # param = [1.06, 1.4, 1.4]  # orig
    # param = [1.08, 1.5, 1.5] #more
    # param = [1.04, 1.3, 1.3] #less
    #mean_hue = np.mean(hsv[:, :, 0])
    #h_max = np.amax([1/mean_hue, h])
    hue = hsv[:, :, 0]
    hue *= random.uniform(1 / h, h)
    #hsv[:, :, 0] = hue % 1
    hsv[:, :, 1] *= random.uniform(1 / s, s)  # 1.3
    hsv[:, :, 2] *= random.uniform(1 / v, v)  # 1.3
    hsv[hsv > 1] = 1
    im = hsv_to_rgb(hsv)
    im = np.uint8(im * 255)
    return im


def hsv_map_augmentation(im, h, s, v):
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    hsv = rgb_to_hsv(im / 255)
    # 1) map hue channel to T:[-0.5, 0.5]
    #mean_map = np.floor(hsv[:, :, 0]/mu) # centered around the mean
    T = hsv[:, :, 0] - 0.5
    # 2) map to infinite real number line
    X = 2*np.tan(T*np.pi)
    # 3) Multiply by factor
    mu = 2*np.tan(0.30*np.pi) #mean factor
    lambd = random.uniform(-h, h)
    #X = ((X - mu)*lambd) + mu
    X = X * lambd
    # 4) Map to circle
    T = np.arctan(X / 2)/np.pi
    # 5) New hue channel
    hsv[:, :, 0] = T + 0.5
    hsv[:, :, 1] *= random.uniform(1 / s, s)  # Saturation
    hsv[:, :, 2] *= random.uniform(1 / v, v)  # Value
    hsv[hsv > 1] = 1
    im = hsv_to_rgb(hsv)
    im = np.uint8(im * 255)
    return im


def grayscale(im):
    from skimage import color
    gray = color.rgb2gray(im)
    return gray

def mean_hue(im):
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    hsv = rgb_to_hsv(im / 255)
    T = hsv[:, :, 0] - 0.5
    mu = np.mean(T)
    return mu


def hue_multiply(im, h=1.08):
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    hsv = rgb_to_hsv(im / 255)
    hsv[:, :, 0] *= random.uniform(1 / h, h)
    hsv[hsv > 1] = 1
    im = hsv_to_rgb(hsv)
    im = np.uint8(im * 255)
    return im


def hsv_jitter(im, h, s, v):
    from torchvision import transforms
    from PIL import Image
    hsv_aug = transforms.ColorJitter(brightness=[1/v, v], saturation=[1/s, s], hue=h)
    pil_img = Image.fromarray(im)
    return np.array(hsv_aug(pil_img))


def hsv_functional_jitter(im, h, s, v, con=1.0):
    import torchvision.transforms.functional as TF
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    import math
    v_factor = float(torch.empty(1).uniform_(1/v, v))
    s_factor = float(torch.empty(1).uniform_(1/s, s))
    h = np.clip(h, -0.4999, 0.4999)
    h_factor = float(torch.empty(1).uniform_(h[0], h[1]))
    h_mult = 2*math.tan(h_factor*math.pi) + 0.9

    transform = transforms.ToTensor()
    torch_img = transform(im/255)
    torch_img = TF.adjust_contrast(torch_img, con)
    torch_img = TF.adjust_brightness(torch_img, v_factor)
    torch_img = TF.adjust_saturation(torch_img, s_factor)
    # torch image is in rgb, this will not work
    numpy_img = np.moveaxis(torch_img.numpy(), source=0, destination=-1)
    numpy_img = rgb_to_hsv(numpy_img)
    numpy_img[0, ...] *= h_mult
    numpy_img = np.clip(numpy_img, 0.0, 1.0)
    numpy_img = hsv_to_rgb(numpy_img)
    numpy_img *= 255
    return numpy_img


def hsv_normal_jitter(im, h, s, v, con=1.0):
    import torchvision.transforms.functional as TF
    v_factor = float(torch.empty(1).uniform_(1/v, v))
    s_factor = float(torch.empty(1).uniform_(1/s, s))
    h_factor = torch.empty(1).normal_(mean=h[0], std=h[1])
    h_factor = h_factor - torch.sign(h_factor - torch.clamp(h_factor, -0.5, 0.5))
    h_factor = float(h_factor)

    im = hue_multiply(im, 1.8)
    transform = transforms.ToTensor()
    torch_img = transform(im/255)
    torch_img = TF.adjust_contrast(torch_img, con)
    torch_img = TF.adjust_brightness(torch_img, v_factor)
    torch_img = TF.adjust_saturation(torch_img, s_factor)
    torch_img = TF.adjust_hue(torch_img, h_factor)
    torch_img *= 255
    return np.moveaxis(torch_img.numpy(), source=0, destination=-1)


class ProstateAugmentationDataset(Dataset):
    def __init__(self,
                 image_paths,
                 transform=None,
                 preprocess='norm',
                 hsv_thresh=235,
                 hsv_factors=[0.25, 1.8, 1.2]):
        self.image_paths = image_paths
        self.transform = transform
        self.preprocess = preprocess
        self.hsv_thresh = hsv_thresh
        [self.h, self.s, self.v] = hsv_factors
        self.inputs_dtype = torch.float32
        self.hue_list = []

        if self.preprocess is None:
            #tkinter.messagebox.showinfo('Note', )
            print("*No preprocessing*")
        elif (self.preprocess != 'norm') and (self.preprocess != 'aug'):
            print("Invalid preprocessing")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Generate one sample of data
        image = imread(self.image_paths[index])
        original = image.copy()

        # Augmentation
        if self.preprocess == 'norm':
            image = hsv_transformation(image, thresh_val=self.hsv_thresh)
        elif self.preprocess == 'aug':
            image = hsv_jitter(image, h=self.h, s=self.s, v=self.v)
        elif self.preprocess == 'augmult':
            image = hsv_augmentation(image, h=self.h, s=self.s, v=self.v)
        elif self.preprocess == 'augf':
            image = hsv_functional_jitter(image, h=self.h, s=self.s, v=self.v)
        elif self.preprocess == 'augmap':
            image = hsv_map_augmentation(image, h=self.h, s=self.s, v=self.v)
        elif self.preprocess == 'meanhue':
            self.hue_list.append(mean_hue(image))
        elif self.preprocess == 'gray':
            image = grayscale(image)

        # Typecasting
        image, original = torch.from_numpy(image).type(self.inputs_dtype), \
                          torch.from_numpy(original).type(self.inputs_dtype)

        return image, original


class ProstateImageDataset2(Dataset):
    def __init__(self,
                 image_paths,
                 target_paths,
                 transform=None,
                 preprocess='norm',
                 hsv_thresh=235,
                 hsv_factors=[0.25, 1.8, 1.2]):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transform = transform
        self.preprocess = preprocess
        self.hsv_thresh = hsv_thresh
        [self.h, self.s, self.v] = hsv_factors
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

        pplist = ['norm', 'aug', 'augf', 'augmult', 'augmap', 'gray']

        if self.preprocess is None:
            print("*No preprocessing*")
        elif self.preprocess not in pplist:
            raise Exception("Invalid preprocessing")



    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Generate one sample of data
        image = imread(self.image_paths[index])
        target = imread(self.target_paths[index])

        # Pre-processing
        target = np.squeeze(target)

        # Augmentation
        if self.preprocess == 'norm':
            image = hsv_transformation(image, thresh_val=self.hsv_thresh)
        elif self.preprocess == 'aug':
            image = hsv_jitter(image, h=self.h, s=self.s, v=self.v)
        elif self.preprocess == 'augmult':
            image = hsv_augmentation(image, h=self.h, s=self.s, v=self.v)
        elif self.preprocess == 'augf':
            image = hsv_functional_jitter(image, h=self.h, s=self.s, v=self.v)
        elif self.preprocess == 'augmap':
            image = hsv_map_augmentation(image, h=self.h, s=self.s, v=self.v)
        elif self.preprocess == 'gray':
            image = grayscale(image)

        if self.transform is not None:
            image, target = self.transform(image, target)

        # Typecasting
        image, target = torch.from_numpy(image).type(self.inputs_dtype), torch.from_numpy(target).type(
            self.targets_dtype)

        return image, target


def test_augmentations():
    base_dir = "/home/fi5666wi/Documents/Prostate images/train_data_with_gt/gt2_256"  # "C:\\Users\\fi5666wi\\Documents\\Prostate images\\Datasets\\small_set"
    img_dir = os.path.join(base_dir, 'Patches')
    this_input_paths = sorted([
        os.path.join(img_dir, fname)
        for fname in os.listdir(img_dir)
        if fname.endswith(".png")
    ])
    input_img_paths = this_input_paths[50:60]

    tar_dir = os.path.join(base_dir, 'Labels')
    this_target_paths = sorted([
        os.path.join(img_dir, fname)
        for fname in os.listdir(img_dir)
        if fname.endswith(".png")
    ])
    target_img_paths = this_target_paths[50:60]

    transform = ComposeDouble([
        FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
    ])

    # dataset = LocalImageDataset(input_img_paths, transform=transform)
    dataset = ProstateImageDataset2(input_img_paths, target_img_paths, transform=transform, preprocess='augmult',
                                    hsv_factors=[1.7, 1.8, 1.2])
    loader = DataLoader(dataset, batch_size=1)

    for im, tar in loader:
        f1 = im.cpu().numpy()
        f1 = np.moveaxis(f1.squeeze(), source=0, destination=-1)
        f1 = cv2.cvtColor(np.uint8(f1), cv2.COLOR_RGB2BGR)
        cv2.imshow("image_1", f1)
        cv2.waitKey(1000)


"""    for y_1, y_2 in loader:
        f1 = y_1.cpu().numpy()
        f2 = y_2.cpu().numpy()
        f1 = np.moveaxis(f1.squeeze(), source=0, destination=-1)
        f2 = np.moveaxis(f2.squeeze(), source=0, destination=-1)
        f1 = cast(f1)
        f2 = cast(f2)
        cv2.imshow("image_1", f1)
        cv2.imshow("image_2", f2)
        cv2.waitKey(1000)"""


def display(data, s):
    img = Image.fromarray(data)
    img.show()


def cast(data):
    img = data / np.amax(data)
    img = img * 255
    return np.uint8(img)


if __name__ == "__main__":
    print("try to run me")
    test_augmentations()
