import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from skimage.io import imread
from torchvision import transforms
from hu_clr.datasets.transformations import *
from matplotlib import pyplot as plt


class LocalImageDataset(Dataset):
    def __init__(self,
                 image_paths,
                 s=1.0,
                 transform=None):
        self.image_paths = image_paths
        self.transform = transform

        self.augmentation = transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)

        self.input_dtype = torch.float32
        self.target_dtype = torch.long

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Generate one sample of data
        image = imread(self.image_paths[index])
        if self.transform is not None:
            image = self.transform(image)

        # Typecasting
        image = torch.from_numpy(image).type(self.input_dtype)

        sample_1 = self.augmentation(image)
        sample_2 = self.augmentation(image)

        return [sample_1, sample_2]


def test_augmentations():
    base_dir = "/Users/filipwinzell/Documents/train_data_with_gt/small_set"
    img_dir = os.path.join(base_dir, 'Patches')
    this_input_paths = sorted([
        os.path.join(img_dir, fname)
        for fname in os.listdir(img_dir)
        if fname.endswith(".png")
    ])
    input_img_paths = this_input_paths

    transform = ComposeSingle([
        FunctionWrapperSingle(np.moveaxis, source=-1, destination=0)
    ])

    dataset = LocalImageDataset(input_img_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=1)

    for f1, f2 in loader:
        f1 = f1.cpu().numpy()
        f2 = f2.cpu().numpy()
        f1 = np.moveaxis(f1.squeeze(), source=0, destination=-1)
        f2 = np.moveaxis(f2.squeeze(), source=0, destination=-1)
        f1 = cast(f1)
        f2 = cast(f2)
        display(f1, "img1.png")
        display(f2, "img2.png")


def display(data, s):
    img = Image.fromarray(data)
    img.show()


def cast(data):
    img = data/np.amax(data)
    img = img*255
    return np.uint8(img)


if __name__ == "__main__":
    print("try to run me")
    test_augmentations()
