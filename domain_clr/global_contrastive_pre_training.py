import torch
from torch import nn
from torchsummary import summary
from models import Encoder
from trainers import PreTrainer
from simclr.data import ContrastiveImageDataset, class_weights
from unet.transformations import ComposeSingle, FunctionWrapperSingle, normalize_01, one_hot_target
from visual import plot_training_opt
from torch.utils.data import DataLoader
from simclr_loss import InfoNCELoss
import argparse
import os
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
assert torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--num_classes", type=int, default=3)
parser.add_argument("--shuffle", type=bool, default=True)
parser.add_argument("--val_split", type=float, default=0.1)
parser.add_argument("--dir",
                    default='/home/fi5666wi/Documents/Prostate images/unlabeled')
parser.add_argument("--input_dim", nargs=3, type=int, default=[256, 256, 3])
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--filters", type=int, default=64)
parser.add_argument("--savedir",
                    default='/home/fi5666wi/Documents/Python/saved_models')

config = parser.parse_args()
device = torch.cuda.current_device()

encoder = Encoder(config).to(device)

# Get image paths
input_img_paths = []
for k, folder in enumerate(os.listdir(config.dir)):
    img_dir = os.path.join(config.dir, folder, 'Patches')

    this_input_paths = sorted([
        os.path.join(img_dir, fname)
        for fname in os.listdir(img_dir)
        if fname.endswith(".png")
    ])
    input_img_paths = input_img_paths + this_input_paths

# Make sure we get an even number of batches
res = len(input_img_paths) % config.batch_size
input_img_paths = input_img_paths[:-res]

# Define transformations
transform = ComposeSingle([
    FunctionWrapperSingle(normalize_01),
    FunctionWrapperSingle(np.moveaxis, source=-1, destination=0)
])

dataset_pretrain = ContrastiveImageDataset(image_paths=input_img_paths, transform=transform)
pretrain_dataloader = DataLoader(dataset_pretrain, batch_size=config.batch_size, shuffle=True)

