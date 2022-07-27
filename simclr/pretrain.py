import torch
from torch import nn
from torchsummary import summary
from model import UnetSimCLR
from trainer import PreTrainer
from data import ContrastiveImageDataset, class_weights
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
parser.add_argument("--decoding_steps", type=int, default=3)
parser.add_argument("--filters", type=int, default=64)
parser.add_argument("--savedir",
                    default='/home/fi5666wi/Documents/Python/saved_models')

config = parser.parse_args()

device = torch.cuda.current_device()
model = UnetSimCLR(config).to(device)
summary = summary(model, (3, 256, 256))

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
# The output from this dataloader? in ordered sets?
# Visualize training example
"""
x_1, x_2 = next(iter(pretrain_dataloader))
for i in range(config.batch_size):
    cv2.imshow('1', np.array(x_1[i, :, :, :].permute(1, 2, 0)))
    cv2.imshow('2', np.array(x_2[i, :, :, :].permute(1, 2, 0)))
    cv2.waitKey(2000)"""

labels = np.array(range(config.num_classes))

# criterion
contrastive_loss_fn = InfoNCELoss(device, config.batch_size)

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=4e-5)

# learning rate schedule
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(3e5))

# trainer
pre_trainer = PreTrainer(model=model,
                  device=torch.device(device),
                  loss_fn=contrastive_loss_fn,
                  optimizer=optimizer,
                  training_dataloader=pretrain_dataloader,
                  lr_scheduler=lr_scheduler,
                  epochs=config.epochs,
                  epoch=0)

# start training
training_losses, lr_rates = pre_trainer.run_trainer()

# plot losses
fig = plot_training_opt(training_losses, lr_rates)
fig.show()

# save the pre trained model
# Create a directory for saving model
savepath = os.path.join(config.savedir, 'simclr_model_' + str(datetime.date.today()))
newpath = savepath
count = 1
while True:
    if os.path.isdir(newpath):
        newpath = savepath + '_' + str(count)
    else:
        break
    count += 1
savepath = newpath
os.mkdir(savepath)

torch.save(model.state_dict(), os.path.join(savepath, 'pretrained_model.pt'))
fig.savefig(os.path.join(savepath, 'pretrain_fig.png'))