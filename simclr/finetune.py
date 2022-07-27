import torch
from torch import nn
from torchsummary import summary
from model import UnetSimCLR, FineTunedUnet
from trainer import Trainer
from data import ProstateImageDataset, class_weights
from unet.transformations import ComposeDouble, FunctionWrapperDouble, normalize_01, one_hot_target
from visual import plot_training_opt
from torch.utils.data import DataLoader
from unet.losses import GeneralizedDiceLoss
import argparse
import os
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
assert torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--num_classes", type=int, default=3)
parser.add_argument("--shuffle", type=bool, default=True)
parser.add_argument("--val_split", type=float, default=0.1)
parser.add_argument("--dir",
                    default='/home/fi5666wi/Documents/Prostate images/train_data_with_gt_3_classes')
parser.add_argument("--model",
            default='/home/fi5666wi/Documents/Python/saved_models/simclr_model_2022-06-23')
parser.add_argument("--input_dim", nargs=3, type=int, default=[256, 256, 3])
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--decoding_steps", type=int, default=3)
parser.add_argument("--filters", type=int, default=64)
parser.add_argument("--savedir",
                    default='/home/fi5666wi/Documents/Python/saved_models')

config = parser.parse_args()

device = torch.cuda.current_device()
pretrained = UnetSimCLR(config).to(device)
pt_weights = torch.load(os.path.join(config.model, 'pretrained_model.pt'))
pretrained.load_state_dict(pt_weights)

model = FineTunedUnet(pretrained, config).to(device)

input_img_paths = []
target_img_paths = []

for k, folder in enumerate(os.listdir(config.dir)):
    img_dir = os.path.join(config.dir, folder, 'Patches')
    target_dir = os.path.join(config.dir, folder, 'Labels')

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

# Split our img paths into a training and a validation set
val_samples = int(len(input_img_paths) * config.val_split)
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Calculate class weights
labels = np.array(range(config.num_classes))
weights = class_weights(target_img_paths[:100], labels)
class_weights = torch.FloatTensor(weights).cuda()

# Define transformations
# FunctionWrapperDouble(one_hot_target, input=False, target=True, nclasses=config.num_classes)
transform = ComposeDouble([
    FunctionWrapperDouble(normalize_01),
    FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    FunctionWrapperDouble(one_hot_target, input=False, target=True, nclasses=config.num_classes)
])

dataset_train = ProstateImageDataset(image_paths=train_input_img_paths, target_paths=train_target_img_paths,
                                     transform=transform)
dataset_val = ProstateImageDataset(image_paths=val_input_img_paths, target_paths=val_target_img_paths,
                                   transform=transform)

train_dataloader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=config.shuffle)
val_dataloader = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=config.shuffle)


# criterion
# MaxDiceLoss(labels)
# GeneralizedDiceLoss(labels)
# torch.nn.CrossEntropyLoss(weight=class_weights)
loss_fn = GeneralizedDiceLoss(labels)

# optimizer
# torch.optim.Adam(model.parameters(), lr=0.001)
# torch.optim.SGD(model.parameters(), lr=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# trainer
# torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5)
trainer = Trainer(model=model,
                  device=torch.device(device),
                  loss_fn=loss_fn,
                  optimizer=optimizer,
                  training_dataloader=train_dataloader,
                  validation_dataloader=val_dataloader,
                  lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5),
                  epochs=config.epochs,
                  epoch=0)

# start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()

# plot losses
fig = plot_training_opt(training_losses, lr_rates, validation_losses=validation_losses)
fig.show()

# save the model
savepath = config.model

torch.save(model.state_dict(), os.path.join(savepath, 'model-3.pt'))
fig.savefig(os.path.join(savepath, 'train_fig-3.png'))


