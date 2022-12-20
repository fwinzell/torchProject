import os
from datetime import date
import argparse

import torch
from torch import nn
import torch.nn.functional as F

from syngan.models import GenUnet
from hu_clr.datasets.two_dim.ProstateImageLoader import load_img_and_target_paths
from unet.data import ProstateImageDataset
from hu_clr.datasets.transformations import *
from torch.utils.data import DataLoader


def parse_config():
    parser = argparse.ArgumentParser("argument for gan pipeline")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--input_shape", nargs=3, type=int, default=[3, 256, 256])
    parser.add_argument("--out_channels", type=int, default=3)

    parser.add_argument("--base_dir", type=str, default="/home/fi5666wi/Documents/Prostate images/train_data_3_classes")
    parser.add_argument("--save_dir", type=str, default="/home/fi5666wi/Documents/Python/saved_models/syngan"
                                                        "/pt_unet")

    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=10e-6)

    # depth of unet encoder
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--start_filters", type=int, default=64)

    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    args = parser.parse_args()
    return args


def load_img_and_mask_paths(base_dir):
    img_paths = []
    mask_paths = []

    for k, folder in enumerate(os.listdir(base_dir)):
        img_dir = os.path.join(base_dir, folder, 'Patches')
        mask_dir = os.path.join(base_dir, folder, 'Masks')

        this_input_paths = sorted([
            os.path.join(img_dir, fname)
            for fname in os.listdir(img_dir)
            if fname.endswith(".png")
        ])
        img_paths = img_paths + this_input_paths

        this_mask_paths = sorted([
            os.path.join(mask_dir, fname)
            for fname in os.listdir(mask_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ])
        mask_paths = mask_paths + this_mask_paths

    return img_paths, mask_paths


def get_loaders(base_dir, batch_size):
    tr_img_paths, tr_mask_paths = load_img_and_mask_paths(base_dir)

    # Define transformations
    transform = ComposeDouble([
        FunctionWrapperDouble(normalize_01),
        FunctionWrapperDouble(np.moveaxis, input=True, target=True, source=-1, destination=0)
    ])

    tr_dataset = ProstateImageDataset(image_paths=tr_img_paths, target_paths=tr_mask_paths,
                                      transform=transform)
    train_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader


def l2_loss(out, tar):
    diff = torch.sub(out, tar)
    return torch.linalg.norm(diff)


def pretrain():
    config = parse_config()
    loader = get_loaders(config.base_dir, config.batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GenUnet(config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)

    for epoch in range(config.epochs):
        print("________Epoch: {}_________".format(epoch))
        model.train()
        n_iter = 0

        for i, (img, mask) in enumerate(loader):
            img = img.float().to(device)
            mask = mask.float().to(device)  # send to device (GPU or CPU)
            optimizer.zero_grad()  # zerograd the parameters
            syn = model(mask)
            loss = l2_loss(syn, img)

            if n_iter % config.log_every_n_steps == 0:
                print("Train:[{0}][{1}][{2}] loss: {loss:.4f}".format(epoch, i, len(loader),
                                                                      loss=loss.item()))

            loss.backward()
            optimizer.step()
            n_iter += 1

        if epoch % config.save_every_n_epochs == 0 or epoch + 1 == config.epochs:
            # save the model weights
            print('____Saving model____')
            torch.save(model.state_dict(), os.path.join(config.save_dir,
                                                        'pt_{}_model.pth'.format(epoch)))


if __name__ == '__main__':
    pretrain()