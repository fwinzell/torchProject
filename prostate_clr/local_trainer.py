import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import shutil
import sys
import pickle

from hu_clr.datasets.two_dim.ProstateImageLoader import ProstateDataset, load_img_paths
from prostate_clr.datasets.image_loaders import LocalImageDataset
from hu_clr.datasets.transformations import *
from prostate_clr.losses.ch_local_loss import ChLocalLoss
from models import LocalUnet, CNN

apex_support = False

import numpy as np

torch.manual_seed(0)


class LocalCLRTrainer(object):

    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(os.path.join(self.config.save_dir, 'tensorboard'))

        self.input_shape = self.config.input_shape
        self.img_rep_sz = config.input_shape[-1] / 2**(config.depth - config.no_decoder_blocks)

        # Local contrastive loss
        # CHECK arguments, config
        self.local_loss = ChLocalLoss(device=self.device, img_rep_size=self.img_rep_sz, config=config)

        # Set up data loaders
        tr_paths, val_paths = load_img_paths(self.config.base_dir, self.config.val_split)
        # Define transformations
        transform = ComposeSingle([
            FunctionWrapperSingle(normalize_01),
            FunctionWrapperSingle(np.moveaxis, source=-1, destination=0)
        ])

        tr_dataset = LocalImageDataset(tr_paths, transform=transform)
        val_dataset = LocalImageDataset(val_paths, transform=transform)

        self.tr_loader = DataLoader(tr_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)

        self.model = LocalUnet(config)
        embed_dim = config.filters * 2**(config.depth - config.no_decoder_blocks)
        # out_channels = embed_dim in Chaintanya et. al.
        self.head = CNN(in_channels=embed_dim, out_channels=embed_dim)

        self.model.to(self.device)
        self.head.to(self.device)

        # Load pre-trained weights
        state_dict = torch.load(config.pretrained_model_path)
        # If strict=True then keys of state_dict must match keys of model
        self.model.load_state_dict(state_dict, strict=True)

        # Freeze encoder
        self.model.freeze_encoder()
        # freeze certain layer if required
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = torch.optim.Adam(parameters, lr=self.config.learning_rate)

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def train(self):
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.tr_loader), eta_min=0,
                                                               last_epoch=-1)

        for epoch_counter in range(self.config.epochs):
            print("=====Training Epoch: %d =====" % epoch_counter)
            for i, x in enumerate(self.tr_loader):
                self.optimizer.zero_grad()

                x = x.float().to(self.device)
                r = self.model(x)  # Representation
                z = self.head(r)  # Projection
                # Normalize projections
                z = F.normalize(z, dim=1)  ### CHECK that this is correct dimension

                loss = self.local_loss(z)

                if n_iter % self.config.log_every_n_steps == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print("Train:[{0}][{1}][{2}] loss: {loss:.4f}".format(epoch_counter, i, len(self.tr_loader),
                                                                          loss=loss.item()))

                loss.backward()
                self.optimizer.step()
                n_iter += 1

            print("===== Validation =====")
            # validate the model if requested
            if epoch_counter % self.config.eval_every_n_epochs == 0:
                valid_loss = self._validate(self.val_loader)
                print("Val:[{0}] loss: {loss:.4f}".format(epoch_counter, loss=valid_loss))
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), os.path.join(self.config.save_dir,
                                                                     'simclr_{}_model.pth'.format(
                                                                         self.config.batch_size)))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

    def _validate(self, valid_loader):

        # validation steps
        with torch.no_grad():
            self.model.eval()

            valid_loss = 0.0
            counter = 0
            for x in valid_loader:
                x = x.float().to(self.device)

                r = self.model(x)  # Representation
                z = self.head(r)  # Projection
                # Normalize projections
                z = F.normalize(z, dim=1)  ### CHECK that this is correct dimension

                loss = self.local_loss(z)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        return valid_loss








