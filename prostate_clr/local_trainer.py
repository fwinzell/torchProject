import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

import os
from datetime import date
import shutil
import sys
import pickle

from hu_clr.datasets.two_dim.ProstateImageLoader import ProstateDataset, load_img_paths, load_img_and_target_paths
from prostate_clr.datasets.image_loaders import LocalImageDataset, LocalSupDataset
from hu_clr.datasets.transformations import *
from prostate_clr.losses.ch_local_loss import ChLocalLoss
from prostate_clr.losses.supcon_loss import BlockConLoss, StrideConLoss
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
        self.img_rep_sz = config.input_shape[-1] / 2 ** (config.depth - config.no_of_decoder_blocks)

        # Local contrastive loss
        self.local_loss = ChLocalLoss(device=self.device, config=self.config)

        # Set up data loaders
        tr_paths, val_paths = load_img_paths(self.config.base_dir, self.config.val_split)
        # Define transformations
        transform = ComposeSingle([
            FunctionWrapperSingle(normalize_01),
            FunctionWrapperSingle(np.moveaxis, source=-1, destination=0)
        ])

        tr_dataset = LocalImageDataset(tr_paths, transform=transform, s=0.75)
        val_dataset = LocalImageDataset(val_paths, transform=transform, s=0.75)

        self.tr_loader = DataLoader(tr_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)

        self.model = LocalUnet(config)
        embed_dim = config.start_filters * 2 ** (config.depth - config.no_of_decoder_blocks)
        # out_channels = embed_dim in Chaintanya et. al.
        self.head = CNN(in_channels=embed_dim, out_channels=embed_dim)

        self.model.to(self.device)
        self.head.to(self.device)

        # Load pre-trained weights
        state_dict = torch.load(config.pretrained_model_path)
        # If strict=True then keys of state_dict must match keys of model
        self.model.load_state_dict(state_dict, strict=False)

        # Freeze encoder
        self.model.encoder.requires_grad_(False)
        # freeze certain layer if required
        # parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        # Only decoder weights in optimizer
        self.optimizer = torch.optim.Adam(self.model.decoder.parameters(), lr=self.config.learning_rate)

        self.date = date.today()

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def train(self):
        n_iter = 0
        valid_n_iter = 0
        last_valid_loss = np.inf

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.tr_loader), eta_min=0,
        #                                                       last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=2, verbose=True)

        for epoch_counter in range(self.config.epochs):
            print("=====Training Epoch: %d =====" % epoch_counter)
            for i, x in enumerate(self.tr_loader):
                self.optimizer.zero_grad()

                x_1 = x[0].float().to(self.device)
                r_1 = self.model(x_1)  # Representation
                z_1 = self.head(r_1)  # Projection
                # Normalize projections
                z_1 = F.normalize(z_1, dim=1)

                x_2 = x[1].float().to(self.device)
                r_2 = self.model(x_2)  # Representation
                z_2 = self.head(r_2)  # Projection
                # Normalize projections
                z_2 = F.normalize(z_2, dim=1)

                loss = self.local_loss(z_1, z_2)

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
                if valid_loss < last_valid_loss:
                    # save the model weights
                    # best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), os.path.join(self.config.save_dir,
                                                                     'local_pt_{}_{}_model.pth'.format(
                                                                         self.config.batch_size, epoch_counter)))
                last_valid_loss = valid_loss
                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                scheduler.step(valid_loss)
                valid_n_iter += 1

            # warmup for the first 10 epochs
            # if epoch_counter >= 10:
            #    scheduler.step()
            self.writer.add_scalar('lr_decay', self.optimizer.param_groups[0]['lr'], global_step=n_iter)

    def _validate(self, valid_loader):

        # validation steps
        with torch.no_grad():
            self.model.eval()

            valid_loss = 0.0
            counter = 0
            for x in valid_loader:
                x_1 = x[0].float().to(self.device)
                r_1 = self.model(x_1)  # Representation
                z_1 = self.head(r_1)  # Projection
                # Normalize projections
                z_1 = F.normalize(z_1, dim=1)

                x_2 = x[1].float().to(self.device)
                r_2 = self.model(x_2)  # Representation
                z_2 = self.head(r_2)  # Projection
                # Normalize projections
                z_2 = F.normalize(z_2, dim=1)

                loss = self.local_loss(z_1, z_2)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        return valid_loss


class SupCLRTrainer(object):

    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(os.path.join(self.config.save_dir, 'tensorboard'))

        self.input_shape = self.config.input_shape
        self.img_rep_sz = config.input_shape[-1] / 2 ** (config.depth - config.no_of_decoder_blocks)

        # Local contrastive loss
        if config.mode == 'block':
            self.criterion = BlockConLoss(temperature=config.loss_temp, block_size=config.block_size)
        elif config.mode == 'stride':
            self.criterion = StrideConLoss(temperature=config.loss_temp, stride=config.stride)
        else:
            raise NotImplementedError('Wrong mode')

        # Set up data loaders
        # unsup_tr_ims, unsup_val_ims = load_img_paths(self.config.base_dir, self.config.val_split)
        tr_img_paths, tr_tar_paths, val_img_paths, val_tar_paths = load_img_and_target_paths(self.config.sup_dir,
                                                                                             self.config.val_split)

        # Define transformations
        # single_transform = ComposeSingle([
        #    FunctionWrapperSingle(normalize_01),
        #    FunctionWrapperSingle(np.moveaxis, source=-1, destination=0)
        # ])
        double_transform = ComposeDouble([
            FunctionWrapperDouble(normalize_01),
            FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0)
        ])

        tr_dataset = LocalSupDataset(tr_img_paths, target_paths=tr_tar_paths, double_transform=double_transform)
        val_dataset = LocalSupDataset(val_img_paths, target_paths=val_tar_paths, double_transform=double_transform)

        self.tr_loader = DataLoader(tr_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)

        self.config.no_of_decoder_blocks = config.depth  # We always use the last layer, before 1x1 conv, as projection
        self.model = LocalUnet(self.config)
        embed_dim = config.start_filters
        # out_channels = 128 in Hu et. al.
        self.head = CNN(in_channels=embed_dim, out_channels=128)

        self.model.to(self.device)
        self.head.to(self.device)

        # Load pre-trained weights
        state_dict = torch.load(config.pretrained_model_path)
        # If strict=True then keys of state_dict must match keys of model
        self.model.load_state_dict(state_dict, strict=False)

        # Freeze encoder
        self.model.encoder.requires_grad_(False)
        # freeze certain layer if required
        # parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        # Only decoder weights in optimizer
        self.optimizer = torch.optim.Adam(self.model.decoder.parameters(), lr=self.config.learning_rate)

        self.date = date.today()

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def train(self):
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.tr_loader), eta_min=0,
        #                                                       last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=2, verbose=True)

        for epoch_counter in range(self.config.epochs):
            print("=====Training Epoch: %d =====" % epoch_counter)
            for i, batch in enumerate(self.tr_loader):
                self.optimizer.zero_grad()
                torch.autograd.set_detect_anomaly(True)
                x = batch[0].float().to(self.device)
                tar = batch[1]

                r = self.model(x)  # Representation
                z = self.head(r)  # Projection
                # Normalize projections
                z = F.normalize(z, dim=1)
                # Need z of shape [batch_size, n_views, c, w, h]
                z = torch.unsqueeze(z, dim=1)
                # Need tar of shape [batch_size, n_views, w, h]
                tar = torch.unsqueeze(tar, dim=1)
                loss = self.criterion(z, tar)

                if n_iter % self.config.log_every_n_steps == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print("Train:[{0}][{1}][{2}] loss: {loss:}".format(epoch_counter, i, len(self.tr_loader),
                                                                          loss=loss.item()))

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), max_norm=0.1)
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
                                                                     'local_pt_{}_{}_model.pth'.format(
                                                                         self.config.batch_size, epoch_counter)))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                scheduler.step(valid_loss)
                valid_n_iter += 1

            # warmup for the first 10 epochs
            # if epoch_counter >= 10:
            #    scheduler.step()
            self.writer.add_scalar('lr_decay', self.optimizer.param_groups[0]['lr'], global_step=n_iter)

    def _validate(self, valid_loader):

        # validation steps
        with torch.no_grad():
            self.model.eval()

            valid_loss = 0.0
            counter = 0
            for x, tar in valid_loader:
                x = x.float().to(self.device)

                r = self.model(x)  # Representation
                z = self.head(r)  # Projection
                # Normalize projections
                z = F.normalize(z, dim=1)
                # Need z of shape [batch_size, n_views, c, w, h]
                z = torch.unsqueeze(z, dim=1)
                # Need tar of shape [batch_size, n_views, w, h]
                tar = torch.unsqueeze(tar, dim=1)
                loss = self.criterion(z, tar)

                loss = self.criterion(z, tar)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter

        return valid_loss
