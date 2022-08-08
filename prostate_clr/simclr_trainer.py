import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from hu_clr.loss_functions.nt_xent import NTXentLoss
import os
import shutil
import sys
import pickle

from hu_clr.datasets.two_dim.ProstateImageLoader import ProstateDataset, load_img_paths
from hu_clr.datasets.transformations import *
from models import Encoder, MLP

apex_support = False

import numpy as np

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(os.path.join(self.config.save_dir, 'tensorboard'))
        self.nt_xent_criterion = NTXentLoss(self.device, self.config.loss_temperature,
                                            self.config.use_cosine_similarity)
        self.input_shape = self.config.input_shape

        tr_paths, val_paths = load_img_paths(self.config.base_dir, self.config.val_split)
        # Define transformations
        transform = ComposeSingle([
            FunctionWrapperSingle(normalize_01),
            FunctionWrapperSingle(np.moveaxis, source=-1, destination=0)
        ])
        tr_dataset = ProstateDataset(tr_paths, mode='global', transform=transform)
        val_dataset = ProstateDataset(val_paths, mode='global', transform=transform)

        self.train_loader = DataLoader(tr_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)

        print(len(self.train_loader))
        self.model = Encoder(self.config)
        self.head = MLP(input_channels=self.config.model_embed_dim, num_class=self.config.model_out_dim)

        self.model.to(self.device)
        self.head.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), 3e-4, weight_decay=eval(self.config.weight_decay))

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, head, xis, xjs, n_iter):

        # get the representations and the projections
        ris = model(xis)  # [N,C]
        zis = head(ris)

        # get the representations and the projections
        rjs = model(xjs)  # [N,C]
        zjs = head(rjs)

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self):

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader), eta_min=0,
                                                               last_epoch=-1)

        for epoch_counter in range(self.config.epochs):
            print("=====Training Epoch: %d =====" % epoch_counter)
            for i, (xis, xjs) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                xis = xis.float().to(self.device)
                xjs = xjs.float().to(self.device)

                loss = self._step(self.model, self.head, xis, xjs, n_iter)

                if n_iter % self.config.log_every_n_steps == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print("Train:[{0}][{1}][{2}] loss: {loss:.4f}".format(epoch_counter, i, len(self.train_loader),
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
                                                                     'b_{}_model.pth'.format(
                                                                         self.config["batch_size"])))

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
            for (xis, xjs) in valid_loader:
                xis = xis.float().to(self.device)
                xjs = xjs.float().to(self.device)

                loss = self._step(self.model, self.head, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        return valid_loss
