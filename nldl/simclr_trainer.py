import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from hu_clr.loss_functions.nt_xent import NTXentLoss
from torchlars import LARS
import os
import shutil
import sys
import pickle

from nldl.image_loaders import SimCLRDataset, HCLDataset
from hu_clr.datasets.transformations import *
from resnet import resnet50, resnet101, resnet34

apex_support = False

import numpy as np

torch.manual_seed(0)


class MLP(nn.Module):
    def __init__(self, input_channels=512, num_class=128):
        super().__init__()
        self.f1 = nn.Linear(input_channels, input_channels)
        self.f2 = nn.Linear(input_channels, num_class)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.f1(x.squeeze())
        y = self.relu(y)
        y = self.f2(y)
        return y


class SimCLRTrainer(object):

    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(os.path.join(self.config.save_dir, 'tensorboard'))
        self.nt_xent_criterion = NTXentLoss(self.device, self.config.loss_temperature,
                                            self.config.use_cosine_similarity)
        self.input_shape = self.config.input_shape

        #tr_dataset = SimCLRDataset(dataset_paths=self.config.data_paths, mode='global')
        tr_dataset = HCLDataset(dataset_paths=self.config.data_paths)
        #val_dataset = ProstateDataset(val_paths, mode='global', transform=transform)

        self.train_loader = DataLoader(tr_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        #self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)

        print(len(self.train_loader))
        if self.config.which_resnet == 'resnet34':
            self.model = resnet34(self.config)
        elif self.config.which_resnet == 'resnet50':
            self.model = resnet50(self.config)
        elif self.config.which_resnet == 'resnet101':
            self.model = resnet101(self.config)
        else:
            NotImplementedError("Architecture not implemented")
        self.head = MLP(input_channels=self.config.model_embed_dim, num_class=self.config.model_out_dim)

        if self.config.resume_training:
            model_weights = torch.load(self.config.transfer_model_path)
            self.model.load_state_dict(model_weights, strict=True)

        self.model.to(self.device)
        self.head.to(self.device)

        base_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
        self.optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), self.config.learning_rate, weight_decay=self.config.weight_decay)

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, head, xis, xjs, n_iter):

        # get the representations and the projections
        ris = model(xis)
        zis = head(ris)

        # get the representations and the projections
        rjs = model(xjs)
        zjs = head(rjs)

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self):
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader), eta_min=0,
        #                                                       last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2,
                                                               verbose=True, threshold=0.001)

        for epoch_counter in range(self.config.start_epoch, self.config.epochs):
            losses = []
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
                    losses.append(loss.item())

                loss.backward()
                self.optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config.save_every_n_epochs == 0 or epoch_counter == self.config.epochs - 1:
                print("===== Saving =====")
                torch.save(self.model.state_dict(), os.path.join(self.config.save_dir,
                                                                'simclr_{}_ep{}_model.pth'.format(
                                                                self.config.date, epoch_counter)))

            avg_loss = np.mean(losses)
            self.writer.add_scalar('epoch loss average', avg_loss, global_step=n_iter)
            scheduler.step(avg_loss)
            self.writer.add_scalar('learning rate', self.optimizer.param_groups[0]['lr'], global_step=n_iter)


            # warmup for the first 10 epochs
            #if epoch_counter >= 10:
            #    scheduler.step()
            #self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

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
