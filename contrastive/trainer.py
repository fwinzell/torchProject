import numpy as np
import torch
from distortions import color_distortion, random_crop_and_resize
from torch import nn
import torch.nn.functional as F


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 loss_fn: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_dataloader: torch.utils.data.Dataset,
                 validation_dataloader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 ):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_dataloader
        self.validation_DataLoader = validation_dataloader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

    def run_trainer(self):
        from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.step(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.step()  # learning rate scheduler step
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):
        from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            # x = x.float()
            # y = y.float()
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input)  # one forward pass
            loss = self.loss_fn(out, target)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):
        from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(input)
                loss = self.loss_fn(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()


class PreTrainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 contrastive_loss: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_dataloader: torch.utils.data.Dataset,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 ):
        self.model = model
        self.contrastive_loss = contrastive_loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_dataloader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch

        self.training_loss = []
        self.learning_rate = []

    def run_trainer(self):
        from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._pretrain()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()  # learning rate scheduler step
        return self.training_loss, self.learning_rate

    def _pretrain(self):
        from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(self.training_DataLoader, 'Training', total=len(self.training_DataLoader),
                          leave=False)

        for (x, y) in batch_iter:
            x, y = torch.squeeze(x), torch.squeeze(y)
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            #start = torch.cuda.Event(enable_timing=True)
            #end = torch.cuda.Event(enable_timing=True)

            distorted = color_distortion(input)
            distorted, dist_tar = random_crop_and_resize(distorted, target)

            # zerograd the parameters
            self.optimizer.zero_grad()
            #for param in self.model.parameters():
            #    param.grad = None

            #start.record()
            # one forward pass
            f_orig = self.model(torch.unsqueeze(input[0], dim=0))  # i
            f_dist = self.model(distorted)  # ii,jj
            dim = f_orig.shape[-1]
            i = self.downsample_label_map(target[0], (dim, dim))
            ii = self.downsample_label_map(dist_tar[0], (dim, dim))
            jj = self.downsample_label_map(dist_tar[1], (dim, dim))
            #end.record()
            #torch.cuda.synchronize()
            #print('Forward pass: ' + str(start.elapsed_time(end)))

            loss = self.contrastive_loss(f_orig[0], f_dist[0], f_dist[1], i, ii, jj) # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            #end.record()
            #torch.cuda.synchronize()
            #print('Loss calc: ' + str(start.elapsed_time(end)))

            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            #end.record()
            #torch.cuda.synchronize()
            #print('Backward pass: ' + str(start.elapsed_time(end)))
            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def downsample_label_map(self, target, dim):
        w = target.shape[-1]
        target = torch.reshape(target, (1, 1, w, w))
        low_res = F.interpolate(target.float(), size=dim, mode='nearest')
        return torch.squeeze(low_res.int())
