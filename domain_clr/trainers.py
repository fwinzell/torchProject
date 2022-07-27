import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class GlobalTrainer:
    def __init__(self,
                 encoder: torch.nn.Module,
                 device: torch.device,
                 optimizer: torch.optim.Optimizer,
                 training_dataloader: torch.utils.data.Dataset,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 enc_out_filters: int = 1024
                 ):
        self.encoder = encoder
        self.device = device
        self.optimizer = optimizer
        self.training_dataloader = training_dataloader
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.epoch = epoch
        self.dim_mlp = enc_out_filters

        self.cos_sim = nn.CosineSimilarity()
        self.temp = 0.05

        self.downsample = nn.AdaptiveAvgPool2d(1)

        self.projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.dim_mlp, self.dim_mlp),
            nn.ReLU(),
            nn.Linear(self.dim_mlp, 128))


    def run_trainer(self):
        from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._global_pre_train()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()  # learning rate scheduler step

        return self.training_loss, self.learning_rate

    def _global_pre_train(self):
        from tqdm import tqdm, trange

        self.encoder.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)

        for (x_1, x_2) in batch_iter:
            x = torch.cat((x_1, x_2))
            input = x.to(self.device)  # send to device (GPU or CPU)

            self.optimizer.zero_grad()  # zerograd the parameters
            enc_out, _ = self.encoder(input)  # one forward pass
            proj_out = self.projection_head(self.downsample(enc_out))

            global_loss = self._global_loss(proj_out)  # calculate loss
            loss_value = global_loss.item()
            train_losses.append(loss_value)
            global_loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()


    def _global_loss(self, z):
        bs = 2 * self.batch_size
        net_global_loss = 0
        for pos_index in range(0, self.batch_size, 1):
            # indexes of positive pair of samples (x_1,x_2)
            num_i1 = torch.arange(pos_index, pos_index + 1, dtype=torch.int)
            j = self.batch_size + pos_index
            num_i2 = torch.arange(j, j + 1, dtype=torch.int)

            # indexes of corresponding negative samples as per positive pair of samples: (x_1,x_2)
            den_indexes = torch.cat(
                torch.arange(0, pos_index, dtype=torch.int),
                torch.arange(pos_index + 1, j, dtype=torch.int),
                torch.arange(j + 1, bs, dtype=torch.int))

            # gather required positive samples x_1,x_2,x_3 for the numerator term
            #  x_num_i1=tf.gather(reg_pred,num_i1)
            x_num_i1 = torch.index_select(z, 0, num_i1)
            x_num_i2 = torch.index_select(z, 0, num_i2)

            # gather required negative samples x_1,x_2,x_3 for the denominator term
            # x_den = tf.gather(reg_pred, den_indexes)
            x_den = torch.index_select(z, 0, den_indexes)

            # calculate cosine similarity score + global contrastive loss for each pair of positive images

            # for positive pair (x_1,x_2);
            # numerator of loss term (num_i1_i2_ss) & denominator of loss term (den_i1_i2_ss) & loss (num_i1_i2_loss)
            num_i1_i2_ss = self.cos_sim(x_num_i1, x_num_i2) / self.temp
            den_i1_i2_ss = self.cos_sim(x_num_i1, x_den)  / self.temp
            num_i1_i2_loss = -torch.log(
                torch.exp(num_i1_i2_ss) / (torch.exp(num_i1_i2_ss) + torch.sum(torch.exp(den_i1_i2_ss))))
            net_global_loss = net_global_loss + num_i1_i2_loss
            # for positive pair (x_2,x_1);
            # numerator same & denominator of loss term (den_i1_i2_ss) & loss (num_i1_i2_loss)
            den_i2_i1_ss = self.cos_sim(x_num_i2, x_den) / self.temp
            num_i2_i1_loss = -torch.log(
                torch.exp(num_i1_i2_ss) / (torch.exp(num_i1_i2_ss) + torch.sum(torch.exp(den_i2_i1_ss))))
            net_global_loss = net_global_loss + num_i2_i1_loss

        return net_global_loss / bs

