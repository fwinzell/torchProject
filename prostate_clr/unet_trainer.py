from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


from hu_clr.datasets.two_dim.ProstateImageLoader import load_img_and_target_paths
from hu_clr.datasets.transformations import *
from unet.data import ProstateImageDataset, class_weights
from unet.losses import GeneralizedDiceLoss
from hu_clr.loss_functions.dice_loss import SoftDiceLoss
from hu_clr.loss_functions.metrics import dice_pytorch
from prostate_clr.models import Unet


class UnetTrainer(object):

    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(os.path.join(self.config.save_dir, 'tensorboard'))

        self.input_shape = self.config.input_shape

        tr_img_paths, tr_tar_paths, val_img_paths, val_tar_paths = load_img_and_target_paths(self.config.base_dir,
                                                                                             self.config.val_split)
        # Define transformations
        transform = ComposeDouble([
            FunctionWrapperDouble(normalize_01),
            FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
            FunctionWrapperDouble(one_hot_target, input=False, target=True, nclasses=config.num_classes)
        ])

        tr_dataset = ProstateImageDataset(image_paths=tr_img_paths, target_paths=tr_tar_paths,
                                             transform=transform)
        val_dataset = ProstateImageDataset(image_paths=val_img_paths, target_paths=val_tar_paths,
                                           transform=transform)

        self.train_loader = DataLoader(tr_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)

        # Set up Unet model
        self.model = Unet(config)
        self.model.to(self.device)

        # Load pre-trained weights
        state_dict = torch.load(config.pretrained_model_path)
        # If strict=True then keys of state_dict must match keys of model
        self.model.load_state_dict(state_dict, strict=True)

        # Loss functions
        # Calculate class weights
        labels = np.array(range(config.num_classes))

        label_weights = class_weights(tr_tar_paths[:100], labels)
        label_weights = torch.FloatTensor(label_weights).cuda()
        self.wcee_loss = nn.CrossEntropyLoss(weight=label_weights)
        self.gd_loss = GeneralizedDiceLoss(labels=labels)
        self.sd_loss = SoftDiceLoss(batch_dice=True, do_bg=False)

        # Optimizer
        # freeze certain layer if required
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        if config.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(parameters, lr=config.learning_rate,
                                              weight_decay=eval(self.config.weight_decay))
        elif config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(parameters, lr=config.learning_rate,
                                             weight_decay=eval(self.config.weight_decay), momentum=config.sgd_momentum)
        else:
            raise NotImplementedError('Invalid optimizer')

    # Train model

    def train(self):
        inp = None
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config.epochs):
            print("=====Training Epoch: %d =====" % epoch_counter)
            self.model.train()  # train mode
            train_losses = []  # accumulate the losses here

            for i, (x, y) in enumerate(self.train_loader):
                inp = x.float().to(self.device)
                tar = y.long().to(self.device)  # send to device (GPU or CPU)
                self.optimizer.zero_grad()  # zerograd the parameters
                pred = self.model(inp)  # one forward pass
                # We calculate a softmax, because our SoftDiceLoss expects that as an input.
                # The CE-Loss does the softmax internally.
                pred_softmax = F.softmax(pred, dim=1)
                loss = self.wcee_loss(pred, tar) + self.sd_loss(pred_softmax, tar)  # calculate loss

                if n_iter % self.config.log_every_n_steps == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print("Train:[{0}][{1}][{2}] loss: {loss:.4f}".format(epoch_counter, i, len(self.train_loader),
                                                                          loss=loss.item()))

                loss.backward()  # one backward pass
                self.optimizer.step()  # update the parameters
                n_iter += 1

            print("===== Validation =====")
            # validate the model if requested
            if epoch_counter % self.config.eval_every_n_epochs == 0:
                valid_loss, valid_dice = self._validate()
                print("Val:[{0}] dice:{dice:.4f} loss: {loss:.4f}".format(epoch_counter, dice=valid_dice, loss=valid_loss))
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), os.path.join(self.config.save_dir,
                                                                     'unet_best_{}_model.pth'.format(
                                                                         self.config.batch_size)))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

        assert inp is not None, 'data is None. Please check if your dataloader works properly'

    def _validate(self):
        self.model.eval()

        data = None
        loss_list = []
        dice_list = []

        with torch.no_grad():
            for x, y in self.val_loader:
                data = x.float().to(self.device)
                target = y.long().to(self.device)

                pred = self.model(data)
                pred_softmax = F.softmax(pred)

                pred_image = torch.argmax(pred_softmax, dim=1)
                dice_result = dice_pytorch(outputs=pred_image, labels=target, N_class=self.config.num_classes)
                dice_list.append(dice_result)

                loss = self.wcee_loss(pred, target) + self.sd_loss(pred_softmax, target)
                # self.sd_loss(pred_softmax, target.squeeze())  self.ce_loss(pred, target.squeeze())
                loss_list.append(loss.item())

        assert data is not None, 'data is None. Please check if your dataloader works properly'

        return np.mean(loss_list), np.mean(dice_list)
