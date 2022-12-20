import random
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

from hu_clr.datasets.two_dim.ProstateImageLoader import load_img_and_target_paths
from hu_clr.datasets.two_dim.distortion import hs_distortion
from hu_clr.datasets.transformations import *
from unet.data import ProstateImageDataset, class_weights
from unet.losses import GeneralizedDiceLoss
from hu_clr.loss_functions.dice_loss import SoftDiceLoss
from hu_clr.loss_functions.metrics import dice_pytorch
from prostate_clr.models import Unet, count_parameters
from prostate_clr.datasets.image_loaders import ProstateImageDataset2

import segmentation_models_pytorch as smp


class UnetTrainer(object):

    def __init__(self, config, load_pt_weights=True, use_imagenet=False, use_gdl=True):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(os.path.join(self.config.save_dir, 'tensorboard'))
        self.use_gdl = use_gdl
        self.load_pt_weights = load_pt_weights
        self.use_imagenet = use_imagenet
        self.preprocess = config.preprocess
        self.hsv_factors = config.hsv_factors

        self.input_shape = self.config.input_shape

        tr_img_paths, tr_tar_paths, val_img_paths, val_tar_paths = load_img_and_target_paths(self.config.base_dir,
                                                                                             self.config.val_split)

        #if self.config.eval_every_n_epochs == 0:
            # If no validation, use all images for training
        # ************Change this later: train on all images, validate on some************
        #tr_img_paths = tr_img_paths + val_img_paths
        #tr_tar_paths = tr_tar_paths + val_tar_paths

        # Define transformations
        transform_hot = ComposeDouble([
            FunctionWrapperDouble(normalize_01),
            FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
            FunctionWrapperDouble(one_hot_target, input=False, target=True, nclasses=config.num_classes)
        ])
        transform_ind = ComposeDouble([
            FunctionWrapperDouble(normalize_01),
            FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0)
        ])

        if use_gdl:
            self.transform = transform_hot
        else:
            self.transform = transform_ind

        tr_dataset = ProstateImageDataset2(image_paths=tr_img_paths, target_paths=tr_tar_paths,
                                           transform=self.transform, preprocess=self.preprocess,
                                           hsv_factors=self.hsv_factors)
        val_dataset = ProstateImageDataset2(image_paths=val_img_paths, target_paths=val_tar_paths,
                                            transform=self.transform, preprocess=self.preprocess,
                                            hsv_factors=self.hsv_factors)

        self.train_loader = DataLoader(tr_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)

        # Set up Unet model
        if use_imagenet:
            self.model = smp.Unet(
                encoder_name='resnet50',
                encoder_depth=5,
                encoder_weights='imagenet',
                classes=config.num_classes,
            )
            self.model.to(self.device)
            count_parameters(self.model)
        else:
            self.model = Unet(config)
            self.model.to(self.device)
            count_parameters(model=self.model)
            #summary(self.model, (3, 256, 256))

            if load_pt_weights:
                # Load pre-trained weights
                state_dict = torch.load(config.pretrained_model_path)
                # If strict=True then keys of state_dict must match keys of model
                self.model.load_state_dict(state_dict, strict=False)

        # Loss functions
        # Calculate class weights
        labels = np.array(range(config.num_classes))

        label_weights = class_weights(tr_tar_paths[:100], labels)
        label_weights = torch.FloatTensor(label_weights).cuda()
        self.wcee_loss = nn.CrossEntropyLoss(weight=label_weights)
        self.gd_loss = GeneralizedDiceLoss(labels=labels)
        self.sd_loss = SoftDiceLoss(batch_dice=True, do_bg=False)

        # Optimizer
        if config.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate,
                                              weight_decay=self.config.weight_decay)
        elif config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.learning_rate,
                                             weight_decay=self.config.weight_decay, momentum=config.sgd_momentum)
        else:
            raise NotImplementedError('Invalid optimizer')

        # Lr scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=10,
                                                                    verbose=True, factor=0.1, threshold=1e-5)

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

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
                if inp.dim() < 4:
                    inp = torch.unsqueeze(inp, 1)
                tar = y.long().to(self.device)  # send to device (GPU or CPU)
                self.optimizer.zero_grad()  # zerograd the parameters
                pred = self.model(inp)  # one forward pass
                # We calculate a softmax, because our SoftDiceLoss expects that as an input.
                # The CE-Loss does the softmax internally.
                pred_softmax = F.softmax(pred, dim=1)
                if self.use_gdl:
                    loss = self.gd_loss(pred, tar)  # calculate loss
                else:
                    loss = self.wcee_loss(pred, tar) + self.sd_loss(pred_softmax, tar)
                #

                if n_iter % self.config.log_every_n_steps == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print("Train:[{0}][{1}][{2}] loss: {loss:.4f}".format(epoch_counter, i, len(self.train_loader),
                                                                          loss=loss.item()))

                loss.backward()  # one backward pass
                self.optimizer.step()  # update the parameters
                n_iter += 1

            print("===== Validation =====")
            # validate the model if requested
            if self.config.eval_every_n_epochs == 0:
                torch.save(self.model.state_dict(), os.path.join(self.config.save_dir,
                                                                 'unet_{}_ep{}_model.pth'.format(
                                                                     self.config.batch_size, epoch_counter)))
            elif epoch_counter % self.config.eval_every_n_epochs == 0:
                valid_loss, valid_dice = self._validate()
                print("Val:[{0}] dice:{dice:.4f} loss: {loss:.4f}".format(epoch_counter, dice=valid_dice,
                                                                          loss=valid_loss))
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    print('____New best model____')
                    torch.save(self.model.state_dict(), os.path.join(self.config.save_dir,
                                                                     'unet_best_{}_model.pth'.format(
                                                                         self.config.batch_size)))

                # else:
                #    torch.save(self.model.state_dict(), os.path.join(self.config.save_dir,
                #                                                     'unet_ep{}_model.pth'.format(
                #                                                         epoch_counter)))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                self.writer.add_scalar('valid_dice_score', valid_dice, global_step=valid_n_iter)
                self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], global_step=valid_n_iter)
                self.scheduler.step(valid_loss)
                valid_n_iter += 1

        assert inp is not None, 'data is None. Please check if your dataloader works properly'

    def _validate(self):
        self.model.eval()

        data = None
        valid_loss = 0.0
        valid_dice = 0.0
        counter = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                data = x.float().to(self.device)
                if data.dim() < 4:
                    data = torch.unsqueeze(data, 1)
                target = y.long().to(self.device)
                if self.use_gdl:
                    labels = torch.argmax(target, dim=1)
                else:
                    labels = target

                pred = self.model(data)
                pred_softmax = F.softmax(pred, dim=1)

                pred_image = torch.argmax(pred_softmax, dim=1)
                dice_result = dice_pytorch(outputs=pred_image, labels=labels, N_class=self.config.num_classes)
                valid_dice += dice_result
                if self.use_gdl:
                    loss = self.gd_loss(pred, target)  # calculate loss
                else:
                    loss = self.wcee_loss(pred, target) + self.sd_loss(pred_softmax, target)
                valid_loss += loss.item()
                counter += 1

        assert data is not None, 'data is None. Please check if your dataloader works properly'
        valid_loss /= counter
        valid_dice /= counter
        return valid_loss, valid_dice


class CrossValTrainer(object):

    def __init__(self, config, load_pt_weights=True, use_imagenet=False, use_gdl=True):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(os.path.join(self.config.save_dir, 'tensorboard'))
        self.use_gdl = use_gdl
        self.load_pt_weights = load_pt_weights
        self.use_imagenet = use_imagenet
        self.preprocess = config.preprocess
        self.hsv_factors = config.hsv_factors

        self.input_shape = self.config.input_shape

        # Load all image paths
        self.all_img_paths, self.all_tar_paths = load_img_and_target_paths(self.config.base_dir, val_split=0.0)
        random.Random(1).shuffle(self.all_img_paths)
        random.Random(1).shuffle(self.all_tar_paths)
        self.n_val_samples = int(len(self.all_img_paths) * 1 / self.config.num_folds)

        # Define transformations
        transform_hot = ComposeDouble([
            FunctionWrapperDouble(normalize_01),
            FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
            FunctionWrapperDouble(one_hot_target, input=False, target=True, nclasses=config.num_classes)
        ])
        transform_ind = ComposeDouble([
            FunctionWrapperDouble(normalize_01),
            FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0)
        ])

        if use_gdl:
            self.transform = transform_hot
        else:
            self.transform = transform_ind

        # Loss functions
        # Calculate class weights
        labels = np.array(range(config.num_classes))

        label_weights = class_weights(self.all_tar_paths[:100], labels)
        label_weights = torch.FloatTensor(label_weights).cuda()
        self.wcee_loss = nn.CrossEntropyLoss(weight=label_weights)
        self.gd_loss = GeneralizedDiceLoss(labels=labels)
        self.sd_loss = SoftDiceLoss(batch_dice=True, do_bg=False)

        self.optimizer = None
        self.scheduler = None

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _load_unet(self):
        # Set up Unet model
        if self.use_imagenet:
            model = smp.Unet(
                encoder_name='resnet50',
                encoder_depth=5,
                encoder_weights='imagenet',
                classes=self.config.num_classes,
            )
            model.to(self.device)
            count_parameters(model)
        else:
            model = Unet(self.config)
            model.to(self.device)
            count_parameters(model=model)

            if self.load_pt_weights:
                # Load pre-trained weights
                state_dict = torch.load(self.config.pretrained_model_path)
                # If strict=True then keys of state_dict must match keys of model
                model.load_state_dict(state_dict, strict=False)

        return model

    def _run_train(self, model, train_loader, val_loader, fold):
        inp = None
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        lr_list = []

        for epoch_counter in range(self.config.epochs):
            print("=====Training Epoch: %d =====" % epoch_counter)
            model.train()  # train mode
            train_losses = []  # accumulate the losses here

            for i, (x, y) in enumerate(train_loader):
                inp = x.float().to(self.device)
                if inp.dim() < 4:
                    inp = torch.unsqueeze(inp, 1) # if grayscale
                tar = y.long().to(self.device)  # send to device (GPU or CPU)
                self.optimizer.zero_grad()  # zerograd the parameters
                pred = model(inp)  # one forward pass
                # We calculate a softmax, because our SoftDiceLoss expects that as an input.
                # The CE-Loss does the softmax internally.
                pred_softmax = F.softmax(pred, dim=1)
                if self.use_gdl:
                    loss = self.gd_loss(pred, tar)  # calculate loss
                else:
                    loss = self.wcee_loss(pred, tar) + self.sd_loss(pred_softmax, tar)
                #

                if n_iter % self.config.log_every_n_steps == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print("Train:[{0}][{1}][{2}] loss: {loss:.4f}".format(epoch_counter, i, len(train_loader),
                                                                          loss=loss.item()))

                loss.backward()  # one backward pass
                self.optimizer.step()  # update the parameters
                n_iter += 1

            print("===== Validation =====")
            # validate the model if requested
            if epoch_counter % self.config.eval_every_n_epochs == 0:
                valid_loss, valid_dice = self._validate(model, val_loader)
                print("Val:[{0}] dice:{dice:.4f} loss: {loss:.4f}".format(epoch_counter, dice=valid_dice,
                                                                          loss=valid_loss))
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    print('____New best model____')
                    torch.save(model.state_dict(), os.path.join(self.config.save_dir,
                                                                'unet_fold_{}_best_{}_model.pth'.format(
                                                                    fold, self.config.batch_size)))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                self.writer.add_scalar('valid_dice_score', valid_dice, global_step=valid_n_iter)
                self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], global_step=valid_n_iter)
                lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(valid_loss)
                if lr != self.optimizer.param_groups[0]['lr']:
                    lr_list.append([epoch_counter, self.optimizer.param_groups[0]['lr']])
                valid_n_iter += 1

            if epoch_counter + 1 == self.config.epochs:
                torch.save(model.state_dict(), os.path.join(self.config.save_dir,
                                                            'unet_fold_{}_last_{}_model.pth'.format(
                                                                fold, self.config.batch_size)))

        print("Learning rate changes:")
        print(lr_list)
        assert inp is not None, 'data is None. Please check if your dataloader works properly'

    def _validate(self, model, val_loader):
        model.eval()

        data = None
        valid_loss = 0.0
        valid_dice = 0.0
        counter = 0

        with torch.no_grad():
            for x, y in val_loader:
                data = x.float().to(self.device)
                if data.dim() < 4:
                    data = torch.unsqueeze(data, 1) #if grayscale
                target = y.long().to(self.device)
                if self.use_gdl:
                    labels = torch.argmax(target, dim=1)
                else:
                    labels = target

                pred = model(data)
                pred_softmax = F.softmax(pred, dim=1)

                pred_image = torch.argmax(pred_softmax, dim=1)
                dice_result = dice_pytorch(outputs=pred_image, labels=labels, N_class=self.config.num_classes)
                valid_dice += dice_result
                if self.use_gdl:
                    loss = self.gd_loss(pred, target)  # calculate loss
                else:
                    loss = self.wcee_loss(pred, target) + self.sd_loss(pred_softmax, target)
                valid_loss += loss.item()
                counter += 1

        assert data is not None, 'data is None. Please check if your dataloader works properly'
        valid_loss /= counter
        valid_dice /= counter
        return valid_loss, valid_dice

    # Train model
    def train(self):
        for f in range(self.config.num_folds):
            model = self._load_unet()
            # Optimizer
            if self.config.optimizer == 'adam':
                self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate,
                                                  weight_decay=self.config.weight_decay)
            elif self.config.optimizer == 'sgd':
                self.optimizer = torch.optim.SGD(model.parameters(), lr=self.config.learning_rate,
                                                 weight_decay=self.config.weight_decay,
                                                 momentum=self.config.sgd_momentum)
            else:
                raise NotImplementedError('Invalid optimizer')

            # Lr scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=10,
                                                                        verbose=True, factor=0.1, threshold=1e-5)

            start = f * self.n_val_samples
            end = (f + 1) * self.n_val_samples
            val_img_paths = self.all_img_paths[start:end]
            val_tar_paths = self.all_tar_paths[start:end]
            tr_img_paths = self.all_img_paths[:start] + self.all_img_paths[end:]
            tr_tar_paths = self.all_tar_paths[:start] + self.all_tar_paths[end:]

            th = 245 #215+5*f
            #self.hsv_factors[2] = 1 + 0.1*f #1 + 0.2*(f+1) #0.08*(f+1)
            tr_dataset = ProstateImageDataset2(image_paths=tr_img_paths, target_paths=tr_tar_paths,
                                               transform=self.transform, preprocess=self.preprocess,
                                               hsv_factors=self.hsv_factors)
            val_dataset = ProstateImageDataset2(image_paths=val_img_paths, target_paths=val_tar_paths,
                                                transform=self.transform, preprocess=self.preprocess,
                                                hsv_factors=self.hsv_factors)

            train_loader = DataLoader(tr_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
            print('=====Starting training, Fold: {}======'.format(f + 1))
            self._run_train(model, train_loader, val_loader, fold=f)
