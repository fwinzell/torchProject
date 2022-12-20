import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, labels, epsilon=0.00001):
        super(GeneralizedDiceLoss, self).__init__()
        self.labels = labels
        self.epsilon = epsilon

    def forward(self, input, target):
        pred = F.softmax(input, dim=1)
        N = input.shape[0]  # batch size
        K = len(self.labels)  # number of classes
        w = torch.zeros(K).cuda()
        for k, l in enumerate(self.labels):
            w[k] = 1 / (torch.sum(target[:, k, :, :]) ** 2 + self.epsilon)

        # flatten prediction and target arrays
        pred = pred.view(N, K, -1)
        target = target.view(N, K, -1)

        # intersection
        intersection = (pred * target).sum(-1)
        intersection = w * intersection
        # denominator
        denominator = (pred + target).sum(-1)
        denominator = (w * denominator).clamp(min=self.epsilon)

        gdl = 1 - 2 * torch.sum(intersection) / torch.sum(denominator)
        return gdl


class ShitDiceLoss(nn.Module):
    def __init__(self, labels, epsilon=0.001):
        super(GeneralizedDiceLoss, self).__init__()
        self.labels = labels
        self.epsilon = epsilon

    def forward(self, input, target):
        pred = F.softmax(input)
        num = torch.zeros(len(self.labels))
        den = torch.zeros(len(self.labels))
        for k, l in enumerate(self.labels):
            y_true = torch.zeros_like(target)
            y_true[target == l] = 1
            w = 1 / (torch.sum(y_true) ** 2 + self.epsilon)
            num[k] = w * torch.sum(torch.mul(pred, y_true))
            den[k] = w * torch.sum(torch.add(pred, y_true))

        gdl = 1 - 2 * torch.sum(num) / torch.sum(den)
        return gdl

    def _with_numpy(self, input, target):
        target.cpu()
        pred = F.softmax(input)
        num = np.zeros(len(self.labels))
        den = np.zeros(len(self.labels))
        for k, l in enumerate(self.labels):
            y_true = np.zeros_like(target)
            y_true[target == l] = 1
            w = 1 / (np.sum(y_true) ** 2 + self.epsilon)
            num[k] = w * np.sum(pred * y_true)
            den[k] = w * (np.sum(pred + y_true))

        gdl = 1 - 2 * np.sum(num) / np.sum(den)
        return gdl


class MaxDiceLoss(nn.Module):
    def __init__(self, labels, epsilon=0.00001):
        super(MaxDiceLoss, self).__init__()
        self.labels = labels
        self.epsilon = epsilon

    def forward(self, input, target):
        pred = F.softmax(input)
        N = input.shape[0]  # batch size
        K = len(self.labels)  # number of classes
        w = torch.zeros(K).cuda()
        for k, l in enumerate(self.labels):
            w[k] = 1 / (torch.sum(target[:, k, :, :]) ** 2 + self.epsilon)

        # flatten prediction and target arrays
        pred = pred.view(N, K, -1)
        target = target.view(N, K, -1)

        # intersection
        intersection = (pred * target).sum(-1)
        intersection = w * intersection
        # denominator
        denominator = (pred + target).sum(-1)
        denominator = (w * denominator).clamp(min=self.epsilon)

        gdl = 1 - 2 * intersection / denominator
        return torch.max(gdl)


class ShitDiceLoss(nn.Module):
    def __init__(self, labels, epsilon=0.001):
        super(GeneralizedDiceLoss, self).__init__()
        self.labels = labels
        self.epsilon = epsilon

    def forward(self, input, target):
        pred = F.softmax(input)
        num = torch.zeros(len(self.labels))
        den = torch.zeros(len(self.labels))
        for k, l in enumerate(self.labels):
            y_true = torch.zeros_like(target)
            y_true[target == l] = 1
            w = 1 / (torch.sum(y_true) ** 2 + self.epsilon)
            num[k] = w * torch.sum(torch.mul(pred, y_true))
            den[k] = w * torch.sum(torch.add(pred, y_true))

        gdl = 1 - 2 * torch.sum(num) / torch.sum(den)
        return gdl

    def _with_numpy(self, input, target):
        target.cpu()
        pred = F.softmax(input)
        num = np.zeros(len(self.labels))
        den = np.zeros(len(self.labels))
        for k, l in enumerate(self.labels):
            y_true = np.zeros_like(target)
            y_true[target == l] = 1
            w = 1 / (np.sum(y_true) ** 2 + self.epsilon)
            num[k] = w * np.sum(pred * y_true)
            den[k] = w * (np.sum(pred + y_true))

        gdl = 1 - 2 * np.sum(num) / np.sum(den)
        return gdl


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
