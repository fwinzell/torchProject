import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


class CrossImageContrastiveLoss(nn.Module):
    def __init__(self, labels, tau, device):
        super(CrossImageContrastiveLoss, self).__init__()
        self.labels = labels
        self.tau = tau
        self.device = device
        self.epsilon = 1e-4

    def forward(self, features_i, features_ii, features_jj, i, ii, jj):
        d = features_i.shape[0]
        N = features_i.shape[1] * features_i.shape[2]
        features_i = torch.reshape(features_i, (d, N))
        features_jj = torch.reshape(features_jj, (d, N))
        features_ii = torch.reshape(features_ii, (d, N))

        denom = torch.zeros(len(self.labels), device=self.device)
        for label in self.labels:
            denom[label] = torch.count_nonzero(ii == label) + torch.count_nonzero(jj == label)

        values = torch.zeros(N, device=self.device)
        for p in range(0, N):
            (y, x) = np.unravel_index(p, i.shape)
            label = i[y, x]
            sim_1 = torch.exp(torch.matmul(features_i[:, p], features_ii) / self.tau)
            diff = (jj == label).flatten() * torch.exp(torch.matmul(features_i[:, p], features_jj) / self.tau)
            term_1 = torch.sum(
                (ii == label).flatten()/(denom[label] + self.epsilon) * torch.log(sim_1[p]/(torch.sum(sim_1) + torch.sum(diff) + self.epsilon)))

            sim_2 = torch.exp(torch.matmul(features_i[:, p], features_jj) / self.tau)
            term_2 = torch.sum(
                (ii == label).flatten()/(denom[label] + self.epsilon) * torch.log(sim_2[p]/(torch.sum(sim_1) + torch.sum(diff) + self.epsilon)))
            values[p] = term_1 + term_2

        loss_value = -1 / N * torch.sum(values)
        return loss_value



 #loss_value = torch.jit.trace(self._calculate_loss, (self.labels, N, tau, features_i, features_ii, features_jj, i, ii, jj))
"""@torch.jit.script
    def _calculate_loss(self, labels, n, tau, features_i, features_ii, features_jj, i, ii, jj):
        denom = torch.zeros(len(labels))
        for label in labels:
            denom[label] = torch.count_nonzero(ii == label) + torch.count_nonzero(jj == label)

        values = torch.zeros(n)
        for p in torch.range(0, n):
            (y, x) = np.unravel_index(p, i.shape)
            label = i[y, x]
            sim_1 = torch.exp(torch.matmul(features_i[:, p], features_ii) / tau)
            diff = (jj == label).flatten() * torch.exp(torch.matmul(features_i[:, p], features_jj) / tau)
            term_1 = torch.sum(
                (ii == label).flatten() / denom[label] * torch.log(sim_1[p] / (torch.sum(sim_1) + torch.sum(diff))))

            sim_2 = torch.exp(torch.matmul(features_i[:, p], features_jj) / tau)
            term_2 = torch.sum(
                (ii == label).flatten() / denom[label] * torch.log(sim_2[p] / (torch.sum(sim_1) + torch.sum(diff))))
            values[p] = term_1 + term_2

        loss_value = -1 / n * torch.sum(values)
        return loss_value"""