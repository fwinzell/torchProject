import torch
import numpy as np

def dice_score(outputs: torch.Tensor, labels: torch.Tensor, N_class):
