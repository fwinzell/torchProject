import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WizLoss(nn.Module):
    def __init__(self,
                 device,
                 img_rep_size,
                 config
                 ):
        super(WizLoss, self).__init__()
        self.device = device

        self.no_of_decoder_blocks = config.no_of_decoder_blocks
        self.no_of_local_regions = config.no_of_local_regions
        self.no_of_neg_local_regions = config.no_of_neg_local_regions
        self.local_reg_size = config.local_reg_size
        self.im_sz = img_rep_size
        self.wgt_en = config.wgt_en

        self.bs = config.batch_size

        self.pos_sample_indexes, self.neg_sample_indexes = self._define_local_regions()

        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.temp_fac = config.temp_fac

    def _define_local_regions(self):
        # Local region size = 3x3

