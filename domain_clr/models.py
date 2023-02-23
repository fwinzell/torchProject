import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from unet.unet_model import EncodingBlock, DecodingBlock


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self._input_dim = config.input_dim
        self._depth = config.depth
        self._start_filters = config.filters

        self._encoder_layers = []

        # create encoder path
        num_filters_out = self._start_filters
        for i in range(self._depth):
            if i == 0:
                num_filters_in = self._input_dim[-1]
            else:
                num_filters_in = num_filters_out
                num_filters_out *= 2

            # print(str(num_filters_in) + ' -> ' + str(num_filters_out))
            self._encoder_layers.append(EncodingBlock(in_channels=num_filters_in,
                                               out_channels=num_filters_out,
                                               pooling=True,
                                               use_separable=False))
        num_filters_in = num_filters_out
        num_filters_out *= 2
        self._encoder_layers.append(
            EncodingBlock(in_channels=num_filters_in, out_channels=num_filters_out, pooling=False, use_separable=False))

        self._encoder_layers = nn.ModuleList(self._encoder_layers)


    def forward(self, x: torch.tensor):
        encoder_output = []

        # Encoder pathway
        for module in self._encoder_layers:
            x, residual = module(x)
            encoder_output.append(residual)

        return x, encoder_output



