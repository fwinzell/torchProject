import torch
from torch import nn
import torch.nn.functional as F

from unet.unet_model import EncodingBlock, DecodingBlock


class GenUnet(nn.Module):
    def __init__(self,
                 config):
        super(GenUnet, self).__init__()

        self.input_dim = config.input_shape
        self.depth = config.depth
        self.start_filters = config.start_filters
        self.out_channels = config.out_channels

        self.encoder = []
        self.decoder = []

        # create encoder path
        num_filters_out = self.start_filters
        for i in range(self.depth):
            if i == 0:
                num_filters_in = self.input_dim[0]
            else:
                num_filters_in = num_filters_out
                num_filters_out *= 2

            # print(str(num_filters_in) + ' -> ' + str(num_filters_out))
            self.encoder.append(EncodingBlock(in_channels=num_filters_in,
                                              out_channels=num_filters_out,
                                              pooling=True,
                                              use_separable=False))
        num_filters_in = num_filters_out
        num_filters_out *= 2
        self.encoder.append(
            EncodingBlock(in_channels=num_filters_in, out_channels=num_filters_out, pooling=False, use_separable=False))

        self.encoder = nn.ModuleList(self.encoder)

        # create decoder path
        for i in range(self.depth):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2

            self.decoder.append(DecodingBlock(in_channels=num_filters_in,
                                              out_channels=num_filters_out
                                              ))

        self.final_layer = nn.Conv2d(num_filters_out, self.out_channels, kernel_size=1, padding='same')

        # self._pt_decoder = nn.ModuleList(self._decoder_layers[:self.n_pt_dec_blocks])
        self.decoder = nn.ModuleList(self.decoder)

    def forward(self, x: torch.tensor):
        # Encoder pathway
        encoder_output = []
        for module in self.encoder:
            x, residual = module(x)
            encoder_output.append(residual)

        # Decoder pathway
        for i, module in enumerate(self.decoder):
            residual = encoder_output[-(i + 2)]
            x = module(x, residual)

        x = self.final_layer(x)

        return x
