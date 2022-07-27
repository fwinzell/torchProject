import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from unet.unet_model import UNet, EncodingBlock, DecodingBlock


class UnetSimCLR(nn.Module):
    def __init__(self,
                 config):
        super(UnetSimCLR, self).__init__()

        self._input_dim = config.input_dim
        self._n_classes = config.num_classes
        self._depth = config.depth
        self._decoding_steps = config.decoding_steps
        self._start_filters = config.filters
        self._interp_dim = 4

        self._encoder = []
        self._decoder = []

        self._layer_dims = []

        # create encoder path
        num_filters_out = self._start_filters
        self._layer_dims.append(self._input_dim[0])
        for i in range(self._depth):
            if i == 0:
                num_filters_in = self._input_dim[-1]
            else:
                num_filters_in = num_filters_out
                num_filters_out *= 2

            # print(str(num_filters_in) + ' -> ' + str(num_filters_out))
            self._encoder.append(EncodingBlock(in_channels=num_filters_in,
                                               out_channels=num_filters_out,
                                               pooling=True,
                                               use_separable=False))
            self._layer_dims.append(self._layer_dims[-1]/2)
        num_filters_in = num_filters_out
        num_filters_out *= 2
        # print(str(num_filters_in) + ' -> ' + str(num_filters_out))
        self._encoder.append(
            EncodingBlock(in_channels=num_filters_in, out_channels=num_filters_out, pooling=False, use_separable=False))

        # create decoder path
        for i in range(self._decoding_steps):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2

            self._decoder.append(DecodingBlock(in_channels=num_filters_in,
                                               out_channels=num_filters_out,
                                               ))
            self._layer_dims.append(self._layer_dims[-1]*2)

        # add the list of modules to current module
        self._encoder = nn.ModuleList(self._encoder)
        self._decoder = nn.ModuleList(self._decoder)
        self._decoder_out_filters = num_filters_out

        dim_mlp = num_filters_out*self._interp_dim**2
        self._projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, dim_mlp))

    def forward(self, x: torch.tensor):
        encoder_output = []

        # Encoder pathway
        for module in self._encoder:
            x, residual = module(x)
            encoder_output.append(residual)

        # Decoder pathway
        for i, module in enumerate(self._decoder):
            residual = encoder_output[-(i + 2)]
            x = module(x, residual)

        # Downsample
        x = F.interpolate(x, size=self._interp_dim)
        x = self._projection_head(x)
        return x

    def get_encoder_decoder(self):
        return self._encoder, self._decoder, self._decoder_out_filters


class FineTunedUnet(nn.Module):
    def __init__(self,
                 pretrained: nn.Module,
                 config):
        super(FineTunedUnet, self).__init__()

        self._encoder, self._decoder, self._out_filters = pretrained.get_encoder_decoder()
        self._n_classes = config.num_classes
        self._depth = config.depth
        self._decoding_steps = config.decoding_steps
        self._start_filters = config.filters

        self._final_layers = []

        num_filters_out = self._out_filters
        for j in range(self._depth - self._decoding_steps):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2

            self._final_layers.append(DecodingBlock(in_channels=num_filters_in,
                                               out_channels=num_filters_out,
                                               ))

        self._conv_layer = nn.Conv2d(num_filters_out, self._n_classes, kernel_size=1, padding='same')

        # add the list of modules to current module
        self._final_layers = nn.ModuleList(self._final_layers)

        # initialize the weights
        self.initialize_parameters()

    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.weight, **kwargs)  # weights

    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.bias, **kwargs)  # bias

    def initialize_parameters(self,
                              method_weights=nn.init.xavier_uniform_,
                              method_bias=nn.init.zeros_,
                              kwargs_weights={},
                              kwargs_bias={}
                              ):
        for module in self._final_layers:
            self.weight_init(module, method_weights, **kwargs_weights)  # initialize weights
            self.bias_init(module, method_bias, **kwargs_bias)  # initialize bias

    def forward(self, x: torch.tensor):
        encoder_output = []

        # Encoder pathway
        for module in self._encoder:
            x, residual = module(x)
            encoder_output.append(residual)

        # Decoder pathway
        for i, module in enumerate(self._decoder):
            residual = encoder_output[-(i + 2)]
            x = module(x, residual)

        i = i + 1
        for j, module in enumerate(self._final_layers):
            residual = encoder_output[-(i + j + 2)]
            x = module(x, residual)

        x = self._conv_layer(x)

        return x

    def __repr__(self):
        attributes = {attr_key: self.__dict__[attr_key] for attr_key in self.__dict__.keys() if
                      '_' not in attr_key[0] and 'training' not in attr_key}
        d = {self.__class__.__name__: attributes}
        return f'{d}'


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


class Decoder(nn.Module):
    def __init__(self,
                 encoder,
                 config):
        super(Decoder, self).__init__()

        self._encoder = encoder
        self._n_classes = config.num_classes
        self._depth = config.depth

        self._decoder = []
        num_filters_out = config.filters*2**config.depth
        # create decoder path
        for i in range(self._depth):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2

            self._decoder.append(DecodingBlock(in_channels=num_filters_in,
                                               out_channels=num_filters_out,
                                               ))

        self._final_layer = nn.Conv2d(num_filters_out, self._n_classes, kernel_size=1, padding='same')

        # add the list of modules to current module
        self._decoder = nn.ModuleList(self._decoder)
        # initialize the weights
        self.initialize_parameters()

    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.weight, **kwargs)  # weights

    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.bias, **kwargs)  # bias

    def initialize_parameters(self,
                              method_weights=nn.init.xavier_uniform_,
                              method_bias=nn.init.zeros_,
                              kwargs_weights={},
                              kwargs_bias={}
                              ):
        for module in self._decoder:
            self.weight_init(module, method_weights, **kwargs_weights)  # initialize weights
            self.bias_init(module, method_bias, **kwargs_bias)  # initialize bias

    def forward(self, x: torch.tensor):
        # Encoder pathway
        x, encoder_output = self._encoder(x)

        # Decoder pathway
        for i, module in enumerate(self._decoder):
            residual = encoder_output[-(i + 2)]
            x = module(x, residual)

        x = self._final_layer(x)

        return x

    def __repr__(self):
        attributes = {attr_key: self.__dict__[attr_key] for attr_key in self.__dict__.keys() if
                      '_' not in attr_key[0] and 'training' not in attr_key}
        d = {self.__class__.__name__: attributes}
        return f'{d}'