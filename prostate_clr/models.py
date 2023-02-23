import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from unet.unet_model import EncodingBlock, DecodingBlock
from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


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

        self._input_dim = config.input_shape
        self._depth = config.depth
        self._start_filters = config.start_filters

        self._encoder_layers = []

        # create encoder path
        num_filters_out = self._start_filters
        for i in range(self._depth):
            if i == 0:
                num_filters_in = self._input_dim[0]
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


class Unet(nn.Module):
    def __init__(self,
                 config,
                 load_pt_weights=True):
        super(Unet, self).__init__()

        self.encoder = Encoder(config)
        self.n_classes = config.num_classes
        self.depth = config.depth
        self.n_pt_dec_blocks = config.no_of_pt_decoder_blocks

        self._decoder_layers = []
        num_filters_out = config.start_filters*2**config.depth
        # create decoder path
        for i in range(self.depth):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2

            self._decoder_layers.append(DecodingBlock(in_channels=num_filters_in,
                                                      out_channels=num_filters_out
                                                      ))

        self.final_layer = nn.Conv2d(num_filters_out, self.n_classes, kernel_size=1, padding='same')

        #self._pt_decoder = nn.ModuleList(self._decoder_layers[:self.n_pt_dec_blocks])
        self.decoder = nn.ModuleList(self._decoder_layers)

        #if load_pt_weights:
            # Load pre-trained weights
        #    state_dict = torch.load(config.pretrained_model_path)
            # If strict=True then keys of state_dict must match keys of model
        #    self._encoder.load_state_dict(state_dict, strict=False)
            # Freeze
        #    self._encoder.requires_grad_(False)
            # Pretrained decoder
        #    if self.n_pt_dec_blocks != 0:
        #        self._pt_decoder.load_state_dict(state_dict, strict=False)
        #        self._pt_decoder.requires_grad_(False)


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
        for module in self.decoder:
            self.weight_init(module, method_weights, **kwargs_weights)  # initialize weights
            self.bias_init(module, method_bias, **kwargs_bias)  # initialize bias

    def forward(self, x: torch.tensor):
        # Encoder pathway
        x, encoder_output = self.encoder(x)

        # Decoder pathway
        for i, module in enumerate(self.decoder):
            residual = encoder_output[-(i + 2)]
            x = module(x, residual)

        #for j, module in enumerate(self._decoder):
        #    residual = encoder_output[-(self.n_pt_dec_blocks + j + 2)]
        #    x = module(x, residual)

        x = self.final_layer(x)

        return x

    def __repr__(self):
        attributes = {attr_key: self.__dict__[attr_key] for attr_key in self.__dict__.keys() if
                      '_' not in attr_key[0] and 'training' not in attr_key}
        d = {self.__class__.__name__: attributes}
        return f'{d}'


class MLP(nn.Module):
    def __init__(self, input_channels=512, num_class=128):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.f1 = nn.Linear(input_channels, input_channels)
        self.f2 = nn.Linear(input_channels, num_class)

    def forward(self, x):
        x = self.gap(x)
        y = self.f1(x.squeeze())
        y = self.f2(y)

        return y


class LocalUnet(nn.Module):
    def __init__(self,
                 config,
                 load_pt_encoder=True):
        super(LocalUnet, self).__init__()

        self.encoder = Encoder(config)
        #if load_pt_encoder:
        #    state_dict = torch.load(config.pretrained_model_path)
        #    self._encoder.load_state_dict(state_dict, strict=True)

        if config.no_of_decoder_blocks > config.depth:
            print('Cannot have more decoding blocks than depth, setting to max')
            config.no_of_decoder_blocks = config.depth

        self._decoder_layers = []
        num_filters_out = config.start_filters * 2**config.depth
        # create decoder path
        for i in range(config.no_of_decoder_blocks):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2

            self._decoder_layers.append(DecodingBlock(in_channels=num_filters_in,
                                                      out_channels=num_filters_out
                                                      ))

        self.decoder = nn.ModuleList(self._decoder_layers)

    def forward(self, x: torch.tensor):
        # Encoder pathway
        x, encoder_output = self.encoder(x)

        # Decoder pathway
        for i, module in enumerate(self.decoder):
            residual = encoder_output[-(i + 2)]
            x = module(x, residual)

        return x


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.g1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.g2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.g1(x)
        y = self.relu(y)
        y = self.bn(y)
        y = self.g2(y)

        return y


class UnetInfer(nn.Module):
    def __init__(self,
                 config):
        super(UnetInfer, self).__init__()

        self._encoder = Encoder(config)
        self.n_classes = config.num_classes
        self.depth = config.depth

        self._decoder_layers = []
        num_filters_out = config.start_filters*2**config.depth
        # create decoder path
        for i in range(self.depth):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2

            self._decoder_layers.append(DecodingBlock(in_channels=num_filters_in,
                                                      out_channels=num_filters_out
                                                      ))

        self._final_layer = nn.Conv2d(num_filters_out, self.n_classes, kernel_size=1, padding='same')
        self._decoder = nn.ModuleList(self._decoder_layers)

        # Load pre-trained weights
        state_dict = torch.load(config.model_path)
        # If strict=True then keys of state_dict must match keys of model
        self._encoder.load_state_dict(state_dict, strict=False)
        self._decoder.load_state_dict(state_dict, strict=False)
        self._final_layer.load_state_dict(state_dict, strict=False)

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