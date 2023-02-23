import torch
from torch import nn
import numpy as np


@torch.jit.script
def autocrop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):
    """
    Center-crops the encoder_layer to the size of the decoder_layer,
    so that merging (concatenation) between levels/blocks is possible.
    This is only necessary for input sizes != 2**n for 'same' padding and always required for 'valid' padding.
    """
    if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
        ds = encoder_layer.shape[2:]
        es = decoder_layer.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        if encoder_layer.dim() == 4:  # 2D
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2)
                            ]
        elif encoder_layer.dim() == 5:  # 3D
            assert ds[2] >= es[2]
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2),
                            ((ds[2] - es[2]) // 2):((ds[2] + es[2]) // 2),
                            ]
    return encoder_layer, decoder_layer


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, layer_1, layer_2):
        x = torch.cat((layer_1, layer_2), 1)
        return x


class EncodingBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation='relu',
                 pooling: bool = True,
                 use_separable=True,
                 bn_layer=nn.BatchNorm2d,
                 ):
        super(EncodingBlock, self).__init__()
        self._out_channels = out_channels
        self._in_channels = in_channels
        self._pooling = pooling

        self._act1 = nn.ReLU()
        self._act2 = nn.ReLU()

        if use_separable:
            self._conv1 = SeparableConv2d(self._in_channels, self._out_channels, kernel_size=3, bias=True)
            self._conv2 = SeparableConv2d(self._out_channels, self._out_channels, kernel_size=3, bias=True)
        else:
            self._conv1 = nn.Conv2d(self._in_channels, self._out_channels, kernel_size=3, stride=1, padding=1, bias=True)
            self._conv2 = nn.Conv2d(self._out_channels, self._out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self._norm1 = bn_layer(self._out_channels)
        self._norm2 = bn_layer(self._out_channels)

        if self._pooling:
            self._pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        y = self._conv1(x)
        y = self._act1(y)
        y = self._norm1(y)
        y = self._conv2(y)
        y = self._act2(y)
        y = self._norm2(y)

        residual = y  # save residual before pooling
        if self._pooling:
            y = self._pool(y)
        return y, residual


class DecodingBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation='relu',
                 bn_layer=nn.BatchNorm2d,
                 ):
        super(DecodingBlock, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels

        self._act0 = nn.ReLU()
        self._act1 = nn.ReLU()
        self._act2 = nn.ReLU()

        self._upconv = nn.ConvTranspose2d(self._in_channels, self._out_channels, kernel_size=2, stride=2)
        self._conv1 = nn.ConvTranspose2d(2 * self._out_channels, self._out_channels, kernel_size=3, padding=1, bias=True)
        self._conv2 = nn.ConvTranspose2d(self._out_channels, self._out_channels, kernel_size=3, padding=1, bias=True)

        self._norm0 = bn_layer(self._out_channels)
        self._norm1 = bn_layer(self._out_channels)
        self._norm2 = bn_layer(self._out_channels)

        self._concat = Concatenate()

    def forward(self, x, residual):
        up = self._upconv(x)  # up-convolution/up-sampling
        cropped_encoder_layer, dec_layer = autocrop(residual, up)  # cropping
        up = self._act0(up)  # activation 0
        up = self._norm0(up)  # normalization 0

        merged_layer = self._concat(up, cropped_encoder_layer)  # concatenation

        y = self._conv1(merged_layer)
        y = self._act1(y)
        y = self._norm1(y)

        y = self._conv2(y)
        y = self._act2(y)
        y = self._norm2(y)

        return y


class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
        self._input_dim = config.input_dim
        self._n_classes = config.num_classes
        self._depth = config.depth
        self._start_filters = config.filters

        self._encoder = []
        self._decoder = []

        # create encoder path
        num_filters_out = self._start_filters
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
        num_filters_in = num_filters_out
        num_filters_out *= 2
        # print(str(num_filters_in) + ' -> ' + str(num_filters_out))
        self._encoder.append(EncodingBlock(in_channels=num_filters_in, out_channels=num_filters_out, pooling=False, use_separable=False))

        # create decoder path
        for i in range(self._depth):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2

            self._decoder.append(DecodingBlock(in_channels=num_filters_in,
                               out_channels=num_filters_out,
                               ))

        self._final_layer = nn.Conv2d(num_filters_out, self._n_classes, kernel_size=1, padding='same')

        # add the list of modules to current module
        self._encoder = nn.ModuleList(self._encoder)
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
        for module in self.modules():
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

        x = self._final_layer(x)

        return x

    def __repr__(self):
        attributes = {attr_key: self.__dict__[attr_key] for attr_key in self.__dict__.keys() if
                      '_' not in attr_key[0] and 'training' not in attr_key}
        d = {self.__class__.__name__: attributes}
        return f'{d}'




