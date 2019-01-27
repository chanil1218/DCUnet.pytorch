import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# NOTE: Use Complex Ops for DCUnet when implemented
# Reference:
#  > Progress: https://github.com/pytorch/pytorch/issues/755
#  > Keras version: https://github.com/ChihebTrabelsi/deep_complex_networks
def pad2d_as(x1, x2):
    # Pad x1 to have same size with x2
    # inputs are NCHW
    diffH = x2.size()[2] - x1.size()[2]
    diffW = x2.size()[3] - x1.size()[3]

    return F.pad(x1, (0, diffW, 0, diffH)) # (L,R,T,B)

def padded_cat(x1, x2, dim):
    # NOTE: Use torch.cat with pad instead when merged
    #  > https://github.com/pytorch/pytorch/pull/11494
    x1 = pad2d_as(x1, x2)
    x1 = torch.cat([x1, x2], dim=dim)
    return x1

class Encoder(nn.Module):
    def __init__(self, conv_cfg, leaky_slope):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(*conv_cfg, bias=False),
            nn.BatchNorm2d(conv_cfg[1]),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self, dconv_cfg, leaky_slope):
        super(Decoder, self).__init__()
        self.dconv = nn.Sequential(
            nn.ConvTranspose2d(*dconv_cfg),
            nn.BatchNorm2d(dconv_cfg[1]),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )

    def forward(self, x, skip=None):
        if skip is not None:
            x = padded_cat(x, skip, dim=1)
        x = self.dconv(x)
        return x

class Unet(nn.Module):
    def __init__(self, cfg):
        super(Unet, self).__init__()
        self.encoders = nn.ModuleList()
        for conv_cfg in cfg['encoders']:
            self.encoders.append(Encoder(conv_cfg, cfg['leaky_slope']))

        self.decoders = nn.ModuleList()
        for dconv_cfg in cfg['decoders'][:-1]:
            self.decoders.append(Decoder(dconv_cfg, cfg['leaky_slope']))

        # Last decoder doesn't use BN & LeakyReLU. Ok for bias?
        self.last_decoder = nn.ConvTranspose2d(*cfg['decoders'][-1])

        if cfg['ratio_mask'] == 'BDT':
            # TODO - Is it proper real value version of complex valued masking?
            # TODO - Should decide cRM(phase and magnitude) or RM(only magnitude)
            # TODO - last decoder output will be 2 channel, real and imaginary part
            self.ratio_mask = lambda x: torch.tanh(torch.abs(x))
        elif cfg['ratio_mask'] == 'BDSS':
            self.ratio_mask = torch.sigmoid
        elif cfg['ratio_mask'] == 'UBD':
            self.ratio_mask = lambda x: x

    def forward(self, x):
        input = x
        skips = list()

        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        skip = skips.pop()
        skip = None # First decoder input x is same as skip, drop skip.
        for decoder in self.decoders:
            x = decoder(x, skip)
            skip = skips.pop()

        x = padded_cat(x, skip, dim=1)
        x = self.last_decoder(x)

        x = pad2d_as(x, input)
        x = self.ratio_mask(x) * input

        return x
