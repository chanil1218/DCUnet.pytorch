import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models.layers.complexnn as dcnn


# NOTE: Use Complex Ops for DCUnet when implemented
# Reference:
#  > Progress: https://github.com/pytorch/pytorch/issues/755
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
        self.conv = dcnn.ComplexConvWrapper(nn.Conv2d, *conv_cfg, bias=False)
        self.bn = dcnn.ComplexBatchNorm(conv_cfg[1])
        self.act = dcnn.CLeakyReLU(leaky_slope, inplace=True)

    def forward(self, xr, xi):
        xr, xi = self.act(*self.bn(*self.conv(xr, xi)))
        return xr, xi

class Decoder(nn.Module):
    def __init__(self, dconv_cfg, leaky_slope):
        super(Decoder, self).__init__()
        self.dconv = dcnn.ComplexConvWrapper(nn.ConvTranspose2d, *dconv_cfg, bias=False)
        self.bn = dcnn.ComplexBatchNorm(dconv_cfg[1])
        self.act = dcnn.CLeakyReLU(leaky_slope, inplace=True)

    def forward(self, xr, xi, skip=None):
        if skip is not None:
            xr, xi = padded_cat(xr, skip[0], dim=1), padded_cat(xi, skip[1], dim=1)
        xr, xi = self.act(*self.bn(*self.dconv(xr, xi)))
        return xr, xi

class Unet(nn.Module):
    def __init__(self, cfg):
        super(Unet, self).__init__()
        self.encoders = nn.ModuleList()
        for conv_cfg in cfg['encoders']:
            self.encoders.append(Encoder(conv_cfg, cfg['leaky_slope']))

        self.decoders = nn.ModuleList()
        for dconv_cfg in cfg['decoders'][:-1]:
            self.decoders.append(Decoder(dconv_cfg, cfg['leaky_slope']))

        # Last decoder doesn't use BN & LeakyReLU. Use bias.
        self.last_decoder = dcnn.ComplexConvWrapper(nn.ConvTranspose2d,
                *cfg['decoders'][-1], bias=True)

        self.ratio_mask_type = cfg['ratio_mask']

    def get_ratio_mask(self, outr, outi):
        def inner_fn(r, i):
            if self.ratio_mask_type == 'BDSS':
                return torch.sigmoid(outr) * r, torch.sigmoid(outi) * i
            else:
                # Polar cordinate masks
                # x1.4 slower
                mag_mask = torch.sqrt(outr**2 + outi**2)
                # M_phase = O/|O| for O = g(X)
                # Same phase rotate(theta), for phase mask O/|O| and O.
                phase_rotate = torch.atan2(outi, outr)

                if self.ratio_mask_type == 'BDT':
                    mag_mask = torch.tanh(mag_mask)
                # else then UBD(Unbounded)

                mag = mag_mask * torch.sqrt(r**2 + i**2)
                phase = phase_rotate + torch.atan2(i, r)

                # return real, imag
                return mag * torch.cos(phase), mag * torch.sin(phase)

        return inner_fn

    def forward(self, xr, xi):
        input_real, input_imag = xr, xi
        skips = list()

        for encoder in self.encoders:
            xr, xi = encoder(xr, xi)
            skips.append((xr, xi))

        skip = skips.pop()
        skip = None # First decoder input x is same as skip, drop skip.
        for decoder in self.decoders:
            xr, xi = decoder(xr, xi, skip)
            skip = skips.pop()

        xr, xi = padded_cat(xr, skip[0], dim=1), padded_cat(xi, skip[1], dim=1)
        xr, xi = self.last_decoder(xr, xi)

        xr, xi = pad2d_as(xr, input_real), pad2d_as(xi, input_imag)
        ratio_mask_fn = self.get_ratio_mask(xr, xi)
        return ratio_mask_fn(input_real, input_imag)
