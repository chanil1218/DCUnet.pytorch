import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.io import wavfile

import utils
from models.unet import Unet
from models.layers.stft import STFT

# NOTE - Use window not supporting stft until pytorch implements istft
#  > https://github.com/pytorch/pytorch/issues/3775
stft_module = STFT()
stft = stft_module.transform
istft = stft_module.inverse

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None, help="Optional, name of the file in --model_dir containing weights to reload before training")  # 'best' or 'train'
args = parser.parse_args()

# NOTE - Used on signal level(time-domain). Backpropagatable istft should be used.
def wSDRLoss(mixed, clean, clean_est):
    def SDRLoss(orig, est):
        # <x, x`> / ||x|| ||x`||
        return torch.mean(orig * est) / (torch.norm(orig, p=1) * torch.norm(est, p=1) + 1e-10)

    noise = mixed - clean
    noise_est = mixed - clean_est

    a = torch.sum(clean**2) / (torch.sum(clean**2) + torch.sum(noise**2))
    return a * SDRLoss(clean, clean_est) + (1 - a) * SDRLoss(noise, noise_est)


# TODO - loader clean speech tempo perturbed as input
# TODO - loader clean speech volume pertubed as input
# TODO - option for (tempo/volume/tempo+volume)
# TODO - loader noise sound as second input
# TODO - loader reverb effect as second input
# TODO - option for (noise/reverb/noise+reverb)

json_path = os.path.join(args.model_dir)
params = utils.Params(json_path)

net = Unet(params.model)

x = torch.randn(*[32, 1, 161, 620])
out = net(x)
print(x, out)

rate, signal = wavfile.read('/home/chanil/4s1d-20140103-1-test.wav')
signal = torch.Tensor(signal).float()
print(signal)
loss = wSDRLoss(signal, signal, signal)
print(loss)
