import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from scipy.io import wavfile
import librosa
from tqdm import tqdm

import utils
from models.unet import Unet
from models.layers.stft import STFT
from se_dataset import AudioDataset
from torch.utils.data import DataLoader

# NOTE - Use window not supporting stft until pytorch implements istft
#  > https://github.com/pytorch/pytorch/issues/3775
stft_module = STFT().cuda()
stft = stft_module.transform
istft = stft_module.inverse

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None, help="Optional, name of the file in --model_dir containing weights to reload before training")  # 'best' or 'train'
parser.add_argument('--batch_size', default=32, type=int, help='train batch size')
parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')
args = parser.parse_args()

def wSDRLoss(mixed, clean, clean_est, eps=2e-7):
    # Used on signal level(time-domain). Backprop-able istft should be used.
    # Batched audio inputs shape (N x T) required.
    bsum = lambda x: torch.sum(x, dim=1) # Batch preserving sum for convenience.
    def mSDRLoss(orig, est):
        # Modified SDR loss, <x, x`> / (||x|| * ||x`||) : L2 Norm.
        # Original SDR Loss: <x, x`>**2 / <x`, x`> (== ||x`||**2)
        #  > Maximize Correlation while producing minimum energy output.
        correlation = bsum(orig * est)
        energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)
        return -(correlation / (energies + eps))

    noise = mixed - clean
    noise_est = mixed - clean_est

    a = bsum(clean**2) / (bsum(clean**2) + bsum(noise**2) + eps)
    wSDR = a * mSDRLoss(clean, clean_est) + (1 - a) * mSDRLoss(noise, noise_est)
    return torch.mean(wSDR)

# TODO - loader clean speech tempo perturbed as input
# TODO - loader clean speech volume pertubed as input
# TODO - option for (tempo/volume/tempo+volume)
# TODO - loader noise sound as second input
# TODO - loader reverb effect as second input
# TODO - option for (noise/reverb/noise+reverb)

def main():
    json_path = os.path.join(args.model_dir)
    params = utils.Params(json_path)

    net = Unet(params.model).cuda()
    # TODO - check exists
    #checkpoint = torch.load('./final.pth.tar')
    #net.load_state_dict(checkpoint)

    train_dataset = AudioDataset(data_type='train')
    test_dataset = AudioDataset(data_type='val')
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
            collate_fn=train_dataset.collate, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
            collate_fn=test_dataset.collate, shuffle=False, num_workers=4)

    torch.set_printoptions(precision=10, profile="full")

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    # Learning rate scheduler
    scheduler = ExponentialLR(optimizer, 0.95)

    for epoch in range(args.num_epochs):
        train_bar = tqdm(train_data_loader)
        for input in train_bar:
            train_mixed, train_clean, seq_len = map(lambda x: x.cuda(), input)
            mag, phase = stft(train_mixed)
            mag, phase = mag.unsqueeze(dim=1), phase.unsqueeze(dim=1)
            out_mag, out_phase = net(mag, phase)
            out_mag, out_phase = torch.squeeze(out_mag, 1), torch.squeeze(out_phase, 1)
            out_audio = istft(out_mag, out_phase)
            out_audio = torch.squeeze(out_audio, dim=1)
            for i, l in enumerate(seq_len):
                out_audio[i, l:] = 0
            librosa.output.write_wav('mixed.wav', train_mixed[0].cpu().data.numpy()[:seq_len[0].cpu().data.numpy()], 16000)
            librosa.output.write_wav('clean.wav', train_clean[0].cpu().data.numpy()[:seq_len[0].cpu().data.numpy()], 16000)
            librosa.output.write_wav('out.wav', out_audio[0].cpu().data.numpy()[:seq_len[0].cpu().data.numpy()], 16000)
            loss = wSDRLoss(train_mixed, train_clean, out_audio)
            print(epoch, loss)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        scheduler.step()
    torch.save(net.state_dict(), './final.pth.tar')

if __name__ == '__main__':
    main()
