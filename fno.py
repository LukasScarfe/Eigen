#####
# fno.py
# Functions to implement the 2D fourier neural operator
#####

import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt

# Implements the fourier layer of the FNO

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, fno_architecture, device=None, padding_frac=1 / 4):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=2)  -- Re and Im of propagated field
        """
        self.modes1 = fno_architecture["modes1"]
        self.modes2 = fno_architecture["modes2"]
        self.width = fno_architecture["width"]
        self.n_layers = fno_architecture["n_layers"]
        self.retrain_fno = fno_architecture["retrain_fno"]

        torch.manual_seed(self.retrain_fno)
        # self.padding = 9 # pad the domain if input is non-periodic
        self.padding_frac = padding_frac
        self.fc0 = nn.Linear(3, self.width)  # input channel is 3: (z, Re(field), Im(field))

        self.conv_list = nn.ModuleList(
            [nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)])
        self.spectral_list = nn.ModuleList(
            [SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.n_layers)])

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 output channels: Re and Im of propagated field

        self.to(device)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1_padding = int(round(x.shape[-1] * self.padding_frac))
        x2_padding = int(round(x.shape[-2] * self.padding_frac))
        x = F.pad(x, [0, x1_padding, 0, x2_padding])

        for k, (s, c) in enumerate(zip(self.spectral_list, self.conv_list)):

            x1 = s(x)
            x2 = c(x)
            x = x1 + x2
            if k != self.n_layers - 1:
                x = F.gelu(x)
        x = x[..., :-x1_padding, :-x2_padding]

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x  # shape: (batch, H, W, 2)


def make_freq_weights(N, k_cutoff_frac=0.25, alpha=2.0, device="cpu"):
    """
    Build an (N, N//2+1) weight map for use with rfft2.
    k_cutoff_frac: cutoff as a fraction of Nyquist (0.25 = lowest quarter).
    alpha:         rolloff steepness. 0 = flat, 2 = soft, 4+ = sharp.
    """
    fy = torch.fft.fftfreq(N, device=device)            # ky in [-0.5, 0.5)
    fx = torch.fft.rfftfreq(N, device=device)           # kx in [0, 0.5]
    KY, KX = torch.meshgrid(fy, fx, indexing="ij")
    k = torch.sqrt(KX**2 + KY**2)
    W = 1.0 / (1.0 + (k / k_cutoff_frac)**2) ** (alpha / 2)
    return W  # shape (N, N//2+1)


def spectral_relative_l2(pred, tgt, weights, eps=1e-8):
    """
    Frequency-weighted relative L2.
    pred, tgt: (B, H, W, C) real  -- channels-last, as output by FNO2d
    weights:   (H, W//2+1) real, broadcast over batch and channel
    """
    # Permute from channels-last (B, H, W, C) -> channels-first (B, C, H, W) for rfft2
    pred = pred.permute(0, 3, 1, 2)
    tgt  = tgt.permute(0, 3, 1, 2)

    P = torch.fft.rfft2(pred, norm="ortho")
    T = torch.fft.rfft2(tgt,  norm="ortho")

    # Apply weights to the complex spectra

    #print(weights)
    #print(P)
    #print(T)
    

    Pw = P * weights
    Tw = T * weights
    diff = torch.abs(Pw - Tw) ** 2
    norm = torch.abs(Tw) ** 2
    num = diff.sum(dim=(1, 2, 3)).sqrt()
    den = norm.sum(dim=(1, 2, 3)).sqrt() + eps
    return (num / den).mean()


