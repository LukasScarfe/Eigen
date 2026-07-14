"""
UNet Architecture (PyTorch)

This is a PyTorch re-implementation of the TensorFlow/Keras U-Net defined in
UNetArchitecture.py. It preserves the same block structure (Encoder/Decoder
blocks with GroupNorm + ReLU, optional Dropout, average-pool at the bottleneck,
max-pool between encoder stages, up-sampling + skip concatenation in the
decoder). The output has 2 channels representing intensity (scaled to [0, 1])
and phase (scaled to [0, 2*pi]).

Notes on translating Keras -> PyTorch:
  - Keras infers input channel counts automatically; PyTorch Conv2d/ConvTranspose2d
    require explicit in_channels, so every block below tracks and returns its
    output channel count so the UNet class can wire blocks together correctly.
  - Keras `GroupNormalization()` defaults to 32 groups (or fewer if the channel
    count doesn't divide evenly). We approximate this by picking the largest
    valid group count <= 32 that evenly divides the number of channels.
  - Keras `padding='same'` with stride 1 is replicated with `padding=size // 2`
    (valid for odd kernel sizes, which is what's used here: 1x1 and 3x3).
"""

import math

import torch
import torch.nn as nn


def _num_groups(channels: int, max_groups: int = 32) -> int:
    """Pick a GroupNorm group count that evenly divides `channels`."""
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


class EncoderBlock(nn.Module):
    """
    Encoding block: (optional AvgPool) -> [Conv2d -> GroupNorm -> (Dropout) -> ReLU] x layers

    in_channels   - number of input channels
    filters       - number of output channels/filters
    size          - convolution kernel size
    layers        - number of conv layers in this block
    middle        - whether this is the bottleneck block (adds an AvgPool first)
    avgpoolsize   - kernel size of the average pooling layer (only if middle=True)
    useDropOut    - whether to apply Dropout after GroupNorm
    dropRate      - dropout probability
    """

    def __init__(self, in_channels, filters, size, layers, middle=False,
                 avgpoolsize=2, useDropOut=False, dropRate=0.1):
        super().__init__()
        self.out_channels = filters

        blocks = []
        if middle:
            blocks.append(nn.AvgPool2d(kernel_size=avgpoolsize))

        current_channels = in_channels
        for _ in range(layers):
            blocks.append(nn.Conv2d(current_channels, filters, kernel_size=size,
                                     padding=size // 2))
            blocks.append(nn.GroupNorm(_num_groups(filters), filters))
            if useDropOut:
                blocks.append(nn.Dropout(dropRate))
            blocks.append(nn.ReLU())
            current_channels = filters

        self.conv = nn.Sequential(*blocks)

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """
    Decoding block: UpSample -> [ConvTranspose2d -> GroupNorm -> (Dropout) -> ReLU] x layers

    in_channels    - number of input channels
    filters        - number of output channels/filters
    size           - convolution kernel size
    layers         - number of conv layers in this block
    upsamplesize   - upsampling scale factor
    useDropOut     - whether to apply Dropout after GroupNorm
    dropRate       - dropout probability
    """

    def __init__(self, in_channels, filters, size, layers, upsamplesize=2,
                 useDropOut=True, dropRate=0.1):
        super().__init__()
        self.out_channels = filters

        blocks = [nn.Upsample(scale_factor=upsamplesize, mode="nearest")]

        current_channels = in_channels
        for _ in range(layers):
            blocks.append(nn.ConvTranspose2d(current_channels, filters, kernel_size=size,
                                              padding=size // 2))
            blocks.append(nn.GroupNorm(_num_groups(filters), filters))
            if useDropOut:
                blocks.append(nn.Dropout(dropRate))
            blocks.append(nn.ReLU())
            current_channels = filters

        self.conv = nn.Sequential(*blocks)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture built from EncoderBlock/DecoderBlock stacks with skip
    connections, matching the structure of the original Keras `uNet` function.

    num_pixel   - image resolution (H == W == num_pixel), kept for parity/reference
    nnType      - 0 for a 6-stage (64x64-style) network, 1 for a 7-stage (128x128-style) network
    kernelSize  - convolution kernel size
    dropRate    - dropout rate used in the deeper decoder blocks
    layers      - number of conv layers per encoder/decoder block
    sixMeasure  - whether the input has 6 channels (True) or 5 channels (False)
    """

    def __init__(self, num_pixel, nnType, kernelSize=3, dropRate=0.1, layers=1, in_channels=6):
        super().__init__()
        self.num_pixel = num_pixel

        if nnType == 0:  # 64 x 64
            enc_filters = [32, 64, 128]
            middle_filters = 256
            dec_filters = [128, 64, 32]
            dec_dropout = [True, False, False]

        elif nnType == 1:  # 128 x 128
            enc_filters = [32, 64, 128, 256, 512, 1024, 2048]
            middle_filters = 4096
            dec_filters = [2048, 1024, 512, 256, 128, 64, 32]
            dec_dropout = [True, True, True, False, False, False, False]
        else:
            raise ValueError(f"Unsupported nnType: {nnType}")

        # ---- Encoder stack ----
        self.down_stack = nn.ModuleList()
        ch = in_channels
        for f in enc_filters:
            self.down_stack.append(EncoderBlock(ch, f, kernelSize, layers))
            ch = f
        self.pool = nn.MaxPool2d(kernel_size=2)

        # ---- Bottleneck ----
        self.middle = EncoderBlock(ch, middle_filters, kernelSize, layers, middle=True)
        ch = middle_filters

        # ---- Decoder stack (skip channel counts are enc_filters reversed) ----
        skip_channels = list(reversed(enc_filters))
        self.up_stack = nn.ModuleList()
        for f, skip_ch, use_drop in zip(dec_filters, skip_channels, dec_dropout):
            self.up_stack.append(
                DecoderBlock(ch, f, kernelSize, layers, useDropOut=use_drop, dropRate=dropRate)
            )
            ch = f + skip_ch  # after concatenation with the skip connection

        # ---- Output projection ----
        # Output has 2 channels: intensity (scaled to [0, 1]) and phase (scaled to [0, 2*pi])

        self.out_conv = nn.ConvTranspose2d(ch, 2, kernel_size=1, stride=1, padding=0)
        self.register_buffer(
            "out_scale", torch.tensor([1.0, 1.0]).view(1, 2, 1, 1)
        )

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down_stack):
            x = block(x)
            skips.append(x)
            if i < len(self.down_stack) - 1:
                x = self.pool(x)

        x = self.middle(x)

        for up, skip in zip(self.up_stack, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)

        x = self.out_conv(x)
        x = torch.sigmoid(x) * self.out_scale
        return x


if __name__ == "__main__":

    model = UNet(num_pixel=64, nnType=0, kernelSize=3, dropRate=0.1, layers=1, sixMeasure=False)
    dummy = torch.randn(2, 5, 64, 64)
    out = model(dummy)
    print("Output shape:", out.shape)  # expected: (2, 2, 64, 64)
