"""Audio features built from fixed convolutions, so they run inside the exported ONNX graph.

The browser passes raw audio to the model. Because the spectrogram lives inside the graph, the
training and browser code paths cannot drift apart. A hand-written TypeScript port would need its
own parity test and could still differ subtly. Convolutions also avoid ONNX's STFT operator, which
WebGPU backends may not support.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

SAMPLE_RATE = 32_000
CLIP_SECONDS = 5
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS
N_FFT = 1024
HOP = 320
N_MELS = 128
F_MIN = 150.0  # Below this is mostly wind and handling noise, and little birdsong.
F_MAX = 15_000.0


def hz_to_mel(hz: float) -> float:
    return 2595 * math.log10(1 + hz / 700)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700 * (10 ** (mel / 2595) - 1)


def mel_filterbank() -> np.ndarray:
    """Triangular filters, with shape (mels, frequency bins)."""
    bins = N_FFT // 2 + 1
    points = mel_to_hz(np.linspace(hz_to_mel(F_MIN), hz_to_mel(F_MAX), N_MELS + 2))
    frequencies = np.linspace(0, SAMPLE_RATE / 2, bins)
    bank = np.zeros((N_MELS, bins), dtype=np.float32)
    for index in range(N_MELS):
        low, centre, high = points[index : index + 3]
        rising = (frequencies - low) / (centre - low)
        falling = (high - frequencies) / (high - centre)
        bank[index] = np.maximum(0, np.minimum(rising, falling))
    return bank


class LogMel(nn.Module):
    """Waveform (batch, samples) to log mel image (batch, 3, mels, frames)."""

    def __init__(self) -> None:
        super().__init__()
        window = torch.hann_window(N_FFT, periodic=True)
        times = torch.arange(N_FFT, dtype=torch.float32)
        real, imaginary = [], []
        for k in range(N_FFT // 2 + 1):
            angle = 2 * torch.pi * k * times / N_FFT
            real.append(torch.cos(angle) * window)
            imaginary.append(-torch.sin(angle) * window)
        self.register_buffer("real", torch.stack(real)[:, None, :])
        self.register_buffer("imaginary", torch.stack(imaginary)[:, None, :])
        self.register_buffer("bank", torch.as_tensor(mel_filterbank()))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = waveform[:, None, :]
        real = nn.functional.conv1d(x, self.real, stride=HOP)
        imaginary = nn.functional.conv1d(x, self.imaginary, stride=HOP)
        power = real * real + imaginary * imaginary  # (batch, bins, frames)
        mel = torch.matmul(self.bank, power)
        log_mel = torch.log(mel + 1e-6)
        # Per-clip normalisation makes the model robust to recording volume and gain.
        mean = log_mel.mean(dim=(1, 2), keepdim=True)
        std = log_mel.std(dim=(1, 2), keepdim=True)
        normalised = (log_mel - mean) / (std + 1e-5)
        # Repeating to three channels lets the model reuse ImageNet-pretrained first-layer weights.
        return normalised[:, None, :, :].repeat(1, 3, 1, 1)


def energy_clips(
    waveform: np.ndarray, count: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Choose the loudest non-overlapping windows.

    Labels apply to a whole recording and a bird may call only sometimes. Windows with the most
    energy are the most likely to contain the bird. Short recordings are zero-padded.
    """
    if len(waveform) <= CLIP_SAMPLES:
        padded = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        padded[: len(waveform)] = waveform
        return padded[None]
    step = CLIP_SAMPLES // 2
    starts = np.arange(0, len(waveform) - CLIP_SAMPLES + 1, step)
    squared = np.concatenate([[0.0], np.cumsum(waveform.astype(np.float64) ** 2)])
    energy = squared[starts + CLIP_SAMPLES] - squared[starts]
    chosen: list[int] = []
    for position in np.argsort(-energy):
        start = int(starts[position])
        if all(abs(start - other) >= CLIP_SAMPLES for other in chosen):
            chosen.append(start)
        if len(chosen) == count:
            break
    return np.stack([waveform[s : s + CLIP_SAMPLES] for s in chosen]).astype(np.float32)


def spec_augment(image: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Time and frequency masking stop the model relying on one narrow band or moment."""
    output = image.clone()
    batch, _, mels, frames = output.shape
    for item in range(batch):
        for _ in range(2):
            width = int(torch.randint(1, mels // 6, (1,), generator=generator))
            start = int(torch.randint(0, mels - width, (1,), generator=generator))
            output[item, :, start : start + width, :] = 0
            width = int(torch.randint(1, frames // 8, (1,), generator=generator))
            start = int(torch.randint(0, frames - width, (1,), generator=generator))
            output[item, :, :, start : start + width] = 0
    return output
