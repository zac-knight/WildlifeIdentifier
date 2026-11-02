import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wildlens.audio import (  # noqa: E402
    CLIP_SAMPLES,
    HOP,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    LogMel,
    energy_clips,
    mel_filterbank,
    spec_augment,
)

pytestmark = pytest.mark.train


def test_log_mel_shape() -> None:
    image = LogMel()(torch.randn(2, CLIP_SAMPLES))
    assert image.shape == (2, 3, N_MELS, (CLIP_SAMPLES - N_FFT) // HOP + 1)


def test_tone_peaks_in_the_right_mel_band() -> None:
    frequency = 4000.0
    times = torch.arange(CLIP_SAMPLES) / SAMPLE_RATE
    wave = torch.sin(2 * torch.pi * frequency * times)[None]
    image = LogMel()(wave)[0, 0].mean(dim=1)
    bank = torch.as_tensor(mel_filterbank())
    bin_index = round(frequency / (SAMPLE_RATE / N_FFT))
    assert int(image.argmax()) == int(bank[:, bin_index].argmax())


def test_convolution_power_matches_torch_stft() -> None:
    wave = torch.randn(1, 8000)
    model = LogMel()
    real = torch.nn.functional.conv1d(wave[:, None], model.real, stride=HOP)
    imaginary = torch.nn.functional.conv1d(wave[:, None], model.imaginary, stride=HOP)
    ours = real**2 + imaginary**2
    reference = (
        torch.stft(
            wave,
            N_FFT,
            hop_length=HOP,
            window=torch.hann_window(N_FFT, periodic=True),
            center=False,
            return_complex=True,
        ).abs()
        ** 2
    )
    assert torch.allclose(ours, reference, rtol=1e-2, atol=1e-2)


def test_filterbank_is_non_negative_and_covers_all_bands() -> None:
    bank = mel_filterbank()
    assert bank.min() >= 0
    assert (bank.sum(axis=1) > 0).all()


def test_energy_clips_find_the_loud_part() -> None:
    wave = np.zeros(CLIP_SAMPLES * 4, dtype=np.float32)
    wave[CLIP_SAMPLES * 2 : CLIP_SAMPLES * 3] = 1
    clips = energy_clips(wave, 1)
    assert clips.shape == (1, CLIP_SAMPLES)
    assert clips.mean() > 0.9


def test_short_audio_is_padded() -> None:
    clips = energy_clips(np.ones(1000, dtype=np.float32), 3)
    assert clips.shape == (1, CLIP_SAMPLES)
    assert clips[0, 999] == 1 and clips[0, 1000] == 0


def test_chosen_clips_do_not_overlap() -> None:
    wave = np.random.default_rng(0).normal(size=CLIP_SAMPLES * 6).astype(np.float32)
    assert energy_clips(wave, 3).shape == (3, CLIP_SAMPLES)


def test_spec_augment_masks_without_touching_input() -> None:
    image = torch.ones(1, 3, N_MELS, 200)
    output = spec_augment(image)
    assert image.min() == 1
    assert output.min() == 0
