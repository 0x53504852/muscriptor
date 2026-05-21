"""Audio loading utility."""

import typing as tp
from pathlib import Path

import torch
import torchaudio


def _load_with_soundfile(path: str) -> tp.Tuple[torch.Tensor, int]:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)  # [C, T]
    return wav, sr


def load_audio(path: tp.Union[str, Path], target_sr: int = 16000) -> torch.Tensor:
    """Load an audio file and return a mono float32 tensor at target_sr.

    Returns:
        Tensor of shape [1, T] at target_sr.
    """
    try:
        wav, sr = torchaudio.load(str(path))
    except Exception:
        wav, sr = _load_with_soundfile(str(path))

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
    return wav
