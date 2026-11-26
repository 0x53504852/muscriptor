"""Band-limited sinc resampling, as `julius.resample_frac` (MIT license, (c) Alexandre Defossez, 2020).

The MuScriptor V2 models were trained (and evaluated) on audio resampled to 16 kHz with julius; this is the same
computation in plain PyTorch, so that the V2 path hears the same 16 kHz signal. The older models keep the scipy
polyphase resampler of `muscriptor.utils.audio.resample`.
"""

import math
from functools import lru_cache

import torch
from torch.nn import functional as F


def _sinc(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x == 0, torch.tensor(1.0, device=x.device, dtype=x.dtype), torch.sin(x) / x
    )


@lru_cache(maxsize=8)
def _kernel(
    old_sr: int, new_sr: int, zeros: int, rolloff: float
) -> tuple[torch.Tensor, int]:
    sr = min(new_sr, old_sr) * rolloff
    width = math.ceil(zeros * old_sr / sr)
    idx = torch.arange(-width, width + old_sr).float()
    kernels = []
    for i in range(new_sr):
        t = (-i / new_sr + idx / old_sr) * sr
        t = t.clamp_(-zeros, zeros)
        t *= math.pi
        window = torch.cos(t / zeros / 2) ** 2
        kernel = _sinc(t) * window
        kernel.div_(kernel.sum())
        kernels.append(kernel)
    return torch.stack(kernels).view(new_sr, 1, -1), width


def resample_frac(
    x: torch.Tensor, old_sr: int, new_sr: int, zeros: int = 24, rolloff: float = 0.945
) -> torch.Tensor:
    """julius.resample_frac(x, old_sr, new_sr): resample along the last dim, output length floor(new_sr * T / old_sr)."""
    gcd = math.gcd(int(old_sr), int(new_sr))
    old_sr, new_sr = int(old_sr) // gcd, int(new_sr) // gcd
    if old_sr == new_sr:
        return x
    kernel, width = _kernel(old_sr, new_sr, zeros, rolloff)
    kernel = kernel.to(x)
    shape = x.shape
    length = x.shape[-1]
    x = x.reshape(-1, length)
    x = F.pad(x[:, None], (width, width + old_sr), mode="replicate")
    ys = F.conv1d(x, kernel, stride=old_sr)
    y = ys.transpose(1, 2).reshape(list(shape[:-1]) + [-1])
    out_len = int(torch.floor(torch.as_tensor(new_sr * length / old_sr)).item())
    return y[..., :out_len]
