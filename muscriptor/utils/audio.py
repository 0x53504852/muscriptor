"""Audio loading and resampling utilities. WAV is handled by the stdlib;
other formats fall back to `soundfile` (optional dependency)."""

import math
import wave
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly


def _read_wav_file(source) -> tuple[torch.Tensor, int]:
    """Load a PCM WAV file using the stdlib `wave` module.

    `source` may be a filesystem path or a binary file-like object.

    Returns:
        (wav, sr) where wav has shape [C, T] and is float32 in [-1, 1].
    """
    if hasattr(source, "read"):
        opened = wave.open(source, "rb")
    else:
        opened = wave.open(str(source), "rb")
    with opened as wf:
        n_channels = wf.getnchannels()
        sr = wf.getframerate()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        as_int32 = (
            bytes_[:, 0].astype(np.int32)
            | (bytes_[:, 1].astype(np.int32) << 8)
            | (bytes_[:, 2].astype(np.int32) << 16)
        )
        as_int32 = np.where(as_int32 >= (1 << 23), as_int32 - (1 << 24), as_int32)
        data = as_int32.astype(np.float32) / float(1 << 23)
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    data = data.reshape(-1, n_channels)
    return torch.from_numpy(np.ascontiguousarray(data.T)), sr


def _read_non_wav_file(path: str) -> tuple[torch.Tensor, int]:
    """Load a non-WAV audio file using `soundfile` (optional dependency).

    Returns:
        (wav, sr) where wav has shape [C, T] and is float32 in [-1, 1].
    """
    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "soundfile is required to read non-WAV audio files. "
            "Install with: `pip install soundfile` or `uvx --with soundfile`"
        ) from e

    data, sample_rate = sf.read(str(path), dtype="float32")
    if data.ndim == 1:
        data = data[:, None]
    wav = torch.from_numpy(np.ascontiguousarray(data.T))
    return wav, sample_rate


def resample(
    waveform: torch.Tensor,
    orig_freq: int,
    new_freq: int,
) -> torch.Tensor:
    """Polyphase resampler via scipy.signal.resample_poly. Operates along the last dim."""
    if orig_freq == new_freq:
        return waveform
    gcd = math.gcd(int(orig_freq), int(new_freq))
    up = int(new_freq) // gcd
    down = int(orig_freq) // gcd
    device = waveform.device
    data = waveform.detach().cpu().numpy()
    out = resample_poly(data, up, down, axis=-1).astype(np.float32, copy=False)
    return torch.from_numpy(np.ascontiguousarray(out)).to(device)


def load_audio(path: str | Path, target_sr: int = 16000) -> torch.Tensor:
    """Load an audio file and return a mono float32 tensor at target_sr.

    WAV files are read with the stdlib `wave` module. Other formats require
    `soundfile` as an optional dependency.

    Returns:
        Tensor of shape [1, T] at target_sr.
    """
    filepath = Path(path)
    if filepath.suffix.lower() == ".wav":
        wav, sr = _read_wav_file(str(filepath))
    else:
        wav, sr = _read_non_wav_file(str(filepath))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = resample(wav, sr, target_sr)
    return wav


# ---------------------------------------------------------------------------
# MuScriptor V2 reading (the training repository's amt/audio.py timeline)
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = (
    ".wav",
    ".flac",
    ".ogg",
    ".mp3",
    ".m4a",
    ".aac",
    ".opus",
    ".webm",
    ".aif",
    ".aiff",
)


def _ffmpeg_binary() -> str | None:
    import os
    import shutil

    return os.environ.get("MUSCRIPTOR_FFMPEG") or shutil.which("ffmpeg")


def _decode_ffmpeg(path: str) -> tuple[torch.Tensor, int]:
    """Full decode with the ffmpeg CLI at the file's own rate: float32 [channels, samples], sample rate."""
    import json
    import subprocess

    ffmpeg = _ffmpeg_binary()
    if ffmpeg is None:
        raise FileNotFoundError(
            f"cannot read {path}: install ffmpeg (or put it in MUSCRIPTOR_FFMPEG), or convert the file to WAV"
        )
    ffprobe = str(Path(ffmpeg).with_name("ffprobe"))
    info = json.loads(
        subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels",
                "-of",
                "json",
                path,
            ],
            capture_output=True,
            check=True,
        ).stdout
    )["streams"][0]
    sr, channels = int(info["sample_rate"]), int(info["channels"])
    out = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-i",
            path,
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            str(channels),
            "-ar",
            str(sr),
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    wav = np.frombuffer(out, dtype=np.float32).reshape(-1, channels).T
    return torch.from_numpy(wav.copy()), sr


def load_audio_v2(path: str | Path, target_sr: int = 16000) -> torch.Tensor:
    """Mono float32 [1, T] at target_sr, read as the V2 training code reads audio (amt/audio.py load_audio):
    WAV / FLAC / OGG sample-exact (stdlib wave for PCM WAV, else soundfile), other containers (mp3, m4a, opus, ...) by a
    full ffmpeg decode, clamped to [-1, 1] (the timeline the V2 labels follow); channel mean; julius-style sinc resampling
    (muscriptor.utils.resample.resample_frac)."""
    from muscriptor.utils.resample import resample_frac

    filepath = Path(path)
    suffix = filepath.suffix.lower()
    clamp = False
    wav = None
    if suffix == ".wav":
        try:
            wav, sr = _read_wav_file(str(filepath))
        except (wave.Error, ValueError, EOFError):
            wav = None  # float or extensible WAV: soundfile / ffmpeg below
    if wav is None and suffix in (".wav", ".flac", ".ogg"):
        try:
            wav, sr = _read_non_wav_file(str(filepath))
        except ImportError:
            wav = None
    if wav is None:
        wav, sr = _decode_ffmpeg(str(filepath))
        clamp = True
    mono = wav.mean(dim=0)
    if sr != target_sr:
        mono = resample_frac(mono, sr, target_sr)
    if clamp:
        mono = mono.clamp(-1, 1)
    return mono[None]
