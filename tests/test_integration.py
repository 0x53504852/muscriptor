"""Integration tests: load real checkpoint and run inference.

These tests require the safetensors weights file to exist.
They are slow (model is large) but run on CPU with short audio clips.

Run with: pytest tests/test_integration.py -v
"""

import tempfile
from pathlib import Path

import pytest
import torch

from muscriptor.tokenizer.notes import Note
from .conftest import SONG_PATH

# Model constants
_SAMPLE_RATE = 16000
_SEGMENT_DURATION = 5  # seconds — must match transcription_model._SEGMENT_DURATION
# One full segment (5 s) is the natural test size; shorter clips are padded internally.
_DURATION_SEC = _SEGMENT_DURATION
# Use exactly two segments for the song clip so we exercise the multi-chunk path.
_SONG_CLIP_SEC = 10


@pytest.fixture(scope="module")
def silence():
    return torch.zeros(1, _SAMPLE_RATE * _DURATION_SEC)


@pytest.fixture(scope="module")
def noise():
    torch.manual_seed(0)
    return torch.randn(1, _SAMPLE_RATE * _DURATION_SEC) * 0.01


@pytest.fixture(scope="module")
def song_clip():
    """First 10 seconds of filling_the_void.wav resampled to 16 kHz mono."""
    if not SONG_PATH.exists():
        pytest.skip(f"Song not found at {SONG_PATH}")
    from muscriptor.utils.audio import load_audio

    wav = load_audio(SONG_PATH, target_sr=_SAMPLE_RATE)
    return wav[:, : _SAMPLE_RATE * _SONG_CLIP_SEC]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def test_model_loads(transcription_model):
    assert transcription_model is not None
    assert transcription_model._model is not None
    assert transcription_model._tokenizer is not None


def test_model_is_eval(transcription_model):
    assert not transcription_model._model.training


def test_model_device_is_cpu(transcription_model):
    p = next(transcription_model._model.parameters())
    assert p.device.type == "cpu"


# ---------------------------------------------------------------------------
# _load_wav preprocessing
# ---------------------------------------------------------------------------


def test_load_wav_1d_tensor(transcription_model):
    wav1d = torch.zeros(_SAMPLE_RATE)
    out = transcription_model._load_wav(wav1d, sample_rate=_SAMPLE_RATE)
    assert out.shape[0] == 1
    assert out.dim() == 2


def test_load_wav_2d_tensor(transcription_model):
    wav2d = torch.zeros(1, _SAMPLE_RATE)
    out = transcription_model._load_wav(wav2d, sample_rate=_SAMPLE_RATE)
    assert out.shape == (1, _SAMPLE_RATE)


def test_load_wav_stereo_to_mono(transcription_model):
    wav_stereo = torch.zeros(2, _SAMPLE_RATE)
    out = transcription_model._load_wav(wav_stereo, sample_rate=_SAMPLE_RATE)
    assert out.shape[0] == 1


def test_load_wav_resamples(transcription_model):
    wav_44k = torch.zeros(1, 44100)
    out = transcription_model._load_wav(wav_44k, sample_rate=44100)
    assert out.shape[-1] == _SAMPLE_RATE  # resampled to 1 second at 16 kHz


# ---------------------------------------------------------------------------
# _build_conditions
# ---------------------------------------------------------------------------


def test_build_conditions_structure(transcription_model):
    wav = torch.zeros(1, _SAMPLE_RATE)
    conditions = transcription_model._build_conditions(wav)
    assert len(conditions) == 1
    cond = conditions[0]
    assert "self_wav" in cond.wav
    assert "instrument_group" in cond.text
    assert "dataset_name" in cond.text


def test_build_conditions_wav_shape(transcription_model):
    wav = torch.zeros(1, _SAMPLE_RATE * _SEGMENT_DURATION)
    conditions = transcription_model._build_conditions(wav)
    wav_cond = conditions[0].wav["self_wav"]
    assert wav_cond.wav.shape == (1, 1, _SAMPLE_RATE * _SEGMENT_DURATION)
    assert wav_cond.length.item() == _SAMPLE_RATE * _SEGMENT_DURATION


# ---------------------------------------------------------------------------
# transcribe — return_notes
# ---------------------------------------------------------------------------


def test_transcribe_returns_list(transcription_model, silence):
    notes = transcription_model.transcribe(
        silence, sample_rate=_SAMPLE_RATE, return_notes=True
    )
    assert isinstance(notes, list)


def test_transcribe_note_types(transcription_model, noise):
    notes = transcription_model.transcribe(
        noise, sample_rate=_SAMPLE_RATE, return_notes=True
    )
    for note in notes:
        assert isinstance(note, Note)


def test_transcribe_note_fields_in_range(transcription_model, noise):
    notes = transcription_model.transcribe(
        noise, sample_rate=_SAMPLE_RATE, return_notes=True
    )
    for note in notes:
        assert 0 <= note.pitch <= 127
        assert 0 <= note.program <= 128  # 128 = drums
        assert note.onset >= 0.0
        assert note.offset >= note.onset


def test_transcribe_returns_none_by_default(transcription_model, silence):
    result = transcription_model.transcribe(silence, sample_rate=_SAMPLE_RATE)
    assert result is None


# ---------------------------------------------------------------------------
# transcribe — MIDI output
# ---------------------------------------------------------------------------


def test_transcribe_saves_midi(transcription_model, silence):
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "out.mid"
        transcription_model.transcribe(
            silence, sample_rate=_SAMPLE_RATE, output_midi=out
        )
        assert out.exists()
        assert out.stat().st_size > 0


def test_transcribe_midi_is_valid(transcription_model, silence):
    from mido import MidiFile

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "out.mid"
        transcription_model.transcribe(
            silence, sample_rate=_SAMPLE_RATE, output_midi=out
        )
        midi = MidiFile(str(out))
        assert len(midi.tracks) > 0


def test_cli_json_output(transcription_model, silence, monkeypatch):
    """End-to-end: `muscriptor transcribe --format json` writes a valid notes list."""
    import json
    import wave

    from typer.testing import CliRunner

    from muscriptor import main as cli_main

    monkeypatch.setattr(
        cli_main.TranscriptionModel,
        "load_model",
        classmethod(lambda cls, **kw: transcription_model),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "in.wav"
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(_SAMPLE_RATE)
            w.writeframes(b"\x00\x00" * (_SAMPLE_RATE * _DURATION_SEC))
        out_path = Path(tmpdir) / "out.json"

        result = CliRunner().invoke(
            cli_main.app,
            ["transcribe", str(wav_path), "-o", str(out_path), "--format", "json"],
        )
        assert result.exit_code == 0, result.output

        assert out_path.exists()
        notes = json.loads(out_path.read_text())
        assert isinstance(notes, list)
        for n in notes:
            assert set(n.keys()) == {"is_drum", "program", "onset", "offset", "pitch"}
            assert isinstance(n["is_drum"], bool)
            assert 0 <= n["pitch"] <= 127
            assert n["offset"] >= n["onset"]


# ---------------------------------------------------------------------------
# transcribe — sampling vs greedy
# ---------------------------------------------------------------------------


def test_transcribe_greedy(transcription_model, silence):
    notes = transcription_model.transcribe(
        silence,
        sample_rate=_SAMPLE_RATE,
        return_notes=True,
        use_sampling=False,
    )
    assert isinstance(notes, list)


def test_transcribe_sampling(transcription_model, noise):
    notes = transcription_model.transcribe(
        noise,
        sample_rate=_SAMPLE_RATE,
        return_notes=True,
        use_sampling=True,
        temperature=1.0,
    )
    assert isinstance(notes, list)


def test_transcribe_cfg_coef_zero(transcription_model, silence):
    # cfg_coef=0 means no guidance; should still run and return notes
    notes = transcription_model.transcribe(
        silence,
        sample_rate=_SAMPLE_RATE,
        return_notes=True,
        cfg_coef=0.0,
    )
    assert isinstance(notes, list)


# ---------------------------------------------------------------------------
# Real audio: filling_the_void.wav (first 10 seconds)
# ---------------------------------------------------------------------------


def test_song_clip_shape(song_clip):
    assert song_clip.dim() == 2
    assert song_clip.shape[0] == 1
    assert song_clip.shape[1] == _SAMPLE_RATE * _SONG_CLIP_SEC


def test_transcribe_song_returns_notes(transcription_model, song_clip):
    notes = transcription_model.transcribe(
        song_clip,
        sample_rate=_SAMPLE_RATE,
        return_notes=True,
    )
    assert isinstance(notes, list)
    assert len(notes) > 0, "expected at least some notes from a real song"


def test_transcribe_song_note_validity(transcription_model, song_clip):
    notes = transcription_model.transcribe(
        song_clip,
        sample_rate=_SAMPLE_RATE,
        return_notes=True,
    )
    for note in notes:
        assert 0 <= note.pitch <= 127
        assert 0 <= note.program <= 128
        assert note.onset >= 0.0
        assert note.offset > note.onset


def test_transcribe_song_saves_midi(transcription_model, song_clip):
    from mido import MidiFile

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "filling_the_void_10s.mid"
        transcription_model.transcribe(
            song_clip,
            sample_rate=_SAMPLE_RATE,
            output_midi=out,
        )
        assert out.exists()
        midi = MidiFile(str(out))
        assert len(midi.tracks) > 0


def test_transcribe_song_from_file_path(transcription_model):
    """Transcribe directly from the file path (exercises load_audio)."""
    if not SONG_PATH.exists():
        pytest.skip(f"Song not found at {SONG_PATH}")
    notes = transcription_model.transcribe(
        SONG_PATH,
        return_notes=True,
    )
    assert isinstance(notes, list)
    assert len(notes) > 0
