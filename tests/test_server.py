"""Hermetic tests for the FastAPI transcription server.

Uses a fake transcriber so no weights / audio decoding is required.
"""

import json
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from muscriptor.events import NoteEndEvent, NoteStartEvent
from muscriptor.server import create_app, event_to_dict


class FakeModel:
    """Minimal transcriber: yields a fixed event stream regardless of input."""

    def __init__(self, events):
        self._events = events
        self.calls: list[str] = []

    def transcribe(self, audio):
        self.calls.append(str(audio))
        yield from self._events


def _wav_bytes(tmp_path: Path) -> bytes:
    """Write a tiny silent WAV so the upload payload is a real file."""
    p = tmp_path / "silent.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    return p.read_bytes()


def _parse_sse(body: str) -> list[dict]:
    """Parse SSE `data: <json>` lines into a list of dicts."""
    out: list[dict] = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        assert chunk.startswith("data: "), f"unexpected SSE chunk: {chunk!r}"
        out.append(json.loads(chunk[len("data: ") :]))
    return out


def test_event_to_dict_start_and_end():
    start = NoteStartEvent(pitch=60, start_time=0.5, index=0, instrument="piano")
    end = NoteEndEvent(end_time=1.5, start_event=start)
    assert event_to_dict(start) == {
        "type": "start",
        "pitch": 60,
        "start_time": 0.5,
        "index": 0,
        "instrument": "piano",
    }
    assert event_to_dict(end) == {
        "type": "end",
        "end_time": 1.5,
        "start_event_index": 0,
    }


def test_transcribe_streams_sse_events(tmp_path):
    s0 = NoteStartEvent(pitch=60, start_time=0.0, index=0, instrument="piano")
    s1 = NoteStartEvent(pitch=64, start_time=0.1, index=1, instrument="guitar")
    events = [
        s0,
        NoteEndEvent(end_time=0.5, start_event=s0),
        s1,
        NoteEndEvent(end_time=0.6, start_event=s1),
    ]
    fake = FakeModel(events)
    client = TestClient(create_app(fake))

    payload = _wav_bytes(tmp_path)
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", payload, "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    parsed = _parse_sse(resp.text)
    assert parsed == [event_to_dict(e) for e in events]
    assert len(fake.calls) == 1


def test_transcribe_empty_stream(tmp_path):
    fake = FakeModel([])
    client = TestClient(create_app(fake))
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
    )
    assert resp.status_code == 200
    assert _parse_sse(resp.text) == []


def test_transcribe_missing_file():
    client = TestClient(create_app(FakeModel([])))
    resp = client.post("/transcribe")
    assert resp.status_code == 422


def test_transcribe_passes_tensor_not_path(tmp_path):
    """Server must hand the model an in-memory (tensor, sr) tuple — no disk."""
    import torch

    captured: dict = {}

    class Capturing(FakeModel):
        def transcribe(self, audio):
            captured["audio"] = audio
            yield from self._events

    model = Capturing([])
    client = TestClient(create_app(model))
    resp = client.post(
        "/transcribe",
        files={"file": ("silent.wav", _wav_bytes(tmp_path), "audio/wav")},
    )
    assert resp.status_code == 200
    audio = captured["audio"]
    assert isinstance(audio, tuple)
    tensor, sr = audio
    assert isinstance(tensor, torch.Tensor)
    assert sr == 16000
    assert tensor.shape[-1] == 1600  # samples we wrote


def test_transcribe_rejects_invalid_wav():
    client = TestClient(create_app(FakeModel([])))
    resp = client.post(
        "/transcribe",
        files={"file": ("garbage.wav", b"not a wav at all", "audio/wav")},
    )
    assert resp.status_code == 400
