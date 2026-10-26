"""CLI tests — checks that `-o -` writes a clean JSONL stream to stdout
and that all progress/timing chatter goes to stderr.

Patches the model loader with a fake so the test stays hermetic.
"""

import json
import wave
from pathlib import Path

import pytest
from typer.testing import CliRunner

import muscriptor.main as main_mod
from muscriptor.events import NoteEndEvent, NoteStartEvent


class _FakeModel:
    @classmethod
    def load_model(cls, **_):
        return cls()

    def transcribe(self, **_):
        s0 = NoteStartEvent(pitch=60, start_time=0.0, index=0, instrument="piano")
        s1 = NoteStartEvent(pitch=64, start_time=0.5, index=1, instrument="guitar")
        yield s0
        yield NoteEndEvent(end_time=0.4, start_event=s0)
        yield s1
        yield NoteEndEvent(end_time=0.9, start_event=s1)

    def transcribe_batch(self, audios, **_):
        for i, _audio in enumerate(audios):
            yield i, list(self.transcribe())

    def events_to_midi(self, events):
        return b"MThd" + bytes([len(list(events))])


@pytest.fixture
def fake_audio(tmp_path: Path) -> Path:
    p = tmp_path / "silent.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 100)
    return p


@pytest.fixture
def patched_model(monkeypatch):
    monkeypatch.setattr(main_mod, "TranscriptionModel", _FakeModel)


def test_jsonl_to_stdout_is_pure_jsonl(patched_model, fake_audio):
    """`-o -` with --format jsonl writes only JSON objects (one per line) to stdout."""
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        ["transcribe", str(fake_audio), "-f", "jsonl", "-o", "-"],
    )
    assert result.exit_code == 0, result.stderr

    # Every non-empty stdout line must be a complete JSON object.
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 4
    parsed = [json.loads(ln) for ln in lines]
    assert parsed[0] == {
        "type": "start",
        "pitch": 60,
        "start_time": 0.0,
        "index": 0,
        "instrument": "piano",
    }
    assert parsed[1] == {"type": "end", "end_time": 0.4, "start_event_index": 0}


def test_jsonl_stdout_has_no_chatter(patched_model, fake_audio):
    """Stdout must contain nothing except JSON — no "Loading model…" etc."""
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        ["transcribe", str(fake_audio), "-f", "jsonl", "-o", "-"],
    )
    assert result.exit_code == 0
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        # Throws if the line isn't valid JSON — proves no banner / "Saved to" /
        # timing line leaked through.
        json.loads(line)


def test_progress_messages_go_to_stderr(patched_model, fake_audio):
    """Loading / transcribing banners must appear on stderr, never stdout."""
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        ["transcribe", str(fake_audio), "-f", "jsonl", "-o", "-"],
    )
    assert result.exit_code == 0
    assert "Loading model" in result.stderr
    assert "Transcribing" in result.stderr
    assert "Loading model" not in result.stdout
    assert "Transcribing" not in result.stdout


def test_jsonl_to_file_keeps_progress_on_stderr(patched_model, fake_audio, tmp_path):
    """Even when writing to a file, banners and "Saved JSONL to …" go to stderr."""
    out = tmp_path / "out.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        ["transcribe", str(fake_audio), "-f", "jsonl", "-o", str(out)],
    )
    assert result.exit_code == 0
    assert result.stdout == ""
    assert "Saved JSONL" in result.stderr
    assert out.read_text().splitlines() == [
        json.dumps(d)
        for d in [
            {
                "type": "start",
                "pitch": 60,
                "start_time": 0.0,
                "index": 0,
                "instrument": "piano",
            },
            {"type": "end", "end_time": 0.4, "start_event_index": 0},
            {
                "type": "start",
                "pitch": 64,
                "start_time": 0.5,
                "index": 1,
                "instrument": "guitar",
            },
            {"type": "end", "end_time": 0.9, "start_event_index": 1},
        ]
    ]


@pytest.fixture
def fake_audios(tmp_path: Path, fake_audio: Path) -> list[Path]:
    (tmp_path / "other").mkdir()
    second = tmp_path / "other" / "second.wav"
    second.write_bytes(fake_audio.read_bytes())
    return [fake_audio, second]


def test_several_inputs_write_next_to_each_input(patched_model, fake_audios):
    runner = CliRunner()
    result = runner.invoke(main_mod.app, ["transcribe", *map(str, fake_audios)])
    assert result.exit_code == 0, result.stderr
    for audio in fake_audios:
        assert audio.with_suffix(".mid").read_bytes() == b"MThd\x04"
    assert "Transcribing 2 files" in result.stderr


def test_several_inputs_to_output_dir(patched_model, fake_audios, tmp_path):
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        [
            "transcribe",
            *map(str, fake_audios),
            "-f",
            "jsonl",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert sorted(p.name for p in out_dir.iterdir()) == ["second.jsonl", "silent.jsonl"]
    lines = (out_dir / "second.jsonl").read_text().splitlines()
    assert [json.loads(ln)["type"] for ln in lines] == ["start", "end", "start", "end"]


def test_several_inputs_reject_single_output(patched_model, fake_audios, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        ["transcribe", *map(str, fake_audios), "-o", str(tmp_path / "x.mid")],
    )
    assert result.exit_code == 1
    assert "--output-dir" in result.stderr


def test_several_inputs_reject_output_name_clash(patched_model, fake_audio, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        [
            "transcribe",
            str(fake_audio),
            str(fake_audio),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 1
    assert "would write to" in result.stderr
