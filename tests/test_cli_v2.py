"""CLI of the note-level pipeline (MuScriptor V2 weights / --ensemble): folder inputs, -o DIR/, --save-notes,
the ensemble recipe picked from --instruments. The model loader is patched with a fake."""

import csv
import json
import wave
from pathlib import Path

import pytest
from typer.testing import CliRunner

import muscriptor.main as main_mod
from muscriptor.tokenizer.notes import Note


class _FakeV2:
    is_v2 = True
    calls: list = []

    @classmethod
    def load_model(cls, **_):
        return cls()

    def resolve_ensemble(self, ensemble, has_instruments):
        if ensemble in (None, "", "none"):
            return None
        return ("A" if has_instruments else "B") if ensemble == "auto" else ensemble

    def transcribe_notes(self, audios, instruments=None, ensemble=None, **kw):
        _FakeV2.calls.append(
            dict(audios=list(audios), instruments=instruments, ensemble=ensemble, **kw)
        )
        for i, _ in enumerate(_FakeV2.calls[-1]["audios"]):
            yield i, [Note(False, 0, 0.5, 1.0, 60), Note(True, 128, 0.25, 0.26, 57)]

    def instrument_name(self, note):
        return "drums" if note.is_drum else "acoustic_piano"

    def notes_to_midi(self, notes):
        return b"MThd" + bytes([len(notes)])

    def notes_to_events(self, notes):
        from muscriptor.events import NoteEndEvent, NoteStartEvent

        for k, n in enumerate(notes):
            s = NoteStartEvent(
                pitch=n.pitch,
                start_time=n.onset,
                index=k,
                instrument=self.instrument_name(n),
            )
            yield s
            yield NoteEndEvent(end_time=n.offset, start_event=s)


def _wav(path: Path):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 100)
    return path


@pytest.fixture
def songs(tmp_path: Path):
    d = tmp_path / "songs"
    d.mkdir()
    for name in ("b.wav", "a.wav"):
        _wav(d / name)
    (d / "notes.txt").write_text("not audio")
    return d


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    _FakeV2.calls = []
    monkeypatch.setattr(main_mod, "TranscriptionModel", _FakeV2)


def test_folder_to_output_dir_with_notes(songs, tmp_path):
    out = tmp_path / "midi"
    r = CliRunner().invoke(
        main_mod.app,
        [
            "transcribe",
            str(songs),
            "-o",
            f"{out}/",
            "--save-notes",
            "csv,json",
            "--ensemble",
            "auto",
            "--instruments",
            "acoustic_piano,drums",
        ],
    )
    assert r.exit_code == 0, r.stderr
    call = _FakeV2.calls[0]
    assert [p.name for p in call["audios"]] == [
        "a.wav",
        "b.wav",
    ]  # sorted, non-audio files skipped
    assert call["ensemble"] == "A" and call["instruments"] == [
        "acoustic_piano",
        "drums",
    ]
    assert sorted(p.name for p in out.iterdir()) == [
        "a.mid",
        "a.notes.csv",
        "a.notes.json",
        "b.mid",
        "b.notes.csv",
        "b.notes.json",
    ]
    rows = list(csv.DictReader(open(out / "a.notes.csv")))
    assert [r["instrument"] for r in rows] == ["drums", "acoustic_piano"]
    assert (
        rows[0]["pitch"] == "49"
    )  # drum pitch mapped to General MIDI (57 crash 2 -> 49 crash 1)
    assert json.loads((out / "b.notes.json").read_text())[1]["pitch"] == 60


def test_auto_without_instruments_is_recipe_b(songs, tmp_path):
    r = CliRunner().invoke(
        main_mod.app,
        [
            "transcribe",
            str(songs / "a.wav"),
            "--ensemble",
            "auto",
            "-o",
            str(tmp_path / "x.mid"),
        ],
    )
    assert r.exit_code == 0, r.stderr
    assert (
        _FakeV2.calls[0]["ensemble"] == "B" and _FakeV2.calls[0]["instruments"] is None
    )
    assert (tmp_path / "x.mid").read_bytes() == b"MThd\x02"


def test_v2_jsonl_events(songs, tmp_path):
    r = CliRunner().invoke(
        main_mod.app, ["transcribe", str(songs / "a.wav"), "-f", "jsonl", "-o", "-"]
    )
    assert r.exit_code == 0, r.stderr
    lines = [json.loads(x) for x in r.stdout.splitlines() if x.strip()]
    assert [x["type"] for x in lines] == ["start", "end", "start", "end"]
    assert _FakeV2.calls[0]["ensemble"] is None


def test_unknown_instrument_is_refused(songs):
    r = CliRunner().invoke(
        main_mod.app, ["transcribe", str(songs / "a.wav"), "--instruments", "kazoo"]
    )
    assert r.exit_code == 1 and "unknown instrument" in r.stderr


def test_bad_save_notes_is_refused(songs):
    r = CliRunner().invoke(
        main_mod.app, ["transcribe", str(songs / "a.wav"), "--save-notes", "xml"]
    )
    assert r.exit_code == 1


def test_list_instruments_has_drums():
    r = CliRunner().invoke(main_mod.app, ["list-instruments"])
    names = r.stdout.split()
    assert "drums" in names and "acoustic_piano" in names and "voice" in names


def test_single_input_to_output_dir_with_slash(songs, tmp_path):
    out = tmp_path / "out"
    r = CliRunner().invoke(
        main_mod.app, ["transcribe", str(songs / "a.wav"), "-o", f"{out}/"]
    )
    assert r.exit_code == 0, r.stderr
    assert (out / "a.mid").exists()
