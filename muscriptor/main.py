"""CLI for muscriptor: audio → MIDI transcription."""

import dataclasses
import json
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from muscriptor.events import NoteEndEvent, NoteStartEvent
from muscriptor.tokenizer.mt3 import INSTRUMENT_GROUP_NAMES
from muscriptor.transcription_model import TranscriptionModel

app = typer.Typer(add_completion=False, help="muscriptor — audio-to-MIDI transcription")


class OutputFormat(str, Enum):
    midi = "midi"
    json = "json"
    jsonl = "jsonl"


def _event_to_dict(ev: NoteStartEvent | NoteEndEvent) -> dict:
    if isinstance(ev, NoteStartEvent):
        return {"type": "start", **dataclasses.asdict(ev)}
    return {
        "type": "end",
        "end_time": ev.end_time,
        "start_event_index": ev.start_event_index,
    }


AUDIO_SUFFIXES = (
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


def _expand_inputs(paths: list[Path], recursive: bool) -> list[Path]:
    """Files as given; a directory -> its audio files (sorted; with --recursive, those of its subfolders too)."""
    out = []
    for p in paths:
        if p.is_dir():
            found = p.rglob("*") if recursive else p.iterdir()
            out += sorted(
                f for f in found if f.is_file() and f.suffix.lower() in AUDIO_SUFFIXES
            )
        else:
            out.append(p)
    return out


@app.command()
def transcribe(
    audio_files: Annotated[
        list[Path],
        typer.Argument(
            help=(
                "Input audio file(s) (wav, flac, mp3, …) or folder(s) of audio files. Several files are "
                "transcribed together, packing their chunks into shared batches."
            )
        ),
    ],
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Output file path. Use '-' to write to stdout (all progress / "
                "timing info is sent to stderr in that case). "
                "Default: <audio_file>.<ext> where ext matches --format. "
                "With several inputs, a directory (an existing one, or a path ending with '/' "
                "or without suffix) receives <audio_file stem>.<ext> for every input."
            ),
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help=(
                "Directory for the outputs, named <audio_file stem>.<ext>. "
                "Default: next to each input."
            ),
        ),
    ] = None,
    format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            "-f",
            help=(
                "Output format: midi (default), json (single array of events), "
                "or jsonl (one event per line, streamed as transcription progresses)"
            ),
            case_sensitive=False,
        ),
    ] = OutputFormat.midi,
    notes: Annotated[
        bool, typer.Option("--notes", help="Print decoded events to stdout")
    ] = False,
    save_notes: Annotated[
        str | None,
        typer.Option(
            "--save-notes",
            help=(
                "Also write the notes next to each output: 'csv', 'json' or 'csv,json' "
                "(<stem>.notes.csv / <stem>.notes.json: onset, offset, pitch, instrument, program, is_drum; "
                "drum pitches as in the MIDI file). Note-level pipeline only (MuScriptor V2 weights, "
                "--ensemble or --overlap-hop)."
            ),
        ),
    ] = None,
    sampling: Annotated[
        bool,
        typer.Option(
            "--sampling", help="Use temperature sampling instead of greedy decoding"
        ),
    ] = False,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature", "-t", help="Sampling temperature (only with --sampling)"
        ),
    ] = 1.0,
    cfg_coef: Annotated[
        float | None,
        typer.Option(
            "--cfg-coef",
            help="Classifier-free guidance coefficient (default: 2.0, or 1.0 for MuScriptor V2 weights)",
        ),
    ] = None,
    model_path: Annotated[
        Path | None,
        typer.Option("--model", "-m", help="Path to safetensors weights file"),
    ] = None,
    device: Annotated[
        str,
        typer.Option(
            "--device", "-d", help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', …"
        ),
    ] = "auto",
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch-size",
            "-b",
            help=(
                "Batch size for generation: 5 s windows decoded together (default: 1 on CPU, 4 on GPU). "
                "Note-level pipeline (MuScriptor V2 weights, --ensemble, --overlap-hop): the most window decodes on "
                "the device at once (default: from the free GPU memory, 256 on CPU); lower it if the GPU runs out "
                "of memory."
            ),
        ),
    ] = None,
    no_eos_is_ok: Annotated[
        bool,
        typer.Option(
            "--no-eos-is-ok",
            help="If set, a chunk failing to emit EOS within max_gen_len produces a warning instead of an error",
        ),
    ] = False,
    instruments: Annotated[
        str | None,
        typer.Option(
            "--instruments",
            help=(
                "Comma-separated list of expected instrument group names "
                "(e.g. 'acoustic_piano,acoustic_guitar,acoustic_bass'). "
                "Run 'muscriptor list-instruments' to see all available names."
            ),
        ),
    ] = None,
    ensemble: Annotated[
        str,
        typer.Option(
            "--ensemble",
            "-e",
            help=(
                "none (single greedy decode, default), auto (recipe A when --instruments is given, B otherwise), "
                "A (13 prompted decodes + vote), B (13 unprompted decodes + vote), or a variant list "
                "'cfg=1.25;hop=2.5;temp=0.5,seed=1;…'."
            ),
        ),
    ] = "none",
    overlap_hop: Annotated[
        float | None,
        typer.Option(
            "--overlap-hop",
            help=(
                "Single decode: a 5 s window every this many seconds, each owning its central region "
                "(default: 4 for MuScriptor V2 weights, back-to-back windows for the released models)."
            ),
        ),
    ] = None,
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive",
            "-r",
            help="With folder inputs, also take the audio files of their subfolders.",
        ),
    ] = False,
):
    """Transcribe audio file(s) to MIDI."""
    instrument_group: str | None = None
    names: list[str] | None = None
    if instruments is not None:
        names = [n.strip() for n in instruments.split(",") if n.strip()]
        unknown = [n for n in names if n not in INSTRUMENT_GROUP_NAMES]
        if unknown:
            typer.echo(
                f"Error: unknown instrument name(s): {', '.join(unknown)}. "
                "Run 'muscriptor list-instruments' to see available names.",
                err=True,
            )
            raise typer.Exit(1)
        instrument_group = " ".join(str(INSTRUMENT_GROUP_NAMES[n]) for n in names)

    missing = [f for f in audio_files if not f.exists()]
    if missing:
        typer.echo(f"Error: file not found: {', '.join(map(str, missing))}", err=True)
        raise typer.Exit(1)
    audio_files = _expand_inputs(audio_files, recursive)
    if not audio_files:
        typer.echo("Error: no audio file found in the given folder(s).", err=True)
        raise typer.Exit(1)

    is_stdout = output == "-"
    if (
        output is not None
        and not is_stdout
        and output_dir is None
        and (
            output.endswith(("/", os.sep))
            or Path(output).is_dir()
            or (len(audio_files) > 1 and Path(output).suffix == "")
        )
    ):
        output_dir, output = Path(output), None
    if output is not None:
        output = Path(output)

    if output is not None and len(audio_files) > 1:
        typer.echo(
            "Error: --output/-o takes a single input, or a directory with several "
            "(a path ending with '/'); or use --output-dir.",
            err=True,
        )
        raise typer.Exit(1)

    suffix = {
        OutputFormat.midi: ".mid",
        OutputFormat.json: ".json",
        OutputFormat.jsonl: ".jsonl",
    }[format]
    if output is not None:
        outputs = [output]
    elif output_dir is not None:
        outputs = [output_dir / f.with_suffix(suffix).name for f in audio_files]
    else:
        outputs = [f.with_suffix(suffix) for f in audio_files]
    duplicates = sorted({str(o) for o in outputs if outputs.count(o) > 1})
    if duplicates:
        typer.echo(
            f"Error: several inputs would write to {', '.join(duplicates)}", err=True
        )
        raise typer.Exit(1)
    note_formats = [x.strip() for x in (save_notes or "").split(",") if x.strip()]
    bad = [x for x in note_formats if x not in ("csv", "json")]
    if bad:
        typer.echo(
            f"Error: --save-notes takes csv and/or json, not {', '.join(bad)}", err=True
        )
        raise typer.Exit(1)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    _device = None if device == "auto" else device

    # All chatty progress/timing info goes to stderr — stdout is reserved for
    # the actual output when `-o -` is used.
    typer.echo("Loading model…", err=True)
    model = TranscriptionModel.load_model(
        weights_path=model_path,
        device=_device,
    )

    use_notes = (
        bool(getattr(model, "is_v2", False))
        or ensemble not in ("none", "")
        or overlap_hop is not None
    )
    if note_formats and not use_notes:
        typer.echo(
            "Error: --save-notes needs the note-level pipeline (MuScriptor V2 weights, --ensemble or --overlap-hop).",
            err=True,
        )
        raise typer.Exit(1)
    if use_notes:
        if sampling:
            typer.echo(
                "Error: --sampling is not available with the note-level pipeline.",
                err=True,
            )
            raise typer.Exit(1)
        recipe = model.resolve_ensemble(ensemble, names is not None)
        what = "single decode" if recipe is None else f"ensemble {recipe}"
        typer.echo(
            f"Transcribing {len(audio_files)} file(s) ({what}, "
            f"{'instruments: ' + ', '.join(names) if names else 'no instrument prompt'}) …",
            err=True,
        )
        if is_stdout and note_formats:
            typer.echo(
                "Error: --save-notes needs a file or directory output, not '-'.",
                err=True,
            )
            raise typer.Exit(1)
        for i, song_notes in model.transcribe_notes(
            audio_files,
            instruments=names,
            ensemble=recipe,
            cfg_coef=cfg_coef,
            overlap_hop=overlap_hop,
            batch_size=batch_size,
        ):
            _write_notes_output(
                model, song_notes, outputs[i], format, notes, note_formats, is_stdout
            )
        return

    kwargs = dict(
        use_sampling=sampling,
        temperature=temperature,
        cfg_coef=cfg_coef,
        instrument_group=instrument_group,
        batch_size=batch_size,
        no_eos_is_ok=no_eos_is_ok,
    )

    if len(audio_files) > 1:
        typer.echo(f"Transcribing {len(audio_files)} files …", err=True)
        for i, events in model.transcribe_batch(audio_files, **kwargs):
            _write_events(model, events, outputs[i], format, notes)
        return

    typer.echo(f"Transcribing {audio_files[0]} …", err=True)
    kwargs["audio"] = audio_files[0]
    output = outputs[0]

    if format == OutputFormat.midi:
        midi_bytes = model.transcribe_to_midi(**kwargs)
        if is_stdout:
            sys.stdout.buffer.write(midi_bytes)
            sys.stdout.buffer.flush()
        else:
            output.write_bytes(midi_bytes)
            typer.echo(f"Saved MIDI to {output}", err=True)
        if notes:
            typer.echo(
                "Re-run with --format json to inspect the event stream.", err=True
            )
    elif format == OutputFormat.jsonl:
        # Stream one JSON object per line, flushing after each event so the
        # file (or stdout pipe) can be consumed live.
        if is_stdout:
            sink = sys.stdout
            close_after = False
        else:
            sink = output.open("w")
            close_after = True
        try:
            for e in model.transcribe(**kwargs):
                sink.write(json.dumps(_event_to_dict(e)) + "\n")
                sink.flush()
                if notes:
                    typer.echo(str(e), err=True)
        finally:
            if close_after:
                sink.close()
        if not is_stdout:
            typer.echo(f"Saved JSONL to {output}", err=True)
    else:  # json
        events = list(model.transcribe(**kwargs))
        payload = json.dumps([_event_to_dict(e) for e in events], indent=2)
        if is_stdout:
            sys.stdout.write(payload + "\n")
            sys.stdout.flush()
        else:
            output.write_text(payload)
            typer.echo(f"Saved JSON to {output}", err=True)
        if notes:
            for e in events:
                typer.echo(str(e), err=True)


def _note_rows(model, song_notes) -> list[dict]:
    """Notes as output rows: drum pitches as in the MIDI file (General MIDI)."""
    from muscriptor.drums import to_general_midi

    rows = []
    for n in sorted(song_notes, key=lambda n: (n.onset, n.is_drum, n.program, n.pitch)):
        rows.append(
            dict(
                onset=round(n.onset, 4),
                offset=round(n.offset, 4),
                pitch=to_general_midi(n.pitch) if n.is_drum else n.pitch,
                instrument=model.instrument_name(n),
                program=n.program,
                is_drum=n.is_drum,
            )
        )
    return rows


def _write_notes_output(
    model, song_notes, output: Path, format, notes: bool, note_formats, is_stdout: bool
) -> None:
    """Write one input's notes (note-level pipeline): MIDI / JSON / JSONL events, and the optional notes files."""
    import csv

    if format == OutputFormat.midi:
        data = model.notes_to_midi(song_notes)
        if is_stdout:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        else:
            output.write_bytes(data)
    else:
        events = [_event_to_dict(e) for e in model.notes_to_events(song_notes)]
        text = (
            json.dumps(events, indent=2)
            if format == OutputFormat.json
            else "".join(json.dumps(e) + "\n" for e in events)
        )
        if is_stdout:
            sys.stdout.write(text if text.endswith("\n") else text + "\n")
            sys.stdout.flush()
        else:
            output.write_text(text)
    if not is_stdout:
        typer.echo(
            f"Saved {format.value.upper()} to {output} ({len(song_notes)} notes)",
            err=True,
        )
    rows = _note_rows(model, song_notes) if note_formats else []
    base = output.with_suffix("")
    if "csv" in note_formats:
        path = base.with_name(base.name + ".notes.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "onset",
                    "offset",
                    "pitch",
                    "instrument",
                    "program",
                    "is_drum",
                ],
            )
            w.writeheader()
            w.writerows(rows)
        typer.echo(f"Saved notes to {path}", err=True)
    if "json" in note_formats:
        path = base.with_name(base.name + ".notes.json")
        path.write_text(json.dumps(rows, indent=1))
        typer.echo(f"Saved notes to {path}", err=True)
    if notes:
        for r in rows or _note_rows(model, song_notes):
            typer.echo(str(r), err=True)


def _write_events(
    model: TranscriptionModel,
    events: list[NoteStartEvent | NoteEndEvent],
    output: Path,
    format: OutputFormat,
    notes: bool,
) -> None:
    """Write one input's events from `transcribe_batch` to `output`."""
    if format == OutputFormat.midi:
        output.write_bytes(model.events_to_midi(events))
    elif format == OutputFormat.jsonl:
        output.write_text("".join(json.dumps(_event_to_dict(e)) + "\n" for e in events))
    else:  # json
        output.write_text(json.dumps([_event_to_dict(e) for e in events], indent=2))
    typer.echo(f"Saved {format.value.upper()} to {output}", err=True)
    if notes:
        for e in events:
            typer.echo(str(e), err=True)


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="Bind address")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to listen on")] = 8000,
    model_path: Annotated[
        Path | None,
        typer.Option("--model", "-m", help="Path to safetensors weights file"),
    ] = None,
    device: Annotated[
        str,
        typer.Option(
            "--device", "-d", help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', …"
        ),
    ] = "auto",
):
    """Run the HTTP transcription server (POST /transcribe → SSE event stream)."""
    import uvicorn

    from muscriptor.server import create_app

    _device = None if device == "auto" else device
    typer.echo("Loading model…")
    model = TranscriptionModel.load_model(weights_path=model_path, device=_device)
    web_dir = Path(__file__).resolve().parent.parent / "web" / "dist"
    fastapi_app = create_app(model, web_dir=web_dir if web_dir.is_dir() else None)
    uvicorn.run(fastapi_app, host=host, port=port)


@app.command()
def list_instruments():
    """List the instrument group names accepted by --instruments."""
    for name in INSTRUMENT_GROUP_NAMES:
        typer.echo(name)


def main():
    app()


if __name__ == "__main__":
    main()
