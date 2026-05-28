"""CLI for muscriptor: audio → MIDI transcription."""

import dataclasses
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from muscriptor.events import NoteEndEvent, NoteStartEvent
from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES
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


@app.command()
def transcribe(
    audio_file: Annotated[
        Path, typer.Argument(help="Input audio file (wav, mp3, flac, …)")
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Output file path. Use '-' to write to stdout (all progress / "
                "timing info is sent to stderr in that case). "
                "Default: <audio_file>.<ext> where ext matches --format."
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
        float, typer.Option("--cfg-coef", help="Classifier-free guidance coefficient")
    ] = 2.0,
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
            help="Batch size for generation (default: 1 on CPU, 4 on GPU)",
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
):
    """Transcribe an audio file to MIDI."""
    instrument_group: str | None = None
    if instruments is not None:
        names = [n.strip() for n in instruments.split(",") if n.strip()]
        unknown = [n for n in names if n not in MT3_FULL_PLUS_GROUP_NAMES]
        if unknown:
            typer.echo(
                f"Error: unknown instrument name(s): {', '.join(unknown)}. "
                "Run 'muscriptor list-instruments' to see available names.",
                err=True,
            )
            raise typer.Exit(1)
        instrument_group = " ".join(str(MT3_FULL_PLUS_GROUP_NAMES[n]) for n in names)

    if not audio_file.exists():
        typer.echo(f"Error: file not found: {audio_file}", err=True)
        raise typer.Exit(1)

    is_stdout = output is not None and str(output) == "-"

    if output is None:
        suffix = {
            OutputFormat.midi: ".mid",
            OutputFormat.json: ".json",
            OutputFormat.jsonl: ".jsonl",
        }[format]
        output = audio_file.with_suffix(suffix)

    _device = None if device == "auto" else device

    # All chatty progress/timing info goes to stderr — stdout is reserved for
    # the actual output when `-o -` is used.
    typer.echo("Loading model…", err=True)
    model = TranscriptionModel.load_model(
        weights_path=model_path,
        device=_device,
    )

    typer.echo(f"Transcribing {audio_file} …", err=True)

    kwargs = dict(
        audio=audio_file,
        use_sampling=sampling,
        temperature=temperature,
        cfg_coef=cfg_coef,
        instrument_group=instrument_group,
        batch_size=batch_size,
        no_eos_is_ok=no_eos_is_ok,
    )

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
    for name in MT3_FULL_PLUS_GROUP_NAMES:
        typer.echo(name)


def main():
    app()


if __name__ == "__main__":
    main()
