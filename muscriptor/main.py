"""CLI for muscriptor: audio → MIDI transcription."""

import dataclasses
import json
from enum import Enum
from pathlib import Path

import typer

from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES
from muscriptor.transcription_model import TranscriptionModel

app = typer.Typer(add_completion=False, help="muscriptor — audio-to-MIDI transcription")


class OutputFormat(str, Enum):
    midi = "midi"
    json = "json"


@app.command()
def transcribe(
    audio_file: Path = typer.Argument(..., help="Input audio file (wav, mp3, flac, …)"),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: <audio_file>.<ext> where ext matches --format)",
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.midi,
        "--format",
        "-f",
        help="Output format: midi (default) or json",
        case_sensitive=False,
    ),
    notes: bool = typer.Option(False, "--notes", help="Print decoded notes to stdout"),
    sampling: bool = typer.Option(
        False, "--sampling", help="Use temperature sampling instead of greedy decoding"
    ),
    temperature: float = typer.Option(
        1.0, "--temperature", "-t", help="Sampling temperature (only with --sampling)"
    ),
    cfg_coef: float = typer.Option(
        2.0, "--cfg-coef", help="Classifier-free guidance coefficient"
    ),
    model_path: Path | None = typer.Option(
        None, "--model", "-m", help="Path to safetensors weights file"
    ),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', …"
    ),
    batch_size: int = typer.Option(
        16,
        "--batch-size",
        "-b",
        help="Batch size for generation (number of 5s chunks per forward pass)",
    ),
    no_eos_is_ok: bool = typer.Option(
        False,
        "--no-eos-is-ok",
        help="If set, a chunk failing to emit EOS within max_gen_len produces a warning instead of an error",
    ),
    instruments: str | None = typer.Option(
        None,
        "--instruments",
        help=(
            "Comma-separated list of expected instrument group names "
            "(e.g. 'acoustic_piano,acoustic_guitar,acoustic_bass'). "
            "Run 'muscriptor list-instruments' to see all available names."
        ),
    ),
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

    if output is None:
        output = audio_file.with_suffix(
            ".mid" if format == OutputFormat.midi else ".json"
        )

    _device = None if device == "auto" else device

    typer.echo("Loading model…")
    model = TranscriptionModel.load_model(
        weights_path=model_path,
        device=_device,
    )

    typer.echo(f"Transcribing {audio_file} …")
    result = model.transcribe(
        audio=audio_file,
        output_midi=output if format == OutputFormat.midi else None,
        return_notes=notes or format == OutputFormat.json,
        use_sampling=sampling,
        temperature=temperature,
        cfg_coef=cfg_coef,
        instrument_group=instrument_group,
        batch_size=batch_size,
        no_eos_is_ok=no_eos_is_ok,
    )

    if format == OutputFormat.json:
        assert result is not None
        output.write_text(json.dumps([dataclasses.asdict(n) for n in result], indent=2))
        typer.echo(f"Saved JSON to {output}")
    else:
        typer.echo(f"Saved MIDI to {output}")

    if notes and result is not None:
        typer.echo(f"\n{len(result)} notes:")
        for n in result:
            typer.echo(
                f"  program={n.program:3d}  pitch={n.pitch:3d}  "
                f"onset={n.onset:.3f}  offset={n.offset:.3f}  "
                f"drum={n.is_drum}"
            )


@app.command()
def serve():
    pass


@app.command()
def list_instruments():
    """List the instrument group names accepted by --instruments."""
    for name in MT3_FULL_PLUS_GROUP_NAMES:
        typer.echo(name)


def main():
    app()


if __name__ == "__main__":
    main()
