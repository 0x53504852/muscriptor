"""CLI for muscriptor: audio → MIDI transcription."""

from pathlib import Path
import typing as tp

import typer

app = typer.Typer(add_completion=False, help="muscriptor — audio-to-MIDI transcription")


@app.command()
def transcribe(
    audio_file: Path = typer.Argument(..., help="Input audio file (wav, mp3, flac, …)"),
    output: tp.Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output MIDI file path (default: <audio_file>.mid)"
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
    model_path: tp.Optional[Path] = typer.Option(
        None, "--model", "-m", help="Path to safetensors weights file"
    ),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', …"
    ),
    instruments: tp.Optional[str] = typer.Option(
        None,
        "--instruments",
        help=(
            "Comma-separated list of expected instrument group names "
            "(e.g. 'acoustic_piano,acoustic_guitar,acoustic_bass'). "
            "Use --list-instruments to see all available names."
        ),
    ),
    list_instruments: bool = typer.Option(
        False,
        "--list-instruments",
        help="Print available instrument group names and exit.",
    ),
):
    """Transcribe an audio file to MIDI."""
    from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES
    from muscriptor.transcription_model import TranscriptionModel

    if list_instruments:
        for name, gid in MT3_FULL_PLUS_GROUP_NAMES.items():
            typer.echo(f"  {gid:2d}  {name}")
        raise typer.Exit(0)

    instrument_group: tp.Optional[str] = None
    if instruments is not None:
        names = [n.strip() for n in instruments.split(",") if n.strip()]
        unknown = [n for n in names if n not in MT3_FULL_PLUS_GROUP_NAMES]
        if unknown:
            typer.echo(
                f"Error: unknown instrument name(s): {', '.join(unknown)}. "
                "Use --list-instruments to see available names.",
                err=True,
            )
            raise typer.Exit(1)
        instrument_group = " ".join(str(MT3_FULL_PLUS_GROUP_NAMES[n]) for n in names)

    if not audio_file.exists():
        typer.echo(f"Error: file not found: {audio_file}", err=True)
        raise typer.Exit(1)

    if output is None:
        output = audio_file.with_suffix(".mid")

    _device = None if device == "auto" else device

    typer.echo("Loading model…")
    model = TranscriptionModel.load_model(
        weights_path=model_path,
        device=_device,
    )

    typer.echo(f"Transcribing {audio_file} …")
    result = model.transcribe(
        audio=audio_file,
        output_midi=output,
        return_notes=notes,
        use_sampling=sampling,
        temperature=temperature,
        cfg_coef=cfg_coef,
        instrument_group=instrument_group,
    )

    typer.echo(f"Saved MIDI to {output}")

    if notes and result is not None:
        typer.echo(f"\n{len(result)} notes:")
        for n in result:
            typer.echo(
                f"  program={n.program:3d}  pitch={n.pitch:3d}  "
                f"onset={n.onset:.3f}  offset={n.offset:.3f}  "
                f"drum={n.is_drum}"
            )


def main():
    app()


if __name__ == "__main__":
    main()
