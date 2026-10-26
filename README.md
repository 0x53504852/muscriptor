# muscriptor

Audio-to-MIDI transcription using a transformer language model.

## Installation

```bash
pip install -e .
```

## Usage

```python
from pathlib import Path
from muscriptor import TranscriptionModel

model = TranscriptionModel.load_model("muscriptor_weights.safetensors")

# Stream events as they're transcribed
for event in model.transcribe("audio.wav"):
    print(event)

# Or get a MIDI file directly
midi_bytes = model.transcribe_to_midi("audio.wav")
Path("out.mid").write_bytes(midi_bytes)

# Several songs at once: their 5 s chunks are packed into shared batches,
# so the GPU stays busy across song boundaries
paths = ["a.wav", "b.wav", "c.wav"]
for i, events in model.transcribe_batch(paths, batch_size=32):
    Path(paths[i]).with_suffix(".mid").write_bytes(model.events_to_midi(events))
```

```python
from dataclasses import dataclass
from typing import Generator


@dataclass
class NoteStartEvent:
    pitch: int
    # Start of the note in seconds, from the beginning of the audio.
    start_time: float
    # A unique index for this note, used to match the corresponding
    # NoteEndEvent.
    index: int
    instrument: str


@dataclass
class NoteEndEvent:
    end_time: float
    # The NoteStartEvent this end matches. Convenient for consumers —
    # when serializing (e.g. to JSON) drop this field and rely on
    # `start_event_index` to refer back to the start by id.
    start_event: NoteStartEvent

    @property
    def start_event_index(self) -> int:
        return self.start_event.index


class TranscriptionModel:
    ...
    def transcribe(
            self,
            audio: str | Path | tuple[torch.Tensor, int],
            use_sampling: bool = False,
            temperature: float = 1.0,
            cfg_coef: float = 2.0,
            instrument_group: str | None = None,
            dataset_name: str | None = None,
            batch_size: int | None = None,
            no_eos_is_ok: bool = False,
        ) -> Generator[NoteStartEvent | NoteEndEvent, None, None]:
        """Transcribe audio into a stream of note events.

        Args:
            audio: Path to an audio file, or a tuple `(tensor, sample_rate)`
                with a float tensor of shape [T] or [1, T] at `sample_rate`
                Hz. The tuple form is useful when the audio is already
                loaded in memory.
            use_sampling: Use temperature sampling instead of greedy decoding.
            temperature: Sampling temperature (only used when use_sampling=True).
            cfg_coef: Classifier-free guidance coefficient.
            instrument_group: Optional instrument group conditioning string.
            dataset_name: Optional dataset name conditioning string.
            batch_size: Number of 5-second chunks processed per forward
                pass. `None` (default) picks a value based on the device:
                1 on CPU, 4 on GPU. Use `batch_size=1` for the lowest
                streaming latency — larger batches process several chunks
                together, so events belonging to later chunks of a batch
                won't arrive until the whole batch finishes. Within a
                batch, events are always yielded in temporal order; all
                events from chunk N are emitted before any event from
                chunk N+1.
            no_eos_is_ok: If True, a chunk that doesn't emit EOS within
                the generation budget produces a warning instead of raising.

        Returns:
            Generator of NoteStartEvent and NoteEndEvent objects. Every
            NoteStartEvent is guaranteed to be followed by exactly one
            matching NoteEndEvent later in the stream (with the same
            `index`). Drum hits appear as a NoteStartEvent immediately
            followed by its matching NoteEndEvent at the same start time
            plus a tiny duration. Note: this tokenizer does not preserve
            velocity (loudness) — only onset/offset timing, pitch, and
            instrument are recovered.
        """

    def transcribe_batch(
            self,
            audios: Iterable[str | Path | tuple[torch.Tensor, int]],
            ...,  # same options as transcribe
        ) -> Iterator[tuple[int, list[NoteStartEvent | NoteEndEvent]]]:
        """Transcribe several inputs, packing the chunks of consecutive
        inputs into the same batches (with `transcribe`, the last batch of
        every input is usually only partly full). Inputs are loaded lazily,
        so `audios` can be a long list or a generator.

        Yields `(index, events)` per input, in input order, as soon as its
        last chunk is decoded; `events` is what `transcribe` would yield.
        A chunk without EOS warns / raises with "input <index> chunk <n>".
        """

    def events_to_midi(self, events) -> bytes:
        """Serialize an event list (e.g. from `transcribe_batch`) to MIDI bytes."""

    def transcribe_to_midi(
            self,
            audio: str | Path | tuple[torch.Tensor, int],
            use_sampling: bool = False,
            temperature: float = 1.0,
            cfg_coef: float = 2.0,
            instrument_group: str | None = None,
            dataset_name: str | None = None,
            batch_size: int | None = None,
            no_eos_is_ok: bool = False,
        ) -> bytes:
        """Same as `transcribe`, but returns a MIDI file as bytes instead
        of a generator of events. Useful when you want to save the MIDI
        to disk or send it over a network without going through the
        event stream.
        """
```


## CLI

```bash
muscriptor transcribe audio.wav -o out.mid
# several files, batched together; outputs next to each input or in --output-dir
muscriptor transcribe a.wav b.wav c.wav --batch-size 32 --output-dir midi/
```

## Web UI

A browser client is included under `web/`. The FastAPI server serves both
the UI and a POST `/transcribe` endpoint that streams `NoteStart`/`NoteEnd`
events back as Server-Sent Events. The UI accepts a WAV via drag-and-drop,
renders a live piano roll while events arrive, auto-plays once enough notes
are available, and crossfades between the original WAV and the synthesized
MIDI playback.

### One-time setup

```bash
uv sync
cd web && npm install && npm run build && cd ..
```

`npm run build` is required once — the FastAPI server auto-mounts
`web/dist/` if it exists.

### Run

```bash
uv run muscriptor serve \
    --model ./muscriptor_weights_8f59580c_750.safetensors \
    --device cuda \
    --host 0.0.0.0 \
    --port 8000
```

Then open <http://127.0.0.1:8000/> (or the LAN address of the host) and drop
a WAV onto the page.

- Drop `--device cuda` if running CPU-only.
- `--host 0.0.0.0` makes it reachable on the LAN; the default `127.0.0.1`
  is local-only.
- The first transcription downloads ~5 MB of General-MIDI soundfonts and
  TR-808 drum samples from `gleitz.github.io` and `smpldsnds.github.io`
  (cached by the browser thereafter), so working outbound HTTPS is
  required on the first run.
