"""TranscriptionModel: main user-facing entry point."""

import contextlib
import io
import itertools
import math
import re
import sys
import time
import warnings
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from safetensors.torch import load_file

from muscriptor.events import (
    NoteEndEvent,
    NoteStartEvent,
    _StreamingEventBuilder,
)
from muscriptor.models.lm import LMModel, TorchAutocast
from muscriptor.utils.download import download_if_necessary
from muscriptor.modules.conditioners import (
    MelSpectrogramConditioner,
    ClassConditioner,
    ConditioningProvider,
    ConditioningAttributes,
    WavCondition,
)
from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES, MT3Tokenizer
from muscriptor.tokenizer.notes import (
    DRUM_PROGRAM,
    Note,
    trim_overlapping_notes,
    validate_notes,
)
from muscriptor.utils.audio import load_audio, resample
from muscriptor.utils.midi import notes_to_midi


@contextlib.contextmanager
def _timed(label: str, store: list[tuple[str, float]] | None = None):
    """Print and (optionally) record how long a block of work takes."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"[muscriptor] {label}: {dt:.2f}s", file=sys.stderr)
    if store is not None:
        store.append((label, dt))


# Placeholder URL — fill in after HuggingFace upload.
@dataclass
class _Chunk:
    """One 5-second chunk of one input, ready for generation."""

    input_index: int
    chunk_index: int
    seek_time: float
    next_seek_time: float | None  # None for the last chunk of an input
    condition: ConditioningAttributes | None  # None for an empty input


AudioInput = str | Path | tuple[torch.Tensor, int]

_HF_URL = "https://huggingface.co/kyutai/muscriptor/resolve/main/muscriptor_weights.safetensors"

_SAMPLE_RATE = 16000
# Must match the segment duration used during training / evaluation.
_SEGMENT_DURATION = 5.0


@dataclass
class _ModelConfig:
    dim: int
    num_heads: int
    num_layers: int
    card: int


# Per-checkpoint configs extracted from .hydra/config.yaml → transformer_lm.
_CONFIGS: dict[str, _ModelConfig] = {
    "01684fbb": _ModelConfig(dim=1536, num_heads=24, num_layers=48, card=1395),
    "0ac4ce03": _ModelConfig(dim=768, num_heads=12, num_layers=14, card=1393),
    "8f59580c": _ModelConfig(dim=1024, num_heads=16, num_layers=24, card=1395),
    "e84904c4": _ModelConfig(dim=1536, num_heads=24, num_layers=48, card=1395),
}

_DEFAULT_CONFIG = _CONFIGS["01684fbb"]


def _config_for_weights(weights_path: Path) -> _ModelConfig:
    """Infer model config from checkpoint hash embedded in the filename."""
    m = re.search(r"_([0-9a-f]{8})_", weights_path.name)
    if m and m.group(1) in _CONFIGS:
        return _CONFIGS[m.group(1)]
    return _DEFAULT_CONFIG


def _build_model(device: torch.device, cfg: _ModelConfig = _DEFAULT_CONFIG) -> LMModel:
    mel_cond = MelSpectrogramConditioner(
        output_dim=cfg.dim,
        device=device,
        sample_rate=_SAMPLE_RATE,
        n_fft=2048,
        frame_rate=100,
        n_mel_bins=512,
        log_scale=True,
        eps=1e-6,
        normalize_audio=False,
    )
    inst_cond = ClassConditioner(num_classes=1000, output_dim=cfg.dim, device=device)
    ds_cond = ClassConditioner(num_classes=4, output_dim=cfg.dim, device=device)

    condition_provider = ConditioningProvider(
        conditioners={
            "self_wav": mel_cond,
            "instrument_group": inst_cond,
            "dataset_name": ds_cond,
        },
        device=device,
    )

    autocast = None
    if device.type == "cuda":
        autocast = TorchAutocast(enabled=True, device_type="cuda", dtype=torch.float16)

    model = LMModel(
        condition_provider=condition_provider,
        delays=[0],
        n_q=1,
        card=cfg.card,
        dim=cfg.dim,
        num_heads=cfg.num_heads,
        hidden_scale=4,
        cfg_coef=1.0,
        autocast=autocast,
        # StreamingTransformer kwargs (forwarded via **kwargs)
        num_layers=cfg.num_layers,
        max_period=10000,
        device=device,
    )
    return model


def _build_instrument_for_program(tokenizer: MT3Tokenizer) -> Callable[[int], str]:
    """Map a decoded program int → human-readable instrument name.

    MT3_FULL_PLUS groups multiple GM programs together; the decoded program
    is always the first program of the group. We map that representative
    back to the readable group name.
    """
    group_map = tokenizer.group_program_map
    program_to_name: dict[int, str] = {}
    for name, gid in MT3_FULL_PLUS_GROUP_NAMES.items():
        if gid in group_map and group_map[gid]:
            program_to_name[group_map[gid][0]] = name

    def lookup(program: int) -> str:
        if program == DRUM_PROGRAM:
            return "drums"
        return program_to_name.get(program, f"program_{program}")

    return lookup


class TranscriptionModel:
    """Transcribes audio to MIDI using the muscriptor model.

    Example::

        from pathlib import Path

        model = TranscriptionModel.load_model()
        for event in model.transcribe("audio.wav"):
            print(event)

        Path("out.mid").write_bytes(model.transcribe_to_midi("audio.wav"))
    """

    def __init__(self, model: LMModel, tokenizer: MT3Tokenizer, device: torch.device):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._instrument_for_program = _build_instrument_for_program(tokenizer)

    @classmethod
    def load_model(
        cls,
        weights_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> "TranscriptionModel":
        """Load model weights and return a ready-to-use TranscriptionModel.

        Args:
            weights_path: Path to a safetensors weights file.  If None, the
                weights are downloaded from HuggingFace to ~/.cache/muscriptor/.
            device: Torch device to use.  Defaults to CUDA if available.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        if weights_path is None:
            weights_path = download_if_necessary(_HF_URL)

        weights_path = Path(weights_path)
        model = _build_model(device, _config_for_weights(weights_path))
        model.eval()

        state_dict = load_file(weights_path, device=str(device))
        model.load_state_dict(state_dict)
        model.to(device)

        tokenizer = MT3Tokenizer(
            instrument_vocabulary="MT3_FULL_PLUS",
            drum_vocabulary="FULL",
            max_shift_steps=1001,
            add_instrument_section=False,
            modulo_max_shift_steps=False,
        )

        return cls(model=model, tokenizer=tokenizer, device=device)

    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio: AudioInput,
        use_sampling: bool = False,
        temperature: float = 1.0,
        cfg_coef: float = 2.0,
        instrument_group: str | None = None,
        dataset_name: str | None = None,
        batch_size: int | None = None,
        no_eos_is_ok: bool = False,
    ) -> Iterator[NoteStartEvent | NoteEndEvent]:
        """Transcribe audio into a stream of note events.

        See the README for full argument documentation and the streaming /
        chunk-ordering guarantees. The audio is split into 5-second chunks;
        within each chunk events arrive in temporal order, and all events
        from chunk N are yielded before any event from chunk N+1.
        """
        t_total = time.perf_counter()
        wav = self._load_input(audio)
        total_duration = wav.shape[-1] / _SAMPLE_RATE
        num_chunks = self._num_chunks(wav)
        print(
            f"[muscriptor] audio: {total_duration:.1f}s → {num_chunks} chunk(s) of {_SEGMENT_DURATION}s",
            file=sys.stderr,
        )

        builder = _StreamingEventBuilder(self._instrument_for_program)
        chunks = self._iter_chunks([wav], instrument_group, dataset_name)
        for (
            _,
            chunk_index,
            seek_time,
            next_seek_time,
            token_list,
        ) in self._generate_chunks(
            chunks,
            batch_size,
            no_eos_is_ok,
            multi_input=False,
            use_sampling=use_sampling,
            temperature=temperature,
            cfg_coef=cfg_coef,
        ):
            if token_list is None:  # empty input
                continue
            note_events, tie_note_events, _, _ = self._tokenizer.tokenizer.decode(
                token_list, start_time=seek_time
            )
            yield from builder.process_chunk(
                note_events, tie_note_events, seek_time, next_seek_time
            )

        yield from builder.flush()

        print(
            f"[muscriptor] transcribe total: {time.perf_counter() - t_total:.2f}s "
            f"({total_duration:.1f}s audio)",
            file=sys.stderr,
        )

    def transcribe_batch(
        self,
        audios: Iterable[AudioInput],
        use_sampling: bool = False,
        temperature: float = 1.0,
        cfg_coef: float = 2.0,
        instrument_group: str | None = None,
        dataset_name: str | None = None,
        batch_size: int | None = None,
        no_eos_is_ok: bool = False,
    ) -> Iterator[tuple[int, list[NoteStartEvent | NoteEndEvent]]]:
        """Transcribe several audio inputs, packing their chunks into shared batches.

        Equivalent to calling :meth:`transcribe` on each input, but the
        5-second chunks of consecutive inputs are batched together, so the
        GPU stays busy across input boundaries (with `transcribe`, the last
        batch of every input is usually only partly full). Inputs are loaded
        lazily, so `audios` can be a long list or a generator.

        Yields `(index, events)` for every input, in input order, as soon as
        its last chunk is decoded. `index` is the position in `audios` and
        `events` is the list :meth:`transcribe` would have yielded.
        """
        t_total = time.perf_counter()
        builders: dict[int, _StreamingEventBuilder] = {}
        events: dict[int, list[NoteStartEvent | NoteEndEvent]] = {}
        chunks = self._iter_chunks(
            (self._load_input(a) for a in audios), instrument_group, dataset_name
        )
        for (
            input_index,
            chunk_index,
            seek_time,
            next_seek_time,
            token_list,
        ) in self._generate_chunks(
            chunks,
            batch_size,
            no_eos_is_ok,
            multi_input=True,
            use_sampling=use_sampling,
            temperature=temperature,
            cfg_coef=cfg_coef,
        ):
            if token_list is not None:
                builder = builders.setdefault(
                    input_index, _StreamingEventBuilder(self._instrument_for_program)
                )
                out = events.setdefault(input_index, [])
                note_events, tie_note_events, _, _ = self._tokenizer.tokenizer.decode(
                    token_list, start_time=seek_time
                )
                out.extend(
                    builder.process_chunk(
                        note_events, tie_note_events, seek_time, next_seek_time
                    )
                )
            if next_seek_time is None:  # last chunk of this input
                out = events.pop(input_index, [])
                if input_index in builders:
                    out.extend(builders.pop(input_index).flush())
                yield input_index, out

        print(
            f"[muscriptor] transcribe_batch total: {time.perf_counter() - t_total:.2f}s",
            file=sys.stderr,
        )

    def _num_chunks(self, wav: torch.Tensor) -> int:
        return math.ceil(wav.shape[-1] / int(_SEGMENT_DURATION * _SAMPLE_RATE))

    def _iter_chunks(
        self,
        wavs: Iterable[torch.Tensor],
        instrument_group: str | None,
        dataset_name: str | None,
    ) -> Iterator[_Chunk]:
        """Split each waveform into 5 s chunks with their conditions, lazily."""
        segment_samples = int(_SEGMENT_DURATION * _SAMPLE_RATE)
        for input_index, wav in enumerate(wavs):
            num_chunks = self._num_chunks(wav)
            if num_chunks == 0:
                # Empty input: no chunk to generate, but still reported.
                yield _Chunk(input_index, 0, 0.0, None, None)
            for i in range(num_chunks):
                chunk = wav[:, i * segment_samples : (i + 1) * segment_samples]
                if chunk.shape[-1] < segment_samples:
                    chunk = F.pad(chunk, (0, segment_samples - chunk.shape[-1]))
                yield _Chunk(
                    input_index,
                    i,
                    i * _SEGMENT_DURATION,
                    (i + 1) * _SEGMENT_DURATION if i + 1 < num_chunks else None,
                    self._build_conditions(chunk, instrument_group, dataset_name)[0],
                )

    def _generate_chunks(
        self,
        chunks: Iterator[_Chunk],
        batch_size: int | None,
        no_eos_is_ok: bool,
        multi_input: bool,
        use_sampling: bool,
        temperature: float,
        cfg_coef: float,
    ) -> Iterator[tuple[int, int, float, float | None, list[int] | None]]:
        """Run generation over `chunks`, `batch_size` chunks at a time.

        Yields `(input_index, chunk_index, seek_time, next_seek_time, tokens)`
        in chunk order; `next_seek_time` is None for the last chunk of an
        input, `tokens` is None for an empty input.
        """
        if batch_size is None:
            batch_size = 4 if self._device.type == "cuda" else 1
        max_gen_len = 512
        eos_id = self._tokenizer.tokenizer.eos_id
        t_gen = time.perf_counter()
        batch_idx = 0
        while batch := list(itertools.islice(chunks, batch_size)):
            todo = [c for c in batch if c.condition is not None]
            tokens = None
            if todo:
                batch_idx += 1
                with _timed(f"generate batch {batch_idx} ({len(todo)} chunks)"):
                    with torch.no_grad():
                        tokens = self._model.generate(
                            conditions=[c.condition for c in todo],
                            max_gen_len=max_gen_len,
                            use_sampling=use_sampling,
                            temp=temperature,
                            top_k=0,
                            top_p=0.0,
                            cfg_coef=cfg_coef,
                            early_stop_on_token=eos_id,
                        )  # [batch, n_q, T_gen]
            j = 0
            for c in batch:
                if c.condition is None:
                    yield c.input_index, c.chunk_index, c.seek_time, None, None
                    continue
                seq = tokens[j]  # [n_q, T_gen]
                j += 1
                eos_mask = seq[0] == eos_id
                if eos_mask.any():
                    seq = seq[:, : eos_mask.int().argmax().item() + 1]
                else:
                    where = f"input {c.input_index} " if multi_input else ""
                    msg = (
                        f"{where}chunk {c.chunk_index} (seek={c.seek_time:.1f}s) "
                        f"did not emit EOS within {max_gen_len} tokens"
                    )
                    if no_eos_is_ok:
                        warnings.warn(msg, RuntimeWarning, stacklevel=2)
                    else:
                        raise RuntimeError(
                            msg
                            + " (pass no_eos_is_ok=True / --no-eos-is-ok to downgrade to a warning)"
                        )
                token_list = self._tokenizer._drop_instrument_section(seq[0].tolist())
                yield (
                    c.input_index,
                    c.chunk_index,
                    c.seek_time,
                    c.next_seek_time,
                    token_list,
                )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(
            f"[muscriptor] generate total: {time.perf_counter() - t_gen:.2f}s",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    def transcribe_to_midi(
        self,
        audio: AudioInput,
        use_sampling: bool = False,
        temperature: float = 1.0,
        cfg_coef: float = 2.0,
        instrument_group: str | None = None,
        dataset_name: str | None = None,
        batch_size: int | None = None,
        no_eos_is_ok: bool = False,
    ) -> bytes:
        """Same as :meth:`transcribe` but returns a MIDI file as bytes.

        Thin wrapper that consumes the event generator, reassembles Notes,
        and serializes them.
        """
        events = self.transcribe(
            audio,
            use_sampling=use_sampling,
            temperature=temperature,
            cfg_coef=cfg_coef,
            instrument_group=instrument_group,
            dataset_name=dataset_name,
            batch_size=batch_size,
            no_eos_is_ok=no_eos_is_ok,
        )
        return self.events_to_midi(events)

    def events_to_midi(self, events: Iterable[NoteStartEvent | NoteEndEvent]) -> bytes:
        """Serialize a transcribed event stream (e.g. one input of
        :meth:`transcribe_batch`) to MIDI file bytes."""
        notes: list[Note] = []
        open_notes: dict[int, Note] = {}
        for ev in events:
            if isinstance(ev, NoteStartEvent):
                is_drum = ev.instrument == "drums"
                program = (
                    DRUM_PROGRAM
                    if is_drum
                    else self._program_for_instrument(ev.instrument)
                )
                note = Note(
                    is_drum=is_drum,
                    program=program,
                    onset=ev.start_time,
                    offset=ev.start_time,  # patched on NoteEndEvent
                    pitch=ev.pitch,
                )
                open_notes[ev.index] = note
            else:  # NoteEndEvent
                note = open_notes.pop(ev.start_event_index)
                note.offset = ev.end_time
                notes.append(note)

        # Match the legacy decoder's note-cleanup pass so the MIDI bytes
        # don't drift from earlier reference outputs.
        notes = validate_notes(notes, fix=True)
        notes = trim_overlapping_notes(notes, sort=True)
        midi = notes_to_midi(notes)
        buf = io.BytesIO()
        midi.save(file=buf)
        return buf.getvalue()

    def _load_input(self, audio: AudioInput) -> torch.Tensor:
        """`_load_wav` for any accepted audio input (path or `(tensor, sr)`)."""
        if isinstance(audio, tuple):
            tensor, sample_rate = audio
            return self._load_wav(tensor, sample_rate)
        return self._load_wav(audio, None)

    def _program_for_instrument(self, instrument: str) -> int:
        """Inverse of `_instrument_for_program` for non-drum instruments."""
        if not hasattr(self, "_inst_to_program"):
            group_map = self._tokenizer.group_program_map
            self._inst_to_program = {
                name: group_map[gid][0]
                for name, gid in MT3_FULL_PLUS_GROUP_NAMES.items()
                if gid in group_map and group_map[gid]
            }
        if instrument in self._inst_to_program:
            return self._inst_to_program[instrument]
        # fallback for unknown names like "program_42"
        if instrument.startswith("program_"):
            return int(instrument.removeprefix("program_"))
        raise ValueError(f"Unknown instrument name: {instrument!r}")

    # ------------------------------------------------------------------
    def _load_wav(
        self, audio: str | Path | torch.Tensor, sample_rate: int | None
    ) -> torch.Tensor:
        """Return mono float32 waveform at 16 kHz, shape [1, T]."""
        if isinstance(audio, (str, Path)):
            wav = load_audio(audio, target_sr=_SAMPLE_RATE)
        else:
            wav = audio.float()
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            if wav.dim() == 3:
                wav = wav.squeeze(0)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sample_rate is not None and sample_rate != _SAMPLE_RATE:
                wav = resample(wav, sample_rate, _SAMPLE_RATE)
        return wav.to(self._device)

    def _build_conditions(
        self,
        wav: torch.Tensor,
        instrument_group: str | None = None,
        dataset_name: str | None = None,
    ) -> list[ConditioningAttributes]:
        """Build a single-element list of ConditioningAttributes for one 5-second chunk."""
        T = wav.shape[-1]
        wav_3d = wav.unsqueeze(0)  # [1, 1, T]
        length = torch.tensor([T], device=self._device)
        wav_cond = WavCondition(
            wav=wav_3d,
            length=length,
            sample_rate=[_SAMPLE_RATE],
            path=[None],
            seek_time=[0.0],
        )
        return [
            ConditioningAttributes(
                wav={"self_wav": wav_cond},
                text={
                    "instrument_group": instrument_group,
                    "dataset_name": dataset_name,
                },
            )
        ]
