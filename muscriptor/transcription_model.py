"""TranscriptionModel: main user-facing entry point."""

import contextlib
import hashlib
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
from muscriptor.tokenizer.mt3 import (
    INSTRUMENT_GROUP_NAMES,
    MT3_FULL_PLUS_GROUP_NAMES,
    MT3Tokenizer,
)
from muscriptor.tokenizer import v2 as v2tok
from muscriptor.tokenizer.notes import (
    DRUM_PROGRAM,
    Note,
    trim_overlapping_notes,
    validate_notes,
)
from muscriptor.utils.audio import load_audio, load_audio_v2, resample
from muscriptor.utils.midi import notes_to_midi, notes_to_multitrack_midi


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


def _v2_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Training-repo key names -> this package's (the single-stream embedding / output layer of audiocraft)."""
    out = {}
    for k, v in state.items():
        if k.startswith("emb."):
            k = "emb.0." + k[len("emb.") :]
        elif k.startswith("linear."):
            k = "linears.0." + k[len("linear.") :]
        out[k] = v
    if "condition_provider.conditioners.self_wav.audio_input_code" in out:
        raise ValueError(
            "two-channel (stereo) MuScriptor V2 weights are not supported by this package"
        )
    return out


def _config_for_state(state: dict[str, torch.Tensor]) -> _ModelConfig:
    """Model size from the weight shapes (64-dimensional heads, as every released size)."""
    dim = state["emb.0.weight"].shape[1]
    card = state["linears.0.weight"].shape[0]
    num_layers = len(
        {k.split(".")[2] for k in state if k.startswith("transformer.layers.")}
    )
    return _ModelConfig(dim=dim, num_heads=dim // 64, num_layers=num_layers, card=card)


def _build_model(
    device: torch.device,
    cfg: _ModelConfig = _DEFAULT_CONFIG,
    autocast_dtype: torch.dtype = torch.float16,
) -> LMModel:
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
        autocast = TorchAutocast(enabled=True, device_type="cuda", dtype=autocast_dtype)

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
    for name, gid in INSTRUMENT_GROUP_NAMES.items():
        if gid >= len(
            MT3_FULL_PLUS_GROUP_NAMES
        ):  # the singleton groups (drums are handled below)
            program_to_name.setdefault(v2tok.PREFIX_GROUPS[gid][0], name)

    def lookup(program: int) -> str:
        if program == DRUM_PROGRAM:
            return "drums"
        return program_to_name.get(program, f"program_{program}")

    return lookup


def _prompt_ids(instrument_group: str | None) -> list[int] | None:
    """The old API's instrument_group string ("0 4 ...") -> group ids."""
    return (
        None if instrument_group is None else [int(x) for x in instrument_group.split()]
    )


class TranscriptionModel:
    """Transcribes audio to MIDI using the muscriptor model.

    Example::

        from pathlib import Path

        model = TranscriptionModel.load_model()
        for event in model.transcribe("audio.wav"):
            print(event)

        Path("out.mid").write_bytes(model.transcribe_to_midi("audio.wav"))
    """

    def __init__(
        self,
        model: LMModel,
        tokenizer: MT3Tokenizer,
        device: torch.device,
        v2: bool = False,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._instrument_for_program = _build_instrument_for_program(tokenizer)
        # MuScriptor V2 weights (training-repo export): bf16, conditions under autocast, julius resampling, 5 s windows
        # every 4 s with central regions, cfg 1.0, sorted instrument prompt, no dataset id, the ensemble recipes.
        self.is_v2 = v2
        # Timings and counts of the last transcribe_notes call.
        self.last_stats: dict = {}

    @classmethod
    def load_model(
        cls,
        weights_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> "TranscriptionModel":
        """Load model weights and return a ready-to-use TranscriptionModel.

        Args:
            weights_path: Path to a safetensors weights file: a released muscriptor checkpoint, or a MuScriptor V2
                export of the training repository (keys `emb.weight` / `linear.weight`; size and compact vocabulary
                read from the shapes). If None, the weights are downloaded from HuggingFace to ~/.cache/muscriptor/.
            device: Torch device to use.  Defaults to CUDA if available.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        if weights_path is None:
            weights_path = download_if_necessary(_HF_URL)

        weights_path = Path(weights_path)
        state_dict = load_file(weights_path, device=str(device))
        v2 = "emb.weight" in state_dict and "linear.weight" in state_dict
        if v2:
            state_dict = _v2_state_dict(state_dict)
            cfg = _config_for_state(state_dict)
            model = _build_model(device, cfg, autocast_dtype=torch.bfloat16)
            if cfg.card == len(v2tok.COMPACT_IDS):
                model.token_of_class = torch.tensor(v2tok.COMPACT_IDS, device=device)
            model.autocast_conditions = True
            model.verbose = False
            model.condition_provider.conditioners["self_wav"].verbose = False
        else:
            model = _build_model(device, _config_for_weights(weights_path))
        model.eval()
        model.load_state_dict(state_dict)
        model.to(device)

        tokenizer = MT3Tokenizer(
            instrument_vocabulary="MT3_FULL_PLUS",
            drum_vocabulary="FULL",
            max_shift_steps=1001,
            add_instrument_section=False,
            modulo_max_shift_steps=False,
        )

        return cls(model=model, tokenizer=tokenizer, device=device, v2=v2)

    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio: AudioInput,
        use_sampling: bool = False,
        temperature: float = 1.0,
        cfg_coef: float | None = None,
        instrument_group: str | None = None,
        dataset_name: str | None = None,
        batch_size: int | None = None,
        no_eos_is_ok: bool = False,
        overlap_hop: float | None = None,
        ensemble: str | None = None,
    ) -> Iterator[NoteStartEvent | NoteEndEvent]:
        """Transcribe audio into a stream of note events.

        See the README for full argument documentation and the streaming /
        chunk-ordering guarantees. The audio is split into 5-second chunks;
        within each chunk events arrive in temporal order, and all events
        from chunk N are yielded before any event from chunk N+1.

        cfg_coef None: 2.0 for the released models, 1.0 for MuScriptor V2 weights. MuScriptor V2 weights, or
        `overlap_hop` / `ensemble` given: decoded by :meth:`transcribe_notes` (overlapping windows / the ensemble),
        and the events come once the whole input is decoded, sorted by time.
        """
        if self._new_path(overlap_hop, ensemble, use_sampling):
            ((_, notes),) = self.transcribe_notes(
                [audio],
                instruments=_prompt_ids(instrument_group),
                ensemble=ensemble,
                cfg_coef=cfg_coef,
                overlap_hop=overlap_hop,
                batch_size=batch_size,
                dataset_name=dataset_name,
            )
            yield from self.notes_to_events(notes)
            return
        cfg_coef = 2.0 if cfg_coef is None else cfg_coef
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
        cfg_coef: float | None = None,
        instrument_group: str | None = None,
        dataset_name: str | None = None,
        batch_size: int | None = None,
        no_eos_is_ok: bool = False,
        overlap_hop: float | None = None,
        ensemble: str | None = None,
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
        if self._new_path(overlap_hop, ensemble, use_sampling):
            for i, notes in self.transcribe_notes(
                audios,
                instruments=_prompt_ids(instrument_group),
                ensemble=ensemble,
                cfg_coef=cfg_coef,
                overlap_hop=overlap_hop,
                batch_size=batch_size,
                dataset_name=dataset_name,
            ):
                yield i, list(self.notes_to_events(notes))
            return
        cfg_coef = 2.0 if cfg_coef is None else cfg_coef
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
        cfg_coef: float | None = None,
        instrument_group: str | None = None,
        dataset_name: str | None = None,
        batch_size: int | None = None,
        no_eos_is_ok: bool = False,
        overlap_hop: float | None = None,
        ensemble: str | None = None,
    ) -> bytes:
        """Same as :meth:`transcribe` but returns a MIDI file as bytes.

        Thin wrapper that consumes the event generator, reassembles Notes,
        and serializes them. MuScriptor V2 weights / overlap_hop / ensemble: the notes of :meth:`transcribe_notes`
        written by :meth:`notes_to_midi` (one track per instrument).
        """
        if self._new_path(overlap_hop, ensemble, use_sampling):
            ((_, notes),) = self.transcribe_notes(
                [audio],
                instruments=_prompt_ids(instrument_group),
                ensemble=ensemble,
                cfg_coef=cfg_coef,
                overlap_hop=overlap_hop,
                batch_size=batch_size,
                dataset_name=dataset_name,
            )
            return self.notes_to_midi(notes)
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

    # ------------------------------------------------------------------
    # Note-level transcription: MuScriptor V2 weights, overlapping windows, ensembles, many songs
    # ------------------------------------------------------------------
    def _new_path(self, overlap_hop, ensemble, use_sampling: bool = False) -> bool:
        if not (self.is_v2 or overlap_hop or (ensemble and ensemble != "none")):
            return False
        if use_sampling:
            raise ValueError(
                "use_sampling is not available with MuScriptor V2 weights / overlap_hop / ensemble "
                "(the ensemble recipes hold their own sampled decodes)"
            )
        return True

    def instrument_prompt(
        self, instruments: Iterable[str | int] | None
    ) -> list[int] | None:
        """Instrument names (`muscriptor list-instruments`) or group ids -> the sorted group ids of the prompt, as the
        V2 training code builds it (None: no prompt)."""
        if instruments is None:
            return None
        ids = []
        for x in instruments:
            if isinstance(x, int) or (isinstance(x, str) and x.lstrip("-").isdigit()):
                g = int(x)
                if not 0 <= g < len(v2tok.PREFIX_GROUPS):
                    raise ValueError(
                        f"instrument group id {g} out of range 0..{len(v2tok.PREFIX_GROUPS) - 1}"
                    )
            elif x in INSTRUMENT_GROUP_NAMES:
                g = INSTRUMENT_GROUP_NAMES[x]
            else:
                raise ValueError(
                    f"unknown instrument name {x!r} (run 'muscriptor list-instruments' for the names)"
                )
            ids.append(g)
        if not ids:
            return None
        ids = sorted(set(ids))
        return ids

    def resolve_ensemble(
        self, ensemble: str | None, has_instruments: bool
    ) -> str | None:
        """'auto' -> recipe A with instruments, B without; None / 'none' / '' -> None (single decode)."""
        if ensemble is None or ensemble in ("", "none", "off"):
            return None
        if ensemble == "auto":
            return "A" if has_instruments else "B"
        return ensemble

    def transcribe_notes(
        self,
        audios: Iterable[AudioInput],
        instruments: Iterable[str | int] | None = None,
        ensemble: str | None = None,
        *,
        song_instruments: list[Iterable[str | int] | None] | None = None,
        cfg_coef: float | None = None,
        overlap_hop: float | None = None,
        batch_size: int | None = None,
        dataset_name: str | None = None,
        max_gen_len: int | None = None,
        seed_keys: list[str] | None = None,
        backend: str | None = None,
        max_rows: int | None = None,
        max_slots: int | None = None,
        return_decodes: bool = False,
        verbose: bool = True,
        decoder: str = "batched",
    ) -> Iterator[tuple[int, list[Note]]]:
        """Transcribe several inputs to note lists (the V2 decoding pipeline), packing their windows into shared batches.

        Single decode (ensemble None): greedy, 5 s windows every `overlap_hop` s (None: 4 s for V2 weights, back to
        back for the released models), each window owning the events of its central region, classifier-free guidance
        `cfg_coef` (None: 1.0 for V2, 2.0 for the released models).
        Ensemble (`ensemble` "A", "B", "auto" or a variant list, see muscriptor.ensemble.parse_variants): every decode
        of the recipe of every song in one batched pass (muscriptor.ensemble.generate_calls), then the consensus vote
        (muscriptor.vote; k 4, 50 ms, median onset / offset, dedup 50 ms). "auto": A when instruments are given, B
        otherwise. Recipe A is meant for prompted decoding and B for unprompted decoding: the prompt is used as given.

        instruments: names (`muscriptor list-instruments`) or group ids for every input; song_instruments: one such
        list (or None) per input, overriding `instruments`. batch_size: the most window decodes (rows) on the device
        at once (None: from the free GPU memory; 256 on CPU). seed_keys: per input, the string whose hash seeds the
        sampled decodes (default: the path as given, `input<i>` for tensors). return_decodes (ensemble): yield
        (index, (voted notes, [notes of each decode])) instead. decoder: "batched" (muscriptor.ensemble, also for
        the single decode: windows of several songs decoded together, finished windows leaving the batch) or
        "generate" (single decode only: LMModel.generate on batches of `batch_size` windows, default 64 on GPU, 4 on
        CPU).

        Yields (index, notes) per input in input order; notes are muscriptor.tokenizer.notes.Note (drums: program 128,
        is_drum True, the pitch written by the model; muscriptor.drums.to_general_midi maps it to a GM drum note).
        Timings of the call: self.last_stats.
        """
        from muscriptor import ensemble as ens

        audios = list(audios)
        n = len(audios)
        if song_instruments is not None and len(song_instruments) != n:
            raise ValueError("song_instruments needs one entry per input")
        prompts = [
            self.instrument_prompt(
                song_instruments[i] if song_instruments is not None else instruments
            )
            for i in range(n)
        ]
        hop = overlap_hop if overlap_hop is not None else (4.0 if self.is_v2 else 0.0)
        cfg = cfg_coef if cfg_coef is not None else (1.0 if self.is_v2 else 2.0)
        max_gen_len = max_gen_len or (1920 if self.is_v2 else 512)
        gen_batch = batch_size or (64 if self._device.type == "cuda" else 4)
        max_rows = max_rows or batch_size
        keys = (
            list(seed_keys)
            if seed_keys is not None
            else [
                str(a) if isinstance(a, (str, Path)) else f"input{i}"
                for i, a in enumerate(audios)
            ]
        )
        stats = self.last_stats = {
            "songs": n,
            "audio_s": 0.0,
            "load_s": 0.0,
            "decode_s": 0.0,
            "notes_s": 0.0,
        }
        t_start = time.perf_counter()
        log = (
            (lambda m: print(f"[muscriptor] {m}", file=sys.stderr))
            if verbose
            else (lambda m: None)
        )

        recipe = None
        if ensemble is not None and ensemble not in ("", "none", "off"):
            # 'auto' per song: A with a prompt, B without
            recipe = ensemble
        if recipe is None and decoder == "generate":
            yield from self._notes_single(
                audios,
                prompts,
                cfg,
                hop,
                gen_batch,
                dataset_name,
                max_gen_len,
                stats,
                log,
            )
        else:
            single = (ens.Variant("single", cfg, 0.0, -1, hop),)
            groups: dict[object, list[int]] = {}
            for i in range(n):
                key = (
                    single
                    if recipe is None
                    else self.resolve_ensemble(recipe, prompts[i] is not None)
                )
                groups.setdefault(key, []).append(i)
            pending: dict[int, object] = {}
            next_out = 0
            for name, idx in groups.items():
                variants = name if isinstance(name, tuple) else ens.parse_variants(name)
                for i, res in self._notes_ensemble(
                    [audios[i] for i in idx],
                    [prompts[i] for i in idx],
                    [keys[i] for i in idx],
                    variants,
                    dataset_name,
                    max_gen_len,
                    backend,
                    max_rows,
                    max_slots,
                    stats,
                    log,
                    verbose,
                ):
                    pending[idx[i]] = res
                    while next_out in pending:
                        voted, decodes = pending.pop(next_out)
                        if recipe is None:
                            yield next_out, decodes[0]
                        else:
                            yield (
                                next_out,
                                ((voted, decodes) if return_decodes else voted),
                            )
                        next_out += 1
        stats["total_s"] = time.perf_counter() - t_start
        log(
            f"{n} input(s), {stats['audio_s']:.1f} s of audio: {stats['total_s']:.1f} s "
            f"(decoding {stats['decode_s']:.1f} s)"
        )

    def _windows(
        self, wav: torch.Tensor, hop_sec: float
    ) -> tuple[torch.Tensor, list[float]]:
        """[1, N] audio -> 5 s windows [num_windows, 80000] and their start times: every hop_sec s (the last window zero
        padded past the end), or back to back for hop_sec 0 (training.evaluate.transcribe_windows)."""
        audio = wav[0]
        seg = int(_SEGMENT_DURATION * _SAMPLE_RATE)
        hop = round(hop_sec * _SAMPLE_RATE) if hop_sec else 0
        if hop:
            if not 0 < hop <= seg or hop % (_SAMPLE_RATE // v2tok.FRAME_RATE):
                raise ValueError(
                    f"overlap_hop {hop_sec} must be in (0, 5] and a multiple of 10 ms"
                )
            num = 1 + max(0, math.ceil((audio.shape[0] - seg) / hop))
            padded = F.pad(audio, (0, (num - 1) * hop + seg - audio.shape[0]))
            return padded.unfold(0, seg, hop), [
                k * hop / _SAMPLE_RATE for k in range(num)
            ]
        num = max(1, math.ceil(audio.shape[0] / seg))
        windows = F.pad(audio, (0, num * seg - audio.shape[0])).view(num, -1)
        return windows, [k * _SEGMENT_DURATION for k in range(num)]

    def _decode_windows(
        self, tokens: list[list[int]], seek_times: list[float], hop_sec: float
    ) -> list[Note]:
        if hop_sec:
            notes, _ = v2tok.decode_track_overlap(tokens, seek_times, _SEGMENT_DURATION)
        else:
            notes, _ = v2tok.decode_track(tokens, seek_times)
        return notes

    def _load_v2(self, audio: AudioInput) -> torch.Tensor:
        """[1, N] float32 at 16 kHz on CPU, read as the V2 training code reads audio (load_audio_v2)."""
        from muscriptor.utils.resample import resample_frac

        if isinstance(audio, tuple):
            wav, sr = audio
            wav = wav.float().cpu()
            if wav.dim() == 1:
                wav = wav[None]
            if wav.dim() == 3:
                wav = wav[0]
            wav = wav.mean(0, keepdim=True)
            return (
                resample_frac(wav, int(sr), _SAMPLE_RATE) if sr != _SAMPLE_RATE else wav
            )
        return load_audio_v2(audio, target_sr=_SAMPLE_RATE)

    def _prompt_text(self, prompt: list[int] | None) -> str | None:
        return None if prompt is None else " ".join(str(g) for g in prompt)

    def _notes_single(
        self,
        audios,
        prompts,
        cfg,
        hop,
        batch_size,
        dataset_name,
        max_gen_len,
        stats,
        log,
    ):
        """Greedy single decode of every input, windows of consecutive inputs packed into batches of batch_size (a
        batch only holds inputs with the same prompt width: no padding positions in the prefix)."""
        eos = self._tokenizer.tokenizer.eos_id
        n = len(audios)
        i_next = 0
        chunk_windows = max(4 * batch_size, 256)
        while i_next < n:
            # load inputs until the chunk holds enough windows
            songs = []  # (index, windows, seek_times, duration)
            total = 0
            while i_next < n and (not songs or total < chunk_windows):
                t0 = time.perf_counter()
                wav = (
                    self._load_v2(audios[i_next])
                    if self.is_v2
                    else self._load_input(audios[i_next]).cpu()
                )
                stats["load_s"] += time.perf_counter() - t0
                windows, seeks = self._windows(wav, hop)
                songs.append((i_next, windows, seeks, wav.shape[-1] / _SAMPLE_RATE))
                stats["audio_s"] += wav.shape[-1] / _SAMPLE_RATE
                total += windows.shape[0]
                i_next += 1
            t0 = time.perf_counter()
            tokens: dict[int, list] = {i: [None] * w.shape[0] for i, w, _, _ in songs}
            by_width: dict[int, list] = {}
            for i, windows, _, _ in songs:
                width = 1 if prompts[i] is None else len(prompts[i])
                by_width.setdefault(width, []).extend(
                    (i, j) for j in range(windows.shape[0])
                )
            win_of = {i: w for i, w, _, _ in songs}
            for items in by_width.values():
                for start in range(0, len(items), batch_size):
                    part = items[start : start + batch_size]
                    conds = []
                    for i, j in part:
                        chunk = win_of[i][j][None].to(self._device)
                        conds += self._build_conditions(
                            chunk, self._prompt_text(prompts[i]), dataset_name
                        )
                    with torch.no_grad():
                        out = self._model.generate(
                            conditions=conds,
                            max_gen_len=max_gen_len,
                            use_sampling=False,
                            cfg_coef=cfg,
                            early_stop_on_token=eos,
                        )  # [B, 1, T] classes
                    out = self._model.classes_to_tokens(out[:, 0]).cpu()
                    for (i, j), seq in zip(part, out.tolist()):
                        if eos in seq:
                            seq = seq[: seq.index(eos) + 1]
                        else:
                            warnings.warn(
                                f"input {i} window {j}: no EOS within {max_gen_len} tokens",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                        tokens[i][j] = seq
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            stats["decode_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            for i, windows, seeks, dur in songs:
                notes = self._decode_windows(tokens[i], seeks, hop)
                log(
                    f"input {i}: {dur:.1f} s, {windows.shape[0]} windows, {len(notes)} notes"
                )
                yield i, notes
            stats["notes_s"] += time.perf_counter() - t0

    def _notes_ensemble(
        self,
        audios,
        prompts,
        keys,
        variants,
        dataset_name,
        max_gen_len,
        backend,
        max_rows,
        max_slots,
        stats,
        log,
        verbose=False,
    ):
        """Every variant of every input in one batched pass per chunk of inputs, then the vote per input."""
        from muscriptor import ensemble as ens
        from muscriptor.vote import note_rows, vote, vote_notes

        n = len(audios)
        hops = list(dict.fromkeys(v.hop for v in variants))
        rows_cap = (
            max_rows
            or ens._capacity(
                self._model, self._device, ens._autocast_dtype(self._model), max_gen_len
            )[0]
        )
        i_next = 0
        carry = None  # an input loaded for the previous chunk that did not fit in it
        while i_next < n or carry is not None:
            songs = []  # (index, {hop: (windows, seeks)}, duration)
            rows = 0
            while carry is not None or i_next < n:
                if carry is not None:
                    i, cuts, dur, r = carry
                    carry = None
                else:
                    t0 = time.perf_counter()
                    wav = (
                        self._load_v2(audios[i_next])
                        if self.is_v2
                        else self._load_input(audios[i_next]).cpu()
                    )
                    stats["load_s"] += time.perf_counter() - t0
                    cuts = {h: self._windows(wav, h) for h in hops}
                    r = sum(
                        cuts[v.hop][0].shape[0] * (2 if v.cfg_coef != 1.0 else 1)
                        for v in variants
                    )
                    i, dur = i_next, wav.shape[-1] / _SAMPLE_RATE
                    i_next += 1
                    stats["audio_s"] += dur
                if songs and rows + r > rows_cap:
                    carry = (i, cuts, dur, r)
                    break
                songs.append((i, cuts, dur))
                rows += r
            batches, calls, owner = [], [], []
            for si, (i, cuts, _) in enumerate(songs):
                batch_of, n_batches = {}, {}
                for h, (windows, _) in cuts.items():
                    starts = list(range(0, windows.shape[0], ens.ENSEMBLE_BATCH))
                    n_batches[h] = len(starts)
                    for b, start in enumerate(starts):
                        batch_of[(h, b)] = len(batches)
                        batches.append(
                            ens.WindowBatch(
                                windows[start : start + ens.ENSEMBLE_BATCH].to(
                                    self._device
                                ),
                                self._prompt_text(prompts[i]),
                                dataset_name,
                            )
                        )
                path_hash = int(hashlib.sha1(keys[i].encode()).hexdigest()[:8], 16)
                gens = []
                for v in variants:
                    g = None
                    if v.temp > 0:
                        g = torch.Generator(device=self._device)
                        if v.seed >= 0:
                            g.manual_seed(v.seed * 1_000_003 + path_hash)
                        else:
                            g.seed()
                    gens.append(g)
                for b in range(
                    max(n_batches.values())
                ):  # batch-major: the variants of a window batch share its slots
                    for vi, v in enumerate(variants):
                        if b < n_batches[v.hop]:
                            calls.append(
                                ens.Call(
                                    batch_of[(v.hop, b)], v.cfg_coef, v.temp, gens[vi]
                                )
                            )
                            owner.append((si, vi))
            t0 = time.perf_counter()
            ens_stats = stats.setdefault("ensemble", {})
            out = ens.generate_calls(
                self._model,
                batches,
                calls,
                max_gen_len=max_gen_len,
                backend=backend,
                max_rows=max_rows,
                max_slots=max_slots,
                stats=ens_stats,
                verbose=verbose,
            )
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            stats["decode_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            tokens = [[[] for _ in variants] for _ in songs]
            for (si, vi), res in zip(owner, out):
                tokens[si][vi] += res
            for si, (i, cuts, dur) in enumerate(songs):
                decodes = []
                for vi, v in enumerate(variants):
                    _, seeks = cuts[v.hop]
                    decodes.append(self._decode_windows(tokens[si][vi], seeks, v.hop))
                if len(variants) == 1:
                    voted = decodes[0]
                    log(f"input {i}: {dur:.1f} s, {len(voted)} notes")
                else:
                    voted = vote_notes(
                        vote(
                            [note_rows(d) for d in decodes],
                            ens.VOTE_K,
                            tol=ens.VOTE_TOL,
                            median=ens.VOTE_MEDIAN,
                            dedup=ens.VOTE_DEDUP,
                        )
                    )
                    log(
                        f"input {i}: {dur:.1f} s, {len(variants)} decodes, {len(voted)} voted notes"
                    )
                yield i, (voted, decodes)
            stats["notes_s"] += time.perf_counter() - t0
            del batches

    # ------------------------------------------------------------------
    def instrument_name(self, note: Note) -> str:
        """Name of a note's instrument ("drums", a group name, or "program_<n>")."""
        return "drums" if note.is_drum else self._instrument_for_program(note.program)

    def notes_to_midi(self, notes: list[Note]) -> bytes:
        """MIDI file bytes (type 1): one named track per instrument, drums on channel 10 with their pitches mapped to
        General MIDI drum notes (muscriptor.drums.to_general_midi), 120 bpm, velocity 100."""
        from muscriptor.drums import to_general_midi

        midi = notes_to_multitrack_midi(notes, self.instrument_name, to_general_midi)
        buf = io.BytesIO()
        midi.save(file=buf)
        return buf.getvalue()

    def notes_to_events(
        self, notes: list[Note]
    ) -> Iterator[NoteStartEvent | NoteEndEvent]:
        """Notes -> NoteStartEvent / NoteEndEvent sorted by time (an end before a start at the same time; every start
        followed by its end). Drum pitches are as written by the model."""
        items = []
        for k, note in enumerate(
            sorted(notes, key=lambda n: (n.onset, n.is_drum, n.program, n.pitch))
        ):
            start = NoteStartEvent(
                pitch=note.pitch,
                start_time=note.onset,
                index=k,
                instrument=self.instrument_name(note),
            )
            items.append((note.onset, 1, k, start))
            items.append(
                (
                    max(note.offset, note.onset),
                    2 if note.offset <= note.onset else 0,
                    k,
                    NoteEndEvent(
                        end_time=max(note.offset, note.onset), start_event=start
                    ),
                )
            )
        items.sort(key=lambda x: (x[0], x[1], x[2]))
        for *_, ev in items:
            yield ev

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
            for name, gid in INSTRUMENT_GROUP_NAMES.items():
                if gid >= len(MT3_FULL_PLUS_GROUP_NAMES) and name != "drums":
                    self._inst_to_program.setdefault(name, v2tok.PREFIX_GROUPS[gid][0])
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
