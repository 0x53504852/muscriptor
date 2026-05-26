"""TranscriptionModel: main user-facing entry point."""

import contextlib
import io
import math
import re
import time
import warnings
from collections.abc import Callable, Iterator
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
    print(f"[muscriptor] {label}: {dt:.2f}s")
    if store is not None:
        store.append((label, dt))


# Placeholder URL — fill in after HuggingFace upload.
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
        audio: str | Path | tuple[torch.Tensor, int],
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
        if batch_size is None:
            batch_size = 4 if self._device.type == "cuda" else 1

        timings: list[tuple[str, float]] = []
        t_total = time.perf_counter()

        if isinstance(audio, tuple):
            tensor, sample_rate = audio
            with _timed("load audio", timings):
                wav = self._load_wav(tensor, sample_rate)
        else:
            with _timed("load audio", timings):
                wav = self._load_wav(audio, None)

        total_samples = wav.shape[-1]
        total_duration = total_samples / _SAMPLE_RATE

        segment_samples = int(_SEGMENT_DURATION * _SAMPLE_RATE)
        num_chunks = math.ceil(total_samples / segment_samples)
        max_gen_len = 512
        eos_id = self._tokenizer.tokenizer.eos_id
        print(
            f"[muscriptor] audio: {total_duration:.1f}s → {num_chunks} chunk(s) of {_SEGMENT_DURATION}s"
        )

        with _timed("build conditions", timings):
            all_conditions: list[ConditioningAttributes] = []
            seek_times: list[float] = []
            for i in range(num_chunks):
                start = i * segment_samples
                chunk = wav[:, start : start + segment_samples]
                if chunk.shape[-1] < segment_samples:
                    chunk = F.pad(chunk, (0, segment_samples - chunk.shape[-1]))
                all_conditions.append(
                    self._build_conditions(chunk, instrument_group, dataset_name)[0]
                )
                seek_times.append(i * _SEGMENT_DURATION)

        builder = _StreamingEventBuilder(self._instrument_for_program)
        num_batches = math.ceil(num_chunks / batch_size)
        t_gen = time.perf_counter()

        for batch_idx, batch_start in enumerate(range(0, num_chunks, batch_size)):
            batch_conditions = all_conditions[batch_start : batch_start + batch_size]
            with _timed(
                f"generate batch {batch_idx + 1}/{num_batches} ({len(batch_conditions)} chunks)",
                timings,
            ):
                with torch.no_grad():
                    tokens = self._model.generate(
                        conditions=batch_conditions,
                        max_gen_len=max_gen_len,
                        use_sampling=use_sampling,
                        temp=temperature,
                        top_k=0,
                        top_p=0.0,
                        cfg_coef=cfg_coef,
                        early_stop_on_token=eos_id,
                    )  # [batch, n_q, T_gen]

            for j in range(tokens.shape[0]):
                chunk_index = batch_start + j
                seek_time = seek_times[chunk_index]
                next_seek_time = (
                    seek_times[chunk_index + 1]
                    if chunk_index + 1 < num_chunks
                    else None
                )

                seq = tokens[j]  # [n_q, T_gen]
                eos_mask = seq[0] == eos_id
                if eos_mask.any():
                    seq = seq[:, : eos_mask.int().argmax().item() + 1]
                else:
                    msg = (
                        f"chunk {chunk_index} (seek={seek_time:.1f}s) "
                        f"did not emit EOS within {max_gen_len} tokens"
                    )
                    if no_eos_is_ok:
                        warnings.warn(msg, RuntimeWarning, stacklevel=2)
                    else:
                        raise RuntimeError(
                            msg
                            + " (pass no_eos_is_ok=True / --no-eos-is-ok to downgrade to a warning)"
                        )

                token_list = seq[0].tolist()
                token_list = self._tokenizer._drop_instrument_section(token_list)
                note_events, tie_note_events, _, _ = self._tokenizer.tokenizer.decode(
                    token_list, start_time=seek_time
                )

                yield from builder.process_chunk(
                    note_events, tie_note_events, seek_time, next_seek_time
                )

        yield from builder.flush()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[muscriptor] generate total: {time.perf_counter() - t_gen:.2f}s")
        print(
            f"[muscriptor] transcribe total: {time.perf_counter() - t_total:.2f}s "
            f"({total_duration:.1f}s audio)"
        )

    # ------------------------------------------------------------------
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
        """Same as :meth:`transcribe` but returns a MIDI file as bytes.

        Thin wrapper that consumes the event generator, reassembles Notes,
        and serializes them.
        """
        notes: list[Note] = []
        open_notes: dict[int, Note] = {}
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
