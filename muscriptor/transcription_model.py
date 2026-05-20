"""TranscriptionModel: main user-facing entry point."""

import dataclasses
import math
import re
import typing as tp
from pathlib import Path

import torch
import torch.nn.functional as F

from muscriptor.models.lm import LMModel, TorchAutocast
from muscriptor.modules.conditioners import (
    MelSpectrogramConditioner, ClassConditioner,
    ConditioningProvider, ConditionFuser,
    ConditioningAttributes, WavCondition,
)
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.tokenizer.notes import Note
from muscriptor.utils.audio import load_audio
from muscriptor.utils.midi import save_midi

# Placeholder URL — fill in after HuggingFace upload.
_HF_URL = "https://huggingface.co/kyutai/muscriptor/resolve/main/muscriptor_weights.safetensors"

_SAMPLE_RATE = 16000
# Must match the segment duration used during training / evaluation.
_SEGMENT_DURATION = 5.0


@dataclasses.dataclass
class _ModelConfig:
    dim: int
    num_heads: int
    num_layers: int
    card: int


# Per-checkpoint configs extracted from .hydra/config.yaml → transformer_lm.
_CONFIGS: tp.Dict[str, _ModelConfig] = {
    "01684fbb": _ModelConfig(dim=1536, num_heads=24, num_layers=48, card=1395),
    "0ac4ce03": _ModelConfig(dim=768,  num_heads=12, num_layers=14, card=1393),
    "8f59580c": _ModelConfig(dim=1024, num_heads=16, num_layers=24, card=1395),
    "e84904c4": _ModelConfig(dim=1536, num_heads=24, num_layers=48, card=1395),
}

_DEFAULT_CONFIG = _CONFIGS["01684fbb"]


def _config_for_weights(weights_path: Path) -> _ModelConfig:
    """Infer model config from checkpoint hash embedded in the filename."""
    m = re.search(r'_([0-9a-f]{8})_', weights_path.name)
    if m and m.group(1) in _CONFIGS:
        return _CONFIGS[m.group(1)]
    return _DEFAULT_CONFIG


def _build_model(device: torch.device, cfg: _ModelConfig = _DEFAULT_CONFIG) -> LMModel:
    mel_cond = MelSpectrogramConditioner(
        output_dim=cfg.dim, device=device,
        sample_rate=_SAMPLE_RATE, n_fft=2048, frame_rate=100,
        n_mel_bins=512, log_scale=True, eps=1e-6, normalize_audio=False)
    inst_cond = ClassConditioner(num_classes=1000, output_dim=cfg.dim, device=device)
    ds_cond = ClassConditioner(num_classes=4, output_dim=cfg.dim, device=device)

    condition_provider = ConditioningProvider(
        conditioners={"self_wav": mel_cond, "instrument_group": inst_cond, "dataset_name": ds_cond},
        device=device)

    fuser = ConditionFuser(fuse2cond={
        "prepend": ["self_wav", "instrument_group", "dataset_name"],
        "cross": [], "sum": [], "input_interpolate": [], "streaming_sum": [],
    })

    autocast = None
    if device.type == 'cuda':
        autocast = TorchAutocast(enabled=True, device_type='cuda', dtype=torch.float16)

    model = LMModel(
        condition_provider=condition_provider,
        fuser=fuser,
        delays=[0],
        n_q=1,
        card=cfg.card,
        dim=cfg.dim,
        num_heads=cfg.num_heads,
        hidden_scale=4,
        norm="layer_norm",
        norm_emb=False,
        weight_init="gaussian",
        depthwise_init="current",
        zero_bias_init=True,
        cfg_dropout=0.0,
        cfg_coef=1.0,
        autocast=autocast,
        # StreamingTransformer kwargs (forwarded via **kwargs)
        num_layers=cfg.num_layers,
        causal=True,
        positional_embedding="sin",
        max_period=10000,
        cross_attention=False,
        context=None,
        layer_scale=None,
        checkpointing="none",
        device=device,
    )
    return model


class TranscriptionModel:
    """Transcribes audio to MIDI using the muscriptor model.

    Example::

        model = TranscriptionModel.load_model()
        notes = model.transcribe("audio.wav", return_notes=True)
        model.transcribe("audio.wav", output_midi="out.mid")
    """

    def __init__(self, model: LMModel, tokenizer: MT3Tokenizer, device: torch.device):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    @classmethod
    def load_model(cls,
                   weights_path: tp.Optional[tp.Union[str, Path]] = None,
                   device: tp.Optional[tp.Union[str, torch.device]] = None
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
            from muscriptor.utils.download import download_if_necessary
            weights_path = download_if_necessary(_HF_URL)

        weights_path = Path(weights_path)
        model = _build_model(device, _config_for_weights(weights_path))
        model.eval()

        from safetensors.torch import load_file
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
        audio: tp.Union[str, Path, torch.Tensor],
        sample_rate: tp.Optional[int] = None,
        output_midi: tp.Optional[tp.Union[str, Path]] = None,
        return_notes: bool = False,
        use_sampling: bool = False,
        temperature: float = 1.0,
        cfg_coef: float = 2.0,
        instrument_group: tp.Optional[str] = None,
        dataset_name: tp.Optional[str] = None,
    ) -> tp.Optional[tp.List[Note]]:
        """Transcribe audio to MIDI / notes.

        Audio longer than _SEGMENT_DURATION seconds is split into non-overlapping
        5-second chunks (matching the evaluation protocol), each decoded
        independently, then merged via decode_segmented_track.

        Args:
            audio: Path to an audio file, or a float tensor of shape [T] or
                [1, T] at `sample_rate` Hz.
            sample_rate: Required when `audio` is a tensor; ignored otherwise.
            output_midi: If given, saves transcription as a MIDI file here.
            return_notes: If True, return a list of :class:`Note` objects.
            use_sampling: Use temperature sampling instead of greedy decoding.
            temperature: Sampling temperature (only used when use_sampling=True).
            cfg_coef: Classifier-free guidance coefficient.
            instrument_group: Optional instrument group conditioning string.
            dataset_name: Optional dataset name conditioning string.

        Returns:
            List of :class:`Note` objects when return_notes is True, else None.
        """
        wav = self._load_wav(audio, sample_rate)
        total_samples = wav.shape[-1]
        total_duration = total_samples / _SAMPLE_RATE

        segment_samples = int(_SEGMENT_DURATION * _SAMPLE_RATE)
        num_chunks = math.ceil(total_samples / segment_samples)
        max_gen_len = self._tokenizer.get_generation_length(_SEGMENT_DURATION)
        eos_id = self._tokenizer.tokenizer.eos_id

        # Build all chunk conditions upfront so we can batch them.
        all_conditions: tp.List[ConditioningAttributes] = []
        seek_times: tp.List[float] = []
        for i in range(num_chunks):
            start = i * segment_samples
            chunk = wav[:, start:start + segment_samples]
            if chunk.shape[-1] < segment_samples:
                chunk = F.pad(chunk, (0, segment_samples - chunk.shape[-1]))
            all_conditions.append(self._build_conditions(chunk, instrument_group, dataset_name)[0])
            seek_times.append(i * _SEGMENT_DURATION)

        # Generate in batches of 16 (matching the evaluation solver).
        all_tokens: tp.List[torch.Tensor] = []
        for i in range(0, len(all_conditions), 16):
            batch_conditions = all_conditions[i:i + 16]
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
            
            # we cut off at the first EOS token, which is included in the output since it signals the end of the sequence
            for j in range(tokens.shape[0]):
                seq = tokens[j]  # [n_q, T_gen]
                eos_mask = seq[0] == eos_id
                if eos_mask.any():
                    seq = seq[:, :eos_mask.int().argmax().item() + 1]
                all_tokens.append(seq)
        # return all_tokens

        notes_dict, _ = self._tokenizer.decode_segmented_track(
            all_tokens, seek_times, total_duration, _SEGMENT_DURATION,
        )
        notes: tp.List[Note] = notes_dict["notes"]

        if output_midi is not None:
            save_midi(notes, output_midi)

        if return_notes:
            return notes
        return None

    # ------------------------------------------------------------------
    def _load_wav(self,
                  audio: tp.Union[str, Path, torch.Tensor],
                  sample_rate: tp.Optional[int]) -> torch.Tensor:
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
                import torchaudio
                wav = torchaudio.transforms.Resample(sample_rate, _SAMPLE_RATE)(wav)
        return wav.to(self._device)

    def _build_conditions(self, wav: torch.Tensor, instrument_group: tp.Optional[str] = None, 
                          dataset_name: tp.Optional[str] = None) -> tp.List[ConditioningAttributes]:
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
        return [ConditioningAttributes(
            wav={"self_wav": wav_cond},
            text={"instrument_group": instrument_group, "dataset_name": dataset_name},
        )]
