"""Minimal conditioners for muscriptor inference.

Contains only the classes needed to run the transcription model:
- ConditioningAttributes, ConditionType, WavCondition
- MelSpectrogramConditioner (audio → mel → linear projection)
- ClassConditioner (class index → embedding)
- ConditioningProvider, ConditionFuser
- ClassifierFreeGuidanceDropout, AttributeDropout (for CFG at inference)
"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import typing as tp

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from muscriptor.modules.streaming import StreamingModule
from muscriptor.modules.transformer import create_sin_embedding
from muscriptor.utils.sampling import length_to_mask

ConditionType = tp.Tuple[
    torch.Tensor, torch.Tensor
]  # (embedding [B, T, D], mask [B, T])


class WavCondition(tp.NamedTuple):
    wav: torch.Tensor
    length: torch.Tensor
    sample_rate: tp.List[int]
    path: tp.List[tp.Optional[str]] = []
    seek_time: tp.List[tp.Optional[float]] = []


@dataclass
class ConditioningAttributes:
    text: tp.Dict[str, tp.Optional[str]] = field(default_factory=dict)
    wav: tp.Dict[str, WavCondition] = field(default_factory=dict)
    joint_embed: tp.Dict[str, tp.Any] = field(default_factory=dict)
    symbolic: tp.Dict[str, tp.Any] = field(default_factory=dict)

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def text_attributes(self):
        return self.text.keys()

    @property
    def wav_attributes(self):
        return self.wav.keys()

    @property
    def joint_embed_attributes(self):
        return self.joint_embed.keys()

    @property
    def symbolic_attributes(self):
        return self.symbolic.keys()

    @property
    def attributes(self):
        return {
            "text": self.text_attributes,
            "wav": self.wav_attributes,
            "joint_embed": self.joint_embed_attributes,
            "symbolic": self.symbolic_attributes,
        }

    @classmethod
    def condition_types(cls) -> list:
        return ["text", "wav", "joint_embed", "symbolic"]


def nullify_condition(condition: ConditionType, dim: int = 1) -> ConditionType:
    assert dim != 0
    cond, mask = condition
    B = cond.shape[0]
    last_dim = cond.dim() - 1
    out = cond.transpose(dim, last_dim)
    out = 0.0 * out[..., :1]
    out = out.transpose(dim, last_dim)
    mask = torch.zeros((B, 1), device=out.device).int()
    return out, mask


def nullify_wav(cond: WavCondition) -> WavCondition:
    null_wav, _ = nullify_condition(
        (cond.wav, torch.zeros_like(cond.wav)), dim=cond.wav.dim() - 1
    )
    return WavCondition(
        wav=null_wav,
        length=torch.tensor([0] * cond.wav.shape[0], device=cond.wav.device),
        sample_rate=cond.sample_rate,
        path=[None] * cond.wav.shape[0],
        seek_time=[None] * cond.wav.shape[0],
    )


def dropout_condition_(
    sample: ConditioningAttributes, condition_type: str, condition: str
) -> None:
    if condition_type == "wav":
        wav_cond = sample.wav[condition]
        sample.wav[condition] = nullify_wav(wav_cond)
    else:
        sample.text[condition] = None


class BaseConditioner(nn.Module):
    def __init__(self, dim: int, output_dim: int):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        if self.output_dim > -1:
            self.output_proj = nn.Linear(dim, output_dim)

    def tokenize(self, *args, **kwargs) -> tp.Any:
        raise NotImplementedError()

    def forward(self, inputs: tp.Any) -> ConditionType:
        raise NotImplementedError()


class TextConditioner(BaseConditioner): ...


class WaveformConditioner(BaseConditioner):
    def __init__(
        self,
        dim: int,
        output_dim: int,
        device: tp.Union[torch.device, str],
        use_null_embedding: bool = False,
    ):
        super().__init__(dim, output_dim)
        self.device = device
        self._use_masking = True
        self.use_null_embedding = use_null_embedding
        final_dim = output_dim if output_dim > -1 else dim
        if use_null_embedding:
            self.null_embedding = nn.Parameter(torch.randn(final_dim) * 0.01)
        else:
            self.null_embedding = None

    def tokenize(self, x: WavCondition) -> WavCondition:
        wav, length, sample_rate, path, seek_time = x
        assert length is not None
        return WavCondition(
            wav.to(self.device), length.to(self.device), sample_rate, path, seek_time
        )

    def _get_wav_embedding(self, x: WavCondition) -> torch.Tensor:
        raise NotImplementedError()

    def _downsampling_factor(self):
        raise NotImplementedError()

    def forward(self, x: WavCondition) -> ConditionType:
        wav, lengths, *_ = x
        with torch.no_grad():
            embeds = self._get_wav_embedding(x)
        if hasattr(self, "output_proj"):
            embeds = embeds.to(self.output_proj.weight)
            embeds = self.output_proj(embeds)

        if lengths is not None and self._use_masking:
            lengths = lengths / self._downsampling_factor()
            mask = length_to_mask(lengths, max_len=embeds.shape[1]).int()
        else:
            mask = torch.ones_like(embeds[..., 0])
        mask_f = mask.float().unsqueeze(-1).to(embeds.device)
        if self.use_null_embedding:
            embeds = embeds * mask_f + self.null_embedding * (1 - mask_f)
        else:
            embeds = embeds * mask_f
        return embeds, mask


class MelSpectrogramConditioner(WaveformConditioner):
    """Log mel spectrogram conditioner. Projects mel bins to transformer dim."""

    def __init__(
        self,
        output_dim: int,
        device: tp.Union[torch.device, str],
        sample_rate: int,
        n_fft: int = 2048,
        frame_rate: int = 100,
        n_mel_bins: int = 512,
        normalize_audio: bool = False,
        log_scale: bool = True,
        eps: float = 1e-6,
        # unused args kept for config compatibility
        filter_drums: bool = False,
        concat_real_im: bool = False,
        separate_stems: bool = False,
        fine_frame_rate: int = None,
        use_null_embedding: bool = False,
    ):
        import torchaudio

        assert not filter_drums, "filter_drums not supported"
        assert not concat_real_im, "concat_real_im not supported"
        assert not separate_stems, "separate_stems not supported"

        self.fine_frame_rate_ratio = 1
        if fine_frame_rate is not None:
            assert fine_frame_rate % frame_rate == 0
            self.fine_frame_rate_ratio = fine_frame_rate // frame_rate

        dim = n_mel_bins * self.fine_frame_rate_ratio
        super().__init__(
            dim=dim,
            output_dim=output_dim,
            device=device,
            use_null_embedding=use_null_embedding,
        )
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.normalize_audio = normalize_audio
        self.log_scale = log_scale
        self.eps = eps

        if self.fine_frame_rate_ratio == 1:
            assert sample_rate % frame_rate == 0
            self.hop_length = sample_rate // frame_rate
        else:
            assert sample_rate % fine_frame_rate == 0
            self.hop_length = sample_rate // fine_frame_rate

        self.mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=self.hop_length,
            n_mels=n_mel_bins,
            power=1.0,
            center=True,
            pad_mode="reflect",
        ).to(device)

    def _downsampling_factor(self) -> int:
        return self.sample_rate // self.frame_rate

    def _get_wav_embedding(self, x: WavCondition) -> torch.Tensor:
        if x.wav.shape[-1] == 1:
            return torch.zeros(x.wav.shape[0], 1, self.dim, device=self.device)
        import time

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            wav = x.wav
            if self.normalize_audio:
                wav = wav / (wav.abs().max(dim=-1, keepdim=True).values + 1e-8)
            mel = self.mel_spec_transform(wav)
            mel = rearrange(mel, "b 1 d t -> b t d")
            if self.fine_frame_rate_ratio > 1:
                mel = rearrange(
                    mel[:, :-1], "b (t f) d -> b t (f d)", f=self.fine_frame_rate_ratio
                )
            if self.log_scale:
                mel = torch.log(mel + self.eps)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(
            f"[muscriptor] mel-spec ({wav.shape[0]} × {wav.shape[-1]} samples): "
            f"{time.perf_counter() - t0:.3f}s"
        )
        return mel


class ClassConditioner(TextConditioner):
    """Conditioner that embeds class indices (e.g., instrument group, dataset name)."""

    def __init__(
        self,
        num_classes: int,
        output_dim: int,
        device: tp.Union[torch.device, str] = "cpu",
    ):
        super().__init__(output_dim, -1)
        self.device = device
        self.embed = nn.Embedding(num_classes + 1, output_dim).to(device)
        self.pad_idx = 0

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> torch.Tensor:
        int_x = [list(map(int, s.split())) if s is not None else [-1] for s in x]
        max_len = max(len(xi) for xi in int_x)
        int_x = [xi + [-1] * (max_len - len(xi)) for xi in int_x]
        int_x = 1 + torch.LongTensor(int_x).to(self.device)
        return int_x

    def forward(self, inputs: torch.Tensor) -> ConditionType:
        embeds = self.embed(inputs + 1)
        mask = torch.ones_like(embeds[..., 0])
        return embeds, mask


class DropoutModule(nn.Module):
    def __init__(self, seed: int = 1234):
        super().__init__()
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)


class AttributeDropout(DropoutModule):
    """Per-attribute dropout (inactive at eval time by default)."""

    def __init__(
        self,
        dropouts: tp.Dict[str, float],
        active_on_eval: bool = False,
        seed: int = 1234,
    ):
        super().__init__(seed=seed)
        self.active_on_eval = active_on_eval
        self.dropouts = dropouts

    def forward(
        self, condition_attributes_batch: tp.List[ConditioningAttributes]
    ) -> tp.List[ConditioningAttributes]:
        if not self.training and not self.active_on_eval:
            return condition_attributes_batch

        condition_attributes_batch = deepcopy(condition_attributes_batch)
        for condition_attributes in condition_attributes_batch:
            for condition_type in ConditioningAttributes.condition_types():
                for condition in getattr(condition_attributes, condition_type):
                    if condition in self.dropouts:
                        if (
                            torch.rand(1, generator=self.rng).item()
                            < self.dropouts[condition]
                        ):
                            dropout_condition_(
                                condition_attributes, condition_type, condition
                            )
        return condition_attributes_batch


class ClassifierFreeGuidanceDropout(DropoutModule):
    """CFG dropout — drops all conditions with probability p (only active during training)."""

    def __init__(self, p: float, seed: int = 1234):
        super().__init__(seed=seed)
        self.p = p

    def forward(
        self,
        samples: tp.List[ConditioningAttributes],
        cond_types: tp.List[str] = ["wav", "text"],
        **kwargs,
    ) -> tp.List[ConditioningAttributes]:
        if not self.training:
            return samples

        drop = torch.rand(1, generator=self.rng).item() < self.p
        if not drop:
            return samples

        samples = deepcopy(samples)
        for condition_type in cond_types:
            for sample in samples:
                for condition in list(sample.attributes[condition_type]):
                    dropout_condition_(sample, condition_type, condition)
        return samples


def collate_wavs(
    samples: tp.List[ConditioningAttributes], wav_conditions: tp.List[str]
) -> tp.Dict[str, WavCondition]:
    """Collate wav conditions from a list of ConditioningAttributes."""
    wavs = defaultdict(list)
    lengths = defaultdict(list)
    sample_rates: tp.Dict[str, list] = defaultdict(list)
    paths: tp.Dict[str, list] = defaultdict(list)
    seek_times: tp.Dict[str, list] = defaultdict(list)
    out: tp.Dict[str, WavCondition] = {}

    for sample in samples:
        for attribute in wav_conditions:
            wav, length, sample_rate, path, seek_time = sample.wav[attribute]
            assert wav.dim() == 3
            B, K, T = wav.shape
            assert B == 1
            if K == 2:
                wav = wav.mean(1, keepdim=True)
            wavs[attribute].append(wav)
            lengths[attribute].append(length)
            sample_rates[attribute].extend(sample_rate)
            paths[attribute].extend(path)
            seek_times[attribute].extend(seek_time)

    for attribute in wav_conditions:
        # Stack along batch dim
        all_wavs = wavs[attribute]
        max_len = max(w.shape[-1] for w in all_wavs)
        padded = torch.cat(
            [F.pad(w, (0, max_len - w.shape[-1])) for w in all_wavs], dim=0
        )
        out[attribute] = WavCondition(
            padded,
            torch.cat(lengths[attribute]),
            sample_rates[attribute],
            paths[attribute],
            seek_times[attribute],
        )
    return out


class ConditioningProvider(nn.Module):
    """Runs all conditioners and returns a dict of condition tensors."""

    def __init__(
        self,
        conditioners: tp.Dict[str, BaseConditioner],
        device: tp.Union[torch.device, str] = "cpu",
    ):
        super().__init__()
        self.device = device
        self.conditioners = nn.ModuleDict(conditioners)

    @property
    def text_conditions(self):
        return [
            k for k, v in self.conditioners.items() if isinstance(v, TextConditioner)
        ]

    @property
    def wav_conditions(self):
        return [
            k
            for k, v in self.conditioners.items()
            if isinstance(v, WaveformConditioner)
        ]

    def tokenize(self, inputs: tp.List[ConditioningAttributes]) -> tp.Dict[str, tp.Any]:
        output = {}
        # Collate text conditions
        text_batch: tp.Dict[str, tp.List[tp.Optional[str]]] = defaultdict(list)
        for sample in inputs:
            for cond in self.text_conditions:
                text_batch[cond].append(sample.text.get(cond))
        for attr, batch in text_batch.items():
            output[attr] = self.conditioners[attr].tokenize(batch)

        # Collate wav conditions
        if self.wav_conditions:
            wav_batch = collate_wavs(inputs, self.wav_conditions)
            for attr, wav_cond in wav_batch.items():
                output[attr] = self.conditioners[attr].tokenize(wav_cond)

        return output

    def forward(self, tokenized: tp.Dict[str, tp.Any]) -> tp.Dict[str, ConditionType]:
        output = {}
        for attribute, inputs in tokenized.items():
            condition, mask = self.conditioners[attribute](inputs)
            output[attribute] = (condition, mask)
        return output


class ConditionFuser(StreamingModule):
    """Fuses conditions into the transformer input (supports prepend, sum, cross modes)."""

    FUSING_METHODS = [
        "sum",
        "prepend",
        "cross",
        "ignore",
        "input_interpolate",
        "streaming_sum",
    ]

    def __init__(
        self,
        fuse2cond: tp.Dict[str, tp.List[str]],
        cross_attention_pos_emb: bool = False,
        cross_attention_pos_emb_scale: float = 1.0,
        prepend_cond_pos_emb_scale: float = 0.0,
    ):
        super().__init__()
        assert all([k in self.FUSING_METHODS for k in fuse2cond.keys()])
        self.cross_attention_pos_emb = cross_attention_pos_emb
        self.cross_attention_pos_emb_scale = cross_attention_pos_emb_scale
        self.prepend_cond_pos_emb_scale = prepend_cond_pos_emb_scale
        self.fuse2cond: tp.Dict[str, tp.List[str]] = fuse2cond
        self.cond2fuse: tp.Dict[str, str] = {}
        for fuse_method, conditions in fuse2cond.items():
            for condition in conditions:
                self.cond2fuse[condition] = fuse_method

    def forward(
        self, input: torch.Tensor, conditions: tp.Dict[str, ConditionType]
    ) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor], int]:
        B, T, _ = input.shape

        if "offsets" in self._streaming_state:
            first_step = False
            offsets = self._streaming_state["offsets"]
        else:
            first_step = True
            offsets = torch.zeros(input.shape[0], dtype=torch.long, device=input.device)

        offset = int(offsets[0].item())
        cross_attention_output = None
        for cond_type, (cond, cond_mask) in conditions.items():
            op = self.cond2fuse[cond_type]
            if op == "sum":
                input = input + cond
            elif op == "streaming_sum":
                zero_frame = torch.zeros(B, 1, cond.shape[-1]).to(cond.device)
                if offset >= cond.shape[1]:
                    input = input + zero_frame
                else:
                    if offset + T > cond.shape[1]:
                        cond = torch.cat(
                            [cond, zero_frame.repeat(1, offset + T - cond.shape[1], 1)],
                            dim=1,
                        )
                    input = input + cond[:, offset : offset + T]
            elif op == "input_interpolate":
                cond = cond.permute(0, 2, 1)
                cond = F.interpolate(cond, size=input.shape[1])
                input = input + cond.permute(0, 2, 1)
            elif op == "prepend":
                if first_step:
                    if self.prepend_cond_pos_emb_scale:
                        positions = torch.arange(
                            cond.shape[1], device=cond.device
                        ).view(1, -1, 1)
                        pos_emb = create_sin_embedding(positions, cond.shape[-1])
                        cond = cond + self.prepend_cond_pos_emb_scale * pos_emb
                    input = torch.cat([cond, input], dim=1)
            elif op == "cross":
                if cross_attention_output is not None:
                    cross_attention_output = torch.cat(
                        [cross_attention_output, cond], dim=1
                    )
                else:
                    cross_attention_output = cond
            elif op == "ignore":
                continue
            else:
                raise ValueError(f"unknown op ({op})")

        prepend_length = input.shape[1] - T if first_step else 0

        if self.cross_attention_pos_emb and cross_attention_output is not None:
            positions = torch.arange(
                cross_attention_output.shape[1], device=cross_attention_output.device
            ).view(1, -1, 1)
            pos_emb = create_sin_embedding(positions, cross_attention_output.shape[-1])
            cross_attention_output = (
                cross_attention_output + self.cross_attention_pos_emb_scale * pos_emb
            )

        if self._is_streaming:
            self._streaming_state["offsets"] = offsets + T

        return input, cross_attention_output, prepend_length
