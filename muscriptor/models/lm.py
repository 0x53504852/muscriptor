"""Language model for MIDI token generation.

Adapted from audiocraft/models/lm.py.
"""

from functools import partial
import logging
import math
import typing as tp

import torch
from torch import nn

from muscriptor.modules.conditioners import (
    ClassifierFreeGuidanceDropout,
    ConditioningProvider,
    ConditionFuser,
    ConditioningAttributes,
    ConditionType,
)
from muscriptor.modules.streaming import StreamingModule, State
from muscriptor.modules.transformer import (
    StreamingTransformer,
    create_norm_fn,
    set_attention_context,
)
import muscriptor.utils.sampling as utils


logger = logging.getLogger(__name__)
ConditionTensors = tp.Dict[str, ConditionType]


# ---------------------------------------------------------------------------
# Weight init helpers
# ---------------------------------------------------------------------------


def _get_init_fn(method: str, input_dim: int, init_depth: tp.Optional[int] = None):
    std = 1 / math.sqrt(input_dim)
    if init_depth is not None:
        std = std / math.sqrt(2 * init_depth)
    if method == "gaussian":
        return partial(
            torch.nn.init.trunc_normal_, mean=0.0, std=std, a=-3 * std, b=3 * std
        )
    elif method == "uniform":
        bound = math.sqrt(3) * std
        return partial(torch.nn.init.uniform_, a=-bound, b=bound)
    else:
        raise ValueError(f"Unsupported init method: {method}")


def _init_layer(
    m: nn.Module,
    method: str,
    init_depth: tp.Optional[int] = None,
    zero_bias_init: bool = False,
):
    if isinstance(m, nn.Linear):
        init_fn = _get_init_fn(method, m.in_features, init_depth=init_depth)
        if m.weight.device.type == "cpu" and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)
        if zero_bias_init and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        init_fn = _get_init_fn(method, m.embedding_dim, init_depth=None)
        if m.weight.device.type == "cpu" and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)


# ---------------------------------------------------------------------------
# ScaledEmbedding  (used for token embeddings, keeps weight compatible with ckpt)
# ---------------------------------------------------------------------------


class ScaledEmbedding(nn.Embedding):
    """Embedding with optional LR scaling and learned zero-token suppression."""

    def __init__(
        self, *args, lr=None, norm: bool = False, zero_idx: int = -1, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.norm = create_norm_fn("layer_norm", self.embedding_dim) if norm else None
        assert zero_idx < 0
        self.zero_idx = zero_idx

    def forward(self, input, *args, **kwargs):
        is_zero = input == self.zero_idx
        input = input.clamp(min=0)
        y = super().forward(input, *args, **kwargs)
        if self.norm is not None:
            y = self.norm(y)
        y = torch.where(is_zero[..., None], torch.zeros_like(y), y)
        return y


# ---------------------------------------------------------------------------
# Delay helpers (kept for generality; no-ops when delays=[0])
# ---------------------------------------------------------------------------


def _delay_sequence(
    delays: tp.List[int], tensor: torch.Tensor, padding: torch.Tensor
) -> torch.Tensor:
    """padding: [B, K, 1]"""
    B, K, T = tensor.shape
    max_delay = max(delays)
    assert len(delays) == K
    if max_delay == 0:
        return tensor
    outs = []
    for k, delay in enumerate(delays):
        delay_right = max_delay - delay
        if delay_right > 0:
            line = torch.cat(
                [
                    padding[:, k].expand(-1, delay),
                    tensor[:, k],
                    padding[:, k].expand(-1, delay_right),
                ],
                dim=1,
            )
        else:
            line = torch.cat([padding[:, k].expand(-1, delay), tensor[:, k]], dim=1)
        outs.append(line)
    return torch.stack(outs, dim=1)


def _undelay_sequence(
    delays: tp.List[int],
    tensor: torch.Tensor,
    fill_value: tp.Union[int, float] = float("NaN"),
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    B, K, T, *_ = tensor.shape
    max_delay = max(delays)
    assert len(delays) == K
    mask = torch.ones(B, K, T, dtype=torch.bool, device=tensor.device)
    if all(d == 0 for d in delays):
        return tensor, mask
    outs = []
    for k, delay in enumerate(delays):
        delay_right = max_delay - delay
        line = (
            tensor[:, k, delay:-delay_right]
            if delay_right > 0
            else tensor[:, k, delay:]
        )
        outs.append(line)
    outs = torch.stack(outs, dim=1)
    mask = torch.ones(B, K, outs.shape[2], dtype=torch.bool, device=tensor.device)
    return outs, mask


# ---------------------------------------------------------------------------
# TorchAutocast
# ---------------------------------------------------------------------------


class TorchAutocast:
    """Minimal autocast context manager (matches the audiocraft interface)."""

    def __init__(
        self,
        enabled: bool = False,
        device_type: str = "cuda",
        dtype: tp.Optional[torch.dtype] = None,
    ):
        self.enabled = enabled
        self.device_type = device_type
        self.dtype = dtype
        self._ctx = None

    def __enter__(self):
        if self.enabled:
            self._ctx = torch.autocast(device_type=self.device_type, dtype=self.dtype)
            self._ctx.__enter__()
        return self

    def __exit__(self, *args):
        if self.enabled and self._ctx is not None:
            self._ctx.__exit__(*args)


# ---------------------------------------------------------------------------
# LMModel
# ---------------------------------------------------------------------------


class LMModel(StreamingModule):
    """Causal transformer LM for MIDI token generation.

    Single-stream
    Supports classifier-free guidance at inference time.
    """

    def __init__(
        self,
        condition_provider: ConditioningProvider,
        fuser: ConditionFuser,
        delays: tp.List[int] = [0],
        n_q: int = 1,
        card: int = 1024,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        norm: str = "layer_norm",
        norm_emb: bool = False,
        emb_lr: tp.Optional[float] = None,
        bias_proj: bool = False,
        weight_init: tp.Optional[str] = None,
        depthwise_init: tp.Optional[str] = None,
        zero_bias_init: bool = False,
        cfg_dropout: float = 0.0,
        cfg_coef: float = 1.0,
        two_step_cfg: bool = False,
        autocast: tp.Optional[TorchAutocast] = None,
        context: tp.Optional[int] = None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.condition_provider = condition_provider
        self.fuser = fuser
        self.n_q = n_q
        self.card = card
        self.dim = dim
        assert len(delays) > 0 and len(delays) <= n_q
        if len(delays) < n_q:
            delays = delays + [delays[-1]] * (n_q - len(delays))
        self.delays = delays
        self.cfg_coef = cfg_coef
        self.cfg_dropout = ClassifierFreeGuidanceDropout(p=cfg_dropout)
        self.two_step_cfg = two_step_cfg
        self.context = context
        self.autocast = (
            autocast if autocast is not None else TorchAutocast(enabled=False)
        )

        EmbeddingFactory = partial(
            ScaledEmbedding,
            norm=norm_emb,
            device=device,
            dtype=dtype,
            lr=emb_lr,
            zero_idx=self.zero_token_id,
        )
        self.emb = nn.ModuleList(
            [EmbeddingFactory(self.card + 1, dim) for _ in range(n_q)]
        )

        kwargs.pop("pattern_provider", None)
        kwargs["context"] = context
        self.transformer = StreamingTransformer(
            d_model=dim,
            num_heads=num_heads,
            dim_feedforward=int(hidden_scale * dim),
            norm=norm,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        self.out_norm = create_norm_fn(norm, dim)
        self.linears = nn.ModuleList(
            [nn.Linear(dim, card, bias=bias_proj) for _ in range(n_q)]
        )

        self._init_weights(weight_init, depthwise_init, zero_bias_init)
        self.__dict__["_fsdp"] = None

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------

    def _init_weights(self, weight_init, depthwise_init, zero_bias_init):
        assert depthwise_init is None or depthwise_init in ("current", "global")
        assert depthwise_init is None or weight_init is not None
        assert not zero_bias_init or weight_init is not None
        if weight_init is None:
            return
        for emb_layer in self.emb:
            _init_layer(emb_layer, method=weight_init, zero_bias_init=zero_bias_init)
        for layer_idx, tr_layer in enumerate(self.transformer.layers):
            depth = (
                layer_idx + 1
                if depthwise_init == "current"
                else len(self.transformer.layers)
                if depthwise_init == "global"
                else None
            )
            tr_layer.apply(
                partial(
                    _init_layer,
                    method=weight_init,
                    init_depth=depth,
                    zero_bias_init=zero_bias_init,
                )
            )
        for linear in self.linears:
            _init_layer(linear, method=weight_init, zero_bias_init=zero_bias_init)

    # ------------------------------------------------------------------
    # Token ID properties
    # ------------------------------------------------------------------

    @property
    def initial_token_id(self) -> int:
        return self.card

    @property
    def zero_token_id(self) -> int:
        return -1

    @property
    def ungenerated_token_id(self) -> int:
        return -2

    @property
    def num_codebooks(self) -> int:
        return self.n_q

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sequence: torch.Tensor,  # [B, K, S]
        conditions: tp.List[ConditioningAttributes],
        condition_tensors: tp.Optional[ConditionTensors] = None,
    ) -> torch.Tensor:  # [B, K, S, card]
        B, K, S = sequence.shape
        assert K == self.n_q

        input_sequence = sequence if self._is_streaming else sequence[:, :, :-1]

        # Sum per-codebook token embeddings
        input_ = sum(
            self.emb[k](input_sequence[:, k]) for k in range(self.n_q)
        )  # [B, S, D]

        if condition_tensors is None:
            assert not self._is_streaming
            conditions = self.cfg_dropout(conditions)
            conditions = self.att_dropout(conditions)
            prepared = self.condition_provider.tokenize(conditions)
            condition_tensors = self.condition_provider(prepared)
        else:
            assert not conditions

        input_, cross_attention_input, prepend_length = self.fuser(
            input_, condition_tensors
        )
        # print("INPUT", input_.shape, input_)
        transformer_out = self.transformer(
            input_,
            cross_attention_src=cross_attention_input,
            prepend_length=prepend_length,
        )
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)

        # Remove prepended conditioning tokens
        if prepend_length > 0:
            transformer_out = transformer_out[:, -input_sequence.shape[2] :]

        logits = torch.stack(
            [self.linears[k](transformer_out) for k in range(self.n_q)], dim=1
        )
        # print("LOGITS", logits.shape, logits)
        return logits  # [B, K, S, card]

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_next_token(
        self,
        sequence: torch.Tensor,
        cfg_conditions: ConditionTensors,
        unconditional_state: State,
        use_sampling: bool = False,
        temp: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        cfg_coef: tp.Optional[float] = None,
    ) -> torch.Tensor:  # [B, K, 1]
        B = sequence.shape[0]
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef

        if cfg_coef == 1.0:
            logits = self(sequence, conditions=[], condition_tensors=cfg_conditions)
        else:
            # Batch-doubled CFG: [cond; uncond] in a single forward pass.
            doubled = torch.cat([sequence, sequence], dim=0)
            all_logits = self(doubled, conditions=[], condition_tensors=cfg_conditions)
            cond_logits, uncond_logits = all_logits.split(B, dim=0)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef

        # logits: [B, K, S, card] — take the last time step
        logits = logits[:, :, -1, :].float()  # [B, K, card]

        # Mask out tokens >= 1393 (reserved / out-of-vocabulary).
        logits[:, :, 1393:] = -torch.inf

        if use_sampling and temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_tokens = utils.sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_tokens = utils.sample_top_k(probs, k=top_k)
            else:
                next_tokens = utils.multinomial(probs, num_samples=1)
        else:
            next_tokens = torch.argmax(logits, dim=-1, keepdim=True)  # [B, K, 1]

        return next_tokens  # [B, K, 1]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: tp.Optional[torch.Tensor] = None,
        conditions: tp.List[ConditioningAttributes] = [],
        num_samples: tp.Optional[int] = None,
        max_gen_len: int = 256,
        use_sampling: bool = True,
        temp: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        cfg_coef: tp.Optional[float] = None,
        callback: tp.Optional[tp.Callable[[int, int], None]] = None,
        early_stop_on_token: tp.Optional[int] = None,
    ) -> torch.Tensor:
        assert not self.training
        device = self.emb[0].weight.device

        if num_samples is None:
            num_samples = (
                len(conditions)
                if conditions
                else (prompt.shape[0] if prompt is not None else 1)
            )

        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef

        # Build condition tensors (with null conditions appended for CFG)
        import time as _time

        if conditions:
            if cfg_coef == 1.0:
                prepared = self.condition_provider.tokenize(conditions)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _t = _time.perf_counter()
                cfg_conditions: ConditionTensors = self.condition_provider(prepared)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(
                    f"[muscriptor] encode conditions (total): {_time.perf_counter() - _t:.3f}s"
                )
            else:
                null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
                all_conditions = conditions + null_conditions
                prepared = self.condition_provider.tokenize(all_conditions)
                print(
                    "[muscriptor] instrument_group tokens:",
                    prepared.get("instrument_group"),
                )
                print(
                    "[muscriptor] dataset_name tokens:    ",
                    prepared.get("dataset_name"),
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _t = _time.perf_counter()
                cfg_conditions = self.condition_provider(prepared)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(
                    f"[muscriptor] encode conditions (total): {_time.perf_counter() - _t:.3f}s"
                )
        else:
            cfg_conditions = {}

        # Initialise generation buffer
        max_delay = max(self.delays)
        ungenerated = self.ungenerated_token_id
        special = torch.full(
            [1, 1, 1], self.initial_token_id, device=device, dtype=torch.long
        )
        initial = special.expand(num_samples, self.n_q, 1)

        gen_sequence = torch.full(
            (num_samples, self.n_q, max_gen_len + max_delay + 1),
            ungenerated,
            device=device,
            dtype=torch.long,
        )

        for cb in range(self.n_q):
            gen_sequence[:, cb, : self.delays[cb] + 1] = initial[:, cb]

        start_offset = 0
        if prompt is not None:
            PT = prompt.shape[-1]
            for cb in range(self.n_q):
                gen_sequence[:, cb, self.delays[cb] + 1 : self.delays[cb] + 1 + PT] = (
                    prompt[:, cb]
                )
            ungenerated_steps = (gen_sequence == ungenerated).nonzero()[:, 2]
            start_offset = max(0, int(ungenerated_steps.amin()) - 1)

        if self.context is not None:
            set_attention_context(
                self.transformer, min(self.context, gen_sequence.shape[-1])
            )

        with self.streaming(), self.autocast:
            unconditional_state: State = {}
            for offset in range(start_offset, max_gen_len + max_delay):
                # Early stop when all samples have produced the stop token
                if early_stop_on_token is not None:
                    done = (
                        (gen_sequence == early_stop_on_token)
                        .all(dim=1)
                        .any(dim=1)
                        .all()
                    )
                    if done:
                        gen_sequence = torch.where(
                            gen_sequence == ungenerated,
                            torch.full_like(gen_sequence, early_stop_on_token),
                            gen_sequence,
                        )
                        break

                input_ = (
                    gen_sequence[:, :, : offset + 1]
                    if offset == start_offset
                    else gen_sequence[:, :, offset : offset + 1]
                )

                next_token = self._sample_next_token(
                    input_,
                    cfg_conditions,
                    unconditional_state,
                    use_sampling=use_sampling,
                    temp=temp,
                    top_k=top_k,
                    top_p=top_p,
                    cfg_coef=cfg_coef,
                )  # [B, K, 1]

                this_gen_step = gen_sequence[:, :, offset + 1]
                next_token = torch.where(
                    this_gen_step == ungenerated, next_token[:, :, 0], this_gen_step
                )
                gen_sequence[:, :, offset + 1] = next_token

                if callback is not None:
                    callback(
                        1 + offset - start_offset,
                        max_gen_len + max_delay - start_offset,
                    )

        output, mask = _undelay_sequence(
            self.delays, gen_sequence[:, :, 1:], fill_value=ungenerated
        )
        assert mask[:, :, :max_gen_len].all()
        output = output[:, :, :max_gen_len]
        assert output.shape == (num_samples, self.n_q, max_gen_len)
        assert not (output == ungenerated).any()
        return output
