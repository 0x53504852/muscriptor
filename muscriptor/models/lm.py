"""Language model for MIDI token generation.

Adapted from audiocraft/models/lm.py.
"""

import contextlib
import logging
import time
from collections.abc import Callable

import torch
from torch import nn

from muscriptor.modules.conditioners import (
    ConditioningProvider,
    ConditioningAttributes,
    ConditionType,
    nullify_all_conditions,
)
from muscriptor.modules.streaming import (
    ModelState,
    increment_steps,
    init_states,
)
from muscriptor.modules.transformer import StreamingTransformer
import muscriptor.utils.sampling as utils


logger = logging.getLogger(__name__)
ConditionTensors = dict[str, ConditionType]


# ---------------------------------------------------------------------------
# ScaledEmbedding  (used for token embeddings, keeps weight compatible with ckpt)
# ---------------------------------------------------------------------------


class ScaledEmbedding(nn.Embedding):
    """Embedding that maps zero_idx (a negative index) to a zero vector."""

    def __init__(self, *args, zero_idx: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        assert zero_idx < 0
        self.zero_idx = zero_idx

    def forward(self, input, *args, **kwargs):
        is_zero = input == self.zero_idx
        input = input.clamp(min=0)
        y = super().forward(input, *args, **kwargs)
        return torch.where(is_zero[..., None], torch.zeros_like(y), y)


# ---------------------------------------------------------------------------
# Delay helpers (kept for generality; no-ops when delays=[0])
# ---------------------------------------------------------------------------


def _undelay_sequence(
    delays: list[int],
    tensor: torch.Tensor,
    fill_value: int | float = float("NaN"),
) -> tuple[torch.Tensor, torch.Tensor]:
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
        dtype: torch.dtype | None = None,
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


class LMModel(nn.Module):
    """Causal transformer LM for MIDI token generation.

    Single-stream
    Supports classifier-free guidance at inference time.
    """

    def __init__(
        self,
        condition_provider: ConditioningProvider,
        delays: list[int] = [0],
        n_q: int = 1,
        card: int = 1024,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        cfg_coef: float = 1.0,
        autocast: TorchAutocast | None = None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.condition_provider = condition_provider
        self.n_q = n_q
        self.card = card
        self.dim = dim
        assert len(delays) > 0 and len(delays) <= n_q
        if len(delays) < n_q:
            delays = delays + [delays[-1]] * (n_q - len(delays))
        self.delays = delays
        self.cfg_coef = cfg_coef
        self.autocast = (
            autocast if autocast is not None else TorchAutocast(enabled=False)
        )

        self.emb = nn.ModuleList(
            [
                ScaledEmbedding(
                    self.card + 1,
                    dim,
                    device=device,
                    dtype=dtype,
                    zero_idx=self.zero_token_id,
                )
                for _ in range(n_q)
            ]
        )

        self.transformer = StreamingTransformer(
            d_model=dim,
            num_heads=num_heads,
            dim_feedforward=int(hidden_scale * dim),
            device=device,
            dtype=dtype,
            **kwargs,
        )
        self.out_norm = nn.LayerNorm(dim, eps=1e-5)
        self.linears = nn.ModuleList(
            [nn.Linear(dim, card, bias=False) for _ in range(n_q)]
        )
        # Tokenizer id of each output class: the identity, or tokenizer.v2.COMPACT_IDS for the compact-vocabulary
        # (MuScriptor V2) models. generate() works on classes; callers map them with `classes_to_tokens`.
        self.register_buffer(
            "token_of_class", torch.arange(card, device=device), persistent=False
        )
        # Run the condition provider under `autocast` too (the V2 training code does: the mel projection then runs in
        # bf16). Off for the older models, whose reference outputs were made without it.
        self.autocast_conditions = False
        # Print the per-batch timing lines.
        self.verbose = True

    def classes_to_tokens(self, classes: torch.Tensor) -> torch.Tensor:
        """Output classes (generate()'s result) -> tokenizer ids."""
        return self.token_of_class[classes]

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
        condition_tensors: ConditionTensors,
        first_step: bool = False,
        model_state: ModelState | None = None,
    ) -> torch.Tensor:  # [B, K, S, card]
        B, K, S = sequence.shape
        assert K == self.n_q

        input_sequence = sequence

        # Sum per-codebook token embeddings
        input_ = sum(
            self.emb[k](input_sequence[:, k]) for k in range(self.n_q)
        )  # [B, S, D]

        prepend_length = 0
        if first_step:
            for cond, _ in condition_tensors.values():
                input_ = torch.cat([cond, input_], dim=1)
            prepend_length = input_.shape[1] - input_sequence.shape[2]

        transformer_out = self.transformer(
            input_,
            prepend_length=prepend_length,
            model_state=model_state,
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
        model_state: ModelState,
        first_step: bool,
        use_sampling: bool = False,
        temp: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        cfg_coef: float | None = None,
    ) -> torch.Tensor:  # [B, K, 1]
        B = sequence.shape[0]
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef

        if cfg_coef == 1.0:
            logits = self(
                sequence,
                cfg_conditions,
                first_step=first_step,
                model_state=model_state,
            )
        else:
            # Batch-doubled CFG: [cond; uncond] in a single forward pass.
            doubled = torch.cat([sequence, sequence], dim=0)
            all_logits = self(
                doubled,
                cfg_conditions,
                first_step=first_step,
                model_state=model_state,
            )
            cond_logits, uncond_logits = all_logits.split(B, dim=0)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef

        # logits: [B, K, S, card] — take the last time step
        logits = logits[:, :, -1, :].float()  # [B, K, card]

        # Mask out tokens >= 1393 (reserved / out-of-vocabulary).
        logits[:, :, 1393:] = -torch.inf

        if use_sampling and temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            next_tokens = utils.sample_from_probs(probs, top_p=top_p, top_k=top_k)
        else:
            next_tokens = torch.argmax(logits, dim=-1, keepdim=True)  # [B, K, 1]

        return next_tokens  # [B, K, 1]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor | None = None,
        conditions: list[ConditioningAttributes] = [],
        num_samples: int | None = None,
        max_gen_len: int = 256,
        use_sampling: bool = True,
        temp: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        cfg_coef: float | None = None,
        callback: Callable[[int, int], None] | None = None,
        early_stop_on_token: int | None = None,
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
        log = print if self.verbose else (lambda *a, **k: None)
        cond_ctx = (
            self.autocast if self.autocast_conditions else contextlib.nullcontext()
        )
        if conditions:
            if cfg_coef == 1.0:
                prepared = self.condition_provider.tokenize(conditions)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _t = time.perf_counter()
                with cond_ctx:
                    cfg_conditions: ConditionTensors = self.condition_provider(prepared)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                log(
                    f"[muscriptor] encode conditions (total): {time.perf_counter() - _t:.3f}s"
                )
            else:
                null_conditions = nullify_all_conditions(conditions)
                all_conditions = conditions + null_conditions
                prepared = self.condition_provider.tokenize(all_conditions)
                log(
                    "[muscriptor] instrument_group tokens:",
                    prepared.get("instrument_group"),
                )
                log(
                    "[muscriptor] dataset_name tokens:    ",
                    prepared.get("dataset_name"),
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _t = time.perf_counter()
                with cond_ctx:
                    cfg_conditions = self.condition_provider(prepared)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                log(
                    f"[muscriptor] encode conditions (total): {time.perf_counter() - _t:.3f}s"
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

        prepend_length = sum(cond.shape[1] for cond, _ in cfg_conditions.values())
        cache_batch_size = num_samples * (1 if cfg_coef == 1.0 else 2)
        cache_seq_len = prepend_length + max_gen_len + max_delay
        model_state = init_states(
            self, batch_size=cache_batch_size, sequence_length=cache_seq_len
        )

        with self.autocast:
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

                first_iter = offset == start_offset
                input_ = (
                    gen_sequence[:, :, : offset + 1]
                    if first_iter
                    else gen_sequence[:, :, offset : offset + 1]
                )

                next_token = self._sample_next_token(
                    input_,
                    cfg_conditions,
                    model_state,
                    first_step=first_iter,
                    use_sampling=use_sampling,
                    temp=temp,
                    top_k=top_k,
                    top_p=top_p,
                    cfg_coef=cfg_coef,
                )  # [B, K, 1]

                input_T = input_.shape[-1]
                increment_steps(
                    self.transformer,
                    model_state,
                    increment=input_T + (prepend_length if first_iter else 0),
                )

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
