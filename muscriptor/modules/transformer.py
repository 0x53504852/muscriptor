"""Transformer model with streaming support."""

import math
import typing as tp

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F

from muscriptor.modules.streaming import StreamingModule


class LayerNormF32(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_f32 = input.float()
        out_f32 = super().forward(x_f32)
        return out_f32.to(input.dtype)


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        use_var: bool = True,
        dtype: tp.Optional[torch.dtype] = None,
        device=None,
    ):
        super().__init__()
        self.eps = eps
        self.dtype = dtype
        self.use_var = use_var
        self.alpha = nn.Parameter(
            torch.full((1, 1, dim), 1.0, requires_grad=True, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor):
        x_dtype = x.dtype
        if self.dtype is not None:
            x = x.to(self.dtype)
        if self.use_var:
            var = self.eps + x.var(dim=2, keepdim=True)
        else:
            var = self.eps + torch.mean(x**2, dim=2, keepdim=True)
        y = (x * (self.alpha.to(var) * torch.rsqrt(var))).to(x_dtype)
        return y


class LayerScale(nn.Module):
    def __init__(
        self,
        channels: int,
        init: float = 1e-4,
        channel_last: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(
            torch.full(
                (channels,), init, requires_grad=True, device=device, dtype=dtype
            )
        )

    def forward(self, x: torch.Tensor):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x


def create_norm_fn(norm_type: str, dim: int, **kwargs) -> nn.Module:
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim, eps=1e-5, **kwargs)
    elif norm_type == "layer_norm_f32":
        kwargs.pop("dtype", None)
        return LayerNormF32(dim, eps=1e-8, **kwargs)
    elif norm_type == "rms_norm":
        return RMSNorm(dim, eps=1e-5, **kwargs)
    elif norm_type == "rms_norm_f32":
        kwargs.pop("dtype", None)
        return RMSNorm(dim, eps=1e-8, dtype=torch.float, **kwargs)
    elif norm_type == "real_rms_norm":
        return RMSNorm(dim, eps=1e-5, use_var=False, **kwargs)
    elif norm_type == "real_rms_norm_f32":
        kwargs.pop("dtype", None)
        return RMSNorm(dim, eps=1e-8, dtype=torch.float, use_var=False, **kwargs)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def create_sin_embedding(
    positions: torch.Tensor,
    dim: int,
    max_period: float = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full([], max_period, device=positions.device, dtype=dtype)
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


def set_attention_context(model: nn.Module, context: tp.Optional[int] = None) -> None:
    for module in model.modules():
        if isinstance(module, StreamingMultiheadAttention):
            if not module.cross_attention:
                module.context = context


class KVCache:
    """Efficient streaming KV cache."""

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        dim_per_head: int,
        context: tp.Optional[int] = None,
        growth: float = 1.2,
        initial_size: int = 100,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
        cache: tp.Optional[torch.Tensor] = None,
        current_end: int = 0,
    ):
        assert growth > 1
        self.growth = growth
        if context is not None:
            initial_size = 1 + int(growth * context)
        self.capacity = initial_size
        self.context = context
        self.current_end = current_end
        if cache is None:
            self.cache = torch.full(
                (2, batch_size, initial_size, num_heads, dim_per_head),
                float("NaN"),
                device=device,
                dtype=dtype,
            )
        else:
            self.cache = cache

    def clone(self) -> "KVCache":
        return KVCache(
            self.cache.shape[1],
            self.cache.shape[3],
            self.cache.shape[4],
            self.context,
            self.growth,
            self.capacity,
            self.cache.device,
            self.cache.dtype,
            self.cache.clone(),
            self.current_end,
        )

    @property
    def current_start(self) -> int:
        if self.context is None:
            return 0
        else:
            return max(self.current_end - self.context, 0)

    def complete(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert k.shape[1] == v.shape[1]
        required_capacity = self.current_end + k.shape[1]
        if required_capacity > self.capacity:
            if self.context is None:
                new_capacity = self.capacity
                while required_capacity > new_capacity:
                    new_capacity = int(math.ceil(new_capacity * self.growth))
                new_shape = list(self.cache.shape)
                new_shape[2] = new_capacity
                new_cache = torch.full(
                    tuple(new_shape),
                    float("NaN"),
                    device=self.cache.device,
                    dtype=self.cache.dtype,
                )
                new_cache[:, :, : self.current_end] = self.cache[
                    :, :, : self.current_end
                ]
                self.cache = new_cache
                self.capacity = new_capacity
            else:
                assert self.current_start > 0
                self.cache[:] = self.cache.roll(-self.current_start, dims=2)
                self.current_end -= self.current_start

        assert self.current_end + k.shape[1] <= self.capacity
        self.cache[0, :, self.current_end : self.current_end + k.shape[1]] = k
        self.cache[1, :, self.current_end : self.current_end + v.shape[1]] = v
        self.current_end += k.shape[1]
        valid = self.cache[:, :, self.current_start : self.current_end]
        return valid[0], valid[1]


class StreamingMultiheadAttention(StreamingModule):
    """Streaming causal multi-head attention with optional xformers backend."""

    _fsdp_final = True

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        causal: bool = False,
        context: tp.Optional[int] = None,
        rope=None,
        cross_attention: bool = False,
        weights_per_step: int = 0,
        weights_per_step_schedule: tp.Optional[tp.List[int]] = None,
        lr: float = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.embed_dim = embed_dim
        self.causal = causal
        self.context = context
        self.rope = rope
        self.cross_attention = cross_attention
        self.num_heads = num_heads
        self.lr = lr
        self.weights_per_step_schedule = weights_per_step_schedule
        self.weights_per_step = weights_per_step

        if cross_attention:
            assert not causal
            assert not context
            assert rope is None

        out_dim = 3 * embed_dim
        mult = 1
        if weights_per_step:
            if weights_per_step_schedule:
                assert len(weights_per_step_schedule) == weights_per_step
                mult = max(weights_per_step_schedule) + 1
            else:
                mult = weights_per_step
        in_proj = nn.Linear(embed_dim, mult * out_dim, bias=False, **factory_kwargs)
        self.in_proj_weight = in_proj.weight
        self.in_proj_bias = in_proj.bias
        self.out_proj = nn.Linear(
            embed_dim, mult * embed_dim, bias=False, **factory_kwargs
        )

    def make_optim_group(self) -> dict:
        group: dict[str, tp.Any] = {}
        if self.lr is not None:
            group["lr"] = self.lr
        if not group:
            return None
        group["params"] = list(self.parameters())
        return group

    def _complete_kv(self, k, v):
        if self.cross_attention:
            return k, v
        if self._is_streaming:
            if "kv_cache" not in self._streaming_state:
                initial_size = self.weights_per_step or 100
                self._streaming_state["kv_cache"] = KVCache(
                    k.shape[0],
                    k.shape[2],
                    k.shape[3],
                    self.context,
                    initial_size=initial_size,
                    device=k.device,
                    dtype=k.dtype,
                )
                self._streaming_state["offset"] = torch.zeros(1)
            kv_cache: KVCache = self._streaming_state["kv_cache"]
            self._streaming_state["offset"] += k.shape[1]
            # if k.shape[1] == 1:  # decode step only
            #     print(f"[muscriptor._complete_kv] k.dtype={k.dtype} k.shape={k.shape} cache_current_end={kv_cache.current_end} k[0,0,0,:4]={k[0,0,0,:4].tolist()} v[0,0,0,:4]={v[0,0,0,:4].tolist()}")
            k, v = kv_cache.complete(k, v)
            return k, v
        else:
            return k, v

    @property
    def _streaming_offset(self) -> int:
        if "offset" in self._streaming_state:
            return int(self._streaming_state["offset"].item())
        else:
            return 0

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        B, T, C = query.shape

        if self.cross_attention:
            dim = self.in_proj_weight.shape[0] // 3
            q = nn.functional.linear(query, self.in_proj_weight[:dim])
            if self._is_streaming and "k_cross" in self._streaming_state:
                k = self._streaming_state["k_cross"]
                v = self._streaming_state["v_cross"]
            else:
                k = nn.functional.linear(key, self.in_proj_weight[dim : 2 * dim])
                v = nn.functional.linear(value, self.in_proj_weight[2 * dim :])
                if self._is_streaming:
                    self._streaming_state["k_cross"] = k
                    self._streaming_state["v_cross"] = v
            q, k, v = [
                rearrange(x, "b t (h d) -> b t h d", h=self.num_heads)
                for x in [q, k, v]
            ]
        else:
            projected = nn.functional.linear(query, self.in_proj_weight)
            packed = rearrange(
                projected, "b t (p h d) -> b t p h d", p=3, h=self.num_heads
            )
            q, k, v = packed.unbind(dim=2)

        k, v = self._complete_kv(k, v)
        dtype = q.dtype

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        is_causal = self.causal and not self.cross_attention
        if is_causal:
            # Use an explicit bottom-right causal mask so that streaming decode steps
            # (T_q=1, T_k=cache_len) correctly attend to all past tokens.
            # PyTorch's is_causal=True uses top-left alignment and masks out all
            # cached tokens except position 0 when T_q < T_k.
            T_q, T_k = q_t.shape[2], k_t.shape[2]
            mask = torch.ones(T_q, T_k, dtype=torch.bool, device=query.device).tril(
                T_k - T_q
            )
            if self.context and self.context < T_k:
                mask &= torch.ones(
                    T_q, T_k, dtype=torch.bool, device=query.device
                ).triu(T_k - T_q - self.context + 1)
            attn_bias = torch.zeros(T_q, T_k, dtype=q.dtype, device=query.device)
            attn_bias.masked_fill_(~mask, float("-inf"))
            x = F.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=attn_bias, dropout_p=0.0
            )
        else:
            x = F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=False, dropout_p=0.0
            )
        x = x.transpose(1, 2).to(dtype)

        x = rearrange(x, "b t h d -> b t (h d)")
        x = self.out_proj(x)
        return x


class StreamingTransformerLayer(StreamingModule):
    """TransformerLayer with Streaming / Causal support."""

    _fsdp_final = True

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        causal: bool = False,
        context: tp.Optional[int] = None,
        cross_attention: bool = False,
        rope=None,
        norm: str = "layer_norm",
        lr_cross: float = None,
        layer_scale: tp.Optional[float] = None,
        gating: str = "none",
        weights_per_step: int = 0,
        weights_per_step_schedule: tp.Optional[tp.List[int]] = None,
        activation=F.gelu,
        skip_self_attn: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        attn_kwargs: tp.Dict[str, tp.Any] = {
            "embed_dim": d_model,
            "num_heads": num_heads,
        }
        if not skip_self_attn:
            self.self_attn = StreamingMultiheadAttention(
                causal=causal,
                context=context,
                rope=rope,
                weights_per_step=weights_per_step,
                weights_per_step_schedule=weights_per_step_schedule,
                **attn_kwargs,
                **factory_kwargs,
            )
            self.norm1 = create_norm_fn(norm, d_model, **factory_kwargs)
        self.norm2 = create_norm_fn(norm, d_model, **factory_kwargs)
        self.weights_per_step = weights_per_step
        self.weights_per_step_schedule = weights_per_step_schedule
        self.gating: tp.Optional[nn.Module] = None
        self.linear1: tp.Optional[nn.Module] = None
        self.linear2: tp.Optional[nn.Module] = None
        self.activation = activation
        self.skip_self_attn = skip_self_attn

        if gating == "none":
            self.linear1 = nn.Linear(
                d_model, dim_feedforward, bias=False, **factory_kwargs
            )
            self.linear2 = nn.Linear(
                dim_feedforward, d_model, bias=False, **factory_kwargs
            )
        else:
            # Gating variants (GLU, etc.) — kept for weight compatibility but not used by default
            try:
                from muscriptor.modules.gating import make_gating

                self.gating = make_gating(
                    gating, d_model, dim_feedforward, **factory_kwargs
                )
            except ImportError:
                raise ImportError(
                    f"Gating '{gating}' requires muscriptor.modules.gating, which is not available."
                )

        self.cross_attention: tp.Optional[StreamingMultiheadAttention] = None
        if cross_attention:
            self.cross_attention = StreamingMultiheadAttention(
                cross_attention=True, lr=lr_cross, **attn_kwargs, **factory_kwargs
            )
            self.norm_cross = nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs)

        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
            if cross_attention:
                self.layer_scale_cross = nn.Identity()
        else:
            self.layer_scale_1 = LayerScale(d_model, layer_scale, **factory_kwargs)
            self.layer_scale_2 = LayerScale(d_model, layer_scale, **factory_kwargs)
            if cross_attention:
                self.layer_scale_cross = LayerScale(
                    d_model, layer_scale, **factory_kwargs
                )

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x_orig = x
        x = self.norm2(x)
        if self.gating is None:
            assert self.linear1 is not None
            assert self.linear2 is not None
            update = self.linear2(self.activation(self.linear1(x)))
        else:
            update = self.gating(x)
        return x_orig + self.layer_scale_2(update)

    def _cross_attention_block(
        self, x: torch.Tensor, cross_attention_src: torch.Tensor
    ) -> torch.Tensor:
        assert self.cross_attention is not None
        x_orig = x
        x = self.norm_cross(x)
        update = self.cross_attention(x, cross_attention_src, cross_attention_src)
        return x_orig + self.layer_scale_cross(update)

    def _sa_block(self, x: torch.Tensor):
        if self.skip_self_attn:
            return x
        x_orig = x
        x = self.norm1(x)
        update = self.self_attn(x, x, x)
        return x_orig + self.layer_scale_1(update)

    def forward(
        self, x: torch.Tensor, cross_attention_src: tp.Optional[torch.Tensor] = None
    ):
        if self._is_streaming:
            if "offset" not in self._streaming_state:
                self._streaming_state["offset"] = torch.tensor(0)

        x = self._sa_block(x)
        if self.cross_attention is not None:
            assert cross_attention_src is not None
            x = self._cross_attention_block(x, cross_attention_src)
        else:
            assert cross_attention_src is None
        x = self._ff_block(x)
        if self._is_streaming:
            self._streaming_state["offset"] += x.shape[1]
        return x


class StreamingTransformer(StreamingModule):
    """Transformer with Streaming / Causal support."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        causal: bool = False,
        context: tp.Optional[int] = None,
        cross_attention: bool = False,
        positional_embedding: str = "sin",
        max_period: float = 10_000,
        positional_scale: float = 1.0,
        lr: tp.Optional[float] = None,
        weight_decay: tp.Optional[float] = None,
        betas: tp.Optional[tp.Tuple[float, float]] = None,
        input_cumsum: bool = False,
        layer_class: tp.Type[StreamingTransformerLayer] = StreamingTransformerLayer,
        checkpointing: str = "none",
        device=None,
        dtype=None,
        pos_embedding_start_at: int = 0,
        pos_embedding_time_factor: float = 1.0,
        rope_restart_at: tp.Optional[int] = None,
        rope_factor_after_restart: float = 1.0,
        dynamic_pos_restart: bool = False,
        pos_restart_extra_offset: int = 0,
        **kwargs,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.positional_embedding = positional_embedding
        self.max_period = max_period
        self.positional_scale = positional_scale
        self.weight_decay = weight_decay
        self.lr = lr
        self.betas = betas
        self.input_cumsum = input_cumsum
        self.pos_embedding_start_at = pos_embedding_start_at
        self.pos_embedding_time_factor = pos_embedding_time_factor
        self.dynamic_pos_restart = dynamic_pos_restart
        self.pos_restart_extra_offset = pos_restart_extra_offset
        assert positional_embedding in {"sin", "rope", "sin_rope", "none"}
        self.rope = None
        if positional_embedding in {"rope", "sin_rope"}:
            try:
                from muscriptor.modules.rope import RotaryEmbedding

                self.rope = RotaryEmbedding(
                    max_period=max_period,
                    restart_at=rope_restart_at,
                    factor_after_restart=rope_factor_after_restart,
                )
            except ImportError:
                raise ImportError(
                    "RoPE positional embedding requires muscriptor.modules.rope."
                )

        self.checkpointing = checkpointing
        assert checkpointing in {"none", "torch", "simple"}

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                layer_class(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    causal=causal,
                    context=context,
                    cross_attention=cross_attention,
                    rope=self.rope,
                    device=device,
                    dtype=dtype,
                    **kwargs,
                )
            )

    def _apply_layer(self, layer, *args, **kwargs) -> tp.Any:
        if self.checkpointing == "none":
            return layer(*args, **kwargs)
        elif self.checkpointing == "torch":
            from torch.utils.checkpoint import checkpoint as torch_checkpoint

            return torch_checkpoint(
                layer,
                *args,
                use_reentrant=False,
                determinism_check="none",
                preserve_rng_state=False,
                **kwargs,
            )
        else:
            raise ValueError(f"Checkpointing method {self.checkpointing} is unknown.")

    def forward(
        self,
        x: torch.Tensor,
        *args,
        prepend_length: int = 0,
        output_intermediates=False,
        **kwargs,
    ):
        B, T, C = x.shape

        if "offsets" in self._streaming_state:
            offsets = self._streaming_state["offsets"]
        else:
            offsets = torch.zeros(B, dtype=torch.long, device=x.device)
        if self.input_cumsum:
            x = torch.cumsum(x, dim=1)
            if "input_cumsum" in self._streaming_state:
                x += self._streaming_state["input_cumsum"]
            if self._is_streaming:
                self._streaming_state["input_cumsum"] = x[:, -1:]

        if self.dynamic_pos_restart:
            if "effective_restart" in self._streaming_state:
                effective_restart = int(
                    self._streaming_state["effective_restart"].item()
                )
            else:
                effective_restart = prepend_length + self.pos_restart_extra_offset
                if self._is_streaming:
                    self._streaming_state["effective_restart"] = torch.tensor(
                        effective_restart
                    )
        else:
            effective_restart = self.pos_embedding_start_at

        if self.positional_embedding in {"sin", "sin_rope"}:
            positions = torch.arange(T, device=x.device).view(1, -1, 1)
            positions = (
                positions + offsets.view(-1, 1, 1) - effective_restart
            ) * self.pos_embedding_time_factor
            pos_emb = create_sin_embedding(
                positions, C, max_period=self.max_period, dtype=x.dtype
            )
            # print(f"[muscriptor.transformer] pos_emb[0,0,:4]={pos_emb[0,0,:4].tolist()} positions[0,0,0]={positions[0,0,0].item()} dtype={pos_emb.dtype}")
            x = x + self.positional_scale * pos_emb * (positions >= 0).float()
        # print(f"[muscriptor.transformer] T={T} offset={offsets.tolist()} effective_restart={effective_restart} pos_emb_type={self.positional_embedding} x[0,0,:4]={x[0,0,:4].tolist()}")

        intermediates = []
        for i, layer in enumerate(self.layers):
            x = self._apply_layer(layer, x, *args, **kwargs)
            # if T == 1:
            # print(f"[muscriptor.layer_{i}] x[0,0,:4]={x[0,0,:4].tolist()}")
            if output_intermediates:
                intermediates.append(x)

        if self._is_streaming:
            self._streaming_state["offsets"] = offsets + T

        if output_intermediates:
            return x, intermediates
        return x

    def make_optim_group(self) -> tp.Dict[str, tp.Any]:
        group: tp.Dict[str, tp.Any] = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        if self.weight_decay is not None:
            group["weight_decay"] = self.weight_decay
        if self.betas is not None:
            group["betas"] = self.betas
        return group
