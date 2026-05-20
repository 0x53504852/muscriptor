"""Tests for muscriptor/modules/transformer.py — CPU only, small tensors."""
import pytest
import torch

from muscriptor.modules.transformer import (
    create_norm_fn,
    create_sin_embedding,
    StreamingTransformer,
    LayerNormF32,
    RMSNorm,
    KVCache,
)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("norm_type", [
    "layer_norm",
    "layer_norm_f32",
    "rms_norm",
    "rms_norm_f32",
])
def test_create_norm_fn_forward(norm_type):
    dim = 16
    norm = create_norm_fn(norm_type, dim)
    x = torch.randn(2, 5, dim)
    out = norm(x)
    assert out.shape == x.shape


def test_layer_norm_f32_output_dtype():
    norm = LayerNormF32(8)
    x = torch.randn(2, 8).to(torch.float16)
    out = norm(x.float())
    assert out.dtype == torch.float32


def test_rms_norm_output_shape():
    norm = RMSNorm(12)
    x = torch.randn(3, 7, 12)
    assert norm(x).shape == x.shape


# ---------------------------------------------------------------------------
# Sinusoidal embeddings
# ---------------------------------------------------------------------------

def test_create_sin_embedding_shape():
    positions = torch.arange(10).float().view(1, 10, 1)  # [B, T, 1]
    emb = create_sin_embedding(positions, dim=16)
    assert emb.shape == (1, 10, 16)


def test_create_sin_embedding_dim_even():
    positions = torch.arange(5).float().view(1, 5, 1)
    emb = create_sin_embedding(positions, dim=8)
    assert emb.shape == (1, 5, 8)


def test_create_sin_embedding_different_positions():
    pos1 = torch.tensor([[[0.0]]])   # [1, 1, 1]
    pos2 = torch.tensor([[[1.0]]])
    e1 = create_sin_embedding(pos1, dim=8)
    e2 = create_sin_embedding(pos2, dim=8)
    assert not torch.allclose(e1, e2)


# ---------------------------------------------------------------------------
# KVCache
# ---------------------------------------------------------------------------

def test_kvcache_complete_shape():
    cache = KVCache(
        batch_size=2, num_heads=2, dim_per_head=4,
        device=torch.device("cpu"), dtype=torch.float32
    )
    k = torch.randn(2, 2, 1, 4)
    v = torch.randn(2, 2, 1, 4)
    k_out, v_out = cache.complete(k, v)
    assert k_out.shape[-1] == 4
    assert k_out.shape[0] == 2


# ---------------------------------------------------------------------------
# StreamingTransformer forward
# ---------------------------------------------------------------------------

def _make_transformer(**kwargs):
    defaults = dict(
        d_model=32,
        num_heads=2,
        num_layers=2,
        dim_feedforward=64,
        causal=True,
    )
    defaults.update(kwargs)
    return StreamingTransformer(**defaults)


def test_streaming_transformer_output_shape():
    model = _make_transformer()
    model.eval()
    x = torch.randn(2, 10, 32)  # [B, T, D]
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10, 32)


def test_streaming_transformer_non_causal():
    model = _make_transformer(causal=False)
    model.eval()
    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 8, 32)


def test_streaming_transformer_with_context():
    model = _make_transformer(context=5)
    model.eval()
    x = torch.randn(1, 12, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 12, 32)


def test_streaming_transformer_streaming_mode():
    """Feed tokens one at a time in streaming mode; check output shape is correct."""
    model = _make_transformer()
    model.eval()
    x = torch.randn(1, 6, 32)

    with torch.no_grad():
        with model.streaming():
            streaming_outs = []
            for t in range(x.shape[1]):
                out_t = model(x[:, t:t+1, :])
                streaming_outs.append(out_t)
        streaming_out = torch.cat(streaming_outs, dim=1)

    assert streaming_out.shape == (1, 6, 32)


def test_streaming_transformer_reset():
    model = _make_transformer()
    model.eval()
    with model.streaming():
        x = torch.randn(1, 3, 32)
        with torch.no_grad():
            model(x)
        model.reset_streaming()
        # After reset, streaming from scratch should still work
        with torch.no_grad():
            out = model(x)
    assert out.shape == (1, 3, 32)
