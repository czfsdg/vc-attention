import math

import pytest
import torch
from vc_attention.core import Config, LayoutCache, attention, expcast_codes, group_values


def dense(q, k, v):
    return torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), -1) @ v


def tensors(seed=3, nq=19, nk=35, heads=2, dim=16):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(1, heads, n, dim, generator=gen) for n in (nq, nk, nk)]


def test_expcast_encoding_and_bit_view():
    # Independently specified IEEE E4M3 codes, including the paper's worked example.
    u = torch.tensor([0.0, -1.0, -1.6, -8.0, -14.0, -100.0, -float("inf")])
    codes = expcast_codes(u / math.log2(math.e))
    assert codes.tolist() == [120, 112, 107, 56, 8, 0, 0]
    assert codes.view(torch.float8_e4m3fn).float().tolist() == [256, 128, 88, 1, 2**-6, 0, 0]


@pytest.mark.parametrize("smooth", [False, True])
def test_unquantized_streaming_matches_independent_dense_with_tail(smooth):
    q, k, v = tensors()
    cfg = Config(v_smooth=smooth, quantize=False, q_block=7, kv_block=8, clusters=3)
    got, _ = attention(q, k, v, cfg)
    torch.testing.assert_close(got, dense(q, k, v), rtol=2e-5, atol=2e-6)


def test_mean_restoration_survives_running_max_update():
    q = torch.ones(1, 1, 2, 4)
    k = torch.cat([torch.full((1, 1, 4, 4), -20.0), torch.full((1, 1, 4, 4), 20.0)], -2)
    v = torch.cat([torch.full((1, 1, 4, 4), 3.0), torch.full((1, 1, 4, 4), 9.0)], -2)
    cfg = Config(v_smooth=True, quantize=False, kv_block=4, clusters=1)
    got, _ = attention(q, k, v, cfg)
    torch.testing.assert_close(got, dense(q, k, v))


@pytest.mark.parametrize("expcast", [False, True])
def test_constant_values_are_restored_with_quantized_probabilities(expcast):
    q, k, v = tensors()
    v.fill_(3.0)
    cfg = Config(v_smooth=True, expcast=expcast, kv_block=8)
    got, _ = attention(q, k, v, cfg)
    torch.testing.assert_close(got, torch.full_like(got, 3), rtol=1e-6, atol=1e-6)


def test_grouping_is_a_permutation_and_preserves_attention():
    q, k, v = tensors()
    pi, centers = group_values(v, clusters=4, iterations=3)
    expected = torch.arange(v.shape[-2]).expand_as(pi)
    assert torch.equal(pi.sort(-1).values, expected)
    gather = pi.unsqueeze(-1).expand_as(v)
    torch.testing.assert_close(dense(q, k.gather(-2, gather), v.gather(-2, gather)), dense(q, k, v))
    assert centers.shape == (1, 2, 4, 16)


def test_expcast_dense_oracle_one_kv_tile():
    q, k, v = tensors(heads=1)
    # Keep Q/K/V exact to isolate direct-code probability encoding.
    cfg = Config(expcast=True, quantize=False, kv_block=64)
    got, _ = attention(q, k, v, cfg)
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    u = ((s - s.amax(-1, keepdim=True)).double() * math.log2(math.e)).numpy()
    import numpy as np

    code = np.clip(np.rint(8 * u + 119.65), 0, 120).astype(np.uint8)
    exponent, mantissa = code >> 3, code & 7
    values = np.where(exponent == 0, mantissa * 2.0**-9, (1 + mantissa / 8) * 2.0 ** (exponent.astype(int) - 7))
    p = torch.tensor(values / values.sum(-1, keepdims=True), dtype=torch.float32)
    torch.testing.assert_close(got, p @ v, atol=2e-6, rtol=2e-5)


def test_cache_reuses_only_layout_and_resets_requests_and_shapes():
    _, _, v = tensors()
    cfg = Config(v_smooth=True, active_fraction=0.25, refresh_interval=4)
    cache = LayoutCache(cfg)
    p0, active, refreshed = cache.layout(v, step=0, total_steps=20, request_id="a", layer_id=0)
    assert active and refreshed
    p1, active, refreshed = cache.layout(v + 1, step=1, total_steps=20, request_id="a", layer_id=0)
    assert active and not refreshed and p1 is p0
    _, active, refreshed = cache.layout(v, step=4, total_steps=20, request_id="a", layer_id=0)
    assert active and refreshed
    p5, active, refreshed = cache.layout(v, step=5, total_steps=20, request_id="a", layer_id=0)
    assert not active and not refreshed
    p6, _, _ = cache.layout(v, step=6, total_steps=20, request_id="a", layer_id=0)
    assert p6 is p5
    _, _, refreshed = cache.layout(v, step=0, total_steps=20, request_id="b", layer_id=0)
    assert refreshed and cache.entries == 1
    _, _, refreshed = cache.layout(v[:, :, :-1], step=1, total_steps=20, request_id="b", layer_id=0)
    assert refreshed


def test_reused_layout_recomputes_current_means_and_residuals():
    q, k, v = tensors()
    cache = LayoutCache(Config(v_smooth=True, kv_block=8, quantize=False))
    a, _ = attention(q, k, v, cache.config, cache=cache, step=0, total_steps=20, request_id="x")
    b, _ = attention(q, k, v + 7, cache.config, cache=cache, step=1, total_steps=20, request_id="x")
    torch.testing.assert_close(b, a + 7, atol=2e-6, rtol=2e-5)


def test_unsupported_semantics_fail_explicitly():
    q, k, v = tensors()
    with pytest.raises(ValueError, match="causal"):
        attention(q, k, v, Config(), causal=True)
    with pytest.raises(ValueError, match="mask"):
        attention(q, k, v, Config(), mask=torch.ones(19, 35))
    with pytest.raises(ValueError, match="heads"):
        attention(q, k[:, :1], v[:, :1], Config())
    with pytest.raises(ValueError, match="finite"):
        attention(q * float("nan"), k, v, Config())


def test_no_smoothing_after_schedule_window():
    q, k, v = tensors()
    cfg = Config(v_smooth=True, quantize=False, kv_block=8)
    cache = LayoutCache(cfg)
    attention(q, k, v, cfg, cache=cache, request_id="a", step=0, total_steps=4)
    got, stats = attention(q, k, v, cfg, cache=cache, request_id="a", step=1, total_steps=4)
    assert not stats["smoothing_active"]
    torch.testing.assert_close(got, dense(q, k, v), atol=2e-6, rtol=2e-5)


def test_expcast_only_keeps_rowwise_exponential(monkeypatch):
    q, k, v = tensors(nq=7, nk=35, heads=1)
    original_exp = torch.exp
    exp_shapes = []

    def tracked_exp(x):
        exp_shapes.append(tuple(x.shape))
        return original_exp(x)

    monkeypatch.setattr(torch, "exp", tracked_exp)
    attention(q, k, v, Config(expcast=True, kv_block=8))
    assert exp_shapes and all(shape[-1] == 1 for shape in exp_shapes)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kv_block": 0},
        {"clusters": -1},
        {"active_fraction": float("nan")},
        {"backend": "automatic"},
        {"backend": "npu_fp8", "kv_block": 96},
    ],
)
def test_config_rejects_invalid_contract(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_npu_mode_never_silently_runs_on_cpu():
    with pytest.raises(ValueError, match="no CPU fallback"):
        attention(*tensors(), Config(backend="npu_fp8"))
