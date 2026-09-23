"""Real Hopper tests. Set VC_REQUIRE_CUDA=1 in remote CI to forbid skips."""
import importlib.util
import math
import os
from dataclasses import replace

import pytest
import torch

from vc_attention.core import Config, LayoutCache, attention, expcast_codes, group_values


def _has_hopper():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


if os.environ.get("VC_REQUIRE_CUDA") == "1":
    if not _has_hopper() or importlib.util.find_spec("vc_attention._cuda_ext") is None:
        raise RuntimeError("VC_REQUIRE_CUDA=1: a Hopper GPU and built extension are required; refusing skipped verification")

hopper = pytest.mark.skipif(not _has_hopper(), reason="requires H800/H100 Hopper GPU")


def test_cuda_rejects_cpu_without_fallback():
    x = torch.ones(1, 1, 3, 16)
    with pytest.raises(ValueError, match="requires CUDA"):
        attention(x, x, x, Config(backend="cuda_fp8"))


@pytest.mark.parametrize("kwargs", [
    {"q_block": 7}, {"kv_block": 512}, {"kv_block": 48},
    {"k_quant_block": 8192}, {"quantize": False},
])
def test_cuda_rejects_unsupported_geometry(kwargs):
    with pytest.raises(ValueError):
        Config(backend="cuda_fp8", **kwargs)


@hopper
@pytest.mark.hopper
def test_preflight():
    from vc_attention.cuda_backend import preflight
    assert preflight()["preflight"] == "passed"


@hopper
@pytest.mark.hopper
def test_expcast_bytes_and_rounding_boundaries():
    from vc_attention.cuda_backend import _extension
    # Include values next to half-integer code boundaries, not just random data.
    thresholds = (torch.arange(120, dtype=torch.float32) + 0.5 - 119.65) / (8 * math.log2(math.e))
    values = torch.cat([
        thresholds,
        torch.nextafter(thresholds, torch.full_like(thresholds, -math.inf)),
        torch.nextafter(thresholds, torch.full_like(thresholds, math.inf)),
        torch.tensor([0, -1e6, -math.inf]),
    ])
    actual = _extension().expcast_codes(values.cuda()).cpu()
    torch.testing.assert_close(actual, expcast_codes(values), rtol=0, atol=0)


def _inputs(shape, dtype):
    batch, heads, nq, nk, d, dv = shape
    g = torch.Generator().manual_seed(1234)
    q = torch.randn(batch, heads, nq, d, generator=g)
    k = torch.randn(batch, heads, nk, d, generator=g)
    # Structured offset makes losing the V mean observable.
    v = torch.randn(batch, heads, nk, dv, generator=g) * 0.2 + 3.0
    return [x.to(dtype) for x in (q, k, v)]


@hopper
@pytest.mark.hopper
@pytest.mark.parametrize("smooth,cast", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("shape", [
    (1, 1, 19, 35, 16, 16),
    (1, 2, 129, 257, 128, 128),
    (2, 2, 33, 145, 37, 21),
])
def test_all_arms_vs_reference(shape, smooth, cast):
    q, k, v = _inputs(shape, torch.float32)
    # One cluster isolates quantization/attention from nondeterministic atomic
    # centroid accumulation. Clustering itself has a separate permutation test.
    cfg = Config(backend="cuda_fp8", v_smooth=smooth, expcast=cast, clusters=1)
    expected, _ = attention(q, k, v, replace(cfg, backend="reference"))
    actual, stats = attention(q.cuda(), k.cuda(), v.cuda(), cfg)
    torch.cuda.synchronize()
    assert stats["native_fp8_matmul"] and not stats["native_fused_attention"]
    assert torch.isfinite(actual).all()
    # Different reduction orders can cross an FP8 rounding boundary. Check
    # both aggregate error and outliers instead of relying only on cosine.
    error = actual.cpu() - expected
    relative = error.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-6)
    assert relative < 0.003
    assert error.abs().max() < 0.04


@hopper
@pytest.mark.hopper
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dtype_noncontiguous_and_distinct_value_width(dtype):
    q, k, v = _inputs((1, 2, 35, 131, 64, 32), dtype)
    # Interleaving creates noncontiguous tensors with exactly the original values.
    def strided(x):
        return torch.stack([x, x], -1)[..., 0]
    cfg = Config(backend="cuda_fp8", expcast=False)
    expected, _ = attention(q, k, v, replace(cfg, backend="reference"))
    actual, _ = attention(*[strided(x.cuda()) for x in (q, k, v)], cfg)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=0.04, rtol=0.005)


@hopper
@pytest.mark.hopper
def test_signed_values_and_shared_k_quantization_scale():
    q, k, v = _inputs((1, 1, 35, 259, 37, 21), torch.float32)
    v = (v - 3) * 5  # Signed values expose incorrect QK/PV layouts.
    k[:, :, :128] *= 0.1
    k[:, :, 128:256] *= 4  # Both halves must retain their shared K scale.
    cfg = Config(backend="cuda_fp8", expcast=False)
    expected, _ = attention(q, k, v, replace(cfg, backend="reference"))
    actual, _ = attention(q.cuda(), k.cuda(), v.cuda(), cfg)
    error = actual.cpu() - expected
    relative = error.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-6)
    assert relative < 0.005
    assert error.abs().max() < 0.015


@hopper
@pytest.mark.hopper
@pytest.mark.parametrize("mean_dtype", ["float16", "bfloat16", "float32"])
def test_mean_precision_and_alternative_tiles(mean_dtype):
    q, k, v = _inputs((1, 1, 33, 97, 37, 21), torch.float32)
    cfg = Config(backend="cuda_fp8", v_smooth=True, clusters=1,
                 mean_dtype=mean_dtype, q_block=64, kv_block=32)
    expected, _ = attention(q, k, v, replace(cfg, backend="reference"))
    actual, _ = attention(q.cuda(), k.cuda(), v.cuda(), cfg)
    torch.testing.assert_close(actual.cpu(), expected, atol=0.015, rtol=0.003)


@hopper
@pytest.mark.hopper
@pytest.mark.parametrize("cast", [False, True])
def test_running_max_mean_recovery_and_tail(cast):
    q = torch.ones(1, 1, 17, 16)
    k = torch.cat([torch.full((1, 1, 128, 16), -20.), torch.full((1, 1, 3, 16), 20.)], 2)
    v = torch.cat([torch.full((1, 1, 128, 16), 3.), torch.full((1, 1, 3, 16), 9.)], 2)
    cfg = Config(backend="cuda_fp8", v_smooth=True, expcast=cast, clusters=1)
    actual, _ = attention(q.cuda(), k.cuda(), v.cuda(), cfg)
    torch.testing.assert_close(actual, torch.full_like(actual, 9), atol=1e-5, rtol=1e-5)


@hopper
@pytest.mark.hopper
def test_clustering_permutation_and_cache_refresh():
    from vc_attention.cuda_backend import group_values as native_group
    q, k, v = [x.cuda() for x in _inputs((1, 2, 19, 259, 16, 16), torch.float32)]
    permutation, centers = native_group(v, clusters=3, iterations=2)
    assert centers.shape == (1, 2, 3, 16)
    torch.testing.assert_close(permutation.sort(-1).values, torch.arange(259, device="cuda").expand_as(permutation))
    # On the same device, C++ and Python use the same ATen Lloyd operations.
    expected_pi, expected_centers = group_values(v, clusters=3, iterations=2)
    torch.testing.assert_close(permutation, expected_pi)
    torch.testing.assert_close(centers, expected_centers, atol=1e-5, rtol=1e-5)
    cfg = Config(backend="cuda_fp8", v_smooth=True, clusters=1)
    cache = LayoutCache(cfg)
    args = dict(cache=cache, total_steps=20, request_id="one", layer_id=0)
    first, first_info = attention(q, k, v, cfg, step=0, **args)
    second, second_info = attention(q, k, v + 8, cfg, step=1, **args)
    assert first_info["layout_refreshed"] and not second_info["layout_refreshed"]
    torch.testing.assert_close(second - first, torch.full_like(first, 8), atol=0.02, rtol=0)
    _, late_info = attention(q, k, v, cfg, step=5, **args)
    assert not late_info["smoothing_active"]


@hopper
@pytest.mark.hopper
def test_current_stream_is_used():
    cfg = Config(backend="cuda_fp8")
    q, k, v = _inputs((1, 1, 17, 35, 16, 16), torch.float32)
    expected, _ = attention(q, k, v, replace(cfg, backend="reference"))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        qd, kd, vd = [x.cuda(non_blocking=True) for x in (q, k, v)]
        actual, _ = attention(qd, kd, vd, cfg)
        # A consumer on the same stream must observe the completed result.
        consumed = actual + 1
    stream.synchronize()
    torch.testing.assert_close(consumed.cpu(), expected + 1, atol=0.02, rtol=0.005)
