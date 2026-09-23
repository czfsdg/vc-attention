"""Independent checks of Eq. 7 and Appendix B, including real CUDA helpers."""
import importlib.util
import math
import struct
from types import SimpleNamespace

import pytest
import torch

from vc_attention.core import Config, attention, expcast_codes, hadamard, prepare_value_tile


def explicit_hadamard(width):
    # Sylvester entries H[i,j] = (-1)**popcount(i & j), independent of FWHT.
    return torch.tensor([
        [(-1.0) ** bin(i & j).count("1") for j in range(width)]
        for i in range(width)
    ]) / math.sqrt(width)


@pytest.mark.parametrize("dim", [1, 16, 37, 64, 128])
def test_hadamard_against_matrix_and_preserves_dot_products(dim):
    gen = torch.Generator().manual_seed(31)
    q, k = [torch.randn(2, 3, n, dim, generator=gen) for n in (7, 11)]
    width = 1 << (dim - 1).bit_length()
    expected = torch.nn.functional.pad(q, (0, width - dim)) @ explicit_hadamard(width)
    torch.testing.assert_close(hadamard(q), expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(hadamard(q) @ hadamard(k).transpose(-1, -2),
                               q @ k.transpose(-1, -2), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("k_smooth,rotate", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("scale", [None, 0.07])
def test_post_rope_preprocessing_preserves_dense_attention(k_smooth, rotate, scale):
    gen = torch.Generator().manual_seed(21)
    q, k = [torch.randn(1, 2, n, 37, generator=gen) for n in (13, 35)]
    k += torch.linspace(-2, 2, 37)  # A channel-dependent K offset.
    v = torch.randn(1, 2, 35, 21, generator=gen)
    cfg = Config(quantize=False, v_smooth=True, clusters=3, q_block=7, kv_block=8,
                 k_smooth=k_smooth, qk_hadamard=rotate)
    actual, _ = attention(q, k, v, cfg, scale=scale)
    # Original D=37 controls softmax scaling even though Hadamard pads to 64.
    expected = ((q @ k.transpose(-1, -2)) * (37**-0.5 if scale is None else scale)).softmax(-1) @ v
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("mean_dtype", ["float16", "bfloat16", "float32"])
def test_value_residual_uses_full_mean_and_storage_uses_scaled_mean(mean_dtype):
    gen = torch.Generator().manual_seed(13)
    tile = torch.randn(1, 2, 35, 21, generator=gen) * 0.17 + 3.14159
    payload, scale, stored = prepare_value_tile(tile, True, Config(mean_dtype=mean_dtype))
    mean = tile.mean(-2, keepdim=True)
    residual = tile - mean
    expected_scale = residual.abs().amax(-2, keepdim=True) / 448
    expected_payload = (residual / expected_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    expected_mean = (mean / expected_scale).to(getattr(torch, mean_dtype)).float()
    torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)
    torch.testing.assert_close(payload.view(torch.uint8), expected_payload.view(torch.uint8), atol=0, rtol=0)
    torch.testing.assert_close(stored, expected_mean, atol=0, rtol=0)
    if mean_dtype != "float32":
        # Rounding mu before demeaning was the old, different contract.
        assert not torch.equal(stored, mean.to(getattr(torch, mean_dtype)).float() / scale)


def test_nearly_constant_values_do_not_overflow_scaled_fp16_mean():
    tile = torch.tensor([1.0, 1.00001, 0.99999]).reshape(1, 1, 3, 1)
    payload, scale, stored = prepare_value_tile(tile, True, Config())
    assert torch.isfinite(stored).all()
    reconstructed = (payload.float() + stored) * scale
    torch.testing.assert_close(reconstructed, tile, atol=1e-6, rtol=0)


def fma_boundary_case():
    # Exact FP32 input bits; expected codes independently checked with C fmaf.
    bits = [0xc1226786, 0xc12104a2, 0xc11cdbf5, 0xc118b349, 0xc1148a9c, 0xc11327b8,
            0xc10ad65f, 0xc106adb2, 0xc1028506, 0xc0ff7e7b, 0xc0fcb8b2, 0xc0f72d21]
    x = torch.tensor([struct.unpack("<f", struct.pack("<I", v))[0] for v in bits])
    return x, torch.tensor([3, 3, 7, 9, 13, 13, 19, 23, 25, 27, 29, 31], dtype=torch.uint8)


def test_expcast_single_fma_regression():
    x, expected = fma_boundary_case()
    torch.testing.assert_close(expcast_codes(x), expected, atol=0, rtol=0)
    separate_rounding = (x * 11.541560173034668 + 119.6500015258789).round().to(torch.uint8)
    assert not torch.equal(separate_rounding, expected)


def test_stale_native_extension_requires_rebuild(monkeypatch):
    from vc_attention import cuda_backend
    monkeypatch.setattr(cuda_backend.importlib, "import_module", lambda _: SimpleNamespace())
    with pytest.raises(RuntimeError, match="Stale CUDA extension: rebuild"):
        cuda_backend._extension()


cuda_helper = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("vc_attention._cuda_ext") is None,
    reason="requires CUDA and a compiled extension; FP8 GEMM is not needed",
)


@cuda_helper
def test_native_expcast_fma_regression():
    from vc_attention.cuda_backend import _extension
    x, expected = fma_boundary_case()
    torch.testing.assert_close(_extension().expcast_codes(x.cuda()).cpu(), expected, atol=0, rtol=0)
    # Cover every code boundary and its two adjacent representable inputs.
    boundary = (torch.arange(120, dtype=torch.float32) + 0.5 - 119.65) / (8 * math.log2(math.e))
    x = torch.cat([boundary, torch.nextafter(boundary, torch.full_like(boundary, -math.inf)),
                   torch.nextafter(boundary, torch.full_like(boundary, math.inf)),
                   torch.tensor([0, -1e6, -math.inf])])
    torch.testing.assert_close(_extension().expcast_codes(x.cuda()).cpu(), expcast_codes(x), atol=0, rtol=0)


@cuda_helper
@pytest.mark.parametrize("dim", [37, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k_smooth,rotate", [(False, False), (False, True), (True, False), (True, True)])
def test_native_qk_preprocessing_matrix_oracle_and_current_stream(dim, dtype, k_smooth, rotate):
    from vc_attention.cuda_backend import _extension
    gen = torch.Generator().manual_seed(14)
    q, k = [torch.randn(2, 2, n, dim, generator=gen) for n in (13, 35)]
    k += torch.linspace(-2, 2, dim)
    q, k = q.to(dtype), k.to(dtype)
    expected_q, expected_k = q.float(), k.float()
    if k_smooth:
        expected_k = expected_k - expected_k.mean(-2, keepdim=True)
    if rotate:
        width = 1 << (dim - 1).bit_length()
        matrix = explicit_hadamard(width)
        expected_q = torch.nn.functional.pad(expected_q, (0, width - dim)) @ matrix
        expected_k = torch.nn.functional.pad(expected_k, (0, width - dim)) @ matrix
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Noncontiguous inputs also exercise the native dtype conversion path.
        qd, kd = [torch.stack([x.cuda(), x.cuda()], -1)[..., 0] for x in (q, k)]
        transformed = _extension().preprocess_qk(qd, kd, k_smooth, rotate)
        consumed = [x + 1 for x in transformed]
    stream.synchronize()
    for got, expected in zip(consumed, (expected_q, expected_k)):
        torch.testing.assert_close(got.cpu(), expected + 1, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(qd.cpu(), q, atol=0, rtol=0)
    torch.testing.assert_close(kd.cpu(), k, atol=0, rtol=0)
