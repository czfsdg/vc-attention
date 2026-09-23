"""Hopper entry point; named to avoid shadowing NVIDIA's cuda namespace."""
import importlib
import math

import torch


def _extension():
    try:
        ext = importlib.import_module("vc_attention._cuda_ext")
    except ImportError as exc:
        raise RuntimeError(
            "CUDA extension is missing or failed to load. On the H800 host run "
            "TORCH_CUDA_ARCH_LIST=9.0 python3 -m pip install . --no-build-isolation. "
            "No reference fallback was used. Original error: " + str(exc)
        ) from exc
    if getattr(ext, "numerical_contract_version", None) != 2:
        raise RuntimeError("Stale CUDA extension: rebuild with python3 -m pip install . --no-build-isolation --no-deps")
    return ext


def group_values(v, *, clusters, iterations, centers=None):
    return _extension().group_values(v, clusters, iterations, centers)


@torch.no_grad()
def _attention(q, k, v, cfg, *, cache, step, total_steps, request_id, layer_id, scale):
    # Shape/dtype/finite checks run in core.attention before dispatch. C++ also
    # validates bounds required for memory safety when called directly.
    if q.device.type != "cuda":
        raise ValueError("cuda_fp8 requires CUDA tensors; no CPU/NPU fallback")
    if torch.cuda.get_device_capability(q.device)[0] != 9:
        raise ValueError("cuda_fp8 requires Hopper (H800/H100/H200, compute capability 9.x)")
    if q.shape[-1] > 128 or v.shape[-1] > 128:
        raise ValueError("cuda_fp8 supports Q/K/V head dimensions <= 128")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    from .core import LayoutCache

    ext = _extension()
    if cache is None:
        cache = LayoutCache(cfg)
        request_id = "one-shot" if request_id is None else request_id
    elif cache.config != cfg:
        raise ValueError("cache.config must match config")
    pi, active, refreshed = cache.layout(
        v.float(), step=step, total_steps=total_steps, request_id=request_id, layer_id=layer_id
    )
    if pi is not None:
        k = k.gather(-2, pi.unsqueeze(-1).expand_as(k))
        v = v.gather(-2, pi.unsqueeze(-1).expand_as(v))
    mean_type = {"float32": 0, "float16": 1, "bfloat16": 2}[cfg.mean_dtype]
    output = ext.forward(q, k, v, cfg.q_block, cfg.k_quant_block, cfg.kv_block,
                         active, cfg.expcast, mean_type, scale, cfg.k_smooth, cfg.qk_hadamard)
    return output, {
        "backend": "cuda_fp8",
        "smoothing_active": active,
        "layout_refreshed": refreshed,
        "expcast": cfg.expcast,
        "k_smooth": cfg.k_smooth,
        "qk_hadamard": cfg.qk_hadamard,
        "numerical_contract_version": 2,
        "tiles": math.ceil(q.shape[-2] / cfg.q_block) * math.ceil(k.shape[-2] / cfg.kv_block),
        "native_fused_attention": False,
        "native_fp8_matmul": True,
    }


def preflight(device="cuda:0"):
    """Fail loudly unless both native FP8 GEMMs and ExpCast work on Hopper."""
    from .core import Config, attention, expcast_codes

    device = torch.device(device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] != 9:
        raise RuntimeError("Preflight requires an H800/H100/H200-class Hopper GPU")
    ext = _extension()
    x = torch.tensor([0, -1, -1.6, -8, -14, -100, -float("inf")], device=device) / math.log2(math.e)
    torch.testing.assert_close(ext.expcast_codes(x), expcast_codes(x), rtol=0, atol=0)
    # Exactly representable FP8 data gives a useful independent GEMM oracle.
    q = torch.zeros(1, 1, 19, 16, device=device)
    k = torch.zeros(1, 1, 35, 16, device=device)
    v = torch.full((1, 1, 35, 16), 3.0, device=device)
    for cast in (False, True):
        actual, _ = attention(q, k, v, Config(backend="cuda_fp8", expcast=cast))
        torch.testing.assert_close(actual, torch.full_like(actual, 3), rtol=2e-5, atol=2e-5)
    torch.cuda.synchronize(device)
    return {"device": str(device), "name": torch.cuda.get_device_name(device),
            "preflight": "passed", **ext.build_info()}


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(preflight(args.device), indent=2))
