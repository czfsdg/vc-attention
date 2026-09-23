"""Four-arm, same-input numerical replay plus separately labelled MindIE baselines."""

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .core import Config, attention

ARMS = {"fp8_control": (False, False), "expcast": (False, True), "v_smooth": (True, False), "combined": (True, True)}


def metrics(actual, reference):
    x, ref = actual.double(), reference.double()
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Nonfinite attention output")
    delta = x - ref
    mse = delta.square().mean().item()
    energy = ref.square().mean().item()
    return {
        "rmse": math.sqrt(mse),
        "relative_rmse": math.sqrt(mse / max(energy, 1e-30)),
        "max_abs": delta.abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(x.flatten(), ref.flatten(), dim=0).item(),
    }


def synchronize(device):
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def synthetic(tokens, heads, dim, seed, clustered):
    generator = torch.Generator().manual_seed(seed)
    q, k, v = [torch.randn((1, heads, tokens, dim), generator=generator) for _ in range(3)]
    if clustered:
        centers = torch.randn((1, heads, 8, dim), generator=generator) * 8
        ids = torch.randint(0, 8, (1, heads, tokens, 1), generator=generator).expand_as(v)
        v = centers.gather(-2, ids) + 0.1 * v
    return q, k, v


def mindie_call(q, k, v, *, fp8, scale, mode="HIGH_PRECISION"):
    import mindiesd

    # Native public wrappers accept FP16/BF16; never label the oracle as FIA.
    q, k, v = [x.to(torch.bfloat16).contiguous() for x in (q, k, v)]
    if fp8:
        return mindiesd.quant_attention(q, k, v, precision="fp8", layout="BNSD", scale=scale, fp8_fa_mode=mode)
    return mindiesd.attention_forward(
        q, k, v, head_first=True, scale=scale, opt_mode="manual", op_type="fused_attn_score", layout="BNSD"
    )


def run(args):
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
    device = torch.device(args.device)
    if args.backend == "npu_fp8":
        from .npu import preflight

        preflight_result = preflight(args.device)
    elif args.backend == "cuda_fp8":
        from .cuda import preflight

        preflight_result = preflight(args.device)
    else:
        preflight_result = None
    if args.capture:
        path = Path(args.capture)
        record = torch.load(path, map_location="cpu", weights_only=True)
        q, k, v = [record[name] for name in ("q", "k", "v")]
        source = {
            "kind": "captured_qkv",
            "file": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "metadata": record["metadata"],
        }
        scale = record["metadata"]["scale"]
    else:
        q, k, v = synthetic(args.tokens, args.heads, args.dim, args.seed, args.clustered)
        source = {"kind": "synthetic_clustered" if args.clustered else "synthetic_gaussian", "seed": args.seed}
        scale = args.dim**-0.5
    if args.q_start % 128 or args.q_start >= q.shape[-2]:
        raise ValueError("q-start must be a valid 128-token block boundary")
    end = q.shape[-2] if args.q_limit == 0 else min(q.shape[-2], args.q_start + args.q_limit)
    # Keep every K/V token. Do not confuse a cropped context with real attention.
    q = q[..., args.q_start : end, :]
    q, k, v = [x.float().to(device) for x in (q, k, v)]
    exact, _ = attention(q, k, v, Config(quantize=False), scale=scale)
    configs = {
        name: Config(
            v_smooth=smooth, expcast=cast, backend=args.backend, clusters=args.clusters, iterations=args.iterations
        )
        for name, (smooth, cast) in ARMS.items()
    }
    calls = {name: (lambda cfg=cfg: attention(q, k, v, cfg, scale=scale)[0]) for name, cfg in configs.items()}
    if args.mindie:
        if device.type != "npu":
            raise ValueError("--mindie requires --device npu:N")
        calls["mindie_bf16"] = lambda: mindie_call(q, k, v, fp8=False, scale=scale)
        calls["mindie_fp8"] = lambda: mindie_call(q, k, v, fp8=True, scale=scale, mode=args.fp8_mode)
    for _ in range(args.warmup):
        for call in calls.values():
            call()
    times = {name: [] for name in calls}
    errors = {}
    for repeat in range(args.repeats):
        order = list(calls) if repeat % 2 == 0 else list(reversed(calls))
        for name in order:
            synchronize(device)
            start = time.perf_counter()
            result = calls[name]()
            synchronize(device)
            times[name].append((time.perf_counter() - start) * 1000)
            errors[name] = metrics(result.float(), exact)
    report = {
        "status": "exploratory",
        "source": source,
        "device": str(device),
        "torch": torch.__version__,
        "host": platform.platform(),
        "preflight": preflight_result,
        "shape_q": list(q.shape),
        "shape_k": list(k.shape),
        "scale": scale,
        "query_range": [args.q_start, end],
        "reference": "FP32 streaming on the same input values",
        "full_kv_context": True,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "scope": "isolated attention; smoothing active; regrouping included on every measured call",
        "warning": "Prototype wall times are NOT fused-kernel or H3 end-to-end speedups. MindIE uses a different V/P quantization contract.",
        "arms": {
            name: {
                "metrics": errors[name],
                "wall_ms": times[name],
                "median_ms": statistics.median(times[name]),
                "config": asdict(configs[name]) if name in configs else {"native_mindie": True},
            }
            for name in calls
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({name: row["metrics"] for name, row in report["arms"].items()}, indent=2))
    print(f"Report: {output.resolve()}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("reference", "npu_fp8", "cuda_fp8"), default="reference")
    parser.add_argument("--mindie", action="store_true")
    parser.add_argument("--fp8-mode", choices=("HIGH_PRECISION", "C8V16_TILING512"), default="HIGH_PRECISION")
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--clustered", action="store_true")
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--q-start", type=int, default=0)
    parser.add_argument("--q-limit", type=int, default=128, help="0 = all queries; all KV tokens are always retained")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="vc-replay.json")
    args = parser.parse_args()
    if min(args.tokens, args.heads, args.dim, args.repeats) < 1 or min(args.warmup, args.q_start, args.q_limit) < 0:
        parser.error("Invalid dimension/count")
    run(args)


if __name__ == "__main__":
    main()
