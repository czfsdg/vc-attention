"""End-to-end latency includes quantization, clustering and Python validation."""
import argparse
import json
import time

import torch
import torch.nn.functional as F

from vc_attention import Config, attention
from vc_attention.cuda_backend import preflight


def measure(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(repeats):
        output = fn()
    torch.cuda.synchronize()
    return output, (time.perf_counter() - begin) * 1000 / repeats


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    args = parser.parse_args()
    if min(args.queries, args.tokens, args.heads, args.dim, args.repeats) <= 0 or args.warmup < 0:
        parser.error("shape/repeats must be positive and warmup nonnegative")
    report = {"environment": preflight(), "arguments": vars(args), "results": {}}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(1234)
    dtype = getattr(torch, args.dtype)
    q, k, v = [torch.randn(1, args.heads, length, args.dim, device="cuda", dtype=dtype)
               for length in (args.queries, args.tokens, args.tokens)]
    # Independent FP32 dense oracle, limited by the explicitly chosen shapes.
    golden = torch.softmax(q.float() @ k.float().transpose(-2, -1) / args.dim**0.5, -1) @ v.float()
    arms = {"torch_sdpa": lambda: F.scaled_dot_product_attention(q, k, v)}
    for name, smooth, cast in [("fp8_control", False, False), ("expcast", False, True),
                               ("v_smooth", True, False), ("combined", True, True)]:
        cfg = Config(backend="cuda_fp8", v_smooth=smooth, expcast=cast)
        arms[name] = lambda cfg=cfg: attention(q, k, v, cfg)[0]
    for name, fn in arms.items():
        output, milliseconds = measure(fn, args.warmup, args.repeats)
        delta = output.float() - golden
        report["results"][name] = {
            "end_to_end_ms": milliseconds,
            "relative_rmse_vs_fp32": (delta.square().mean().sqrt() / golden.square().mean().sqrt().clamp_min(1e-12)).item(),
            "max_abs_error_vs_fp32": delta.abs().max().item(),
        }
    report["notes"] = [
        "Random synthetic data; not H3 model quality validation.",
        "Each native call includes preprocessing and first-step clustering when enabled; no layout cache is reused.",
        "FP8 and SDPA have different numerical contracts. No speedup is presumed.",
    ]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
