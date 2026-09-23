"""Compare installed FA2/FA3/FA4/SageAttention and VC CUDA API latency.

All arms receive identical values in their native contiguous layouts. Quantization
and other internal preprocessing are timed; layout copies and first-use JIT are not.
"""
import argparse
from dataclasses import asdict, dataclass, replace
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import random
import statistics
import time

import torch


EXTERNAL = {
    "flash2": ("flash-attn", "flash_attn", "flash_attn_func", "BSHD"),
    "flash3": ("flash-attn-3", "flash_attn_3.flash_attn_interface", "flash_attn_func", "BSHD"),
    "flash4": ("flash-attn-4", "flash_attn.cute", "flash_attn_func", "BSHD"),
    "sage": ("sageattention", "sageattention", "sageattn", "BHND"),
}
DEFAULT_BACKENDS = (*EXTERNAL, "vc")


@dataclass
class Backend:
    name: str
    call: object
    layout: str
    info: dict


def distribution_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown (distribution metadata unavailable)"


def load_external(name):
    distribution, module_name, function_name, layout = EXTERNAL[name]
    try:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # Older FA3 source installations expose a top-level interface.
            # A missing dependency/binary inside FA3 must remain a real failure.
            if name != "flash3" or exc.name not in (
                "flash_attn_3", "flash_attn_3.flash_attn_interface"
            ):
                raise
            module_name = "flash_attn_interface"
            module = importlib.import_module(module_name)
        function = getattr(module, function_name)
        if not callable(function):
            raise TypeError(f"{module_name}.{function_name} is not callable")
    except Exception as exc:
        raise RuntimeError(
            f"{name}: cannot load {distribution}: {type(exc).__name__}: {exc}. "
            "Fix this installation or explicitly choose a subset with --backends. "
            "No substitute backend is used."
        ) from exc
    return function, layout, {
        "distribution": distribution,
        "version": distribution_version(distribution),
        "interface": f"{module_name}.{function_name}",
        "module_file": str(getattr(module, "__file__", "unknown")),
        "input_layout": layout,
    }


def external_backend(name, qkv_hnd, qkv_shd):
    function, layout, info = load_external(name)
    scale = qkv_hnd[0].shape[-1] ** -0.5
    if name == "sage":
        kwargs = dict(tensor_layout="HND", is_causal=False, sm_scale=scale, return_lse=False)
    else:
        kwargs = dict(softmax_scale=scale, causal=False)
        if name == "flash2":
            kwargs["dropout_p"] = 0.0
    inputs = qkv_shd if layout == "BSHD" else qkv_hnd
    info["call_kwargs"] = kwargs
    return Backend(name, lambda: function(*inputs, **kwargs), layout, info)


def make_backends(names, qkv_hnd, vc_ablation):
    # Make layout copies once, outside every timed call, for all Flash versions.
    qkv_shd = tuple(x.transpose(1, 2).contiguous() for x in qkv_hnd) if any(
        name in ("flash2", "flash3", "flash4") for name in names
    ) else None
    backends = []
    for name in names:
        if name != "vc":
            backends.append(external_backend(name, qkv_hnd, qkv_shd))
            continue
        from vc_attention import Config, attention

        variants = [("vc", True, True)]
        if vc_ablation:
            variants += [("vc_control", False, False), ("vc_expcast", False, True),
                         ("vc_v_smooth", True, False)]
        for label, smooth, cast in variants:
            cfg = Config(backend="cuda_fp8", v_smooth=smooth, expcast=cast)
            backends.append(Backend(
                label, lambda cfg=cfg: attention(*qkv_hnd, cfg), "BHND",
                {"distribution": "vc-attention", "version": distribution_version("vc-attention"),
                 "interface": "vc_attention.attention", "input_layout": "BHND",
                 "config": asdict(cfg), "layout_cache_reused": False},
            ))
    return backends


def output_hnd(output, layout, shape, dtype):
    # FA3/FA4 releases may return (output, lse); VC returns (output, stats).
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError("attention must return a Tensor or a tuple/list starting with one")
    if layout == "BSHD":
        output = output.transpose(1, 2)
    if output.shape != shape or output.dtype != dtype:
        raise ValueError(f"unexpected output {output.shape}/{output.dtype}; expected {shape}/{dtype}")
    if not torch.isfinite(output).all().item():
        raise ValueError("attention output contains NaN/Inf")
    return output


def query_indices(length, count):
    count = length if count == 0 else min(count, length)
    if count == 1:
        return [0]
    return [i * (length - 1) // (count - 1) for i in range(count)]


@torch.no_grad()
def fp32_reference(q, k, v, indices, chunk):
    """Sample query rows only, attending over ALL keys with an independent oracle."""
    k32, v32 = k.float(), v.float()
    rows = []
    for start in range(0, len(indices), chunk):
        ids = torch.tensor(indices[start:start + chunk], device=q.device)
        scores = q.index_select(2, ids).float() @ k32.transpose(-2, -1)
        probabilities = torch.softmax(scores * (q.shape[-1] ** -0.5), dim=-1)
        rows.append((probabilities @ v32).cpu())
    return torch.cat(rows, dim=2)


def error_metrics(sample, golden, reference_name="fp32"):
    delta = sample.double() - golden.double()
    return {
        f"relative_rmse_vs_{reference_name}": (delta.square().mean().sqrt()
                                               / golden.double().square().mean().sqrt().clamp_min(1e-12)).item(),
        f"max_abs_error_vs_{reference_name}": delta.abs().max().item(),
    }


@torch.no_grad()
def compare_vc_reference(qkv, native_config, native_sample, golden, indices):
    """Compare one timed native output with the same quantized Python algorithm.

    Keep ALL query rows until after attention: sampling Q before quantization
    changes the per-block Q scales. The reference runs on the same device and
    independently performs preprocessing, including clustering when enabled.
    """
    from vc_attention import Config, attention

    reference_config = replace(Config(**native_config), backend="reference")
    output, stats = attention(*qkv, reference_config)
    output = output_hnd(output, "BHND", qkv[0].shape, qkv[0].dtype)
    ids = torch.tensor(indices, device=output.device)
    reference_sample = output.index_select(2, ids).float().cpu()
    if native_sample.shape != reference_sample.shape or golden.shape != reference_sample.shape:
        raise ValueError("native/reference/FP32 samples must have the same shape")
    return {
        "reference_config": asdict(reference_config),
        "reference_device": str(qkv[0].device),
        "reference_stats": stats,
        "native_vs_fp32": error_metrics(native_sample, golden),
        "reference_vs_fp32": error_metrics(reference_sample, golden),
        "native_vs_reference": error_metrics(native_sample, reference_sample, reference_name="reference"),
    }


def measure_trial(fn, repeats, device):
    torch.cuda.synchronize(device)
    begin = time.perf_counter()
    for _ in range(repeats):
        output = fn()
    torch.cuda.synchronize(device)
    return output, (time.perf_counter() - begin) * 1000 / repeats


def add_speedups(results, baseline):
    baseline_ms = results[baseline]["median_ms"]
    for result in results.values():
        result["speedup_vs_baseline"] = baseline_ms / result["median_ms"]
    if "vc" not in results:
        return {}
    return {name: results[name]["median_ms"] / results["vc"]["median_ms"]
            for name in EXTERNAL if name in results}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", choices=DEFAULT_BACKENDS, default=list(DEFAULT_BACKENDS))
    parser.add_argument("--baseline", choices=DEFAULT_BACKENDS, help="reference latency for speedup (default: flash3, or first selected)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=1024, help="K/V sequence length")
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--check-queries", type=int, default=128, help="evenly spaced rows for accuracy; 0 checks all")
    parser.add_argument("--reference-chunk", type=int, default=32, help="FP32 oracle query rows per chunk")
    parser.add_argument("--vc-ablation", action="store_true", help="also measure VC control/ExpCast-only/V-Smooth-only")
    parser.add_argument("--vc-reference-check", action="store_true", help="compare VC to the same-config Python quantized reference after timing")
    parser.add_argument("--output", help="save complete JSON report")
    args = parser.parse_args(argv)
    if min(args.batch, args.queries, args.tokens, args.heads, args.warmup,
           args.repeats, args.trials, args.reference_chunk) <= 0 or args.check_queries < 0:
        parser.error("shapes/warmup/repeats/trials/reference-chunk must be positive; check-queries must be >= 0")
    if len(set(args.backends)) != len(args.backends):
        parser.error("--backends must not contain duplicates")
    if args.vc_ablation and "vc" not in args.backends:
        parser.error("--vc-ablation requires vc in --backends")
    if args.vc_reference_check and "vc" not in args.backends:
        parser.error("--vc-reference-check requires vc in --backends")
    args.baseline = args.baseline or ("flash3" if "flash3" in args.backends else args.backends[0])
    if args.baseline not in args.backends:
        parser.error("--baseline must be present in --backends")
    return args


@torch.no_grad()
def run(args):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to benchmark the installed attention kernels")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "device": str(device), "tf32_enabled": False,
    }
    if "vc" in args.backends:
        from vc_attention.cuda_backend import preflight
        environment["vc_preflight"] = preflight(device)
    print(json.dumps(environment, indent=2), flush=True)
    # Preflight may consume random numbers; subsets must still get the same Q/K/V.
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    qkv = tuple(torch.randn(args.batch, args.heads, n, args.dim, device=device, dtype=dtype)
                for n in (args.queries, args.tokens, args.tokens))
    backends = make_backends(args.backends, qkv, args.vc_ablation)
    indices = query_indices(args.queries, args.check_queries)
    print(f"FP32 reference: {len(indices)} query rows, all {args.tokens} keys", flush=True)
    golden = fp32_reference(*qkv, indices, args.reference_chunk)
    if not torch.isfinite(golden).all().item():
        raise ValueError("FP32 reference contains NaN/Inf")
    ids = torch.tensor(indices, device=device)
    results = {}
    for backend in backends:
        print(f"Warmup {backend.name} ({backend.info['version']}); first-use JIT may take time", flush=True)
        try:
            for _ in range(args.warmup):
                output = backend.call()
            torch.cuda.synchronize(device)
            normalized = output_hnd(output, backend.layout, qkv[0].shape, dtype)
            metrics = error_metrics(normalized.index_select(2, ids).float().cpu(), golden)
            del output, normalized
        except Exception as exc:
            raise RuntimeError(f"{backend.name} warmup/validation failed: {exc}") from exc
        results[backend.name] = {**backend.info, **metrics, "trial_ms": []}
    rng = random.Random(args.seed)
    trial_order = []
    native_samples = {}
    for trial in range(args.trials):
        order = list(backends)
        rng.shuffle(order)
        trial_order.append([backend.name for backend in order])
        for backend in order:
            print(f"Trial {trial + 1}/{args.trials}: {backend.name}", flush=True)
            try:
                output, milliseconds = measure_trial(backend.call, args.repeats, device)
                # Check a timed output outside the measured interval too.
                normalized = output_hnd(output, backend.layout, qkv[0].shape, dtype)
                sample = normalized.index_select(2, ids).float().cpu()
                metrics = error_metrics(sample, golden)
                if args.vc_reference_check and backend.info.get("config", {}).get("backend") == "cuda_fp8":
                    # Overwrite per trial; compare the last timed sample, not a
                    # new native call that might use a different clustering order.
                    native_samples[backend.name] = sample
                del output, normalized
            except Exception as exc:
                raise RuntimeError(f"{backend.name} timed run/validation failed: {exc}") from exc
            results[backend.name]["trial_ms"].append(milliseconds)
            for key, value in metrics.items():
                results[backend.name][key] = max(results[backend.name][key], value)
    for result in results.values():
        samples = result["trial_ms"]
        result.update(median_ms=statistics.median(samples), min_ms=min(samples), max_ms=max(samples))
    report = {
        "environment": environment, "arguments": vars(args), "results": results,
        "vc_speedup_vs": add_speedups(results, args.baseline), "trial_order": trial_order,
        "accuracy": {"query_indices": indices, "all_keys": True,
                     "reference": "chunked FP32 dense softmax, TF32 disabled",
                     "scope": "worst error over warmup and last output of each timed trial; no tolerance gate"},
        "notes": [
            "Dense noncausal forward MHA, no dropout/backward; identical random Q/K/V values.",
            "Synchronized wall time per API call; median of trial means. Not isolated GPU kernel time.",
            "Input layout conversion, output checks and first-use JIT excluded; internal preprocessing included.",
            "VC includes quantization/allocation/validation and first-step clustering every call; no layout cache.",
            "Sage uses its public sageattn auto dispatch, including its internal preprocessing.",
            "Different internal precisions; synthetic accuracy is not model quality validation.",
            "speedup_vs_baseline = baseline_ms / row_ms; vc_speedup_vs = other_ms / vc_ms; >1 is faster.",
        ],
    }
    if args.vc_reference_check:
        report["vc_reference_check"] = {
            "scope": "last output of the final timed trial for each VC variant; same Q/K/V and query indices",
            "reference": "same-device Python reference, FP8 encoding/decoding with FP32 matmuls; TF32 disabled",
            "preprocessing": "native and reference independently compute clustering and quantization",
            "full_q_before_sampling": True, "included_in_timing": False,
            "acceptance_threshold": None, "results": {},
        }
        for backend in backends:
            if backend.name not in native_samples:
                continue
            print(f"Python quantized reference: {backend.name} (all Q/K/V; outside timing)", flush=True)
            try:
                comparison = compare_vc_reference(
                    qkv, backend.info["config"], native_samples[backend.name], golden, indices
                )
            except Exception as exc:
                raise RuntimeError(f"{backend.name} Python reference check failed: {exc}") from exc
            report["vc_reference_check"]["results"][backend.name] = comparison
        report["notes"].append(
            "VC reference diagnostics report three pairwise errors for the same final timed output. "
            "They do not impose a tolerance or establish model quality."
        )
    return report


def print_report(report):
    baseline = report["arguments"]["baseline"]
    print(f"\n{'backend':<14} {'version':<16} {'median ms':>12} {'min ms':>12} {'max ms':>12} "
          f"{'speedup':>10} {'rel RMSE':>12} {'max abs':>12}")
    for name, row in report["results"].items():
        print(f"{name:<14} {row['version']:<16} {row['median_ms']:12.4f} {row['min_ms']:12.4f} "
              f"{row['max_ms']:12.4f} {row['speedup_vs_baseline']:9.3f}x "
              f"{row['relative_rmse_vs_fp32']:12.6f} {row['max_abs_error_vs_fp32']:12.6f}")
    print(f"speedup = {baseline} latency / row latency; >1 means this row is faster.")
    for name, speedup in report["vc_speedup_vs"].items():
        print(f"VC speedup vs {name}: {speedup:.4f}x (other_ms / vc_ms; >1 means VC is faster)")
    count = len(report["accuracy"]["query_indices"])
    total = report["arguments"]["queries"]
    scope = "all" if count == total else "sampled"
    print(f"Latency includes each API's internal preprocessing. Error uses {scope} {count}/{total} query rows and all keys.")
    if "vc_reference_check" in report:
        print("\nVC reference check: relative RMSE in PERCENT (last timed output; outside timing)")
        print(f"{'backend':<14} {'CUDA/FP32 %':>14} {'REF/FP32 %':>14} {'CUDA/REF %':>14} {'CUDA/REF max abs':>18}")
        for name, row in report["vc_reference_check"]["results"].items():
            print(f"{name:<14} {100 * row['native_vs_fp32']['relative_rmse_vs_fp32']:14.6f} "
                  f"{100 * row['reference_vs_fp32']['relative_rmse_vs_fp32']:14.6f} "
                  f"{100 * row['native_vs_reference']['relative_rmse_vs_reference']:14.6f} "
                  f"{row['native_vs_reference']['max_abs_error_vs_reference']:18.8f}")
        print("REF is the Python quantized algorithm; FP32 is standard dense attention.")
        print("CUDA/REF measures implementation agreement, including independent preprocessing; no pass/fail threshold.")


def main():
    args = parse_args()
    report = run(args)
    print_report(report)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Report saved to {path}")


if __name__ == "__main__":
    main()
