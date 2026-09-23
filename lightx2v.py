"""Opt-in LightX2V MiniMax-H3 registry adapter, capture and inference launcher."""

import argparse
import atexit
import json
import os
import runpy
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from .core import Config, LayoutCache, attention
from .replay import ARMS, mindie_call


def validate_layout(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, kwargs):
    allowed = {"scheduler", "block_idx", "causal", "softmax_scale", "dropout", "attn_mask", "mask", "pse"}
    if set(kwargs) - allowed:
        raise ValueError(f"Unknown attention options: {sorted(set(kwargs) - allowed)}")
    if kwargs.get("causal", False) or kwargs.get("dropout", 0):
        raise ValueError("Experiment requires noncausal attention and dropout=0")
    if any(kwargs.get(name) is not None for name in ("mask", "attn_mask", "pse")):
        raise ValueError("Mask/bias is unsupported; do not silently discard it")
    if q.ndim == 4 and q.shape[0] == 1:
        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)
    if any(x.ndim != 3 for x in (q, k, v)) or q.shape != k.shape or k.shape != v.shape:
        raise ValueError("H3 adapter requires one dense self-attention sequence in SND/1SND with equal heads")
    for cu, maximum, length in ((cu_seqlens_q, max_seqlen_q, q.shape[0]), (cu_seqlens_kv, max_seqlen_kv, k.shape[0])):
        if cu is not None and cu.cpu().tolist() != [0, length]:
            raise ValueError("Packed/multi-sequence attention is unsupported")
        if maximum is not None and int(maximum) != length:
            raise ValueError("max_seqlen does not match the single sequence")
    return [x.transpose(0, 1).unsqueeze(0) for x in (q, k, v)]


class ExperimentAdapter:
    def __init__(self, options, audit):
        self.options, self.audit = options, audit
        self.config = {}
        smooth, cast = ARMS.get(options.mode, (False, False))
        self.vc_config = Config(v_smooth=smooth, expcast=cast, backend=options.backend)
        self.cache = LayoutCache(self.vc_config)
        self._marker = None
        self._generation = 0
        self._last_steps = {}

    @torch.no_grad()
    def apply(self, q, k, v, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, **kwargs):
        q, k, v = validate_layout(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, kwargs)
        scheduler, layer = kwargs.get("scheduler"), kwargs.get("block_idx")
        if scheduler is None or layer is None:
            raise ValueError("H3 DiT scheduler/block_idx missing; set refiner_attn_type=npu_flash_attn")
        marker = getattr(scheduler, "layout_cpu", None)
        if marker is None:
            raise ValueError("This adapter targets H3 scheduler.layout_cpu; unsupported framework version")
        step, steps = int(scheduler.step_index), int(scheduler.infer_steps)
        if marker is not self._marker or (step == 0 and layer in self._last_steps):
            self.cache.reset()
            self._marker = marker  # Hold a strong reference; do not rely on recycled id().
            self._generation += 1
            self._last_steps.clear()
        if layer in self._last_steps and step <= self._last_steps[layer]:
            raise ValueError("Repeated/reversed layer-step; head_parallel/CFG/reentrant calls are unsupported")
        self._last_steps[layer] = step
        scale = kwargs.get("softmax_scale")
        scale = q.shape[-1] ** -0.5 if scale is None else scale
        self.audit["calls"] += 1
        if self.options.mode == "capture":
            self._capture(q, k, v, step, steps, layer, scale)
        if self.options.mode in ("capture", "mindie_fp8", "mindie_bf16"):
            out = mindie_call(q, k, v, fp8=self.options.mode != "mindie_bf16", scale=scale, mode=self.options.fp8_mode)
        else:
            out, stats = attention(
                q,
                k,
                v,
                self.vc_config,
                cache=self.cache,
                step=step,
                total_steps=steps,
                request_id=self._generation,
                layer_id=layer,
                scale=scale,
            )
            self.audit["smoothing_calls"] += int(stats["smoothing_active"])
            self.audit["regroup_calls"] += int(stats["layout_refreshed"])
            self.audit["expcast_calls"] += int(stats["expcast"])
        return out.squeeze(0).transpose(0, 1).reshape(q.shape[-2], -1).to(q.dtype)

    def _capture(self, q, k, v, step, steps, layer, scale):
        rank = int(os.environ.get("RANK", "0"))
        if (
            rank != self.options.capture_rank
            or step not in self.options.capture_steps
            or layer not in self.options.capture_layers
        ):
            return
        key = f"rank{rank}_step{step}_layer{layer}"
        if key in self.audit["captures"] or len(self.audit["captures"]) >= self.options.max_captures:
            return
        tensors = [x[:, : self.options.capture_heads].detach() for x in (q, k, v)]
        byte_count = sum(x.numel() * x.element_size() for x in tensors)
        if self.audit["capture_bytes"] + byte_count > self.options.max_capture_mib * 1024**2:
            raise ValueError("Capture budget exceeded; reduce heads/captures or explicitly increase max-capture-mib")
        target = Path(self.options.output_dir) / "captures" / f"{key}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "model": "MiniMax-H3",
            "layout": "BNSD",
            "step": step,
            "total_steps": steps,
            "layer": layer,
            "rank": rank,
            "scale": scale,
            "full_shape": list(q.shape),
            "post_rope": True,
            "attention_boundary": "post_rope_and_optional_ulysses",
            "full_kv_context": True,
            "dtype": str(q.dtype),
        }
        torch.save(dict(zip(("q", "k", "v"), [x.cpu().contiguous() for x in tensors]), metadata=metadata), target)
        self.audit["captures"][key] = str(target.resolve())
        self.audit["capture_bytes"] += byte_count


def configure(base):
    config = dict(base)
    parallel = config.get("parallel", {})
    if config.get("use_compile", False):
        raise ValueError("Disable use_compile for this eager experimental adapter, in every comparison arm")
    if config.get("warmup", False):
        raise ValueError("Disable warmup for this experiment; warmup tensors must not be captured as the real request")
    config["warmup"] = False
    if parallel.get("seq_p_head_parallel", False):
        raise ValueError("Disable seq_p_head_parallel; head identity is not supplied to the adapter")
    if parallel.get("seq_p_attn_type", "ulysses") != "ulysses":
        raise ValueError("Only Ulysses is supported, not ring/local-softmax composition")
    if config.get("enable_cfg", False):
        raise ValueError("CFG is unsupported by this H3 experiment")
    config["attn_type"] = "mindie_vc_experiment"
    config["refiner_attn_type"] = "npu_flash_attn"
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("capture", "mindie_bf16", "mindie_fp8", *ARMS), required=True)
    parser.add_argument("--backend", choices=("reference", "npu_fp8"), default="reference")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fp8-mode", choices=("HIGH_PRECISION", "C8V16_TILING512"), default="HIGH_PRECISION")
    parser.add_argument("--capture-steps", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--capture-layers", type=int, nargs="+", default=[0, 12])
    parser.add_argument("--capture-rank", type=int, default=0)
    parser.add_argument("--capture-heads", type=int, default=1)
    parser.add_argument("--max-captures", type=int, default=4)
    parser.add_argument("--max-capture-mib", type=int, default=512)
    parser.add_argument("--prepare-only", action="store_true", help="Write patched config without loading LightX2V/NPU")
    parser.add_argument("lightx_args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    argv = options.lightx_args
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if "--config_json" not in argv:
        parser.error("Pass existing LightX2V arguments after --, including --config_json PATH")
    if min(options.capture_heads, options.max_captures, options.max_capture_mib) < 1:
        parser.error("Capture limits must be positive")
    index = argv.index("--config_json") + 1
    if index >= len(argv):
        parser.error("--config_json requires a path")
    source_path = Path(argv[index])
    base = json.loads(source_path.read_text(encoding="utf-8"))
    config = configure(base)
    out = Path(options.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    patched_path = out / f"config.rank{rank}.json"
    patched_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    argv[index] = str(patched_path.resolve())
    if options.prepare_only:
        print(patched_path.resolve())
        return
    import torch_npu  # noqa: F401

    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if options.backend == "npu_fp8" and options.mode in ARMS:
        from .npu import preflight

        print(json.dumps(preflight(f"npu:{torch.npu.current_device()}")))
    from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
    from lightx2v_platform.ops.attn.template import AttnWeightTemplate

    smooth, cast = ARMS.get(options.mode, (False, False))
    audit = {
        "mode": options.mode,
        "backend": options.backend,
        "status": "running",
        "rank": rank,
        "calls": 0,
        "smoothing_calls": 0,
        "regroup_calls": 0,
        "expcast_calls": 0,
        "captures": {},
        "capture_bytes": 0,
        "argv": argv,
        "source_config": str(source_path.resolve()),
        "vc_config": asdict(Config(v_smooth=smooth, expcast=cast, backend=options.backend)),
        "fp8_mode": options.fp8_mode,
    }

    class RegisteredAdapter(ExperimentAdapter, AttnWeightTemplate):
        def __init__(self):
            super().__init__(options, audit)

    ATTN_WEIGHT_REGISTER.register(RegisteredAdapter, key="mindie_vc_experiment")

    def save_audit():
        (out / f"audit.rank{rank}.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    atexit.register(save_audit)
    print(f"[VC experiment] mode={options.mode} backend={options.backend}; eager prototype, no fused speed claim")
    sys.argv = ["lightx2v.infer", *argv]
    try:
        runpy.run_module("lightx2v.infer", run_name="__main__")
    except BaseException:
        audit["status"] = "failed"
        raise
    else:
        if audit["calls"] == 0:
            audit["status"] = "failed_no_attention_calls"
            raise RuntimeError("Experiment was configured but no attention call reached the adapter")
        audit["status"] = "completed"
    finally:
        save_audit()


if __name__ == "__main__":
    main()
