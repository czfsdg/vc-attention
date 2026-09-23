"""Executable VC numerical contract, reconstructed from arXiv:2609.15810v1.

This eager prototype streams tiles, but is NOT a fused Ascend attention kernel.
The reference backend decodes E4M3 operands and uses FP32 matmuls. It measures
algorithmic error, not FP8 hardware speed. NPU baselines live in replay.py.
Only dense, unmasked, noncausal MHA is supported (the MiniMax-H3 DiT contract).
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Config:
    expcast: bool = False
    v_smooth: bool = False
    quantize: bool = True
    q_block: int = 128
    k_quant_block: int = 256
    kv_block: int = 128
    clusters: int = 8
    iterations: int = 2
    active_fraction: float = 0.25
    refresh_interval: int = 4
    mean_dtype: str = "float16"
    backend: str = "reference"

    def __post_init__(self):
        for name in ("q_block", "k_quant_block", "kv_block", "clusters", "iterations", "refresh_interval"):
            if (
                isinstance(getattr(self, name), bool)
                or not isinstance(getattr(self, name), int)
                or getattr(self, name) < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.active_fraction) or not 0 <= self.active_fraction <= 1:
            raise ValueError("active_fraction must be in [0, 1]")
        if self.mean_dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError("Unsupported mean_dtype")
        if self.backend not in ("reference", "npu_fp8", "cuda_fp8"):
            raise ValueError("backend must be reference, npu_fp8, or cuda_fp8")
        if self.backend in ("npu_fp8", "cuda_fp8") and not self.quantize:
            raise ValueError(f"{self.backend} requires quantize=True")
        if self.backend in ("npu_fp8", "cuda_fp8") and self.k_quant_block % self.kv_block:
            raise ValueError(f"{self.backend} requires kv_block to divide k_quant_block")
        if self.backend == "cuda_fp8":
            if any(x < 32 or x > 256 or x % 32 for x in (self.q_block, self.kv_block)):
                raise ValueError("cuda_fp8 q_block/kv_block must be multiples of 32 in [32, 256]")
            if self.k_quant_block > 4096:
                raise ValueError("cuda_fp8 k_quant_block must be <= 4096")


def expcast_codes(shifted_scores):
    """Eq. 7: RNE and byte reinterpretation, NOT uint8-to-FP8 numeric cast.

    Input is in natural-log units and already row-max shifted. Eager PyTorch
    multiply/add need not fuse; a future native implementation must test FMA
    halfway cases independently. -inf maps to zero; masked rows are rejected
    by the public experiment entry rather than receiving invented semantics.
    """
    return torch.round(shifted_scores.float() * (8.0 * math.log2(math.e)) + 119.65).clamp(0, 120).to(torch.uint8)


def _encode(x, dims):
    scale = x.abs().amax(dim=dims, keepdim=True) / 448.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    payload = (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return payload, scale


def _quantize_qk(x, block):
    # One scale per sequence block and 128 channels. Match the public MindIE
    # HIGH_PRECISION block geometry; the quantizer itself is an explicit oracle,
    # not claimed byte-identical to npu_dynamic_block_quant without device tests.
    out = torch.empty_like(x)
    for s in range(0, x.shape[-2], block):
        for d in range(0, x.shape[-1], 128):
            tile = x[..., s : s + block, d : d + 128]
            payload, scale = _encode(tile, (-2, -1))
            out[..., s : s + block, d : d + 128] = payload.float() * scale
    return out


def group_values(v, *, clusters, iterations, centers=None):
    """Deterministic Lloyd prototype, batched per head; bounded assignment memory.

    Cluster count/iterations/initialization are experimental choices, not author
    code. Stable sorting preserves token order inside each cluster. Empty
    clusters retain their previous centers. Only layout/centers may be cached.
    """
    x = v.float()
    count = min(clusters, x.shape[-2])
    if centers is None or centers.shape[-2] != count:
        indices = torch.linspace(0, x.shape[-2] - 1, count, device=x.device).long()
        centers = x.index_select(-2, indices).clone()
    labels = torch.empty(x.shape[:-1], dtype=torch.long, device=x.device)
    for _ in range(iterations):
        sums = torch.zeros_like(centers)
        counts = torch.zeros_like(centers[..., :1])
        center_norms = centers.square().sum(-1).unsqueeze(-2)
        for start in range(0, x.shape[-2], 2048):
            rows = x[..., start : start + 2048, :]
            distances = rows.square().sum(-1, keepdim=True) + center_norms - 2 * (rows @ centers.transpose(-2, -1))
            ids = distances.argmin(-1)
            labels[..., start : start + 2048] = ids
            sums.scatter_add_(-2, ids.unsqueeze(-1).expand_as(rows), rows)
            counts.scatter_add_(-2, ids.unsqueeze(-1), torch.ones_like(rows[..., :1]))
        centers = torch.where(counts > 0, sums / counts.clamp_min(1), centers)
    return labels.argsort(dim=-1, stable=True), centers


class LayoutCache:
    """Request-scoped layout state. Callers must supply a real request identifier."""

    def __init__(self, config):
        self.config = config
        self._request_id = None
        self._states = {}

    @property
    def entries(self):
        return len(self._states)

    def reset(self):
        self._request_id = None
        self._states.clear()

    def layout(self, v, *, step, total_steps, request_id, layer_id):
        if request_id is None:
            raise ValueError("A persistent layout cache requires request_id")
        if total_steps < 1 or not 0 <= step < total_steps:
            raise ValueError("Invalid denoising step/total_steps")
        if request_id != self._request_id:
            self.reset()
            self._request_id = request_id
        signature = (tuple(v.shape), str(v.device), v.dtype, total_steps)
        state = self._states.get(layer_id)
        if state is not None and (state["signature"] != signature or step < state["step"]):
            state = None
            self._states.pop(layer_id, None)
        active = self.config.v_smooth and step < math.ceil(total_steps * self.config.active_fraction)
        refresh = active and (state is None or step - state["refresh_step"] >= self.config.refresh_interval)
        if refresh:
            grouping = group_values
            if self.config.backend == "cuda_fp8":
                from .cuda_backend import group_values as grouping
            pi, centers = grouping(
                v,
                clusters=self.config.clusters,
                iterations=self.config.iterations,
                centers=None if state is None else state["centers"],
            )
            state = {"signature": signature, "pi": pi, "centers": centers, "refresh_step": step, "step": step}
            self._states[layer_id] = state
        if state is not None:
            state["step"] = step
        return (None if state is None else state["pi"]), active, refresh


def _validate(q, k, v, causal, mask):
    if causal:
        raise ValueError("This experiment does not support causal attention")
    if mask is not None:
        raise ValueError("This experiment does not support a mask/bias")
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("Expected BNSD tensors")
    if q.shape[:2] != k.shape[:2] or k.shape[:2] != v.shape[:2]:
        raise ValueError("Batch and heads must match; GQA is not implemented")
    if k.shape[-2] != v.shape[-2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Incompatible Q/K/V shape")
    if any(0 in x.shape for x in (q, k, v)):
        raise ValueError("Empty Q/K/V are not supported")
    if any(x.device != q.device or x.dtype != q.dtype for x in (k, v)):
        raise ValueError("Q/K/V must have the same device and dtype")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Inputs must be FP16/BF16/FP32 before quantization")
    if not all(bool(torch.isfinite(x).all()) for x in (q, k, v)):
        raise ValueError("Q/K/V must be finite")


def _product(a, b, backend):
    if backend == "reference":
        return a.float() @ b.float()
    from .npu import fp8_product

    return fp8_product(a, b)


@torch.no_grad()
def attention(
    q,
    k,
    v,
    config=None,
    *,
    cache=None,
    step=0,
    total_steps=1,
    request_id=None,
    layer_id=0,
    scale=None,
    causal=False,
    mask=None,
):
    """Return (output, counters); online FP32 accumulator includes mean restoration.

    ExpCast uses decoded probabilities for BOTH row sum and mean restoration.
    Without ExpCast Eq. 6 uses exact exp row sums, while PV consumes rounded P.
    Quantization disabled => exact streaming reference (unless ExpCast enabled).
    """
    cfg = config or Config()
    _validate(q, k, v, causal, mask)
    if cfg.backend == "cuda_fp8":
        from .cuda_backend import _attention
        return _attention(q, k, v, cfg, cache=cache, step=step, total_steps=total_steps,
                          request_id=request_id, layer_id=layer_id,
                          scale=q.shape[-1] ** -0.5 if scale is None else scale)
    if cfg.backend == "npu_fp8" and q.device.type != "npu":
        raise ValueError("npu_fp8 requires NPU tensors; no CPU fallback")
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    if cfg.backend == "npu_fp8" and q.shape[-1] > 128:
        raise ValueError("npu_fp8 prototype currently supports head_dim <= 128")
    qf, kf, vf = q.float(), k.float(), v.float()
    if cache is None:
        cache = LayoutCache(cfg)
        request_id = "one-shot" if request_id is None else request_id
    elif cache.config != cfg:
        raise ValueError("cache.config must match config")
    pi, active, refreshed = cache.layout(
        vf, step=step, total_steps=total_steps, request_id=request_id, layer_id=layer_id
    )
    if pi is not None:
        kf = kf.gather(-2, pi.unsqueeze(-1).expand_as(kf))
        vf = vf.gather(-2, pi.unsqueeze(-1).expand_as(vf))
    if cfg.backend == "npu_fp8":
        native_q = [_encode(qf[..., s : s + cfg.q_block, :], (-2, -1)) for s in range(0, qf.shape[-2], cfg.q_block)]
        native_k = [
            _encode(kf[..., s : s + cfg.k_quant_block, :], (-2, -1)) for s in range(0, kf.shape[-2], cfg.k_quant_block)
        ]
    elif cfg.quantize:
        qf = _quantize_qk(qf, cfg.q_block)
        kf = _quantize_qk(kf, cfg.k_quant_block)
    # Recompute values/means every call, even when the permutation is reused.
    values = []
    for start in range(0, vf.shape[-2], cfg.kv_block):
        tile = vf[..., start : start + cfg.kv_block, :]
        mean = tile.mean(-2, keepdim=True) if active else torch.zeros_like(tile[..., :1, :])
        if cfg.quantize:
            mean = mean.to(getattr(torch, cfg.mean_dtype)).float()
        residual = tile - mean
        if cfg.quantize:
            payload, vscale = _encode(residual, (-2,))
        else:
            payload, vscale = residual, torch.ones_like(mean)
        values.append((payload, vscale, mean))
    output = torch.empty((*q.shape[:-1], v.shape[-1]), device=q.device, dtype=torch.float32)
    tiles = 0
    for qs in range(0, q.shape[-2], cfg.q_block):
        qt = qf[..., qs : qs + cfg.q_block, :]
        shape = (*qt.shape[:-1], 1)
        rowmax = torch.full(shape, -float("inf"), device=q.device)
        denom = torch.zeros(shape, device=q.device)
        accum = torch.zeros((*qt.shape[:-1], v.shape[-1]), device=q.device)
        for j, ks in enumerate(range(0, k.shape[-2], cfg.kv_block)):
            kt = kf[..., ks : ks + cfg.kv_block, :]
            if cfg.backend == "npu_fp8":
                # Slice payloads without requantizing the smaller KV tile.
                qa, qa_scale = native_q[qs // cfg.q_block]
                kb, kb_scale = native_k[ks // cfg.k_quant_block]
                offset = ks % cfg.k_quant_block
                kb = kb[..., offset : offset + cfg.kv_block, :]
                scores = _product(qa, kb.transpose(-2, -1), cfg.backend) * qa_scale * kb_scale * scale
            else:
                scores = (qt @ kt.transpose(-2, -1)) * scale
            newmax = torch.maximum(rowmax, scores.amax(-1, keepdim=True))
            alpha = torch.exp(rowmax - newmax)  # Per-row online rescaling remains.
            shifted = scores - newmax
            if cfg.expcast:
                p8 = expcast_codes(shifted).view(torch.float8_e4m3fn)
                probability = p8.float() / 256.0
            else:
                probability = torch.exp(shifted)
                p8 = (probability * 256.0).to(torch.float8_e4m3fn) if cfg.quantize else probability * 256.0
            rowmass = probability.sum(-1, keepdim=True)
            residual, vscale, mean = values[j]
            pv = _product(p8, residual, cfg.backend) * (vscale / 256.0)
            accum = alpha * accum + pv + rowmass * mean
            denom = alpha * denom + rowmass
            rowmax = newmax
            tiles += 1
        output[..., qs : qs + cfg.q_block, :] = accum / denom
    return output.to(q.dtype), {
        "backend": cfg.backend,
        "smoothing_active": active,
        "layout_refreshed": refreshed,
        "expcast": cfg.expcast,
        "tiles": tiles,
        "native_fused_attention": False,
    }
