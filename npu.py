"""Unfused A5 FP8 matmul route. Requires device preflight; no silent fallback.

Uses the documented npu_add_quant_matmul_ MX path with E8M0 scales fixed at 1
(byte 127). Actual operands are E4M3; ordinary float scales apply outside the
matmul, preserving the same quantization contract as the CPU reference.
Source: Ascend/op-plugin docs/zh/custom_APIs/torch_npu/torch_npu-npu_add_quant_matmul_.md
blob e0c917fbc50059790bace68697cee9e159ac0fff. This is not a fused FIA kernel.
"""

import math

import torch


def fp8_product(a, b):
    import torch_npu

    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise TypeError("FP8 matmul requires E4M3 operands")
    if a.device.type != "npu" or b.device != a.device:
        raise ValueError("FP8 matmul requires same-device NPU tensors")
    op = getattr(torch_npu, "npu_add_quant_matmul_", None)
    if op is None:
        raise RuntimeError("This experiment needs torch_npu.npu_add_quant_matmul_; use reference explicitly if absent")
    batch = a.shape[:-2]
    m, k = a.shape[-2:]
    n = b.shape[-1]
    if b.shape[:-2] != batch or b.shape[-2] != k:
        raise ValueError("Mismatched FP8 matmul shapes")
    pm, pk, pn = (math.ceil(x / 64) * 64 for x in (m, k, n))
    outputs = []
    for ai, bi in zip(a.reshape(-1, m, k), b.reshape(-1, k, n)):
        # The MX path requires transposed storage for x1 and its scales.
        at = torch.zeros((pk, pm), dtype=a.dtype, device=a.device)
        bt = torch.zeros((pk, pn), dtype=b.dtype, device=b.device)
        at[:k, :m] = ai.transpose(0, 1)
        bt[:k, :n] = bi
        x1_scale = torch.full((pk // 64, pm, 2), 127, dtype=torch.uint8, device=a.device)
        x2_scale = torch.full((pk // 64, pn, 2), 127, dtype=torch.uint8, device=a.device)
        out = torch.zeros((pm, pn), dtype=torch.float32, device=a.device)
        op(
            out,
            at.transpose(0, 1),
            bt,
            x2_scale,
            x1_scale=x1_scale.transpose(0, 1),
            x1_scale_dtype=torch_npu.float8_e8m0fnu,
            x2_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
        )
        outputs.append(out[:m, :n])
    return torch.stack(outputs).reshape(*batch, m, n)


def preflight(device="npu:0"):
    """Check bit reinterpretation, RNE and real FP8 products before model runs."""
    from .core import Config, attention, expcast_codes

    code = expcast_codes(torch.tensor([0.0, -1.6 / math.log2(math.e), -float("inf")], device=device))
    if code.cpu().tolist() != [120, 107, 0]:
        raise RuntimeError("ExpCast integer encoding mismatch")
    if code.view(torch.float8_e4m3fn).float().cpu().tolist() != [256.0, 88.0, 0.0]:
        raise RuntimeError("E4M3 byte reinterpretation mismatch")
    ties = torch.tensor([2.5, 3.5, 4.5, 5.5], device=device).round().cpu().tolist()
    if ties != [2.0, 4.0, 4.0, 6.0]:
        raise RuntimeError("round is not nearest-even on this runtime")
    generator = torch.Generator().manual_seed(7)
    for m, k, n in ((64, 128, 128), (13, 35, 19)):
        a = torch.randn((1, 2, m, k), generator=generator).to(torch.float8_e4m3fn)
        b = torch.randn((1, 2, k, n), generator=generator).to(torch.float8_e4m3fn)
        expected = a.float() @ b.float()
        actual = fp8_product(a.to(device), b.to(device)).cpu()
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    # Compare all four arms to the independent execution backend on identical data.
    q, k, v = [torch.randn((1, 1, s, 128), generator=generator) for s in (67, 259, 259)]
    errors = {}
    for smooth, cast in ((False, False), (False, True), (True, False), (True, True)):
        cpu, _ = attention(q, k, v, Config(v_smooth=smooth, expcast=cast))
        npu, _ = attention(
            q.to(device), k.to(device), v.to(device), Config(v_smooth=smooth, expcast=cast, backend="npu_fp8")
        )
        # Parallel clustering/reductions can change boundaries; inspect beyond tolerance.
        rmse = ((cpu - npu.cpu()).square().mean() / cpu.square().mean().clamp_min(1e-20)).sqrt().item()
        if not math.isfinite(rmse) or rmse > 0.01:
            raise RuntimeError(f"NPU/reference mismatch: smooth={smooth}, expcast={cast}, relative_RMSE={rmse}")
        errors[f"smooth={smooth},expcast={cast}"] = rmse
    return {"device": str(device), "fp8_matmul": "passed", "bitcast": "passed", "relative_rmse": errors}


if __name__ == "__main__":
    import argparse
    import json

    import torch_npu  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    print(json.dumps(preflight(args.device), indent=2))
