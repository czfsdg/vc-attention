"""Real device checks: skipped, never mocked as an A5 pass, on CPU hosts."""

import importlib.util

import pytest
import torch
from vc_attention.npu import preflight
from vc_attention.replay import metrics, mindie_call

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch_npu") is None, reason="A5/torch_npu unavailable")


def test_a5_fp8_preflight():
    import torch_npu  # noqa: F401

    assert preflight("npu:0")["fp8_matmul"] == "passed"


def test_real_mindie_baselines():
    import torch_npu  # noqa: F401

    gen = torch.Generator().manual_seed(10)
    q, k, v = [torch.randn(1, 2, 512, 128, generator=gen).bfloat16() for _ in range(3)]
    exact = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    for fp8, tolerance in ((False, 0.02), (True, 0.15)):
        output = mindie_call(q.npu(), k.npu(), v.npu(), fp8=fp8, scale=128**-0.5)
        assert metrics(output.float().cpu(), exact)["relative_rmse"] < tolerance
