"""CPU regression checks for benchmark adapters, reference and measurement math.

These do not replace running the actual third-party kernels on Hopper.
"""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


spec = importlib.util.spec_from_file_location(
    "vc_benchmark_cuda", Path(__file__).resolve().parents[1] / "scripts" / "benchmark_cuda.py"
)
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


@pytest.mark.parametrize("name", ["flash2", "flash3", "flash4", "sage"])
def test_adapter_preserves_values_layout_scale_and_output(monkeypatch, name):
    torch.manual_seed(12)
    qkv = tuple(torch.randn(2, 3, n, 8) for n in (5, 7, 7))
    shd = tuple(x.transpose(1, 2).contiguous() for x in qkv)
    calls = []

    def installed_function(q, k, v, **kwargs):
        calls.append(kwargs)
        assert all(x.is_contiguous() for x in (q, k, v))
        if name == "sage":
            assert kwargs == dict(tensor_layout="HND", is_causal=False, sm_scale=8**-0.5, return_lse=False)
        else:
            assert kwargs["causal"] is False
            assert kwargs["softmax_scale"] == 8**-0.5
            if name == "flash2":
                assert kwargs == dict(softmax_scale=8**-0.5, causal=False, dropout_p=0.0)
            else:
                assert set(kwargs) == {"softmax_scale", "causal"}
            assert q.shape == (2, 5, 3, 8)
            q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        for actual, expected in zip((q, k, v), qkv):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        result = F.scaled_dot_product_attention(q, k, v)
        if name != "sage":
            result = result.transpose(1, 2).contiguous()
        return (result, "unused lse") if name in ("flash3", "flash4") else result

    distribution, module_name, function_name, _ = benchmark.EXTERNAL[name]
    module = SimpleNamespace(__file__="installed/interface.py", **{function_name: installed_function})
    monkeypatch.setattr(benchmark.importlib, "import_module", lambda path: module if path == module_name else None)
    monkeypatch.setattr(benchmark.metadata, "version", lambda package: "test-version" if package == distribution else None)
    backend = benchmark.external_backend(name, qkv, shd)
    actual = benchmark.output_hnd(backend.call(), backend.layout, qkv[0].shape, qkv[0].dtype)
    expected = torch.softmax(qkv[0] @ qkv[1].transpose(-2, -1) / 8**0.5, -1) @ qkv[2]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert len(calls) == 1
    assert backend.info["version"] == "test-version"
    assert backend.info["interface"] == f"{module_name}.{function_name}"


def test_fa3_legacy_interface_only_when_modern_module_is_absent(monkeypatch):
    seen = []
    legacy = SimpleNamespace(flash_attn_func=lambda: None, __file__="legacy.py")

    def import_module(name):
        seen.append(name)
        if name == "flash_attn_3.flash_attn_interface":
            raise ModuleNotFoundError("No module named flash_attn_3", name="flash_attn_3")
        assert name == "flash_attn_interface"
        return legacy

    monkeypatch.setattr(benchmark.importlib, "import_module", import_module)
    monkeypatch.setattr(benchmark, "distribution_version", lambda _: "source-install")
    function, layout, info = benchmark.load_external("flash3")
    assert function is legacy.flash_attn_func
    assert layout == "BSHD"
    assert info["interface"] == "flash_attn_interface.flash_attn_func"
    assert seen == ["flash_attn_3.flash_attn_interface", "flash_attn_interface"]


@pytest.mark.parametrize("name,missing", [("flash3", "flash_attn_3._C"), ("flash4", "flash_attn.cute")])
def test_broken_installation_is_not_silently_replaced(monkeypatch, name, missing):
    seen = []

    def import_module(module):
        seen.append(module)
        raise ModuleNotFoundError(f"No module named {missing}", name=missing)

    monkeypatch.setattr(benchmark.importlib, "import_module", import_module)
    with pytest.raises(RuntimeError, match=f"{name}: cannot load") as error:
        benchmark.load_external(name)
    assert isinstance(error.value.__cause__, ModuleNotFoundError)
    assert len(seen) == 1


@pytest.mark.parametrize("count", [0, 1, 3, 100])
def test_sampled_reference_uses_all_keys_and_handles_chunk_tail(count):
    torch.manual_seed(1)
    q, k, v = (torch.randn(2, 3, n, 8) for n in (7, 11, 11))
    indices = benchmark.query_indices(7, count)
    assert indices == sorted(set(indices))
    assert len(indices) == (7 if count == 0 else min(count, 7))
    assert indices[0] == 0
    if len(indices) > 1:
        assert indices[-1] == 6
    dense = F.scaled_dot_product_attention(q, k, v)
    actual = benchmark.fp32_reference(q, k, v, indices, chunk=2)
    torch.testing.assert_close(actual, dense[:, :, indices, :], atol=1e-6, rtol=1e-5)
    assert actual.device.type == "cpu"


def test_timer_waits_for_gpu_and_converts_to_per_call_milliseconds(monkeypatch):
    events = []
    clock = iter([10.0, 10.06])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", lambda device: events.append(("sync", device)))

    def call():
        events.append("call")
        return "output"

    output, ms = benchmark.measure_trial(call, repeats=3, device="cuda:1")
    assert output == "output"
    assert ms == pytest.approx(20)
    assert events == [("sync", "cuda:1"), "call", "call", "call", ("sync", "cuda:1")]


def test_speedup_direction_and_error_units():
    rows = {"flash3": {"median_ms": 1.0}, "flash4": {"median_ms": 0.5}, "vc": {"median_ms": 2.0}}
    vc_ratios = benchmark.add_speedups(rows, "flash3")
    assert rows["flash4"]["speedup_vs_baseline"] == 2.0
    assert rows["vc"]["speedup_vs_baseline"] == 0.5
    assert vc_ratios == {"flash3": 0.5, "flash4": 0.25}
    metrics = benchmark.error_metrics(torch.tensor([2.0, 2.0]), torch.ones(2))
    assert metrics == {"relative_rmse_vs_fp32": 1.0, "max_abs_error_vs_fp32": 1.0}


def test_vc_ablation_dispatch_freezes_each_config(monkeypatch):
    import vc_attention

    configs = []
    def attention(q, k, v, cfg):
        configs.append(cfg)
        return q, {}

    monkeypatch.setattr(vc_attention, "attention", attention)
    qkv = (torch.zeros(1, 2, 3, 8),) * 3
    backends = benchmark.make_backends(["vc"], qkv, vc_ablation=True)
    for backend in backends:
        backend.call()
    assert [x.name for x in backends] == ["vc", "vc_control", "vc_expcast", "vc_v_smooth"]
    assert [(x.v_smooth, x.expcast) for x in configs] == [(True, True), (False, False), (False, True), (True, False)]
    assert all(x.backend == "cuda_fp8" for x in configs)


def test_nonfinite_output_is_rejected():
    output = torch.full((1, 2, 3, 8), float("nan"))
    with pytest.raises(ValueError, match="NaN/Inf"):
        benchmark.output_hnd(output, "BHND", output.shape, output.dtype)


def test_explicit_subset_uses_selected_baseline():
    args = benchmark.parse_args(["--backends", "sage", "vc"])
    assert args.baseline == "sage"
    assert args.backends == ["sage", "vc"]
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--backends", "sage", "--baseline", "flash3"])
