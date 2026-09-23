from types import SimpleNamespace

import pytest
import torch
from vc_attention.core import Config, attention
from vc_attention.lightx2v import ExperimentAdapter, configure, validate_layout
from vc_attention.replay import mindie_call


def options(tmp_path, mode="combined"):
    return SimpleNamespace(
        mode=mode,
        backend="reference",
        output_dir=str(tmp_path),
        fp8_mode="HIGH_PRECISION",
        capture_rank=0,
        capture_steps=[0],
        capture_layers=[0],
        capture_heads=1,
        max_captures=1,
        max_capture_mib=1,
    )


def audit():
    return {
        "calls": 0,
        "smoothing_calls": 0,
        "regroup_calls": 0,
        "expcast_calls": 0,
        "captures": {},
        "capture_bytes": 0,
    }


def inputs():
    gen = torch.Generator().manual_seed(4)
    return [torch.randn(17, 2, 16, generator=gen) for _ in range(3)]


def test_adapter_output_matches_direct_call_and_tracks_schedule(tmp_path):
    opt, log = options(tmp_path), audit()
    adapter = ExperimentAdapter(opt, log)
    q, k, v = inputs()
    scheduler = SimpleNamespace(layout_cpu=object(), step_index=0, infer_steps=4)
    got = adapter.apply(q, k, v, scheduler=scheduler, block_idx=0, causal=False)
    expected, _ = attention(
        *[x.transpose(0, 1).unsqueeze(0) for x in (q, k, v)], Config(v_smooth=True, expcast=True), total_steps=4
    )
    torch.testing.assert_close(got, expected.squeeze(0).transpose(0, 1).reshape(17, -1))
    scheduler.step_index = 1
    adapter.apply(q, k, v, scheduler=scheduler, block_idx=0)
    assert log["calls"] == 2 and log["smoothing_calls"] == 1 and log["expcast_calls"] == 2
    scheduler.layout_cpu, scheduler.step_index = object(), 0
    adapter.apply(q, k, v, scheduler=scheduler, block_idx=0)
    assert log["regroup_calls"] == 2 and adapter.cache.entries == 1


def test_capture_keeps_full_context_and_baseline_result(tmp_path, monkeypatch):
    opt, log = options(tmp_path, "capture"), audit()
    adapter = ExperimentAdapter(opt, log)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr("vc_attention.lightx2v.mindie_call", lambda q, k, v, **kw: q + 9)
    q, k, v = inputs()
    scheduler = SimpleNamespace(layout_cpu=object(), step_index=0, infer_steps=4)
    result = adapter.apply(q, k, v, scheduler=scheduler, block_idx=0)
    torch.testing.assert_close(result, (q + 9).reshape(17, -1))
    path = next(iter(log["captures"].values()))
    record = torch.load(path, weights_only=True)
    assert record["k"].shape == (1, 1, 17, 16)
    assert record["metadata"]["post_rope"] and record["metadata"]["full_kv_context"]
    torch.testing.assert_close(record["v"], v[:, :1].transpose(0, 1).unsqueeze(0))


@pytest.mark.parametrize("kw", [{"causal": True}, {"attn_mask": torch.ones(17, 17)}, {"dropout": 0.1}, {"unknown": 1}])
def test_adapter_never_silently_discards_semantics(kw):
    with pytest.raises(ValueError):
        validate_layout(*inputs(), None, None, None, None, kw)


def test_packed_sequences_are_rejected():
    with pytest.raises(ValueError, match="Packed"):
        validate_layout(*inputs(), torch.tensor([0, 8, 17]), None, None, None, {})


def test_config_is_opt_in_and_preserves_input():
    original = {"attn_type": "npu_flash_attn", "infer_steps": 29}
    changed = configure(original)
    assert changed["attn_type"] == "mindie_vc_experiment"
    assert original["attn_type"] == "npu_flash_attn" and changed["infer_steps"] == 29
    with pytest.raises(ValueError, match="compile"):
        configure({"use_compile": True})
    with pytest.raises(ValueError, match="head_parallel"):
        configure({"parallel": {"seq_p_head_parallel": True}})
    with pytest.raises(ValueError, match="warmup"):
        configure({"warmup": True})
    assert changed["warmup"] is False


def test_mindie_baseline_uses_explicit_bnsd_and_fp8_mode(monkeypatch):
    import sys

    calls = []

    def spy(q, k, v, **kw):
        calls.append(kw)
        return q

    monkeypatch.setitem(sys.modules, "mindiesd", SimpleNamespace(quant_attention=spy, attention_forward=spy))
    q = torch.ones(1, 2, 17, 16)
    mindie_call(q, q, q, fp8=False, scale=0.25)
    assert calls[-1]["head_first"] and calls[-1]["layout"] == "BNSD"
    mindie_call(q, q, q, fp8=True, scale=0.25, mode="C8V16_TILING512")
    assert calls[-1]["precision"] == "fp8" and calls[-1]["fp8_fa_mode"] == "C8V16_TILING512"
