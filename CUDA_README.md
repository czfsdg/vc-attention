# H800 CUDA C++ implementation

This adds `Config(backend="cuda_fp8")` to the existing VC-Attention API. It is
**VC-Attention (V-Smooth + ExpCast), not the SageAttention K-smoothing recipe**.

## Implementation and current validation

- `csrc/kernels.cu`: actual CUDA kernels for Q/K E4M3 quantization, per-channel V
  residual quantization, ExpCast, online Softmax, and V-mean recovery.
- Both QK and PV use **cuBLASLt E4M3 x E4M3 GEMM with FP32 accumulation/output**.
  There is no FP32-matmul simulation fallback. FP8 fast accumulation is disabled.
- `csrc/kernels.cu` also contains the C++ host tile loops.
- `csrc/bindings.cpp`: PyTorch allocation/stream integration and a C++/ATen
  implementation of the Lloyd clustering preprocessing.
- `cuda.py`: extension loading and the existing request/step layout cache.
  Python does not loop over the Attention tiles. K/V gathers remain PyTorch
  operations on CUDA. Clustering is GPU ATen orchestration, not a fused custom
  clustering kernel.
- This is a **multi-kernel correctness baseline**, not a fully fused FlashAttention
  kernel. Tiles and states pass through device memory between calls. No
  performance parity or speedup relative to FlashAttention/SDPA is claimed.
- **H800 runtime correctness and speed must be verified on the remote machine.**
  Local syntax/build checks are not a substitute for the Hopper tests below.

Local checks completed on 2026-09-23: the extension compiled for `sm_90` and
loaded successfully with CUDA 12.1 / PyTorch 2.3.1 on Windows; 34 CPU/reference
and argument-validation tests passed. The 24 Hopper tests and 2 NPU tests were
skipped because the local GPU is an RTX 3060. These results do not establish
Linux build compatibility or H800 runtime correctness. See
`results/cuda_local_validation.json` for the recorded scope.

## Requirements

- Target: Linux, NVIDIA H800/H100 (compute capability 9.x).
- CUDA Toolkit 12.1 or newer with `nvcc` and cuBLASLt. Use a Toolkit compatible
  with the CUDA-enabled PyTorch installed in the target environment.
- PyTorch >= 2.3, Python >= 3.9, a CUDA-supported C++17 compiler.
- Python build/test tools: setuptools >= 64, wheel, ninja, pytest, numpy.
- CANN and torch_npu are not required by this backend.

## Compile and install on the remote H800

Run inside the Python environment containing your intended PyTorch build:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name())"
nvcc --version
python -m pip install 'setuptools>=64' wheel ninja pytest numpy

export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=4
python -m pip install . --no-build-isolation --no-deps
```

`CUDA_HOME` may be set to the Toolkit directory if `nvcc` is not discovered.
Building does not require an attached GPU: the architecture is explicit. The
default build targets `9.0`, even if the build host has a different GPU.
Never copy a Windows `.pyd` to Linux; compile from source on the remote host.
When changing `.cpp` or `.cu`, rerun the install command to rebuild the extension.

No manual wheel step is required. To distribute the result to compatible hosts:

```bash
python -m pip wheel . --no-build-isolation --no-deps -w dist
```

For CPU/reference-only installation explicitly opt out of building the extension:

```bash
VC_ATTENTION_BUILD_CUDA=0 python -m pip install . --no-build-isolation --no-deps
```

This opt-out never changes the behavior of `backend="cuda_fp8"`: choosing it
still fails if the native extension or Hopper hardware is unavailable.

## Calling the operator

```python
import torch
from vc_attention import Config, attention

q = torch.randn(1, 2, 129, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 2, 257, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
cfg = Config(backend="cuda_fp8", expcast=True, v_smooth=True)
out, stats = attention(q, k, v, cfg)
print(out.shape, stats)
```

The default `Config()` still selects the existing `reference` backend. Existing
`npu_fp8` behavior is unchanged.

Supported v1 contract:

- Forward-only inference, dense noncausal unmasked MHA in `[B,H,N,D]` layout.
- FP16/BF16/FP32 inputs, output uses the input dtype; no backward implementation.
- Q/K head dimension and V head dimension each in `[1,128]`; they may differ.
- Noncontiguous inputs and sequence tails are supported by copies/padding.
- Q and KV block sizes are multiples of 32 in `[32,256]`, default 128.
- K quantization block is a multiple of KV block and <= 4096, default 256.
- Q/K share one scale per quantization block/head. V residual scales are per
  KV block/channel. Internal channel padding is to 32 and is excluded from
  maxima/means; padded keys get zero probability, not Softmax mass.
- Mean precision, denoising window, permutation refresh, and request isolation
  follow the original `Config` / `LayoutCache` contract.
- No causal mask, GQA, dropout, sparse selection, CUDA Graph/torch.compile
  integration, or native FP8 acceleration claims on Ampere.

With ExpCast disabled, denominator and V-mean recovery use the FP32 exponential
row sum; PV uses rounded E4M3 probabilities. With ExpCast enabled, both use the
decoded E4M3 values. This distinction is intentional and matches `core.py`.

## Validate before benchmarking

```bash
python -m vc_attention.cuda --device cuda:0
VC_REQUIRE_CUDA=1 python -m pytest tests/test_core.py tests/test_adapter.py tests/test_cuda.py -q
```

Or, after installing build/test dependencies, run `bash scripts/validate_h800.sh`
to build, preflight, and test. `VC_REQUIRE_CUDA=1` turns a missing Hopper GPU or
extension into an error; a run that skips hardware tests is not accepted.

Tests cover raw ExpCast bytes including rounding boundaries, all four ablation
arms, sequence tails, padded channels, multiple batches/heads, FP16/BF16,
noncontiguous inputs, rising online maxima, mean recovery, clustering/cache
behavior, and a non-default CUDA stream. Device/reference differences can come
from reduction order and FP8 boundary crossings; aggregate and maximum errors
are checked rather than relying only on cosine similarity.

```bash
python scripts/benchmark_cuda.py --queries 128 --tokens 1024 --heads 2 --dim 128
# Existing four-arm replay works with real captured tensors as well:
python -m vc_attention.replay --device cuda:0 --backend cuda_fp8 --tokens 1024 --heads 2 --dim 128 --output results/h800_replay.json
```

The benchmark measures the complete API call, including validation,
quantization, allocation, and first-step clustering. SDPA uses a different
numerical contract. Report measured speed and error together.

## Next performance work

Once Hopper correctness is established, profile launch overhead, small GEMMs,
and memory traffic. The planned next implementation is a fused Hopper
CUTLASS/CuTe kernel keeping score/probability tiles on chip, with explicit
WGMMA/TMA scheduling. That implementation is not part of this baseline.

## Technical references

- [PyTorch CUDAExtension](https://docs.pytorch.org/docs/stable/cpp_extension.html)
- [CUDA 12.1 cuBLASLt FP8 requirements](https://docs.nvidia.com/cuda/archive/12.1.0/cublas/index.html#cublasltmatmul)
- [CUDA FP8 conversions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html)
