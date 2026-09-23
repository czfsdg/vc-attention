#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Activate your target Python environment before running this script. It does
# not install/upgrade torch, CUDA, or the NVIDIA driver.
python3 -c 'import torch; print("torch:", torch.__version__, "CUDA:", torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name()); assert torch.cuda.get_device_capability()[0] == 9, "Hopper GPU required"'
nvcc --version
# This validation script targets Hopper, including H200. Override any broad
# architecture list inherited from the container to compile only sm_90.
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS="${MAX_JOBS:-4}"
export VC_ATTENTION_BUILD_CUDA=1
printf 'Building for TORCH_CUDA_ARCH_LIST=%s\n' "$TORCH_CUDA_ARCH_LIST"
python3 -m pip install . --no-build-isolation --no-deps
python3 -m vc_attention.cuda_backend --device cuda:0
VC_REQUIRE_CUDA=1 python3 -m pytest tests/test_core.py tests/test_adapter.py tests/test_cuda.py tests/test_paper_alignment.py tests/test_imports.py tests/test_benchmark.py -q
