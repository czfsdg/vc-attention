"""Build against the torch already installed in the target environment."""
import os
import sys

from setuptools import setup
extensions = []
commands = {}
if os.environ.get("VC_ATTENTION_BUILD_CUDA", "1") != "0":
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

    if CUDA_HOME is None or torch.version.cuda is None:
        raise RuntimeError(
            "CUDA Toolkit and a CUDA-enabled PyTorch are required. Install with "
            "pip install . --no-build-isolation. For reference-only use, set "
            "VC_ATTENTION_BUILD_CUDA=0 explicitly."
        )
    # The build host need not have a GPU. H800/H100 are the intended targets.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    extensions = [CUDAExtension(
        "vc_attention._cuda_ext",
        sources=["csrc/bindings.cpp", "csrc/kernels.cu"],
        libraries=["cublasLt"],
        extra_compile_args={
            "cxx": ["/O2", "/std:c++17"] if sys.platform == "win32" else ["-O3", "-std=c++17"],
            # ExpCast explicitly uses __fmaf_rn; keep other arithmetic rounding
            # stable for reference comparisons, independent of auto-contraction.
            "nvcc": ["-O3", "-std=c++17", "--fmad=false", "-lineinfo"],
        },
    )]
    commands["build_ext"] = BuildExtension

setup(
    name="vc-attention",
    version="0.2.0",
    description="VC-Attention reference and CUDA C++ FP8 backend for Hopper",
    packages=["vc_attention"],
    package_dir={"": "src"},
    ext_modules=extensions,
    cmdclass=commands,
    python_requires=">=3.9",
    install_requires=["torch>=2.3"],
    extras_require={"test": ["pytest", "numpy"]},
    include_package_data=False,
)
