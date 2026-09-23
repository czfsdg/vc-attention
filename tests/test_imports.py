"""Check import resolution from the source checkout without CUDA dependencies."""
from pathlib import Path
import subprocess
import sys


def test_torch_can_import_cuda_bindings_from_checkout(tmp_path):
    # Recent PyTorch imports NVIDIA's cuda.bindings during its initialization.
    # Model that import order in a fresh interpreter. In particular, cuda is a
    # namespace package, so a checkout-level cuda.py would shadow it.
    torch_package = tmp_path / "torch"
    torch_package.mkdir()
    (torch_package / "__init__.py").write_text(
        "from cuda.bindings import driver\n"
        "loaded_driver = driver.marker\n",
        encoding="utf-8",
    )
    bindings = tmp_path / "cuda" / "bindings"
    bindings.mkdir(parents=True)
    (bindings / "__init__.py").write_text("", encoding="utf-8")
    (bindings / "driver.py").write_text(
        'marker = "nvidia-cuda-bindings"\n', encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; sys.path.append(sys.argv[1]); import torch; "
            "assert torch.loaded_driver == 'nvidia-cuda-bindings'",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
