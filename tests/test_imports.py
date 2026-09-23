"""Check import resolution from the source checkout without CUDA dependencies."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

import pytest


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


def test_pytest_runs_in_hyphenated_checkout(tmp_path):
    # A checkout named vc-attention must not become a Python package itself.
    # Reproduce pytest's package setup, not just collection (--collect-only).
    repository = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "vc-attention"
    checkout.mkdir()
    shutil.copy2(repository / "pyproject.toml", checkout)
    for module in repository.glob("*.py"):
        shutil.copy2(module, checkout)
    probe_tests = checkout / "tests"
    probe_tests.mkdir()
    (probe_tests / "test_installed_package.py").write_text(
        "from vc_attention import Config, attention\n"
        "import torch\n"
        "def test_public_api():\n"
        "    x = torch.ones(1, 1, 3, 16)\n"
        "    result, _ = attention(x, x, x, Config(quantize=False))\n"
        "    torch.testing.assert_close(result, x)\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    # Also support runners that load pytest from an isolated --target install.
    pytest_path = str(Path(pytest.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [pytest_path, env.get("PYTHONPATH")])
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "--tb=short"],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
