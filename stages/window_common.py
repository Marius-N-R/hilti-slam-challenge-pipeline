"""Shared helpers for the parallel Window image-processing stages."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from runtime_backend import ROOT, BindMount

WINDOW_ROOT_DEFAULT = str(ROOT / "third_party" / "window")
WINDOW_CONTAINER_PYTHON = Path("/opt/window_venv/bin/python")
WINDOW_MODEL_CACHE = ROOT / ".cache" / "window-models"
WINDOW_MODEL_CACHE_TARGET = "/window_models"


def window_root(config) -> Path:
    raw = getattr(config, "window_root", None) or WINDOW_ROOT_DEFAULT
    root = Path(str(raw)).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Window root does not exist: {root}")
    return root


def window_python(root: Path) -> Path:
    python = root / "3dv_venv" / "bin" / "python"
    if python.is_file():
        return python
    return WINDOW_CONTAINER_PYTHON


def window_mount(root: Path) -> BindMount:
    # Mount at the same absolute path so copied venvs keep their original layout.
    return BindMount(root, str(root), read_only=True)


def window_model_cache_mount() -> BindMount:
    WINDOW_MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    return BindMount(WINDOW_MODEL_CACHE, WINDOW_MODEL_CACHE_TARGET, read_only=False)


def groundingdino_checkpoint_path() -> Path:
    return Path(WINDOW_MODEL_CACHE_TARGET) / "groundingdino_swint_ogc.pth"


def window_container_env(root: Path, *python_paths: Path) -> dict[str, str]:
    env = base_env(root, *python_paths)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("HF_HOME", f"{WINDOW_MODEL_CACHE_TARGET}/huggingface")
    return env


def window_uses_gpu(config) -> bool:
    return getattr(config, "window_device", "auto") == "cuda"


def window_preflight_script(python_path: Path) -> str:
    return f'''
python_bin="{python_path}"
"$python_bin" - <<'PY'
import sys
print(sys.executable)
print(sys.version)
import encodings
PY
'''


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"\n[status] {result.returncode}\n")
    return result.returncode


def base_env(root: Path, *python_paths: Path) -> dict[str, str]:
    env = os.environ.copy()
    extra_paths = [str(path) for path in python_paths]
    if extra_paths:
        existing = env.get("PYTHONPATH", "")
        if existing:
            extra_paths.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(extra_paths)
    env.setdefault("MPLBACKEND", "Agg")
    env["WINDOW_ROOT"] = str(root)
    return env


def selected_image_paths(input_dir: Path) -> list[Path]:
    images_dir = input_dir / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Expected selected images directory: {images_dir}")
    images = sorted(
        [
            *images_dir.glob("*.png"),
            *images_dir.glob("*.jpg"),
            *images_dir.glob("*.jpeg"),
        ]
    )
    if not images:
        raise FileNotFoundError(f"No selected images found in {images_dir}")
    return images
