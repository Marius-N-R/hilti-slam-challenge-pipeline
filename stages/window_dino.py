"""Window GroundingDINO stage for selected images."""

import json
from pathlib import Path

from runtime_backend import ExecutionSpec

from .base import Stage, StageConfig
from .window_common import (
    groundingdino_checkpoint_path,
    window_container_env,
    window_model_cache_mount,
    window_mount,
    window_preflight_script,
    window_python,
    window_root,
    window_uses_gpu,
)


class WindowDinoStage(Stage):
    """Run the Window GroundingDINO code on selected images."""

    @property
    def name(self) -> str:
        return "window_dino"

    @property
    def description(self) -> str:
        return "Run Window GroundingDINO window detection on selected images"

    @property
    def requires_container(self) -> bool:
        return True

    @property
    def container_profile(self) -> str:
        return "window"

    @property
    def input_type(self) -> str:
        return "images"

    @property
    def output_type(self) -> str:
        return "directory"

    def run(self, runner, input_dir: Path, config: StageConfig) -> Path:
        root = window_root(config)
        python = window_python(root)
        grounding_root = root / "GroundingDINO"
        config_path = grounding_root / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
        checkpoint_path = groundingdino_checkpoint_path()
        if not config_path.is_file():
            raise FileNotFoundError(f"GroundingDINO config not found: {config_path}")

        args = {
            "python": str(python),
            "grounding_root": str(grounding_root),
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "prompt": config.window_prompt,
            "box_threshold": config.window_box_threshold,
            "text_threshold": config.window_text_threshold,
            "cpu_only": config.window_device != "cuda",
        }
        wrapper = f"""#!/bin/bash
set +e
mkdir -p /output
cp -r /input/. /output/
exec > >(tee /output/{self.name}.log) 2>&1
{window_preflight_script(python)}
PREFLIGHT_STATUS=$?
if [ "$PREFLIGHT_STATUS" -ne 0 ]; then
  echo "$PREFLIGHT_STATUS" > /output/{self.name}.status
  exit 0
fi
{python} /stage_runtime/window_dino_driver.py '{json.dumps(args)}'
STATUS=$?
echo "$STATUS" > /output/{self.name}.status
exit 0
"""
        return runner.run_stage(
            container_profile=self.container_profile,
            input_dir=input_dir,
            config=config,
            spec=ExecutionSpec(
                stage_name=self.name,
                command=["/bin/bash", "/stage_runtime/run_window_dino.sh"],
                files={
                    "run_window_dino.sh": wrapper,
                    "window_dino_driver.py": _DINO_DRIVER,
                },
                env=window_container_env(root, grounding_root),
                extra_mounts=[window_mount(root), window_model_cache_mount()],
                workdir=str(grounding_root),
                use_gpu=window_uses_gpu(config),
            ),
        )


_DINO_DRIVER = r'''#!/usr/bin/env python3
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

DINO_CHECKPOINT_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    "v0.1.0-alpha/groundingdino_swint_ogc.pth"
)


def selected_image_paths(root):
    images_dir = root / "images"
    images = sorted([
        *images_dir.glob("*.png"),
        *images_dir.glob("*.jpg"),
        *images_dir.glob("*.jpeg"),
    ])
    if not images:
        raise FileNotFoundError(f"No selected images found in {images_dir}")
    return images


def ensure_checkpoint(path):
    path = Path(path)
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    print(f"[window_dino] Downloading GroundingDINO checkpoint to: {path}", flush=True)
    urllib.request.urlretrieve(DINO_CHECKPOINT_URL, temp_path)
    temp_path.replace(path)


def main():
    args = json.loads(sys.argv[1])
    ensure_checkpoint(args["checkpoint_path"])
    image_paths = selected_image_paths(Path("/output"))
    for image_path in image_paths:
        frame_output = Path("/output") / "grounding_dino" / image_path.stem
        command = [
            args["python"],
            "demo/inference_on_a_image.py",
            "-c",
            args["config_path"],
            "-p",
            args["checkpoint_path"],
            "-i",
            str(image_path),
            "-o",
            str(frame_output),
            "-t",
            args["prompt"],
            "--box_threshold",
            str(args["box_threshold"]),
            "--text_threshold",
            str(args["text_threshold"]),
        ]
        if args["cpu_only"]:
            command.append("--cpu-only")
        print("$ " + " ".join(command), flush=True)
        status = subprocess.run(command, cwd=args["grounding_root"], check=False).returncode
        if status != 0:
            return status
    print(f"[window_dino] Processed {len(image_paths)} selected image(s)")
    print("[window_dino] Output: /output/grounding_dino")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
