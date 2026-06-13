"""Window mask rectification stage."""

import json
from pathlib import Path

from runtime_backend import ExecutionSpec

from .base import Stage, StageConfig
from .window_common import (
    window_container_env,
    window_mount,
    window_preflight_script,
    window_python,
    window_root,
    window_uses_gpu,
)


class WindowRectifyStage(Stage):
    """Rectify Window SAM3 window masks using py360convert."""

    @property
    def name(self) -> str:
        return "window_rectify"

    @property
    def description(self) -> str:
        return "Rectify Window SAM3 masks with py360convert"

    @property
    def requires_container(self) -> bool:
        return True

    @property
    def container_profile(self) -> str:
        return "window"

    def run(self, runner, input_dir: Path, config: StageConfig) -> Path:
        root = window_root(config)
        python = window_python(root)
        py360_root = root / "py360convert"

        args = {
            "python": str(python),
            "root": str(root),
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
{python} /stage_runtime/window_rectify_driver.py '{json.dumps(args)}'
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
                command=["/bin/bash", "/stage_runtime/run_window_rectify.sh"],
                files={
                    "run_window_rectify.sh": wrapper,
                    "window_rectify_driver.py": _RECTIFY_DRIVER,
                    "window_rectify_runner.py": _RECTIFY_RUNNER,
                },
                env=window_container_env(root, py360_root),
                extra_mounts=[window_mount(root)],
                workdir=str(root),
                use_gpu=window_uses_gpu(config),
            ),
        )


_RECTIFY_DRIVER = r'''#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path


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


def main():
    args = json.loads(sys.argv[1])
    image_paths = selected_image_paths(Path("/output"))
    for image_path in image_paths:
        mask_path = Path("/output") / "sam3" / image_path.stem / "windows_masks.npy"
        frame_output = Path("/output") / "rectified" / image_path.stem
        command = [
            args["python"],
            "/stage_runtime/window_rectify_runner.py",
            "--mask",
            str(mask_path),
            "--output-dir",
            str(frame_output),
        ]
        print("$ " + " ".join(command), flush=True)
        status = subprocess.run(command, cwd=args["root"], check=False).returncode
        if status != 0:
            return status
    print(f"[window_rectify] Processed {len(image_paths)} selected image(s)")
    print("[window_rectify] Output: /output/rectified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


_RECTIFY_RUNNER = r'''#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import py360convert


def as_mask_image(array):
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D mask or single-channel mask, got shape {array.shape}")
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    if np.issubdtype(array.dtype, np.floating):
        if array.max(initial=0.0) <= 1.0:
            array = array * 255.0
        return np.clip(array, 0, 255).astype(np.uint8)
    return np.clip(array, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    mask = as_mask_image(np.load(args.mask))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    perspective_img = py360convert.e2p(
        e_img=mask,
        fov_deg=(200, 200),
        u_deg=0,
        v_deg=0,
        out_hw=mask.shape[:2],
        in_rot_deg=0,
        mode="nearest",
    )
    Image.fromarray(as_mask_image(perspective_img)).save(output_dir / "mask_undistorted.png")


if __name__ == "__main__":
    main()
'''
