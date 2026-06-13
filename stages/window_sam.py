"""Window SAM3 stage for selected image detections."""

import json
from pathlib import Path

from runtime_backend import ExecutionSpec

from .base import Stage, StageConfig
from .window_common import (
    window_container_env,
    window_model_cache_mount,
    window_mount,
    window_preflight_script,
    window_python,
    window_root,
    window_uses_gpu,
)


class WindowSamStage(Stage):
    """Run the Window SAM3 segmentation code on GroundingDINO boxes."""

    @property
    def name(self) -> str:
        return "window_sam"

    @property
    def description(self) -> str:
        return "Run Window SAM3 window segmentation on selected images"

    @property
    def requires_container(self) -> bool:
        return True

    @property
    def container_profile(self) -> str:
        return "window"

    @property
    def input_type(self) -> str:
        return "directory"

    @property
    def output_type(self) -> str:
        return "directory"

    def run(self, runner, input_dir: Path, config: StageConfig) -> Path:
        root = window_root(config)
        python = window_python(root)
        sam_root = root / "sam3"
        if not sam_root.is_dir():
            raise FileNotFoundError(f"SAM3 root not found: {sam_root}")

        args = {
            "python": str(python),
            "root": str(root),
            "device": config.window_device,
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
{python} /stage_runtime/window_sam_driver.py '{json.dumps(args)}'
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
                command=["/bin/bash", "/stage_runtime/run_window_sam.sh"],
                files={
                    "run_window_sam.sh": wrapper,
                    "window_sam_driver.py": _SAM_DRIVER,
                    "window_sam_runner.py": _SAM_RUNNER,
                },
                env=window_container_env(root, sam_root),
                extra_mounts=[window_mount(root), window_model_cache_mount()],
                workdir=str(root),
                use_gpu=window_uses_gpu(config),
            ),
        )


_SAM_DRIVER = r'''#!/usr/bin/env python3
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
        boxes_path = Path("/output") / "grounding_dino" / image_path.stem / "bb.npy"
        frame_output = Path("/output") / "sam3" / image_path.stem
        command = [
            args["python"],
            "/stage_runtime/window_sam_runner.py",
            "--image",
            str(image_path),
            "--boxes",
            str(boxes_path),
            "--output-dir",
            str(frame_output),
            "--device",
            args["device"],
        ]
        print("$ " + " ".join(command), flush=True)
        status = subprocess.run(command, cwd=args["root"], check=False).returncode
        if status != 0:
            return status
    print(f"[window_sam] Processed {len(image_paths)} selected image(s)")
    print("[window_sam] Output: /output/sam3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


_SAM_RUNNER = r'''#!/usr/bin/env python3
import argparse
import contextlib
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def is_cuda_device(value):
    if value == "cuda":
        return True
    if isinstance(value, torch.device):
        return value.type == "cuda"
    return False


@contextlib.contextmanager
def cpu_safe_sam3_build(device):
    if device != "cpu":
        yield
        return

    patched_names = ("arange", "empty", "full", "ones", "tensor", "zeros")
    originals = {name: getattr(torch, name) for name in patched_names}
    original_tensor_cuda = torch.Tensor.cuda
    original_module_cuda = torch.nn.Module.cuda

    def cpu_redirect_factory(name):
        original = originals[name]

        def cpu_redirect(*args, **kwargs):
            if is_cuda_device(kwargs.get("device")):
                kwargs = dict(kwargs)
                kwargs["device"] = "cpu"
            return original(*args, **kwargs)

        return cpu_redirect

    def cuda_cpu_noop(self, *args, **kwargs):
        return self

    for name in patched_names:
        setattr(torch, name, cpu_redirect_factory(name))
    torch.Tensor.cuda = cuda_cpu_noop
    torch.nn.Module.cuda = cuda_cpu_noop
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(torch, name, original)
        torch.Tensor.cuda = original_tensor_cuda
        torch.nn.Module.cuda = original_module_cuda


@contextlib.contextmanager
def cpu_safe_sam3_inference(device):
    if device != "cpu":
        yield
        return

    import sam3.model.vitdet as vitdet
    import sam3.perflib.fused as fused

    patched_names = ("arange", "empty", "full", "ones", "tensor", "zeros")
    originals = {name: getattr(torch, name) for name in patched_names}
    original_tensor_cuda = torch.Tensor.cuda
    original_tensor_pin_memory = torch.Tensor.pin_memory
    original_module_cuda = torch.nn.Module.cuda
    original_fused_addmm_act = fused.addmm_act
    original_vitdet_addmm_act = vitdet.addmm_act

    def cpu_redirect_factory(name):
        original = originals[name]

        def cpu_redirect(*args, **kwargs):
            if is_cuda_device(kwargs.get("device")):
                kwargs = dict(kwargs)
                kwargs["device"] = "cpu"
            return original(*args, **kwargs)

        return cpu_redirect

    def cuda_cpu_noop(self, *args, **kwargs):
        return self

    def pin_memory_cpu_noop(self, *args, **kwargs):
        return self

    def addmm_act_float32(activation, linear, mat1):
        y = linear(mat1.float())
        if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
            return torch.nn.functional.relu(y)
        if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
            return torch.nn.functional.gelu(y)
        raise ValueError(f"Unexpected activation {activation}")

    for name in patched_names:
        setattr(torch, name, cpu_redirect_factory(name))
    torch.Tensor.cuda = cuda_cpu_noop
    torch.Tensor.pin_memory = pin_memory_cpu_noop
    torch.nn.Module.cuda = cuda_cpu_noop
    fused.addmm_act = addmm_act_float32
    vitdet.addmm_act = addmm_act_float32
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(torch, name, original)
        torch.Tensor.cuda = original_tensor_cuda
        torch.Tensor.pin_memory = original_tensor_pin_memory
        torch.nn.Module.cuda = original_module_cuda
        fused.addmm_act = original_fused_addmm_act
        vitdet.addmm_act = original_vitdet_addmm_act


def crop_center_band(image):
    width, height = image.size
    left = int(width * 0.25)
    right = int(width * 0.75)
    center_img = image.crop((left, 0, right, height))
    result_img = Image.new(image.mode, (width, height), (0, 0, 0))
    result_img.paste(center_img, (left, 0))
    return result_img


def filter_boxes(boxes):
    filtered = []
    for box in boxes:
        if box[2] > 0.6 or box[3] > 0.6:
            continue
        if box[2] < 0.025 or box[3] < 0.025:
            continue
        duplicate = any(np.linalg.norm(box[:2] - accepted[:2]) < 0.05 for accepted in filtered)
        if not duplicate:
            filtered.append(box)
    return filtered


def save_largest_mask(image, masks, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = image.size
    mask_img = np.zeros((height, width, 1), dtype=np.uint8)
    if torch.is_tensor(masks):
        masks = masks.cpu().float().numpy()
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    if len(masks) > 0:
        binary_masks = [(mask > 0.5).astype(np.uint8) for mask in masks]
        largest_mask = max(binary_masks, key=lambda mask: mask.sum())
        mask_img = largest_mask.reshape(height, width, 1) * 255
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.gca().imshow(largest_mask, alpha=0.45)
        plt.axis("off")
        plt.savefig(output_dir / "windows_segmented.png", bbox_inches="tight", pad_inches=0)
        plt.close()
    np.save(output_dir / "windows_masks.npy", mask_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--boxes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA for SAM3 but no GPU is available")

    image = crop_center_band(Image.open(args.image).convert("RGB"))
    boxes = np.load(args.boxes).astype(np.float32)
    filtered_boxes = filter_boxes(boxes)
    print(f"[window_sam] Filtered boxes: {len(filtered_boxes)}")

    if not os.environ.get("HF_TOKEN", "").strip():
        print("[window_sam] HF_TOKEN is not set; SAM3 model loading may fail if no local cache exists")
    with cpu_safe_sam3_build(device):
        model = build_sam3_image_model(device=device)

    autocast_enabled = device == "cuda"
    with cpu_safe_sam3_inference(device):
        processor = Sam3Processor(model, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            inference_state = processor.set_image(image)
            processor.reset_all_prompts(inference_state)
            for box in filtered_boxes:
                processor.add_geometric_prompt(box=box.tolist(), label=True, state=inference_state)
            output = processor._forward_grounding(state=inference_state)

    save_largest_mask(image, output["masks"], Path(args.output_dir))


if __name__ == "__main__":
    main()
'''
