"""Public window-perception stage with normalized output artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .base import Stage, StageConfig


class WindowsStage(Stage):
    @property
    def name(self) -> str:
        return "windows"

    @property
    def description(self) -> str:
        return "Finalize normalized window-perception artifacts"

    @property
    def container_profile(self) -> str:
        return "windows"

    @property
    def input_type(self) -> str:
        return "directory"

    @property
    def output_type(self) -> str:
        return "directory"

    def run(
        self,
        runner,
        input_dir: Path,
        config: StageConfig,
    ) -> Path:
        windows_dir = input_dir / "windows"
        windows_dir.mkdir(parents=True, exist_ok=True)

        artifacts = {
            "boxes_npy": (
                input_dir / "grounding_dino" / "bb.npy",
                windows_dir / "boxes.npy",
            ),
            "dino_input_image": (
                input_dir / "grounding_dino" / "raw_image.jpg",
                windows_dir / "dino_input.jpg",
            ),
            "dino_overlay_image": (
                input_dir / "grounding_dino" / "pred.jpg",
                windows_dir / "dino_overlay.jpg",
            ),
            "masks_npy": (
                input_dir / "windows_masks.npy",
                windows_dir / "masks.npy",
            ),
            "segmented_overlay": (
                input_dir / "windows_segmented.png",
                windows_dir / "segmented_overlay.png",
            ),
            "rectified_mask": (
                input_dir / "undistorted" / "mask_undistorted.png",
                windows_dir / "rectified_mask.png",
            ),
        }

        copied_artifacts: dict[str, str] = {}
        for key, (source, target) in artifacts.items():
            if not source.exists():
                continue
            shutil.copy2(source, target)
            copied_artifacts[key] = str(target.relative_to(input_dir))

        required = ["boxes_npy", "masks_npy", "rectified_mask"]
        missing = [name for name in required if name not in copied_artifacts]
        if missing:
            raise FileNotFoundError(
                "Missing required window artifacts for finalization: "
                + ", ".join(missing)
            )

        stats: dict[str, object] = {}
        boxes_path = input_dir / copied_artifacts["boxes_npy"]
        masks_path = input_dir / copied_artifacts["masks_npy"]

        try:
            boxes = np.load(boxes_path)
            stats["num_boxes"] = int(len(boxes))
            stats["boxes_shape"] = list(boxes.shape)
        except Exception:
            pass

        try:
            masks = np.load(masks_path)
            stats["masks_shape"] = list(masks.shape)
            stats["mask_dtype"] = str(masks.dtype)
        except Exception:
            pass

        metadata = {
            "schema_version": 1,
            "stage": self.name,
            "input_name": str(config.extra.get("current_input_name", "")),
            "device": config.windows_device,
            "prompt": config.windows_prompt,
            "grounding_dino": {
                "box_threshold": config.windows_box_threshold,
                "text_threshold": config.windows_text_threshold,
            },
            "artifacts": copied_artifacts,
            "stats": stats,
        }

        metadata_path = windows_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"[windows] Output bundle: {windows_dir}")
        return input_dir
