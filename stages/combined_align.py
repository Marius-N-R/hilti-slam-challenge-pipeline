"""Fuse floorplan and window realignments into one trajectory."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np

from .base import Stage, StageConfig, stage_output_path
from .trajectory_utils import (
    apply_rigid_2d_to_trajectory,
    estimate_rigid_2d,
    load_pose_csv,
    match_by_timestamp,
    rotation_from_yaw,
    weighted_yaw,
    write_pose_csv,
)

OUTPUT_CSV = "trajectory_combined_aligned.csv"
SUMMARY_JSON = "combined_alignment.json"


class CombinedAlignStage(Stage):
    """Combine floorplan and window trajectory realignments."""

    @property
    def name(self) -> str:
        return "combined_align"

    @property
    def description(self) -> str:
        return "Fuse floorplan and Window realignments with configurable weights"

    @property
    def requires_container(self) -> bool:
        return False

    @property
    def container_profile(self) -> str:
        return "host"

    @property
    def input_type(self) -> str:
        return "trajectory_csv"

    @property
    def output_type(self) -> str:
        return "trajectory_csv"

    def run(self, runner, input_dir: Path, config: StageConfig) -> Path:
        base_path = stage_output_path(config, "align") / "trajectory_aligned.csv"
        floor_path = stage_output_path(config, "floorplan_align") / "trajectory_floor_aligned.csv"
        window_path = stage_output_path(config, "window_align") / "trajectory_window_aligned.csv"

        base_t, base_xyz, base_quats = load_pose_csv(base_path)
        floor_transform = _estimate_named_transform(
            name="floorplan",
            base_t=base_t,
            base_xyz=base_xyz,
            target_path=floor_path,
            max_dt=config.eval_max_time_delta,
        )
        window_transform = _estimate_named_transform(
            name="window",
            base_t=base_t,
            base_xyz=base_xyz,
            target_path=window_path,
            max_dt=config.eval_max_time_delta,
        )

        floor_weight = max(0.0, float(config.floorplan_realign_weight))
        window_weight = max(0.0, float(config.window_realign_weight))
        if floor_weight == 0.0 and window_weight == 0.0:
            raise ValueError("At least one realignment weight must be greater than zero")

        blend_yaw = weighted_yaw(
            [floor_transform["yaw_rad"], window_transform["yaw_rad"]],
            [floor_weight, window_weight],
        )
        denom = floor_weight + window_weight
        blend_translation = (
            floor_weight * floor_transform["translation"]
            + window_weight * window_transform["translation"]
        ) / denom
        blend_rotation = rotation_from_yaw(blend_yaw)

        combined_xyz, combined_quats = apply_rigid_2d_to_trajectory(
            base_xyz,
            base_quats,
            blend_rotation,
            blend_translation,
        )

        stage_root = Path(tempfile.mkdtemp(prefix=f"{self.name}-", dir=runner.runtime_dir))
        output_dir = stage_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_csv = output_dir / OUTPUT_CSV
        write_pose_csv(output_csv, base_t, combined_xyz, combined_quats)
        summary = {
            "base_trajectory": str(base_path),
            "floorplan_trajectory": str(floor_path),
            "window_trajectory": str(window_path),
            "weights": {
                "floorplan": floor_weight,
                "window": window_weight,
            },
            "floorplan_transform": _serializable_transform(floor_transform),
            "window_transform": _serializable_transform(window_transform),
            "combined_transform": {
                "yaw_deg": math.degrees(blend_yaw),
                "translation_m": {
                    "x": float(blend_translation[0]),
                    "y": float(blend_translation[1]),
                },
            },
            "output_trajectory": str(output_csv),
        }
        (output_dir / SUMMARY_JSON).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        log_lines = [
            f"Base trajectory: {base_path} ({len(base_t)} poses)",
            f"Floorplan trajectory: {floor_path}",
            f"Window trajectory: {window_path}",
            f"Weights: floorplan={floor_weight:.3f}, window={window_weight:.3f}",
            f"Combined yaw correction: {math.degrees(blend_yaw):+.3f} deg",
            (
                "Combined translation correction: "
                f"dx={blend_translation[0]:+.3f}, dy={blend_translation[1]:+.3f} m"
            ),
            f"Output trajectory: {output_csv}",
        ]
        (output_dir / f"{self.name}.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        (output_dir / f"{self.name}.status").write_text("0", encoding="utf-8")
        for line in log_lines:
            print(f"[{self.name}] {line}")
        return output_dir


def _estimate_named_transform(
    name: str,
    base_t: np.ndarray,
    base_xyz: np.ndarray,
    target_path: Path,
    max_dt: float,
):
    target_t, target_xyz, _ = load_pose_csv(target_path)
    pairs = match_by_timestamp(base_t, target_t, max_dt=max_dt)
    if len(pairs) < 1:
        raise RuntimeError(f"No timestamp matches found for {name} trajectory: {target_path}")
    base_points = np.asarray([base_xyz[base_idx, :2] for base_idx, _, _ in pairs], dtype=float)
    target_points = np.asarray([target_xyz[target_idx, :2] for _, target_idx, _ in pairs], dtype=float)
    rotation, translation, yaw, residuals = estimate_rigid_2d(base_points, target_points)
    return {
        "name": name,
        "path": str(target_path),
        "matches": len(pairs),
        "rotation": rotation,
        "translation": translation,
        "yaw_rad": yaw,
        "rms_residual_m": float(math.sqrt(np.mean(residuals * residuals))),
        "max_abs_dt_s": float(max(abs(dt) for _, _, dt in pairs)),
    }


def _serializable_transform(transform: dict) -> dict:
    return {
        "path": transform["path"],
        "matches": transform["matches"],
        "yaw_deg": math.degrees(transform["yaw_rad"]),
        "translation_m": {
            "x": float(transform["translation"][0]),
            "y": float(transform["translation"][1]),
        },
        "rms_residual_m": transform["rms_residual_m"],
        "max_abs_dt_s": transform["max_abs_dt_s"],
    }
