"""Window-based trajectory realignment stage."""

import csv
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from ._geometry import quat_to_rot
from .base import Stage, StageConfig, stage_output_path
from .trajectory_utils import (
    apply_rigid_2d_to_trajectory,
    estimate_rigid_2d,
    load_pose_csv,
    nearest_point_on_segments,
    write_pose_csv,
)

OUTPUT_CSV = "trajectory_window_aligned.csv"
OBSERVATIONS_CSV = "window_alignment_observations.csv"
TRANSFORM_JSON = "window_alignment_transform.json"


class WindowAlignStage(Stage):
    """Realign the trajectory using selected-frame window detections."""

    @property
    def name(self) -> str:
        return "window_align"

    @property
    def description(self) -> str:
        return "Realign trajectory from Window detections matched to floorplan edges"

    @property
    def requires_container(self) -> bool:
        return False

    @property
    def container_profile(self) -> str:
        return "host"

    @property
    def input_type(self) -> str:
        return "window_pose"

    @property
    def output_type(self) -> str:
        return "trajectory_csv"

    def run(self, runner, input_dir: Path, config: StageConfig) -> Path:
        base_traj_path = stage_output_path(config, "align") / "trajectory_aligned.csv"
        edges_path = stage_output_path(config, "floorplan_edges") / "floorplan_edges.csv"
        metadata_path = stage_output_path(config, "image_selector") / "selected_frames.json"
        window_pose_dir = stage_output_path(config, "window_pose")

        timestamps, xyz, quats = load_pose_csv(base_traj_path)
        edges = _load_edges(edges_path)
        frame_timestamps = _load_frame_timestamps(metadata_path)

        observations = _build_observations(
            window_pose_dir=window_pose_dir,
            frame_timestamps=frame_timestamps,
            traj_t=timestamps,
            traj_xyz=xyz,
            traj_quats=quats,
            edges=edges,
            max_dt=config.eval_max_time_delta,
        )
        if not observations:
            raise RuntimeError("No usable window observations found for window alignment")

        source_points = np.asarray(
            [point for obs in observations for point in (obs["observed_bl"], obs["observed_br"])],
            dtype=float,
        )
        target_points = np.asarray(
            [point for obs in observations for point in (obs["target_bl"], obs["target_br"])],
            dtype=float,
        )
        rotation, translation, yaw, residuals = estimate_rigid_2d(source_points, target_points)
        corrected_xyz, corrected_quats = apply_rigid_2d_to_trajectory(
            xyz,
            quats,
            rotation,
            translation,
        )

        stage_root = Path(tempfile.mkdtemp(prefix=f"{self.name}-", dir=runner.runtime_dir))
        output_dir = stage_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        out_csv = output_dir / OUTPUT_CSV
        write_pose_csv(out_csv, timestamps, corrected_xyz, corrected_quats)
        _write_observations(output_dir / OBSERVATIONS_CSV, observations)

        transform = {
            "base_trajectory": str(base_traj_path),
            "floorplan_edges": str(edges_path),
            "selected_frames": str(metadata_path),
            "window_pose_dir": str(window_pose_dir),
            "observations": len(observations),
            "points": int(len(source_points)),
            "yaw_correction_deg": math.degrees(yaw),
            "translation_correction_m": {
                "x": float(translation[0]),
                "y": float(translation[1]),
            },
            "rms_point_residual_m": float(math.sqrt(np.mean(residuals * residuals))),
            "mean_point_residual_m": float(np.mean(residuals)),
            "max_point_residual_m": float(np.max(residuals)),
        }
        (output_dir / TRANSFORM_JSON).write_text(
            json.dumps(transform, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        log_lines = [
            f"Base trajectory: {base_traj_path} ({len(timestamps)} poses)",
            f"Floorplan edges: {edges_path} ({len(edges)} segments)",
            f"Window observations: {len(observations)} frames / {len(source_points)} points",
            f"Yaw correction: {math.degrees(yaw):+.3f} deg",
            f"Translation correction: dx={translation[0]:+.3f}, dy={translation[1]:+.3f} m",
            f"Point residual RMS: {transform['rms_point_residual_m']:.4f} m",
            f"Output trajectory: {out_csv}",
        ]
        (output_dir / f"{self.name}.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        (output_dir / f"{self.name}.status").write_text("0", encoding="utf-8")
        for line in log_lines:
            print(f"[{self.name}] {line}")
        return output_dir


def _load_edges(path: Path) -> np.ndarray:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("x1"):
                continue
            parts = stripped.replace(",", " ").split()
            try:
                values = [float(part) for part in parts]
            except ValueError:
                continue
            if len(values) == 4:
                rows.append(values)
    if not rows:
        raise ValueError(f"No floorplan edges found in {path}")
    return np.asarray(rows, dtype=float)


def _load_frame_timestamps(path: Path) -> dict[str, float]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    frame_timestamps = {}
    for item in metadata.get("extracted", []):
        image_path = Path(item["image"])
        frame_timestamps[image_path.stem] = float(item["timestamp_ns"]) * 1e-9
    if not frame_timestamps:
        raise ValueError(f"No extracted frame timestamps found in {path}")
    return frame_timestamps


def _build_observations(
    *,
    window_pose_dir: Path,
    frame_timestamps: dict[str, float],
    traj_t: np.ndarray,
    traj_xyz: np.ndarray,
    traj_quats: np.ndarray,
    edges: np.ndarray,
    max_dt: float,
) -> list[dict]:
    observations = []
    for summary_path in sorted((window_pose_dir / "pose").glob("*/pose_summary.json")):
        frame = summary_path.parent.name
        timestamp = frame_timestamps.get(frame)
        if timestamp is None:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        try:
            bottom_left = np.asarray(summary["bottom_left_m"], dtype=float)
            bottom_right = np.asarray(summary["bottom_right_m"], dtype=float)
        except KeyError:
            continue
        traj_idx = int(np.argmin(np.abs(traj_t - timestamp)))
        dt = float(traj_t[traj_idx] - timestamp)
        if abs(dt) > max_dt:
            continue
        observed_bl = _window_point_to_map_xy(traj_xyz[traj_idx], traj_quats[traj_idx], bottom_left)
        observed_br = _window_point_to_map_xy(traj_xyz[traj_idx], traj_quats[traj_idx], bottom_right)
        target_bl, edge_bl, dist_bl = nearest_point_on_segments(observed_bl, edges)
        target_br, edge_br, dist_br = nearest_point_on_segments(observed_br, edges)
        observations.append(
            {
                "frame": frame,
                "timestamp": timestamp,
                "trajectory_timestamp": float(traj_t[traj_idx]),
                "dt_s": dt,
                "observed_bl": observed_bl,
                "observed_br": observed_br,
                "target_bl": target_bl,
                "target_br": target_br,
                "edge_bl": edge_bl,
                "edge_br": edge_br,
                "dist_bl": dist_bl,
                "dist_br": dist_br,
            }
        )
    return observations


def _window_point_to_map_xy(cam_xyz: np.ndarray, cam_quat: np.ndarray, point: np.ndarray) -> np.ndarray:
    rotation = quat_to_rot(*cam_quat)
    # Window pose summaries use x/lateral, y/up, z/forward in a gravity-aligned
    # camera-local frame. Project that local x/z displacement through the cam0
    # orientation so it lands in the same map xy frame as the trajectory.
    offset_xy = rotation[:2, 0] * point[0] + rotation[:2, 2] * point[2]
    return cam_xyz[:2] + offset_xy


def _write_observations(path: Path, observations: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "timestamp",
                "trajectory_timestamp",
                "dt_s",
                "observed_bl_x",
                "observed_bl_y",
                "observed_br_x",
                "observed_br_y",
                "target_bl_x",
                "target_bl_y",
                "target_br_x",
                "target_br_y",
                "edge_bl",
                "edge_br",
                "dist_bl_m",
                "dist_br_m",
            ]
        )
        for obs in observations:
            writer.writerow(
                [
                    obs["frame"],
                    f"{obs['timestamp']:.9f}",
                    f"{obs['trajectory_timestamp']:.9f}",
                    f"{obs['dt_s']:.9f}",
                    f"{obs['observed_bl'][0]:.9f}",
                    f"{obs['observed_bl'][1]:.9f}",
                    f"{obs['observed_br'][0]:.9f}",
                    f"{obs['observed_br'][1]:.9f}",
                    f"{obs['target_bl'][0]:.9f}",
                    f"{obs['target_bl'][1]:.9f}",
                    f"{obs['target_br'][0]:.9f}",
                    f"{obs['target_br'][1]:.9f}",
                    obs["edge_bl"],
                    obs["edge_br"],
                    f"{obs['dist_bl']:.9f}",
                    f"{obs['dist_br']:.9f}",
                ]
            )
