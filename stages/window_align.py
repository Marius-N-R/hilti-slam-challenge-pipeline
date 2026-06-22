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
SKIPPED_CSV = "window_alignment_skipped.csv"
TRANSFORM_JSON = "window_alignment_transform.json"
BASE_TRAJECTORY_CSV = "trajectory_aligned.csv"
FLOORPLAN_OFFSET_FILENAME = "floorplan_offset.txt"
WINDOW_FLOORPLAN_OFFSET_FILENAME = "window_floorplan_offset.txt"


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
        base_traj_path = _resolve_base_trajectory(config)
        edges_path = stage_output_path(config, "floorplan_edges") / "floorplan_edges.csv"
        metadata_path = stage_output_path(config, "image_selector") / "selected_frames.json"
        window_pose_dir = stage_output_path(config, "window_pose")

        timestamps, xyz, quats = load_pose_csv(base_traj_path)
        edges, edge_offset_note = _load_edges(edges_path, config)
        frame_timestamps = _load_frame_timestamps(metadata_path)

        observations, skipped = _build_observations(
            window_pose_dir=window_pose_dir,
            frame_timestamps=frame_timestamps,
            traj_t=timestamps,
            traj_xyz=xyz,
            traj_quats=quats,
            edges=edges,
            max_dt=config.eval_max_time_delta,
            max_observation_width=config.window_max_observation_width,
            max_observation_distance=config.window_max_observation_distance,
            max_edge_distance=config.window_max_edge_distance,
        )
        if not observations:
            stage_root = Path(tempfile.mkdtemp(prefix=f"{self.name}-", dir=runner.runtime_dir))
            output_dir = stage_root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_skipped(output_dir / SKIPPED_CSV, skipped)
            log_lines = [
                f"Base trajectory: {base_traj_path} ({len(timestamps)} poses)",
                f"Floorplan edges: {edges_path} ({len(edges)} segments; {edge_offset_note})",
                f"Window observations: 0 frames / 0 points",
                f"Skipped window observations: {len(skipped)}",
                f"Skipped details: {output_dir / SKIPPED_CSV}",
            ]
            (output_dir / f"{self.name}.log").write_text(
                "\n".join(log_lines) + "\n", encoding="utf-8"
            )
            (output_dir / f"{self.name}.status").write_text("1", encoding="utf-8")
            for line in log_lines:
                print(f"[{self.name}] {line}")
            raise RuntimeError(
                "No usable window observations found for window alignment; "
                f"see {output_dir / SKIPPED_CSV}"
            )

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
        _write_skipped(output_dir / SKIPPED_CSV, skipped)

        transform = {
            "base_trajectory": str(base_traj_path),
            "start_alignment_enabled": bool(config.align_start_position),
            "base_coordinate_frame": (
                "map"
                if config.align_start_position
                else "openvins_world_cam0"
            ),
            "floorplan_edges": str(edges_path),
            "floorplan_edges_offset_note": edge_offset_note,
            "selected_frames": str(metadata_path),
            "window_pose_dir": str(window_pose_dir),
            "observations": len(observations),
            "skipped_observations": len(skipped),
            "filters": {
                "max_observation_width_m": float(config.window_max_observation_width),
                "max_observation_distance_m": float(config.window_max_observation_distance),
                "max_edge_distance_m": float(config.window_max_edge_distance),
            },
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
            (
                "Start alignment: enabled; Window observations are projected in map frame"
                if config.align_start_position
                else (
                    "Start alignment: disabled; Window observations are projected in "
                    "OpenVINS-world cam0 coordinates before floorplan matching"
                )
            ),
            f"Floorplan edges: {edges_path} ({len(edges)} segments; {edge_offset_note})",
            f"Window observations: {len(observations)} frames / {len(source_points)} points",
            f"Skipped window observations: {len(skipped)}",
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


def _resolve_base_trajectory(config: StageConfig) -> Path:
    """Return the cam0 trajectory produced by the align stage.

    The Window observations are camera-relative, so the base trajectory must be
    the align-stage CSV. That stage always converts OpenVINS IMU poses to cam0
    and, when requested, applies the initial-position map transform.
    """
    try:
        path = stage_output_path(config, "align") / BASE_TRAJECTORY_CSV
    except Exception as exc:
        raise FileNotFoundError(
            "Window alignment requires align/trajectory_aligned.csv. "
            "Run the `align` stage before `window_align`."
        ) from exc
    if path.is_file():
        return path
    raise FileNotFoundError(
        f"Window alignment requires {path}. Run `slam align` before `window_align`; "
        "include --align-start-position to place the Window pipeline in the initial "
        "map frame."
    )


def _load_edges(path: Path, config: StageConfig) -> tuple[np.ndarray, str]:
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
    edges = np.asarray(rows, dtype=float)
    delta, note = _window_edge_offset_delta(config)
    if delta is not None:
        edges[:, [0, 2]] += delta[0]
        edges[:, [1, 3]] += delta[1]
    return edges, note


def _window_edge_offset_delta(config: StageConfig) -> tuple[np.ndarray | None, str]:
    original = config.extra.get("current_input_path", "")
    if not original:
        return None, "using floorplan_edges offset"
    original_dir = Path(original)
    window_offset_path = original_dir / WINDOW_FLOORPLAN_OFFSET_FILENAME
    if not window_offset_path.is_file():
        return None, "using floorplan_edges offset"

    floor_offset_path = original_dir / FLOORPLAN_OFFSET_FILENAME
    floor_offset = _load_offset_pair(floor_offset_path) if floor_offset_path.is_file() else (0.0, 0.0)
    window_offset = _load_offset_pair(window_offset_path)
    delta = np.asarray(
        [
            float(window_offset[0] - floor_offset[0]),
            float(window_offset[1] - floor_offset[1]),
        ],
        dtype=float,
    )
    return (
        delta,
        (
            f"window-specific offset {window_offset_path.name}: "
            f"({window_offset[0]:+.3f}, {window_offset[1]:+.3f}) m; "
            f"delta from floorplan_edges=({delta[0]:+.3f}, {delta[1]:+.3f}) m"
        ),
    )


def _load_offset_pair(path: Path) -> tuple[float, float]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            try:
                values = [float(part) for part in parts]
            except ValueError:
                continue
            if len(values) == 2:
                return float(values[0]), float(values[1])
    raise ValueError(f"No (offset_x, offset_y) row found in {path}")


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
    max_observation_width: float,
    max_observation_distance: float,
    max_edge_distance: float,
) -> tuple[list[dict], list[dict]]:
    observations = []
    skipped = []
    for summary_path in sorted((window_pose_dir / "pose").glob("*/pose_summary.json")):
        frame = summary_path.parent.name
        timestamp = frame_timestamps.get(frame)
        if timestamp is None:
            skipped.append({"frame": frame, "reason": "missing_frame_timestamp"})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        reason = _window_summary_rejection_reason(
            summary,
            max_observation_width=max_observation_width,
            max_observation_distance=max_observation_distance,
        )
        if reason is not None:
            skipped.append({"frame": frame, "reason": reason})
            continue
        try:
            bottom_left = np.asarray(summary["bottom_left_m"], dtype=float)
            bottom_right = np.asarray(summary["bottom_right_m"], dtype=float)
        except KeyError:
            skipped.append({"frame": frame, "reason": "missing_bottom_points"})
            continue
        traj_idx = int(np.argmin(np.abs(traj_t - timestamp)))
        dt = float(traj_t[traj_idx] - timestamp)
        if abs(dt) > max_dt:
            skipped.append(
                {
                    "frame": frame,
                    "reason": (
                        "timestamp_dt_exceeds_limit:"
                        f"dt={dt:.6f},"
                        f"frame_t={timestamp:.6f},"
                        f"nearest_traj_t={float(traj_t[traj_idx]):.6f},"
                        f"traj_range={float(traj_t[0]):.6f}..{float(traj_t[-1]):.6f}"
                    ),
                }
            )
            continue
        observed_bl = _window_point_to_map_xy(
            traj_xyz[traj_idx],
            traj_quats[traj_idx],
            bottom_left,
        )
        observed_br = _window_point_to_map_xy(
            traj_xyz[traj_idx],
            traj_quats[traj_idx],
            bottom_right,
        )
        target_bl, edge_bl, dist_bl = nearest_point_on_segments(observed_bl, edges)
        target_br, edge_br, dist_br = nearest_point_on_segments(observed_br, edges)
        if max(dist_bl, dist_br) > max_edge_distance:
            skipped.append(
                {
                    "frame": frame,
                    "reason": (
                        "edge_distance_exceeds_limit:"
                        f"bl={dist_bl:.3f},br={dist_br:.3f}"
                    ),
                }
            )
            continue
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
    return observations, skipped


def _window_summary_rejection_reason(
    summary: dict,
    *,
    max_observation_width: float,
    max_observation_distance: float,
) -> str | None:
    try:
        bottom_left = np.asarray(summary["bottom_left_m"], dtype=float)
        bottom_right = np.asarray(summary["bottom_right_m"], dtype=float)
    except KeyError:
        return "missing_bottom_points"
    if bottom_left.shape != (3,) or bottom_right.shape != (3,):
        return "invalid_bottom_point_shape"
    if not np.isfinite(bottom_left).all() or not np.isfinite(bottom_right).all():
        return "nonfinite_bottom_points"

    local_distance = max(float(np.linalg.norm(bottom_left)), float(np.linalg.norm(bottom_right)))
    if local_distance > max_observation_distance:
        return f"local_distance_exceeds_limit:{local_distance:.3f}"

    width_candidates = [
        float(np.linalg.norm(bottom_right - bottom_left)),
        float(summary.get("estimated_width_m", 0.0) or 0.0),
        float(summary.get("estimated_width_horizontal_m", 0.0) or 0.0),
    ]
    finite_widths = [abs(value) for value in width_candidates if np.isfinite(value)]
    if not finite_widths:
        return "nonfinite_width"
    width = max(finite_widths)
    if width > max_observation_width:
        return f"width_exceeds_limit:{width:.3f}"
    return None


def _window_point_to_map_xy(cam_xyz: np.ndarray, cam_quat: np.ndarray, point: np.ndarray) -> np.ndarray:
    rotation = quat_to_rot(*cam_quat)
    # Window pose summaries use x/lateral, y/up, and negative z as forward in a
    # gravity-aligned camera-local frame. Project local x/(-z) displacement
    # through cam0 orientation so it lands in the trajectory's map xy frame.
    offset_xy = rotation[:2, 0] * point[0] - rotation[:2, 2] * point[2]
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


def _write_skipped(path: Path, skipped: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "reason"])
        writer.writeheader()
        for item in skipped:
            writer.writerow(
                {
                    "frame": item.get("frame", ""),
                    "reason": item.get("reason", ""),
                }
            )
