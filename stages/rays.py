"""Rays stage - back-projects 2D line detections to 3D rays in the world frame.

Adapted from rework/Floorplan-Alignment/src/vizualize_results_edge_alignment.py.
The EUCM back-projection (`calculateRay`) is kept verbatim. Since the upstream
`align` stage now outputs cam0 pose directly (in map frame), this stage simply
transforms cam-frame rays via T_wrld_cam taken straight from each pose row -
no extrinsic composition required.
"""

import csv
import math
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np

from ._geometry import quat_to_rot
from .base import Stage, StageConfig, stage_output_path

OUTPUT_CSV = "rays.csv"
LINES_FILENAME = "lines.csv"
PCA_ALIGNED_TRAJ_FILENAME = "trajectory_pca_aligned.csv"
ALIGNED_TRAJ_FILENAME = "trajectory_aligned.csv"
SLAM_TRAJ_FILENAME = "trajectory.txt"

# EUCM camera intrinsics for cam0. Copied verbatim from
# vizualize_results_edge_alignment.py (sourced from
# Floorplan-Alignment/intrinsics/kalibr_imucam_chain.yaml).
ALPHA = 0.6899954350657926
GAMMA = 1 - ALPHA
BETA = 0.8911981210457725
FU = 465.2979536302252
FV = 465.3194431883040
PU = 730.0455886686005
PV = 720.14270076712060

# Magnitude is arbitrary; we only use ray *direction* downstream. Matches the
# reference script's `ray_scale = 2000.0` to keep numbers in a comparable range.
RAY_SCALE = 2000.0


class RaysStage(Stage):
    """Back-project per-frame line detections to 3D rays in world coordinates."""

    @property
    def name(self) -> str:
        return "rays"

    @property
    def description(self) -> str:
        return "Back-project 2D line detections to 3D rays in world frame"

    @property
    def requires_container(self) -> bool:
        return False

    @property
    def container_profile(self) -> str:
        return "host"

    def run(self, runner, input_dir: Path, config: StageConfig) -> Path:
        lines_csv = _find_lines_csv(input_dir, config)
        traj_csv, traj_kind = _find_trajectory(config)

        lines = _load_lines(lines_csv)
        pose_t, pose_xyz, pose_q = _load_trajectory(traj_csv)

        if traj_kind not in {"aligned", "pca_aligned"}:
            print(
                f"[{self.name}] WARNING: using {traj_kind} trajectory; "
                "rays expect cam0 poses in world from `align`, not raw IMU poses"
            )
        elif not config.align_start_position:
            print(
                f"[{self.name}] WARNING: start alignment is disabled; rays are in "
                "OpenVINS world, not the input map frame"
            )

        stage_root = Path(
            tempfile.mkdtemp(prefix=f"{self.name}-", dir=runner.runtime_dir)
        )
        output_dir = stage_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        out_csv = output_dir / OUTPUT_CSV
        kept = 0
        skipped_nan = 0
        with out_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "timestamp",
                    "ox", "oy", "oz",
                    "r1x", "r1y", "r1z",
                    "r2x", "r2y", "r2z",
                ]
            )

            for ts, sx, sy, ex, ey in lines:
                ray1_cam = _calculate_ray(sx, sy)
                ray2_cam = _calculate_ray(ex, ey)
                if ray1_cam is None or ray2_cam is None:
                    skipped_nan += 1
                    continue

                # Nearest-timestamp pose (matches the reference script). The
                # aligned trajectory is already cam0-in-world after the align
                # stage chains through T_cam_imu, so this pose IS T_wrld_cam.
                idx = int(np.argmin(np.abs(pose_t - ts)))
                R_wrld_cam = quat_to_rot(*pose_q[idx])
                T_wrld_cam = np.eye(4)
                T_wrld_cam[:3, :3] = R_wrld_cam
                T_wrld_cam[:3, 3] = pose_xyz[idx]

                origin = T_wrld_cam[:3, 3]
                ray1_wrld = (T_wrld_cam @ ray1_cam)[:3, 0]
                ray2_wrld = (T_wrld_cam @ ray2_cam)[:3, 0]

                writer.writerow(
                    [
                        f"{ts:.6f}",
                        f"{origin[0]:.6f}", f"{origin[1]:.6f}", f"{origin[2]:.6f}",
                        f"{ray1_wrld[0]:.6f}", f"{ray1_wrld[1]:.6f}", f"{ray1_wrld[2]:.6f}",
                        f"{ray2_wrld[0]:.6f}", f"{ray2_wrld[1]:.6f}", f"{ray2_wrld[2]:.6f}",
                    ]
                )
                kept += 1

        log_lines = [
            f"Lines CSV: {lines_csv} ({len(lines)} entries)",
            f"Trajectory CSV: {traj_csv} ({traj_kind}, {len(pose_t)} poses)",
            f"Rays written: {kept}; skipped (outside EUCM valid radius): {skipped_nan}",
            f"Output: {out_csv}",
        ]
        (output_dir / f"{self.name}.log").write_text(
            "\n".join(log_lines) + "\n", encoding="utf-8"
        )
        (output_dir / f"{self.name}.status").write_text("0", encoding="utf-8")
        for line in log_lines:
            print(f"[{self.name}] {line}")

        return output_dir


def _find_lines_csv(input_dir: Path, config: StageConfig) -> Path:
    """Look in the chained input first, then fall back to line_extractor's output dir."""
    candidate = input_dir / LINES_FILENAME
    if candidate.is_file():
        return candidate
    try:
        path = stage_output_path(config, "line_extractor") / LINES_FILENAME
        if path.is_file():
            return path
    except Exception:
        pass
    raise FileNotFoundError(
        f"Could not find {LINES_FILENAME} in {input_dir} or in the line_extractor output dir"
    )


def _find_trajectory(config: StageConfig) -> Tuple[Path, str]:
    """Prefer PCA/align CSV trajectories; fall back to slam's raw trajectory.txt."""
    try:
        pca_aligned = stage_output_path(config, "pca_align") / PCA_ALIGNED_TRAJ_FILENAME
        if pca_aligned.is_file():
            return pca_aligned, "pca_aligned"
    except Exception:
        pass
    try:
        aligned = stage_output_path(config, "align") / ALIGNED_TRAJ_FILENAME
        if aligned.is_file():
            return aligned, "aligned"
    except Exception:
        pass
    try:
        slam = stage_output_path(config, "slam") / SLAM_TRAJ_FILENAME
        if slam.is_file():
            return slam, "slam"
    except Exception:
        pass
    raise FileNotFoundError(
        "No trajectory found. Expected pca_align/trajectory_pca_aligned.csv, "
        "align/trajectory_aligned.csv, or slam/trajectory.txt"
    )


def _load_lines(path: Path):
    """Yield (timestamp, startX, startY, endX, endY) tuples."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if len(vals) != 5:
                continue
            rows.append(tuple(vals))
    return rows


def _load_trajectory(path: Path):
    """Return (timestamps[N], positions[Nx3], quats[Nx4 xyzw])."""
    ts, xyz, q = [], [], []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if len(vals) != 8:
                continue
            ts.append(vals[0])
            xyz.append(vals[1:4])
            q.append(vals[4:8])
    return np.asarray(ts), np.asarray(xyz), np.asarray(q)


def _calculate_ray(u: float, v: float):
    """EUCM back-projection. Returns a 4x1 homogeneous point in cam frame, or None.

    Verbatim port of `calculateRay` from vizualize_results_edge_alignment.py.
    Sources for the EUCM equations: https://hal.science/hal-01722264v1/document
    """
    x_p = (u - PU) / FU
    y_p = (v - PV) / FV
    r = math.sqrt(x_p * x_p + y_p * y_p)
    # Eqn. 39 - point is outside the back-projectable disk for this alpha/beta.
    if r > 1 / ((ALPHA - GAMMA) * BETA):
        return None
    z_p = (1 - ALPHA * ALPHA * BETA * r * r) / (
        ALPHA * math.sqrt(1 - (ALPHA - GAMMA) * BETA * r * r) + GAMMA
    )
    return np.array(
        [[x_p * RAY_SCALE], [y_p * RAY_SCALE], [z_p * RAY_SCALE], [1.0]]
    )
