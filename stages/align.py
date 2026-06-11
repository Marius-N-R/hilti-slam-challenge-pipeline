"""Align a SLAM trajectory rigidly to a known initial pose.

Runs on the host (pure numpy). Reads:
  - <previous stage output>/trajectory.txt   (TUM format: `timestamp tx ty tz qx qy qz qw`)
  - <original input folder>/initial-pos.txt  (CSV: header `# timestamp tx ty tz qx qy qz qw`,
                                              single data row)

Anchors the SLAM pose whose timestamp is closest to the initial-pose timestamp to the initial
pose, then applies the same rigid SE(3) transform to every SLAM pose. Writes
`trajectory_aligned.csv` to the stage output.
"""

import csv
import tempfile
from pathlib import Path

import numpy as np

from ._geometry import matrix_to_pose, pose_to_matrix
from .base import Stage, StageConfig

TRAJECTORY_FILENAME = "trajectory.txt"
INITIAL_POSE_FILENAME = "initial-pos.txt"
OUTPUT_CSV = "trajectory_aligned.csv"


class AlignStage(Stage):
    """Rigidly align a SLAM trajectory to a known initial pose."""

    @property
    def name(self) -> str:
        return "align"

    @property
    def description(self) -> str:
        return "Rigidly align SLAM trajectory to a known initial pose"

    @property
    def requires_container(self) -> bool:
        return False

    @property
    def container_profile(self) -> str:
        return "host"

    def run(self, runner, input_dir: Path, config: StageConfig) -> Path:
        trajectory_path = input_dir / TRAJECTORY_FILENAME
        if not trajectory_path.is_file():
            raise FileNotFoundError(
                f"Expected '{TRAJECTORY_FILENAME}' in previous stage output: {input_dir}"
            )

        original_input_str = config.extra.get("current_input_path", "")
        if not original_input_str:
            raise RuntimeError("Original input path not set in config.extra")
        original_input = Path(original_input_str)
        if not original_input.is_dir():
            raise FileNotFoundError(
                f"Original input folder no longer exists: {original_input}"
            )
        initial_pose_path = original_input / INITIAL_POSE_FILENAME
        if not initial_pose_path.is_file():
            raise FileNotFoundError(
                f"Expected '{INITIAL_POSE_FILENAME}' in original input folder: {original_input}"
            )

        slam_poses = _load_pose_table(trajectory_path)
        init_row = _load_initial_pose(initial_pose_path)

        anchor_idx = int(np.argmin(np.abs(slam_poses[:, 0] - init_row[0])))
        anchor_t = float(slam_poses[anchor_idx, 0])
        init_t = float(init_row[0])

        T = _se3_align(slam_poses[anchor_idx, 1:], init_row[1:])
        aligned = _apply_se3(slam_poses[:, 1:], T)

        stage_root = Path(
            tempfile.mkdtemp(prefix=f"{self.name}-", dir=runner.runtime_dir)
        )
        output_dir = stage_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        _write_csv(output_dir / OUTPUT_CSV, slam_poses[:, 0], aligned)

        log_lines = [
            f"Trajectory: {trajectory_path} ({len(slam_poses)} poses)",
            f"Initial pose: {initial_pose_path}",
            (
                f"Anchor SLAM pose index {anchor_idx} at t={anchor_t:.6f} "
                f"(init t={init_t:.6f}, dt={anchor_t - init_t:+.6f})"
            ),
            f"Aligned trajectory written: {output_dir / OUTPUT_CSV}",
        ]
        (output_dir / f"{self.name}.log").write_text(
            "\n".join(log_lines) + "\n", encoding="utf-8"
        )
        (output_dir / f"{self.name}.status").write_text("0", encoding="utf-8")
        for line in log_lines:
            print(f"[{self.name}] {line}")

        return output_dir


def _load_pose_table(path: Path) -> np.ndarray:
    """Load whitespace- or comma-separated rows of `timestamp tx ty tz qx qy qz qw`."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) != 8:
                continue
            rows.append([float(p) for p in parts])
    if not rows:
        raise ValueError(f"No pose rows found in {path}")
    return np.asarray(rows, dtype=float)


def _load_initial_pose(path: Path) -> np.ndarray:
    rows = _load_pose_table(path)
    if rows.shape[0] > 1:
        print(f"[align] WARNING: {path} has {rows.shape[0]} data rows; using the first")
    return rows[0]


def _se3_align(slam_anchor: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
    M_slam = pose_to_matrix(slam_anchor)
    M_init = pose_to_matrix(init_pose)
    return M_init @ np.linalg.inv(M_slam)


def _apply_se3(poses: np.ndarray, T: np.ndarray) -> np.ndarray:
    out = np.empty_like(poses)
    for i in range(poses.shape[0]):
        M = pose_to_matrix(poses[i])
        out[i] = matrix_to_pose(T @ M)
    return out


def _write_csv(path: Path, timestamps: np.ndarray, poses: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["# timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        for t, p in zip(timestamps, poses):
            writer.writerow(
                [
                    f"{t:.9f}",
                    f"{p[0]:.9f}",
                    f"{p[1]:.9f}",
                    f"{p[2]:.9f}",
                    f"{p[3]:.9f}",
                    f"{p[4]:.9f}",
                    f"{p[5]:.9f}",
                    f"{p[6]:.9f}",
                ]
            )
