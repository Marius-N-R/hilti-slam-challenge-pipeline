"""Convert a SLAM trajectory to cam0 CSV, optionally aligned to the initial pose.

Runs on the host (pure numpy). Reads:
  - <previous stage output>/trajectory.txt   (TUM format: `timestamp tx ty tz qx qy qz qw`)
  - <original input folder>/initial-pos.txt  when --align-start-position is enabled
    (CSV: header `# timestamp tx ty tz qx qy qz qw`, single data row)

The SLAM trajectory from `/ov_msckf/poseimu` is the **IMU** body pose in OpenVINS
world. This stage always converts it to a **cam0** pose CSV. If start alignment
is disabled, that cam0 pose remains in OpenVINS world. If start alignment is
enabled, `initial-pos.txt` is used to place cam0 in the map frame.

Naively chaining `T_map_ovworld = M_init @ T_cam_imu @ inv(M_slam_anchor)` produces
an SE(3) that "lands" cam0 at the GT pose exactly, but it bakes in any small
inconsistency between the three measurements (Kalibr cam-imu extrinsic, GT cam0
quaternion, SLAM IMU quaternion) as a non-vertical rotation. Empirically on
floor_1, that residual is ~43 degrees, which tips the gravity-aligned SLAM
trajectory on its side (z range explodes from <1m to ~28m).

Both `ovworld` and `map` are gravity-aligned, so the *true* T_map_ovworld is a
pure yaw rotation plus a 3D translation. We therefore extract just the yaw
component from the full SE(3) alignment, discard the residual pitch/roll, and
recompute the translation so that cam0 at the anchor still lands at the GT init
position. This matches the approach the reference script
`Floorplan-Alignment/src/vizualize_results_global_alignment.py` takes (using a
precomputed yaw from `map -> global` static TF), but we derive the yaw from the
initial-pos.txt + Kalibr extrinsic instead.

Trade-off: the output cam0 orientation at the anchor will only match the GT in
yaw, not in roll/pitch. For the challenge's Localization task that is fine -
the metric is xy distance and z is excluded.
"""

import csv
import math
import tempfile
from pathlib import Path

import numpy as np

from ._geometry import T_CAM_IMU, T_IMU_CAM, matrix_to_pose, pose_to_matrix
from .base import Stage, StageConfig

TRAJECTORY_FILENAME = "trajectory.txt"
INITIAL_POSE_FILENAME = "initial-pos.txt"
OUTPUT_CSV = "trajectory_aligned.csv"


class AlignStage(Stage):
    """Convert SLAM poses to cam0 CSV, optionally aligned to a known initial pose."""

    @property
    def name(self) -> str:
        return "align"

    @property
    def description(self) -> str:
        return "Convert SLAM trajectory to cam0 CSV, optionally start-aligned"

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

        slam_poses = _load_pose_table(trajectory_path)
        T_out_ovworld = np.eye(4)
        log_lines = [
            f"Trajectory: {trajectory_path} ({len(slam_poses)} poses)",
        ]

        if config.align_start_position:
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

            init_row = _load_initial_pose(initial_pose_path)
            anchor_idx = int(np.argmin(np.abs(slam_poses[:, 0] - init_row[0])))
            anchor_t = float(slam_poses[anchor_idx, 0])
            init_t = float(init_row[0])

            T_out_ovworld, yaw_rad, residual_tilt_deg = _compute_T_map_ovworld(
                slam_anchor_imu=slam_poses[anchor_idx, 1:],
                init_cam=init_row[1:],
            )
            log_lines.extend(
                [
                    "Start alignment: enabled",
                    f"Initial pose: {initial_pose_path}",
                    (
                        f"Anchor SLAM pose index {anchor_idx} at t={anchor_t:.6f} "
                        f"(init t={init_t:.6f}, dt={anchor_t - init_t:+.6f})"
                    ),
                    f"Yaw correction (ovworld -> map): {math.degrees(yaw_rad):+.3f} deg",
                    (
                        "Residual non-yaw tilt in full SE(3) alignment: "
                        f"{residual_tilt_deg:.2f} deg "
                        "(discarded; see module docstring)"
                    ),
                ]
            )
        else:
            log_lines.append(
                "Start alignment: disabled; output cam0 poses remain in OpenVINS world"
            )

        aligned = _apply_to_imu_get_cam(slam_poses[:, 1:], T_out_ovworld)

        stage_root = Path(
            tempfile.mkdtemp(prefix=f"{self.name}-", dir=runner.runtime_dir)
        )
        output_dir = stage_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        _write_csv(output_dir / OUTPUT_CSV, slam_poses[:, 0], aligned)

        log_lines.append(f"Trajectory CSV written: {output_dir / OUTPUT_CSV}")
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


def _compute_T_map_ovworld(slam_anchor_imu: np.ndarray, init_cam: np.ndarray):
    """Return (T_map_ovworld, yaw_rad, residual_tilt_deg).

    `T_map_ovworld` is a yaw rotation around +z plus a 3D translation. It is the
    closest gravity-preserving SE(3) to the full alignment

        T_full = M_init @ T_cam_imu @ inv(M_slam_anchor)

    so we keep only the yaw (atan2 of R_full[1,0], R_full[0,0]) and recompute
    the translation such that cam0 at the anchor still lands at the GT init
    position:

        translation = init.position - R_yaw @ (M_slam_anchor @ T_imu_cam).position

    `residual_tilt_deg` is the angle that the full alignment's R sits away from
    pure yaw (acos(R_full[2,2])). Large values flag a calibration / convention
    mismatch worth investigating; for the challenge data this is ~43 degrees.
    """
    M_slam = pose_to_matrix(slam_anchor_imu)
    M_init = pose_to_matrix(init_cam)

    T_full = M_init @ T_CAM_IMU @ np.linalg.inv(M_slam)
    R_full = T_full[:3, :3]
    yaw = math.atan2(R_full[1, 0], R_full[0, 0])
    residual_tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, R_full[2, 2]))))

    c, s = math.cos(yaw), math.sin(yaw)
    R_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    slam_anchor_cam_pos = (M_slam @ T_IMU_CAM)[:3, 3]
    translation = M_init[:3, 3] - R_yaw @ slam_anchor_cam_pos

    T_map_ovworld = np.eye(4)
    T_map_ovworld[:3, :3] = R_yaw
    T_map_ovworld[:3, 3] = translation
    return T_map_ovworld, yaw, residual_tilt_deg


def _apply_to_imu_get_cam(slam_imu_poses: np.ndarray, T_map_ovworld: np.ndarray) -> np.ndarray:
    """Map each SLAM IMU pose to the corresponding cam0 pose in map frame (eq. 1)."""
    out = np.empty_like(slam_imu_poses)
    for i in range(slam_imu_poses.shape[0]):
        M_slam_i = pose_to_matrix(slam_imu_poses[i])
        T_map_cam_i = T_map_ovworld @ M_slam_i @ T_IMU_CAM
        out[i] = matrix_to_pose(T_map_cam_i)
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
