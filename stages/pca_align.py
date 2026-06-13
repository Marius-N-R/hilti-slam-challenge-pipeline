"""PCA-align an already CSV-converted trajectory."""

import csv
import shutil
import tempfile
from pathlib import Path

import numpy as np

from ._geometry import quat_to_rot, rot_to_quat
from .base import Stage, StageConfig, stage_output_path

INPUT_CSV = "trajectory_aligned.csv"
OUTPUT_CSV = "trajectory_pca_aligned.csv"
MATRIX_TXT = "pca_alignment_matrix.txt"
INFO_TXT = "pca_alignment_info.txt"


class PcaAlignStage(Stage):
    """Reorient an aligned trajectory using PCA-derived axes."""

    @property
    def name(self) -> str:
        return "pca_align"

    @property
    def description(self) -> str:
        return "PCA-align the CSV trajectory after initial pose alignment"

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
        input_csv = _find_input_csv(input_dir, config)
        timestamps, positions, quats = _load_pose_csv(input_csv)

        mean, basis, eigenvalues = _build_pca_basis(positions)
        anchor_idx, anchor_source = _resolve_anchor_idx(timestamps, config)
        anchor_position = positions[anchor_idx].copy()
        aligned_positions = (positions - anchor_position) @ basis + anchor_position

        # Position rows are multiplied by `basis`, so column-vector rotations
        # need the equivalent world-frame transform `basis.T`.
        pca_world_from_original = basis.T
        aligned_quats = np.empty_like(quats)
        for idx, quat in enumerate(quats):
            corrected_rot = pca_world_from_original @ quat_to_rot(*quat)
            aligned_quats[idx] = rot_to_quat(corrected_rot)

        stage_root = Path(
            tempfile.mkdtemp(prefix=f"{self.name}-", dir=runner.runtime_dir)
        )
        output_dir = stage_root / "output"
        if input_dir.exists():
            shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        output_csv = output_dir / OUTPUT_CSV
        _write_pose_csv(output_csv, timestamps, aligned_positions, aligned_quats)
        np.savetxt(output_dir / MATRIX_TXT, basis, fmt="%.9f")

        log_lines = [
            f"Input trajectory: {input_csv} ({len(timestamps)} poses)",
            f"Output trajectory: {output_csv}",
            f"Anchor: index={anchor_idx}, source={anchor_source}, position={anchor_position.tolist()}",
            f"Mean: {mean.tolist()}",
            f"Eigenvalues: {eigenvalues.tolist()}",
            f"Basis matrix: {output_dir / MATRIX_TXT}",
        ]
        (output_dir / INFO_TXT).write_text(
            "\n".join(
                [
                    f"points={len(timestamps)}",
                    f"anchor_index={anchor_idx}",
                    f"anchor_source={anchor_source}",
                    f"anchor_position={anchor_position.tolist()}",
                    f"mean={mean.tolist()}",
                    f"eigenvalues={eigenvalues.tolist()}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (output_dir / f"{self.name}.log").write_text(
            "\n".join(log_lines) + "\n", encoding="utf-8"
        )
        (output_dir / f"{self.name}.status").write_text("0", encoding="utf-8")
        for line in log_lines:
            print(f"[{self.name}] {line}")

        return output_dir


def _find_input_csv(input_dir: Path, config: StageConfig) -> Path:
    candidate = input_dir / INPUT_CSV
    if candidate.is_file():
        return candidate
    try:
        candidate = stage_output_path(config, "align") / INPUT_CSV
        if candidate.is_file():
            return candidate
    except Exception:
        pass
    raise FileNotFoundError(
        f"Could not find {INPUT_CSV}. Run the align stage before pca_align."
    )


def _load_pose_csv(path: Path):
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            try:
                values = [float(part) for part in parts]
            except ValueError:
                continue
            if len(values) != 8:
                continue
            rows.append(values)
    if len(rows) < 2:
        raise ValueError(f"Need at least two pose rows for PCA alignment: {path}")
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1:4], arr[:, 4:8]


def _resolve_anchor_idx(timestamps: np.ndarray, config: StageConfig):
    if config.align_start_position:
        original_input = config.extra.get("current_input_path", "")
        if original_input:
            initial_pose_path = Path(original_input) / "initial-pos.txt"
            if initial_pose_path.is_file():
                try:
                    initial_ts = _load_initial_timestamp(initial_pose_path)
                    idx = int(np.argmin(np.abs(timestamps - initial_ts)))
                    return idx, f"initial-pos.txt timestamp {initial_ts:.9f}"
                except Exception as exc:
                    print(
                        f"[pca_align] WARNING: failed to read {initial_pose_path}: {exc}; "
                        "falling back to first pose anchor"
                    )
    return 0, "first trajectory pose"


def _load_initial_timestamp(path: Path) -> float:
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) < 1:
                continue
            return float(parts[0])
    raise ValueError(f"No timestamp row found in {path}")


def _build_pca_basis(points: np.ndarray):
    mean = points.mean(axis=0)
    centered = points - mean

    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    primary = eigenvectors[:, 0]
    if primary[0] < 0:
        primary = -primary

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    secondary = np.cross(world_up, primary)
    secondary_norm = np.linalg.norm(secondary)
    if secondary_norm < 1e-8:
        secondary = eigenvectors[:, 1]
        if secondary[1] < 0:
            secondary = -secondary
    else:
        secondary /= secondary_norm

    tertiary = np.cross(primary, secondary)
    tertiary_norm = np.linalg.norm(tertiary)
    if tertiary_norm < 1e-8:
        tertiary = eigenvectors[:, 2]
    else:
        tertiary /= tertiary_norm

    if tertiary[2] < 0:
        tertiary = -tertiary
        secondary = -secondary

    basis = np.column_stack((primary, secondary, tertiary))
    return mean, basis, eigenvalues


def _write_pose_csv(
    path: Path,
    timestamps: np.ndarray,
    positions: np.ndarray,
    quats: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["# timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        for timestamp, position, quat in zip(timestamps, positions, quats):
            writer.writerow(
                [
                    f"{timestamp:.9f}",
                    f"{position[0]:.9f}",
                    f"{position[1]:.9f}",
                    f"{position[2]:.9f}",
                    f"{quat[0]:.9f}",
                    f"{quat[1]:.9f}",
                    f"{quat[2]:.9f}",
                    f"{quat[3]:.9f}",
                ]
            )
