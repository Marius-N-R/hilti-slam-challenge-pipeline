"""Align a SLAM trajectory using PCA-derived axes."""

import dagger

from .base import Stage, StageConfig


class PcaAlignStage(Stage):
    """Create a PCA-aligned trajectory from trajectory.txt."""

    @property
    def name(self) -> str:
        return "pca_align"

    @property
    def description(self) -> str:
        return "Align SLAM trajectory using PCA"

    @property
    def input_type(self) -> str:
        return "trajectory"

    @property
    def output_type(self) -> str:
        return "trajectory"

    async def run(
        self,
        container: dagger.Container,
        input_dir: dagger.Directory,
        config: StageConfig,
    ) -> dagger.Directory:
        """Read trajectory.txt, compute PCA alignment, and overwrite trajectory.txt."""

        align_script = """#!/usr/bin/env python3
import os
import sys

import numpy as np

INPUT_PATH = "/input/trajectory.txt"
OUTPUT_PATH = "/output/trajectory.txt"
MATRIX_PATH = "/output/pca_alignment_matrix.txt"
INFO_PATH = "/output/pca_alignment_info.txt"


def load_trajectory():
    header_lines = []
    rows = []
    with open(INPUT_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                header_lines.append(line.rstrip("\\n"))
                continue
            parts = stripped.split()
            if len(parts) < 8:
                continue
            rows.append(parts[:8])
    if len(rows) < 2:
        raise RuntimeError("Not enough trajectory rows to align")
    return header_lines, rows


def build_rotation(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    rotation = np.column_stack((primary, secondary, tertiary))
    return mean, rotation, eigenvalues


def main() -> None:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError("/input/trajectory.txt not found")

    os.makedirs("/output", exist_ok=True)
    headers, rows = load_trajectory()

    positions = np.array([[float(row[1]), float(row[2]), float(row[3])] for row in rows], dtype=np.float64)
    orientations = np.array([[float(row[4]), float(row[5]), float(row[6]), float(row[7])] for row in rows], dtype=np.float64)

    mean, rotation, eigenvalues = build_rotation(positions)
    aligned_positions = (positions - mean) @ rotation

    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        if headers:
            for header in headers:
                handle.write(header + "\\n")
        else:
            handle.write("# timestamp tx ty tz qx qy qz qw\\n")

        for row, aligned_position, orientation in zip(rows, aligned_positions, orientations):
            timestamp = row[0]
            handle.write(
                f"{timestamp} "
                f"{aligned_position[0]:.9f} {aligned_position[1]:.9f} {aligned_position[2]:.9f} "
                f"{orientation[0]:.9f} {orientation[1]:.9f} {orientation[2]:.9f} {orientation[3]:.9f}\\n"
            )

    np.savetxt(MATRIX_PATH, rotation, fmt="%.9f")
    with open(INFO_PATH, "w", encoding="utf-8") as handle:
        handle.write(f"points={len(rows)}\\n")
        handle.write(f"mean={mean.tolist()}\\n")
        handle.write(f"eigenvalues={eigenvalues.tolist()}\\n")

    print(f"[pca_align] Wrote {OUTPUT_PATH}")
    print(f"[pca_align] Wrote {MATRIX_PATH}")
    print(f"[pca_align] Wrote {INFO_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[pca_align] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
"""

        wrapper_script = """#!/bin/bash
set +e
mkdir -p /output
cp -a /input/. /output/ 2>/dev/null || true
python3 /tmp/pca_align.py 2>&1 | tee /output/pca_align.log
STATUS=${PIPESTATUS[0]}
echo "$STATUS" > /output/pca_align.status
exit 0
"""

        result = (
            container
            .with_mounted_directory("/input", input_dir)
            .with_exec(["/bin/bash", "-c", "mkdir -p /output"])
            .with_new_file("/tmp/pca_align.py", contents=align_script, permissions=0o755)
            .with_new_file("/tmp/run_pca_align.sh", contents=wrapper_script, permissions=0o755)
            .with_exec(["/bin/bash", "/tmp/run_pca_align.sh"])
        )

        return result.directory("/output")
