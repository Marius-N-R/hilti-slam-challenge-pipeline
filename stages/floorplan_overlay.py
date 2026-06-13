"""Floorplan overlay stage - renders the final aligned trajectory on the floorplan PNG.

Mirrors the rendering convention used by rework/Floorplan-Alignment/src/
vizualize_results_global_alignment.py: convert world meters to PNG pixels at
100 px/m, then `imshow(np.flipud(img), origin="lower")` so the world's +y axis
points up on the rendered image.

The trajectory is taken from the most-refined output available:
  1. trajectory_floor_aligned.csv (floorplan_align)
  2. trajectory_aligned.csv        (align)

The PNG is taken from:
  1. <input>/<input_name>.png  (e.g., data/floor_1/floor_1.png)
  2. <input>/floorplan.png
"""

import csv
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np

# Force a non-interactive backend before pyplot is imported - matplotlib must
# not try to open a window when running headless.
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from .base import Stage, StageConfig, stage_output_path

OUTPUT_PNG = "overlay.png"

# Dataset convention - the reference scripts assume 100 px per meter when
# converting between world coordinates and the floorplan PNG.
PIXELS_PER_METER = 100.0


class FloorplanOverlayStage(Stage):
    """Render the final aligned trajectory on the floorplan PNG."""

    @property
    def name(self) -> str:
        return "floorplan_overlay"

    @property
    def description(self) -> str:
        return "Render the final aligned trajectory on the floorplan PNG"

    @property
    def requires_container(self) -> bool:
        return False

    @property
    def container_profile(self) -> str:
        return "host"

    def run(self, runner, input_dir: Path, config: StageConfig) -> Path:
        png_path = _find_floorplan_png(config)
        traj_path, traj_kind = _find_trajectory(input_dir, config)

        # Optional "before" trajectory for comparison.
        before_traj_path = None
        if traj_kind == "floor_aligned":
            try:
                candidate = stage_output_path(config, "align") / "trajectory_aligned.csv"
                if candidate.is_file():
                    before_traj_path = candidate
            except Exception:
                pass

        # Optional floorplan_edges overlay (sanity check that the world->pixel
        # mapping is consistent between the DXF-derived edges and the PNG).
        edges_path = None
        try:
            candidate = stage_output_path(config, "floorplan_edges") / "floorplan_edges.csv"
            if candidate.is_file():
                edges_path = candidate
        except Exception:
            pass

        traj_xy = _load_traj_xy(traj_path)
        before_xy = _load_traj_xy(before_traj_path) if before_traj_path is not None else None
        edges = _load_edges(edges_path) if edges_path is not None else None

        stage_root = Path(
            tempfile.mkdtemp(prefix=f"{self.name}-", dir=runner.runtime_dir)
        )
        output_dir = stage_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_png = output_dir / OUTPUT_PNG

        _render(
            png_path=png_path,
            traj_xy_m=traj_xy,
            traj_label=f"trajectory ({traj_kind})",
            before_xy_m=before_xy,
            edges_m=edges,
            out_path=out_png,
            title=(
                f"Floorplan overlay - {config.extra.get('current_input_name', 'run')} "
                f"({len(traj_xy)} poses)"
            ),
        )

        log_lines = [
            f"PNG: {png_path}",
            f"Trajectory: {traj_path} ({traj_kind}, {len(traj_xy)} poses)",
            (
                f"Before-comparison trajectory: {before_traj_path}"
                if before_traj_path is not None
                else "Before-comparison trajectory: (none; rendering corrected trajectory only)"
            ),
            (
                f"Floorplan edges: {edges_path} ({len(edges)} segments)"
                if edges is not None
                else "Floorplan edges: (none; rendering trajectory only)"
            ),
            f"Pixels per meter: {PIXELS_PER_METER}",
            f"Output: {out_png}",
        ]
        (output_dir / f"{self.name}.log").write_text(
            "\n".join(log_lines) + "\n", encoding="utf-8"
        )
        (output_dir / f"{self.name}.status").write_text("0", encoding="utf-8")
        for line in log_lines:
            print(f"[{self.name}] {line}")

        return output_dir


def _find_floorplan_png(config: StageConfig) -> Path:
    original = config.extra.get("current_input_path", "")
    if not original:
        raise RuntimeError("current_input_path not set in config.extra")
    original_dir = Path(original)
    name = config.extra.get("current_input_name", "")

    candidates = []
    if name:
        candidates.append(original_dir / f"{name}.png")
    candidates.append(original_dir / "floorplan.png")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No floorplan PNG found. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + ". Drop a PNG named '<input_folder_name>.png' (or 'floorplan.png') "
          "into the input folder."
    )


def _find_trajectory(input_dir: Path, config: StageConfig) -> Tuple[Path, str]:
    candidate = input_dir / "trajectory_floor_aligned.csv"
    if candidate.is_file():
        return candidate, "floor_aligned"
    try:
        path = stage_output_path(config, "floorplan_align") / "trajectory_floor_aligned.csv"
        if path.is_file():
            return path, "floor_aligned"
    except Exception:
        pass
    try:
        path = stage_output_path(config, "align") / "trajectory_aligned.csv"
        if path.is_file():
            return path, "aligned"
    except Exception:
        pass
    raise FileNotFoundError(
        "No trajectory found. Expected floorplan_align/trajectory_floor_aligned.csv "
        "or align/trajectory_aligned.csv"
    )


def _load_traj_xy(path: Path) -> np.ndarray:
    """Return (N, 2) array of (x, y) positions in meters."""
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
            if len(vals) != 8:
                continue
            rows.append((vals[1], vals[2]))
    if not rows:
        raise ValueError(f"No pose rows found in {path}")
    return np.asarray(rows, dtype=float)


def _load_edges(path: Path) -> np.ndarray:
    """Return (M, 4) array of (x1, y1, x2, y2) edge endpoints in meters."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("x1"):
                continue
            parts = stripped.replace(",", " ").split()
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if len(vals) != 4:
                continue
            rows.append(vals)
    return np.asarray(rows, dtype=float)


def _render(
    png_path: Path,
    traj_xy_m: np.ndarray,
    traj_label: str,
    before_xy_m,
    edges_m,
    out_path: Path,
    title: str,
) -> None:
    img = mpimg.imread(str(png_path))
    h, w = img.shape[:2]

    fig, ax = plt.subplots(figsize=(14, 9))
    cmap = "gray" if img.ndim == 2 else None
    ax.imshow(np.flipud(img), origin="lower", cmap=cmap, extent=(0, w, 0, h))

    if edges_m is not None and len(edges_m) > 0:
        for x1, y1, x2, y2 in edges_m:
            ax.plot(
                [x1 * PIXELS_PER_METER, x2 * PIXELS_PER_METER],
                [y1 * PIXELS_PER_METER, y2 * PIXELS_PER_METER],
                color="#8888ff",
                linewidth=0.8,
                alpha=0.6,
                zorder=2,
            )

    if before_xy_m is not None and len(before_xy_m) > 0:
        ax.plot(
            before_xy_m[:, 0] * PIXELS_PER_METER,
            before_xy_m[:, 1] * PIXELS_PER_METER,
            color="gray",
            linewidth=1.0,
            linestyle="--",
            alpha=0.7,
            label="trajectory (aligned, pre floorplan_align)",
            zorder=3,
        )

    tx_px = traj_xy_m[:, 0] * PIXELS_PER_METER
    ty_px = traj_xy_m[:, 1] * PIXELS_PER_METER
    ax.plot(tx_px, ty_px, color="#00d0ff", linewidth=1.8, label=traj_label, zorder=4)
    ax.scatter(
        tx_px[0],
        ty_px[0],
        color="#00cc44",
        marker="o",
        s=90,
        label="start",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
    )
    ax.scatter(
        tx_px[-1],
        ty_px[-1],
        color="#ff3333",
        marker="X",
        s=110,
        label="end",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
    )

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_xlabel("image x [px]")
    ax.set_ylabel("image y [px]")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.85, fontsize=9)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
