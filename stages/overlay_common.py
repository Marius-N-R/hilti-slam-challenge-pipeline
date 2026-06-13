"""Shared floorplan overlay rendering helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from .base import StageConfig, stage_output_path

PIXELS_PER_METER = 100.0


def find_floorplan_png(config: StageConfig) -> Path:
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
    raise FileNotFoundError("No floorplan PNG found. Looked in: " + ", ".join(map(str, candidates)))


def load_traj_xy(path: Path) -> np.ndarray:
    rows = []
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
            if len(values) == 8:
                rows.append((values[1], values[2]))
    if not rows:
        raise ValueError(f"No trajectory rows found in {path}")
    return np.asarray(rows, dtype=float)


def load_edges(config: StageConfig) -> np.ndarray | None:
    try:
        path = stage_output_path(config, "floorplan_edges") / "floorplan_edges.csv"
    except Exception:
        return None
    if not path.is_file():
        return None
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("x1"):
                continue
            parts = stripped.replace(",", " ").split()
            try:
                values = [float(part) for part in parts]
            except ValueError:
                continue
            if len(values) == 4:
                rows.append(values)
    return np.asarray(rows, dtype=float) if rows else None


def load_groundtruth(config: StageConfig) -> np.ndarray | None:
    original = config.extra.get("current_input_path", "")
    if not original:
        return None
    path = Path(original) / "groundtruth.txt"
    if not path.is_file():
        return None
    return load_traj_xy(path)


def render_overlay(
    *,
    config: StageConfig,
    trajectories: list[dict],
    out_path: Path,
    title: str,
    observations: list[dict] | None = None,
) -> None:
    png_path = find_floorplan_png(config)
    img = mpimg.imread(str(png_path))
    height, width = img.shape[:2]

    fig, ax = plt.subplots(figsize=(14, 9))
    cmap = "gray" if img.ndim == 2 else None
    ax.imshow(np.flipud(img), origin="lower", cmap=cmap, extent=(0, width, 0, height))

    edges = load_edges(config)
    if edges is not None:
        for x1, y1, x2, y2 in edges:
            ax.plot(
                [x1 * PIXELS_PER_METER, x2 * PIXELS_PER_METER],
                [y1 * PIXELS_PER_METER, y2 * PIXELS_PER_METER],
                color="#8f8fff",
                linewidth=0.8,
                alpha=0.45,
                zorder=2,
            )

    groundtruth = load_groundtruth(config)
    if groundtruth is not None:
        ax.plot(
            groundtruth[:, 0] * PIXELS_PER_METER,
            groundtruth[:, 1] * PIXELS_PER_METER,
            color="#ff8800",
            linewidth=2.2,
            alpha=0.9,
            label=f"ground truth ({len(groundtruth)} poses)",
            zorder=3,
        )

    if observations:
        for obs in observations:
            observed = np.asarray([obs["observed_bl"], obs["observed_br"]], dtype=float)
            target = np.asarray([obs["target_bl"], obs["target_br"]], dtype=float)
            ax.plot(
                observed[:, 0] * PIXELS_PER_METER,
                observed[:, 1] * PIXELS_PER_METER,
                color="#dd55ff",
                linewidth=1.0,
                alpha=0.55,
                zorder=4,
            )
            ax.plot(
                target[:, 0] * PIXELS_PER_METER,
                target[:, 1] * PIXELS_PER_METER,
                color="#222222",
                linewidth=1.2,
                alpha=0.75,
                zorder=4,
            )

    for trajectory in trajectories:
        xy = load_traj_xy(trajectory["path"])
        ax.plot(
            xy[:, 0] * PIXELS_PER_METER,
            xy[:, 1] * PIXELS_PER_METER,
            color=trajectory.get("color", "#00d0ff"),
            linewidth=trajectory.get("linewidth", 1.8),
            linestyle=trajectory.get("linestyle", "-"),
            alpha=trajectory.get("alpha", 0.9),
            label=f"{trajectory['label']} ({len(xy)} poses)",
            zorder=trajectory.get("zorder", 5),
        )
        if trajectory.get("mark_endpoints", True):
            ax.scatter(
                xy[0, 0] * PIXELS_PER_METER,
                xy[0, 1] * PIXELS_PER_METER,
                color="#00cc44",
                marker="o",
                s=60,
                edgecolors="black",
                linewidths=0.6,
                zorder=7,
            )
            ax.scatter(
                xy[-1, 0] * PIXELS_PER_METER,
                xy[-1, 1] * PIXELS_PER_METER,
                color="#ff3333",
                marker="X",
                s=80,
                edgecolors="black",
                linewidths=0.6,
                zorder=7,
            )

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.set_xlabel("image x [px]")
    ax.set_ylabel("image y [px]")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.85, fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
