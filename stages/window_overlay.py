"""Render the window-aligned trajectory on the floorplan."""

import csv
import tempfile
from pathlib import Path

import numpy as np

from .base import Stage, StageConfig, stage_output_path
from .overlay_common import render_overlay

OUTPUT_PNG = "window_overlay.png"


class WindowOverlayStage(Stage):
    """Render the window-aligned trajectory and window constraints."""

    @property
    def name(self) -> str:
        return "window_overlay"

    @property
    def description(self) -> str:
        return "Render the Window-aligned trajectory on the floorplan PNG"

    @property
    def requires_container(self) -> bool:
        return False

    @property
    def container_profile(self) -> str:
        return "host"

    def run(self, runner, input_dir: Path, config: StageConfig) -> Path:
        base_path = stage_output_path(config, "align") / "trajectory_aligned.csv"
        window_path = stage_output_path(config, "window_align") / "trajectory_window_aligned.csv"
        observations_path = stage_output_path(config, "window_align") / "window_alignment_observations.csv"
        observations = _load_observations(observations_path)

        stage_root = Path(tempfile.mkdtemp(prefix=f"{self.name}-", dir=runner.runtime_dir))
        output_dir = stage_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_png = output_dir / OUTPUT_PNG

        render_overlay(
            config=config,
            trajectories=[
                {
                    "path": base_path,
                    "label": "base aligned",
                    "color": "#777777",
                    "linestyle": "--",
                    "linewidth": 1.0,
                    "alpha": 0.65,
                    "zorder": 4,
                },
                {
                    "path": window_path,
                    "label": "window aligned",
                    "color": "#dd55ff",
                    "linewidth": 2.0,
                    "zorder": 6,
                },
            ],
            observations=observations,
            out_path=out_png,
            title=f"Window overlay - {config.extra.get('current_input_name', 'run')}",
        )

        log_lines = [
            f"Base trajectory: {base_path}",
            f"Window trajectory: {window_path}",
            f"Window observations: {observations_path} ({len(observations)} frames)",
            f"Output: {out_png}",
        ]
        (output_dir / f"{self.name}.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        (output_dir / f"{self.name}.status").write_text("0", encoding="utf-8")
        for line in log_lines:
            print(f"[{self.name}] {line}")
        return output_dir


def _load_observations(path: Path) -> list[dict]:
    observations = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            observations.append(
                {
                    "observed_bl": np.array(
                        [float(row["observed_bl_x"]), float(row["observed_bl_y"])],
                        dtype=float,
                    ),
                    "observed_br": np.array(
                        [float(row["observed_br_x"]), float(row["observed_br_y"])],
                        dtype=float,
                    ),
                    "target_bl": np.array(
                        [float(row["target_bl_x"]), float(row["target_bl_y"])],
                        dtype=float,
                    ),
                    "target_br": np.array(
                        [float(row["target_br_x"]), float(row["target_br_y"])],
                        dtype=float,
                    ),
                }
            )
    return observations
