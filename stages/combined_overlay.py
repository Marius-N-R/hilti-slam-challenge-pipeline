"""Render base, individual, and combined realignments on the floorplan."""

import tempfile
from pathlib import Path

from .base import Stage, StageConfig, stage_output_path
from .overlay_common import render_overlay

OUTPUT_PNG = "combined_overlay.png"


class CombinedOverlayStage(Stage):
    """Render all trajectory realignments together."""

    @property
    def name(self) -> str:
        return "combined_overlay"

    @property
    def description(self) -> str:
        return "Render floorplan, Window, and combined aligned trajectories"

    @property
    def requires_container(self) -> bool:
        return False

    @property
    def container_profile(self) -> str:
        return "host"

    def run(self, runner, input_dir: Path, config: StageConfig) -> Path:
        base_path = stage_output_path(config, "align") / "trajectory_aligned.csv"
        floor_path = stage_output_path(config, "floorplan_align") / "trajectory_floor_aligned.csv"
        window_path = stage_output_path(config, "window_align") / "trajectory_window_aligned.csv"
        combined_path = stage_output_path(config, "combined_align") / "trajectory_combined_aligned.csv"

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
                    "alpha": 0.55,
                    "zorder": 4,
                },
                {
                    "path": floor_path,
                    "label": "floorplan aligned",
                    "color": "#00a6d6",
                    "linewidth": 1.6,
                    "alpha": 0.8,
                    "zorder": 5,
                },
                {
                    "path": window_path,
                    "label": "window aligned",
                    "color": "#dd55ff",
                    "linewidth": 1.6,
                    "alpha": 0.8,
                    "zorder": 5,
                },
                {
                    "path": combined_path,
                    "label": "combined aligned",
                    "color": "#00cc44",
                    "linewidth": 2.4,
                    "zorder": 7,
                },
            ],
            out_path=out_png,
            title=f"Combined overlay - {config.extra.get('current_input_name', 'run')}",
        )

        log_lines = [
            f"Base trajectory: {base_path}",
            f"Floorplan trajectory: {floor_path}",
            f"Window trajectory: {window_path}",
            f"Combined trajectory: {combined_path}",
            f"Output: {out_png}",
        ]
        (output_dir / f"{self.name}.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        (output_dir / f"{self.name}.status").write_text("0", encoding="utf-8")
        for line in log_lines:
            print(f"[{self.name}] {line}")
        return output_dir
