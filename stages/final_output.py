"""Collect final trajectories, overlays, and evaluation artifacts."""

import json
import shutil
import tempfile
from pathlib import Path

from .base import Stage, StageConfig, stage_output_path


class FinalOutputStage(Stage):
    """Collect the final pipeline outputs into one folder."""

    @property
    def name(self) -> str:
        return "final_output"

    @property
    def description(self) -> str:
        return "Collect realigned trajectories, overlays, and evaluation into one folder"

    @property
    def requires_container(self) -> bool:
        return False

    @property
    def container_profile(self) -> str:
        return "host"

    def run(self, runner, input_dir: Path, config: StageConfig) -> Path:
        stage_root = Path(tempfile.mkdtemp(prefix=f"{self.name}-", dir=runner.runtime_dir))
        output_dir = stage_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        artifacts = [
            ("trajectories/base_aligned.csv", "align", "trajectory_aligned.csv"),
            ("trajectories/floorplan_aligned.csv", "floorplan_align", "trajectory_floor_aligned.csv"),
            ("trajectories/window_aligned.csv", "window_align", "trajectory_window_aligned.csv"),
            ("trajectories/combined_aligned.csv", "combined_align", "trajectory_combined_aligned.csv"),
            ("overlays/floorplan_overlay.png", "floorplan_overlay", "overlay.png"),
            ("overlays/window_overlay.png", "window_overlay", "window_overlay.png"),
            ("overlays/combined_overlay.png", "combined_overlay", "combined_overlay.png"),
            ("alignment/window_alignment_transform.json", "window_align", "window_alignment_transform.json"),
            (
                "alignment/window_alignment_observations.csv",
                "window_align",
                "window_alignment_observations.csv",
            ),
            ("alignment/combined_alignment.json", "combined_align", "combined_alignment.json"),
            ("evaluation/summary.json", "final_eval", "summary.json"),
            ("evaluation/matched_errors.csv", "final_eval", "matched_errors.csv"),
        ]

        manifest = {
            "input": config.extra.get("current_input_path", ""),
            "input_name": config.extra.get("current_input_name", ""),
            "artifacts": [],
            "missing": [],
        }
        for relative_target, stage_name, filename in artifacts:
            source = stage_output_path(config, stage_name) / filename
            target = output_dir / relative_target
            if source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                manifest["artifacts"].append(
                    {
                        "name": relative_target,
                        "source_stage": stage_name,
                        "source": str(source),
                    }
                )
            else:
                manifest["missing"].append(
                    {
                        "name": relative_target,
                        "source_stage": stage_name,
                        "source": str(source),
                    }
                )

        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / f"{self.name}.log").write_text(
            "\n".join(
                [
                    f"Collected artifacts: {len(manifest['artifacts'])}",
                    f"Missing artifacts: {len(manifest['missing'])}",
                    f"Output: {output_dir}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (output_dir / f"{self.name}.status").write_text("0", encoding="utf-8")
        print(f"[{self.name}] Collected {len(manifest['artifacts'])} artifact(s)")
        if manifest["missing"]:
            print(f"[{self.name}] Missing {len(manifest['missing'])} optional artifact(s)")
        print(f"[{self.name}] Output: {output_dir}")
        return output_dir
