"""Aggregate stage that expands to the full ordered pipeline."""

from pathlib import Path
from typing import Any

from .base import Stage, StageConfig

FULL_PIPELINE_STAGES = [
    "slam",
    "align",
    "line_extractor",
    "floorplan_edges",
    "rays",
    "floorplan_align",
    "floorplan_overlay",
    "image_selector",
    "window_dino",
    "window_sam",
    "window_rectify",
    "window_pose",
    "window_align",
    "window_overlay",
    "combined_align",
    "combined_overlay",
    "final_eval",
    "final_output",
]


class AllStage(Stage):
    """Run every registered processing stage in dependency order."""

    @property
    def name(self) -> str:
        return "all"

    @property
    def description(self) -> str:
        return "Run the complete pipeline in dependency order"

    @property
    def requires_container(self) -> bool:
        return False

    @property
    def container_profile(self) -> str:
        return "aggregate"

    @property
    def expanded_stage_names(self) -> list[str]:
        return list(FULL_PIPELINE_STAGES)

    def run(
        self,
        runner: Any,
        input_dir: Path,
        config: StageConfig,
    ) -> Path:
        raise RuntimeError(
            "The 'all' aggregate stage must be expanded by the pipeline orchestrator"
        )
