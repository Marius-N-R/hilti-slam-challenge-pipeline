"""Clean stage - removes pipeline outputs on the host."""

from pathlib import Path

from .base import Stage, StageConfig


class CleanStage(Stage):
    """Remove pipeline-generated outputs for a run."""

    @property
    def name(self) -> str:
        return "clean"

    @property
    def description(self) -> str:
        return "Remove pipeline outputs (leaves input data intact)"

    @property
    def input_type(self) -> str:
        return "directory"

    @property
    def output_type(self) -> str:
        return "directory"

    def run(
        self,
        runner,
        input_dir: Path,
        config: StageConfig,
    ) -> Path:
        """No-op in container; host cleanup is handled in pipeline."""
        return input_dir
