"""
Example custom stage - use this as a template for adding new stages.

To add a new stage:
1. Copy this file and rename it (e.g., my_stage.py)
2. Modify the class to implement your processing
3. Register it in stages/__init__.py:

   from .my_stage import MyStage
   registry.register(MyStage())

4. Run with: python pipeline.py --stages my_stage --input <bag>
"""

from pathlib import Path

from runtime_backend import ExecutionSpec

from .base import Stage, StageConfig


class ExampleStage(Stage):
    """
    Example stage that demonstrates how to create custom processing.

    This stage simply copies the input to output with a marker file.
    Replace the run() method with your actual processing logic.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this stage (used in --stages argument)."""
        return "example"

    @property
    def description(self) -> str:
        """Human-readable description shown in --list-stages."""
        return "Example custom stage (template)"

    @property
    def requires_ros_runtime(self) -> bool:
        """Set to True if this stage needs ros2 launch/nodes running."""
        return False

    @property
    def input_type(self) -> str:
        """What this stage expects: 'rosbag', 'euroc', 'directory', 'trajectory'."""
        return "rosbag"

    @property
    def output_type(self) -> str:
        """What this stage produces."""
        return "directory"

    def run(
        self,
        runner,
        input_dir: Path,
        config: StageConfig,
    ) -> Path:
        """
        Execute the stage processing.

        Args:
            runner: Container runtime backend.
            input_dir: Input directory (from previous stage or host).
            config: Stage configuration options.

        Returns:
            Output directory with results.
        """
        # Example: Run a simple command that processes the input
        # Replace this with your actual processing logic

        process_cmd = f"""
{self.get_ros_source_cmd()} && \\
echo "Processing input directory..." && \\
ls -la /input && \\
mkdir -p /output && \\
cp -r /input/* /output/ && \\
echo "example_stage_completed" > /output/stage_marker.txt && \\
echo "Processing complete!"
"""

        wrapper_script = f"""#!/bin/bash
set +e
mkdir -p /output
/bin/bash -c {process_cmd!r} 2>&1 | tee /output/{self.name}.log
STATUS=${{PIPESTATUS[0]}}
echo "$STATUS" > /output/{self.name}.status
exit 0
"""

        return runner.run_stage(
            container_profile=self.container_profile,
            input_dir=input_dir,
            config=config,
            spec=ExecutionSpec(
                stage_name=self.name,
                command=["/bin/bash", f"/stage_runtime/{self.name}_wrapper.sh"],
                files={f"{self.name}_wrapper.sh": wrapper_script},
            ),
        )


# Uncomment the following to auto-register when importing this module:
# from . import registry
# registry.register(ExampleStage())
