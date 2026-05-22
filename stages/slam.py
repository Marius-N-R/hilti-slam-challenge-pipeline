"""SLAM stage - runs OpenVINS visual-inertial odometry."""

from pathlib import Path

from runtime_backend import ExecutionSpec

from .base import Stage, StageConfig


class SlamStage(Stage):
    """Run OpenVINS visual-inertial SLAM."""

    @property
    def name(self) -> str:
        return "slam"

    @property
    def description(self) -> str:
        return "Run OpenVINS visual-inertial SLAM"

    @property
    def requires_ros_runtime(self) -> bool:
        return True

    @property
    def input_type(self) -> str:
        return "rosbag"

    @property
    def output_type(self) -> str:
        return "trajectory"

    def run(
        self,
        runner,
        input_dir: Path,
        config: StageConfig,
    ) -> Path:
        """Run OpenVINS SLAM on the input bag."""

        # Read the runner script
        runner_path = Path(__file__).parent / "slam_runner.py"
        runner_script = runner_path.read_text()

        # Create wrapper that handles output capture
        wrapper = f"""#!/bin/bash
set +e
mkdir -p /output

# Preserve floorplan assets (if present) for downstream overlay stage.
for candidate in /input/floorplan.* /input/map.*; do
  if [ -f "$candidate" ]; then
    cp "$candidate" /output/
  fi
done

python3 /stage_runtime/slam_runner.py {config.slam_rate} {config.slam_timeout} 2>&1 | tee /output/slam.log
STATUS=${{PIPESTATUS[0]}}
echo "$STATUS" > /output/slam.status
exit 0
"""

        return runner.run_stage(
            container_profile=self.container_profile,
            input_dir=input_dir,
            config=config,
            spec=ExecutionSpec(
                stage_name=self.name,
                command=["/bin/bash", "/stage_runtime/run_slam.sh"],
                files={
                    "slam_runner.py": runner_script,
                    "run_slam.sh": wrapper,
                },
            ),
        )
