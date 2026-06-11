"""Line extractor stage - detects near-horizontal line segments in cam0 frames."""

from pathlib import Path

from runtime_backend import ExecutionSpec

from .base import Stage, StageConfig


class LineExtractorStage(Stage):
    """Extract near-horizontal line segments from /cam0 frames in a ROS2 bag.

    Mirrors the behavior of the C++ `line_extractor` node from
    rework/Floorplan-Alignment: detects line segments with OpenCV inside a fixed
    ROI polygon, keeps only those within 30 deg of horizontal and at least 75
    pixels long, and writes them to `lines.csv`.
    """

    @property
    def name(self) -> str:
        return "line_extractor"

    @property
    def description(self) -> str:
        return "Extract near-horizontal line segments from cam0 frames"

    @property
    def requires_ros_runtime(self) -> bool:
        return True

    @property
    def input_type(self) -> str:
        return "rosbag"

    @property
    def output_type(self) -> str:
        return "lines"

    def run(
        self,
        runner,
        input_dir: Path,
        config: StageConfig,
    ) -> Path:
        original_input_str = config.extra.get("current_input_path", "")
        if not original_input_str:
            raise RuntimeError("Original input path not set in config.extra")
        original_input = Path(original_input_str)
        bag_dir = original_input / "rosbag"
        if not bag_dir.is_dir():
            raise FileNotFoundError(
                f"Expected 'rosbag' subdirectory inside {original_input}"
            )

        runner_path = Path(__file__).parent / "line_extractor_runner.py"
        runner_script = runner_path.read_text()

        wrapper = """#!/bin/bash
set +e
mkdir -p /output
source /opt/ros/jazzy/setup.bash
source /root/ros2_ws/install/setup.bash

python3 /stage_runtime/line_extractor_runner.py 2>&1 | tee /output/line_extractor.log
STATUS=${PIPESTATUS[0]}
echo "$STATUS" > /output/line_extractor.status
exit 0
"""

        return runner.run_stage(
            container_profile=self.container_profile,
            input_dir=bag_dir,
            config=config,
            spec=ExecutionSpec(
                stage_name=self.name,
                command=["/bin/bash", "/stage_runtime/run_line_extractor.sh"],
                files={
                    "line_extractor_runner.py": runner_script,
                    "run_line_extractor.sh": wrapper,
                },
            ),
        )
