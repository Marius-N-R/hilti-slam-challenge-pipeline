"""Format conversion stage - converts ROS2 bags to EuRoC format."""

from pathlib import Path

from runtime_backend import ExecutionSpec

from .base import Stage, StageConfig


class ConvertStage(Stage):
    """Convert ROS2 bag to EuRoC dataset format."""

    @property
    def name(self) -> str:
        return "convert"

    @property
    def description(self) -> str:
        return "Convert ROS2 bag to EuRoC dataset format"

    @property
    def input_type(self) -> str:
        return "rosbag"

    @property
    def output_type(self) -> str:
        return "euroc"

    def run(
        self,
        runner,
        input_dir: Path,
        config: StageConfig,
    ) -> Path:
        """Convert ROS2 bag to EuRoC format."""

        script_path = (
            "/root/ros2_ws/src/hilti-trimble-slam-challenge-2026"
            "/challenge_tools_ros/bag_helper/ros2bag_to_euroc.py"
        )

        # Get custom topic names from config if provided
        cam_topics = config.extra.get(
            "cam_topics",
            "/cam0/image_raw/compressed /cam1/image_raw/compressed"
        )
        imu_topic = config.extra.get("imu_topic", "/imu/data")

        convert_cmd = f"""
{self.get_ros_source_cmd()} && \\
python3 {script_path} \\
    --bag /input \\
    --out /output \\
    --cam-topics {cam_topics} \\
    --imu-topic {imu_topic}
"""

        wrapper_script = f"""#!/bin/bash
set +e
mkdir -p /output
/bin/bash -c {convert_cmd!r} 2>&1 | tee /output/{self.name}.log
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
