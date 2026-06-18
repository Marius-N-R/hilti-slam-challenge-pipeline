"""SLAM stage - runs OpenVINS visual-inertial odometry."""

from pathlib import Path

from runtime_backend import BindMount, ExecutionSpec

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
        """Run OpenVINS SLAM on the bag inside <input_dir>/rosbag/."""

        bag_dir = input_dir / "rosbag"
        if not bag_dir.is_dir():
            raise FileNotFoundError(
                f"Expected 'rosbag' subdirectory inside {input_dir}"
            )

        # Get original input path for accessing init_pos.txt
        original_input_str = config.extra.get("current_input_path", "")
        if not original_input_str:
            raise RuntimeError("Original input path not set in config.extra")
        original_input_path = Path(original_input_str)

        runner_path = Path(__file__).parent / "slam_runner.py"
        runner_script = runner_path.read_text()

        # Load static TF publisher script
        static_tf_publisher_path = Path(__file__).parent / "static_tf_publisher_custom.py"
        static_tf_publisher_script = static_tf_publisher_path.read_text()

        # Create wrapper that handles output capture
        wrapper = f"""#!/bin/bash
set +e
mkdir -p /output

python3 /stage_runtime/slam_runner.py {config.slam_rate} {config.slam_timeout} 2>&1 | tee /output/slam.log
STATUS=${{PIPESTATUS[0]}}
echo "$STATUS" > /output/slam.status
exit 0
"""

        return runner.run_stage(
            container_profile=self.container_profile,
            input_dir=bag_dir,
            config=config,
            spec=ExecutionSpec(
                stage_name=self.name,
                command=["/bin/bash", "/stage_runtime/run_slam.sh"],
                files={
                    "slam_runner.py": runner_script,
                    "static_tf_publisher_custom.py": static_tf_publisher_script,
                    "run_slam.sh": wrapper,
                },
                extra_mounts=[
                    BindMount(source=original_input_path, target="/original_input", read_only=True),
                ],
            ),
        )
