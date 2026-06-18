"""SaveTF stage - extracts the map->global static TF.

If init_pos.txt is available in the original input folder, computes T_map_global
by matching it with the SLAM trajectory. Otherwise, falls back to reading /tf_static
from the rosbag.

Writes orientation.json in the same format as the Floorplan-Alignment save_tf.py node.

The stage passes through all files from its input (slam output) and adds
orientation.json, so the align stage that follows can read trajectory.txt and
orientation.json from the same directory.
"""

import shutil
import tempfile
from pathlib import Path
from textwrap import dedent

from runtime_backend import BindMount, ExecutionSpec

from .base import Stage, StageConfig

OUTPUT_JSON = "orientation.json"


class SaveTfStage(Stage):
    """Extract map->global static TF from the rosbag and write orientation.json."""

    @property
    def name(self) -> str:
        return "save_tf"

    @property
    def description(self) -> str:
        return "Extract map->global static TF from rosbag (writes orientation.json)"

    @property
    def requires_container(self) -> bool:
        return True

    @property
    def container_profile(self) -> str:
        return "ros"

    @property
    def input_type(self) -> str:
        return "trajectory"

    @property
    def output_type(self) -> str:
        return "trajectory"

    def run(self, runner, input_dir: Path, config: StageConfig) -> Path:
        original_input_str = config.extra.get("current_input_path", "")
        if not original_input_str:
            raise RuntimeError("Original input path not set in config.extra")
        original_input_path = Path(original_input_str)
        bag_dir = original_input_path / "rosbag"
        if not bag_dir.is_dir():
            raise FileNotFoundError(f"Expected rosbag/ subdirectory at {bag_dir}")

        runner_script = (Path(__file__).parent / "save_tf_runner.py").read_text(encoding="utf-8")
        wrapper = dedent("""\
            #!/bin/bash
            set +e
            mkdir -p /output
            source /opt/ros/jazzy/setup.bash && source /root/ros2_ws/install/setup.bash
            python3 /stage_runtime/save_tf_runner.py 2>&1 | tee /output/save_tf.log
            STATUS=${PIPESTATUS[0]}
            echo "$STATUS" > /output/save_tf.status
            exit $STATUS
        """)

        # Mount SLAM output for trajectory.txt and original input for init_pos.txt
        extra_mounts = [
            BindMount(source=original_input_path, target="/original_input", read_only=True),
        ]

        # Also mount the SLAM output directory if it exists (contains trajectory.txt)
        if input_dir.exists():
            extra_mounts.append(
                BindMount(source=input_dir, target="/slam_output", read_only=True)
            )

        container_output = runner.run_stage(
            container_profile=self.container_profile,
            input_dir=bag_dir,
            config=config,
            spec=ExecutionSpec(
                stage_name=self.name,
                command=["/bin/bash", "/stage_runtime/run_save_tf.sh"],
                files={
                    "save_tf_runner.py": runner_script,
                    "run_save_tf.sh": wrapper,
                },
                extra_mounts=extra_mounts,
            ),
        )

        orientation_src = container_output / OUTPUT_JSON
        if not orientation_src.is_file():
            raise FileNotFoundError(
                f"Container did not produce {OUTPUT_JSON}; check save_tf.log"
            )

        # Merge: pass through slam output + add orientation.json and logs
        stage_root = Path(
            tempfile.mkdtemp(prefix=f"{self.name}-", dir=runner.runtime_dir)
        )
        output_dir = stage_root / "output"
        if input_dir.exists():
            shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(orientation_src, output_dir / OUTPUT_JSON)
        for src in container_output.iterdir():
            if src.name == OUTPUT_JSON:
                continue
            dest = output_dir / src.name
            if not dest.exists():
                shutil.copy2(src, dest)

        return output_dir
