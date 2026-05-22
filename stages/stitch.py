"""Image stitching stage - converts dual fisheye to 360 equirectangular."""

from pathlib import Path

from runtime_backend import ExecutionSpec

from .base import Stage, StageConfig


class StitchStage(Stage):
    """Convert dual fisheye images to 360° equirectangular panorama."""

    @property
    def name(self) -> str:
        return "stitch"

    @property
    def description(self) -> str:
        return "Convert dual fisheye images to 360° equirectangular panorama"

    @property
    def input_type(self) -> str:
        return "rosbag"

    @property
    def output_type(self) -> str:
        return "rosbag"

    def run(
        self,
        runner,
        input_dir: Path,
        config: StageConfig,
    ) -> Path:
        """Run image stitching on the input bag."""

        # Prepare torch args; verify availability inside the container at runtime.
        torch_args = ""
        if config.use_torch:
            torch_args = f"--use-torch --device {config.torch_device}"

        # The stitching script path in the container
        script_path = (
            "/root/ros2_ws/src/hilti-trimble-slam-challenge-2026"
            "/challenge_tools_ros/bag_helper/image_stitching.py"
        )
        config_base = (
            "/root/ros2_ws/src/hilti-trimble-slam-challenge-2026"
            "/config/hilti_openvins"
        )

        stitch_cmd = f"""#!/bin/bash
set -euxo pipefail

set +u
source /opt/ros/jazzy/setup.bash
source /root/ros2_ws/install/setup.bash
set -u

echo "[stitch] Python: $(python3 --version)"
echo "[stitch] ros2 path: $(command -v ros2 || true)"
echo "[stitch] colcon path: $(command -v colcon || true)"
echo "[stitch] git path: $(command -v git || true)"
echo "[stitch] Input bag contents:"
ls -la /input
echo "[stitch] Script path: {script_path}"
test -f {script_path}
echo "[stitch] Config path: {config_base}/kalibr_imucam_chain.yaml"
test -f {config_base}/kalibr_imucam_chain.yaml
test -f {config_base}/mask_cam0.png
test -f {config_base}/mask_cam1.png
echo "[stitch] Running stitching script..."
rm -rf /output/rosbag_pano

TORCH_ARGS=""
if {str(config.use_torch).lower()}; then
  if python3 - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("torch") else 1)
PY
  then
    TORCH_ARGS="{torch_args}"
  else
    echo "[stitch] PyTorch not available; falling back to OpenCV stitching"
  fi
fi

{{
python3 {script_path} \
    --bag /input \
    --yaml {config_base}/kalibr_imucam_chain.yaml \
    --mask0 {config_base}/mask_cam0.png \
    --mask1 {config_base}/mask_cam1.png \
    --out /output/rosbag_pano \
    --jpeg-quality {config.jpeg_quality} \
    $TORCH_ARGS
}}

touch /output/rosbag_pano/.stitched
echo "[stitch] Output bag contents:"
ls -la /output/rosbag_pano
"""

        wrapper_script = """#!/bin/bash
set +e
mkdir -p /output
/bin/bash /stage_runtime/run_stitch.sh 2>&1 | tee /output/stitch.log
STATUS=${PIPESTATUS[0]}
echo "$STATUS" > /output/stitch.status
if [ $STATUS -ne 0 ]; then
  echo "[stitch] Wrapper detected failure, preserving logs" | tee -a /output/stitch.log
fi
exit 0
"""

        return runner.run_stage(
            container_profile=self.container_profile,
            input_dir=input_dir,
            config=config,
            spec=ExecutionSpec(
                stage_name=self.name,
                command=["/bin/bash", "/stage_runtime/run_stitch_wrapper.sh"],
                files={
                    "run_stitch.sh": stitch_cmd,
                    "run_stitch_wrapper.sh": wrapper_script,
                },
            ),
        )
