"""Mask rectification stage for the vendored window pipeline."""

import json
from pathlib import Path

from runtime_backend import ExecutionSpec

from .base import Stage, StageConfig


class WindowsRectifyStage(Stage):
    @property
    def name(self) -> str:
        return "window_rectify"

    @property
    def description(self) -> str:
        return "Undistort the SAM3 window mask with py360convert"

    @property
    def container_profile(self) -> str:
        return "windows"

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
        runtime_args = json.dumps(
            {
                "mask_path": "/output/windows_masks.npy",
            }
        )

        stage_cmd = """#!/bin/bash
set -euo pipefail
mkdir -p /output
cp -a /input/. /output/
export WINDOWS_PIPELINE_ROOT=/opt/windows_pipeline
export PYTHONPATH=/opt/windows_pipeline/py360convert:${PYTHONPATH:-}
python /opt/pipeline_scripts/windows/run_rectify.py '""" + runtime_args + """'
"""

        wrapper_cmd = f"""#!/bin/bash
set +e
/bin/bash /stage_runtime/{self.name}.sh 2>&1 | tee /output/{self.name}.log
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
                files={
                    f"{self.name}.sh": stage_cmd,
                    f"{self.name}_wrapper.sh": wrapper_cmd,
                },
                use_gpu=config.windows_device == "cuda",
            ),
        )
