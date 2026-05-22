"""Mask rectification stage for the vendored window pipeline."""

import json

import dagger

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

    async def run(
        self,
        container: dagger.Container,
        input_dir: dagger.Directory,
        config: StageConfig,
    ) -> dagger.Directory:
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
/bin/bash /tmp/{self.name}.sh 2>&1 | tee /output/{self.name}.log
STATUS=${{PIPESTATUS[0]}}
echo "$STATUS" > /output/{self.name}.status
exit 0
"""

        result = (
            container
            .with_mounted_directory("/input", input_dir)
            .with_exec(["/bin/bash", "-c", "mkdir -p /output"])
            .with_new_file(f"/tmp/{self.name}.sh", contents=stage_cmd, permissions=0o755)
            .with_new_file(
                f"/tmp/{self.name}_wrapper.sh",
                contents=wrapper_cmd,
                permissions=0o755,
            )
            .with_exec(["/bin/bash", f"/tmp/{self.name}_wrapper.sh"])
        )
        return result.directory("/output")
