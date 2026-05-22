"""Render a SLAM trajectory as a 2D path image."""

from pathlib import Path

from runtime_backend import ExecutionSpec

from .base import Stage, StageConfig


class PlotPathStage(Stage):
    """Render SLAM trajectory to an image."""

    @property
    def name(self) -> str:
        return "plot_path"

    @property
    def description(self) -> str:
        return "Render SLAM trajectory to an image"

    @property
    def input_type(self) -> str:
        return "trajectory"

    @property
    def output_type(self) -> str:
        return "directory"

    def run(
        self,
        runner,
        input_dir: Path,
        config: StageConfig,
    ) -> Path:
        """Render a 2D top-down path from trajectory.txt."""

        render_script = """#!/usr/bin/env python3
import os
import sys

import cv2
import numpy as np

traj_path = "/input/trajectory.txt"
out_path = "/output/trajectory_path.png"

if not os.path.exists(traj_path):
    print("[plot_path] ERROR: /input/trajectory.txt not found", file=sys.stderr)
    sys.exit(1)

points = []
with open(traj_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            x = float(parts[1])
            y = float(parts[2])
        except ValueError:
            continue
        points.append((x, y))

if len(points) < 2:
    print("[plot_path] ERROR: Not enough points to render", file=sys.stderr)
    sys.exit(1)

pts = np.array(points, dtype=np.float32)
min_x, min_y = pts.min(axis=0)
max_x, max_y = pts.max(axis=0)

pad = 60
width, height = 1200, 1200
range_x = max(max_x - min_x, 1e-6)
range_y = max(max_y - min_y, 1e-6)
scale = min((width - 2 * pad) / range_x, (height - 2 * pad) / range_y)

img = np.full((height, width, 3), 255, dtype=np.uint8)

draw_pts = []
for x, y in pts:
    px = int((x - min_x) * scale + pad)
    py = int((max_y - y) * scale + pad)
    draw_pts.append([px, py])

draw_pts = np.array(draw_pts, dtype=np.int32).reshape((-1, 1, 2))
cv2.polylines(img, [draw_pts], isClosed=False, color=(30, 30, 30), thickness=2)

start = tuple(draw_pts[0][0])
end = tuple(draw_pts[-1][0])
cv2.circle(img, start, 6, (0, 200, 0), -1)
cv2.circle(img, end, 6, (0, 0, 200), -1)

cv2.putText(
    img,
    f"points: {len(points)}",
    (pad, height - 20),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (80, 80, 80),
    2,
)

os.makedirs("/output", exist_ok=True)
cv2.imwrite(out_path, img)
print(f"[plot_path] Wrote {out_path}")
"""

        wrapper_script = """#!/bin/bash
set +e
mkdir -p /output
cp -a /input/. /output/ 2>/dev/null || true
python3 /stage_runtime/render_path.py 2>&1 | tee /output/plot_path.log
STATUS=${PIPESTATUS[0]}
echo "$STATUS" > /output/plot_path.status
exit 0
"""

        return runner.run_stage(
            container_profile=self.container_profile,
            input_dir=input_dir,
            config=config,
            spec=ExecutionSpec(
                stage_name=self.name,
                command=["/bin/bash", "/stage_runtime/run_plot_path.sh"],
                files={
                    "render_path.py": render_script,
                    "run_plot_path.sh": wrapper_script,
                },
            ),
        )
