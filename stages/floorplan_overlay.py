"""Overlay a SLAM trajectory on top of a floorplan image."""

import base64
from pathlib import Path

from runtime_backend import ExecutionSpec

from .base import Stage, StageConfig


class FloorplanOverlayStage(Stage):
    """Render a placeholder floorplan overlay from trajectory data."""

    @property
    def name(self) -> str:
        return "floorplan_overlay"

    @property
    def description(self) -> str:
        return "Overlay SLAM trajectory on a floorplan image"

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
        """Render floorplan overlay image.

        Expected input: a directory containing `trajectory.txt`.
        Optional floorplan source:
          - `--floorplan` host path (injected into container)
          - `/input/floorplan.(png|jpg|jpeg)`
          - `/input/map.(png|jpg|jpeg)`
        """

        floorplan_hint = str(config.extra.get("floorplan_path", "") or "")
        floorplan_payload = ""
        floorplan_ext = ".png"

        if floorplan_hint:
            host_floorplan = Path(floorplan_hint).expanduser()
            if host_floorplan.is_file():
                floorplan_payload = base64.b64encode(host_floorplan.read_bytes()).decode("ascii")
                floorplan_ext = host_floorplan.suffix.lower() or ".png"
            elif config.verbose:
                print(
                    "[floorplan_overlay] WARNING: --floorplan path not found on host "
                    f"(will try to resolve inside input): {host_floorplan}"
                )

        render_script = f"""#!/usr/bin/env python3
import base64
import os
import sys
from pathlib import Path

import cv2
import numpy as np


TRAJECTORY_PATH = Path("/input/trajectory.txt")
OUTPUT_PATH = Path("/output/floorplan_overlay.png")
FLOORPLAN_HINT = {floorplan_hint!r}
FLOORPLAN_EXT = {floorplan_ext!r}


def load_floorplan() -> tuple[np.ndarray, str]:
    candidates: list[str] = []

    cli_b64 = Path("/stage_runtime/floorplan_cli.b64")
    if cli_b64.exists():
        try:
            payload = cli_b64.read_text(encoding="utf-8").strip()
            if payload:
                suffix = FLOORPLAN_EXT if FLOORPLAN_EXT.startswith(".") else f".{{FLOORPLAN_EXT}}"
                cli_path = Path("/tmp/floorplan_cli").with_suffix(suffix)
                cli_path.write_bytes(base64.b64decode(payload))
                candidates.append(str(cli_path))
        except Exception as exc:
            print(f"[floorplan_overlay] WARNING: Failed to decode CLI floorplan: {{exc}}")

    if FLOORPLAN_HINT:
        hint_path = Path(FLOORPLAN_HINT)
        if hint_path.is_absolute():
            candidates.append(str(hint_path))
        else:
            candidates.append(str(Path("/input") / hint_path))
            candidates.append(str(Path("/output") / hint_path))

    candidates.extend(
        [
            "/input/floorplan.png",
            "/input/floorplan.jpg",
            "/input/floorplan.jpeg",
            "/input/map.png",
            "/input/map.jpg",
            "/input/map.jpeg",
        ]
    )

    for candidate in candidates:
        candidate_path = Path(candidate)
        if not candidate_path.exists():
            continue
        image = cv2.imread(str(candidate_path), cv2.IMREAD_COLOR)
        if image is not None:
            return image, str(candidate_path)

    placeholder = np.full((1200, 1200, 3), 245, dtype=np.uint8)
    for value in range(0, 1201, 80):
        cv2.line(placeholder, (value, 0), (value, 1200), (230, 230, 230), 1)
        cv2.line(placeholder, (0, value), (1200, value), (230, 230, 230), 1)
    cv2.putText(
        placeholder,
        "Placeholder floorplan",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (120, 120, 120),
        2,
    )
    return placeholder, "generated_placeholder"


def load_points() -> np.ndarray:
    if not TRAJECTORY_PATH.exists():
        raise FileNotFoundError("/input/trajectory.txt not found")

    points: list[tuple[float, float]] = []
    with TRAJECTORY_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                continue
            points.append((float(parts[1]), float(parts[2])))

    if len(points) < 2:
        raise RuntimeError("Not enough points in trajectory.txt to draw overlay")

    return np.array(points, dtype=np.float32)


def draw_overlay(floorplan: np.ndarray, points: np.ndarray) -> np.ndarray:
    canvas = floorplan.copy()
    height, width = canvas.shape[:2]
    padding = max(20, int(min(height, width) * 0.05))

    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    range_x = max(float(max_xy[0] - min_xy[0]), 1e-6)
    range_y = max(float(max_xy[1] - min_xy[1]), 1e-6)

    scale = min((width - 2 * padding) / range_x, (height - 2 * padding) / range_y)
    draw_points = []
    for x_coord, y_coord in points:
        px = int((x_coord - min_xy[0]) * scale + padding)
        py = int((max_xy[1] - y_coord) * scale + padding)
        draw_points.append([px, py])

    polyline = np.array(draw_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [polyline], isClosed=False, color=(40, 40, 220), thickness=3)

    start = tuple(polyline[0][0])
    end = tuple(polyline[-1][0])
    cv2.circle(canvas, start, 8, (0, 180, 0), -1)
    cv2.circle(canvas, end, 8, (0, 0, 220), -1)

    return canvas


def main() -> None:
    os.makedirs("/output", exist_ok=True)
    points = load_points()
    floorplan, source = load_floorplan()
    overlay = draw_overlay(floorplan, points)

    if not cv2.imwrite(str(OUTPUT_PATH), overlay):
        raise RuntimeError("Failed to write floorplan_overlay.png")

    print(f"[floorplan_overlay] Wrote {{OUTPUT_PATH}}")
    print(f"[floorplan_overlay] Floorplan source: {{source}}")
    print(f"[floorplan_overlay] Points drawn: {{len(points)}}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[floorplan_overlay] ERROR: {{exc}}", file=sys.stderr)
        sys.exit(1)
"""

        wrapper_script = """#!/bin/bash
set +e
mkdir -p /output
cp -a /input/. /output/ 2>/dev/null || true
python3 /stage_runtime/render_floorplan_overlay.py 2>&1 | tee /output/floorplan_overlay.log
STATUS=${PIPESTATUS[0]}
echo "$STATUS" > /output/floorplan_overlay.status
exit 0
"""

        files = {
            "render_floorplan_overlay.py": render_script,
            "run_floorplan_overlay.sh": wrapper_script,
        }
        if floorplan_payload:
            files["floorplan_cli.b64"] = floorplan_payload

        return runner.run_stage(
            container_profile=self.container_profile,
            input_dir=input_dir,
            config=config,
            spec=ExecutionSpec(
                stage_name=self.name,
                command=["/bin/bash", "/stage_runtime/run_floorplan_overlay.sh"],
                files=files,
            ),
        )
