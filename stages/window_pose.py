"""Window per-window pose metric stage."""

import csv
import json
from pathlib import Path

from runtime_backend import ExecutionSpec

from .base import Stage, StageConfig
from .window_common import (
    window_container_env,
    window_mount,
    window_preflight_script,
    window_python,
    window_root,
    window_uses_gpu,
)


class WindowPoseStage(Stage):
    """Compute Window floorplan pose metrics from SAM3 masks."""

    @property
    def name(self) -> str:
        return "window_pose"

    @property
    def description(self) -> str:
        return "Compute Window window pose metrics from segmented masks"

    @property
    def requires_container(self) -> bool:
        return True

    @property
    def container_profile(self) -> str:
        return "window"

    def run(self, runner, input_dir: Path, config: StageConfig) -> Path:
        root = window_root(config)
        python = window_python(root)

        args = {
            "python": str(python),
            "root": str(root),
            "camera_height": config.window_camera_height,
        }
        wrapper = f"""#!/bin/bash
set +e
mkdir -p /output
cp -r /input/. /output/
exec > >(tee /output/{self.name}.log) 2>&1
{window_preflight_script(python)}
PREFLIGHT_STATUS=$?
if [ "$PREFLIGHT_STATUS" -ne 0 ]; then
  echo "$PREFLIGHT_STATUS" > /output/{self.name}.status
  exit 0
fi
{python} /stage_runtime/window_pose_driver.py '{json.dumps(args)}'
STATUS=$?
echo "$STATUS" > /output/{self.name}.status
exit 0
"""
        return runner.run_stage(
            container_profile=self.container_profile,
            input_dir=input_dir,
            config=config,
            spec=ExecutionSpec(
                stage_name=self.name,
                command=["/bin/bash", "/stage_runtime/run_window_pose.sh"],
                files={
                    "run_window_pose.sh": wrapper,
                    "window_pose_driver.py": _POSE_DRIVER,
                    "window_pose_runner.py": _POSE_RUNNER,
                },
                env=window_container_env(root),
                extra_mounts=[window_mount(root)],
                workdir=str(root),
                use_gpu=window_uses_gpu(config),
            ),
        )


def _write_summary_csv(path: Path, summaries: list[dict]) -> None:
    fields = [
        "frame",
        "distance_m",
        "yaw_deg",
        "lateral_offset_m",
        "pitch_deg",
        "roll_deg",
        "estimated_width_m",
        "estimated_width_horizontal_m",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field, "") for field in fields})


_POSE_DRIVER = r'''#!/usr/bin/env python3
import csv
import json
import subprocess
import sys
from pathlib import Path


def selected_image_paths(root):
    images_dir = root / "images"
    images = sorted([
        *images_dir.glob("*.png"),
        *images_dir.glob("*.jpg"),
        *images_dir.glob("*.jpeg"),
    ])
    if not images:
        raise FileNotFoundError(f"No selected images found in {images_dir}")
    return images


def write_summary_csv(path, summaries):
    fields = [
        "frame",
        "distance_m",
        "yaw_deg",
        "lateral_offset_m",
        "pitch_deg",
        "roll_deg",
        "estimated_width_m",
        "estimated_width_horizontal_m",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field, "") for field in fields})


def main():
    args = json.loads(sys.argv[1])
    image_paths = selected_image_paths(Path("/output"))
    summaries = []
    for image_path in image_paths:
        mask_path = Path("/output") / "sam3" / image_path.stem / "windows_masks.npy"
        frame_output = Path("/output") / "pose" / image_path.stem
        summary_path = frame_output / "pose_summary.json"
        command = [
            args["python"],
            "/stage_runtime/window_pose_runner.py",
            "--image",
            str(image_path),
            "--mask",
            str(mask_path),
            "--output-dir",
            str(frame_output),
            "--camera-height",
            str(args["camera_height"]),
        ]
        print("$ " + " ".join(command), flush=True)
        status = subprocess.run(command, cwd=args["root"], check=False).returncode
        if status != 0:
            return status
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    write_summary_csv(Path("/output") / "window_pose_summary.csv", summaries)
    print(f"[window_pose] Processed {len(image_paths)} selected image(s)")
    print("[window_pose] Output: /output/pose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


_POSE_RUNNER = r'''#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def unproject_eucm(pixel_coords, intrinsics):
    alpha, beta, fx, fy, cx, cy = intrinsics
    mx = (pixel_coords[:, 0] - cx) / fx
    my = (pixel_coords[:, 1] - cy) / fy
    r2 = mx**2 + my**2
    term = 1 - beta * alpha * (2 * alpha - 1) * r2
    z = (1 - alpha**2 * beta * r2) / (alpha * np.sqrt(np.maximum(term, 0)) + (1 - alpha))
    rays = np.stack([mx, my, z], axis=-1)
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def get_corners_from_mask(mask_array):
    mask_uint8 = (mask_array > 0).astype(np.uint8) * 255
    if len(mask_uint8.shape) == 3 and mask_uint8.shape[2] == 3:
        mask_uint8 = cv2.cvtColor(mask_uint8, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2).astype(np.float32)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()

    def best_corner(qpts, prefer_x, prefer_y):
        if len(qpts) == 0:
            return None
        return qpts[np.argmax(prefer_x * qpts[:, 0] + prefer_y * qpts[:, 1])]

    tl = best_corner(pts[(pts[:, 0] <= cx) & (pts[:, 1] <= cy)], -1, -1)
    tr = best_corner(pts[(pts[:, 0] > cx) & (pts[:, 1] <= cy)], +1, -1)
    br = best_corner(pts[(pts[:, 0] > cx) & (pts[:, 1] > cy)], +1, +1)
    bl = best_corner(pts[(pts[:, 0] <= cx) & (pts[:, 1] > cy)], -1, +1)
    if any(c is None for c in [tl, tr, br, bl]):
        return None
    return np.array([tl, tr, br, bl])


def calculate_3d_position_gravity_aligned(rays, camera_height):
    r_tl, r_tr, r_br, r_bl = rays[0], rays[1], rays[2], rays[3]
    n_left = np.cross(r_tl, r_bl)
    n_right = np.cross(r_tr, r_br)
    v_v = np.cross(n_left, n_right)
    v_v /= np.linalg.norm(v_v)
    if v_v[1] > 0:
        v_v = -v_v

    y_world = v_v
    roll_deg = np.degrees(np.arctan2(y_world[0], -y_world[1]))
    pitch_deg = np.degrees(np.arcsin(y_world[2]))
    x_world = np.cross(np.array([0.0, 0.0, 1.0]), y_world)
    x_world /= np.linalg.norm(x_world)
    z_world = np.cross(x_world, y_world)
    r_c2w = np.vstack([x_world, y_world, z_world])

    ray_w_bl = r_c2w @ r_bl
    ray_w_br = r_c2w @ r_br
    d_bl = -camera_height / ray_w_bl[1]
    d_br = -camera_height / ray_w_br[1]
    return d_bl * ray_w_bl, d_br * ray_w_br, pitch_deg, roll_deg


def compute_floorplan_pose(pos_bl, pos_br):
    mid_xz = np.array([(pos_bl[0] + pos_br[0]) / 2, (pos_bl[2] + pos_br[2]) / 2])
    edge_xz = np.array([pos_br[0] - pos_bl[0], pos_br[2] - pos_bl[2]])
    edge_xz /= np.linalg.norm(edge_xz)
    normal_xz = np.array([-edge_xz[1], edge_xz[0]])
    if np.dot(normal_xz, mid_xz) > 0:
        normal_xz = -normal_xz
    distance = abs(np.dot(mid_xz, normal_xz))
    lateral_offset = np.dot(mid_xz, edge_xz)
    cam_forward_xz = np.array([0.0, 1.0])
    cos_yaw = np.clip(np.dot(cam_forward_xz, normal_xz), -1, 1)
    yaw_deg = np.degrees(np.arccos(cos_yaw))
    cross = cam_forward_xz[0] * normal_xz[1] - cam_forward_xz[1] * normal_xz[0]
    if cross < 0:
        yaw_deg = -yaw_deg
    return distance, yaw_deg, lateral_offset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-height", type=float, default=2.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask = np.load(args.mask)
    corners = get_corners_from_mask(mask)
    if corners is None:
        raise RuntimeError("Could not extract four window corners from mask")

    image = cv2.imread(args.image)
    if image is not None:
        labels = ["TL", "TR", "BR", "BL"]
        colors = [(0, 200, 0), (0, 200, 0), (0, 0, 255), (0, 0, 255)]
        for coords, label, color in zip(corners, labels, colors):
            point = (int(coords[0]), int(coords[1]))
            cv2.circle(image, point, 6, color, -1)
            cv2.putText(image, label, (point[0] + 8, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imwrite(str(output_dir / "corners_debug.png"), image)

    intrinsics = [0.689995, 0.891198, 465.2979, 465.3194, 730.0455, 720.1427]
    original_res = (1472, 1440)
    if original_res[0] != mask.shape[1]:
        scale = np.array([original_res[0] / mask.shape[1], original_res[1] / mask.shape[0]])
    else:
        scale = 1
    rays = unproject_eucm(corners * scale, intrinsics)
    pos_bl, pos_br, pitch_deg, roll_deg = calculate_3d_position_gravity_aligned(
        rays,
        args.camera_height,
    )
    distance, yaw_deg, lateral_offset = compute_floorplan_pose(pos_bl, pos_br)
    estimated_width = np.linalg.norm(pos_br - pos_bl)
    estimated_width_horizontal = abs(pos_br[0] - pos_bl[0])

    summary = {
        "frame": Path(args.image).stem,
        "corners_px": corners.tolist(),
        "bottom_left_m": pos_bl.tolist(),
        "bottom_right_m": pos_br.tolist(),
        "distance_m": float(distance),
        "yaw_deg": float(yaw_deg),
        "lateral_offset_m": float(lateral_offset),
        "pitch_deg": float(pitch_deg),
        "roll_deg": float(roll_deg),
        "estimated_width_m": float(estimated_width),
        "estimated_width_horizontal_m": float(estimated_width_horizontal),
    }
    (output_dir / "pose_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
'''
