# Hilti-Trimble SLAM Pipeline (Dagger)

Containerized ROS2 pipeline for the Hilti-Trimble SLAM Challenge 2026.

This project runs stage-based processing on ROS2 bag data using Dagger + Docker, while keeping your input bags on the host filesystem.

## What This Pipeline Does

- Orchestrates processing stages from a single CLI (`pipeline.py`)
- Runs ROS tooling in a reproducible container workspace
- Supports stage chaining (`stitch`, `slam`, `plot_path`, etc.)
- Supports PCA-based trajectory alignment as an intermediate step
- Preserves per-stage logs/status files for debugging
- Exports artifacts to a structured output directory per bag

## Repository Layout

```text
pipeline.py                # CLI entrypoint + orchestration
stages/                    # Stage implementations
  base.py                  # Stage interfaces + config
  stitch.py                # Fisheye -> equirectangular ROS bag
  convert.py               # ROS2 bag -> EuRoC format
  slam.py                  # OpenVINS SLAM stage wrapper
  slam_runner.py           # In-container SLAM runtime + diagnostics
  pca_align.py             # PCA-based trajectory alignment
  plot_path.py             # Render trajectory image
  floorplan_overlay.py     # Overlay trajectory on floorplan placeholder/image
  clean.py                 # Host-side output cleanup stage
data/                      # Example ROS2 bag inputs
results/                   # Common output location
Dockerfile.workspace       # ROS2/OpenVINS workspace image
```

## Requirements

- Python `>=3.10`
- Docker (daemon running)
- Network access for first workspace image build (clones challenge repos)
- Recommended local environment: Conda env `3dvis`

## Setup

```bash
# Optional but recommended
conda activate 3dvis

# Runtime dependencies
pip install -r requirements.txt

# Dev tools (pytest, ruff)
pip install -e ".[dev]"
```

### First-Run Note

On first run, the pipeline builds `slam-workspace:latest` from `Dockerfile.workspace`, then exports it to `.cache/slam-workspace.tar`. This can take several minutes.

## Quick Start

```bash
# Show available stages
python pipeline.py --list-stages

# Run SLAM only
python pipeline.py --stages slam \
  --input data/floor_1/2025-05-05/run_1/rosbag \
  --output results/

# SLAM + rendered trajectory image
python pipeline.py --stages slam plot_path \
  --input data/floor_1/2025-05-05/run_1/rosbag \
  --output results/

# SLAM + PCA alignment + rendered trajectory image
python pipeline.py --stages pca_align plot_path \
  --input data/floor_1/2025-05-05/run_1/rosbag \
  --output results/

# SLAM + floorplan overlay placeholder
python pipeline.py --stages slam floorplan_overlay \
  --input data/floor_1/2025-05-05/run_1/rosbag \
  --output results/
```

## CLI Reference

```bash
python pipeline.py --help
```

Key arguments:

- `--stages/-s`: ordered stage list to run
- `--input/-i`: one or more bag paths (supports quoted globs)
- `--output/-o`: output root directory
- `--list-stages/-l`: print stages and exit
- `--verbose/-v`: extra console output

Stitching options:

- `--no-torch`: force OpenCV stitching path
- `--device {auto,cpu,cuda}`: torch device
- `--jpeg-quality`: stitched JPEG quality

SLAM options:

- `--slam-rate`: bag playback rate (use `<1.0` if init is unstable)
- `--slam-timeout`: hard timeout in seconds (`0` disables)

Visualization options:

- `--floorplan`: optional host path to floorplan image for `floorplan_overlay`

## Stages

| Stage | Purpose | Typical Input | Typical Output |
|---|---|---|---|
| `stitch` | Convert dual fisheye streams to panorama topics | ROS2 bag | stitched ROS2 bag (`rosbag_pano`) |
| `convert` | Export ROS2 bag to EuRoC layout | ROS2 bag | EuRoC directory |
| `slam` | Run OpenVINS visual-inertial SLAM | ROS2 bag with `cam0/cam1 + imu` | `trajectory.txt` + SLAM logs |
| `pca_align` | Reorient trajectory using PCA axes | `trajectory.txt` | aligned `trajectory.txt` |
| `plot_path` | Draw 2D trajectory image | `trajectory.txt` | `trajectory_path.png` |
| `floorplan_overlay` | Draw trajectory over floorplan image | `trajectory.txt` (+ optional floorplan image) | `floorplan_overlay.png` |
| `clean` | Remove output directory for the input run | output folder as input | cleaned host output |

## Automatic Stage Dependencies

The pipeline auto-adds dependencies when needed:

- `pca_align` automatically includes `slam`
- `plot_path` automatically includes `slam`
- `floorplan_overlay` automatically includes `slam`

If `pca_align` is explicitly included together with `plot_path` or `floorplan_overlay`, the aligned trajectory is used by downstream stages.

Smart skip behavior is also implemented:

- If a valid `trajectory.txt` already exists in the target output for the bag, `slam` is skipped.
- If input is a non-rosbag directory containing `trajectory.txt`, `slam` is filtered out for `plot_path`/`floorplan_overlay` runs.

## Input Conventions

Expected bag layout:

```text
data/<site>/<date>/run_<n>/rosbag/
  metadata.yaml
  rosbag.db3
```

Example:

```text
data/floor_1/2025-05-05/run_1/rosbag
```

For `slam`, the bag must include:

- `/cam0/image_raw/compressed`
- `/cam1/image_raw/compressed`
- `/imu/data_raw`

## Output Structure and Artifacts

For a rosbag input, results are exported under:

```text
<output>/<site>_<date>_run_<n>/
```

Common SLAM artifacts:

- `trajectory.txt`: TUM-like pose lines (`timestamp tx ty tz qx qy qz qw`)
- `slam.log`: stage-level log stream
- `slam_debug.log`: structured debug messages
- `openvins.log`: OpenVINS launch/runtime logs
- `bag_info.txt`: `ros2 bag info` output
- `pose_topic_info.txt`: publisher/subscriber info for `/ov_msckf/poseimu`
- `trajectory_logger.log`: logger node output
- `bag_play.log`: `ros2 bag play` output

Failure artifacts:

- Failed stage outputs are exported to `_failed/<stage_name>/`
- For SLAM failures, `slam_diagnosis.txt` is created when a known init pattern is detected

Visualization artifacts:

- `trajectory_path.png` from `plot_path`
- `floorplan_overlay.png` from `floorplan_overlay`

PCA alignment artifacts:

- `trajectory.txt`: rewritten with aligned positions
- `pca_alignment_matrix.txt`: alignment basis matrix
- `pca_alignment_info.txt`: point count, mean, eigenvalues

## Floorplan Overlay Behavior

`floorplan_overlay` tries floorplan images in this order:

1. `--floorplan` host path (if provided)
2. `/input/floorplan.(png|jpg|jpeg)`
3. `/input/map.(png|jpg|jpeg)`
4. Fallback generated placeholder floorplan

Scaling and orientation are intentionally placeholder-level for now.

## SLAM Debugging Guide

If SLAM produces no trajectory:

1. Check `slam.log` for final pose count and warnings.
2. Inspect `slam_diagnosis.txt` (if present).
3. Review `openvins.log` for repeated init failures.
4. Confirm `pose_topic_info.txt` and `trajectory_logger.log` indicate the logger attached.

Recommended fixes:

- Lower playback rate, e.g. `--slam-rate 0.5`
- Ensure you run `slam` on original dual-camera bag (not stitched-only topics)
- Re-run and inspect `_failed/slam/` artifacts

Note: `stdin is not a terminal device. Keyboard handling disabled.` from `ros2 bag play` is expected in non-interactive container execution.

## Examples

### Process multiple runs with glob

```bash
python pipeline.py --stages slam \
  --input "data/floor_1/*/run_*/rosbag" \
  --output results/
```

### Force safer SLAM rate + overlay

```bash
python pipeline.py --stages slam floorplan_overlay \
  --input data/floor_1/2025-05-05/run_1/rosbag \
  --output results/ \
  --slam-rate 0.5
```

### PCA-align, then plot path

```bash
python pipeline.py --stages pca_align plot_path \
  --input data/floor_1/2025-05-05/run_1/rosbag \
  --output results/ \
  --slam-rate 0.5
```

### PCA-align, then overlay on floorplan

```bash
python pipeline.py --stages pca_align floorplan_overlay \
  --input data/floor_1/2025-05-05/run_1/rosbag \
  --output results/ \
  --slam-rate 0.5 \
  --floorplan ./data/floorplans/masks_no_windows/floor_1.png
```

### Use an explicit floorplan image

```bash
python pipeline.py --stages slam floorplan_overlay \
  --input data/floor_1/2025-05-05/run_1/rosbag \
  --output results/ \
  --slam-rate 0.5 \
  --floorplan /path/to/floorplan.png
```

### Clean a run output

```bash
python pipeline.py --stages clean \
  --input results/floor_1_2025-05-05_run_1 \
  --output results/
```

## Development

Lint:

```bash
ruff check .
```

Tests (if present):

```bash
pytest
```

### Adding a New Stage

1. Add a class in `stages/` that inherits `Stage`.
2. Implement required properties and `run(...)`.
3. Register it in `stages/__init__.py`.
4. Run `python pipeline.py --list-stages` to verify registration.

## Notes

- Input `data/` is mounted at runtime; avoid writing processing outputs into `data/`.
- SLAM artifacts are mirrored to the selected output and to `results/` for convenience.
- The pipeline intentionally uses a fast process exit after final export because Dagger disconnect can hang on this machine during teardown.
