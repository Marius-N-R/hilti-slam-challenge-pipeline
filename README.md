# Hilti-Trimble SLAM Pipeline

Containerized ROS2 pipeline for the Hilti-Trimble SLAM Challenge 2026.

This project runs stage-based processing on ROS2 bag data using a native Python execution backend with Docker or Apptainer, while keeping input bags on the host filesystem.

## What This Pipeline Does

- Orchestrates processing stages from a single CLI (`pipeline.py`)
- Runs ROS tooling in a reproducible container workspace
- Supports stage chaining (`stitch`, `slam`, `plot_path`, etc.)
- Targets ROS-native perception stages that consume bag topics via subscribers
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
Dockerfile.windows.*       # Transitional image-only window pipeline images
third_party/windows_pipeline/ # Vendored GroundingDINO / SAM3 / py360convert
```

## Requirements

- Python `>=3.10`
- Docker for image builds and Docker runtime execution
- Apptainer or Singularity for `--container-runtime apptainer`
- Network access for first workspace image build (clones challenge repos)
- Network access for the first `window_dino` run (downloads the GroundingDINO checkpoint)
- Recommended local environment: Python virtual environment (`venv`)
- For `window_*` GPU runs: NVIDIA GPU, `nvidia-smi`, and runtime GPU support (`docker --gpus all` or `apptainer/singularity --nv`)

## Window Detection Direction

The intended long-term integration for window detection is a ROS-native stage that:

- subscribes directly to bag-replayed image topics inside the container
- runs detection/segmentation from ROS callbacks instead of from a standalone image file
- optionally consumes IMU or SLAM pose topics for downstream projection
- exports masks, boxes, debug images, and per-frame metadata to the stage output directory

In practice, that means the preferred architecture is:

1. a ROS-enabled container profile that contains both the existing ROS workspace and the vendored ML stack
2. a `window_detect_ros` stage with `requires_ros_runtime = True`
3. a ROS2 node that subscribes to camera topics during `ros2 bag play`
4. optional downstream stages such as `window_project` for floor projection or SLAM alignment

The current `window_dino`, `window_sam`, and `window_rectify` stages are a transitional, file-based integration used to stabilize the vendored model stack before moving the full window pipeline into the ROS execution model.

## Setup

```bash
# Optional but recommended
python -m venv .venv
source .venv/bin/activate

# Runtime dependencies
pip install -r requirements.txt

# Optional: dev tools (pytest, ruff)
pip install -e ".[dev]"
```

### First-Run Note

On first run, the pipeline builds `slam-workspace:latest` from `Dockerfile.workspace`. Docker runs use that image directly. Apptainer runs also build a cached `.sif` image from the Docker image, which can take several additional minutes.

The first `window_*` run also builds either `windows-pipeline-cpu:latest` or `windows-pipeline-gpu:latest`. Apptainer runs additionally cache matching `.sif` images under `.cache/`.

For the transitional `window_*` stages, the first window-image build downloads the upstream GroundingDINO checkpoint into the image. The runtime also re-downloads it if the file is missing.

These image-only stages are transitional. The planned ROS-native window stage will eventually live in a ROS-capable container profile instead of a separate file-processing image.

## Quick Start

```bash
# Show available stages
python pipeline.py --list-stages

# Run SLAM only
python pipeline.py --stages slam \
  --input data/floor_1/2025-05-05/run_1/rosbag \
  --output results/ \
  --slam-rate 0.5

# SLAM + rendered trajectory image
python pipeline.py --stages slam plot_path \
  --input data/floor_1/2025-05-05/run_1/rosbag \
  --output results/ \
  --slam-rate 0.5

# SLAM + PCA alignment + rendered trajectory image
python pipeline.py --stages pca_align plot_path \
  --input data/floor_1/2025-05-05/run_1/rosbag \
  --output results/ \
  --slam-rate 0.5

# SLAM + floorplan overlay placeholder
python pipeline.py --stages slam floorplan_overlay \
  --input data/floor_1/2025-05-05/run_1/rosbag \
  --output results/ \
  --slam-rate 0.5

# Transitional file-based window pipeline on CPU
python pipeline.py --stages window_rectify \
  --input /path/to/input.png \
  --output results/ \
  --windows-device cpu
```

## CLI Reference

```bash
python pipeline.py --help
```

Key arguments:

- `--stages/-s`: ordered stage list to run
- `--input/-i`: one or more bag paths (supports quoted globs)
- `--output/-o`: output root directory
- `--container-runtime {docker,apptainer}`: execution backend
  `apptainer` mode accepts either the `apptainer` or `singularity` host binary
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

Window segmentation options:

- `--windows-device {auto,cpu,cuda}`: device for `window_*` stages
- `--windows-prompt`: GroundingDINO prompt
- `--windows-box-threshold`: GroundingDINO box threshold
- `--windows-text-threshold`: GroundingDINO text threshold
- `--sam3-checkpoint`: optional local SAM3 checkpoint path for the transitional file-based pipeline

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
| `window_dino` | Transitional file-based window box detection | image file | `grounding_dino/bb.npy` + previews |
| `window_sam` | Transitional file-based SAM3 segmentation | image file or `window_dino` output | `windows_masks.npy`, `windows_segmented.png` |
| `window_rectify` | Transitional file-based mask rectification | `window_sam` output | `undistorted/mask_undistorted.png` |

Planned, not yet implemented:

- `window_detect_ros`: ROS-native subscriber stage for bag-replayed camera topics
- `window_project`: downstream projection/alignment stage for window detections

## Automatic Stage Dependencies

The pipeline auto-adds dependencies when needed:

- `pca_align` automatically includes `slam`
- `plot_path` automatically includes `slam`
- `floorplan_overlay` automatically includes `slam`
- `window_sam` automatically includes `window_dino`
- `window_rectify` automatically includes `window_sam`

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

For the planned ROS-native window stage, the expected source is bag-replayed camera topics, not a dedicated standalone image file.

For the current transitional `window_*` stages, pass a single image file as `--input`.

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

Window segmentation artifacts:

- `grounding_dino/raw_image.jpg`
- `grounding_dino/pred.jpg`
- `grounding_dino/bb.npy`
- `windows_segmented.png`
- `windows_masks.npy`
- `undistorted/mask_undistorted.png`

For the planned ROS-native window stage, expected artifacts are:

- per-frame masks
- per-frame boxes
- debug overlays
- optional projected floor points
- optional ROS-topic-derived summaries

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

### Transitional: run only GroundingDINO on an image

```bash
python pipeline.py --stages window_dino \
  --input /path/to/input.png \
  --output results/ \
  --windows-device cpu
```

### Transitional: run the full window pipeline on GPU

```bash
python pipeline.py --stages window_rectify \
  --input /path/to/input.png \
  --output results/ \
  --windows-device cuda
```

### Transitional: force CPU fallback for the full window pipeline

```bash
python pipeline.py --stages window_rectify \
  --input /path/to/input.png \
  --output results/ \
  --windows-device cpu
```

### Planned ROS-native window flow

```bash
# Target direction, not yet implemented as a concrete stage name:
# replay rosbag -> ROS subscriber node -> window detections/masks in /output
python pipeline.py --stages window_detect_ros \
  --input data/floor_1/2025-05-05/run_1/rosbag \
  --output results/
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
- Docker remains the image build source of truth because the repo currently ships Dockerfiles, while Apptainer runtime uses cached `.sif` images derived from those builds.
