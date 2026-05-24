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
- Docker for Docker runtime execution, and optionally for faster Apptainer image conversion
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

### Fresh Clone

```bash
git clone <your-repo-url> hilti-slam-challenge-pipeline
cd hilti-slam-challenge-pipeline
```

### Python Environment

Use a standard virtual environment. The project no longer assumes Conda.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Runtime dependencies
pip install -r requirements.txt

# Optional: dev tools (pytest, ruff)
pip install -e ".[dev]"
```

### Host Prerequisites

Pick the subset you need:

- Docker for Docker runtime execution and for the fastest Apptainer conversion path
- Apptainer or Singularity for `--container-runtime apptainer`
- `nvidia-smi` for GPU runs
- network access for first-time model/image downloads

Quick checks:

```bash
python --version
docker --version
apptainer --version   # or singularity --version
nvidia-smi            # only required for GPU runs
```

### Runtime Dependency Notes

- `windows` and `windows_rosbag` do not require `numpy` in the host venv to run successfully
- host-side `numpy` is only used opportunistically for richer bundle statistics in `windows/metadata.json`
- container images carry the ML stack; the host venv only needs the Python orchestration dependencies

## Image Build Flows

### Docker Runtime

For Docker runs, the pipeline builds images on demand:

- `slam-workspace:latest` from `Dockerfile.workspace`
- `windows-pipeline-cpu:latest` from `Dockerfile.windows.cpu`
- `windows-pipeline-gpu:latest` from `Dockerfile.windows.gpu`

You can also build them manually:

```bash
docker build -t slam-workspace:latest -f Dockerfile.workspace .
docker build -t windows-pipeline-cpu:latest -f Dockerfile.windows.cpu .
docker build -t windows-pipeline-gpu:latest -f Dockerfile.windows.gpu .
```

### Apptainer Runtime

For Apptainer runs, the backend tries these paths in order:

1. use a prebuilt `.sif` from environment variables
2. if Docker is available, convert a Docker image/archive into `.sif`
3. if Docker is not available, build from native definition files in `container_defs/`

Supported override variables:

```bash
export PIPELINE_APPTAINER_ROS_IMAGE=/path/to/slam-workspace.sif
export PIPELINE_APPTAINER_WINDOWS_CPU_IMAGE=/path/to/windows-pipeline-cpu.sif
export PIPELINE_APPTAINER_WINDOWS_GPU_IMAGE=/path/to/windows-pipeline-gpu.sif
```

Only set the images you actually need.

### Recommended Apptainer Build Workflow

If Docker is available on the build machine, this is the most reliable path:

```bash
docker build --no-cache -t windows-pipeline-gpu:latest -f Dockerfile.windows.gpu .
docker save windows-pipeline-gpu:latest -o windows-pipeline-gpu.tar
apptainer build windows-pipeline-gpu.sif docker-archive://$(pwd)/windows-pipeline-gpu.tar
```

For CPU:

```bash
docker build --no-cache -t windows-pipeline-cpu:latest -f Dockerfile.windows.cpu .
docker save windows-pipeline-cpu:latest -o windows-pipeline-cpu.tar
apptainer build windows-pipeline-cpu.sif docker-archive://$(pwd)/windows-pipeline-cpu.tar
```

For the ROS workspace:

```bash
docker build -t slam-workspace:latest -f Dockerfile.workspace .
docker save slam-workspace:latest -o slam-workspace.tar
apptainer build slam-workspace.sif docker-archive://$(pwd)/slam-workspace.tar
```

If Docker is not available but Apptainer build is permitted, build from definition files directly:

```bash
apptainer build windows-pipeline-gpu.sif container_defs/windows_gpu.def
apptainer build windows-pipeline-cpu.sif container_defs/windows_cpu.def
apptainer build slam-workspace.sif container_defs/workspace.def
```

### Validate a Fresh GPU Window Image

Before copying a new GPU `.sif` anywhere, validate the Docker image locally:

```bash
docker run --rm windows-pipeline-gpu:latest \
  python -c "import torch; print(torch.__version__); print(torch.version.cuda)"

docker run --rm windows-pipeline-gpu:latest \
  python -c "import cv2, groundingdino, sam3; print('imports ok')"
```

For the current GPU image, expected values are:

- `torch 2.7.0+cu128`
- `CUDA 12.8`

## First-Run Behavior

- the first SLAM run builds the ROS workspace image if missing
- the first window run builds either the CPU or GPU window image if missing
- the first GroundingDINO run downloads `groundingdino_swint_ogc.pth` automatically
- `window_sam` still needs either:
  - `HF_TOKEN` with access to gated `facebook/sam3`, or
  - `--sam3-checkpoint /absolute/path/to/sam3.pt`

Example:

```bash
export HF_TOKEN=<your_token>
```

or:

```bash
python pipeline.py --stages windows \
  --input /path/to/image.png \
  --output results/ \
  --windows-device cuda \
  --sam3-checkpoint /absolute/path/to/sam3.pt
```

These image-only stages are still transitional. The long-term target remains a ROS-native subscriber-based window stage.

## Cluster / Prebuilt SIF Deployment

On systems where local `apptainer build` is restricted, build `.sif` files on another machine and copy them over.

Example copy to ETH cluster:

```bash
scp windows-pipeline-gpu.sif \
  fniederer@student-cluster1.inf.ethz.ch:/work/courses/3dv/team6/hilti-slam-challenge-pipeline/
```

Then on the target system:

```bash
export PIPELINE_APPTAINER_WINDOWS_GPU_IMAGE=/work/courses/3dv/team6/hilti-slam-challenge-pipeline/windows-pipeline-gpu.sif
```

Run the pipeline normally after that.

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

# Public file-based window pipeline on CPU
python pipeline.py --stages windows \
  --input /path/to/input.png \
  --output results/ \
  --windows-device cpu

# Public ROS2 bag window pipeline
python pipeline.py --stages windows_rosbag \
  --input data/floor_1/2025-05-05/run_1/rosbag \
  --output results/ \
  --container-runtime apptainer \
  --windows-device cuda
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

- `--windows-device {auto,cpu,cuda}`: device for `windows` / `window_*` stages
- `--windows-prompt`: GroundingDINO prompt
- `--windows-box-threshold`: GroundingDINO box threshold
- `--windows-text-threshold`: GroundingDINO text threshold
- `--sam3-checkpoint`: optional local SAM3 checkpoint path for the transitional file-based pipeline
- `--windows-topic`: preferred image topic for `windows_rosbag` extraction
- `--windows-frame-index`: frame index for `windows_rosbag` extraction (`-1` means middle frame)

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
| `windows` | Public normalized window-perception bundle | image file | `windows/metadata.json` + canonical artifacts |
| `windows_rosbag` | Public normalized window-perception bundle from a ROS2 bag | ROS2 bag | `windows/metadata.json` + canonical artifacts |
| `windows_extract` | Internal bag-to-image adapter for window inference | ROS2 bag | extracted `input.png` + source metadata |
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
- `windows` automatically includes `window_rectify`
- `windows_rosbag` automatically includes `windows_extract` and `window_rectify`
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

## Apptainer Build Workarounds

These are the concrete issues encountered during integration and the workarounds that proved useful.

### 1. `/tmp` quota too small during `apptainer build`

Symptom:

```text
disk quota exceeded
```

Cause:

- Apptainer unpacks OCI / Docker archives under `/tmp` by default
- large GPU images exceed temporary storage quotas quickly

Workaround:

```bash
mkdir -p .apptainer-tmp .apptainer-cache
export APPTAINER_TMPDIR=$PWD/.apptainer-tmp
export TMPDIR=$PWD/.apptainer-tmp
export APPTAINER_CACHEDIR=$PWD/.apptainer-cache
```

Use those variables before `apptainer build`.

### 2. Cluster forbids unprivileged `%post` execution

Symptom:

```text
Could not write info to setgroups: Permission denied
Error while waiting event for user namespace mappings
```

Cause:

- the cluster allows Apptainer runtime but not full unprivileged image builds for these definitions

Practical workaround:

- do not build the `.sif` on the cluster
- build it on a workstation where Docker or Apptainer build works
- copy the finished `.sif` to the cluster
- point the pipeline at it with `PIPELINE_APPTAINER_*_IMAGE`

This ended up being the reliable deployment model.

### 3. Only `singularity` exists, not `apptainer`

Symptom:

- host has `singularity` but no `apptainer` binary

Current behavior:

- `--container-runtime apptainer` now accepts either binary
- no symlink hack is required anymore

### 4. No Docker on the runtime machine

If Docker is unavailable:

- the backend can still run with prebuilt `.sif` images
- or attempt native `.def` builds if the machine permits them

Recommended approach:

- on restricted systems, treat image build and image run as separate steps

### 5. Bind-mounted script permission problems

Observed issues included:

- Docker bind-mounted wrapper scripts not executable under SELinux
- Apptainer output copy failures when preserving permissions

Current workarounds already implemented in the repo:

- Docker backend uses `--security-opt label=disable`
- stage wrappers use `cp -r`, not `cp -a`

### 6. GPU image compatibility on newer NVIDIA GPUs

Problem we hit:

- older PyTorch/CUDA stacks did not support newer GPU architectures such as `sm_120`

Current fix in the repo:

- GPU image uses a newer CUDA/PyTorch stack
- local validation should always include:

```bash
docker run --rm windows-pipeline-gpu:latest \
  python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

### 7. OpenCV GUI / GL / X11 issues under Apptainer

Symptoms included:

- `GLIBC_2.38 not found`
- `libX11.so.6: cannot open shared object file`

Current fix in the repo:

- the window images use `opencv-python-headless`
- GUI-linked OpenCV variants are removed during image build

### 8. GroundingDINO weights and read-only container paths

Problem:

- Apptainer runtime mounted `/opt/windows_pipeline` read-only, so downloading weights into the vendored tree failed

Current fix in the repo:

- GroundingDINO weights auto-download into the writable stage output cache:
  - `grounding_dino/.model_cache/`

### 9. Prebuilt image override path must be real

Symptom:

```text
PIPELINE_APPTAINER_WINDOWS_GPU_IMAGE points to a missing Apptainer image
```

Cause:

- the environment variable pointed to a placeholder path or missing file

Check first:

```bash
ls -lh "$PIPELINE_APPTAINER_WINDOWS_GPU_IMAGE"
```

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

### Run the public window pipeline on GPU

```bash
python pipeline.py --stages windows \
  --input /path/to/input.png \
  --output results/ \
  --windows-device cuda
```

### Force CPU fallback for the public window pipeline

```bash
python pipeline.py --stages windows \
  --input /path/to/input.png \
  --output results/ \
  --windows-device cpu
```

Canonical output bundle:

```text
results/<image_stem>/windows/
  metadata.json
  boxes.npy
  dino_input.jpg
  dino_overlay.jpg
  masks.npy
  segmented_overlay.png
  rectified_mask.png
```

### Run the public window pipeline on a ROS2 bag

```bash
python pipeline.py --stages windows_rosbag \
  --input data/floor_1/2025-05-05/run_1/rosbag \
  --output results/ \
  --container-runtime apptainer \
  --windows-device cuda \
  --windows-topic /cam0/image_raw/compressed \
  --windows-frame-index -1
```

The current `windows_rosbag` implementation is a bag adapter:
- it extracts one representative frame from the chosen image topic
- it runs the validated image-based window stack on that frame
- it emits the same normalized `windows/` output bundle

This is the first integration step before a future ROS-native subscriber stage.

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
