# Hilti-Trimble SLAM Pipeline

Containerized ROS2 pipeline for the Hilti-Trimble SLAM Challenge 2026.

This project runs the OpenVINS SLAM stage on ROS2 bag data using a Python execution backend with Docker or Apptainer, while keeping input bags on the host filesystem.

## What This Pipeline Does

- Orchestrates the SLAM stage from a single CLI (`pipeline.py`)
- Runs ROS tooling in a reproducible container workspace
- Preserves per-stage logs/status files for debugging
- Exports artifacts to a structured output directory per bag

## Repository Layout

```text
pipeline.py                # CLI entrypoint + orchestration
stages/                    # Stage implementations
  base.py                  # Stage interfaces + config
  slam.py                  # OpenVINS SLAM stage wrapper
  slam_runner.py           # In-container SLAM runtime + diagnostics
data/                      # Example ROS2 bag inputs
results/                   # Common output location
Dockerfile.workspace       # ROS2/OpenVINS workspace image
container_defs/workspace.def # Apptainer/Singularity definition
```

## Requirements

- Python `>=3.10`
- Docker for Docker runtime execution, and optionally for faster Apptainer image conversion
- Apptainer or Singularity for `--container-runtime apptainer`
- Network access for first workspace image build (clones challenge repos)
- Recommended local environment: Python virtual environment (`venv`)

## Setup

### Fresh Clone

```bash
git clone <your-repo-url> hilti-slam-challenge-pipeline
cd hilti-slam-challenge-pipeline
```

### Python Environment

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

- Docker for Docker runtime execution and for the fastest Apptainer conversion path
- Apptainer or Singularity for `--container-runtime apptainer`
- network access for first-time image downloads

Quick checks:

```bash
python --version
docker --version
apptainer --version   # or singularity --version
```

## Image Build Flows

### Docker Runtime

For Docker runs, the pipeline builds `slam-workspace:latest` from `Dockerfile.workspace` on demand. You can also build it manually:

```bash
docker build -t slam-workspace:latest -f Dockerfile.workspace .
```

### Apptainer Runtime

For Apptainer runs, the backend tries these paths in order:

1. use a prebuilt `.sif` from `PIPELINE_APPTAINER_ROS_IMAGE`
2. if Docker is available, convert a Docker image/archive into `.sif`
3. if Docker is not available, build from `container_defs/workspace.def`

Override:

```bash
export PIPELINE_APPTAINER_ROS_IMAGE=/path/to/slam-workspace.sif
```

### Recommended Apptainer Build Workflow

If Docker is available on the build machine:

```bash
docker build -t slam-workspace:latest -f Dockerfile.workspace .
docker save slam-workspace:latest -o slam-workspace.tar
apptainer build slam-workspace.sif docker-archive://$(pwd)/slam-workspace.tar
```

If Docker is not available but Apptainer build is permitted:

```bash
apptainer build slam-workspace.sif container_defs/workspace.def
```

## Quick Start

```bash
# Show available stages
python pipeline.py --list-stages

# Run the complete dependency-ordered pipeline on one run folder
python pipeline.py --stages all \
  --input data/floor_1 \
  --output ./out \
  --slam-rate 0.5

# Run the complete pipeline with optional PCA alignment after align
python pipeline.py --stages all --include-pca-align \
  --input data/floor_1 \
  --output ./out \
  --slam-rate 0.5

# Run SLAM (input is a run folder containing a 'rosbag/' subdir)
python pipeline.py --stages slam \
  --input data/floor_1 \
  --output ./out \
  --slam-rate 0.5
# -> writes artifacts to ./out/slam/floor_1/
```

## CLI Reference

```bash
python pipeline.py --help
```

Key arguments:

- `--stages/-s`: ordered stage list to run (default: `slam`). Use `all` to run the complete pipeline.
- `--input/-i`: one or more run folders that each contain a `rosbag/` subdir (supports quoted globs)
- `--output/-o`: output root directory (default: `./out`). Each stage writes to `<output>/<stage>/<input_folder_name>/`.
- `--container-runtime {docker,apptainer}`: execution backend
  `apptainer` mode accepts either the `apptainer` or `singularity` host binary
- `--list-stages/-l`: print stages and exit
- `--include-pca-align`: when using `all`, insert `pca_align` immediately after `align`
- `--verbose/-v`: extra console output

SLAM options:

- `--slam-rate`: bag playback rate (use `<1.0` if init is unstable)
- `--slam-timeout`: hard timeout in seconds (`0` disables)

## Stages

| Stage | Purpose | Typical Input | Typical Output |
|---|---|---|---|
| `all` | Expand to the complete dependency-ordered pipeline | Run folder | All stage artifacts |
| `slam` | Run OpenVINS visual-inertial SLAM | ROS2 bag with `cam0/cam1 + imu` | `trajectory.txt` + SLAM logs |
| `align` | Align SLAM trajectory to `initial-pos.txt` | `slam` output + run folder | `trajectory_aligned.csv` |
| `pca_align` | Reorient the aligned CSV trajectory using PCA axes | `align` output | `trajectory_pca_aligned.csv` + PCA diagnostics |
| `line_extractor` | Extract near-horizontal cam0 line detections | Run folder ROS2 bag | `lines.csv` |
| `floorplan_edges` | Extract wall segments from the run DXF | Run folder DXF | `floorplan_edges.csv` |
| `rays` | Back-project line detections using aligned poses | `line_extractor` + `pca_align` or `align` outputs | `rays.csv` |
| `floorplan_align` | Refine trajectory against floorplan edges | `rays`, `floorplan_edges`, `pca_align` or `align` outputs | `trajectory_floor_aligned.csv` |
| `floorplan_overlay` | Render final trajectory on the floorplan PNG | floorplan PNG + final trajectory | `overlay.png` |

## Smart Skip

If a valid `trajectory.txt` already exists in the target output for the run, `slam` is skipped on re-run.

## Input Conventions

Each `--input` path is a **run folder** that contains a `rosbag/` subdirectory with the actual ROS2 bag files:

```text
data/floor_1/
  rosbag/
    metadata.yaml
    rosbag.db3
```

Example invocation:

```bash
python pipeline.py --stages slam --input data/floor_1 --output ./out
```

For `slam`, the bag must include:

- `/cam0/image_raw/compressed`
- `/cam1/image_raw/compressed`
- `/imu/data_raw`

## Output Structure and Artifacts

Each stage writes to `<output>/<stage>/<input_folder_name>/`. For example:

```text
out/slam/floor_1/
```

SLAM artifacts:

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

## Apptainer Build Workarounds

### 1. `/tmp` quota too small during `apptainer build`

Apptainer unpacks OCI / Docker archives under `/tmp` by default; large images may exceed quotas.

```bash
mkdir -p .apptainer-tmp .apptainer-cache
export APPTAINER_TMPDIR=$PWD/.apptainer-tmp
export TMPDIR=$PWD/.apptainer-tmp
export APPTAINER_CACHEDIR=$PWD/.apptainer-cache
```

### 2. Cluster forbids unprivileged `%post` execution

- do not build the `.sif` on the cluster
- build it on a workstation where Docker or Apptainer build works
- copy the finished `.sif` to the cluster
- point the pipeline at it with `PIPELINE_APPTAINER_ROS_IMAGE`

### 3. Only `singularity` exists, not `apptainer`

`--container-runtime apptainer` accepts either binary; no symlink hack is required.

### 4. Bind-mounted script permission problems

- Docker backend uses `--security-opt label=disable`
- stage wrappers use `cp -r`, not `cp -a`

### 5. Prebuilt image override path must be real

```bash
ls -lh "$PIPELINE_APPTAINER_ROS_IMAGE"
```

## SLAM Debugging Guide

If SLAM produces no trajectory:

1. Check `slam.log` for final pose count and warnings.
2. Inspect `slam_diagnosis.txt` (if present).
3. Review `openvins.log` for repeated init failures.
4. Confirm `pose_topic_info.txt` and `trajectory_logger.log` indicate the logger attached.

Recommended fixes:

- Lower playback rate, e.g. `--slam-rate 0.5`
- Re-run and inspect `_failed/slam/` artifacts

Note: `stdin is not a terminal device. Keyboard handling disabled.` from `ros2 bag play` is expected in non-interactive container execution.

## Examples

### Process multiple runs with glob

```bash
python pipeline.py --stages slam \
  --input "data/floor_*" \
  --output ./out
```

### Force safer SLAM rate

```bash
python pipeline.py --stages slam \
  --input data/floor_1 \
  --output ./out \
  --slam-rate 0.5
```

### Run the complete pipeline

```bash
python pipeline.py --stages all \
  --input data/floor_1 \
  --output ./out \
  --slam-rate 0.5
```

### Run the complete pipeline with PCA alignment

```bash
python pipeline.py --stages all --include-pca-align \
  --input data/floor_1 \
  --output ./out \
  --slam-rate 0.5
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
- Docker remains the image build source of truth because the repo currently ships a Dockerfile, while Apptainer runtime uses cached `.sif` images derived from that build.
