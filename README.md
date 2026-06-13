# Hilti-Trimble SLAM Pipeline

Containerized ROS2 pipeline for the Hilti-Trimble SLAM Challenge 2026.

This project runs a staged SLAM, floorplan, Window-detection, realignment, overlay, and evaluation workflow on ROS2 bag data. ROS-heavy stages run in Docker or Apptainer, while host-only geometry and Window stages read and write normal filesystem artifacts.

## What This Pipeline Does

- Orchestrates the full processing chain from a single CLI (`pipeline.py`)
- Runs ROS tooling in a reproducible container workspace
- Extracts floorplan wall edges and image line rays for floorplan-based realignment
- Runs an optional Window detection/segmentation flow and derives a second realignment
- Combines floorplan and Window realignments with configurable weights
- Renders overlays and evaluates the final trajectory against ground truth
- Preserves per-stage logs/status files for debugging
- Exports artifacts to a structured output directory per bag

## Repository Layout

```text
pipeline.py                # CLI entrypoint + orchestration
stages/                    # Stage implementations
  base.py                  # Stage interfaces + config
  all.py                   # Aggregate full-pipeline stage
  slam.py                  # OpenVINS SLAM stage wrapper
  slam_runner.py           # In-container SLAM runtime + diagnostics
  align.py                 # SLAM-to-cam0 CSV conversion and optional start alignment
  pca_align.py             # Optional PCA trajectory reorientation
  line_extractor.py        # Containerized image-line extraction wrapper
  floorplan_edges.py       # DXF wall-segment extraction
  rays.py                  # Camera line back-projection
  floorplan_align.py       # Floorplan-based trajectory realignment
  window_*.py              # Window image-flow, pose, alignment, and overlay stages
  combined_*.py            # Weighted realignment fusion and overlay
  final_eval.py            # Ground-truth evaluation
  final_output.py          # Final artifact collector
data/                      # Example ROS2 bag inputs
results/                   # Common output location
requirements-window.txt    # Window container Python dependencies
third_party/window/        # Vendored Window source: GroundingDINO, SAM3, py360convert
Dockerfile.workspace       # ROS2/OpenVINS workspace image
Dockerfile.window          # Window/GroundingDINO/SAM runtime image
container_defs/workspace.def # ROS Apptainer/Singularity definition
container_defs/window.def  # Window Apptainer/Singularity definition
```

## Getting Started

### 1. Clone the Repository

```bash
git clone <your-repo-url> hilti-slam-challenge-pipeline
cd hilti-slam-challenge-pipeline
```

### 2. Install Required Host Dependencies

Required on the host:

- Python `>=3.10`
- `pip` and `venv`
- Docker, or Apptainer/Singularity
- Network access for the first container image build
- Host Python packages from `requirements.txt`: `numpy`, `ezdxf`, `matplotlib`

Recommended for development:

- `pytest`
- `ruff`

Create the local Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

Quick checks:

```bash
python --version
python -c "import numpy, ezdxf, matplotlib"
```

### 3. Install or Configure a Container Runtime

The ROS/OpenVINS stages run in a container. You need one of these:

- Docker for `--container-runtime docker` (default)
- Apptainer or Singularity for `--container-runtime apptainer`

Check the runtime:

```bash
docker --version
apptainer --version   # or singularity --version
```

The ROS container includes ROS Jazzy, OpenVINS, challenge packages, OpenCV, `cv_bridge`, `rosbag2_py`, and the in-container Python packages needed by `slam`, `line_extractor`, and `image_selector`.

### 4. Configure the Window Pipeline Dependencies

The Window stages run in a separate `window-workspace:repo-local-v2` container. The required Window source code is vendored in this repository:

```text
third_party/window/
  GroundingDINO/
  sam3/
  py360convert/
```

Python dependencies are installed into `/opt/window_venv` during the Window container build from `requirements-window.txt`. The large model files are not checked into Git:

- GroundingDINO downloads `groundingdino_swint_ogc.pth` into `.cache/window-models/` on first use.
- SAM3 and transformer assets download through Hugging Face into `.cache/window-models/huggingface/`.

The default source path is `third_party/window`. Override it only if you need a different checkout:

```bash
python pipeline.py ... --window-root /path/to/window-code
```

`window_sam` loads Meta SAM3 from the gated Hugging Face repository `facebook/sam3`.
Before running `window_sam` or `all`, export a Hugging Face access token that has been granted access to that model, unless the SAM3 checkpoint is already cached in the Window runtime environment:

```bash
export HF_TOKEN=...
```

Do not commit or paste the token into tracked files. If the token is missing or does not have SAM3 access, the stage fails while downloading `config.json` or `sam3.pt`. Downloaded models remain in `.cache/window-models/`, which is ignored by Git.

CUDA is optional. Use `--window-device auto`, `--window-device cpu`, or `--window-device cuda`.
With Docker, only `--window-device cuda` requests `docker run --gpus all`. The default `auto` is CPU-safe for GroundingDINO and does not request Docker GPU access.

### 5. Prepare an Input Run Folder

Each `--input` is a run folder. It must contain a ROS2 bag:

```text
data/floor_1/
  rosbag/
    metadata.yaml
    rosbag.db3
```

Additional files are required by specific stages:

- `initial-pos.txt`: required only when `--align-start-position` is used.
- At least one `*.dxf` file: required by `floorplan_edges`; if multiple exist, the first sorted path is used.
- `floorplan_offset.txt`: optional `(offset_x, offset_y)` in meters for the DXF floorplan edges.
- `<input_folder_name>.png` or `floorplan.png`: required by overlay stages.
- `groundtruth.txt`: required by `final_eval`.

The main camera/IMU topics expected by the container stages are:

- `/cam0/image_raw/compressed`
- `/cam1/image_raw/compressed`
- `/imu/data_raw`

### 6. Run the Pipeline

Show available stages:

```bash
python pipeline.py --list-stages
```

Run the full pipeline:

```bash
export HF_TOKEN=...  # required by window_sam/all unless SAM3 is cached locally

python pipeline.py --stages all --align-start-position \
  --input data/floor_1 \
  --output ./out \
  --image-frames 100,250,400 \
  --floorplan-realign-weight 1.0 \
  --window-realign-weight 0.1 \
  --slam-rate 0.5 --window-device cpu
```

Run the full pipeline with start alignment, optional PCA, and weighted combined realignment:

```bash
export HF_TOKEN=...  # required by window_sam/all unless SAM3 is cached locally

python pipeline.py --stages all --align-start-position --include-pca-align \
  --input data/floor_1 \
  --output ./out \
  --image-frames 100,250,400 \
  --floorplan-realign-weight 1.0 \
  --window-realign-weight 0.1 \
  --slam-rate 0.5 --window-device cpu
```

Run only SLAM:

```bash
python pipeline.py --stages slam \
  --input data/floor_1 \
  --output ./out \
  --slam-rate 0.5
```

## Image Build Flows

### Docker Runtime

For Docker runs, the pipeline builds images on demand:

- `slam-workspace:latest` from `Dockerfile.workspace` for ROS/OpenVINS stages
- `window-workspace:repo-local-v2` from `Dockerfile.window` for Window/GroundingDINO/SAM stages

You can also build them manually:

```bash
docker build -t slam-workspace:latest -f Dockerfile.workspace .
docker build -t window-workspace:repo-local-v2 -f Dockerfile.window .
```

### Apptainer Runtime

For Apptainer runs, the backend tries these paths in order:

1. use a prebuilt `.sif` from `PIPELINE_APPTAINER_ROS_IMAGE`
2. if Docker is available, convert a Docker image/archive into `.sif`
3. if Docker is not available, build from `container_defs/workspace.def`

Override:

```bash
export PIPELINE_APPTAINER_ROS_IMAGE=/path/to/slam-workspace.sif
export PIPELINE_APPTAINER_WINDOW_IMAGE=/path/to/window-workspace.sif
```

### Recommended Apptainer Build Workflow

If Docker is available on the build machine:

```bash
docker build -t slam-workspace:latest -f Dockerfile.workspace .
docker build -t window-workspace:repo-local-v2 -f Dockerfile.window .
docker save slam-workspace:latest -o slam-workspace.tar
docker save window-workspace:repo-local-v2 -o window-workspace-repo-local-v2.tar
apptainer build slam-workspace.sif docker-archive://$(pwd)/slam-workspace.tar
apptainer build window-workspace-repo-local-v2.sif docker-archive://$(pwd)/window-workspace-repo-local-v2.tar
```

If Docker is not available but Apptainer build is permitted:

```bash
apptainer build slam-workspace.sif container_defs/workspace.def
apptainer build window-workspace.sif container_defs/window.def
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
- `--align-start-position`: use `initial-pos.txt` in the `align` stage to place the CSV trajectory in the map frame
- `--include-pca-align`: when using `all`, insert `pca_align` immediately after `align`
- `--eval-max-dt`: maximum timestamp difference for `final_eval` matches in seconds (default: `0.05`)
- `--verbose/-v`: extra console output

Parallel Window image-flow options:

- `--image-frames`: comma-separated frame numbers/ranges for `image_selector`, e.g. `10,20,30-35`
- `--image-topic`: preferred image topic for `image_selector` (default: `/cam0/image_raw/compressed`)
- `--window-root`: path to the Window source directory (default: `third_party/window`)
- `--window-prompt`: GroundingDINO text prompt (default: `windows`)
- `--window-device {auto,cpu,cuda}`: device for Window DINO/SAM stages
- `--window-box-threshold`, `--window-text-threshold`: GroundingDINO thresholds
- `--window-camera-height`: camera height in meters for `window_pose`

Combined realignment options:

- `--floorplan-realign-weight`: weight for the floorplan realignment in `combined_align` (default: `1.0`)
- `--window-realign-weight`: weight for the Window realignment in `combined_align` (default: `1.0`)

SLAM options:

- `--slam-rate`: bag playback rate (use `<1.0` if init is unstable)
- `--slam-timeout`: hard timeout in seconds (`0` disables)

## Stages

| Stage | Purpose | Typical Input | Typical Output |
|---|---|---|---|
| `all` | Expand to the complete dependency-ordered pipeline | Run folder | All stage artifacts |
| `slam` | Run OpenVINS visual-inertial SLAM | ROS2 bag with `cam0/cam1 + imu` | `trajectory.txt` + SLAM logs |
| `align` | Convert SLAM IMU poses to cam0 CSV; optionally align to `initial-pos.txt` | `slam` output + run folder | `trajectory_aligned.csv` |
| `pca_align` | Reorient the aligned CSV trajectory using PCA axes | `align` output | `trajectory_pca_aligned.csv` + PCA diagnostics |
| `line_extractor` | Extract near-horizontal cam0 line detections | Run folder ROS2 bag | `lines.csv` |
| `floorplan_edges` | Extract wall segments from the run DXF | Run folder DXF | `floorplan_edges.csv` |
| `rays` | Back-project line detections using aligned poses | `line_extractor` + `align` outputs | `rays.csv` |
| `floorplan_align` | Refine trajectory against floorplan edges | `rays`, `floorplan_edges`, `align` outputs | `trajectory_floor_aligned.csv` |
| `floorplan_overlay` | Render final trajectory on the floorplan PNG | floorplan PNG + final trajectory | `overlay.png` |
| `image_selector` | Extract selected camera frames from the ROS2 bag | run folder + `--image-frames` | `images/frame_*.png` |
| `window_dino` | Run Window GroundingDINO on selected frames | `image_selector` output | per-frame `grounding_dino/*/bb.npy` |
| `window_sam` | Run Window SAM3 on DINO boxes | `window_dino` output | per-frame `sam3/*/windows_masks.npy` |
| `window_rectify` | Rectify Window SAM3 masks | `window_sam` output | per-frame `rectified/*/mask_undistorted.png` |
| `window_pose` | Compute Window mask-derived window pose metrics | `window_sam` output | `window_pose_summary.csv` |
| `window_align` | Realign trajectory from selected-frame window detections | `window_pose`, `align`, `floorplan_edges` outputs | `trajectory_window_aligned.csv` |
| `window_overlay` | Render the Window-aligned trajectory and window constraints | floorplan PNG + `window_align` output | `window_overlay.png` |
| `combined_align` | Fuse floorplan and Window realignments by weight | `floorplan_align`, `window_align`, `align` outputs | `trajectory_combined_aligned.csv` |
| `combined_overlay` | Render base, floorplan, Window, and combined trajectories | all aligned trajectories | `combined_overlay.png` |
| `final_eval` | Evaluate final trajectory against `groundtruth.txt` | final trajectory + run folder ground truth | `summary.json` + `matched_errors.csv` |
| `final_output` | Collect final trajectories, overlays, and evaluation files | completed pipeline outputs | `manifest.json` + copied artifacts |

## Stage Reference

### `all`

Aggregate stage. It expands to the full ordered pipeline:

```text
slam -> align -> line_extractor -> floorplan_edges -> rays -> floorplan_align -> floorplan_overlay
-> image_selector -> window_dino -> window_sam -> window_rectify -> window_pose
-> window_align -> window_overlay -> combined_align -> combined_overlay -> final_eval -> final_output
```

`--include-pca-align` inserts `pca_align` immediately after `align`. Because `all` includes the Window flow, it requires `--image-frames`.

### `slam`

Runs OpenVINS visual-inertial odometry inside the ROS container.

- Input: run folder with `rosbag/`.
- Required topics: `/cam0/image_raw/compressed`, `/cam1/image_raw/compressed`, `/imu/data_raw`.
- Key options: `--slam-rate`, `--slam-timeout`.
- Output: `trajectory.txt`, `trajectory_ov.txt`, SLAM logs, bag diagnostics.
- Notes: existing valid `trajectory.txt` causes the stage to skip on reruns.

### `align`

Converts SLAM IMU poses to cam0 pose CSV and optionally maps the trajectory into the input map frame.

- Input: `slam/trajectory.txt`.
- Optional input: `initial-pos.txt` when `--align-start-position` is enabled.
- Output: `trajectory_aligned.csv`.
- Notes: start alignment keeps only yaw plus translation, avoiding pitch/roll drift from calibration convention mismatch.

### `pca_align`

Optional PCA reorientation of the CSV trajectory.

- Input: `align/trajectory_aligned.csv`.
- Enabled by: `--include-pca-align` with aggregate stages, or by running `--stages pca_align`.
- Output: `trajectory_pca_aligned.csv`, `pca_alignment_matrix.txt`, `pca_alignment_info.txt`.
- Notes: this is diagnostic/optional; floorplan ray matching still uses the base `align` trajectory.

### `line_extractor`

Detects near-horizontal line segments in cam0 frames inside the ROS container.

- Input: run folder `rosbag/`.
- Dependencies: ROS container with OpenCV, `cv_bridge`, `rosbag2_py`.
- Output: `lines.csv`, `line_extractor.log`.
- Notes: filters line segments by ROI, orientation, and minimum pixel length.

### `floorplan_edges`

Extracts 2D wall segments from the run folder DXF.

- Input: first sorted `*.dxf` file in the original run folder.
- Optional input: `floorplan_offset.txt` with two floats in meters.
- Host dependency: `ezdxf`.
- Output: `floorplan_edges.csv`.
- Notes: emits meter-space wall segments used by both floorplan and window realignment.

### `rays`

Back-projects 2D image line detections to 3D rays in the aligned trajectory frame.

- Input: `line_extractor/lines.csv`, `align/trajectory_aligned.csv`.
- Host dependency: `numpy`.
- Output: `rays.csv`.
- Notes: uses fixed cam0 EUCM intrinsics from the reference alignment scripts.

### `floorplan_align`

Computes the floorplan-based residual trajectory correction.

- Input: `rays/rays.csv`, `floorplan_edges/floorplan_edges.csv`, `align/trajectory_aligned.csv`.
- Host dependency: `numpy`.
- Output: `trajectory_floor_aligned.csv`, `floorplan_align.log`.
- Notes: searches for a yaw and translation correction that best matches ray planes to floorplan wall edges.

### `floorplan_overlay`

Renders the floorplan-aligned trajectory on the floorplan image.

- Input: floorplan PNG from `<input_name>.png` or `floorplan.png`.
- Optional inputs: `groundtruth.txt`, `floorplan_edges.csv`.
- Host dependencies: `matplotlib`, `numpy`.
- Output: `overlay.png`.
- Notes: uses the dataset convention of `100 px/m`.

### `image_selector`

Extracts explicit camera frame numbers from the ROS2 bag.

- Input: run folder `rosbag/`.
- Required option: `--image-frames`, for example `100,250,400` or `10,20-25`.
- Optional option: `--image-topic`.
- Dependencies: ROS container with OpenCV, `cv_bridge`, `rosbag2_py`.
- Output: `images/frame_*.png`, `selected_frames.json`.
- Notes: stores image header timestamps when available so selected frames can be matched against trajectory timestamps.

### `window_dino`

Runs GroundingDINO window detection on selected images.

- Input: `image_selector` output.
- Source/dependencies: `third_party/window/GroundingDINO` plus `/opt/window_venv` in the Window container.
- Model cache: downloads `groundingdino_swint_ogc.pth` into `.cache/window-models/` if missing.
- Key options: `--window-root`, `--window-prompt`, `--window-device`, `--window-box-threshold`, `--window-text-threshold`.
- Output: per-frame `grounding_dino/<frame>/bb.npy`, `pred.jpg`, raw image artifacts.
- Notes: runs in the Window container, not the ROS/OpenVINS container.

### `window_sam`

Runs SAM3 segmentation using the GroundingDINO boxes.

- Input: `window_dino` output.
- Source/dependencies: `third_party/window/sam3` plus `/opt/window_venv` in the Window container.
- Required environment: `HF_TOKEN` with access to the gated `facebook/sam3` Hugging Face model, unless the SAM3 model is already cached.
- Model cache: Hugging Face assets are stored under `.cache/window-models/huggingface/`.
- Key option: `--window-device`.
- Output: per-frame `sam3/<frame>/windows_masks.npy`, `windows_segmented.png`.

### `window_rectify`

Rectifies SAM3 window masks with `py360convert`.

- Input: `window_sam` output.
- Source/dependencies: `third_party/window/py360convert` plus `/opt/window_venv` in the Window container.
- Output: per-frame `rectified/<frame>/mask_undistorted.png`.
- Notes: this stage preserves the Window pipeline artifact flow; `window_pose` currently reads the SAM3 mask output.

### `window_pose`

Computes per-frame window geometry from SAM3 masks.

- Input: `window_sam` output.
- Source/dependencies: `/opt/window_venv` in the Window container.
- Key option: `--window-camera-height`.
- Output: `window_pose_summary.csv`, per-frame `pose/<frame>/pose_summary.json`, `corners_debug.png`.
- Notes: estimates window distance, yaw, lateral offset, pitch/roll, and width metrics.

### `window_align`

Builds a trajectory realignment from selected-frame window detections.

- Input: `window_pose` summaries, `image_selector/selected_frames.json`, `align/trajectory_aligned.csv`, `floorplan_edges/floorplan_edges.csv`.
- Host dependency: `numpy`.
- Output: `trajectory_window_aligned.csv`, `window_alignment_transform.json`, `window_alignment_observations.csv`.
- Notes: converts window-relative observations into the map frame, snaps them to nearest floorplan wall segments, then solves a 2D yaw plus translation correction.

### `window_overlay`

Renders the window-aligned trajectory and window constraints.

- Input: floorplan PNG, `window_align` outputs, `align/trajectory_aligned.csv`.
- Host dependencies: `matplotlib`, `numpy`.
- Output: `window_overlay.png`.
- Notes: shows the base trajectory, window-aligned trajectory, observed window edges, and target wall-edge projections.

### `combined_align`

Fuses the floorplan and window realignments.

- Input: `align/trajectory_aligned.csv`, `floorplan_align/trajectory_floor_aligned.csv`, `window_align/trajectory_window_aligned.csv`.
- Key options: `--floorplan-realign-weight`, `--window-realign-weight`.
- Output: `trajectory_combined_aligned.csv`, `combined_alignment.json`.
- Notes: estimates each branch’s residual 2D transform relative to the same base `align` trajectory, then blends yaw and translation by weight.

### `combined_overlay`

Renders all trajectory variants together.

- Input: base, floorplan-aligned, window-aligned, and combined trajectories.
- Host dependencies: `matplotlib`, `numpy`.
- Output: `combined_overlay.png`.
- Notes: useful for visually comparing the two realignment branches against the weighted result.

### `final_eval`

Evaluates the best available trajectory against ground truth.

- Input: `groundtruth.txt` plus the best available estimate.
- Estimate preference order: `combined_align`, `floorplan_align`, `window_align`, `pca_align`, `align`, `slam`.
- Key option: `--eval-max-dt`.
- Output: `summary.json`, `matched_errors.csv`.
- Notes: reports XY, Z, XYZ, and timestamp error statistics.

### `final_output`

Collects the final artifacts into a single folder.

- Input: completed upstream outputs.
- Output: copied trajectories, overlays, alignment metadata, evaluation files, and `manifest.json`.
- Typical output folder: `<output>/final_output/<input_name>/`.
- Notes: missing optional artifacts are listed in `manifest.json` rather than silently ignored.

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

Evaluation artifacts:

- `summary.json`: matched-pose counts and XY/XYZ/Z/time error statistics
- `matched_errors.csv`: per-estimate nearest-ground-truth match and error columns

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
export HF_TOKEN=...  # required by window_sam/all unless SAM3 is cached locally

python pipeline.py --stages all --align-start-position \
  --input data/floor_1 \
  --output ./out \
  --image-frames 100,250,400 \
  --floorplan-realign-weight 1.0 \
  --window-realign-weight 0.1 \
  --slam-rate 0.5 --window-device cpu
```

### Run the complete pipeline with start and PCA alignment

```bash
export HF_TOKEN=...  # required by window_sam/all unless SAM3 is cached locally

python pipeline.py --stages all --align-start-position --include-pca-align \
  --input data/floor_1 \
  --output ./out \
  --image-frames 100,250,400 \
  --floorplan-realign-weight 1.0 \
  --window-realign-weight 0.1 \
  --slam-rate 0.5 --window-device cpu
```

### Evaluate existing outputs

```bash
python pipeline.py --stages final_eval \
  --input data/floor_1 \
  --output ./out
```

### Run the parallel Window image flow

```bash
export HF_TOKEN=...  # required by window_sam unless SAM3 is cached locally

python pipeline.py \
  --stages image_selector window_dino window_sam window_rectify window_pose window_align window_overlay \
  --input data/floor_1 \
  --output ./out \
  --image-frames 100,250,400 \
  --window-device cpu
```

The Window stages can still be run independently, but they are now also included in `all`.
The public stage and option names use `window_*`; by default `--window-root` points at the vendored source under `third_party/window`.

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
