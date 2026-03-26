#!/usr/bin/env python3
"""
Dagger Pipeline for Hilti-Trimble SLAM Challenge 2026

This pipeline orchestrates the SLAM processing workflow in containers.
Data files (rosbags) remain outside the container and are mounted at runtime.

Available stages:
  - stitch   : Convert dual fisheye to 360° equirectangular
  - convert  : Convert ROS2 bags to EuRoC format
  - slam     : Run OpenVINS visual-inertial odometry
  - pca_align: Align SLAM trajectory using PCA
  - plot_path: Render SLAM trajectory to an image
  - floorplan_overlay: Overlay trajectory on a floorplan image
  - clean    : Remove pipeline outputs (leaves input data intact)

Usage:
    python pipeline.py --stages stitch slam --input data/floor_1/2025-05-05/run_1/rosbag
    python pipeline.py --stages convert --input data/floor_1/2025-05-05/run_1/rosbag --output results/
    python pipeline.py --list-stages

Adding Custom Stages:
    1. Create a new stage class in stages/ that inherits from Stage
    2. Register it in stages/__init__.py
"""

import argparse
import asyncio
import os
import shutil
import sys
import time
from glob import glob
from pathlib import Path
from typing import List

import dagger

from stages import registry
from stages.base import StageConfig

# =============================================================================
# Container Building
# =============================================================================

WORKSPACE_IMAGE = "slam-workspace:latest"
WORKSPACE_TARBALL = Path(__file__).parent / ".cache" / "slam-workspace.tar"
DOCKERFILE_PATH = Path(__file__).parent / "Dockerfile.workspace"


def ensure_workspace_image() -> None:
    """Ensure the workspace Docker image exists, building if necessary."""
    import subprocess

    # Check if image exists
    result = subprocess.run(
        ["docker", "images", "-q", WORKSPACE_IMAGE],
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        return  # Image exists

    # Build the image
    print("[build] Building workspace Docker image (this may take 5-10 minutes)...")
    if not DOCKERFILE_PATH.exists():
        raise FileNotFoundError(f"Dockerfile not found: {DOCKERFILE_PATH}")

    result = subprocess.run(
        ["docker", "build", "-t", WORKSPACE_IMAGE, "-f", str(DOCKERFILE_PATH), "."],
        cwd=DOCKERFILE_PATH.parent,
    )
    if result.returncode != 0:
        raise RuntimeError("Failed to build workspace Docker image")
    print("[build] Workspace image built successfully")


def ensure_workspace_tarball() -> Path:
    """Ensure the workspace tarball exists, creating from Docker image if needed."""
    import subprocess

    WORKSPACE_TARBALL.parent.mkdir(parents=True, exist_ok=True)

    if WORKSPACE_TARBALL.exists():
        return WORKSPACE_TARBALL

    ensure_workspace_image()

    print("[build] Exporting workspace image to tarball...")
    result = subprocess.run(
        ["docker", "save", WORKSPACE_IMAGE, "-o", str(WORKSPACE_TARBALL)],
    )
    if result.returncode != 0:
        raise RuntimeError("Failed to export workspace image")
    print(f"[build] Workspace tarball created: {WORKSPACE_TARBALL}")
    return WORKSPACE_TARBALL


async def load_workspace(client: dagger.Client) -> dagger.Container:
    """Load the workspace container from tarball.

    This works around a Dagger issue where containers built with many
    with_exec() calls hang when additional commands are run on them.
    """
    tarball_path = ensure_workspace_tarball()
    tarball = client.host().file(str(tarball_path))
    return client.container().import_(tarball)


# =============================================================================
# Pipeline Orchestration
# =============================================================================

async def run_pipeline(
    stage_names: List[str],
    input_bags: List[str],
    output_dir: str,
    config: StageConfig,
):
    """Run the pipeline with the given stages and inputs."""

    print("=" * 60)
    print("Hilti-Trimble SLAM Challenge Pipeline")
    print("=" * 60)
    print(f"Stages: {', '.join(stage_names)}")
    print(f"Input bags: {len(input_bags)}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    # Validate stages exist
    for name in stage_names:
        if registry.get(name) is None:
            print(f"[error] Unknown stage: {name}")
            print("Use --list-stages to see available stages")
            return 1

    dagger_config = dagger.Config()

    async with dagger.connection(dagger_config):
        client = dagger.dag
        # Load workspace container (from pre-built tarball)
        print("\n[build] Loading workspace container...")
        workspace = await load_workspace(client)
        print("[build] Workspace ready")

        # Process each input bag
        for bag_path in input_bags:
            print(f"\n[pipeline] Processing: {bag_path}")

            # Get bag directory from host
            bag_dir = Path(bag_path)
            if not bag_dir.exists():
                print(f"[error] Bag path does not exist: {bag_path}")
                continue

            # If bag_path is a file, use parent directory
            if bag_dir.is_file():
                bag_dir = bag_dir.parent

            # Resolve to absolute path
            bag_dir = bag_dir.resolve()
            input_data = client.host().directory(str(bag_dir))

            # Determine output subdirectory based on input path
            bag_name = bag_dir.name
            if bag_name == "rosbag":
                # Use parent directories for naming: floor_X_date_run_Y
                # Build safely so absolute roots never become part of the name.
                parent_names: list[str] = []
                current_parent = bag_dir.parent
                for _ in range(3):
                    if not current_parent.name:
                        break
                    parent_names.append(current_parent.name)
                    current_parent = current_parent.parent

                if parent_names:
                    bag_name = "_".join(reversed(parent_names))
                else:
                    bag_name = "rosbag"

            output_subdir = Path(output_dir) / bag_name

            per_bag_stages = filter_stages_for_input(stage_names, bag_dir, output_dir)
            print(f"[pipeline] Stages for this bag: {', '.join(per_bag_stages)}")
            if per_bag_stages != stage_names:
                skipped = [name for name in stage_names if name not in per_bag_stages]
                if skipped:
                    print(f"[pipeline] Skipping stage(s) for this input: {', '.join(skipped)}")

            if not per_bag_stages:
                print("[pipeline] No stages to run after filtering")
                return 1

            # Run selected stages in sequence
            current_data = input_data

            stages = []
            for name in per_bag_stages:
                stage = registry.get(name)
                if stage is None:
                    print(f"[error] Unknown stage: {name}")
                    print("Use --list-stages to see available stages")
                    return 1
                stages.append(stage)

            stage_failed = False
            failed_stage_name = None
            for stage in stages:
                print(f"\n[{stage.name}] Running stage: {stage.description}")
                try:
                    if stage.name == "clean":
                        shutil.rmtree(output_subdir, ignore_errors=True)
                        current_data = input_data
                        continue
                    if stage.name == "slam":
                        trajectory_path = output_subdir / "trajectory.txt"
                        existing_poses = count_trajectory_poses(trajectory_path)
                        if existing_poses >= 2:
                            print(
                                "[slam] Skipping stage: found existing trajectory "
                                f"with {existing_poses} poses at {trajectory_path}"
                            )
                            copied_floorplans = copy_floorplan_assets(bag_dir, output_subdir)
                            if copied_floorplans > 0:
                                print(
                                    "[slam] Copied floorplan asset(s) from input bag to output: "
                                    f"{copied_floorplans}"
                                )
                            current_data = client.host().directory(str(output_subdir.resolve()))
                            continue
                        expected_time = estimate_bag_duration_seconds(bag_dir, config.slam_rate)
                        current_data = await run_stage_with_progress(
                            stage,
                            workspace,
                            current_data,
                            config,
                            expected_time,
                        )
                    else:
                        current_data = await stage.run(workspace, current_data, config)
                    await current_data.id()
                    status = await read_stage_status(current_data, stage.name)
                    if status is not None and status != 0:
                        print(f"[{stage.name}] ERROR: stage exited with status {status}")
                        await print_stage_log_tail(current_data, stage.name)
                        stage_failed = True
                        failed_stage_name = stage.name
                        break
                except Exception as e:
                    print(f"[{stage.name}] ERROR: {e}")
                    stage_failed = True
                    failed_stage_name = stage.name
                    break

            # Export results to host
            if per_bag_stages == ["clean"]:
                print(f"\n[pipeline] Outputs removed: {output_subdir}")
                continue

            if stage_failed:
                if failed_stage_name is not None:
                    failed_dir = output_subdir / "_failed" / failed_stage_name
                    if failed_dir.exists():
                        shutil.rmtree(failed_dir)
                    failed_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        await current_data.export(str(failed_dir))
                        print(f"[pipeline] Exported failed stage artifacts to: {failed_dir}")
                        if failed_stage_name == "slam":
                            mirror_slam_artifacts(
                                source_dir=failed_dir,
                                output_subdir=output_subdir,
                                bag_name=bag_name,
                            )
                    except Exception as export_error:
                        print(f"[pipeline] Failed to export failed stage artifacts: {export_error}")
                print(f"[pipeline] Skipping export because a stage failed for: {bag_path}")
                continue

            output_subdir.mkdir(parents=True, exist_ok=True)
            await current_data.export(str(output_subdir))
            print(f"\n[pipeline] Results exported to: {output_subdir}")

            if "slam" in per_bag_stages:
                mirror_slam_artifacts(
                    source_dir=output_subdir,
                    output_subdir=output_subdir,
                    bag_name=bag_name,
                )

        # Force exit BEFORE the async with block tries to close
        # This avoids hanging on Dagger's connection cleanup
        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print("=" * 60)
        os._exit(0)


def mirror_slam_artifacts(source_dir: Path, output_subdir: Path, bag_name: str) -> None:
    """Mirror core SLAM artifacts to both selected output and results/."""
    artifact_names = [
        "trajectory.txt",
        "slam_debug.log",
        "openvins.log",
        "bag_info.txt",
        "pose_topic_info.txt",
    ]

    found_any = False
    mirrored_any = False
    for artifact_name in artifact_names:
        source_file = source_dir / artifact_name
        if not source_file.exists():
            continue
        found_any = True

        targets = [
            output_subdir / artifact_name,
            Path("results") / bag_name / artifact_name,
        ]

        seen_targets = set()
        source_resolved = source_file.resolve()
        for target in targets:
            target_resolved = target.resolve()
            if target_resolved == source_resolved:
                continue
            if target_resolved in seen_targets:
                continue
            seen_targets.add(target_resolved)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target)
            mirrored_any = True
            print(f"[pipeline] Mirrored {artifact_name} to: {target}")

    if not found_any:
        print(
            "[pipeline] WARNING: No SLAM artifacts found to mirror "
            f"from {source_dir}"
        )
    elif not mirrored_any:
        print("[pipeline] SLAM artifacts already present in target location(s)")


def count_trajectory_poses(trajectory_path: Path) -> int:
    """Count valid pose lines in trajectory.txt."""
    if not trajectory_path.exists():
        return 0

    poses = 0
    try:
        with trajectory_path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                poses += 1
    except Exception:
        return 0
    return poses


def copy_floorplan_assets(input_dir: Path, output_dir: Path) -> int:
    """Copy floorplan-like image assets from input dir to output dir."""
    if not input_dir.exists():
        return 0

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    candidates = []
    candidates.extend(input_dir.glob("floorplan.*"))
    candidates.extend(input_dir.glob("map.*"))

    copied = 0
    seen_targets = set()
    for source in candidates:
        if not source.is_file():
            continue
        if source.suffix.lower() not in image_extensions:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / source.name
        target_resolved = target.resolve()
        if target_resolved in seen_targets:
            continue
        seen_targets.add(target_resolved)

        if source.resolve() == target_resolved:
            continue

        shutil.copy2(source, target)
        copied += 1

    return copied


def estimate_bag_duration_seconds(bag_dir: Path, rate: float) -> float | None:
    """Estimate bag playback time from metadata.yaml, adjusted by rate."""
    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.exists():
        return None
    try:
        with metadata_path.open() as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("nanoseconds:"):
                    value = stripped.split(":", 1)[1].strip()
                    nanoseconds = int(value)
                    if rate <= 0:
                        return None
                    return nanoseconds / 1e9 / rate
    except Exception:
        return None
    return None


async def run_stage_with_progress(
    stage,
    container: dagger.Container,
    input_dir: dagger.Directory,
    config: StageConfig,
    expected_time: float | None,
) -> dagger.Directory:
    """Run a stage while showing a simple progress bar."""
    if not expected_time:
        return await stage.run(container, input_dir, config)

    bar_width = 30
    start = time.time()
    output_dir = await stage.run(container, input_dir, config)
    task = asyncio.create_task(output_dir.id())

    while not task.done():
        elapsed = time.time() - start
        pct = min(100.0, (elapsed / expected_time) * 100.0)
        filled = int((pct / 100.0) * bar_width)
        bar = f"{'#' * filled}{'-' * (bar_width - filled)}"
        line = (
            f"[slam] Progress: [{bar}] {pct:5.1f}% "
            f"({elapsed:5.0f}s/{expected_time:5.0f}s)"
        )
        sys.stdout.write(f"\r{line}")
        sys.stdout.flush()
        await asyncio.sleep(1)

    elapsed = time.time() - start
    pct = min(100.0, (elapsed / expected_time) * 100.0)
    filled = int((pct / 100.0) * bar_width)
    bar = f"{'#' * filled}{'-' * (bar_width - filled)}"
    line = (
        f"[slam] Progress: [{bar}] {pct:5.1f}% "
        f"({elapsed:5.0f}s/{expected_time:5.0f}s)"
    )
    sys.stdout.write(f"\r{line}\n")
    sys.stdout.flush()

    await task
    return output_dir


async def read_stage_status(output_dir: dagger.Directory, stage_name: str) -> int | None:
    """Return non-zero stage status if a status file is present."""
    status_path = f"{stage_name}.status"
    try:
        contents = await output_dir.file(status_path).contents()
    except Exception:
        return None
    try:
        return int(contents.strip())
    except ValueError:
        return None


async def print_stage_log_tail(output_dir: dagger.Directory, stage_name: str) -> None:
    """Print the tail of a stage log if it exists."""
    log_path = f"{stage_name}.log"
    try:
        contents = await output_dir.file(log_path).contents()
    except Exception:
        return
    lines = contents.splitlines()
    tail = lines[-40:] if len(lines) > 40 else lines
    if tail:
        print(f"[{stage_name}] Log tail:")
        for line in tail:
            print(line)


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Dagger Pipeline for Hilti-Trimble SLAM Challenge 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run stitching on a bag
    python pipeline.py --stages stitch --input data/floor_1/2025-05-05/run_1/rosbag

    # Run full pipeline (stitch + slam)
    python pipeline.py --stages stitch slam --input data/floor_1/2025-05-05/run_1/rosbag

    # Process multiple bags with glob patterns
    python pipeline.py --stages slam --input "data/floor_1/*/run_*/rosbag"

    # Convert to EuRoC format
    python pipeline.py --stages convert --input data/floor_1/2025-05-05/run_1/rosbag

    # Overlay trajectory on floorplan placeholder
    python pipeline.py --stages slam floorplan_overlay --input data/floor_1/2025-05-05/run_1/rosbag

    # List available stages
    python pipeline.py --list-stages

Adding Custom Stages:
    Create a Python file in stages/ directory that defines a class
    inheriting from Stage, then register it in stages/__init__.py
        """
    )

    parser.add_argument(
        "--stages", "-s",
        nargs="+",
        help="Stages to run (in order). Use --list-stages to see options."
    )

    parser.add_argument(
        "--input", "-i",
        nargs="+",
        dest="input_bags",
        help="Input rosbag path(s). Supports glob patterns in quotes."
    )

    parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory (default: ./output)"
    )

    parser.add_argument(
        "--list-stages", "-l",
        action="store_true",
        help="List available pipeline stages and exit"
    )

    # Stitching options
    stitch_group = parser.add_argument_group("Stitching options")
    stitch_group.add_argument(
        "--no-torch",
        action="store_true",
        help="Disable PyTorch acceleration for stitching (use OpenCV)"
    )
    stitch_group.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for PyTorch stitching (default: auto)"
    )
    stitch_group.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for stitched images (1-100, default: 95)"
    )

    # SLAM options
    slam_group = parser.add_argument_group("SLAM options")
    slam_group.add_argument(
        "--slam-rate",
        type=float,
        default=1.0,
        help="Playback rate for SLAM (default: 1.0, use <1 for slower)"
    )
    slam_group.add_argument(
        "--slam-timeout",
        type=int,
        default=0,
        help="Timeout for SLAM stage in seconds (0 disables, default: 0)"
    )

    # Visualization options
    viz_group = parser.add_argument_group("Visualization options")
    viz_group.add_argument(
        "--floorplan",
        default="",
        help=(
            "Optional floorplan image path for floorplan_overlay stage. "
            "If omitted, stage uses floorplan/map image from input when available."
        ),
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def expand_stage_dependencies(stage_names: List[str]) -> List[str]:
    """Expand stage list to include required dependencies in order.

    Note: slam runs on the original dual fisheye bag (cam0/cam1 topics),
    NOT on the stitched panorama. Stitch produces /pano/... topics which
    OpenVINS doesn't understand - it expects /cam0/... and /cam1/...
    """
    deps = {
        # slam has no dependencies - runs on original bag with cam0/cam1 topics
        "pca_align": ["slam"],
        "plot_path": ["slam"],
        "floorplan_overlay": ["slam"],
    }
    ordered = []
    seen = set()

    def add_stage(name: str) -> None:
        for dep in deps.get(name, []):
            add_stage(dep)
        if name not in seen:
            ordered.append(name)
            seen.add(name)

    for stage in stage_names:
        add_stage(stage)

    if "pca_align" in stage_names:
        updated = []
        for name in ordered:
            updated.append(name)
            if name == "pca_align":
                continue
            if name in {"plot_path", "floorplan_overlay"} and "pca_align" not in updated:
                updated.append("pca_align")
        ordered = updated

    return ordered


def filter_stages_for_input(
    stage_names: List[str],
    bag_dir: Path,
    output_dir: str,
) -> List[str]:
    """Filter stages based on input data markers."""
    try:
        bag_dir.relative_to(Path(output_dir).resolve())
        is_pipeline_output = True
    except ValueError:
        is_pipeline_output = False

    stitched_bag_layout = (
        (bag_dir / "metadata.yaml").exists()
        and (bag_dir / "rosbag.db3").exists()
        and (bag_dir / ".stitched").exists()
    )

    rosbag_layout = (
        (bag_dir / "metadata.yaml").exists()
        and (bag_dir / "rosbag.db3").exists()
    )
    has_trajectory = (bag_dir / "trajectory.txt").exists()

    if has_trajectory and not rosbag_layout:
        if "plot_path" in stage_names or "floorplan_overlay" in stage_names:
            filtered = [name for name in stage_names if name != "slam"]
            if filtered:
                return filtered

    if is_pipeline_output and stitched_bag_layout:
        return [name for name in stage_names if name != "stitch"]
    return stage_names


def expand_input_paths(patterns: List[str]) -> List[str]:
    """Expand glob patterns in input paths."""
    expanded = []
    for pattern in patterns:
        matches = glob(pattern, recursive=True)
        if matches:
            expanded.extend(sorted(matches))
        else:
            # Keep as-is, will be checked later
            expanded.append(pattern)
    return expanded


def main():
    args = parse_args()

    # List stages and exit
    if args.list_stages:
        registry.print_stages()
        return 0

    # Validate required arguments
    if not args.stages:
        print("Error: No stages specified. Use --stages or --list-stages")
        return 1

    if not args.input_bags:
        print("Error: No input bags specified. Use --input")
        return 1

    # Expand glob patterns
    input_bags = expand_input_paths(args.input_bags)

    if not input_bags:
        print("Error: No input bags found matching the given patterns")
        return 1

    extra_config = {}
    if args.floorplan:
        extra_config["floorplan_path"] = args.floorplan

    # Build stage configuration
    config = StageConfig(
        verbose=args.verbose,
        use_torch=not args.no_torch,
        torch_device=args.device,
        jpeg_quality=args.jpeg_quality,
        slam_rate=args.slam_rate,
        slam_timeout=args.slam_timeout,
        extra=extra_config,
    )

    expanded_stages = expand_stage_dependencies(args.stages)
    if expanded_stages != args.stages:
        print(f"[pipeline] Auto-expanded stages: {', '.join(expanded_stages)}")

    return asyncio.run(run_pipeline(
        stage_names=expanded_stages,
        input_bags=input_bags,
        output_dir=args.output,
        config=config,
    ))


if __name__ == "__main__":
    sys.exit(main())
