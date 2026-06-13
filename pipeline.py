#!/usr/bin/env python3
"""
Hilti-Trimble SLAM Challenge 2026 pipeline.

This pipeline orchestrates the SLAM processing workflow in containers.
Data files (rosbags) remain outside the container and are mounted at runtime.

Available stages:
  - all      : Run the complete pipeline in dependency order
  - slam     : Run OpenVINS visual-inertial odometry
  - pca_align: Optionally PCA-align the CSV trajectory after initial alignment

Usage:
    python pipeline.py --stages all --include-pca-align --input data/floor_1 --output ./out
    python pipeline.py --stages slam --input data/floor_1 --output ./out
    python pipeline.py --list-stages

Input layout:
    Each --input path is a run folder that contains a 'rosbag/' subdirectory
    with the actual ROS2 bag (metadata.yaml + db3).

Output layout:
    Artifacts are written to <output>/<stage_name>/<input_folder_name>/.
    Example: --input data/floor_1 --output ./out  ->  ./out/slam/floor_1/

Adding Custom Stages:
    1. Create a new stage class in stages/ that inherits from Stage
    2. Register it in stages/__init__.py
"""

import argparse
import shutil
import sys
from glob import glob
from pathlib import Path
from typing import List

from runtime_backend import ContainerBackend, StageExecutionError
from stages import registry
from stages.base import Stage, StageConfig


def run_pipeline(
    stage_names: List[str],
    input_bags: List[str],
    output_dir: str,
    config: StageConfig,
    container_runtime: str,
    include_pca_align: bool = False,
):
    """Run the pipeline with the given stages and inputs."""
    try:
        requested_stage_names = list(stage_names)
        stage_names = expand_stage_names(stage_names)
        if include_pca_align:
            stage_names = include_stage_after(
                stage_names,
                stage_name="pca_align",
                after_stage_name="align",
            )
    except ValueError as exc:
        print(f"[error] {exc}")
        print("Use --list-stages to see available stages")
        return 1

    print("=" * 60)
    print("Hilti-Trimble SLAM Challenge Pipeline")
    print("=" * 60)
    print(f"Stages: {', '.join(stage_names)}")
    if requested_stage_names != stage_names:
        print(f"Requested stages: {', '.join(requested_stage_names)}")
    print(f"Input bags: {len(input_bags)}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    runner = ContainerBackend(runtime=container_runtime)

    for input_path_str in input_bags:
        print(f"\n[pipeline] Processing: {input_path_str}")

        input_path = Path(input_path_str)
        if not input_path.exists():
            print(f"[error] Input path does not exist: {input_path_str}")
            continue
        if not input_path.is_dir():
            print(f"[error] Input must be a directory containing 'rosbag/': {input_path_str}")
            continue

        input_path = input_path.resolve()
        folder_name = input_path.name
        config.extra["current_input_path"] = str(input_path)
        config.extra["current_input_name"] = folder_name

        current_data = input_path

        stages: list[Stage] = []
        for name in stage_names:
            stage = registry.get(name)
            if stage is None:
                print(f"[error] Unknown stage: {name}")
                print("Use --list-stages to see available stages")
                return 1
            stages.append(stage)

        for stage in stages:
            stage_output_subdir = (Path(output_dir) / stage.name / folder_name).resolve()
            print(f"\n[{stage.name}] Running stage: {stage.description}")
            print(f"[{stage.name}] Output: {stage_output_subdir}")
            if stage.requires_container:
                print(f"[build] Loading container profile: {stage.container_profile} ({container_runtime})")

            stage_failed = False
            try:
                if stage.name == "slam":
                    trajectory_path = stage_output_subdir / "trajectory.txt"
                    existing_poses = count_trajectory_poses(trajectory_path)
                    if existing_poses >= 2:
                        print(
                            "[slam] Skipping stage: found existing trajectory "
                            f"with {existing_poses} poses at {trajectory_path}"
                        )
                        current_data = stage_output_subdir
                        continue

                stage_temp_output = stage.run(runner, current_data, config)
                status = read_stage_status(stage_temp_output, stage.name)
                if status is not None and status != 0:
                    print(f"[{stage.name}] ERROR: stage exited with status {status}")
                    print_stage_log_tail(stage_temp_output, stage.name)
                    stage_failed = True
            except StageExecutionError as exc:
                stage_temp_output = exc.output_dir
                print(f"[{stage.name}] ERROR: stage exited with status {exc.returncode}")
                print_stage_log_tail(stage_temp_output, stage.name)
                stage_failed = True
            except Exception as e:
                print(f"[{stage.name}] ERROR: {e}")
                stage_failed = True
                stage_temp_output = None

            if stage_failed:
                if stage_temp_output is not None:
                    failed_dir = stage_output_subdir / "_failed" / stage.name
                    if failed_dir.exists():
                        shutil.rmtree(failed_dir)
                    failed_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        copy_tree(stage_temp_output, failed_dir)
                        print(f"[pipeline] Exported failed stage artifacts to: {failed_dir}")
                    except Exception as export_error:
                        print(f"[pipeline] Failed to export failed stage artifacts: {export_error}")
                print(
                    "[pipeline] Skipping remaining stages because a stage failed for: "
                    f"{input_path_str}"
                )
                break

            stage_output_subdir.mkdir(parents=True, exist_ok=True)
            copy_tree(stage_temp_output, stage_output_subdir)
            print(f"[{stage.name}] Results exported to: {stage_output_subdir}")
            current_data = stage_output_subdir

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    return 0


def include_stage_after(
    stage_names: List[str],
    stage_name: str,
    after_stage_name: str,
) -> List[str]:
    """Return stage_names with stage_name inserted after after_stage_name if missing."""
    if stage_name in stage_names:
        return stage_names
    if registry.get(stage_name) is None:
        raise ValueError(f"Unknown stage: {stage_name}")
    try:
        insert_at = stage_names.index(after_stage_name) + 1
    except ValueError as exc:
        raise ValueError(
            f"Cannot include {stage_name}: {after_stage_name} is not in the stage list"
        ) from exc
    return [*stage_names[:insert_at], stage_name, *stage_names[insert_at:]]


def expand_stage_names(stage_names: List[str]) -> List[str]:
    """Expand aggregate stage names into concrete stage names."""
    expanded: list[str] = []
    stack: list[str] = []

    def visit(name: str) -> None:
        stage = registry.get(name)
        if stage is None:
            raise ValueError(f"Unknown stage: {name}")
        if name in stack:
            cycle = " -> ".join([*stack, name])
            raise ValueError(f"Aggregate stage cycle detected: {cycle}")

        children = stage.expanded_stage_names
        if not children:
            expanded.append(name)
            return

        stack.append(name)
        for child_name in children:
            visit(child_name)
        stack.pop()

    for stage_name in stage_names:
        visit(stage_name)
    return expanded


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


def read_stage_status(output_dir: Path, stage_name: str) -> int | None:
    """Return non-zero stage status if a status file is present."""
    status_path = output_dir / f"{stage_name}.status"
    try:
        contents = status_path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        return int(contents.strip())
    except ValueError:
        return None


def print_stage_log_tail(output_dir: Path, stage_name: str) -> None:
    """Print the tail of a stage log if it exists."""
    log_path = output_dir / f"{stage_name}.log"
    try:
        contents = log_path.read_text(encoding="utf-8")
    except Exception:
        return
    lines = contents.splitlines()
    tail = lines[-40:] if len(lines) > 40 else lines
    if tail:
        print(f"[{stage_name}] Log tail:")
        for line in tail:
            print(line)


def copy_tree(source_dir: Path, target_dir: Path) -> None:
    """Replace target directory contents with a copy of source_dir."""
    source_resolved = source_dir.resolve()
    target_resolved = target_dir.resolve()
    if source_resolved == target_resolved:
        return

    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hilti-Trimble SLAM Challenge 2026 pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run the complete dependency-ordered pipeline on one input folder
    python pipeline.py --stages all --input data/floor_1 --output ./out

    # Run the complete pipeline with start alignment and PCA after trajectory CSV conversion
    python pipeline.py --stages all --align-start-position --include-pca-align --input data/floor_1 --output ./out

    # Run SLAM on a single run folder (which contains a 'rosbag/' subdir)
    python pipeline.py --stages slam --input data/floor_1 --output ./out

    # Process multiple run folders with glob patterns
    python pipeline.py --stages slam --input "data/floor_*" --output ./out

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
        default=["slam"],
        help="Stages to run (in order). Use --list-stages to see options."
    )

    parser.add_argument(
        "--input", "-i",
        nargs="+",
        dest="input_bags",
        help="Run folder(s) containing a 'rosbag/' subdirectory. Supports glob patterns in quotes."
    )

    parser.add_argument(
        "--output", "-o",
        default="./out",
        help="Output root (default: ./out). Stages write to <output>/<stage>/<input_folder_name>/."
    )

    parser.add_argument(
        "--container-runtime",
        default="docker",
        choices=["docker", "apptainer"],
        help="Container runtime backend to use (default: docker)",
    )

    parser.add_argument(
        "--list-stages", "-l",
        action="store_true",
        help="List available pipeline stages and exit"
    )

    parser.add_argument(
        "--include-pca-align",
        action="store_true",
        help="When using aggregate stages, insert pca_align after align",
    )

    parser.add_argument(
        "--align-start-position",
        action="store_true",
        help="Use initial-pos.txt to align the converted cam0 CSV trajectory to the map frame",
    )

    parser.add_argument(
        "--eval-max-dt",
        type=float,
        default=0.05,
        help="Maximum timestamp difference for final_eval matches in seconds (default: 0.05)",
    )

    window_group = parser.add_argument_group("Parallel Window image-flow options")
    window_group.add_argument(
        "--image-frames",
        default="",
        help="Comma-separated frame numbers/ranges for image_selector, e.g. 10,20,30-35",
    )
    window_group.add_argument(
        "--image-topic",
        default="/cam0/image_raw/compressed",
        help="Preferred image topic for image_selector",
    )
    window_group.add_argument(
        "--window-root",
        default="third_party/window",
        help="Path to the Window source directory (default: third_party/window)",
    )
    window_group.add_argument(
        "--window-prompt",
        default="windows",
        help='GroundingDINO prompt for window_dino (default: "windows")',
    )
    window_group.add_argument(
        "--window-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for Window DINO/SAM stages (default: auto)",
    )
    window_group.add_argument(
        "--window-box-threshold",
        type=float,
        default=0.3,
        help="GroundingDINO box threshold for window_dino (default: 0.3)",
    )
    window_group.add_argument(
        "--window-text-threshold",
        type=float,
        default=0.25,
        help="GroundingDINO text threshold for window_dino (default: 0.25)",
    )
    window_group.add_argument(
        "--window-camera-height",
        type=float,
        default=2.0,
        help="Camera height in meters for window_pose (default: 2.0)",
    )

    combined_group = parser.add_argument_group("Combined realignment options")
    combined_group.add_argument(
        "--floorplan-realign-weight",
        type=float,
        default=1.0,
        help="Weight for the floorplan realignment in combined_align (default: 1.0)",
    )
    combined_group.add_argument(
        "--window-realign-weight",
        type=float,
        default=1.0,
        help="Weight for the Window realignment in combined_align (default: 1.0)",
    )

    slam_group = parser.add_argument_group("SLAM options")
    slam_group.add_argument(
        "--slam-rate",
        type=float,
        default=0.5,
        help="Playback rate for SLAM (default: 0.5; OpenVINS needs <=0.5x on this dataset for stable init)"
    )
    slam_group.add_argument(
        "--slam-timeout",
        type=int,
        default=0,
        help="Timeout for SLAM stage in seconds (0 disables, default: 0)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def expand_input_paths(patterns: List[str]) -> List[str]:
    """Expand glob patterns in input paths."""
    expanded = []
    for pattern in patterns:
        matches = glob(pattern, recursive=True)
        if matches:
            expanded.extend(sorted(matches))
        else:
            expanded.append(pattern)
    return expanded


def parse_frame_numbers(raw: str) -> List[int]:
    """Parse comma-separated frame numbers and inclusive ranges."""
    frames: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_raw, end_raw = item.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"Invalid frame range: {item}")
            frames.update(range(start, end + 1))
        else:
            frames.add(int(item))
    return sorted(frames)


def main():
    args = parse_args()

    if args.list_stages:
        registry.print_stages()
        return 0

    if not args.stages:
        print("Error: No stages specified. Use --stages or --list-stages")
        return 1

    if not args.input_bags:
        print("Error: No input bags specified. Use --input")
        return 1

    input_bags = expand_input_paths(args.input_bags)

    if not input_bags:
        print("Error: No input bags found matching the given patterns")
        return 1

    try:
        image_frame_numbers = parse_frame_numbers(args.image_frames)
    except ValueError as exc:
        print(f"Error: invalid --image-frames value: {exc}")
        return 1

    config = StageConfig(
        verbose=args.verbose,
        input_root=str(Path(args.output).resolve()),
        slam_rate=args.slam_rate,
        slam_timeout=args.slam_timeout,
        align_start_position=args.align_start_position,
        eval_max_time_delta=args.eval_max_dt,
        image_frame_numbers=image_frame_numbers,
        image_topic=args.image_topic,
        window_root=args.window_root,
        window_prompt=args.window_prompt,
        window_device=args.window_device,
        window_box_threshold=args.window_box_threshold,
        window_text_threshold=args.window_text_threshold,
        window_camera_height=args.window_camera_height,
        floorplan_realign_weight=args.floorplan_realign_weight,
        window_realign_weight=args.window_realign_weight,
    )

    return run_pipeline(
        stage_names=args.stages,
        input_bags=input_bags,
        output_dir=args.output,
        config=config,
        container_runtime=args.container_runtime,
        include_pca_align=args.include_pca_align,
    )


if __name__ == "__main__":
    sys.exit(main())
