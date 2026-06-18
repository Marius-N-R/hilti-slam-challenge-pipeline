#!/usr/bin/env python3
"""SLAM runner script - executed inside the container."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from textwrap import dedent

DEBUG_LOG = Path("/output/slam_debug.log")
OPENVINS_LOG = Path("/output/openvins.log")
LOGGER_LOG = Path("/output/trajectory_logger.log")
BAG_PLAY_LOG = Path("/output/bag_play.log")
TRAJECTORY_FILE = Path("/output/trajectory.txt")
OPENVINS_TRAJECTORY_FILE = Path("/output/trajectory_ov.txt")
DIAGNOSIS_FILE = Path("/output/slam_diagnosis.txt")


def log(message: str) -> None:
    """Log to stdout and debug file with a timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[slam][{timestamp}] {message}"
    print(line, flush=True)
    with DEBUG_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def count_data_lines(path: Path) -> int:
    """Count non-empty, non-comment lines in a text file."""
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
    return count


def log_file_tail(path: Path, label: str, max_lines: int = 60) -> None:
    """Print and persist a tail excerpt from a file."""
    if not path.exists():
        log(f"{label} log not found: {path}")
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        log(f"{label} log is empty: {path}")
        return
    log(f"{label} log tail ({min(len(lines), max_lines)} lines):")
    for line in lines[-max_lines:]:
        log(f"[{label}] {line}")


def openvins_log_has_failure() -> bool:
    """Return True if OpenVINS log contains a known hard-failure marker."""
    if not OPENVINS_LOG.exists():
        return False
    content = OPENVINS_LOG.read_text(encoding="utf-8", errors="replace")
    failure_markers = (
        "process has died",
        "unable to parse all parameters",
        "terminate called after throwing",
    )
    return any(marker in content for marker in failure_markers)


def diagnose_initialization_failure(rate: float) -> str | None:
    """Summarize common OpenVINS initialization failures."""
    if not OPENVINS_LOG.exists():
        return None

    content = OPENVINS_LOG.read_text(encoding="utf-8", errors="replace")
    no_jerk_count = content.count("failed static init: no accel jerk detected")
    low_feature_count = content.count("valid features of required")
    high_excitation_count = content.count("to much IMU excitation")

    if no_jerk_count == 0 and low_feature_count == 0 and high_excitation_count == 0:
        return None

    reasons = ["OpenVINS stayed in initialization and never produced poses"]
    if no_jerk_count > 0:
        reasons.append(f"static init reported no accel jerk {no_jerk_count} time(s)")
    if low_feature_count > 0:
        reasons.append(f"dynamic init had too few valid features {low_feature_count} time(s)")
    if high_excitation_count > 0:
        reasons.append(
            "static init reported excessive IMU excitation "
            f"{high_excitation_count} time(s)"
        )

    if rate > 1.0:
        reasons.append("playback rate is high; try --slam-rate 1.0 or 0.5")

    return "; ".join(reasons)


def get_topic_subscription_count(ros_setup: str, topic: str) -> int:
    """Return ROS2 topic subscription count, or zero if unknown."""
    result = subprocess.run(
        f"{ros_setup} && ros2 topic info {topic}",
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return 0

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("Subscription count:"):
            continue
        try:
            return int(line.split(":", 1)[1].strip())
        except ValueError:
            return 0
    return 0


def wait_for_topic_subscribers(
    ros_setup: str,
    topic: str,
    min_subscribers: int = 1,
    timeout_seconds: int = 25,
) -> bool:
    """Wait until a topic has at least min_subscribers subscriptions."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        subscribers = get_topic_subscription_count(ros_setup, topic)
        if subscribers >= min_subscribers:
            log(f"Topic ready: {topic} has {subscribers} subscriber(s)")
            return True
        time.sleep(1)
    subscribers = get_topic_subscription_count(ros_setup, topic)
    log(
        f"WARNING: Timed out waiting for subscribers on {topic} "
        f"(have {subscribers}, need {min_subscribers})"
    )
    return False


def stop_process(process: subprocess.Popen, name: str, timeout: int = 5) -> int | None:
    """Terminate a subprocess and return the exit code."""
    if process.poll() is not None:
        log(f"{name} already exited with code {process.returncode}")
        return process.returncode

    process.terminate()
    try:
        exit_code = process.wait(timeout=timeout)
        log(f"{name} terminated with code {exit_code}")
        return exit_code
    except subprocess.TimeoutExpired:
        process.kill()
        exit_code = process.wait(timeout=timeout)
        log(f"{name} killed after timeout, code {exit_code}")
        return exit_code


def create_config() -> str:
    """Create OpenVINS config with dynamic initialization enabled."""
    src_config = Path("/root/ros2_ws/src/hilti-trimble-slam-challenge-2026/config/hilti_openvins")
    config_dir = Path("/tmp/slam_config")

    if config_dir.exists():
        shutil.rmtree(config_dir)
    shutil.copytree(src_config, config_dir)

    config_path = config_dir / "estimator_config.yaml"
    config = config_path.read_text(encoding="utf-8")

    replacements = [
        ("init_dyn_use: false", "init_dyn_use: true"),
        ("init_dyn_num_pose: 6", "init_dyn_num_pose: 4"),
        ("save_total_state: false", "save_total_state: true"),
        ('filepath_est: "/tmp/ov_estimate.txt"', 'filepath_est: "/output/trajectory_ov.txt"'),
        (
            'filepath_std: "/tmp/ov_estimate_std.txt"',
            'filepath_std: "/output/trajectory_std.txt"',
        ),
        ('filepath_gt: "/tmp/ov_groundtruth.txt"', 'filepath_gt: "/output/groundtruth.txt"'),
    ]
    for old, new in replacements:
        if old not in config:
            raise RuntimeError(f"Expected config key not found: {old}")
        config = config.replace(old, new, 1)

    config_path.write_text(config, encoding="utf-8")

    header = config.splitlines()[0] if config.splitlines() else ""
    if not header.startswith("%YAML:1.0"):
        raise RuntimeError(f"Invalid config header: '{header}'")

    log(f"Wrote OpenVINS config: {config_path}")
    return str(config_path)


def main() -> None:
    rate = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    os.makedirs("/output", exist_ok=True)
    DEBUG_LOG.write_text("", encoding="utf-8")

    log(f"SLAM runner starting (rate={rate}, timeout={timeout})")

    config_path = create_config()
    ros_setup = "source /opt/ros/jazzy/setup.bash && source /root/ros2_ws/install/setup.bash"

    log("Checking input bag")
    bag_info = subprocess.run(
        f"{ros_setup} && ros2 bag info /input",
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
    )
    Path("/output/bag_info.txt").write_text(
        bag_info.stdout + bag_info.stderr,
        encoding="utf-8",
    )

    print(bag_info.stdout)
    if bag_info.returncode != 0:
        log(f"ERROR: Could not read bag: {bag_info.stderr.strip()}")
        sys.exit(1)

    if "/cam0/image_raw" not in bag_info.stdout or "/cam1/image_raw" not in bag_info.stdout:
        log("ERROR: Missing camera topics in bag")
        if "/pano/" in bag_info.stdout:
            log("NOTE: Input appears stitched. Run SLAM on original bag with cam0/cam1 topics")
        sys.exit(1)

    duration = None
    for line in bag_info.stdout.splitlines():
        if "Duration:" not in line:
            continue
        try:
            duration = float(line.split()[1].rstrip("s"))
        except Exception:
            duration = None
        break

    expected_time = duration / rate if duration else None
    if expected_time:
        log(f"Bag duration={duration:.3f}s, expected playback={expected_time:.0f}s")
    else:
        log(f"Bag duration unavailable, playback rate={rate}")

    log("Starting OpenVINS")
    openvins_log_handle = OPENVINS_LOG.open("w", encoding="utf-8")
    openvins = subprocess.Popen(
        (
            f"{ros_setup} && ros2 launch challenge_tools_ros run_openvins.launch.py "
            f"rviz_enable:=false use_sim_time:=true config_path:={config_path}"
        ),
        shell=True,
        executable="/bin/bash",
        stdout=openvins_log_handle,
        stderr=subprocess.STDOUT,
    )

    time.sleep(8)
    if openvins.poll() is not None:
        log(f"ERROR: OpenVINS exited during startup with code {openvins.returncode}")
        openvins_log_handle.flush()
        log_file_tail(OPENVINS_LOG, "openvins", max_lines=120)
        sys.exit(1)

    log("OpenVINS started successfully")
    openvins_log_handle.flush()

    ros_nodes = subprocess.run(
        f"{ros_setup} && ros2 node list",
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
    )
    Path("/output/ros_nodes.txt").write_text(ros_nodes.stdout + ros_nodes.stderr, encoding="utf-8")

    ros_topics = subprocess.run(
        f"{ros_setup} && ros2 topic list -t",
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
    )
    Path("/output/ros_topics.txt").write_text(ros_topics.stdout + ros_topics.stderr, encoding="utf-8")

    pose_topic_info = subprocess.run(
        f"{ros_setup} && ros2 topic info /ov_msckf/poseimu -v",
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
    )
    Path("/output/pose_topic_info.txt").write_text(
        pose_topic_info.stdout + pose_topic_info.stderr,
        encoding="utf-8",
    )
    if pose_topic_info.returncode != 0:
        log("ERROR: /ov_msckf/poseimu not available; OpenVINS likely failed during startup")
        openvins_log_handle.flush()
        log_file_tail(OPENVINS_LOG, "openvins", max_lines=120)
        sys.exit(1)

    log("Waiting for OpenVINS/input topic subscriptions to be ready")
    wait_for_topic_subscribers(ros_setup, "/imu/data_raw")
    wait_for_topic_subscribers(ros_setup, "/cam0/image_raw/compressed")
    wait_for_topic_subscribers(ros_setup, "/cam1/image_raw/compressed")
    wait_for_topic_subscribers(ros_setup, "/cam0/image_raw")
    wait_for_topic_subscribers(ros_setup, "/cam1/image_raw")

    # Check if we need to launch static TF publisher for map->global transform
    init_pos_file = Path("/original_input/init_pos.txt")
    static_tf_publisher = None
    static_tf_log_handle = None
    if init_pos_file.exists():
        log(f"Found {init_pos_file}, starting static TF publisher for map->global transform")
        STATIC_TF_LOG = Path("/output/static_tf_publisher.log")
        static_tf_log_handle = STATIC_TF_LOG.open("w", encoding="utf-8")
        static_tf_publisher = subprocess.Popen(
            f"{ros_setup} && python3 /stage_runtime/static_tf_publisher_custom.py {init_pos_file}",
            shell=True,
            executable="/bin/bash",
            stdout=static_tf_log_handle,
            stderr=subprocess.STDOUT,
        )
        time.sleep(2)
        if static_tf_publisher.poll() is not None:
            log(f"WARNING: Static TF publisher exited early with code {static_tf_publisher.returncode}")
        else:
            log("Static TF publisher started successfully")
    else:
        log(f"No {init_pos_file} found, skipping static TF publisher (no map->global transform)")

    log("Starting trajectory logger")
    logger_script = dedent(
        """\
        import rclpy
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from rclpy.node import Node

        class Logger(Node):
            def __init__(self):
                super().__init__("traj_logger")
                self.f = open("/output/trajectory.txt", "w", encoding="utf-8")
                self.f.write("# timestamp tx ty tz qx qy qz qw\\n")
                self.sub = self.create_subscription(
                    PoseWithCovarianceStamped,
                    "/ov_msckf/poseimu",
                    self.cb,
                    10,
                )
                self.count = 0

            def cb(self, msg):
                t = msg.header.stamp
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                self.f.write(
                    f"{t.sec}.{t.nanosec:09d} {p.x} {p.y} {p.z} {q.x} {q.y} {q.z} {q.w}\\n"
                )
                self.f.flush()
                self.count += 1
                if self.count % 100 == 0:
                    print(f"[logger] {self.count} poses", flush=True)

        rclpy.init()
        node = Logger()
        try:
            rclpy.spin(node)
        except Exception:
            pass
        finally:
            try:
                node.destroy_node()
            except Exception:
                pass
            try:
                rclpy.shutdown()
            except Exception:
                pass
        """
    )

    logger_script_path = Path("/tmp/trajectory_logger.py")
    logger_script_path.write_text(logger_script, encoding="utf-8")
    logger_log_handle = LOGGER_LOG.open("w", encoding="utf-8")
    logger = subprocess.Popen(
        f"{ros_setup} && python3 {logger_script_path}",
        shell=True,
        executable="/bin/bash",
        stdout=logger_log_handle,
        stderr=subprocess.STDOUT,
    )

    time.sleep(2)
    if logger.poll() is not None:
        log(f"WARNING: Trajectory logger exited early with code {logger.returncode}")
        logger_log_handle.flush()
        log_file_tail(LOGGER_LOG, "trajectory_logger", max_lines=60)

    pose_subscribers = get_topic_subscription_count(ros_setup, "/ov_msckf/poseimu")
    if pose_subscribers < 1:
        log(
            "WARNING: /ov_msckf/poseimu still has 0 subscribers after starting "
            "trajectory logger"
        )
    else:
        log(f"Trajectory logger attached to /ov_msckf/poseimu ({pose_subscribers} subscriber)")

    log(f"Playing bag at rate {rate}")
    start_time = time.time()
    bag_play_log_handle = BAG_PLAY_LOG.open("w", encoding="utf-8")
    bag_play = subprocess.Popen(
        f"{ros_setup} && ros2 bag play /input --rate {rate} --clock --delay 2",
        shell=True,
        executable="/bin/bash",
        stdout=bag_play_log_handle,
        stderr=subprocess.STDOUT,
    )

    openvins_failed = False
    last_poses = 0
    bar_width = 30

    def render_progress(elapsed: float, poses: int) -> None:
        if expected_time:
            pct = min(100.0, elapsed / expected_time * 100.0)
            filled = int((pct / 100.0) * bar_width)
            bar = f"{'#' * filled}{'-' * (bar_width - filled)}"
            line = (
                f"[slam] Progress: [{bar}] {pct:5.1f}% "
                f"({elapsed:5.0f}s/{expected_time:5.0f}s) | Poses: {poses}"
            )
            sys.stdout.write(f"\r{line}")
            sys.stdout.flush()
            return
        log(f"Progress: {elapsed:.0f}s | Poses: {poses}")

    while bag_play.poll() is None:
        time.sleep(10)
        elapsed = time.time() - start_time
        poses = count_data_lines(TRAJECTORY_FILE)
        last_poses = poses
        render_progress(elapsed, poses)

        if timeout > 0 and elapsed > timeout:
            log(f"ERROR: Timeout after {timeout}s")
            bag_play.terminate()
            break

        if openvins.poll() is not None:
            openvins_failed = True
            log(f"ERROR: OpenVINS exited during playback with code {openvins.returncode}")
            bag_play.terminate()
            break

        openvins_log_handle.flush()
        if openvins_log_has_failure():
            openvins_failed = True
            log("ERROR: OpenVINS reported a runtime failure (see openvins.log)")
            bag_play.terminate()
            break

        if elapsed > 30 and poses == 0:
            log("WARNING: No poses yet - initialization may be failing")

    if expected_time:
        elapsed = time.time() - start_time
        render_progress(elapsed, last_poses)
        print()

    bag_play.wait(timeout=30)
    bag_play_log_handle.flush()
    total_time = time.time() - start_time
    log(f"Bag playback complete after {total_time:.0f}s")

    time.sleep(2)
    stop_process(logger, "trajectory logger")
    if static_tf_publisher is not None:
        stop_process(static_tf_publisher, "static TF publisher")
    stop_process(openvins, "OpenVINS")

    logger_log_handle.close()
    if static_tf_log_handle is not None:
        static_tf_log_handle.flush()
        static_tf_log_handle.close()
    openvins_log_handle.flush()
    openvins_log_handle.close()
    bag_play_log_handle.close()

    poses = count_data_lines(TRAJECTORY_FILE)
    openvins_states = count_data_lines(OPENVINS_TRAJECTORY_FILE)
    log(f"Collected poses in trajectory.txt: {poses}")
    log(f"Collected OpenVINS states in trajectory_ov.txt: {openvins_states}")

    if openvins_failed:
        log_file_tail(OPENVINS_LOG, "openvins", max_lines=120)
        sys.exit(1)

    if poses < 2:
        log(f"ERROR: Only {poses} poses - SLAM failed")
        diagnosis = diagnose_initialization_failure(rate)
        if diagnosis:
            DIAGNOSIS_FILE.write_text(diagnosis + "\n", encoding="utf-8")
            log(f"DIAGNOSIS: {diagnosis}")
        log_file_tail(OPENVINS_LOG, "openvins", max_lines=120)
        log_file_tail(LOGGER_LOG, "trajectory_logger", max_lines=80)
        sys.exit(1)

    log("==============================================================")
    log("SLAM complete")
    log(f"Total time: {total_time:.0f}s")
    log(f"Poses: {poses}")
    log(f"Output: {TRAJECTORY_FILE}")
    log("==============================================================")


if __name__ == "__main__":
    main()
