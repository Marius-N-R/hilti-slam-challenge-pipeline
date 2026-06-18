#!/usr/bin/env python3
"""Extract the map->global static TF and write orientation.json.

If init_pos.txt is provided, computes T_map_global by matching the initial pose
with the corresponding SLAM trajectory pose. Otherwise, falls back to reading
/tf_static from the rosbag (original behavior).

Executed inside the ROS container.
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

OUTPUT = Path("/output/orientation.json")
INIT_POS_FILE = Path("/original_input/init_pos.txt")
SLAM_TRAJECTORY_FILE = Path("/slam_output/trajectory.txt")
BAG_PATH = "/input"
PARENT_FRAME = "map"
CHILD_FRAME = "global"

# Hardcoded T_cam0_imu calibration from Hilti challenge
T_CAM0_IMU = np.array([
    [-0.00680499, -0.0153215, 0.99985, 0.00198158],
    [-0.999977, 0.000334627, -0.00680328, -0.120996],
    [-0.000230383, -0.999883, -0.0153224, -0.0219206],
    [0.0, 0.0, 0.0, 1.0]
])


def quat_xyzw_to_rotmat(qx, qy, qz, qw):
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    return np.array([
        [1.0 - (yy + zz),       xy - wz,         xz + wy],
        [xy + wz,               1.0 - (xx + zz), yz - wx],
        [xz - wy,               yz + wx,         1.0 - (xx + yy)],
    ])


def pose_to_matrix(pos, quat):
    """Convert position and quaternion (xyzw) to 4x4 transformation matrix."""
    qx, qy, qz, qw = quat
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_rotmat(qx, qy, qz, qw)
    T[:3, 3] = pos
    return T


def matrix_to_quat(T):
    """Convert 4x4 transformation matrix to quaternion [x, y, z, w]."""
    R = T[:3, :3]
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return [qx, qy, qz, qw]


def quaternion_to_euler(qx, qy, qz, qw):
    """Convert quaternion to Euler angles (roll, pitch, yaw)."""
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to quaternion [x, y, z, w]."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return [qx, qy, qz, qw]


def load_init_pose(csv_path):
    """Load initial cam0 pose in map frame from init_pos.txt."""
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue

            # Parse: seq_name, floor_plan, timestamp, tx, ty, tz, qx, qy, qz, qw
            timestamp_str = row[2]
            timestamp = float(timestamp_str)

            tx, ty, tz = float(row[3]), float(row[4]), float(row[5])
            qx, qy, qz, qw = float(row[6]), float(row[7]), float(row[8]), float(row[9])

            return timestamp, [tx, ty, tz], [qx, qy, qz, qw]

    raise ValueError(f"No valid pose found in {csv_path}")


def load_slam_trajectory(traj_path):
    """Load SLAM trajectory (IMU poses in global frame) from trajectory.txt."""
    poses = []
    with open(traj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) != 8:
                continue

            timestamp = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

            poses.append((timestamp, [tx, ty, tz], [qx, qy, qz, qw]))

    return poses


def find_matching_slam_pose(slam_poses, target_timestamp):
    """Find the SLAM pose closest to the target timestamp."""
    best_pose = None
    best_diff = float('inf')

    for timestamp, pos, quat in slam_poses:
        diff = abs(timestamp - target_timestamp)
        if diff < best_diff:
            best_diff = diff
            best_pose = (timestamp, pos, quat)

        # Optimization: if we've passed the target, we can stop
        if timestamp > target_timestamp + 1.0:
            break

    return best_pose


def compute_map_global_transform(init_timestamp, p_map_cam0_pos, p_map_cam0_quat, slam_poses):
    """Compute T_map_global using initial pose and matching SLAM pose."""
    # Find matching SLAM pose
    slam_match = find_matching_slam_pose(slam_poses, init_timestamp)
    if slam_match is None:
        raise RuntimeError("No matching SLAM pose found")

    slam_timestamp, p_global_imu_pos, p_global_imu_quat = slam_match
    time_diff = abs(slam_timestamp - init_timestamp)

    print(f"[save_tf] Initial pose timestamp: {init_timestamp:.9f}", flush=True)
    print(f"[save_tf] Matched SLAM pose timestamp: {slam_timestamp:.9f}", flush=True)
    print(f"[save_tf] Time difference: {time_diff:.6f} s", flush=True)

    if time_diff > 0.5:
        print(f"[save_tf] WARNING: Large time difference ({time_diff:.3f}s) between init pose and SLAM pose", flush=True)

    # Compute transforms
    T_global_imu = pose_to_matrix(p_global_imu_pos, p_global_imu_quat)
    T_imu_cam0 = np.linalg.inv(T_CAM0_IMU)
    T_global_cam0 = T_global_imu @ T_imu_cam0

    T_map_cam0 = pose_to_matrix(p_map_cam0_pos, p_map_cam0_quat)
    T_cam0_global = np.linalg.inv(T_global_cam0)
    T_map_global = T_map_cam0 @ T_cam0_global

    # Extract quaternion and yaw
    quat = matrix_to_quat(T_map_global)
    roll, pitch, yaw = quaternion_to_euler(*quat)

    print(f"[save_tf] Raw T_map_global:", flush=True)
    print(f"[save_tf]   Translation: {T_map_global[:3, 3]}", flush=True)
    print(f"[save_tf]   Roll:  {np.degrees(roll):.3f} deg", flush=True)
    print(f"[save_tf]   Pitch: {np.degrees(pitch):.3f} deg", flush=True)
    print(f"[save_tf]   Yaw:   {np.degrees(yaw):.3f} deg", flush=True)

    # Zero out pitch and roll (gravity-aligned frames)
    quat_aligned = euler_to_quaternion(0, 0, yaw)
    T_map_global_aligned = np.eye(4)
    T_map_global_aligned[:3, :3] = quat_xyzw_to_rotmat(*quat_aligned)
    T_map_global_aligned[:3, 3] = T_map_global[:3, 3]

    return T_map_global_aligned, yaw


def compute_from_init_pos():
    """Compute orientation.json from init_pos.txt and SLAM trajectory."""
    print(f"[save_tf] Using init_pos.txt to compute T_map_global", flush=True)

    if not INIT_POS_FILE.exists():
        raise FileNotFoundError(f"init_pos.txt not found: {INIT_POS_FILE}")

    if not SLAM_TRAJECTORY_FILE.exists():
        raise FileNotFoundError(f"SLAM trajectory not found: {SLAM_TRAJECTORY_FILE}")

    # Load initial pose
    init_timestamp, p_map_cam0_pos, p_map_cam0_quat = load_init_pose(INIT_POS_FILE)
    print(f"[save_tf] Loaded initial cam0 pose from {INIT_POS_FILE}", flush=True)
    print(f"[save_tf]   Position: {p_map_cam0_pos}", flush=True)
    print(f"[save_tf]   Quaternion: {p_map_cam0_quat}", flush=True)

    # Load SLAM trajectory
    slam_poses = load_slam_trajectory(SLAM_TRAJECTORY_FILE)
    print(f"[save_tf] Loaded {len(slam_poses)} SLAM poses from {SLAM_TRAJECTORY_FILE}", flush=True)

    # Compute T_map_global
    T_map_global, yaw = compute_map_global_transform(
        init_timestamp, p_map_cam0_pos, p_map_cam0_quat, slam_poses
    )

    # Extract final values
    tx, ty, tz = T_map_global[:3, 3]
    qx, qy, qz, qw = matrix_to_quat(T_map_global)

    payload = {
        "parent_frame": PARENT_FRAME,
        "child_frame": CHILD_FRAME,
        "translation_xyz": [float(tx), float(ty), float(tz)],
        "quaternion_xyzw": [float(qx), float(qy), float(qz), float(qw)],
        "yaw_rad": float(yaw),
        "yaw_deg": float(np.degrees(yaw)),
        "T_parent_child": T_map_global.tolist(),
    }

    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[save_tf] Saved {PARENT_FRAME} -> {CHILD_FRAME} transform to {OUTPUT}", flush=True)
    print(f"[save_tf] Final Yaw: {math.degrees(yaw):.3f} deg", flush=True)
    print(f"[save_tf] Final Translation: x={tx:.6f}, y={ty:.6f}, z={tz:.6f}", flush=True)


def read_from_rosbag():
    """Original behavior: read /tf_static from rosbag."""
    print(f"[save_tf] Reading /tf_static from rosbag (fallback method)", flush=True)
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from tf2_msgs.msg import TFMessage
    except ImportError as exc:
        print(f"[save_tf] ERROR: Missing ROS2 dependency: {exc}", flush=True)
        sys.exit(1)

    print(f"[save_tf] Opening bag: {BAG_PATH}", flush=True)
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=BAG_PATH, storage_id="")
    converter_options = rosbag2_py.ConverterOptions("cdr", "cdr")
    reader.open(storage_options, converter_options)

    storage_filter = rosbag2_py.StorageFilter(topics=["/tf_static"])
    reader.set_filter(storage_filter)

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != "/tf_static":
            continue
        msg = deserialize_message(data, TFMessage)
        for tf in msg.transforms:
            if tf.header.frame_id != PARENT_FRAME or tf.child_frame_id != CHILD_FRAME:
                continue

            tx = tf.transform.translation.x
            ty = tf.transform.translation.y
            tz = tf.transform.translation.z
            qx = tf.transform.rotation.x
            qy = tf.transform.rotation.y
            qz = tf.transform.rotation.z
            qw = tf.transform.rotation.w

            T = np.eye(4)
            T[:3, :3] = quat_xyzw_to_rotmat(qx, qy, qz, qw)
            T[:3, 3] = [tx, ty, tz]
            yaw = float(np.arctan2(T[1, 0], T[0, 0]))

            payload = {
                "parent_frame": PARENT_FRAME,
                "child_frame": CHILD_FRAME,
                "translation_xyz": [tx, ty, tz],
                "quaternion_xyzw": [qx, qy, qz, qw],
                "yaw_rad": yaw,
                "yaw_deg": float(np.degrees(yaw)),
                "T_parent_child": T.tolist(),
            }
            OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[save_tf] Saved {PARENT_FRAME} -> {CHILD_FRAME} transform to {OUTPUT}", flush=True)
            print(f"[save_tf] Yaw: {math.degrees(yaw):.3f} deg", flush=True)
            print(f"[save_tf] Translation: x={tx:.6f}, y={ty:.6f}, z={tz:.6f}", flush=True)
            return

    print(
        f"[save_tf] ERROR: '{PARENT_FRAME}' -> '{CHILD_FRAME}' not found in /tf_static",
        flush=True,
    )
    sys.exit(1)


def main():
    """Main entry point - try init_pos.txt first, fallback to rosbag."""
    # Try to use init_pos.txt if available
    if INIT_POS_FILE.exists() and SLAM_TRAJECTORY_FILE.exists():
        try:
            compute_from_init_pos()
            sys.exit(0)
        except Exception as exc:
            print(f"[save_tf] WARNING: Failed to compute from init_pos.txt: {exc}", flush=True)
            print(f"[save_tf] Falling back to rosbag method", flush=True)

    # Fallback to reading from rosbag
    read_from_rosbag()


if __name__ == "__main__":
    main()
