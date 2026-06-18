#!/usr/bin/env python3
"""Custom static transform publisher that reads initial pose from init_pos.txt.

This is a simplified version of the challenge's static_transform_publisher.py
that reads from a custom CSV file instead of the built-in init_gt_poses.csv.

Usage: static_tf_publisher_custom.py /path/to/init_pos.txt
"""

import csv
import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class ReactTime:
    """Simple time representation."""
    def __init__(self, sec, nsec):
        self.sec = int(sec)
        self.nsec = int(nsec)

    def __gt__(self, other):
        if self.sec > other.sec:
            return True
        if self.sec == other.sec and self.nsec > other.nsec:
            return True
        return False


class ReactPose:
    """Simple pose representation."""
    def __init__(self, pos, quat):
        self.pos = pos  # [x, y, z]
        self.quat = quat  # [x, y, z, w]


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


def pose_to_matrix(pose):
    """Convert ReactPose to 4x4 transformation matrix."""
    qx, qy, qz, qw = pose.quat
    # Normalize quaternion
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pose.pos
    return T


def matrix_to_pose(T):
    """Convert 4x4 transformation matrix to ReactPose."""
    pos = T[:3, 3].tolist()

    # Convert rotation matrix to quaternion
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

    quat = [qx, qy, qz, qw]
    return ReactPose(pos, quat)


def load_imu_to_cam0_mat():
    """Load T_cam0_imu from calibration file."""
    # Hardcoded calibration from the Hilti challenge
    # This is T_cam_imu for cam0
    T_cam_imu = np.array([
        [-0.00680499, -0.0153215, 0.99985, 0.00198158],
        [-0.999977, 0.000334627, -0.00680328, -0.120996],
        [-0.000230383, -0.999883, -0.0153224, -0.0219206],
        [0.0, 0.0, 0.0, 1.0]
    ])
    return T_cam_imu


class StaticTransformPublisher(Node):
    """Publishes static map->global TF based on initial pose and first SLAM pose."""

    def __init__(self, init_pose_path: str):
        super().__init__('static_transform_publisher_custom')

        self.broadcaster = StaticTransformBroadcaster(self)
        self.static_initialized = False

        # Load initial pose from CSV
        self.pose_gt0 = self.load_initial_pose(init_pose_path)
        self.get_logger().info(f'Loaded initial pose: timestamp={self.pose_gt0[0].sec}.{self.pose_gt0[0].nsec:09d}')
        self.get_logger().info(f'  Position: {self.pose_gt0[1].pos}')
        self.get_logger().info(f'  Quaternion: {self.pose_gt0[1].quat}')

        # Load IMU to cam0 transform
        self.imu_to_cam0_mat = load_imu_to_cam0_mat()

        # Subscribe to OpenVINS poses
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/ov_msckf/poseimu',
            self.pose_callback,
            10
        )

    def load_initial_pose(self, csv_path: str):
        """Load initial pose from init_pos.txt CSV file."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Initial pose file not found: {csv_path}")

        with path.open('r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith('#'):
                    continue

                # Parse: seq_name, floor_plan, timestamp, tx, ty, tz, qx, qy, qz, qw
                timestamp_str = row[2]
                timestamp_parts = timestamp_str.split('.')
                sec = int(timestamp_parts[0])
                nsec = int(timestamp_parts[1]) if len(timestamp_parts) > 1 else 0

                tx, ty, tz = float(row[3]), float(row[4]), float(row[5])
                qx, qy, qz, qw = float(row[6]), float(row[7]), float(row[8]), float(row[9])

                time = ReactTime(sec, nsec)
                pose = ReactPose([tx, ty, tz], [qx, qy, qz, qw])

                return (time, pose)

        raise ValueError(f"No valid pose found in {csv_path}")

    def publish_static_transform(self, pose, time):
        """Publish static TF from map to global frame."""
        t = TransformStamped()
        t.header.stamp.sec = time.sec
        t.header.stamp.nanosec = time.nsec
        t.header.frame_id = 'map'
        t.child_frame_id = 'global'

        t.transform.translation.x = pose.pos[0]
        t.transform.translation.y = pose.pos[1]
        t.transform.translation.z = pose.pos[2]

        t.transform.rotation.x = pose.quat[0]
        t.transform.rotation.y = pose.quat[1]
        t.transform.rotation.z = pose.quat[2]
        t.transform.rotation.w = pose.quat[3]

        self.broadcaster.sendTransform(t)
        self.get_logger().info('Static transform map->global published to /tf_static')

    def pose_callback(self, msg):
        """Match first OpenVINS pose with ground truth to compute map->global TF."""
        if self.static_initialized:
            return

        current_t = ReactTime(msg.header.stamp.sec, msg.header.stamp.nanosec)

        if current_t > self.pose_gt0[0]:
            self.get_logger().info(f'Matched SLAM pose at {current_t.sec}.{current_t.nsec:09d}')

            p_map_cam0 = self.pose_gt0[1]

            # Unpack OpenVINS IMU pose (in global frame)
            pos = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ]
            quat = [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ]

            p_global_imu = ReactPose(pos, quat)
            T_global_imu = pose_to_matrix(p_global_imu)

            # Transform from IMU to cam0
            T_cam0_imu = self.imu_to_cam0_mat
            T_imu_cam0 = np.linalg.inv(T_cam0_imu)
            T_global_cam0 = np.matmul(T_global_imu, T_imu_cam0)

            # Compute map->global transform
            T_map_cam0 = pose_to_matrix(p_map_cam0)
            T_cam0_global = np.linalg.inv(T_global_cam0)
            T_map_global = np.matmul(T_map_cam0, T_cam0_global)
            p_map_global = matrix_to_pose(T_map_global)

            # Extract yaw and zero out pitch/roll
            quat = p_map_global.quat
            roll, pitch, yaw = quaternion_to_euler(quat[0], quat[1], quat[2], quat[3])

            self.get_logger().info(f'Computed T_map_global:')
            self.get_logger().info(f'  Translation: {p_map_global.pos}')
            self.get_logger().info(f'  Roll:  {np.degrees(roll):.3f} deg')
            self.get_logger().info(f'  Pitch: {np.degrees(pitch):.3f} deg')
            self.get_logger().info(f'  Yaw:   {np.degrees(yaw):.3f} deg')

            # Force pitch and roll to zero (gravity-aligned frames)
            p_map_global_aligned = ReactPose(
                p_map_global.pos,
                euler_to_quaternion(0, 0, yaw)
            )

            self.publish_static_transform(p_map_global_aligned, self.pose_gt0[0])
            self.static_initialized = True


def main(args=None):
    if len(sys.argv) < 2:
        print("Usage: static_tf_publisher_custom.py /path/to/init_pos.txt")
        print("This script publishes a static map->global TF based on the initial pose")
        print("and the first matching OpenVINS pose.")
        sys.exit(1)

    init_pose_path = sys.argv[1]

    rclpy.init(args=args)
    node = StaticTransformPublisher(init_pose_path)

    print(f"Waiting for OpenVINS poses to compute map->global transform...")
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
