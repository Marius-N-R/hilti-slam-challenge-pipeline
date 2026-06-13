"""Shared trajectory and 2D rigid-transform helpers."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from ._geometry import quat_to_rot, rot_to_quat


def load_pose_csv(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            try:
                values = [float(part) for part in parts]
            except ValueError:
                continue
            if len(values) != 8:
                continue
            rows.append(values)
    if not rows:
        raise ValueError(f"No pose rows found in {path}")
    arr = np.asarray(rows, dtype=float)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    return arr[:, 0], arr[:, 1:4], arr[:, 4:8]


def write_pose_csv(path: Path, timestamps: np.ndarray, xyz: np.ndarray, quats: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["# timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        for timestamp, position, quat in zip(timestamps, xyz, quats):
            writer.writerow(
                [
                    f"{timestamp:.9f}",
                    f"{position[0]:.9f}",
                    f"{position[1]:.9f}",
                    f"{position[2]:.9f}",
                    f"{quat[0]:.9f}",
                    f"{quat[1]:.9f}",
                    f"{quat[2]:.9f}",
                    f"{quat[3]:.9f}",
                ]
            )


def estimate_rigid_2d(source_xy: np.ndarray, target_xy: np.ndarray, weights=None):
    """Estimate yaw+translation mapping source_xy onto target_xy."""
    if len(source_xy) != len(target_xy):
        raise ValueError("source and target point arrays must have the same length")
    if len(source_xy) == 0:
        raise ValueError("Need at least one point to estimate a 2D transform")

    source_xy = np.asarray(source_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    if weights is None:
        weights = np.ones(len(source_xy), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 0.0)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(source_xy), dtype=float)
    weights = weights / weights.sum()

    source_centroid = np.sum(source_xy * weights[:, None], axis=0)
    target_centroid = np.sum(target_xy * weights[:, None], axis=0)
    source_centered = source_xy - source_centroid
    target_centered = target_xy - target_centroid

    if len(source_xy) < 2 or np.linalg.norm(source_centered) < 1e-9:
        rotation = np.eye(2)
    else:
        covariance = source_centered.T @ (target_centered * weights[:, None])
        u, _, vt = np.linalg.svd(covariance)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1, :] *= -1
            rotation = vt.T @ u.T

    translation = target_centroid - rotation @ source_centroid
    transformed = (rotation @ source_xy.T).T + translation
    residuals = np.linalg.norm(transformed - target_xy, axis=1)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return rotation, translation, yaw, residuals


def apply_rigid_2d_to_trajectory(
    xyz: np.ndarray,
    quats: np.ndarray,
    rotation_2d: np.ndarray,
    translation_2d: np.ndarray,
):
    corrected_xyz = xyz.copy()
    corrected_xyz[:, :2] = (rotation_2d @ xyz[:, :2].T).T + translation_2d

    yaw = math.atan2(rotation_2d[1, 0], rotation_2d[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    rotation_3d = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    corrected_quats = np.empty_like(quats)
    for idx, quat in enumerate(quats):
        corrected_quats[idx] = rot_to_quat(rotation_3d @ quat_to_rot(*quat))
    return corrected_xyz, corrected_quats


def match_by_timestamp(source_t: np.ndarray, target_t: np.ndarray, max_dt: float | None = None):
    pairs = []
    for source_idx, timestamp in enumerate(source_t):
        target_idx = int(np.searchsorted(target_t, timestamp))
        candidates = []
        if target_idx < len(target_t):
            candidates.append(target_idx)
        if target_idx > 0:
            candidates.append(target_idx - 1)
        if not candidates:
            continue
        best_idx = min(candidates, key=lambda idx: abs(target_t[idx] - timestamp))
        dt = float(timestamp - target_t[best_idx])
        if max_dt is not None and abs(dt) > max_dt:
            continue
        pairs.append((source_idx, best_idx, dt))
    return pairs


def rotation_from_yaw(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=float)


def weighted_yaw(yaws: list[float], weights: list[float]) -> float:
    sx = sum(weight * math.cos(yaw) for yaw, weight in zip(yaws, weights))
    sy = sum(weight * math.sin(yaw) for yaw, weight in zip(yaws, weights))
    if abs(sx) < 1e-12 and abs(sy) < 1e-12:
        return 0.0
    return math.atan2(sy, sx)


def nearest_point_on_segments(point_xy: np.ndarray, segments: np.ndarray):
    best_point = None
    best_idx = -1
    best_dist = math.inf
    for idx, segment in enumerate(segments):
        a = np.array([segment[0], segment[1]], dtype=float)
        b = np.array([segment[2], segment[3]], dtype=float)
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            projection = a
        else:
            t = float(np.clip(np.dot(point_xy - a, ab) / denom, 0.0, 1.0))
            projection = a + t * ab
        dist = float(np.linalg.norm(point_xy - projection))
        if dist < best_dist:
            best_point = projection
            best_idx = idx
            best_dist = dist
    if best_point is None:
        raise ValueError("No valid segments provided")
    return best_point, best_idx, best_dist
