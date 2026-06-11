#!/usr/bin/env python3
"""Line extractor runner - executed inside the container.

Iterates over /cam0/image_raw/compressed messages in the mounted rosbag at
/input, runs cv2 line detection inside a fixed ROI polygon, keeps near-horizontal
segments above a minimum length, and writes them to /output/lines.csv.
"""

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as exc:
    print(f"FATAL: missing ROS Python deps: {exc}", file=sys.stderr)
    sys.exit(2)


CAM_TOPIC = "/cam0/image_raw/compressed"
INPUT_DIR = Path("/input")
OUTPUT_CSV = Path("/output/lines.csv")
DEBUG_LOG = Path("/output/line_extractor_debug.log")

# ROI polygon from the C++ line_extractor node.
ROI_POLYGON = np.array(
    [[625, 200], [1200, 200], [1400, 400], [1400, 450], [625, 450]],
    dtype=np.int32,
)

# Within 30 deg of horizontal, length >= 75 px.
ACCEPTABLE_ANGLE_RAD = 30.0 * np.pi / 180.0
ACCEPTABLE_LENGTH_PX = 75.0


def log(message: str) -> None:
    line = f"[line_extractor][{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with DEBUG_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


class _KeyLine:
    __slots__ = (
        "startPointX",
        "startPointY",
        "endPointX",
        "endPointY",
        "angle",
        "lineLength",
    )


def _wrap_segments_as_keylines(segments: np.ndarray) -> list[_KeyLine]:
    """Convert (N,4) [x1,y1,x2,y2] segments into objects with the C++ keyline fields."""
    out: list[_KeyLine] = []
    for x1, y1, x2, y2 in segments:
        kl = _KeyLine()
        kl.startPointX = float(x1)
        kl.startPointY = float(y1)
        kl.endPointX = float(x2)
        kl.endPointY = float(y2)
        dx = kl.endPointX - kl.startPointX
        dy = kl.endPointY - kl.startPointY
        kl.angle = float(np.arctan2(dy, dx))
        kl.lineLength = float(np.hypot(dx, dy))
        out.append(kl)
    return out


def _build_detector():
    """Pick the best available cv2 line detector and return a (name, fn) pair.

    The returned `fn(image, mask)` produces a list of `_KeyLine`-shaped objects.
    Preference order matches the C++ node:
      1. cv2.line_descriptor.BinaryDescriptor (contrib, exact parity)
      2. cv2.ximgproc.FastLineDetector (contrib, modern LSD replacement)
      3. cv2.HoughLinesP (always available, less accurate)
    """
    if hasattr(cv2, "line_descriptor"):
        try:
            detector = cv2.line_descriptor.BinaryDescriptor_createBinaryDescriptor()
        except AttributeError:
            detector = cv2.line_descriptor.BinaryDescriptor.createBinaryDescriptor()

        def detect(img: np.ndarray, mask: np.ndarray) -> list[_KeyLine]:
            keylines = detector.detect(img, mask)
            if keylines is None:
                return []
            return list(keylines)

        return "cv2.line_descriptor.BinaryDescriptor", detect

    if hasattr(cv2, "ximgproc"):
        detector = cv2.ximgproc.createFastLineDetector()

        def detect(img: np.ndarray, mask: np.ndarray) -> list[_KeyLine]:
            masked = cv2.bitwise_and(img, mask)
            segments = detector.detect(masked)
            if segments is None:
                return []
            return _wrap_segments_as_keylines(segments.reshape(-1, 4))

        return "cv2.ximgproc.FastLineDetector", detect

    def detect(img: np.ndarray, mask: np.ndarray) -> list[_KeyLine]:
        masked = cv2.bitwise_and(img, mask)
        edges = cv2.Canny(masked, 50, 150)
        segments = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=60,
            minLineLength=int(ACCEPTABLE_LENGTH_PX),
            maxLineGap=10,
        )
        if segments is None:
            return []
        return _wrap_segments_as_keylines(segments.reshape(-1, 4))

    return "cv2.HoughLinesP (fallback)", detect


def _is_horizontal(angle_rad: float) -> bool:
    """Match the C++ filter: keep if within ACCEPTABLE_ANGLE_RAD of 0 or pi (mod 2pi)."""
    a = angle_rad
    if a < 0:
        a += 2.0 * np.pi
    near_zero = a <= ACCEPTABLE_ANGLE_RAD or a >= 2.0 * np.pi - ACCEPTABLE_ANGLE_RAD
    near_pi = abs(a - np.pi) <= ACCEPTABLE_ANGLE_RAD
    return near_zero or near_pi


def _open_reader(bag_dir: Path):
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def main() -> None:
    DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    DEBUG_LOG.write_text("", encoding="utf-8")
    log("Line extractor starting")
    log(f"Input bag: {INPUT_DIR}")

    try:
        reader = _open_reader(INPUT_DIR)
    except Exception as exc:
        log(f"ERROR: failed to open rosbag at {INPUT_DIR}: {exc}")
        sys.exit(1)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if CAM_TOPIC not in type_map:
        log(f"ERROR: topic {CAM_TOPIC} not found. Available: {sorted(type_map)}")
        sys.exit(1)

    reader.set_filter(rosbag2_py.StorageFilter(topics=[CAM_TOPIC]))
    msg_class = get_message(type_map[CAM_TOPIC])

    detector_name, detect = _build_detector()
    log(f"Line detector: {detector_name}")

    mask = None
    frames = 0
    kept = 0
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "startX", "startY", "endX", "endY"])

        while reader.has_next():
            _, raw, _ = reader.read_next()
            msg = deserialize_message(raw, msg_class)
            frames += 1

            buf = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            if mask is None or mask.shape != img.shape:
                mask = np.zeros(img.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [ROI_POLYGON], 255)

            keylines = detect(img, mask)
            if not keylines:
                continue

            t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            for kl in keylines:
                if kl.lineLength < ACCEPTABLE_LENGTH_PX:
                    continue
                if not _is_horizontal(kl.angle):
                    continue
                writer.writerow(
                    [
                        f"{t_sec:.6f}",
                        f"{kl.startPointX:.5f}",
                        f"{kl.startPointY:.5f}",
                        f"{kl.endPointX:.5f}",
                        f"{kl.endPointY:.5f}",
                    ]
                )
                kept += 1

            if frames % 200 == 0:
                log(f"Processed {frames} frames, {kept} lines kept")

    log(f"Done. Frames processed: {frames}, lines written: {kept}")
    log(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
