"""Select explicit image frames from a run folder's ROS2 bag."""

import json
from pathlib import Path

from runtime_backend import ExecutionSpec

from .base import Stage, StageConfig


class ImageSelectorStage(Stage):
    """Extract user-selected camera frames from the input ROS2 bag."""

    @property
    def name(self) -> str:
        return "image_selector"

    @property
    def description(self) -> str:
        return "Extract selected camera frame numbers as images"

    @property
    def requires_ros_runtime(self) -> bool:
        return True

    @property
    def input_type(self) -> str:
        return "rosbag"

    @property
    def output_type(self) -> str:
        return "images"

    def run(self, runner, input_dir: Path, config: StageConfig) -> Path:
        if not config.image_frame_numbers:
            raise ValueError("image_selector requires --image-frames, e.g. --image-frames 10,20,30")

        run_dir = input_dir
        if not (run_dir / "rosbag").is_dir():
            original_input = config.extra.get("current_input_path", "")
            if original_input:
                run_dir = Path(original_input)

        bag_dir = run_dir / "rosbag"
        if not bag_dir.is_dir():
            raise FileNotFoundError(f"Expected rosbag directory inside input folder: {run_dir}")

        runtime_args = json.dumps(
            {
                "bag_path": "/input",
                "output_dir": "/output/images",
                "metadata_path": "/output/selected_frames.json",
                "frame_numbers": config.image_frame_numbers,
                "preferred_topics": [
                    config.image_topic,
                    "/cam0/image_raw/compressed",
                    "/cam1/image_raw/compressed",
                    "/cam0/image_raw",
                    "/cam1/image_raw",
                ],
            }
        )

        extractor = _EXTRACT_SELECTED_FRAMES_SCRIPT
        wrapper = f"""#!/bin/bash
set +e
mkdir -p /output/images
{self.get_ros_source_cmd()}
python3 /stage_runtime/extract_selected_frames.py '{runtime_args}' 2>&1 | tee /output/{self.name}.log
STATUS=${{PIPESTATUS[0]}}
echo "$STATUS" > /output/{self.name}.status
exit 0
"""

        return runner.run_stage(
            container_profile=self.container_profile,
            input_dir=bag_dir,
            config=config,
            spec=ExecutionSpec(
                stage_name=self.name,
                command=["/bin/bash", "/stage_runtime/run_image_selector.sh"],
                files={
                    "extract_selected_frames.py": extractor,
                    "run_image_selector.sh": wrapper,
                },
            ),
        )


_EXTRACT_SELECTED_FRAMES_SCRIPT = r'''#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def unique_preserving_order(items):
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def open_reader(bag_path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    return reader


def decode_image(message, msg_type, bridge):
    if msg_type == "sensor_msgs/msg/CompressedImage":
        array = np.frombuffer(message.data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Failed to decode CompressedImage payload")
        return image
    if msg_type == "sensor_msgs/msg/Image":
        return bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
    raise RuntimeError(f"Unsupported image message type: {msg_type}")


def main():
    args = json.loads(sys.argv[1])
    bag_path = Path(args["bag_path"])
    output_dir = Path(args["output_dir"])
    metadata_path = Path(args["metadata_path"])
    requested_frames = sorted(set(int(frame) for frame in args["frame_numbers"]))
    preferred_topics = unique_preserving_order(args["preferred_topics"])

    reader = open_reader(bag_path)
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}

    selected_topic = None
    for topic in preferred_topics:
        if topic in topic_types:
            selected_topic = topic
            break
    if selected_topic is None:
        raise RuntimeError(
            "No suitable image topic found. Tried "
            f"{preferred_topics}. Available: {sorted(topic_types)}"
        )

    selected_type = topic_types[selected_topic]
    if selected_type not in {"sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/Image"}:
        raise RuntimeError(f"Unsupported selected topic type: {selected_type}")

    output_dir.mkdir(parents=True, exist_ok=True)
    bridge = CvBridge()
    message_class = get_message(selected_type)
    wanted = set(requested_frames)
    extracted = []
    topic_frame_idx = -1

    reader = open_reader(bag_path)
    while reader.has_next() and wanted:
        topic_name, data, timestamp = reader.read_next()
        if topic_name != selected_topic:
            continue
        topic_frame_idx += 1
        if topic_frame_idx not in wanted:
            continue
        message = deserialize_message(data, message_class)
        image = decode_image(message, selected_type, bridge)

        header_stamp = getattr(getattr(message, "header", None), "stamp", None)
        if header_stamp is not None:
            frame_timestamp_ns = int(header_stamp.sec) * 1000000000 + int(header_stamp.nanosec)
        else:
            frame_timestamp_ns = int(timestamp)
        image_name = f"frame_{topic_frame_idx:06d}.png"
        image_path = output_dir / image_name
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(f"Failed to write {image_path}")
        extracted.append(
            {
                "frame_number": topic_frame_idx,
                "timestamp_ns": frame_timestamp_ns,
                "bag_timestamp_ns": int(timestamp),
                "topic": selected_topic,
                "type": selected_type,
                "image": f"images/{image_name}",
                "shape": list(image.shape),
            }
        )
        wanted.remove(topic_frame_idx)

    metadata = {
        "schema_version": 1,
        "bag_path": str(bag_path),
        "selected_topic": selected_topic,
        "selected_type": selected_type,
        "requested_frames": requested_frames,
        "missing_frames": sorted(wanted),
        "extracted": extracted,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if wanted:
        raise RuntimeError(f"Failed to extract requested frames: {sorted(wanted)}")
    print(f"[image_selector] Topic: {selected_topic}")
    print(f"[image_selector] Extracted frames: {requested_frames}")
    print(f"[image_selector] Output: {output_dir}")


if __name__ == "__main__":
    main()
'''
