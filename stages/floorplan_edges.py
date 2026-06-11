"""Floorplan edge extractor - reads a DXF floor plan and emits 2D wall segments.

Adapted from rework/Floorplan-Alignment/src/floorplan_edge_extractor.py. The
walk over LWPOLYLINE entities is unchanged; we drop the x100 pixel scaling and
PNG-vs-DXF offset because downstream stages run in meters/world frame. The
pixel scale is only needed by floorplan visualization (not on the alignment
critical path), which can apply it as a one-line transform when rendering.
"""

import csv
import math
import tempfile
from pathlib import Path

try:
    import ezdxf
except ImportError as exc:  # pragma: no cover - reported at runtime
    ezdxf = None  # type: ignore[assignment]
    _EZDXF_IMPORT_ERROR = exc

from .base import Stage, StageConfig

OUTPUT_CSV = "floorplan_edges.csv"
OFFSET_FILENAME = "floorplan_offset.txt"

# Minimum DXF segment length to keep, in DXF units. The reference script used
# 40 in pixel units after a x100 scaling -> 0.4 m. Kept identical.
MIN_LENGTH_M = 0.4


class FloorplanEdgesStage(Stage):
    """Convert the per-run DXF floor plan to a CSV of 2D wall segments (meters)."""

    @property
    def name(self) -> str:
        return "floorplan_edges"

    @property
    def description(self) -> str:
        return "Extract 2D wall segments from the floor plan DXF"

    @property
    def requires_container(self) -> bool:
        return False

    @property
    def container_profile(self) -> str:
        return "host"

    def run(self, runner, input_dir: Path, config: StageConfig) -> Path:
        if ezdxf is None:
            raise RuntimeError(
                "ezdxf is required for the floorplan_edges stage; "
                f"`pip install ezdxf` ({_EZDXF_IMPORT_ERROR})"
            )

        original_input_str = config.extra.get("current_input_path", "")
        if not original_input_str:
            raise RuntimeError("Original input path not set in config.extra")
        original_input = Path(original_input_str)
        dxf_candidates = sorted(original_input.glob("*.dxf"))
        if not dxf_candidates:
            raise FileNotFoundError(
                f"No *.dxf file found in original input folder: {original_input}"
            )
        if len(dxf_candidates) > 1:
            print(
                f"[{self.name}] WARNING: multiple .dxf files found; using "
                f"{dxf_candidates[0].name}"
            )
        dxf_path = dxf_candidates[0]

        offset_path = original_input / OFFSET_FILENAME
        offset = _load_offset(offset_path) if offset_path.is_file() else (0.0, 0.0)

        segments = _extract_segments(dxf_path)

        stage_root = Path(
            tempfile.mkdtemp(prefix=f"{self.name}-", dir=runner.runtime_dir)
        )
        output_dir = stage_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / OUTPUT_CSV
        ox, oy = offset
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x1", "y1", "x2", "y2"])
            for (x1, y1), (x2, y2) in segments:
                writer.writerow(
                    [
                        f"{x1 + ox:.6f}",
                        f"{y1 + oy:.6f}",
                        f"{x2 + ox:.6f}",
                        f"{y2 + oy:.6f}",
                    ]
                )

        offset_note = (
            f"offset: ({ox:+.4f}, {oy:+.4f}) m from {offset_path}"
            if offset_path.is_file()
            else "offset: (0, 0) m (no floorplan_offset.txt found)"
        )
        log_lines = [
            f"DXF: {dxf_path}",
            offset_note,
            f"Kept {len(segments)} segments (min length {MIN_LENGTH_M} m)",
            f"Output: {csv_path}",
        ]
        (output_dir / f"{self.name}.log").write_text(
            "\n".join(log_lines) + "\n", encoding="utf-8"
        )
        (output_dir / f"{self.name}.status").write_text("0", encoding="utf-8")
        for line in log_lines:
            print(f"[{self.name}] {line}")

        return output_dir


def _load_offset(path: Path):
    """Load an (offset_x, offset_y) pair in meters.

    Format mirrors initial-pos.txt: whitespace- or comma-separated, `#` header lines
    are ignored. The first data row must contain exactly two floats.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if len(vals) != 2:
                continue
            return float(vals[0]), float(vals[1])
    raise ValueError(f"No (offset_x, offset_y) row found in {path}")


def _extract_segments(dxf_path: Path):
    """Walk LWPOLYLINE entities and yield (p1, p2) segment pairs (in meters).

    Adapted verbatim from floorplan_edge_extractor.py, sans pixel scale / offset.
    """
    drawing = ezdxf.readfile(str(dxf_path))
    msp = drawing.modelspace()

    segments = []
    for entity in msp:
        if entity.dxftype() != "LWPOLYLINE":
            continue
        points = entity.get_points()
        for idx in range(len(points) - 1):
            x1, y1 = points[idx][:2]
            x2, y2 = points[idx + 1][:2]
            if math.hypot(x2 - x1, y2 - y1) > MIN_LENGTH_M:
                segments.append(((float(x1), float(y1)), (float(x2), float(y2))))
    return segments
