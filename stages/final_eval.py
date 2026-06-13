"""Final trajectory evaluation against input ground truth."""

import csv
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from .base import Stage, StageConfig, stage_output_path

GROUNDTRUTH_FILENAME = "groundtruth.txt"
MATCHED_ERRORS_CSV = "matched_errors.csv"
SUMMARY_JSON = "summary.json"

TRAJECTORY_CANDIDATES = [
    ("combined_align", "trajectory_combined_aligned.csv", "combined_aligned"),
    ("floorplan_align", "trajectory_floor_aligned.csv", "floor_aligned"),
    ("window_align", "trajectory_window_aligned.csv", "window_aligned"),
    ("pca_align", "trajectory_pca_aligned.csv", "pca_aligned"),
    ("align", "trajectory_aligned.csv", "aligned"),
    ("slam", "trajectory.txt", "slam"),
]


class FinalEvalStage(Stage):
    """Evaluate the final estimated path against ground truth."""

    @property
    def name(self) -> str:
        return "final_eval"

    @property
    def description(self) -> str:
        return "Evaluate the final aligned trajectory against ground truth"

    @property
    def requires_container(self) -> bool:
        return False

    @property
    def container_profile(self) -> str:
        return "host"

    @property
    def input_type(self) -> str:
        return "trajectory"

    @property
    def output_type(self) -> str:
        return "evaluation"

    def run(self, runner, input_dir: Path, config: StageConfig) -> Path:
        gt_path = _find_groundtruth(config)
        est_path, est_kind = _find_estimate(input_dir, config)

        gt = _load_pose_table(gt_path)
        est = _load_pose_table(est_path)
        matches = _match_by_timestamp(
            estimate=est,
            groundtruth=gt,
            max_dt=config.eval_max_time_delta,
        )
        if matches.size == 0:
            raise RuntimeError(
                "No timestamp matches found between estimate and ground truth "
                f"within {config.eval_max_time_delta:.3f}s"
            )

        summary = _summarize(matches)
        summary.update(
            {
                "estimate_path": str(est_path),
                "estimate_kind": est_kind,
                "groundtruth_path": str(gt_path),
                "estimate_poses": int(len(est)),
                "groundtruth_poses": int(len(gt)),
                "matched_poses": int(len(matches)),
                "unmatched_estimate_poses": int(len(est) - len(matches)),
                "max_time_delta_s": float(config.eval_max_time_delta),
            }
        )

        stage_root = Path(
            tempfile.mkdtemp(prefix=f"{self.name}-", dir=runner.runtime_dir)
        )
        output_dir = stage_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        _write_matches(output_dir / MATCHED_ERRORS_CSV, matches)
        (output_dir / SUMMARY_JSON).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        log_lines = [
            f"Estimate: {est_path} ({est_kind}, {len(est)} poses)",
            f"Ground truth: {gt_path} ({len(gt)} poses)",
            (
                f"Matched poses: {len(matches)}/{len(est)} "
                f"(max dt {config.eval_max_time_delta:.3f}s)"
            ),
            f"XY RMSE: {summary['xy']['rmse_m']:.4f} m",
            f"XY mean / median / p95: "
            f"{summary['xy']['mean_m']:.4f} / "
            f"{summary['xy']['median_m']:.4f} / "
            f"{summary['xy']['p95_m']:.4f} m",
            f"XYZ RMSE: {summary['xyz']['rmse_m']:.4f} m",
            f"Summary: {output_dir / SUMMARY_JSON}",
            f"Matched errors: {output_dir / MATCHED_ERRORS_CSV}",
        ]
        (output_dir / f"{self.name}.log").write_text(
            "\n".join(log_lines) + "\n", encoding="utf-8"
        )
        (output_dir / f"{self.name}.status").write_text("0", encoding="utf-8")
        for line in log_lines:
            print(f"[{self.name}] {line}")

        return output_dir


def _find_groundtruth(config: StageConfig) -> Path:
    original_input = config.extra.get("current_input_path", "")
    if not original_input:
        raise RuntimeError("Original input path not set in config.extra")
    gt_path = Path(original_input) / GROUNDTRUTH_FILENAME
    if not gt_path.is_file():
        raise FileNotFoundError(f"Expected {GROUNDTRUTH_FILENAME} in {original_input}")
    return gt_path


def _find_estimate(input_dir: Path, config: StageConfig):
    for stage_name, filename, kind in TRAJECTORY_CANDIDATES:
        candidate = input_dir / filename
        if candidate.is_file():
            return candidate, kind
        try:
            candidate = stage_output_path(config, stage_name) / filename
            if candidate.is_file():
                return candidate, kind
        except Exception:
            pass
    expected = ", ".join(f"{stage}/{filename}" for stage, filename, _ in TRAJECTORY_CANDIDATES)
    raise FileNotFoundError(f"No estimate trajectory found. Expected one of: {expected}")


def _load_pose_table(path: Path) -> np.ndarray:
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
    rows.sort(key=lambda row: row[0])
    return np.asarray(rows, dtype=float)


def _match_by_timestamp(
    estimate: np.ndarray,
    groundtruth: np.ndarray,
    max_dt: float,
) -> np.ndarray:
    gt_t = groundtruth[:, 0]
    matched_rows = []
    for est_row in estimate:
        idx = int(np.searchsorted(gt_t, est_row[0]))
        candidates = []
        if idx < len(groundtruth):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        if not candidates:
            continue

        best_idx = min(candidates, key=lambda gt_idx: abs(gt_t[gt_idx] - est_row[0]))
        gt_row = groundtruth[best_idx]
        dt = est_row[0] - gt_row[0]
        if abs(dt) > max_dt:
            continue

        diff = est_row[1:4] - gt_row[1:4]
        xy_error = math.hypot(diff[0], diff[1])
        xyz_error = float(np.linalg.norm(diff))
        matched_rows.append(
            [
                est_row[0],
                gt_row[0],
                dt,
                est_row[1],
                est_row[2],
                est_row[3],
                gt_row[1],
                gt_row[2],
                gt_row[3],
                diff[0],
                diff[1],
                diff[2],
                xy_error,
                abs(diff[2]),
                xyz_error,
            ]
        )
    return np.asarray(matched_rows, dtype=float)


def _summarize(matches: np.ndarray) -> dict:
    xy = matches[:, 12]
    z_abs = matches[:, 13]
    xyz = matches[:, 14]
    dt_abs = np.abs(matches[:, 2])
    return {
        "xy": _stats(xy, unit_suffix="m"),
        "z_abs": _stats(z_abs, unit_suffix="m"),
        "xyz": _stats(xyz, unit_suffix="m"),
        "time_delta_abs": _stats(dt_abs, unit_suffix="s"),
    }


def _stats(values: np.ndarray, unit_suffix: str) -> dict:
    return {
        f"mean_{unit_suffix}": float(np.mean(values)),
        f"median_{unit_suffix}": float(np.median(values)),
        f"rmse_{unit_suffix}": float(math.sqrt(np.mean(values * values))),
        f"p90_{unit_suffix}": float(np.percentile(values, 90)),
        f"p95_{unit_suffix}": float(np.percentile(values, 95)),
        f"max_{unit_suffix}": float(np.max(values)),
    }


def _write_matches(path: Path, matches: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "estimate_timestamp",
                "groundtruth_timestamp",
                "dt_s",
                "estimate_x",
                "estimate_y",
                "estimate_z",
                "groundtruth_x",
                "groundtruth_y",
                "groundtruth_z",
                "dx",
                "dy",
                "dz",
                "xy_error_m",
                "z_abs_error_m",
                "xyz_error_m",
            ]
        )
        for row in matches:
            writer.writerow([f"{value:.9f}" for value in row])
