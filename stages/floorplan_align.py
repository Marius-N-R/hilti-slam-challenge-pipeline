"""Floorplan-align stage - SE(3) residual correction by matching rays to floorplan edges.

Adapted from rework/Floorplan-Alignment/src/vizualize_results_edge_alignment.py.
The angleRangeCheck helper, the per-ray edge-matching loop, the theta grid
search, and the (dx, dy, dz) grid search are kept faithful to the reference.
Visualization (matplotlib sliders, 3D triad plots) is dropped.

Inputs:
  * rays.csv from `rays` stage (origin + two ray endpoints per detected line)
  * floorplan_edges.csv from `floorplan_edges` stage (2D wall segments in meters)
  * trajectory_pca_aligned.csv from `pca_align` when present, otherwise
    trajectory_aligned.csv from `align` (start-pose-anchored trajectory)

Output:
  * trajectory_floor_aligned.csv: the input trajectory with the residual SE(3)
    correction (yaw rotation + xyz translation) applied to every pose.
"""

import csv
import math
import tempfile
from pathlib import Path

import numpy as np

from ._geometry import quat_to_rot, rot_to_quat
from .base import Stage, StageConfig, stage_output_path

OUTPUT_CSV = "trajectory_floor_aligned.csv"
PCA_ALIGNED_TRAJ_CSV = "trajectory_pca_aligned.csv"
ALIGNED_TRAJ_CSV = "trajectory_aligned.csv"

# Search ranges. Theta range matches the reference (-20..+20 deg). For
# translations we depart from the reference because:
#   * X is widened to +/-5 m so the search can absorb a few meters of slack in
#     the user-provided floorplan_offset.txt without pegging at the boundary.
#   * Y stays at +/-2.5 m (the reference range was already fine for floor_1).
#   * Z is pinned to 0: the floorplan is 2D so the plane-to-edge distance
#     metric does not constrain z; an unpinned search slides z to whichever
#     boundary nudges other dims best, producing an unphysical sub-floor offset.
THETA_RANGE_DEG = (-20.0, 20.0)
THETA_STEPS = 81
TRANS_X_RANGE_M = (-5.0, 5.0)
TRANS_X_STEPS = 41
TRANS_Y_RANGE_M = (-2.5, 2.5)
TRANS_Y_STEPS = 11
TRANS_Z_RANGE_M = (0.0, 0.0)
TRANS_Z_STEPS = 1


class FloorplanAlignStage(Stage):
    """Compute a residual rigid SE(3) correction by matching rays to floorplan edges."""

    @property
    def name(self) -> str:
        return "floorplan_align"

    @property
    def description(self) -> str:
        return "Refine aligned trajectory by matching detected line rays to floorplan edges"

    @property
    def requires_container(self) -> bool:
        return False

    @property
    def container_profile(self) -> str:
        return "host"

    def run(self, runner, input_dir: Path, config: StageConfig) -> Path:
        rays_path = _resolve(input_dir, config, "rays", "rays.csv")
        edges_path = _resolve(None, config, "floorplan_edges", "floorplan_edges.csv")
        traj_path = _resolve_trajectory(config)

        rays = _load_rays(rays_path)
        edges = _load_edges(edges_path)
        traj_t, traj_xyz, traj_q = _load_aligned_traj(traj_path)

        if not config.align_start_position:
            print(
                "[floorplan_align] WARNING: start alignment is disabled; "
                "floorplan matching is using OpenVINS-world coordinates"
            )

        if rays.size == 0:
            raise RuntimeError(f"No rays found in {rays_path}")
        if edges.size == 0:
            raise RuntimeError(f"No floorplan edges found in {edges_path}")

        # 1) Match each ray pair to the closest floorplan edge.
        ray_pairs = _match_rays_to_edges(rays, edges)
        matched = sum(1 for rp in ray_pairs if rp[3] is not None)
        print(f"[floorplan_align] Matched {matched}/{len(ray_pairs)} ray pairs to a floorplan edge")

        # 2) Grid-search yaw correction.
        theta_correction = _grid_search_theta(ray_pairs)
        print(f"[floorplan_align] theta correction = {math.degrees(theta_correction):+.3f} deg")

        # 3) Grid-search translation correction with rotation applied first.
        rotation = _rotz(theta_correction)
        translation = _grid_search_translation(ray_pairs, rotation)
        print(
            f"[floorplan_align] translation correction "
            f"(dx,dy,dz) = ({translation[0]:+.3f}, {translation[1]:+.3f}, {translation[2]:+.3f}) m"
        )

        # 4) Apply correction to every pose in the aligned trajectory.
        traj_xyz_rot = (rotation @ traj_xyz.T).T + translation
        traj_q_corrected = np.empty_like(traj_q)
        for i in range(len(traj_q)):
            R_corrected = rotation @ quat_to_rot(*traj_q[i])
            qx, qy, qz, qw = rot_to_quat(R_corrected)
            traj_q_corrected[i] = (qx, qy, qz, qw)

        stage_root = Path(
            tempfile.mkdtemp(prefix=f"{self.name}-", dir=runner.runtime_dir)
        )
        output_dir = stage_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        out_csv = output_dir / OUTPUT_CSV
        _write_traj(out_csv, traj_t, traj_xyz_rot, traj_q_corrected)

        log_lines = [
            f"Rays: {rays_path} ({len(rays)})",
            f"Floorplan edges: {edges_path} ({len(edges)})",
            f"Input trajectory: {traj_path} ({len(traj_t)} poses)",
            f"Matched ray pairs: {matched}/{len(ray_pairs)}",
            f"theta correction: {math.degrees(theta_correction):+.3f} deg",
            (
                f"translation correction: dx={translation[0]:+.3f}, "
                f"dy={translation[1]:+.3f}, dz={translation[2]:+.3f} m"
            ),
            f"Output: {out_csv}",
        ]
        (output_dir / f"{self.name}.log").write_text(
            "\n".join(log_lines) + "\n", encoding="utf-8"
        )
        (output_dir / f"{self.name}.status").write_text("0", encoding="utf-8")

        return output_dir


# ---------------------------------------------------------------------------
# Resolver / loaders
# ---------------------------------------------------------------------------


def _resolve(input_dir, config, stage_name, filename):
    if input_dir is not None:
        candidate = input_dir / filename
        if candidate.is_file():
            return candidate
    try:
        path = stage_output_path(config, stage_name) / filename
        if path.is_file():
            return path
    except Exception:
        pass
    raise FileNotFoundError(
        f"Could not find {filename} (looked in current_data and the {stage_name} output dir)"
    )


def _resolve_trajectory(config):
    try:
        path = stage_output_path(config, "pca_align") / PCA_ALIGNED_TRAJ_CSV
        if path.is_file():
            return path
    except Exception:
        pass
    return _resolve(None, config, "align", ALIGNED_TRAJ_CSV)


def _read_numeric_rows(path: Path, expected_cols: int) -> np.ndarray:
    rows = []
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
            if len(vals) != expected_cols:
                continue
            rows.append(vals)
    return np.asarray(rows)


def _load_rays(path: Path) -> np.ndarray:
    return _read_numeric_rows(path, 10)


def _load_edges(path: Path) -> np.ndarray:
    return _read_numeric_rows(path, 4)


def _load_aligned_traj(path: Path):
    arr = _read_numeric_rows(path, 8)
    if arr.size == 0:
        return np.empty(0), np.empty((0, 3)), np.empty((0, 4))
    return arr[:, 0], arr[:, 1:4], arr[:, 4:8]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _rotz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _angle_range_check(edge, ray1_vec, ray2_vec, plane_point) -> bool:
    """Verbatim port of `angleRangeCheck` from vizualize_results_edge_alignment.py."""
    theta_ray1 = math.atan2(ray1_vec[1], ray1_vec[0])
    theta_ray2 = math.atan2(ray2_vec[1], ray2_vec[0])
    theta_vertex1 = math.atan2(edge[1] - plane_point[1], edge[0] - plane_point[0])
    theta_vertex2 = math.atan2(edge[3] - plane_point[1], edge[2] - plane_point[0])

    if theta_ray1 > math.pi / 2 and theta_ray2 < -math.pi / 2:
        if theta_vertex1 > theta_ray1 or theta_vertex1 < theta_ray2:
            return True
        if theta_vertex2 > theta_ray1 or theta_vertex2 < theta_ray2:
            return True
    elif theta_ray2 > math.pi / 2 and theta_ray1 < -math.pi / 2:
        if theta_vertex1 > theta_ray2 or theta_vertex1 < theta_ray1:
            return True
        if theta_vertex2 > theta_ray2 or theta_vertex2 < theta_ray1:
            return True
    else:
        if (theta_vertex1 >= theta_ray1 and theta_vertex1 < theta_ray2) or (
            theta_vertex1 < theta_ray1 and theta_vertex1 >= theta_ray2
        ):
            return True
        if (theta_vertex2 >= theta_ray1 and theta_vertex2 < theta_ray2) or (
            theta_vertex2 < theta_ray1 and theta_vertex2 >= theta_ray2
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Matching + grid searches (verbatim algorithm structure from the reference)
# ---------------------------------------------------------------------------


def _match_rays_to_edges(rays: np.ndarray, edges: np.ndarray):
    """Return list of (origin, ray1_vec, ray2_vec, best_edge_or_None)."""
    pairs = []
    for row in rays:
        origin = row[1:4]
        ray1_world_end = row[4:7]
        ray2_world_end = row[7:10]
        ray1_vec = ray1_world_end - origin
        ray2_vec = ray2_world_end - origin

        plane_normal = np.cross(ray1_vec, ray2_vec)
        nrm = np.linalg.norm(plane_normal)
        if nrm < 1e-9:
            pairs.append((origin, ray1_vec, ray2_vec, None))
            continue
        plane_normal = plane_normal / nrm

        best_alignment_score = math.inf
        best_distance_score = math.inf
        best_edge = None
        for edge in edges:
            edge_vec = np.array(
                [edge[2] - edge[0], edge[3] - edge[1], 0.0]
            )
            ev_norm = np.linalg.norm(edge_vec)
            if ev_norm < 1e-9:
                continue
            edge_vec_n = edge_vec / ev_norm

            if not _angle_range_check(edge, ray1_vec, ray2_vec, origin):
                continue

            alignment_score = abs(np.dot(plane_normal, edge_vec_n))
            if alignment_score <= best_alignment_score:
                dist_1 = abs(
                    np.dot(plane_normal, np.array([edge[0], edge[1], 0.0]) - origin)
                )
                dist_2 = abs(
                    np.dot(plane_normal, np.array([edge[2], edge[3], 0.0]) - origin)
                )
                distance_score = dist_1 + dist_2
                if distance_score < best_distance_score:
                    best_edge = edge
                    best_alignment_score = alignment_score
                    best_distance_score = distance_score

        pairs.append((origin, ray1_vec, ray2_vec, best_edge))
    return pairs


def _grid_search_theta(ray_pairs) -> float:
    thetas = np.linspace(THETA_RANGE_DEG[0], THETA_RANGE_DEG[1], THETA_STEPS) * math.pi / 180.0
    best_thetas = []
    for origin, ray1_vec, ray2_vec, edge in ray_pairs:
        if edge is None:
            continue
        edge_vec = np.array([edge[2] - edge[0], edge[3] - edge[1], 0.0])

        best_score = math.inf
        best_theta = 0.0
        for theta in thetas:
            R = _rotz(theta)
            r1 = R @ ray1_vec
            r2 = R @ ray2_vec
            plane_normal = np.cross(r1, r2)
            score = abs(np.dot(edge_vec, plane_normal))
            if score < best_score:
                best_score = score
                best_theta = theta
        best_thetas.append(best_theta)

    if not best_thetas:
        return 0.0
    return float(np.mean(best_thetas))


def _grid_search_translation(ray_pairs, rotation: np.ndarray) -> np.ndarray:
    rotated_origins = []
    rotated_r1s = []
    rotated_r2s = []
    edge_vecs = []
    edge_vertices = []
    for origin, r1_vec, r2_vec, edge in ray_pairs:
        if edge is None:
            continue
        edge_vec = np.array([edge[2] - edge[0], edge[3] - edge[1], 0.0])
        if np.linalg.norm(edge_vec) < 1e-9:
            continue
        rotated_origins.append(rotation @ origin)
        rotated_r1s.append(rotation @ r1_vec)
        rotated_r2s.append(rotation @ r2_vec)
        edge_vecs.append(edge_vec)
        edge_vertices.append(np.array([edge[0], edge[1], 0.0]))

    if not rotated_origins:
        return np.array([0.0, 0.0, 0.0])

    xs = np.linspace(TRANS_X_RANGE_M[0], TRANS_X_RANGE_M[1], TRANS_X_STEPS)
    ys = np.linspace(TRANS_Y_RANGE_M[0], TRANS_Y_RANGE_M[1], TRANS_Y_STEPS)
    zs = np.linspace(TRANS_Z_RANGE_M[0], TRANS_Z_RANGE_M[1], TRANS_Z_STEPS)

    best_total = math.inf
    best = (0.0, 0.0, 0.0)
    for dx in xs:
        for dy in ys:
            for dz in zs:
                delta = np.array([dx, dy, dz])
                total = 0.0
                for ro, r1, r2, ev, vx in zip(
                    rotated_origins,
                    rotated_r1s,
                    rotated_r2s,
                    edge_vecs,
                    edge_vertices,
                ):
                    plane1 = np.cross(r1, ev)
                    plane2 = np.cross(r2, ev)
                    nrm1 = np.linalg.norm(plane1)
                    nrm2 = np.linalg.norm(plane2)
                    if nrm1 < 1e-9 or nrm2 < 1e-9:
                        continue
                    ray_origin_vertex = ro + delta
                    D_1_edge = -np.dot(plane1, vx)
                    D_1_ray = -np.dot(plane1, ray_origin_vertex)
                    D_2_edge = -np.dot(plane2, vx)
                    D_2_ray = -np.dot(plane2, ray_origin_vertex)
                    d1 = abs(D_1_edge - D_1_ray) / nrm1
                    d2 = abs(D_2_edge - D_2_ray) / nrm2
                    total += d1 + d2
                if total < best_total:
                    best_total = total
                    best = (dx, dy, dz)
    return np.array(best)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _write_traj(path: Path, ts: np.ndarray, xyz: np.ndarray, q: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["# timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        for i in range(len(ts)):
            writer.writerow(
                [
                    f"{ts[i]:.9f}",
                    f"{xyz[i, 0]:.9f}", f"{xyz[i, 1]:.9f}", f"{xyz[i, 2]:.9f}",
                    f"{q[i, 0]:.9f}", f"{q[i, 1]:.9f}", f"{q[i, 2]:.9f}", f"{q[i, 3]:.9f}",
                ]
            )
