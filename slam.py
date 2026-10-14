"""Dense RGB-D SLAM over a TUM RGB-D sequence with Open3D's tensor pipeline.

Each frame is tracked against a raycast of the map built so far (frame-to-model ICP), then
fused into a TSDF voxel grid at the tracked pose: localization and mapping, simultaneously.
There is no loop closure, so drift accumulates on long sequences; the absolute trajectory
error against the dataset's motion-capture ground truth makes that visible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import open3d as o3d
import open3d.core as o3c

# TUM depth PNGs store millimetres * 5; Open3D divides raw values by this to get metres.
DEPTH_SCALE = 5000.0
DEPTH_MIN = 0.1
DEPTH_MAX = 3.0
VOXEL_SIZE = 0.0058
BLOCK_RESOLUTION = 16
BLOCK_COUNT = 40_000
TRUNC_VOXEL_MULTIPLIER = 8.0
ODOMETRY_DISTANCE_THR = 0.07
# Mesh extraction: voxels observed fewer times than this are treated as noise, the mesh is
# simplified by merging vertices within this many metres, and fragments with fewer triangles than
# this are dropped.
MESH_WEIGHT_THRESHOLD = 3.0
MESH_CLUSTER_M = 0.02
MESH_MIN_COMPONENT_TRIANGLES = 300
# Frames whose depth and colour timestamps differ by more than this are not paired.
MAX_ASSOCIATION_DT_S = 0.02

# Per-camera pinhole intrinsics published with the dataset (fx, fy, cx, cy).
INTRINSICS = {
    "freiburg1": (517.3, 516.5, 318.6, 255.3),
    "freiburg2": (520.9, 521.0, 325.1, 249.7),
    "freiburg3": (535.4, 539.2, 320.1, 247.6),
}


@dataclass(frozen=True)
class Sequence:
    name: str
    root: Path
    timestamps: list[float]
    depth_files: list[Path]
    color_files: list[Path]
    intrinsics: tuple[float, float, float, float]
    ground_truth: dict[float, np.ndarray]


def _read_list(path: Path) -> list[tuple[float, str]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        stamp, rest = line.split(maxsplit=1)
        rows.append((float(stamp), rest.strip()))
    return rows


def _associate(a: list[tuple[float, str]], b: list[tuple[float, str]], max_dt: float) -> list[tuple[float, str, str]]:
    """Greedily pairs each entry of `a` with the nearest unused entry of `b` in time."""
    stamps_b = np.array([t for t, _ in b])
    used: set[int] = set()
    out = []
    for t, value in a:
        j = int(np.argmin(np.abs(stamps_b - t)))
        if j not in used and abs(stamps_b[j] - t) <= max_dt:
            used.add(j)
            out.append((t, value, b[j][1]))
    return out


def _pose_matrix(tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
    m[:3, 3] = [tx, ty, tz]
    return m


def load_sequence(root: Path, stride: int) -> Sequence:
    """Loads a TUM RGB-D sequence directory, keeping every `stride`-th associated frame."""
    pairs = _associate(_read_list(root / "depth.txt"), _read_list(root / "rgb.txt"), MAX_ASSOCIATION_DT_S)[::stride]
    gt = {}
    for t, rest in _read_list(root / "groundtruth.txt"):
        gt[t] = _pose_matrix(*map(float, rest.split()))
    camera = next(k for k in INTRINSICS if k in root.name)
    return Sequence(
        name=root.name,
        root=root,
        timestamps=[t for t, _, _ in pairs],
        depth_files=[root / d for _, d, _ in pairs],
        color_files=[root / c for _, _, c in pairs],
        intrinsics=INTRINSICS[camera],
        ground_truth=gt,
    )


@dataclass(frozen=True)
class FrameResult:
    index: int
    pose: np.ndarray
    fitness: float
    seconds: float


class DenseSlam:
    """One run of tracking + TSDF integration over a sequence."""

    def __init__(self, seq: Sequence, device: o3c.Device) -> None:
        super().__init__()
        self.seq = seq
        self.device = device
        fx, fy, cx, cy = seq.intrinsics
        self.intrinsic = o3c.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], o3c.Dtype.Float64)
        self.model = o3d.t.pipelines.slam.Model(
            VOXEL_SIZE,
            BLOCK_RESOLUTION,
            BLOCK_COUNT,
            o3c.Tensor(np.eye(4)),
            device,
        )
        first = o3d.t.io.read_image(str(seq.depth_files[0]))
        self.input_frame = o3d.t.pipelines.slam.Frame(first.rows, first.columns, self.intrinsic, device)
        self.raycast_frame = o3d.t.pipelines.slam.Frame(first.rows, first.columns, self.intrinsic, device)
        self.pose = o3c.Tensor(np.eye(4))
        self.poses: list[np.ndarray] = []

    def frames(self) -> Iterator[FrameResult]:
        for i, (depth_file, color_file) in enumerate(zip(self.seq.depth_files, self.seq.color_files)):
            t0 = time.perf_counter()
            depth = o3d.t.io.read_image(str(depth_file)).to(self.device)
            color = o3d.t.io.read_image(str(color_file)).to(self.device)
            self.input_frame.set_data_from_image("depth", depth)
            self.input_frame.set_data_from_image("color", color)
            fitness = 1.0
            if i > 0:
                result = self.model.track_frame_to_model(
                    self.input_frame,
                    self.raycast_frame,
                    DEPTH_SCALE,
                    DEPTH_MAX,
                    ODOMETRY_DISTANCE_THR,
                )
                self.pose = self.pose @ result.transformation
                fitness = float(result.fitness)
            self.model.update_frame_pose(i, self.pose)
            self.model.integrate(self.input_frame, DEPTH_SCALE, DEPTH_MAX, TRUNC_VOXEL_MULTIPLIER)
            self.model.synthesize_model_frame(
                self.raycast_frame, DEPTH_SCALE, DEPTH_MIN, DEPTH_MAX, TRUNC_VOXEL_MULTIPLIER, False
            )
            pose = self.pose.cpu().numpy()
            self.poses.append(pose)
            yield FrameResult(i, pose, fitness, time.perf_counter() - t0)

    def frame_points(self, index: int, max_points: int) -> tuple[np.ndarray, np.ndarray]:
        """A subsample of frame `index`'s depth, back-projected into the map frame with colours."""
        depth = o3d.t.io.read_image(str(self.seq.depth_files[index]))
        color = o3d.t.io.read_image(str(self.seq.color_files[index]))
        rgbd = o3d.t.geometry.RGBDImage(color, depth)
        pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            self.intrinsic,
            o3c.Tensor(np.linalg.inv(self.poses[index])),
            DEPTH_SCALE,
            DEPTH_MAX,
            4,
        )
        pts = pcd.point.positions.numpy()
        cols = pcd.point.colors.numpy()
        if len(pts) > max_points:
            keep = np.random.default_rng(index).choice(len(pts), max_points, replace=False)
            pts, cols = pts[keep], cols[keep]
        return pts.astype(np.float32), cols.astype(np.float32)

    def mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The fused TSDF surface as a coloured triangle mesh: (positions, colours, faces).

        Marching cubes yields ~5M triangles at this voxel size. Vertex clustering brings that to a
        few hundred thousand, a size a browser renders smoothly, and dropping small disconnected
        pieces removes floaters left by noisy depth at object edges.
        """
        mesh = self.model.voxel_grid.extract_triangle_mesh(weight_threshold=MESH_WEIGHT_THRESHOLD)
        mesh = (
            mesh.cpu()
            .to_legacy()
            .simplify_vertex_clustering(
                voxel_size=MESH_CLUSTER_M, contraction=o3d.geometry.SimplificationContraction.Average
            )
        )
        mesh.remove_degenerate_triangles()
        ids, counts, _ = mesh.cluster_connected_triangles()
        mesh.remove_triangles_by_mask(np.asarray(counts)[np.asarray(ids)] < MESH_MIN_COMPONENT_TRIANGLES)
        mesh.remove_unreferenced_vertices()
        return (
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.vertex_colors, dtype=np.float32),
            np.asarray(mesh.triangles, dtype=np.uint32),
        )


def ground_truth_trajectory(seq: Sequence, count: int) -> np.ndarray | None:
    """Ground-truth camera positions for the first `count` frames, nearest in time."""
    if not seq.ground_truth:
        return None
    stamps = np.array(sorted(seq.ground_truth))
    out = []
    for t in seq.timestamps[:count]:
        j = int(np.argmin(np.abs(stamps - t)))
        out.append(seq.ground_truth[float(stamps[j])][:3, 3])
    return np.array(out)


def align_and_ate(estimated: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, float]:
    """Rigidly aligns `truth` onto `estimated` (Horn/Umeyama, no scale) and returns the
    aligned truth plus the RMS absolute trajectory error in metres."""
    mu_e, mu_t = estimated.mean(0), truth.mean(0)
    h = (truth - mu_t).T @ (estimated - mu_e)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    aligned = (r @ (truth - mu_t).T).T + mu_e
    ate = float(np.sqrt(np.mean(np.sum((aligned - estimated) ** 2, axis=1))))
    return aligned, ate
