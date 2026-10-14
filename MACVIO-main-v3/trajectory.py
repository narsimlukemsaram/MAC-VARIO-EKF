"""
trajectory.py — UMRP4 MAC-VIO
Trajectory accumulator + ATE (Sim3) + RPE (rrel/trel per-frame).

ATE: Horn 1987 closed-form Sim3 alignment (scale+rot+trans).
RPE: per-frame relative rotation error (deg) and translation error (m).
TUM format: timestamp tx ty tz qx qy qz qw  (space-separated).

Sim3 alignment follows evo / EuRoC evaluation convention:
  - align estimated trajectory to GT via Sim3
  - report RMSE of translation differences after alignment
"""

import numpy as np
from pathlib import Path


def _filter_finite_pairs(p_est, p_gt):
    # remove rows where either array has nan/inf
    mask = np.isfinite(p_est).all(axis=1) & np.isfinite(p_gt).all(axis=1)
    n_bad = (~mask).sum()
    if n_bad > 0:
        print(f"  [trajectory] WARNING: dropping {n_bad} non-finite frames before ATE/Sim3")
    return p_est[mask], p_gt[mask], mask


# Sim3 closed-form (Horn 1987 + Umeyama 1991)
def _sim3_align(
    p_est: np.ndarray,   # (N, 3) estimated positions
    p_gt:  np.ndarray,   # (N, 3) ground-truth positions
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (R, t, s) such that s*R*p_est + t ≈ p_gt  (Umeyama 1991).
    """
    N = p_est.shape[0]
    assert N >= 3, "Need >= 3 points for Sim3 alignment"

    # guard: any remaining nan/inf causes SVD to fail
    if not (np.isfinite(p_est).all() and np.isfinite(p_gt).all()):
        raise ValueError("_sim3_align received non-finite positions after filtering")

    mu_e = p_est.mean(0)
    mu_g = p_gt.mean(0)
    e_c  = p_est - mu_e
    g_c  = p_gt  - mu_g

    sigma2_e = (e_c ** 2).sum() / N   # variance of est
    if sigma2_e < 1e-10:
        # degenerate: all estimates identical, return identity alignment
        return np.eye(3), mu_g - mu_e, 1.0

    H = (g_c.T @ e_c) / N             # cross-covariance (3,3)

    # clamp H to prevent overflow in SVD (EuRoC positions are O(10) m)
    H = np.clip(H, -1e6, 1e6)

    try:
        U, d, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        # fallback: identity rotation, zero scale, centroid translation
        print("  [trajectory] WARNING: SVD failed — using centroid-only alignment")
        return np.eye(3), mu_g - mu_e, 1.0

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1   # handle reflection

    R = U @ S @ Vt                       # (3,3) rotation
    s = (d * S.diagonal()).sum() / sigma2_e if sigma2_e > 1e-10 else 1.0
    t = mu_g - s * R @ mu_e

    return R, t, float(s)


def _apply_sim3(
    p: np.ndarray,  # (N, 3)
    R: np.ndarray,  # (3, 3)
    t: np.ndarray,  # (3,)
    s: float,
) -> np.ndarray:
    return (s * (R @ p.T)).T + t


def compute_ate(
    positions_est: np.ndarray,   # (N, 3)
    positions_gt:  np.ndarray,   # (N, 3)
    align: bool = True,
) -> float:
    """ATE RMSE in meters. Sim3 alignment by default. Skips non-finite frames."""
    assert positions_est.shape == positions_gt.shape

    # filter non-finite before alignment
    p_est, p_gt, _ = _filter_finite_pairs(positions_est, positions_gt)

    if len(p_est) < 3:
        print(f"  [trajectory] WARNING: only {len(p_est)} finite frames — ATE unreliable")
        if len(p_est) == 0:
            return float('nan')

    if align and len(p_est) >= 3:
        R, t, s = _sim3_align(p_est, p_gt)
        p_aligned = _apply_sim3(p_est, R, t, s)
    else:
        p_aligned = p_est

    errors = np.linalg.norm(p_aligned - p_gt, axis=1)
    return float(np.sqrt((errors ** 2).mean()))


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """xyzw quaternion → (3,3) rotation matrix."""
    x, y, z, w = q
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float64)
    return R


def compute_rpe(
    poses_est: np.ndarray,   # (N, 7) [tx ty tz qx qy qz qw]
    poses_gt:  np.ndarray,   # (N, 7)
) -> tuple[float, float]:
    """
    Per-frame RPE: rrel (deg/frame) and trel (m/frame).
    Relative pose error between consecutive frame pairs.
    Returns (rrel_mean_deg, trel_mean_m). Skips non-finite frames.
    """
    N = len(poses_est)
    assert N >= 2
    rrels, trels = [], []

    for i in range(N - 1):
        # skip if any pose is non-finite
        if not (np.isfinite(poses_est[i]).all() and np.isfinite(poses_est[i+1]).all()):
            continue
        if not (np.isfinite(poses_gt[i]).all() and np.isfinite(poses_gt[i+1]).all()):
            continue

        # GT relative pose
        R_gt_k  = _quat_to_rot(poses_gt[i,  3:])
        R_gt_k1 = _quat_to_rot(poses_gt[i+1, 3:])
        t_gt_k  = poses_gt[i,  :3]
        t_gt_k1 = poses_gt[i+1, :3]
        R_gt_rel = R_gt_k.T @ R_gt_k1
        t_gt_rel = R_gt_k.T @ (t_gt_k1 - t_gt_k)

        # Est relative pose
        R_e_k  = _quat_to_rot(poses_est[i,  3:])
        R_e_k1 = _quat_to_rot(poses_est[i+1, 3:])
        t_e_k  = poses_est[i,  :3]
        t_e_k1 = poses_est[i+1, :3]
        R_e_rel = R_e_k.T @ R_e_k1
        t_e_rel = R_e_k.T @ (t_e_k1 - t_e_k)

        # Error
        R_err = R_gt_rel.T @ R_e_rel
        trace = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
        angle_deg = np.degrees(np.arccos(trace))
        trans_err = np.linalg.norm(t_gt_rel - t_e_rel)

        rrels.append(angle_deg)
        trels.append(trans_err)

    if not rrels:
        return float('nan'), float('nan')
    return float(np.mean(rrels)), float(np.mean(trels))


class TrajectoryAccumulator:
    """
    Collects (timestamp_ns, position, quaternion_xyzw) per frame.
    Provides ATE + RPE vs ground truth.
    """

    def __init__(self):
        self._times:  list[int]       = []
        self._poses:  list[np.ndarray] = []  # (7,) each: tx ty tz qx qy qz qw

    def add(
        self,
        time_ns: int,
        position: np.ndarray,          # (3,)
        quaternion_xyzw: np.ndarray= None,  # (4,) optional
    ):
        """Add one keyframe to trajectory. Non-finite positions stored as-is but flagged."""
        q = np.array([0., 0., 0., 1.], dtype=np.float64)
        if quaternion_xyzw is not None:
            q = np.array(quaternion_xyzw, dtype=np.float64)
            # normalize only if finite
            nrm = np.linalg.norm(q)
            if np.isfinite(nrm) and nrm > 1e-12:
                q /= nrm
        self._times.append(int(time_ns))
        self._poses.append(np.concatenate([np.array(position, dtype=np.float64), q]))

    def __len__(self):
        return len(self._times)

    def positions(self) -> np.ndarray:
        """(N, 3) positions."""
        if not self._poses:
            return np.zeros((0, 3))
        return np.stack(self._poses)[:, :3]

    def poses(self) -> np.ndarray:
        """(N, 7) [tx ty tz qx qy qz qw]."""
        if not self._poses:
            return np.zeros((0, 7))
        return np.stack(self._poses)

    def timestamps_ns(self) -> np.ndarray:
        return np.array(self._times, dtype=np.int64)

    def ate(self, gt_positions: np.ndarray, align: bool = True) -> float:
        """ATE RMSE vs gt_positions (N,3)."""
        return compute_ate(self.positions(), gt_positions, align=align)

    def rpe(self, gt_poses: np.ndarray) -> tuple[float, float]:
        """RPE (rrel_deg, trel_m) vs gt_poses (N,7)."""
        return compute_rpe(self.poses(), gt_poses)

    def save_tum(self, path: object):
        """
        Save TUM format trajectory.
        Format: timestamp[s] tx ty tz qx qy qz qw
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            for t_ns, pose in zip(self._times, self._poses):
                t_s = t_ns * 1e-9
                line = f"{t_s:.9f} " + " ".join(f"{v:.9f}" for v in pose)
                f.write(line + "\n")

    @classmethod
    def load_tum(cls, path: object) -> 'TrajectoryAccumulator':
        """Load TUM-format trajectory file."""
        acc = cls()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                vals = list(map(float, line.split()))
                t_ns = int(vals[0] * 1e9)
                pos  = np.array(vals[1:4])
                quat = np.array(vals[4:8]) if len(vals) >= 8 else None
                acc.add(t_ns, pos, quat)
        return acc


def load_euroc_gt(
    data_root: str,
    sensor: str = 'state_groundtruth_estimate0',
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load EuRoC ground-truth from state_groundtruth_estimate0/data.csv.
    Returns (timestamps_ns (N,), poses (N,7) [tx ty tz qx qy qz qw]).
    Note: EuRoC GT stores quaternion as [qw qx qy qz] — we reorder to xyzw.
    """
    csv = Path(data_root) / sensor / 'data.csv'
    if not csv.exists():
        raise FileNotFoundError(f"GT CSV not found: {csv}")

    rows = []
    with open(csv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = list(map(float, line.split(',')))
            # EuRoC GT: ts,px,py,pz,qw,qx,qy,qz,vx,vy,vz,...
            t_ns = int(vals[0])
            tx, ty, tz = vals[1], vals[2], vals[3]
            qw, qx, qy, qz = vals[4], vals[5], vals[6], vals[7]
            rows.append([t_ns, tx, ty, tz, qx, qy, qz, qw])  # xyzw

    rows = np.array(rows, dtype=np.float64)
    ts   = rows[:, 0].astype(np.int64)
    poses = rows[:, 1:8]   # (N, 7) tx ty tz qx qy qz qw
    return ts, poses


def interpolate_gt_at(
    gt_ts: np.ndarray,     # (M,) sorted int64 ns
    gt_poses: np.ndarray,  # (M, 7)
    query_ts: np.ndarray,  # (K,) int64 ns
) -> np.ndarray:
    """
    Linear interpolation of GT poses at query timestamps.
    Returns (K, 7). Clamps queries outside GT range to boundary.
    """
    K = len(query_ts)
    out = np.zeros((K, 7), dtype=np.float64)
    for i, qt in enumerate(query_ts):
        idx = np.searchsorted(gt_ts, qt)
        if idx == 0:
            out[i] = gt_poses[0]
        elif idx >= len(gt_ts):
            out[i] = gt_poses[-1]
        else:
            t0, t1 = gt_ts[idx-1], gt_ts[idx]
            dt = float(t1 - t0)
            if dt < 1:
                out[i] = gt_poses[idx]
                continue
            alpha = float(qt - t0) / dt
            # Linear interpolation of position
            out[i, :3] = (1-alpha)*gt_poses[idx-1, :3] + alpha*gt_poses[idx, :3]
            # SLERP for quaternion (xyzw)
            q0 = gt_poses[idx-1, 3:]
            q1 = gt_poses[idx,   3:]
            dot = np.dot(q0, q1)
            if dot < 0:   # shorter arc
                q1 = -q1
                dot = -dot
            dot = min(dot, 1.0)
            if dot > 0.9995:
                q = q0 + alpha*(q1-q0)
            else:
                theta0 = np.arccos(dot)
                theta  = alpha * theta0
                q = np.sin(theta0 - theta)/np.sin(theta0)*q0 + np.sin(theta)/np.sin(theta0)*q1
            out[i, 3:] = q / (np.linalg.norm(q) + 1e-12)
    return out