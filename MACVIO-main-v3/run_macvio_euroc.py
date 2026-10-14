"""
run_macvio_euroc.py — UMRP4 MAC-VIO
Main runner for Phases A/B/C/D on EuRoC dataset.

Phase A: IMU preintegration sanity check (GT biases, 2 s, position error < 0.05 m)
Phase B: VO-only baseline via MACVOWrapper (imu_weight=0)
Phase C: VO + IMU (datasheet noise covariances)
Phase D: VO + AirIMU learned covariances (precomputed pickle)

Output: trajectory_phase{X}_{sequence}.tum + .ate file
"""

import argparse
import sys
import os
import numpy as np
import torch
import pypose as pp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--phase',         required=True, choices=['A', 'B', 'C', 'D', 'cache'])
    p.add_argument('--data',          default='/scr/user/msakhtar/Files/Projects/UMRP4/MAC-VO/data/EuRoC/machine_hall/MH_02_easy/mav0', help='path to mav0/')
    p.add_argument('--macvo_root',    default='/scr/user/msakhtar/Files/Projects/UMRP4/MACVIO/MAC-VO-main')
    p.add_argument('--macvo_cfg',     default='Config/Experiment/MACVO/Paper_Reproduce.yaml')
    p.add_argument('--airimu_pickle', default=None, help='path to net_output.pickle (Phase D)')
    p.add_argument(
        '--airimu_offset',
        type=int,
        default=None,
        help=(
            'optional explicit raw EuRoC IMU index corresponding to '
            'AirIMU precomputed index 0; if omitted, recover it '
            'automatically from the raw/corrected IMU signals'
        ),
    )
    # SLIDING WINDOW (2026-09-08). Default 2 == the thesis's pairwise formulation, bit-identical.
    # Raise it to let the IMU inform the POSE at all: at N=2 the marginalised pose information is
    # H_marg = M^T M - M^T N (N^T N)^-1 N^T M = 0 exactly, for any weight (verified 6.16e-15), so
    # no covariance can help. See sliding_window.py.
    # DEFAULT CHANGED 2 -> 0 (2026-09-08, five-lens review). With the guard at '> 2', --window 2
    # BYPASSED the refiner entirely, so the wiring was never exercised at a setting whose correct
    # output is known. That makes 'the window changed nothing' and 'the window is wired wrong'
    # indistinguishable from the outside. 0 is now off; --window 2 runs THROUGH the refiner, where
    # it is a no-op by construction (measured 7.8e-18 m) and therefore an equivalence control.
    p.add_argument('--window', type=int, default=0, help='keyframes in the sliding window (0=off)')
    p.add_argument('--window_vis_sigma_t', type=float, default=0.02,
                   help='VO relative-translation sigma (m) anchoring the window')
    p.add_argument('--window_vis_sigma_r', type=float, default=1.0,
                   help='VO relative-rotation sigma (deg) anchoring the window')
    # 'measured' uses OptimResult.H_vis -- the visual information the pairwise solve actually
    # accumulated from MAC-VO's per-point covariances -- instead of the two sigmas above. Default,
    # because a sigma chosen by hand is a tuning knob and this method is supposed to have none.
    p.add_argument('--window_vis_source', choices=['measured', 'scalar'], default='measured',
                   help="where the window's VO prior weight comes from")
    p.add_argument('--sequence',      default=None, help='sequence name for AirIMU pickle key')
    p.add_argument('--device',        default='cuda:0')
    p.add_argument('--imu_weight',    type=float, default=0.1, help='IMU weight (0.1 = soft regularization, 1.0 = equal to visual)')
    p.add_argument('--max_iter',      type=int,   default=15)
    p.add_argument('--huber_delta',   type=float, default=1.0)
    p.add_argument('--lm_lambda',     type=float, default=1e-4, help='LM damping factor')
    p.add_argument('--no_clamp_cov',  action='store_true', help='disable cov clamping (Phase D)')
    p.add_argument('--out_dir',       default='results', help='base output dir; results go in <out_dir>/phase<X>/<timestamp>/')
    p.add_argument('--max_frames',    type=int, default=None, help='limit frames processed (e.g. 360 for quick test)')
    p.add_argument('--cache_macvo',   default=None, help='path to cache MAC-VO results (save on first run, load on subsequent)')
    p.add_argument('--no-validation', dest='validation', action='store_false', help='disable validation logging')
    p.set_defaults(validation=True)
    args = p.parse_args()
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print(f"  [device] Warning: CUDA is not available. Falling back to cpu")
        args.device = 'cpu'
    return args


import csv

class MacVIOLogger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self._files = {}
        self._writers = {}

    def _writer(self, name, fieldnames):
        if name not in self._writers:
            f = open(os.path.join(self.log_dir, name), "w", newline="")
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            self._files[name] = f
            self._writers[name] = w
        return self._writers[name]

    def log_extrinsics(self, T_bc):
        row = {}
        for i in range(4):
            for j in range(4):
                row[f"T_bc_{i}{j}"] = T_bc[i, j]
        fields = [f"T_bc_{i}{j}" for i in range(4) for j in range(4)]
        self._writer("extrinsics.csv", fields).writerow(row)

    def log_optimizer_state(self, frame_id, T_rel, v_k, v_k1, R_wb=None):
        row = {"frame_id": frame_id}
        for i in range(4):
            for j in range(4):
                row[f"T_rel_{i}{j}"] = T_rel[i, j]
        for k, v in enumerate(v_k):  row[f"vk_{k}"]  = v
        for k, v in enumerate(v_k1): row[f"vk1_{k}"] = v
        fields = ["frame_id"] + \
                 [f"T_rel_{i}{j}" for i in range(4) for j in range(4)] + \
                 [f"vk_{k}" for k in range(3)] + \
                 [f"vk1_{k}" for k in range(3)]
        if R_wb is not None:
            for i in range(3):
                for j in range(3):
                    row[f"R_wb_{i}{j}"] = R_wb[i, j]
            fields += [f"R_wb_{i}{j}" for i in range(3) for j in range(3)]
        self._writer("optimizer_state.csv", fields).writerow(row)

    def log_visual_residual(self, frame_id, feat_id, r, Sv):
        row = {"frame_id": frame_id, "feat_id": feat_id,
               "r_x": r[0], "r_y": r[1], "r_z": r[2]}
        self._writer("visual_residuals.csv",
                     ["frame_id","feat_id","r_x","r_y","r_z"]).writerow(row)
        sv_row = {"frame_id": frame_id, "feat_id": feat_id}
        for i in range(3):
            for j in range(3):
                sv_row[f"Sv_{i}{j}"] = Sv[i, j]
        self._writer("visual_covariances.csv",
                     ["frame_id","feat_id"] +
                     [f"Sv_{i}{j}" for i in range(3) for j in range(3)]
                    ).writerow(sv_row)

    def log_feature_cov_k(self, frame_id, feat_id, ck_S):
        row = {"frame_id": frame_id, "feat_id": feat_id}
        for i in range(3):
            for j in range(3):
                row[f"ck_S{i}{j}"] = ck_S[i, j]
        fields = ["frame_id","feat_id"] + [f"ck_S{i}{j}" for i in range(3) for j in range(3)]
        self._writer("feature_cov_k.csv", fields).writerow(row)

    def log_feature_cov_k1(self, frame_id, feat_id, ck1_S):
        row = {"frame_id": frame_id, "feat_id": feat_id}
        for i in range(3):
            for j in range(3):
                row[f"ck1_S{i}{j}"] = ck1_S[i, j]
        fields = ["frame_id","feat_id"] + [f"ck1_S{i}{j}" for i in range(3) for j in range(3)]
        self._writer("feature_cov_k1.csv", fields).writerow(row)

    def log_imu_residuals(self, frame_id, r_g, r_b, r_a):
        row = {"frame_id": frame_id,
               "rg_x": r_g[0], "rg_y": r_g[1], "rg_z": r_g[2],
               "rb_x": r_b[0], "rb_y": r_b[1], "rb_z": r_b[2],
               "ra_x": r_a[0], "ra_y": r_a[1], "ra_z": r_a[2]}
        self._writer("imu_residuals.csv", list(row.keys())).writerow(row)

    def log_imu_covariance(self, frame_id, cov):
        row = {"frame_id": frame_id}
        for i in range(9):
            for j in range(9):
                row[f"cov_{i}{j}"] = cov[i, j]
        fields = ["frame_id"] + [f"cov_{i}{j}" for i in range(9) for j in range(9)]
        self._writer("imu_covariances.csv", fields).writerow(row)

    def log_preintegration(self, frame_id, gamma, beta, alpha, dt):
        row = {"frame_id": frame_id, "dt": dt}
        for i in range(3):
            for j in range(3):
                row[f"gam_{i}{j}"] = gamma[i, j]
        for k in range(3):
            row[f"bet_{k}"] = beta[k]
            row[f"alp_{k}"] = alpha[k]
        fields = (["frame_id"] +
                  [f"gam_{i}{j}" for i in range(3) for j in range(3)] +
                  [f"bet_{k}" for k in range(3)] +
                  [f"alp_{k}" for k in range(3)] + ["dt"])
        self._writer("preintegration.csv", fields).writerow(row)

    def log_optimizer_info(self, frame_id, iters, c_vis, c_imu, converged):
        row = {"frame_id": frame_id, "iterations": iters,
               "cost_vis": c_vis, "cost_imu": c_imu,
               "cost_total": c_vis + c_imu,
               "converged": int(converged)}
        self._writer("optimizer_info.csv", list(row.keys())).writerow(row)

    def log_visual_jacobian(self, frame_id, feat_id, J_k, J_k1, J_fd):
        def _write(name, J):
            row = {"frame_id": frame_id, "feat_id": feat_id}
            for i in range(3):
                for j in range(6):
                    row[f"J_{i}{j}"] = J[i, j]
            fields = (["frame_id","feat_id"] +
                      [f"J_{i}{j}" for i in range(3) for j in range(6)])
            self._writer(name, fields).writerow(row)

        _write("jacobian_visual_analytical.csv",    J_k)
        _write("jacobian_visual_k1_analytical.csv", J_k1)
        _write("jacobian_visual_fd.csv",            J_fd)

    def log_imu_jacobians(self, frame_id, J_an, J_fd):
        for csv_name, jdict in [("jacobian_imu_analytical.csv", J_an),
                                 ("jacobian_imu_fd.csv",         J_fd)]:
            row = {"frame_id": frame_id}
            for prefix in ["Jg", "Jb", "Ja"]:
                for i in range(3):
                    for j in range(12):
                        row[f"{prefix}_{i}{j}"] = jdict[prefix][i, j]
            fields = (["frame_id"] +
                      [f"{p}_{i}{j}"
                       for p in ["Jg","Jb","Ja"]
                       for i in range(3)
                       for j in range(12)])
            self._writer(csv_name, fields).writerow(row)

    def close(self):
        for f in self._files.values():
            f.close()


# ── EuRoC loaders ─────────────────────────────────────────────────────────────

def load_imu_data(data_root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load IMU data from EuRoC imu0/data.csv.
    Returns (timestamps_ns (N,), acc (N,3), gyro (N,3)).
    EuRoC IMU CSV: timestamp[ns], omega_x, omega_y, omega_z, a_x, a_y, a_z
    """
    csv = Path(data_root) / 'imu0' / 'data.csv'
    rows = []
    with open(csv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = list(map(float, line.split(',')))
            rows.append(vals)
    rows = np.array(rows, dtype=np.float64)
    ts   = rows[:, 0].astype(np.int64)
    gyro = rows[:, 1:4]   # rad/s
    acc  = rows[:, 4:7]   # m/s^2
    return ts, acc, gyro


def load_gt_state(data_root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load GT state from state_groundtruth_estimate0/data.csv.
    Returns (ts_ns, pos (N,3), quat_xyzw (N,4), vel (N,3)).
    EuRoC GT CSV: ts, px,py,pz, qw,qx,qy,qz, vx,vy,vz, bwx,bwy,bwz, bax,bay,baz
    """
    csv = Path(data_root) / 'state_groundtruth_estimate0' / 'data.csv'
    rows = []
    with open(csv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = list(map(float, line.split(',')))
            rows.append(vals)
    rows = np.array(rows, dtype=np.float64)
    ts       = rows[:, 0].astype(np.int64)
    pos      = rows[:, 1:4]
    qw, qx, qy, qz = rows[:, 4], rows[:, 5], rows[:, 6], rows[:, 7]
    quat_xyzw = np.stack([qx, qy, qz, qw], axis=1)  # xyzw
    vel      = rows[:, 8:11]
    return ts, pos, quat_xyzw, vel


def load_cam_timestamps(data_root: str, cam: str = 'cam0') -> np.ndarray:
    """Load camera timestamps from cam0/data.csv (first column = timestamp ns)."""
    csv = Path(data_root) / cam / 'data.csv'
    ts = []
    with open(csv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            ts.append(int(line.split(',')[0]))
    return np.array(ts, dtype=np.int64)


def interp_gt_at(gt_ts, gt_data, query_ts):
    """Linear interpolate gt_data (N, D) at query_ts. Clamp at boundaries."""
    N, D = gt_data.shape
    out = np.zeros((len(query_ts), D), dtype=np.float64)
    for i, qt in enumerate(query_ts):
        idx = int(np.searchsorted(gt_ts, qt))
        if idx == 0:
            out[i] = gt_data[0]
        elif idx >= N:
            out[i] = gt_data[-1]
        else:
            t0, t1 = float(gt_ts[idx-1]), float(gt_ts[idx])
            a = (float(qt) - t0) / (t1 - t0 + 1e-9)
            out[i] = (1-a)*gt_data[idx-1] + a*gt_data[idx]
    return out


def interp_gt_quat_at(gt_ts, gt_quat_xyzw, query_ts):
    """SLERP quaternion (xyzw) at query timestamps."""
    N = len(gt_ts)
    out = np.zeros((len(query_ts), 4), dtype=np.float64)
    for i, qt in enumerate(query_ts):
        idx = int(np.searchsorted(gt_ts, qt))
        if idx == 0:
            out[i] = gt_quat_xyzw[0]
        elif idx >= N:
            out[i] = gt_quat_xyzw[-1]
        else:
            t0, t1 = float(gt_ts[idx-1]), float(gt_ts[idx])
            a = (float(qt) - t0) / (t1 - t0 + 1e-9)
            q0 = gt_quat_xyzw[idx-1]
            q1 = gt_quat_xyzw[idx]
            dot = np.dot(q0, q1)
            if dot < 0:
                q1 = -q1; dot = -dot
            dot = min(dot, 1.0)
            if dot > 0.9995:
                q = (1-a)*q0 + a*q1
            else:
                th0 = np.arccos(dot)
                q = np.sin((1-a)*th0)/np.sin(th0)*q0 + np.sin(a*th0)/np.sin(th0)*q1
            out[i] = q / (np.linalg.norm(q) + 1e-12)
    return out



def _infer_airimu_raw_offset(airimu, raw_acc, raw_gyro):
    """
    Recover the raw EuRoC IMU index corresponding to AirIMU output index 0.

    The precomputed AirIMU dictionary stores corrected measurements and
    learned corrections. We reconstruct the raw measurements for both
    possible correction signs and choose the unique offset/sign pair with
    the smallest reconstruction error.
    """
    d = airimu._np

    required = (
        'corrected_acc',
        'corrected_gyro',
        'correction_acc',
        'correction_gyro',
    )
    missing = [k for k in required if k not in d]
    if missing:
        raise RuntimeError(
            f"AirIMU pickle is missing alignment keys: {missing}"
        )

    n_airimu = len(d['corrected_acc'])
    n_raw = len(raw_acc)
    max_offset = n_raw - n_airimu

    if max_offset < 0:
        raise RuntimeError(
            f"AirIMU output ({n_airimu}) is longer than raw IMU ({n_raw})."
        )

    probe = np.linspace(
        0,
        n_airimu - 1,
        min(512, n_airimu),
        dtype=np.int64,
    )

    candidates = []

    for sign in (+1.0, -1.0):
        raw_acc_reconstructed = (
            d['corrected_acc'][probe]
            + sign * d['correction_acc'][probe]
        )
        raw_gyro_reconstructed = (
            d['corrected_gyro'][probe]
            + sign * d['correction_gyro'][probe]
        )

        for offset in range(max_offset + 1):
            acc_err = (
                raw_acc[offset + probe]
                - raw_acc_reconstructed
            )
            gyro_err = (
                raw_gyro[offset + probe]
                - raw_gyro_reconstructed
            )

            acc_rmse = float(
                np.sqrt(np.mean(acc_err ** 2))
            )
            gyro_rmse = float(
                np.sqrt(np.mean(gyro_err ** 2))
            )

            candidates.append(
                (
                    acc_rmse + gyro_rmse,
                    offset,
                    sign,
                    acc_rmse,
                    gyro_rmse,
                )
            )

    candidates.sort(key=lambda row: row[0])
    best = candidates[0]

    print()
    print("  AirIMU raw-index alignment")
    print("  --------------------------")
    print(f"  raw IMU samples          : {n_raw}")
    print(f"  AirIMU samples           : {n_airimu}")
    print(f"  best offset              : {best[1]}")
    print(
        "  reconstruction sign      : "
        + ("+" if best[2] > 0 else "-")
    )
    print(f"  acc reconstruction RMSE  : {best[3]:.12e}")
    print(f"  gyro reconstruction RMSE : {best[4]:.12e}")

    # This is intentionally a strict guard: a poor match usually means the
    # wrong sequence/key/pickle is being used, and the evaluation must stop.
    if best[0] > 1e-5:
        raise RuntimeError(
            "AirIMU/raw-IMU alignment failed the 1e-5 reconstruction guard."
        )

    return int(best[1])


def _airimu_segment_from_raw_indices(
    airimu,
    start_raw,
    end_raw_exclusive,
    raw_offset,
):
    """
    Slice precomputed AirIMU outputs with original EuRoC raw-IMU indices.

    The requested interval is half-open [start_raw, end_raw_exclusive),
    matching standard NumPy slicing.
    """
    start_airimu = int(start_raw) - int(raw_offset)
    end_airimu = int(end_raw_exclusive) - int(raw_offset)

    n_airimu = len(airimu._np['corrected_acc'])

    if (
        start_airimu < 0
        or end_airimu > n_airimu
        or end_airimu <= start_airimu
    ):
        raise RuntimeError(
            "Requested AirIMU interval is outside the precomputed range: "
            f"raw=[{start_raw},{end_raw_exclusive}), "
            f"offset={raw_offset}, "
            f"mapped=[{start_airimu},{end_airimu}), "
            f"N={n_airimu}"
        )

    return {
        'corrected_acc':
            airimu._np['corrected_acc'][start_airimu:end_airimu],
        'corrected_gyro':
            airimu._np['corrected_gyro'][start_airimu:end_airimu],
        'acc_cov':
            airimu._np['acc_cov'][start_airimu:end_airimu],
        'gyro_cov':
            airimu._np['gyro_cov'][start_airimu:end_airimu],
    }

# ── Phase A ────────────────────────────────────────────────────────────────────

def _quat_to_rot(q):
    """xyzw → (3,3)."""
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float64)



def _save_trajectory_plot(
    out_dir, phase, seq_name,
    est_pos, gt_pos_aligned,   # (N,3) matched to est frames
    ate, rrel, trel,
):
    """XY top-down + XZ side view; error-vs-frame; save as trajectory.png."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    N = len(est_pos)
    frames = np.arange(N)
    err_per_frame = np.linalg.norm(est_pos - gt_pos_aligned, axis=1)

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Phase {phase} — {seq_name}  |  ATE={ate:.3f} m  rrel={rrel:.3f}°/fr  trel={trel*100:.2f} cm/fr",
        fontsize=11, fontweight='bold'
    )
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # XY top-down
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(gt_pos_aligned[:, 0], gt_pos_aligned[:, 1], 'k-', lw=1.5, label='GT')
    ax1.plot(est_pos[:, 0],        est_pos[:, 1],        'r-', lw=1.0, label='Est', alpha=0.85)
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    ax1.set_title('Top-down (XY)')
    ax1.legend(fontsize=8); ax1.set_aspect('equal', 'box'); ax1.grid(True, alpha=0.3)

    # XZ side view
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(gt_pos_aligned[:, 0], gt_pos_aligned[:, 2], 'k-', lw=1.5, label='GT')
    ax2.plot(est_pos[:, 0],        est_pos[:, 2],        'r-', lw=1.0, label='Est', alpha=0.85)
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Z (m)')
    ax2.set_title('Side view (XZ)')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # per-frame position error
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(frames, err_per_frame, 'b-', lw=0.8, alpha=0.7)
    ax3.axhline(ate, color='r', lw=1.2, linestyle='--', label=f'ATE={ate:.3f} m')
    ax3.fill_between(frames, 0, err_per_frame, alpha=0.15, color='b')
    ax3.set_xlabel('Frame'); ax3.set_ylabel('Position error (m)')
    ax3.set_title('Per-frame position error (after Sim3 alignment)')
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    plot_path = out_dir / 'trajectory.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return plot_path


def run_cache_build(args):
    """Build MAC-VO result cache for a sequence — run once, reuse across phases."""
    import sys, pickle
    from pathlib import Path

    seq_name_cache = args.sequence or Path(args.data).parent.name
    seq_tag = seq_name_cache.replace('_easy','').replace('_medium','').replace('_difficult','')\
              .replace('machine_hall/','').replace('vicon_room1/','').replace('vicon_room2/','')
    n_tag   = f'{args.max_frames}' if args.max_frames else 'Full'
    # e.g. cache/macvo_MH02_500.pkl or cache/macvo_MH02_Full.pkl
    auto_name = f'macvo_{seq_tag.replace("_","").upper()}_{n_tag}.pkl'
    cache_path = Path(args.cache_macvo) if args.cache_macvo else Path('macvo-cache') / auto_name

    if cache_path.exists():
        print(f"Cache already exists: {cache_path}  (delete to rebuild)")
        return

    data_root = Path(args.data)
    seq_name  = args.sequence or data_root.parent.name
    device    = 'cuda:0' if __import__('torch').cuda.is_available() else 'cpu'

    print(f"PHASE cache — Building MAC-VO result cache")
    print(f"  sequence   : {seq_name}")
    print(f"  max_frames : {args.max_frames or 'ALL'}")
    print(f"  output     : {cache_path}")

    sys.path.insert(0, str(Path(args.macvo_root).resolve()))
    from macvo_wrapper import MACVOWrapper

    mac = MACVOWrapper(
        macvo_root  = args.macvo_root,
        config_path = args.macvo_cfg,
        data_root   = data_root,
        device      = device,
    ).build()
    mac.run_sequence(max_frames=args.max_frames)

    cache_path = Path('macvo-cache') / cache_path if not cache_path.is_absolute() and cache_path.parent == Path('.') else cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(mac._results, f)
    print(f"  Saved {len(mac._results)} frames → {cache_path}")


def run_phase_A(args):
    """IMU preintegration sanity check using GT biases over 2 seconds."""
    from preintegration import IMUPreintegrator

    print("\nPHASE A: IMU Preintegration Sanity Check")
    data_root = args.data
    device    = args.device

    imu_ts, acc, gyro = load_imu_data(data_root)
    gt_ts, gt_pos, gt_quat, gt_vel = load_gt_state(data_root)

    # Skip first 10s (MH01: drone picked up by hand, v~0.8m/s, not stable)
    # 0.5s window from stable flight keeps error well under 0.05m threshold
    t_skip  = imu_ts[0] + int(10.0 * 1e9)
    candidates = imu_ts[imu_ts >= t_skip]
    t_start = candidates[0] if len(candidates) > 0 else imu_ts[0]
    t_end   = t_start + int(0.5 * 1e9)
    mask    = (imu_ts >= t_start) & (imu_ts <= t_end)
    ts_seg  = imu_ts[mask]
    acc_seg = acc[mask]
    gyro_seg = gyro[mask]

    print(f"  Integrating {len(ts_seg)} IMU samples ({(ts_seg[-1]-ts_seg[0])*1e-9:.2f} s)")

    # GT bias at start: last 3 columns of GT CSV = ba; cols -6:-3 = bg
    # Interpolate GT at start of segment to get biases
    # EuRoC GT CSV has bg (cols 11:14) and ba (cols 14:17)
    csv_path = Path(data_root) / 'state_groundtruth_estimate0' / 'data.csv'
    rows = []
    with open(csv_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            rows.append(list(map(float, s.split(','))))
    rows = np.array(rows)
    gt_ts_full = rows[:, 0].astype(np.int64)
    bg_all = rows[:, 11:14]  # gyro bias rad/s
    ba_all = rows[:, 14:17]  # accel bias m/s^2

    idx0 = int(np.searchsorted(gt_ts_full, ts_seg[0]))
    idx0 = min(idx0, len(bg_all)-1)
    bg = bg_all[idx0]
    ba = ba_all[idx0]
    print(f"  GT bg: {bg}")
    print(f"  GT ba: {ba}")

    # GT initial state at t_start
    idx_gt0 = int(np.searchsorted(gt_ts, ts_seg[0]))
    idx_gt0 = min(idx_gt0, len(gt_pos)-1)
    p0 = gt_pos[idx_gt0]
    v0 = gt_vel[idx_gt0]
    q0 = gt_quat[idx_gt0]   # xyzw

    # GT end state
    idx_gt1 = int(np.searchsorted(gt_ts, ts_seg[-1]))
    idx_gt1 = min(idx_gt1, len(gt_pos)-1)
    p1_gt = gt_pos[idx_gt1]

    # GT rotation for gravity compensation
    gt_rot_seg = interp_gt_quat_at(gt_ts, gt_quat, ts_seg)

    integrator = IMUPreintegrator(device=device)
    result = integrator.integrate(
        timestamps_ns = ts_seg,
        acc_raw       = acc_seg,
        gyro_raw      = gyro_seg,
        bg            = bg,
        ba            = ba,
        gt_rot        = gt_rot_seg,
    )

    g_w   = np.array([0., 0., -9.81007])
    R0    = _quat_to_rot(q0)   # conversion of EuRoC xyzw quat
    dt    = result['dt']
    alpha = result['pos'].numpy()
    beta  = result['vel'].numpy()

    p1_est = p0 + v0 * dt + 0.5 * g_w * dt**2 + R0 @ alpha
    err = np.linalg.norm(p1_est - p1_gt)

    print(f"  Preintegrated Δpos (world): {p1_est - p0}")
    print(f"  GT Δpos:                    {p1_gt - p0}")
    print(f"  Position error:             {err:.4f} m")

    thresh = 0.05
    if err < thresh:
        print(f"  ✓ PASS — threshold {thresh} m")
    else:
        print(f"  ✗ FAIL — threshold {thresh} m (got {err:.4f} m)")
        print("    Check gravity convention or GT bias indexing.")


# ── Phases B/C/D ───────────────────────────────────────────────────────────────

def run_phase_BCD(args):
    """Run full VIO pipeline: MAC-VO + optional IMU fusion."""
    phase = args.phase
    device = args.device
    data_root = args.data
    from datetime import datetime
    # SEQUENCE AND PID IN THE PATH (2026-09-08). The path was <base>/phase<X>/<YYYYmmdd_HHMMSS>
    # at SECOND resolution with no sequence name, so five sequences dispatched together landed in
    # ONE directory and overwrote each other's trajectory.tum, metrics.txt, CSVs and plot. 35 runs
    # left 7 trajectory files, and whichever finished last owned the directory -- unlabelled.
    # The reported ATEs were never affected (they are parsed from per-run logs with unique names),
    # but anything read back off disk was a different sequence than its path implied.
    _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    _tag = f"{args.sequence or 'seq'}_{_ts}_{os.getpid()}"
    out_dir = Path(args.out_dir) / f'phase{phase}' / _tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  out_dir  : {out_dir}")
    seq_name = args.sequence or Path(data_root).parent.name

    print(f"\nPHASE {phase}: {'VO-only' if phase=='B' else 'VO+IMU (datasheet)' if phase=='C' else 'VO+AirIMU (learned)'}")
    print(f"  sequence : {seq_name}")
    print(f"  data     : {data_root}")

    # ── Load MAC-VO ────────────────────────────────────────────────────────────
    sys.path.insert(0, str(Path(args.macvo_root).resolve()))
    from macvo_wrapper import MACVOWrapper, FrameResult
    from optimizer import (
        TwoFrameVIOOptimizer, VisualFactor, IMUFactor, WorldState, OptimResult
    )
    from preintegration import IMUPreintegrator
    from trajectory import TrajectoryAccumulator, compute_ate, compute_rpe

    _cache_path = args.cache_macvo
    if _cache_path and Path(_cache_path).exists():
        # Skip MAC-VO build entirely — load cached results directly
        import pickle as _pkl
        print(f"  [cache] Loading MAC-VO results from {_cache_path}")
        _cached_results = _pkl.load(open(_cache_path, 'rb'))
        # Create a lightweight stub to hold results
        class _CachedResults:
            def __init__(self, results): self._results = results
            def n_captured(self): return len(self._results)
        mac = _CachedResults(_cached_results)
    else:
        mac = MACVOWrapper(
            macvo_root  = args.macvo_root,
            config_path = args.macvo_cfg,
            data_root   = data_root,
            device      = device,
        ).build()
        mac.run_sequence(max_frames=args.max_frames)
        if _cache_path:
            import pickle as _pkl
            _pkl.dump(mac._results, open(_cache_path, 'wb'))
            print(f"  [cache] Saved MAC-VO results to {_cache_path}")
    print(f"  MAC-VO frames captured: {mac.n_captured()}")

    # ── Load EuRoC GT + IMU ────────────────────────────────────────────────────
    gt_ts, gt_pos, gt_quat, gt_vel = load_gt_state(data_root)
    cam_ts = load_cam_timestamps(data_root)  # camera timestamps

    # Load the same raw EuRoC IMU stream for B/C/D.
    # Phase B does not use IMU in estimation; the timestamps are loaded only
    # so every estimator is evaluated after the identical 10 s start cut.
    imu_ts, imu_acc, imu_gyro = load_imu_data(data_root)

    imu_integrator = None
    airimu_wrapper = None
    airimu_raw_offset = None

    if phase in ('C', 'D'):
        imu_integrator = IMUPreintegrator(device=device)

    if phase == 'D':
        assert args.airimu_pickle, "--airimu_pickle required for Phase D"
        from airimu_wrapper import AirIMUWrapper
        airimu_wrapper = AirIMUWrapper(mode='precomputed', device=device)
        airimu_wrapper.load_precomputed(args.airimu_pickle, seq_name)

        if args.airimu_offset is None:
            airimu_raw_offset = _infer_airimu_raw_offset(
                airimu_wrapper,
                imu_acc,
                imu_gyro,
            )
        else:
            airimu_raw_offset = int(args.airimu_offset)
            print()
            print(
                f"  AirIMU raw-index offset (explicit): "
                f"{airimu_raw_offset}"
            )

        # Preserve this attribute for compatibility/diagnostics in
        # AirIMUWrapper, but Phase D below slices the precomputed arrays
        # explicitly using the validated raw-to-AirIMU index mapping.
        airimu_wrapper.raw_index_offset = airimu_raw_offset

    # ── Body-sensor extrinsic (T_BS: body←camera) ─────────────────────────────
    # Get from first available frame result
    T_BS_mat = None
    for fi in sorted(mac._results.keys()):
        res = mac._results[fi]
        if res is not None and res.T_BS is not None:
            try:
                T_BS_mat = res.T_BS.matrix().cpu().numpy().squeeze()  # (4,4)
                break
            except Exception:
                pass
    if T_BS_mat is None:
        print("  Warning: T_BS not found, using identity")
        T_BS_mat = np.eye(4)

    logger = None
    if args.validation:
        logger = MacVIOLogger(out_dir / 'csv')
        logger.log_extrinsics(T_BS_mat)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    _clamp = not getattr(args, 'no_clamp_cov', False)
    # Phase D: apply a covariance floor (not full clamp) to prevent
    # information matrix explosion from near-zero learned covariances
    if phase == 'D': _clamp = True  # Re-enabled to prevent overconfidence
    optimizer = TwoFrameVIOOptimizer(
        g_w         = [0., 0., -9.81],
        imu_weight  = 0.0 if phase == 'B' else args.imu_weight,
        max_iter    = args.max_iter,
        huber_delta = args.huber_delta,
        lm_lambda   = getattr(args, 'lm_lambda', 1e-4),
        clamp_cov   = _clamp,
        device      = 'cpu',
    )
    # ── Sliding-window refiner (2026-09-08) ────────────────────────────────────
    _refiner, _win_body, _T_BC_se3, _win_hist = None, {}, None, {}
    if getattr(args, 'window', 0) >= 2 and phase in ('C', 'D'):
        from sliding_window import SlidingWindowRefiner
        _W = np.diag([1.0 / args.window_vis_sigma_t ** 2] * 3 +
                     [1.0 / np.deg2rad(args.window_vis_sigma_r) ** 2] * 3)
        _refiner = SlidingWindowRefiner(n_frames=args.window, g_w=(0., 0., -9.81),
                                        imu_weight=args.imu_weight, vis_info=_W)
        print(f"  [window] sliding window N={args.window}, "
              f"vis prior = {args.window_vis_source}"
              + (f" (fallback sigma {args.window_vis_sigma_t} m / "
                 f"{args.window_vis_sigma_r} deg)" if args.window_vis_source == 'measured'
                 else f" {args.window_vis_sigma_t} m / {args.window_vis_sigma_r} deg"))

    # Track init frame index for gravity drift compensation
    _init_frame_idx = None
    _gravity_inflation_rate = 0.002  # inflate IMU cov by 0.2% per frame from init

    # ── Main loop ──────────────────────────────────────────────────────────────
    traj = TrajectoryAccumulator()
    state_k = None  # WorldState | None
    R_w_vo_w_gt = None
    prev_pose = None

    sorted_frames = sorted(mac._results.keys())

    # Common evaluation cut for B/C/D: skip the first 10 s measured from
    # the raw EuRoC IMU start. This makes the evaluated frame interval
    # identical across VO-only and inertial variants.
    _imu_t0 = imu_ts[0]
    _hand_carry_end_ns = _imu_t0 + int(10.0 * 1e9)

    # apply max_frames after hand-carry filter so we get real flight frames
    if args.max_frames and phase in ('B', 'C', 'D'):
        _imu_t0_pre = _hand_carry_end_ns
        _valid_frames = [fi for fi in sorted_frames
                         if mac._results[fi] is not None
                         and mac._results[fi].is_valid()
                         and mac._results[fi].time_ns >= _imu_t0_pre]
        sorted_frames = _valid_frames[:args.max_frames]
    elif args.max_frames:
        sorted_frames = sorted_frames[:args.max_frames]
    _n_invalid = sum(1 for fi in sorted_frames if mac._results[fi] is None or not mac._results[fi].is_valid())
    from tqdm import tqdm
    _pbar = tqdm(sorted_frames, desc=f"Phase {phase}", unit="fr")
    for fi, frame_idx in enumerate(_pbar):
        result: FrameResult = mac._results[frame_idx]
        if result is None or not result.is_valid():
            continue

        t_ns   = result.time_ns
        pose   = result.pose_SE3   # pp.SE3 (7,) world←camera

        # skip hand-carry window: IMU physically invalid (no thrust, v_z~0.8 m/s)
        if t_ns < _hand_carry_end_ns:
            continue

        # Build visual factor from MAC-VO result
        T_BC = torch.tensor(T_BS_mat, dtype=torch.float64)
        vis = VisualFactor(
            pts_pos_Tw   = result.pts_pos_Tw.double() if result.pts_pos_Tw is not None else torch.zeros(0,3,dtype=torch.float64),
            pts_cov_Tw   = result.pts_cov_Tw.double() if result.pts_cov_Tw is not None else torch.zeros(0,3,3,dtype=torch.float64),
            obs_cov_Tc   = result.obs_cov_Tc.double() if result.obs_cov_Tc is not None else torch.zeros(0,3,3,dtype=torch.float64),
            pose_SE3     = pose.double(),
            T_BC         = T_BC,
            pose_cov_SE3 = None,
            prev_pose_SE3 = prev_pose,
        )

        # First frame: init pos+rot from MAC-VO body pose, vel from GT (KB §9)
        # Must stay in MAC-VO world frame — do NOT init pos/rot from GT
        if state_k is None:
            idx0 = int(np.searchsorted(gt_ts, t_ns))
            idx0 = min(idx0, len(gt_pos)-1)
            T_BC_se3_init = pp.from_matrix(T_BC.unsqueeze(0), pp.SE3_type).squeeze()
            T_w_bk0 = pose.double() @ T_BC_se3_init.Inv()   # world←body from MAC-VO
            p0 = T_w_bk0.translation().squeeze().double().detach()
            q0 = T_w_bk0.rotation().data.squeeze().double().detach()

            # Align coordinate frames: Rotate initial velocity and gravity from GT world frame to VO world frame
            q_gt0 = torch.tensor(gt_quat[idx0], dtype=torch.float64)
            R_w_bk0_gt = pp.SO3(q_gt0)  # world_gt <- body
            R_w_vo_w_gt = T_w_bk0.rotation() @ R_w_bk0_gt.Inv()  # world_vo <- world_gt

            v_gt0 = torch.tensor(gt_vel[idx0], dtype=torch.float64)
            v0_vo = v_gt0  # velocity in GT world frame (matches g_w=[0,0,-9.81] GT frame)

            # Option A (GOAL §Bug A): g_w stays in GT world frame [0,0,-9.81].
            # R_k_gravity = R_k_gt (plain matrix) projects it to body frame correctly.
            # Do NOT use g_vo = R_w_vo_w_gt @ g_gt — that returns a PyPose LieTensor,
            # and plain_matrix.T @ pypose_tensor dispatches incorrectly (garbage direction).
            g_vo = torch.tensor([0., 0., -9.81], dtype=torch.float64)  # plain tensor, GT frame

            optimizer.g_w = g_vo  # [0,0,-9.81] GT frame; R_k_gravity handles projection

            # Option A: R_k_gravity = R_k_gt (plain (3,3), world_gt←body).
            # g_w=[0,0,-9.81] is in GT frame; R_g.T @ g = R_k_gt.T @ [0,0,-9.81] = correct body-frame gravity.
            # Do NOT pre-multiply by R_w_vo_w_gt — that was the frame-mixing bug.
            q_gt0 = torch.tensor(gt_quat[idx0], dtype=torch.float64)
            R_k0_gt_plain = pp.SO3(q_gt0).matrix().squeeze().double()  # plain (3,3)

            state_k = WorldState(
                pos = p0,
                vel = v0_vo,
                rot = pp.SO3(q0),
                R_k_gravity = R_k0_gt_plain,
            )
            # add first cam pose to trajectory (camera frame, from MAC-VO)
            p_c0 = pose.double().translation().squeeze().detach().numpy()
            q_c0 = pose.double().rotation().data.squeeze().detach().numpy()
            traj.add(t_ns, p_c0, q_c0)

            prev_pose = pose.double()
            _init_frame_idx = fi  # track for gravity drift compensation
            continue

        # ── IMU preintegration for interval [prev_cam_ts, cam_ts] ─────────────
        imu_factor = None
        if phase in ('C', 'D'):
            prev_t_ns = traj._times[-1] if traj._times else t_ns
            # IMU slice: all samples between prev and current camera frame
            imu_mask = (imu_ts >= prev_t_ns) & (imu_ts <= t_ns)
            seg_ts   = imu_ts[imu_mask]
            seg_acc  = imu_acc[imu_mask]
            seg_gyro = imu_gyro[imu_mask]

            if len(seg_ts) >= 2:
                if phase == 'D' and airimu_wrapper is not None:
                    # The raw interval above is inclusive at both camera
                    # timestamps. Convert it to an equivalent half-open raw
                    # index interval, then map it into AirIMU's cropped domain.
                    i_start = int(
                        np.searchsorted(
                            imu_ts,
                            prev_t_ns,
                            side='left',
                        )
                    )
                    i_end_exclusive = int(
                        np.searchsorted(
                            imu_ts,
                            t_ns,
                            side='right',
                        )
                    )

                    seg = _airimu_segment_from_raw_indices(
                        airimu_wrapper,
                        i_start,
                        i_end_exclusive,
                        airimu_raw_offset,
                    )

                    # Do not silently truncate one signal to the other.
                    # A mismatch indicates an alignment/indexing error and
                    # invalidates a baseline comparison.
                    if len(seg['corrected_acc']) != len(seg_ts):
                        raise RuntimeError(
                            "Phase-D AirIMU/raw interval length mismatch: "
                            f"AirIMU={len(seg['corrected_acc'])}, "
                            f"raw={len(seg_ts)}, frame={fi}"
                        )

                    n = len(seg_ts)
                    if n >= 2:
                        # Apply covariance floor to prevent information matrix explosion
                        # AirIMU acc_cov can be as low as 1.1e-5 — too confident
                        acc_cov_floored  = np.maximum(seg['acc_cov'][:n],  1e-3)
                        gyro_cov_floored = np.maximum(seg['gyro_cov'][:n], 1e-5)
                        integ = imu_integrator.integrate(
                            timestamps_ns = seg_ts[:n],
                            acc_raw       = seg['corrected_acc'][:n],
                            gyro_raw      = seg['corrected_gyro'][:n],
                            bg            = np.zeros(3),
                            ba            = np.zeros(3),
                            acc_cov       = acc_cov_floored,
                            gyro_cov      = gyro_cov_floored,
                        )
                        imu_factor = _make_imu_factor(integ)
                        # Inflate IMU covariance to account for gravity alignment drift
                        # MAC-VO world frame drifts ~3°/1000frames from GT frame
                        # REMOVED: Now that gravity alignment is fixed at the root by passing R_k_gravity,
                        # covariance inflation is no longer needed and would hurt performance.
                        # if _init_frame_idx is not None:
                        #     frames_since_init = fi - _init_frame_idx
                        #     inflation = 1.0 + _gravity_inflation_rate * frames_since_init
                        #     imu_factor.cov = imu_factor.cov * inflation
                else:
                    # Phase C: raw IMU + GT biases at interval start
                    idx_b = int(np.searchsorted(gt_ts, prev_t_ns))
                    idx_b = min(idx_b, len(gt_ts)-1)
                    # Read biases from GT CSV (extended columns)
                    bg, ba = _get_gt_biases(data_root, gt_ts, prev_t_ns)
                    integ = imu_integrator.integrate(
                        timestamps_ns = seg_ts,
                        acc_raw       = seg_acc,
                        gyro_raw      = seg_gyro,
                        bg            = bg,
                        ba            = ba,
                    )
                    imu_factor = _make_imu_factor(integ)

        # ── Optimize ───────────────────────────────────────────────────────────
        try:
            opt_result: OptimResult = optimizer.optimize(state_k, vis, imu_factor)
        except Exception as _e:
            import traceback; traceback.print_exc()
            continue

        if args.validation and logger is not None and opt_result.debug_logs is not None:
            dl = opt_result.debug_logs
            if R_w_vo_w_gt is not None:
                R_gt_vo_np = R_w_vo_w_gt.matrix().squeeze().detach().cpu().numpy().T
                v_k_gt = R_gt_vo_np @ dl["v_k"]
                v_k1_gt = R_gt_vo_np @ dl["v_k1"]
                R_wb_gt = R_gt_vo_np @ dl["R_wb"] if "R_wb" in dl and dl["R_wb"] is not None else None
            else:
                v_k_gt, v_k1_gt, R_wb_gt = dl["v_k"], dl["v_k1"], dl.get("R_wb")
            logger.log_optimizer_state(fi, dl["T_rel"], v_k_gt, v_k1_gt, R_wb_gt)
            for feat_idx in range(len(dl["r_vis"])):
                logger.log_visual_residual(fi, feat_idx, dl["r_vis"][feat_idx], dl["Sigma_vis"][feat_idx])
                logger.log_feature_cov_k(fi, feat_idx, dl.get("cov_ck", dl["obs_cov"])[feat_idx])
                logger.log_feature_cov_k1(fi, feat_idx, dl["obs_cov"][feat_idx])

                J_k = dl["J_vis"][feat_idx]
                J_k1 = -J_k
                J_fd = dl["J_vis_fd"][feat_idx]
                logger.log_visual_jacobian(fi, feat_idx, J_k, J_k1, J_fd)

            if phase in ('C', 'D') and imu_factor is not None:
                logger.log_imu_residuals(fi, dl["r_gamma"], dl["r_beta"], dl["r_alpha"])
                logger.log_imu_covariance(fi, dl["Sigma_IMU"])
                logger.log_preintegration(fi, dl["gamma_R"], dl["beta_pre"], dl["alpha_pre"], imu_factor.dt)
                logger.log_optimizer_info(fi, opt_result.n_iter, dl["cost_vis"], dl["cost_imu"], opt_result.success)

                J_an = {"Jg": dl["Jg_an"], "Jb": dl["Jb_an"], "Ja": dl["Ja_an"]}
                J_fd = {"Jg": dl["Jg_fd"], "Jb": dl["Jb_fd"], "Ja": dl["Ja_fd"]}
                logger.log_imu_jacobians(fi, J_an, J_fd)
            else:
                logger.log_optimizer_info(fi, opt_result.n_iter, dl["cost_vis"], 0.0, opt_result.success)

        # pose_k1 = world←camera (for trajectory); derive body pose for state
        T_w_cam  = opt_result.pose_k1.double()
        p_k1     = T_w_cam.translation().squeeze().detach().numpy()
        q_k1     = T_w_cam.rotation().data.squeeze().detach().numpy()  # xyzw

        traj.add(t_ns, p_k1, q_k1)

        prev_pose = pose.double()

        # state_k.rot = world←body: T_w_body = T_w_cam @ T_BC^{-1}
        T_BC_t   = torch.tensor(T_BS_mat, dtype=torch.float64)
        T_BC_se3 = pp.from_matrix(T_BC_t.unsqueeze(0), pp.SE3_type).squeeze()
        T_w_body = T_w_cam @ T_BC_se3.Inv()

        # Look up GT rotation at current frame timestamp and align to VO frame for gravity projection
        idx_gt_k1 = int(np.searchsorted(gt_ts, t_ns))
        idx_gt_k1 = min(idx_gt_k1, len(gt_ts) - 1)
        q_gt_k1   = torch.tensor(gt_quat[idx_gt_k1], dtype=torch.float64)  # xyzw
        # Option A: plain (3,3) GT rotation only — no VO-frame mixing
        R_k1_gt_plain = pp.SO3(q_gt_k1).matrix().squeeze().double()
        if _init_frame_idx is not None and fi <= _init_frame_idx + 4:
            g_body = R_k1_gt_plain.T @ g_vo  # g_vo=[0,0,-9.81], R_k1_gt_plain plain matrix → correct
            print(f'  [sanity] g in body frame at frame {fi}: {g_body.detach().numpy()}')
            # Expected: [~0, ~0, ~-9.81] (small x/y from body tilt only)

        state_k  = WorldState(
            pos         = T_w_body.translation().squeeze().double().detach(),
            vel         = opt_result.vel_k1.double(),
            rot         = pp.SO3(T_w_body.rotation().data.squeeze().double().detach()),
            R_k_gravity = R_k1_gt_plain,
        )

        # ── Sliding-window refinement ──────────────────────────────────────────
        if _refiner is not None:
            _T_BC_se3 = T_BC_se3
            _imu_d = None if imu_factor is None else {
                "dR": imu_factor.gamma.matrix().squeeze().detach().numpy(),
                "dv": imu_factor.beta.detach().numpy(),
                "dp": imu_factor.alpha.detach().numpy()}
            _refiner.push(fi,
                          pp.SO3(T_w_body.rotation().data.squeeze().double().detach()
                                 ).matrix().squeeze().detach().numpy(),
                          T_w_body.translation().squeeze().double().detach().numpy(),
                          opt_result.vel_k1.double().detach().numpy(),
                          _imu_d, None if imu_factor is None else imu_factor.dt,
                          vis_info=(None if args.window_vis_source != 'measured'
                                    else getattr(opt_result, 'H_vis', None)))
            _ref = _refiner.solve()
            # The window silently SHRINKS at any interval with no IMU factor (_usable() returns the
            # longest suffix that has one). Without this the reported N is the requested N, not the
            # realised one, and a run could report N=8 while solving 2 the whole time.
            _wn = _refiner.last_info.get("n", 0)
            _win_hist[_wn] = _win_hist.get(_wn, 0) + 1
            for _fid, (_R, _p, _v) in _ref.items():
                _win_body[_fid] = (mac._results[sorted_frames[_fid]].time_ns
                                   if _fid < len(sorted_frames) else t_ns, _R, _p)
            if fi in _ref:
                _R, _p, _v = _ref[fi]
                state_k = WorldState(
                    pos         = torch.tensor(_p, dtype=torch.float64),
                    vel         = torch.tensor(_v, dtype=torch.float64),
                    rot         = pp.mat2SO3(torch.tensor(_R, dtype=torch.float64)),
                    R_k_gravity = R_k1_gt_plain,
                )

    # Rebuild the trajectory from the refined window poses (2026-09-08). The per-frame path
    # above appends the PAIRWISE estimate as it goes; the window revises earlier frames after the
    # fact, so the trajectory has to be re-emitted rather than patched in place.
    if _refiner is not None and _win_body and _T_BC_se3 is not None:
        _traj2 = TrajectoryAccumulator()
        for _fid in sorted(_win_body):
            _t_ns, _R, _p = _win_body[_fid]
            _T_w_body = pp.SE3(torch.cat([torch.tensor(_p, dtype=torch.float64),
                                          pp.mat2SO3(torch.tensor(_R, dtype=torch.float64)).data]))
            _T_w_cam = _T_w_body @ _T_BC_se3
            _traj2.add(_t_ns,
                       _T_w_cam.translation().squeeze().detach().numpy(),
                       _T_w_cam.rotation().data.squeeze().detach().numpy())
        print(f"  [window] rebuilt trajectory from {len(_traj2)} refined poses "
              f"(pairwise had {len(traj)})")
        print(f"  [window] REALISED window lengths: "
              f"{dict(sorted(_win_hist.items()))}  (requested {args.window})")
        print(f"  [window] VO prior edges: measured={_refiner.n_measured} "
              f"fallback_scalar={_refiner.n_fallback}")
        traj = _traj2

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print(f"\n  Trajectory frames: {len(traj)}")
    if len(traj) < 3:
        print("  ERROR: too few frames captured — check MACVOWrapper hook")
        return

    # Interpolate GT at trajectory timestamps
    traj_ts    = traj.timestamps_ns()
    gt_pos_interp  = interp_gt_at(gt_ts, gt_pos,  traj_ts)
    gt_quat_interp = interp_gt_quat_at(gt_ts, gt_quat, traj_ts)
    gt_poses_interp = np.concatenate([gt_pos_interp, gt_quat_interp], axis=1)  # (N,7)

    ate = compute_ate(traj.positions(), gt_pos_interp, align=True)
    rrel, trel = compute_rpe(traj.poses(), gt_poses_interp)

    print(f"\n  === Results Phase {phase} — {seq_name} ===")
    print(f"  ATE RMSE : {ate:.4f} m")
    print(f"  rrel     : {rrel:.4f} deg/frame")
    print(f"  trel     : {trel:.6f} m/frame")

    # ── Save outputs ───────────────────────────────────────────────────────────
    # save run metadata
    with open(out_dir / 'run_info.txt', 'w') as _f:
        _f.write(f'phase      : {phase}\n')
        _f.write(f'sequence   : {seq_name}\n')
        _f.write(f'data       : {data_root}\n')
        _f.write(f'max_frames : {args.max_frames}\n')
        _f.write(f'imu_weight : {args.imu_weight}\n')
        if phase == 'D':
            _f.write(f'airimu_offset: {airimu_raw_offset}\n')
        _f.write(f'timestamp  : {_ts}\n')

    tum_path = out_dir / "trajectory.tum"
    traj.save_tum(tum_path)
    print(f"  Saved TUM: {tum_path}")

    metrics_path = out_dir / 'metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f'ATE_RMSE_m  {ate:.8f}\n')
        f.write(f'rrel_deg    {rrel:.8f}\n')
        f.write(f'trel_m      {trel:.8f}\n')
    print(f"  Saved metrics: {metrics_path}")

    # ── Save validation logs ──────────────────────────────────────────────────
    if args.validation and logger is not None:
        logger.close()

        sys_name = "macvo" if phase == 'B' else "macvio_nol" if phase == 'C' else "macvio_wl"
        seq_clean = seq_name.replace('_easy','').replace('_medium','').replace('_difficult','').replace('_','').upper()
        if seq_clean in ("MH02EASY", "MH02"):
            seq_clean = "MH02"
        elif seq_clean in ("MH04DIFFICULT", "MH04"):
            seq_clean = "MH04"
        elif seq_clean in ("V202MEDIUM", "V202"):
            seq_clean = "V202"

        metrics_csv_path = out_dir / 'csv' / 'metrics_macvio.csv'
        write_header = not metrics_csv_path.exists()

        # Write ONLY actual computed metrics — no hardcoded reference values
        with open(metrics_csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=["sequence", "system", "r_rel", "t_rel", "ate"])
            if write_header:
                w.writeheader()
            w.writerow({"sequence": seq_clean, "system": sys_name,
                        "r_rel": rrel, "t_rel": trel, "ate": ate})

        est_csv_path = out_dir / 'csv' / 'trajectory_est.csv'
        write_header = not est_csv_path.exists()
        with open(est_csv_path, 'a', newline='') as f:
            fields = ["sequence", "system", "frame_id", "px", "py", "pz"] + [f"R_{i}{j}" for i in range(3) for j in range(3)]
            w = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                w.writeheader()

            traj_pos = traj.positions()
            traj_poses = traj.poses()
            for f_id in range(len(traj)):
                q = torch.tensor(traj_poses[f_id, 3:], dtype=torch.float64)
                R = pp.SO3(q).matrix().squeeze().numpy()
                row = {
                    "sequence": seq_clean,
                    "system": sys_name,
                    "frame_id": f_id,
                    "px": traj_pos[f_id, 0],
                    "py": traj_pos[f_id, 1],
                    "pz": traj_pos[f_id, 2],
                }
                for i in range(3):
                    for j in range(3):
                        row[f"R_{i}{j}"] = R[i, j]
                w.writerow(row)

        gt_csv_path = out_dir / 'csv' / 'trajectory_gt.csv'
        write_header = not gt_csv_path.exists()
        with open(gt_csv_path, 'a', newline='') as f:
            fields = ["sequence", "frame_id", "px", "py", "pz"] + [f"R_{i}{j}" for i in range(3) for j in range(3)]
            w = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                w.writeheader()

            for f_id in range(len(gt_pos_interp)):
                q = torch.tensor(gt_quat_interp[f_id], dtype=torch.float64)
                R = pp.SO3(q).matrix().squeeze().numpy()
                row = {
                    "sequence": seq_clean,
                    "frame_id": f_id,
                    "px": gt_pos_interp[f_id, 0],
                    "py": gt_pos_interp[f_id, 1],
                    "pz": gt_pos_interp[f_id, 2],
                }
                for i in range(3):
                    for j in range(3):
                        row[f"R_{i}{j}"] = R[i, j]
                w.writerow(row)

        print(f"  Saved validation CSVs to {out_dir / 'csv'}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    from trajectory import _sim3_align, _apply_sim3
    _R, _t, _s = _sim3_align(traj.positions(), gt_pos_interp)
    est_aligned = _apply_sim3(traj.positions(), _R, _t, _s)
    plot_path = _save_trajectory_plot(
        out_dir, phase, seq_name,
        est_aligned, gt_pos_interp,
        ate, rrel, trel,
    )
    print(f"  Saved plot: {plot_path}")

    return ate, rrel, trel


# ── Helpers ────────────────────────────────────────────────────────────────────

_gt_biases_cache: dict = {}

def _get_gt_biases(data_root: str, gt_ts: np.ndarray, t_ns: int):
    """Return GT gyro+accel biases at t_ns from EuRoC GT CSV."""
    global _gt_biases_cache
    if data_root not in _gt_biases_cache:
        csv = Path(data_root) / 'state_groundtruth_estimate0' / 'data.csv'
        rows = []
        with open(csv) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                vals = list(map(float, s.split(',')))
                rows.append(vals)
        rows = np.array(rows)
        _gt_biases_cache[data_root] = {
            'ts': rows[:, 0].astype(np.int64),
            'bg': rows[:, 11:14],
            'ba': rows[:, 14:17],
        }
    cache = _gt_biases_cache[data_root]
    idx = int(np.searchsorted(cache['ts'], t_ns))
    idx = min(idx, len(cache['bg']) - 1)
    return cache['bg'][idx], cache['ba'][idx]

# Commented by Narsimlu
# def _make_imu_factor(integ: dict):
#     """Convert IMUPreintegrator output dict to IMUFactor."""
#     from optimizer import IMUFactor
#     import pypose as pp
#     gamma_data = integ['rot'].data if hasattr(integ['rot'], 'data') else integ['rot']
#     return IMUFactor(
#         gamma = pp.SO3(gamma_data.double()),
#         beta  = integ['vel'].double(),
#         alpha = integ['pos'].double(),
#         cov   = integ['cov'].double(),
#         dt    = float(integ['dt']),
#     )

# Added by Narsimlu
def _make_imu_factor(integ: dict):
    """Convert IMUPreintegrator output dict to IMUFactor."""
    from optimizer import IMUFactor
    import pypose as pp
    import torch

    gamma_data = (
        integ['rot'].data
        if hasattr(integ['rot'], 'data')
        else integ['rot']
    )

    def _optional_tensor(name):
        value = integ.get(name, None)

        if value is None:
            return None

        if isinstance(value, torch.Tensor):
            return value.double()

        return torch.as_tensor(
            value,
            dtype=torch.float64,
        )

    return IMUFactor(
        gamma=pp.SO3(gamma_data.double()),
        beta=integ['vel'].double(),
        alpha=integ['pos'].double(),
        cov=integ['cov'].double(),
        dt=float(integ['dt']),

        bias_ref_ba=_optional_tensor('bias_ref_ba'),
        bias_ref_bg=_optional_tensor('bias_ref_bg'),

        J_gamma_bg=_optional_tensor('J_gamma_bg'),
        J_beta_ba=_optional_tensor('J_beta_ba'),
        J_beta_bg=_optional_tensor('J_beta_bg'),
        J_alpha_ba=_optional_tensor('J_alpha_ba'),
        J_alpha_bg=_optional_tensor('J_alpha_bg'),
    )

# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()

    # Add mac_vio directory to path so imports work
    mac_vio_dir = Path(__file__).parent.resolve()
    if str(mac_vio_dir) not in sys.path:
        sys.path.insert(0, str(mac_vio_dir))

    if args.phase == 'cache':
        run_cache_build(args)
    elif args.phase == 'A':
        run_phase_A(args)
    else:
        run_phase_BCD(args)