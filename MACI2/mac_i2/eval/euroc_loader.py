"""
EuRoC CSV loader — produces windowed preintegration segments
matching the paper's 10KF / 5-second evaluation setup (§3.8.1/3.8.2).

CSV format:
  imu : timestamp[ns], wx, wy, wz, ax, ay, az
  gt  : timestamp[ns], px,py,pz, qw,qx,qy,qz, vx,vy,vz, bgx,bgy,bgz, bax,bay,baz
"""
import os
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class EuRoCWindow:
    """One 5-second / 10-KF initialization window."""
    acc:  torch.Tensor      # (N, 3) m/s²  body frame
    gyro: torch.Tensor      # (N, 3) rad/s body frame
    dt:   torch.Tensor      # (N,)   seconds

    kf_pos:  torch.Tensor   # (K, 3) world frame positions [m]
    kf_quat: torch.Tensor   # (K, 4) [qw,qx,qy,qz]
    kf_vel:  torch.Tensor   # (K, 3) world frame velocities [m/s]
    kf_bg:   torch.Tensor   # (3,)   mean gyro bias over window
    kf_ba:   torch.Tensor   # (3,)   mean accel bias over window
    kf_ts:   torch.Tensor   # (K,)   timestamps [s]
    gravity: torch.Tensor   # (3,)   gravity in world frame [m/s²]

    seq_name: str
    win_idx:  int

    def __reduce__(self):
        """Pickle tensors as numpy bytes — zero FDs, safe for ProcessPoolExecutor."""
        return (_rebuild_euroc_window, (
            self.acc.numpy(), self.gyro.numpy(), self.dt.numpy(),
            self.kf_pos.numpy(), self.kf_quat.numpy(), self.kf_vel.numpy(),
            self.kf_bg.numpy(), self.kf_ba.numpy(), self.kf_ts.numpy(),
            self.gravity.numpy(),
            self.seq_name, self.win_idx,
        ))


def _rebuild_euroc_window(acc, gyro, dt, kf_pos, kf_quat, kf_vel,
                          kf_bg, kf_ba, kf_ts, gravity, seq_name, win_idx):
    """Reconstruct EuRoCWindow from numpy arrays after pickling."""
    return EuRoCWindow(
        acc      = torch.from_numpy(acc),
        gyro     = torch.from_numpy(gyro),
        dt       = torch.from_numpy(dt),
        kf_pos   = torch.from_numpy(kf_pos),
        kf_quat  = torch.from_numpy(kf_quat),
        kf_vel   = torch.from_numpy(kf_vel),
        kf_bg    = torch.from_numpy(kf_bg),
        kf_ba    = torch.from_numpy(kf_ba),
        kf_ts    = torch.from_numpy(kf_ts),
        gravity  = torch.from_numpy(gravity),
        seq_name = seq_name,
        win_idx  = win_idx,
    )

def load_euroc_csvs(imu_csv: str, gt_csv: str) -> dict:
    """Load raw numpy arrays from EuRoC CSV files."""
    imu_npy = imu_csv + ".npy"
    gt_npy  = gt_csv + ".npy"
    if os.path.exists(imu_npy) and os.path.exists(gt_npy):
        imu_raw = np.load(imu_npy)
        gt_raw  = np.load(gt_npy)
    else:
        with open(imu_csv, 'r') as f:
            imu_raw = np.array([line.strip().split(',') for line in f if line and not line.startswith('#')], dtype=np.float64)
        with open(gt_csv, 'r') as f:
            gt_raw = np.array([line.strip().split(',') for line in f if line and not line.startswith('#')], dtype=np.float64)
        try:
            np.save(imu_npy, imu_raw)
            np.save(gt_npy, gt_raw)
        except Exception:
            pass
    return {
        "imu_t":   imu_raw[:, 0] * 1e-9,
        "gyro":    imu_raw[:, 1:4],
        "acc":     imu_raw[:, 4:7],
        "gt_t":    gt_raw[:, 0] * 1e-9,
        "gt_pos":  gt_raw[:, 1:4],
        "gt_quat": gt_raw[:, 4:8],
        "gt_vel":  gt_raw[:, 8:11],
        "gt_bg":   gt_raw[:, 11:14],
        "gt_ba":   gt_raw[:, 14:17],
    }


def interp_xyz(t_query, t_data, x_data):
    return np.column_stack([
        np.interp(t_query, t_data, x_data[:, i]) for i in range(x_data.shape[1])
    ])


def interp_quat(t_query, t_data, q_data):
    """Simple linear interp + renorm for quaternions [qw,qx,qy,qz]."""
    out = np.zeros((len(t_query), 4))
    for i, t in enumerate(t_query):
        idx = int(np.searchsorted(t_data, t))
        idx = np.clip(idx, 1, len(t_data) - 1)
        t0, t1 = t_data[idx-1], t_data[idx]
        alpha = (t - t0) / max(t1 - t0, 1e-12)
        q0, q1 = q_data[idx-1], q_data[idx]
        if np.dot(q0, q1) < 0:
            q1 = -q1
        q = (1 - alpha) * q0 + alpha * q1
        q /= np.linalg.norm(q) + 1e-12
        out[i] = q
    return out


def make_windows(
    data: dict,
    seq_name: str,
    window_sec: float = 5.0,
    n_keyframes: int = 10,
) -> list:
    """
    Slice sequence into non-overlapping 5-second windows, 10 keyframes each.
    Matches §3.8.2 protocol exactly.
    """
    imu_t = data["imu_t"]
    gt_t  = data["gt_t"]

    t_start = max(imu_t[0], gt_t[0]) + 0.1
    t_end   = min(imu_t[-1], gt_t[-1]) - 0.1

    # EuRoC gravity: IMU measures g upward ≈ [0,0,9.81] in sensor frame
    # World frame (NWU): gravity = [0, 0, -9.81007]
    gravity = torch.tensor([0., 0., -9.81007], dtype=torch.float64)

    windows = []
    t_win   = t_start
    idx     = 0

    while t_win + window_sec <= t_end:
        t_end_win = t_win + window_sec
        kf_ts = np.linspace(t_win, t_end_win, n_keyframes)

        mask = (imu_t >= t_win) & (imu_t < t_end_win)
        if mask.sum() < 5:
            t_win += window_sec
            continue

        t_imu   = imu_t[mask]
        acc_np  = data["acc"][mask]
        gyro_np = data["gyro"][mask]
        dt_np   = np.diff(t_imu, prepend=t_imu[0] - (t_imu[1]-t_imu[0]))

        kf_pos  = interp_xyz(kf_ts, gt_t, data["gt_pos"])
        kf_vel  = interp_xyz(kf_ts, gt_t, data["gt_vel"])
        kf_quat = interp_quat(kf_ts, gt_t, data["gt_quat"])
        kf_bg   = interp_xyz(kf_ts, gt_t, data["gt_bg"]).mean(0)
        kf_ba   = interp_xyz(kf_ts, gt_t, data["gt_ba"]).mean(0)

        windows.append(EuRoCWindow(
            acc      = torch.tensor(acc_np,  dtype=torch.float64),
            gyro     = torch.tensor(gyro_np, dtype=torch.float64),
            dt       = torch.tensor(dt_np,   dtype=torch.float64),
            kf_pos   = torch.tensor(kf_pos,  dtype=torch.float64),
            kf_quat  = torch.tensor(kf_quat, dtype=torch.float64),
            kf_vel   = torch.tensor(kf_vel,  dtype=torch.float64),
            kf_bg    = torch.tensor(kf_bg,   dtype=torch.float64),
            kf_ba    = torch.tensor(kf_ba,   dtype=torch.float64),
            kf_ts    = torch.tensor(kf_ts,   dtype=torch.float64),
            gravity  = gravity,
            seq_name = seq_name,
            win_idx  = idx,
        ))
        t_win += window_sec
        idx   += 1

    return windows
