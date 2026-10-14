"""
Real gated multi-window MAC-I2 evaluation on the full 2981-frame local CPU-built MH02 cache.
Builds Omega_vis per KF-edge via compose_edge_omega_vis (real co-visible match data from the
fresh npz), runs gate_and_solve per window at multiple Mahalanobis thresholds (matching the
20/30/50 sweep already documented in MACI2_REPRODUCTION.md), for both raw and learned IMU.

No fabricated numbers -- every value here comes from an actual solver call in this process.
"""
import sys
import os
import numpy as np
import torch

REMOTE = "/scr/user/msakhtar/Files/Projects/UMRP4/MACVIO/MACI2/remote_cache_reference"
sys.path.insert(0, os.path.join(REMOTE, "mac_i2"))

DATA = "/scr/user/msakhtar/Files/Projects/UMRP4/MAC-VO/data/EuRoC/machine_hall/MH_02_easy/mav0"
IMU_CSV = os.path.join(DATA, "imu0", "data.csv")
GT_CSV = os.path.join(DATA, "state_groundtruth_estimate0", "data.csv")
FULL_NPZ = "/scr/user/msakhtar/Files/Projects/UMRP4/MACVIO/MACI2/local_cache_build/out_mh02_full/MACVO-Performant@MH02slice/08_12_231608/tensor_map.npz"

from eval.window_build import make_windows_from_npz
from eval.covariance import load_airimu, compose_edge_omega_vis
from eval.gating import gate_and_solve
import eval.map_solver as MS

DT = torch.float64


def main():
    ckpt = os.path.join(REMOTE, "mac_i2", "checkpoints", "airimu_best.pt")
    model = load_airimu(ckpt)

    W = make_windows_from_npz(FULL_NPZ, IMU_CSV, GT_CSV, "MH02_local_full", n_keyframes=10, kf_stride=10)
    print(f"[window build] {len(W)} full stride-10 windows built from the full local MH02 cache "
          f"({2981} frames)")

    z = np.load(FULL_NPZ)

    results_raw, results_learned = [], []
    for win, fidx in W:
        K = len(fidx)
        Ov = torch.zeros(K - 1, 6, 6, dtype=DT)
        for k in range(K - 1):
            Ov[k] = compose_edge_omega_vis(z, fidx[k], fidx[k + 1])

        MS.USE_AIRIMU = False
        r_raw = gate_and_solve(win, Ov, model, infl=100.0, M_THR=1e9)  # M_THR huge: record maha, don't filter yet
        MS.USE_AIRIMU = True
        r_learned = gate_and_solve(win, Ov, model, infl=100.0, M_THR=1e9)

        results_raw.append(r_raw)
        results_learned.append(r_learned)
        print(f"  win {win.win_idx:3d} fidx=[{fidx[0]}..{fidx[-1]}]  "
              f"maha/dof={r_raw['maha_per_dof']:9.2f}  "
              f"vel_rmse(raw)={r_raw['vel_rmse']:.4f}  vel_rmse(learned)={r_learned['vel_rmse']:.4f}  "
              f"g_dir_err(raw)={r_raw['g_dir_err']:.2f}")

    print(f"\n=== Unfiltered aggregate over all {len(W)} windows ===")
    print(f"raw IMU:     mean vel_rmse = {np.mean([r['vel_rmse'] for r in results_raw]):.4f} m/s")
    print(f"learned IMU: mean vel_rmse = {np.mean([r['vel_rmse'] for r in results_learned]):.4f} m/s")

    print(f"\n=== Mahalanobis liar-filter sweep (raw IMU config) -- matching MACI2_REPRODUCTION.md's own 20/30/50 sweep ===")
    for m_thr in [20, 30, 50]:
        passed = [r for r in results_raw if r["maha_per_dof"] < m_thr]
        if passed:
            mean_v = np.mean([r["vel_rmse"] for r in passed])
            print(f"  M_THR={m_thr:3d}: {len(passed)}/{len(W)} windows pass, gated mean vel_rmse = {mean_v:.4f} m/s")
        else:
            print(f"  M_THR={m_thr:3d}: 0/{len(W)} windows pass")

    print(f"\n=== Mahalanobis liar-filter sweep (learned IMU config) ===")
    for m_thr in [20, 30, 50]:
        passed = [r for r in results_learned if r["maha_per_dof"] < m_thr]
        if passed:
            mean_v = np.mean([r["vel_rmse"] for r in passed])
            print(f"  M_THR={m_thr:3d}: {len(passed)}/{len(W)} windows pass, gated mean vel_rmse = {mean_v:.4f} m/s")
        else:
            print(f"  M_THR={m_thr:3d}: 0/{len(W)} windows pass")

    print(f"\npaper target: 0.018 m/s, G.Dir 0.889 deg")


if __name__ == "__main__":
    main()
