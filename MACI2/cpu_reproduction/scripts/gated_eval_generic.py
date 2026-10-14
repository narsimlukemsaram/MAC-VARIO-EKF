"""
Generalized version of gated_eval_full_mh02.py -- same real gated multi-window MAC-I2
evaluation, parametrized by sequence so it runs against any of the 5 fresh local caches
(MH02/MH04/V101/V103/V202) without duplicating the logic per sequence.

Usage: python3 gated_eval_generic.py <seq_name> <data_root> <npz_path>
"""
import sys
import os
import numpy as np
import torch

REMOTE = "/scr/user/msakhtar/Files/Projects/UMRP4/MACVIO/MACI2/remote_cache_reference"
sys.path.insert(0, os.path.join(REMOTE, "mac_i2"))

from eval.window_build import make_windows_from_npz
from eval.covariance import load_airimu, compose_edge_omega_vis
from eval.gating import gate_and_solve
import eval.map_solver as MS

DT = torch.float64


def main():
    seq_name, data_root, npz_path = sys.argv[1], sys.argv[2], sys.argv[3]
    imu_csv = os.path.join(data_root, "imu0", "data.csv")
    gt_csv = os.path.join(data_root, "state_groundtruth_estimate0", "data.csv")

    ckpt = os.path.join(REMOTE, "mac_i2", "checkpoints", "airimu_best.pt")
    model = load_airimu(ckpt)

    W = make_windows_from_npz(npz_path, imu_csv, gt_csv, seq_name, n_keyframes=10, kf_stride=10)
    print(f"[{seq_name}] [window build] {len(W)} full stride-10 windows built")

    z = np.load(npz_path)

    results_raw, results_learned = [], []
    for win, fidx in W:
        K = len(fidx)
        Ov = torch.zeros(K - 1, 6, 6, dtype=DT)
        for k in range(K - 1):
            Ov[k] = compose_edge_omega_vis(z, fidx[k], fidx[k + 1])

        MS.USE_AIRIMU = False
        r_raw = gate_and_solve(win, Ov, model, infl=100.0, M_THR=1e9)
        MS.USE_AIRIMU = True
        r_learned = gate_and_solve(win, Ov, model, infl=100.0, M_THR=1e9)

        results_raw.append(r_raw)
        results_learned.append(r_learned)
        print(f"[{seq_name}]   win {win.win_idx:3d} fidx=[{fidx[0]}..{fidx[-1]}]  "
              f"maha/dof={r_raw['maha_per_dof']:9.2f}  "
              f"vel_rmse(raw)={r_raw['vel_rmse']:.4f}  vel_rmse(learned)={r_learned['vel_rmse']:.4f}  "
              f"g_dir_err(raw)={r_raw['g_dir_err']:.2f}")

    print(f"\n[{seq_name}] === Unfiltered aggregate over all {len(W)} windows ===")
    print(f"[{seq_name}] raw IMU:     mean vel_rmse = {np.mean([r['vel_rmse'] for r in results_raw]):.4f} m/s")
    print(f"[{seq_name}] learned IMU: mean vel_rmse = {np.mean([r['vel_rmse'] for r in results_learned]):.4f} m/s")

    for cfg_name, results in [("raw", results_raw), ("learned", results_learned)]:
        print(f"\n[{seq_name}] === Mahalanobis sweep ({cfg_name} IMU) ===")
        for m_thr in [20, 30, 50]:
            passed = [r for r in results if r["maha_per_dof"] < m_thr]
            if passed:
                mean_v = np.mean([r["vel_rmse"] for r in passed])
                mean_g = np.mean([r["g_dir_err"] for r in passed])
                print(f"[{seq_name}]   M_THR={m_thr:3d}: {len(passed)}/{len(W)} pass, "
                      f"gated mean vel_rmse={mean_v:.4f} m/s, mean g_dir_err={mean_g:.2f} deg")
            else:
                print(f"[{seq_name}]   M_THR={m_thr:3d}: 0/{len(W)} windows pass")

    print(f"\n[{seq_name}] paper target: 0.018 m/s, G.Dir 0.889 deg")
    print(f"[{seq_name}] GATED_EVAL_DONE")


if __name__ == "__main__":
    main()
