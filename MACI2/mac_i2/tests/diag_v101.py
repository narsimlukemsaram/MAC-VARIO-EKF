#!/usr/bin/env python3
# diagnose V101 per-window G.Dir distribution
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from eval.euroc_loader import load_euroc_csvs, make_windows
from eval.vi_init_eval  import initialize_window

DATA = "/tmp/euroc_flat"
for wsec in [2.5, 5.0]:
    data = load_euroc_csvs(f"{DATA}/V101_imu.csv", f"{DATA}/V101_gt.csv")
    wins = make_windows(data, "V101", window_sec=wsec, n_keyframes=10)
    gd, vl = [], []
    for w in wins:
        r = initialize_window(w)
        gd.append(r["g_dir_err"]); vl.append(r["vel_rmse"])
    gd = np.array(gd); vl = np.array(vl)
    p = [50,75,90,95,99]
    print(f"\nV101 wsec={wsec}  n={len(wins)}")
    print(f"  G.Dir pct {p}: {np.percentile(gd,p).round(2)}")
    print(f"  Vel   pct {p}: {np.percentile(vl,p).round(4)}")
    print(f"  G.Dir>5°: {(gd>5).sum()}  G.Dir>2°: {(gd>2).sum()}  G.Dir>1°: {(gd>1).sum()}")
    print(f"  worst 5 G.Dir: {np.sort(gd)[-5:][::-1].round(2)}")
    # inspect a bad window
    worst_idx = np.argmax(gd)
    w = wins[worst_idx]
    print(f"  worst win idx={worst_idx}  t_start={float(w.kf_ts[0]):.2f}  vel_mean={float(w.kf_vel.norm(dim=-1).mean()):.3f}")
