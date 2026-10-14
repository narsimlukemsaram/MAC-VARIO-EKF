#!/usr/bin/env python3
"""
run_eval_macvo.py — evaluates MAC-I² VI-init using real MAC-VO visual estimates.

Usage (on HPC):
    cd /scr/user/msakhtar/Files/Projects/UMRP4/MACVIO/MACVIO-main-v3
    python3 mac_i2/eval/run_eval_macvo.py \
        --cache  macvo-cache \
        --euroc  /scr/user/msakhtar/Files/Projects/UMRP4/MAC-VO/data/EuRoC \
        --seqs   MH02 MH04 V101 V103 V202

Paper targets (Table 3.1, 10KFs, w/o Learned IMU):
    MH02: Vel=0.011  G.Dir=0.820°
    MH04: Vel=0.022  G.Dir=0.800°
    V101: Vel=0.010  G.Dir=0.872°
    V103: Vel=0.025  G.Dir=1.149°
    V202: Vel=0.020  G.Dir=0.804°
    Avg:  Vel=0.018  G.Dir=0.889°  SR=1.00
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/scr/user/msakhtar/Files/Projects/UMRP4/MACVIO/MACVIO-main-v3')

import numpy as np
from eval.macvo_cache_loader import make_macvo_windows

# Sequence → EuRoC path mapping
SEQ_PATHS = {
    "MH01": "machine_hall/MH_01_easy",
    "MH02": "machine_hall/MH_02_easy",
    "MH03": "machine_hall/MH_03_medium",
    "MH04": "machine_hall/MH_04_difficult",
    "MH05": "machine_hall/MH_05_difficult",
    "V101": "vicon_room1/V1_01_easy",
    "V102": "vicon_room1/V1_02_medium",
    "V103": "vicon_room1/V1_03_difficult",
    "V201": "vicon_room2/V2_01_easy",
    "V202": "vicon_room2/V2_02_medium",
    "V203": "vicon_room2/V2_03_difficult",
}

PAPER = {
    "MH02": (0.011, 0.820),
    "MH04": (0.022, 0.800),
    "V101": (0.010, 0.872),
    "V103": (0.025, 1.149),
    "V202": (0.020, 0.804),
}


def _eval_worker_macvo(win_and_params):
    win, max_iters, identity_vis, identity_imu = win_and_params
    import torch
    torch.set_num_threads(1)
    from eval.vi_init_macvo import initialize_macvo_window
    return initialize_macvo_window(win, max_iters=max_iters,
                                   identity_vis=identity_vis, identity_imu=identity_imu)


def run_sequence(seq: str, cache_dir: str, euroc_root: str, fast: bool = False, workers: int = None, executor = None, identity_vis: bool = False, identity_imu: bool = False) -> dict | None:
    seq_upper = seq.upper()
    cache_pkl = os.path.join(cache_dir, f"macvo_{seq_upper}_Full.pkl")

    rel = SEQ_PATHS.get(seq_upper)
    if rel is None:
        print(f"  [SKIP] {seq}: unknown sequence name")
        return None
    if not os.path.exists(cache_pkl):
        print(f"  [SKIP] {seq}: cache not found at {cache_pkl}")
        return None

    imu_csv = os.path.join(euroc_root, rel, "mav0/imu0/data.csv")
    gt_csv  = os.path.join(euroc_root, rel, "mav0/state_groundtruth_estimate0/data.csv")

    if not os.path.exists(imu_csv):
        print(f"  [SKIP] {seq}: IMU CSV not found at {imu_csv}")
        return None

    import pickle as pkl
    fast_suffix = "_fast" if fast else "_full"
    win_cache_pkl = cache_pkl.replace(".pkl", f"{fast_suffix}_windows.pkl")

    if os.path.exists(win_cache_pkl):
        with open(win_cache_pkl, 'rb') as f:
            windows = pkl.load(f)
    else:
        windows = make_macvo_windows(cache_pkl, imu_csv, gt_csv, seq_upper)
        if fast:
            if len(windows) > 2:
                step = max(len(windows) // 2, 1)
                windows = windows[::step][:2]
        try:
            with open(win_cache_pkl, 'wb') as f:
                pkl.dump(windows, f)
        except Exception:
            pass

    print(f"  {seq}: {len(windows)} windows")
    if not windows:
        return None

    from eval.vi_init_macvo import initialize_macvo_window
    max_iters = 15 if fast else 50
    if workers == 1 or len(windows) <= 2:
        results = [initialize_macvo_window(w, max_iters=max_iters,
                   identity_vis=identity_vis, identity_imu=identity_imu) for w in windows]
    elif executor is not None:
        tasks = [(w, max_iters, identity_vis, identity_imu) for w in windows]
        results = list(executor.map(_eval_worker_macvo, tasks))
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        ctx = mp.get_context('spawn')
        tasks = [(w, max_iters, identity_vis, identity_imu) for w in windows]
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor_local:
            results = list(executor_local.map(_eval_worker_macvo, tasks))

    vels  = [r["vel_rmse"]     for r in results]
    gdirs = [r["g_dir_err"]   for r in results]
    succs = [r["success"]     for r in results]
    vels_ls    = [r.get("vel_rmse_ls",    float("nan")) for r in results]
    gdirs_ls   = [r.get("g_dir_ls",       float("nan")) for r in results]
    alpha_res  = [r.get("mean_alpha_res", float("nan")) for r in results]

    return {
        "seq":            seq_upper,
        "vel_mean":       np.mean(vels),
        "gdir_mean":      np.mean(gdirs),
        "sr":             np.mean(succs),
        "vel_mean_ls":    np.nanmean(vels_ls),
        "gdir_mean_ls":   np.nanmean(gdirs_ls),
        "mean_alpha_res": np.nanmean(alpha_res),
        "n":              len(results),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache",  default="macvo-cache",
                    help="Directory with macvo_*.pkl files")
    ap.add_argument("--euroc",  required=True,
                    help="EuRoC dataset root (contains machine_hall/, vicon_room*/)")
    ap.add_argument("--seqs",   nargs="+",
                    default=["MH02","MH04","V101","V103","V202"],
                    help="Sequences to evaluate")
    ap.add_argument("--fast", action="store_true",
                    help="Accelerated evaluation: runs 2 windows per sequence with capped max_iters")
    ap.add_argument("--workers", type=int, default=32,
                    help="Number of parallel worker processes (default: 32)")
    ap.add_argument("--identity-vis", action="store_true",
                    help="Use identity visual covariance (Table 3.6 'Id Vis' config)")
    ap.add_argument("--identity-imu", action="store_true",
                    help="Use identity IMU covariance (Table 3.6 'Id IMU' config)")
    args = ap.parse_args()

    print("=" * 70)
    print("MAC-I² Evaluation — w/o Learned IMU — using real MAC-VO poses")
    print("=" * 70)

    import time
    all_r = {}

    executor = None
    if args.workers > 1 and not args.fast:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        ctx = mp.get_context('spawn')
        executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx)

    try:
        for seq in args.seqs:
            print(f"\n▶ {seq}")
            t0 = time.time()
            r = run_sequence(seq, args.cache, args.euroc, args.fast, args.workers,
                             executor=executor, identity_vis=args.identity_vis,
                             identity_imu=args.identity_imu)
            t1 = time.time()
            if r:
                elapsed = t1 - t0
                r["time_elapsed"] = elapsed
                all_r[seq.upper()] = r
                n_win = r.get("n", 0)
                fps = n_win / elapsed if elapsed > 0 else 0.0
                print(f"  Completed in {elapsed:.2f}s ({fps:.1f} windows/sec)")
    finally:
        if executor is not None:
            executor.shutdown()

    print("\n" + "=" * 110)
    print(f"{'Seq':<6} {'Vel(GN)':>9} {'Vel(LS)':>9} {'G.Dir':>9} {'SR':>5} {'αRes':>8} {'Time':>8} {'Speed':>12} "
          f"{'pVel':>8} {'pGDir':>9}")
    print("-" * 110)

    all_vel, all_gdir, all_sr, all_time = [], [], [], []
    for seq in [s.upper() for s in args.seqs]:
        if seq not in all_r: continue
        r = all_r[seq]
        pv, pg = PAPER.get(seq, (None, None))
        pv_s = f"{pv:.3f}" if pv else "  n/a"
        pg_s = f"{pg:.3f}" if pg else "  n/a"
        n_win = r.get("n", 0)
        fps = n_win / r['time_elapsed'] if r['time_elapsed'] > 0 else 0.0
        fps_s = f"{fps:.1f} win/s"
        vls  = r.get('vel_mean_ls',    float('nan'))
        ares = r.get('mean_alpha_res', float('nan'))
        print(f"{seq:<6} {r['vel_mean']:>9.3f} {vls:>9.3f} {r['gdir_mean']:>8.3f}° "
              f"{r['sr']:>5.2f} {ares:>8.4f} {r['time_elapsed']:>7.2f}s {fps_s:>12} {pv_s:>8} {pg_s:>8}°")
        all_vel.append(r["vel_mean"])
        all_gdir.append(r["gdir_mean"])
        all_sr.append(r["sr"])
        all_time.append(r["time_elapsed"])

    if all_vel:
        pv_avg = np.mean([v for v,_ in PAPER.values()])
        pg_avg = np.mean([g for _,g in PAPER.values()])
        avg_v = np.mean(all_vel);  avg_g = np.mean(all_gdir); avg_sr = np.mean(all_sr)
        sum_time = np.sum(all_time)
        total_windows = sum([r.get("n", 0) for r in all_r.values()])
        avg_fps = total_windows / sum_time if sum_time > 0 else 0.0
        avg_fps_s = f"{avg_fps:.1f} win/s"
        avg_vls  = np.nanmean([r.get("vel_mean_ls",    float("nan")) for r in all_r.values()])
        avg_ares = np.nanmean([r.get("mean_alpha_res", float("nan")) for r in all_r.values()])
        print("-" * 110)
        print(f"{'Avg':<6} {avg_v:>9.3f} {avg_vls:>9.3f} {avg_g:>8.3f}° {avg_sr:>5.2f} {avg_ares:>8.4f} {sum_time:>7.2f}s {avg_fps_s:>12}  "
              f"{pv_avg:>8.3f} {pg_avg:>8.3f}°")
    print("=" * 110)


if __name__ == "__main__":
    main()