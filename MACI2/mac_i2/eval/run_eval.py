#!/usr/bin/env python3
"""
MAC-I² evaluation against paper Table 3.1.

Runs the w/o Learned IMU variant (§3.7.3) on all 5 EuRoC test sequences.

Usage:
  python3 eval/run_eval.py --data /path/to/euroc_test_data

Paper targets (Table 3.1, 10KFs, MAC-I² w/o Learned IMU):
  MH02: Vel=0.011 G.Dir=0.820°
  MH04: Vel=0.022 G.Dir=0.800°
  V101: Vel=0.010 G.Dir=0.872°
  V103: Vel=0.025 G.Dir=1.149°
  V202: Vel=0.020 G.Dir=0.804°
  Avg:  Vel=0.018 G.Dir=0.889°  SR=1.00
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from eval.euroc_loader import load_euroc_csvs, make_windows
from eval.vi_init_eval  import initialize_window

SEQUENCES = ["MH02", "MH04", "V101", "V103", "V202"]

PAPER_RESULTS = {
    # (vel_rmse, g_dir_deg)  w/o Learned IMU  Table 3.1
    "MH02": (0.011, 0.820),
    "MH04": (0.022, 0.800),
    "V101": (0.010, 0.872),
    "V103": (0.025, 1.149),
    "V202": (0.020, 0.804),
}


def _eval_worker(win_and_params):
    win, max_iters, verbose = win_and_params
    import torch
    # new
    torch.set_num_threads(1)
    from eval.vi_init_eval import initialize_window
    return initialize_window(win, max_iters=max_iters, verbose=verbose)


def run_sequence(seq: str, data_dir: str, fast: bool = False, workers: int = None, executor = None, window_sec: float = 5.0) -> dict:
    imu_csv = os.path.join(data_dir, f"{seq}_imu.csv")
    gt_csv  = os.path.join(data_dir, f"{seq}_gt.csv")
    if not os.path.exists(imu_csv):
        print(f"  [SKIP] {seq}: file not found")
        return None

    import pickle as pkl
    fast_suffix = "_fast" if fast else "_full"
    win_cache_pkl = os.path.join(data_dir, f"{seq}{fast_suffix}_w{window_sec:.1f}_windows.pkl")

    if os.path.exists(win_cache_pkl):
        with open(win_cache_pkl, 'rb') as f:
            windows = pkl.load(f)
    else:
        data    = load_euroc_csvs(imu_csv, gt_csv)
        windows = make_windows(data, seq, window_sec=window_sec, n_keyframes=10)
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

    max_iters = 15 if fast else 50
    if workers == 1 or len(windows) <= 2:
        results = [initialize_window(w, max_iters=max_iters) for w in windows]
    elif executor is not None:
        tasks = [(w, max_iters, False) for w in windows]
        results = list(executor.map(_eval_worker, tasks))
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        ctx = mp.get_context('spawn')
        tasks = [(w, max_iters, False) for w in windows]
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor_local:
            results = list(executor_local.map(_eval_worker, tasks))

    vels    = [r["vel_rmse"]  for r in results]
    gdirs   = [r["g_dir_err"] for r in results]
    succ    = [r["success"]   for r in results]

    return {
        "seq":       seq,
        "vel_mean":  np.mean(vels),
        "gdir_mean": np.mean(gdirs),
        "sr":        np.mean(succ),
        "n_windows": len(results),
        "vels":      vels,
        "gdirs":     gdirs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/tmp/euroc_test_data",
                        help="Directory with {SEQ}_imu.csv and {SEQ}_gt.csv files")
    parser.add_argument("--fast", action="store_true",
                        help="Accelerated evaluation: runs 2 windows per sequence with capped max_iters")
    parser.add_argument("--window-sec", type=float, default=5.0, dest="window_sec",
                        help="window duration seconds (paper 1.5-2.5)")
    parser.add_argument("--workers", type=int, default=32,
                        help="Number of parallel worker processes (default: 32)")
    args = parser.parse_args()

    print("=" * 72)
    print("MAC-I² EuRoC Evaluation  —  w/o Learned IMU (§3.7.3)")
    print("=" * 72)

    import time
    all_results = {}

    executor = None
    if args.workers > 1 and not args.fast:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        ctx = mp.get_context('spawn')
        executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx)

    try:
        for seq in SEQUENCES:
            print(f"\n▶ {seq}")
            t0 = time.time()
            r = run_sequence(seq, args.data, args.fast, args.workers, executor=executor, window_sec=args.window_sec)
            t1 = time.time()
            if r:
                elapsed = t1 - t0
                r["time_elapsed"] = elapsed
                all_results[seq] = r
                n_win = r.get("n_windows", 0)
                fps = n_win / elapsed if elapsed > 0 else 0.0
                print(f"  Completed in {elapsed:.2f}s ({fps:.1f} windows/sec)")
    finally:
        if executor is not None:
            executor.shutdown()

    # ── Print table matching paper format ──────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Sequence':<8} {'Vel RMSE':>10} {'G.Dir Err':>11} {'SR':>6} {'Time':>8} {'Speed':>12} "
          f"{'PaperVel':>10} {'PaperGDir':>11}")
    print("-" * 90)

    all_vel, all_gdir, all_sr, all_time = [], [], [], []
    for seq in SEQUENCES:
        if seq not in all_results:
            continue
        r = all_results[seq]
        pv, pg = PAPER_RESULTS[seq]
        n_win = r.get("n_windows", 0)
        fps = n_win / r['time_elapsed'] if r['time_elapsed'] > 0 else 0.0
        fps_s = f"{fps:.1f} win/s"
        print(f"{seq:<8} {r['vel_mean']:>10.3f} {r['gdir_mean']:>10.3f}° "
              f"{r['sr']:>6.2f} {r['time_elapsed']:>7.2f}s {fps_s:>12} "
              f"{pv:>10.3f} {pg:>10.3f}°")
        all_vel.append(r["vel_mean"])
        all_gdir.append(r["gdir_mean"])
        all_sr.append(r["sr"])
        all_time.append(r["time_elapsed"])

    print("-" * 90)
    avg_vel  = np.mean(all_vel);  pv_avg = np.mean([v for v,_ in PAPER_RESULTS.values()])
    avg_gdir = np.mean(all_gdir); pg_avg = np.mean([g for _,g in PAPER_RESULTS.values()])
    avg_sr   = np.mean(all_sr)
    sum_time = np.sum(all_time)
    total_windows = sum([r.get("n_windows", 0) for r in all_results.values()])
    avg_fps = total_windows / sum_time if sum_time > 0 else 0.0
    avg_fps_s = f"{avg_fps:.1f} win/s"
    print(f"{'Avg/Total':<8} {avg_vel:>10.3f} {avg_gdir:>10.3f}° {avg_sr:>6.2f} {sum_time:>7.2f}s {avg_fps_s:>12} "
          f"{pv_avg:>10.3f} {pg_avg:>10.3f}°")
    print("=" * 90)

    return all_results


if __name__ == "__main__":
    main()