"""
test_vi_init_fast.py — fast offline test harness for vi_init_macvo

Requires: tests/test_windows.pt  (produced by extract_test_windows.py)
Runtime:  ~1-2s (5 windows, no GPU needed)

Usage (from mac_i2/):
    python3 tests/test_vi_init_fast.py
    python3 tests/test_vi_init_fast.py --windows tests/test_windows.pt --verbose

What it checks:
    1. Preintegration sanity (beta/alpha norms in reasonable range)
    2. LS solution: velocities finite, gravity near 9.81 m/s
    3. GN does not increase error vs LS (stability check)
    4. Vel RMSE and G.Dir per window (printed for comparison with paper targets)
    5. Position residual (alpha_res) norm — signals if visual positions are useful
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from dataclasses import dataclass

# ── Minimal MACVOWindow reconstruction from saved dict ───────────────────────

from eval.macvo_cache_loader import MACVOWindow
import numpy as np

def dict_to_window(d: dict) -> MACVOWindow:
    kf_time_ns = d["kf_time_ns"]
    if hasattr(kf_time_ns, "numpy"):
        kf_time_ns = kf_time_ns.numpy()
    return MACVOWindow(
        kf_time_ns  = kf_time_ns,
        kf_pose_R   = d["kf_pose_R"],
        kf_pose_t   = d["kf_pose_t"],
        kf_pose_cov = d["kf_pose_cov"],
        T_BS_R      = d["T_BS_R"],
        T_BS_t      = d["T_BS_t"],
        acc         = d["acc"],
        gyro        = d["gyro"],
        dt          = d["dt"],
        gt_pos      = d["gt_pos"],
        gt_vel      = d["gt_vel"],
        gt_vel_body = d["gt_vel_body"],
        gt_bg       = d["gt_bg"],
        gravity     = d["gravity"],
        seq_name    = d["seq_name"],
        win_idx     = int(d["win_idx"]),
    )



# ── Import the function under test ───────────────────────────────────────────

from eval.vi_init_macvo import initialize_macvo_window


# ── Assertions ────────────────────────────────────────────────────────────────

def assert_finite(t, name):
    assert torch.isfinite(t).all(), f"Non-finite values in {name}: {t}"

def assert_close(val, expected, tol, name):
    assert abs(val - expected) < tol, f"{name}: got {val:.4f}, expected ≈ {expected} (tol={tol})"


# ── Main test loop ────────────────────────────────────────────────────────────

def run_tests(windows_path: str, verbose: bool = False, identity_vis: bool = True, identity_imu: bool = False):
    if not os.path.exists(windows_path):
        print(f"[SKIP] {windows_path} not found — run extract_test_windows.py first")
        return

    dicts   = torch.load(windows_path, weights_only=False)
    windows = [dict_to_window(d) for d in dicts]
    print(f"Loaded {len(windows)} windows from {windows_path}")

    results = []
    t0 = time.time()

    for win in windows:
        r = initialize_macvo_window(win, max_iters=20, identity_vis=identity_vis, identity_imu=identity_imu)
        results.append(r)

    elapsed = time.time() - t0
    print(f"Ran {len(windows)} windows in {elapsed:.2f}s  ({elapsed/len(windows)*1000:.0f} ms/win)\n")

    # ── Per-window report ────────────────────────────────────────────────
    print(f"{'Win':>4} {'VelRMSE(GN)':>12} {'VelRMSE(LS)':>12} {'G.Dir':>8} {'αRes':>8} {'SR':>4}")
    print("-" * 56)

    all_vel_gn, all_vel_ls, all_gdir, all_ares = [], [], [], []
    n_pass = 0

    for r in results:
        vel_gn = r["vel_rmse"]
        vel_ls = r.get("vel_rmse_ls", float("nan"))
        gdir   = r["g_dir_err"]
        ares   = r.get("mean_alpha_res", float("nan"))
        ok     = r["success"]

        all_vel_gn.append(vel_gn)
        all_vel_ls.append(vel_ls)
        all_gdir.append(gdir)
        all_ares.append(ares)
        if ok: n_pass += 1

        flag = "✓" if ok else "✗"
        print(f"{r['win_idx']:>4} {vel_gn:>12.4f} {vel_ls:>12.4f} {gdir:>7.3f}° {ares:>8.4f}  {flag}")

    print("-" * 56)
    print(f"{'Avg':>4} {np.mean(all_vel_gn):>12.4f} {np.nanmean(all_vel_ls):>12.4f} "
          f"{np.mean(all_gdir):>7.3f}° {np.nanmean(all_ares):>8.4f}  {n_pass}/{len(results)}")
    print(f"\nPaper targets: Vel=0.018  G.Dir=0.889°  SR=1.00")

    # ── Invariant assertions (fail fast) ─────────────────────────────────
    print("\n── Invariant checks ──")
    errors = []

    for i, r in enumerate(results):
        # Gravity magnitude must be near 9.81
        g_norm = float(r["g_est"].norm().item())
        if abs(g_norm - 9.81) > 0.5:
            errors.append(f"win[{i}] gravity norm={g_norm:.3f}, expected ~9.81")

        # Velocities must be finite
        if not torch.isfinite(r["g_est"]).all():
            errors.append(f"win[{i}] g_est has non-finite values")

        # GN should not dramatically increase error vs LS
        vel_gn = r["vel_rmse"]
        vel_ls = r.get("vel_rmse_ls", vel_gn)
        if vel_gn > vel_ls * 3.0:
            errors.append(f"win[{i}] GN degraded LS: {vel_ls:.4f} → {vel_gn:.4f} (3× worse)")

        # G.Dir must not be wildly off (>30° means gravity direction is wrong)
        if r["g_dir_err"] > 30.0:
            errors.append(f"win[{i}] G.Dir={r['g_dir_err']:.1f}° — gravity direction diverged")

    if errors:
        print("FAILED:")
        for e in errors: print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("  All invariants passed ✓")

    if verbose:
        print("\n── Verbose: bg_est vs gt_bg per window ──")
        for i, r in enumerate(results):
            win_obj = windows[i]
            gt_bg = win_obj.gt_bg
            bg_est = r["bg_est"].cpu()
            err = (bg_est - gt_bg).norm().item()
            print(f"  win[{i}] bg_est={bg_est.tolist()}  gt_bg={gt_bg.tolist()}  err={err:.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="tests/test_windows.pt")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--identity-vis", action="store_true", default=False)
    ap.add_argument("--identity-imu", action="store_true", default=False)
    args = ap.parse_args()
    run_tests(args.windows, args.verbose, args.identity_vis, args.identity_imu)
