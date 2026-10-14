"""
extract_test_windows.py — run ONCE on the GPU cluster to produce test_windows.pt

Usage (from mac_i2/):
    python3 tests/extract_test_windows.py \
        --cache /scr/user/msakhtar/Files/Projects/UMRP4/MACVIO/MACVIO-main-v3/macvo-cache \
        --euroc /scr/user/msakhtar/Files/Projects/UMRP4/MAC-VO/data/EuRoC \
        --out   tests/test_windows.pt \
        --seq   MH02 \
        --n     5

Output: tests/test_windows.pt
    A list of 5 MACVOWindow-compatible dicts with all tensors serialised.
    Can be loaded on any machine with torch — no pkl or EuRoC data needed.
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from eval.macvo_cache_loader import make_macvo_windows


def window_to_dict(w):
    """Convert MACVOWindow dataclass to a plain dict of tensors + scalars."""
    return dict(
        kf_time_ns  = torch.tensor(w.kf_time_ns,  dtype=torch.int64),
        kf_pose_R   = w.kf_pose_R.double(),
        kf_pose_t   = w.kf_pose_t.double(),
        kf_pose_cov = w.kf_pose_cov.double(),
        T_BS_R      = w.T_BS_R.double(),
        T_BS_t      = w.T_BS_t.double(),
        acc         = w.acc.double(),
        gyro        = w.gyro.double(),
        dt          = w.dt.double(),
        gt_pos      = w.gt_pos.double(),
        gt_vel      = w.gt_vel.double(),
        gt_vel_body = w.gt_vel_body.double(),
        gt_bg       = w.gt_bg.double(),
        gravity     = w.gravity.double(),
        seq_name    = w.seq_name,
        win_idx     = w.win_idx,
    )


SEQ_PATHS = {
    "MH02": ("machine_hall/MH_02_easy",   "mav0/imu0/data.csv",   "mav0/state_groundtruth_estimate0/data.csv"),
    "MH04": ("machine_hall/MH_04_difficult", "mav0/imu0/data.csv", "mav0/state_groundtruth_estimate0/data.csv"),
    "V101": ("vicon_room1/V1_01_easy",     "mav0/imu0/data.csv",   "mav0/state_groundtruth_estimate0/data.csv"),
    "V103": ("vicon_room1/V1_03_difficult","mav0/imu0/data.csv",   "mav0/state_groundtruth_estimate0/data.csv"),
    "V202": ("vicon_room2/V2_02_medium",   "mav0/imu0/data.csv",   "mav0/state_groundtruth_estimate0/data.csv"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--euroc", required=True)
    ap.add_argument("--out",   default="tests/test_windows.pt")
    ap.add_argument("--seq",   default="MH02")
    ap.add_argument("--n",     type=int, default=5)
    args = ap.parse_args()

    seq_upper = args.seq.upper()
    seq_sub, imu_rel, gt_rel = SEQ_PATHS[seq_upper]

    cache_pkl = os.path.join(args.cache, f"macvo_{seq_upper}_Full.pkl")
    imu_csv   = os.path.join(args.euroc, seq_sub, imu_rel)
    gt_csv    = os.path.join(args.euroc, seq_sub, gt_rel)

    print(f"Loading windows from {seq_upper}...")
    windows = make_macvo_windows(cache_pkl, imu_csv, gt_csv, seq_name=seq_upper)
    print(f"  {len(windows)} windows available, extracting first {args.n}")

    selected = windows[:args.n]
    dicts    = [window_to_dict(w) for w in selected]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(dicts, args.out)
    print(f"Saved {len(dicts)} windows → {args.out}")

    # Quick sanity print
    for i, d in enumerate(dicts):
        K = d["kf_pose_R"].shape[0]
        print(f"  win[{i}]  K={K}  gt_vel_body norm={d['gt_vel_body'].norm(dim=-1).mean():.3f}"
              f"  gravity={d['gravity'].tolist()}")


if __name__ == "__main__":
    main()