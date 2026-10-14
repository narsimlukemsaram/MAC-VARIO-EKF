#!/usr/bin/env python3
# symlink EuRoC raw CSVs into flat dir expected by run_eval.py
import os, sys, argparse

SEQ_MAP = {
    "MH02": "machine_hall/MH_02_easy",
    "MH04": "machine_hall/MH_04_difficult",
    "V101": "vicon_room1/V1_01_easy",
    "V103": "vicon_room1/V1_03_difficult",
    "V202": "vicon_room2/V2_02_medium",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--euroc", required=True)  # e.g. .../data/EuRoC
    ap.add_argument("--out",   required=True)  # flat output dir
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for seq, sub in SEQ_MAP.items():
        src_imu = os.path.join(args.euroc, sub, "mav0/imu0/data.csv")
        src_gt  = os.path.join(args.euroc, sub, "mav0/state_groundtruth_estimate0/data.csv")
        dst_imu = os.path.join(args.out, f"{seq}_imu.csv")
        dst_gt  = os.path.join(args.out, f"{seq}_gt.csv")
        for src, dst in [(src_imu, dst_imu), (src_gt, dst_gt)]:
            if os.path.exists(dst): os.remove(dst)
            os.symlink(src, dst)
            print(f"  {dst} -> {src}")

if __name__ == "__main__":
    main()
