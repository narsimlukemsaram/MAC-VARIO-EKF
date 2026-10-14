"""
Compares the fresh local CPU MAC-VO cache against (a) the remote "clean" cache and (b) GT,
using the gauge-invariant frame-adjacent rotation-error falsification test from
MACI2_REPRODUCTION.md section 7. Then runs the real MAC-I2 window evaluation (raw-IMU and
learned-IMU configs) on the fresh cache and reports vel_rmse / g_dir_err against the paper's
target (0.018 m/s, G.Dir 0.889 deg).

No fabricated numbers -- every number printed here comes from an actual tensor read or an
actual solver call in this script's own process, not from memory of what any prior run reported.
"""
import sys
import os
import numpy as np

REMOTE = "/scr/user/msakhtar/Files/Projects/UMRP4/MACVIO/MACI2/remote_cache_reference"
LOCAL_RUN_ROOT = "/scr/user/msakhtar/Files/Projects/UMRP4/MACVIO/MACI2/local_cache_build/out_local_run"
sys.path.insert(0, os.path.join(REMOTE, "mac_i2"))

DATA = "/scr/user/msakhtar/Files/Projects/UMRP4/MAC-VO/data/EuRoC/machine_hall/MH_02_easy/mav0"
IMU_CSV = os.path.join(DATA, "imu0", "data.csv")
GT_CSV = os.path.join(DATA, "state_groundtruth_estimate0", "data.csv")


def find_local_npz():
    for root, _, files in os.walk(LOCAL_RUN_ROOT):
        for f in files:
            if f == "tensor_map.npz":
                return os.path.join(root, f)
    return None


def quat_to_R(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    X, Y, Z = x * s, y * s, z * s
    wx, wy, wz = w * X, w * Y, w * Z
    xx, xy, xz = x * X, x * Y, x * Z
    yy, yz, zz = y * Y, y * Z, z * Z
    return np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)],
    ])


def rot_angle_deg(R):
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def load_gt_interp(gt_csv):
    gt = np.array([l.strip().split(",") for l in open(gt_csv) if l and not l.startswith("#")], dtype=np.float64)
    return gt[:, 0] * 1e-9, gt[:, 1:4], gt[:, 4:8]  # t_sec, pos(xyz), quat(wxyz)


def gt_pose_at(t_query, gt_t, gt_pos, gt_quat):
    """Nearest-neighbour GT pose lookup (sufficient for a gauge-invariant frame-adjacent check)."""
    idx = np.searchsorted(gt_t, t_query)
    idx = np.clip(idx, 0, len(gt_t) - 1)
    return gt_pos[idx], gt_quat[idx]


def frame_adjacent_rot_err_vs_gt(npz_path, gt_t, gt_pos, gt_quat, label):
    npz = np.load(npz_path)
    poses = npz["frames//pose"]  # tx,ty,tz,qx,qy,qz,qw
    time_ns = npz["frames//time_ns"]
    n = len(time_ns)
    errs = []
    for i in range(n - 1):
        t_a, t_b = time_ns[i] * 1e-9, time_ns[i + 1] * 1e-9
        q_est_a, q_est_b = poses[i, 3:7], poses[i + 1, 3:7]  # xyzw
        R_est_a, R_est_b = quat_to_R(q_est_a), quat_to_R(q_est_b)
        _, q_gt_a_wxyz = gt_pose_at(t_a, gt_t, gt_pos, gt_quat)
        _, q_gt_b_wxyz = gt_pose_at(t_b, gt_t, gt_pos, gt_quat)
        q_gt_a = [q_gt_a_wxyz[1], q_gt_a_wxyz[2], q_gt_a_wxyz[3], q_gt_a_wxyz[0]]
        q_gt_b = [q_gt_b_wxyz[1], q_gt_b_wxyz[2], q_gt_b_wxyz[3], q_gt_b_wxyz[0]]
        R_gt_a, R_gt_b = quat_to_R(q_gt_a), quat_to_R(q_gt_b)
        dR_est = R_est_a.T @ R_est_b
        dR_gt = R_gt_a.T @ R_gt_b
        err_R = dR_est.T @ dR_gt
        errs.append(rot_angle_deg(err_R))
    errs = np.array(errs)
    print(f"[{label}] n_frames={n}  mean_dR_err={errs.mean():.4f} deg  median={np.median(errs):.4f} deg  "
          f"max={errs.max():.4f} deg  min={errs.min():.4f} deg")
    return errs


def direct_frame_comparison(npz_a_path, npz_b_path, label_a, label_b):
    """Compare fresh-local vs remote-cache directly on their shared timestamps."""
    a = np.load(npz_a_path)
    b = np.load(npz_b_path)
    ta, tb = a["frames//time_ns"], b["frames//time_ns"]
    shared = np.intersect1d(ta, tb)
    print(f"[{label_a} vs {label_b}] shared timestamps: {len(shared)} / ({len(ta)}, {len(tb)})")
    if len(shared) < 2:
        print("  too few shared timestamps for a direct frame-adjacent comparison")
        return
    idx_a = {t: i for i, t in enumerate(ta)}
    idx_b = {t: i for i, t in enumerate(tb)}
    errs = []
    for i in range(len(shared) - 1):
        t0, t1 = shared[i], shared[i + 1]
        pa0, pa1 = a["frames//pose"][idx_a[t0]], a["frames//pose"][idx_a[t1]]
        pb0, pb1 = b["frames//pose"][idx_b[t0]], b["frames//pose"][idx_b[t1]]
        Ra0, Ra1 = quat_to_R(pa0[3:7]), quat_to_R(pa1[3:7])
        Rb0, Rb1 = quat_to_R(pb0[3:7]), quat_to_R(pb1[3:7])
        dRa = Ra0.T @ Ra1
        dRb = Rb0.T @ Rb1
        errs.append(rot_angle_deg(dRa.T @ dRb))
    errs = np.array(errs)
    print(f"  mean frame-adjacent ΔR disagreement: {errs.mean():.4f} deg  median={np.median(errs):.4f} deg")


def run_mac_i2_eval(fresh_npz_path):
    from eval.window_build import make_windows_from_npz
    from eval.covariance import load_airimu
    import eval.map_solver as MS

    ckpt = os.path.join(REMOTE, "mac_i2", "checkpoints", "airimu_best.pt")
    model = load_airimu(ckpt)

    W = make_windows_from_npz(fresh_npz_path, IMU_CSV, GT_CSV, "MH02_local")
    print(f"\n[MAC-I2 window build] {len(W)} full stride-10 windows built from the fresh local cache")
    if len(W) == 0:
        print("  0 windows -- fresh cache did not span enough frames for a full 10-KF stride-10 window. "
              "Cannot run the real MAC-I2 evaluation on this cache; reporting the falsification-test "
              "results above only.")
        return

    win, fidx = W[0]
    print(f"[MAC-I2 window 0] fidx={fidx}")

    MS.USE_AIRIMU = False
    r_raw = MS.initialize_scale(win, model, max_iters=10)
    MS.USE_AIRIMU = True
    r_learned = MS.initialize_scale(win, model, max_iters=10)

    print("\n=== MAC-I2 real evaluation results (fresh local CPU-built cache) ===")
    print(f"{'config':<20}{'vel_rmse (m/s)':<18}{'g_dir_err (deg)':<18}{'s_est':<10}")
    print(f"{'raw IMU':<20}{r_raw['vel_rmse']:<18.4f}{r_raw['g_dir_err']:<18.4f}{r_raw['s_est']:<10.4f}")
    print(f"{'learned IMU':<20}{r_learned['vel_rmse']:<18.4f}{r_learned['g_dir_err']:<18.4f}{r_learned['s_est']:<10.4f}")
    print(f"{'paper target':<20}{'0.018':<18}{'0.889':<18}{'--':<10}")


if __name__ == "__main__":
    gt_t, gt_pos, gt_quat = load_gt_interp(GT_CSV)

    remote_npz = os.path.join(REMOTE, "tensor_map.npz")
    local_npz = find_local_npz()
    if local_npz is None:
        print("ERROR: no local tensor_map.npz found under", LOCAL_RUN_ROOT)
        sys.exit(1)
    print("local npz:", local_npz)
    print("remote npz:", remote_npz)

    print("\n=== Falsification test: frame-adjacent rotation error vs GT (gauge-invariant) ===")
    frame_adjacent_rot_err_vs_gt(remote_npz, gt_t, gt_pos, gt_quat, "REMOTE cache")
    frame_adjacent_rot_err_vs_gt(local_npz, gt_t, gt_pos, gt_quat, "LOCAL fresh cache")

    print("\n=== Direct comparison: local fresh cache vs remote cache (shared timestamps) ===")
    direct_frame_comparison(local_npz, remote_npz, "LOCAL", "REMOTE")

    run_mac_i2_eval(local_npz)
