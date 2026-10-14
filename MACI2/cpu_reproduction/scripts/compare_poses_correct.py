"""
Corrected comparison: uses MACVO's own poses.npy/ref_poses.npy (already GT-aligned by MACVO.py
internally, avoiding the camera-vs-body frame convention mismatch that made the first version of
this comparison (tensor_map.npz frames//pose vs raw EuRoC body-frame GT) produce inflated,
methodologically-invalid numbers -- flagging that bug rather than hiding it.
"""
import numpy as np


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


def frame_adjacent_err(poses, ref_poses, label):
    n = len(poses)
    assert len(ref_poses) == n
    errs = []
    for i in range(n - 1):
        q_est_a, q_est_b = poses[i, 4:8], poses[i + 1, 4:8]
        q_gt_a, q_gt_b = ref_poses[i, 4:8], ref_poses[i + 1, 4:8]
        R_est_a, R_est_b = quat_to_R(q_est_a), quat_to_R(q_est_b)
        R_gt_a, R_gt_b = quat_to_R(q_gt_a), quat_to_R(q_gt_b)
        dR_est = R_est_a.T @ R_est_b
        dR_gt = R_gt_a.T @ R_gt_b
        errs.append(rot_angle_deg(dR_est.T @ dR_gt))
    errs = np.array(errs)
    print(f"[{label}] n={n}  mean={errs.mean():.4f} deg  median={np.median(errs):.4f} deg  "
          f"max={errs.max():.4f} deg  min={errs.min():.4f} deg")
    return errs


REMOTE = "/scr/user/msakhtar/Files/Projects/UMRP4/MACVIO/MACI2/remote_cache_reference"
LOCAL = "/scr/user/msakhtar/Files/Projects/UMRP4/MACVIO/MACI2/local_cache_build/out_local_run/MACVO-Performant@MH02slice/08_12_230807"

print("=== corrected falsification test: MACVO's own aligned poses.npy vs ref_poses.npy ===")
rp = np.load(f"{REMOTE}/poses.npy")
rr = np.load(f"{REMOTE}/ref_poses.npy")
print(f"remote: poses.npy shape={rp.shape} ref_poses.npy shape={rr.shape}")
print(f"remote timestamp span: {(rp[-1,0]-rp[0,0])/1e9:.3f}s, n={len(rp)}")
frame_adjacent_err(rp, rr, "REMOTE cache (corrected)")

lp = np.load(f"{LOCAL}/poses.npy")
lr = np.load(f"{LOCAL}/ref_poses.npy")
print(f"\nlocal: poses.npy shape={lp.shape} ref_poses.npy shape={lr.shape}")
print(f"local timestamp span: {(lp[-1,0]-lp[0,0])/1e9:.3f}s, n={len(lp)}")
frame_adjacent_err(lp, lr, "LOCAL fresh cache (corrected)")

# Direct local-vs-remote on shared timestamps, still using the correctly-aligned pose arrays
shared = np.intersect1d(lp[:, 0].astype(np.int64), rp[:, 0].astype(np.int64))
print(f"\nshared timestamps (corrected arrays): {len(shared)}")
if len(shared) >= 2:
    li = {int(t): i for i, t in enumerate(lp[:, 0])}
    ri = {int(t): i for i, t in enumerate(rp[:, 0])}
    errs = []
    for i in range(len(shared) - 1):
        t0, t1 = int(shared[i]), int(shared[i + 1])
        Ra0, Ra1 = quat_to_R(lp[li[t0], 4:8]), quat_to_R(lp[li[t1], 4:8])
        Rb0, Rb1 = quat_to_R(rp[ri[t0], 4:8]), quat_to_R(rp[ri[t1], 4:8])
        dRa = Ra0.T @ Ra1
        dRb = Rb0.T @ Rb1
        errs.append(rot_angle_deg(dRa.T @ dRb))
    errs = np.array(errs)
    print(f"LOCAL vs REMOTE direct (corrected): mean={errs.mean():.4f} deg  median={np.median(errs):.4f} deg")
else:
    print("too few shared timestamps for direct comparison")
