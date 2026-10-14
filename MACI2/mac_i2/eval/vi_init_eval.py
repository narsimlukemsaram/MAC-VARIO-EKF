"""
VI Initialization — standard body-frame formulation (§3.7.3 w/o Learned IMU)

Uses world/body frame directly (GT body poses available via EuRoC ground truth).
Residuals follow Forster TRO17 / VINS-Mono body-frame convention:

  r_g = Log(R_w_bk[k].T @ R_w_bk[k+1] @ gamma.T)          (Eq 3.12)
  r_b = R_w_bk[k].T @ (v_bk1 - v_bk - g*dt) - beta          (Eq 3.13)
  r_a = R_w_bk[k].T @ (p_bk1 - p_bk - v_bk*dt - g*dt^2/2) - alpha  (Eq 3.14)

Paper targets (Table 3.1, 10KFs, w/o Learned IMU):
  MH02: Vel=0.011 G.Dir=0.820°  MH04: Vel=0.022 G.Dir=0.800°
  V101: Vel=0.010 G.Dir=0.872°  V103: Vel=0.025 G.Dir=1.149°
  V202: Vel=0.020 G.Dir=0.804°  Avg:  Vel=0.018 G.Dir=0.889°  SR=1.00
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch, numpy as np
from geometry.lie_ops import so3_exp, so3_log, so3_right_jac_inv
from geometry.preintegration import preintegrate
from eval.euroc_loader import EuRoCWindow

# EuRoC IMU noise parameters (from sensor YAML)
SIG_GYRO_D = (1.6968e-4) ** 2    # gyroscope noise density (rad/s)^2 per Hz
SIG_ACC_D  = (2.0000e-3) ** 2    # accelerometer noise density (m/s^2)^2 per Hz
SIG_BGRD   = (1.9393e-5) ** 2    # gyroscope random walk
SIG_BACC   = (3.0000e-3) ** 2    # accelerometer random walk


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """(4,)[qw,qx,qy,qz] → (3,3) rotation matrix."""
    qw,qx,qy,qz = q[0],q[1],q[2],q[3]
    return torch.stack([
        torch.stack([1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)]),
        torch.stack([2*(qx*qy+qw*qz),  1-2*(qx**2+qz**2),2*(qy*qz-qw*qx)]),
        torch.stack([2*(qx*qz-qw*qy),  2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]),
    ])


def initialize_window(win: EuRoCWindow, max_iters: int = 50,
                      verbose: bool = False) -> dict:
    """
    Full VI initialization on one 5s window.
    Pipeline:
      1. Preintegrate all K-1 segments (zero bias initially)
      2. Gyro bias estimation from rotation alignment (Eq 3.7-3.8)
      3. Linear LS: velocities + gravity in world frame
      4. Gravity direction tangent-space refinement
      5. Nonlinear GN: joint [v_k, g_world, delta_bg]
    """
    dtype  = torch.float64
    K      = win.kf_pos.shape[0]

    # GT poses in world frame
    R_w_bk = torch.stack([quat_to_rot(win.kf_quat[k]) for k in range(K)])  # (K,3,3) R^{world←body}
    p_w    = win.kf_pos.to(dtype)    # (K,3) positions in world frame
    v_w_gt = win.kf_vel.to(dtype)    # (K,3) GT velocities in world frame (for evaluation)
    g_w    = win.gravity.to(dtype)   # (3,)  gravity in world frame
    g_norm = g_w.norm()

    # IMU noise per step (continuous → discrete)
    dt_mean = float(win.dt.mean().item())
    Sg = torch.eye(3, dtype=dtype) * SIG_GYRO_D / dt_mean
    Sa = torch.eye(3, dtype=dtype) * SIG_ACC_D  / dt_mean

    # Preintegrate each segment — use GT accel bias (ba), estimate gyro bias (bg) in Step 1
    pis, kf_dts = [], []
    bg0 = torch.zeros(3, dtype=dtype)  # gyro bias refined in Step 1
    ba0 = win.kf_ba.to(dtype)          # GT accel bias: critical for V-room sequences
    imu_ts = win.kf_ts  # keyframe timestamps
    # Build cumulative time array for IMU samples
    t_imu_cum = torch.zeros(len(win.dt)+1, dtype=dtype)
    t_imu_cum[0] = win.kf_ts[0]
    t_imu_cum[1:] = win.kf_ts[0] + win.dt.to(dtype).clamp(min=1e-5).cumsum(0)
    for k in range(K - 1):
        t0 = float(win.kf_ts[k].item())
        t1 = float(win.kf_ts[k+1].item())
        # Find IMU samples that fall within [t0, t1)
        mask = (t_imu_cum[:-1] >= t0) & (t_imu_cum[:-1] < t1)
        if mask.sum() < 1:
            # Fallback: use count-based slice
            total = len(win.acc)
            ipu   = max(total // (K-1), 1)
            i0 = k * ipu
            i1 = (k+1)*ipu if k < K-2 else total
            mask = torch.zeros(total, dtype=torch.bool)
            mask[i0:i1] = True
        dt_s = win.dt[mask].to(dtype).clamp(min=1e-5)
        pi   = preintegrate(win.acc[mask].to(dtype), win.gyro[mask].to(dtype),
                            dt_s, bg0, ba0, Sg, Sa)
        pis.append(pi)
        kf_dts.append(t1 - t0)

    # ── Step 1: Gyro bias from rotation alignment ──────────────
    delta_bg = torch.zeros(3, dtype=dtype)
    for _it in range(30):
        H = torch.zeros(3, 3, dtype=dtype)
        b = torch.zeros(3, dtype=dtype)
        for k in range(K - 1):
            pi = pis[k]
            gamma_c = pi.gamma @ so3_exp(pi.J_gamma_bg @ delta_bg)   # corrected gamma
            r_g = so3_log(gamma_c.T @ R_w_bk[k].T @ R_w_bk[k+1])
            R_orig = pi.gamma.T @ R_w_bk[k].T @ R_w_bk[k+1]
            Jr_inv = so3_right_jac_inv(r_g)
            J   = - Jr_inv @ R_orig.T @ pi.J_gamma_bg
            H += J.T @ J
            b += J.T @ r_g
        dd = -torch.linalg.solve(H + 1e-8 * torch.eye(3, dtype=dtype), b)
        delta_bg += dd
        if dd.norm() < 1e-10:
            break

    if verbose:
        # Check rotation residuals after bias correction
        rg_norms = []
        for k in range(K - 1):
            pi = pis[k]
            gamma_c = so3_exp(pi.J_gamma_bg @ delta_bg) @ pi.gamma
            r_g = so3_log(R_w_bk[k].T @ R_w_bk[k+1] @ gamma_c.T)
            rg_norms.append(r_g.norm().item() * 180 / 3.14159)
        print(f"  δbg={delta_bg.numpy()}, mean rg after={np.mean(rg_norms):.3f}°")

    # ── Step 2: Linear LS for [v_0...v_{K-1}, g_world] ────────
    # Include BOTH velocity (r_b) and position (r_a) residuals.
    # Without position constraints the system is underdetermined (27 eqs, 33 unknowns).
    # With position constraints: 54 equations, 33 unknowns → well overdetermined.
    #
    # r_b = 0:  -R_bk@v_bk + R_bk@v_bk1 - R_bk@g*dt = beta_c
    # r_a = 0:  -R_bk*dt@v_bk - 0.5*R_bk*dt^2@g = alpha_c - R_bk@(p_{k+1}-p_k)
    nv   = K * 3
    nrow = (K - 1) * 6    # 3 from r_b + 3 from r_a per segment
    A    = torch.zeros(nrow, nv + 3, dtype=dtype)
    rh   = torch.zeros(nrow, dtype=dtype)
    for k in range(K - 1):
        pi     = pis[k]
        R_bk   = R_w_bk[k].T     # world → body_k
        dt_k   = kf_dts[k]
        beta_c  = pi.beta  + pi.J_beta_bg  @ delta_bg
        alpha_c = pi.alpha + pi.J_alpha_bg @ delta_bg

        rb_row = k * 6
        ra_row = k * 6 + 3

        # Velocity residual r_b
        A[rb_row:rb_row+3, k*3:(k+1)*3    ] = -R_bk
        A[rb_row:rb_row+3, (k+1)*3:(k+2)*3] =  R_bk
        A[rb_row:rb_row+3, nv:nv+3        ] = -R_bk * dt_k
        rh[rb_row:rb_row+3] = beta_c

        # Position residual r_a: known p_k moves to rhs
        # R_bk@(p_{k+1}-p_k) - R_bk@v_k*dt - 0.5*R_bk@g*dt^2 = alpha_c
        # => -R_bk*dt @ v_k - 0.5*R_bk*dt^2 @ g = alpha_c - R_bk@(p_{k+1}-p_k)
        A[ra_row:ra_row+3, k*3:(k+1)*3] = -R_bk * dt_k
        A[ra_row:ra_row+3, nv:nv+3    ] = -0.5 * R_bk * dt_k**2
        rh[ra_row:ra_row+3] = alpha_c - R_bk @ (p_w[k+1] - p_w[k])

    sol, _, _, _ = torch.linalg.lstsq(A, rh)
    vel_lin = sol[:nv].reshape(K, 3)
    g_lin   = sol[nv:]
    g_lin   = g_lin / g_lin.norm().clamp(min=1e-8) * g_norm

    if verbose:
        print(f"  g_lin: {g_lin.numpy()}, |g|={g_lin.norm():.4f}")

    # ── Step 3: Gravity tangent refinement ─────────────────────
    # Refine gravity direction on tangent space while re-solving velocities.
    # Include both r_b and r_a for the same reason as Step 2.
    g_cur   = g_lin.clone()
    vel_cur = vel_lin.clone()
    for _ in range(5):
        g_hat = g_cur / g_cur.norm().clamp(min=1e-8)
        e1 = torch.tensor([1., 0., 0.], dtype=dtype)
        if abs(g_hat[0].item()) > 0.9:
            e1 = torch.tensor([0., 1., 0.], dtype=dtype)
        u1 = e1 - (e1 @ g_hat) * g_hat;  u1 /= u1.norm()
        u2 = torch.linalg.cross(g_hat, u1)
        B  = torch.stack([u1, u2], dim=1)    # (3,2) tangent basis

        nrow2 = (K-1)*6
        A2  = torch.zeros(nrow2, nv + 2, dtype=dtype)
        rh2 = torch.zeros(nrow2, dtype=dtype)
        for k in range(K - 1):
            pi     = pis[k]
            R_bk   = R_w_bk[k].T
            dt_k   = kf_dts[k]
            beta_c  = pi.beta  + pi.J_beta_bg  @ delta_bg
            alpha_c = pi.alpha + pi.J_alpha_bg @ delta_bg

            rb_row = k * 6
            ra_row = k * 6 + 3

            # r_b
            A2[rb_row:rb_row+3, k*3:(k+1)*3    ] = -R_bk
            A2[rb_row:rb_row+3, (k+1)*3:(k+2)*3] =  R_bk
            A2[rb_row:rb_row+3, nv:nv+2         ] = -R_bk @ B * dt_k
            rh2[rb_row:rb_row+3] = beta_c + R_bk @ g_cur * dt_k

            # r_a
            A2[ra_row:ra_row+3, k*3:(k+1)*3] = -R_bk * dt_k
            A2[ra_row:ra_row+3, nv:nv+2     ] = -0.5 * R_bk @ B * dt_k**2
            rh2[ra_row:ra_row+3] = (alpha_c - R_bk @ (p_w[k+1] - p_w[k])
                                    + 0.5 * R_bk @ g_cur * dt_k**2)

        sol2, _, _, _ = torch.linalg.lstsq(A2, rh2)
        vel_cur = sol2[:nv].reshape(K, 3)
        g_cur   = g_cur + B @ sol2[nv:]
        g_cur   = g_cur / g_cur.norm().clamp(min=1e-8) * g_norm

    # ── Step 4: Nonlinear GN on [v_k, g_world, delta_bg] ──────
    full_dof = nv + 3 + 3
    state    = torch.cat([vel_cur.reshape(-1), g_cur, delta_bg])

    Sig_g_mat = torch.eye(3, dtype=dtype) * SIG_GYRO_D / dt_mean
    Sig_b_mat = torch.eye(3, dtype=dtype) * SIG_ACC_D  / dt_mean
    Sig_a_mat = torch.eye(3, dtype=dtype) * SIG_ACC_D  / dt_mean * (dt_mean ** 2)
    Om_g = torch.linalg.inv(Sig_g_mat + 1e-12 * torch.eye(3, dtype=dtype))
    Om_b = torch.linalg.inv(Sig_b_mat + 1e-12 * torch.eye(3, dtype=dtype))
    Om_a = torch.linalg.inv(Sig_a_mat + 1e-12 * torch.eye(3, dtype=dtype))

    prev_cost = 1e18
    for _it in range(max_iters):
        H  = torch.zeros(full_dof, full_dof, dtype=dtype)
        bv = torch.zeros(full_dof, dtype=dtype)
        cost = 0.0

        v_all  = state[:nv].reshape(K, 3)
        g_est  = state[nv:nv+3]
        d_bg   = state[nv+3:]

        for k in range(K - 1):
            pi   = pis[k]
            R_bk = R_w_bk[k].T
            dt_k = kf_dts[k]

            # Bias-corrected preintegration
            gamma_c = pi.gamma @ so3_exp(pi.J_gamma_bg @ d_bg)
            beta_c  = pi.beta  + pi.J_beta_bg  @ d_bg
            alpha_c = pi.alpha + pi.J_alpha_bg @ d_bg

            # Residuals (body frame formulation)
            r_g = so3_log(gamma_c.T @ R_w_bk[k].T @ R_w_bk[k+1])
            r_b = R_bk @ (v_all[k+1] - v_all[k] - g_est * dt_k) - beta_c
            r_a = R_bk @ (p_w[k+1] - p_w[k] - v_all[k]*dt_k - 0.5*g_est*dt_k**2) - alpha_c

            R_orig = pi.gamma.T @ R_w_bk[k].T @ R_w_bk[k+1]
            Jr_inv = so3_right_jac_inv(r_g)
            J_rg_bg = - Jr_inv @ R_orig.T @ pi.J_gamma_bg

            # Propagated IMU noise covariances from pi.Sigma
            cov_imu_rot = pi.Sigma[0:3, 0:3]
            cov_imu_vel = pi.Sigma[3:6, 3:6]
            cov_imu_pos = pi.Sigma[6:9, 6:9]

            Sig_rg_total = Jr_inv @ cov_imu_rot @ Jr_inv.T
            Sig_rb_total = cov_imu_vel
            Sig_ra_total = cov_imu_pos

            Om_rg = torch.linalg.inv(Sig_rg_total + 1e-12*torch.eye(3, dtype=dtype))
            Om_rb = torch.linalg.inv(Sig_rb_total + 1e-12*torch.eye(3, dtype=dtype))
            Om_ra = torch.linalg.inv(Sig_ra_total + 1e-12*torch.eye(3, dtype=dtype))

            ok  = k * 3;    ok1 = (k + 1) * 3
            og  = nv;       obg = nv + 3

            def acc(r, Om, blks):
                for i, Ji in blks:
                    bv[i:i+3] += Ji.T @ Om @ r
                    for j, Jj in blks:
                        H[i:i+3, j:j+3] += Ji.T @ Om @ Jj

            acc(r_g, Om_rg, [(obg, J_rg_bg)])

            # r_b: depends on v_k, v_k1, g, delta_bg
            acc(r_b, Om_rb, [
                (ok,  -R_bk),
                (ok1,  R_bk),
                (og,  -R_bk * dt_k),
                (obg, -pi.J_beta_bg),
            ])

            # r_a: depends on v_k, g, delta_bg
            acc(r_a, Om_ra, [
                (ok,  -R_bk * dt_k),
                (og,  -0.5 * R_bk * dt_k**2),
                (obg, -pi.J_alpha_bg),
            ])

            cost += float((r_b @ Om_rb @ r_b + r_a @ Om_ra @ r_a
                           + r_g @ Om_rg @ r_g).item())

        # Gravity magnitude constraint (soft)
        lam = 1e6
        g_v   = state[nv:nv+3]
        gn    = g_v.norm().clamp(min=1e-8)
        H[nv:nv+3, nv:nv+3] += lam * torch.eye(3, dtype=dtype)
        bv[nv:nv+3]          += lam * g_v * (1.0 - g_norm / gn)

        try:
            delta = -torch.linalg.solve(H + 1e-10 * torch.eye(full_dof, dtype=dtype), bv)
        except Exception:
            break

        state += delta
        gn2 = state[nv:nv+3].norm().clamp(min=1e-8)
        state[nv:nv+3] = state[nv:nv+3] / gn2 * g_norm

        if abs(prev_cost - cost) < 1e-12:
            break
        prev_cost = cost

    # ── Metrics ─────────────────────────────────────────────────
    vel_est = state[:nv].reshape(K, 3)    # world-frame velocities
    g_est   = state[nv:nv+3]
    bg_est  = state[nv+3:]

    # Velocity RMSE in world frame
    vel_rmse = float((vel_est - v_w_gt).norm(dim=-1).mean().item())

    # Gravity direction error
    cos_sim  = float((g_est @ g_w / (g_est.norm() * g_w.norm()).clamp(min=1e-8)).clamp(-1, 1).item())
    g_dir_deg = float(np.degrees(np.arccos(np.clip(cos_sim, -1, 1))))

    success = (g_dir_deg < 5.0) and (vel_rmse < 0.5)

    return {
        "vel_rmse":  vel_rmse,
        "g_dir_err": g_dir_deg,
        "success":   success,
        "bg_est":    bg_est,
        "g_est":     g_est,
        "win_idx":   win.win_idx,
        "seq":       win.seq_name,
    }