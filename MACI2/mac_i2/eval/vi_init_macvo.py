"""
VI initialization using real MAC-VO visual estimates (§3.7 full pipeline).

Key difference from gt_pose path (eval/vi_init_eval.py):
  - Poses come from MAC-VO (estimated), not GT body poses
  - Visual residual uses camera-frame SE3 with Eq 3.10-3.11
  - Visual covariance Σ_vis from Hessian inverse (Eq 3.2) already in MACVOWindow
  - Body-frame GT velocity used for evaluation metrics only

Residual formulation (§3.7.1, camera-frame):
  r^vis_k = Log(T^{c_k}_{c_{k-1}}  ·  (T^{c_0}_{b_{k+1}})^{-1}  ·  T^{c_0}_{b_k}  ·  T^b_c)
           where T^{c_0}_{b_k} comes from composing MAC-VO camera pose with T_BS extrinsic

  For the w/o-learned-IMU path (Table 3.1 baseline):
  We work directly in the camera/world frame using MAC-VO pose estimates.
  The body-frame positions and gravity are recovered via T_BS.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from geometry.lie_ops import so3_exp, so3_log, so3_right_jac, so3_right_jac_inv
from geometry.preintegration import preintegrate
from eval.macvo_cache_loader import MACVOWindow, se3_inv, se3_compose

SIG_GYRO_D = (1.6968e-4) ** 2
SIG_ACC_D  = (2.0000e-3) ** 2


def initialize_macvo_window(win: MACVOWindow, max_iters: int = 50,
                            identity_vis: bool = True,
                            identity_imu: bool = False) -> dict:
    """
    VI initialization on one MACVOWindow.

    Uses MAC-VO camera poses (estimated) instead of GT body poses.
    Body-frame quantities derived via T_BS extrinsic.

    State: [v_0..v_{K-1} (world-frame velocities), g_world (3,), delta_bg (3,)]
    Total DOF: K*3 + 3 + 3

    Returns: vel_rmse, g_dir_err, success, bg_est, g_est
    """
    dtype = torch.float64
    K     = win.kf_pose_R.shape[0]

    # ── Camera pose → body pose via T_BS ─────────────────────────────────────
    # T_BS: body ← camera  →  R_cb = T_BS_R,  R_bc = R_cb.T
    # Camera pose: R_wc (world←camera)
    # Body pose:   R_wb = R_wc @ R_cb
    R_bc = win.T_BS_R.to(dtype)      # body←camera
    R_cb = R_bc.T                     # camera←body

    # Body rotations from MAC-VO camera poses (used for preintegration alignment)
    R_w_bk = torch.stack([win.kf_pose_R[k] @ R_cb for k in range(K)])  # (K,3,3)

    # GAP 1 FIX: Estimated body positions from MAC-VO camera translations + T_BS extrinsic.
    # p_w_b[k] = t_wc[k] + R_wc[k] @ t_cb   where t_cb = -R_cb @ t_bc
    # (Previously used win.gt_pos — oracle GT positions that inflated constraint quality.)
    t_bc = win.T_BS_t.to(dtype)          # (3,) body←camera translation
    t_cb = -(R_cb @ t_bc)               # camera←body translation
    p_w_gt = torch.stack([
        win.kf_pose_t[k].to(dtype) + win.kf_pose_R[k].to(dtype) @ t_cb
        for k in range(K)
    ])  # (K,3) estimated body positions in world frame

    # GT body velocities (world frame) — for evaluation ONLY
    v_gt_w = win.gt_vel.to(dtype)    # (K,3)
    g_w    = win.gravity.to(dtype)
    g_norm = g_w.norm()

    # ── IMU noise ──────────────────────────────────────────────────────────
    dt_mean = float(win.dt.mean().item())
    Sg = torch.eye(3, dtype=dtype) * SIG_GYRO_D / dt_mean
    Sa = torch.eye(3, dtype=dtype) * SIG_ACC_D  / dt_mean

    # ── Preintegrate per KF segment (timestamp-based slice) ─────────────
    K_seg = K - 1
    pis, kf_dts = [], []
    bg0 = ba0 = torch.zeros(3, dtype=dtype)

    # Build cumulative IMU time array anchored at first KF
    t_imu_cum = torch.zeros(len(win.dt) + 1, dtype=dtype)
    t_imu_cum[0] = win.kf_time_ns[0] * 1e-9
    t_imu_cum[1:] = t_imu_cum[0] + win.dt.to(dtype).clamp(min=1e-5).cumsum(0)

    for k in range(K_seg):
        t0 = float(win.kf_time_ns[k]   * 1e-9)
        t1 = float(win.kf_time_ns[k+1] * 1e-9)
        mask = (t_imu_cum[:-1] >= t0) & (t_imu_cum[:-1] < t1)
        if mask.sum() < 1:
            # Fallback: count-based slice
            total = len(win.acc)
            ipu = max(total // K_seg, 1)
            i0 = k * ipu; i1 = (k+1)*ipu if k < K_seg-1 else total
            mask = torch.zeros(total, dtype=torch.bool); mask[i0:i1] = True
        dt_s = win.dt[mask].to(dtype).clamp(min=1e-5)
        pi   = preintegrate(win.acc[mask].to(dtype), win.gyro[mask].to(dtype),
                            dt_s, bg0, ba0, Sg, Sa)
        pis.append(pi)
        kf_dts.append(t1 - t0)

    # ── Step 1: Gyro bias from rotation alignment ─────────────────────────
    delta_bg = torch.zeros(3, dtype=dtype)
    for _ in range(30):
        H = torch.zeros(3, 3, dtype=dtype)
        b = torch.zeros(3, dtype=dtype)
        for k in range(K_seg):
            pi = pis[k]
            gamma_c = pi.gamma @ so3_exp(pi.J_gamma_bg @ delta_bg)
            r_g = so3_log(gamma_c.T @ R_w_bk[k].T @ R_w_bk[k+1])
            R_orig = pi.gamma.T @ R_w_bk[k].T @ R_w_bk[k+1]
            Jr_inv = so3_right_jac_inv(-r_g)  # J_l⁻¹(r) = J_r⁻¹(-r)
            J   = - Jr_inv @ so3_right_jac(pi.J_gamma_bg @ delta_bg) @ pi.J_gamma_bg
            H  += J.T @ J
            b  += J.T @ r_g
        dd = -torch.linalg.solve(H + 1e-8*torch.eye(3, dtype=dtype), b)
        delta_bg += dd
        if dd.norm() < 1e-10:
            break

    # ── Step 2: Linear LS [v^b_0..v^b_{K-1}, g_c0] — body-frame velocities ──
    # Thesis state: v_bk_bk = velocity of body k in body-k frame (Eq 3.13).
    # r_beta = R_bk_c0 @ R_c0_bk1 @ v_{k+1} - v_k - R_bk_c0 @ g * dt - beta_c = 0
    # r_alpha = R_bk_c0 @ (p_{k+1} - p_k) - v_k*dt - 0.5*R_bk_c0@g*dt^2 - alpha_c = 0
    # State: [v^b_0(3), v^b_1(3), ..., v^b_{K-1}(3), g_c0(3)]  dim = K*3+3
    nv   = K * 3
    nrow = K_seg * 6
    A    = torch.zeros(nrow, nv+3, dtype=dtype)
    rh   = torch.zeros(nrow, dtype=dtype)
    for k in range(K_seg):
        pi      = pis[k]
        R_bk    = R_w_bk[k].T                    # R^{b_k}_{c_0}  (world→body_k)
        R_bk1   = R_w_bk[k+1].T                  # R^{b_{k+1}}_{c_0}
        dt_k    = kf_dts[k]
        beta_c  = pi.beta  + pi.J_beta_bg  @ delta_bg
        alpha_c = pi.alpha + pi.J_alpha_bg @ delta_bg
        rb_row  = k * 6
        ra_row  = k * 6 + 3
        # r_beta: R_bk @ R_c0_bk1 @ v_{k+1} - v_k - R_bk @ g * dt = beta_c
        #   → -I * v_k  +  R_bk @ R_bk1.T * v_{k+1}  -  R_bk * dt * g = beta_c
        A[rb_row:rb_row+3,  k*3:(k+1)*3    ] = -torch.eye(3, dtype=dtype)
        A[rb_row:rb_row+3, (k+1)*3:(k+2)*3 ] =  R_bk @ R_bk1.T
        A[rb_row:rb_row+3,  nv:nv+3        ] = -R_bk * dt_k
        rh[rb_row:rb_row+3] = beta_c
        # r_alpha: R_bk @ (p_{k+1} - p_k) - v_k*dt - 0.5*R_bk@g*dt^2 = alpha_c
        #   → -I*dt * v_k  -  0.5*R_bk*dt^2 * g = alpha_c - R_bk@(p_{k+1}-p_k)
        w_alpha = 1.0
        A[ra_row:ra_row+3,  k*3:(k+1)*3    ] = (-torch.eye(3, dtype=dtype) * dt_k) * w_alpha
        A[ra_row:ra_row+3,  nv:nv+3        ] = (-0.5 * R_bk * dt_k**2) * w_alpha
        rh[ra_row:ra_row+3] = (alpha_c - R_bk @ (p_w_gt[k+1] - p_w_gt[k])) * w_alpha

    sol, _, _, _ = torch.linalg.lstsq(A, rh)
    vel_cur = sol[:nv].reshape(K, 3)   # body-frame velocities v^b_k
    g_cur   = sol[nv:]
    g_cur   = g_cur / g_cur.norm().clamp(min=1e-8) * g_norm

    # ── Step 3: Gravity tangent refinement (body-frame velocity state) ────
    for _ in range(5):
        g_hat = g_cur / g_cur.norm().clamp(min=1e-8)
        e1 = torch.tensor([1.,0.,0.], dtype=dtype)
        if abs(g_hat[0].item()) > 0.9:
            e1 = torch.tensor([0.,1.,0.], dtype=dtype)
        u1 = e1 - (e1@g_hat)*g_hat;  u1 /= u1.norm()
        u2 = torch.linalg.cross(g_hat, u1)
        B  = torch.stack([u1, u2], dim=1)    # (3,2)

        A2  = torch.zeros(K_seg*6, nv+2, dtype=dtype)
        rh2 = torch.zeros(K_seg*6, dtype=dtype)
        for k in range(K_seg):
            pi      = pis[k]
            R_bk    = R_w_bk[k].T
            R_bk1   = R_w_bk[k+1].T
            dt_k    = kf_dts[k]
            beta_c  = pi.beta  + pi.J_beta_bg  @ delta_bg
            alpha_c = pi.alpha + pi.J_alpha_bg @ delta_bg
            rb_row  = k * 6
            ra_row  = k * 6 + 3
            # r_beta with g refined: g = g_cur + B@dw
            A2[rb_row:rb_row+3,  k*3:(k+1)*3   ] = -torch.eye(3, dtype=dtype)
            A2[rb_row:rb_row+3, (k+1)*3:(k+2)*3] =  R_bk @ R_bk1.T
            A2[rb_row:rb_row+3,  nv:nv+2        ] = -R_bk @ B * dt_k
            rh2[rb_row:rb_row+3] = beta_c + R_bk @ g_cur * dt_k
            # r_alpha with g refined
            w_alpha = 1.0
            A2[ra_row:ra_row+3,  k*3:(k+1)*3   ] = ((-torch.eye(3, dtype=dtype) * dt_k)) * w_alpha
            A2[ra_row:ra_row+3,  nv:nv+2        ] = ((-0.5 * R_bk @ B * dt_k**2)) * w_alpha
            rh2[ra_row:ra_row+3] = ((alpha_c - R_bk @ (p_w_gt[k+1]-p_w_gt[k])
                                    + 0.5 * R_bk @ g_cur * dt_k**2)) * w_alpha
        sol2, _, _, _ = torch.linalg.lstsq(A2, rh2)
        vel_cur = sol2[:nv].reshape(K, 3)
        g_cur   = g_cur + B @ sol2[nv:]
        g_cur   = g_cur / g_cur.norm().clamp(min=1e-8) * g_norm

    # ── Pre-GN metrics (LS output) for diagnosis ─────────────────────────
    # vel_cur is body-frame v^b_k — compare directly to gt_vel_body.
    v_gt_body = win.gt_vel_body.to(dtype)
    vel_rmse_ls = float((vel_cur - v_gt_body).norm(dim=-1).mean().item())
    cs_ls = float((g_cur @ g_w / (g_cur.norm()*g_w.norm()).clamp(min=1e-8)).clamp(-1,1).item())
    g_dir_ls = float(np.degrees(np.arccos(np.clip(cs_ls, -1, 1))))

    # Diagnostic: alpha residual norm at LS solution (position constraint quality)
    alpha_res_norms = []
    for k in range(K_seg):
        pi      = pis[k]
        R_bk    = R_w_bk[k].T
        dt_k    = kf_dts[k]
        alpha_c = pi.alpha + pi.J_alpha_bg @ delta_bg
        r_a = R_bk @ (p_w_gt[k+1] - p_w_gt[k]) - vel_cur[k]*dt_k \
              - 0.5 * R_bk @ g_cur * dt_k**2 - alpha_c
        alpha_res_norms.append(float(r_a.norm().item()))
    mean_alpha_res = float(np.mean(alpha_res_norms))


    # ── Step 4: SE(3) joint pose+velocity GN/LM (paper Eq 3.9-3.14, App A.4-A.5) ──
    # State: [δξ_k=(δρ,δφ)(6D) k=0..K-1, v^b_k(3D) k=0..K-1, g(3D), δb^g(3D)]
    # Residuals per seg: r^vis(6D) + r_γ(3D) + r_β(3D) + r_α(3D) = 15D
    from geometry.lie_ops import se3_log, se3_exp, se3_adjoint, se3_left_jac_inv, skew

    def jl_inv_so3(r): return so3_right_jac_inv(-r)  # J_l^{-1}(φ) = J_r^{-1}(-φ)

    # Body pose estimates (SE3: R_wb_k, p_wb_k)
    R_pose = R_w_bk.clone()   # (K,3,3) body-to-world rotation
    p_pose = p_w_gt.clone()   # (K,3)   body positions in world

    # Camera poses for visual relative motion (fixed from MAC-VO)
    R_wc  = win.kf_pose_R.to(dtype); t_wc = win.kf_pose_t.to(dtype)
    t_bc  = win.T_BS_t.to(dtype);    t_cb = -(R_cb @ t_bc)
    # T^{ck}_{ck+1}: relative camera motion (measured, fixed)
    R_rel_cam = torch.stack([R_wc[k].T @ R_wc[k+1]                      for k in range(K_seg)])
    t_rel_cam = torch.stack([R_wc[k].T @ (t_wc[k+1] - t_wc[k])         for k in range(K_seg)])

    v_cur_gn  = vel_cur.clone()   # (K,3) body-frame velocities
    g_cur_gn  = g_cur.clone()     # (3,)
    bg_cur_gn = delta_bg.clone()  # (3,)

    np_  = K * 6; nv_ = K * 3
    g_i  = np_ + nv_; bg_i = np_ + nv_ + 3
    dof  = np_ + nv_ + 6   # K*9 + 6

    def T_err_vis(k, Rk, pk, Rk1, pk1):
        # T^{c0}_{b_{k+1}}^{-1} @ T^{c0}_{b_k}: relative body motion k+1→k
        R_rel = Rk1.T @ Rk;  t_rel = Rk1.T @ (pk - pk1)
        # T^c_b @ (R_rel, t_rel) @ T^b_c
        R_m = R_cb @ R_rel @ R_bc
        t_m = R_cb @ (R_rel @ t_bc + t_rel) + t_cb
        # T^{ck}_{ck+1} @ M
        return R_rel_cam[k] @ R_m, R_rel_cam[k] @ t_m + t_rel_cam[k]

    def A_k_mat(k, Rk1, pk1):
        # A_k = T^{ck}_{ck+1} . T^c_b . inv(T^{c0}_{b_{k+1}})
        R_A = R_rel_cam[k] @ R_cb @ Rk1.T
        t_A = R_rel_cam[k] @ (R_cb @ (-Rk1.T @ pk1) + t_cb) + t_rel_cam[k]
        return R_A, t_A

    def residuals_and_jac(k, Rk, pk, Rk1, pk1, v_k, v_k1, g_, bg_):
        pi   = pis[k]; dt_k = kf_dts[k]
        gc   = pi.gamma @ so3_exp(pi.J_gamma_bg @ bg_)
        bc   = pi.beta  + pi.J_beta_bg  @ bg_
        ac   = pi.alpha + pi.J_alpha_bg @ bg_

        # -- r^vis (Eq 3.10, A.13)
        Re, te   = T_err_vis(k, Rk, pk, Rk1, pk1)
        r_vis    = se3_log(Re, te)
        Jl_v     = se3_left_jac_inv(r_vis)           # (6,6)
        R_A, t_A = A_k_mat(k, Rk1, pk1)
        Ad_A     = se3_adjoint(R_A, t_A)             # (6,6)
        J_v_k    =  Jl_v @ Ad_A                       # ∂r^vis/∂δξ_k  (6,6)
        J_v_k1   = -Jl_v @ Ad_A                       # ∂r^vis/∂δξ_{k+1}

        # -- r_γ (Eq A.21)
        E_g   = Rk1.T @ Rk @ gc
        r_g   = so3_log(E_g)
        Jl_g  = jl_inv_so3(r_g)                      # (3,3)
        J_g_phik  =  Jl_g @ Rk1.T                    # ∂r_γ/∂δφ_k   (Eq A.23)
        J_g_phik1 = -Jl_g @ Rk1.T
        Jr_g  = so3_right_jac_inv(r_g)
        J_g_bg = Jr_g @ so3_right_jac(pi.J_gamma_bg @ bg_) @ pi.J_gamma_bg  # (Eq A.26)

        # -- r_β (Eq A.27)
        a_w   = Rk1 @ v_k1                            # R_{k+1} v_{k+1}
        r_b   = Rk.T @ (a_w - g_ * dt_k) - v_k - bc
        J_b_phik  =  Rk.T @ skew(a_w) - dt_k * Rk.T @ skew(g_)  # (Eq A.28)
        J_b_phik1 = -Rk.T @ skew(a_w)
        J_b_vk  = -torch.eye(3, dtype=dtype)
        J_b_vk1 =  Rk.T @ Rk1
        J_b_g   = -dt_k * Rk.T
        J_b_bg  = -pi.J_beta_bg                       # (Eq A.32)

        # -- r_α (Eq A.33)
        r_a   = (Rk.T @ pk1 - Rk.T @ pk
                 - v_k * dt_k - 0.5 * Rk.T @ g_ * dt_k**2 - ac)
        J_a_phik  =  Rk.T @ (skew(pk1) - 0.5 * dt_k**2 * skew(g_))   # (Eq A.34)
        J_a_rhok  = -Rk.T
        J_a_phik1 = -Rk.T @ skew(pk1)                                  # (Eq A.35)
        J_a_rhok1 =  Rk.T
        J_a_vk  = -dt_k * torch.eye(3, dtype=dtype)
        J_a_g   = -0.5 * dt_k**2 * Rk.T
        J_a_bg  = -pi.J_alpha_bg                                        # (Eq A.38)

        return (r_vis, r_g, r_b, r_a,
                J_v_k, J_v_k1,
                J_g_phik, J_g_phik1, J_g_bg,
                J_b_phik, J_b_phik1, J_b_vk, J_b_vk1, J_b_g, J_b_bg,
                J_a_phik, J_a_rhok, J_a_phik1, J_a_rhok1, J_a_vk, J_a_g, J_a_bg)

    def compute_cost(R_p, p_p, v_p, g_p, bg_p):
        c = 0.0
        for k in range(K_seg):
            pi  = pis[k]; dt_k = kf_dts[k]
            Rk  = R_p[k]; pk  = p_p[k]; Rk1 = R_p[k+1]; pk1 = p_p[k+1]
            gc  = pi.gamma @ so3_exp(pi.J_gamma_bg @ bg_p)
            bc  = pi.beta  + pi.J_beta_bg  @ bg_p
            ac  = pi.alpha + pi.J_alpha_bg @ bg_p
            Re, te = T_err_vis(k, Rk, pk, Rk1, pk1)
            rv = se3_log(Re, te)
            rg = so3_log(Rk1.T @ Rk @ gc)
            rb = Rk.T @ (Rk1 @ v_p[k+1] - g_p * dt_k) - v_p[k] - bc
            ra = Rk.T @ pk1 - Rk.T @ pk - v_p[k]*dt_k - 0.5*Rk.T@g_p*dt_k**2 - ac
            cT = (torch.eye(6, dtype=dtype) * 1e-4) if identity_vis else (
                 (win.kf_pose_cov[k].to(dtype) + win.kf_pose_cov[k+1].to(dtype)) * 10000.0)
            Jv = se3_left_jac_inv(rv); Jrg = so3_right_jac_inv(rg)
            Sv = Jv @ cT @ Jv.T
            I3 = torch.eye(3,dtype=dtype); I6 = torch.eye(6,dtype=dtype)
            Ov = torch.linalg.inv(Sv + 1e-12*I6)
            if identity_imu:
                Og = I3; Ob = I3; Oa = I3
            else:
                Sg = Jrg @ pi.Sigma[0:3,0:3] @ Jrg.T
                Og = torch.linalg.inv(Sg + 1e-12*I3)
                Ob = torch.linalg.inv(pi.Sigma[3:6,3:6] + 1e-12*I3)
                Oa = torch.linalg.inv(pi.Sigma[6:9,6:9] + 1e-12*I3)
            c += float((rv@Ov@rv + rg@Og@rg + rb@Ob@rb + ra@Oa@ra).item())
        return c

    def acc_H(H, bv, r, Om, blks):
        # blks: list of (state_offset, Jacobian(r_dim × s_dim))
        for oi, Ji in blks:
            s = Ji.shape[1]
            bv[oi:oi+s] += Ji.T @ Om @ r
            for oj, Jj in blks:
                H[oi:oi+s, oj:oj+Jj.shape[1]] += Ji.T @ Om @ Jj

    lm_lambda = 1e-3; lm_up = 10.0; lm_dn = 0.1; prev_cost = 1e18

    for _ in range(max_iters):
        H  = torch.zeros(dof, dof, dtype=dtype)
        bv = torch.zeros(dof, dtype=dtype)
        cost = 0.0

        for k in range(K_seg):
            Rk = R_pose[k]; pk = p_pose[k]; Rk1 = R_pose[k+1]; pk1 = p_pose[k+1]
            (r_vis, r_g, r_b, r_a,
             J_v_k, J_v_k1,
             J_g_pk, J_g_pk1, J_g_bg,
             J_b_pk, J_b_pk1, J_b_vk, J_b_vk1, J_b_g, J_b_bg,
             J_a_pk, J_a_rk, J_a_pk1, J_a_rk1, J_a_vk, J_a_g, J_a_bg
             ) = residuals_and_jac(k, Rk, pk, Rk1, pk1,
                                   v_cur_gn[k], v_cur_gn[k+1], g_cur_gn, bg_cur_gn)

            pi  = pis[k]
            opk = k*6; opk1 = (k+1)*6   # pose block offsets (ρ at +0:3, φ at +3:6)
            ovk = np_+k*3; ovk1 = np_+(k+1)*3

            # Visual pose covariance (Eq 3.11): propagate through J_l^{-1}
            # identity_vis = True  → "Id Vis" config (Table 3.6): Σ_vis = I_6
            # identity_vis = False → "Learned Vis": Hessian-derived from MAC-VO
            Jl_v = se3_left_jac_inv(r_vis)
            if identity_vis:
                cov_T = torch.eye(6, dtype=dtype) * 1e-4
            else:
                cov_T = (win.kf_pose_cov[k].to(dtype) + win.kf_pose_cov[k+1].to(dtype)) * 10000.0
            Sv    = Jl_v @ cov_T @ Jl_v.T
            I3 = torch.eye(3,dtype=dtype); I6 = torch.eye(6,dtype=dtype)
            Om_v = torch.linalg.inv(Sv + 1e-12*I6)
            # identity_imu = True → "Id IMU": Ω_γ = Ω_β = Ω_α = I₃
            # identity_imu = False → proper preintegration covariance
            if identity_imu:
                Om_g = I3; Om_b = I3; Om_a = I3
            else:
                Jr_g_ = so3_right_jac_inv(r_g)
                Sg    = Jr_g_ @ pi.Sigma[0:3,0:3] @ Jr_g_.T
                Om_g = torch.linalg.inv(Sg + 1e-12*I3)
                Om_b = torch.linalg.inv(pi.Sigma[3:6,3:6] + 1e-12*I3)
                Om_a = torch.linalg.inv(pi.Sigma[6:9,6:9] + 1e-12*I3)

            # r^vis: Jacobians w.r.t. full δξ_k (6D) and δξ_{k+1} (6D)
            acc_H(H, bv, r_vis, Om_v, [(opk, J_v_k), (opk1, J_v_k1)])
            # r_γ: Jacobians w.r.t. δφ_k (at opk+3), δφ_{k+1} (at opk1+3), δb^g
            acc_H(H, bv, r_g,   Om_g, [(opk+3, J_g_pk), (opk1+3, J_g_pk1), (bg_i, J_g_bg)])
            # r_β: δφ_k, δφ_{k+1}, v_k, v_{k+1}, g, δb^g
            acc_H(H, bv, r_b,   Om_b, [(opk+3, J_b_pk), (opk1+3, J_b_pk1),
                                         (ovk, J_b_vk), (ovk1, J_b_vk1),
                                         (g_i, J_b_g),  (bg_i, J_b_bg)])
            # r_α: δφ_k, δρ_k, δφ_{k+1}, δρ_{k+1}, v_k, g, δb^g
            acc_H(H, bv, r_a,   Om_a, [(opk+3, J_a_pk), (opk, J_a_rk),
                                         (opk1+3, J_a_pk1),(opk1, J_a_rk1),
                                         (ovk, J_a_vk), (g_i, J_a_g), (bg_i, J_a_bg)])
            cost += float((r_vis@Om_v@r_vis + r_g@Om_g@r_g + r_b@Om_b@r_b + r_a@Om_a@r_a).item())

        # Gravity magnitude constraint
        gn_ = g_cur_gn.norm().clamp(min=1e-8)
        H[g_i:g_i+3, g_i:g_i+3] += 1e6 * torch.eye(3, dtype=dtype)
        bv[g_i:g_i+3]            += 1e6 * g_cur_gn * (1.0 - g_norm / gn_)

        try:
            delta = -torch.linalg.solve(H + lm_lambda * torch.eye(dof, dtype=dtype), bv)
        except Exception:
            break

        dxi = delta[:np_].reshape(K, 6)           # (K,6) pose perturbations
        dv  = delta[np_:np_+nv_].reshape(K, 3)
        dg  = delta[g_i:g_i+3]; dbg = delta[bg_i:bg_i+3]

        # SE(3) left retraction: T_new = Exp(δξ) @ T  (Eq A.3-A.4)
        R_try = torch.stack([so3_exp(dxi[k,3:]) @ R_pose[k] for k in range(K)])
        p_try = torch.stack([so3_exp(dxi[k,3:]) @ p_pose[k] +
                             so3_right_jac(dxi[k,3:]) @ dxi[k,:3] for k in range(K)])
        v_try = v_cur_gn + dv
        g_try = g_cur_gn + dg; g_try = g_try / g_try.norm().clamp(min=1e-8) * g_norm
        bg_try = bg_cur_gn + dbg

        cost_new = compute_cost(R_try, p_try, v_try, g_try, bg_try)

        if cost_new < cost:
            R_pose = R_try; p_pose = p_try
            v_cur_gn = v_try; g_cur_gn = g_try; bg_cur_gn = bg_try
            lm_lambda = max(lm_lambda * lm_dn, 1e-10)
            if abs(prev_cost - cost_new) < 1e-8 * (1.0 + abs(prev_cost)):
                break
            prev_cost = cost_new
        else:
            lm_lambda = min(lm_lambda * lm_up, 1e6)

    # ── Metrics ────────────────────────────────────────────────────────────
    v_gt_body = win.gt_vel_body.to(dtype)
    vel_rmse  = float((v_cur_gn - v_gt_body).norm(dim=-1).mean().item())
    cs        = float((g_cur_gn @ g_w / (g_cur_gn.norm()*g_w.norm()).clamp(min=1e-8)).clamp(-1,1).item())
    g_dir_deg = float(np.degrees(np.arccos(np.clip(cs, -1, 1))))

    return {
        "vel_rmse":       vel_rmse,
        "g_dir_err":      g_dir_deg,
        "success":        (g_dir_deg < 5.0) and (vel_rmse < 0.5),
        "vel_rmse_ls":    vel_rmse_ls,
        "g_dir_ls":       g_dir_ls,
        "mean_alpha_res": mean_alpha_res,
        "bg_est":         bg_cur_gn,
        "g_est":          g_cur_gn,
        "win_idx":        win.win_idx,
        "seq":            win.seq_name,
    }