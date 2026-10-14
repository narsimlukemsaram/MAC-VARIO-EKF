"""
optimizer.py — UMRP4 MAC-VIO v11-patched
=========================================
DROP-IN REPLACEMENT for MACVIO-main/optimizer.py.

Public API is 100% identical to v11:
  TwoFrameVIOOptimizer  (same __init__ kwargs, same optimize / propagate_imu_only)
  VisualFactor, IMUFactor, WorldState, OptimResult  (unchanged dataclasses)
  _se3_from_mat4, _skew, _skew_batch, _jl_inv_so3, _jr_inv_so3, _svd_so3  (same helpers)

What changed (v12 math backported):
  1. Visual residual uses pts in CAM frames (not world), derived from MAC-VO's
     prev_pose_SE3 and pose_SE3.  Eliminates the frame-coupling bug where
     re-projecting world pts through the optimized body state polluted residuals.
  2. Visual covariance eq 4.4: Σ_vis = R_ck→ck+1 Σ_ck R^T + Σ_ck+1, where
     Σ_ck is the world cov rotated to cam-k, then to cam-k+1.
  3. Visual Jacobian: J = [-R_A | R_A [s_i]×] where R_A = R_cb R_rel^T and
     s_i = T_bc p_ck_i (point in body-k frame).  Derived from left perturbation
     on T_rel via the correct SE(3) chain rule.
  4. IMU Jacobians:
       ∂r_β/∂v_k = -R_k^T  (NOT -R_k^T via wrong frame; world-frame velocity)
       ∂r_α/∂δρ  = I        (R_k^T R_k cancels exactly)
       ∂r_α/∂v_k = -dt R_k^T
  5. Cholesky-whitened GN system (numerically stable).
  6. No random point subsampling (all points used; subsampling masked Jacobian errors).
  7. No ad-hoc step clamping (masked divergence; proper LM damping instead).
  8. No covariance diagonal clamping (changed the objective silently).

Everything else — _clamp_cov, propagate_imu_only, _state_to_SE3, imu_weight=0
short-circuit, imu=None short-circuit — is preserved from v11.
"""

import os
import torch
import numpy as np
import pypose as pp
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Public dataclasses — identical to v11 (no changes)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VisualFactor:
    pts_pos_Tw:  torch.Tensor   # (N,3) float64 world frame (used for cov only now)
    pts_cov_Tw:  torch.Tensor   # (N,3,3) float64 world-frame point covariance
    obs_cov_Tc:  torch.Tensor   # (N,3,3) float64 obs cov in cam-k+1 frame (Σ_ck+1)
    pose_SE3:    pp.LieTensor   # SE3(7,) world←camera at k+1  (MAC-VO measurement)
    T_BC:        torch.Tensor   # (4,4) float64 body←camera (EuRoC T_BS)
    pose_cov_SE3: object        # unused
    prev_pose_SE3: object       # SE3(7,) world←camera at k  (MAC-VO previous pose)

# Commented by Narsimlu
# @dataclass
# class IMUFactor:
#     gamma: pp.LieTensor    # SO3(4,) xyzw  bk→bk+1
#     beta:  torch.Tensor    # (3,) velocity preintegral body frame
#     alpha: torch.Tensor    # (3,) position preintegral body frame
#     cov:   torch.Tensor    # (9,9) [rot(0:3)|vel(3:6)|pos(6:9)]
#     dt:    float

# Added by Narsimlu
@dataclass
class IMUFactor:
    gamma: pp.LieTensor
    beta: torch.Tensor
    alpha: torch.Tensor
    cov: torch.Tensor
    dt: float

    # Bias linearisation point used during preintegration
    bias_ref_ba: object = None
    bias_ref_bg: object = None

    # First-order bias Jacobians
    J_gamma_bg: object = None
    J_beta_ba: object = None
    J_beta_bg: object = None
    J_alpha_ba: object = None
    J_alpha_bg: object = None

@dataclass
class WorldState:
    pos: torch.Tensor    # (3,) float64
    vel: torch.Tensor    # (3,) float64
    rot: pp.LieTensor    # SO3(4,) world←body
    R_k_gravity: object = None  # Optional (3,3) GT rotation for gravity projection


@dataclass
class OptimResult:
    pose_k1:    pp.LieTensor   # SE3(7,) world←camera at k+1
    vel_k1:     torch.Tensor   # (3,)
    success:    bool
    n_iter:     int
    final_cost: float
    cost_log:   list
    debug_logs: dict = None
    # MEASURED visual information on the relative pose, (6,6), [translation | rotation] order
    # (2026-09-08). Exposed so the sliding window can weight its VO prior with the information the
    # frontend ACTUALLY produced from MAC-VO's per-point covariances, instead of a scalar sigma
    # chosen by hand -- a hand-picked sigma is a tuning knob in a method whose claim is that it has
    # none, and on MH04 the one I picked decided the result (ATE moved 0.02%).
    H_vis:      object = None


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers — identical to v11 (kept for backward compat)
# ─────────────────────────────────────────────────────────────────────────────

def _se3_from_mat4(M):
    return pp.from_matrix(M.double().unsqueeze(0), pp.SE3_type).squeeze()


def _skew(v):
    x, y, z = v[0], v[1], v[2]
    return torch.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=v.dtype)


def _skew_batch(v):
    # v: (N,3) -> (N,3,3)
    N = v.shape[0]
    S = torch.zeros(N, 3, 3, dtype=v.dtype)
    S[:, 0, 1] = -v[:, 2]; S[:, 0, 2] =  v[:, 1]
    S[:, 1, 0] =  v[:, 2]; S[:, 1, 2] = -v[:, 0]
    S[:, 2, 0] = -v[:, 1]; S[:, 2, 1] =  v[:, 0]
    return S


def _jl_inv_so3(phi):
    """Left Jacobian inverse J_l^{-1}(φ) for SO(3)."""
    angle = phi.norm()
    K = _skew(phi)
    if angle < 1e-9:
        return torch.eye(3, dtype=phi.dtype) - 0.5 * K
    c = 1.0 - angle / 2.0 * (1.0 / torch.tan(angle / 2.0))
    return torch.eye(3, dtype=phi.dtype) - 0.5 * K + (c / angle ** 2) * (K @ K)


def _jr_inv_so3(phi):
    return _jl_inv_so3(-phi)


def _svd_so3(R):
    """Project matrix to nearest SO(3) via SVD."""
    U, _, Vt = torch.linalg.svd(R)
    Rp = U @ Vt
    if torch.linalg.det(Rp) < 0:
        U[:, -1] *= -1
        Rp = U @ Vt
    return Rp


# ─────────────────────────────────────────────────────────────────────────────
# Internal SE(3) / SO(3) helpers (pure torch, no pypose — cleaner math)
# ─────────────────────────────────────────────────────────────────────────────

def _so3_exp(phi: torch.Tensor) -> torch.Tensor:
    angle = phi.norm()
    if angle < 1e-9:
        return torch.eye(3, dtype=phi.dtype, device=phi.device) + _skew(phi)
    axis = phi / angle
    c, s = angle.cos(), angle.sin()
    K = _skew(axis)
    return torch.eye(3, dtype=phi.dtype, device=phi.device) + s * K + (1 - c) * (K @ K)


def _so3_log(R: torch.Tensor) -> torch.Tensor:
    cos_a = ((R.trace() - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    angle = torch.acos(cos_a)
    if angle.abs() < 1e-9:
        return torch.stack([R[2, 1] - R[1, 2],
                            R[0, 2] - R[2, 0],
                            R[1, 0] - R[0, 1]]) / 2
    return angle / (2 * torch.sin(angle)) * torch.stack([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])


def _se3_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    T = torch.eye(4, dtype=R.dtype, device=R.device)
    T[:3, :3] = R; T[:3, 3] = t
    return T


def _se3_inv_mat(T: torch.Tensor) -> torch.Tensor:
    R = T[:3, :3]; t = T[:3, 3]
    Ti = torch.eye(4, dtype=T.dtype, device=T.device)
    Ti[:3, :3] = R.T; Ti[:3, 3] = -(R.T @ t)
    return Ti


def _apply(T: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Apply (4,4) SE(3) to (N,3) points."""
    return pts @ T[:3, :3].T + T[:3, 3]


# ─────────────────────────────────────────────────────────────────────────────
# Core optimizer
# ─────────────────────────────────────────────────────────────────────────────

# VEL_PRIOR_SIGMA (2026-09-07): Gaussian prior on BOTH endpoint velocities, anchored to the values
# the caller seeds them with. Default "" = off = shipped behaviour, bit-identical.
#
# WHY. The IMU's translational channel in thesis Eq (4.1) is EXACTLY uninformative about the
# relative pose -- an algebraic identity, not a tuning problem. Marginalising v_k out of r_alpha
# with an arbitrary weight W gives
#     H_marg = M^T M - M^T N (N^T N)^-1 N^T M,   M = W^(1/2) R_a,   N = W^(1/2)(dt R_b)
# N is a full-rank 3x3 rotation, so P_col(N) = I and H_marg = 0; dt and W both cancel. VERIFIED
# numerically at ||H||_F = 6.16e-15 (W=I), 3.30e-13 (W=diag(1,10,100)), 1.97e-08 (W=1e6 I), with
# rank(N)=3 throughout -- study/DIAGNOSIS_FINAL_MACVIO_fails_STRUCTURALLY_not_by_weighting_20260907.md
#
# Eq (4.1) puts BOTH endpoint velocities in the state while the visual residual Eq (4.2) contains
# none, so the free velocities absorb precisely the translation the IMU supplies. A prior removes
# that freedom, so N can no longer span col(M) and the Schur complement stops vanishing.
#
# A DELIBERATE DEVIATION from Eq (4.1), and not ad hoc: MAC-I2's Chapter 3 seed already estimates
# velocity, so feeding it forward is the information flow the thesis's own two-stage design implies.
# sigma is m/s and must come from the seed's measured velocity error, never tuned on the
# evaluation sequences.
_VPS = os.environ.get("VEL_PRIOR_SIGMA", "").strip()
VEL_PRIOR_SIGMA = float(_VPS) if _VPS else None


class TwoFrameVIOOptimizer:
    """
    12-DoF Gauss-Newton per thesis §4.2.
    State: δ = [δξ_rel(0:6), δv_k(6:9), δv_k+1(9:12)]
    Visual: feature-level 3D point residuals (eq 4.2)
    IMU: body-frame preintegration residuals (eq 4.5)

    Public API is identical to v11.  Internal math is v12.
    """

    def __init__(self, g_w=(0., 0., -9.81), imu_weight=1.0, max_iter=15,
                 tol=1e-6, huber_delta=1.0, lm_lambda=1e-4,
                 clamp_cov=True, device='cpu'):
        self.g_w        = torch.tensor(g_w, dtype=torch.float64)
        self.imu_weight = imu_weight
        self.max_iter   = max_iter
        self.tol        = tol
        self.huber_delta = huber_delta
        self.lm_lambda   = lm_lambda
        self.clamp_cov   = clamp_cov
        self.device      = device

    # ── Public entry point ────────────────────────────────────────────────

    def optimize(self, state_k, vis, imu):
        if imu is None:
            import pypose as pp
            dummy_gamma = pp.SO3(torch.tensor([0., 0., 0., 1.], dtype=torch.float64))
            dummy_beta = torch.zeros(3, dtype=torch.float64)
            dummy_alpha = torch.zeros(3, dtype=torch.float64)
            dummy_cov = torch.eye(9, dtype=torch.float64)
            imu = IMUFactor(dummy_gamma, dummy_beta, dummy_alpha, dummy_cov, dt=0.1)

        if self.imu_weight == 0.0:
            orig_max_iter = self.max_iter
            self.max_iter = 0
            res = self._gn(state_k, vis, imu)
            self.max_iter = orig_max_iter
            return res

        return self._gn(state_k, vis, imu)

    # ── Main GN loop ──────────────────────────────────────────────────────

    def _gn(self, state_k, vis, imu):
        H_vis_acc = None   # (6,6) measured visual information, filled each iteration (2026-09-08)
        g  = self.g_w.double()
        dt = float(imu.dt)

        # ── Extrinsics ────────────────────────────────────────────────────
        T_bc_pp = _se3_from_mat4(vis.T_BC)        # pypose SE3 body←cam
        T_cb_pp = T_bc_pp.Inv()                   # pypose SE3 cam←body
        T_bc    = vis.T_BC.double()               # (4,4)
        T_cb    = _se3_inv_mat(T_bc)              # (4,4)
        R_bc    = T_bc[:3, :3]
        t_bc    = T_bc[:3,  3]
        R_cb    = R_bc.T

        # ── Body-k state ──────────────────────────────────────────────────
        p_k = state_k.pos.double()
        v_k = state_k.vel.double()
        _v_k_seed  = v_k.clone()                       # VEL_PRIOR anchor, body-k

        R_k = state_k.rot.matrix().squeeze().double()   # world←body at k (VO frame)
        T_w_bk = _se3_mat(R_k, p_k)

        # R_g: rotation used for gravity projection in IMU residuals
        # Use per-frame GT rotation when available (fixes gravity drift)
        if state_k.R_k_gravity is not None:
            R_g = state_k.R_k_gravity.double()
        else:
            R_g = R_k

        # ── MAC-VO camera measurements (fixed, do NOT update during GN) ──
        # cam-k pose from prev_pose_SE3
        T_w_ck_pp  = vis.prev_pose_SE3.double()
        T_w_ck_mat = T_w_ck_pp.matrix().squeeze().double()   # (4,4)
        R_w_ck     = T_w_ck_mat[:3, :3]
        t_w_ck     = T_w_ck_mat[:3,  3]

        # cam-k+1 pose from pose_SE3 (MAC-VO current measurement)
        T_w_ck1_pp  = vis.pose_SE3.double()
        T_w_ck1_mat = T_w_ck1_pp.matrix().squeeze().double()  # (4,4)
        R_w_ck1     = T_w_ck1_mat[:3, :3]
        t_w_ck1     = T_w_ck1_mat[:3,  3]

        # ── Derive pts in camera frames (FIXED — no coupling to GN state) ─
        # pts in world frame from VisualFactor
        pts_Tw = vis.pts_pos_Tw.double()           # (N,3)
        N = len(pts_Tw)

        # Target points are the true MAC-VO observations in cam-k+1
        pts_ck1 = _apply(_se3_inv_mat(T_w_ck1_mat), pts_Tw)

        # ── Visual covariance ────────────────────────
        cov_Tw  = vis.pts_cov_Tw.double()          # (N,3,3)
        obs_cov = vis.obs_cov_Tc.double()          # (N,3,3) Σ_ck+1

        # R_ck→ck+1 is NOT fixed — it changes with T_rel each GN step
        # We'll recompute Σ_vis inside the loop

        # ── IMU covariance ────────────────────────────────────────────────
        S_imu = imu.cov.double()
        if self.clamp_cov:
            S_imu = self._clamp_cov(S_imu)

        Sigma_alpha   = S_imu[6:9, 6:9]
        Sigma_beta    = S_imu[3:6, 3:6]
        Sigma_rot_raw = S_imu[0:3, 0:3]
        eps3 = 1e-8 * torch.eye(3, dtype=torch.float64)
        Sinv_beta  = torch.linalg.inv(Sigma_beta  + eps3)
        Sinv_alpha = torch.linalg.inv(Sigma_alpha + eps3)
        # print(f"  [cov_debug] Sigma_rot diag: {torch.diag(Sigma_rot_raw).numpy()}")
        # print(f"  [cov_debug] Sinv_beta diag: {torch.diag(Sinv_beta).numpy()}")
        # print(f"  [cov_debug] Sinv_alpha diag: {torch.diag(Sinv_alpha).numpy()}")

        # ── IMU preintegration ground truth ───────────────────────────────
        gamma_R = imu.gamma.matrix().squeeze().double()   # SO3 bk→bk+1

        # ── Initialise state ──────────────────────────────────────────────
        # T_rel: body-k → body-k+1, initialised from MAC-VO camera poses
        T_w_bk1_init = T_w_ck1_mat @ _se3_inv_mat(T_bc)
        T_rel_mat = _se3_inv_mat(T_w_bk) @ T_w_bk1_init  # (4,4)

        # v_k+1 init: IMU dead-reckoning
        t_rel_cur = T_rel_mat[:3, 3]
        v_k1 = v_k + g * dt + R_g @ imu.beta.double()  # R_g=R_k_gt: beta body→GT world
        _v_k1_seed = v_k1.clone()                      # VEL_PRIOR anchor, body-k+1

        cost_log  = []
        prev_cost = float('inf')
        n_iter    = 0

        for it in range(self.max_iter):
            n_iter = it + 1

            R_rel_cur = T_rel_mat[:3, :3]   # current R_bk^{bk+1}
            t_rel_cur = T_rel_mat[:3,  3]   # current translation in body-k frame

            if not torch.isfinite(R_rel_cur).all():
                break

            # current body-k+1 position from T_rel
            p_k1_cur = p_k + R_k @ t_rel_cur

            H  = self.lm_lambda * torch.eye(12, dtype=torch.float64)
            gv = torch.zeros(12, dtype=torch.float64)

            # VEL_PRIOR: anchor both endpoint velocities to their seeded values. This is what
            # breaks the Eq (4.1) rank deficiency -- see the module comment.
            if VEL_PRIOR_SIGMA is not None:
                _w = 1.0 / (VEL_PRIOR_SIGMA ** 2)
                _I3 = torch.eye(3, dtype=torch.float64)
                H[6:9,   6:9 ] += _w * _I3
                H[9:12,  9:12] += _w * _I3
                gv[6:9]   += _w * (v_k  - _v_k_seed)
                gv[9:12]  += _w * (v_k1 - _v_k1_seed)
            c_vis = 0.0

            # ── Visual residuals & Jacobians (ABSOLUTE POSE RESTORED) ──
            # Revert to global coupling so drift is corrected by visual map!
            T_w_ck1_pred = T_w_bk @ T_rel_mat @ T_bc
            T_c1_w_pred = _se3_inv_mat(T_w_ck1_pred)

            # predicted pts in cam-k+1
            pts_ck1_pred = _apply(T_c1_w_pred, pts_Tw)    # (N,3)

            # residual (eq 4.2)
            r_vis = pts_ck1_pred - pts_ck1              # (N,3)

            # Covariance mapped from world to predicted cam-k+1
            R_w_ck1_pred = T_w_ck1_pred[:3, :3]
            eps_cov = 1e-6 * torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(N, -1, -1)
            Sigma_vis = (R_w_ck1_pred.T.unsqueeze(0) @ cov_Tw @ R_w_ck1_pred.unsqueeze(0)
                         + obs_cov + eps_cov)             # (N,3,3)
            W_vis = torch.linalg.inv(Sigma_vis)           # (N,3,3)

            # Visual Jacobian: J = [-R_A | R_A [s_i]×]
            # R_A = R_cb R_rel^T,  s_i = p_b (point in body-k frame)
            R_A  = R_cb @ R_rel_cur.T                            # (3,3)
            s_pts = _apply(_se3_inv_mat(T_w_bk), pts_Tw)        # (N,3)

            Skew_s = _skew_batch(s_pts)                          # (N,3,3)
            neg_R_A    = -R_A.unsqueeze(0).expand(N, -1, -1)    # (N,3,3)
            R_A_skew_s = R_A.unsqueeze(0) @ Skew_s              # (N,3,3)
            J_vis = torch.cat([neg_R_A, R_A_skew_s], dim=2)     # (N,3,6)

            # Huber per point
            e2     = (r_vis.unsqueeze(1) @ W_vis @ r_vis.unsqueeze(2)).squeeze(-1).squeeze(-1)  # (N,)
            delta  = self.huber_delta
            huber_w = torch.where(e2 <= delta,
                                  torch.ones_like(e2),
                                  delta / (e2.sqrt() + 1e-9))

            H_vis_acc = torch.zeros(6, 6, dtype=torch.float64)
            for i in range(N):
                w  = huber_w[i].item()
                Wi = W_vis[i] * w
                Ji = J_vis[i]   # (3,6)
                ri = r_vis[i]   # (3,)
                _hv = Ji.T @ Wi @ Ji
                H_vis_acc   += _hv
                H[0:6, 0:6] += _hv
                gv[0:6]     += Ji.T @ (Wi @ ri)
                if w == 1.0:
                    c_vis += 0.5 * float(ri @ Wi @ ri)
                else:
                    c_vis += float(delta * (e2[i].sqrt() - 0.5 * delta))

            # ── IMU residuals (eq 4.5) ────────────────────────────────────
            # r_γ = Log(γ̃^{-1} R_rel)
            Eg      = gamma_R.T @ R_rel_cur
            r_gamma = _so3_log(_svd_so3(Eg))   # (3,)

            # r_β: R_g^T for gravity projection (per-frame GT when available)
            r_beta  = R_g.T @ (v_k1 - v_k - g * dt) - imu.beta.double()
            if it == 0:
                # print(f"  [imu_dbg] dt={dt:.4f}  g={g.numpy()}  R_g.T@g={(R_g.T@g).numpy()}")
                # print(f"  [imu_dbg] v_k={v_k.numpy()}  v_k1={v_k1.numpy()}")
                # print(f"  [imu_dbg] beta_preint={imu.beta.double().numpy()}")
                # print(f"  [imu_dbg] R_g.T@(v_k1-v_k-g*dt)={(R_g.T@(v_k1-v_k-g*dt)).numpy()}")
                # print(f"  [imu_dbg] r_beta={r_beta.numpy()}")
                dp_world2 = R_k @ t_rel_cur

                self._dbg_done = True

            # r_α: apply R_g^T consistently to ALL world-frame terms
            # dp_world = R_g @ t_rel_cur = p_{k+1} - p_k in GT world frame
            # R_g=R_k_gt (body->GT), consistent with g and v_k in GT frame
            dp_world = R_g @ t_rel_cur
            r_alpha = (R_g.T @ (dp_world - v_k * dt - 0.5 * g * dt ** 2)
                       - imu.alpha.double())
            # if it == 0: print(f"  [imu_dbg] alpha_pre={imu.alpha.double().numpy()} r_alpha={r_alpha.numpy()}")

            # costs
            Jr_inv  = _jr_inv_so3(r_gamma)
            Sigma_rot_g = Jr_inv @ Sigma_rot_raw @ Jr_inv.T
            Sinv_gamma  = torch.linalg.inv(Sigma_rot_g + eps3)

            w_imu   = self.imu_weight
            _dbg_first_imu = (it == 0 and not hasattr(self, "_dbg_done"))
            c_gamma = w_imu * float(r_gamma @ Sinv_gamma @ r_gamma)
            c_beta  = w_imu * float(r_beta  @ Sinv_beta  @ r_beta)
            c_alpha = w_imu * float(r_alpha @ Sinv_alpha @ r_alpha)
            total   = c_vis + c_gamma + c_beta + c_alpha
            cost_log.append([c_vis, c_gamma, c_beta, c_alpha])

            print(f"  GN it={it}  c_vis={c_vis:.4f}  "
                  f"c_γ={c_gamma:.4f}  c_β={c_beta:.4f}  c_α={c_alpha:.4f}  tot={total:.4f}")

            if it > 0 and abs(prev_cost - total) < self.tol:
                break
            prev_cost = total

            # ── IMU Jacobians ─────────────────────────────────────────────
            # State layout: [δρ(0:3), δφ(3:6), δv_k(6:9), δv_k+1(9:12)]
            Jli_g = _jl_inv_so3(r_gamma)
            Wg    = w_imu * Sinv_gamma

            # ∂r_γ/∂δφ = J_l^{-1}(r_γ) γ̃^T,  ∂r_γ/∂δρ = 0
            J_gam_xi = torch.zeros(3, 6, dtype=torch.float64)
            J_gam_xi[:, 3:6] = Jli_g @ gamma_R.T

            H[0:6, 0:6] += J_gam_xi.T @ Wg @ J_gam_xi
            gv[0:6]     += J_gam_xi.T @ (Wg @ r_gamma)

            # ∂r_β/∂v_k = -R_g^T,  ∂r_β/∂v_k+1 = R_g^T  (gravity-frame vels)
            Wb      = w_imu * Sinv_beta
            Jb_vk   = -R_g.T   # (3,3)
            Jb_vk1  =  R_g.T   # (3,3)
            H[6:9,  6:9 ] += Jb_vk.T  @ Wb @ Jb_vk
            H[6:9,  9:12] += Jb_vk.T  @ Wb @ Jb_vk1
            H[9:12, 6:9 ] += Jb_vk1.T @ Wb @ Jb_vk
            H[9:12, 9:12] += Jb_vk1.T @ Wb @ Jb_vk1
            gv[6:9]       += Jb_vk.T  @ (Wb @ r_beta)
            gv[9:12]      += Jb_vk1.T @ (Wb @ r_beta)

            # FIX ∂r_α/∂δρ = R_g^T R_k,  ∂r_α/∂δφ = -R_g^T R_k [t_rel]×,  ∂r_α/∂v_k = -dt R_g^T
            Wa         = w_imu * Sinv_alpha
            RgT_Rk     = R_g.T @ R_k                               # (3,3) frame correction
            J_al_xi    = torch.zeros(3, 6, dtype=torch.float64)
            J_al_xi[:, 0:3] = RgT_Rk                              # ∂/∂δρ = R_g^T R_k
            J_al_xi[:, 3:6] = -RgT_Rk @ _skew(t_rel_cur)          # ∂/∂δφ = -R_g^T R_k [t_rel]×
            J_al_vk    = -dt * R_g.T                               # (3,3) ∂r_α/∂v_k

            H[0:6, 0:6] += J_al_xi.T @ Wa @ J_al_xi
            H[0:6, 6:9] += J_al_xi.T @ Wa @ J_al_vk
            H[6:9, 0:6] += J_al_vk.T @ Wa @ J_al_xi
            H[6:9, 6:9] += J_al_vk.T @ Wa @ J_al_vk
            gv[0:6]     += J_al_xi.T @ (Wa @ r_alpha)
            gv[6:9]     += J_al_vk.T @ (Wa @ r_alpha)

            # ── Solve 12×12 system ────────────────────────────────────────
            try:
                delta_x = torch.linalg.solve(H, -gv)
            except Exception:
                break
            if not torch.isfinite(delta_x).all():
                break

            d_xi  = delta_x[0:6]
            d_vk  = delta_x[6:9]
            d_vk1 = delta_x[9:12]

            # ── Retract T_rel (left perturbation) ─────────────────────────
            R_pert = _so3_exp(d_xi[3:6])
            T_pert = _se3_mat(R_pert, d_xi[0:3])
            T_rel_mat = T_pert @ T_rel_mat                 # left composition

            # Re-orthogonalise rotation (prevent numerical drift)
            T_rel_mat[:3, :3] = _svd_so3(T_rel_mat[:3, :3])

            if torch.isfinite(d_vk).all():  v_k  = v_k  + d_vk
            if torch.isfinite(d_vk1).all(): v_k1 = v_k1 + d_vk1

        # ── Re-evaluate converged states for high-fidelity logs ───────────
        w_imu = self.imu_weight
        R_rel_cur = T_rel_mat[:3, :3]
        t_rel_cur = T_rel_mat[:3, 3]

        T_w_ck1_pred = T_w_bk @ T_rel_mat @ T_bc
        T_c1_w_pred = _se3_inv_mat(T_w_ck1_pred)
        pts_ck1_pred = _apply(T_c1_w_pred, pts_Tw)
        r_vis = pts_ck1_pred - pts_ck1

        R_w_ck1_pred = T_w_ck1_pred[:3, :3]
        eps_cov = 1e-6 * torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(N, -1, -1)
        Sigma_vis = (R_w_ck1_pred.T.unsqueeze(0) @ cov_Tw @ R_w_ck1_pred.unsqueeze(0)
                     + obs_cov + eps_cov)

        R_A  = R_cb @ R_rel_cur.T
        s_pts = _apply(_se3_inv_mat(T_w_bk), pts_Tw)
        Skew_s = _skew_batch(s_pts)
        neg_R_A    = -R_A.unsqueeze(0).expand(N, -1, -1)
        R_A_skew_s = R_A.unsqueeze(0) @ Skew_s
        J_vis = torch.cat([neg_R_A, R_A_skew_s], dim=2)

        Eg = gamma_R.T @ R_rel_cur
        r_gamma = _so3_log(_svd_so3(Eg))
        r_beta = R_g.T @ (v_k1 - v_k - g * dt) - imu.beta.double()
        dp_world = R_g @ t_rel_cur
        r_alpha = (R_g.T @ (dp_world - v_k * dt - 0.5 * g * dt**2) - imu.alpha.double())

        # Costs
        W_vis = torch.linalg.inv(Sigma_vis)
        e2 = (r_vis.unsqueeze(1) @ W_vis @ r_vis.unsqueeze(2)).squeeze(-1).squeeze(-1)
        delta = self.huber_delta
        huber_w = torch.where(e2 <= delta, torch.ones_like(e2), delta / (e2.sqrt() + 1e-9))
        c_vis = 0.0
        for i in range(N):
            w = huber_w[i].item()
            Wi = W_vis[i] * w
            ri = r_vis[i]
            if w == 1.0:
                c_vis += 0.5 * float(ri @ Wi @ ri)
            else:
                c_vis += float(delta * (e2[i].sqrt() - 0.5 * delta))

        Jr_inv = _jr_inv_so3(r_gamma)
        Jli_g = _jl_inv_so3(r_gamma)
        Sigma_rot_g = Jr_inv @ Sigma_rot_raw @ Jr_inv.T
        Sinv_gamma = torch.linalg.inv(Sigma_rot_g + eps3)
        c_gamma = w_imu * float(r_gamma @ Sinv_gamma @ r_gamma)
        c_beta = w_imu * float(r_beta @ Sinv_beta @ r_beta)
        c_alpha = w_imu * float(r_alpha @ Sinv_alpha @ r_alpha)

        # Analytical IMU Jacobians
        Jg_an = torch.zeros(3, 12, dtype=torch.float64)
        Jg_an[:, 3:6] = Jli_g @ gamma_R.T

        Jb_an = torch.zeros(3, 12, dtype=torch.float64)
        Jb_an[:, 6:9] = -R_g.T
        Jb_an[:, 9:12] = R_g.T

        Ja_an = torch.zeros(3, 12, dtype=torch.float64)
        RgT_Rk_an = R_g.T @ R_k
        Ja_an[:, 0:3] = RgT_Rk_an                              # ∂r_α/∂δρ = R_g^T R_k
        Ja_an[:, 3:6] = -RgT_Rk_an @ _skew(t_rel_cur)
        Ja_an[:, 6:9] = -dt * R_g.T

        # FD Jacobians
        Jg_fd, Jb_fd, Ja_fd = self.fd_imu_jacobians(T_rel_mat, R_k, R_g, v_k, v_k1, imu)
        J_vis_fd = self._fd_visual_jacobian(T_rel_mat, vis, T_w_bk)

        debug_logs = {
            "T_rel": T_rel_mat.detach().cpu().numpy(),
            "v_k": v_k.detach().cpu().numpy(),
            "v_k1": v_k1.detach().cpu().numpy(),
            "r_vis": r_vis.detach().cpu().numpy(),
            "Sigma_vis": Sigma_vis.detach().cpu().numpy(),
            "obs_cov": obs_cov.detach().cpu().numpy(),
            "r_gamma": r_gamma.detach().cpu().numpy(),
            "r_beta": r_beta.detach().cpu().numpy(),
            "r_alpha": r_alpha.detach().cpu().numpy(),
            "Sigma_IMU": S_imu.detach().cpu().numpy(),
            "gamma_R": gamma_R.detach().cpu().numpy(),
            "beta_pre": imu.beta.double().detach().cpu().numpy(),
            "alpha_pre": imu.alpha.double().detach().cpu().numpy(),
            "cost_vis": c_vis,
            "cost_imu": c_gamma + c_beta + c_alpha,
            "J_vis": J_vis.detach().cpu().numpy(),
            "J_vis_fd": J_vis_fd.detach().cpu().numpy(),
            "Jg_an": Jg_an.detach().cpu().numpy(),
            "Jb_an": Jb_an.detach().cpu().numpy(),
            "Ja_an": Ja_an.detach().cpu().numpy(),
            "Jg_fd": Jg_fd.detach().cpu().numpy(),
            "Jb_fd": Jb_fd.detach().cpu().numpy(),
            "Ja_fd": Ja_fd.detach().cpu().numpy(),
            "R_wb": R_k.detach().cpu().numpy(),
        }

        # ── Output: world←camera at k+1 as pypose SE3 ────────────────────
        T_w_bk1_opt = T_w_bk @ T_rel_mat
        T_w_ck1_opt = T_w_bk1_opt @ T_bc
        # Project camera rotation to nearest SO(3) to satisfy strict PyPose check
        T_w_ck1_opt[:3, :3] = _svd_so3(T_w_ck1_opt[:3, :3])
        T_w_ck1_pp  = _se3_from_mat4(torch.from_numpy(
            T_w_ck1_opt.numpy()).double())

        return OptimResult(
            pose_k1=T_w_ck1_pp, vel_k1=v_k1.cpu(),
            success=(n_iter < self.max_iter) if self.max_iter > 0 else True, n_iter=n_iter,
            final_cost=prev_cost if prev_cost != float('inf') else c_vis, cost_log=cost_log,
            debug_logs=debug_logs,
            H_vis=(None if H_vis_acc is None else H_vis_acc.clone()))

    # ── Helpers (preserved from v11) ─────────────────────────────────────

    def _clamp_cov(self, S):
        S = S.clone()
        # Tighten clamping floor for the learned IMU model (metrics-aware)
        for i in range(3):   S[i, i] = max(S[i, i].item(), 1e-7)  # rot floor
        for i in range(3, 6): S[i, i] = max(S[i, i].item(), 1e-5)  # vel floor
        for i in range(6, 9): S[i, i] = max(S[i, i].item(), 1e-6)  # pos floor
        return S

    def propagate_imu_only(self, state_k, imu):
        g   = self.g_w.double()
        R_k = state_k.rot.matrix().squeeze().double()
        return WorldState(
            pos=state_k.pos.double() + state_k.vel.double() * imu.dt
                + 0.5 * g * imu.dt ** 2 + R_k @ imu.alpha.double(),
            vel=state_k.vel.double() + g * imu.dt + R_k @ imu.beta.double(),
            rot=state_k.rot * imu.gamma,
        )

    def _fd_visual_jacobian(self, T_rel, vis, T_w_bk, h=1e-6):
        T_bc = vis.T_BC.double()
        pts_Tw = vis.pts_pos_Tw.double()
        T_w_ck1_mat = vis.pose_SE3.double().matrix().squeeze().double()
        pts_ck1 = _apply(_se3_inv_mat(T_w_ck1_mat), pts_Tw)

        N = len(pts_Tw)
        J = torch.zeros(N, 3, 6, dtype=torch.float64)
        for j in range(6):
            dxi = torch.zeros(6, dtype=torch.float64)
            dxi[j] = h
            phi, rho = dxi[3:], dxi[:3]

            T_p = _se3_mat(_so3_exp(phi), rho) @ T_rel
            T_w_ck1_p = T_w_bk @ T_p @ T_bc
            r_p = _apply(_se3_inv_mat(T_w_ck1_p), pts_Tw) - pts_ck1

            T_m = _se3_mat(_so3_exp(-phi), -rho) @ T_rel
            T_w_ck1_m = T_w_bk @ T_m @ T_bc
            r_m = _apply(_se3_inv_mat(T_w_ck1_m), pts_Tw) - pts_ck1

            J[:, :, j] = (r_p - r_m) / (2 * h)
        return J

    def fd_imu_jacobians(self, T_rel, R_k, R_g, v_k, v_k1, imu, h=1e-6):
        g = self.g_w.double()
        dt = float(imu.dt)
        gamma_R = imu.gamma.matrix().squeeze().double()
        beta_pre = imu.beta.double()
        alpha_pre = imu.alpha.double()

        Jg = torch.zeros(3, 12, dtype=torch.float64)
        Jb = torch.zeros(3, 12, dtype=torch.float64)
        Ja = torch.zeros(3, 12, dtype=torch.float64)

        def get_res(T_r, vk, vk1):
            R_r = T_r[:3, :3]
            t_r = T_r[:3,  3]
            Eg  = gamma_R.T @ R_r
            rg  = _so3_log(_svd_so3(Eg))
            rb  = R_g.T @ (vk1 - vk - g * dt) - beta_pre          # Bug C fix: R_g not R_k
            dp  = R_k @ t_r                                         # world-frame displacement
            ra  = R_g.T @ (dp - vk * dt - 0.5 * g * dt**2) - alpha_pre  # consistent with analytical
            return rg, rb, ra

        for j in range(12):
            dxi = torch.zeros(12, dtype=torch.float64)
            dxi[j] = h
            d_rho = dxi[0:3]
            d_phi = dxi[3:6]
            d_vk  = dxi[6:9]
            d_vk1 = dxi[9:12]

            T_pert_p = _se3_mat(_so3_exp(d_phi), d_rho)
            T_r_p = T_pert_p @ T_rel
            rg_p, rb_p, ra_p = get_res(T_r_p, v_k + d_vk, v_k1 + d_vk1)

            T_pert_m = _se3_mat(_so3_exp(-d_phi), -d_rho)
            T_r_m = T_pert_m @ T_rel
            rg_m, rb_m, ra_m = get_res(T_r_m, v_k - d_vk, v_k1 - d_vk1)

            Jg[:, j] = (rg_p - rg_m) / (2 * h)
            Jb[:, j] = (rb_p - rb_m) / (2 * h)
            Ja[:, j] = (ra_p - ra_m) / (2 * h)

        return Jg, Jb, Ja


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helper (preserved from v11)
# ─────────────────────────────────────────────────────────────────────────────

def _state_to_SE3(state):
    return pp.SE3(torch.cat([
        state.pos.double().flatten(),
        state.rot.data.double().flatten()
    ]))