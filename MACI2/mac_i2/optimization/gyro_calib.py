"""
Gyroscope bias and extrinsic rotation estimation  (thesis §3.6)

Solves:  min_{δb_g, q_cb}  Σ_{k∈B} ||Log(q_bc q_{c_{k+1}}^{c_k} (q_bc)^{-1} γ̂_{b_{k+1}}^{b_k})||²_{Σ^r_k}

Eq 3.7:  cost function (weighted on tangent space)
Eq 3.8:  first-order bias correction of preintegration  γ̂ = γ Exp(J^γ_{b_g} δb_g)

Termination:  §3.6 — σ_max of Hessian inverse < σ_th

Two modes:
  1. Known extrinsics  → only estimate δb_g
  2. Calibration mode  → jointly estimate (δb_g, R_cb)
"""

import torch
from geometry.lie_ops import so3_exp, so3_log, so3_right_jac
from typing import Optional


def gyro_bias_extrinsic_estimation(
    gamma_raw: list[torch.Tensor],    # list of K rotation preintegrations γ_{b_{k+1}}^{b_k}  each (3,3)
    J_gamma_bg: list[torch.Tensor],   # list of K Jacobians  ∂γ/∂b_g  each (3,3)
    R_c_meas: list[torch.Tensor],     # list of K relative camera rotations  q_{c_{k+1}}^{c_k}  each (3,3)
    Sigma_r: list[torch.Tensor],      # list of K rotation residual covariances each (3,3)
    R_bc_init: Optional[torch.Tensor] = None,  # (3,3) initial extrinsic rotation  (calibration mode)
    sigma_th: float = 0.05,           # convergence threshold on σ_max of H^{-1}
    max_iters: int = 100,
    calibrate_extrinsic: bool = False,
) -> dict:
    """
    Gauss-Newton solver for gyroscope bias + (optionally) camera-IMU extrinsic.

    Returns:
      bg      : (3,)  estimated gyroscope bias
      R_bc    : (3,3) extrinsic rotation (refined or identity if not calibrating)
      H_inv   : Hessian inverse (for uncertainty)
      converged: bool
    """
    device = gamma_raw[0].device
    dtype  = gamma_raw[0].dtype
    K      = len(gamma_raw)

    # State vector:  δb_g (3)  [+ δφ_bc (3) in calibration mode]
    dof = 6 if calibrate_extrinsic else 3

    bg    = torch.zeros(3, device=device, dtype=dtype)
    R_bc  = R_bc_init if R_bc_init is not None else torch.eye(3, device=device, dtype=dtype)
    R_cb  = R_bc.T

    converged = False
    H_inv = torch.eye(dof, device=device, dtype=dtype)

    for _it in range(max_iters):
        H = torch.zeros(dof, dof, device=device, dtype=dtype)
        b = torch.zeros(dof, device=device, dtype=dtype)

        for k in range(K):
            # Eq 3.8: corrected preintegration with current bias
            gamma_hat = gamma_raw[k] @ so3_exp(J_gamma_bg[k] @ bg)

            # Relative rotation from vision side: R_bc * R_c * R_cb
            R_vision  = R_bc @ R_c_meas[k] @ R_cb

            # Residual  r_k = Log(R_vision  γ̂^{-1})  (Eq 3.7)
            r_k = so3_log(R_vision @ gamma_hat.T)     # (3,)

            # Information matrix Ω_k = Σ^{-1}_{r_k}
            Omega = torch.linalg.inv(Sigma_r[k] + 1e-6 * torch.eye(3, device=device, dtype=dtype))

            # ── Jacobian w.r.t. δb_g  (first-order) ──────────────
            # ∂r/∂δb_g = J_l^{-1}(r_k) * R_vision * J_gamma_bg[k]
            # Approximate J_l^{-1} ≈ I for small residuals
            J_r = -R_vision @ J_gamma_bg[k]   # (3, 3)

            if calibrate_extrinsic:
                # Fix: ∂r/∂δφ_bc = d/dδφ [Log(Exp(δφ) R_bc R_c R_cb Exp(-δφ) γˆ^{-1})] |_{δφ=0}
                # = (I - R_vision)  (not skew of camera measurement log)
                # Bug was: phi_c = so3_log(R_c_meas[k]); J_phi = skew(phi_c)  — wrong frame/formula
                J_phi = torch.eye(3, device=device, dtype=dtype) - R_vision  # (3, 3)
                J = torch.cat([J_r, J_phi], dim=1)  # (3, 6)
            else:
                J = J_r                        # (3, 3)

            H = H + J.T @ Omega @ J
            b = b + J.T @ Omega @ r_k

        # Gauss-Newton step
        try:
            H_inv = torch.linalg.inv(H + 1e-6 * torch.eye(dof, device=device, dtype=dtype))
        except Exception:
            break
        delta = -H_inv @ b

        # ── Apply update ──────────────────────────────────────────
        bg = bg + delta[:3]
        if calibrate_extrinsic and dof == 6:
            R_bc = so3_exp(delta[3:6]) @ R_bc
            R_cb = R_bc.T

        # ── Convergence check  §3.6 ───────────────────────────────
        sigma_max = torch.linalg.eigvalsh(H_inv).max().item()
        if sigma_max < sigma_th:
            converged = True
            break

    return {
        "bg":       bg,
        "R_bc":     R_bc,
        "H_inv":    H_inv,
        "converged": converged,
        "sigma_max": sigma_max,
    }
