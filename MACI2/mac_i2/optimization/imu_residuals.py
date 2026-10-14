"""
IMU residuals for VI initialization and calibration  (thesis §3.7.2 + §3.7.3)

Two cases:
  With learned IMU model  (§3.7.2, Eq 3.12–3.15):
    - Uses corrected preintegration  γ̃, β̃, α̃
    - No explicit bias in residual (correction absorbs it)
    - Cov from predicted preintegration covariance + Jac transformation

  Without learned IMU model  (§3.7.3, Eq 3.16–3.18):
    - Uses raw preintegration + first-order bias correction
    - Explicit b^g, b^a in residual
    - Cov from calibrated/fixed noise params
"""

import torch
from geometry.lie_ops import so3_log, so3_right_jac_inv, skew
from geometry.preintegration import PreintegrationResult


def imu_residual_with_learned_model(
    R_c0_bk:    torch.Tensor,    # (3,3)  rotation part of T^{c_0}_{b_k}
    R_c0_bk1:   torch.Tensor,    # (3,3)  rotation part of T^{c_0}_{b_{k+1}}
    p_c0_bk:    torch.Tensor,    # (3,)   position of b_k in c_0 frame
    p_c0_bk1:   torch.Tensor,    # (3,)   position of b_{k+1} in c_0 frame
    v_bk_bk:    torch.Tensor,    # (3,)   velocity of b_k in b_k frame
    v_bk1_bk1:  torch.Tensor,    # (3,)   velocity of b_{k+1} in b_{k+1} frame
    g_c0:       torch.Tensor,    # (3,)   gravity in c_0 frame
    dt:         float,            # time interval t_{k+1} - t_k
    gamma_corr: torch.Tensor,    # (3,3)  corrected rotation preintegration  γ̃
    beta_corr:  torch.Tensor,    # (3,)   corrected velocity preintegration  β̃
    alpha_corr: torch.Tensor,    # (3,)   corrected position preintegration  α̃
    Sigma_preint: torch.Tensor,  # (9,9)  predicted preintegration covariance
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      r_gamma (3,)   rotation residual  Eq 3.12
      r_beta  (3,)   velocity residual  Eq 3.13
      r_alpha (3,)   position residual  Eq 3.14
      Sigma_imu (9,9) covariance of stacked [r_gamma, r_beta, r_alpha]  Eq 3.15
    """
    R_bk_c0 = R_c0_bk.T     # R^{b_k}_{c_0}

    # Eq 3.12:  r_γ = Log(R^{b_{k+1}}_{c_0} R^{c_0}_{b_k} γ̃)
    # Fix: must be R_c0_bk1.T @ R_c0_bk — relative rotation b_{k+1}→b_k (thesis Eq A.21)
    # Bug was: R_c0_bk1 @ R_bk_c0 which conjugates into the wrong frame
    r_gamma = so3_log(R_c0_bk1.T @ R_c0_bk @ gamma_corr)   # (3,)

    # Eq 3.13:  r_β = R^{b_k}_{c_0} R^{c_0}_{b_{k+1}} v^{b_{k+1}} - v^{b_k} - R^{b_k}_{c_0} g^{c_0} Δt - β̃
    r_beta = (R_bk_c0 @ R_c0_bk1 @ v_bk1_bk1
              - v_bk_bk
              - R_bk_c0 @ g_c0 * dt
              - beta_corr)   # (3,)

    # Eq 3.14:  r_α = R^{b_k}_{c_0}(p^{c_0}_{b_{k+1}} - R^{b_k}_{c_0} p^{c_0}_{b_k})
    #                  - v^{b_k}_{b_k} Δt - ½ R^{b_k}_{c_0} g^{c_0} Δt² - α̃
    r_alpha = (R_bk_c0 @ (p_c0_bk1 - p_c0_bk)
               - v_bk_bk * dt
               - 0.5 * R_bk_c0 @ g_c0 * dt**2
               - alpha_corr)   # (3,)

    # ── Eq 3.15: covariance — block-diagonal per §3.7.2 ─────────────────
    Sig_dR = Sigma_preint[0:3, 0:3]
    Sig_dv = Sigma_preint[3:6, 3:6]
    Sig_dp = Sigma_preint[6:9, 6:9]

    Jr_inv    = so3_right_jac_inv(r_gamma)
    Sig_rg    = Jr_inv @ Sig_dR @ Jr_inv.T
    Sigma_imu = torch.block_diag(Sig_rg, Sig_dv, Sig_dp)

    return r_gamma, r_beta, r_alpha, Sigma_imu


def imu_residual_without_learned_model(
    R_c0_bk:    torch.Tensor,
    R_c0_bk1:   torch.Tensor,
    p_c0_bk:    torch.Tensor,
    p_c0_bk1:   torch.Tensor,
    v_bk_bk:    torch.Tensor,
    v_bk1_bk1:  torch.Tensor,
    g_c0:       torch.Tensor,
    dt:         float,
    preint:     PreintegrationResult,
    delta_bg:   torch.Tensor,    # (3,) gyroscope bias increment
    delta_ba:   torch.Tensor,    # (3,) accelerometer bias increment (may be zero)
    Sigma_noise: torch.Tensor,   # (9,9) preintegration noise cov from calibrated params
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    IMU residual with first-order bias correction  (Eq 3.16–3.18).

    r_γ = Log(R^{c_0}_{b_{k+1}} R^{b_k}_{c_0} γ Exp(J^γ_{b_g} δb_g))
    r_β = ... - β - J^β_{b_g} δb_g - J^β_{b_a} δb_a
    r_α = ... - α - J^α_{b_g} δb_g - J^α_{b_a} δb_a
    """
    from geometry.lie_ops import so3_exp

    R_bk_c0 = R_c0_bk.T

    # First-order corrected rotation
    gamma_corr = preint.gamma @ so3_exp(preint.J_gamma_bg @ delta_bg)

    # Fix: same as learned-model case — R_c0_bk1.T @ R_c0_bk (thesis Eq A.21)
    r_gamma = so3_log(R_c0_bk1.T @ R_c0_bk @ gamma_corr)

    r_beta = (R_bk_c0 @ R_c0_bk1 @ v_bk1_bk1
              - v_bk_bk
              - R_bk_c0 @ g_c0 * dt
              - preint.beta
              - preint.J_beta_bg  @ delta_bg
              - preint.J_beta_ba  @ delta_ba)

    r_alpha = (R_bk_c0 @ (p_c0_bk1 - p_c0_bk)
               - v_bk_bk * dt
               - 0.5 * R_bk_c0 @ g_c0 * dt**2
               - preint.alpha
               - preint.J_alpha_bg @ delta_bg
               - preint.J_alpha_ba @ delta_ba)

    # Covariance — block-diagonal per §3.7.3, same structure as with-learned case.
    Sig_dR = Sigma_noise[0:3, 0:3]
    Sig_dv = Sigma_noise[3:6, 3:6]
    Sig_dp = Sigma_noise[6:9, 6:9]
    Jr_inv    = so3_right_jac_inv(r_gamma)
    Sig_rg    = Jr_inv @ Sig_dR @ Jr_inv.T
    Sigma_imu = torch.block_diag(Sig_rg, Sig_dv, Sig_dp)

    return r_gamma, r_beta, r_alpha, Sigma_imu
