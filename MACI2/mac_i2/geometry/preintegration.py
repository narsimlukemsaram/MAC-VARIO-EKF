"""
On-manifold IMU preintegration (Forster et al. TRO 2017).

Implements Eq 2.4-2.8 from thesis:
  - Preintegration terms: γ (rotation), β (velocity), α (position)
  - Covariance propagation: Σ_{k+1} = A Σ_k A^T + B diag(Σ_g, Σ_a) B^T
  - Bias Jacobians for first-order correction

SIGN CONVENTION for bias Jacobians:
  Following Forster TRO17 Appendix and VINS-Mono:
    J_gamma_bg_{k+1} = ΔR_{k,k+1}^T * J_gamma_bg_k  -  Jr_k * dt_k   (NEGATIVE)
  This ensures:  γ(bg + δbg) ≈ γ * Exp(J_gamma_bg * δbg)
"""

import torch
from geometry.lie_ops import so3_exp, so3_log, so3_right_jac, skew
from dataclasses import dataclass
from typing import Optional


@dataclass
class PreintegrationResult:
    gamma:      torch.Tensor   # (3,3)  ΔR  rotation preintegration
    beta:       torch.Tensor   # (3,)   Δv  velocity preintegration
    alpha:      torch.Tensor   # (3,)   Δp  position preintegration
    Sigma:      torch.Tensor   # (9,9)  covariance [γ(0:3), β(3:6), α(6:9)]
    J_gamma_bg: torch.Tensor   # (3,3)  ∂γ/∂bg
    J_beta_bg:  torch.Tensor   # (3,3)  ∂β/∂bg
    J_beta_ba:  torch.Tensor   # (3,3)  ∂β/∂ba
    J_alpha_bg: torch.Tensor   # (3,3)  ∂α/∂bg
    J_alpha_ba: torch.Tensor   # (3,3)  ∂α/∂ba
    dt_total:   float


def preintegrate(
    a_meas:    torch.Tensor,       # (N,3) raw accelerometer  â_k
    w_meas:    torch.Tensor,       # (N,3) raw gyroscope      ω̂_k
    dt:        torch.Tensor,       # (N,)  time steps Δt
    bg:        torch.Tensor,       # (3,)  gyroscope bias
    ba:        torch.Tensor,       # (3,)  accelerometer bias
    Sigma_g:   torch.Tensor,       # (3,3) gyroscope noise cov
    Sigma_a:   torch.Tensor,       # (3,3) accelerometer noise cov
    init_Sigma: Optional[torch.Tensor] = None,   # (9,9) initial Σ_0
) -> PreintegrationResult:
    """
    Integrate N raw IMU measurements and propagate covariance.
    State order in 9×9 covariance: [γ(0:3), β(3:6), α(6:9)]
    """
    device, dtype = a_meas.device, a_meas.dtype
    N = a_meas.shape[0]

    a_corr = a_meas - ba.unsqueeze(0)    # â - ba
    w_corr = w_meas - bg.unsqueeze(0)    # ω̂ - bg

    gamma = torch.eye(3, device=device, dtype=dtype)
    beta  = torch.zeros(3, device=device, dtype=dtype)
    alpha = torch.zeros(3, device=device, dtype=dtype)

    Sigma = init_Sigma.clone() if init_Sigma is not None else \
            torch.zeros(9, 9, device=device, dtype=dtype)

    # Bias Jacobians — SIGN: J_gamma_bg uses NEGATIVE Jr (Forster TRO17 sign convention)
    J_gamma_bg = torch.zeros(3, 3, device=device, dtype=dtype)
    J_beta_bg  = torch.zeros(3, 3, device=device, dtype=dtype)
    J_beta_ba  = torch.zeros(3, 3, device=device, dtype=dtype)
    J_alpha_bg = torch.zeros(3, 3, device=device, dtype=dtype)
    J_alpha_ba = torch.zeros(3, 3, device=device, dtype=dtype)

    # R_{b_i}^{b_k} — accumulated body rotation from start to current step
    R_ik = torch.eye(3, device=device, dtype=dtype)

    I3  = torch.eye(3, device=device, dtype=dtype)
    Z33 = torch.zeros(3, 3, device=device, dtype=dtype)
    noise_cov = torch.block_diag(Sigma_g, Sigma_a)

    for k in range(N):
        dt_k = dt[k]
        w_k  = w_corr[k]
        a_k  = a_corr[k]

        dR   = so3_exp(w_k * dt_k)                    # ΔR_{k,k+1}
        Jr_k = so3_right_jac(w_k * dt_k)              # J_r(ω̂-bg)*Δt

        # ── Eq 2.7: state transition A ──────────────────────────
        ak_skew = skew(a_k)
        A_row0  = torch.cat([dR.T,                              Z33,        Z33], dim=1)
        A_row1  = torch.cat([-R_ik @ ak_skew * dt_k,           I3,         Z33], dim=1)
        A_row2  = torch.cat([-0.5*R_ik @ ak_skew * dt_k**2,    I3*dt_k,    I3 ], dim=1)
        A = torch.cat([A_row0, A_row1, A_row2], dim=0)         # (9,9)

        # ── Eq 2.8: noise input B ────────────────────────────────
        Bg_col = torch.cat([Jr_k * dt_k,   Z33,             Z33            ], dim=0)
        Ba_col = torch.cat([Z33,            R_ik * dt_k,     0.5*R_ik*dt_k**2], dim=0)
        B = torch.cat([Bg_col, Ba_col], dim=1)                 # (9,6)

        # ── Eq 2.6: covariance propagation ──────────────────────
        Sigma = A @ Sigma @ A.T + B @ noise_cov @ B.T

        # ── Eq 2.4: preintegration update ───────────────────────
        alpha = alpha + beta * dt_k + 0.5 * R_ik @ a_k * dt_k**2
        beta  = beta  + R_ik @ a_k * dt_k
        gamma = gamma @ dR

        # ── Bias Jacobian recursion (VINS-Mono / Forster TRO17 sign) ──
        # J_gamma_bg_{k+1} = ΔR^T * J_gamma_bg_k  - Jr_k * dt_k   ← NEGATIVE Jr
        J_alpha_bg = J_alpha_bg + J_beta_bg * dt_k \
                     - 0.5 * R_ik @ ak_skew @ J_gamma_bg * dt_k**2
        J_alpha_ba = J_alpha_ba + J_beta_ba * dt_k + 0.5 * R_ik * dt_k**2
        J_beta_bg  = J_beta_bg  - R_ik @ ak_skew @ J_gamma_bg * dt_k
        J_beta_ba  = J_beta_ba  + R_ik * dt_k
        J_gamma_bg = dR.T @ J_gamma_bg - Jr_k * dt_k            # ← FIXED SIGN

        R_ik = R_ik @ dR

    return PreintegrationResult(
        gamma=gamma, beta=beta, alpha=alpha, Sigma=Sigma,
        J_gamma_bg=J_gamma_bg, J_beta_bg=J_beta_bg, J_beta_ba=J_beta_ba,
        J_alpha_bg=J_alpha_bg, J_alpha_ba=J_alpha_ba,
        dt_total=float(dt.sum().item()),
    )


def apply_bias_correction(
    result: PreintegrationResult,
    delta_bg: torch.Tensor,    # (3,)
    delta_ba: torch.Tensor,    # (3,)
) -> PreintegrationResult:
    """First-order bias correction without re-integration."""
    gamma_corr = result.gamma @ so3_exp(result.J_gamma_bg @ delta_bg)
    beta_corr  = result.beta  + result.J_beta_bg  @ delta_bg + result.J_beta_ba  @ delta_ba
    alpha_corr = result.alpha + result.J_alpha_bg @ delta_bg + result.J_alpha_ba @ delta_ba
    import dataclasses
    return dataclasses.replace(result, gamma=gamma_corr, beta=beta_corr, alpha=alpha_corr)
