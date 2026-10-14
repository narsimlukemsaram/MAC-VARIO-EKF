"""
Lie group operations for SO(3) and SE(3).

Numerically stable for all rotation angles including near-zero and near-pi.
All equations reference thesis Ch2/Ch3 notation.
"""

import torch

EPS      = 1e-8
NEAR_ZERO = 1e-6
NEAR_PI   = 3.14159265358979 - 1e-4


def skew(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) → (..., 3, 3) antisymmetric matrix."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([ z,          -v[..., 2],  v[..., 1]], dim=-1),
        torch.stack([ v[..., 2],   z,          -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1],   v[..., 0],   z        ], dim=-1),
    ], dim=-2)


def vee(S: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) → (..., 3): extract axial vector from skew matrix."""
    return torch.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], dim=-1)


def so3_exp(phi: torch.Tensor) -> torch.Tensor:
    """Exp: R^3 → SO(3).  phi: (..., 3) → R: (..., 3, 3)."""
    angle = phi.norm(dim=-1, keepdim=True).clamp(min=EPS)
    axis  = phi / angle
    K     = skew(axis)
    I     = torch.eye(3, device=phi.device, dtype=phi.dtype).expand_as(K)
    s     = angle.unsqueeze(-1).sin()
    c     = angle.unsqueeze(-1).cos()
    return I + s * K + (1.0 - c) * (K @ K)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """
    Log: SO(3) → R^3.  R: (..., 3, 3) → phi: (..., 3)

    Three stable branches:
      near-zero (θ<1e-6) : Taylor  φ ≈ vee(R-Rᵀ)/2
      normal               : vee((R-Rᵀ)·θ/(2sinθ))
      near-pi  (θ>π-1e-4) : symmetric-matrix eigenvector axis
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_a = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    angle = cos_a.acos()                                       # (...)

    # ── Branch 1: standard ──────────────────────────────────────
    safe_sin  = angle.sin().clamp(min=EPS)
    coeff     = angle / (2.0 * safe_sin)
    phi_std   = vee((R - R.transpose(-1, -2)) * coeff[..., None, None])

    # ── Branch 2: near-zero Taylor ──────────────────────────────
    phi_small = vee(R - R.transpose(-1, -2)) * 0.5

    # ── Branch 3: near-pi — axis from (R+Rᵀ)/2 ─────────────────
    # (R+Rᵀ)/2 = cosθ·I + (1-cosθ)·nnᵀ   →   nnᵀ = [(R+Rᵀ)/2 - cosθ·I]/(1-cosθ)
    # Diagonal of nnᵀ gives n_i² directly.
    one_minus_c = (1.0 - cos_a).clamp(min=EPS)                 # (...)
    sym_diag    = (R + R.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1) / 2.0  # (...,3)
    n2          = (sym_diag - cos_a[..., None]) / one_minus_c[..., None]      # (...,3)
    n2          = n2.clamp(min=0.0)
    axis_abs    = n2.sqrt()                                     # (...,3)  |n_i|

    # Sign: use sign of vee(R-Rᵀ) where available; product of cross-diagonal for off-diag
    skew_part   = vee(R - R.transpose(-1, -2))                  # (...,3)
    # Fix: adding 1e-12 prevents exact-zero collapse during backprop when true axis has
    # negative components near π — bare torch.sign returns 0 for exact zeros, flipping axis
    sign = torch.sign(skew_part + 1e-12)  # Prevents zero-collapse on exact pi
    axis_pi     = axis_abs * sign
    norm_pi     = axis_pi.norm(dim=-1, keepdim=True).clamp(min=EPS)
    axis_pi     = axis_pi / norm_pi
    phi_pi      = axis_pi * angle[..., None]

    # ── Select per element ────────────────────────────────────────
    small = (angle < NEAR_ZERO)[..., None]
    pi    = (angle > NEAR_PI  )[..., None]
    return torch.where(small, phi_small, torch.where(pi, phi_pi, phi_std))


def so3_right_jac(phi: torch.Tensor) -> torch.Tensor:
    """
    Right Jacobian J_r(φ): (..., 3, 3)
    J_r = I − (1−cosθ)/θ² φ^ + (θ−sinθ)/θ³ (φ^)²
    Near zero: J_r ≈ I − ½φ^
    """
    theta = phi.norm(dim=-1, keepdim=True).clamp(min=EPS)
    K, K2 = skew(phi), skew(phi) @ skew(phi)
    I  = torch.eye(3, device=phi.device, dtype=phi.dtype).expand_as(K)
    th = theta.squeeze(-1)
    c1 = ((1.0 - th.cos()) / th.pow(2))[..., None, None]
    c2 = ((th - th.sin())  / th.pow(3))[..., None, None]
    Jr = I - c1 * K + c2 * K2
    sm = (th < NEAR_ZERO)[..., None, None]
    return torch.where(sm, I - 0.5 * K, Jr)


def so3_right_jac_inv(phi: torch.Tensor) -> torch.Tensor:
    """
    Inverse right Jacobian J_r⁻¹(φ): (..., 3, 3)
    J_r⁻¹ = I + ½φ^ + (1/θ² − (1+cosθ)/(2θ sinθ)) (φ^)²
    Near zero: J_r⁻¹ ≈ I + ½φ^
    """
    theta = phi.norm(dim=-1, keepdim=True).clamp(min=EPS)
    K, K2 = skew(phi), skew(phi) @ skew(phi)
    I  = torch.eye(3, device=phi.device, dtype=phi.dtype).expand_as(K)
    th = theta.squeeze(-1)
    s  = th.sin().clamp(min=EPS)
    c2 = (1.0 / th.pow(2) - (1.0 + th.cos()) / (2.0 * th * s))[..., None, None]
    Jr_inv = I + 0.5 * K + c2 * K2
    sm = (th < NEAR_ZERO)[..., None, None]
    return torch.where(sm, I + 0.5 * K, Jr_inv)


def se3_left_jac_inv(xi: torch.Tensor) -> torch.Tensor:
    """
    Analytical J_l⁻¹(ξ) for ξ=[ρ,φ]∈R^6.
    (..., 6) → (..., 6, 6)

    Structure:
      J_l⁻¹(ξ) = [[J_l⁻¹(φ),  -J_l⁻¹(φ) @ Q @ J_l⁻¹(φ)],
                  [0,         J_l⁻¹(φ)]]
    where J_l⁻¹(φ) = J_r⁻¹(-φ).
    """
    # Extract translation rho and rotation phi
    rho = xi[..., :3]
    phi = xi[..., 3:]

    # 1. Compute J_l⁻¹(φ) = J_r⁻¹(-φ)
    Jl_inv = so3_right_jac_inv(-phi)

    # 2. Compute Q matrix
    theta = phi.norm(dim=-1, keepdim=True).clamp(min=EPS)
    th = theta.squeeze(-1)

    rx = skew(rho)
    px = skew(phi)

    px_rx = px @ rx
    rx_px = rx @ px
    px2 = px @ px
    px_rx_px = px_rx @ px

    # Components
    t1 = rx
    t2 = px_rx + rx_px + px_rx_px
    t3 = px2 @ rx + rx @ px2 - 3.0 * px_rx_px
    t4 = px_rx_px @ px + px2 @ px_rx

    th2 = th.pow(2)
    th3 = th.pow(3)
    th4 = th.pow(4)
    th5 = th.pow(5)
    s = th.sin()
    c = th.cos()

    m1 = 0.5
    m2 = ((th - s) / th3)[..., None, None]
    m3 = ((0.5 * th2 + c - 1.0) / th4)[..., None, None]
    m4 = ((th - 1.5 * s + 0.5 * th * c) / th5)[..., None, None]

    Q = m1 * t1 + m2 * t2 + m3 * t3 + m4 * t4

    # Handle singularity when theta is near zero
    sm = (th < NEAR_ZERO)[..., None, None]
    Q = torch.where(sm, 0.5 * rx, Q)

    # 3. Assemble block matrix J_l⁻¹
    # Top-right block: -Jl_inv @ Q @ Jl_inv
    top_right = -Jl_inv @ Q @ Jl_inv

    Z = torch.zeros_like(Jl_inv)
    r0 = torch.cat([Jl_inv, top_right], dim=-1)
    r1 = torch.cat([Z,       Jl_inv], dim=-1)
    return torch.cat([r0, r1], dim=-2)



def se3_exp(xi: torch.Tensor) -> tuple:
    # xi=[rho,phi] ∈ R^6 → (R,t) ∈ SE(3). rho=translation part, phi=rotation part.
    phi = xi[3:]; rho = xi[:3]
    R = so3_exp(phi)
    Jr = so3_right_jac(phi)
    t = Jr @ rho
    return R, t


def se3_log(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # (R,t) ∈ SE(3) → xi=[rho,phi] ∈ R^6
    phi = so3_log(R)
    Jr_inv = so3_right_jac_inv(phi)
    rho = Jr_inv @ t
    return torch.cat([rho, phi])


def se3_adjoint(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # Ad_T for T=(R,t): 6x6 matrix acting on xi=[rho,phi]
    # Ad_T = [[R, [t]xR], [0, R]]  (Eq A.1, translation-first convention)
    tx = skew(t)
    Z = torch.zeros_like(R)
    row0 = torch.cat([R,    tx @ R], dim=-1)
    row1 = torch.cat([Z,    R     ], dim=-1)
    return torch.cat([row0, row1], dim=-2)