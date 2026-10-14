"""
Metrics-aware 3D covariance model  (thesis §2.4.4)

Transforms per-keypoint 2D pixel uncertainty (σ_u², σ_v²) + depth uncertainty
σ_d² into the full 3×3 covariance matrix  cΣ_i  in the camera frame.

Equations implemented:
  Eq 2.11 : pinhole unprojection  x, y, z
  Eq 2.12 : diagonal covariance terms σ_x², σ_y², σ_z²
  Eq 2.13 : off-diagonal terms σ_xz, σ_yz, σ_xy
  Eq 2.14 : assemble full 3×3 Σ
  Eq 2.15 : depth-discontinuity correction  σ_d² (patch-based)
"""

import torch
from typing import Optional


# ─────────────────────────────────────────────
# Core covariance projection  (Eq 2.11 – 2.14)
# ─────────────────────────────────────────────

def project_cov_2d_to_3d(
    u:   torch.Tensor,   # (..., N)  pixel x
    v:   torch.Tensor,   # (..., N)  pixel y
    d:   torch.Tensor,   # (..., N)  depth [m]
    su2: torch.Tensor,   # (..., N)  σ_u²
    sv2: torch.Tensor,   # (..., N)  σ_v²
    sd2: torch.Tensor,   # (..., N)  σ_d²
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> torch.Tensor:
    """
    Returns full 3×3 covariance matrices  cΣ_i  for N keypoints.
    Output shape: (..., N, 3, 3)

    All computations use the exact variance-of-products formula
    for independent (u, v, d) random variables.
    """
    dx = u - cx        # (u - c_x)
    dy = v - cy        # (v - c_y)

    # ── Eq 2.12: diagonal terms ──────────────────────────────────
    # σ_x² = (σ_u² σ_d² + σ_u² d² + (u-cx)² σ_d²) / fx²
    sx2 = (su2 * sd2  +  su2 * d.pow(2)  +  dx.pow(2) * sd2) / (fx * fx)
    # σ_y² = (σ_v² σ_d² + σ_v² d² + (v-cy)² σ_d²) / fy²
    sy2 = (sv2 * sd2  +  sv2 * d.pow(2)  +  dy.pow(2) * sd2) / (fy * fy)
    # σ_z² = σ_d²
    sz2 = sd2

    # ── Eq 2.13: off-diagonal terms ──────────────────────────────
    # σ_xz = σ_d² (u-cx) / fx
    sxz = sd2 * dx / fx
    # σ_yz = σ_d² (v-cy) / fy
    syz = sd2 * dy / fy
    # σ_xy = σ_d² (u-cx)(v-cy) / (fx fy)
    sxy = sd2 * dx * dy / (fx * fy)

    # ── Eq 2.14: assemble symmetric 3×3 matrix ───────────────────
    #   [[sx2, sxy, sxz],
    #    [sxy, sy2, syz],
    #    [sxz, syz, sz2]]
    zeros = torch.zeros_like(sx2)
    # rows
    row0 = torch.stack([sx2, sxy, sxz], dim=-1)   # (..., N, 3)
    row1 = torch.stack([sxy, sy2, syz], dim=-1)
    row2 = torch.stack([sxz, syz, sz2], dim=-1)
    Sigma = torch.stack([row0, row1, row2], dim=-2)  # (..., N, 3, 3)
    return Sigma


# ─────────────────────────────────────────────
# Depth discontinuity correction  (Eq 2.15)
# ─────────────────────────────────────────────

def depth_uncertainty_correction(
    depth_patch: torch.Tensor,   # (..., N, H_p, W_p)  local depth patch
    su2: torch.Tensor,            # (..., N)  σ_u²
    sv2: torch.Tensor,            # (..., N)  σ_v²
) -> torch.Tensor:
    """
    Corrected depth uncertainty  σ_d²  using local depth variation.

    Eq 2.15:  σ_d²_i = Σ_j  φ_j (d_j - μ_{D_i})²

    where φ is a 2D Gaussian kernel with covariance matching (σ_u², σ_v²).

    depth_patch : local patch around each keypoint (size ~3×3 up to 32×32)
    Returns:  (..., N)  corrected depth variance
    """
    *B, N, Hp, Wp = depth_patch.shape

    # ── Build Gaussian weights φ  ────────────────────────────────
    cy_p = (Hp - 1) / 2.0
    cx_p = (Wp - 1) / 2.0
    yi = torch.arange(Hp, device=depth_patch.device, dtype=depth_patch.dtype) - cy_p
    xi = torch.arange(Wp, device=depth_patch.device, dtype=depth_patch.dtype) - cx_p
    grid_y, grid_x = torch.meshgrid(yi, xi, indexing='ij')   # (Hp, Wp)

    # φ_j = exp(-0.5*(x²/σ_u² + y²/σ_v²))
    # su2/sv2 shape (..., N) → (..., N, 1, 1)
    su2_e = su2.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-6)
    sv2_e = sv2.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-6)
    # broadcast grid over batch+keypoint dims
    gx = grid_x.expand(*B, N, Hp, Wp)
    gy = grid_y.expand(*B, N, Hp, Wp)
    phi = torch.exp(-0.5 * (gx.pow(2) / su2_e + gy.pow(2) / sv2_e))
    phi = phi / phi.sum(dim=(-1, -2), keepdim=True).clamp(min=1e-12)  # normalise

    # ── Weighted mean  μ  ────────────────────────────────────────
    mu = (phi * depth_patch).sum(dim=(-1, -2))   # (..., N)

    # ── Weighted variance  Eq 2.15  ──────────────────────────────
    diff2 = (depth_patch - mu.unsqueeze(-1).unsqueeze(-1)).pow(2)
    sd2   = (phi * diff2).sum(dim=(-1, -2))       # (..., N)
    return sd2
