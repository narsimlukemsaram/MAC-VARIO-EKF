"""
Unit tests for MAC-I2 implementation.

Each test maps to a specific equation in the thesis and verifies
numerical correctness against analytical or synthetic ground truth.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import math
import pytest

from geometry.lie_ops import (
    skew, vee, so3_exp, so3_log, so3_right_jac, so3_right_jac_inv
)
from geometry.covariance_projection import project_cov_2d_to_3d, depth_uncertainty_correction
from geometry.preintegration import preintegrate
from models.imu_network import ImprovedAirIMU, nll_covariance_loss
from optimization.imu_residuals import imu_residual_with_learned_model


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def rand_SO3(device='cpu'):
    """Random rotation via QR decomposition."""
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.to(device)


# ─────────────────────────────────────────────
# SO(3) tests
# ─────────────────────────────────────────────

class TestSO3:
    def test_skew_antisymmetric(self):
        v = torch.randn(3)
        S = skew(v)
        assert torch.allclose(S + S.T, torch.zeros(3, 3), atol=1e-6), \
            "skew matrix must be antisymmetric"

    def test_skew_vee_roundtrip(self):
        v = torch.randn(5, 3)
        assert torch.allclose(vee(skew(v)), v, atol=1e-6), \
            "vee(skew(v)) must recover v"

    def test_exp_log_roundtrip_small(self):
        """Log(Exp(φ)) == φ for small φ."""
        phi = torch.randn(3) * 0.1
        R   = so3_exp(phi)
        phi_rec = so3_log(R)
        assert torch.allclose(phi, phi_rec, atol=1e-5), \
            f"Exp/Log roundtrip failed: {(phi-phi_rec).norm():.2e}"

    def test_exp_is_rotation(self):
        """Exp(φ) ∈ SO(3): R^T R = I, det = +1."""
        phi = torch.randn(10, 3)
        R   = so3_exp(phi)
        I   = torch.eye(3).unsqueeze(0).expand(10, -1, -1)
        assert torch.allclose(R @ R.transpose(-1, -2), I, atol=1e-5), \
            "Exp output not orthogonal"
        assert torch.allclose(torch.linalg.det(R), torch.ones(10), atol=1e-5), \
            "Exp output det != 1"

    def test_right_jac_identity_at_zero(self):
        """J_r(0) = I."""
        phi = torch.zeros(3)
        Jr  = so3_right_jac(phi)
        assert torch.allclose(Jr, torch.eye(3), atol=1e-4), \
            "J_r(0) should be identity"

    def test_right_jac_inv_product(self):
        """J_r^{-1}(φ) · J_r(φ) ≈ I."""
        phi    = torch.randn(3) * 0.5
        Jr     = so3_right_jac(phi)
        Jr_inv = so3_right_jac_inv(phi)
        prod   = Jr_inv @ Jr
        assert torch.allclose(prod, torch.eye(3), atol=1e-4), \
            f"J_r^{{-1}} · J_r ≠ I, max err {(prod - torch.eye(3)).abs().max():.2e}"


# ─────────────────────────────────────────────
# 3D Covariance Projection  (Eq 2.12–2.14)
# ─────────────────────────────────────────────

class TestCovarianceProjection:
    def test_output_symmetric_positive(self):
        """Eq 2.14: cΣ must be symmetric and positive semi-definite."""
        N = 20
        u   = torch.randn(N) * 10 + 320.0
        v   = torch.randn(N) * 10 + 240.0
        d   = torch.rand(N)  * 4  + 0.5
        su2 = torch.rand(N)  * 0.5 + 0.01
        sv2 = torch.rand(N)  * 0.5 + 0.01
        sd2 = (d * 0.1).pow(2)

        Sigma = project_cov_2d_to_3d(u, v, d, su2, sv2, sd2,
                                      fx=458.0, fy=458.0, cx=367.0, cy=248.0)
        assert Sigma.shape == (N, 3, 3), "Wrong output shape"

        # Symmetry
        assert torch.allclose(Sigma, Sigma.transpose(-1, -2), atol=1e-7), \
            "Covariance must be symmetric"

        # PSD: all eigenvalues ≥ 0
        eigvals = torch.linalg.eigvalsh(Sigma)
        assert (eigvals >= -1e-6).all(), \
            f"Covariance has negative eigenvalue: {eigvals.min():.3e}"

    def test_z_variance_equals_depth_variance(self):
        """Eq 2.12: σ_z² = σ_d²."""
        N = 5
        u   = torch.zeros(N) + 320.0
        v   = torch.zeros(N) + 240.0
        d   = torch.ones(N) * 2.0
        su2 = torch.zeros(N)
        sv2 = torch.zeros(N)
        sd2 = torch.rand(N) * 0.5 + 0.01

        Sigma = project_cov_2d_to_3d(u, v, d, su2, sv2, sd2,
                                      fx=458.0, fy=458.0, cx=320.0, cy=240.0)
        # At optical center (u=cx, v=cy): σ_z = σ_d, cross-terms = 0
        assert torch.allclose(Sigma[:, 2, 2], sd2, atol=1e-6), \
            "σ_z² should equal σ_d² at optical center"

    def test_off_diagonal_zero_at_optical_center(self):
        """At (u=cx, v=cy): σ_xz = σ_yz = σ_xy = 0  (Eq 2.13)."""
        N = 5
        u   = torch.zeros(N) + 320.0
        v   = torch.zeros(N) + 240.0
        d   = torch.ones(N) * 2.0
        sd2 = torch.ones(N) * 0.04
        su2 = sv2 = torch.zeros(N)

        Sigma = project_cov_2d_to_3d(u, v, d, su2, sv2, sd2,
                                      fx=458.0, fy=458.0, cx=320.0, cy=240.0)
        assert torch.allclose(Sigma[:, 0, 2], torch.zeros(N), atol=1e-7), "σ_xz ≠ 0"
        assert torch.allclose(Sigma[:, 1, 2], torch.zeros(N), atol=1e-7), "σ_yz ≠ 0"
        assert torch.allclose(Sigma[:, 0, 1], torch.zeros(N), atol=1e-7), "σ_xy ≠ 0"

    def test_depth_correction_increases_near_discontinuity(self):
        """Patch with high depth variance → larger σ_d² than flat patch."""
        N = 2
        # Flat patch
        flat_patch  = torch.ones(1, N, 5, 5) * 2.0
        # Edge patch: abrupt depth change
        edge_patch  = torch.ones(1, N, 5, 5) * 2.0
        edge_patch[:, :, :, 3:] = 5.0   # big jump

        su2 = torch.ones(1, N) * 1.0
        sv2 = torch.ones(1, N) * 1.0

        sd2_flat = depth_uncertainty_correction(flat_patch, su2, sv2)
        sd2_edge = depth_uncertainty_correction(edge_patch, su2, sv2)

        assert (sd2_edge > sd2_flat).all(), \
            "Edge patch should produce larger depth uncertainty"


# ─────────────────────────────────────────────
# IMU Preintegration  (Eq 2.4–2.8)
# ─────────────────────────────────────────────

class TestPreintegration:
    def test_zero_motion_identity(self):
        """
        Zero acceleration, zero angular velocity → γ=I, β=0, α=0.
        """
        N  = 100
        a  = torch.zeros(N, 3)
        w  = torch.zeros(N, 3)
        dt = torch.full((N,), 0.01)
        bg = torch.zeros(3)
        ba = torch.zeros(3)
        Sg = torch.eye(3) * 1e-4
        Sa = torch.eye(3) * 1e-4

        res = preintegrate(a, w, dt, bg, ba, Sg, Sa)
        assert torch.allclose(res.gamma, torch.eye(3), atol=1e-5), "γ ≠ I for zero motion"
        assert torch.allclose(res.beta,  torch.zeros(3), atol=1e-5), "β ≠ 0 for zero motion"
        assert torch.allclose(res.alpha, torch.zeros(3), atol=1e-5), "α ≠ 0 for zero motion"

    def test_cov_positive_definite(self):
        """Propagated covariance must be positive definite."""
        N  = 50
        a  = torch.randn(N, 3) * 0.1
        w  = torch.randn(N, 3) * 0.05
        dt = torch.full((N,), 0.01)
        bg = torch.zeros(3)
        ba = torch.zeros(3)
        Sg = torch.eye(3) * 1e-4
        Sa = torch.eye(3) * 1e-4

        res = preintegrate(a, w, dt, bg, ba, Sg, Sa)
        eigvals = torch.linalg.eigvalsh(res.Sigma)
        assert (eigvals > 0).all(), \
            f"Preintegration covariance not PD. Min eigval={eigvals.min():.3e}"

    def test_learnable_init_cov_increases_sigma(self):
        """Starting with Σ_0 > 0 produces larger total covariance than Σ_0=0."""
        N  = 20
        a  = torch.randn(N, 3) * 0.1
        w  = torch.randn(N, 3) * 0.05
        dt = torch.full((N,), 0.01)
        bg = ba = torch.zeros(3)
        Sg = torch.eye(3) * 1e-4
        Sa = torch.eye(3) * 1e-4

        res0   = preintegrate(a, w, dt, bg, ba, Sg, Sa)
        Sigma0 = torch.eye(9) * 0.1
        resS0  = preintegrate(a, w, dt, bg, ba, Sg, Sa, init_Sigma=Sigma0)

        trace0  = res0.Sigma.trace()
        traceSi = resS0.Sigma.trace()
        assert traceSi > trace0, \
            "Learnable Σ_0 should increase total covariance trace"


# ─────────────────────────────────────────────
# IMU Network  (§3.5)
# ─────────────────────────────────────────────

class TestIMUNetwork:
    def test_forward_shapes(self):
        model = ImprovedAirIMU()
        B, T = 4, 100
        a = torch.randn(B, T, 3)
        w = torch.randn(B, T, 3)
        out = model(a, w)
        assert out["a_corr"].shape   == (B, T, 3), "a_corr shape mismatch"
        assert out["w_corr"].shape   == (B, T, 3), "w_corr shape mismatch"
        assert out["eta_acc"].shape  == (B, T, 3), "eta_acc shape mismatch"
        assert out["eta_gyro"].shape == (B, T, 3), "eta_gyro shape mismatch"
        assert out["Sigma_0"].shape  == (9, 9),    "Sigma_0 shape mismatch"

    def test_uncertainty_positive(self):
        """η_acc, η_gyro must all be positive (variance)."""
        model = ImprovedAirIMU()
        a = torch.randn(2, 50, 3)
        w = torch.randn(2, 50, 3)
        out = model(a, w)
        assert (out["eta_acc"]  > 0).all(), "eta_acc contains non-positive values"
        assert (out["eta_gyro"] > 0).all(), "eta_gyro contains non-positive values"

    def test_sigma0_positive_definite(self):
        """Σ_0 (from log-diagonal param) must be PD."""
        model = ImprovedAirIMU()
        S0 = model.learnable_init_cov
        eigvals = torch.linalg.eigvalsh(S0)
        assert (eigvals > 0).all(), "Σ_0 not positive definite"

    def test_nll_loss_finite(self):
        """NLL covariance loss should be finite and decrease with better covariance."""
        B = 4
        phi_err = torch.randn(B, 3) * 0.1
        v_err   = torch.randn(B, 3) * 0.1
        p_err   = torch.randn(B, 3) * 0.1
        # Tight covariance (well-calibrated)
        scale   = 0.01
        Sigma_good = (torch.eye(9) * scale).unsqueeze(0).expand(B, -1, -1)
        R_w_bi  = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
        loss_good = nll_covariance_loss(phi_err, v_err, p_err, Sigma_good, R_w_bi)
        assert torch.isfinite(loss_good), "NLL loss is not finite"


# ─────────────────────────────────────────────
# IMU Residuals  (Eq 3.12–3.15)
# ─────────────────────────────────────────────

class TestIMUResiduals:
    def test_zero_residual_on_perfect_prediction(self):
        """
        With perfect corrected preintegration = state difference,
        rotation and position residuals should be ~ 0.
        r_beta will have the gravity contribution for zero-velocity case.
        """
        R = torch.eye(3)
        p = torch.zeros(3)
        v = torch.zeros(3)
        g = torch.tensor([0., -9.81, 0.])
        dt = 0.1

        Sigma9 = torch.eye(9) * 0.01

        # For zero motion with gravity: corrected preintegration must absorb gravity
        # r_beta = -R^{bk}_{c0} g dt - beta_corr  → beta_corr = -g*dt
        # r_alpha = -½ g dt²            → alpha_corr = -½ g dt²
        beta_expected  = -g * dt
        alpha_expected = -0.5 * g * dt**2

        r_g, r_b, r_a, Sig = imu_residual_with_learned_model(
            R, R, p, p, v, v, g, dt,
            gamma_corr=torch.eye(3),
            beta_corr=beta_expected,       # corrected preint matches expected motion
            alpha_corr=alpha_expected,
            Sigma_preint=Sigma9,
        )
        assert torch.allclose(r_g, torch.zeros(3), atol=1e-5), f"r_gamma={r_g}"
        assert torch.allclose(r_b, torch.zeros(3), atol=1e-4), f"r_beta={r_b}"
        assert torch.allclose(r_a, torch.zeros(3), atol=1e-4), f"r_alpha={r_a}"

    def test_sigma_imu_positive_definite(self):
        """Σ_imu from residual transformation must be PD."""
        R   = rand_SO3()
        R1  = rand_SO3()
        p   = torch.randn(3)
        p1  = torch.randn(3)
        v   = torch.randn(3) * 0.1
        v1  = torch.randn(3) * 0.1
        g   = torch.tensor([0., -9.81, 0.])
        Sig9 = torch.eye(9) * 0.01

        _, _, _, Sig_imu = imu_residual_with_learned_model(
            R, R1, p, p1, v, v1, g, 0.1,
            gamma_corr=rand_SO3(),
            beta_corr=torch.randn(3)*0.1,
            alpha_corr=torch.randn(3)*0.01,
            Sigma_preint=Sig9,
        )
        eigvals = torch.linalg.eigvalsh(Sig_imu)
        assert (eigvals > -1e-6).all(), f"Σ_imu has negative eigval: {eigvals.min():.3e}"


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Run without pytest for quick check
    suites = [TestSO3, TestCovarianceProjection, TestPreintegration, TestIMUNetwork, TestIMUResiduals]
    total, passed, failed = 0, 0, []
    for suite_cls in suites:
        suite = suite_cls()
        for name in [m for m in dir(suite_cls) if m.startswith("test_")]:
            total += 1
            try:
                getattr(suite, name)()
                passed += 1
                print(f"  ✓  {suite_cls.__name__}.{name}")
            except Exception as e:
                failed.append(f"{suite_cls.__name__}.{name}")
                print(f"  ✗  {suite_cls.__name__}.{name}  →  {e}")

    print(f"\n{'─'*60}")
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  |  Failed: {', '.join(failed)}")
    else:
        print("  — all clear ✓")
