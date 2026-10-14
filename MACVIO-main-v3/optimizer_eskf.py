"""
optimizer_eskf.py

EKF-aware extension of the existing two-frame VIO optimiser.

The validated TwoFrameVIOOptimizer used by Phases B/C/D is left
untouched. This module will be used only by the new
MAC-VARIO-EKF experimental path.

Final optimisation increment:

    dz = [
        dxi_rel,      # 6
        dv_k,         # 3
        dv_k1,        # 3
        dba,           # 3
        dbg            # 3
    ]

Total: 18 DoF

The previous pose is fixed by the recursive ES-EKF. The two endpoint
velocities and the interval bias variables are jointly refined so the
preintegrated inertial factor has the same endpoint-velocity structure
as the validated two-frame VIO formulation.

The ES-EKF prior itself remains a 15-D navigation-state prior:

    dx_ekf = [
        dp, dv, dtheta, dba, dbg
    ]
"""

from dataclasses import dataclass

import torch

# Commented out by Narsimlu
# from optimizer import (
#     TwoFrameVIOOptimizer,
#     WorldState,
#     VisualFactor,
#     IMUFactor,
#     OptimResult,
# )

# Added by Narsimlu
from optimizer import (
    TwoFrameVIOOptimizer,
    WorldState,
    VisualFactor,
    IMUFactor,
    OptimResult,
    _se3_from_mat4,
    _se3_mat,
    _se3_inv_mat,
    _apply,
    _skew,
    _skew_batch,
    _so3_exp,
    _so3_log,
    _svd_so3,
    _jl_inv_so3,
    _jr_inv_so3,
)

# Added by Narsimlu
from preintegration import (
    EUROC_SIGMA_BA,
    EUROC_SIGMA_BG,
)

# Added by Narsimlu
@dataclass
class ESKFWorldState:
    """
    Previous recursive-filter state at image time k.

    R convention:
        world <- body
    """
    p: torch.Tensor
    v: torch.Tensor
    R: torch.Tensor
    b_a: torch.Tensor
    b_g: torch.Tensor

@dataclass
class ESKFPrior:
    """
    ES-EKF prediction supplied to the image-rate optimiser.

    All quantities correspond to time k+1.

    p   : predicted body position in world, (3,)
    v   : predicted body velocity in world, (3,)
    R   : predicted rotation world <- body, (3,3)
    b_a : predicted accelerometer bias, (3,)
    b_g : predicted gyroscope bias, (3,)
    P   : 15x15 ES-EKF error covariance

    Error-state order:
        [dp, dv, dtheta, dba, dbg]
    """

    p: torch.Tensor
    v: torch.Tensor
    R: torch.Tensor
    b_a: torch.Tensor
    b_g: torch.Tensor
    P: torch.Tensor

    # Prior tempering factor from the manuscript.
    eta: float = 1.0


@dataclass
class ESKFOptimResult:
    """
    Result returned by the 18-DoF EKF-aware optimiser.
    """

    base: OptimResult

    b_a_k1: torch.Tensor
    b_g_k1: torch.Tensor

    # Navigation-state correction to be injected into ES-EKF:
    # [dp, dv, dtheta, dba, dbg]
    delta_nav: torch.Tensor

    # Optional approximate posterior information matrix in optimizer
    # coordinates and covariance in the ESKF navigation-error coordinates.
    H_nav: object = None
    P_nav: object = None


class TwoFrameESKFOptimizer(TwoFrameVIOOptimizer):
    """
    MAC-VARIO-EKF image-rate nonlinear optimiser.

    The MAC-VARIO-EKF path uses:

      1. bias-corrected IMU preintegration,
      2. an 18-DoF two-frame nonlinear state
         [relative pose, v_k, v_k+1, b_a, b_g],
      3. a tempered 15-D ES-EKF prediction prior at k+1, and
      4. a 15-D navigation correction/covariance for recursive feedback.

    The validated base TwoFrameVIOOptimizer used by Phases B/C/D remains
    untouched. Legacy 15-D residual/Jacobian helpers are retained below
    only for regression diagnostics; the public EKF-aware entry points
    dispatch to the 18-DoF solver.
    """

    def __init__(
        self,
        g_w=(0.0, 0.0, -9.81),
        imu_weight=1.0,
        max_iter=15,
        tol=1e-6,
        rel_tol=1e-5,
        huber_delta=1.0,
        lm_lambda=1e-4,
        clamp_cov=True,
        device="cpu",
        prior_eta=1.0,
    ):
        super().__init__(
            g_w=g_w,
            imu_weight=imu_weight,
            max_iter=max_iter,
            tol=tol,
            huber_delta=huber_delta,
            lm_lambda=lm_lambda,
            clamp_cov=clamp_cov,
            device=device,
        )

        self.prior_eta = float(prior_eta)
        self.rel_tol = float(rel_tol)


    def optimize_with_eskf_prior(
        self,
        state_k: ESKFWorldState,
        vis: VisualFactor,
        imu: IMUFactor,
        eskf_prior: ESKFPrior,
    ):
        """Full MAC-VARIO-EKF optimisation using the 18-DoF solver."""
        return self.optimize_18d_with_prior(
            state_k,
            vis,
            imu,
            eskf_prior,
        )

    def _has_bias_jacobians(self, imu: IMUFactor) -> bool:
        """Return True when first-order preintegration bias Jacobians exist."""
        names = (
            "bias_ref_ba",
            "bias_ref_bg",
            "J_gamma_bg",
            "J_beta_ba",
            "J_beta_bg",
            "J_alpha_ba",
            "J_alpha_bg",
        )
        return all(getattr(imu, name, None) is not None for name in names)

    def _bias_correct_preintegration(
        self,
        imu: IMUFactor,
        state_k: ESKFWorldState,
        b_a_candidate: torch.Tensor = None,
        b_g_candidate: torch.Tensor = None,
    ):
        """
        First-order bias correction of the preintegrated IMU quantities.

        If candidate biases are supplied they are used as the current
        optimisation variables. With no candidates, this falls back to
        the recursive state at time k, preserving the legacy behaviour.
        """
        gamma_R = imu.gamma.matrix().squeeze().double()
        beta = imu.beta.double()
        alpha = imu.alpha.double()

        if not self._has_bias_jacobians(imu):
            return gamma_R, beta, alpha

        ba_ref = imu.bias_ref_ba.double()
        bg_ref = imu.bias_ref_bg.double()

        if b_a_candidate is None:
            b_a_candidate = state_k.b_a
        if b_g_candidate is None:
            b_g_candidate = state_k.b_g

        dba = b_a_candidate.double() - ba_ref
        dbg = b_g_candidate.double() - bg_ref

        beta_corr = (
            beta
            + imu.J_beta_ba.double() @ dba
            + imu.J_beta_bg.double() @ dbg
        )

        alpha_corr = (
            alpha
            + imu.J_alpha_ba.double() @ dba
            + imu.J_alpha_bg.double() @ dbg
        )

        dtheta = imu.J_gamma_bg.double() @ dbg
        gamma_corr = gamma_R @ _so3_exp(dtheta)

        return gamma_corr, beta_corr, alpha_corr

    # Legacy 15-D diagnostic helper retained for regression tests.
    def _imu_residuals_15d(
        self,
        T_rel: torch.Tensor,
        state_k: ESKFWorldState,
        v_k1: torch.Tensor,
        b_a_k1: torch.Tensor,
        b_g_k1: torch.Tensor,
        imu: IMUFactor,
    ):
        """
        15-D IMU + bias-random-walk residual.

        Optimisation state:
            [drho, dphi, dv_k1, dba_k1, dbg_k1]

        State at time k is fixed and supplied by the recursive ES-EKF.
        """

        g = self.g_w.double()
        dt = float(imu.dt)

        R_k = state_k.R.double()
        v_k = state_k.v.double()

        R_rel = T_rel[:3, :3]
        t_rel = T_rel[:3, 3]

        # Bias-correct preintegrated measurements using bias at time k.
        gamma_R, beta_corr, alpha_corr = (
            self._bias_correct_preintegration(
                imu,
                state_k,
            )
        )

        # ----------------------------------------------------------
        # Rotation residual
        # ----------------------------------------------------------

        Eg = gamma_R.T @ R_rel

        r_gamma = _so3_log(
            _svd_so3(Eg)
        )

        # ----------------------------------------------------------
        # Velocity residual
        #
        # R_k is world <- body.
        # ----------------------------------------------------------

        r_beta = (
            R_k.T
            @ (
                v_k1
                - v_k
                - g * dt
            )
            - beta_corr
        )

        # ----------------------------------------------------------
        # Position residual
        #
        # p_{k+1} - p_k = R_k * t_rel
        # ----------------------------------------------------------

        dp_world = R_k @ t_rel

        r_alpha = (
            R_k.T
            @ (
                dp_world
                - v_k * dt
                - 0.5 * g * dt * dt
            )
            - alpha_corr
        )

        # ----------------------------------------------------------
        # Bias random walk
        # ----------------------------------------------------------

        r_ba = (
            b_a_k1.double()
            - state_k.b_a.double()
        )

        r_bg = (
            b_g_k1.double()
            - state_k.b_g.double()
        )

        return (
            r_gamma,
            r_beta,
            r_alpha,
            r_ba,
            r_bg,
        )


    def _imu_jacobian_15d(
        self,
        T_rel: torch.Tensor,
        state_k: ESKFWorldState,
        v_k1: torch.Tensor,
        b_a_k1: torch.Tensor,
        b_g_k1: torch.Tensor,
        imu: IMUFactor,
    ):
        """
        Analytical Jacobian for the stacked residual

            [r_gamma,
             r_beta,
             r_alpha,
             r_ba,
             r_bg]

        with respect to

            [drho,
             dphi,
             dv_k1,
             dba_k1,
             dbg_k1].

        Returns:
            J : (15,15)
        """

        dtype = torch.float64

        R_k = state_k.R.double()

        R_rel = T_rel[:3, :3]
        t_rel = T_rel[:3, 3]

        gamma_R, _, _ = (
            self._bias_correct_preintegration(
                imu,
                state_k,
            )
        )

        (
            r_gamma,
            _,
            _,
            _,
            _,
        ) = self._imu_residuals_15d(
            T_rel,
            state_k,
            v_k1,
            b_a_k1,
            b_g_k1,
            imu,
        )

        J = torch.zeros(
            (15, 15),
            dtype=dtype,
        )

        # ==========================================================
        # r_gamma
        # ==========================================================
        #
        # Same left-perturbation convention as the validated
        # TwoFrameVIOOptimizer.
        #
        # Existing solver:
        #   J_gamma_phi =
        #       J_l^{-1}(r_gamma) gamma^T
        # ==========================================================

        Jli_g = _jl_inv_so3(r_gamma)

        J[
            0:3,
            3:6,
        ] = Jli_g @ gamma_R.T

        # ==========================================================
        # r_beta
        # ==========================================================
        #
        # state_k velocity is fixed.
        #
        # dr_beta / dv_k1 = R_k^T
        # ==========================================================

        J[
            3:6,
            6:9,
        ] = R_k.T

        # ==========================================================
        # r_alpha
        # ==========================================================
        #
        # Under:
        #
        #   T_rel' = Exp(drho,dphi) T_rel
        #
        # t_rel' ≈
        #   t_rel + drho - [t_rel]x dphi
        #
        # With R_g = R_k:
        #
        # dr_alpha/drho = I
        # dr_alpha/dphi = -[t_rel]x
        # ==========================================================

        J[
            6:9,
            0:3,
        ] = torch.eye(
            3,
            dtype=dtype,
        )

        J[
            6:9,
            3:6,
        ] = -_skew(t_rel)

        # ==========================================================
        # Accelerometer-bias random walk
        # ==========================================================

        J[
            9:12,
            9:12,
        ] = torch.eye(
            3,
            dtype=dtype,
        )

        # ==========================================================
        # Gyroscope-bias random walk
        # ==========================================================

        J[
            12:15,
            12:15,
        ] = torch.eye(
            3,
            dtype=dtype,
        )

        return J


    def _stack_imu_residuals_15d(
        self,
        T_rel,
        state_k,
        v_k1,
        b_a_k1,
        b_g_k1,
        imu,
    ):
        residuals = self._imu_residuals_15d(
            T_rel,
            state_k,
            v_k1,
            b_a_k1,
            b_g_k1,
            imu,
        )

        return torch.cat(
            residuals,
            dim=0,
        )

    def _imu_residuals_18d(
        self,
        T_rel: torch.Tensor,
        state_k: ESKFWorldState,
        v_k_opt: torch.Tensor,
        v_k1: torch.Tensor,
        b_a_k1: torch.Tensor,
        b_g_k1: torch.Tensor,
        imu: IMUFactor,
    ):
        """
        IMU plus bias-random-walk residual used by the 18-DoF solver.

        Optimisation increment order:
            [drho, dphi, dv_k, dv_k1, dba, dbg]

        The candidate bias variables directly correct the preintegrated
        rotation, velocity and position increments. This removes the
        structural disconnect in the interim 15-DoF implementation where
        the optimised biases appeared only in the random-walk residuals.
        """
        g = self.g_w.double()
        dt = float(imu.dt)

        R_k = state_k.R.double()
        R_rel = T_rel[:3, :3]
        t_rel = T_rel[:3, 3]

        gamma_corr, beta_corr, alpha_corr = self._bias_correct_preintegration(
            imu,
            state_k,
            b_a_candidate=b_a_k1,
            b_g_candidate=b_g_k1,
        )

        r_gamma = _so3_log(
            _svd_so3(gamma_corr.T @ R_rel)
        )

        r_beta = (
            R_k.T
            @ (
                v_k1.double()
                - v_k_opt.double()
                - g * dt
            )
            - beta_corr
        )

        dp_world = R_k @ t_rel

        r_alpha = (
            R_k.T
            @ (
                dp_world
                - v_k_opt.double() * dt
                - 0.5 * g * dt * dt
            )
            - alpha_corr
        )

        r_ba = b_a_k1.double() - state_k.b_a.double()
        r_bg = b_g_k1.double() - state_k.b_g.double()

        return r_gamma, r_beta, r_alpha, r_ba, r_bg

    def _imu_jacobian_18d(
        self,
        T_rel: torch.Tensor,
        state_k: ESKFWorldState,
        v_k_opt: torch.Tensor,
        v_k1: torch.Tensor,
        b_a_k1: torch.Tensor,
        b_g_k1: torch.Tensor,
        imu: IMUFactor,
    ):
        """
        Jacobian of [r_gamma, r_beta, r_alpha, r_ba, r_bg] with respect to

            [drho, dphi, dv_k, dv_k1, dba, dbg].

        Returns:
            J : (15, 18)

        The pose/velocity blocks follow the validated two-frame solver.
        Translational preintegration bias blocks use the stored first-order
        Jacobians directly. The small SO(3) bias block is evaluated by a
        three-axis central difference around the current candidate bias so
        the implemented Jacobian exactly follows the preintegration
        correction convention used by this module.
        """
        dtype = torch.float64
        R_k = state_k.R.double()
        R_rel = T_rel[:3, :3]
        t_rel = T_rel[:3, 3]
        dt = float(imu.dt)

        gamma_corr, _, _ = self._bias_correct_preintegration(
            imu,
            state_k,
            b_a_candidate=b_a_k1,
            b_g_candidate=b_g_k1,
        )

        r_gamma, _, _, _, _ = self._imu_residuals_18d(
            T_rel,
            state_k,
            v_k_opt,
            v_k1,
            b_a_k1,
            b_g_k1,
            imu,
        )

        J = torch.zeros((15, 18), dtype=dtype)
        I3 = torch.eye(3, dtype=dtype)

        # Rotation residual vs. left relative-pose perturbation.
        Jli_g = _jl_inv_so3(r_gamma)
        J[0:3, 3:6] = Jli_g @ gamma_corr.T

        # Velocity residual.
        J[3:6, 6:9] = -R_k.T
        J[3:6, 9:12] = R_k.T

        # Position residual under T_rel' = Exp(dxi) T_rel.
        J[6:9, 0:3] = I3
        J[6:9, 3:6] = -_skew(t_rel)
        J[6:9, 6:9] = -dt * R_k.T

        if self._has_bias_jacobians(imu):
            J[3:6, 12:15] = -imu.J_beta_ba.double()
            J[3:6, 15:18] = -imu.J_beta_bg.double()
            J[6:9, 12:15] = -imu.J_alpha_ba.double()
            J[6:9, 15:18] = -imu.J_alpha_bg.double()

            # Robustly match the implemented SO(3) bias correction.
            eps_b = 1e-6
            for j in range(3):
                ej = torch.zeros(3, dtype=dtype)
                ej[j] = eps_b

                gamma_p, _, _ = self._bias_correct_preintegration(
                    imu,
                    state_k,
                    b_a_candidate=b_a_k1,
                    b_g_candidate=b_g_k1 + ej,
                )
                gamma_m, _, _ = self._bias_correct_preintegration(
                    imu,
                    state_k,
                    b_a_candidate=b_a_k1,
                    b_g_candidate=b_g_k1 - ej,
                )

                r_p = _so3_log(_svd_so3(gamma_p.T @ R_rel))
                r_m = _so3_log(_svd_so3(gamma_m.T @ R_rel))
                J[0:3, 15 + j] = (r_p - r_m) / (2.0 * eps_b)

        # Bias random walks.
        J[9:12, 12:15] = I3
        J[12:15, 15:18] = I3

        return J

    def _stack_imu_residuals_18d(
        self,
        T_rel,
        state_k,
        v_k_opt,
        v_k1,
        b_a_k1,
        b_g_k1,
        imu,
    ):
        return torch.cat(
            self._imu_residuals_18d(
                T_rel,
                state_k,
                v_k_opt,
                v_k1,
                b_a_k1,
                b_g_k1,
                imu,
            ),
            dim=0,
        )

    def optimize_18d_no_prior(
        self,
        state_k: ESKFWorldState,
        vis: VisualFactor,
        imu: IMUFactor,
    ):
        """18-DoF bias-aware VIO ablation without an ES-EKF prior."""
        return self._gn18(state_k, vis, imu, prior=None)

    def optimize_18d_with_prior(
        self,
        state_k: ESKFWorldState,
        vis: VisualFactor,
        imu: IMUFactor,
        prior: ESKFPrior,
    ):
        """Full 18-DoF MAC-VARIO-EKF optimisation."""
        return self._gn18(state_k, vis, imu, prior=prior)

    # Backward-compatible names used by the existing validation runners.
    # They intentionally dispatch to the new 18-DoF implementation.
    def optimize_15d_no_prior(
        self,
        state_k: ESKFWorldState,
        vis: VisualFactor,
        imu: IMUFactor,
    ):
        return self.optimize_18d_no_prior(state_k, vis, imu)

    def optimize_15d_with_prior(
        self,
        state_k: ESKFWorldState,
        vis: VisualFactor,
        imu: IMUFactor,
        prior: ESKFPrior,
    ):
        return self.optimize_18d_with_prior(state_k, vis, imu, prior)

    def _gn18(
        self,
        state_k: ESKFWorldState,
        vis: VisualFactor,
        imu: IMUFactor,
        prior=None,
    ):
        """
        18-DoF Gauss-Newton / scale-aware LM optimisation:

            z = [
                relative pose  6,
                v_k            3,
                v_k+1          3,
                b_a            3,
                b_g            3
            ]

        The previous pose remains fixed. v_k is a local two-frame nuisance
        variable, while the returned navigation correction is defined only
        for the k+1 ES-EKF state.
        """

        dtype = torch.float64

        g = self.g_w.double()
        dt = float(imu.dt)

        # ==========================================================
        # Fixed recursive-filter state at image time k
        # ==========================================================

        p_k = state_k.p.double()
        v_k_seed = state_k.v.double()
        v_k_opt = v_k_seed.clone()
        R_k = state_k.R.double()

        b_a_k = state_k.b_a.double()
        b_g_k = state_k.b_g.double()

        T_w_bk = _se3_mat(
            R_k,
            p_k,
        )

        # ==========================================================
        # Camera/body extrinsics
        # ==========================================================

        T_bc = vis.T_BC.double()

        R_bc = T_bc[:3, :3]
        R_cb = R_bc.T

        # ==========================================================
        # Fixed MAC-VO camera observations
        # ==========================================================

        T_w_ck1_mat = (
            vis.pose_SE3
            .double()
            .matrix()
            .squeeze()
            .double()
        )

        pts_Tw = vis.pts_pos_Tw.double()
        cov_Tw = vis.pts_cov_Tw.double()
        obs_cov = vis.obs_cov_Tc.double()

        N = len(pts_Tw)

        pts_ck1 = _apply(
            _se3_inv_mat(T_w_ck1_mat),
            pts_Tw,
        )

        # ==========================================================
        # IMU covariance
        # ==========================================================

        S_imu = imu.cov.double()

        if self.clamp_cov:
            S_imu = self._clamp_cov(
                S_imu
            )

        Sigma_alpha = S_imu[6:9, 6:9]
        Sigma_beta = S_imu[3:6, 3:6]
        Sigma_rot_raw = S_imu[0:3, 0:3]

        eps3 = (
            1e-8
            * torch.eye(
                3,
                dtype=dtype,
            )
        )

        Sinv_beta = torch.linalg.inv(
            Sigma_beta + eps3
        )

        Sinv_alpha = torch.linalg.inv(
            Sigma_alpha + eps3
        )

        # ==========================================================
        # Bias-corrected preintegrated measurements
        # ==========================================================

        gamma_corr, beta_corr, alpha_corr = (
            self._bias_correct_preintegration(
                imu,
                state_k,
            )
        )

        # ==========================================================
        # Initial relative pose from MAC-VO
        # ==========================================================

        T_w_bk1_init = (
            T_w_ck1_mat
            @ _se3_inv_mat(T_bc)
        )

        T_rel_mat = (
            _se3_inv_mat(T_w_bk)
            @ T_w_bk1_init
        )

        # ==========================================================
        # Initial k+1 state
        # ==========================================================

        v_k1 = (
            v_k_opt
            + g * dt
            + R_k @ beta_corr
        )

        b_a_k1 = b_a_k.clone()
        b_g_k1 = b_g_k.clone()

        # ==========================================================
        # Bias random-walk covariance
        #
        # EuRoC sigma values are noise densities.
        # Q_b ~= sigma_b^2 * dt
        # ==========================================================

        dt_safe = max(
            dt,
            1e-6,
        )

        var_ba = max(
            float(EUROC_SIGMA_BA) ** 2
            * dt_safe,
            1e-12,
        )

        var_bg = max(
            float(EUROC_SIGMA_BG) ** 2
            * dt_safe,
            1e-14,
        )

        W_ba = (
            torch.eye(
                3,
                dtype=dtype,
            )
            / var_ba
        )

        W_bg = (
            torch.eye(
                3,
                dtype=dtype,
            )
            / var_bg
        )

        cost_log = []

        prev_cost = float("inf")
        n_iter = 0
        converged = False
        stable_rel_count = 0

        H_vis_acc = None
        H_info_last = None

        # ==========================================================
        # Gauss-Newton
        # ==========================================================

        for it in range(self.max_iter):

            n_iter = it + 1

            R_rel_cur = T_rel_mat[:3, :3]
            t_rel_cur = T_rel_mat[:3, 3]

            if not torch.isfinite(
                R_rel_cur
            ).all():
                break

            # ------------------------------------------------------
            # Normal equations
            # ------------------------------------------------------

            H = torch.zeros(
                (18, 18),
                dtype=dtype,
            )

            gv = torch.zeros(
                18,
                dtype=dtype,
            )

            # ======================================================
            # VISUAL FACTOR
            #
            # Identical visual construction to the validated
            # TwoFrameVIOOptimizer.
            # ======================================================

            T_w_ck1_pred = (
                T_w_bk
                @ T_rel_mat
                @ T_bc
            )

            T_c1_w_pred = _se3_inv_mat(
                T_w_ck1_pred
            )

            pts_ck1_pred = _apply(
                T_c1_w_pred,
                pts_Tw,
            )

            r_vis = (
                pts_ck1_pred
                - pts_ck1
            )

            R_w_ck1_pred = (
                T_w_ck1_pred[:3, :3]
            )

            eps_cov = (
                1e-6
                * torch.eye(
                    3,
                    dtype=dtype,
                )
                .unsqueeze(0)
                .expand(N, -1, -1)
            )

            Sigma_vis = (
                R_w_ck1_pred.T.unsqueeze(0)
                @ cov_Tw
                @ R_w_ck1_pred.unsqueeze(0)
                + obs_cov
                + eps_cov
            )

            W_vis = torch.linalg.inv(
                Sigma_vis
            )

            # Same visual Jacobian convention as existing solver.
            R_A = (
                R_cb
                @ R_rel_cur.T
            )

            s_pts = _apply(
                _se3_inv_mat(T_w_bk),
                pts_Tw,
            )

            Skew_s = _skew_batch(
                s_pts
            )

            neg_R_A = (
                -R_A.unsqueeze(0)
                .expand(N, -1, -1)
            )

            R_A_skew_s = (
                R_A.unsqueeze(0)
                @ Skew_s
            )

            J_vis = torch.cat(
                [
                    neg_R_A,
                    R_A_skew_s,
                ],
                dim=2,
            )

            e2 = (
                r_vis.unsqueeze(1)
                @ W_vis
                @ r_vis.unsqueeze(2)
            ).squeeze(-1).squeeze(-1)

            delta_h = self.huber_delta

            huber_w = torch.where(
                e2 <= delta_h,
                torch.ones_like(e2),
                delta_h
                / (
                    e2.sqrt()
                    + 1e-9
                ),
            )

            c_vis = 0.0

            H_vis_acc = torch.zeros(
                (6, 6),
                dtype=dtype,
            )

            for i in range(N):

                w = huber_w[i].item()

                Wi = (
                    W_vis[i]
                    * w
                )

                Ji6 = J_vis[i]
                ri = r_vis[i]

                hv = (
                    Ji6.T
                    @ Wi
                    @ Ji6
                )

                H_vis_acc += hv

                H[0:6, 0:6] += hv

                gv[0:6] += (
                    Ji6.T
                    @ (
                        Wi @ ri
                    )
                )

                if w == 1.0:

                    c_vis += (
                        0.5
                        * float(
                            ri
                            @ Wi
                            @ ri
                        )
                    )

                else:

                    c_vis += float(
                        delta_h
                        * (
                            e2[i].sqrt()
                            - 0.5
                            * delta_h
                        )
                    )

            # ======================================================
            # IMU + BIAS RANDOM-WALK RESIDUAL
            # ======================================================

            (
                r_gamma,
                r_beta,
                r_alpha,
                r_ba,
                r_bg,
            ) = self._imu_residuals_18d(
                T_rel_mat,
                state_k,
                v_k_opt,
                v_k1,
                b_a_k1,
                b_g_k1,
                imu,
            )

            # Rotation residual covariance depends on current residual.
            Jr_inv = _jr_inv_so3(
                r_gamma
            )

            Sigma_rot_g = (
                Jr_inv
                @ Sigma_rot_raw
                @ Jr_inv.T
            )

            Sinv_gamma = torch.linalg.inv(
                Sigma_rot_g
                + eps3
            )

            J_imu = self._imu_jacobian_18d(
                T_rel_mat,
                state_k,
                v_k_opt,
                v_k1,
                b_a_k1,
                b_g_k1,
                imu,
            )

            r_imu = torch.cat(
                [
                    r_gamma,
                    r_beta,
                    r_alpha,
                    r_ba,
                    r_bg,
                ],
                dim=0,
            )

            W_imu = torch.zeros(
                (15, 15),
                dtype=dtype,
            )

            w_imu = float(
                self.imu_weight
            )

            W_imu[0:3, 0:3] = (
                w_imu
                * Sinv_gamma
            )

            W_imu[3:6, 3:6] = (
                w_imu
                * Sinv_beta
            )

            W_imu[6:9, 6:9] = (
                w_imu
                * Sinv_alpha
            )

            W_imu[9:12, 9:12] = (
                w_imu
                * W_ba
            )

            W_imu[12:15, 12:15] = (
                w_imu
                * W_bg
            )

            H += (
                J_imu.T
                @ W_imu
                @ J_imu
            )

            gv += (
                J_imu.T
                @ (
                    W_imu
                    @ r_imu
                )
            )

            # Added by Narsimlu
                        # ======================================================
            # TEMPERED ES-EKF PRIOR
            #
            # Omega_EKF = eta * (P^- + eps I)^(-1)
            #
            # Prior error-state ordering:
            #   [dp, dv, dtheta, dba, dbg]
            # ======================================================

            c_prior = 0.0

            if prior is not None:

                r_prior = self._prior_residual_15d(
                    T_rel_mat,
                    state_k,
                    v_k1,
                    b_a_k1,
                    b_g_k1,
                    prior,
                )

                J_prior = self._prior_jacobian_18d(
                    T_rel_mat,
                    state_k,
                    v_k1,
                    b_a_k1,
                    b_g_k1,
                    prior,
                )

                # Keep optimiser arithmetic on the same device.
                r_prior = r_prior.to(
                    dtype=dtype,
                    device=H.device,
                )

                J_prior = J_prior.to(
                    dtype=dtype,
                    device=H.device,
                )

                P_prior = (
                    prior.P
                    .detach()
                    .to(
                        dtype=dtype,
                        device=H.device,
                    )
                )

                # Numerical symmetrisation.
                P_prior = 0.5 * (
                    P_prior
                    + P_prior.T
                )

                I15 = torch.eye(
                    15,
                    dtype=dtype,
                    device=H.device,
                )

                # Manuscript regularisation:
                # (P^- + epsilon I)^-1
                P_reg = (
                    P_prior
                    + 1e-9 * I15
                )

                # Eigendecomposition gives a stable inverse and
                # prevents tiny numerical negative eigenvalues
                # producing an indefinite information matrix.
                eigval, eigvec = torch.linalg.eigh(
                    P_reg
                )

                eigval = torch.clamp(
                    eigval,
                    min=1e-12,
                )

                P_inv = (
                    eigvec
                    @ torch.diag(
                        1.0 / eigval
                    )
                    @ eigvec.T
                )

                eta_eff = (
                    float(self.prior_eta)
                    * float(prior.eta)
                )

                if eta_eff < 0.0:
                    raise ValueError(
                        "ES-EKF prior eta must be non-negative"
                    )

                W_prior = (
                    eta_eff
                    * P_inv
                )

                H += (
                    J_prior.T
                    @ W_prior
                    @ J_prior
                )

                gv += (
                    J_prior.T
                    @ (
                        W_prior
                        @ r_prior
                    )
                )

                c_prior = float(
                    r_prior
                    @ W_prior
                    @ r_prior
                )

            # ======================================================
            # Local posterior information approximation
            #
            # H contains only residual information. Numerical LM damping is
            # applied separately to H_solve, so the posterior covariance
            # is formed from the undamped information matrix.
            # ======================================================

            H_info_last = (
                0.5 * (H + H.T)
            )

            # ======================================================
            # Costs
            # ======================================================

            c_gamma = (
                w_imu
                * float(
                    r_gamma
                    @ Sinv_gamma
                    @ r_gamma
                )
            )

            c_beta = (
                w_imu
                * float(
                    r_beta
                    @ Sinv_beta
                    @ r_beta
                )
            )

            c_alpha = (
                w_imu
                * float(
                    r_alpha
                    @ Sinv_alpha
                    @ r_alpha
                )
            )

            c_ba = (
                w_imu
                * float(
                    r_ba
                    @ W_ba
                    @ r_ba
                )
            )

            c_bg = (
                w_imu
                * float(
                    r_bg
                    @ W_bg
                    @ r_bg
                )
            )

            # total = (
            #     c_vis
            #     + c_gamma
            #     + c_beta
            #     + c_alpha
            #     + c_ba
            #     + c_bg
            # )

            total = (
                c_vis
                + c_gamma
                + c_beta
                + c_alpha
                + c_ba
                + c_bg
                + c_prior
            )

            # cost_log.append(
            #     [
            #         c_vis,
            #         c_gamma,
            #         c_beta,
            #         c_alpha,
            #         c_ba,
            #         c_bg,
            #     ]
            # )

            cost_log.append(
                [
                    c_vis,
                    c_gamma,
                    c_beta,
                    c_alpha,
                    c_ba,
                    c_bg,
                    c_prior,
                ]
            )

            # print(
            #     f"  GN18 it={it} "
            #     f"c_vis={c_vis:.4f} "
            #     f"c_g={c_gamma:.4f} "
            #     f"c_b={c_beta:.4f} "
            #     f"c_a={c_alpha:.4f} "
            #     f"c_ba={c_ba:.4f} "
            #     f"c_bg={c_bg:.4f} "
            #     f"tot={total:.4f}"
            # )

            print(
                f"  GN18 it={it} "
                f"c_vis={c_vis:.4f} "
                f"c_g={c_gamma:.4f} "
                f"c_b={c_beta:.4f} "
                f"c_a={c_alpha:.4f} "
                f"c_ba={c_ba:.4f} "
                f"c_bg={c_bg:.4f} "
                f"c_prior={c_prior:.4f} "
                f"tot={total:.4f}"
            )

            if it > 0:
                cost_change = abs(
                    prev_cost
                    - total
                )

                cost_scale = max(
                    1.0,
                    abs(prev_cost),
                )

                rel_change = (
                    cost_change
                    / cost_scale
                )

                # Absolute convergence remains the strictest test.
                if cost_change < self.tol:
                    converged = True
                    prev_cost = total
                    break

                # Relative convergence detects a numerical plateau.
                # Use the magnitude of normalized objective change
                # irrespective of sign, because fixed-LM Gauss-Newton
                # can oscillate by tiny amounts around a stationary
                # solution. Requiring five consecutive plateau steps
                # avoids accepting a single accidental small change.
                if rel_change < self.rel_tol:
                    stable_rel_count += 1
                else:
                    stable_rel_count = 0

                if stable_rel_count >= 5:
                    converged = True
                    prev_cost = total
                    break

            prev_cost = total

            # ======================================================
            # Solve 18 x 18 system
            # ======================================================

            try:

                # Scale-aware Levenberg-Marquardt damping.
                # A fixed lambda*I is ineffective when the normal-equation
                # diagonal spans many orders of magnitude.
                diag_H = torch.clamp(
                    torch.abs(torch.diag(H)),
                    min=1.0,
                )

                H_solve = (
                    H
                    + float(self.lm_lambda)
                    * torch.diag(diag_H)
                )

                delta_x = (
                    torch.linalg.solve(
                        H_solve,
                        -gv,
                    )
                )

            except Exception:

                break

            if not torch.isfinite(
                delta_x
            ).all():
                break

            d_xi = delta_x[0:6]
            d_vk = delta_x[6:9]
            d_vk1 = delta_x[9:12]
            d_ba = delta_x[12:15]
            d_bg = delta_x[15:18]

            # ======================================================
            # Retract relative pose
            # ======================================================

            R_pert = _so3_exp(
                d_xi[3:6]
            )

            T_pert = _se3_mat(
                R_pert,
                d_xi[0:3],
            )

            T_rel_mat = (
                T_pert
                @ T_rel_mat
            )

            T_rel_mat[:3, :3] = (
                _svd_so3(
                    T_rel_mat[:3, :3]
                )
            )

            # ======================================================
            # Euclidean-state updates
            # ======================================================

            if torch.isfinite(
                d_vk
            ).all():

                v_k_opt = (
                    v_k_opt
                    + d_vk
                )

            if torch.isfinite(
                d_vk1
            ).all():

                v_k1 = (
                    v_k1
                    + d_vk1
                )

            if torch.isfinite(
                d_ba
            ).all():

                b_a_k1 = (
                    b_a_k1
                    + d_ba
                )

            if torch.isfinite(
                d_bg
            ).all():

                b_g_k1 = (
                    b_g_k1
                    + d_bg
                )

        # ==========================================================
        # Construct final camera pose
        # ==========================================================

        T_w_bk1_opt = (
            T_w_bk
            @ T_rel_mat
        )

        T_w_ck1_opt = (
            T_w_bk1_opt
            @ T_bc
        )

        T_w_ck1_opt[:3, :3] = (
            _svd_so3(
                T_w_ck1_opt[:3, :3]
            )
        )

        T_w_ck1_pp = _se3_from_mat4(
            T_w_ck1_opt
        )

        # ==========================================================
        # Base result — compatible with existing logging
        # ==========================================================

        base = OptimResult(
            pose_k1=T_w_ck1_pp,
            vel_k1=v_k1.cpu(),

            success=converged,

            n_iter=n_iter,

            final_cost=(
                prev_cost
                if prev_cost != float("inf")
                else 0.0
            ),

            cost_log=cost_log,

            debug_logs={
                "converged": converged,
                "rel_tol": self.rel_tol,
                "stable_rel_required": 5,
                # "solver": "15d_no_ekf_prior",
                "solver": (
                    "18d_with_ekf_prior"
                    if prior is not None
                    else "18d_no_ekf_prior"
            ),

                "T_rel": (
                    T_rel_mat
                    .detach()
                    .cpu()
                    .numpy()
                ),

                "v_k_opt": (
                    v_k_opt
                    .detach()
                    .cpu()
                    .numpy()
                ),

                "v_k1": (
                    v_k1
                    .detach()
                    .cpu()
                    .numpy()
                ),

                "b_a_k1": (
                    b_a_k1
                    .detach()
                    .cpu()
                    .numpy()
                ),

                "b_g_k1": (
                    b_g_k1
                    .detach()
                    .cpu()
                    .numpy()
                ),
            },

            H_vis=(
                None
                if H_vis_acc is None
                else H_vis_acc.clone()
            ),
        )

        # zero_nav = torch.zeros(
        #     15,
        #     dtype=dtype,
        # )

        if prior is not None:

            delta_nav = (
                self._prior_residual_15d(
                    T_rel_mat,
                    state_k,
                    v_k1,
                    b_a_k1,
                    b_g_k1,
                    prior,
                )
                .detach()
                .clone()
            )

        else:

            delta_nav = torch.zeros(
                15,
                dtype=dtype,
            )

        # ==========================================================
        # Posterior covariance for recursive feedback
        # ==========================================================

        P_nav = None

        if (
            prior is not None
            and H_info_last is not None
        ):

            H_info = (
                H_info_last
                .detach()
                .clone()
                .double()
            )

            H_info = 0.5 * (
                H_info
                + H_info.T
            )

            # Stable inverse of the local information matrix.
            eig_h, U_h = torch.linalg.eigh(
                H_info
            )

            eig_h = torch.clamp(
                eig_h,
                min=1e-12,
            )

            P_z = (
                U_h
                @ torch.diag(
                    1.0 / eig_h
                )
                @ U_h.T
            )

            # Map optimizer perturbations
            # [drho,dphi,dv_k,dv_k1,dba,dbg]
            # to the ESKF navigation error ordering at k+1
            # [dp,dv,dtheta,dba,dbg].
            J_nav = self._prior_jacobian_18d(
                T_rel_mat,
                state_k,
                v_k1,
                b_a_k1,
                b_g_k1,
                prior,
            ).double()

            P_nav = (
                J_nav
                @ P_z
                @ J_nav.T
            )

            P_nav = 0.5 * (
                P_nav
                + P_nav.T
            )

        return ESKFOptimResult(
            base=base,

            b_a_k1=(
                b_a_k1
                .detach()
                .clone()
            ),

            b_g_k1=(
                b_g_k1
                .detach()
                .clone()
            ),

            # Optimised navigation correction relative to the ES-EKF prior.
            delta_nav=delta_nav,

            H_nav=(
                None
                if H_info_last is None
                else H_info_last.detach().clone()
            ),

            P_nav=(
                None
                if P_nav is None
                else P_nav.detach().clone()
            ),
        )

    def _prior_residual_15d(
        self,
        T_rel: torch.Tensor,
        state_k: ESKFWorldState,
        v_k1: torch.Tensor,
        b_a_k1: torch.Tensor,
        b_g_k1: torch.Tensor,
        prior: ESKFPrior,
    ):
        """
        ES-EKF prior residual in the covariance ordering:

            [dp, dv, dtheta, dba, dbg]

        P in ESKFPrior uses this same 15-D ordering.
        """

        R_k = state_k.R.double()
        p_k = state_k.p.double()

        R_rel = T_rel[:3, :3]
        t_rel = T_rel[:3, 3]

        # Candidate navigation state at k+1.
        p_k1 = p_k + R_k @ t_rel
        R_k1 = R_k @ R_rel

        r_p = p_k1 - prior.p.double()

        r_v = (
            v_k1.double()
            - prior.v.double()
        )

        E_R = (
            prior.R.double().T
            @ R_k1
        )

        r_theta = _so3_log(
            _svd_so3(E_R)
        )

        r_ba = (
            b_a_k1.double()
            - prior.b_a.double()
        )

        r_bg = (
            b_g_k1.double()
            - prior.b_g.double()
        )

        return torch.cat(
            [
                r_p,
                r_v,
                r_theta,
                r_ba,
                r_bg,
            ],
            dim=0,
        )


    def _prior_jacobian_15d(
        self,
        T_rel: torch.Tensor,
        state_k: ESKFWorldState,
        v_k1: torch.Tensor,
        b_a_k1: torch.Tensor,
        b_g_k1: torch.Tensor,
        prior: ESKFPrior,
    ):
        """
        Jacobian of the 15-D ES-EKF prior residual with respect to

            [drho, dphi, dv_k1, dba_k1, dbg_k1].

        Relative pose uses the same left perturbation as the
        validated VIO optimiser.
        """

        dtype = torch.float64

        R_k = state_k.R.double()

        R_rel = T_rel[:3, :3]
        t_rel = T_rel[:3, 3]

        R_k1 = (
            R_k
            @ R_rel
        )

        E_R = (
            prior.R.double().T
            @ R_k1
        )

        r_theta = _so3_log(
            _svd_so3(E_R)
        )

        J = torch.zeros(
            (15, 15),
            dtype=dtype,
        )

        # ------------------------------------------------------
        # Position residual
        #
        # T_rel' = Exp(drho,dphi) T_rel
        #
        # t_rel' ~= t_rel + drho - [t_rel]x dphi
        # p_k1   = p_k + R_k t_rel
        # ------------------------------------------------------

        J[0:3, 0:3] = R_k

        J[0:3, 3:6] = (
            -R_k
            @ _skew(t_rel)
        )

        # ------------------------------------------------------
        # Velocity
        # ------------------------------------------------------

        J[3:6, 6:9] = torch.eye(
            3,
            dtype=dtype,
        )

        # ------------------------------------------------------
        # Orientation
        #
        # E = R_prior^T R_k R_rel
        #
        # Left perturbation of R_rel maps through
        # A = R_prior^T R_k.
        # ------------------------------------------------------

        A = (
            prior.R.double().T
            @ R_k
        )

        J[6:9, 3:6] = (
            _jl_inv_so3(r_theta)
            @ A
        )

        # ------------------------------------------------------
        # Biases
        # ------------------------------------------------------

        J[9:12, 9:12] = torch.eye(
            3,
            dtype=dtype,
        )

        J[12:15, 12:15] = torch.eye(
            3,
            dtype=dtype,
        )

        return J

    def _prior_jacobian_18d(
        self,
        T_rel: torch.Tensor,
        state_k: ESKFWorldState,
        v_k1: torch.Tensor,
        b_a_k1: torch.Tensor,
        b_g_k1: torch.Tensor,
        prior: ESKFPrior,
    ):
        """
        Jacobian of the 15-D k+1 ES-EKF prior residual with respect to

            [drho, dphi, dv_k, dv_k1, dba, dbg].

        The prior does not constrain the local nuisance velocity v_k;
        its information enters through the two-frame IMU factor.
        """
        dtype = torch.float64
        R_k = state_k.R.double()
        R_rel = T_rel[:3, :3]
        t_rel = T_rel[:3, 3]

        R_k1 = R_k @ R_rel
        E_R = prior.R.double().T @ R_k1
        r_theta = _so3_log(_svd_so3(E_R))

        J = torch.zeros((15, 18), dtype=dtype)
        I3 = torch.eye(3, dtype=dtype)

        # Position at k+1.
        J[0:3, 0:3] = R_k
        J[0:3, 3:6] = -R_k @ _skew(t_rel)

        # Velocity at k+1. Columns 6:9 (v_k) are deliberately zero.
        J[3:6, 9:12] = I3

        # Orientation at k+1.
        A = prior.R.double().T @ R_k
        J[6:9, 3:6] = _jl_inv_so3(r_theta) @ A

        # Biases at k+1 / interval bias variables.
        J[9:12, 12:15] = I3
        J[12:15, 15:18] = I3

        return J
