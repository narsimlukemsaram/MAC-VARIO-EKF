"""
preintegration.py — UMRP4 MAC-VIO  v4
Manual Forster body-frame preintegration, exactly matching thesis eq 2.4 / 2.5.

Thesis notation (eq 2.4):
    gamma_bj^bi = prod_k Exp(w_k * dt)           # accumulated rotation bi→bj
    beta_bj^bi  = sum_k  R_bi^bk @ a_k * dt      # R_bi^bk = dR_k^T  (bk→bi)
    alpha_bj^bi = sum_k (beta_bk^bi * dt + 0.5 * R_bi^bk @ a_k * dt^2)

R_bi^bk means "rotation from bk to bi" = (accumulated dR from bi to bk)^T = dR_k^T

Thesis eq 2.5 reconstruction:
    p_j = p_i + v_i*dt + 0.5*g_w*dt^2 + R_w^bi @ alpha
    v_j = v_i + g_w*dt + R_w^bi @ beta
    R_j = R_i ⊗ gamma

EuRoC: g_w = [0,0,-9.81] (world z-up, gravity downward = negative z)
"""

import torch
import pypose as pp
import numpy as np
from scipy.spatial.transform import Rotation

EUROC_SIGMA_GYRO  = 1.6968e-4
EUROC_SIGMA_ACCEL = 2.0000e-3
EUROC_SIGMA_BG    = 1.9393e-5
EUROC_SIGMA_BA    = 3.0000e-3
_DT_TYPICAL = 0.005  # 200 Hz


def _default_gyro_cov(device, dt=_DT_TYPICAL):
    return torch.tensor([EUROC_SIGMA_GYRO**2/dt]*3, dtype=torch.float64).to(device)


def _default_acc_cov(device, dt=_DT_TYPICAL):
    return torch.tensor([EUROC_SIGMA_ACCEL**2/dt]*3, dtype=torch.float64).to(device)


def _extract_cov(cov_tensor):
    c = cov_tensor[0, -1].cpu()
    if c.dim() == 1:
        return torch.diag(c.double())
    return c.double()


class IMUPreintegrator:
    """
    Integrates raw or AirIMU-corrected IMU data between two keyframes.
    Manual Forster body-frame integration (thesis eq 2.4): R_bi^bk = dR_k.T
    PyPose used only for 9x9 covariance propagation.

    Returns:
        rot     : pp.SO3 (4,)   — gamma: relative rotation (bi→bj), xyzw
        vel     : Tensor(3,)    — beta:  velocity preintegral in frame bi
        pos     : Tensor(3,)    — alpha: position preintegral in frame bi
        rot_mat : ndarray(3,3)  — gamma as rotation matrix
        cov     : Tensor(9,9)   — [rot|vel|pos] blocks
        dt      : float         — total interval seconds
    """

    def __init__(self, device='cuda:0'):
        self.device = device
        # gravity=0: only cov output used from PyPose
        self._integrator = pp.module.IMUPreintegrator(
            prop_cov=True, reset=True, gravity=0.0
        ).to(device).double()
        self._default_acc_cov  = _default_acc_cov(device)
        self._default_gyro_cov = _default_gyro_cov(device)

    def integrate(self, timestamps_ns, acc_raw, gyro_raw, bg, ba,
                  acc_cov=None, gyro_cov=None, gt_rot=None):
        N = len(timestamps_ns)
        assert N >= 2
        dev = self.device

        dt_np = np.diff(timestamps_ns.astype(np.float64)) * 1e-9
        dt_np = np.clip(dt_np, 1e-4, 0.1)

        acc_b  = acc_raw  - ba  # (N,3)
        gyro_b = gyro_raw - bg  # (N,3)

        # ── Manual Forster preintegration (thesis eq 2.4) ─────────────────────
        # dR accumulates rotation from bi to bk: dR_k = R_{bi→bk}
        # R_bi^bk = dR_k.T  (maps bk-frame vectors to bi-frame)
        dR  = np.eye(3, dtype=np.float64)  # R_{bi→bk}, starts at identity
        dv  = np.zeros(3, dtype=np.float64)  # beta in bi frame
        dp  = np.zeros(3, dtype=np.float64)  # alpha in bi frame

        for k in range(N - 1):
            dt_k = dt_np[k]
            a_k  = acc_b[k]
            w_k  = gyro_b[k]

            # R_bi^bk = dR.T  (thesis notation: rotation from bk to bi)
            R_bi_bk = dR.T

            # update alpha then beta (Forster ordering: alpha uses beta before update)
            dp += dv * dt_k + 0.5 * (R_bi_bk @ a_k) * dt_k**2
            dv += R_bi_bk @ a_k * dt_k

            # update accumulated rotation dR = R_{bi→bk+1}
            angle = float(np.linalg.norm(w_k * dt_k))
            if angle > 1e-10:
                dR = dR @ Rotation.from_rotvec(w_k * dt_k).as_matrix()

        # Added by Narsimlu for bias Jacobians
        # --------------------------------------------------------------
        # Bias Jacobians around the current preintegration
        # linearisation point.
        #
        # These are evaluated by central finite differences using the
        # exact same manual preintegration convention above. This keeps
        # the existing Phase C/D nominal trajectory unchanged while
        # exposing first-order bias sensitivity for the EKF-aware solver.
        #
        # Definitions:
        #   dR(b_g + db_g) ~= dR(b_g) Exp(J_gamma_bg db_g)
        #
        #   dv(b_a + db_a, b_g + db_g)
        #       ~= dv + J_beta_ba db_a + J_beta_bg db_g
        #
        #   dp(b_a + db_a, b_g + db_g)
        #       ~= dp + J_alpha_ba db_a + J_alpha_bg db_g
        # --------------------------------------------------------------

        bg_ref = np.asarray(bg, dtype=np.float64).reshape(3)
        ba_ref = np.asarray(ba, dtype=np.float64).reshape(3)

        def _reintegrate_with_bias(bg_test, ba_test):
            """Repeat only the nominal manual preintegration."""

            bg_test = np.asarray(
                bg_test, dtype=np.float64
            ).reshape(3)

            ba_test = np.asarray(
                ba_test, dtype=np.float64
            ).reshape(3)

            acc_test = acc_raw - ba_test.reshape(1, 3)
            gyro_test = gyro_raw - bg_test.reshape(1, 3)

            dR_test = np.eye(3, dtype=np.float64)
            dv_test = np.zeros(3, dtype=np.float64)
            dp_test = np.zeros(3, dtype=np.float64)

            for kk in range(N - 1):
                dt_kk = float(dt_np[kk])

                a_kk = acc_test[kk]
                w_kk = gyro_test[kk]

                # Same convention as nominal preintegration above.
                R_bi_bk_test = dR_test.T

                dp_test += (
                    dv_test * dt_kk
                    + 0.5
                    * (R_bi_bk_test @ a_kk)
                    * (dt_kk ** 2)
                )

                dv_test += (
                    R_bi_bk_test @ a_kk
                ) * dt_kk

                angle_kk = float(
                    np.linalg.norm(w_kk * dt_kk)
                )

                if angle_kk > 1e-10:
                    dR_test = (
                        dR_test
                        @ Rotation.from_rotvec(
                            w_kk * dt_kk
                        ).as_matrix()
                    )

            return dR_test, dv_test, dp_test

        J_gamma_bg = np.zeros(
            (3, 3), dtype=np.float64
        )

        J_beta_ba = np.zeros(
            (3, 3), dtype=np.float64
        )

        J_beta_bg = np.zeros(
            (3, 3), dtype=np.float64
        )

        J_alpha_ba = np.zeros(
            (3, 3), dtype=np.float64
        )

        J_alpha_bg = np.zeros(
            (3, 3), dtype=np.float64
        )

        # Perturbation sizes.
        eps_ba = 1e-5       # m/s^2
        eps_bg = 1e-6       # rad/s

        # Accelerometer-bias derivatives
        for j in range(3):

            dba = np.zeros(
                3, dtype=np.float64
            )

            dba[j] = eps_ba

            _, dv_p, dp_p = _reintegrate_with_bias(
                bg_ref,
                ba_ref + dba,
            )

            _, dv_m, dp_m = _reintegrate_with_bias(
                bg_ref,
                ba_ref - dba,
            )

            J_beta_ba[:, j] = (
                dv_p - dv_m
            ) / (2.0 * eps_ba)

            J_alpha_ba[:, j] = (
                dp_p - dp_m
            ) / (2.0 * eps_ba)

        # Gyroscope-bias derivatives
        for j in range(3):

            dbg = np.zeros(
                3, dtype=np.float64
            )

            dbg[j] = eps_bg

            dR_p, dv_p, dp_p = _reintegrate_with_bias(
                bg_ref + dbg,
                ba_ref,
            )

            dR_m, dv_m, dp_m = _reintegrate_with_bias(
                bg_ref - dbg,
                ba_ref,
            )

            # Right-local rotation perturbation relative to
            # the nominal preintegrated rotation.
            phi_p = Rotation.from_matrix(
                dR.T @ dR_p
            ).as_rotvec()

            phi_m = Rotation.from_matrix(
                dR.T @ dR_m
            ).as_rotvec()

            J_gamma_bg[:, j] = (
                phi_p - phi_m
            ) / (2.0 * eps_bg)

            J_beta_bg[:, j] = (
                dv_p - dv_m
            ) / (2.0 * eps_bg)

            J_alpha_bg[:, j] = (
                dp_p - dp_m
            ) / (2.0 * eps_bg)
        # --------------------------------------------------------------

        # ── PyPose for covariance only ────────────────────────────────────────
        dt_t    = torch.tensor(dt_np, dtype=torch.float64).view(1,N-1,1).to(dev)
        acc_bt  = torch.tensor(acc_b,  dtype=torch.float64).unsqueeze(0).to(dev)
        gyro_bt = torch.tensor(gyro_b, dtype=torch.float64).unsqueeze(0).to(dev)

        if acc_cov is None:
            acc_cov_t = self._default_acc_cov.view(1,1,3).expand(1,N-1,3).clone()
        else:
            arr = np.clip(np.asarray(acc_cov, np.float64), 1e-8, 1e2)
            acc_cov_t = torch.tensor(arr[:N-1], dtype=torch.float64).unsqueeze(0).to(dev)

        if gyro_cov is None:
            gyro_cov_t = self._default_gyro_cov.view(1,1,3).expand(1,N-1,3).clone()
        else:
            arr = np.clip(np.asarray(gyro_cov, np.float64), 1e-12, 1e2)
            gyro_cov_t = torch.tensor(arr[:N-1], dtype=torch.float64).unsqueeze(0).to(dev)

        init_state = {
            'pos': torch.zeros(1,1,3,dtype=torch.float64).to(dev),
            'vel': torch.zeros(1,1,3,dtype=torch.float64).to(dev),
            'rot': pp.identity_SO3(1,1).to(dev).double(),
        }
        result = self._integrator(
            init_state=init_state, dt=dt_t,
            acc=acc_bt[:,:N-1,:], gyro=gyro_bt[:,:N-1,:],
            rot=None, acc_cov=acc_cov_t, gyro_cov=gyro_cov_t,
        )

        # gamma = dR as pp.SO3 (xyzw)
        dR_xyzw = Rotation.from_matrix(dR).as_quat()
        rot_so3  = pp.SO3(torch.tensor(dR_xyzw, dtype=torch.float64))

        # return {
        #     'rot'     : rot_so3,
        #     'vel'     : torch.tensor(dv, dtype=torch.float64),
        #     'pos'     : torch.tensor(dp, dtype=torch.float64),
        #     'rot_mat' : dR,
        #     'cov'     : _extract_cov(result['cov']),
        #     'dt'      : float(dt_np.sum()),
        # }

        # Added by Narsimlu: return bias Jacobians as well
        return {
            'rot'     : rot_so3,
            'vel'     : torch.tensor(
                dv, dtype=torch.float64
            ),
            'pos'     : torch.tensor(
                dp, dtype=torch.float64
            ),
            'rot_mat' : dR,
            'cov'     : _extract_cov(
                result['cov']
            ),
            'dt'      : float(dt_np.sum()),

            # Bias linearisation point
            'bias_ref_ba': torch.tensor(
                ba_ref,
                dtype=torch.float64,
            ),
            'bias_ref_bg': torch.tensor(
                bg_ref,
                dtype=torch.float64,
            ),

            # First-order bias Jacobians
            'J_gamma_bg': torch.tensor(
                J_gamma_bg,
                dtype=torch.float64,
            ),
            'J_beta_ba': torch.tensor(
                J_beta_ba,
                dtype=torch.float64,
            ),
            'J_beta_bg': torch.tensor(
                J_beta_bg,
                dtype=torch.float64,
            ),
            'J_alpha_ba': torch.tensor(
                J_alpha_ba,
                dtype=torch.float64,
            ),
            'J_alpha_bg': torch.tensor(
                J_alpha_bg,
                dtype=torch.float64,
            ),
        }

    @staticmethod
    def cov_rot(cov): return cov[0:3, 0:3]

    @staticmethod
    def cov_vel(cov): return cov[3:6, 3:6]

    @staticmethod
    def cov_pos(cov): return cov[6:9, 6:9]