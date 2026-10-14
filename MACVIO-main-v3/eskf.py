"""
eskf.py

15-state Error-State EKF propagation for MAC-VARIO-EKF.

Nominal state:
    p_WB : position of body in world, (3,)
    v_WB : velocity in world, (3,)
    R_WB : rotation world <- body, (3,3)
    b_a  : accelerometer bias, (3,)
    b_g  : gyroscope bias, (3,)

Error state:
    dx = [dp, dv, dtheta, dba, dbg]   (15,)

Rotation convention matches WorldState.rot in optimizer.py:
    R_WB = world <- body
"""

from dataclasses import dataclass

import numpy as np
import torch


DTYPE = torch.float64


def skew(v: torch.Tensor) -> torch.Tensor:
    """Return 3x3 skew-symmetric matrix for a 3-vector."""
    v = v.reshape(3)
    x, y, z = v[0], v[1], v[2]

    return torch.stack([
        torch.stack([torch.zeros((), dtype=v.dtype, device=v.device), -z, y]),
        torch.stack([z, torch.zeros((), dtype=v.dtype, device=v.device), -x]),
        torch.stack([-y, x, torch.zeros((), dtype=v.dtype, device=v.device)]),
    ])


def so3_exp(phi: torch.Tensor) -> torch.Tensor:
    """
    Exponential map SO(3), phi in radians.
    Returns a 3x3 rotation matrix.
    """
    phi = phi.reshape(3)
    theta = torch.linalg.norm(phi)
    I = torch.eye(3, dtype=phi.dtype, device=phi.device)

    if theta.item() < 1e-10:
        A = skew(phi)
        return I + A + 0.5 * (A @ A)

    axis = phi / theta
    A = skew(axis)

    return (
        I
        + torch.sin(theta) * A
        + (1.0 - torch.cos(theta)) * (A @ A)
    )


@dataclass
class ESKFState:
    p: torch.Tensor       # (3,) world position
    v: torch.Tensor       # (3,) world velocity
    R: torch.Tensor       # (3,3) world <- body
    b_a: torch.Tensor     # (3,) accelerometer bias
    b_g: torch.Tensor     # (3,) gyroscope bias
    P: torch.Tensor       # (15,15) error covariance


class AdaptiveESKF:
    """
    Propagation part of the MAC-VARIO-EKF recursive filter.

    Error-state order:
        [dp, dv, dtheta, dba, dbg]
    """

    def __init__(
        self,
        p0,
        v0,
        R0,
        b_a0=None,
        b_g0=None,
        P0=None,
        gravity=None,
        device="cpu",
        alpha_a=1.0,
        alpha_g=1.0,
        acc_var_floor=1e-3,
        gyro_var_floor=1e-5,
        ba_rw_var=(3.0000e-3)**2,
        bg_rw_var=(1.9393e-5)**2,
    ):
        self.device = torch.device(device)

        def t(x):
            return torch.as_tensor(
                x, dtype=DTYPE, device=self.device
            ).clone()

        if b_a0 is None:
            b_a0 = np.zeros(3)

        if b_g0 is None:
            b_g0 = np.zeros(3)

        if gravity is None:
            gravity = np.array([0.0, 0.0, -9.81])

        if P0 is None:
            P0 = torch.diag(
                torch.tensor(
                    [
                        # position
                        1e-4, 1e-4, 1e-4,

                        # velocity
                        1e-3, 1e-3, 1e-3,

                        # orientation error
                        1e-4, 1e-4, 1e-4,

                        # accelerometer bias
                        1e-4, 1e-4, 1e-4,

                        # gyro bias
                        1e-5, 1e-5, 1e-5,
                    ],
                    dtype=DTYPE,
                    device=self.device,
                )
            )

        self.state = ESKFState(
            p=t(p0).reshape(3),
            v=t(v0).reshape(3),
            R=t(R0).reshape(3, 3),
            b_a=t(b_a0).reshape(3),
            b_g=t(b_g0).reshape(3),
            P=t(P0).reshape(15, 15),
        )

        self.gravity = t(gravity).reshape(3)

        self.alpha_a = float(alpha_a)
        self.alpha_g = float(alpha_g)

        self.acc_var_floor = float(acc_var_floor)
        self.gyro_var_floor = float(gyro_var_floor)

        self.ba_rw_var = float(ba_rw_var)
        self.bg_rw_var = float(bg_rw_var)

    def propagate(
        self,
        acc_m,
        gyro_m,
        dt,
        acc_var=None,
        gyro_var=None,
    ):
        """
        Propagate nominal state and 15x15 covariance for one IMU sample.

        acc_m, gyro_m:
            corrected or raw IMU measurement in body frame.

        acc_var, gyro_var:
            learned per-axis variances from AirIMU.
            If None, the fixed noise floors are used.
        """

        dt = float(dt)

        if dt <= 0.0:
            return

        s = self.state

        acc_m = torch.as_tensor(
            acc_m, dtype=DTYPE, device=self.device
        ).reshape(3)

        gyro_m = torch.as_tensor(
            gyro_m, dtype=DTYPE, device=self.device
        ).reshape(3)

        # Bias-corrected measurements
        omega = gyro_m - s.b_g
        acc_b = acc_m - s.b_a

        # Save current orientation for translational propagation
        R_k = s.R.clone()

        # Specific force -> world-frame acceleration
        a_w = R_k @ acc_b + self.gravity

        # Nominal-state propagation
        s.p = s.p + s.v * dt + 0.5 * a_w * (dt * dt)
        s.v = s.v + a_w * dt
        s.R = R_k @ so3_exp(omega * dt)

        # ------------------------------------------------------------
        # Error-state dynamics
        # dx = [dp, dv, dtheta, dba, dbg]
        # ------------------------------------------------------------

        F = torch.zeros(
            (15, 15), dtype=DTYPE, device=self.device
        )

        F[0:3, 3:6] = torch.eye(
            3, dtype=DTYPE, device=self.device
        )

        F[3:6, 6:9] = -R_k @ skew(acc_b)
        F[3:6, 9:12] = -R_k

        F[6:9, 6:9] = -skew(omega)
        F[6:9, 12:15] = -torch.eye(
            3, dtype=DTYPE, device=self.device
        )

        # Noise vector:
        # [n_a, n_g, n_ba, n_bg]
        G = torch.zeros(
            (15, 12), dtype=DTYPE, device=self.device
        )

        G[3:6, 0:3] = -R_k
        G[6:9, 3:6] = -torch.eye(
            3, dtype=DTYPE, device=self.device
        )
        G[9:12, 6:9] = torch.eye(
            3, dtype=DTYPE, device=self.device
        )
        G[12:15, 9:12] = torch.eye(
            3, dtype=DTYPE, device=self.device
        )

        # ------------------------------------------------------------
        # Discrete IMU sample variances
        #
        # AirIMU / PyPose covariance inputs are per-sample variances.
        # Therefore their state covariance contribution scales as dt^2.
        # ------------------------------------------------------------

        if acc_var is None:

            # EuRoC accelerometer noise density -> sample variance
            sigma_acc = 2.0000e-3

            qa = torch.full(
                (3,),
                (sigma_acc ** 2) / dt,
                dtype=DTYPE,
                device=self.device,
            )

        else:

            qa = torch.as_tensor(
                acc_var,
                dtype=DTYPE,
                device=self.device,
            ).reshape(3)

            # Same covariance-floor convention as the
            # learned-covariance VIO path.
            qa = torch.clamp(
                self.alpha_a * qa,
                min=self.acc_var_floor,
            )

        if gyro_var is None:

            # EuRoC gyroscope noise density -> sample variance
            sigma_gyro = 1.6968e-4

            qg = torch.full(
                (3,),
                (sigma_gyro ** 2) / dt,
                dtype=DTYPE,
                device=self.device,
            )

        else:

            qg = torch.as_tensor(
                gyro_var,
                dtype=DTYPE,
                device=self.device,
            ).reshape(3)

            qg = torch.clamp(
                self.alpha_g * qg,
                min=self.gyro_var_floor,
            )

        # ------------------------------------------------------------
        # Bias random-walk continuous spectral densities
        #
        # EuRoC:
        #   accelerometer random walk = 3.0000e-3
        #   gyroscope random walk     = 1.9393e-5
        # ------------------------------------------------------------

        qba = torch.full(
            (3,),
            self.ba_rw_var,
            dtype=DTYPE,
            device=self.device,
        )

        qbg = torch.full(
            (3,),
            self.bg_rw_var,
            dtype=DTYPE,
            device=self.device,
        )

        # ------------------------------------------------------------
        # First-order state transition
        # ------------------------------------------------------------

        I15 = torch.eye(
            15, dtype=DTYPE, device=self.device
        )

        Phi = I15 + F * dt

        # ------------------------------------------------------------
        # Discrete process covariance
        #
        # Sample IMU variances integrate with dt^2, while continuous
        # bias random-walk spectral densities integrate with dt.
        # ------------------------------------------------------------

        G_imu = G[:, 0:6]

        R_imu_sample = torch.diag(
            torch.cat(
                [
                    qa,
                    qg,
                ]
            )
        )

        Qd_imu = (
            G_imu
            @ R_imu_sample
            @ G_imu.T
        ) * (dt ** 2)

        G_bias = G[:, 6:12]

        Qc_bias = torch.diag(
            torch.cat(
                [
                    qba,
                    qbg,
                ]
            )
        )

        Qd_bias = (
            G_bias
            @ Qc_bias
            @ G_bias.T
        ) * dt

        Qd = (
            Qd_imu
            + Qd_bias
        )

        s.P = Phi @ s.P @ Phi.T + Qd

        # Numerical symmetry safeguard
        s.P = 0.5 * (s.P + s.P.T)

    def propagate_batch(
        self,
        acc,
        gyro,
        dt,
        acc_var=None,
        gyro_var=None,
    ):
        """
        Propagate over an IMU segment containing N timestamped samples.

        There are N-1 physical intervals between N samples.

        acc, gyro:
            (N,3)

        dt:
            scalar interval, or array of length N-1

        acc_var, gyro_var:
            optional (N,3)

        The measurement at sample i is used over interval
        [t_i, t_{i+1}), consistent with the preintegration code.
        """

        acc = np.asarray(acc)
        gyro = np.asarray(gyro)

        N = len(acc)

        if N < 2:
            return

        if len(gyro) != N:
            raise ValueError(
                "acc and gyro must contain the same number of samples"
            )

        if np.isscalar(dt):

            dts = np.full(
                N - 1,
                float(dt),
                dtype=np.float64,
            )

        else:

            dts = np.asarray(
                dt,
                dtype=np.float64,
            ).reshape(-1)

            if len(dts) != N - 1:
                raise ValueError(
                    f"For N={N} IMU samples, dt must contain "
                    f"N-1={N-1} intervals; got {len(dts)}."
                )

        for i in range(N - 1):

            av = (
                None
                if acc_var is None
                else acc_var[i]
            )

            gv = (
                None
                if gyro_var is None
                else gyro_var[i]
            )

            self.propagate(
                acc[i],
                gyro[i],
                dts[i],
                acc_var=av,
                gyro_var=gv,
            )


    def inject(
        self,
        delta_x,
        P_new=None,
    ):
        """
        Inject a 15-D optimized error-state correction.

        Error-state order:
            [dp, dv, dtheta, dba, dbg]

        Rotation uses the right-multiplicative convention:

            R <- R @ Exp(dtheta)

        If P_new is supplied, it must already be the posterior
        covariance expressed about the corrected nominal state.

        If P_new is None, the currently propagated covariance is
        retained and transformed by the standard error-state reset
        Jacobian.
        """

        dx = torch.as_tensor(
            delta_x,
            dtype=DTYPE,
            device=self.device,
        ).reshape(15)

        dp = dx[0:3]
        dv = dx[3:6]
        dtheta = dx[6:9]
        dba = dx[9:12]
        dbg = dx[12:15]

        s = self.state

        # Nominal-state injection
        s.p = s.p + dp
        s.v = s.v + dv

        # Right-multiplicative orientation correction.
        s.R = s.R @ so3_exp(dtheta)

        s.b_a = s.b_a + dba
        s.b_g = s.b_g + dbg

        if P_new is not None:

            P_new = torch.as_tensor(
                P_new,
                dtype=DTYPE,
                device=self.device,
            ).reshape(15, 15)

            # P_new is assumed to already be expressed around
            # the corrected nominal state.
            s.P = 0.5 * (
                P_new + P_new.T
            )

        else:

            # Error-state reset Jacobian for the right-multiplicative
            # attitude error.
            G_reset = torch.eye(
                15,
                dtype=DTYPE,
                device=self.device,
            )

            G_reset[6:9, 6:9] = (
                torch.eye(
                    3,
                    dtype=DTYPE,
                    device=self.device,
                )
                - 0.5 * skew(dtheta)
            )

            s.P = (
                G_reset
                @ s.P
                @ G_reset.T
            )

            s.P = 0.5 * (
                s.P + s.P.T
            )

        # Final numerical rotation safeguard.
        U, _, Vh = torch.linalg.svd(
            s.R
        )

        s.R = U @ Vh

        if torch.linalg.det(s.R) < 0:
            U[:, -1] *= -1.0
            s.R = U @ Vh

    def copy_state(self):
        """Return detached copies for logging / optimiser prior."""
        s = self.state

        return {
            "p": s.p.detach().clone(),
            "v": s.v.detach().clone(),
            "R": s.R.detach().clone(),
            "b_a": s.b_a.detach().clone(),
            "b_g": s.b_g.detach().clone(),
            "P": s.P.detach().clone(),
        }