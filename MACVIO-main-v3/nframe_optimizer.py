"""N-frame visual-inertial optimizer — the structural fix for thesis Eq (4.1).  2026-09-08.

WHY THIS EXISTS
Eq (4.1) optimises X = {T_bk_bk+1, v_bk, v_bk+1}: ONE relative pose and TWO free velocities. That is
2 velocities against 2 IMU residual blocks (r_beta, r_alpha), exactly determined, so the free
velocities absorb every scrap of translational information the IMU supplies. Marginalising v_k out
of r_alpha with an ARBITRARY weight W gives

    H_marg = M^T M - M^T N (N^T N)^-1 N^T M,   M = W^(1/2) R_a,   N = W^(1/2)(dt R_b)

with N a full-rank 3x3 rotation, so P_col(N) = I and H_marg = 0. Both dt and W cancel. VERIFIED on
the real Jacobians at ||H||_F = 6.16e-15 (W=I), 3.30e-13 (W=diag(1,10,100)), 1.97e-08 (W=1e6 I).

MEASURED CONSEQUENCE OF ADDING FRAMES (same Jacobians, marginalising ALL velocities):
    2 frames (Eq 4.1)   0.00%  of the IMU's translational information reaches the pose
    3 frames           62.84%
    4 frames           76.04%
    5 frames           82.26%
The counting: N frames give N velocities but 2(N-1) constraints. At N=2 that is 2-vs-2, exactly
determined. From N=3 the system is over-determined and the surplus reaches the pose.

WHY NOT A VELOCITY PRIOR. A prior also recovers ~100% of the information, and was implemented and
measured end-to-end on 2026-09-07: MH02 identical, MH04 15.6x WORSE. It removes the null space by
ASSERTING the seed's velocity, and our seed velocity is our worst metric (~2.4x the thesis target).
This module asserts nothing -- it adds constraints. See
study/REFUTED_the_velocity_prior_FAILS_end_to_end_20260907.md.

GATE 3, recovery over 20 seeds (2026-09-08). A single-seed run first suggested velocity got WORSE
at N=3; that was seed noise, and drawing a direction from n=1 violated this project's own
never-publish-below-n=8 rule. Over 20 seeds, at the loose visual weight:
    N=2   translation  -0.1%   velocity  +0.7%    <- dead, as the algebra requires
    N=3   translation -34.3%   velocity -30.2%
    N=5   translation -50.7%   velocity -46.1%
Both channels improve from N=3 and improve further with window size. Raising the visual weight
DEGRADES the result (at 1e4: N=3 translation -0.2%), which is the expected behaviour -- an
over-constrained visual factor leaves the IMU nothing to contribute.

VALIDATION, both structural gates passed 2026-09-08:
  gate 1, synthetic-but-structured inputs   N=2  0.00%   N=3 70.62%   N=4 81.51%   N=5 86.42%
  gate 2, the REAL deployed Jacobians       N=2  0.00%   N=3 62.77%   N=4 75.94%   N=5 82.16%
Gate 2 reproduces the standalone analysis (62.84 / 76.04 / 82.26) to two decimals from an
independent code path, which is the point of running it twice.

Gate 1 FAILED on its first run at 70.71% for N=2, and the gate was what was wrong: 70.71% is
sqrt(0.5), the ROTATION channel surviving marginalisation exactly as it must, since r_gamma has no
velocity dependence. The deficiency is in TRANSLATION only and the full 6-DoF block averaged the
dead channel with the live one. Fixed by slicing translation rows/columns.

STATUS: NOT YET WIRED into run_macvio_euroc. Both structural gates pass, so the algebra and the
implementation agree; what remains is the caller, which currently iterates consecutive PAIRS and
must iterate WINDOWS. No end-to-end number exists yet, and after the velocity prior -- which was
also algebraically correct and made MH04 15.6x worse -- no ATE claim may be made from algebra alone.
"""
import torch

DT = torch.float64


def _skew(v):
    return torch.tensor([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]], dtype=DT)


def _so3_exp(phi):
    th = phi.norm()
    if th < 1e-12:
        return torch.eye(3, dtype=DT) + _skew(phi)
    K = _skew(phi / th)
    return torch.eye(3, dtype=DT) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def _so3_log(R):
    c = ((R.trace() - 1) * 0.5).clamp(-1., 1.)
    th = torch.acos(c)
    if th < 1e-12:
        return torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    return (th / (2 * torch.sin(th))) * torch.stack(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


class NFrameVIOOptimizer:
    """Joint GN over N keyframes.

    State, 6(N-1) + 3N dimensional:
        [ dxi_0 .. dxi_{N-2} | dv_0 .. dv_{N-1} ]
    where dxi_k is the 6-DoF increment on the relative pose T_{b_k, b_{k+1}} and dv_k the 3-DoF
    increment on the world-frame velocity at keyframe k.

    Residuals per consecutive pair k:
        r_gamma_k = Log( dR_k^T (R_k^T R_{k+1}) )                                    (3)
        r_beta_k  = R_k^T (v_{k+1} - v_k - g dt_k) - dv_ij_k                         (3)
        r_alpha_k = R_k^T (p_{k+1} - p_k - v_k dt_k - 0.5 g dt_k^2) - dp_ij_k        (3)
    plus the visual relative-pose residual on each dxi_k.

    At N=2 this reduces exactly to thesis Eq (4.1); the gate below asserts that numerically.
    """

    def __init__(self, n_frames, g_w=(0., 0., -9.81007), imu_weight=1.0,
                 max_iter=15, tol=1e-8, lm_lambda=1e-4):
        assert n_frames >= 2, "N-frame optimizer needs at least 2 frames"
        self.N = int(n_frames)
        self.g = torch.tensor(g_w, dtype=DT)
        self.imu_weight = float(imu_weight)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.lm_lambda = float(lm_lambda)
        self.n_pose = self.N - 1
        self.dim = 6 * self.n_pose + 3 * self.N

    def _idx_pose(self, k):
        return slice(6 * k, 6 * k + 6)

    def _idx_vel(self, k):
        return slice(6 * self.n_pose + 3 * k, 6 * self.n_pose + 3 * k + 3)

    def imu_block(self, k, R_k, p_k, p_k1, v_k, v_k1, imu, dt):
        """Residual and Jacobian blocks for IMU pair k. Returns (r, J_pose, J_vk, J_vk1)."""
        Rt = R_k.T
        r_gamma = _so3_log(imu["dR"].T @ (Rt @ imu["R_k1"]))
        r_beta = Rt @ (v_k1 - v_k - self.g * dt) - imu["dv"]
        r_alpha = Rt @ (p_k1 - p_k - v_k * dt - 0.5 * self.g * dt * dt) - imu["dp"]
        r = torch.cat([r_gamma, r_beta, r_alpha])

        Z, I3 = torch.zeros(3, 3, dtype=DT), torch.eye(3, dtype=DT)
        # d r_alpha / d p_rel is R_k^T; d r_beta / d v_k = -R_k^T, / d v_k1 = +R_k^T
        J_pose = torch.zeros(9, 6, dtype=DT)
        J_pose[6:9, 0:3] = Rt                      # translation part of dxi
        J_pose[0:3, 3:6] = I3                      # rotation part, first order
        J_vk = torch.cat([Z, -Rt, -Rt * dt], dim=0)
        J_vk1 = torch.cat([Z, Rt, Z], dim=0)
        return r, J_pose, J_vk, J_vk1

    def marginalised_pose_information(self, imu_blocks, W=None):
        """H_marg after eliminating ALL velocities. The quantity that is 0 at N=2.

        This is the diagnostic that motivated the module and it is exact, not a simulation:
        assemble Hxx, Hxv, Hvv from the same Jacobians the solver uses, then Schur-complement.
        """
        npose, nvel = self.n_pose, self.N
        Hxx = torch.zeros(6 * npose, 6 * npose, dtype=DT)
        Hxv = torch.zeros(6 * npose, 3 * nvel, dtype=DT)
        Hvv = torch.zeros(3 * nvel, 3 * nvel, dtype=DT)
        for k, (_, Jp, Jvk, Jvk1) in enumerate(imu_blocks):
            Wk = torch.eye(9, dtype=DT) if W is None else W[k]
            Hxx[6*k:6*k+6, 6*k:6*k+6] += Jp.T @ Wk @ Jp
            Hxv[6*k:6*k+6, 3*k:3*k+3] += Jp.T @ Wk @ Jvk
            Hxv[6*k:6*k+6, 3*(k+1):3*k+6] += Jp.T @ Wk @ Jvk1
            for (a, Ja) in ((k, Jvk), (k + 1, Jvk1)):
                for (b, Jb) in ((k, Jvk), (k + 1, Jvk1)):
                    Hvv[3*a:3*a+3, 3*b:3*b+3] += Ja.T @ Wk @ Jb
        H = Hxx - Hxv @ torch.linalg.pinv(Hvv) @ Hxv.T
        # TRANSLATION ONLY. The gate's first version measured the full 6-DoF pose block and read
        # 70.71% at N=2 -- which is sqrt(0.5), i.e. the ROTATION channel surviving marginalisation
        # exactly as it should (r_gamma has no velocity dependence at all, so nothing can absorb
        # it). The rank deficiency is in TRANSLATION only, and mixing the two hid it. Slice out the
        # translation rows/cols of every pose block before measuring.
        ti = torch.cat([torch.arange(6 * k, 6 * k + 3) for k in range(npose)])
        return H[ti][:, ti], Hxx[ti][:, ti]


    def solve(self, poses, vels, imus, dts, vis_blocks=None, verbose=False, vis_ref=None):
        """Joint Gauss-Newton over the whole window.

        poses  : list of N-1 dicts {"R": 3x3 relative rotation, "t": 3 relative translation}
        vels   : list of N world-frame velocities (initial guess)
        imus   : list of N-1 dicts {"dR","dv","dp"} preintegrated between consecutive keyframes
        dts    : list of N-1 interval durations
        vis_blocks : optional list of N-1 (r_vis(6), J_vis(6x6), W_vis(6x6)) on each relative pose
        vis_ref    : optional list of N-1 ({"R","t"}, W(6x6)) VO measurements to anchor each
                     relative pose against. ADDED 2026-09-08 for the sliding-window wiring.

                     Why this exists and `vis_blocks` was not enough: `vis_blocks` is evaluated
                     ONCE, outside the iteration, so its residual never updates. Anchoring the
                     window at its own initial value that way gives r_vis = 0 forever, and a
                     residual pinned at zero contributes `J^T W r = 0` to the gradient -- curvature
                     but no restoring force. The pose would drift away from the VO measurement with
                     nothing pulling it back, and the run would still look like it converged.
                     `vis_ref` recomputes the residual against a FIXED measurement each iteration,
                     which is what a prior has to do to be a prior.

        Returns (poses, vels, info). Increments are applied on the manifold: rotation by
        right-multiplied exponential, translation and velocity additively.

        CONVERGENCE IS TESTED AND REPORTED, not assumed -- fixed-iteration solves were finding B4
        in the PGO, where a differently-weighted arm could be ranked by how fast it converged.
        """
        N, npose = self.N, self.n_pose
        R = [p["R"].clone() for p in poses]
        t = [p["t"].clone() for p in poses]
        v = [x.clone() for x in vels]
        prev, iters, converged = float("inf"), 0, False

        for it in range(self.max_iter):
            H = self.lm_lambda * torch.eye(self.dim, dtype=DT)
            g = torch.zeros(self.dim, dtype=DT)
            cost = 0.0

            for k in range(npose):
                # accumulate the chain up to keyframe k so absolute quantities are available
                R_k = torch.eye(3, dtype=DT)
                p_k = torch.zeros(3, dtype=DT)
                for j in range(k):
                    p_k = p_k + R_k @ t[j]
                    R_k = R_k @ R[j]
                p_k1 = p_k + R_k @ t[k]
                imu = dict(imus[k]); imu["R_k1"] = R_k @ R[k]
                r, Jp, Jvk, Jvk1 = self.imu_block(k, R_k, p_k, p_k1, v[k], v[k + 1], imu, dts[k])
                r = r * self.imu_weight
                Jp, Jvk, Jvk1 = (J * self.imu_weight for J in (Jp, Jvk, Jvk1))
                cost += float(r @ r)

                ip, ivk, ivk1 = self._idx_pose(k), self._idx_vel(k), self._idx_vel(k + 1)
                for (ia, Ja) in ((ip, Jp), (ivk, Jvk), (ivk1, Jvk1)):
                    g[ia] += Ja.T @ r
                    for (ib, Jb) in ((ip, Jp), (ivk, Jvk), (ivk1, Jvk1)):
                        H[ia, ib] += Ja.T @ Jb

                if vis_blocks is not None:
                    r_v, J_v, W_v = vis_blocks[k]
                    cost += float(r_v @ W_v @ r_v)
                    g[ip] += J_v.T @ W_v @ r_v
                    H[ip, ip] += J_v.T @ W_v @ J_v

                if vis_ref is not None:
                    # Re-linearised each iteration against the fixed VO measurement (2026-09-08).
                    ref, W_v = vis_ref[k]
                    r_v = torch.cat([t[k] - ref["t"], _so3_log(ref["R"].T @ R[k])])
                    J_v = torch.eye(6, dtype=DT)   # first order, matching imu_block's convention
                    cost += float(r_v @ W_v @ r_v)
                    g[ip] += J_v.T @ W_v @ r_v
                    H[ip, ip] += J_v.T @ W_v @ J_v

            try:
                d = torch.linalg.solve(H, -g)
            except Exception:
                d = -torch.linalg.pinv(H) @ g
            if not torch.isfinite(d).all():
                break

            for k in range(npose):
                dx = d[self._idx_pose(k)]
                t[k] = t[k] + dx[0:3]
                R[k] = R[k] @ _so3_exp(dx[3:6])
            for k in range(N):
                v[k] = v[k] + d[self._idx_vel(k)]

            iters = it + 1
            if verbose:
                print(f"    it={it} cost={cost:.6f} |d|={float(d.norm()):.3e}")
            # BUG FOUND BY GATE 3 (2026-09-08): with prev = inf the test is abs(inf - cost) <=
            # tol * inf, i.e. inf <= inf, which is TRUE -- so the solver "converged" after ONE
            # iteration on every case and gate 3 read it as a solver that does not recover. The
            # first iteration can never be a convergence decision; there is nothing to compare to.
            if prev < float("inf") and abs(prev - cost) <= self.tol * max(abs(prev), 1.0):
                converged = True
                break
            prev = cost

        return ([{"R": R[k], "t": t[k]} for k in range(npose)], v,
                {"iters": iters, "converged": converged, "cost": cost})


def nframe_equivalence_check(verbose=True):
    """GATE: N=2 must be rank-deficient and N>=3 must not. Run before reporting any number.

    This is the same discipline that caught the --gpu_resident batch-order bug: assert the
    property the change is supposed to have, on synthetic-but-structured data, before trusting it.
    """
    torch.manual_seed(0)
    out = {}
    for N in (2, 3, 4, 5):
        opt = NFrameVIOOptimizer(N)
        blocks = []
        for k in range(N - 1):
            R_k = _so3_exp(torch.randn(3, dtype=DT) * 0.1)
            imu = {"dR": _so3_exp(torch.randn(3, dtype=DT) * 0.05),
                   "R_k1": _so3_exp(torch.randn(3, dtype=DT) * 0.1),
                   "dv": torch.randn(3, dtype=DT) * 0.1,
                   "dp": torch.randn(3, dtype=DT) * 0.1}
            b = opt.imu_block(k, R_k, torch.randn(3, dtype=DT), torch.randn(3, dtype=DT),
                              torch.randn(3, dtype=DT), torch.randn(3, dtype=DT), imu, 0.05)
            blocks.append(b)
        H, Hxx = opt.marginalised_pose_information(blocks)
        frac = float(torch.linalg.norm(H) / torch.linalg.norm(Hxx))
        out[N] = frac
        if verbose:
            print(f"  N={N}  retained {100*frac:6.2f}%  "
                  f"{'DEAD (expected at N=2)' if frac < 1e-6 else 'ALIVE'}")
    ok = out[2] < 1e-6 and all(out[n] > 1e-3 for n in (3, 4, 5))
    if verbose:
        print(f"\n  GATE {'PASS' if ok else 'FAIL'} — N=2 dead, N>=3 alive")
    return ok, out


if __name__ == "__main__":
    print("N-frame optimizer structural gate:")
    nframe_equivalence_check()
