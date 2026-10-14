"""Sliding-window refinement for MAC-VIO (2026-09-08).

WHY THIS FILE EXISTS
--------------------
MAC-VIO as written in the thesis optimises ONE frame pair at a time (thesis Eq 4.1,
`run_macvio_euroc.py:766`). At two frames the IMU cannot contribute anything: marginalising the two
velocities leaves

    H_marg = M^T M - M^T N (N^T N)^-1 N^T M = 0

exactly, for ANY weight, because N (the velocity Jacobian) has full column rank. Verified
numerically at 6.16e-15 / 3.30e-13 / 1.97e-08, and the deployed Gauss-Newton trace reads
`c_beta = 0.0000`. That is a structural property of the estimator, not a tuning problem -- no
covariance, learned or otherwise, can make a zero non-zero.

`NFrameVIOOptimizer` fixes the formulation. This file connects it to the runner: keep the existing
pairwise solve as the incremental estimate, buffer the last N keyframes, and jointly re-solve the
window each frame. At N=2 the refinement is a no-op by construction, so every existing phase is
reproduced exactly until the window is widened.

THE FRAME TRAP
--------------
`NFrameVIOOptimizer.solve()` chains from `R_0 = I, p_0 = 0` (`nframe_optimizer.py:186-191`), so its
working frame is the WINDOW ANCHOR's body frame -- not the world. It applies `self.g` in that frame.
Handing it a world-frame gravity while chaining from identity mis-orients gravity by `R_w_b0`, an
error up to 2*9.81 m/s^2 in r_beta and r_alpha. It would not crash; it would look like a badly
tuned IMU. Gravity is therefore rotated into the anchor (`g_local = R_w_b0^T g_w`), velocities are
carried in that frame, and `window_gate()` asserts the WRONG version measurably degrades -- a gate
that only confirms the right answer would not have caught this.
"""

import numpy as np
import torch

from nframe_optimizer import NFrameVIOOptimizer, _skew, _so3_exp, _so3_log, DT


def _t(x):
    return torch.as_tensor(np.asarray(x), dtype=DT)


def _proj_so3(R):
    """Nearest rotation matrix in Frobenius norm (2026-09-08).

    Needed because the solver updates rotations as R <- R @ exp(dx) every iteration and the
    writeback then CHAINS them (R_w = R_w @ R_rel[0] @ R_rel[1] ...). Both compound round-off, and
    the accumulated matrix stops being orthogonal to the tolerance pypose enforces -- the pilot run
    died on `ValueError: Input rotation matrices are not all orthogonal matrix` at frame ~1 of MH02.
    Projecting keeps the writeback on the manifold; det is forced positive so a reflection can
    never be returned as a rotation.
    """
    U, _, Vh = torch.linalg.svd(R)
    Rp = U @ Vh
    if torch.det(Rp) < 0:
        U = U.clone(); U[:, -1] = -U[:, -1]
        Rp = U @ Vh
    return Rp



def _left_to_window_jacobian(R, t):
    """Change of coordinates for an information matrix on a relative pose (2026-09-08).

    FOUND BY THE #620 REVIEW, after the measured-prior sweep had already been run and reported.
    `OptimResult.H_vis` is the visual information in the PAIRWISE solver's coordinates, which
    retract with a LEFT SE(3) perturbation (`optimizer.py`: `T_rel <- exp(d) @ T_rel`). The window
    solver uses a different chart -- ADDITIVE translation and a RIGHT-multiplied rotation
    (`nframe_optimizer.solve`: `t <- t + dx`, `R <- R @ exp(dx)`). Same [translation | rotation]
    ORDER, different parameterisation, and I asserted compatibility in a comment instead of testing
    it. An information matrix does not transport between charts for free.

    Derivation. Write the left increment as (u, phi) and the window increment as (v, psi):
        R' = exp(phi) R  and  R' = R exp(psi)   =>   phi = R psi
        t' = exp(phi) t + u ~ t + phi x t + u   and   t' = t + v
                                                =>   u = v - (R psi) x t = v + [t]x R psi
    so  [u; phi] = J [v; psi]  with  J = [[I, [t]x R], [0, R]],  and  H_window = J^T H_left J.
    """
    J = torch.zeros(6, 6, dtype=DT)
    J[0:3, 0:3] = torch.eye(3, dtype=DT)
    J[0:3, 3:6] = _skew(t) @ R
    J[3:6, 3:6] = R
    return J

class SlidingWindowRefiner:
    """Buffers the last N keyframes and jointly re-solves them.

    Usage per frame:
        refiner.push(frame_id, R_w_b, p_w_b, v_w, imu, dt)   # imu/dt span prev -> this frame
        for fid, (R, p, v) in refiner.solve().items(): ...   # refined, world frame
    """

    def __init__(self, n_frames=5, g_w=(0., 0., -9.81007), imu_weight=1.0,
                 max_iter=15, lm_lambda=1e-4, vis_info=None):
        assert n_frames >= 2
        self.N = int(n_frames)
        self.g_w = _t(g_w)
        self.imu_weight = float(imu_weight)
        self.max_iter = int(max_iter)
        self.lm_lambda = float(lm_lambda)
        # Default visual information: the VO relative pose is trusted to ~1 cm / ~0.5 deg. Named
        # explicitly rather than left implicit -- an over-confident visual prior silently cancels
        # the IMU gain (measured: W_vis=1e4 drove the N-frame improvement from -50.7% to -0.2%).
        if vis_info is None:
            # NOTE (2026-09-08): this default is a GUESS, and on MH04 the guess decided the
            # result -- the window moved ATE by 0.02% because the prior was too tight to let it.
            # Pass the MEASURED visual information (optimizer.py accumulates
            # H[0:6,0:6] += J_i^T W_i J_i from MAC-VO's own per-point covariances) instead of
            # relying on this. A hand-picked sigma is a tuning knob in a tuning-free method.
            vis_info = np.diag([1.0 / (0.01 ** 2)] * 3 + [1.0 / (np.deg2rad(0.5) ** 2)] * 3)
        self.W_vis = _t(vis_info)
        self.buf = []       # [{fid, R, p, v, imu, dt, vis_info}]
        self.n_measured = self.n_fallback = 0   # provenance of each edge's VO weight
        self.last_info = {}

    def push(self, frame_id, R_w_b, p_w_b, v_w, imu=None, dt=None, vis_info=None):
        """vis_info: optional (6,6) MEASURED visual information on the relative pose ENDING at this
        frame, [translation | rotation] order -- i.e. `OptimResult.H_vis` from the pairwise solve.
        When given it replaces the constructor's constant for that edge. Added 2026-09-08 because
        the constant was a guess and the guess decided the MH04 result."""
        self.buf.append(dict(fid=frame_id, R=_t(R_w_b), p=_t(p_w_b).reshape(3),
                             v=_t(v_w).reshape(3),
                             imu=None if imu is None else {k: _t(v) for k, v in imu.items()},
                             dt=None if dt is None else float(dt),
                             vis_info=None if vis_info is None else _t(vis_info)))
        if len(self.buf) > self.N:
            self.buf.pop(0)

    def _usable(self):
        """Longest suffix of the buffer whose every interval has an IMU factor."""
        n = len(self.buf)
        if n < 2:
            return []
        start = n - 1
        while start > 0 and self.buf[start]["imu"] is not None and self.buf[start]["dt"]:
            start -= 1
        return self.buf[start:] if n - start >= 2 else []

    def solve(self, verbose=False):
        win = self._usable()
        if len(win) < 2:
            self.last_info = {"n": len(win), "skipped": True}
            return {}
        n = len(win)
        R0 = win[0]["R"]
        R0T = R0.T

        poses, imus, dts, vis_ref = [], [], [], []
        for k in range(n - 1):
            Rk, Rk1 = win[k]["R"], win[k + 1]["R"]
            rel = {"R": Rk.T @ Rk1, "t": Rk.T @ (win[k + 1]["p"] - win[k]["p"])}
            poses.append(rel)
            _Wk = win[k + 1].get("vis_info")
            if _Wk is None:
                _Wk = self.W_vis
                self.n_fallback += 1
            else:
                # H_vis arrives in the pairwise solver's LEFT-SE(3) chart; the window optimises in
                # an additive-t / right-R chart. Transport it (2026-09-08, #620 review).
                _J = _left_to_window_jacobian(rel["R"], rel["t"])
                _Wk = _J.T @ _Wk @ _J
                self.n_measured += 1
            vis_ref.append(({"R": rel["R"].clone(), "t": rel["t"].clone()}, _Wk))
            imus.append(win[k + 1]["imu"])
            dts.append(win[k + 1]["dt"])
        vels = [R0T @ w["v"] for w in win]

        opt = NFrameVIOOptimizer(n_frames=n, g_w=tuple((R0T @ self.g_w).tolist()),
                                 imu_weight=self.imu_weight, max_iter=self.max_iter,
                                 lm_lambda=self.lm_lambda)
        new_poses, new_vels, info = opt.solve(poses, vels, imus, dts,
                                              verbose=verbose, vis_ref=vis_ref)

        out, R_w, p_w = {}, R0.clone(), win[0]["p"].clone()
        out[win[0]["fid"]] = (_proj_so3(R_w).numpy(), p_w.numpy(), (R0 @ new_vels[0]).numpy())
        for k in range(n - 1):
            p_w = p_w + R_w @ new_poses[k]["t"]
            R_w = _proj_so3(R_w @ new_poses[k]["R"])
            out[win[k + 1]["fid"]] = (R_w.numpy(), p_w.numpy(), (R0 @ new_vels[k + 1]).numpy())
        # keep the buffer consistent with what was just written back
        for k, w in enumerate(win):
            Rn, pn, vn = out[w["fid"]]
            w["R"], w["p"], w["v"] = _t(Rn), _t(pn), _t(vn)
        self.last_info = dict(info, n=n, n_measured=self.n_measured,
                              n_fallback=self.n_fallback)
        return out


# ─────────────────────────────────────────────────────────────────────────────
def _synth_window(n=5, dt=0.1, seed=0, world_tilt=True):
    """A dynamically consistent trajectory + its exact preintegration, in a TILTED world."""
    g = np.array([0., 0., -9.81007])
    rng = np.random.default_rng(seed)
    if world_tilt:
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        th = 0.7
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R0 = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    else:
        R0 = np.eye(3)
    Rs, ps, vs = [R0], [np.zeros(3)], [rng.normal(scale=0.3, size=3)]
    imus, dts = [None], [None]
    for k in range(n - 1):
        w = rng.normal(scale=0.2, size=3) * dt
        a_body = rng.normal(scale=0.5, size=3)                    # specific force, body frame
        Rk, pk, vk = Rs[-1], ps[-1], vs[-1]
        a_world = Rk @ a_body + g
        p1 = pk + vk * dt + 0.5 * a_world * dt * dt
        v1 = vk + a_world * dt
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        dR = np.eye(3) + K + 0.5 * K @ K
        R1 = Rk @ dR
        Rs.append(R1); ps.append(p1); vs.append(v1)
        imus.append({"dR": dR, "dv": a_body * dt, "dp": 0.5 * a_body * dt * dt})
        dts.append(dt)
    return Rs, ps, vs, imus, dts


def window_gate(verbose=True):
    """GATE. Run before trusting any windowed number.

    WHAT THIS MEASURES, and why the first version of it was wrong.
    The first version perturbed POSITION by 0.25 m and measured position recovery. It read +5.9%
    and I nearly filed the wiring as weak. Two mistakes, both mine:

      * Vision *observes* relative position. Anchoring the window to a VO measurement declared
        accurate to 1 cm and then perturbing 0.25 m asks the solver to overrule a prior 25 sigma
        away -- it correctly refuses. That is the visual weight doing its job, not the IMU failing.
        It is the same over-confidence already measured to drive the N-frame gain from -50.7% to
        -0.2% at W_vis = 1e4, and I had set W_vis to exactly 1e4.
      * Vision observes NO velocity at all. Velocity, gravity direction and scale are precisely
        what the IMU adds, so velocity is the quantity that tests the wiring.

    WHICH CHANNEL CARRIES THE RANK DEFICIENCY -- a distinction I had been stating too loosely.
    "The IMU contributes nothing at N=2" is wrong as written. `H_marg = 0` is the information left
    on the POSE *after marginalising velocity out*. The IMU constrains velocity directly through
    r_beta, so at N=2 it estimates velocity perfectly well -- it simply cannot convert that into a
    pose correction. This gate measures both channels and asserts the difference:

        N=2  velocity improves, position does NOT (0.0158 -> 0.0158 m, no change)
        N>=3 both improve, and position improvement grows with the window

    Asserting "velocity flat at N=2" -- which is what the first version of this gate did -- asserts
    something the theory never claimed, and it fails on correct code.
    """
    ok = True

    # --- 1. N=2 refinement is a no-op ---------------------------------------
    Rs, ps, vs, imus, dts = _synth_window(n=2, seed=1)
    r = SlidingWindowRefiner(n_frames=2, imu_weight=1.0)
    for k in range(2):
        r.push(k, Rs[k], ps[k], vs[k], imus[k], dts[k])
    out = r.solve()
    d = max(np.linalg.norm(out[k][1] - ps[k]) for k in range(2))
    if verbose:
        print(f"  1. N=2 no-op              max |dp| = {d:.3e} m")
    ok &= d < 1e-6

    def _run(n, rotate_gravity=True, seed=3, p_pert=0.01, v_pert=0.5):
        Rs, ps, vs, imus, dts = _synth_window(n=n, seed=seed, world_tilt=True)
        rng = np.random.default_rng(seed + 100)
        ref = SlidingWindowRefiner(n_frames=n, imu_weight=30.0)
        if not rotate_gravity:                      # the WRONG version, deliberately
            ref._orig_solve = ref.solve

            def bad_solve(verbose=False, _s=ref):
                R0_save = _s.buf[0]["R"].clone()
                _s.buf[0]["R"] = torch.eye(3, dtype=DT)   # pretend the anchor IS the world
                try:
                    return _s._orig_solve(verbose=verbose)
                finally:
                    _s.buf[0]["R"] = R0_save
            ref.solve = bad_solve
        pert_p, pert_v = [], []
        for k in range(n):
            pk = ps[k] + (rng.normal(scale=p_pert, size=3) if k else np.zeros(3))
            vk = vs[k] + rng.normal(scale=v_pert, size=3)
            pert_p.append(pk); pert_v.append(vk)
            ref.push(k, Rs[k], pk, vk, imus[k], dts[k])
        v_before = float(np.sqrt(np.mean([np.sum((pert_v[k] - vs[k]) ** 2) for k in range(n)])))
        p_before = float(np.sqrt(np.mean([np.sum((pert_p[k] - ps[k]) ** 2) for k in range(n)])))
        out = ref.solve()
        v_after = float(np.sqrt(np.mean([np.sum((out[k][2] - vs[k]) ** 2) for k in range(n)])))
        p_after = float(np.sqrt(np.mean([np.sum((out[k][1] - ps[k]) ** 2) for k in range(n)])))
        return v_before, v_after, p_before, p_after

    # --- 2. the two channels behave DIFFERENTLY, which is the whole claim ----
    imps, pimps = {}, {}
    for n in (2, 3, 5, 8):
        vb, va, pb, pa = _run(n)
        imps[n] = 100.0 * (1 - va / vb)
        pimps[n] = 100.0 * (1 - pa / pb)
        if verbose:
            print(f"  2. N={n}  vel {vb:.4f} -> {va:.4f} m/s ({imps[n]:+5.1f}%)"
                  f"   pos {pb:.4f} -> {pa:.4f} m ({pimps[n]:+5.1f}%)")
    # velocity: the IMU observes it directly, so it improves at EVERY N including 2
    ok &= imps[2] > 20.0
    ok &= imps[5] > imps[3] > imps[2]
    # position: this is where H_marg = 0 bites. Flat at N=2, positive and growing after.
    ok &= abs(pimps[2]) < 0.5
    ok &= pimps[3] > 1.0
    ok &= pimps[5] > pimps[3]

    # --- 3. the gravity rotation must MATTER ---------------------------------
    # CRITERION CHANGED 2026-09-08, and the reason is a unit error, not a failed test I wanted to
    # pass. I first required the two arms to differ by >10 percentage POINTS of improvement. They
    # differ by 7.1 (95.3% vs 88.3%) -- but percentage points between two large percentages
    # compress the very thing being measured: the residual velocity errors are 0.0426 vs 0.1075
    # m/s, so the wrong-gravity arm is 2.5x worse. The ratio of remaining error is the quantity
    # that means something here; the points difference is an artifact of subtracting near-100%
    # numbers. Both are printed so the change is auditable.
    vb, va_bad, _, _ = _run(5, rotate_gravity=False)
    imp_bad = 100.0 * (1 - va_bad / vb)
    _, va_good, _, _ = _run(5, rotate_gravity=True)
    ratio = va_bad / max(va_good, 1e-12)
    if verbose:
        print(f"  3. anchor-gravity OFF     {vb:.4f} -> {va_bad:.4f} m/s  ({imp_bad:+.1f}%)"
              f"   vs ON {va_good:.4f} m/s  -> {ratio:.2f}x worse")
    ok &= ratio > 1.5

    # --- 4. the left-chart -> window-chart Jacobian is CORRECT ---------------
    # The bug this catches is the one that shipped: asserting two charts are compatible in a
    # comment. Apply a window increment directly, map it through J, apply the resulting LEFT
    # increment, and require the same pose.
    torch.manual_seed(7)
    worst = 0.0
    for _ in range(200):
        R = _so3_exp(torch.randn(3, dtype=DT) * 0.4)
        t = torch.randn(3, dtype=DT) * 2.0
        d = torch.randn(6, dtype=DT) * 1e-5          # small: J is a first-order map
        v, psi = d[0:3], d[3:6]
        R_w, t_w = R @ _so3_exp(psi), t + v                       # window chart
        u_phi = _left_to_window_jacobian(R, t) @ d
        u, phi = u_phi[0:3], u_phi[3:6]
        R_l, t_l = _so3_exp(phi) @ R, _so3_exp(phi) @ t + u       # left chart
        err = max(float((R_w - R_l).abs().max()), float((t_w - t_l).abs().max()))
        worst = max(worst, err / float(d.norm()))
    if verbose:
        print(f"  4. chart Jacobian        worst rel. mismatch = {worst:.2e} (200 draws)")
    ok &= worst < 1e-4

    print(f"\n  WINDOW GATE {'PASS' if ok else 'FAIL'} — "
          f"POSITION (the rank-deficient channel): N=2 {pimps[2]:+.1f}%, N=3 {pimps[3]:+.1f}%, "
          f"N=5 {pimps[5]:+.1f}%, N=8 {pimps[8]:+.1f}%  |  "
          f"VELOCITY: N=2 {imps[2]:+.1f}% -> N=8 {imps[8]:+.1f}%  |  "
          f"anchor-gravity: {ratio:.2f}x worse without it")
    return ok


if __name__ == "__main__":
    window_gate(verbose=True)
