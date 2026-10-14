#!/usr/bin/env python3
"""
train_airimu.py — trains improved AirIMU (§3.5) on EuRoC training split.

Architecture (§3.5.2):
  - Shared CNN encoder, GRU correction decoder, CNN-only uncertainty decoder
  - Learnable Σ_0 initial covariance

Training strategy:
  - Correction loss: direct supervised bias prediction (stable, no backprop
    through 200-step preintegration which causes NaN gradients)
  - Covariance loss: NLL on preintegration error (epochs > 10 only)

Training sequences: MH01, MH03, V102, V201, V203
Test  sequences:    MH02, MH04, V101, V103, V202
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.imu_network import ImprovedAirIMU, nll_covariance_loss
from geometry.lie_ops   import so3_log

TRAIN_SEQS = {
    "MH01": "machine_hall/MH_01_easy",
    "MH03": "machine_hall/MH_03_medium",
    "V102": "vicon_room1/V1_02_medium",
    "V201": "vicon_room2/V2_01_easy",
    "V203": "vicon_room2/V2_03_difficult",
}

SIG_GYRO = (1.6968e-4) ** 2
SIG_ACC  = (2.0000e-3) ** 2


# ── Dataset ───────────────────────────────────────────────────────────────────

class EuRoCIMUDataset(Dataset):
    def __init__(self, euroc_root, seqs, window=200, stride=20):
        self.segments = []
        for name, rel in seqs.items():
            imu_csv = os.path.join(euroc_root, rel, "mav0/imu0/data.csv")
            gt_csv  = os.path.join(euroc_root, rel,
                                   "mav0/state_groundtruth_estimate0/data.csv")
            if not os.path.exists(imu_csv):
                print(f"  [SKIP] {name}"); continue
            imu = np.loadtxt(imu_csv, delimiter=',', comments='#')
            gt  = np.loadtxt(gt_csv,  delimiter=',', comments='#')
            imu_t  = imu[:,0]*1e-9
            gyro   = imu[:,1:4].astype(np.float32)
            acc    = imu[:,4:7].astype(np.float32)
            gt_t   = gt[:,0]*1e-9
            gt_vel = gt[:,8:11].astype(np.float32)
            gt_bg  = gt[:,11:14].astype(np.float32)
            gt_ba  = gt[:,14:17].astype(np.float32)
            gt_quat= gt[:,4:8].astype(np.float32)
            gt_pos = gt[:,1:4].astype(np.float32)
            def interp(x):
                return np.column_stack([np.interp(imu_t,gt_t,x[:,i]) for i in range(x.shape[1])])
            vel_i  = interp(gt_vel); bg_i = interp(gt_bg); ba_i = interp(gt_ba)
            pos_i  = interp(gt_pos)
            quat_i = self._interp_quat(imu_t, gt_t, gt_quat)
            dt_i   = np.diff(imu_t, prepend=imu_t[0]-(imu_t[1]-imu_t[0])).astype(np.float32)
            N = len(imu_t)
            for s in range(0, N-window, stride):
                e = s+window
                self.segments.append({
                    "acc":   acc[s:e],   "gyro":  gyro[s:e],
                    "dt":    dt_i[s:e],
                    "pos0":  pos_i[s],   "pos1":  pos_i[e-1],
                    "vel0":  vel_i[s],   "vel1":  vel_i[e-1],
                    "quat0": quat_i[s],  "quat1": quat_i[e-1],
                    "bg":    bg_i[s:e].mean(0),
                    "ba":    ba_i[s:e].mean(0),
                })
        print(f"  Dataset: {len(self.segments)} windows from {len(seqs)} sequences")

    @staticmethod
    def _interp_quat(t_q, t_d, q_d):
        out = np.zeros((len(t_q),4), dtype=np.float32)
        for i,t in enumerate(t_q):
            idx = int(np.searchsorted(t_d,t)); idx=np.clip(idx,1,len(t_d)-1)
            a=(t-t_d[idx-1])/max(t_d[idx]-t_d[idx-1],1e-12)
            q0,q1=q_d[idx-1],q_d[idx]
            if np.dot(q0,q1)<0: q1=-q1
            q=(1-a)*q0+a*q1; out[i]=(q/(np.linalg.norm(q)+1e-12)).astype(np.float32)
        return out

    def __len__(self): return len(self.segments)
    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k,v in self.segments[idx].items()}


# ── Batched skew ──────────────────────────────────────────────────────────────

def _skew_batch(v):
    """(B,3) → (B,3,3)"""
    B = v.shape[0]
    z = torch.zeros(B, device=v.device, dtype=v.dtype)
    return torch.stack([
        torch.stack([ z,      -v[:,2],  v[:,1]], -1),
        torch.stack([ v[:,2],  z,      -v[:,0]], -1),
        torch.stack([-v[:,1],  v[:,0],  z     ], -1),
    ], -2)


# ── Batched preintegration (GPU, vectorized) ──────────────────────────────────

def batched_preintegrate(a_corr, w_corr, dt, Sigma_0, eta_acc, eta_gyro,
                         sig_gyro=SIG_GYRO, sig_acc=SIG_ACC):
    """
    (B,T,3), (B,T,3), (B,T) → gamma(B,3,3), beta(B,3), alpha(B,3), Sigma(B,9,9)
    Fully on GPU. Taylor-safe Rodrigues Exp for small angles.
    """
    B, T, _ = a_corr.shape
    device, dtype = a_corr.device, a_corr.dtype
    I3 = torch.eye(3, device=device, dtype=dtype)
    I_b = I3.unsqueeze(0).expand(B,-1,-1)

    gamma = I_b.clone()
    beta  = torch.zeros(B,3, device=device, dtype=dtype)
    alpha = torch.zeros(B,3, device=device, dtype=dtype)
    Sigma = Sigma_0.unsqueeze(0).expand(B,-1,-1).clone().to(dtype)

    for t in range(T):
        dt_k  = dt[:,t]                              # (B,)
        w_k   = w_corr[:,t,:]                        # (B,3)
        a_k   = a_corr[:,t,:]
        eta_g = eta_gyro[:,t,:].mean(-1)             # (B,)
        eta_a = eta_acc[:,t,:].mean(-1)

        phi    = w_k * dt_k.unsqueeze(-1)            # (B,3)
        angle2 = (phi*phi).sum(-1, keepdim=True).clamp(min=1e-12)  # (B,1)
        angle  = angle2.sqrt()
        K      = _skew_batch(phi)                    # (B,3,3)

        # Taylor-safe sinc and (1-cos)/angle^2
        a2 = angle2.squeeze(-1)
        sinc = torch.where(a2 < 1e-6,
                           1.0 - a2/6.0,
                           angle.squeeze(-1).sin() / angle.squeeze(-1))
        cosc = torch.where(a2 < 1e-6,
                           0.5 - a2/24.0,
                           (1.0 - angle.squeeze(-1).cos()) / a2.clamp(min=1e-12))

        dR = I_b + sinc.view(B,1,1)*K + cosc.view(B,1,1)*(K@K)

        # Right Jacobian
        c2 = torch.where(a2 < 1e-6,
                         1.0/6.0 - a2/120.0,
                         (angle.squeeze(-1) - angle.squeeze(-1).sin())
                         / (angle.squeeze(-1).pow(3).clamp(min=1e-12)))
        Jr = I_b - cosc.view(B,1,1)*K + c2.view(B,1,1)*(K@K)

        # Noise matrices
        Sg = (sig_gyro * eta_g.pow(2) / dt_k).view(B,1,1) * I_b
        Sa = (sig_acc  * eta_a.pow(2) / dt_k).view(B,1,1) * I_b

        # State transition A (9×9)
        ak_skew = _skew_batch(a_k)
        dt_     = dt_k.view(B,1,1)
        A = torch.zeros(B,9,9, device=device, dtype=dtype)
        A[:,0:3,0:3] = dR.transpose(-1,-2)
        A[:,3:6,0:3] = -gamma @ ak_skew * dt_
        A[:,6:9,0:3] = -0.5 * gamma @ ak_skew * dt_.pow(2)
        A[:,3:6,3:6] = I_b
        A[:,6:9,3:6] = I_b * dt_
        A[:,6:9,6:9] = I_b

        # Noise input B (9×6)
        Bmat = torch.zeros(B,9,6, device=device, dtype=dtype)
        Bmat[:,0:3,0:3] = Jr * dt_
        Bmat[:,3:6,3:6] = gamma * dt_
        Bmat[:,6:9,3:6] = 0.5 * gamma * dt_.pow(2)

        Q = torch.zeros(B,6,6, device=device, dtype=dtype)
        Q[:,0:3,0:3] = Sg;  Q[:,3:6,3:6] = Sa

        Sigma = A @ Sigma @ A.transpose(-1,-2) + Bmat @ Q @ Bmat.transpose(-1,-2)

        alpha = alpha + beta*dt_k.unsqueeze(-1) \
              + 0.5*(gamma @ a_k.unsqueeze(-1)).squeeze(-1)*dt_k.unsqueeze(-1)**2
        beta  = beta  + (gamma @ a_k.unsqueeze(-1)).squeeze(-1)*dt_k.unsqueeze(-1)
        gamma = gamma @ dR

    # Re-orthogonalize gamma via SVD (prevents SO(3) drift)
    U, _, Vh = torch.linalg.svd(gamma)
    gamma = U @ Vh
    gamma = gamma * torch.linalg.det(gamma).unsqueeze(-1).unsqueeze(-1).sign()

    return gamma, beta, alpha, Sigma


def quat_to_rot_batch(q):
    qw,qx,qy,qz = q[:,0],q[:,1],q[:,2],q[:,3]
    return torch.stack([
        torch.stack([1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],-1),
        torch.stack([2*(qx*qy+qw*qz),  1-2*(qx**2+qz**2),2*(qy*qz-qw*qx)],-1),
        torch.stack([2*(qx*qz-qw*qy),  2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)],-1),
    ], -2)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = EuRoCIMUDataset(args.euroc, TRAIN_SEQS,
                               window=args.window, stride=args.stride)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                         num_workers=2, pin_memory=True, drop_last=True)

    model = ImprovedAirIMU().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # Fast-forward scheduler so LR matches the correct position on the cosine
    # curve when resuming mid-training (no-op when start_epoch == 1).
    for _ in range(args.start_epoch - 1):
        sched.step()

    os.makedirs(args.out, exist_ok=True)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    best_loss = float('inf')
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        # Restore best_loss so we only save checkpoints that genuinely improve
        if "loss" in ckpt:
            best_loss = ckpt["loss"]
        print(f"Resumed from {args.resume}  "
              f"(saved epoch {ckpt.get('epoch', '?')}, loss={best_loss:.5f})")

    G = torch.tensor([0., 0., -9.81007], device=device)

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    for epoch in range(args.start_epoch, args.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()

        it = tqdm(loader, desc=f"Ep {epoch:3d}", leave=False) if use_tqdm else loader

        for batch in it:
            acc   = batch["acc"].to(device)      # (B,W,3)
            gyro  = batch["gyro"].to(device)
            dt    = batch["dt"].to(device)
            bg    = batch["bg"].to(device)       # (B,3) GT gyro bias
            ba    = batch["ba"].to(device)       # (B,3) GT accel bias
            quat0 = batch["quat0"].to(device)
            quat1 = batch["quat1"].to(device)
            vel0  = batch["vel0"].to(device)
            vel1  = batch["vel1"].to(device)
            pos0  = batch["pos0"].to(device)
            pos1  = batch["pos1"].to(device)

            # Normalize inputs: acc ~ N(2,5), gyro ~ N(0,0.3)
            acc_n = acc / 10.0
            out   = model(acc_n, gyro)

            # Scale corrections back to physical units
            a_corr = out["a_corr"] * 10.0   # (B,W,3)
            w_corr = out["w_corr"]           # (B,W,3)
            eta_a  = out["eta_acc"]
            eta_g  = out["eta_gyro"]
            S0     = out["Sigma_0"]

            # ── Correction loss: supervise bias prediction directly ──────────
            # Network correction = (corrected - raw) should equal GT bias
            # No backprop through preintegration → stable gradients
            sigma_gyro = w_corr - gyro          # (B,W,3)
            sigma_acc  = a_corr - acc           # (B,W,3)
            L_corr = (F.huber_loss(sigma_gyro.mean(dim=1), bg.float(), delta=0.005)
                    + F.huber_loss(sigma_acc.mean(dim=1),  ba.float(), delta=0.005))

            # ── Covariance loss: NLL on preintegration errors ────────────────
            # Only after warmup (epoch > 10); detach corrections so only
            # eta_acc, eta_gyro, Sigma_0 get gradients from this loss
            if epoch > 10:
                with torch.no_grad():
                    R0 = quat_to_rot_batch(quat0)
                    R1 = quat_to_rot_batch(quat1)
                    dtt = dt.sum(-1)
                    gd,bd,ad,_ = batched_preintegrate(
                        a_corr.detach(), w_corr.detach(), dt,
                        S0.detach(), eta_a.detach(), eta_g.detach())
                    v1p = vel0+(R0@bd.unsqueeze(-1)).squeeze(-1)+G*dtt.unsqueeze(-1)
                    p1p = pos0+vel0*dtt.unsqueeze(-1)\
                        +(R0@ad.unsqueeze(-1)).squeeze(-1)\
                        +0.5*G*dtt.unsqueeze(-1)**2
                    phi_e = so3_log(R1.transpose(-1,-2)@R0@gd)
                    v_e   = v1p - vel1
                    p_e   = p1p - pos1

                _,_,_,Sigma = batched_preintegrate(
                    a_corr.detach(), w_corr.detach(), dt, S0, eta_a, eta_g)
                L_cov = nll_covariance_loss(phi_e, v_e, p_e, Sigma,
                                             R0.detach())
                loss = L_corr + 0.1 * L_cov
            else:
                loss = L_corr

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        sched.step()
        avg     = np.mean([l for l in losses if l == l])   # skip NaN
        elapsed = time.time() - t0
        print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg:.5f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  t={elapsed:.0f}s")

        if avg < best_loss:
            best_loss = avg
            ckpt = os.path.join(args.out, "airimu_best.pt")
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "loss": best_loss}, ckpt)
            print(f"  → best checkpoint: {ckpt}  (loss={best_loss:.5f})")

        if epoch % 10 == 0:
            torch.save({"epoch": epoch, "state_dict": model.state_dict()},
                       os.path.join(args.out, f"airimu_ep{epoch:04d}.pt"))

    print(f"\nDone. Best loss: {best_loss:.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--euroc",       required=True)
    ap.add_argument("--out",         default="checkpoints")
    ap.add_argument("--epochs",      type=int,   default=100)
    ap.add_argument("--batch",       type=int,   default=32)
    ap.add_argument("--window",      type=int,   default=200)
    ap.add_argument("--stride",      type=int,   default=20)
    ap.add_argument("--lr",          type=float, default=1e-4)
    ap.add_argument("--device",      default="cuda:0")
    ap.add_argument("--resume",      default=None,
                    help="Path to checkpoint .pt to resume from")
    ap.add_argument("--start_epoch", type=int,   default=1,
                    help="Epoch number to start/resume from (1-indexed)")
    args = ap.parse_args()
    train(args)

if __name__ == "__main__":
    main()