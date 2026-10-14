"""
Improved AirIMU network  (thesis §2.5 + §3.5)

Architecture changes from original AirIMU  (§3.5.2):
  1. GRU removed from uncertainty decoder — CNN only
  2. Learnable initial covariance Σ_0  replaces Σ_ii = 0
  3. Shared CNN encoder for both correction + uncertainty branches
  4. Higher-frequency CNN encoder  (§3.5.5)

Training losses  (§3.5.3):
  Correction:   L_r, L_v, L_p  — Huber  (Eq 3.4)
  Covariance:   L^cov_r, L^cov_v, L^cov_p  — NLL  (Eq 3.5)
  World-frame cov correction  (Eq 3.6):
    Σ^r_ij = J_r^{-1}(e_r) Σ^ΔR_ij (J_r^{-1})^T
    Σ^v_ij = R^w_{b_i} Σ^Δv_ij (R^w_{b_i})^T
    Σ^p_ij = R^w_{b_i} Σ^Δp_ij (R^w_{b_i})^T
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from geometry.lie_ops import so3_log, so3_right_jac_inv


# ─────────────────────────────────────────────
# CNN Encoder
# ─────────────────────────────────────────────

class IMUCNNEncoder(nn.Module):
    """
    Shared CNN encoder for both correction and uncertainty branches.
    §3.5.5: higher-frequency encoder (smaller temporal stride = 1).

    Input:  (B, T, 6)  — concatenated [accel(3), gyro(3)]
    Output: (B, T, C)  — features per time step
    """
    def __init__(self, in_channels: int = 6, hidden: int = 128, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # Conv along time dimension (kernel_size=3, stride=1 for high-freq)
            nn.Conv1d(in_channels, hidden,     kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden,      hidden,     kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden,      out_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 6)
        x = x.transpose(1, 2)             # (B, 6, T)
        feat = self.net(x)                 # (B, C, T)
        return feat.transpose(1, 2)        # (B, T, C)


# ─────────────────────────────────────────────
# Correction Decoder  (GRU retained here — Eq 2.17)
# ─────────────────────────────────────────────

class CorrectionDecoder(nn.Module):
    """
    Predicts per-sample IMU correction  (σ^acc, σ^gyro)  via GRU.
    §2.5.1 / §3.5.2: GRU is kept in the *correction* branch only.
    """
    def __init__(self, in_channels: int = 64, hidden: int = 64):
        super().__init__()
        self.gru  = nn.GRU(in_channels, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 6)   # 3 accel + 3 gyro corrections

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        feat  : (B, T, C)
        Returns σ_acc(B,T,3), σ_gyro(B,T,3)
        """
        h, _ = self.gru(feat)              # (B, T, hidden)
        out   = self.head(h)               # (B, T, 6)
        return out[..., :3], out[..., 3:]


# ─────────────────────────────────────────────
# Uncertainty Decoder  (CNN only — no GRU)
# ─────────────────────────────────────────────

class UncertaintyDecoder(nn.Module):
    """
    Predicts per-sample uncertainty  (η^acc, η^gyro)  — CNN-only.
    §3.5.2: GRU removed; patterns from Σ_0 absorb initial error spike.
    Outputs are log-scale to guarantee positivity.
    """
    def __init__(self, in_channels: int = 64, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),          # log(η_acc(3)) + log(η_gyro(3))
        )

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns η_acc(B,T,3), η_gyro(B,T,3)  — in variance units (squared)
        """
        log_eta = self.net(feat)            # (B, T, 6)
        # exp ensures positivity; clamp prevents collapse/explosion
        eta = torch.exp(log_eta.clamp(-10, 10))
        return eta[..., :3], eta[..., 3:]


# ─────────────────────────────────────────────
# Full improved AirIMU model
# ─────────────────────────────────────────────

class ImprovedAirIMU(nn.Module):
    """
    Full improved AirIMU  (§3.5).

    Forward pass produces:
      - corrected measurements  ã, ω̃  (Eq 2.17)
      - per-sample uncertainty   η_acc, η_gyro

    Key novelty: learnable Σ_0  (init_log_cov) passed into preintegration
    as starting covariance  Σ_ii = Σ_0 instead of 0  (§3.5.2).
    """
    def __init__(
        self,
        enc_hidden:  int = 128,
        enc_out:     int = 64,
        dec_hidden:  int = 64,
    ):
        super().__init__()
        self.encoder      = IMUCNNEncoder(6, enc_hidden, enc_out)
        self.corr_decoder = CorrectionDecoder(enc_out, dec_hidden)
        self.unc_decoder  = UncertaintyDecoder(enc_out, dec_hidden)

        # §3.5.2: learnable initial covariance Σ_0
        # Parameterised as log-diagonal of a 9×9 covariance (state: [γ,β,α])
        # Full matrix via Cholesky could also be used; log-diag is simpler & stable.
        self.init_log_cov = nn.Parameter(torch.full((9,), -4.0))   # initialise small

    def forward(
        self,
        a_raw: torch.Tensor,   # (B, T, 3)  raw accelerometer
        w_raw: torch.Tensor,   # (B, T, 3)  raw gyroscope
    ) -> dict:
        imu = torch.cat([a_raw, w_raw], dim=-1)   # (B, T, 6)
        feat = self.encoder(imu)                   # (B, T, C)

        sigma_acc, sigma_gyro = self.corr_decoder(feat)     # (B,T,3) each
        eta_acc,   eta_gyro   = self.unc_decoder(feat)      # (B,T,3) each

        # Corrected measurements  Eq 2.17
        a_corr = a_raw + sigma_acc   # ã = â + σ^acc  (η excluded from mean)
        w_corr = w_raw + sigma_gyro  # ω̃ = ω̂ + σ^gyro

        # Learnable Σ_0  (9×9 diagonal)
        Sigma_0 = torch.diag(torch.exp(self.init_log_cov))   # (9, 9)

        return {
            "a_corr":    a_corr,       # (B, T, 3)
            "w_corr":    w_corr,       # (B, T, 3)
            "eta_acc":   eta_acc,      # (B, T, 3)  per-frame accel uncertainty
            "eta_gyro":  eta_gyro,     # (B, T, 3)  per-frame gyro  uncertainty
            "Sigma_0":   Sigma_0,      # (9, 9)     learnable initial covariance
        }

    @property
    def learnable_init_cov(self) -> torch.Tensor:
        return torch.diag(torch.exp(self.init_log_cov))


# ─────────────────────────────────────────────
# Training Losses
# ─────────────────────────────────────────────

def correction_loss(
    q_pred: torch.Tensor,   # (B, 4) or (B, 3, 3) quaternion/rotation
    q_gt:   torch.Tensor,
    v_pred: torch.Tensor,   # (B, 3)
    v_gt:   torch.Tensor,
    p_pred: torch.Tensor,   # (B, 3)
    p_gt:   torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    Eq 3.4:  L_r = ||Log(q̂^w_{b_j} q^w_{b_j})||_h
             L_v = ||v^w_{b_j} - v̂^w_{b_j}||_h
             L_p = ||p^w_{b_j} - p̂^w_{b_j}||_h
    """
    # Rotation error via SO(3) log
    if q_pred.shape[-1] == 3 and q_pred.ndim >= 3:
        # rotation matrix path
        R_err = q_gt.transpose(-1, -2) @ q_pred
        phi_err = so3_log(R_err)
    else:
        # treat as already a tangent-space error
        phi_err = q_pred - q_gt
    Lr = F.huber_loss(phi_err, torch.zeros_like(phi_err), delta=delta, reduction='mean')
    Lv = F.huber_loss(v_pred, v_gt, delta=delta, reduction='mean')
    Lp = F.huber_loss(p_pred, p_gt, delta=delta, reduction='mean')
    return Lr + Lv + Lp


def nll_covariance_loss(
    phi_err: torch.Tensor,     # (B, 3)  rotation error in tangent space
    v_err:   torch.Tensor,     # (B, 3)  velocity error
    p_err:   torch.Tensor,     # (B, 3)  position error
    Sigma:   torch.Tensor,     # (B, 9, 9)  full preintegration covariance
    R_w_bi:  torch.Tensor,     # (B, 3, 3)  rotation world←body_i  for frame correction
) -> torch.Tensor:
    """
    Eq 3.5 + 3.6 (world-frame covariance correction):

    Σ^r_ij = J_r^{-1}(e_r) Σ^ΔR_ij (J_r^{-1})^T
    Σ^v_ij = R^w_{b_i} Σ^Δv_ij (R^w_{b_i})^T
    Σ^p_ij = R^w_{b_i} Σ^Δp_ij (R^w_{b_i})^T

    L^cov = ½ (||e_r||²_{Σ^r} + ln det Σ^r)
          + ½ (||e_v||²_{Σ^v} + ln det Σ^v)
          + ½ (||e_p||²_{Σ^p} + ln det Σ^p)
    """
    # Extract 3×3 blocks from 9×9 Sigma  (order: [γ(0:3), β(3:6), α(6:9)])
    Sig_dR = Sigma[:, 0:3, 0:3]   # (B, 3, 3)
    Sig_dv = Sigma[:, 3:6, 3:6]
    Sig_dp = Sigma[:, 6:9, 6:9]

    # ── World-frame correction  Eq 3.6 ──────────────────────────
    Jr_inv = so3_right_jac_inv(phi_err)             # (B, 3, 3)
    Sig_r  = Jr_inv @ Sig_dR @ Jr_inv.transpose(-1, -2)
    Sig_v  = R_w_bi @ Sig_dv @ R_w_bi.transpose(-1, -2)
    Sig_p  = R_w_bi @ Sig_dp @ R_w_bi.transpose(-1, -2)

    # ── NLL per component  ───────────────────────────────────────
    def _nll_block(err, Sig):
        # err : (B, 3)
        # ½ e^T Σ^{-1} e  +  ½ ln det Σ
        eps = 1e-6 * torch.eye(3, device=Sig.device, dtype=Sig.dtype).unsqueeze(0)
        Sig_reg = Sig + eps
        L_inv = torch.linalg.cholesky(Sig_reg)                     # (B, 3, 3)
        maha  = torch.linalg.solve_triangular(L_inv, err.unsqueeze(-1), upper=False)
        maha  = maha.squeeze(-1).pow(2).sum(dim=-1)                 # (B,)
        logdet = 2.0 * L_inv.diagonal(dim1=-1, dim2=-2).log().sum(dim=-1)  # (B,)
        return 0.5 * (maha + logdet)

    loss = _nll_block(phi_err, Sig_r).mean() \
         + _nll_block(v_err,   Sig_v).mean() \
         + _nll_block(p_err,   Sig_p).mean()
    return loss
