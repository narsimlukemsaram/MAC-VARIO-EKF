"""
Visual pose covariance via PGO Hessian  (thesis §3.4, Eq 3.2)

The full 6-DoF pose covariance is extracted from the inverse of the Hessian
at convergence of the MAC-VO pose graph optimization.

Left perturbation model:
  r_{i,k}(δξ) = p^{c_{k-1}}_i  -  Exp(δξ) T^{c_{k-1}}_{c_k}  p^{c_k}_i

Jacobian  J_{i,k} = ∂r_{i,k}/∂δξ = [-I₃ₓ₃  | (T^{c_{k-1}}_{c_k} p^{c_k}_i)^]

Hessian accumulation:
  H_k ≈ Σ_i  J_{i,k}^T  Σ_i^{-1}  J_{i,k}
  Σ_{T^{c_{k-1}}_{c_k}} ≈ H_k^{-1}
"""

import torch
from geometry.lie_ops import skew, so3_log, se3_left_jac_inv


def compute_pose_covariance(
    p_world:  torch.Tensor,    # (N, 3)  3D keypoints in world/prev-camera frame
    p_cam:    torch.Tensor,    # (N, 3)  same keypoints in current camera frame
    Sigma_3d: torch.Tensor,    # (N, 3, 3)  per-keypoint 3D covariance  cΣ_i
    T_c_prev: torch.Tensor,    # (4, 4) or (3, 4)  estimated relative pose T^{c_{k-1}}_{c_k}
    reg: float = 1e-6,
) -> torch.Tensor:
    """
    Returns:  Σ_pose  (6, 6)  left-perturbation pose covariance  H_k^{-1}

    Eq 3.2:
      J_{i,k} = ∂r_{i,k}/∂δξ = [-I | (T p^c_i)^]
      H_k ≈ Σ_i J^T Σ_i^{-1} J
      Σ_T ≈ H_k^{-1}
    """
    N = p_cam.shape[0]
    device, dtype = p_cam.device, p_cam.dtype

    # Transform p_cam into prev-camera frame  T^{c_{k-1}}_{c_k} p^{c_k}_i
    R = T_c_prev[:3, :3]   # (3, 3)
    t = T_c_prev[:3,  3]   # (3,)
    p_transformed = (R @ p_cam.T).T + t   # (N, 3)

    # Build Jacobians  J_{i,k}  each (3, 6)
    # Left part: -I_{3×3}   Right part: (T p^c_i)^  skew
    I3   = torch.eye(3, device=device, dtype=dtype)          # (3, 3)
    neg_I = -I3.unsqueeze(0).expand(N, -1, -1)              # (N, 3, 3)
    p_skew = skew(p_transformed)                             # (N, 3, 3)
    J = torch.cat([neg_I, p_skew], dim=-1)                   # (N, 3, 6)

    # Accumulate Hessian  H = Σ_i J^T Σ_i^{-1} J
    H = torch.zeros(6, 6, device=device, dtype=dtype)
    for i in range(N):
        Omega_i = torch.linalg.inv(
            Sigma_3d[i] + reg * torch.eye(3, device=device, dtype=dtype)
        )                                                     # (3, 3)
        H = H + J[i].T @ Omega_i @ J[i]                     # (6, 6)

    # Regularise and invert
    Sigma_pose = torch.linalg.inv(
        H + reg * torch.eye(6, device=device, dtype=dtype)
    )
    return Sigma_pose   # (6, 6)


def visual_residual_and_covariance(
    T_c_k_prev: torch.Tensor,    # (4, 4) estimated visual pose T^{c_k}_{c_{k-1}} (odometry output)
    T_cb_k:     torch.Tensor,    # (4, 4) world pose of body at k  T^{c_0}_{b_k}
    T_cb_k1:    torch.Tensor,    # (4, 4) world pose of body at k+1 T^{c_0}_{b_{k+1}}
    T_bc:        torch.Tensor,    # (4, 4) extrinsic  T^b_c (camera-to-body)
    Sigma_pose:  torch.Tensor,    # (6, 6) pose covariance from Eq 3.2
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Visual residual for VI initialization  (Eq 3.10 + 3.11):

    r^visual_k = Log(T^{c_k}_{c_{k-1}}  T^{c_0}_{b_k+1}^{-1}  T^{c_0}_{b_k}  T^b_c)

    Σ^visual_k = (∂r/∂δξ)  Σ_{T}  (∂r/∂δξ)^T
    with ∂r/∂δξ = J_l^{-1}(r^visual_k)

    Returns:  (residual (6,), covariance (6,6))
    """
    def mat2SE3(T):
        return T[:4, :4] if T.shape[0] >= 4 else T

    # Compose the relative transformation from inertial state estimate
    # T_vis_from_imu = T_c0_bk  T_bc    (maps body→camera)
    # Full chain:  T_c0_bk T_bc  →  (T_c0_bk1)^{-1}  to get relative cam-to-cam
    T_imu = torch.linalg.inv(T_cb_k1) @ T_cb_k @ T_bc

    # Residual in SE(3) tangent space  (log of relative rotation + translation)
    T_err = T_c_k_prev @ torch.linalg.inv(T_imu)   # should be ~ I if consistent
    R_err = T_err[:3, :3]
    t_err = T_err[:3,  3]
    phi_err = so3_log(R_err)
    r_visual = torch.cat([t_err, phi_err], dim=0)   # (6,)  [trans | rot]

    # Jacobian  ∂r/∂δξ = J_l^{-1}(r)
    J_inv = se3_left_jac_inv(r_visual)              # (6, 6)
    Sigma_vis = J_inv @ Sigma_pose @ J_inv.T         # (6, 6)

    return r_visual, Sigma_vis
