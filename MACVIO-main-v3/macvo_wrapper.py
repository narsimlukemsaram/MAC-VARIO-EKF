import os
"""
macvo_wrapper.py — UMRP4 MAC-VIO  v3
Hooks into MAC-VO's MACVO odometry class to expose per-frame visual covariances.

HOOK TIMING (verified against Odometry/MACVO.py line 188):
    on_optimize_writeback fires inside run_pair(), after:
        self.Optimizer.write_map(self.graph)   ← commits pose for prev_keyframe[1]
        for func in self.on_optimize_writeback: func(self)
    At this point:
        system.prev_keyframe[1] = integer index of the JUST-WRITTEN frame (frame_k)
        The NEXT frame (frame_k+1) has not yet been pushed to graph.

    Result stored under frame index = system.prev_keyframe[1].
    Caller accesses via get_frame_result(i) for frame i (no +1 offset needed).

GRAPH API (verified against Module/Map/VisualMap.py):
    graph.frame2match  : Scaling_DenseEdge_Multi — use .project(frame_node.index)
    graph.match2frame2 : Scaling_SingleEdge      — stores frame2 index per match
    graph.match2point  : Scaling_SingleEdge      — stores point index per match
    Points data keys   : 'pos_Tw' (N,3), 'cov_Tw' (N,3,3)
    Match data keys    : 'obs2_covTc' (N,3,3), 'pixel2_uv' (N,2), 'pixel2_d' (N,1)

DATA CONFIG NOTE:
    Paper_Reproduce.yaml has  Data: !include EvaluationDataset.yaml  (list of seqs).
    We bypass cfg.Data and build EuRoC_NoIMU directly with SimpleNamespace(root=...).

USAGE:
    mac_vo = MACVOWrapper(
        macvo_root  = '/path/to/MAC-VO-main',
        config_path = 'Config/Experiment/MACVO/Paper_Reproduce.yaml',
        data_root   = '/data/EuRoC/MH_01_easy/mav0',
        device      = 'cuda:0',
    ).build()
    mac_vo.run_sequence()
    result = mac_vo.get_frame_result(1)   # FrameResult for frame index 1
"""

import sys
import torch
import numpy as np
import pypose as pp
from pathlib import Path
from types import SimpleNamespace


class FrameResult:
    """
    Container for per-frame MAC-VO outputs consumed by MAC-VIO optimizer.
    All tensors: CPU, float64.
    """
    __slots__ = [
        'frame_idx',    # int — MAC-VO frame index in VisualMap
        'time_ns',      # int — camera timestamp in nanoseconds
        'pose_SE3',     # pp.LieTensor SE3 (7,) — world←camera pose
        'T_BS',         # pp.LieTensor SE3 (7,) — body←sensor (cam) extrinsic
        'pts_pos_Tw',   # Tensor(N, 3)   — 3D point positions in world frame
        'pts_cov_Tw',   # Tensor(N, 3, 3)— 3D point covariances in world frame
        'obs_cov_Tc',   # Tensor(N, 3, 3)— obs covariances in camera frame (frame k+1)
        'kp2_uv',       # Tensor(N, 2)   — matched keypoint pixel coords
        'kp2_d',        # Tensor(N, 1)   — matched keypoint depths
        # GLOBAL POINT IDS (2026-09-08). The one field that turns a pile of per-frame point clouds
        # into a MAP. `graph.get_match2point()` already returns a PointNode bundle whose `.index` is
        # the global landmark id -- it was computed on every frame and thrown away, so the caches
        # hold N frames of anonymous points and co-observation across frames is unrecoverable.
        # A landmark seen by frames i and j with |i-j| large IS a loop closure, with data
        # association from MAC-VO's own matcher. That is the signal geometric proximity cannot give:
        # proximity returns 101,521 candidates at 0.5 m on a 14 m trajectory. Optional, so old
        # caches keep loading.
        'pts_id',       # Tensor(N,) int64 — global landmark id in VisualMap, or None
    ]

    def __init__(self, **kwargs):
        for k in self.__slots__:
            setattr(self, k, None)          # so a field absent from an old cache reads as None
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getattr__(self, name):
        """Missing slots read as None (2026-09-08).

        __init__ defaults do not help an UNPICKLED object -- pickle restores __slots__ directly and
        never calls __init__, so a field added after a cache was written raises AttributeError on
        every old cache. Caught by testing the new `pts_id` field against a 2981-frame cache from
        2026-09-06 rather than assuming backward compatibility.
        """
        if name in FrameResult.__slots__:
            return None
        raise AttributeError(name)

    def __repr__(self):
        n = len(self.pts_pos_Tw) if self.pts_pos_Tw is not None else 0
        return f"FrameResult(idx={self.frame_idx}, time_ns={self.time_ns}, n_pts={n})"

    def is_valid(self) -> bool:
        """True if visual data is available (non-None and non-empty)."""
        return (
            self.pts_pos_Tw is not None
            and len(self.pts_pos_Tw) > 0
        )


class _LimitedSequence:
    """Wraps a SequenceBase to stop after max_frames (for quick testing)."""
    def __init__(self, seq, max_frames: int):
        self._seq = seq
        self._max = max_frames

    def __len__(self):
        return min(len(self._seq), self._max)

    def __iter__(self):
        for i, item in enumerate(self._seq):
            if i >= self._max:
                break
            yield item

    def __getattr__(self, name):
        return getattr(self._seq, name)


class MACVOWrapper:
    """
    Thin wrapper around MAC-VO's MACVO system that:
      1. Configures MAC-VO from a YAML config + explicit data root
      2. Installs a hook to capture per-frame optimization results
      3. Exposes FrameResult objects for MAC-VIO consumption
    """

    def __init__(
        self,
        macvo_root:  str,
        config_path: str,
        data_root:   str,
        device:      str = 'cuda:0',
    ):
        self.macvo_root  = Path(macvo_root).resolve()
        self.config_path = config_path
        self.data_root   = str(Path(data_root).resolve())
        self.device      = device

        self._results:  dict[int, FrameResult] = {}
        self._system    = None
        self._sequence  = None

        self._add_macvo_to_path()

    def _add_macvo_to_path(self):
        s = str(self.macvo_root)
        if s not in sys.path:
            sys.path.insert(0, s)

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self) -> 'MACVOWrapper':
        """
        Import MAC-VO modules and construct the MACVO system + sequence.
        Call after __init__ — deferred so GPU allocation is explicit.
        """
        from Utility.Config import load_config
        from DataLoader import SequenceBase, StereoFrame, smart_transform
        from Odometry.MACVO import MACVO

        cfg_path = self.macvo_root / self.config_path
        # chdir to macvo_root so relative weight paths (./Model/...) resolve correctly
        _orig_cwd = os.getcwd()
        os.chdir(str(self.macvo_root))
        cfg, _   = load_config(cfg_path)

        # Build EuRoC sequence directly (bypass cfg.Data which is a list)
        # EuRoC_NoIMU requires SimpleNamespace with root (str) and gt_pose (bool)
        seq_cfg  = SimpleNamespace(root=self.data_root, gt_pose=True)
        seq_raw  = SequenceBase[StereoFrame].instantiate('EuRoC_NoIMU', seq_cfg)
        self._sequence = smart_transform(seq_raw, cfg.Preprocess)

        # Build MACVO system from config (still in macvo_root dir for weight loading)
        self._system = MACVO[StereoFrame].from_config(cfg)
        os.chdir(_orig_cwd)  # restore CWD after model load

        # Install hook — register_on_optimize_finish appends to on_optimize_writeback
        self._system.register_on_optimize_finish(self._on_optimize_finish)

        print(f"[MACVOWrapper] Built MACVO system")
        print(f"  data root  : {self.data_root}")
        print(f"  num frames : {len(self._sequence)}")
        return self

    # ── Hook ──────────────────────────────────────────────────────────────────

    def _on_optimize_finish(self, system) -> None:
        """
        Hook called by MACVO after each PGO write-back.

        Fires inside run_pair(frame_k, frame_{k+1}) after:
            Optimizer.write_map(graph)       # commits optimized pose for frame_k
            func(self)                       # ← we are called here

        system.prev_keyframe = (frame_k_data, frame_k_idx, depth_k)
        So system.prev_keyframe[1] is the integer index of the just-written frame.
        """
        if system.prev_keyframe is None:
            return  # first frame, no optimization yet

        graph         = system.graph
        opt_frame_idx = int(system.prev_keyframe[1])  # frame just written back

        # ── Read frame-level data ──────────────────────────────────────────────
        try:
            frame_node = graph.frames[torch.tensor([opt_frame_idx], dtype=torch.long)]
            time_ns    = int(frame_node.data['time_ns'][0].item())
            pose_SE3   = pp.SE3(frame_node.data['pose'][0].double().cpu())
            T_BS       = pp.SE3(frame_node.data['T_BS'][0].double().cpu())
        except Exception as e:
            print(f"[MACVOWrapper] Warning: frame node read failed for {opt_frame_idx}: {e}")
            return

        # ── Read match/point data via graph API ───────────────────────────────
        pts_pos = pts_cov = obs_cov = kp2_uv = kp2_d = pts_id = None
        try:
            frame_node_for_match = graph.frames[torch.tensor([opt_frame_idx], dtype=torch.long)]
            # get_frame2match: returns MatchObs (TensorBundle) for matches where this is frame2
            match_obs = graph.get_frame2match(frame_node_for_match)

            if len(match_obs) > 0:
                # match_obs.data holds all match features for these matches
                kp2_uv_all  = match_obs.data['pixel2_uv'].float().cpu()     # (M, 2)
                kp2_d_all   = match_obs.data['pixel2_d' ].float().cpu()     # (M, 1)
                obs_cov_all = match_obs.data['obs2_covTc'].double().cpu()   # (M, 3, 3)

                # get_match2point: returns PointNode for each match (valid points only)
                point_obs = graph.get_match2point(match_obs)                # PointNode TensorBundle

                if len(point_obs) > 0:
                    # point_obs.index gives which matches had valid points
                    # Align: match_obs rows that map to valid points
                    pt_match_mask = graph.match2point.mapping[match_obs.index] >= 0  # (M,) bool
                    pts_pos = point_obs.data['pos_Tw'].double().cpu()  # (K, 3)
                    pts_cov = point_obs.data['cov_Tw'].double().cpu()  # (K, 3, 3)
                    # the global landmark ids -- already right here, previously discarded
                    pts_id  = point_obs.index.long().cpu()             # (K,)
                    obs_cov = obs_cov_all[pt_match_mask]               # (K, 3, 3)
                    kp2_uv  = kp2_uv_all[pt_match_mask]               # (K, 2)
                    kp2_d   = kp2_d_all[pt_match_mask]                # (K, 1)

        except Exception as e:
            print(f"[MACVOWrapper] Warning: point extraction failed for frame {opt_frame_idx}: {e}")

        self._results[opt_frame_idx] = FrameResult(
            frame_idx  = opt_frame_idx,
            time_ns    = time_ns,
            pose_SE3   = pose_SE3,
            T_BS       = T_BS,
            pts_pos_Tw = pts_pos,
            pts_cov_Tw = pts_cov,
            obs_cov_Tc = obs_cov,
            kp2_uv     = kp2_uv,
            kp2_d      = kp2_d,
            pts_id     = pts_id,
        )

    # ── Sequence execution ─────────────────────────────────────────────────────

    def run_sequence(self, max_frames=None):  # int | None
        """Run MAC-VO on full or truncated sequence. max_frames limits frames processed."""
        assert self._system and self._sequence, "Call build() first"
        from Utility.Sandbox import Sandbox
        import itertools

        seq = self._sequence
        if max_frames is not None:
            # wrap sequence in a length-limited view
            seq = _LimitedSequence(seq, max_frames)
            print(f"[MACVOWrapper] Limiting to {max_frames} frames")

        exp_space = Sandbox.create(Path("/tmp/mac_vio_macvo_run"), "run")
        self._system.receive_frames(seq, exp_space)
        print(f"[MACVOWrapper] Sequence complete. Captured {len(self._results)} frames.")

    def step(self, frame):
        """Process a single frame (for online/custom loop)."""
        assert self._system, "Call build() first"
        self._system.run(frame)

    # ── Result access ──────────────────────────────────────────────────────────

    def get_frame_result(self, frame_idx: int):
        """
        Return FrameResult for frame_idx (MAC-VO VisualMap frame index).
        Returns None if frame not yet captured or optimized.
        """
        return self._results.get(frame_idx, None)

    def get_all_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (timestamps_ns, poses) for all captured frames.
        poses: (F, 7) float64 — [tx,ty,tz,qx,qy,qz,qw] world←camera
        """
        frames_sorted = sorted(self._results.keys())
        ts    = np.array([self._results[i].time_ns           for i in frames_sorted], dtype=np.int64)
        poses = torch.stack([self._results[i].pose_SE3.data  for i in frames_sorted]).numpy()
        return ts, poses

    def get_visual_map(self):
        """Return the raw VisualMap."""
        return self._system.graph if self._system else None

    def get_T_BS(self, frame_idx: int = 1) -> np.ndarray:
        """Return body←sensor 4×4 matrix for given frame (from sensor.yaml)."""
        result = self._results.get(frame_idx)
        if result is None or result.T_BS is None:
            raise ValueError(f"No T_BS available for frame {frame_idx}")
        return result.T_BS.matrix().cpu().numpy().squeeze()   # (4, 4) float64

    def n_captured(self) -> int:
        return len(self._results)

    # ── Standalone EuRoC loader ────────────────────────────────────────────────

    @classmethod
    def load_euroc_stereo_sequence(cls, macvo_root: str, data_root: str):
        """
        Load just the EuRoC stereo sequence for custom frame iteration.
        Returns SequenceBase[StereoFrame].
        """
        s = str(Path(macvo_root).resolve())
        if s not in sys.path:
            sys.path.insert(0, s)
        from DataLoader import SequenceBase, StereoFrame
        return SequenceBase[StereoFrame].instantiate(
            'EuRoC_NoIMU',
            SimpleNamespace(root=str(data_root), gt_pose=True),
        )