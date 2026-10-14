"""
airimu_wrapper.py — UMRP4 MAC-VIO  v2
Loads pretrained AirIMU CodeNet and provides corrected IMU + per-frame covariances.

Two modes:
  'precomputed' : reads from net_output.pickle saved by AirIMU's inference.py
  'online'      : runs CodeNet forward pass directly (for deployment)

PICKLE KEY NOTES (verified from AirIMU inference.py):
  The pickle from inference.py stores per-sequence dicts.
  Sequence keys vary by dataset config; common formats:
    'MH_01_easy', 'MH01', 'machine_hall/MH_01_easy'  etc.
  Call wrapper.load_precomputed(..., fuzzy_match=True) to auto-match by substring.

ACC_COV MEANING:
  acc_cov output is exp-activated variance: values in ~[1e-3, 1e-1] m²/s⁴/Hz
  Do NOT sqrt before passing to IMUPreintegrator — it expects variance not std.

BIAS CONVENTION:
  When using AirIMU corrections:
    - Pass bg=np.zeros(3), ba=np.zeros(3) to IMUPreintegrator
    - AirIMU corrected_acc/gyro already have bias removed during training
"""

import pickle
import numpy as np
import torch
from pathlib import Path


class AirIMUWrapper:
    """
    Wraps AirIMU CodeNet inference for MAC-VIO.

    Quick-start (precomputed):
        wrapper = AirIMUWrapper(mode='precomputed')
        wrapper.load_precomputed('experiments/EuRoC/codenet/net_output.pickle', 'MH_01_easy')
        seg = wrapper.get_segment(imu_idx_start, imu_idx_end)
        # seg['corrected_acc']  : (N, 3)
        # seg['corrected_gyro'] : (N, 3)
        # seg['acc_cov']        : (N, 3)  per-axis variance (NOT std)
        # seg['gyro_cov']       : (N, 3)

    Quick-start (online):
        wrapper = AirIMUWrapper(mode='online', ckpt_path='...', conf_path='...')
        seg = wrapper.run_online(acc_raw, gyro_raw, timestamps_ns)
    """

    def __init__(
        self,
        mode:      str        = 'precomputed',
        ckpt_path: object= None,
        conf_path: object= None,
        device:    str        = 'cuda:0',
    ):
        self.mode   = mode
        self.device = device

        self._data: object= None
        self._np:   dict[str, np.ndarray]= None

        # Raw EuRoC IMU index corresponding to AirIMU precomputed index 0.
        # This is sequence/preprocessing dependent and should be set explicitly.
        self.raw_index_offset = 0

        if mode == 'online':
            assert ckpt_path and conf_path, \
                "online mode requires ckpt_path and conf_path"
            self._load_network(ckpt_path, conf_path, device)

    # ── Network loading ───────────────────────────────────────────────────────

    def _load_network(self, ckpt_path: str, conf_path: str, device: str):
        """Load AirIMU CodeNet weights."""
        try:
            from pyhocon import ConfigFactory
            from model.net import net_dict
        except ImportError as e:
            raise ImportError(
                "AirIMU source must be on PYTHONPATH for online mode. "
                f"Add AirIMU-main/ to sys.path.\nOriginal: {e}"
            )

        conf = ConfigFactory.parse_file(conf_path)
        train_cfg = conf.train

        self.network = net_dict['codenet'](train_cfg).to(device).double()
        ckpt = torch.load(ckpt_path, map_location=device)
        self.network.load_state_dict(ckpt['model_state_dict'])
        self.network.eval()

        print(f"[AirIMUWrapper] Loaded CodeNet from {ckpt_path}")
        print(f"  checkpoint epoch : {ckpt.get('epoch', '?')}")

    # ── Precomputed mode ──────────────────────────────────────────────────────

    def load_precomputed(
        self,
        pickle_path:   str,
        sequence_name: str,
        fuzzy_match:   bool = True,
    ):
        """
        Load net_output.pickle from AirIMU's inference.py.

        Expected pickle structure:
            { sequence_name: {
                'corrected_acc':  Tensor(T, 3),
                'corrected_gyro': Tensor(T, 3),
                'acc_cov':        Tensor(T, 3),   # exp-activated variance
                'gyro_cov':       Tensor(T, 3),
                'rot':            Tensor(T, 4),   # GT rotation xyzw
                'dt':             Tensor(T, 1),
            }, ... }

        fuzzy_match: if True, tries substring matching when exact key not found.
        """
        with open(pickle_path, 'rb') as f:
            full = pickle.load(f)

        # Exact match first
        if sequence_name in full:
            key = sequence_name
        elif fuzzy_match:
            # Try substring match (e.g. 'MH_01_easy' matches 'machine_hall/MH_01_easy')
            candidates = [k for k in full.keys() if sequence_name in k or k in sequence_name]
            if len(candidates) == 0:
                raise KeyError(
                    f"Sequence '{sequence_name}' not found in pickle.\n"
                    f"Available keys: {list(full.keys())}\n"
                    f"Tip: pass the exact key from the pickle (check with pickle.load + list(d.keys()))"
                )
            key = candidates[0]
            if len(candidates) > 1:
                print(f"[AirIMUWrapper] Warning: multiple fuzzy matches {candidates}, using '{key}'")
        else:
            raise KeyError(
                f"Sequence '{sequence_name}' not in pickle. "
                f"Available: {list(full.keys())}"
            )

        self._data = full[key]

        # Convert all array-like values to numpy float64
        self._np = {}
        for k, v in self._data.items():
            if isinstance(v, torch.Tensor):
                arr = v.detach().cpu().numpy().astype(np.float64)
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr.squeeze(0)  # (1, T, 3) → (T, 3)
                self._np[k] = arr
            elif isinstance(v, np.ndarray):
                self._np[k] = v.astype(np.float64)
            # skip non-array entries

        T = self._np['corrected_acc'].shape[0]
        print(f"[AirIMUWrapper] Loaded precomputed output for '{key}'")
        print(f"  total IMU frames : {T}")
        print(f"  available keys   : {list(self._np.keys())}")
        # Sanity-check acc_cov range (should be ~1e-3 to 1e-1 for EuRoC)
        if 'acc_cov' in self._np:
            acov = self._np['acc_cov']
            print(f"  acc_cov range    : [{acov.min():.3e}, {acov.max():.3e}]  mean={acov.mean():.3e}")

    def get_segment(self, start_idx: int, end_idx: int) -> dict:
        """
        Return corrected IMU + covariances for IMU index range [start, end).
        Returns numpy arrays shape (N, 3) where N = end_idx - start_idx.

        IMPORTANT: pass bg=np.zeros(3), ba=np.zeros(3) to IMUPreintegrator
        when using these outputs — AirIMU already corrected biases.
        """
        assert self._np is not None, \
            "Call load_precomputed() first (precomputed mode) or use run_online()"

        # Added 2026-06-30: convert raw EuRoC IMU indices to AirIMU precomputed indices using raw_index_offset.
        #  s, e = int(start_idx), int(end_idx)

        # Guard bounds
        # T = self._np['corrected_acc'].shape[0]
        # s = max(0, s)
        # e = min(T, e)

        # start_idx/end_idx are indices in the original EuRoC IMU stream.
        # Convert them to indices in the cropped AirIMU output.
        T = self._np['corrected_acc'].shape[0]

        s = int(start_idx) - int(self.raw_index_offset)
        e = int(end_idx)   - int(self.raw_index_offset)

        # Clamp safely to the AirIMU output range.
        s = min(T, max(0, s))
        e = min(T, max(0, e))

        if e < s:
            e = s

        return {
            'corrected_acc' : self._np['corrected_acc'] [s:e],   # (N, 3) m/s²
            'corrected_gyro': self._np['corrected_gyro'][s:e],   # (N, 3) rad/s
            'acc_cov'       : self._np['acc_cov']       [s:e],   # (N, 3) variance
            'gyro_cov'      : self._np['gyro_cov']      [s:e],   # (N, 3) variance
        }

    # ── Online mode ───────────────────────────────────────────────────────────

    def run_online(
        self,
        acc_raw:       np.ndarray,        # (N, 3) raw accelerometer m/s²
        gyro_raw:      np.ndarray,        # (N, 3) raw gyroscope rad/s
        timestamps_ns: np.ndarray,        # (N,)   nanosecond timestamps
        gt_rot:        np.ndarray= None,  # (N, 4) xyzw for gravity comp
    ) -> dict:
        """
        Run CodeNet forward pass on a segment.
        Returns same dict as get_segment().
        """
        assert self.mode == 'online', "run_online() requires mode='online'"
        assert hasattr(self, 'network'), "Network not loaded — check __init__"

        N = len(timestamps_ns)
        dt = np.diff(timestamps_ns.astype(np.float64)) * 1e-9  # (N-1,) seconds
        dt = np.clip(dt, 1e-4, 0.1)

        acc_t  = torch.tensor(acc_raw,  dtype=torch.float64).unsqueeze(0).to(self.device)
        gyro_t = torch.tensor(gyro_raw, dtype=torch.float64).unsqueeze(0).to(self.device)
        dt_t   = torch.tensor(dt,       dtype=torch.float64).unsqueeze(0).unsqueeze(-1).to(self.device)

        rot_t = None
        if gt_rot is not None:
            import pypose as pp
            rot_t = pp.SO3(
                torch.tensor(gt_rot, dtype=torch.float64).unsqueeze(0).to(self.device)
            )

        with torch.no_grad():
            # AirIMU CodeNet forward: (acc, gyro, dt, [rot]) → corrected outputs
            out = self.network(
                acc  = acc_t,
                gyro = gyro_t,
                dt   = dt_t,
                rot  = rot_t,
            )

        def _to_np(key):
            if key not in out or out[key] is None:
                return None
            v = out[key]
            if isinstance(v, torch.Tensor):
                return v[0].detach().cpu().numpy().astype(np.float64)
            return v

        return {
            'corrected_acc' : _to_np('corrected_acc'),
            'corrected_gyro': _to_np('corrected_gyro'),
            'acc_cov'       : _to_np('acc_cov'),
            'gyro_cov'      : _to_np('gyro_cov'),
        }

    # ── Timestamp alignment ────────────────────────────────────────────────────

    def imu_index_for_camera_frame(
        self,
        imu_timestamps_ns: np.ndarray,
        cam_timestamps_ns: np.ndarray,
    ) -> np.ndarray:
        """
        For each camera timestamp, find the first IMU index >= that timestamp.
        Returns int64 array of shape (len(cam_timestamps_ns),).

        Usage for frame pair (k, k+1):
            cam_imu_idx = wrapper.imu_index_for_camera_frame(imu_ts, cam_ts)
            seg = wrapper.get_segment(cam_imu_idx[k], cam_imu_idx[k+1])
        """
        return np.searchsorted(imu_timestamps_ns, cam_timestamps_ns).astype(np.int64)

    def describe(self):
        """Print summary of the loaded precomputed data."""
        if self._np is None:
            print("[AirIMUWrapper] No precomputed data loaded.")
            return
        for k, v in self._np.items():
            print(f"  {k:25s}: {v.shape}  "
                  f"mean={v.mean():.4e}  std={v.std():.4e}  "
                  f"range=[{v.min():.3e}, {v.max():.3e}]")