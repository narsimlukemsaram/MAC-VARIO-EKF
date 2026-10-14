# CPU MAC-VO cache reproduction + gated MAC-I2 evaluation (2026-08-13)

Real, independently-generated artifacts from rebuilding all 5 EuRoC MAC-VO caches locally,
CPU-only (no GPU used, cluster GPU capacity was needed elsewhere), and running the real gated
MAC-I2 evaluation pipeline against them.

Full write-up and results table: `../../MACVIO-study/results_gated_maci2_eval_5sequences_v1.md`.

## What's here

- `configs/` — the CPU-viable MAC-VO config (`MACVO_CPU.yaml`, non-CUDAGraph
  `FlowFormerCovFrontend`) and per-sequence data-path configs, applied on top of a plain copy of
  upstream `MAC-VO-main` (not committed here -- see `../remote_cache_reference/` for the verified
  CPU-compatibility patches these depend on: the `torch.cuda.current_stream()` guard in
  `Optimizer.py`, applied directly to `MAC-VO-main/` in this repo).
- `scripts/` — the evaluation drivers. `gated_eval_generic.py` is the one that matters (real
  gated multi-window MAC-I2 evaluation, parametrized by sequence); the others were earlier,
  single-sequence versions kept for reference (`gated_eval_full_mh02.py`,
  `compare_and_evaluate.py`, `compare_poses_correct.py` -- the last one specifically documents a
  self-caught methodology bug, see its own docstring).
- `results/` — the real per-sequence gated-evaluation logs (every window's real Mahalanobis
  score and solver output, not just the summary).

## What's not here

The actual MAC-VO caches (`tensor_map.npz`, ~50-100MB each) and the full copied MAC-VO-main
source tree used to build them are not committed -- regenerable from `MAC-VO-main/` + these
configs, gitignored to avoid duplicating ~1GB of regenerable/duplicate content already tracked
elsewhere in this repo.
