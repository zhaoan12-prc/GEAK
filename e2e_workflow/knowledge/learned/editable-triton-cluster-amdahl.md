---
key: editable Triton FLA/mamba cluster · gfx942 · prefill-dominated hybrid
type: routing
confidence: ★★★
effect: per-kernel iso 1.10–1.18× real, but each ~1–3% GPU → solo e2e below the 0.5% noise band
confirms: 3
last_seen: 2026-06-09
---
# Editable Triton linear-attn cluster → STACK-and-compound, don't expect a solo e2e pass
- lever: the gated-delta / FLA / mamba Triton kernels (chunk_gated_delta_rule_fwd_h, chunk_fwd_kernel_o,
  causal_conv1d, recompute_w_u_fwd) are the best EDITABLE targets on hybrid gfx942, with large *isolated*
  wins — but each sits at ~1–3% GPU, so by Amdahl no single one moves e2e past noise in the
  prefill regime (where ~80% GPU is dense GEMM). Optimize them as a COMBINED cluster and let the
  Director's final stacked gate decide.
- apply: spend the head/config budget on dense GEMM FIRST; route the whole cluster as carry-forward and
  measure the SUM. Extract seam = modules under `sglang.srt.layers.attention.{fla,mamba}`.
- verify: **pre-dispatch screen** — if `pct_gpu × (1 − 1/plausible_iso) < NOISE_BAND_PCT (~0.5%)`, mark
  carry-forward-only (don't expect a solo gate to pass). Engagement via the overlay banner — see
  [[method-verify-engagement]]. e2e A/B via [[method-e2e-ab-harness]].
- caution: also verify the candidate is above the isolated harness's MEASUREMENT floor before
  spending the round the Amdahl screen cleared. On a low-concurrency decode regime the same
  screen passes kernels at 3-5% GPU whose per-launch time (~5-12 us) is under the harness's
  ~17 us graph-replay floor, so they cannot be RANKED at all and return ~1.00x by construction —
  a different failure from "optimized and no win". See method-decode-microkernel-measurement-floor.md.
- source: exp/e2e_*Qwen3.5-27B*/ 2026-06-05 / 06-07 / 06-09; floor caution:
  exp/e2e_*DeepSeek-R1*/ 2026-08-28 (TP8, conc 4)
