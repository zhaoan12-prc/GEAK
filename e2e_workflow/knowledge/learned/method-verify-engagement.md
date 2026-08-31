---
key: engagement verification · any gfx · any backend
type: method
confidence: ★★★
effect: turns "did my kernel actually run live?" from a guess into proof
confirms: 4
last_seen: 2026-08-30
---
# Prove the optimized kernel ran on the LIVE serving path (don't infer from an e2e wiggle)
- lever: instrument the candidate kernel with a one-shot stderr banner and grep the server log — this
  PROVES engagement on both the bench and parity legs, instead of inferring it from a throughput delta
  (which can move for unrelated reasons).
- apply: emit `[overlay-mark] <kernel> OPTIMIZED kernel CALLED` (once) from inside the candidate; for an
  overlay rebind also look for `[overlay] injected module <path>` (N hits = N workers) and `[OVERLAY_ENGAGED]`.
- verify: ≥1 banner per worker on the live run = engaged. ZERO banners but server healthy = the seam
  missed (wrong rebind target / a self-capturing wrapper fell back to eager — see
  [[method-cudagraph-safe-integration]]). For cudagraph paths, verify engagement INSIDE the captured
  region, not just at module-injection time.
- source: exp/e2e_*Qwen3.5-27B*/ FLA overlay runs 2026-06-07 / 06-09
- 2026-08-30 (DSR1 gfx942 TP8): **the ENGAGED banner is not proof of execution.** An overlay
  printed `[overlay-e04] ENGAGED` on all 8 ranks while its fused branch ran **zero** times, because a
  gate one level above it was closed. Always emit per-rank fused AND fallback counters and require
  8/8 ranks fused with 0 fallbacks; and tag them by phase, or a legitimate prefill size-guard decline
  reads as a decode fallback.
- caution: a `sitecustomize.py` that appends to `sys.argv` must NOT identify the server by
  `sys.argv[0]`. Under `python3 -m sglang.launch_server`, `site` imports sitecustomize BEFORE runpy
  rewrites argv[0], so argv[0] is still `-m` and the guard silently does nothing — the server then
  starts without the flag and the A/B measures a same-code replicate. Match on an argument only the
  server carries (e.g. `--model-path`), and VERIFY by grepping the server log for the parsed value
  (`enable_aiter_allreduce_fusion=True`).
- source: 2026-08-30 (05_FUSION_APPLYBACK.md; applyback/COMBINED_ATTRIBUTION.md section 5)
