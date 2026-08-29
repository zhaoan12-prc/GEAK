---
name: atom-dense-gemm-seam-select-decode-gfx950
description: On an ATOM/aiter fp8 block-scale stack the dense-GEMM seam is env-selected; the CK/asm to Triton flip won +6.49% e2e at conc=4 decode AND is the prerequisite that makes Triton per-shape tables bind.
keywords: [config-tune, backend-select, seam-discovery, dense-gemm, fp8-blockscale, decode, low-concurrency, atom, aiter, triton, engagement-verification, noise-band, mla, moe]
kernels: [gemm_a8w8_blockscale_bpreshuffle_triton, kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle, fmoe_bf16_blockscaleFp8_g1u1_vs_silu]
platforms: [gfx950]
kernel_class: dense_gemm
regime: decode
key: which backend the dense fp8 a8w8 block-scale GEMM seam dispatches on an ATOM-class aiter serving stack (MLA + fused-MoE model) · gfx950/MI355X · low-concurrency decode
type: lever
confidence: ★★★
confirms: 1
effect: env-only, ZERO HBM. +6.49% e2e attributable (n=3 same-session, own-node) from flipping the dense-GEMM seam CK->Triton; composes multiplicatively with the per-shape M-bucket tune it unlocks (1.0649 x 1.0424 ~= the Director-validated 1.1169x / +11.69%, TPOT -10.69%, parity pass). Head share of the CK dense GEMM went 15.47% -> 0.97% prefill-only.
lifecycle: active
last_seen: 2026-08-29
---
# The dense-GEMM seam on an ATOM-class stack is an ENV choice — sweep the COMBINATION, at n>=3
- lever: an ATOM-class server picks its dense-GEMM / MoE / MLA implementations from a family of coarse
  `*_USE_TRITON_{GEMM,MOE,MLA}` + fusion-toggle envs. They are **not independent and not one quality
  tier**: the winning arm here was a PARTIAL combination — GEMM->Triton ON, MoE->Triton OFF, the
  DS input-rmsnorm+quant fusion OFF. Sweep combinations, not one flag at a time.
- apply: env only, no overlay, no extra HBM; it rides the final bundle's launch env. Treat it as a
  **kernel-engagement prerequisite**, not an optional config nicety: once the seam is Triton, a CK tune
  DB binds to nothing and the Triton per-shape JSON tables become the load-bearing tuning artifact
  (see gemm-mbucket-lowconc-decode-triton-tune.md). Accept the two together or neither.
- verify: two-sided, per rank — a server with the flag OFF prints a "<flag> is not turned on" banner once
  per TP rank, so the count must be N/N in the reference arm and 0/N in the candidate arm. Then RE-PROFILE:
  the flip moves the head, and any queued CK candidate is instantly stale
  (method-reprofile-after-config-change.md).
- caution: also verify at n>=3 before believing ANY arm of this family. Single-flag `USE_TRITON_GEMM=1`
  read -10.1% and the 2-flag arm -9.0%, both at n=1; the 3-flag arm read -5.1% at n=1 and **+6.49% at n=3**.
  An n=1 arm reports `spread = 0.0`, which is the ABSENCE of a noise floor, not a tight one — a real win
  was one rejection away. Also verify each sibling flag on its own merits: `USE_TRITON_MLA=1` on the same
  stack was a **-87%** catastrophe, and a `--enable-tbo all` style batch-overlap flag -53%; a shared name
  prefix implies nothing about maturity.
- caution: also verify accuracy with something better than a small gsm8k sample. A ~200-sample gsm8k has a
  measured same-config spread of 0.80-0.855 on this stack — a blow-up detector only. Here it was
  non-vacuous (arm 0.835 vs baseline 0.815, a sibling arm 0.525), but a backend swap is NOT bit-exact and
  deserves a sized accuracy gate when it ships.
- source: exp/e2e_*DeepSeek-R1*_atom_*/ 2026-08-29 (fp8 [128,128] block-scale MLA+MoE, TP8,
  ISL8k/OSL1k/conc4; config sweep n=3 + Director same-session validation, validated_win, 3 replicas/arm).
