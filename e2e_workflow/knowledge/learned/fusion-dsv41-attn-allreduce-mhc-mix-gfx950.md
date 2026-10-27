---
name: fusion-dsv41-attn-allreduce-mhc-mix-gfx950
description: fuse the attention-output TP2 all-reduce with mHC hc_post + collapse + FFN-side mixing GEMV in one HIP kernel → DSv4.1-Flash c32 decode ≈+1.2% e2e (stack +5.1% P90)
keywords: [allreduce, mhc, hc-boundary, mixing-gemv, tp2, decode, cudagraph, deepseek-v4, collective-fusion]
kernels: [ar_ll128_unroll2, _hc_boundary_partial_kernel, _rmsnorm_sinkhorn_kernel, _gemm_afp8wfp8_kernel]
platforms: [gfx950]
kernel_class: collective_fusion
regime: decode
key: DeepSeek-V4.1-Flash attention-output mHC boundary · gfx950/MI355X · sglang TP2 decode, cudagraph (allreduce + hc_post + RMSNorm + mixing GEMM)
type: lever
confidence: ★★
effect: ≈+1.2% e2e attributed on DeepSeek-V4.1-Flash AgentX c32 (iso 1.439x c32-weighted, M 1–138; seam 1.65x@M6 → 1.25x@M138). The 8-fusion stack it ships in measured +5.1% P90 interactivity / TPOT p50 −10.7% / per-request 1.095–1.102x over 2 clean interleaved 3600 s AgentX pairs; per-fusion share is by unit-ceiling attribution, not a solo A/B.
lifecycle: active
last_seen: 2026-10-27
---
# DSv4.1 mHC: TP2 attention all-reduce + hc_post + collapse + FFN mixing GEMV → one kernel
- lever: after `wo_b`, the chain `ar_ll128` (TP2 AR) → `_hc_boundary_partial` (hc_post/collapse/sum-of-squares) → FFN-side mixing GEMV is launch- and round-trip-bound at small M. One HIP kernel (80 blocks × 64-column hidden slices) reads the peer buffer, does hc_post + collapse + per-row sumsq, and runs the mixing GEMV via bf16×3 MFMA; output contract = live `_hc_boundary_partial` (+ ysq for the next norm). Per layer 3 → 2 launches.
- apply: rebind `deepseek_v4_fused_mhc.{attention_mhc_fusion, apply_attention_mhc, hc_boundary}`; the AR uses aiter CustomAllreduce IPC buffers (TP2 only). When the FFN norm+GEMV fusion (A) engages at that M, set its pending-norm state; otherwise hand off to live `rmsnorm_with_sinkhorn`, so it engages alone and above A's M range.
- verify: `[overlay-E] ENGAGED` on every rank; in a graph trace `ar_ll128_unroll2` + `_hc_boundary_partial_kernel` disappear from the attention→FFN boundary; res_out and y must be bit-exact vs live (it is, under graph).
- caution: also verify it is unit-tested on the WHOLE TP group — on a single card the AR cannot run, so a TP=1 harness silently drops it and measures only the GEMM/norm ends (0.84–1.02x). Collective-name lists that omit `ar_ll128*` route it to TP=1. Engages only under CUDA graph capture.
- caution: an earlier variant that hung the mixing GEMV off the next MoE GEMV launch was slower at every M; keep the mix inside the AR kernel.
- source: DeepSeek-V4.1-Flash InferenceX AgentX c32 study (`exp/e2e_*DeepSeek-V4.1-Flash*/` 2026-10-23..27; overlay `kernels/E/ar_hc_mix_tp2.cuh`)
