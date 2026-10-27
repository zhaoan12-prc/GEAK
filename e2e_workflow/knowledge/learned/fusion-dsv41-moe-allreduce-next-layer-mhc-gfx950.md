---
name: fusion-dsv41-moe-allreduce-next-layer-mhc-gfx950
description: fuse the MoE-output TP2 all-reduce + hc_post with the NEXT layer's attention-side mHC boundary/norm/sinkhorn (cross-layer, 5→4 launches) → DSv4.1-Flash c32 ≈+0.6% e2e
keywords: [allreduce, mhc, hc-boundary, rmsnorm, sinkhorn, cross-layer, tp2, decode, cudagraph, deepseek-v4, collective-fusion]
kernels: [ar_ll128_unroll2, _hc_boundary_partial_kernel, _rmsnorm_sinkhorn_kernel]
platforms: [gfx950]
kernel_class: collective_fusion
regime: decode
key: DeepSeek-V4.1-Flash MoE-output → next-layer mHC boundary · gfx950/MI355X · sglang TP2 decode, cudagraph (allreduce + hc_post + hc_boundary RMSNorm sinkhorn)
type: lever
confidence: ★★
effect: ≈+0.6% e2e attributed on DeepSeek-V4.1-Flash AgentX c32 (iso 1.123x weighted, faster at every M 6–138: 1.085–1.162x; kernel alone 2.0x small-M → 1.3x large-M). Stack of 8 fusions measured +5.1% P90 interactivity, per-request 1.095–1.102x (2 clean interleaved 3600 s pairs); share by unit-ceiling attribution.
lifecycle: active
last_seen: 2026-10-27
---
# DSv4.1: MoE-output TP2 all-reduce + hc_post + next layer's attention-side mHC boundary (cross-layer)
- lever: layer i's MoE output AR → hc_post and layer i+1's `_hc_boundary_partial` → `_rmsnorm_sinkhorn` → wqkv_a → q_norm are one residual-stream seam that spans a layer boundary. AR + hc_post + next-layer collapse/sumsq in one kernel (shared with the attention-output AR fusion), mixing GEMV + input-norm rows in one Triton launch (no grid barrier), and the reduce + sinkhorn folded into q_norm. Per layer 5 → 4 launches.
- apply: turn on sglang's delayed-AR hook (`deepseek_v4_fused_mhc.moe_mhc_fusion` / `apply_moe_mhc`, normally reserved for TP4) at TP2 and rebind `hc_boundary`, `deepseek_v4_hip.{input_norm, q_norm_for_wq_b}` behind identity checks. Skip layers whose next layer is an Engram layer and the last layer (37/40 engage per forward on Flash).
- verify: `[overlay-F2...] ENGAGED`; residual, y, x_norm, x_q/scale, qkv, q_norm bit-exact; only part_sq reduction order differs (~1e-7 rel on pre/post/comb). Wrapper ordering: it must be the outermost wrapper of the attention-side boundary.
- caution: also verify the screen can see it — a window rule that requires both ends in the same layer (or ≤4 members) never enumerates this seam, and a collective-name list without `ar_ll128*` unit-tests it on one card without the AR. Engages only under CUDA graph, TP2.
- source: DeepSeek-V4.1-Flash InferenceX AgentX c32 study (`exp/e2e_*DeepSeek-V4.1-Flash*/` 2026-10-23..27; overlay `kernels/F2_moe_ar_mhc/`)
