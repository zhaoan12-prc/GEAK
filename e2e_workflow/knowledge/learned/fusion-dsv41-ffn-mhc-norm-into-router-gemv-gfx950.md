---
name: fusion-dsv41-ffn-mhc-norm-into-router-gemv-gfx950
description: fold the FFN-side mHC boundary RMSNorm/sinkhorn into the fused router GEMV + route launches (stacked on the router/top-k fusion) → DSv4.1-Flash c32 ≈+0.6% e2e
keywords: [mhc, hc-boundary, rmsnorm, sinkhorn, router, gemv-prologue, moe-prologue, decode, cudagraph, deepseek-v4]
kernels: [_hc_boundary_partial_kernel, _rmsnorm_sinkhorn_kernel, _router_gemv_split_k_kernel]
platforms: [gfx950]
kernel_class: moe_prologue
regime: decode
key: DeepSeek-V4.1-Flash FFN-side mHC norm · gfx950/MI355X · sglang TP2 decode, cudagraph (hc_boundary RMSNorm sinkhorn + router GEMM + topk)
type: lever
confidence: ★★
effect: ≈+0.6% e2e attributed on DeepSeek-V4.1-Flash AgentX c32 (incremental iso 1.175x over the router/top-k fusion; together 1.403x, M 6–60). Stack of 8 fusions measured +5.1% P90 interactivity, per-request 1.095–1.102x (2 clean interleaved 3600 s pairs); share by unit-ceiling attribution.
lifecycle: active
last_seen: 2026-10-27
---
# DSv4.1 FFN-side mHC boundary + RMSNorm/sinkhorn folded into the router GEMV and route launch
- lever: the `_rmsnorm_sinkhorn` launch between the FFN-side mHC boundary and the router can disappear: compute RMSNorm inside the router split-K GEMV (row sumsq from the boundary partials) and the sinkhorn rows inside the route/sort launch. Three launches: `_hc_boundary_partial_ysq`, `_ffn_norm_gemv_quant`, `_route_sort_sink`.
- apply: rebind `deepseek_v4_fused_mhc.{hc_boundary, _gfx95_dense_post_attention_norm}` and `deepseek_v4_platform_stage.forward_hc_pre_from_prev_fused_boundary`; it REQUIRES the router/top-k fusion (fusion-dsv41-router-topk-sort-mxquant-gfx950) to host the norm and sinkhorn.
- verify: `[overlay-A] ENGAGED`; residual, ids, sorted buffers bit-exact; x_norm may show 1-ulp bf16 flips (rstd from 80 segmented sums vs live reduction order) — equivalent rounding.
- caution: also verify the M range — at M ≥ 66 the router is `hgemm_bf16` + `topk_gating`, which has no launch to host the norm, so the boundary stays live there (a guard, not a failure).
- source: DeepSeek-V4.1-Flash InferenceX AgentX c32 study (`exp/e2e_*DeepSeek-V4.1-Flash*/` 2026-10-23..27; overlay `kernels/A/`)
