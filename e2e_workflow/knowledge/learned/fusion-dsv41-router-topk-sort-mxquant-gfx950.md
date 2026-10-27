---
name: fusion-dsv41-router-topk-sort-mxquant-gfx950
description: fold router split-K GEMV + gate/top-k + small MoE sort + MXFP8 quant into 2 launches → DSv4.1-Flash c32 decode ≈+1.1% e2e (stack +5.1% P90)
keywords: [router, topk, moe-sorting, mxfp8, quant, moe-prologue, decode, cudagraph, deepseek-v4]
kernels: [_router_gemv_split_k_kernel, _router_gate_kernel, _moe_sorting_small_kernel_sorted, fused_mx_quant_moe_sort_kernel]
platforms: [gfx950]
kernel_class: moe_prologue
regime: decode
key: DeepSeek-V4.1-Flash MoE prologue · gfx950/MI355X · sglang TP2 decode, cudagraph (router GEMM + topk gating + moe_sorting + quant)
type: lever
confidence: ★★
effect: ≈+1.1% e2e attributed on DeepSeek-V4.1-Flash AgentX c32 (iso 1.346x c32-weighted over M 6–60: bs1–5 1.277x, bs6–10 1.442x). Stack of 8 fusions measured +5.1% P90 interactivity, per-request 1.095–1.102x (2 clean interleaved 3600 s pairs); share by unit-ceiling attribution.
lifecycle: active
last_seen: 2026-10-27
---
# DSv4.1 MoE prologue: router GEMV + gate/top-k + sort + MXFP8 quant → 2 launches
- lever: the decode MoE prologue `rocm_router_gemv_split_k` → `rocm_router_gate(num_shared=1)` → small `moe_sorting` → `fused_dynamic_mxfp8_quant_moe_sort` is 4 tiny launches per layer. Keep the live 16-row split-K GEMV tiling (bit-exact logits) and fuse gate + top-k + sort + MXFP8 quant into one route kernel; per-expert token bitmaps widen to NWB 32-bit words for M > 32 (block_m=32 sort output identical to aiter opus/small sort).
- apply: rebind `aiter.fused_moe.{moe_sorting, fused_dynamic_mxfp8_quant_moe_sort}`, `sglang.srt.layers.moe.topk.rocm_router_gate` and `deepseek_v2_hip_moe.forward_gate`; size-guard by rows (M 6–60 here).
- verify: `[overlay-e01] ENGAGED`; ids, logits, sorted ids/weights/expert ids and fp8 a1 bit-exact vs live in eager, inference_mode and graph; 20k-replay soak with zero bad windows.
- caution: also verify which router path the served M actually takes — with `ROCM_ROUTER_MAX_TOKENS=64`, M ≥ 66 runs `hgemm_bf16` → `topk_gating_kernel_opt` → `_fused_append_shared_experts` → opus sort instead (a separate seam). Set the size guard from the observed M distribution (c32 AgentX decode spans M 6–138), not from the sampled points only.
- caution: a narrower window (sort+quant only, M ≤ 36) measured ~1.27x in isolation but leaves the GEMV/gate launches and ~45% of c32 decode time uncovered.
- source: DeepSeek-V4.1-Flash InferenceX AgentX c32 study (`exp/e2e_*DeepSeek-V4.1-Flash*/` 2026-10-23..27; overlay `kernels/e01/`)
