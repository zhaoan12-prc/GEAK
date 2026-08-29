# Learned — index of distilled experience cards

Open the cards matching your run's `(kernel_class, gfx, regime)` as **additional, advisory priors** —
they only ADD candidates to try, never remove any or replace measurement. The on-box bake-off + e2e gate
is always the judge (see `README.md` philosophy). One line per card, grouped by reuse key. **Cap: ≤40 lines.**
Confidence (a hint strength, not authority): ★ noise/unverified · ★★ single non-overlap or ≥2 consistent · ★★★ ≥2 non-overlap or verified e2e.

## dense GEMM
- [gfx950 · vLLM MXFP8 E8M0 decode-bound] dense-linear split-K/fused decode-tile Triton rewrite ★★★ **+21.8% e2e (verified, gsm8k-clean); decode-driven (converts only at high conc); grouped-MoE GEMM resists (~1.1× ceiling)** — (mxfp8-linear-decode-rewrite-gfx950.md)
- [gfx942 · sglang bf16] aiter per-shape DB tune ★★★ **+2.23% e2e (verified)** — (aiter-bf16-tuned-gemm-gfx942.md)
- [gfx942 · sglang fp8 a8w8 blockscale] **MANDATED LEVER = the CK skill** `gemm_tuning/fp8_gemm_tuning_sglang_aiter.md` (capture live (M,N,K) → aiter CK tuner → fp8_utils Triton→CK switch overlay + `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE`); baseline = the UNTUNED Triton default, so CK-tuned is the real win. The old per-(N,K) Triton config-JSON overlay is **DEPRECATED for this op (do NOT use it — it keeps the slow Triton seam live and bypasses the skill)** — (fp8-a8w8-blockscale-overlay-gfx942.md)
- [gfx950 · vLLM MXFP8 E8M0] dense `tl.dot_scaled` STATIC tiles (decode BK256/prefill BM128) ★★★ part of +12.1% e2e — (mxfp8-microscale-gemm-gfx950.md)

## MoE grouped GEMM
- [gfx950 · vLLM MXFP8 E8M0] grouped `dot_scaled` STATIC tiles (GEMM1-decode BN64+BK256) ★★★ part of +12.1% e2e — (mxfp8-microscale-gemm-gfx950.md)
- [gfx942 · vLLM int4 W4A16] per-shape fused-MoE Triton config tune via `VLLM_TUNED_CONFIG_FOLDER` (env, ZERO HBM; N=moe_int//TP) ★★★ +11-18% e2e (10 confirms, TP8 & TP4) — (moe-int4-w4a16-tune-gfx942.md)
- [gfx942 · vLLM bf16 MoE] SAME per-shape fused-MoE config-tune lever works for DENSE bf16 (dtype=None filename, gelu_tanh); no shipped E=128/N=704 config → default fallback ★★ iso 1.06-1.25×/bucket, ZERO HBM, e2e gate pending — (moe-bf16-tune-gfx942.md)

## attention
- [gfx942 · sglang hybrid prefill] `--attention-backend triton` cheap flag win ★★★ +~5% e2e — (attention-backend-triton-gfx942.md)
- [gfx942 · vLLM decode/prefill, pow2+non-pow2 KV block, +MLA TRITON_MLA, +0.21 UNIFIED_ATTENTION] live=editable in-tree Triton → Tier-C rewrite (pow2 ROCm/CK→author); op bake-off N/A ★★★ ~+1-4% — (paged-attn-nonpow2-gfx942.md)
- [gfx950 · vLLM block-sparse NSA GQA prefill] custom kernel, no lib swap; live = editable in-tree Triton → Tier-C rewrite ★★ ~5.6% head — (sparse-attn-nsa-triton-gfx950.md)

## linear-attention / FLA / mamba (editable Triton)
- [gfx942 · prefill-dominated hybrid] stack-and-compound cluster; Amdahl pre-dispatch screen ★★★ — (editable-triton-cluster-amdahl.md)

## kernel fusion
- [gfx942 · sglang MLA fp8 decode, cudagraph] fold a per-step fp8 weight DEQUANT into the GEMM (V-absorb -> `batched_gemm_a8w8_..._prequant_...`); the dequant kernels must reach n=0, not just get faster; the capture path can differ from eager ★★★ **-9.13% decode kernel time; +11.80% e2e output throughput (verified, gsm8k-clean) for the 2-patch stack** — (fusion-vabsorb-fp8-batched-gemm-gfx942.md)
- [gfx942 · sglang fp8 decode TP8] aiter fused AllReduce+add+RMSNorm+per-group-quant (`--enable-aiter-allreduce-fusion`); the banner can print while the arg is False, and the fused collective silently falls back above a size guard ★★ -1.81% decode kernel time, launches -50.6%/iter — (fusion-allreduce-rmsnorm-quant-gfx942.md)
- fusion coverage: give every phase a DENOMINATOR (fusible regions -> candidates -> execution list -> 单侧 verdict -> disposition); omission looks identical to success ★★★ — (method-fusion-coverage-denominator.md)
- split prefill/decode capture: both traces must reach the table, and rows-without-shapes is as fatal as no rows ★★ — (method-split-trace-two-phase.md)

## method (cross-model, applies to any run)
- engagement verification: one-shot stderr banner + log grep ★★★ — (method-verify-engagement.md)
- e2e A/B: pinned port, interleaved, non-overlap gate ★★★ — (method-e2e-ab-harness.md)
- cuda/HIP-graph-safe integration (the #1 e2e killer) ★★★ — (method-cudagraph-safe-integration.md)
