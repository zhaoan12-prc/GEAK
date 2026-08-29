---
key: kernel fusion (weight dequant into GEMM) · gfx942 · sglang MLA fp8 decode cudagraph
type: lever
confidence: ★★★
effect: -9.13% decode kernel time; e2e TPOT -10.94% / output throughput +11.80% (verified, gsm8k-clean) for the 2-patch stack
confirms: 1
last_seen: 2026-08-26
---
# The win is deleting the per-step weight dequant, not swapping the GEMM
- lever: when an fp8 weight is dequantized to bf16 every layer every decode step just to feed a bf16
  GEMM, the fusion payoff is the dequant, not the GEMM. On DSR1 MLA V-absorb, routing the capture-mode
  path to `batched_gemm_a8w8_..._prequant_...` removed BOTH elementwise dequant kernels outright
  (17.669 + 14.793 us/layer-step -> 0) and cost 5.909 us for the fp8 batched GEMM.
- apply: sglang `srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; the eager path
  already took the fp8 route, the CUDA-graph CAPTURE path did not. A branch that differs between eager
  and capture is a good place to look — decode runs under capture, so a capture-only miss costs every step.
- verify: the dequant kernels must go to n=0/layer-step, not merely get faster. Anchor the trace
  comparison on a kernel neither patch touches (`aiter::mla_a16w8`, one per attn layer per step) and
  normalize per layer-step; check the same delta reproduces on ≥2 TP ranks (here -11.31% / -11.39%).
- caution: also verify accuracy against a SAME-CODE control, not against a single baseline. Under
  continuous batching, gsm8k moved 0.45 pp between two identical-code baseline runs — the same order as
  the baseline-vs-candidate gap, so a single-baseline comparison would have read as a real change.
- caution: a WIDER superset exists at this same seam — aiter Triton `fused_fp8_bmm_rope_cat_and_cache_mla`
  (aiter.ops.triton.fusions.fused_bmm_rope_kv_cache) fuses this V-absorb BMM together with RoPE + concat +
  the fp8 KV-cache write into ONE launch (单侧 1.11-1.12x isolated vs the live split, gfx942 decode shape
  [4,512], parity-exact). It is wireable: its fp8 output feeds `attn_mqa` (the BUILT aiter MLA decode core),
  NOT the un-built MoE preshuffle_off consumer. Seam = forward_absorb_prepare BMM + forward_absorb_core cat/
  cache-write; on gfx942 `_use_aiter_gfx95=False` so the stock path is the SPLIT form (the narrower
  `fused_qk_rope_cat_and_cache_mla` is gfx95-only). ALSO VERIFY it end-to-end next round: a reversible
  env-only overlay was authored + kernel-availability-gated PASS but could NOT be e2e-gated (2026-08-29 run:
  DeepSeek-R1-0528 weights absent + /mnt/raid0 100% full -> no server). Overlay ready to gate:
  geak_fusion_result/20260828_e2e/dsr1/fusion/fusion_overlays/dsr1/mla_headprep_decode/.
- source: /raid/users/zhaoan/fusion_kernel_result/20260826_e2e/dsr1/round1 (apply/FINAL_RESULT.md §2-4 patch 01)
