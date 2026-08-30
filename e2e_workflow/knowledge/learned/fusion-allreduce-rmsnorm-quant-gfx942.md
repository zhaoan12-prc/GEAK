---
key: kernel fusion (collective+norm+quant) · gfx942 · sglang fp8 decode TP8
type: lever
confidence: ★★★
effect: reproduced on 2 independent runs. Decode kernel time -20.0% over the fusion set (51.96 -> 41.55 us/layer); launches 245 -> 121/iter (-50.6%). e2e transfer, ISOLATED at last: **+1.583% output tok/s, non-overlapping** (167.330 -> 169.979 tok/s) — i.e. this rung ALONE is worth ~1.6%, not the +11.80% of the stack it previously shipped inside.
confirms: 2
last_seen: 2026-08-30
---
# aiter fused AllReduce+add+RMSNorm+per-group-quant — and the flag that lies about being on
- lever: on TP decode, the AllReduce → add → RMSNorm → per-group-quant chain after each
  attention/MLP block is one of the largest non-donor runs in the layer. aiter ships it fused
  (`custom_fused_ar_rms_per_group_quant` / `fused_allreduce_rmsnorm_quant_per_group`); sglang has the
  flag. Worth trying on any TP>1 decode-bound serving run.
- apply: `--enable-aiter-allreduce-fusion` (sglang `server_args.py`). On DSR1/v0.5.12 the flag was
  advertised but not actually set — the baseline printed "Enable Aiter AllReduce Fusion" while the
  args dump had `enable_aiter_allreduce_fusion=False`. Check the dump, not the banner. There are TWO
  gates on the same flag: `apply_aiter_all_reduce_fusion` and
  `LayerCommunicator.should_fuse_mlp_allreduce_with_next_layer` — neutralise both, or the seam is
  half-live. Seam for an overlay: `self.input_layernorm.forward_with_allreduce_fusion` in `prepare_attn`.
- verify: in the trace, `cross_device_reduce_1stage` + `add_rmsnorm_quant_kernel` collapse into
  `allreduce_fusion_kernel_1stage<...>` (2026-08-26: 15.376+9.727 -> 18.216 us/layer-step;
  2026-08-30: 16.28+9.72 -> 20.78 — the same numbers twice). The stronger check is that the standalone
  `dynamic_per_group_scaled_quant_kernel` loses exactly ONE call per layer at unchanged us/call.
  ENGAGED is not execution — log the path actually taken; the fused helper prints its banner even if it
  falls back on every call.
- caution: also verify the ops you predicted would NOT change (`add_rmsnorm_quant` was pre-registered
  as "already enabled, should not move" and was in fact absorbed). And **also verify the PREFILL
  shape separately**: the fused collective carries a byte guard (64 MiB) and at ISL=8192/TP8 the
  prefill collectives are 101.9 MiB, so the fusion is decode-only at that shape and TTFT does not move
  (-0.53%, not significant) — the decode win is real, the prefill win is not there to be had.
- source: /raid/users/zhaoan/fusion_kernel_result/20260826_e2e/dsr1/round1 (apply/FINAL_RESULT.md §4 patch 02)
- source: /raid/users/zhaoan/fusion_kernel_result/20260829_e2e_v1/dsr1 (COMBINED_ATTRIBUTION.md §4.1, §5 e02)
