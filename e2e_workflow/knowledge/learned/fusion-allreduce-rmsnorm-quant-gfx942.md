---
key: kernel fusion (collective+norm+quant) · gfx942 · sglang fp8 decode TP8
type: lever
confidence: ★★
effect: -1.81% decode kernel time; launches 245 -> 121/iter (-50.6%); part of a verified +11.80% e2e stack
confirms: 1
last_seen: 2026-08-26
---
# aiter fused AllReduce+add+RMSNorm+per-group-quant — and the flag that lies about being on
- lever: on TP decode, the AllReduce → add → RMSNorm → per-group-quant chain after each
  attention/MLP block is one of the largest non-donor runs in the layer. aiter ships it fused
  (`aiter.ops.custom_all_reduce.fused_allreduce_rmsnorm_quant_per_group`); sglang has the flag.
  Worth trying on any TP>1 decode-bound serving run.
- apply: `--enable-aiter-allreduce-fusion` (sglang `server_args.py`). On DSR1/v0.5.12 the flag was
  advertised but not actually set — the baseline printed "Enable Aiter AllReduce Fusion" while the
  args dump had `enable_aiter_allreduce_fusion=False`. Check the dump, not the banner.
- verify: in the trace, `cross_device_reduce_1stage` + `add_rmsnorm_quant_kernel` should collapse into
  `allreduce_fusion_kernel_1stage<...>` (15.376+9.727 -> 18.216 us/layer-step here). The fused
  collective carries a SIZE GUARD and silently falls back above a byte threshold — confirm the fused
  kernel is in the trace at your real shape, not just that the flag is set (see [[method-verify-engagement]]).
- caution: also verify the ops you predicted would NOT change. Here `add_rmsnorm_quant` was
  pre-registered as "already enabled, should not move" and was in fact absorbed by the fused kernel —
  a correct pre-registration check catches this, an after-the-fact rewrite hides it.
- source: /raid/users/zhaoan/fusion_kernel_result/20260826_e2e/dsr1/round1 (apply/FINAL_RESULT.md §4 patch 02)
