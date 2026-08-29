---
name: method-decode-microkernel-measurement-floor
description: The isolated graph-replay harness has a ~17us per-op floor, so a decode kernel under ~20us reads ~1.00x whatever the candidate does - screen for the floor before spending a round.
keywords: [method, measurement-floor, decode, low-concurrency, launch-bound, isolated-ab, graph-replay, amdahl, pre-dispatch-screen, dispatch-overhead]
kernels: [dynamic_per_group_scaled_quant_kernel, wv_splitk_small_fp16_bf16_kernel, opus_moe_sorting_entry, kn_mla_reduce_v1_ps]
platforms: [gfx942, gfx950]
kernel_class: method
regime: decode
key: can a small decode-regime kernel be RANKED at all on the isolated per-op A/B harness - low-concurrency serving (M = conc <= 8) - any gfx, any framework
type: method
confidence: ★★
confirms: 1
effect: explains and prevents a whole kernel round of unresolvable ~1.00x results; costs one baseline-only size sweep per candidate (seconds)
lifecycle: active
last_seen: 2026-08-28
---
# Screen decode kernels against the HARNESS FLOOR, not just against Amdahl
- lever: the isolated harness times a captured graph REPLAY with a cache flush per sample, and that
  scaffolding has a fixed cost — measured **~17 us/op, FLAT**. A per-group fp8 quant read 17.08/16.84/
  16.68/17.12 us at N=7168/2304/2048/512 (identical across a **14x** byte span) vs 5.5 us/call in the
  live trace; a skinny bf16 GEMV read 16.96/18.04/19.84 us at M=1/2/4 vs 9.0 us live. Under the floor
  the harness measures itself: every candidate scores ~1.00x with spread wider than any real win.
- apply: before dispatching a decode candidate, run the BASELINE ALONE over a >=10x size sweep. Flat
  `baseline_ms` => on the floor; do not rank candidates there. Re-route to a seam that aggregates many
  launches (time the fused/whole-operation call, not the leaf kernel) or to a config/tuning table the
  kernel already reads. Tell-tale: harness baseline 2-3x the traced us/call.
- verify: quote BOTH numbers in the result — traced us/call vs harness baseline us — so a ~1.00x is
  filed as UNRESOLVABLE, not as a measured no-win; only the latter should discourage a retry.
- caution: also verify the Amdahl screen did not clear the candidate for the wrong reason. At low
  concurrency the profile degenerates into a long tail of small launches (one run: 381k launches /
  4.7 s GPU = 12.4 us mean; the 3-5%-of-GPU rows were 4.8-11.7 us each), so several kernels clear a
  3-5% pct_gpu bar while all of them sit under the floor — four such directions returned four nulls in
  one round while a 25%-of-GPU fused head went untouched. Rank by time-per-launch too, not pct_gpu alone.
- source: exp/e2e_*DeepSeek-R1*/ 2026-08-28 kernel lane (fp8 block-scale MoE+MLA, TP8, conc 4; three
  independent op tasks, seven shapes); complements editable-triton-cluster-amdahl.md.
