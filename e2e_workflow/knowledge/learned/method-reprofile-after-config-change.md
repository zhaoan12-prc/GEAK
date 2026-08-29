---
name: method-reprofile-after-config-change
description: An accepted config/tuning change re-routes the live seam, so a stale head queue makes the extractor's correct self-correction fail the kernel-identity gate and burns the round.
keywords: [method, reprofile, stale-queue, seam-discovery, kernel-identity, routing, budget]
kernels: [extract_op, get_gemm_config]
platforms: [gfx942, gfx950]
kernel_class: method
regime: both
key: routing hygiene - re-derive head/kernel queues from a profile of the ACCEPTED stack before dispatching extraction · any gfx · any framework
type: method
confidence: ★★
confirms: 2
effect: prevents a whole round being spent on candidates that no longer exist or no longer matter; re-ranks by the post-change bottleneck. Reproduced on the following kernel round of the same run: four sub-5% candidates consumed the round while the re-profiled #1 head (25% of GPU) was never dispatched
lifecycle: active
last_seen: 2026-08-28
---
# Re-profile after every accepted config/tuning change, BEFORE dispatching heads
- lever: a config win (backend-select env, quant swap, tuned table) changes which device kernel runs. A
  queue built from the pre-change profile names a kernel that the accepted stack no longer launches, and
  its `pct_gpu_time` is the wrong denominator. Rebuild the queue from a profile of the accepted stack.
- apply: make the re-profile a hard precondition of head/kernel dispatch, and carry the profile round id
  with each candidate so a consumer can tell whether it is stale. Re-rank by the NEW top of the profile -
  the largest remaining consumer is often a head that was never dispatched at all.
- verify: for each queued candidate, confirm its device symbol appears in the accepted-stack trace. If it
  does not, retarget the candidate to whatever the seam dispatches now rather than dropping it.
- caution: also verify the *identity gate* your harness uses. A contract that certifies an extraction only
  when the chosen seam launches the QUEUED symbol is correct against vacuous passes, but against a stale
  queue it rejects the extractor's runtime-evidence self-correction as a violation - the agent found the
  real live path and was failed for it. When a certification says "certifies '' " while the agent's notes
  name a different-but-live kernel, suspect the queue, not the agent. Related: a serial head loop with
  retries can consume the entire phase budget on candidate #1, so the biggest head never gets a turn -
  order the queue by the post-change profile and cap per-candidate retries by wall clock.
- confirm: the ordering half of this card failed again one round later — the queue was rebuilt from a
  fresh profile of the accepted stack, but ordered by editability rather than by the new pct_gpu_time,
  so four 3-5% kernels were dispatched and the 25%-of-GPU fused head was not. Order the queue by the
  post-change profile FIRST and only then filter by editability; a non-editable #1 is a routing
  question (config/tuning table, fused-op hook), not a reason to skip past it.
- source: exp/e2e_*DeepSeek-R1*/ 2026-08-28 head phase (queue was the pre-tuning round-0 order; the live
  seam had moved CK -> Triton; 3 head slots, ~2.5 h, 0 candidates, on a head worth ~1.3% of decode time)
