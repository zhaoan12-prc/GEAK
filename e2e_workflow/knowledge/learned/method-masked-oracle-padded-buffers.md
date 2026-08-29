---
name: method-masked-oracle-padded-buffers
description: Ops that write capacity-padded buffers (MoE sorting/scatter, split-K workspaces) leave the tail past the valid count undefined and nondeterministic - mask the oracle to the valid prefix.
keywords: [method, oracle, correctness, extraction, moe-sorting, padding, nondeterminism, unittest]
kernels: [moe_sorting, grouped_topk, sorted_token_ids, num_valid_ids]
platforms: [gfx942, gfx950]
kernel_class: method
regime: both
key: building the immutable correctness oracle for an op whose outputs are capacity-sized with a runtime valid-length · any gfx · any framework
type: method
confidence: ★★
confirms: 1
effect: turns a "kernel is non-reproducible / extraction failed" dead end into a measurable direction; costs one masking pass over the reference tensors
lifecycle: active
last_seen: 2026-08-28
---
# Mask the oracle to the DEFINED prefix when an op writes capacity-padded outputs
- lever: MoE sorting/scatter style ops allocate worst-case-sized outputs and only define the first
  `num_valid_*` entries; the tail keeps whatever the allocator had. A full-buffer bit-exact comparison then
  fails on garbage and the direction is written off as unreproducible, even though the op is deterministic
  where it matters.
- apply: read the op's own valid-length output first, then compare only `out[:num_valid]` (and the derived
  block/expert arrays only up to their own valid block count). Keep the mask in the immutable oracle so a
  candidate cannot pass by writing a different tail.
- verify: run the BASELINE against ITSELF twice before trusting any mismatch. Same-input run-to-run
  differences in the tail (observed: thousands of differing entries past the valid prefix, and 0 within it,
  at M=1/2/4/10240) prove the tail is undefined rather than the candidate being wrong.
- caution: also verify any pre-fill/initialization argument the launcher takes - initializing the buffer to
  a sentinel changes the tail but not the defined prefix, so a harness that compares full buffers will show
  "correctness" swinging with an unrelated flag. Prefix-masked comparison is invariant to it.
- caution: also verify the op clears the harness's measurement floor before spending the round the
  fixed oracle unlocks — masking made this op measurable, and it then read ~1.00x because at
  11.7 us/launch it sits under the isolated graph-replay floor
  (method-decode-microkernel-measurement-floor.md). Fixing the oracle is necessary, not sufficient.
- source: exp/e2e_*DeepSeek-R1*/ 2026-08-28 kernel lane (aiter MoE sorting entry, 4.6% GPU; self-vs-self
  diagnostic at four M values)
