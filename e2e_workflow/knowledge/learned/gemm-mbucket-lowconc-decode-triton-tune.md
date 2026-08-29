---
name: gemm-mbucket-lowconc-decode-triton-tune
description: At conc<=4 decode the aiter M-bucket ladder has no M_LEQ_4 key, so decode runs an M_LEQ_8/16 config; adding the bucket gave +4.24% e2e attributable, bit-identical, and shipped inside a Director-validated 1.117x stack.
keywords: [gemm, m-bucket, decode, low-concurrency, config-tune, per-shape-tune, aiter, triton, fp8-blockscale, batched-gemm, mla, split-k, engagement-verification, final-bundle-recheck, negative-control]
kernels: [gemm_a8w8_blockscale_preshuffle, _gemm_a16w8_blockscale_kernel, _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant, _gemm_a8w8_blockscale_reduce_kernel, get_gemm_config]
platforms: [gfx950, gfx942]
kernel_class: dense_gemm
regime: decode
key: aiter Triton per-shape GEMM config tables (A8W8 block-scale preshuffled dense + batched MLA-absorb) · gfx950/MI355X · low-concurrency decode where live M <= 4
type: lever
confidence: ★★★
confirms: 4
effect: iso 1.30-1.51x per kernel on 4 shapes totalling ~13.7% GPU; +4.24% ATTRIBUTABLE e2e on its own interleaved same-session A/B (301.5 -> 314.2 tok/s), arms DISJOINT on tput/ITL/TPOT, median-ITL -4.385% (= +4.586% decode rate); reproduced on 3 nodes whose ABSOLUTE throughput differed by 17% yet whose DELTAS agreed within 0.5 pp; output bitwise identical on all 4 shapes. E2E GATE NOW CLOSED: shipped in the final bundle and Director-validated same-session (3 cold-start replicas/arm, non-overlapping, 23x the 0.5% band) at 332.2 -> 371.1 tok/s = 1.1169x (+11.69%), TPOT -10.69%, output parity pass -- the tables are ~38% of that composed gain, the backend-select env that made them live is the rest (see atom-dense-gemm-seam-select-decode-gfx950.md)
lifecycle: active
last_seen: 2026-08-29
---
# Decode can fall off the BOTTOM of a per-shape tuning table's M ladder
- lever: a per-shape config table is keyed by M-buckets (`M_LEQ_4/8/16/...`) and the lookup returns the
  first bucket with `M <= x` (aiter: `get_gemm_config` over `STANDARD_M_BOUNDS=(4,8,16,...)` against
  `configs/gemm/<gfx>-{GEMM,BATCHED_GEMM}-<variant>-N=<n>-K=<k>.json`). Shipped tables often start at
  `M_LEQ_8`, so serving at concurrency 4 (and
  any TP where per-rank M stays tiny) silently executes a config tuned for a batch 2-4x larger. Check the
  live M against the LOWEST key present for that (N,K) before concluding "this shape is already tuned".
- apply: pick targets from a trace of the ACCEPTED stack (a Triton symbol encodes its constexprs, so the
  live name tells you which bucket was used); coordinate-descent the new low-M bucket (`NUM_KSPLIT`,
  `BLOCK_SIZE_M/N`, `num_stages`, `cache_modifier=.cg`, `waves_per_eu` were the movers); ship as a JSON/CSV
  edit. Zero extra HBM. The lookup is `lru_cache`d per worker and framework compile caches do NOT hash the
  table, so INSTALL BEFORE THE SERVER STARTS and clear the framework compile cache. This is a DATA-only
  win: it rides the final bundle's patch/deploy step, NOT a `PYTHONPATH` overlay. Adopting `NUM_KSPLIT>1`
  adds a `..._reduce_kernel_..._ACTUAL_KSPLIT_k` companion launch, so time (and quote) the WHOLE call.
- verify: prove the bucket resolves live (`get_gemm_config(..., M=<live>, N, K)` returns the new key on the
  installed tree) AND that the production entry point reproduces the tuned per-op time with `config=None`;
  a table the runtime never reads fails silently. Confirm the new Triton symbols appear in the JIT cache
  only after the post leg. Make the profiler check TWO-SIDED and PER RANK: the shipped symbol must go to
  **0 on N/N ranks** and the tuned symbol appear the expected count on **N/N** — at TP=N a partially engaged
  rank reads as no gain at all, because the fast ranks wait on the slow one.
  Re-verify on the SHIPPED bundle too, as a two-directional negative control: revert the tables for the
  reference arm (assert the low-M key is absent) and re-install them for the candidate arm (assert the
  files' sha256 match the accepted arm byte for byte) — that is what turns a data-only edit into a
  defensible final-gate result rather than an assertion.
- caution: also verify WHICH backend the seam actually dispatches before tuning any table - a table binds
  only to its own backend, so a CK/CSV artifact is dead weight when a `USE_TRITON_GEMM`-style switch has
  put the Triton path live (and some builds drop `kernelName` for block-scale CK, making a CK table
  unselectable at all). Retarget the tune to the live backend rather than the profiled one. Also verify (a) that any
  split-K you adopt partitions K on the QUANT BLOCK-SCALE boundary - that is what kept this bitwise
  identical; a split crossing a scale block changes accumulation order and then you owe an accuracy gate;
  (b) RE-TIME every first-pass winner interleaved against the shipped config - 1 of 5 shapes here showed
  1.14x on first pass and 0.4% (inside spread) on a paired re-race, and would have shipped a null row;
  (c) on an aiter/ATOM-class stack the biggest profile heads are often PREBUILT ASM / C++-extension kernels
  with no editable source seam (fused MoE, MoE sorting, MLA decode) - probe seam editability before
  budgeting them; the reachable lever there is the config/tuning table the kernel itself reads, not a rewrite.
- source: exp/e2e_*DeepSeek-R1*_atom_*/ 2026-08-28 (fp8 block-scale, TP8, ISL8k/OSL1k/conc4, 3 nodes)
  + 2026-08-29 Director same-session validation (validated_win, 3 replicas/arm, parity pass);
  see also gemm_tuning/ and knowledge/tuning_skillset_integration.md
