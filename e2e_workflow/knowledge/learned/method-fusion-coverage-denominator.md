---
key: kernel fusion coverage · any gfx · any backend
type: method
confidence: ★★★
effect: turns "did we find every fusion?" from a judgement call into a number each phase is scored on
confirms: 1
last_seen: 2026-08-26
---
# Every fusion phase needs a denominator, or the misses are invisible
- lever: fusion phases fail by OMISSION, and omission looks exactly like success — a report reading
  "16 verdicts: pass 16, status=pass" was 16 of 42 candidates. Give each phase a denominator computed
  from the previous phase's artifact, and make the gap red: fusible regions (Phase 1) -> a candidate per
  region (2.1) -> the execution list (2.2) -> a 单侧 verdict per row (3.0) -> a disposition per row (3.1).
- apply: the four `scripts/fusion_*_harness.py` gates enforce this; `scripts/report_index.py` surfaces
  each denominator on one line. A **fusible region** = a maximal contiguous same-stream run of non-donor
  rows between two donors (GEMM / attention / MoE / collective), since a fused kernel cannot cross a
  donor body. That makes the opportunity set a pure function of the table rather than of judgement —
  which is what makes two runs of the same table produce the same candidate set.
- verify: read the coverage line before the results line. "Not mentioned" is never "not applicable":
  an unbenched candidate is `not_validated`, not a pass; an unmentioned execution row is `unaccounted`,
  not a skip. What silently gets dropped is biased — the hardest candidates to microbench (paged-KV
  state, MoE routing state) have no correlation with payoff, so an unforced loop selects for easy.
- caution: also check WHY a row was dropped before accepting it. A mutual-exclusion group is a pairwise
  conflict graph, not an equivalence class: collapsing "A conflicts B, B conflicts C" into "pick one of
  three" forbids a legal combination. And declining to choose inside a group is an open decision, not
  an absent opportunity.
- source: /raid/users/zhaoan/fusion_kernel_result/20260826_e2e/dsr1/round1; GEAK e2e_workflow scripts/fusion_*_harness.py + tests
