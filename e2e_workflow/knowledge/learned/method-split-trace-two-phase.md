---
key: trace capture completeness · any gfx · any backend
type: method
confidence: ★★
effect: prevents a whole phase (usually decode) from being silently absent from the analysis
confirms: 1
last_seen: 2026-08-26
---
# A prefill/decode split capture means TWO traces must reach the table
- lever: when the profiler captures prefill and decode as separate traces (EXTEND + DECODE), the
  semantic table must ingest BOTH. Ingesting one leaves the other phase with no rows at all — and
  every downstream phase then reports zero candidates there, which reads as "no opportunity" rather
  than "never looked". On DSR1 that hid the single highest-payoff decode fusion.
- apply: keep `--table-phases all`; `scripts/semantic_report.py` names any expected phase absent from
  the tables in red, and lists every trace path + sha256 so two distinct captures are visibly two.
- verify: check per-phase SHAPE resolution too, not just row counts. Under CUDA-graph replay decode
  emits almost no `nn.Module` spans, so a decode table can have rows and `0/N` shapes resolved — which
  is equally fatal (no shape, no candidate) and needs an eager probe to fix.
- caution: the table's declared `phase_coverage` record can go stale when a later step grafts shapes
  onto the rows; measure coverage off the rows and treat a mismatch as a warning, not a correction.
- source: /raid/users/zhaoan/fusion_kernel_result/20260826_e2e/dsr1/round1 (p1_trace/, semantics_production_shaped/)
