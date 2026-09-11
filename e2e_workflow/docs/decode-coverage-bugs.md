# Decode coverage: eight defects that made the semantics pipeline prefill-only

Found 2026-08-26 while attributing a DeepSeek-R1 fusion A/B whose TTFT improved
(+0.55%, replicated) but whose TPOT did not move at all. Root-caused to the
Phase-1 semantics stage never having observed a single decode forward.

Symptom chain, as it actually happened:

```
SHAPE_CAPTURE_PLAN.json  trace_path = ...-TP-0-EXTEND.trace.json.gz    [B2]
                         (the -TP-0-DECODE sibling sat unread beside it)
  -> target_buckets = 2 buckets, both phase:'prefill'
  -> pattern_layer_kernel_table.json says table_phases:["all"]          [B1]  <-- lie
  -> _required_phases(plan) = ['prefill']                               [B4]
  -> GEAK_SEMANTICS_PHASES=prefill
  -> eager-decode probe never auto-enabled                              [B4]
  -> semantic_runtime_capture.py:223 drops every decode forward
  -> shape.jsonl = 150 records, all phase='extend', zero decode
  -> every fusion candidate derived from prefill shapes only
  -> the winning candidate targets `q_a_layernorm`, a seam that does not
     exist on the decode path (decode runs aiter::fused_qk_rmsnorm)
  -> measured: rung-4 fused kernel fires 61x in EXTEND, 0x in DECODE
  -> TPOT gain: zero, exactly as the missing coverage predicts
```

## B1 - `table_phases: ["all"]` is a lie (severity: high, silent)

`semantic_kernel_mapping.py:1759`

```python
"table_phases": sorted(table_phases) if table_phases else ["all"],
```

`table_phases=None` means "apply no phase filter", i.e. *all phases present in
this trace*. It does not mean "all phases of the model". With a single-phase
EXTEND trace as input, the emitted table claims `["all"]` while containing the
string `decode` exactly zero times. Every downstream consumer, human or
machine, reads `["all"]` as full coverage.

**Fix:** report the phases actually observed, and keep the request separate.

## B2 - `build()` accepts one trace and never mentions the other (severity: high)

`semantic_kernel_mapping.py:1712`

SGLang's `profile_by_stage` writes `<stem>-TP-<rank>-EXTEND.trace.json.gz` and
`<stem>-TP-<rank>-DECODE.trace.json.gz` as separate files. `build()` takes a
single `trace_path`, so the mapper can only ever see one phase per invocation,
and nothing checks whether the sibling exists. The two files are timestamp
disjoint and correctly ordered (EXTEND `[...061043074, ...064726031]`, DECODE
`[...265735895, ...267481518]`), so concatenating their event lists is safe.

**Fix:** accept multiple traces, auto-discover the phase sibling, and refuse to
silently emit a single-phase plan.

## B3 - `decode_capture_windows` is dead metadata (severity: high, actively misleading)

`semantic_kernel_mapping.py:1673`

```python
"decode_capture_windows": ["graph_capture", "warmup", "enforce_eager_probe"],
```

This string literal is the only occurrence of `decode_capture_windows` in the
entire repository. Nothing reads it. It is written into every
`SHAPE_CAPTURE_PLAN.json`, where it reads as a commitment that decode is
captured through three named windows. None of the three is wired to anything.

**Fix:** derive the field from what the run will actually do, or drop it.

## B4 - phase narrowing is silent, and the eager probe never auto-arms (severity: high)

`run_semantic_shape_capture.py:110,131`

```python
def _required_phases(plan):
    return sorted({str(b.get("phase")).lower() for b in plan.get("target_buckets", []) if b.get("phase")})
...
if not phases:
    phases = _required_phases(plan)
```

Deriving capture phases from the plan's buckets is correct in isolation, but it
turns B2's single-phase plan into a single-phase *capture* with no warning.

Worse: `_with_disable_cuda_graph()` at line 99 already implements the
eager-decode probe, and the generated script is even named
`benchmark_eager_decode.sh` - but it only arms when the caller passes
`--disable-cuda-graph`.

What the probe does and does not buy, measured on the real traces:

| | EXTEND | DECODE (replay) |
|---|---|---|
| `nn.Module:` spans | 981 | **1** |
| kernel events | 2027 | 649 |
| P0 table rows (ordered sequence) | 29 | **29** |
| rows with resolved shapes | 22/29 | **0/29** |

Sequence coverage and shape coverage fail **independently**, and conflating them
is what kept this invisible. A replay-mode DECODE trace still carries every
kernel event, so the ordered per-layer sequence reconstructs fine - that alone
would have shown that decode runs `aiter::fused_qk_rmsnorm` where prefill runs
`q_a_layernorm`, i.e. that the chosen fusion seam does not exist in decode. What
it cannot give is dims/dtypes: with no module or cpu_op spans there is nothing to
attach `Input Dims` to, so all 29 decode rows come back `unresolved`. Candidate
*discovery* needs the sequence; candidate *generation and benchmarking* need the
shapes.

**Fix:** requesting decode implies the eager probe unless explicitly overridden;
a narrowed phase set is reported loudly; and coverage is reported as two separate
booleans (`decode_sequence_covered`, `decode_shapes_covered`) so a partial win is
never rounded up to a full one.

## B5 - rank-0 trace selection never matches under `profile_by_stage` (severity: medium)

`run_semantic_shape_capture.py:92`

```python
rank_zero = [path for path in candidates if "-TP-0.trace.json" in path]
```

With `profile_by_stage` on, filenames are `-TP-0-EXTEND.trace.json.gz`. The
substring `-TP-0.trace.json` never matches, `rank_zero` is empty, and selection
falls through to `max(candidates, key=os.path.getmtime)` - newest file across
**all 8 ranks and both phases**. Which rank and phase you get is a race.

**Fix:** match the rank prefix, keep phase explicit.

## B6 - `pkill -f` pattern-kill inside a shared container (severity: low)

`run_semantic_shape_capture.py:36-43` (before the fix)

`_stop_service` pattern-killed on `[s]glang.launch_server.*--port(=| )<port>`,
both before the run (to clear a stale service) and in the `finally` block.
It is port-scoped, which bounds the blast radius, but it is still a
pattern-kill executed inside a container that may be shared, and it does not
protect PID 1 -- which, in the e2e harness, *is* the orchestration process.
Two concrete hazards:

* another session serving the same port number in the same container is
  indistinguishable from ours and gets killed;
* the pre-run call kills before we have started anything, i.e. it kills a
  process this run provably does not own.

### Fix

Own the process group, then kill only that group.

1. The benchmark is launched under `setsid`, and the new session leader
   writes its own PID -- which is the PGID -- to `<out_dir>/server.pgid`:

   ```sh
   setsid bash -c 'echo $$ > "$GEAK_SERVER_PGID_FILE"; exec bash "$GEAK_BENCHMARK_SCRIPT"' &
   geak_wrapper=$!
   wait "$geak_wrapper"
   ```

   `$$` inside the `setsid`-created session is the group leader regardless of
   whether `setsid` forked, so this is robust where `$!` alone is not.

2. `_stop_service(container, pgid_path, port)` reads that file and issues
   `kill -TERM -"$pgid"` / `kill -KILL -"$pgid"` -- the leading `-` is the
   whole point. An absent or empty file means the run never started anything
   and teardown is a no-op. Liveness is polled on the **port**, not on
   `kill -0` of the group: the server leaves unreaped children behind and
   container PID 1 does not reap them, so a group-liveness poll never clears.

3. The pre-run call becomes `_assert_port_free(container, port)`, which
   probes `/dev/tcp/127.0.0.1/<port>` and *raises* if the port is busy
   instead of killing. A stale service is now a reported error the operator
   resolves from the session that owns it.

`pkill`/`pgrep` no longer appear anywhere in the script.

### Tests

`tests/test_run_semantic_shape_capture.py::OwnProcessGroupTeardownTest`

* teardown group-kills the recorded PGID and contains no `pkill`/`pgrep`;
* teardown is a no-op when no PGID was recorded;
* a busy port raises rather than kills;
* a free port passes.

## B7 — module-less windows are carved into the configured layer count

**Symptom.** With the B1–B6 fixes in place, a real two-phase run still produced
a DECODE `P1_MLA_MOE_SHARED` representative of **2 kernels / 11.2 µs**. A MoE
layer cannot be two kernels. Every quality gate reported `pass`.

**Mechanism.** `_stage_sequence_partition` ends in

```python
for layer_id in range(layer_count):   # layer_count = 61, from config
```

so it always emits exactly `layer_count` segments, whatever the window holds.
SGLang's DECODE profiling window is truncated — it stops mid-forward — so the
window physically contained **21** layer bodies (a clean 35-kernel period,
verified by the recurrence of the attention block). Forcing 61 cuts onto 21
bodies produced 40 degenerate one- and two-kernel "layers". The step was
partitioned by `forced_best_alignment` with `mean_pattern_similarity = 0.274`.

**Why no gate caught it.** `representative_layer_integrity` checks only internal
self-consistency: expected count == actual count, rows ordered, interval
complete, durations sum. A mis-cut window satisfies all of them — the 2-kernel
table was internally perfect. Nothing checked whether the representative could
*be* the layer it claimed to be.

**Fix, two parts.**

1. `_anchor_runs` finds the stage that starts every layer body by recurrence,
   and maps the bodies that are physically present instead of assuming the
   configured count. Candidates are scored on whether the segments they produce
   look like the declared layers — `segment_validity` (does each body contain the
   stages its pattern requires), then `window_coverage` (a router stage fires
   only in MoE layers, so it segments those perfectly and leaves the dense ones
   unassigned), then the dense/MoE class agreement against the configured chain.
   That last term also recovers the layer-id offset, so a window starting
   mid-model is labelled correctly. Bodies at the two ends are trimmed while
   incomplete: a truncated window has a partial body at each end by
   construction. The path is entered only when module spans are absent and the
   observed body count differs from the configured one, so module-backed
   prefill segmentation is untouched.

2. A new gating check, `representative_pattern_plausibility`, requires each
   representative to contain the stages its declared pattern implies —
   `attn` + `gemm`, plus `moe` + `topk` for a routed-expert FFN. The requirement
   is intersected with the stages the trace actually demonstrates, so a capture
   genuinely without expert kernels is not failed for lacking them; only a
   representative that omits stages the same trace proves are there is.

**Caveat recorded in the output.** No stage in the decode window sits at the true
module entry (the input layernorm fires twice per layer, so it cannot anchor).
Bodies are therefore cut at the anchor stage and each is a *cyclic rotation* of
the true layer: the kernel set and the intra-layer order are faithful, but the
first row is not necessarily the layer's first kernel. The partition diagnostics
carry `boundary_reference: "anchor_stage_rotation"` to say so.

**Measured effect** (real MI308X DSR1 run, TP-0, EXTEND+DECODE):

| bucket | before | after |
|---|---|---|
| `P0_MLA_DENSE_MLP` / prefill | 29 rows | 29 rows |
| `P1_MLA_MOE_SHARED` / prefill | 33 rows | 33 rows |
| `P0_MLA_DENSE_MLP` / decode | 29 rows | 29 rows |
| `P1_MLA_MOE_SHARED` / decode | **2 rows** | **35 rows** |

The decode MoE chain — `grouped_topk` → `fused_append_shared_experts` →
`moe_sorting` → `quant` → `moe_gemm` → `act_and_mul` → `quant` → `moe_gemm` —
was absent from the candidate pool entirely before this fix.

## B8 — decode shapes attach to nothing, and the merge still reports `pass`

**Symptom.** With B1–B7 fixed and an eager shape probe (B4) that provably
captured decode — `shape.jsonl` held 144 decode records against 60 prefill
ones — `semantic_shape_merge.py` produced a shaped table in which **every
decode row was at evidence level `U`**: 64 decode rows, 0 resolved. Prefill
resolved 45 of 62. `verification.status` was `pass`.

**Mechanism.** The merge attaches a shape record to a kernel row by the row's
`parent_operator`, which `_candidate_groups` uses to pick the module the
kernel was launched from. A table built from a production **CUDA-graph**
DECODE trace has `parent_operator: "unresolved"` on every row — the CPU
replays a captured graph and never walks the module tree, so the trace has no
`nn.Module` spans to resolve against. `_candidate_groups` returns `[]` for all
of them, and each row falls through to `U`.

The shape log was fine. The table was fine *as a timing table*. The two simply
could not be joined, and nothing said so.

**Why no gate caught it.** The existing checks are identity checks: the
representative-table checks compare the merged table against the source table
and confirm nothing was altered. Attaching zero shapes alters nothing, so they
pass. Evidence-level distribution was reported but not gated — `U` is a legal
level (a kernel whose operands genuinely cannot be determined), so a table
made entirely of `U` looked like a conservative result rather than a failed
join.

**Fix, two parts.**

1. *At the source.* Build the decode table from the **eager** capture trace
   (`--disable-cuda-graph`), which does carry module spans, and keep the
   production graph trace for structure and timing. Decode resolution went
   from 0 % to 82–87 %.

2. *As a gate.* `_phase_resolution(audits, groups)` is now part of
   `verification` and contributes to `verification.status`. The rule: if the
   shape log carries rank-0 records for a phase, and the table has rows for
   that phase, then **at least one** of those rows must resolve. Zero is a
   join failure, not a conservative answer. The failing phase carries a note
   naming the likely cause and the remedy — rebuild from the eager trace.

   Coverage is read from rank-0 groups only, matching the rank the probe
   actually instrumented; a phase the probe never covered is reported with
   `shape_log_covers_phase: false` and is not gated, since that is a capture
   coverage gap the capture step already reports.

**Measured effect** (same run, TP-0):

| table built from | decode rows | resolved | `verification.status` |
|---|---|---|---|
| CUDA-graph trace | 64 | 0 | `fail` (was `pass`) |
| eager trace | 72 | 49 | `pass` |

Prefill is unchanged at 45/62 in both.

### Tests

`tests/test_semantic_shape_merge.py::PhaseResolutionGateTest`

* a covered phase that resolves nothing fails the merge and names the cause;
* partial resolution passes;
* a phase absent from the shape log is reported but not gated;
* only rank-0 shape records define which phases count as covered.
