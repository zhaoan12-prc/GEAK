# KernelFusion — optional stage before the GEAK e2e workflow

A separate module, **on by default**: every external `interface/run_e2e.py` call runs it to
completion **before** the main e2e workflow (baseline → trace → topk → author → apply back → the
e2e workflow). When it accepts a fused overlay, the main workflow starts from that overlay (its own
baseline is measured on the fused stack). Switched off, nothing changes.

```
trace ──► topk ──► author ──► apply
(GEAK     (fusion  (UT: GEAK  (GEAK e2e_integrator
profiler)  strategy skill;     A/B, compounding
                    kernel:    overlay)
                    agent)
```

| Step | What runs | Reused GEAK code |
|---|---|---|
| Setup | eval dir + baseline throughput (one fresh server, one timed round, no warmup round) | `roles/director.md` PHASE=setup, `bench_e2e.sh` |
| `trace` | AgentX: the kernel-extract skill's workload measurement + sampling capture (points planned from the baseline log, one bounded probe each with Python stacks, `kernel_workload.json` weights); other workloads: one warm serving trace. Skipped when `trace_dir` is given | `knowledge/kernel_extraction_skills/kernel-extract` (stages 1-2), else `roles/profiler.md` |
| `topk` | `kf_trace_digest.py` (deterministic: per-step kernel sequences, plus the live model classes and files read from the eager Python stacks, where every seam must live), then the analyst agent follows the **`fusion-analysis` skill** (`knowledge/kernel_fusion_skills/fusion-analysis/SKILL.md`) plus the run's prompt(s) → `analysis/topk.json` + the report the skill asks for | `parse_profile.py` |
| `author` | a triage agent reads the whole Top-K and picks, by its own judgement, the patterns worth a kernel (no cap, no gain floor; the one hard line is a Python seam); each: region UT, then a fused kernel | AgentX: kernel-extract skill `PHASE=prepare` (one joint-boundary task per region); else `roles/kernel_extractor.md`. The fused kernel itself: one agent per pattern, given the region and the UT, no GEAK kernel layer |
| `apply` | once, after every kernel is written: a select agent reads the UT results and picks which kernels go on (no speedup floor; one kernel per seam / overlapping region); they go in together as ONE combined overlay on ONE A/B vs the reference (gsm8k on the whole stack first, then validation-length legs). Only if the stack fails accuracy or engagement is each kernel checked alone (correctness only) and the A/B re-run without the failing ones; only a measured e2e win is handed to the main workflow | `roles/e2e_integrator.md` PHASE=integrate |

## L3 MegaKernel through GEAK mode=mega (opt-in)

The fusion-analysis skill (`knowledge/kernel_fusion_skills/fusion-analysis`) defines four levels (L1
element-wise, L2 operator-wise, L3 stage-wise, L4 layer-wise) and splits L3 into two forms:
- **stage fusion**: authored by an agent like L1/L2;
- **MegaKernel**: by default an unsupported pattern (`unsupported_patterns`).

With `l3_mega` on, the **expert-parallel MoE MegaKernel** (the MegaMoE pattern) is implementable: it
leaves the unsupported patterns, and GEAK mode=mega builds it.

**Region.** The whole MoE layer as the live path runs it: routing (gate, top-k, sorting), quant,
dispatch, expert GEMM1 + activation + GEMM2, combine, and the shared expert(s) when computed there.

**Recognition.** From the stage's own kernels, which must include expert GEMMs plus either:
- EP library kernels (megamoe, DeepEP, mori, all-to-all); or
- an EP deployment (`ep` > 1). This covers a from-scratch path whose dispatch/combine carry no
  library name.

An L3 stage fusion without expert GEMMs (router + top-k + dispatch preparation) stays with the author
agents. This is the pattern mode=mega has been validated on:
- DeepSeek-V4-Pro EP8 on MI355X;
- Qwen3-235B EP8 on MI300X, over both an AITER/Mori path and a Triton-from-scratch path.

Any EP size works: mode=mega runs on the stage's own rank count (`gpus_per_job` = the EP group, e.g.
EP4 on 4 serving GPUs). The measured results above are EP8. The MegaMoE expert skill
(`use_expert_skills`) is written for EP8 / gfx950, so leave it off for other EP sizes. A finished run
is reused only for the same rank count.

Other multi-rank stages (a TP collective with its compute) are wired but stay off until mode=mega is
measured on them. `l3_stage_kinds: "ep_moe,collective"` turns them on.

| step | L1 / L2 / L3 stage fusion | L3 EP MoE MegaKernel |
|---|---|---|
| `topk` | ranked as before | ranked with `level: "L3"`, `implementable_by: "mega"`, `exclusive_with` the L1/L2 candidates inside it; it goes to mode=mega only if its kernels are an EP MoE stage |
| triage | agent's pick | same agent; at most `l3_max` stages |
| `author` | region UT + one author agent, in parallel on single GPUs | after those: `l3_mega.md` PHASE=prepare (operator, frozen source tree, multi-rank UT) → `kernel_workflow/kernel_mega.js` on the serving GPUs (`gpus_per_job` = the stage's rank group) → PHASE=package (overlay `[kf-<id>] ENGAGED`, checked with the UT) |
| select / gsm8k / A/B | one stack, one A/B; accuracy failure → per-kernel screen | the L3 kernel is one more kernel of that stack; if the stack holding it does not win, the L1/L2 kernels (with the ones only it had excluded) get one A/B without it |

Mode=mega args are layered, so a run needs nothing beyond `l3_mega: true`:

| layer | keys | set by |
|---|---|---|
| search policy | profile, pairs, replays, timeouts, accuracy metric/threshold, `launch_target` / `analysis_skill` per stage kind | `l3_mega_defaults.json` (`common`, then `ep_moe` or `collective`) |
| this stage | cases, guards, accuracy cases, positive control, task, import modules | prepare (from the trace and the serving config), or `l3_operator` |
| where / how long | `workflow_dir`, `kernel_path`, `benchmark_harness`, GPUs, `exp_root`, `state_dir`, `budget`, `mega_time_budget_s` | the stage (`l3_operator`, `l3_rounds`, `l3_budget_s`); `l3_mega_args` cannot change them |
| caller | any policy or stage key | `l3_mega_args`, applied last |

Timing. The stage's usual path is ~7 h (baseline 1 h, trace 0.8 h, topk 0.5 h, kernels + UTs 1.5 h,
gsm8k 0.2 h, A/B 2 h). A finished mode=mega run (`l3_mega_run`) only adds packaging. A fresh search
adds mode=mega's own budget (`l3_budget_s`, 8 h); with it the stage budget defaults to 12 h + that,
capped at half the run. `result.json["kernel_fusion"]["l3"]` records each stage: `applied`,
`ab_rejected`, `not_selected`, `mega_no_deliverable`, `mega_error`, `prepare_blocked`, `package_blocked`.

## Switch

Precedence: CLI > env > handoff.

```
run_e2e.py handoff.json result.json                        # on (default)
run_e2e.py handoff.json result.json --no-kernel-fusion     # off for this run
GEAK_KERNEL_FUSION=0|1                                     # env
"kernel_fusion": false                                     # handoff, or an object that configures it:
"kernel_fusion": {"prompt": "/path/deploy.txt", "trace_dir": "...", ...}
```

| key (handoff) | env | default | meaning |
|---|---|---|---|
| `strategy` | `GEAK_KF_STRATEGY` | `knowledge/kernel_fusion_skills/fusion-analysis/SKILL.md` | the analysis method (fusion levels L1–L4, requirements incl. the boundary ledger, procedure, estimation, report contents); another file replaces it, `false` drops it. Output formats are defined by `kf_workflow.js`, and the implementable scope is injected (`kf_implementable_levels`, default `L1,L2,L3`; `kf_unsupported_patterns`, default L3 MegaKernel and L4) |
| `prompt` | `GEAK_KF_PROMPT` | — | the run's own prompt(s), comma-separated, read after the strategy (e.g. the deployment / workload description `muse_agentx_con16.txt`) |
| `trace_dir` | `GEAK_KF_TRACE_DIR` | — | existing traces (any depth; one dir per sampling point); skips the capture |
| `workload_summary` | `GEAK_KF_WORKLOAD_SUMMARY` | — | per-point weights for the analysis |
| `topk_json` | `GEAK_KF_TOPK_JSON` | — | a finished `topk.json` (REPORT.md / MODEL_NOTES.md beside it are carried over); the analysis is not re-run |
| `reuse_journal` | `GEAK_KF_REUSE_JOURNAL` | — | an earlier run's Workflow `journal.jsonl`: its finished UT-prepare and author agents are reused, not re-run |
| `baseline_dir` | `GEAK_KF_BASELINE_DIR` | — | a measured baseline dir of this same stack (`bench_summary.json`, `server.log`); copied, not re-measured |
| `source_dirs` | `GEAK_KF_SOURCE_DIRS` | installed packages | framework/op-library source for the analyst when it is not importable where the stage runs |
| `steps` | `GEAK_KF_STEPS` | `trace,topk,author,apply` | subset to run; `topk` alone with `trace_dir` needs no GPU |
| `author_k` | `GEAK_KF_AUTHOR_K` | 0 (no cap) | optional cap on the patterns the triage agent sends to authoring |
| `baseline_mode` | `GEAK_KF_BASELINE_MODE` | `isolated_server` | lifecycle of the stage's baseline: one fresh server, one timed round, no discarded warmup round |
| `implementable_levels` | `GEAK_KF_IMPLEMENTABLE_LEVELS` | `L1,L2,L3` | fusion levels the analysis may recommend and the triage may author (all levels are still ranked) |
| `unsupported_patterns` | `GEAK_KF_UNSUPPORTED_PATTERNS` | L3 MegaKernel (e.g. MegaMoE), L4 MonoKernel | patterns inside those levels that are ranked but not recommended or authored |
| `accuracy_confirm_limit` | `GEAK_KF_ACCURACY_CONFIRM_LIMIT` | 1319 (full gsm8k test set) | when the fused stack misses the gsm8k gate on the 200-question sample, both sides are re-scored on this many questions and that result decides; the per-kernel screen (only if it still fails) uses it too |
| `min_gain_pct` | `GEAK_KF_MIN_GAIN_PCT` | 0.5 | the combined A/B hands the overlay on only when its gain exceeds max(measured noise band, this) |
| `validation_mode` / `validation_replicas` | `GEAK_KF_VALIDATION_MODE` / `_REPLICAS` | `isolated_server` / 1 | measurement mode / replicas of the combined A/B (one A/B) |
| `task` | `GEAK_KF_TASK` | — | operator constraints appended to every agent prompt (e.g. directories it must not read) |
| `budget_s` | `GEAK_KF_BUDGET_S` | min(12 h, ⅓ of the run) | stage wall clock; the main workflow gets the rest |
| `kernel_language` | `GEAK_KF_KERNEL_LANGUAGE` | — | language every fused kernel is written in (e.g. `flydsl`), or `auto` for the GPU's default (gfx950: FlyDSL). Unset: L1/L2/L3 stage fusion keep the analyst's choice (Triton by default), and an L3 mode=mega search follows the GPU's default, else its source's language |
| `l3_mega` | `GEAK_KF_L3_MEGA` | false | L3 stage fusion through GEAK mode=mega (above) |
| `l3_mega_run` | `GEAK_KF_L3_MEGA_RUN` | — | a finished mode=mega run of the stage (state_dir / eval dir / mega_result.json): package it, no search |
| `l3_max` | `GEAK_KF_L3_MAX` | 1 | L3 stages handed to mode=mega |
| `l3_budget_s` / `l3_rounds` | `GEAK_KF_L3_BUDGET_S` / `_ROUNDS` | 28800 / 12 | mode=mega time budget and candidate rounds of a search |
| `l3_mega_args` | `GEAK_KF_L3_MEGA_ARGS` (JSON) | — | merged last into the mode=mega args, e.g. `{"use_expert_skills": "true"}`; paths / GPUs / budget keys are ignored |
| `l3_defaults` | `GEAK_KF_L3_DEFAULTS` | `l3_mega_defaults.json` | search-policy defaults file; `false` leaves them to mode=mega |
| `l3_operator` | `GEAK_KF_L3_OPERATOR` (JSON) | — | what is known about the stage's operator (`kernel_path`, `benchmark_harness`, `call_site`, `package`, `ep_size`, `mega_args`): prepare verifies it instead of discovering it |
| `l3_kernel_regex` | `GEAK_KF_L3_KERNEL_REGEX` | — | extra EP MoE kernel names to recognize (another EP library) |
| `l3_stage_kinds` | `GEAK_KF_L3_STAGE_KINDS` | `ep_moe` | stage kinds handed to mode=mega; `ep_moe,collective` adds TP collective stages (not yet measured) |
| `kf_dir` | `GEAK_KF_DIR` | `<exp_root>/kf_<main eval dir suffix>` | stage output dir |

Synthetic speculative acceptance (`SGLANG_SIMULATE_ACC_*`, `TILERT_SIMULATE_ACC_*`,
`TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`, as InferenceX AgentX throughput runs set) stays on for
throughput legs; every parity / gsm8k check of the apply step runs on servers without it.

## Outputs

`<kf_dir>/analysis/` (digest, `topk.json`, the prompt's report), `kernels/` (task dirs + kernel-layer
runs), `overlay/` (integrator candidates), `kf_return.json`. `result.json` gains a `kernel_fusion`
section with the stage A/B (`throughput_speedup`), and `combined_speedup_vs_original` = stage ×
main workflow speedup. A finished `kf_return.json` is reused when the same eval dir is resumed.

## Running pieces on their own

```
python3 kf_trace_digest.py --trace-dir <traces> --out /tmp/digest           # digest only
node interface/runtime/engine/run_workflow.mjs e2e_workflow/kernel_fusion/kf_workflow.js --agent claude \
  --args '{"workflow_dir":"<repo>/e2e_workflow","kf_dir":"/tmp/kf","kf_steps":"topk",
           "kf_trace_dir":"<traces>","kf_prompt":"<prompt>","model_path":"<model>"}'   # analysis only
python3 -m pytest e2e_workflow/kernel_fusion/tests                          # tests (node optional)
```
