# L3 stage fusion through GEAK mode=mega

KernelFusion levels (`knowledge/kernel_fusion_skills/fusion-analysis/references/fusion-levels.md`):
L1 element-wise, L2 operator-wise, L3 stage-wise (stage fusion or MegaKernel), L4 layer-wise. You handle
the L3 MegaKernel form. An author agent writes L1/L2 kernels against a single-card region UT. A multi-rank
stage runs over several GPUs at once, so no single-card UT can hold it. It is the kernels of one
functional stage that include cross-GPU communication, for example:
- an expert-parallel MoE: dispatch → expert GEMMs → combine over the EP group;
- a collective with the compute around it: GEMM + all-reduce / reduce-scatter, all-reduce + norm +
  quant, all-gather + GEMM.

By default only the expert-parallel MoE kind runs (`kf_l3_stage_kinds`, default `ep_moe`); the
collective kind is opt-in. With `kf_l3_mega` on, such an L3 candidate goes to GEAK mode=mega
(`GEAK_KERNEL_WORKFLOW_DIR/kernel_mega.js`; see the "Mega mode" section of
`GEAK_KERNEL_WORKFLOW_DIR/README.md`), which searches a fused multi-rank kernel for it.

The stage calls you after every author agent has finished, so the serving GPUs are free: no
server of this stage is running. Never start a serving server. Run GPU work through a GEAK group
lease, which checks the GPUs are idle and reaps its own job on timeout:
```bash
GEAK_GPU_RUN_TIMEOUT=1800 bash "$GEAK_KERNEL_WORKFLOW_DIR/scripts/gpu_lock.sh" group:$SERVING_GPU \
  torchrun --standalone --nproc_per_node=$RANKS <ut> correctness --replays 16
```
Each lease gets its own `MASTER_PORT`; never hard-code `--master_port`. Write only under `L3_DIR`.

`L3_STAGE` is the Top-K candidate: `kernels` (in order), `phases`, `seam.target_callable`,
`m_range`, `occurrences`, `exclusive_with`.

## PHASE=prepare

`OPERATOR_HINT`, when given, is what the caller already knows (any of `kernel_path`,
`benchmark_harness`, `call_site`, `package`, `ep_size`, `mega_args`). Use it and only verify it:
the call site must be on the served path, and the UT must meet the contract and pass on the frozen
tree. Discover only what it leaves out.

1. **Find the operator.** Start from `seam.target_callable` and the installed serving code, and
   find the call that launches this stage on the served path under `SERVER_FLAGS` / `SERVER_ENV`
   (follow the branch those flags select). Record:
   - the call site (`module:function`);
   - the package that owns the kernels and its source tree (the editable checkout or the
     installed package directory);
   - the rank count: the group the stage's communication runs over (the EP group of an MoE, the
     TP group of an all-reduce), from the serving flags (default `EP_DEFAULT`). It is mega's
     `gpus_per_job`; return it as `ep_size`.

   If the package is installed without sources, return `blocked`.
2. **Freeze the baseline tree.** Copy the owning source tree to `L3_DIR/baseline_tree`, or
   `git worktree add` it at its current commit, and note the commit. This is mega's
   `kernel_path`; nothing edits it after this step.
3. **Write the UT** inside the frozen tree; mega copies the tree, and the UT goes with it. Run one
   process per rank under `torchrun`, with modes `correctness` and `performance` and a repeatable
   `--case` flag.
   - **Operator:** build it exactly as the served model does (shapes, dtype, quantization, how
     weights and tokens are split over the ranks, the communication backend, the call from
     step 1). The UT runs the whole stage, communication included, from its input on each rank
     to its output on each rank. For an EP MoE stage that is everything in `L3_STAGE.kernels`:
     - routing: gate GEMM, top-k, sorting / routing metadata;
     - the activation quant before dispatch;
     - dispatch, expert GEMM1 + activation + GEMM2 (with their quant), combine and the weighted
       reduction;
     - the shared expert(s), when the live path runs them in this stage (fused into the routed
       experts, or alongside them).

     The input is the hidden states (plus the router weights when routing is inside the stage),
     and the fp32 reference computes the same math. Leave a part out only when the live path
     runs it outside the stage's call site, and say so. Import the package from the tree under test (working
     directory / `PYTHONPATH`), so that a candidate tree is what runs. Weights may be random,
     seeded the same on every rank.
   - **Path marker:** print `[op] path=<NAME> rank=<r>` on every rank.
   - **Correctness:** compute relL2 against an fp32 reference of the same math. Capture a
     CUDA graph and check that N replays equal the eager output. Print
     `[RESULT] case=<id> relL2=<x> replays=<n> pass=<0|1>`.
   - **Performance:** time one call under CUDA-graph replay, take the mean per rank and the MAX
     over ranks, and print `Perf: <ms> ms (<case>)`.
   - **Cases:** use the tokens per rank the stage runs at in the served workload
     (`WORKLOAD_SUMMARY` / the traces under `TRACE_DIR`, the candidate's `m_range`), named
     `<tokens>_uniform`, with each case's weight.

   An existing operator benchmark in the tree may be adapted instead, as long as it meets this
   contract.
4. **Check it** on `SERVING_GPU` under a group lease, on the frozen tree. Correctness must pass
   for every case, and performance must print `Perf` for every case.
5. **Mega args.** Return only what is specific to this stage. The stage sets the rest:
   - paths, GPUs and budget: `mode`, `workflow_dir`, `kernel_path`, `benchmark_harness`,
     `gpu_ids`, `gpus_per_job`, `exp_root`, `state_dir`, `budget`, `mega_time_budget_s`;
   - search policy from `kernel_fusion/l3_mega_defaults.json`: accuracy metric and threshold,
     replays, graph capture, pairs, timeouts, and `launch_target` per stage kind (2 for an EP MoE,
     1 for a collective with its compute).

   Write:
   - **Guards:** `target_guards` is the case or cases carrying the most weight.
     `regression_guards` is the other cases (veto only), each with
     `regression_floor_by_guard` 0.98.
   - **Accuracy cases:** `required_accuracy_cases` = every case's token count. Set
     `accuracy_threshold` only when the model needs tighter than the default 0.1.
   - **Fusion target:** set `launch_target` only when this stage needs a different count from
     its kind's default.
   - **Environment:**
     - `candidate_import_modules` = `[<package>]`;
     - `device_description` = `"<EP|TP><n>, <n>x <gpu>/<gfx>"`.
   - **`positive_control`:** a calibrated device-side delay on the measured operator stream,
     giving -4% to -9% on the main target guard, built in an isolated copy.
   - **Language:** when `KERNEL_LANGUAGE` is given, the candidates are written in it (the stage
     also states it in the task); otherwise they follow the source they start from.
   - **`task`:** one paragraph covering the model, the operator and its call site, dtype, the rank
     group, the exact UT commands, the fact that the baseline at `kernel_path` is frozen, and the
     PASS criteria. Do not include paths of other runs or earlier state.

   Write your return to `L3_DIR/prepared.json`.
6. **Return** one of:
   - `{status: "ready", kernel_path, benchmark_harness, ep_size, package, call_site, cases:
     [{case, tokens, weight}], mega_args}`;
   - `{status: "blocked", reason}`, naming what is missing.

## PHASE=package

You receive one of:
- `MEGA_RESULT` (mega's verdict and its selected candidate: `final_tree` (else `tree`), `patch`,
  `activation`, `per_case`) together with `PREPARED`;
- `MEGA_RUN`, a finished mode=mega run (its `state_dir`, eval dir or a `mega_result.json`).

1. **Find the delivered candidate.**
   - With `MEGA_RUN`, read it: the candidate registry in `state_dir`, or the eval dir's
     `tech_lead_report.md` and `final_patch.diff`. It must be a delivered result
     (`mega_deliverable` true, or a selected candidate whose final validation passed). It must
     also be for this stage: the same communication kernels, the same rank count (an EP8 result
     does not serve an EP4 deployment: the experts are split differently), and the same call site as
     `L3_STAGE.seam` / `OPERATOR_HINT`. Otherwise return `correctness: "fail"` with the reason.
     Recover its UT and frozen tree from the run.
   - Write `L3_DIR/mega_result.json` with what you used.
2. **Overlay.** Write `OVERLAY_DIR/sitecustomize.py`. The stage stacks it with the other fused
   kernels by renaming each `sitecustomize.py`, so follow the same rules as an author agent's
   overlay:
   - **Imports:** only the standard library at top level.
   - **Code swap:** put the candidate's code first on `sys.path`. Either apply mega's final
     patch to a copy of the INSTALLED package under `OVERLAY_DIR`, or use the candidate tree
     when it is a complete package. Build JIT modules once here, with the activation set, so
     ranks do not race on a JIT build at server start.
   - **Activation:** set the candidate's activation switches in `os.environ` at top level,
     before the package is imported.
   - **Banner:** print `BANNER` to stderr once per process, from inside the stage's call site
     (wrap the call site lazily, after its module is first imported), the first time it takes
     the mega path. Never print it at patch time.
   - **Rank-uniform path:** the mega path must be chosen the same way on every rank, because
     mega's activation is a static capacity decision. Add no per-rank token-count gate: it would
     desynchronize the stage's collectives.
3. **Check.**
   - On CPU: `PYTHONPATH=OVERLAY_DIR python3 -c "import <package>, <backend>"`.
   - On GPU: run the UT in correctness mode for the target case on `SERVING_GPU`, under a group
     lease, with `OVERLAY_DIR` first on `PYTHONPATH` (`--replays 16`). It must pass and print the
     candidate's path marker and `BANNER` on every rank.
4. **Return** `final_patch` (the patch you applied, plus the overlay files, written to
   `L3_DIR/final_patch.diff`), `overlay_dir`, `banner`, `isolated_speedup` (mega's verified
   speedup on the target case), `correctness`, `launches_after`, `ut_command` and `notes`.
