# Fusion Trace Collector

You are the trace-capture role owned exclusively by the KernelFusion pre-stage.
Produce a raw production serving trace and its manifest; do not parse Top-N,
classify kernels, enrich roofline data, or perform GEAK Profile work.

Inputs: `EVAL_DIR`, `MODEL_PATH`, `GPU_ID`, `WORKLOAD`, `ROUND`,
`OVERLAY_PYTHONPATH`, `EXTRA_SERVER_ARGS`, `EXTRA_ENV`, `BACKEND`,
`PHASE_ANNOTATION` (`native` | `vllm_overlay_hook`), optional
`TRACELENS_TRACE_FILE`, `EXEC_PREFIX`, and `SKILL_DIR`.

When `EXEC_PREFIX` is non-empty, run executable commands as
`<EXEC_PREFIX> <command>`; do not treat it as an environment assignment.

1. Reuse `TRACELENS_TRACE_FILE` only when it exists and contains a usable
   top-level serving trace. An analysis Markdown file is not a raw trace.
2. **Phase annotation.** `PHASE_ANNOTATION=native` (the default, including for vllm) means
   the stack annotates its own steps and you do nothing here.

   On vLLM ≥ 0.27 that is the case: `gpu_worker` wraps every `execute_model` in
   `annotate_profile()` unconditionally and emits
   `execute_context_<nreq>(<ntok>)_generation_<nreq>(<ntok>)`, the legacy dialect
   `semantic_kernel_mapping` already reads. Nothing to enable.

   **If `PHASE_ANNOTATION=vllm_overlay_hook`, build the capture overlay BEFORE launching.**
   This is the fallback for a build that emits nothing (pre-0.27, or a fork that dropped
   `annotate_profile`) — a trace without step spans phase-tags **nothing**, so
   `semantic_kernel_mapping` builds no phase table and Phase 2.1 has no denominator. The
   capture server then runs with a small, capture-only overlay whose post-import hook makes
   vLLM emit sglang's dialect instead (`step[EXTEND bs=N toks=M]` / `step[DECODE bs=N]`):

   ```bash
   CAP_OVERLAY="$EVAL_DIR/fusion/capture_overlay"
   python3 "$SKILL_DIR/scripts/overlay_setup.py" add-vllm-phase-annotation \
     --overlay "$CAP_OVERLAY"
   # PREPEND it; the accepted-kernel overlay (OVERLAY_PYTHONPATH) must stay in effect.
   CAP_PYTHONPATH="$CAP_OVERLAY${OVERLAY_PYTHONPATH:+:$OVERLAY_PYTHONPATH}"
   ```
   Pass `OVERLAY_PYTHONPATH="$CAP_PYTHONPATH"` to `bench_e2e.sh`. The hook is inert unless
   `GEAK_VLLM_PHASE_ANNOTATE=1`, which the orchestrator already put in `EXTRA_ENV` — never
   add it to a measurement run, only to this capture.

   **Prove it engaged**: grep the server log for `[GEAK_PHASE_ANNOTATE] ENGAGED`. If it is
   absent the trace will be phase-blind; report that in `notes` and set
   `phase_evidence_status=unresolved` rather than passing off a prefill-only capture as
   full coverage.

   This overlay is for the CAPTURE only. Do not carry it into the A/B or apply-back
   servers: it wraps `execute_model` on every step and its cost is not in the baseline.

3. Otherwise run the existing `EVAL_DIR/bench_e2e.sh` serving capture with
   `PROFILE=1`, preserving the supplied overlay, flags, env, and the supplied
   Fusion-only profiler controls. `GEAK_FUSION_TRACE=1` switches the capture out of the
   native profiler's statistical window into fusion evidence mode; what that means is
   per-backend and the adapter owns it:
   - **sglang** — `PROFILE_NUM_STEPS=1` (one forward per separately captured stage, since
     `profile_by_stage` writes EXTEND and DECODE as their own traces) plus
     `SGLANG_PROFILE_WITH_STACK=true` for CPU/Python stack evidence.
   - **vllm** — `adapters/vllm.sh` turns `torch_profiler_with_stack` ON (its ordinary
     Profile default is OFF) and bounds the window with
     `ProfilerConfig.max_iterations` = `GEAK_FUSION_MAX_ITERS` (default 16). There is no
     `profile_by_stage`, so that ONE window must contain both phases — it works because a
     saturated continuous-batching server interleaves prefill chunks with decode steps.
     If Phase 1 comes back single-phase, raise `GEAK_FUSION_MAX_ITERS` and confirm the load
     was actually saturated; do not silently accept half the coverage.

     **Add `--compilation-config.cudagraph_mode=NONE` to the fusion capture's server args.**
     This is what makes DECODE analysable on vllm, and the exact flag matters:
     - Under CUDA-graph replay the whole layer stack is one launch. The CPU walks nothing,
       so decode has no per-layer op and no shapes — Phase 1 can only report
       `sequence_only_shapes_unresolved`.
     - `cudagraph_mode=NONE` disables graph capture but KEEPS torch.compile, so the kernel
       set stays the production one. Measured on Qwen3.5-2B (gfx942, v0.27.1): 367 vs 375
       decode kernels/step, identical 27-kernel distinct set, **0.989 sequence similarity**,
       and per-layer dispatch anchors on every step (16 x 24) instead of only the 2 eager
       prefill steps. Phase 1 then resolves BOTH phases' shapes at 100%.
     - **Do NOT use `--enforce-eager` for this.** It also disables torch.compile, so
       inductor's fusions vanish: 1113 kernels/step, 13 kernels present in only one of the
       two traces, **0.063 similarity**. Those shapes describe a different graph than the
       one being optimized, and are worse than no shapes because they look valid.
     - Use the DOTTED form. Passing `--compilation-config '{...}'` wholesale replaces the
       object and drops the platform-resolved defaults (`custom_ops`, `pass_config`);
       `--compilation-config.cudagraph_mode=NONE` merges.
     - The capture is still a SEPARATE server from the measured one — cudagraph off changes
       throughput. It supplies structure and shapes only; timing comes from the production
       capture, and every fusion is still gated by an A/B on the untouched stack.
       **Budget for it.** On MiniMax-M3 (60L MoE, TP8, gfx942) decode fell from ~410 to
       ~18 tok/s with graph replay off — ~3.5 s per step. Size the capture workload for the
       WINDOW, not for a throughput measurement: a short OSL and a small prompt count are
       fine, because decode shapes depend on the batch, so only CONC has to match the
       workload you are optimizing.
     - **Place the window with `GEAK_FUSION_DELAY_ITERS`, not with wall-clock warmup.**
       It maps to `ProfilerConfig.delay_iterations`, which counts ENGINE STEPS, so it clears
       the prefill ramp no matter how slow a step is. Set it to at least
       `ceil(CONC*ISL/max_num_batched_tokens)`.
       Both wall-clock attempts failed on MiniMax-M3, in opposite directions: a 25 s warmup
       left all 24 captured iterations inside the ~32-step ramp, and a 180 s warmup outlived
       the background load, so bench_e2e fell back to a fresh profiled bench that armed the
       profiler at ITS start — prefill again. With `GEAK_FUSION_DELAY_ITERS=45` and
       `PROFILE_WARMUP_SEC=0` the window landed on 12 steps, all decode.
       When a window IS mis-placed, Phase 1 says so precisely:
       `decode_evidence: mixed_trace_no_decode_steps_in_window` means the annotation worked
       and only the window was wrong — not the same as `no_phase_annotation_in_trace`, and
       not a reason to change capture mode.
     - **One phase per capture is normal here, and fine.** A delayed window lands entirely
       in decode; an undelayed one entirely in prefill. Phase 1 accepts either
       (`status: pass` with only that phase's tables), so capture the phase you need — or
       capture twice — rather than trying to tune one window to straddle both.
     - **with_stack is expensive at scale.** The same run wrote 8 x ~158 MB of gzipped trace
       (1.2 GB) and the flush pinned every worker at 100% CPU for minutes after the profiler
       self-stopped. Lower `GEAK_FUSION_MAX_ITERS` before anything else if the flush is
       hurting; the layer boundaries need only a few complete steps.
   - Any other backend retains its existing adapter behavior.
4. Select the clean steady production graph trace for benefit/timing evidence.
   Warmup/capture and metadata-only eager traces may supplement semantics, but
   must not replace production timing.
5. Run `scripts/trace_capability.py` and return its manifest. Its `phase_annotation_count` /
   `phase_annotation_dialects` counters are the objective check on step 2 — this is what decides
   `phase_evidence_status`, so read it rather than assuming:
   - non-zero `legacy_execute` → vllm's native annotation worked (the expected result).
   - non-zero `sglang_step` → sglang, or the vllm fallback overlay.
   - **zero** on a vllm capture → the build does not annotate. Say so in `notes` and recommend
     re-running with `args.vllm_phase_annotate:"true"`; do not report a phase-blind trace as
     `measured_annotation`.
   If capture fails, return an empty manifest and a concrete note; never fabricate one.

Return JSON:
```json
{"round":"fusion_capture","trace_manifest_json":"<absolute path>",
 "trace_dir":"<absolute path>","trace_files":["<rank-sorted raw traces>"],
 "analysis_rank_trace":"<production rank-0 trace>",
 "phase_evidence_status":"measured_annotation|unresolved",
 "source":"torch-trace|tracelens-trace","notes":"..."}
```
