# vllm serving adapter for bench_e2e.sh.  Sourced (not executed). Defines the contract functions.
# Reads the same env the dispatcher exports; sets SERVER_PID in adapter_launch; appends canonical
# result lines to $RESULT_JSONL.
#
# VERSION NOTE (read scripts/../knowledge/preflight.md): the vllm CLI surface drifts across releases.
#   - `vllm serve` and `vllm bench serve` exist on current vllm (>=0.6 / v1). On older builds the
#     equivalents are `python -m vllm.entrypoints.openai.api_server` and
#     `python benchmarks/benchmark_serving.py` (needs the repo checkout).
#   - `--gpu-memory-utilization` is the vllm analogue of sglang's `--mem-fraction-static`.
#   - profiling: vllm >=0.19 MOVED torch-profiler config from the VLLM_TORCH_PROFILER_DIR env var to the
#     `--profiler-config` CLI flag (the env var is now an UNKNOWN var -> warned + ignored -> NO trace is
#     written, so TraceLens gets no input). We emit `--profiler-config '{"profiler":"torch",...}'` so the
#     bench's `--profile` dumps a *.pt.trace.json.gz into PROFILE_DIR.
#     CROSS-VERSION: we DON'T blindly pass `--profiler-config` — old (<0.19) builds' argparse rejects the
#     unknown flag and the server never starts. We detect support by importing vllm.config.ProfilerConfig
#     (only present on builds that have the flag): if it imports -> use the flag (new builds); otherwise
#     fall back to the VLLM_TORCH_PROFILER_DIR env (old builds). This probe is device-independent (unlike
#     `vllm serve --help[=all]`, which initializes config/device and CRASHES on a driver-less host -> empty
#     output -> false-negative -> profiling silently lost) and far cheaper (no full server spin-up).
#   - fusion capture: GEAK_FUSION_TRACE=1 (via EXTRA_ENV) switches the profiler to the KernelFusion
#     evidence mode: with_stack ON (module hierarchy for layer boundaries) and an iteration-bounded
#     window via ProfilerConfig.max_iterations. Ordinary Profile rounds keep with_stack OFF.
#   - PHASE ANNOTATIONS ARE FREE, and this file used to claim otherwise. vllm's gpu_worker wraps every
#     execute_model in annotate_profile() UNCONDITIONALLY; its default branch emits
#     `execute_context_<nreq>(<ntok>)_generation_<nreq>(<ntok>)` as a record_function, which lands in
#     gpu_user_annotation and is exactly the dialect parse_profile._seg reads. Measured on v0.27.1 /
#     gfx942 / conc 8: 774 spans in a 25 s window -> a full serving block (steady=true,
#     decode_batch_captured == CONC) and a per-kernel prefill/decode/both phase. Nothing to enable.
#     `detailed_trace_annotation` is a RICHER opt-in (adds per-phase sq/sk/sqsq/sqsk roofline terms)
#     that v0.27.1 does accept -- but it is still not passed by default: the plain annotation already
#     gives the split, and every extra key is one more thing a strict schema can reject.
# The Director's preflight step should smoke-test these two commands on the target image and record
# any needed EXTRA_SERVER_ARGS BEFORE the run relies on them. This adapter targets the current CLI.

adapter_default_port() { echo 8000; }

# ---------------------------------------------------------------------------
# KernelFusion capture mode.  Set by the orchestrator via EXTRA_ENV
# (GEAK_FUSION_TRACE=1).  Fusion has a DIFFERENT evidence goal from the native Top-N
# profiler: it needs Python/module spans and a short window that contains BOTH serving
# phases, not a long statistical sample.  See bench_e2e.sh's sizing block.
_geak_fusion_capture() {
  case " ${EXTRA_ENV:-} " in *" GEAK_FUSION_TRACE=1 "*) return 0 ;; esac
  [ "${GEAK_FUSION_TRACE:-0}" = "1" ]
}

# Which ProfilerConfig keys does THIS build accept?
#
# vllm's ProfilerConfig is a strict (pydantic extra=forbid) schema: passing ONE key it does
# not know ABORTS the server at startup.  The old all-or-nothing probe (`import
# ProfilerConfig` -> assume the whole current key set) is why `detailed_trace_annotation`
# could kill a run.  Introspect the real field names instead and emit only what is present,
# so a key that appears/disappears across releases degrades to "not passed" rather than to a
# dead server.  Device-independent (no config/device init), so it is also safe on a
# driver-less host, unlike `vllm serve --help`.
_vllm_profiler_fields() {
  python3 - <<'PY' 2>/dev/null
try:
    from vllm.config import ProfilerConfig
except Exception:
    raise SystemExit(1)
names = []
try:
    import dataclasses
    names = [f.name for f in dataclasses.fields(ProfilerConfig)]
except Exception:
    pass
if not names:
    names = list(getattr(ProfilerConfig, "model_fields", {}) or {})
print(" ".join(names))
PY
}

_vllm_has_field() { case " $1 " in *" $2 "*) return 0 ;; esac; return 1; }

adapter_launch() {
  # Pin GPU_ARCHS so aiter's JIT skips rocm_agent_enumerator/_detect_native (see sglang.sh / gpu_lock.sh).
  local _ga="${GPU_ARCHS:-$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || true)}"
  # Enable the server-side torch profiler in a version-portable way. Two mutually-exclusive paths:
  #   new vllm (>=0.19): pass --profiler-config (the env var is rejected/ignored there).
  #   old vllm (<0.19) : pass VLLM_TORCH_PROFILER_DIR env (the CLI flag does NOT exist -> argparse would
  #                      abort the launch, so we MUST NOT pass it on old builds).
  # We pick the path by importing ProfilerConfig (device-independent capability probe). The JSON is held
  # in an array so it stays ONE argument (no word-split / brace-expansion). When PROFILE_DIR is unset,
  # profiling is off: the array is empty and we don't export the env var.
  local -a _prof=()
  local -a _prof_env=()
  if [ -n "${PROFILE_DIR:-}" ]; then
    # with_stack: vllm's ProfilerConfig DEFAULTS torch_profiler_with_stack to TRUE, the exact
    # OPPOSITE of the sglang adapter's default. Left implicit, every ordinary Profile round
    # records python_function spans for the whole window -> a trace an order of magnitude
    # larger, whose ROCm flush can block the scheduler (the failure sglang.sh documents at
    # length). So state it EXPLICITLY in both directions rather than inheriting a default:
    #   normal profile  -> false  (kernel timeline + Input Dims is all parse_profile needs)
    #   fusion capture  -> true   (semantics needs the nn.Module hierarchy for layer boundaries)
    # Override with VLLM_PROFILE_WITH_STACK=true|false.
    local _stack="${VLLM_PROFILE_WITH_STACK:-false}"
    local _max_iters=""
    if _geak_fusion_capture; then
      _stack="${VLLM_PROFILE_WITH_STACK:-true}"
      # Iteration-bounded window. NOTE this contradicts the older comment in this file that
      # "vllm's /start_profile takes NO num_steps": it does not, but ProfilerConfig carries
      # delay_iterations/max_iterations and WorkerProfiler.step() counts engine iterations,
      # auto-stopping (and flushing) at the limit. That is the vllm analogue of sglang's
      # PROFILE_NUM_STEPS, and it is what keeps a stack-heavy fusion trace bounded.
      #
      # It must still be big enough to contain BOTH phases. Unlike sglang there is no
      # profile_by_stage, so prefill and decode are only separable because a saturated
      # continuous-batching server interleaves them within one window -- one iteration (the
      # sglang setting) would capture one phase and silently halve coverage. If Phase 1
      # reports a single phase, raise this.
      _max_iters="${GEAK_FUSION_MAX_ITERS:-16}"
      # Skip this many engine iterations before the profiler actually starts.
      #
      # Landing the window in DECODE by wall-clock warmup is a race, and it is one you
      # lose quietly: at ISL 8192 / CONC 64 the prefill ramp is ~32 steps, so a short
      # warmup captures 24 iterations of pure prefill (Phase 1 then reports
      # `mixed_trace_no_decode_steps_in_window`), while a warmup long enough to clear the
      # ramp can outlive the background load entirely and fall back to a cold-start
      # capture -- prefill again, by a different route. Both were observed on MiniMax-M3.
      #
      # delay_iterations counts ENGINE STEPS, so it clears the ramp deterministically
      # regardless of how slow those steps are. Set it to at least
      # ceil(CONC*ISL/max_num_batched_tokens) to skip the prefill ramp.
      _delay_iters="${GEAK_FUSION_DELAY_ITERS:-0}"
    fi
    local _fields
    if _fields="$(_vllm_profiler_fields)" && [ -n "$_fields" ]; then
      # Build the JSON from the keys this build actually declares.
      local _json="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\""
      _vllm_has_field "$_fields" torch_profiler_record_shapes \
        && _json="$_json,\"torch_profiler_record_shapes\":true"
      _vllm_has_field "$_fields" torch_profiler_with_stack \
        && _json="$_json,\"torch_profiler_with_stack\":$_stack"
      if [ -n "${_delay_iters:-}" ] && [ "${_delay_iters:-0}" -gt 0 ] \
           && _vllm_has_field "$_fields" delay_iterations; then
        _json="$_json,\"delay_iterations\":$_delay_iters"
      fi
      if [ -n "$_max_iters" ] && _vllm_has_field "$_fields" max_iterations; then
        _json="$_json,\"max_iterations\":$_max_iters"
      elif [ -n "$_max_iters" ]; then
        echo "!!! fusion capture: this build's ProfilerConfig has no max_iterations; the window" >&2
        echo "!!! falls back to PROFILE_WINDOW_SEC timing (trace size is NOT iteration-bounded)" >&2
      fi
      _json="$_json}"
      _prof=(--profiler-config "$_json")
    else
      # Old (<0.19) builds: the CLI flag does not exist and argparse would abort the launch,
      # so configure through the (now deprecated, but honored there) env vars instead.
      _prof_env=(VLLM_TORCH_PROFILER_DIR="$PROFILE_DIR"
                 VLLM_TORCH_PROFILER_RECORD_SHAPES=1
                 VLLM_TORCH_PROFILER_WITH_STACK="$([ "$_stack" = true ] && echo 1 || echo 0)")
    fi
  fi
  # Launch through $SERVER_LAUNCH_PREFIX (adapter contract): it puts the server in its
  # own session so teardown can prove the process group is ours. Empty when unset.
  # shellcheck disable=SC2086
  ${SERVER_LAUNCH_PREFIX:-} env $EXTRA_ENV \
    ${_ga:+GPU_ARCHS=$_ga} \
    HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU \
    "${_prof_env[@]}" \
    PYTHONPATH="${OVERLAY_PYTHONPATH:+$OVERLAY_PYTHONPATH:}${PYTHONPATH:-}" \
    vllm serve "$MODEL" \
      --host "$HOST" --port "$PORT" \
      --tensor-parallel-size "$TP" \
      --gpu-memory-utilization "$MEM_FRACTION" \
      "${_prof[@]}" \
      $EXTRA_SERVER_ARGS \
      > "$LOG" 2>&1 &
  SERVER_PID=$!
}

adapter_health() { curl -sf "${BASE_URL}/health" >/dev/null 2>&1; }

# adapter_bench NUM_PROMPTS MAX_CONC PROFILE_FLAG
adapter_bench() {
  local NUMP="$1" MAXC="$2" PROF="${3:-0}"
  local res_json="$PROFILE_DIR/.vllm_bench_$$_${RANDOM}.json"
  local extra=()
  [ "$PROF" = "1" ] && extra=(--profile)
  # Custom-tokenizer models (e.g. Kimi-K2.6) need the bench client to trust remote code to load
  # the tokenizer; mirror the server's trust setting (BENCH_TRUST_REMOTE_CODE from the dispatcher).
  [ "${BENCH_TRUST_REMOTE_CODE:-0}" = "1" ] && extra+=(--trust-remote-code)
  # GREEDY (--temperature 0) + --ignore-eos: deterministic, fixed-length OSL output. This is the
  # correct protocol for optimization work — it makes throughput reproducible, output parity byte-exact,
  # and speculative-decoding (MTP/EAGLE) acceptance meaningful (recent vllm dropped the temp==0 default).
  vllm bench serve \
    --backend vllm --base-url "$BASE_URL" --model "$MODEL" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --num-prompts "$NUMP" --max-concurrency "$MAXC" \
    --seed "$SEED" --temperature 0 --ignore-eos \
    --save-result --result-filename "$res_json" "${extra[@]}"
  # vllm writes ONE result object (keys: output_throughput, median_ttft_ms, median_tpot_ms, ...).
  # Append it as a single jsonl line into the dispatcher's canonical results file.
  if [ -f "$res_json" ]; then
    python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))" "$res_json" \
      >> "$RESULT_JSONL" 2>/dev/null || cat "$res_json" >> "$RESULT_JSONL"
    rm -f "$res_json"
  fi
}

# adapter_profile_window — capture a profiler window on the ALREADY-RUNNING, warm, mid-load server via
# vllm's HTTP profiler, so the trace reflects the real continuous-batching steady-state mix (prefill
# chunks + decode interleaved) instead of the cold prefill ramp `vllm bench serve --profile` would catch.
# Requires the server to have been launched with the torch profiler enabled (adapter_launch does:
# --profiler-config / VLLM_TORCH_PROFILER_DIR, with record_shapes=true so the parser gets Input Dims).
#
# DIFFERS FROM sglang: vllm's /start_profile takes NO num_steps — it runs until /stop_profile. So the
# window is TIME-controlled: start, sleep PROFILE_WINDOW_SEC of steady-state load, then stop. The trace
# is flushed on /stop_profile (the server blocks until the flush completes), so we allow a long curl
# timeout and then confirm a new trace landed.
adapter_profile_window() {
  local before after
  before=$(ls "$PROFILE_DIR"/*.trace.json* 2>/dev/null | wc -l)
  if ! curl -sf -X POST "${BASE_URL}/start_profile" >/dev/null 2>&1; then
    echo "!!! /start_profile request failed (vllm torch profiler not enabled at launch?)" >&2
    return 1
  fi
  # Window duration. With max_iterations set (fusion capture) the worker self-stops at the
  # iteration limit and this sleep is only an upper bound -- keep it short so we are not
  # idling long after the profiler already flushed.
  if _geak_fusion_capture; then
    sleep "${GEAK_FUSION_WINDOW_SEC:-20}"
  else
    sleep "${PROFILE_WINDOW_SEC:-40}"
  fi
  # /stop_profile flushes the trace; the server waits for the flush, so give curl a generous timeout.
  curl -s --max-time "${PROFILE_WINDOW_TIMEOUT:-180}" -X POST "${BASE_URL}/stop_profile" \
    >/dev/null 2>&1 || echo "!!! /stop_profile request errored (checking for a trace anyway)" >&2
  # wait for a NEW trace to land (flush is async on some builds even after the stop returns)
  local deadline=$(( $(date +%s) + ${PROFILE_WINDOW_TIMEOUT:-180} ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    after=$(ls "$PROFILE_DIR"/*.trace.json* 2>/dev/null | wc -l)
    [ "$after" -gt "$before" ] && { sleep 2; return 0; }   # +2s for the write to flush
    sleep 3
  done
  after=$(ls "$PROFILE_DIR"/*.trace.json* 2>/dev/null | wc -l)
  [ "$after" -gt "$before" ]
}
