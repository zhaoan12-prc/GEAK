#!/usr/bin/env python3
"""Launch the real GEAK Semantics 1.2 metadata+marker replay."""
import argparse
import base64
import json
import os
import shlex
import subprocess
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_CAPTURE = os.path.join(
    SCRIPT_DIR, "semantic_runtime_capture.py")
DEFAULT_MODEL_RUNNER = (
    "/sgl-workspace/sglang/python/sglang/srt/"
    "model_executor/model_runner.py")
DEFAULT_RUNTIME_MODULE = (
    "/sgl-workspace/sglang/python/sglang/srt/"
    "geak_semantic_runtime_capture.py")
SENTINEL = "# GEAK_SEMANTICS_CAPTURE_BOOTSTRAP_V1"


def _run(command, stdout=None):
    return subprocess.run(command, check=True, stdout=stdout,
                          stderr=subprocess.STDOUT)


def _docker(container, shell_command, stdout=None):
    return _run(
        ["docker", "exec", container, "bash", "-lc", shell_command],
        stdout=stdout)


def _port_state(container, port):
    """Return "BUSY" or "FREE" for `port` inside `container`."""
    probe = (
        "if (exec 3<>/dev/tcp/127.0.0.1/%d) 2>/dev/null; then "
        "exec 3<&- 3>&-; echo BUSY; else echo FREE; fi" % int(port))
    result = subprocess.run(
        ["docker", "exec", container, "bash", "-lc", probe],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return "BUSY" if b"BUSY" in result.stdout else "FREE"


def _assert_port_free(container, port):
    """Refuse to start when the port is already serving.

    The capture container is shared: PID 1 is the orchestration process and
    other sessions may hold their own servers.  A pattern-kill (`pkill -f`)
    cannot tell those apart from ours, so a stale service is reported and
    the run stops -- never killed on a guess.
    """
    if _port_state(container, port) == "BUSY":
        raise RuntimeError(
            "port %s is already serving inside container %s; this run will "
            "not pattern-kill a process it did not start -- stop it from the "
            "session that owns it, then retry" % (port, container))


def _stop_service(container, pgid_path, port, timeout=90):
    """Group-kill only the process group this run started.

    `pgid_path` holds the PGID written by the setsid wrapper in `capture()`.
    Absent or empty means we never got as far as starting anything, so there
    is nothing of ours to stop.  The wait polls the port rather than process
    liveness because the server leaves unreaped children behind and PID 1 in
    the container does not reap them.
    """
    command = """
pgid_file=%s
[ -s "$pgid_file" ] || exit 0
pgid=$(cat "$pgid_file")
case "$pgid" in ''|*[!0-9]*) exit 0;; esac
kill -TERM -"$pgid" 2>/dev/null || true
for _ in $(seq 1 %d); do
  if ! (exec 3<>/dev/tcp/127.0.0.1/%d) 2>/dev/null; then exit 0; fi
  exec 3<&- 3>&-
  sleep 1
done
kill -KILL -"$pgid" 2>/dev/null || true
""" % (shlex.quote(str(pgid_path)), int(timeout), int(port))
    _docker(container, command)


def _deploy(container, model_runner, runtime_module):
    runtime_backup = runtime_module + ".geak_semantics_bak"
    _docker(container, """
set -e
if [ -e %s ] && [ ! -e %s ]; then cp %s %s; fi
""" % tuple(shlex.quote(path) for path in (
        runtime_module, runtime_backup, runtime_module, runtime_backup)))
    _run(["docker", "cp", RUNTIME_CAPTURE,
          "%s:%s" % (container, runtime_module)])
    bootstrap = """

# GEAK_SEMANTICS_CAPTURE_BOOTSTRAP_V1
import os as _geak_os
if _geak_os.environ.get("GEAK_SEMANTICS_CAPTURE", "0") in ("1", "true", "True"):
    import sys as _geak_sys
    try:
        from sglang.srt import geak_semantic_runtime_capture as _geak_capture
        _geak_original_load_model = ModelRunner.load_model
        def _geak_load_model(self, *args, **kwargs):
            result = _geak_original_load_model(self, *args, **kwargs)
            _geak_capture.install_on_model(self.model)
            return result
        ModelRunner.load_model = _geak_load_model
        _geak_sys.stderr.write("[GEAK_SEMANTICS] ModelRunner.load_model wrapped\\n")
    except Exception as _geak_error:
        _geak_sys.stderr.write("[GEAK_SEMANTICS] bootstrap failed: %s\\n" % _geak_error)
"""
    program = """
import os, shutil
path = %r
sentinel = %r
bootstrap = %s
with open(path) as fh:
    text = fh.read()
backup = path + ".geak_semantics_bak"
if not os.path.exists(backup):
    shutil.copyfile(path, backup)
if sentinel not in text:
    with open(path, "a") as fh:
        fh.write(bootstrap)
""" % (model_runner, SENTINEL, repr(bootstrap))
    encoded = base64.b64encode(program.encode("utf-8")).decode("ascii")
    _docker(container, "python3 -c \"import base64;exec(base64.b64decode('%s'))\"" %
            encoded)


def _restore(container, model_runner, runtime_module):
    """Restore both runtime files changed by `_deploy`.

    A failed capture must not leave the serving installation instrumented.  A
    pre-existing runtime module is restored from its backup; a module created
    only for this capture is removed.
    """
    model_backup = model_runner + ".geak_semantics_bak"
    runtime_backup = runtime_module + ".geak_semantics_bak"
    command = """
set -e
if [ -e %s ]; then cp %s %s; rm -f %s; fi
if [ -e %s ]; then
  cp %s %s
  rm -f %s
else
  rm -f %s
fi
""" % tuple(shlex.quote(path) for path in (
        model_backup, model_backup, model_runner, model_backup,
        runtime_backup, runtime_backup, runtime_module, runtime_backup,
        runtime_module))
    _docker(container, command)


_MODEL_PHASES = ("prefill", "decode")


def _required_phases(plan):
    return sorted({
        str(bucket.get("phase")).lower()
        for bucket in plan.get("target_buckets", [])
        if bucket.get("phase")
    })


def _warn_narrowed_phases(phases, plan, source):
    """B4: a phase set derived from a single-phase plan used to narrow the
    whole capture to that phase with no signal at all."""
    absent = [phase for phase in _MODEL_PHASES if phase not in set(phases)]
    if not absent:
        return []
    coverage = plan.get("phase_coverage") or {}
    notes = [
        "capture phases = %s (from %s); %s will NOT be observed."
        % (",".join(phases) or "<none>", source, ",".join(absent)),
        "Fusion candidates derived from this capture apply only to %s."
        % (",".join(phases) or "<none>"),
    ]
    if "decode" in absent:
        notes.append(
            "To cover decode: pass --phase decode. Decode shapes are captured "
            "during CUDA/HIP graph construction; no enforce-eager trace is "
            "required.")
    if coverage.get("phases_absent_from_tables"):
        notes.append(
            "upstream plan phase_coverage reports absent phases: %s"
            % ",".join(coverage["phases_absent_from_tables"]))
    for note in notes:
        print("[GEAK_SEMANTICS][warn] %s" % note, file=sys.stderr)
    return notes


def _observed_phases(shape_log):
    """Phases actually present in the emitted shape log."""
    alias = {"extend": "prefill", "prompt": "prefill", "generation": "decode"}
    seen = set()
    try:
        with open(shape_log) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                phase = str((record.get("context") or record).get(
                    "phase", "")).strip().lower()
                if phase:
                    seen.add(alias.get(phase, phase))
    except IOError:
        pass
    return seen


def capture(setup_path, capture_plan_path, out_dir, phases=None,
            forwards_per_bucket=1):
    with open(setup_path) as fh:
        setup = json.load(fh)
    phases = list(phases or setup.get("capture_phases", []))
    if forwards_per_bucket == 1 and setup.get("forwards_per_bucket") is not None:
        forwards_per_bucket = int(setup["forwards_per_bucket"])
    with open(capture_plan_path) as fh:
        plan = json.load(fh)
    phase_source = "--phase / setup.capture_phases"
    if not phases:
        phases = _required_phases(plan)
        phase_source = "plan.target_buckets"
    phases = [str(phase).strip().lower() for phase in phases if str(phase).strip()]
    phase_notes = _warn_narrowed_phases(phases, plan, phase_source)

    decode_requested = "decode" in phases
    required = ("container", "model", "benchmark", "port",
                "tensor_parallel_size", "workload")
    missing = [name for name in required if setup.get(name) is None]
    if missing:
        raise ValueError(
            "shape capture setup missing: %s" % ", ".join(missing))

    container = setup["container"]
    model_runner = setup.get("sglang_model_runner", DEFAULT_MODEL_RUNNER)
    runtime_module = setup.get(
        "geak_runtime_capture_module", DEFAULT_RUNTIME_MODULE)
    layers = sorted(set(
        int(target["representative_layer_id"])
        for target in plan.get("capture_targets", [])
        if target.get("representative_layer_id") is not None))
    if not layers:
        raise ValueError("capture plan has no representative layers")

    os.makedirs(out_dir, exist_ok=True)
    shape_log = os.path.join(out_dir, "shape.jsonl")
    graph_capture_trace = os.path.join(
        out_dir, "graph_capture-TP-{rank}.trace.json")
    benchmark_log = os.path.join(out_dir, "benchmark.log")
    pgid_path = os.path.join(out_dir, "server.pgid")
    for path in (shape_log,):
        if os.path.exists(path):
            os.remove(path)

    _assert_port_free(container, setup["port"])
    workload = setup["workload"]
    repository = setup.get(
        "benchmark_repository",
        "/mnt/raid0/zhaoan12/repo/InferenceX")
    benchmark = setup["benchmark"]
    extra_server_args = str(setup.get("extra_server_args", "")).strip()
    if (setup.get("disable_cuda_graph")
            or "--disable-cuda-graph" in extra_server_args):
        raise ValueError(
            "semantic shape capture requires CUDA/HIP graph construction; "
            "remove --disable-cuda-graph")
    if "--enable-profile-cuda-graph" not in extra_server_args:
        extra_server_args = (
            extra_server_args + " --enable-profile-cuda-graph").strip()
    extra_env = str(setup.get("extra_env", "")).strip()
    callable_targets = list(setup.get("callable_targets", []))
    # V-absorb uses a functional BMM, so nn.Module hooks only expose the broad
    # self-attention wrapper. Probe torch.bmm during graph construction; logging
    # remains restricted to selected layers and capture buckets.
    if decode_requested and "torch:bmm" not in callable_targets:
        callable_targets.append("torch:bmm")
    command = """
set -e
export GEAK_SEMANTICS_CAPTURE=1
export GEAK_SEMANTICS_SHAPE_LOG=%s
export GEAK_SEMANTICS_RANK=0
export GEAK_SEMANTICS_LAYERS=%s
export GEAK_SEMANTICS_PHASES=%s
export GEAK_SEMANTICS_FORWARDS_PER_BUCKET=%s
export GEAK_SEMANTICS_CALLABLE_TARGETS=%s
export GEAK_SEMANTICS_REQUIRE_PROFILER=0
export GEAK_SEMANTICS_GRAPH_CAPTURE_TRACE=%s
export EXTRA_SERVER_ARGS=%s
export EXTRA_ENV=%s
export PROFILE=0
export REPEATS=%s
export PROFILE_NUM_STEPS=%s
export OUT_DIR=%s
export MODEL=%s
export TP=%s
export GPU=%s
export CONC=%s
export ISL=%s
export OSL=%s
export RANDOM_RANGE_RATIO=%s
export NUM_PROMPTS=%s
export NUM_WARMUPS=%s
export SEED=%s
export MEM_FRACTION=%s
export BENCH_CLIENT=%s
export INFERENCEX_PATH=%s
export BENCH_COLD_FINAL=%s
export PORT=%s
export RESULT_FILENAME=geak_semantics_1_2_capture
export EVAL_ONLY=false
export RUN_EVAL=false
export ROCR_VISIBLE_DEVICES=%s
export GEAK_SERVER_PGID_FILE=%s
export GEAK_BENCHMARK_SCRIPT=%s
cd %s
: > "$GEAK_SERVER_PGID_FILE"
setsid bash -c 'echo $$ > "$GEAK_SERVER_PGID_FILE"; exec bash "$GEAK_BENCHMARK_SCRIPT"' &
geak_wrapper=$!
wait "$geak_wrapper"
""" % (
        shape_log, ",".join(str(layer) for layer in layers),
        ",".join(phases or []), forwards_per_bucket,
        ",".join(callable_targets),
        graph_capture_trace,
        shlex.quote(extra_server_args), shlex.quote(extra_env),
        int(setup.get("repeats", 1)),
        int(setup.get("profile_num_steps", 1)),
        out_dir,
        setup["model"], setup["tensor_parallel_size"],
        setup.get("gpu_ids", "0"),
        workload["concurrency"], workload["input_length"],
        workload["output_length"], workload.get("random_range_ratio", 0.8),
        int(workload.get("num_prompts", 20)),
        int(workload.get("num_warmups", min(workload["concurrency"], 8))),
        int(workload.get("seed", 0)),
        float(setup.get("mem_fraction", 0.8)),
        setup.get("bench_client", "inferencex"),
        setup.get("inferencex_path", repository),
        "1" if setup.get("bench_cold_final", False) else "0",
        setup["port"], setup.get("gpu_ids", "0"), pgid_path, benchmark,
        repository)
    started = time.time()
    try:
        _deploy(container, model_runner, runtime_module)
        with open(benchmark_log, "w") as log:
            _docker(container, command, stdout=log)
    finally:
        # Stop only the process group this replay started.
        try:
            _stop_service(container, pgid_path, setup["port"])
        finally:
            _restore(container, model_runner, runtime_module)
    graph_capture_trace_rank0 = graph_capture_trace.format(rank=0)
    trace = graph_capture_trace_rank0
    if not os.path.exists(shape_log) or os.path.getsize(shape_log) == 0:
        raise RuntimeError(
            "GEAK runtime capture produced no shape metadata: %s" %
            benchmark_log)
    if not trace:
        raise RuntimeError(
            "GEAK runtime capture produced no profiler trace: %s" %
            benchmark_log)
    if not os.path.exists(trace):
        raise RuntimeError(
            "GEAK runtime capture trace is missing: %s (see %s)" %
            (trace, benchmark_log))
    result = {
        "schema_version": 1,
        "status": "pass",
        "capture_mode": "graph_capture_metadata_plus_runtime_markers",
        "shape_capture_execution": "graph_capture",
        "capture_phases": list(phases or []),
        "forwards_per_bucket": int(forwards_per_bucket),
        "container": container,
        "representative_layers": layers,
        "callable_targets": callable_targets,
        "callable_kernel_map": list(
            setup.get("callable_kernel_map", [])),
        "source_wrapper_map": list(
            setup.get("source_wrapper_map", [])),
        "shape_log": shape_log,
        "capture_trace": trace,
        "graph_capture_trace": graph_capture_trace_rank0,
        "phase_notes": phase_notes,
        "observed_phases": sorted(_observed_phases(shape_log)),
        "benchmark_log": benchmark_log,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    observed = _observed_phases(shape_log)
    unmet = sorted(set(phases) - observed)
    if unmet:
        result["status"] = "fail"
        result["failure"] = (
            "requested phase(s) %s produced zero shape records (observed: %s). "
            "See %s" % (",".join(unmet), ",".join(sorted(observed)) or "none",
                        benchmark_log))
        print("[GEAK_SEMANTICS][fail] %s" % result["failure"], file=sys.stderr)
    result_path = os.path.join(out_dir, "CAPTURE_RESULT.json")
    result["result_json"] = result_path
    with open(result_path, "w") as fh:
        json.dump(result, fh, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", required=True)
    parser.add_argument("--capture-plan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--phase", action="append", default=[])
    parser.add_argument("--forwards-per-bucket", type=int, default=1)
    parser.add_argument("--result-json", default="")
    args = parser.parse_args()
    result = capture(
        args.setup, args.capture_plan, args.out_dir,
        args.phase, args.forwards_per_bucket)
    if args.result_json:
        with open(args.result_json, "w") as fh:
            json.dump(result, fh, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
