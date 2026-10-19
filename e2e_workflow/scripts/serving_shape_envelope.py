#!/usr/bin/env python3
"""Token-count envelope the serving engine can hand a fused seam, per phase.

A unit-side verdict is measured on the captured steady-state shape (e.g. decode M=24).
The server runs the same patched seam on many more shapes: at startup it captures a
CUDA graph for every capture size (vLLM up to --max-cudagraph-capture-size, sglang up to
--cuda-graph-max-bs x tokens per request), and prefill chunks go up to the chunk budget.
An overlay that is only correct at the steady-state shape and does not decline the rest
crashes the apply-back server (vLLM DSv4.1 e02, 2026-10-19: unit-tested at M=24, wrote
out of bounds while the server captured the M=1008 graph).

This derives, from the serving flags only, the range [min_tokens, max_tokens] each phase
class can see:
  graph   -- decode / target-verify steps (CUDA-graph captured)
  prefill -- prefill / extend chunks
A bound that the flags do not pin is left out (never guessed); the unit-side harness
then only warns for that phase.

  python3 serving_shape_envelope.py --backend vllm --server-args "$EXTRA_SERVER_ARGS" \
      --out "$EVAL_DIR/fusion/serving_shape_envelope.json"
"""
import argparse
import json
import shlex
import sys

GRAPH_PHASES = ("decode", "verify", "target_verify", "draft", "draft_extend", "mtp")
PREFILL_PHASES = ("prefill", "extend", "mixed", "chunked_prefill")


def phase_class(phase):
    p = str(phase or "").strip().lower()
    if p in GRAPH_PHASES:
        return "graph"
    if p in PREFILL_PHASES:
        return "prefill"
    return None


def _flag_map(server_args):
    """--a b --c=d --flag -> {"a": "b", "c": "d", "flag": ""}; last occurrence wins."""
    tokens = shlex.split(server_args or "")
    out = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            key = tok[2:]
            if "=" in key:
                key, val = key.split("=", 1)
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                val = tokens[i + 1]
                i += 1
            else:
                val = ""
            out[key.replace("_", "-")] = val
        i += 1
    return out


def _int(flags, *names):
    for name in names:
        val = flags.get(name)
        if val not in (None, ""):
            try:
                return int(val), name
            except ValueError:
                pass
    return None, None


def _json_flag(flags, name):
    raw = flags.get(name)
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {}
    except ValueError:
        return {}


def _vllm(flags):
    env, notes = {}, []
    comp = _json_flag(flags, "compilation-config")
    sizes = comp.get("cudagraph_capture_sizes") or []
    graph_max, src = _int(flags, "max-cudagraph-capture-size")
    if graph_max is None and comp.get("max_cudagraph_capture_size"):
        graph_max, src = int(comp["max_cudagraph_capture_size"]), "compilation-config"
    if graph_max is None and sizes:
        graph_max, src = max(int(s) for s in sizes), "compilation-config.cudagraph_capture_sizes"
    graph_min = min(int(s) for s in sizes) if sizes else 1
    mode = str(comp.get("cudagraph_mode", "")).upper()
    if "enforce-eager" in flags or mode == "NONE":
        notes.append("CUDA graphs off: decode steps run eagerly at any batch")
        graph_max, src = None, None
    if graph_max is not None:
        env["graph"] = {"min_tokens": graph_min, "max_tokens": graph_max, "source": src}
    else:
        notes.append("graph max not pinned by flags (vLLM default depends on max_num_seqs)")
    pre_max, src = _int(flags, "max-num-batched-tokens")
    if pre_max is not None:
        env["prefill"] = {"min_tokens": 1, "max_tokens": pre_max, "source": src}
    else:
        notes.append("prefill max not pinned (--max-num-batched-tokens absent)")
    return env, notes


def _sglang(flags):
    env, notes = {}, []
    bs, src = _int(flags, "cuda-graph-max-bs-decode", "cuda-graph-max-bs")
    if "disable-cuda-graph" in flags:
        notes.append("CUDA graphs off: decode steps run eagerly at any batch")
        bs = None
    per_req, per_src = 1, "no speculative decoding"
    block, bsrc = _int(flags, "speculative-dspark-block-size")
    drafts, dsrc = _int(flags, "speculative-num-draft-tokens")
    if block is not None:
        per_req, per_src = block + 1, bsrc  # verify = block draft tokens + the bonus token
    elif drafts is not None:
        per_req, per_src = drafts, dsrc
    if bs is not None:
        env["graph"] = {"min_tokens": 1, "max_tokens": bs * per_req,
                        "source": "%s x %d tokens/request (%s)" % (src, per_req, per_src)}
    else:
        notes.append("graph max not pinned (--cuda-graph-max-bs[-decode] absent)")
    pre_max, src = _int(flags, "chunked-prefill-size", "max-prefill-tokens")
    if pre_max is not None and pre_max > 0:
        env["prefill"] = {"min_tokens": 1, "max_tokens": pre_max, "source": src}
    else:
        notes.append("prefill max not pinned (--chunked-prefill-size absent)")
    return env, notes


def derive(backend, server_args):
    flags = _flag_map(server_args)
    backend = str(backend or "").strip().lower()
    if backend == "vllm":
        phases, notes = _vllm(flags)
    elif backend == "sglang":
        phases, notes = _sglang(flags)
    else:
        phases, notes = {}, ["backend %r: no envelope rules; unit-side only warns" % backend]
    return {"backend": backend, "phases": phases, "notes": notes}


def required_points(envelope, phase):
    """The token counts a verdict for `phase` must cover: the two ends of the range."""
    cls = phase_class(phase)
    rng = ((envelope or {}).get("phases") or {}).get(cls) if cls else None
    if not rng:
        return None
    return sorted({int(rng["min_tokens"]), int(rng["max_tokens"])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--server-args", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    env = derive(a.backend, a.server_args)
    text = json.dumps(env, indent=2)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
