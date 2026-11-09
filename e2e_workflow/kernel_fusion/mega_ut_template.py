#!/usr/bin/env python3
"""Multi-rank UT template for an L3 MegaKernel region authored by the kernel workflow's mode=mega (kf_mega).

    torchrun --standalone --nproc_per_node=<ranks> mega_ut.py correctness [--replays 256]
    torchrun --standalone --nproc_per_node=<ranks> mega_ut.py performance [--case <id>] [--json-output f.json]

Output contract mode=mega reads (keep it when adapting):
  correctness  [RESULT] case=<id> relL2=<x> replays=<n> pass=<0|1>   (vs a reference; graph replays == eager)
  performance  Perf: <ms> ms (<case id>)                             (CUDA-graph replay of one call, rank-max)
  per rank     [op] path=<NAME> rank=<r>                             (which path ran)
One operator per process, inputs seeded by rank. Replace the three hooks; the default is a toy
all-reduce + linear so the template runs as is.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist

# case id -> (tokens per rank, route): every served phase, see the prepare step
CASES = {
    "64_uniform": (64, "uniform"),
    "4096_uniform": (4096, "uniform"),
}
HIDDEN = 4096


# ---------------------------------------------------------------- TODO hooks
def build_operator(case_id, tokens, route, rank, world):
    """Return op(x) -> y for this rank. Build weights/buffers here, once per process."""
    torch.manual_seed(1234)                      # same weights on every rank
    w = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.bfloat16) / HIDDEN ** 0.5
    print(f"[op] path=BASELINE rank={rank}", flush=True)

    def op(x):
        y = x.clone()
        dist.all_reduce(y)
        return y @ w
    op.weights = w
    return op


def make_inputs(case_id, tokens, route, rank):
    g = torch.Generator(device="cuda").manual_seed(1000 + rank)
    return torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g)


def reference(op, x):
    """fp32 reference of the same math (no fused kernels)."""
    y = x.float().clone()
    dist.all_reduce(y)
    return y @ op.weights.float()
# ----------------------------------------------------------------------------


def rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def capture(op, x):
    out = op(x)                                  # warm / JIT outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(x)
    return graph, out


def run_correctness(case_ids, replays):
    ok_all = True
    for cid in case_ids:
        tokens, route = CASES[cid]
        rank, world = dist.get_rank(), dist.get_world_size()
        op = build_operator(cid, tokens, route, rank, world)
        x = make_inputs(cid, tokens, route, rank)
        eager = op(x)
        err = rel_l2(eager, reference(op, x))
        graph, out = capture(op, x)
        replay_ok = True
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
        replay_ok = torch.equal(out, eager) or rel_l2(out, eager) < 1e-6
        flag = torch.tensor([1.0 if (err <= 0.1 and replay_ok) else 0.0], device="cuda")
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        ok = flag.item() == 1.0
        ok_all &= ok
        if rank == 0:
            print(f"[RESULT] case={cid} relL2={err:.4f} replays={replays} pass={int(ok)}", flush=True)
    return ok_all


def run_performance(case_ids, iters, rounds):
    results = []
    for cid in case_ids:
        tokens, route = CASES[cid]
        rank, world = dist.get_rank(), dist.get_world_size()
        op = build_operator(cid, tokens, route, rank, world)
        x = make_inputs(cid, tokens, route, rank)
        graph, _ = capture(op, x)
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        per_round = []
        for _ in range(rounds):
            dist.barrier()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                graph.replay()
            e.record()
            torch.cuda.synchronize()
            ms = torch.tensor([s.elapsed_time(e) / iters], device="cuda")
            dist.all_reduce(ms, op=dist.ReduceOp.MAX)             # rank-max
            per_round.append(ms.item())
        med = sorted(per_round)[len(per_round) // 2]
        results.append({"test_case_id": cid, "execution_time_ms": med, "rounds_ms": per_round})
        if rank == 0:
            print(f"Perf: {med:.4f} ms ({cid})", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["correctness", "performance"])
    ap.add_argument("--case", action="append", help="case id (repeatable); default: all")
    ap.add_argument("--replays", type=int, default=256)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--json-output", default="")
    args = ap.parse_args()

    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local}"))
    cases = args.case or list(CASES)
    if args.mode == "correctness":
        ok = run_correctness(cases, args.replays)
        code = 0 if ok else 1
    else:
        res = run_performance(cases, args.iters, args.rounds)
        if args.json_output and dist.get_rank() == 0:
            with open(args.json_output, "w") as f:
                json.dump({"test_cases": res}, f, indent=1)
        code = 0
    dist.destroy_process_group()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
