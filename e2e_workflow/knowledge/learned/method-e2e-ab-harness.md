---
name: method-e2e-ab-harness
description: Honest e2e A/B - interleave both arms in ONE session on ONE node, gate on non-overlap, and never divide by a throughput measured elsewhere.
keywords: [method, e2e-ab, noise-band, interleaved, non-overlap, cross-node, warmup, jit, replicas, output-parity, negative-control]
kernels: [bench_serving]
platforms: [gfx942, gfx950]
kernel_class: method
regime: both
key: e2e A/B measurement · any gfx · sglang/vllm/atom
type: method
confidence: ★★★
effect: stops false wins AND false regressions - a positive median inside the noise band is a NULL, a cross-node denominator can invert a real win into a double-digit "regression", and an n=1 arm can invert a +6.5% win into a -5% rejection. Also stops a false PARITY failure: cross-arm output agreement is only interpretable against a same-arm across-restart control
confirms: 5
last_seen: 2026-08-29
---
# Honest e2e A/B: tight interleave + non-overlap gate (not just a positive median)
- lever: run a tight INTERLEAVED A/B (REF, CAND, REF, CAND, ...) with a PINNED port on the SAME GPU set in
  the SAME session, then gate on BOTH `delta_med > noise_band` AND non-overlapping distributions
  (`cand_min > ref_max`). The ~0.5% noise band is real: clean ref/cand medians overlap routinely, so a
  sub-band delta with overlapping [min,max] is a NULL. Cross-check the throughput delta against a
  steady-state decode metric (median ITL) - the two agreeing is much stronger than either alone.
- apply: >=5-7 repeats/leg (>=4 legs interleaved across restarts), back-to-back, same GPU. Combine an
  accepted-config stack and gate the SUM vs the TRUE baseline (small real wins only count when stacked).
- verify: sglang derives `grpc_port = port + 10000` and rejects >65535 -> an OS ephemeral port >55535
  crashes launch; ALWAYS pin PORT to a low value. Budget for grpc-port-flake retries.
- caution: also verify the DENOMINATOR came from the same node and session. ABSOLUTE throughput is a
  machine property: the same artifact reproduced +4.24%/+4.72%/+4.43% on three nodes whose absolute
  throughput differed by 17%, and had one node's number been used as the reference the same win would have
  printed as a -10.9% REGRESSION. Any speedup computed as `final_here / baseline_there` is meaningless -
  recompute it from the local pair. Also verify the FIRST leg after a container start is discarded or
  warmed: a one-shot JIT/compile stall (~0.75 s p99.9 ITL, seen on 5 independent nodes) inflates leg 1's
  latency, and aggregating all legs instead of comparing warm pairs overstated the same win by >2x
  (+10.3% aggregate vs +4.24% warm-pair).
- lever (restart-scale drift): when each leg needs its own cold server, interleave the ARMS across restarts
  (`pre, post, pre, post`) rather than running arm A's legs then arm B's — restart-scale drift is larger than
  many real wins. Report the per-arm cross-restart spread as the floor you beat, and say n.
- caution: the FIRST timed leg in a fresh container pays JIT/autotune of any newly-live seam. Its
  steady-state can be identical to a later same-arm leg (median ITL within 0.04%) while aggregate throughput
  reads ~11% lower, purely from one ~700 ms one-shot stall (p99.9 ITL ~750 ms vs ~186 ms elsewhere; 4
  independent reproductions). So (a) discard or explicitly exclude the first leg and say which direction the
  exclusion cuts, and (b) cross-check the throughput delta against a **stall-immune steady-state metric**
  (median ITL / TPOT) that keeps ALL legs — two metrics agreeing to <0.5 pp is much stronger than one.
  Note a cache-clear helper may drop only the framework compile cache and not `~/.triton/cache`.
- caution: NEVER use a throughput measured on another node/session as the A/B denominator. Nodes of the same
  SKU have been seen 17% apart in absolute throughput while agreeing to 0.5 pp on the DELTA — the same
  artifact read +4.2% against its own-session pre leg and would have read -10.9% against a foreign baseline.
  Every claim needs an own-session, same-node reference leg.
- caution: also verify n. **`n=1` reports `spread = 0.0`, which is the ABSENCE of a noise floor, not a tight
  one.** A config arm that ended up the run's biggest single lever measured -10.1% / -9.0% / -5.1% across
  three n=1 probes and **+6.49% at n=3**; three other n=1 arms were logged `UNRESOLVED_at_n1` rather than
  ranked, correctly. Never accept OR reject a config arm at n=1 — and when the budget forbids n=3, rank on
  median ITL, which is stall-insensitive and usable at n=1, instead of on aggregate throughput.
- caution: an OUTPUT-PARITY probe needs its own NEGATIVE CONTROL, or it manufactures a failure. A TP=N
  batched server is not reproducible across restarts even within ONE arm: the same-arm, different-server
  control agreed on only 2.33/8 and 3.00/8 whole greedy completions, while cross-arm agreed on 2.00/8 —
  i.e. INSIDE the control envelope, and at every prefix truncation from 8 to 1024 chars cross-arm matched
  or beat the same-arm control. Always run same-arm-across-restart first and report the cross-arm number
  against it; a bare "cross-arm differs" is free-running continuation drift after a greedy tie-break, not
  evidence about the change. Treat this probe as a blow-up detector and keep the load-bearing correctness
  evidence elsewhere (bitwise identity for a data-only edit; a sized accuracy gate for a backend swap).
- source: exp/e2e_*Qwen3.5-27B*/ 2026-06-07 / 06-09; exp/e2e_*DeepSeek-R1*/ 2026-08-28 (3-node
  cross-validation, 4-leg interleaved isolated-server A/B); exp/e2e_*DeepSeek-R1*_atom_*/ 2026-08-28
  (4-leg interleaved isolated-server A/B; first-leg JIT stall characterised on 3 nodes);
  exp/e2e_*DeepSeek-R1*_atom_*/ 2026-08-29 (Director validation, 3 cold-start replicas per arm,
  n=1-vs-n=3 config inversion, same-arm parity control).
