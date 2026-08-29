# Backend Playbook — Which Backend Suits Which Kernel (persistent experience)

This is the **experience library** the System Architect owns. It maps a kernel CLASS to a ranked list
of backends worth trying, and it GROWS: after every run, append confirmed POSITIVE methods/routing to the
"Learned" section (with model, shape regime, and the measured result). Seeded from MI300X (gfx942)
experience; treat the seed as priors, not gospel — the unittest is the judge.

## Backend menu (what each is good at on MI300X)
- **aiter** — AMD's fused op library (GEMM, rmsnorm+quant, MoE, some attention). **On sglang/gfx942 it
  IS the live dense-GEMM dispatcher** (`tuned_gemm.py` → hipBLASLt `Cijk_*`/asm/triton/skinny). Tune
  its per-shape DB (`bf16_tuned_gemm.csv`) — this is THE GEMM lever (see `gemm_tuning/aiter_gemm_tuning.md`).
  Also fuses norm+quant; often wins skinny/decode GEMM.
- **hipBLASLt / Tensile** — the kernels aiter actually executes for dense GEMM. NOT separately tunable
  via `HIPBLASLT_TUNING_FILE` on this stack (aiter bypasses the PyTorch/hipBLASLt C dispatch for its
  tuned shapes). The "not found tuned config" warnings = aiter shapes you haven't tuned yet = target list.
- **CK / ck_tile (Composable Kernel)** — attention (FmhaBatchPrefill/paged), some GEMM. Best paged
  attention on MI300X today; tunable via instance selection.
- **Triton** — custom/novel kernels (mamba/gated-delta linear attn, fused norms, activations,
  bespoke fusions). Fastest to iterate; good for memory-bound and fusion. The kernel squad's home.
- **HIP / raw** — when you need warp-cooperative control Triton can't express, or to hand-fuse.
- **asm (MFMA intrinsics / hand asm)** — last 10–20% on a proven-hot compute-bound kernel; high cost,
  only for a kernel that is large pct_gpu_time and already backend-chosen.

## Class → ranked backends (priors)
| kernel class | try in this order | notes |
|---|---|---|
| dense GEMM (prefill, large M) | **tune aiter `bf16_tuned_gemm.csv`** (capture→gradlib→`AITER_CONFIG_GEMM_BF16`) | confirmed +1.22% (partial) on hybrid-dense gfx942; NOT TunableOp/HIPBLASLT_TUNING_FILE — see `gemm_tuning/aiter_gemm_tuning.md` |
| skinny GEMM (decode, M=batch) | tune aiter DB (decode M-buckets) → skinny default | aiter dispatches skinny kernels; tune M=16/32/48/64 buckets |
| paged/prefill attention | CK(ck_tile) → aiter → triton FA | `--attention-backend` swap is free to try |
| mamba / gated-delta linear attn | triton (tune) → HIP | almost always Triton; tune tiling/scan |
| rmsnorm (+quant/residual) | aiter fused → triton fused | fuse with neighbor add/quant |
| rope / qk-norm | triton fused → aiter | fold into attention pre-step |
| activation (silu/gelu + mul) | fused act_and_mul (aiter/triton) | collapse into the producing GEMM epilogue if possible |
| elementwise/fill/cast/copy | fuse away (host_runtime) / cuda-graph | usually shouldn't be its own kernel |

## Roofline prior calibration (predicted vs measured — one line per direction)
- 2026-08-19 · gfx950 vLLM mxfp4 grouped fused-MoE (`_matmul_ogs...swiglu`, gpt-oss-120b, decode): roofline
  predicted `attainable_speedup=1.0`, `expected_e2e_gain_pct=0.0` (memory-bound, `roofline_pct` 0.95–1.0,
  headroom `saturated`, confidence **low**). MEASURED e2e **+26.9%** via a whole-file Triton rewrite. The
  device-time byte/FLOP roofline model was WRONG for this seam — it cannot see the win, which is
  host launch-overhead / decode-seam collapse, not a byte reduction. Correct behavior held (confidence was
  low → not ranked on, head not dropped). Lesson: for a fused MoE dispatcher at decode, do NOT trust a
  `saturated`/`attainable=1.0` roofline verdict to size the opportunity; the launch-overhead win is invisible
  to it. This is the exact "measured EXCEEDS predicted attainable" failure mode — flag loudly.
- 2026-08-28 · gfx950 ATOM DeepSeek-R1 fp8 block-scale, TP8, conc=4 (LATENCY-bound decode): roofline
  round_0 predicted the dense-projection unit at `attainable_speedup=7.05` / `expected_e2e_gain_pct=+14.74`.
  MEASURED +6.49% e2e from the config swap that replaced it. Direction and TARGET both correct, magnitude
  **over-predicted ~2.3x**: `attainable_speedup` assumes the whole gap to `target_eff` is recoverable, but at
  conc=4 on TP=8 most of that gap is occupancy/launch latency a backend swap cannot recover. Calibration to
  carry forward: on a low-concurrency latency-bound decode regime, **DISCOUNT `expected_e2e_gain_pct` by
  ~2-3x before budgeting a target** (order with it, never size with it).
- 2026-08-28 · same run, whole-step sanity: ~13.1 GB of weights+KV per decode step per rank would take
  1.64 ms at 8 TB/s against a measured 10.12 ms step = ~16% of the memory roofline. When EVERY modelled head
  comes back `underperforming` on a roof the step as a whole is nowhere near, the honest read is "the regime
  is latency/occupancy bound", not "every kernel has 3-5x of headroom" — and CONFIG levers (collective
  cost, dispatch-floor launches) stay competitive with any single kernel rewrite.
- 2026-08-28 · same run, kernel lane: `attainable ~4-6x` predicted for a 1-wave/CU skinny bf16 GEMV
  (`wv_splitk_small`, 2.7% of peak BW) and for a dispatch-floor fp8 group-quant (~100x above its own
  roofline time). Both measured **no isolated speedup within the round's budget** — a latency/occupancy
  headroom estimate is NOT a speedup estimate; it says the time is theoretically recoverable, not that a
  rewrite recovers it. Treat sub-5% latency-bound kernels as low priority even when their roofline gap is
  enormous.

- 2026-08-28 · same run, TUNING lane (round_config prior): roofline predicted the decode dense-projection
  unit (`o_proj + q_b + qkv_a`, 11.82% GPU) at `attainable_speedup=4.959` / `expected_e2e_gain_pct=+10.46`
  (bound_type `latency`, headroom `underperforming`, confidence medium). MEASURED: per-shape M-bucket tune
  gave iso **1.30-1.51x** on a 13.70% head and **+4.240% e2e** (+4.586% decode rate on median ITL). Prior
  over-predicted ~3.3x on isolated and ~2.5x on e2e — a SECOND confirm of the "discount latency-bound
  `expected_e2e_gain_pct` by 2-3x" rule above. Ordering was right (this unit was the correct target); sizing
  was not. Use it to RANK, never to budget.
- 2026-08-28 · same run, round-1 kernel lane, the failure mode the roofline cannot see: `fused_kv_bmm`
  launch-collapse (batched MLA-absorb GEMM + `fuse_qk_rope_concat_and_cache_mla`) and
  `opus_moe_sorting_entry + grouped_topk_opt_sort` both carried large latency headroom, and both died at
  **EXTRACTION** (aiter prebuilt ASM / C++-extension seam, non-editable) rather than at measurement. Where
  the collapse did build, it measured **1.03x at M=4 and 0.96x at M=1** (weighted 1.031, geomean 0.996) —
  i.e. the modelled headroom was real but not reachable by that rewrite. Lesson: a latency-bound
  `underperforming` verdict says the time is theoretically recoverable, NOT that a seam exists to recover it.
  Run a **seam-editability probe before budgeting** any latency-headroom head on an aiter/ATOM-class stack;
  the reachable lever on prebuilt-kernel heads is the config/tuning table the kernel itself reads.

- 2026-08-28 · same run, round-1 kernel lane, CALIBRATION LIMIT: for the two directions that did build
  (`dynamic_per_group_scaled_quant` 4.7% GPU / 5.5 us per call, `wv_splitk_small` bf16 GEMV 3.5% / 9.0 us)
  predicted-vs-actual is UNDEFINED, not "prior was wrong": the isolated graph-replay harness has a ~17 us
  per-op floor on this box (baseline flat at 16.7-17.1 us across a 14x byte span; 2-3x the traced us/call),
  so both candidates and their baselines land on the floor and every result reads 0.98-1.03x with wider
  spread. Roofline `attainable 4-6x` was neither confirmed nor refuted. **Do not feed such a run back into
  the prior as a measured no-win** — record it as unresolvable and screen for the floor (baseline-only
  size sweep) before budgeting any sub-20-us decode kernel. See
  `learned/method-decode-microkernel-measurement-floor.md`.

- 2026-08-29 · same run, FINAL-GATE reconciliation — **the calibration correction that matters most.** The
  two "prior over-predicted 2-3x" entries above compared ONE lever against a prediction made for a WHOLE
  logical unit. Stack every lever that actually targets that unit and the prior looks very different:
  round_0 predicted the dense-projection/dense-GEMM unit at `expected_e2e_gain_pct = +14.74`; the two
  accepted levers on that unit (backend-select env +6.49% × M-bucket table tune +4.24%) Director-validated
  at **+11.689% e2e** (332.242 → 371.078 tok/s, 1.1169×, n=3/arm non-overlapping, TPOT −10.69%, parity pass).
  Prediction was only **~1.26x optimistic** against the FULL lever set, not 2-3x. Carry forward: apply the
  "discount 2-3x" rule when budgeting a SINGLE lever, but do NOT let it talk you out of a target — the
  roofline `expected_e2e_gain_pct` sized the unit's TOTAL recoverable time about right.
- 2026-08-29 · same run: that round_0 entry carried `confidence: low` (derived peaks), so by doctrine it was
  display-and-annotate only — and it was nonetheless the correct headline target, the only one of the run to
  pay. A `low`-confidence roofline row still must not be RANKED on, but it must not be dropped either; here
  the `pct_gpu_time` order and the roofline order agreed, which is the case where the low-confidence row is
  safe to act on.
- 2026-08-29 · same run, the two heads roofline ranked ABOVE it: fused MoE (25.2% GPU, `attainable 2.427`,
  `+16.43` expected, confidence medium) and MLA decode (6.4%, `3.292`, `+4.98`) both went unrealized — not
  refuted, but blocked at the SEAM (prebuilt asm / C++-extension, and a build whose `kernelName` dispatch
  could not bind a CK tune table). Predicted-vs-actual is UNDEFINED for both. The standing lesson: pair every
  roofline ranking with a seam-editability + tune-hook probe before budgeting, because the roofline model
  cannot see whether a lever exists.

## How to use this in a run
1. Architect reads the Profiler Top-N classification + shapes.
2. For `library_*` kernels → hand to Config Tuner with the ranked swaps above (no source edit).
3. For editable kernels → hand to Extractor + kernel squad; pass the ranked backends as the
   squad's "candidate backends" so it compares them via the (immutable) unittest.
4. **CURATE** `knowledge/learned/` after the run (read INDEX → merge/insert ≥★★ / archive
   contradicted), per `knowledge/learned/README.md`.

## Learned experience → `knowledge/learned/`
Confirmed routing/method findings are NOT appended here anymore. They live as distilled, evidence-cited
cards in **`knowledge/learned/`**, read via **`knowledge/learned/INDEX.md`** (grouped by reuse key
`kernel_class · gfx`). Open only the cards matching the current run's `(model_class, gfx, regime)`;
rank by `EV = Amdahl_ceiling × confidence`; honor each card's `dead-end:` lines.
