---
key: engagement verification · any gfx · any backend
type: method
confidence: ★★★
confirms: 4
effect: turns "did my kernel actually run live?" from a guess into proof
last_seen: 2026-10-01
---
# Prove the optimized kernel ran on the LIVE serving path (don't infer it from an e2e wiggle)

- path: instrument the candidate with a one-shot stderr banner and grep the server log. This PROVES
  engagement on both the bench and the parity leg, instead of inferring it from a throughput delta
  (which moves for unrelated reasons).
- apply: emit `[overlay-mark] <kernel> OPTIMIZED kernel CALLED` once from inside the candidate; for an
  overlay rebind also grep `[overlay] injected module <path>` (N hits = N workers) and
  `[OVERLAY_ENGAGED]`.
- verify: ≥1 banner per worker on the live run = engaged. Zero banners with a healthy server = the seam
  missed — wrong rebind target, or a self-capturing wrapper fell back to eager (see
  [[method-cudagraph-safe-integration]]). On cudagraph paths, verify engagement INSIDE the captured
  region, not just at module-injection time.
- caution: an in-place fast-path patch (a runtime-gated `apply()` that switches to a native path, rather
  than an overlay rebind) can be missing its banner and still compile and pass. There, exact-zero parity
  (`max_rel_err == 0.0`) against a LIVE baseline is itself the reliable tell of a silent fallback. Make
  the fail-closed engagement assert UNCONDITIONAL — an assert gated on an env var the verify harness
  never sets protects nothing (two rounds shipped a no-op that passed correctness).
- caution (closure-captured launcher): a dispatcher that binds its authored launcher by DIRECT reference
  at import time (`fn = _load_authored()` held in a closure) is NOT re-pointed by a marker/capture you
  install on the launcher module's attribute AFTERWARD — the closure already holds the original object, so
  the marker fires zero times (`installed_but_never_live`) and the OUTER wrapper is what looks "deepest":
  a FALSE-POSITIVE deepest-seam certification. Fix: install the capture + launcher marker via a
  POST-IMPORT hook on the authored module ITSELF, firing the instant it executes — BEFORE the dispatcher's
  `getattr`/`_load_authored` returns — so the live closure holds the wrapped fn and the marker fires on
  every call. Then certify with per-call marked-vs-device-kernel MATCH COUNTS (e.g. 32/32 + empty
  `deeper_live_candidates`/`installed_but_never_live`), never "a marker exists". Same class of bug as the
  silent-fallback caution above: the thing you instrumented is bypassed, so absence of a banner (or a
  marker on the wrong object) is not proof of absence of the kernel.
- source: exp/e2e_*Qwen3.5-27B*/ FLA overlay runs 2026-06-07 / 06-09; exp/e2e_*MXFP4*/ 2026-08-16
  (native-mxfp4 fast path fell back twice on a TP4 shard); exp/e2e_*Qwen3-14B-FP8*/ 2026-10-01
  (sglang fp8 a8w8 blockscale prefill GEMM deepest-seam capture: closure-captured launcher bypassed a
  post-hoc marker → false-positive deepest_verified on the fp8_utils wrapper; post-import hook fixed it).
