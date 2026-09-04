# probes/

Small, numerically-inert instruments used to ANSWER a question about a runtime, not to
change it. They are not part of any workflow phase; a role invokes one when it needs
evidence that a contract actually holds on this stack.

## geak_engage_sentinel.py

Answers "did my overlay actually run?" — the question `fusion_integrator.md`'s
`[overlay-…] ENGAGED` banner only appears to answer.

It rebinds a seam to a wrapper that adds a uniquely-named `torch.profiler.record_function`
and calls straight through, so the run is numerically identical. Reprofile, then grep the
trace for `GEAK_SENTINEL_<tag>`. Present with GPU annotations under it = the seam executes.
Absent = the rebind was inert, whatever the banner said.

Wire it with the overlay hook mechanism and arm it with `GEAK_SENTINEL=1`:

```bash
python3 scripts/overlay_setup.py add-hook --overlay "$OV" \
  --module vllm.model_executor.layers.utils \
  --impl-module geak_engage_sentinel --impl-attr install \
  --impl-file scripts/probes/geak_engage_sentinel.py
# then launch with OVERLAY_PYTHONPATH="$OV", EXTRA_ENV="GEAK_SENTINEL=1 VLLM_DISABLE_COMPILE_CACHE=1"
```

Edit the seam list in `install()` for the seam you are about to fuse. Running it on a
candidate seam BEFORE authoring the real fusion is cheap and tells you whether the seam is
reachable at all — see the measured table in `roles/fusion_integrator.md`, where the banner
reported ENGAGED for a seam that never executed.
