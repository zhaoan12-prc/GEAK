---
key: kernel fusion (RoPE + cat + fp8 KV write) · gfx942 · sglang MLA fp8 decode
type: lever
confidence: ★★★
effect: +7.302% e2e output throughput marginal on top of an already-fused stack, non-overlapping (base_max 171.733 < cand_min 182.980), TPOT -7.057%, gsm8k 0.935 -> 0.940; lands TWO exec_ids with one overlay
confirms: 1
last_seen: 2026-08-30
---
# A caller-side arch gate is a DISPATCH decision, not a capability statement
- lever: sglang gates `fused_qk_rope_cat_and_cache_mla` behind `_use_aiter_gfx95`, so on gfx942 the
  stock path is the SPLIT form: RoPE gather + `torch.cat` + a separate fp8 KV-cache write. The kernel
  itself runs fine on gfx942. Flipping the flag inside a sitecustomize overlay (and injecting the
  symbol, which is imported only under that same `if`) collapses four launches per layer into one.
  Trace evidence: `CatArrayBatchedCopy` 44->0, `kn_entry_2c_sbhd_cached_indirect_inplace` 23->0,
  `index_elementwise_kernel` 22->0, `_fused_qk_rope_cat_and_cache_mla_kernel` 0->44.
- THE GENERAL LESSON, and it is worth more than this one kernel: a `_use_aiter_gfx95` /
  `dtype == float8_e4m3fn` / "no call site in sglang" gate is the vendor's DISPATCH policy for the
  shipped path. It is NOT evidence the kernel cannot run here. An apply-back overlay replaces the seam
  directly and BYPASSES the dispatcher, so the gate never participates. Marking a candidate `blocked`
  because a caller-side gate is False is a MISJUDGEMENT — it was made in the 2026-08-29 v1 run for four
  candidates, and on retry three of the four turned out to be real, non-overlapping e2e wins
  (+7.302%, +10.340%, +7.328%). Always ask "would the overlay even reach this gate?" before blocking.
  Note the flip side: on this box `dtype == torch.float8_e4m3fn` is *permanently* False because the
  ROCm weight loader (`deepseek_weight_loader.py`) runs `normalize_e4m3fn_to_e4m3fnuz` — so supply the
  fnuz variant in the overlay rather than expecting the fn branch to ever fire.
- apply: the overlay must supply what the gate would have supplied. Two concrete traps here:
  (1) the gfx950 call site passes `q_out_dtype=kv_cache_dtype` (fp8 q), but the gfx942 stock arm builds
  q as a BF16 `torch.cat`; force `torch.bfloat16` or an unrelated dtype change confounds the A/B.
  (2) flipping the flag also enables `_skip_rope_for_aiter_fused_mla`. Wrap it and RAISE if it returns
  False — that would mean upstream RoPE is still running and the model silently double-RoPEs.
  Audit every reference to a flag before flipping it (11 here).
- verify: `PATH fused` per-rank counts with 0 fallback, not just the ENGAGED banner.
- source: /raid/users/zhaoan/fusion_kernel_result/20260829_e2e_v2/dsr1 (05_FUSION_APPLYBACK.md; leg c05;
  overlay fusion/fusion_overlays/dsr1/e05_rope_kv_write)
