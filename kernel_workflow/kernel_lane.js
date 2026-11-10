export const meta = {
  name: 'kernel-lane',
  description: 'SINGLE-LANGUAGE kernel optimization worker (Director/TechLead/specialist Engineers) with budget-controlled rounds, independent verification, and integration. Optimizes ONE kernel in ONE language (mode=optimize) or authors a fresh seed then optimizes it (mode=author); mode=mega searches a multi-rank operator (e.g. an EP MoE layer) as independent whole-kernel candidates, each verified on-card against the frozen baseline (validated on EP MoE only). This is the worker invoked per lane by the kernel-workflow dispatcher (kernel_workflow.js) and by e2e_workflow; prefer calling kernel-workflow directly unless you specifically want one unchanged lane. Target: AMD Instinct MI-series (CDNA gfx942/gfx950) and the validated RDNA4 product Radeon AI PRO R9700 (gfx1201) — auto-detected on-box; on R9700 use knowledge/amd_rdna4.md (wave32, WMMA). Generic gfx1201 without R9700 identity is not a calibrated target.',
  whenToUse: 'Internal single-language worker. Prefer the kernel-workflow dispatcher (kernel_workflow.js) as the entry point; invoke this directly only to run one unchanged lane. Pass args.kernel_path (required), args.mode, args.target_language, args.budget, args.gpu_ids, args.gpu_mode, args.gpus_per_job, args.job_gpu_ids, args.task; mode=mega also takes its mega_* / guard / control args (see README "Mega mode").',
  phases: [
    { title: 'Setup', detail: 'director builds the isolated eval dir + canonical workspace' },
    { title: 'Author', detail: 'author_engineer writes a fresh optimize-loop seed (only when mode=author); speedup denominator stays the frozen online kernel' },
    { title: 'Analyze', detail: 'tech_lead analyzes kernel + writes roadmap' },
    { title: 'Benchmark', detail: 'benchmark_engineer builds the COMMANDMENT + baseline' },
    { title: 'Profile', detail: 'profile_engineer classifies the bottleneck; mode=mega: optional analysis_engineer validates the evidence' },
    { title: 'Research', detail: 'OPT-IN (args.dra_enabled): researcher fans research questions out in parallel via native WebSearch/WebFetch, writes a ranked-directions brief the planner seeds from' },
    { title: 'WarmStart', detail: 'search the experience KB by canonical id (geak:kernel:*) — the shared service when credentialed, else the on-disk store, brought level with kb_artifacts/ first — for the best curated patch per optimization direction for this (kernel,language,gfx), validate each through the verify gate, adopt the first that passes [warm_start!=off]' },
    { title: 'Optimize', detail: 'budget loop: tech_lead plans, specialist OR deep_explore engineers optimize, reprofile; mode=mega: mega_search_lead picks one whole-kernel candidate per round, an engineer authors it in its own tree' },
    { title: 'Verify', detail: 'each candidate patch independently re-benchmarked' },
    { title: 'Merge', detail: 'integrator combines the round winners' },
    { title: 'Report', detail: 'tech_lead writes the final report + patch' },
    { title: 'Validate', detail: 'director independently validates vs the true baseline, then the TechLead curates ONE distilled card into knowledge/learned/ on a measured win [update_experience!=off]' },
  ],
};

// ---------------------------------------------------------------------------
// Args + defaults. (The script cannot touch the filesystem or read its own
// path; agents do all FS work, and every path is supplied/derived from args —
// nothing about the install location is hard-coded.)
// ---------------------------------------------------------------------------
const A = args || {};
if (!A.kernel_path) throw new Error('args.kernel_path is required (absolute path to the kernel/model directory)');

// WORKFLOW_DIR = the directory that holds this script + roles/ + knowledge/ + scripts/.
// A JS workflow script can't read its own path, so the caller passes it (it is just the
// dirname of the scriptPath used to launch the workflow).
const WORKFLOW_DIR = String(A.workflow_dir || '').replace(/\/+$/, '');
if (!WORKFLOW_DIR) {
  throw new Error('args.workflow_dir is required: absolute path to the directory containing ' +
    'kernel_workflow.js, roles/, knowledge/, scripts/ (i.e. the dirname of this script).');
}
// EXP_ROOT = where timestamped run dirs are written. Default: a sibling "exp/" next to the
// kernel_workflow dir (…/<parent>/kernel_workflow -> …/<parent>/exp). Override with args.exp_root.
const EXP_ROOT = String(A.exp_root || (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/exp')).replace(/\/+$/, '');

const KERNEL_PATH_ORIG = A.kernel_path;
const EXPECTED_GFX = String(A.expected_gfx || '').trim().toLowerCase();
const EXPECTED_TARGET = String(A.expected_target || '').trim().toLowerCase();
const EXPECTED_DEVICE_NAME = String(A.expected_device_name || '').trim();
const EXPECTED_PHYSICAL_CU_COUNT = Number(A.expected_physical_cu_count || 0);
if (['r9700', 'r9600d'].includes(EXPECTED_TARGET) && EXPECTED_GFX !== 'gfx1201') {
  throw new Error(`expected_target=${EXPECTED_TARGET} requires expected_gfx=gfx1201`);
}
if (EXPECTED_GFX === 'gfx1200' || EXPECTED_TARGET === 'gfx1200') {
  throw new Error('gfx1200 is not supported: this workflow is hardware-validated only on R9700 / gfx1201');
}
const BUDGET = parseInt(A.budget != null ? A.budget : 6, 10);
// Minimum verified geomean improvement over the cumulative best for a round winner to be COMMITTED
// into the canonical workspace (default 2%). Kept as a knob rather than a hard-coded constant so the
// gate is tunable per run (e.g. raise it on a noisy box, lower it to bank small compounding wins).
const MIN_IMPROVE = (() => {
  const v = parseFloat(A.min_improve != null ? A.min_improve : 0.02);
  return Number.isFinite(v) && v >= 0 ? v : 0.02;
})();
// Minimum verified speedup for a candidate to enter the round's candidate list (default 1.0 = only a
// candidate that beats the baseline is worth looking at). A knob for the same reason MIN_IMPROVE is:
// a transcription (plain Triton -> Gluon/TileLang/HIP) lands BELOW the comparator by construction, and
// at 1.0 its recovery phase is invisible -- no patch saved, no verify, `winner` null every round. The
// COMMIT gate is separate and still requires beating `cumulative` by MIN_IMPROVE, so a sub-baseline
// candidate can be TRACKED but never BANKED.
const CANDIDATE_FLOOR = (() => {
  const v = parseFloat(A.candidate_floor != null ? A.candidate_floor : 1.0);
  return Number.isFinite(v) && v > 0 ? v : 1.0;
})();
// Rendered into the Optimize prompt, where `${1.0}` would stringify to "1" and silently reword a
// prompt that has always said "geomean>1.0". Keeps the default run byte-identical.
const CANDIDATE_FLOOR_TXT = Number.isInteger(CANDIDATE_FLOOR)
  ? CANDIDATE_FLOOR.toFixed(1) : String(CANDIDATE_FLOOR);
// How far BELOW the best candidate ever seen a round may land and still count as "the search is
// advancing" (default +MIN_IMPROVE = the historical test). A knob because a climb that starts under
// the comparator advances for many rounds without ever clearing `cumulative`, and scoring those as
// stalls ends it at MAX_NO_IMPROVE however much budget was given -- and no static counter substitutes,
// since it would have to pre-guess how many rounds the climb takes. Negative admits a round that gives
// ground (a layout experiment that does not pay is information, not a stall).
const PROGRESS_DELTA = (() => {
  const v = parseFloat(A.progress_delta != null ? A.progress_delta : MIN_IMPROVE);
  return Number.isFinite(v) && v > -1 ? v : MIN_IMPROVE;
})();
// Budget cost of ONE `deep_explore` direction. The deep-explore engineer does far more than a single
// specialist — broad rewrite authority, its own multi-iteration measure→profile→rewrite loop — so it
// is charged more than 1 against the direction budget (default 2). It also always runs in a DEDICATED
// round (no other directions that round), enforced below.
const DEEP_COST = (() => {
  const v = parseInt(A.deep_cost != null ? A.deep_cost : 2, 10);
  return Number.isFinite(v) && v >= 1 ? v : 2;
})();
const GPU_IDS = String(A.gpu_ids != null ? A.gpu_ids : '0');
const GPU_LIST = GPU_IDS.split(',').map(s => s.trim()).filter(Boolean);
// Every GPU consumer gets the WHOLE pool rather than a pinned lane. gpu_lock.sh
// resolves a comma spec by flocking whichever lane is free AND idle at acquire
// time, so placement follows what work actually costs instead of an index fixed
// before the cost is known. Measured task costs span 23x on campaign20, and a
// pinned round ends with the slowest LANE, not the slowest task -- at 4-way that
// idles ~43% of lane time. Pinning is also fragile on a shared box: GPU_LIST[0]
// has no fallback when lane 0 has a foreign tenant.
// gpu_mode='pin' restores the pre-scheduler behavior (direction i pinned to GPU_LIST[i % n]).
// It exists ONLY so the scheduler can be A/B'd as a single-variable change against the arm it
// replaced; leaving it out would make the "before" arm unreachable and force the comparison to be
// made across two different scripts, where any other drift would be indistinguishable from the
// scheduler's effect. Default 'pool'.
const GPU_MODE = String(A.gpu_mode || 'pool') === 'pin' ? 'pin' : 'pool';
// Multi-GPU jobs (multi-rank ops). gpus_per_job > 1 leases any N GPUs of gpu_ids per command
// (gpu_lock.sh pool:N:<ids>); job_gpu_ids pins one fixed group (group:<ids>). Either may also come
// from op_spec.resource. Without any of them the count is what Setup found in the task's own
// test/benchmark (`detected`, see scripts/detect_gpu_requirement.py). Every GPU consumer then gets
// that one spec, so each command holds the whole group. One GPU per job leaves everything unchanged.
// Setup's finding is a static hint: one that cannot run in gpu_ids (unknown, no usable count, more
// GPUs than the pool) keeps the lane single-GPU with a warning. Only an explicit request throws.
function resolveGpuRequest(A, gpuList, detected) {
  const resource = (A.op_spec && A.op_spec.resource) || {};
  const parseIds = (name, value) => {
    const parts = String(value).split(',').map(s => s.trim()).filter(Boolean);
    if (!parts.length) throw new Error(`${name} must contain at least one GPU id`);
    if (parts.some(id => !/^\d+$/.test(id))) throw new Error(`${name} contains a non-numeric GPU id`);
    const ids = parts.map(id => String(Number(id)));
    if (new Set(ids).size !== ids.length) throw new Error(`${name} contains duplicate GPU ids`);
    return ids;
  };
  const pool = parseIds('gpu_ids', gpuList.join(','));
  const fixedValue = String(A.job_gpu_ids || '').trim() || resource.job_gpu_ids;
  const fixedIds = String(fixedValue || '').trim() ? parseIds('job_gpu_ids', fixedValue) : [];
  const explicit = A.gpus_per_job != null || resource.gpus_per_job != null || fixedIds.length > 0;
  const found = (!explicit && detected) || {};
  let foundCount = found.kind === 'scalable' ? pool.length : 1;
  let problem = '';
  if (found.kind === 'unknown') {
    problem = 'Setup found multi-GPU code in this task but could not tell how many GPUs one run needs';
  } else if (found.kind === 'fixed') {
    const n = Number(found.gpus_per_job);
    if (!Number.isSafeInteger(n) || n < 1) {
      problem = `Setup reported no usable GPU count (${found.gpus_per_job})`;
    } else if (n > pool.length) {
      problem = `Setup found ${n} GPUs per run but gpu_ids holds ${pool.length}`;
    } else {
      foundCount = n;
    }
  }
  const warning = problem ? `${problem} (${found.evidence || 'no evidence'}); running single-GPU. ` +
    'Pass gpus_per_job=<N> (or job_gpu_ids) with enough gpu_ids to run it on N GPUs.' : '';
  const countValue = A.gpus_per_job != null ? A.gpus_per_job
    : (resource.gpus_per_job != null ? resource.gpus_per_job : (fixedIds.length || foundCount));
  const countText = String(countValue).trim();
  if (!/^[1-9]\d*$/.test(countText)) throw new Error('gpus_per_job must be a positive integer');
  const gpusPerJob = Number(countText);
  if (!Number.isSafeInteger(gpusPerJob)) throw new Error('gpus_per_job is too large');
  if (fixedIds.length && fixedIds.length !== gpusPerJob) {
    throw new Error(`gpus_per_job=${gpusPerJob} does not match job_gpu_ids size ${fixedIds.length}`);
  }
  if (!fixedIds.length && gpusPerJob > pool.length) {
    throw new Error(`gpus_per_job=${gpusPerJob} exceeds gpu_ids pool size ${pool.length}; ` +
      `pass gpu_ids with at least ${gpusPerJob} GPUs`);
  }
  if (fixedIds.some(id => !pool.includes(id))) throw new Error('job_gpu_ids must be a subset of gpu_ids');
  const jobSpec = fixedIds.length ? `group:${fixedIds.join(',')}`
    : (gpusPerJob > 1 ? `pool:${gpusPerJob}:${pool.join(',')}` : '');
  return { gpusPerJob, fixedIds, jobSpec, explicit, warning };
}
// Validated now so a bad request fails before Setup; re-resolved from Setup's finding when implicit.
let GPU_RESOURCE = null;
let GPU_JOB = '';
let GPU_POOL = '';
let GPU_SINGLE = '';
function applyGpuResource(r) {
  GPU_RESOURCE = r;
  GPU_JOB = r.jobSpec;
  GPU_POOL = GPU_JOB || (GPU_MODE === 'pin' ? GPU_LIST[0] : GPU_LIST.join(','));
  GPU_SINGLE = GPU_JOB || GPU_LIST[0];
}
applyGpuResource(resolveGpuRequest(A, GPU_LIST));
const TASK = A.task || '';
const EVAL_DIR_OVERRIDE = A.eval_dir || '';
const APPLY_TO_ORIGINAL = String(A.apply_to_original != null ? A.apply_to_original : 'false');
const KERNEL_NAME_HINT = KERNEL_PATH_ORIG.replace(/\/+$/, '').split('/').pop();

// --- author mode: when there is NO existing source, write a fresh from-scratch SEED first, then optimize it.
// mode=optimize (default) keeps the exact original behavior (backward compatible). mode=author seeds
// the workspace from an op task dir (immutable oracle + frozen online kernel in baseline_src/), the
// author_engineer writes a passing seed, then the SAME optimize loop runs — always timing against the
// frozen online kernel, never against the seed's own language. KERNEL_KNOWLEDGE_DIR is the AMD authoring
// knowledge base — REFERENCE ONLY (facts/how-to, never decisions; the author always measures regardless). Default:
// sibling perf_knowledge/ so standalone runs use it too; empty if WORKFLOW_DIR is unset (no behavior change).
const MODE = String(A.mode != null ? A.mode : 'optimize').trim() || 'optimize';
const MEGA = MODE === 'mega';   // multi-rank candidate search: see the mode=mega backend section
const TARGET_LANGUAGE = String(A.target_language != null ? A.target_language : 'triton').trim() || 'triton';
const OP_SPEC = A.op_spec || {};
// When the op will run on the CUDA/HIP-graph-captured decode path (e2e sets op_spec.cuda_graph_safe=true),
// the isolated oracle alone CANNOT catch a kernel that passes iso but host-syncs or lazily-compiles under
// graph capture — the "wins isolated, crashes serving" class (cuda_graph_capture_unsafe / NO_BINARY_FOR_GPU).
// This turns on an OPTIONAL capture+replay smoke in the verify step so that failure is caught at the cheap
// isolated stage. Unset (standalone single-kernel runs / non-graph ops) => byte-identical to before.
const REQUIRE_GRAPH_CAPTURE = !!(OP_SPEC && OP_SPEC.cuda_graph_safe === true);
// WORKLOAD ALIGNMENT (optional). When the caller supplies the real-workload shape/dtype case
// distribution, the benchmark harness benchmarks EXACTLY those (shape, dtype) cases, weights each
// by its total time contribution in the workload (weight = count * baseline_latency), and the
// optimization target becomes the time-weighted ratio-of-sums instead of an unweighted geomean.
//   workload_spec_path : path to a workload-v1 json (produced by parse_profile.py --workload-out,
//                        or hand-written). The benchmark_engineer reads it (JS can't touch FS).
//   op_spec.workload   : inline cases, same shape as a workload-v1 "kernels[].cases" list (or the
//                        full object). Takes precedence; weight_source becomes "caller".
// Both unset => unweighted behavior, byte-identical to before. Correctness ALWAYS stays on the
// frozen immutable oracle (parity vs baseline_src/, + a recorded reference_io.pt where the task dir
// came from e2e's kernel_extractor); this only shapes the PERFORMANCE measurement.
const WORKLOAD_SPEC_PATH = String(A.workload_spec_path || (OP_SPEC && OP_SPEC.workload_path) || '').trim();
const WORKLOAD_SPEC = (OP_SPEC && OP_SPEC.workload) || A.workload || null;
const HAS_WORKLOAD = !!(WORKLOAD_SPEC_PATH ||
  (Array.isArray(WORKLOAD_SPEC) && WORKLOAD_SPEC.length) ||
  (WORKLOAD_SPEC && Array.isArray(WORKLOAD_SPEC.kernels) && WORKLOAD_SPEC.kernels.length));
// PRIMARY-metric selector: prefer the time-weighted number when a workload spec is in play and the
// agent reported one; otherwise fall back to the geomean (unweighted runs => unchanged behavior).
const primSpeedup = (o) => {
  if (!o) return 0;
  const w = o.verified_weighted != null ? o.verified_weighted
          : (o.speedup_weighted != null ? o.speedup_weighted : null);
  if (HAS_WORKLOAD && Number.isFinite(w)) return w;
  const g = o.verified_geomean != null ? o.verified_geomean : o.speedup_geomean;
  return Number.isFinite(g) ? g : 0;
};
// Gate fields like `correctness` / `status` are free strings in the schemas, so an agent legitimately
// answers "PASS - 15/15 draws" and a `=== 'pass'` test silently drops a genuinely verified candidate.
// Match the leading word instead: "PASS - ..." / "passed" gate open, "FAIL"/"did not pass" stay shut.
const says = (v, w) => String(v == null ? '' : v).trim().toLowerCase().startsWith(w);
let KERNEL_KNOWLEDGE_DIR = String(A.perf_knowledge_dir ||
  (WORKFLOW_DIR ? WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/perf_knowledge' : '')).replace(/\/+$/, '');
// Expert skills = human-authored, validated kernel recipes (perf_knowledge/expert_skills/). ADVISORY
// priors only: a matched `validated` skill is a HIGH-PRIOR author/optimize candidate the planning/author
// roles reproduce, then gate by the isolated A/B vs the oracle — it NEVER overrides measurement. Default
// OFF (opt-in: pass use_expert_skills="true"). When OFF (the default) NOTHING is injected -> byte-identical
// to a build without this feature. When invoked by the e2e layer the flag + dir are passed down.
const USE_EXPERT_SKILLS_REQUESTED = String(A.use_expert_skills != null ? A.use_expert_skills : 'false') === 'true';
let USE_EXPERT_SKILLS = USE_EXPERT_SKILLS_REQUESTED;
const EXPERT_SKILLS_DIR = String(A.expert_skills_dir ||
  (KERNEL_KNOWLEDGE_DIR ? KERNEL_KNOWLEDGE_DIR + '/expert_skills' : '')).replace(/\/+$/, '');
// Only planning + authoring roles consult skills; every other role gets no injection.
const EXPERT_SKILL_ROLES = new Set(['tech_lead', 'author_engineer', 'engineer', 'deep_engineer']);
// Expected-identity isolation must apply before Setup (no CDNA skills/KB on a declared R9700 run).
// Detected identity is OR'd in after director setup — do not read GFX here (TDZ).
let SKILLS_ISOLATE = EXPECTED_GFX === 'gfx1201';

// --- Deep Research Agent (DRA) -------------------------------------------------------------------
// OPT-IN: a v4-native research phase that runs AFTER Profile and BEFORE the optimize loop (so the
// COMMANDMENT + baseline profile + analysis exist). The `researcher` persona extracts facts and a
// ranked set of research QUESTIONS; the script fans those out in PARALLEL (each question = one
// hang-guarded agent using native WebSearch/WebFetch), then a synthesis pass writes a ranked
// directions portfolio (deep_search.md / deep_search_brief.md / deep_search.json) into EVAL_DIR that
// the TechLead's plan_round seeds from. DEFAULT OFF: when dra_enabled is not "true" NOTHING runs and
// behavior is byte-identical to a build without this feature (existing runs unchanged).
const DRA_ENABLED = String(A.dra_enabled != null ? A.dra_enabled : 'false') === 'true';
const DRA_MAX_QUESTIONS = (() => {
  const v = parseInt(A.dra_max_questions != null ? A.dra_max_questions : 8, 10);
  return Number.isFinite(v) && v >= 1 ? v : 8;
})();
// Optional Stage 5/6 blindspot-critique pass (one more parallel research wave). Default OFF (budget).
const DRA_BLINDSPOT = String(A.dra_blindspot != null ? A.dra_blindspot : 'false') === 'true';
const DRA_MAX_BLINDSPOTS = (() => {
  const v = parseInt(A.dra_max_blindspots != null ? A.dra_max_blindspots : 4, 10);
  return Number.isFinite(v) && v >= 1 ? v : 4;
})();

// ---------------------------------------------------------------------------
// LEARNED-KNOWLEDGE SINK. Derived from WORKFLOW_DIR and nothing else, so a lane spawned by
// e2e_workflow still curates into kernel_workflow/knowledge/learned/ — e2e's learned/ is a
// separate memory owned by its own system_architect step.
//   update_experience = on (default) | off | false | none
// The bake-off dispatcher passes `off` and curates once centrally instead (N parallel lanes would
// file N near-duplicate cards for one op).
const LEARNED_DIR = `${WORKFLOW_DIR}/knowledge/learned`;
// How much of ONE round the learned KB may steer. Measured, not guessed: an arm whose planner was
// handed at most 3 matched cards and could KB-seed at most one direction reached 4.45x geomean on 16
// kernels; the same 16 under a planner that reads the whole 84-line index and picks freely reached
// 3.44x — a 22.7% regression, with the losses concentrated on kernels where the bounded arm had found
// an unusual win (one went 25.77x -> 2.15x). Both arms averaged ~3 rounds, so it was not lost rounds.
// The index-reading design is better at RECALL; what it lost was the profile-driven direction that
// nothing from the KB had touched — the per-round control. `knowledge/learned/README.md` says the KB
// must be "an accelerant, NOT a cage", and this is the line that keeps that true.
//   KB_DIR_CAP        directions per round that may draw on cards (default 1)
//   KB_COLD_DIRECTION at least one direction per round must be profile-only (default on)
// Whether the planner is being pointed at the learned KB at all. Defined HERE because this lane is
// where it is USED: the constant of this name lives in the dispatcher, and referencing it from the
// lane threw `USE_LEARNED_READ is not defined` at the first plan step of every kernel — 13 of them
// before the run was stopped. A parse test cannot catch that: the file compiles, and the reference
// only resolves when the line runs. Default ON, matching this branch's design where the tech_lead
// role reads INDEX.md unconditionally; `use_learned_kb=false` turns the budget block off with it.
let USE_LEARNED_READ = String(A.use_learned_kb != null ? A.use_learned_kb : 'true') === 'true';
const KB_DIR_CAP = Math.max(0, parseInt(A.kb_dir_cap != null ? A.kb_dir_cap : 1, 10));
const KB_COLD_DIRECTION = String(A.kb_cold_direction != null ? A.kb_cold_direction : 'true') === 'true';
let kbCapBound = 0;      // rounds where the cap actually had to strip something
const UPDATE_EXPERIENCE = String(A.update_experience != null ? A.update_experience : 'on').trim().toLowerCase() || 'on';
const UPDATE_EXPERIENCE_ON = UPDATE_EXPERIENCE !== 'off' && UPDATE_EXPERIENCE !== 'false' && UPDATE_EXPERIENCE !== 'none';
// Did the CALLER freeze this kernel through oracle_freezer? Only the bake-off dispatcher does, so this
// tells director whether a GEAK_TIMING_RECEIPT is producible at all before it gates on one. Must be an
// ARG: this file never runs Freeze itself, and a workflow script has no filesystem access to go looking.
const FROZEN_ORACLE = String(A.frozen_oracle != null ? A.frozen_oracle : 'false') === 'true';

// ---------------------------------------------------------------------------
// WARM-START (the experience KB). Before the optimize loop, search the machine-produced KB —
// addressed by canonical id on both planes — for the top-3 best patches for THIS (kernel, language,
// gfx), validate each through the SAME verify_engineer gate, and adopt the first that passes; after
// write this run's own win back.
//   on (default)      | read + validate top-3, ADOPT the first that passes.
//   reference         | read top-3 as prose only, never auto-apply.
//   return_after_read | adopt then RETURN before the optimize loop.
//   off/false/none, a STATE_DIR resume, or no arch => cold start (byte-identical to pre-feature).
const WARM_START = String(A.warm_start != null ? A.warm_start : 'on').trim().toLowerCase() || 'on';
let WARM_START_ON = WARM_START !== 'off' && WARM_START !== 'false' && WARM_START !== 'none';
if (SKILLS_ISOLATE) {
  KERNEL_KNOWLEDGE_DIR = '';
  USE_LEARNED_READ = false;
  WARM_START_ON = false;
  USE_EXPERT_SKILLS = false;
}
const WARM_START_REF_ONLY = WARM_START === 'reference';
const WARM_START_RETURN_AFTER = WARM_START === 'return_after_read';
const KB_ARTIFACTS_DIR = String(A.kb_artifacts_dir ||
  (WORKFLOW_DIR ? WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/kb_artifacts' : '')).replace(/\/+$/, '');
const EXPERIENCE_STORE = `${WORKFLOW_DIR}/scripts/experience_store.py`;
// Every candidate costs a full on-box verify to reject, so a recorded near-tie is not worth reading.
const WARM_START_MIN_SPEEDUP = Number.isFinite(parseFloat(A.warm_start_min_speedup))
  ? parseFloat(A.warm_start_min_speedup) : 1.05;
// How hard the resolver may work to map this run's kernel name onto a stored page. The name is
// layout-derived (`fused_moe_kernel` standalone, `fused_moe_kernel_task` from an e2e head extraction),
// so `exact` alone would make the head path miss its own history; `fuzzy` also accepts an op_kind.
const WARM_START_MATCH = ['exact', 'normalized', 'fuzzy'].includes(String(A.warm_start_match || '').trim())
  ? String(A.warm_start_match).trim() : 'fuzzy';
// Which PLANE the experience is READ from. Same phases, same schemas, same verify gate either way
// — only the command string differs, because the store subcommands were built to print the same
// JSON as the directory ones.
//   store (default)  a KB Store on disk, in the shape the service uses, keyed by canonical id.
//   local            the kb_artifacts/ tree, keyed by slug. The original scheme, kept as an escape
//                    hatch for a box whose store is unusable.
//
// `store` is the default because it is the SAME addressing the service uses. Two schemes side by
// side drifted — `resolve` and `resolve-remote` assembled their filter chains separately, which is
// how the keyed read lost the `--include-retired` the slug read always had. Nothing moves: the tree
// is still written (`write-remote --plane both`) and `sync-local` carries the backlog across before
// each read; the tree stays the assembly buffer and the surface a curation pass edits.
const KB_MODE = String(A.kb_mode || 'store').trim().toLowerCase() === 'local' ? 'local' : 'store';
const KB_STORE_DIR = String(A.kb_store_dir ||
  (KB_ARTIFACTS_DIR ? KB_ARTIFACTS_DIR.replace(/\/[^/]*$/, '') + '/kb_store_local' : '')).replace(/\/+$/, '');
// The key carries a rocm <major>.<minor>; on a box without /opt/rocm the measured stack is empty and
// every record would file under `unspecified`, which splits one kernel's history across two keys.
const KB_FRAMEWORK_VERSION = String(A.kb_framework_version || '').trim();
const KB_VERSION_FLAG = KB_FRAMEWORK_VERSION ? ` --framework-version ${JSON.stringify(KB_FRAMEWORK_VERSION)}` : '';
const KB_ROOT_OK = KB_MODE === 'store' ? !!KB_STORE_DIR : !!KB_ARTIFACTS_DIR;
// Whether this run may also talk to the KB Store SERVICE, on top of whichever local plane KB_MODE
// selected. `auto` (default) uses it when credentials are present and silently does not when they
// are not; `off` restores the directory-only behaviour byte for byte.
const KB_REMOTE = String(A.kb_remote || 'auto').trim().toLowerCase() === 'off' ? 'off' : 'auto';
// The credentials are NOT visible from this process. They live in the user's profile, and the
// shells these commands run in are non-interactive, so `process.env.KB_STORE_TOKEN` is empty here
// even on a box where the service is configured. Deciding the plane in JS would therefore mean
// deciding it wrong. Instead every emitted command exports its own credentials from the profile's
// two sources and then branches on what it actually got — the test happens where the answer is
// knowable. The token is read from a 0600 file into a variable, never passed in argv, because
// `ps` is world-readable on this box.
// The gateway's internal AMD CA is not in a stock container trust store, so a KB command run inside
// one fails TLS (the old workaround was `curl -k`). DETECT then heal: only when the caller has NOT
// already established trust (SSL_CERT_FILE unset) do we point urllib/requests/curl/node at the first
// readable AMD-root bundle we find. Most callers here run OUTSIDE the warm-start launcher (which sets
// these at `docker run`), so this self-heal is what keeps their KB reachable. Path-only, no CA
// content; overridable with KB_CA_BUNDLE; a no-op when SSL_CERT_FILE is already set or no bundle is
// readable (so CI and already-trusting images are byte-identical). DNS (the host has none
// in-container) is a launch concern, handled with `docker run --add-host`.
// GEAK_KB_STORE_* wins over the bare KB_STORE_* (so a host that also runs Hyperloom, which carries
// its own KB_STORE_*, can point GEAK at a distinct store); re-exported under the bare names every
// downstream reader here branches on.
const KB_ENV_PRELUDE =
  'export KB_STORE_URL="${GEAK_KB_STORE_URL:-${KB_STORE_URL:-https://global.primus-safe.amd.com/knowledge-base}}"; ' +
  'export KB_STORE_TOKEN="${GEAK_KB_STORE_TOKEN:-${KB_STORE_TOKEN:-$(cat ~/.geak_kb_token 2>/dev/null)}}"; ' +
  'if [ -z "${SSL_CERT_FILE:-}" ]; then for _ca in "${KB_CA_BUNDLE:-}" ' +
  '/shared_nfs/hyperloom/ca/amd-ca-combined.pem "$HOME/amd-extra-ca-bundle.pem"; do ' +
  '[ -n "$_ca" ] && [ -r "$_ca" ] && { export SSL_CERT_FILE="$_ca" REQUESTS_CA_BUNDLE="$_ca" ' +
  'CURL_CA_BUNDLE="$_ca" NODE_EXTRA_CA_CERTS="$_ca"; break; }; done; fi; ';
// Writing in store mode records BOTH planes in one call, so it needs both roots: the directory tree
// stays the source of truth a curation pass edits, and the store is derived from it.
const KB_WRITE_OK = KB_ROOT_OK && !!KB_ARTIFACTS_DIR;
const kebab = s => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60);

// ---------------------------------------------------------------------------
// DEEP-MODE continuation + cross-backend / e2e-feedback hooks. ALL OPTIONAL.
// When none are passed (every normal/fast e2e run, and every standalone run) these are '' / the
// current defaults and are NEVER threaded into a prompt — so behavior is byte-identical to the
// pre-feature build. They are set ONLY by e2e_workflow's deep_mode head scheduler.
//   STATE_DIR        a STABLE dir for THIS (kernel,backend) ACROSS deep waves. When set the run
//                    RESUMES: director seeds the canonical from STATE_DIR/best (the cumulative-best
//                    code) and returns prior_state (cumulative + history) so re-invocation CONTINUES
//                    instead of restarting (no lost experience, no re-explored directions). The frozen
//                    oracle baseline (immutable unittest.py/meta.json) stays the reference, so speedups
//                    remain comparable to the TRUE baseline across waves. update_memory writes STATE.json
//                    + syncs best/ each round.
//   SHARED_KB        cross-backend blackboard file (read by plan+engineers, appended by update_memory).
//   GLOBAL_KB        run-global cross-KERNEL technique blackboard (deep): techniques that generalize
//                    across head ops/backends. Optional; unset (default/fast) => byte-identical prompts.
//   E2E_FEEDBACK     path to the latest end-to-end A/B result + problems from e2e_workflow (engaged?,
//                    cudagraph behavior, mem footprint, decode regression, e2e delta) — steers planning.
//   HARNESS_ADDENDUM path to an e2e-refined harness addendum (timing-weight / cudagraph-capture / hard
//                    constraint gates). The IMMUTABLE oracle is NEVER touched; this only refines what the
//                    perf bench emphasizes so the isolated target aligns with e2e.
//   MAX_NO_IMPROVE   consecutive non-improving rounds before stopping (default 2 = current behavior).
const STATE_DIR = String(A.state_dir || '').replace(/\/+$/, '');
const SHARED_KB = String(A.shared_kb || '').trim();
const GLOBAL_KB = String(A.global_kb || '').trim();   // run-global cross-KERNEL technique blackboard (deep)
const E2E_FEEDBACK = String(A.e2e_feedback || '').trim();
const HARNESS_ADDENDUM = String(A.harness_addendum || '').trim();
// P2 (deep continuation): on a RESUMED wave (STATE_DIR holds prior work) the cold re-derivation of
// Analyze + baseline Profile is largely redundant. INCREMENTAL_RESUME tells those two ADVISORY agents to
// load the prior roadmap/profile and return it with only delta updates instead of re-deriving from
// scratch, so each burst spends its budget on optimization ROUNDS, not re-analysis. Benchmark is NEVER
// incremental (it re-pins a fresh in-window baseline every wave \u2014 the matched-A/B correctness rail).
// Unset (default/fast / first deep burst) => spreading {} adds nothing => byte-identical prompts.
const INCREMENTAL = !!STATE_DIR && String(A.incremental_analyze || '') === 'true';
const RESUME_INPUT = INCREMENTAL ? { INCREMENTAL_RESUME: '1' } : {};
const MAX_NO_IMPROVE = Math.max(1, parseInt(A.max_no_improve != null ? A.max_no_improve : 2, 10));
// Conditional inputs: spreading {} adds NOTHING to a prompt (byte-identical) when a hook is unset.
const KB_INPUTS = {
  ...(SHARED_KB ? { SHARED_KB } : {}),
  ...(GLOBAL_KB ? { GLOBAL_KB } : {}),
  ...(E2E_FEEDBACK ? { E2E_FEEDBACK } : {}),
  ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
};

// ---------------------------------------------------------------------------
// Reusable JSON-schema fragments.
// ---------------------------------------------------------------------------
const perCase = {
  type: 'array',
  items: {
    type: 'object',
    properties: {
      name: { type: 'string' },
      baseline_ms: { type: 'number' },
      optimized_ms: { type: 'number' },
      speedup: { type: 'number' },
      // Workload-alignment fields (present only when a WORKLOAD_SPEC drives the harness; absent
      // on a normal unweighted run). weight = this case's baseline time SHARE in the real workload;
      // it is the coefficient of the time-weighted metric Σweight / Σ(weight/speedup). count is
      // optional/informational (regime-attributed cases have no per-call count).
      weight: { type: 'number' },
      count: { type: 'number' },
      dims: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
      dtypes: { type: 'array', items: { type: 'string' } },
      weight_source: { type: 'string' }, // trace | regime | regime_floor | prior | caller
    },
    required: ['name', 'speedup'],
  },
};
const obj = (props, required) => ({ type: 'object', properties: props, required: required || [], additionalProperties: true });

const SETUP_SCHEMA = obj({
  eval_dir: { type: 'string' }, workspace: { type: 'string' }, baseline_dir: { type: 'string' },
  kernel_name: { type: 'string' },
  device_gfx: { type: 'string' },
  device_target: { type: 'string' },
  device_name: { type: 'string' },
  physical_cu_count: { type: 'number' },
  source_files: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
  // Frozen-baseline verdict (BOTH modes). The unittest's timing + random-value parity baseline MUST be
  // the real online kernel — the immutable baseline_src/ dir OR an importable meta.baseline_callable —
  // never kernel_src/ (the candidate's own scaffold). The director sets baseline_frozen=true after it
  // copies baseline_src/ + confirms meta.baseline_callable; the script aborts the run if neither holds.
  baseline_frozen: { type: 'boolean' }, baseline_callable: { type: 'string' },
  // GPUs one run of the task's test/benchmark needs (roles/director.md "GPU requirement").
  gpu_requirement: obj({
    kind: { type: 'string', enum: ['single', 'fixed', 'scalable', 'unknown'] },
    gpus_per_job: { type: 'number' }, evidence: { type: 'string' },
  }, ['kind']),
  // DEEP-MODE resume only: populated by the director ONLY when STATE_DIR was provided AND a prior best
  // exists there. Lets a continued wave restore its cumulative speedup + insight/ledger history so it
  // does not re-explore dead directions. Absent (undefined) on a fresh run -> no behavior change.
  resumed: { type: 'boolean' },
  prior_state: obj({
    cumulative: { type: 'number' }, insights: { type: 'array', items: { type: 'string' } },
    ledger: { type: 'array', items: { type: 'object', additionalProperties: true } },
    bottleneck_now: { type: 'string' }, best_per_case: perCase,
  }, []),
}, ['eval_dir', 'workspace', 'kernel_name', 'device_gfx', 'device_target', 'physical_cu_count']);

const AUTHOR_SCHEMA = obj({
  authored: { type: 'boolean' }, target_language: { type: 'string' }, correctness: { type: 'string' },
  baseline_ms: { type: 'number' }, kernel_src_path: { type: 'string' }, entry_point: { type: 'string' },
  build: { type: 'boolean' }, notes: { type: 'string' },
}, ['authored', 'correctness']);

const ANALYZE_SCHEMA = obj({
  kernel_type: { type: 'string' }, kernel_file: { type: 'string' }, entry_point: { type: 'string' },
  modifiable_files: { type: 'array', items: { type: 'string' } },
  bottleneck_guess: { type: 'string' }, roadmap_summary: { type: 'string' },
  candidate_directions: { type: 'array', items: { type: 'object', additionalProperties: true } },
  // perf_knowledge resolution (REFERENCE ONLY): the operator/language this kernel maps to in the
  // AMD perf_knowledge base, plus the most relevant card paths, so engineers read focused context
  // instead of re-navigating the whole base. Empty string / [] / null when no card applies.
  kk_operator: { type: ['string', 'null'] }, kk_language: { type: ['string', 'null'] },
  kk_refs: { type: 'array', items: { type: 'string' } },
}, ['kernel_type', 'roadmap_summary']);

const BENCH_SCHEMA = obj({
  commandment_path: { type: 'string' }, correctness_cmd: { type: 'string' },
  benchmark_cmd: { type: 'string' }, profile_cmd: { type: 'string' }, parse_hint: { type: 'string' },
  baseline_per_case: { type: 'array', items: { type: 'object', additionalProperties: true } },
  baseline_geomean_ms: { type: 'number' }, num_test_cases: { type: 'number' },
  // Workload-aligned outputs: present when a WORKLOAD_SPEC drove case selection + weights.
  // baseline_weighted_total_ms = the baseline time the weights represent (Σ weight_i in time units).
  // The metric is Σ weight_i / Σ (weight_i/speedup_i). workload_aligned flags weights are real (not 1).
  workload_aligned: { type: 'boolean' },
  baseline_weighted_total_ms: { type: 'number' },
  weights_provenance: { type: 'string' }, // e.g. "trace" | "regime" | "regime_floor" | "prior" | "caller" | "mixed"
  reliable: { type: 'boolean' }, notes: { type: 'string' },
}, ['commandment_path', 'baseline_per_case', 'baseline_geomean_ms']);

const PROFILE_SCHEMA = obj({
  bottleneck: { type: 'string' }, profiler_used: { type: 'string' }, dispatch_count: { type: 'number' },
  // The accelerator detected on-box (e.g. "MI300X / gfx942 / CDNA3, 304 CU, ~5.3 TB/s"), so the
  // roofline ceiling + grid-sizing advice downstream use the real card instead of an assumed MI300X.
  device: { type: 'string' },
  key_metrics: { type: 'object', additionalProperties: true },
  top_kernels: { type: 'array', items: { type: 'object', additionalProperties: true } },
  top_opportunities: { type: 'array', items: { type: 'string' } },
  summary_path: { type: 'string' }, shift_note: { type: 'string' },
}, ['bottleneck', 'top_opportunities']);

// --- Deep Research Agent (DRA) schemas (opt-in via args.dra_enabled) -------
// Stage 0+1/2: the researcher extracts facts and returns a ranked set of research QUESTIONS the
// script then fans out in parallel. Stages 3/4: one answer per question (run concurrently). Stage 5:
// optional blindspot critique. Stage 7: the ranked-directions portfolio + brief the planner consumes.
const RESEARCH_PLAN_SCHEMA = obj({
  facts: { type: 'object', additionalProperties: true },
  questions: {
    type: 'array',
    items: obj({
      id: { type: 'string' }, question: { type: 'string' },
      search_queries: { type: 'array', items: { type: 'string' } },
      rationale: { type: 'string' }, tests_hypothesis: { type: 'string' },
      mode: { type: 'string', enum: ['bottleneck', 'design_space'] },
      rank_score: { type: 'number' },
    }, ['question']),
  },
  notes: { type: 'string' },
}, ['questions']);

const RESEARCH_QUESTION_SCHEMA = obj({
  question_id: { type: 'string' }, question: { type: 'string' },
  mode: { type: 'string' }, tests_hypothesis: { type: 'string' },
  answer: { type: 'string' },
  status: { type: 'string', enum: ['prefer', 'deprioritize', 'reject', 'open'] },
  affected: { type: 'array', items: { type: 'string' } },
  evidence: { type: 'array', items: { type: 'object', additionalProperties: true } },
  taskgen_implications: { type: 'string' }, notes: { type: 'string' },
}, ['answer', 'status']);

const RESEARCH_BLINDSPOT_SCHEMA = obj({
  blindspots: {
    type: 'array',
    items: obj({
      description: { type: 'string' }, why_it_matters: { type: 'string' },
      follow_up_question: { type: 'string' },
    }, ['description']),
  },
}, ['blindspots']);

// Stage 7 final portfolio. `directions` is kept COMPACT here (mirrors deep_search_brief.md): the
// planner reads the brief on disk, so this structured echo is just for logging/validation.
const RESEARCH_SCHEMA = obj({
  num_questions: { type: 'number' }, num_directions: { type: 'number' },
  brief_path: { type: 'string' }, full_path: { type: 'string' }, json_path: { type: 'string' },
  directions: {
    type: 'array',
    items: obj({
      id: { type: 'string' }, title: { type: 'string' },
      specialty: { type: 'string', enum: ['algorithm', 'memory', 'compute', 'host_runtime', 'deep_explore'] },
      mechanism: { type: 'string' }, expected_upside: { type: 'string' },
      confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
      rank_score: { type: 'number' },
    }, ['title']),
  },
  notes: { type: 'string' },
}, ['num_directions', 'brief_path']);

const PLAN_SCHEMA = obj({
  stop: { type: 'boolean' }, reasoning: { type: 'string' },
  directions: {
    type: 'array',
    items: obj({
      id: { type: 'string' }, title: { type: 'string' },
      specialty: { type: 'string', enum: ['algorithm', 'memory', 'compute', 'host_runtime', 'deep_explore'] },
      focus_files: { type: 'array', items: { type: 'string' } },
      expected_speedup: { type: 'number' }, prompt: { type: 'string' },
      kk_refs: { type: 'array', items: { type: 'string' } }, // optional: perf_knowledge card paths for THIS direction (REFERENCE ONLY)
      // Learned cards that SEEDED this direction, by filename. Structural attribution: the planner
      // declares what it opened, the script joins that against what the VERIFIER independently
      // measured. Declared here rather than inferred because the read path is semantic — the planner
      // reads INDEX.md and judges by meaning, so nothing downstream can reconstruct which card it
      // acted on. A citation is not a causal claim (the planner saw the profile too, and one run
      // holds no counterfactual); it is the only way a card can ever LOSE standing.
      learned_refs: { type: 'array', items: { type: 'string' } },
    }, ['id', 'title', 'specialty', 'prompt']),
  },
}, ['stop', 'directions']);

const ENG_SCHEMA = obj({
  engineer_id: { type: 'string' }, specialty: { type: 'string' }, task: { type: 'string' },
  strategy: { type: 'string' }, speedup_geomean: { type: 'number' }, speedup_arithmetic: { type: 'number' },
  // Time-weighted ratio-of-sums vs the TRUE baseline (PRIMARY metric when workload_aligned).
  // = Σ weight_i / Σ (weight_i / speedup_i). Omitted on unweighted runs.
  speedup_weighted: { type: 'number' },
  per_case: perCase, status: { type: 'string' }, patch_file: { type: 'string' },
  measurement_valid: { type: 'boolean' },
  strategies_tried: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
}, ['status', 'speedup_geomean', 'measurement_valid']);

const VERIFY_SCHEMA = obj({
  status: { type: 'string' }, correctness: { type: 'string' },
  verified_geomean: { type: 'number' }, verified_arithmetic: { type: 'number' },
  verified_weighted: { type: 'number' }, // time-weighted ratio-of-sums (PRIMARY when workload_aligned)
  per_case: perCase, variance_note: { type: 'string' }, notes: { type: 'string' },
  graph_safe: { type: 'string' },
}, ['status', 'verified_geomean']);

const INTEGRATE_SCHEMA = obj({
  attempted: { type: 'boolean' },
  combos_tried: { type: 'array', items: { type: 'object', additionalProperties: true } },
  best: { type: 'object', additionalProperties: true },
  improved_over_best_individual: { type: 'boolean' },
  conclusion: { type: 'string' }, notes: { type: 'string' },
}, ['attempted', 'conclusion']);

const MEMORY_SCHEMA = obj({
  insights: { type: 'array', items: { type: 'string' } },
  ledger: { type: 'array', items: { type: 'object', additionalProperties: true } },
  bottleneck_now: { type: 'string' }, suggest_next: { type: 'string' },
}, ['insights']);

const COMMIT_SCHEMA = obj({
  committed: { type: 'boolean' }, current_best_diff: { type: 'string' }, note: { type: 'string' },
}, ['committed']);

const REPORT_SCHEMA = obj({
  final_speedup_geomean: { type: 'number' }, final_speedup_arithmetic: { type: 'number' },
  final_speedup_weighted: { type: 'number' }, // time-weighted ratio-of-sums (PRIMARY when workload_aligned)
  rounds: { type: 'number' }, budget_used: { type: 'number' },
  report_path: { type: 'string' }, final_patch: { type: 'string' }, per_case: perCase,
}, ['final_speedup_geomean', 'report_path', 'final_patch']);

const UPDATE_EXPERIENCE_SCHEMA = obj({
  action: { type: 'string' },      // created | merged | skipped
  card_path: { type: 'string' },   // path under knowledge/learned/, or "" if nothing distilled
  key: { type: 'string' }, note: { type: 'string' },
}, []);

const VALIDATE_SCHEMA = obj({
  kernel_name: { type: 'string' },
  director_verified_speedup_geomean: { type: 'number' },
  director_verified_speedup_arithmetic: { type: 'number' },
  director_verified_speedup_weighted: { type: 'number' }, // PRIMARY when workload_aligned
  tech_lead_reported_speedup_geomean: { type: 'number' },
  validation_status: { type: 'string' }, correctness: { type: 'string' },
  // clean | host_bound | unprimed | unknown | not_applicable. Declared so the relaying agent has no
  // reason to drop it -- the gate calls it REQUIRED so a speedup never travels downstream unlabelled.
  timing_basis: { type: 'string' },
  per_case: perCase, applied_to_original: { type: 'string' },
  arbitration_note: { type: 'string' }, final_patch: { type: 'string' },
}, ['director_verified_speedup_geomean', 'validation_status']);

// Warm-start resolver output = experience_store.py `resolve` JSON, verbatim.
const WARMSTART_RESOLVE_SCHEMA = obj({
  read_reason: { type: 'string' }, slug: { type: 'string' },
  // What this run's name derived to vs the page actually served, so a surprising match is visible.
  requested_slug: { type: 'string' }, match_tier: { type: 'string' },
  // store mode only: the key the candidates came from. Declared rather than left to
  // additionalProperties so the agent relaying this JSON has no reason to drop it.
  canonical_id: { type: 'string' },
  // Which of the keyed read's planes answered ("remote" | "local"). Both are addressed by canonical
  // id, so this is the ONLY field telling them apart — and the attest step picks its flags from it.
  read_plane: { type: 'string' },
  // Same kernel, another language: the wrong target_language was passed, not an empty store.
  other_language_pages: { type: 'array', items: { type: 'string' } },
  filtered: obj({
    total: { type: 'number' }, retired: { type: 'number' }, below_min_speedup: { type: 'number' },
    same_direction_collapsed: { type: 'number' }, demoted_by_hint: { type: 'number' },
    // The caveat on every number above: the service pages `--scan` rows, so a saturated read
    // describes a prefix of the page and reads identically to a thin page.
    scanned: { type: 'number' }, scan_saturated: { type: 'boolean' },
  }),
  candidates: {
    type: 'array',
    items: obj({
      rank: { type: 'number' }, slug: { type: 'string' }, speedup: { type: 'number' },
      exp_dir: { type: 'string' }, arch: { type: 'string' },
      // Declared for the same reason canonical_id is: without it the attest step below has no
      // address to write this run's verdict back to.
      session_id: { type: 'string' },
      patch_path: { type: 'string' }, prose_path: { type: 'string' },
      strategy: { type: 'string' }, status: { type: 'string' },
      // direction = the IDEA (one rank each). comparable=false: recorded on a different bench key
      // than rank 1's, so the order between them is a prior only.
      direction: { type: 'string' }, bench_key: { type: 'string' }, comparable: { type: 'boolean' },
    }, ['rank', 'patch_path']),
  },
}, ['read_reason']);
// Warm-start writer output = experience_store.py `write` JSON, verbatim. written=false with reason
// "duplicate_impl" is a success: the store already held this code and counted a reproduction.
const WARMSTART_WRITE_SCHEMA = obj({
  written: { type: 'boolean' }, reason: { type: 'string' }, slug: { type: 'string' },
  dir: { type: 'string' }, speedup: { type: 'number' },
  reproduced: { type: 'string' }, reproductions: { type: 'number' },
  // store mode only: where the same result landed on the KB plane. `replaced` says which of the
  // two outcomes it was — a new candidate under the key, or this patch's own one measured again.
  remote: obj({
    written: { type: 'boolean' }, reason: { type: 'string' }, canonical_id: { type: 'string' },
    session_id: { type: 'string' }, champion: { type: 'boolean' }, replaced: { type: 'boolean' },
  }),
}, ['written']);

// ---------------------------------------------------------------------------
// Prompt helpers. Every agent reads its role file from WORKFLOW_DIR and the
// relevant knowledge files itself; the script only passes paths + JSON inputs.
// ---------------------------------------------------------------------------
const cfg = (o) => Object.entries(o).map(([k, v]) =>
  `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');

// --- Hung-agent guard ------------------------------------------------------
// An agent LLM call that HANGS (no response, no terminal error) blocks a
// parallel()/pipeline() round-barrier forever (observed: engineer agents frozen
// mid-turn wedged the whole optimize round for >30min). The harness resolves
// terminal API errors to null but NOT an indefinite hang. So bound every agent()
// call: if it has not returned after AGENT_TIMEOUT_MS, resolve it to null (which
// every .filter(Boolean)/null-check downstream already tolerates) and let the
// round proceed. VERY generous default (60min): a true hang never returns, so this only fires on a
// hang, NEVER on a legitimately-long agent. Inner agents include benchmark/profile/verify that build
// (hipcc/ninja) and run benches — minutes, well under 60min — plus the LLM-heavy optimize engineers
// (the ones observed hanging). A too-short bound would kill legit long agents (e.g. a slow rocprof or
// build), so keep it large. Cache keys (prompt, opts) are unchanged so resume still works. Falls back
// to raw agent() if setTimeout is unavailable. args.agent_timeout_ms=0 disables.
// API-FAULT TOLERANCE: a transient API failure (gateway 4xx/5xx, rate-limit, dropped connection, the
// model API going down mid-run) must NOT crash the whole workflow. agentT retries the call up to
// AGENT_RETRIES times on a thrown API/agent error, then resolves to null (every .filter(Boolean)/
// null-check downstream — incl. the Director validate + final report — already degrades on null rather
// than exiting). A timeout (hang) resolves null immediately and is NOT retried (a real hang would just
// burn another full timeout window). args.agent_retries tunes the count. If the failure is PERSISTENT
// (e.g. an auth/header requirement the client doesn't send), retries are exhausted then the run
// degrades — re-run with Workflow({resumeFromRunId}) once the client/API is fixed; cached agent results
// make resume cheap.
const AGENT_TIMEOUT_MS = parseInt(A.agent_timeout_ms != null ? A.agent_timeout_ms : 3600000, 10);
const AGENT_RETRIES = Math.max(1, parseInt(A.agent_retries != null ? A.agent_retries : 4, 10));
// GEAK-ABLATION-ARMS-v1 (B6): registered mechanical dispatches are folded into the NEXT
// agent instead of getting an agent of their own. In the historical Qwen3 run the
// storage-reclaim agent -- whose whole job is to run one fixed bash command -- cost
// 152 calls across 8 agents ($13.90), because every one of them paid the full
// bootstrap floor and then held a conversation. Folding removes the dispatch; the
// command still runs, verbatim, as the next agent's first action.
const ABLATION_ARM = String(A.ablation_arm || '').trim().toUpperCase();
const ABL_B6 = ABLATION_ARM === 'B6';
const ABL_PENDING = [];
const ABL_DRAINED = [];
function ablDrainPrelude() {
  if (!ABL_PENDING.length) return '';
  const cmds = ABL_PENDING.splice(0, ABL_PENDING.length);
  for (const c of cmds) ABL_DRAINED.push(c);
  return '## FIRST, run these registered maintenance commands EXACTLY as written, then continue with your task.\n' +
    cmds.map((c) => '```bash\n' + c + '\n```').join('\n') + '\n\n';
}

// LLM token+time accounting (PURELY ADDITIVE; args.llm_stats="false" makes it a no-op).
// DELIBERATELY NO TIMESTAMPS: Date.now()/new Date() are unavailable in workflow scripts. Every
// duration in the report comes from the transcripts (scripts/llm_ledger.py); this records only the
// role/phase/attempt identity of each agent call so the ledger attributes tokens to the right agent.
const LLM_STATS = String(A.llm_stats != null ? A.llm_stats : 'true').trim().toLowerCase() !== 'false';
const LLM_TL = { schema: 'geak.agent_timeline/1', workflow: 'kernel_lane', events: [], nested: [] };
const TL_ROLE_RE = /You are the ([A-Za-z0-9_.\-]+)\.\s*PHASE=([A-Za-z0-9_.\-]+)\./;
// Called AT DISPATCH (before the await) so LLM_TL.events is in dispatch order, not completion order.
// Returns the event so the caller flips `ok` once the attempt resolves (null when off).
function tlAgent(prompt, o, attempt) {
  if (!LLM_STATS) return null;
  const m = TL_ROLE_RE.exec(String(prompt || ''));
  const e = {
    seq: LLM_TL.events.length,
    phase: (o && o.phase) || '',
    label: (o && o.label) || 'agent',
    role: m ? m[1] : '',
    sub_phase: m ? m[2] : '',
    attempt: attempt,
    ok: false,
  };
  LLM_TL.events.push(e);
  return e;
}

// Per-call options (stripped before the agent call): timeout_ms overrides AGENT_TIMEOUT_MS, max_retries
// overrides AGENT_RETRIES, and timeout_marker makes a timeout return {__agent_timed_out, label} instead of
// null, for callers that must tell a hang from an empty answer (mode=mega).
async function agentT(p, o) {
  const label = (o && o.label) ? o.label : 'agent';
  const timeoutMs = Number(o && o.timeout_ms) > 0 ? Number(o.timeout_ms) : AGENT_TIMEOUT_MS;
  const timeoutMarker = !!(o && o.timeout_marker);
  const retries = Number(o && o.max_retries) > 0 ? Math.max(1, Number(o.max_retries)) : AGENT_RETRIES;
  if (o && ('timeout_ms' in o || 'timeout_marker' in o || 'max_retries' in o)) {
    o = Object.fromEntries(Object.entries(o).filter(([k]) => !['timeout_ms', 'timeout_marker', 'max_retries'].includes(k)));
  }
  if (ABL_B6 && typeof p === 'string') { const pre = ablDrainPrelude(); if (pre) p = pre + p; }
  for (let attempt = 1; attempt <= retries; attempt++) {
    const ev = tlAgent(p, o, attempt);   // record AT DISPATCH; ok=false until it resolves
    try {
      if (typeof setTimeout !== 'function' || !(timeoutMs > 0)) {
        const r0 = await agent(p, o); if (ev && r0) ev.ok = true; return r0;
      }
      let to;
      const guard = new Promise((resolve) => {
        to = setTimeout(() => {
          log(`  [hung-agent guard] ${label} exceeded ${Math.round(timeoutMs / 60000)}min with no return — ` +
            `${timeoutMarker ? 'returning a timeout marker' : 'resolving null so the round proceeds'}.`);
          resolve(timeoutMarker ? { __agent_timed_out: true, label } : null);
        }, timeoutMs);
      });
      // A timeout resolves null (returned as-is, no retry). An API/agent error rejects -> caught below.
      const r = await Promise.race([
        agent(p, o).then((rr) => { clearTimeout(to); return rr; }, (e) => { clearTimeout(to); throw e; }),
        guard,
      ]);
      if (ev && r) ev.ok = true;
      return r;
    } catch (e) {
      const msg = String(e && e.message ? e.message : e).slice(0, 200);
      if (attempt < retries) {
        log(`  [api-fault guard] ${label} attempt ${attempt}/${retries} hit an API/agent error (${msg}) — retrying so a transient outage doesn't kill the run.`);
        continue;
      }
      log(`  [api-fault guard] ${label} still failing after ${retries} attempts (${msg}) — resolving null so the workflow degrades gracefully instead of exiting.`);
      return null;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// WALL-CLOCK DEADLINE (opt-in; absent => byte-identical to a build without it).
//   DEADLINE_EPOCH  unix seconds after which NO new optimization round may start.
//   NO_STOP_S       while more than this many seconds remain, the TechLead may not end the run on
//                   its own convergence gates; it gets exactly one forced re-plan.
// In CODE and not in the caller's prose because the prose version was measured: a "fixed 4h window
// per kernel" comparison whose window had no implementation anywhere realised 1.55h to 15.22h, a
// 9.81x spread, and the comparison it existed to make was void. The loop below is
// `while (dispatched < BUDGET && noImprove < MAX_NO_IMPROVE)` — no time term. Emphatic prompt text
// is not a constraint, it is one more input to a judgement call.
// A workflow script has no clock (Date.now() is unavailable — it would break resume), so the time is
// read by a tiny low-effort agent. If that read FAILS the run is NOT killed: an unreadable clock
// returns Infinity, degrading to the old budget-only behaviour rather than truncating a healthy run.
const DEADLINE_EPOCH = (() => {
  const v = parseInt(A.deadline_epoch != null ? A.deadline_epoch : 0, 10);
  return Number.isFinite(v) && v > 0 ? v : 0;
})();
const NO_STOP_S = Math.max(0, parseInt(A.no_stop_s != null ? A.no_stop_s : 900, 10));
// Run-level cap on how many TechLead stops may be refused (forcedReplans is not reset per round).
// Not 1: the refusal must scale with the window, or a long budget is handed back early.
// Not unbounded: a role that truly cannot name a direction would spend the window planning.
const MAX_FORCED_REPLANS = Math.max(1, parseInt(A.max_forced_replans != null ? A.max_forced_replans : 6, 10));
const CLOCK_SCHEMA = { type: 'object', properties: { epoch: { type: 'number' } },
                       required: ['epoch'], additionalProperties: true };
let deadlineHit = false;
let forcedReplans = 0;

async function secondsLeft(tag) {
  if (!DEADLINE_EPOCH) return Infinity;
  const r = await agentT(
    `Run EXACTLY this command and nothing else:
\`\`\`bash
date +%s
\`\`\`
Return {"epoch": <the integer it printed>}. Do NOT modify any file and do NOT run anything else.`,
    { phase: 'Optimize', label: `clock ${tag}`, effort: 'low', schema: CLOCK_SCHEMA });
  if (!r || !Number.isFinite(r.epoch)) {
    log(`  [deadline] clock read failed at ${tag} — treating time as unlimited for this check.`);
    return Infinity;
  }
  return DEADLINE_EPOCH - r.epoch;
}

// Expert-skills injection. PURELY ADDITIVE: '' when OFF or the role is not a skills consumer, so both
// call sites (roleAgent, and the inline Optimize prompt) are byte-identical to the pre-feature build in
// those cases. When ON, appends an advisory pointer telling the agent to Read the fragment + query the
// skills index (scripts have no fs access).
function expertSkillsBlock(role) {
  if (!USE_EXPERT_SKILLS || !EXPERT_SKILL_ROLES.has(role) || !EXPERT_SKILLS_DIR) return '';
  return `\n\n## Expert skills (ADVISORY — opt-in, enabled this run)\n` +
    `Also Read ${WORKFLOW_DIR}/roles/_fragments/expert_skills.md and follow it: query ` +
    `${EXPERT_SKILLS_DIR}/index.yaml for skills whose \`match\` fits this op (operator/dtype/regime, and ` +
    `from_backend->to_backend for migration skills), whose \`scope\` is \`kernel\` (\`tuning\` entries ` +
    `belong to the e2e tuning phase and match every operator), and whose validation_status is ` +
    `\`validated\` and \`auto_apply\` is true, and ` +
    `treat each as a HIGH-PRIOR candidate to reproduce — advisory only, never overriding your isolated ` +
    `A/B vs the oracle, never reducing a result below the measured baseline.`;
}

// Appended to every role prompt only for multi-GPU jobs, so single-GPU prompts are unchanged.
function multiGpuBlock(role) {
  if (!GPU_JOB) return '';
  const n = GPU_RESOURCE.gpusPerJob;
  return `\n\n## Multi-GPU job (GPUS_PER_JOB=${n})\n` +
    `Every GPU command of this op needs ${n} GPUs. \`GPU_ID\` is the group spec \`${GPU_JOB}\`: one ` +
    `\`gpu_lock.sh $GPU_ID <cmd>\` holds the whole group for the whole command and sets ` +
    `HIP_VISIBLE_DEVICES to the leased ids, so start every rank inside that one call (e.g. ` +
    `\`torchrun --nproc_per_node=${n} ...\`). Never call gpu_lock.sh from inside a leased command (it is ` +
    `rejected) and never background a lease (\`&\`, nohup). A leased command is killed after ` +
    `GEAK_GPU_RUN_TIMEOUT seconds (default 900); export a larger value for long runs such as a cold ` +
    `JIT build. Leases on the same GPUs serialize, so put the arms you compare into one run.`;
}

function roleAgent(role, phase, intro, inputs) {
  const base = `You are the ${role}. PHASE=${phase}.
First Read ${WORKFLOW_DIR}/roles/${MEGA ? 'mega/' : ''}${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${WORKFLOW_DIR}/knowledge/.
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}

## Inputs
${cfg(inputs)}

Return ONLY the structured JSON the role file specifies (a StructuredOutput tool is forced).`;
  if (MEGA) return base + megaExpertSkillsBlock(role, phase) + analysisSkillBlock(role, phase);
  return base + expertSkillsBlock(role) + multiGpuBlock(role);
}

// ===========================================================================
// mode=mega backend: a multi-rank operator (e.g. an EP MoE layer) searched as independent whole-kernel
// candidates, each verified on-card against the frozen baseline. The lane phases below run for every
// mode; under MEGA they call the hooks of this section, which hold only what mega adds. Hooks that can
// end the run early return {exit: result}.
// ===========================================================================

// ---------------------------------------------------------------------------
// mode=mega args (parsed for every mode; checked only under MEGA)
// ---------------------------------------------------------------------------
// Resume-safe deterministic modeled time. Charge allocated phase slices rather
// than wall time: replay/cached agents must follow the same budget path.
const WORKFLOW_STARTED_MS = 0;
let MEGA_CLOCK_MS = 0;
const MEGA_PREP_MODEL_MS = 90 * 1000;

// Positive control calibrates detection; without it 1.000x is unfalsifiable. Null calibrates noise.
const POSITIVE_CONTROL = (A.positive_control && typeof A.positive_control === 'object')
  ? A.positive_control : null;
const PC_ABORT = POSITIVE_CONTROL ? (A.positive_control.abort_on_fail !== false) : false;
const GPU_MIN_FREE_GIB = Number(A.gpu_min_free_gib || 0);
// working_kernel preserves one runnable candidate across rounds; speedup keeps the normal score gate.
const OBJECTIVE = (() => {
  const v = String(A.objective || 'speedup').trim();
  if (MEGA && v !== 'speedup' && v !== 'working_kernel') {
    throw new Error(`args.objective must be 'speedup' or 'working_kernel', got '${v}'`);
  }
  return v;
})();
const WORKING_KERNEL = OBJECTIVE === 'working_kernel';

const MEGA_DEFAULT_SPECIALTY = String(A.mega_default_specialty ||
  (GPU_RESOURCE.gpusPerJob > 1 ? 'distributed' : 'compute'));
// Mega keeps independent workflow-authored whole-kernel candidates. A matched Expert Skill supplies
// knowledge and an optional contract; external reference trees are never candidate bases.
const MEGA_PROFILE = String(A.mega_profile || 'production').trim();
if (MEGA && !['production', 'audit'].includes(MEGA_PROFILE)) {
  throw new Error(`args.mega_profile must be 'production' or 'audit', got '${MEGA_PROFILE}'`);
}
const MEGA_PRODUCTION = MEGA_PROFILE === 'production';
const MEGA_STOP_ON_DELIVERABLE = A.mega_stop_on_deliverable != null
  ? String(A.mega_stop_on_deliverable) === 'true' : MEGA_PRODUCTION;
const MEGA_MIN_SEARCH_ATTEMPTS = Math.max(0, Number(A.mega_min_search_attempts != null
  ? A.mega_min_search_attempts : 2));
const MEGA_CANDIDATE_TIMEOUT_S = Math.max(300, Number(A.mega_candidate_timeout_s ||
  3600));
const MEGA_STRUCTURAL_ONLY = String(A.mega_structural_only != null ? A.mega_structural_only : 'false') === 'true';
const MEGA_ROUTE_ONLY = String(A.mega_route_only != null ? A.mega_route_only : 'false') === 'true';
const MEGA_FINAL_TIMEOUT_S = Math.max(600, Number(A.mega_final_timeout_s ||
  (MEGA_PRODUCTION ? 7200 : 3600)));
const MEGA_TIME_BUDGET_S = Math.max(1800, Number(A.mega_time_budget_s ||
  28800));
const MEGA_FINAL_RESERVE_S = Math.max(600, Math.min(MEGA_TIME_BUDGET_S / 2,
  Number(A.mega_final_reserve_s || (MEGA_PRODUCTION ? 7200 : 3600))));
const MEGA_CLOSEOUT_RESERVE_S = Math.max(300, Number(A.mega_closeout_reserve_s || 900));
const EXPERT_SKILL_ID = String(A.expert_skill_id || '');
if (MEGA && EXPERT_SKILL_ID && !/^[a-z0-9][a-z0-9_-]*$/.test(EXPERT_SKILL_ID)) {
  throw new Error('expert_skill_id must be a safe slug');
}
const EXPERT_SKILL_REVISION = String(A.expert_skill_revision || '');
const EXPERT_SKILL_BUNDLE_SHA256 = String(A.expert_skill_bundle_sha256 || '');
const EXPERT_SKILL_PLANNER_EXTENSION_SHA256 =
  String(A.expert_skill_planner_extension_sha256 || '');
const EXPERT_SKILL_CONTRACT_SHA256 = String(A.expert_skill_contract_sha256 || '');
const optionalNumber = (value) => {
  const number = Number(value);
  return value == null || value === '' || !Number.isFinite(number) ? null : number;
};
const EXPERT_SKILL_RECORDED_SCORE = optionalNumber(A.expert_skill_recorded_score);
const EXPERT_SKILL_RECORDED_LOW = optionalNumber(A.expert_skill_recorded_low);
const EXPERT_SKILL_RECORDED_HIGH = optionalNumber(A.expert_skill_recorded_high);
const MEGA_FINAL_TOP_K = Math.max(1, Number(A.mega_final_top_k ||
  (MEGA_PRODUCTION ? 1 : 2)));
const MEGA_FINAL_FALLBACK_K = Math.max(MEGA_FINAL_TOP_K,
  Number(A.mega_final_fallback_k || (MEGA_PRODUCTION ? 2 : MEGA_FINAL_TOP_K)));
const MEGA_TIE_NOISE_PCT = Math.max(0, Number(
  A.mega_tie_noise_pct != null ? A.mega_tie_noise_pct : 1.45));
// Graph-captured targets require an isolated capture/replay smoke.
const MEGA_REQUIRE_GRAPH_CAPTURE = A.require_graph_capture != null
  ? String(A.require_graph_capture) === 'true'
  : !!(OP_SPEC && OP_SPEC.cuda_graph_safe === true);
const DIRECT_GRAPH_ACCURACY = String(A.direct_graph_accuracy || 'false') === 'true';
const argList = (value) => (Array.isArray(value) ? value : String(value || '').split(','))
  .map((s) => String(s).trim()).filter(Boolean);
// Guards that create credit are separate from guards that can only veto, so a win on a
// veto-only guard cannot be promoted as progress on the target.
const TARGET_GUARDS = argList(A.target_guards);
const REGRESSION_GUARDS = argList(A.regression_guards);
const BIMODAL_GUARDS = argList(A.bimodal_guards);
const BASELINE_ACTIVATION = A.baseline_activation &&
  typeof A.baseline_activation === 'object' ? A.baseline_activation : {};
// operator_e2e is the only metric comparable across a launch-structure change; per-stage timers are
// diagnostics. changed_kernel needs a same-timeline changed_us/replaced_sum_us measurement; legacy
// keeps the old global score.
const PROMOTION_METRIC = String(A.promotion_metric || 'legacy').trim();
if (MEGA && !['legacy', 'operator_e2e', 'changed_kernel'].includes(PROMOTION_METRIC)) {
  throw new Error(`args.promotion_metric must be 'legacy', 'operator_e2e' or 'changed_kernel', got ${PROMOTION_METRIC}`);
}
// Strict proof runs must state the metric; inheriting historical mixed behavior is not auditable.
if (MEGA && (String(A.strict_autonomy || 'false') === 'true' && PROMOTION_METRIC === 'legacy')) {
  throw new Error('args.strict_autonomy requires an explicit promotion_metric');
}
const EXPERT_SKILL_DIR = EXPERT_SKILL_ID
  ? `${EXPERT_SKILLS_DIR}/skills/${EXPERT_SKILL_ID}` : '';
const PINNED_MEGA_SKILL = USE_EXPERT_SKILLS && !!EXPERT_SKILL_DIR;
const EXPERT_SKILL_PLANNER_GUIDE_FILE = `${EXPERT_SKILL_DIR}/planner_guide.md`;
const EXPERT_SKILL_AUTHOR_GUIDE_FILE = `${EXPERT_SKILL_DIR}/author_guide.md`;
const EXPERT_SKILL_BUNDLE_TOOL = PINNED_MEGA_SKILL
  ? `${EXPERT_SKILLS_DIR}/_contribute/validate_skill.py`
  : String(A.expert_skill_bundle_tool || `${EXPERT_SKILLS_DIR}/_contribute/validate_skill.py`);
const EXPERT_SKILL_PLAYBOOK_FILE = PINNED_MEGA_SKILL
  ? `${EXPERT_SKILL_DIR}/skill.md` : String(A.expert_skill_playbook || '');
const EXPERT_SKILL_PLANNER_EXTENSION_FILE =
  PINNED_MEGA_SKILL ? `${EXPERT_SKILL_DIR}/planner_extension.yaml`
    : String(A.expert_skill_planner_extension || '');
const EXPERT_SKILL_CONTRACT_FILE = PINNED_MEGA_SKILL
  ? `${EXPERT_SKILL_DIR}/contract.yaml` : String(A.expert_skill_contract || '');
const EXPERT_SKILL_VALIDATION_FILE = PINNED_MEGA_SKILL
  ? `${EXPERT_SKILL_DIR}/validation.yaml` : String(A.expert_skill_validation || '');
const EXPERT_SKILL_RUNTIME_FILE = PINNED_MEGA_SKILL
  ? `${EXPERT_SKILL_DIR}/runtime_validation.py` : String(A.expert_skill_runtime_file || '');
const EXPERT_SKILL_VALIDATION_STATUS = String(
  A.expert_skill_validation_status || ''
);
const EXPERT_SKILL_USAGE = String(A.expert_skill_usage || '');
const REQUIRE_EXPERT_SKILL_BUNDLE_IDENTITY = USE_EXPERT_SKILLS &&
  (PINNED_MEGA_SKILL ||
   String(A.require_expert_skill_bundle_identity || 'false') === 'true');
const validSha256 = (value) => /^[a-f0-9]{64}$/.test(String(value || ''));
if (MEGA && (REQUIRE_EXPERT_SKILL_BUNDLE_IDENTITY &&
    (!EXPERT_SKILL_ID ||
     !EXPERT_SKILL_PLANNER_EXTENSION_FILE || !EXPERT_SKILL_CONTRACT_FILE ||
     !validSha256(EXPERT_SKILL_BUNDLE_SHA256) ||
     !validSha256(EXPERT_SKILL_PLANNER_EXTENSION_SHA256) ||
     !validSha256(EXPERT_SKILL_CONTRACT_SHA256)))) {
  throw new Error(
    'require_expert_skill_bundle_identity needs skill id/revision, Planner Extension/contract ' +
    'paths, and canonical bundle/extension/contract SHA-256 values'
  );
}
const EXPERT_SKILL_CONTRACT_TOOL = PINNED_MEGA_SKILL
  ? `${WORKFLOW_DIR}/tools/expert_skill_contract.py`
  : String(A.expert_skill_contract_tool || `${WORKFLOW_DIR}/tools/expert_skill_contract.py`);
const EXPERT_SKILL_REFERENCE_PATH = String(A.expert_skill_reference_path || '').replace(/\/+$/, '');
const CHECK_EXPERT_SKILL_CONTRACT = USE_EXPERT_SKILLS &&
  !!EXPERT_SKILL_ID && !!EXPERT_SKILL_CONTRACT_FILE &&
  String(A.verify_expert_skill_contract != null
    ? A.verify_expert_skill_contract : 'true') === 'true';
const REQUIRE_EXPERT_SKILL_CONTRACT = CHECK_EXPERT_SKILL_CONTRACT &&
  (PINNED_MEGA_SKILL || MEGA_STRUCTURAL_ONLY ||
    String(A.require_expert_skill_contract || 'false') === 'true');
// Skill source-shape checks are advisory diagnostics. Runtime admission is
// decided by the activated on-card behavior gates below.
const STRUCTURAL_CONTRACT_BLOCKING =
  String(A.block_on_expert_skill_contract || 'false') === 'true';
// Structural check + on-card Verify are reserved BEFORE the Author is sized; a leftover-based
// Verify budget was always <300s after a full Author turn, so no turn ever reached the card.
const MEGA_VERIFY_RESERVE_S = Math.max(300, Number(A.mega_verify_reserve_s || 1200));
const MEGA_POST_AUTHOR_RESERVE_S = MEGA_STRUCTURAL_ONLY ? 0
  : (CHECK_EXPERT_SKILL_CONTRACT ? 600 : 0) + MEGA_VERIFY_RESERVE_S;
const MEGA_STALL_LIMIT = Math.max(0, Number(A.mega_stall_limit != null ? A.mega_stall_limit : 3));
const MEASURE_FIRST_TEXT = (n) => MEGA_STALL_LIMIT && n >= MEGA_STALL_LIMIT
  ? `MEASURE_FIRST: ${n} rounds produced no new runtime evidence; before any source edit run a ` +
    'bounded on-card smoke of the current HEAD and report its concrete result. ' : '';
if (MEGA && MEGA_PRODUCTION &&
    MEGA_CANDIDATE_TIMEOUT_S - 150 - MEGA_POST_AUTHOR_RESERVE_S < 900) {
  throw new Error(`mega_candidate_timeout_s=${MEGA_CANDIDATE_TIMEOUT_S} cannot fit a 900s Author ` +
    `plus the ${MEGA_POST_AUTHOR_RESERVE_S}s structural+on-card Verify reserve`);
}
if (MEGA && MEGA_STRUCTURAL_ONLY && !CHECK_EXPERT_SKILL_CONTRACT) {
  throw new Error('mega_structural_only requires an enabled Expert Skill contract');
}
if (MEGA && USE_EXPERT_SKILLS && EXPERT_SKILL_VALIDATION_STATUS &&
    !['validated', 'experimental'].includes(EXPERT_SKILL_VALIDATION_STATUS)) {
  throw new Error('only validated or explicitly pinned experimental Expert Skills may run');
}
if (MEGA && USE_EXPERT_SKILLS && EXPERT_SKILL_ID && !EXPERT_SKILL_VALIDATION_STATUS) {
  throw new Error('an explicitly pinned Expert Skill requires validation status');
}
if (MEGA && USE_EXPERT_SKILLS && PINNED_MEGA_SKILL) {
  if (!['authoring', 'candidate_validation'].includes(EXPERT_SKILL_USAGE) ||
      !REQUIRE_EXPERT_SKILL_BUNDLE_IDENTITY || !EXPERT_SKILL_VALIDATION_FILE) {
    throw new Error(
      'pinned Mega Expert Skill requires exact bundle identity, validation file, and ' +
      'expert_skill_usage=authoring|candidate_validation'
    );
  }
  if (EXPERT_SKILL_USAGE === 'authoring' && !MEGA_STRUCTURAL_ONLY) {
    throw new Error('pinned authoring usage requires mega_structural_only=true');
  }
  if (EXPERT_SKILL_USAGE === 'candidate_validation' &&
      !CHECK_EXPERT_SKILL_CONTRACT) {
    throw new Error('pinned candidate_validation requires contract diagnostics');
  }
}
const GRAPH_CONTRACT_TOOL = PINNED_MEGA_SKILL
  ? `${WORKFLOW_DIR}/tools/expert_skill_runtime.py` : String(A.graph_contract_tool ||
    (!USE_EXPERT_SKILLS ? `${WORKFLOW_DIR}/tools/generic_graph_contract.py` : ''));
const FAST_TEST_KEY_TOOL = `${WORKFLOW_DIR}/tools/fast_test_key.py`;
const BENCH_HARNESS = String(A.benchmark_harness || '');
const CANDIDATE_IMPORT_MODULES = Object.freeze(argList(
  A.candidate_import_modules || [],
));
const EXPERT_SKILL_SOURCE_FILES = argList(A.expert_skill_source_files || []);
const EXPERT_SKILL_ACCURACY_CASES = Object.freeze(argList(A.expert_skill_accuracy_cases || []));
const REQUIRED_ACCURACY_CASES = Object.freeze(argList(A.required_accuracy_cases != null
  ? A.required_accuracy_cases : (USE_EXPERT_SKILLS ? EXPERT_SKILL_ACCURACY_CASES : [])));
const FORBIDDEN_CANDIDATE_TREE_DIGESTS = new Set(argList(
  A.forbidden_candidate_tree_digests || [],
).filter(validSha256));
const MEGA_EXPERT_SKILL_ROLES = new Set([
  'tech_lead', 'author_engineer', 'engineer', 'deep_engineer', 'mega_search_lead',
]);

const CAPABILITY_EVAL = String(A.capability_eval ?? 'false') === 'true';
const STRICT_AUTONOMY = String(A.strict_autonomy != null ? A.strict_autonomy : 'false') === 'true';
if (MEGA && STRICT_AUTONOMY && !CAPABILITY_EVAL) {
  throw new Error('args.strict_autonomy requires capability_eval=true');
}
if (MEGA && STRICT_AUTONOMY && TARGET_GUARDS.length === 0) {
  throw new Error('args.strict_autonomy requires at least one explicit target_guards entry');
}
const positiveIntArg = (name, value, fallback) => {
  const n = Number(value == null ? fallback : value);
  if (Number.isSafeInteger(n) && n > 0) return n;
  if (STRICT_AUTONOMY) throw new Error(`args.${name} must be a positive integer`);
  return fallback;
};
const REQUIRED_REPLAYS = positiveIntArg('required_replays', A.required_replays, 1000);
const REQUIRED_PAIRS = positiveIntArg('required_pairs', A.required_pairs, 5);
const REQUIRED_PAIRS_BY_GUARD = (A.required_pairs_by_guard &&
  typeof A.required_pairs_by_guard === 'object') ? A.required_pairs_by_guard : {};
if (MEGA && STRICT_AUTONOMY) {
  for (const [guard, count] of Object.entries(REQUIRED_PAIRS_BY_GUARD)) {
    if (!(Number.isSafeInteger(Number(count)) && Number(count) > 0)) {
      throw new Error(`args.required_pairs_by_guard[${guard}] must be a positive integer`);
    }
  }
}
const RF = Object(A.regression_floor_by_guard);
const RFI = { ...(A.regression_floor_by_guard ? { REGRESSION_FLOORS: RF } : {}),
  ...(A.mega_min_speedup ? { MIN_TARGET_SPEEDUP: +A.mega_min_speedup } : {}) };
const REQUIRE_OVERLAP = String(A.require_overlap != null ? A.require_overlap : 'false') === 'true';
const REQUIRE_ATTRIBUTION = String(A.require_attribution != null ? A.require_attribution : 'false') === 'true';
const REQUIRE_ARTIFACT_DISTINCT = String(
  A.require_artifact_distinct != null ? A.require_artifact_distinct : 'false') === 'true';
// Require an activated switched path to have actually launched on hardware.
const REQUIRE_HW_ACTIVATION = String(
  A.require_hardware_activation != null ? A.require_hardware_activation : 'false') === 'true';
const ACCURACY_METRIC = String(A.accuracy_metric || 'relL2');
const ACCURACY_THRESHOLD = Number(A.accuracy_threshold != null ? A.accuracy_threshold : 0.10);
if (MEGA && (STRICT_AUTONOMY && (!(ACCURACY_THRESHOLD > 0) || !Number.isFinite(ACCURACY_THRESHOLD)))) {
  throw new Error('args.accuracy_threshold must be a positive finite number');
}
const launchTargetArg = Number(A.launch_target != null ? A.launch_target : 2);
if (MEGA && STRICT_AUTONOMY && !(Number.isSafeInteger(launchTargetArg) && launchTargetArg > 0)) {
  throw new Error('args.launch_target must be a positive integer');
}
const LAUNCH_TARGET = Number.isSafeInteger(launchTargetArg) && launchTargetArg > 0
  ? launchTargetArg : 2;
const ALLOW_PARTIAL_FUSION =
  String(A.allow_partial_fusion != null ? A.allow_partial_fusion : 'true') === 'true';
if (MEGA && PINNED_MEGA_SKILL &&
    EXPERT_SKILL_USAGE === 'candidate_validation') {
  const missing = [
    [!MEGA_STRUCTURAL_ONLY, 'mega_structural_only=false'],
    [MEGA_REQUIRE_GRAPH_CAPTURE, 'require_graph_capture'],
    [DIRECT_GRAPH_ACCURACY, 'direct_graph_accuracy'],
    [REQUIRE_HW_ACTIVATION, 'require_hardware_activation'],
    [REQUIRE_ARTIFACT_DISTINCT, 'require_artifact_distinct'],
    [REQUIRE_ATTRIBUTION, 'require_attribution'],
    [REQUIRED_REPLAYS >= 256, 'required_replays>=256'],
    [REQUIRED_PAIRS >= 5, 'required_pairs>=5'],
    [LAUNCH_TARGET === 1 || LAUNCH_TARGET === 2, 'launch_target in {1,2}'],
    [!ALLOW_PARTIAL_FUSION, 'allow_partial_fusion=false'],
    [REQUIRED_ACCURACY_CASES.length > 0, 'required_accuracy_cases'],
  ].filter(([ok]) => !ok).map(([, label]) => label);
  if (missing.length) {
    throw new Error(`pinned candidate_validation missing gates: ${missing.join(', ')}`);
  }
}
const CONTAINMENT_PREFLIGHT = (A.containment_preflight &&
  typeof A.containment_preflight === 'object') ? A.containment_preflight : {};
const REFERENCE_LEAK_MARKER_FILE = String(A.reference_leak_marker_file || '');
if (MEGA && STRICT_AUTONOMY && CONTAINMENT_PREFLIGHT.clean !== true) {
  throw new Error(
    'args.strict_autonomy requires a clean containment_preflight supplied by the trusted launcher');
}
// Trees whose contents would constitute an imported answer. Given to VERIFY (to compare against) and
// deliberately NOT to engineers — handing them the list would be handing them the location.
const KNOWN_REFERENCE_PATHS = (Array.isArray(A.known_reference_paths) ? A.known_reference_paths
  : String(A.known_reference_paths || '').split(',')).map(s => String(s).trim()).filter(Boolean);
const KNOWN_REFERENCE_HASHES = Array.isArray(A.known_reference_hashes)
  ? A.known_reference_hashes.filter((r) => r && r.path_sha256 &&
    (r.sha256 || r.normalized_sha256)) : [];
if (MEGA && STRICT_AUTONOMY && KNOWN_REFERENCE_PATHS.length) {
  throw new Error(
    'args.strict_autonomy refuses known_reference_paths in workflow args; supply ' +
    'known_reference_hashes generated out of band so no agent-readable address is published.');
}
if (MEGA && CAPABILITY_EVAL) {
  log(`CAPABILITY EVAL mode: prior art is advisory CONCLUSIONS only; engineers get no reference ` +
      `paths, and verify rejects byte-identical files as status:"plagiarized". ` +
      `reference evidence=${KNOWN_REFERENCE_HASHES.length
        ? `${KNOWN_REFERENCE_HASHES.length} out-of-band file hash(es)`
        : (KNOWN_REFERENCE_PATHS.length ? `${KNOWN_REFERENCE_PATHS.length} hidden path(s)` :
          '(none given — provenance check DISABLED)')}`);
}

// Reference containment: under capability_eval a reference tree must live outside the ancestor that
// holds the task, workflow and run artifacts, because engineers can reach anything one `ls ..` away.
// Set here and enforced by the post-Setup gate, which sweeps the same ancestor.
let RUN_TREE_ANCESTOR = '';
if (MEGA && CAPABILITY_EVAL) {
  // Normalized by hand: workflow scripts run without Node's `path`/`fs`, so require() would throw
  // here at runtime and turn a safety check into a crash.
  const abs = (p) => {
    const out = [];
    for (const seg of String(p).split('/')) {
      if (!seg || seg === '.') continue;
      if (seg === '..') out.pop(); else out.push(seg);
    }
    return '/' + out.join('/');
  };
  const parts = [KERNEL_PATH_ORIG, EXP_ROOT, WORKFLOW_DIR].filter(Boolean).map(abs);
  let ancestor = parts.length ? parts[0] : '';
  for (const p of parts.slice(1)) {                       // longest common directory prefix
    const a = ancestor.split('/'), b = p.split('/');
    let i = 0; while (i < a.length && i < b.length && a[i] === b[i]) i++;
    ancestor = a.slice(0, i).join('/') || '/';
  }
  RUN_TREE_ANCESTOR = ancestor;
  const inside = KNOWN_REFERENCE_PATHS.filter(
    (p) => ancestor && (abs(p) === ancestor || abs(p).startsWith(ancestor + '/')));
  if (inside.length) {
    throw new Error(
      `REFERENCE NOT CONTAINED: capability_eval is on, but ${inside.length} reference tree(s) sit ` +
      `inside the run's own tree (${ancestor}): ${inside.join(' ')}. An engineer reaches these by ` +
      `walking up one directory, so the run would measure a copy and report it as a derivation — ` +
      `that has already happened twice here. Move each tree outside ${ancestor} (for a git worktree: ` +
      `git worktree move <old> <new>) and pass the new path in known_reference_paths, so verify can ` +
      `still compare against it. Set capability_eval=false if you WANT the reference ported.`);
  }
  log(`REFERENCE CONTAINMENT ok: ${KNOWN_REFERENCE_PATHS.length} reference tree(s) resolve outside ` +
      `the run tree ${ancestor}. NOTE this checks CONFIGURED PATHS ONLY, and that is a small fraction ` +
      `of the exposure. Sweeping this tree for reference-only markers afterwards found the answer in ` +
      `five more places, none of them a known_reference_path: the positive control's own applied ` +
      `workspace left behind under artifacts/, a previous wave's output tree, the SIBLING EVAL DIRS ` +
      `inside exp_root (one directory up from the dir every engineer is handed by name), the durable ` +
      `patches in state_dir, and a git bundle of the candidate branch. Run ` +
      `scripts/reference_leak_sweep.sh --tree ${ancestor} before trusting capability_eval, and note ` +
      `it still cannot see a reference reachable as a branch in another repository the run can read ` +
      `(JIT cache, sibling clone) — only verify's byte-identity check stands between that and a ` +
      `counted win.`);
}

// Optional profile-analysis skill (see knowledge/analysis_skills/INDEX.md): after Profile, an
// analysis_engineer adds operator-specific structure (e.g. MoE route imbalance, stage time-share).
// Analysis only, never directions. Default none: no inputs and no prompt text are added.
const ANALYSIS_SKILL = String(A.analysis_skill != null ? A.analysis_skill : 'none').trim();
const ANALYSIS_SKILL_ON = !!ANALYSIS_SKILL && ANALYSIS_SKILL !== 'none' && ANALYSIS_SKILL !== 'false';
if (MEGA && ANALYSIS_SKILL_ON && !/^[a-z0-9][a-z0-9_-]*$/.test(ANALYSIS_SKILL)) {
  throw new Error(`args.analysis_skill must be a safe skill slug, got ${JSON.stringify(ANALYSIS_SKILL)}`);
}
const ANALYSIS_SKILL_INPUTS = ANALYSIS_SKILL_ON ? {
  ANALYSIS_SKILL: ANALYSIS_SKILL,
  ANALYSIS_SKILL_DIR: `${WORKFLOW_DIR}/knowledge/analysis_skills/${ANALYSIS_SKILL}`,
} : {};
if (MEGA && ANALYSIS_SKILL_ON) log(`Profile-analysis skill: ${ANALYSIS_SKILL} (analysis only; Step-3 TechLead owns directions).`);
// Generic mega (Expert Skills off) only; skill-on and non-mega runs get {}.
const GMI = !USE_EXPERT_SKILLS ? {
  MEGA_MEASUREMENT_GUIDE: `${WORKFLOW_DIR}/knowledge/mega_measurement.md`,
  GENERIC_MEGA_NOTE: 'Read MEGA_MEASUREMENT_GUIDE\'s role table and MEGA_GENERIC_LESSONS before acting; they apply to your role here.',
  MEGA_GENERIC_LESSONS: `${WORKFLOW_DIR}/knowledge/mega_generic_lessons.md`,
  MEGA_AB_TOOL: `${WORKFLOW_DIR}/tools/mega_graph_ab.py`,
} : {};
// MoE regime guide + paired graph A/B for runs pinning a megamoe Skill (it names Skill designs, so a
// generic run never gets it by default); args.mega_regimes='true'|'false' overrides.
const MEGA_REGIMES_ON = A.mega_regimes != null
  ? String(A.mega_regimes) === 'true' : /^megamoe/.test(EXPERT_SKILL_ID);
const MRI = MEGA_REGIMES_ON ? {
  MEGA_REGIMES_GUIDE: `${WORKFLOW_DIR}/knowledge/mega_regimes.md`,
  MEGA_AB_TOOL: `${WORKFLOW_DIR}/tools/mega_graph_ab.py`,
  MEGA_REGIME_NOTE: 'Read MEGA_REGIMES_GUIDE for the target size\'s regime before choosing a direction; use MEGA_AB_TOOL for fast paired graph A/B while iterating.',
  ...(USE_EXPERT_SKILLS ? { MEGA_GENERIC_LESSONS: `${WORKFLOW_DIR}/knowledge/mega_generic_lessons.md` } : {}),
} : {};

const MEGA_RESUME_STATE = !!STATE_DIR &&
  String(A.mega_resume_state != null ? A.mega_resume_state : 'false') === 'true';
// Fast-test cache (mega + STATE_DIR only, default off): reuse a prior wave's front-matter measurements
// (control calibration, profile replays) when the validity key matches. It never touches the scoring
// denominator: every candidate A/B re-measures the frozen baseline.
const MEGA_FAST_TEST = !!STATE_DIR &&
  String(A.mega_fast_test != null ? A.mega_fast_test : 'false') === 'true';
   // run-global cross-KERNEL technique blackboard (deep)

// Verdict vocabulary keeps clean/leak/skipped/unknown distinct. "The scanner could not run" can
// never be reported as clean — that collapse is the exact failure this subsystem exists to prevent.
const CONTAINMENT_GATE_SCHEMA = obj({
  verdict: { type: 'string', enum: ['clean', 'leak', 'skipped', 'unknown'] },
  findings: { type: 'array', items: { type: 'string' } },
  note: { type: 'string' },
}, ['verdict']);

// Three-valued so that "could not read the pool" never collapses into "free".
const POOL_SCHEMA = obj({
  verdict: { type: 'string', enum: ['free', 'occupied', 'unknown'] },
  cards_total: { type: 'number' },
  cards_free: { type: 'number' },
  min_free_gib: { type: 'number' },
  foreign_pids: { type: 'array', items: { type: 'string' } },
  note: { type: 'string' },
}, ['verdict']);

const MEGA_SETUP_SCHEMA = obj({
  eval_dir: { type: 'string' }, workspace: { type: 'string' }, baseline_dir: { type: 'string' },
  kernel_name: { type: 'string' }, source_files: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
  // Frozen baseline: timing and parity must use the real online kernel (baseline_src/ or
  // meta.baseline_callable), never kernel_src/. The run aborts if neither is confirmed.
  baseline_frozen: { type: 'boolean' }, baseline_callable: { type: 'string' },
  mega_workspace_from_original: { type: 'boolean' },
  mega_candidate_state_checked: { type: 'boolean' },
  // Resume only: the director fills this when STATE_DIR holds a prior best, so a continued wave keeps
  // its cumulative speedup and history. Absent on a fresh run.
  resumed: { type: 'boolean' },
  workflow_revision: { type: 'string' },
  workflow_clean: { type: 'boolean' },
  prior_state: obj({
    cumulative: { type: 'number' }, insights: { type: 'array', items: { type: 'string' } },
    // Declared, because memoryMerge keys rows by `id` and dedupes on it, and the ladder is read back
  // out of `roadmap_rung`. A row missing `id` merges as a new row every round; a row missing
  // `verdict` is treated by the next plan_round as untried. Both read as normal ledger growth.
  ledger: { type: 'array', items: obj({
    id: { type: 'string' }, title: { type: 'string' }, specialty: { type: 'string' },
    // confirmed | partial | unresolved | dead_end — see tech_lead.md "Verdict by SPEC, not by
    // result". `unresolved` is what keeps a rung re-openable instead of closing everything above it.
    verdict: { type: 'string', enum: ['confirmed', 'partial', 'unresolved', 'dead_end'] },
    roadmap_rung: { type: ['string', 'null'] },
    expected: { type: 'number' }, actual: { type: 'number' },
    lesson: { type: 'string' },
    first_round: { type: 'number' }, last_round: { type: 'number' },
  }, ['id', 'verdict']) },
    bottleneck_now: { type: 'string' }, best_per_case: perCase,
    // Shelf entries are keyed by id, aged by base_round and offered by files; an entry without files is
    // marked footprint=unknown and never offered.
    shelf: { type: 'array', items: obj({
      id: { type: 'string' }, title: { type: 'string' }, specialty: { type: 'string' },
      geomean: { type: 'number' }, patch: { type: 'string' },
      files: { type: 'array', items: { type: 'string' } },
      // Fusion chain id (its terminal rung), carried across waves so a chain's producer commit does not
      // age its consumer half off the shelf. Empty for a standalone optimisation.
      chain_id: { type: 'string' },
      footprint: { type: 'string', enum: ['known', 'unknown'] },
      base_round: { type: 'number' }, shelved_round: { type: 'number' },
      absorbed: { type: 'boolean' },
    }, ['id', 'geomean']) },
    candidate_registry: { type: 'array', items: { type: 'object', additionalProperties: true } },
    measurement_calibration: { type: 'object', additionalProperties: true },
    state_sequence: { type: 'number' },
    // round -> files CANONICAL took on that round. Keys are round numbers as strings.
    absorbed_files: { type: 'object' },
  }, []),
}, ['eval_dir', 'workspace', 'kernel_name']);

// mode=mega only. Every whole-kernel lane — validated artifact, skill reproduction, or open search —
// uses one lifecycle. A WIP may advance its own lane without becoming the global incumbent.
const MEGA_CANDIDATE_SCHEMA = obj({
  candidate_id: { type: 'string' },
  candidate_source: { type: 'string', enum: ['search', 'integrated'] },
  base_candidate_id: { type: 'string' },
  candidate_status: {
    type: 'string',
    enum: ['authoring', 'runnable', 'scored', 'finalist', 'rejected'],
    description: 'A finalized GPU-free source checkpoint remains authoring until runtime verification.',
  },
  claim_complete: {
    type: 'boolean',
    description: 'True when this turn finalized its committed artifact, including a GPU-free static checkpoint; hardware gates are separate fields.',
  },
  checkpoint_complete: { type: 'boolean' },
  structural_verified: { type: 'boolean' },
  runtime_verified: { type: 'boolean' },
  score_complete: { type: 'boolean' },
  verification_status: { type: 'string' },
  gpu_executed: { type: 'boolean' },
  structural_report: { type: 'string' },
  structural_skill_id: { type: 'string' },
  structural_candidate_head: { type: 'string' },
  structural_candidate_tree_digest: { type: 'string' },
  structural_skill_bundle_sha256: { type: 'string' },
  structural_planner_extension_sha256: { type: 'string' },
  structural_contract_revision: { type: 'string' },
  structural_contract_sha256: { type: 'string' },
  contract_failures: { type: 'array', items: obj({
    id: { type: 'string' }, category: { type: 'string' },
    severity: { type: 'string' },
    messages: { type: 'array', items: { type: 'string' } },
  }, ['id', 'category', 'severity', 'messages']) },
  failed_required_checks: { type: 'array', items: { type: 'string' } },
  plan_consistency_errors: { type: 'array', items: { type: 'string' } },
  input_errors: { type: 'array', items: { type: 'string' } },
  provenance_attestation_valid: { type: 'boolean' },
  attempt_id: { type: 'string' },
  evidence_manifest: { type: 'string' },
  target_guard: { type: 'string' },
  correctness: { type: 'string' },
  build: { type: 'boolean' },
  tree: { type: 'string' },
  head: { type: 'string' },
  patch_file: { type: 'string' },
  changed_files: { type: 'array', items: { type: 'string' } },
  topology: obj({}, []),
  absolute_score: { type: 'number' },
  per_case: perCase,
  activation_on_hardware: { type: 'string' },
  path_marker: { type: 'string' },
  launches: { type: 'number' },
  liveness_replays: { type: 'number' },
  graph_safe: { type: 'string' },
  topology_sig: { type: 'string' },
  provenance: { type: 'string' },
  next_blocker: { type: 'string' },
  notes: { type: 'string' },
  activation: obj({
    mode: { type: 'string' }, switch_name: { type: 'string' }, switch_value: { type: 'string' },
    path_marker: { type: 'string' }, marker_how: { type: 'string' },
    switches: {
      type: 'array',
      items: obj({ switch_name: { type: 'string' }, switch_value: { type: 'string' } }, []),
    },
  }, []),
}, ['candidate_id', 'candidate_source', 'candidate_status', 'claim_complete',
  'contract_failures']);

const EXPERT_SKILL_CONTRACT_VERIFY_SCHEMA = obj({
  candidate_id: { type: 'string' },
  candidate_head: { type: 'string' },
  candidate_tree_digest: { type: 'string' },
  skill_id: { type: 'string' },
  skill_bundle_sha256: { type: 'string' },
  planner_extension_sha256: { type: 'string' },
  claim_complete: { type: 'boolean' },
  report_path: { type: 'string' },
  structural_compatible: { type: 'boolean' },
  independent_structure_pass: { type: 'boolean' },
  capability_eligible: { type: 'boolean' },
  reference_copy_detected: { type: 'boolean' },
  reference_copy_suspected: { type: 'boolean' },
  provenance_status: { type: 'string' },
  contract_revision: { type: 'string' },
  contract_sha256: { type: 'string' },
  contract_failures: { type: 'array', items: obj({
    id: { type: 'string' }, category: { type: 'string' },
    severity: { type: 'string' },
    messages: { type: 'array', items: { type: 'string' } },
  }, ['id', 'category', 'severity', 'messages']) },
  plan_consistent: { type: 'boolean' },
  semantic_features_passed: { type: 'number' },
  semantic_features_total: { type: 'number' },
  hardware_verified: { type: 'boolean' },
  accuracy_verified: { type: 'boolean' },
  performance_verified: { type: 'boolean' },
  next_blocker: { type: 'string' },
  notes: { type: 'string' },
}, [
  'candidate_id', 'candidate_head', 'claim_complete', 'structural_compatible',
  'candidate_tree_digest', 'skill_id', 'skill_bundle_sha256',
  'planner_extension_sha256',
  'independent_structure_pass', 'capability_eligible', 'hardware_verified',
  'accuracy_verified', 'performance_verified', 'plan_consistent',
  'contract_revision', 'contract_sha256', 'contract_failures',
  'failed_required_checks', 'plan_consistency_errors', 'input_errors',
  'provenance_attestation_valid',
]);

// Operator-neutral whole-kernel topology.
const MEGA_TOPOLOGY_SCHEMA = obj({
  launch_count: { type: 'number' },
  included_regions: { type: 'array', items: { type: 'string' } },
  included_queues: { type: 'array', items: { type: 'string' } },
  capabilities: { type: 'array', items: { type: 'string' } },
  require_overlap: { type: 'boolean' },
  parameters: { type: 'object', additionalProperties: true },
  notes: { type: 'string' },
});

const MEGA_ANALYZE_BASE_SCHEMA = obj({
  kernel_type: { type: 'string' }, kernel_file: { type: 'string' }, entry_point: { type: 'string' },
  modifiable_files: { type: 'array', items: { type: 'string' } },
  bottleneck_guess: { type: 'string' }, roadmap_summary: { type: 'string' },
  // The ladder: Analyze's ranked rungs and what each is conditional on. Each rung declares id,
  // gated_on (rungs that must run to spec first; [] is an explicit answer), mandatory_arms and
  // is_positive_control, so roadmapLadderGate can check dispatch order against them; prose cannot be
  // checked.
  candidate_directions: { type: 'array', items: obj({
    id: { type: 'string' },              // short, stable, quotable: D0, D1, ...
    title: { type: 'string' },
    candidate_id: { type: 'string' },
    candidate_source: { type: 'string', enum: ['search', 'integrated'] },
    base_candidate_id: { type: 'string' },
    specialty: { type: 'string' },
    focus_files: { type: 'array', items: { type: 'string' } },
    prompt: { type: 'string' },
    rationale: { type: 'string' },
    rung_deviation: { type: 'string' },
    expected_speedup: { type: 'number' },
    gated_on: { type: 'array', items: { type: 'string' } },
    mandatory_arms: { type: 'array', items: { type: 'string' } },
    is_positive_control: { type: 'boolean' },
    // How a rung closes: terminal on a measurement; enabling on functional acceptance, since a
    // prerequisite step has no standalone speedup. See stepRoleOf.
    step_role: { type: 'string', enum: ['enabling', 'terminal'] },
    enables: { type: 'string' },
    cost_budget_pct: { type: 'number' },
    // A terminal rung must say what shape closes it. Without this a three-launch partial fusion and
    // the requested two-launch terminal are indistinguishable to the rung tracker.
    target_shape: {
      type: 'object',
      properties: {
        launches: { type: 'number' },
        stages_fused: { type: 'array', items: { type: 'string' } },
        require_overlap: { type: 'boolean' },
      },
      required: ['launches'],
      additionalProperties: true,
    },
    target_topology: MEGA_TOPOLOGY_SCHEMA,
  }, ['id', 'title', 'gated_on']) },
  // perf_knowledge mapping (reference only): operator/language and the most relevant card paths.
  // Empty when no card applies.
  kk_operator: { type: ['string', 'null'] }, kk_language: { type: ['string', 'null'] },
  kk_refs: { type: 'array', items: { type: 'string' } },
  baseline_operator_map: { type: 'array', items: obj({
    role: { type: 'string' },
    file: { type: 'string' },
    symbol: { type: 'string' },
    kind: { type: 'string' },
    semantics: { type: 'string' },
    inputs: { type: 'array', items: { type: 'string' } },
    outputs: { type: 'array', items: { type: 'string' } },
    layout: { type: 'string' },
    synchronization: { type: 'string' },
    action: { type: 'string', enum: [
      'reuse', 'extract_shared', 'inline', 'rewrite_equivalent', 'rewire', 'extend',
    ] },
    target_role: { type: 'string' },
    proof: { type: 'string' },
  }, ['role', 'file', 'symbol', 'semantics', 'action']) },
  prior_art: { type: 'array', items: { type: 'object', additionalProperties: true } },
  // Fine-grained dependency graph; taskGraphGate owns semantic validation.
  task_graph: {
    type: ['object', 'null'],
    additionalProperties: true,
    properties: {
      nodes: { type: 'array', items: obj({
        id: { type: 'string' }, stage: { type: 'string' }, duration_us: { type: 'number' },
        // measured | assumed. The gate counts assumed durations, because a graph whose every
        // duration is assumed is a hypothesis wearing a measurement's formatting.
        source: { type: 'string', enum: ['measured', 'assumed'] },
      }, ['id', 'source']) },
      edges: { type: 'array', items: obj({
        from: { type: 'string' }, to: { type: 'string' },
        // data | launch_boundary | barrier | assumed. `launch_boundary` is the load-bearing value:
        // it marks an ordering the DATA does not require, which is the only kind fusion can delete.
        enforced_by: { type: 'string' },
        scope: { type: 'string' },
        fan_in: { type: 'number' },
        // Consumer operands the producer does not supply (weights, scales, descriptors). They can load while
        // the producer runs and have no node in the graph, so they are named here. [] is a real answer.
        producer_independent_operands: { type: 'array', items: { type: 'string' } },
      }, ['from', 'to', 'enforced_by']) },
      unknowns: { type: 'array', items: { type: 'object', additionalProperties: true } },
      critical_path_us: { type: 'number' },
      measured_e2e_us: { type: 'number' },
    },
  },
  // Per-pipe resource timeline, required alongside task_graph: the graph says what MAY overlap, this
  // says which unit is idle and what could fill it.
  //   {pipes: [{stage, pipe, utilization_pct, source}], interkernel_gap_us: {median, max, n_boundaries},
  //    class: throughput_bound|latency_bound|launch_bound|mixed, stall_reason: [...],
  //    idle_pipe_opportunities: [...], closed_axes: [...], unknowns: [...]}
  // `class` rules out lever families early (knowledge/pipe_occupancy.md). The gate drops pipe rows
  // without a finite utilization_pct, so the fields are declared.
  resource_timeline: {
    type: ['object', 'null'],
    additionalProperties: true,
    properties: {
      pipes: { type: 'array', items: obj({
        stage: { type: 'string' },
        pipe: { type: 'string' },   // valu|mfma|vmem|lds|hbm|interconnect|scalar
        utilization_pct: { type: ['number', 'null'] },
        source: { type: 'string' },
      }, ['stage', 'pipe', 'utilization_pct']) },
      interkernel_gap_us: obj({
        median: { type: 'number' }, max: { type: 'number' }, n_boundaries: { type: 'number' },
      }, ['median']),
      // The gate DERIVES the class from the numbers and compares it against this one; a mismatch is
      // itself reportable, so the stated value must be present to be contradicted.
      class: { type: 'string', enum: ['throughput_bound', 'latency_bound', 'launch_bound', 'mixed'] },
      // id, the size fields and rides_on are read by the gate, so they are declared; rides_on records that
      // another rung absorbs this hole.
      idle_pipe_opportunities: { type: 'array', items: obj({
        id: { type: 'string' }, window: { type: 'string' }, idle_pipe: { type: 'string' },
        recoverable_us: { type: ['number', 'null'] }, pct_of_e2e: { type: ['number', 'null'] },
        rides_on: { type: 'string' },
      }, []) },
      closed_axes: { type: 'array', items: { type: 'object', additionalProperties: true } },
      unknowns: { type: 'array', items: { type: 'object', additionalProperties: true } },
    },
  },
  // Lowerable whole-kernel schedule contract. Unlike roadmap prose, every downstream Mega role
  // receives this object verbatim and static Verify checks the authored source against it.
  mega_plan_ir: {
    type: ['object', 'null'],
    additionalProperties: true,
    properties: {
      plan_version: { type: 'string' },
      expert_skill_id: { type: ['string', 'null'] },
      expert_skill_revision: { type: ['string', 'null'] },
      expert_skill_bundle_sha256: { type: ['string', 'null'] },
      expert_skill_planner_extension_sha256: { type: ['string', 'null'] },
      expert_skill_validation_status: { type: ['string', 'null'] },
      expert_skill_auto_apply: { type: ['boolean', 'null'] },
      expert_skill_explicit_pin_modes: {
        type: 'array', items: { type: 'string' },
      },
      target: obj({
        launch_count: { type: 'number' },
        required_regions: { type: 'array', items: { type: 'string' } },
        required_queues: { type: 'array', items: { type: 'string' } },
        required_capabilities: { type: 'array', items: { type: 'string' } },
      }, ['launch_count', 'required_regions', 'required_queues']),
      work_domains: { type: 'array', items: obj({
        id: { type: 'string' }, unit: { type: 'string' },
        extent: { type: 'string' }, index_type: { type: 'string' },
      }, ['id', 'unit', 'extent', 'index_type']) },
      regions: { type: 'array', items: obj({
        id: { type: 'string' }, role: { type: 'string' },
        work_domain: { type: 'string' }, engine: { type: 'string' },
      }, ['id', 'role', 'work_domain']) },
      buffers: { type: 'array', items: {
        type: 'object', additionalProperties: true,
        properties: {
          id: { type: 'string' }, role: { type: 'string' },
          element_type: { type: 'string' }, capacity: { type: 'string' },
          address_space: { type: 'string' }, format: { type: 'string' },
          producers: { type: 'array', items: { type: 'string' } },
          consumers: { type: 'array', items: { type: 'string' } },
          lifetime: { type: 'string' }, alias_group: { type: 'string' },
        },
        required: ['id', 'role', 'element_type', 'capacity', 'address_space'],
      } },
      counters: { type: 'array', items: {
        type: 'object', additionalProperties: true,
        properties: {
          id: { type: 'string' }, element_type: { type: 'string' },
          capacity: { type: 'string' }, scope: { type: 'string' },
          lifecycle: { type: 'string' }, reset_owner: { type: 'string' },
          generation: { type: 'string' },
          producers: { type: 'array', items: { type: 'string' } },
          consumers: { type: 'array', items: { type: 'string' } },
          publish: { type: 'string' }, wait: { type: 'string' },
        },
        required: [
          'id', 'element_type', 'capacity', 'scope', 'lifecycle',
          'producers', 'consumers',
        ],
      } },
      queues: { type: 'array', items: obj({
        id: { type: 'string' }, work_domain: { type: 'string' },
        claim_unit: { type: 'string' }, owner: { type: 'string' },
        shards: { type: 'number' }, stride_bytes: { type: 'number' },
        chunk_policy: { type: 'string' }, ready_when: { type: 'string' },
        priority: { type: 'number' },
      }, ['id', 'work_domain', 'claim_unit', 'owner']) },
      events: { type: 'array', items: obj({
        id: { type: 'string' }, producer: { type: 'string' },
        consumer: { type: 'string' }, scope: { type: 'string' },
        counter: { type: 'string' }, publish: { type: 'string' },
        wait: { type: 'string' },
      }, ['id', 'producer', 'consumer', 'scope']) },
      abi: {
        type: 'object',
        additionalProperties: true,
        properties: {
          entry_point: { type: 'string' },
          arguments: { type: 'array', items: {
            type: 'object', additionalProperties: true,
            properties: {
              id: { type: 'string' }, type: { type: 'string' },
              role: { type: 'string' }, source: { type: 'string' },
              consumers: { type: 'array', items: { type: 'string' } },
              optional: { type: 'boolean' },
            },
            required: ['id', 'type', 'role'],
          } },
          parameters: { type: 'object', additionalProperties: true },
        },
        required: ['entry_point', 'arguments'],
      },
      resources: {
        type: 'object',
        additionalProperties: true,
        properties: {
          arch: { type: 'string' },
          workgroup: obj({
            wave_size: { type: 'number' }, wave_count: { type: 'number' },
            thread_count: { type: 'number' },
          }, ['wave_size', 'wave_count', 'thread_count']),
          local_memory: obj({
            total_bytes: { type: 'number' },
            limit_bytes: { type: 'number' },
            allocation_rule: { type: 'string' },
            allocations: { type: 'array', items: {
              type: 'object', additionalProperties: true,
            } },
          }, ['total_bytes', 'limit_bytes', 'allocation_rule', 'allocations']),
          registers: obj({
            vgpr_max: { type: ['number', 'null'] },
            sgpr_max: { type: ['number', 'null'] },
            scratch_bytes_max: { type: ['number', 'null'] },
            source: { type: 'string' },
          }, ['vgpr_max', 'sgpr_max', 'scratch_bytes_max', 'source']),
          occupancy: obj({
            compute_units: { type: 'number' },
            grid_blocks: { type: 'number' },
            min_workgroups_per_cu: { type: 'number' },
            requires_full_grid_residency: { type: 'boolean' },
          }, [
            'compute_units', 'grid_blocks', 'min_workgroups_per_cu',
            'requires_full_grid_residency',
          ]),
          engines: { type: 'array', items: obj({
            region: { type: 'string' }, pipe: { type: 'string' },
          }, ['region', 'pipe']) },
          parameters: { type: 'object', additionalProperties: true },
        },
        required: [
          'arch', 'workgroup', 'local_memory', 'registers', 'occupancy', 'engines',
        ],
      },
      schedule: {
        type: 'object',
        additionalProperties: true,
        properties: {
          primary_loop: obj({
            kind: { type: 'string' },
            carried_state: { type: 'array', items: { type: 'string' } },
            queue_priority: { type: 'array', items: { type: 'string' } },
            queue_selection: { type: 'string' },
            progress_invariants: { type: 'array', items: { type: 'string' } },
          }, ['kind', 'carried_state', 'progress_invariants']),
          policies: { type: 'object', additionalProperties: true },
          parameters: { type: 'object', additionalProperties: true },
        },
        required: ['primary_loop', 'policies'],
      },
      compiler_constraints: { type: 'array', items: {
        type: 'object', additionalProperties: true,
        properties: { id: { type: 'string' }, rule: { type: 'string' } },
        required: ['id', 'rule'],
      } },
      evidence_requirements: { type: 'object', additionalProperties: true },
      known_unknowns: { type: 'array', items: { type: 'object', additionalProperties: true } },
    },
    required: [
      'plan_version', 'target', 'work_domains', 'regions', 'buffers', 'counters',
      'queues', 'events', 'abi', 'resources', 'schedule',
      'compiler_constraints', 'evidence_requirements', 'known_unknowns',
    ],
  },
}, ['kernel_type', 'roadmap_summary']);
const MEGA_ANALYZE_SCHEMA = {
  ...MEGA_ANALYZE_BASE_SCHEMA,
  properties: {
    ...MEGA_ANALYZE_BASE_SCHEMA.properties,
    candidate_directions: {
      ...MEGA_ANALYZE_BASE_SCHEMA.properties.candidate_directions,
      items: {
        ...MEGA_ANALYZE_BASE_SCHEMA.properties.candidate_directions.items,
        required: [...new Set([
          ...(MEGA_ANALYZE_BASE_SCHEMA.properties.candidate_directions.items.required || []),
          'candidate_id', 'candidate_source', 'base_candidate_id', 'specialty',
          'focus_files', 'prompt', 'target_shape', 'target_topology',
        ])],
      },
    },
  },
  required: [...new Set([
    ...(MEGA_ANALYZE_BASE_SCHEMA.required || []),
    'modifiable_files', 'candidate_directions', 'prior_art',
    'baseline_operator_map', 'task_graph', 'resource_timeline', 'mega_plan_ir',
  ])],
};

const MEGA_BENCH_SCHEMA = obj({
  commandment_path: { type: 'string' }, correctness_cmd: { type: 'string' },
  benchmark_cmd: { type: 'string' }, profile_cmd: { type: 'string' }, parse_hint: { type: 'string' },
  baseline_per_case: { type: 'array', items: { type: 'object', additionalProperties: true } },
  baseline_geomean_ms: { type: 'number' }, num_test_cases: { type: 'number' },
  // Workload-aligned outputs (present with a workload spec): metric = Σweight / Σ(weight/speedup);
  // baseline_weighted_total_ms is the baseline time the weights represent.
  workload_aligned: { type: 'boolean' },
  baseline_weighted_total_ms: { type: 'number' },
  weights_provenance: { type: 'string' }, // e.g. "trace" | "regime" | "regime_floor" | "prior" | "caller" | "mixed"
  // args.positive_control result. switch_present=false means both arms ran the same code. Fields are
  // declared so a dropped one cannot validate as a passing control.
  positive_control: obj({
    ran: { type: 'boolean' },
    claim_complete: { type: 'boolean' },
    attempt_id: { type: 'string' },
    evidence_manifest: { type: 'string' },
    switch_present: { type: ['boolean', 'null'] },
    switch_checked: { type: 'string' },
    measured_pct: { type: ['number', 'null'] },
    expected_lo: { type: 'number' }, expected_hi: { type: 'number' },
    passed: { type: 'boolean' }, reps: { type: 'number' },
    control_pairs_pct: { type: 'array', items: { type: 'number' } },
    null_pairs_pct: { type: 'array', items: { type: 'number' } },
    null_arm_pct: { type: 'number' },
    note: { type: 'string' },
    // Required only when the object exists at all: a run with no control omits it entirely. When it
    // IS reported, these three are what separate a control from a number, so none may be defaulted.
  }, ['ran', 'claim_complete', 'switch_present', 'measured_pct']),
  reliable: { type: 'boolean' }, notes: { type: 'string' },
}, ['commandment_path', 'baseline_per_case', 'baseline_geomean_ms']);

// Fast-test cache load: the agent recomputes the validity key from disk (frozen base revision, harness
// hash, control spec, guards) and compares it with the stored key; only an exact match (key_valid)
// reuses prior measurements. bench/analysis have the same shape as the live Benchmark/Profile results.
const FAST_TEST_LOAD_SCHEMA = obj({
  cache_present: { type: 'boolean' },   // any cache artifact found at all
  key_valid: { type: 'boolean' },       // cache present AND its stored key matches the current key
  commandment_materialized: { type: 'boolean' },
  commandment_current_identity: { type: 'boolean' },
  current_key: { type: 'string' }, cached_key: { type: 'string' },
  note: { type: 'string' },
  bench: { type: 'object', additionalProperties: true },     // reconstructed BENCH_SCHEMA payload (CACHE_KIND=bench)
  analysis: { type: 'object', additionalProperties: true },  // reconstructed MEGA_ANALYSIS payload (CACHE_KIND=profile)
}, [
  'cache_present', 'key_valid', 'commandment_materialized',
  'commandment_current_identity',
]);

// Fast-test cache publish: copy this wave's fresh artifacts into STATE_DIR/fast_test_cache and write
// fast_test_key.json with the validity key computed from disk.
const FAST_TEST_PUBLISH_SCHEMA = obj({
  published: { type: 'boolean' }, key: { type: 'string' },
  cache_dir: { type: 'string' }, files: { type: 'array', items: { type: 'string' } },
  note: { type: 'string' },
}, ['published']);

const ANALYSIS_RESULT_SCHEMA = obj({
  status: { type: 'string', enum: ['ready', 'degraded'] },
  analysis_skill: { type: 'string' },
  analysis_schema_version: { type: 'string' },
  analysis_status: {
    type: 'string',
    enum: ['awaiting_measurement', 'evidence_complete', 'unavailable'],
  },
  analysis_json: { type: 'string' },
  failure_note: { type: 'string' },
}, [
  'status',
  'analysis_skill',
  'analysis_schema_version',
  'analysis_status',
  'analysis_json',
  'failure_note',
]);

const MEGA_PLAN_BASE_SCHEMA = obj({
  stop: { type: 'boolean' }, reasoning: { type: 'string' },
  directions: {
    type: 'array',
    items: obj({
      id: { type: 'string' }, title: { type: 'string' },
      specialty: { type: 'string', enum: ['algorithm', 'memory', 'compute', 'host_runtime', 'distributed', 'deep_explore'] },
      focus_files: { type: 'array', items: { type: 'string' } },
      expected_speedup: { type: 'number' }, prompt: { type: 'string' },
      kk_refs: { type: 'array', items: { type: 'string' } }, // optional: perf_knowledge card paths for THIS direction (REFERENCE ONLY)
      // Binds the direction to Analyze's pipe table: fills_pipe is the unit it loads or relieves,
      // pipe_util_pct that unit's measured utilization, headroom_basis which bound caps the claim (idle
      // fraction or measured_e2e_us - critical_path_us). See knowledge/pipe_occupancy.md.
      fills_pipe: { type: 'string' },
      pipe_util_pct: { type: 'number' },
      headroom_basis: { type: 'string' },
      graph_refs: { type: 'array', items: { type: 'string' } }, // task_graph node/edge ids this direction acts on
      // The idle_pipe_opportunities id this direction fills, so each hole can be checked as owned or not.
      fills_hole: { type: 'string' },
      // Binds the direction to the ladder: roadmap_rung is the rung id or `off_ladder`. rung_deviation is
      // required when the rung is off the ladder, out of order, or has unmet gated_on: say what is skipped
      // and what happens to it. Skipping is allowed; skipping silently is not.
      roadmap_rung: { type: 'string' },
      rung_deviation: { type: 'string' },
      // Step kind: terminal (default) is judged on speed; enabling is a prerequisite in a multi-step fusion,
      // judged on function (builds, path taken, correct, no deadlock). enables names the terminal rung and is
      // required for enabling steps; cost_budget_pct is the slowdown this step is allowed.
      step_role: { type: 'string', enum: ['enabling', 'terminal'] },
      enables: { type: 'string' },
      cost_budget_pct: { type: 'number' },
      gated_on: { type: 'array', items: { type: 'string' } },
      mandatory_arms: { type: 'array', items: { type: 'string' } },
      target_shape: {
        type: 'object',
        properties: {
          launches: { type: 'number' },
          stages_fused: { type: 'array', items: { type: 'string' } },
          require_overlap: { type: 'boolean' },
        },
        required: ['launches'],
        additionalProperties: true,
      },
      // Operator-neutral multi-lever descriptor for a whole-kernel candidate.
      // Skill-specific region/queue/capability names remain data values.
      target_topology: MEGA_TOPOLOGY_SCHEMA,
      candidate_id: { type: 'string' },
      candidate_source: { type: 'string' },
      base_candidate_id: { type: 'string' },
    }, ['id', 'title', 'specialty', 'prompt']),
  },
}, ['stop', 'directions']);

const MEGA_PLAN_SCHEMA = {
  ...MEGA_PLAN_BASE_SCHEMA,
  properties: {
    ...MEGA_PLAN_BASE_SCHEMA.properties,
    directions: {
      ...MEGA_PLAN_BASE_SCHEMA.properties.directions,
      items: {
        ...MEGA_PLAN_BASE_SCHEMA.properties.directions.items,
        properties: {
          ...MEGA_PLAN_BASE_SCHEMA.properties.directions.items.properties,
          candidate_source: {
            type: 'string', enum: ['search', 'integrated'],
          },
        },
        required: [...new Set([
          ...(MEGA_PLAN_BASE_SCHEMA.properties.directions.items.required || []),
          'candidate_id', 'candidate_source', 'base_candidate_id',
          'focus_files', 'roadmap_rung', 'step_role', 'gated_on',
          'target_shape', 'target_topology',
        ])],
      },
    },
  },
  required: [...new Set([...(MEGA_PLAN_BASE_SCHEMA.required || []), 'reasoning'])],
};

const MEGA_VERIFY_SCHEMA = obj({
  status: { type: 'string' }, correctness: { type: 'string' },
  claim_complete: { type: 'boolean' }, attempt_id: { type: 'string' },
  timeout_recovery_safe: { type: 'boolean' },
  active_gpu_processes: { type: 'number' },
  lane_lock_free: { type: 'boolean' },
  evidence_complete: { type: 'boolean' },
  evidence_manifest: { type: 'string' }, candidate_id: { type: 'string' },
  candidate_source: { type: 'string' }, candidate_tree: { type: 'string' },
  candidate_head: { type: 'string' },
  activation: obj({
    mode: { type: 'string' }, switch_name: { type: 'string' }, switch_value: { type: 'string' },
    path_marker: { type: 'string' }, marker_how: { type: 'string' },
    switches: {
      type: 'array',
      items: obj({ switch_name: { type: 'string' }, switch_value: { type: 'string' } }, []),
    },
  }, []),
  verified_geomean: { type: ['number', 'null'] }, verified_arithmetic: { type: ['number', 'null'] },
  verified_weighted: { type: ['number', 'null'] }, // time-weighted ratio-of-sums (PRIMARY when workload_aligned)
  per_case: perCase, variance_note: { type: 'string' }, notes: { type: 'string' },
  graph_safe: { type: 'string' },
  liveness: { type: 'string' }, // pass|fail|n/a — deadlock/stale-read stress (distributed specialty)
  replay_count: { type: 'number' },
  replay_results: { type: 'array', items: obj({
    guard: { type: 'string' }, count: { type: 'number' },
    status: { type: 'string' }, graph_safe: { type: 'string' },
  }, ['guard', 'count', 'status']) },
  arms_run: { type: 'array', items: { type: 'string' } },
  // Accuracy as a number (metric, value, threshold, guard/route) so functionalAcceptance can check
  // value < threshold. Missing fields fall back to the `correctness` string check.
  accuracy: obj({
    metric: { type: 'string' }, value: { type: 'number', minimum: 0 },
    threshold: { type: 'number', exclusiveMinimum: 0 },
    guard: { type: 'string' }, method: { type: 'string' },
  }, []),
  accuracy_results: { type: 'array', items: obj({
    metric: { type: 'string' }, value: { type: 'number', minimum: 0 },
    threshold: { type: 'number', exclusiveMinimum: 0 },
    guard: { type: 'string' }, method: { type: 'string' },
  }, ['metric', 'value', 'threshold', 'guard']) },
  // Launches per call for base and candidate, read paired; target is caller-owned; how_counted is the
  // evidence (trace record count, marker tally). Missing fields leave the launch criterion unjudged.
  launch_shape: obj({
    launches_base: { type: 'number' }, launches_cand: { type: 'number' },
    per_rank: { type: 'boolean' }, target: { type: 'number' },
    stages_fused: { type: 'array', items: { type: 'string' } }, how_counted: { type: 'string' },
  }, []),
  // reps = interleaved A/B pairs; null_arm_pct = the delta of an arm doing identical work (the noise
  // floor). A win smaller than |null_arm_pct| is unreadable. Missing or low values make the result
  // provisional rather than void.
  reps: { type: 'number' }, null_arm_pct: { type: 'number' },
  // Raw per-pair readings tagged by guard/route, so the driver can split caller-declared bimodal guards
  // by state (modeSplit/pairedModeAware) and check the number came from a target guard. Absent:
  // verified_geomean is used as before.
  paired_readings: { type: 'array', items: obj({
    guard: { type: 'string' }, base: { type: 'number' }, cand: { type: 'number' },
  }, ['guard', 'base', 'cand']) },
  // Did the patched path execute in the measured run? yes requires the observed path_marker; no or
  // unknown voids the comparison rather than making it negative.
  activation_confirmed: { type: 'string' },  // yes|no|unknown
  activation_evidence: { type: 'string' },   // the command + the marker output that proves it
  // Did the switched path run on the device this round? yes requires a gpu_lock-wrapped run with the
  // switch set and a device-side observable (timing, trace record, on-device marker). Compile-only or
  // CPU checks are no; n/a means there is no switched path. With REQUIRE_HW_ACTIVATION, a committable or
  // enabling candidate must be yes.
  activation_on_hardware: { type: 'string' }, // yes|no|n/a|unknown
  hardware_evidence: { type: 'string' },       // the gpu_lock cmd + the device-side observable it produced
  // A marker proves the host path ran, not that the arms compiled differently: a JIT disk cache can
  // serve one binary to both arms if the switch is not in the cache key. For JIT candidates Verify
  // reports cache keys / IR-ISA hashes and they must differ; n/a is for non-JIT candidates.
  artifact_distinct: { type: 'string' },     // yes|no|n/a|unknown
  artifact_hash_base: { type: 'string' },
  artifact_hash_candidate: { type: 'string' },
  // Files the patch touches. Required: an empty list would read as "touches nothing" and make the
  // patch look combinable. The candidate shelf uses it to decide whether a patch still applies.
  touched_files: { type: 'array', items: { type: 'string' } },
  // Overlap (see knowledge/overlap_instrument.md). After fusion there is one trace record, so overlap
  // needs its own meter. measured is three-valued: no = measured and none found; unknown = not
  // measured.
  overlap: obj({
    measured: { type: 'string', enum: ['yes', 'no', 'unknown'] },
    fraction: { type: 'number' },        // wall-clock, >=2 distinct roles active
    cu_fraction: { type: 'number' },     // CU-weighted; wall-clock alone is manufacturable
    method: { type: 'string' },
    // The meter's controls: base_reading on the unfused path must read ~0 (else the meter is broken);
    // forced_reading must read high (else the meter is dead).
    base_reading: { type: 'number' },
    forced_reading: { type: 'number' },
    clock_skew_ns: { type: 'number' },   // s_memrealtime coherence across XCDs — assumed at your peril
    meter_overhead_pct: { type: 'number' },
    note: { type: 'string' },
  }, ['measured']),
  // Attribution (see knowledge/fusion_preconditions.md): for any patch that adds, removes or merges a
  // launch, report changed_us (patched kernel) and replaced_sum_us (the kernels it replaces), paired at
  // the same guard. An end-to-end win can come from inter-kernel gaps rather than the kernel. A sign
  // mismatch rejects under promotion_metric=changed_kernel; under operator_e2e it only withholds the
  // mechanism claim.
  attribution: obj({
    changed_us: { type: 'number' },
    replaced_sum_us: { type: 'number' },
    guard: { type: 'string' },
    // e2e minus the sum of all kernel times, both arms. Where a gap-win hides.
    residual_ms_base: { type: 'number' },
    residual_ms_cand: { type: 'number' },
    method: { type: 'string' },
    absolute_to_frozen: { type: 'boolean' },
    note: { type: 'string' },
  }, []),
}, ['status', 'verified_geomean', 'touched_files']);

const MEGA_MEMORY_SCHEMA = obj({
  insights: { type: 'array', items: { type: 'string' } },
  candidate_registry: { type: 'array', items: { type: 'object', additionalProperties: true } },
  measurement_calibration: { type: 'object', additionalProperties: true },
  state_written: { type: 'boolean' }, state_round: { type: 'number' },
  state_path: { type: 'string' }, state_generation: { type: 'string' },
  state_sequence: { type: 'number' },
  // memoryMerge keys and dedupes rows by id and reads the ladder from roadmap_rung; a row without
  // verdict is treated as untried.
  ledger: { type: 'array', items: obj({
    id: { type: 'string' }, title: { type: 'string' }, specialty: { type: 'string' },
    // confirmed | partial | unresolved | dead_end — see tech_lead.md "Verdict by SPEC, not by
    // result". `unresolved` is what keeps a rung re-openable instead of closing everything above it.
    verdict: { type: 'string', enum: ['confirmed', 'partial', 'unresolved', 'dead_end'] },
    roadmap_rung: { type: ['string', 'null'] },
    expected: { type: 'number' }, actual: { type: 'number' },
    lesson: { type: 'string' },
    first_round: { type: 'number' }, last_round: { type: 'number' },
  }, ['id', 'verdict']) },
  // Durable hardware/toolchain facts (lowerings true for every kernel on this box), as opposed to
  // insights about this operator. knowledge/ is read-only to the script, so candidates are surfaced in
  // the log and report for a human to merge. See tech_lead.md for the bar.
  knowledge_delta: { type: 'array', items: obj({
    fact: { type: 'string' },
    // Which card it belongs in, so the merge is a placement decision and not a search.
    card: { type: 'string' },
    // The artifact that establishes it: an ISA dump path, a counter, a sha256, a measured number.
    // A fact whose evidence is "the engineer said so" is an insight, not a knowledge delta.
    evidence: { type: 'string' },
    // What a reader would have believed without this fact; a lowering earns a card only when the
    // source-level reading says otherwise.
    contradicts: { type: 'string' },
    // true = holds for any kernel on this target; false = this operator's shape. Only the first
    // kind is a card edit; the second belongs in the report and nowhere else.
    generalizes: { type: 'boolean' },
  }, ['fact', 'evidence', 'generalizes']) },
  bottleneck_now: { type: 'string' }, suggest_next: { type: 'string' },
}, ['insights']);

const MEGA_SELECTION_SCHEMA = obj({
  claim_complete: { type: 'boolean' },
  attempt_id: { type: 'string' },
  evidence_manifest: { type: 'string' },
  selected_candidate_id: { type: 'string' },
  selected_source: { type: 'string' },
  selected_tree: { type: 'string' },
  selected_head: { type: 'string' },
  materialized_from_head: { type: 'string' },
  selected_score: { type: 'number' },
  selected_patch: { type: 'string' },
  candidates: { type: 'array', items: obj({
    candidate_id: { type: 'string' }, source: { type: 'string' },
    tree: { type: 'string' }, head: { type: 'string' }, score: { type: 'number' },
    status: { type: 'string' },
    touched_files: { type: 'array', items: { type: 'string' } },
    activation: obj({
      mode: { type: 'string' }, switch_name: { type: 'string' }, switch_value: { type: 'string' },
      path_marker: { type: 'string' }, marker_how: { type: 'string' },
      switches: {
        type: 'array',
        items: obj({ switch_name: { type: 'string' }, switch_value: { type: 'string' } }, []),
      },
    }, []),
    correctness: { type: 'string' }, guards_pass: { type: 'boolean' },
    path_marker: { type: 'string' }, path_marker_count: { type: 'number' },
    world_size: { type: 'number' },
    launches: { type: 'number' }, launches_base: { type: 'number' },
    graph_safe: { type: 'string' }, artifact_distinct: { type: 'string' },
    artifact_hash_base: { type: 'string' }, artifact_hash_candidate: { type: 'string' },
    overlap_measured: { type: 'string' }, overlap_fraction: { type: 'number' },
    overlap_cu_fraction: { type: 'number' },
    overlap_base_reading: { type: 'number' }, overlap_forced_reading: { type: 'number' },
    attribution_complete: { type: 'boolean' },
    attribution: obj({
      changed_us: { type: 'number' }, replaced_sum_us: { type: 'number' },
      residual_ms_base: { type: 'number' }, residual_ms_cand: { type: 'number' },
      guard: { type: 'string' }, method: { type: 'string' },
      absolute_to_frozen: { type: 'boolean' },
    }, []),
    accuracy_metric: { type: 'string' }, accuracy_value: { type: 'number' },
    accuracy_results: { type: 'array', items: obj({
      metric: { type: 'string' }, value: { type: 'number' },
      threshold: { type: 'number' }, guard: { type: 'string' },
      method: { type: 'string' },
    }, ['metric', 'value', 'guard']) },
    liveness_replays: { type: 'number' }, claim_complete: { type: 'boolean' },
    replay_results: { type: 'array', items: obj({
      guard: { type: 'string' }, count: { type: 'number' },
      status: { type: 'string' }, graph_safe: { type: 'string' },
      arrival_jitter: { type: 'boolean' },
    }, ['guard', 'count', 'status']) },
    paired_readings: { type: 'array', items: obj({
      guard: { type: 'string' }, base: { type: 'number' }, cand: { type: 'number' },
    }, ['guard', 'base', 'cand']) },
    null_arm_pct: { type: 'number' },
    per_case: perCase,
    reason: { type: 'string' },
  }, ['candidate_id', 'claim_complete']) },
  notes: { type: 'string' },
}, ['claim_complete', 'selected_candidate_id', 'selected_tree']);

const MEGA_IDENTITY_SCHEMA = obj({
  passed: { type: 'boolean' },
  selected_candidate_id: { type: 'string' },
  source_head: { type: 'string' },
  materialized_head: { type: 'string' },
  note: { type: 'string' },
}, ['passed', 'selected_candidate_id', 'source_head', 'materialized_head']);

const MEGA_VALIDATE_SCHEMA = obj({
  kernel_name: { type: 'string' },
  director_verified_speedup_geomean: { type: 'number' },
  director_verified_speedup_arithmetic: { type: 'number' },
  director_verified_speedup_weighted: { type: 'number' }, // PRIMARY when workload_aligned
  tech_lead_reported_speedup_geomean: { type: 'number' },
  validation_status: { type: 'string' }, correctness: { type: 'string' },
  autonomy_acceptance_confirmed: { type: 'boolean' },
  workflow_unchanged: { type: 'boolean' },
  workflow_revision_end: { type: 'string' },
  materialization_matches: { type: 'boolean' },
  selected_source_head: { type: 'string' },
  selected_materialized_head: { type: 'string' },
  final_patch_verified: { type: 'boolean' },
  attribution: obj({
    changed_us: { type: 'number' }, replaced_sum_us: { type: 'number' },
    guard: { type: 'string' }, residual_ms_base: { type: 'number' },
    residual_ms_cand: { type: 'number' }, method: { type: 'string' },
    absolute_to_frozen: { type: 'boolean' },
  }, []),
  per_case: perCase, applied_to_original: { type: 'string' },
  // Count of GPU-lease jobs still alive for this EVAL_DIR when validation finished. Non-zero means
  // some direction backgrounded a lease and produced a measurement that is NOT in the report.
  orphan_leases_swept: { type: 'number' },
  arbitration_note: { type: 'string' }, final_patch: { type: 'string' },
}, ['director_verified_speedup_geomean', 'validation_status', 'correctness']);

// ---------------------------------------------------------------------------
// mode=mega helpers and run state
// ---------------------------------------------------------------------------
function megaNowMs() { return WORKFLOW_STARTED_MS + MEGA_CLOCK_MS; }
function megaAdvanceMs(ms) { MEGA_CLOCK_MS += Math.max(0, Number(ms) || 0); }
function megaChargeMs(allocatedMs) { megaAdvanceMs(allocatedMs); }

// <<REPLAY:structural_evidence_identity>>
function structuralEvidenceIdentityVerdict(report, expected) {
  const r = report || {};
  const e = expected || {};
  const reasons = [];
  const requireEqual = (field, wanted, label = field) => {
    if (String(r[field] || '') !== String(wanted || '')) {
      reasons.push(`${label} mismatch: got ${String(r[field] || '(missing)')}, ` +
        `expected ${String(wanted || '(missing)')}`);
    }
  };
  requireEqual('candidate_id', e.candidateId, 'candidate id');
  requireEqual('candidate_head', e.candidateHead, 'candidate HEAD');
  requireEqual('skill_id', e.skillId, 'Skill id');
  requireEqual('contract_revision', e.skillRevision, 'Skill revision');
  if (e.contractSha256) {
    requireEqual('contract_sha256', e.contractSha256, 'contract digest');
  }
  if (e.plannerExtensionSha256) {
    requireEqual(
      'planner_extension_sha256',
      e.plannerExtensionSha256,
      'Planner Extension digest'
    );
  }
  if (e.skillBundleSha256) {
    requireEqual('skill_bundle_sha256', e.skillBundleSha256, 'Skill bundle digest');
  }
  if (!String(r.candidate_tree_digest || '')) {
    reasons.push('candidate tree digest missing');
  }
  return { pass: reasons.length === 0, reasons };
}
// <</REPLAY:structural_evidence_identity>>

// <<REPLAY:mega_topology_contract>>
function topologyLaunchCount(topo) {
  if (!topo || typeof topo !== 'object') return NaN;
  const value = topo.launch_count != null ? topo.launch_count : topo.launches;
  return Number(value);
}

function candidateClaimsPlanTarget(direction, planIr) {
  const planned = Number(planIr && planIr.target && planIr.target.launch_count);
  const declared = topologyLaunchCount(direction && direction.target_topology);
  // launch_count is an upper bound.
  return Number.isFinite(planned) && Number.isFinite(declared) && declared >= 1 &&
    declared <= planned;
}

function megaEffectiveLaneCheckpoint(candidate) {
  const c = candidate || {};
  const w = c.working_snapshot && typeof c.working_snapshot === 'object'
    ? c.working_snapshot : {};
  return String(c.working_head || '') && String(w.head || '') === String(c.working_head)
    ? w : c;
}

function authoringContractNeedsPreflight(
  direction, planIr, candidate, contractEnabled, expected = {}
) {
  if (!contractEnabled || !candidateClaimsPlanTarget(direction, planIr)) return false;
  const c = megaEffectiveLaneCheckpoint(candidate);
  const identityMatches =
    (!expected.skillId || c.structural_skill_id === expected.skillId) &&
    (!expected.revision || c.structural_contract_revision === expected.revision) &&
    (!expected.bundle || c.structural_skill_bundle_sha256 === expected.bundle) &&
    (!expected.planner ||
      c.structural_planner_extension_sha256 === expected.planner) &&
    (!expected.contract || c.structural_contract_sha256 === expected.contract);
  return !(c.structural_verified === true &&
    String(c.structural_candidate_head || '') &&
    String(c.structural_candidate_head) === String(c.head || '') &&
    identityMatches);
}

// Convert the operator-neutral descriptor into Verify's historical TARGET_SHAPE vocabulary.
function megaShapeFromTopology(topo) {
  if (!topo || typeof topo !== 'object') {
    return { launches: LAUNCH_TARGET, stages_fused: [], require_overlap: false };
  }
  return {
    launches: Number.isFinite(topologyLaunchCount(topo))
      ? topologyLaunchCount(topo) : LAUNCH_TARGET,
    stages_fused: Array.isArray(topo.included_regions)
      ? topo.included_regions.map(String)
      : (Array.isArray(topo.fused_stages) ? topo.fused_stages.map(String) : []),
    queues: Array.isArray(topo.included_queues) ? topo.included_queues.map(String) : [],
    capabilities: Array.isArray(topo.capabilities) ? topo.capabilities.map(String) : [],
    require_overlap: topo.require_overlap === true,
    parameters: topo.parameters && typeof topo.parameters === 'object'
      ? { ...topo.parameters } : {},
  };
}

function megaTopologyVerdict(direction, planIr, allowPartialFusion = true) {
  const topo = direction && direction.target_topology;
  if (!topo || typeof topo !== 'object') {
    return { pass: false, reason: 'target_topology is missing' };
  }
  if (direction.topology_inferred === true) {
    return { pass: false, reason:
      'planner omitted target_topology; orchestrator inferred a compatibility fallback' };
  }
  const launches = topologyLaunchCount(topo);
  const regions = Array.isArray(topo.included_regions)
    ? topo.included_regions.filter(Boolean)
    : (Array.isArray(topo.fused_stages) ? topo.fused_stages.filter(Boolean) : []);
  if (!Number.isFinite(launches) || launches < 1 || !regions.length) {
    return { pass: false, reason:
      `target_topology must declare finite launch_count and non-empty included_regions; got ` +
      `${JSON.stringify(topo)}` };
  }
  const terminal = String(direction.step_role || 'terminal') === 'terminal';
  // A direction naming its bucket is held to that bucket's target (one per regime), else to `target`.
  const bt = planIr && planIr.target_by_bucket;
  const target = bt && bt[String(direction.bucket)] || planIr && planIr.target || {};
  const planned = Number(target.launch_count); // an upper bound
  if (terminal && Number.isFinite(planned) && launches > planned &&
      !allowPartialFusion) {
    return { pass: false, reason:
      `partial fusion is disabled: terminal declares ${launches} launches, target is <= ${planned}` };
  }
  if (terminal && Number.isFinite(planned) && launches > planned &&
      !String(direction.rung_deviation || '').trim()) {
    return { pass: false, reason:
      `terminal topology declares ${launches} launches but mega_plan_ir target is ${planned}; ` +
      'declare rung_deviation for an intentional partial-fusion terminal' };
  }
  const requireAll = (declared, required, label) => {
    const have = new Set(Array.isArray(declared) ? declared.map(String) : []);
    const missing = (Array.isArray(required) ? required : [])
      .map(String).filter((value) => !have.has(value));
    return missing.length ? `${label} missing ${missing.join(',')}` : '';
  };
  if (terminal && launches <= planned) {
    const regionError = requireAll(regions, target.required_regions, 'included_regions');
    if (regionError) return { pass: false, reason: regionError };
    const queueError = requireAll(
      topo.included_queues, target.required_queues, 'included_queues');
    if (queueError) return { pass: false, reason: queueError };
    const capabilityError = requireAll(
      topo.capabilities, target.required_capabilities, 'capabilities');
    if (capabilityError) return { pass: false, reason: capabilityError };
  }
  return { pass: true, reason: '' };
}
// <</REPLAY:mega_topology_contract>>

// Add Skill context only to roles that consume it.
function megaExpertSkillsBlock(role, phase) {
  if (!USE_EXPERT_SKILLS || !MEGA_EXPERT_SKILL_ROLES.has(role) || !EXPERT_SKILLS_DIR) return '';
  if (EXPERT_SKILL_DIR) {
    if (!['mega_search_lead', 'engineer', 'deep_engineer'].includes(role)) return '';
    const guide = role === 'mega_search_lead'
      ? (phase === 'analyze' ? EXPERT_SKILL_PLANNER_EXTENSION_FILE
        : EXPERT_SKILL_PLANNER_GUIDE_FILE)
      : EXPERT_SKILL_AUTHOR_GUIDE_FILE;
    return `\n\n## MEGA EXPERT SKILL — ${
      EXPERT_SKILL_VALIDATION_STATUS === 'experimental'
        ? 'EXPERIMENTAL TRANSFER, EXPLICIT PIN' : 'VALIDATED PRIOR'
    }\n` +
      `Read ${guide}. ${
        role === 'mega_search_lead' && phase === 'analyze'
          ? 'Use its canonical Planner Extension to derive MegaPlanIR.'
          : 'Do not read or restate the full Skill contract; independent Verify owns it.'
      } ` +
      `The Skill is one validated solution for its stated size range (its skill.md "When to use"): ` +
      `inside that range treat it as a seed recipe — reproduce it first, then search beyond it; ` +
      `outside the range use it only for mechanisms${MRI.MEGA_REGIMES_GUIDE ? ' and let MEGA_REGIMES_GUIDE decide what transfers' : ''}. ` +
      `validation_status=${
        EXPERT_SKILL_VALIDATION_STATUS || 'legacy_validated'}, usage=${
        EXPERT_SKILL_USAGE || 'legacy'}. Candidate source remains search/integrated.` +
      (role === 'mega_search_lead' && REQUIRE_EXPERT_SKILL_BUNDLE_IDENTITY
        ? ` Before using the Skill, run python ${EXPERT_SKILL_BUNDLE_TOOL} ${EXPERT_SKILL_ID} ` +
          `--emit-bundle and require bundle_sha256=${EXPERT_SKILL_BUNDLE_SHA256}, ` +
          `planner_extension_sha256=${EXPERT_SKILL_PLANNER_EXTENSION_SHA256}, and ` +
          `contract_sha256=${EXPERT_SKILL_CONTRACT_SHA256}. A mismatch is a RunContract failure, ` +
          `not a reason to reinterpret stale knowledge.`
        : '');
  }
  return `\n\n## Expert skills (ADVISORY — opt-in, enabled this run)\n` +
    `Also Read ${WORKFLOW_DIR}/roles/mega/_fragments/expert_skills.md and follow it: query ` +
    `${EXPERT_SKILLS_DIR}/index.yaml for skills whose \`match\` fits this op (operator/dtype/regime, and ` +
    `from_backend->to_backend for migration skills) and whose validation_status is \`validated\` and ` +
    `auto_apply is true, and ` +
    `treat each as a HIGH-PRIOR candidate to reproduce — advisory only, never overriding your isolated ` +
    `A/B vs the oracle, never reducing a result below the measured baseline.` +
    // Capability evaluation may use mechanisms, never reference addresses.
    (CAPABILITY_EVAL
      ? ` **Capability eval: use a skill's MECHANISM and MEASUREMENTS, never its addresses.** Its ` +
        `Sources/Procedure sections may cite a branch, commit, or sibling checkout containing the ` +
        `finished implementation. Do not read, port, copy, or diff against any of them — write the ` +
        `code yourself from the described mechanism. A patch containing a file byte-identical to ` +
        `such a reference is rejected at verify as \`plagiarized\`, whatever it measures.`
      : '');
}

function analysisSkillBlock(role, phase) {
  if (!ANALYSIS_SKILL_ON) return '';
  if (role === 'analysis_engineer') {
    return `\n\n## Profile-analysis skill (ANALYSIS ONLY — explicitly enabled this run)\n` +
      `Read ${ANALYSIS_SKILL_INPUTS.ANALYSIS_SKILL_DIR}/SKILL.md, execute only its checked-in ` +
      `builder/runner/analyzer/validator chain, and return the validated artifact contract. ` +
      `Never improvise arithmetic or change the generic Profile result.`;
  }
  if (role === 'tech_lead' && phase === 'plan_round') {
    return `\n\n## Profile-analysis evidence (explicitly enabled this run)\n` +
      `Inspect PROFILE_SUMMARY.analysis_result. Only Read analysis_json when status=ready and ` +
      `analysis_schema_version is the Skill's current schema. If analysis_status=awaiting_measurement, ` +
      `use findings only at their declared confidence; do not treat hypotheses, bounds, or claims as ` +
      `root cause and do not rank a direction from them. If evidence_complete, use only resolved ` +
      `high/medium-confidence evidence alongside generic profile/per-case data. A degraded or unavailable ` +
      `analysis contributes nothing; continue from the generic Profile result.`;
  }
  if (role === 'mega_search_lead' && phase === 'plan_round') return analysisSkillBlock('tech_lead', 'plan_round');
  return '';
}

// Pool sample: the script reads the GPU pool itself each round and hands the lead the result, so a
// round is never planned for idle cards that an external tenant holds. It does not abort the run; an
// occupied pool redirects the round to lease-free work. Sampled per round because tenants come and go.
async function samplePool(round, timeoutMs) {
  if (!(GPU_MIN_FREE_GIB > 0)) return null;
  const p = await agentT(
    `Sample the GPU pool and report what you see. Do NOT acquire a lease, do NOT run any workload, ` +
    `do NOT try to free anything. Reading is the whole job.\n\n` +
    `1. Per-card free VRAM: read /sys/class/drm/card*/device/mem_info_vram_{total,used} (or ` +
    `\`rocm-smi --showmeminfo vram --csv\`) and compute free = total - used in GiB for every card.\n` +
    `2. Per-card utilisation: /sys/class/drm/card*/device/gpu_busy_percent.\n` +
    `3. \`rocm-smi --showpids\`. For each KFD PID it lists, check whether that PID exists in your own ` +
    `/proc. A PID that does NOT is in a FOREIGN NAMESPACE — that is an external tenant, not a stale ` +
    `lease of ours, and the two have opposite remedies. List those PIDs in foreign_pids.\n\n` +
    `verdict:"free" only if EVERY card has at least ${GPU_MIN_FREE_GIB} GiB free. If any card is ` +
    `below that floor, or a foreign tenant holds memory, verdict:"occupied". If you cannot read the ` +
    `pool at all, verdict:"unknown" — do not report an unreadable pool as free.`,
    { phase: 'Optimize', label: `pool sample r${round}`, schema: POOL_SCHEMA,
      ...(Number(timeoutMs) > 0 ? { timeout_ms: Number(timeoutMs), max_retries: 1 } : {}) });
  if (!p) {
    log(`POOL SAMPLE r${round}: sampler returned nothing. Treating as UNKNOWN, not free.`);
    return { verdict: 'unknown', note: 'sampler agent returned nothing' };
  }
  const detail = `${p.cards_free != null ? `${p.cards_free}/${p.cards_total} cards free, ` : ''}` +
    `min ${p.min_free_gib != null ? p.min_free_gib : '?'} GiB vs a ${GPU_MIN_FREE_GIB} GiB floor` +
    `${(p.foreign_pids || []).length ? `, foreign PIDs ${p.foreign_pids.join(',')}` : ''}`;
  if (p.verdict === 'free') {
    log(`POOL SAMPLE r${round}: FREE — ${detail}.`);
  } else {
    log(`POOL ${p.verdict.toUpperCase()} r${round}: ${detail}. ${p.note || ''}\n` +
        `  Plan this round GPU-LESS. A GPU direction dispatched now will spend its clock inside flock ` +
        `and return nothing; an instrument authored now, with its driver, is a partial the next round ` +
        `can run in one lease.`);
  }
  return p;
}

async function runProfileAnalysis(profileSummary, round, label) {
  if (!ANALYSIS_SKILL_ON || !profileSummary) return null;
  const result = await agentT(
    roleAgent(
      'analysis_engineer',
      'analyze_profile',
      'Execute and validate the optional Step-2 analysis pipeline.',
      {
        WORKSPACE: CANONICAL,
        EVAL_DIR,
        ROUND: round,
        COMMANDMENT,
        PROFILE_SUMMARY: profileSummary,
        SKILL_DIR: WORKFLOW_DIR,
        ...ANALYSIS_SKILL_INPUTS, ...GMI, ...MRI,
      },
    ),
    {
      phase: 'Profile',
      label,
      schema: ANALYSIS_RESULT_SCHEMA,
      ...(MEGA_PRODUCTION ? { timeout_ms: 300000, max_retries: 1 } : {}),
    },
  );
  return result || {
    status: 'degraded',
    analysis_skill: ANALYSIS_SKILL,
    analysis_schema_version: '',
    analysis_status: 'unavailable',
    analysis_json: '',
    failure_note: 'analysis agent failed or returned invalid structured output',
  };
}

// Mega candidate registry (no blocking pre-loop reproduction phase).

// One candidate portfolio and lifecycle, with or without Expert Skills.
const MEGA_CANDIDATE_STATES = ['authoring', 'runnable', 'scored', 'finalist', 'rejected', 'parked'];
let megaCandidateRegistry = [];
let megaStateSequenceBase = 0;
let megaUnsafeTimeout = null;
// Consecutive rounds without new runtime evidence (independent Verify, a better score, or an
// on-card Author result that changed the blocker). Drives measure-first and the stall stop.
let megaStall = 0;
let megaLastBlocker = '';
// A refused plan re-plans with the reason instead of ending the wave; 3 in a row still stops.
let megaRefusal = null;
let megaPcRemeasured = 0;
let megaMeasurementCalibration = {
  ready: false, candidate_id: '', attempt_id: '', evidence_manifest: '', note: 'not measured',
};

// <<REPLAY:analyze_resume_fallback>>
// A resumed Analyze with no ladder did not resume anything; rerun it once without the fast path.
function megaPlanIRVerdict(
  plan,
  expectedSkillRevision = '',
  expectedSkillId = '',
  expectedPlannerExtensionSha256 = '',
  expectedSkillBundleSha256 = '',
  expectedValidationStatus = '',
  expectedSkillUsage = ''
) {
  const errors = [];
  if (!plan || typeof plan !== 'object') return { pass: false, errors: ['plan missing'] };
  if (plan.plan_version !== 'mega-plan-v2') errors.push('plan_version must be mega-plan-v2');
  if (expectedSkillRevision &&
      String(plan.expert_skill_revision || '') !== String(expectedSkillRevision)) {
    errors.push(`expert_skill_revision must be ${expectedSkillRevision}`);
  }
  if (expectedSkillId && String(plan.expert_skill_id || '') !== String(expectedSkillId)) {
    errors.push(`expert_skill_id must be ${expectedSkillId}`);
  }
  if (expectedPlannerExtensionSha256 &&
      String(plan.expert_skill_planner_extension_sha256 || '') !==
        String(expectedPlannerExtensionSha256)) {
    errors.push('expert_skill_planner_extension_sha256 does not match the RunContract');
  }
  if (expectedSkillBundleSha256 &&
      String(plan.expert_skill_bundle_sha256 || '') !== String(expectedSkillBundleSha256)) {
    errors.push('expert_skill_bundle_sha256 does not match the RunContract');
  }
  if (expectedValidationStatus &&
      String(plan.expert_skill_validation_status || '') !== expectedValidationStatus) {
    errors.push('expert_skill_validation_status does not match bundle metadata');
  }
  if (expectedValidationStatus === 'experimental') {
    if (plan.expert_skill_auto_apply !== false) {
      errors.push('experimental Skill must report auto_apply=false');
    }
    if (!Array.isArray(plan.expert_skill_explicit_pin_modes) ||
        !plan.expert_skill_explicit_pin_modes.includes(expectedSkillUsage)) {
      errors.push('experimental Skill bundle does not permit the requested usage');
    }
  }
  const collections = [
    'work_domains', 'regions', 'buffers', 'counters', 'queues', 'events',
    'compiler_constraints', 'known_unknowns',
  ];
  const mayBeEmpty = new Set(['counters', 'known_unknowns']);
  for (const name of collections) {
    if (!Array.isArray(plan[name]) || (!mayBeEmpty.has(name) && !plan[name].length)) {
      errors.push(`${name} must be a ${mayBeEmpty.has(name) ? '' : 'non-empty '}array`);
    }
  }
  const ids = (values) => new Set((Array.isArray(values) ? values : [])
    .map((value) => String(value && value.id || '')).filter(Boolean));
  const domains = ids(plan.work_domains);
  const regions = ids(plan.regions);
  const counters = ids(plan.counters);
  const queues = ids(plan.queues);
  const requireUnique = (values, label) => {
    const present = (Array.isArray(values) ? values : [])
      .map((value) => String(value && value.id || '')).filter(Boolean);
    if (present.length !== new Set(present).size) errors.push(`${label} ids must be unique`);
  };
  for (const name of ['work_domains', 'regions', 'buffers', 'counters', 'queues', 'events']) {
    requireUnique(plan[name], name);
  }
  for (const d of Array.isArray(plan.work_domains) ? plan.work_domains : []) {
    if (d.parameters && d.parameters.owned_subdomain)
      errors.push(`work_domain ${d.id || '?'} owned_subdomain must be direct`);
    const o = d.owned_subdomain;
    if (o && ['unit', 'extent_per_claim', 'linear_index'].some((k) => !o[k]))
      errors.push(`work_domain ${d.id || '?'} owned_subdomain incomplete`);
  }
  for (const region of Array.isArray(plan.regions) ? plan.regions : []) {
    if (!domains.has(String(region.work_domain || ''))) {
      errors.push(`region ${region.id || '?'} references unknown work_domain`);
    }
  }
  const checkEndpoints = (item, label) => {
    for (const producer of Array.isArray(item.producers) ? item.producers : []) {
      if (!regions.has(String(producer))) errors.push(`${label} has unknown producer ${producer}`);
    }
    for (const consumer of Array.isArray(item.consumers) ? item.consumers : []) {
      if (!regions.has(String(consumer))) errors.push(`${label} has unknown consumer ${consumer}`);
    }
  };
  for (const buffer of Array.isArray(plan.buffers) ? plan.buffers : []) {
    checkEndpoints(buffer, `buffer ${buffer.id || '?'}`);
  }
  for (const counter of Array.isArray(plan.counters) ? plan.counters : []) {
    checkEndpoints(counter, `counter ${counter.id || '?'}`);
  }
  for (const queue of Array.isArray(plan.queues) ? plan.queues : []) {
    if (!domains.has(String(queue.work_domain || ''))) {
      errors.push(`queue ${queue.id || '?'} references unknown work_domain`);
    }
  }
  for (const event of Array.isArray(plan.events) ? plan.events : []) {
    if (!regions.has(String(event.producer || ''))) {
      errors.push(`event ${event.id || '?'} has unknown producer`);
    }
    if (!regions.has(String(event.consumer || ''))) {
      errors.push(`event ${event.id || '?'} has unknown consumer`);
    }
    if (event.counter && !counters.has(String(event.counter))) {
      errors.push(`event ${event.id || '?'} references unknown counter`);
    }
  }
  const target = plan.target || {};
  if (!(Number(target.launch_count) > 0)) errors.push('target.launch_count must be positive');
  const requireKnown = (values, known, label) => {
    for (const value of Array.isArray(values) ? values : []) {
      if (!known.has(String(value))) errors.push(`target ${label} has unknown id ${value}`);
    }
  };
  requireKnown(target.required_regions, regions, 'required_regions');
  requireKnown(target.required_queues, queues, 'required_queues');
  const resources = plan.resources || {};
  const workgroup = resources.workgroup || {};
  const localMemory = resources.local_memory || {};
  if (!String(resources.arch || '')) errors.push('resources.arch missing');
  if (!(Number(workgroup.wave_size) > 0) || !(Number(workgroup.thread_count) > 0)) {
    errors.push('resources.workgroup is incomplete');
  }
  if (!Number.isFinite(Number(localMemory.total_bytes)) ||
      Number(localMemory.total_bytes) < 0 ||
      Number(localMemory.total_bytes) > Number(localMemory.limit_bytes)) {
    errors.push('resources.local_memory is invalid');
  }
  const primaryLoop = plan.schedule && plan.schedule.primary_loop || {};
  if (!String(primaryLoop.kind || '')) errors.push('schedule.primary_loop.kind missing');
  if (!String(primaryLoop.queue_selection || '') &&
      !Array.isArray(primaryLoop.queue_priority)) {
    errors.push('schedule.primary_loop requires queue_selection or queue_priority');
  }
  for (const queue of Array.isArray(primaryLoop.queue_priority) ? primaryLoop.queue_priority : []) {
    if (!queues.has(String(queue))) errors.push(`primary loop references unknown queue ${queue}`);
  }
  if (!(plan.abi && String(plan.abi.entry_point || '') &&
      Array.isArray(plan.abi.arguments) && plan.abi.arguments.length)) {
    errors.push('abi entry_point/arguments missing');
  }
  if (!(plan.evidence_requirements && Object.keys(plan.evidence_requirements).length)) {
    errors.push('evidence_requirements must be non-empty');
  }
  return { pass: errors.length === 0, errors };
}

function analyzeResumeDegenerate(
  incremental,
  ver,
  requireCompleteGraph,
  requireMegaPlan = false,
  expectedSkillRevision = '',
  expectedSkillId = '',
  expectedPlannerExtensionSha256 = '',
  expectedSkillBundleSha256 = '',
  expectedValidationStatus = '',
  expectedSkillUsage = ''
) {
  const rungs = (ver && Array.isArray(ver.candidate_directions) ? ver.candidate_directions : [])
    .filter((c) => c && (c.id || c.title));
  const graphMissing = !!requireCompleteGraph &&
    (!(ver && ver.task_graph && Array.isArray(ver.task_graph.nodes) &&
       ver.task_graph.nodes.length) ||
     !(ver && ver.resource_timeline && Array.isArray(ver.resource_timeline.pipes)));
  const plan = ver && ver.mega_plan_ir;
  const planVerdict = megaPlanIRVerdict(
    plan,
    expectedSkillRevision,
    expectedSkillId,
    expectedPlannerExtensionSha256,
    expectedSkillBundleSha256,
    expectedValidationStatus,
    expectedSkillUsage
  );
  const megaPlanMissing = !!requireMegaPlan && !planVerdict.pass;
  if (rungs.length && !graphMissing && !megaPlanMissing) return { retry: false, reason: '' };
  if (!incremental && !graphMissing && !megaPlanMissing) return { retry: false, reason: '' };
  return { retry: true, reason:
    `${incremental ? 'ANALYZE RESUME DEGENERATE' : 'MEGA ANALYZE CONTRACT INCOMPLETE'}: ` +
    `${!rungs.length ? 'no structured candidate_directions' : ''}` +
    `${!rungs.length && graphMissing ? ' and ' : ''}` +
    `${graphMissing ? 'required task_graph/resource_timeline missing' : ''}` +
    `${(graphMissing || !rungs.length) && megaPlanMissing ? ' and ' : ''}` +
    `${megaPlanMissing
      ? `required mega_plan_ir invalid (${planVerdict.errors.slice(0, 4).join('; ')})`
      : ''}. ` +
    'Re-running one full frozen-source analysis before Benchmark.' };
}
// <</REPLAY:analyze_resume_fallback>>
function applyBaselineOperatorMap(value) {
  if (!value || !Array.isArray(value.baseline_operator_map)) return value;
  const editable = [...new Set(value.baseline_operator_map
    .filter((row) => row && row.file && row.action !== 'reuse')
    .map((row) => String(row.file)))];
  value.modifiable_files = [...new Set([...(value.modifiable_files || []), ...editable])];
  for (const direction of (value.candidate_directions || [])) {
    direction.focus_files = [...new Set([...(direction.focus_files || []), ...editable])];
  }
  if (editable.length) {
    log(`Baseline operator map added ${editable.length} transformation file(s) to author scope.`);
  }
  return value;
}

// Task-graph gate (args.require_task_graph): makes ordering claims checkable as an enumerated graph.
// Advisory: a partial graph with honest unknowns beats an invented one. Pure function, lifted verbatim
// by tests/test_task_graph_gate.js between these markers; keep it free of log() and run state.
// <<REPLAY:task_graph_gate>>
function taskGraphGate(tg) {
  const nodes = tg && Array.isArray(tg.nodes) ? tg.nodes.length : 0;
  const edges = tg && Array.isArray(tg.edges) ? tg.edges.length : 0;
  if (!tg || (!nodes && !edges)) {
    return { verdict: 'MISSING', summary: '', caveat:
      'TASK GRAPH MISSING: this run required a tile-level dependency graph from Analyze and none ' +
      'was returned. Every direction ranked below is therefore asserted, not derived — there is no ' +
      'record of which orderings are required by the data and which are artifacts of the current ' +
      'code, no critical path to bound what any change can win, and no slack to say which nodes ' +
      'are worth touching. Read the rankings as hypotheses.' };
  }
  // Print the launch_boundary edge count: zero is a finding, all edges is a kernel-granularity graph.
  const eArr = Array.isArray(tg.edges) ? tg.edges : [];
  const launchEdges = eArr.filter((e) => e && e.enforced_by === 'launch_boundary').length;
  const unknowns = Array.isArray(tg.unknowns) ? tg.unknowns.length : 0;
  const assumed = (Array.isArray(tg.nodes) ? tg.nodes : [])
    .filter((n) => n && n.source === 'assumed').length;
  const cp = Number(tg.critical_path_us);
  const e2e = Number(tg.measured_e2e_us);
  const quantified = Number.isFinite(cp) && Number.isFinite(e2e) && e2e > 0;
  const summary = `Task graph: ${nodes} nodes, ${edges} edges, ${launchEdges} enforced only by a ` +
    `launch boundary, ${unknowns} declared unknown, ${assumed} node duration(s) assumed. ` +
    (quantified
      // 1 - cp/e2e is the share of e2e the dependence chain does not explain, not a ceiling on gains: on a
      // throughput-bound operator the gap is arithmetic, which overlap cannot remove.
      ? `critical_path=${cp.toFixed(1)}µs vs e2e=${e2e.toFixed(1)}µs ` +
        `(dependence explains ${(100 * (cp / e2e)).toFixed(1)}% of e2e; the remaining ` +
        `${(100 * (1 - cp / e2e)).toFixed(1)}% is resource-bound work plus bubbles, and only the ` +
        'bubble set is addressable by reordering)'
      : 'critical path NOT quantified');

  if (quantified && assumed > 0) {
    return { verdict: 'UNMEASURED_NUMERIC', summary, caveat:
      `TASK GRAPH USES NUMERIC PLACEHOLDERS: ${assumed} node duration(s) are marked assumed while ` +
      `critical_path_us/measured_e2e_us are numeric. Before Benchmark these numbers are not ` +
      'measurements and must not rank directions; preserve them as unknown/null until current-run data exists.' };
  }
  if (quantified && cp > e2e) {
    return { verdict: 'INCONSISTENT', summary, caveat:
      `TASK GRAPH INCONSISTENT: critical_path_us (${cp}) exceeds measured_e2e_us (${e2e}). The ` +
      'longest path through the graph cannot be longer than the thing it is a path through, so ' +
      'either a node duration is wrong or an edge is spurious. Do not use this graph\'s headroom ' +
      'number to size any proposal.' };
  }
  // Every edge a launch boundary and none narrower means the scope column was not filled in; the graph
  // cannot then tell a required barrier from an incidental one. Advisory.
  const narrow = eArr.filter((e) => e && e.scope && e.scope !== 'cross_rank' && e.scope !== 'hbm').length;
  if (edges && launchEdges === edges && narrow === 0) {
    return { verdict: 'KERNEL_GRANULARITY', summary, caveat:
      `TASK GRAPH IS KERNEL-GRANULAR: all ${edges} edges came back enforced_by=launch_boundary with ` +
      'no edge at a narrower scope, which is what a graph whose nodes are KERNELS looks like. Such ' +
      'a graph restates the launch order and cannot separate an ordering the data requires from one ' +
      'the current code imposes. Treat any fusion direction ranked from it as unsupported.' };
  }
  // WHAT THE CONSUMER NEEDS that the producer does not supply (weights, scales, descriptors): graph nodes
  // are output tiles, so these never appear as edges. Ask per edge; [] is a valid answer. Advisory:
  // missing answers make prefetch unrankable, not the graph invalid.
  const fanInEdges = eArr.filter((e) => e && Number(e.fan_in) > 0);
  const unanswered = fanInEdges
    .filter((e) => !Array.isArray(e.producer_independent_operands))
    .map((e) => `${e.from || '?'}->${e.to || '?'}`);
  if (unanswered.length) {
    return { verdict: 'OK', summary, caveat:
      `EDGES THAT DID NOT SAY WHAT THE CONSUMER LOADS ANYWAY (${unanswered.length}): ` +
      `${unanswered.join(', ')}. Each names an operand the consumer waits for; none names the ` +
      'operands it needs that the producer does not supply — weights, scales, descriptors, index ' +
      'arrays. Those depend on nothing and can be loaded during the producer\'s own execution, ' +
      'which is a different and usually cheaper lever than moving the dependent operand earlier. ' +
      'It cannot be ranked from this graph because it has no node here.' };
  }
  return { verdict: 'OK', summary, caveat: '' };
}
// <</REPLAY:task_graph_gate>>

// <<REPLAY:pipe_occupancy_gate>>
// Pipe table gate, separate from taskGraphGate: the graph says what may overlap, this says whether the
// pipe is even busy. It prints the table, derives the class from the numbers, and names directions
// that claim more than their pipe's idle fraction.
const PIPE_SATURATED_PCT = 80;
const PIPE_LOW_PCT = 50;
const GAP_NEGLIGIBLE_US = 0.5;

function pipeOccupancyGate(rt, directions) {
  const dirs = Array.isArray(directions) ? directions : [];
  const pipes = rt && Array.isArray(rt.pipes)
    ? rt.pipes.filter((p) => p && p.utilization_pct != null &&
      Number.isFinite(Number(p.utilization_pct))) : [];
  if (!pipes.length) {
    return { verdict: 'MISSING', summary: '', class_derived: null, caveat:
      'PIPE TABLE MISSING: this run required a per-pipe resource timeline from Analyze and none was ' +
      'returned. Nothing below is priced against what the machine was actually doing — there is no ' +
      'record of which functional unit was the constraint, so a direction targeting an already-' +
      'saturated pipe and one filling an idle pipe are indistinguishable here, and the classic ' +
      'reflexes (raise occupancy, fuse launches) cannot be ruled out before they cost a lease. ' +
      'Read every expected_speedup as unbounded above.' };
  }
  const maxUtil = Math.max(...pipes.map((p) => Number(p.utilization_pct)));
  const satur = pipes.filter((p) => Number(p.utilization_pct) >= PIPE_SATURATED_PCT);
  const gap = rt.interkernel_gap_us && Number.isFinite(Number(rt.interkernel_gap_us.median))
    ? Number(rt.interkernel_gap_us.median) : null;
  const opps = Array.isArray(rt.idle_pipe_opportunities) ? rt.idle_pipe_opportunities.length : 0;
  const closed = Array.isArray(rt.closed_axes) ? rt.closed_axes.length : 0;

  // Derive the class from the numbers; a stated class the table does not support is worse than none.
  let derived = 'mixed';
  // Throughput-bound only when every measured engine is saturated (or only one was measured);
  // otherwise mixed, since idle engines can still take cross-engine work.
  if (satur.length && satur.length === pipes.length) derived = 'throughput_bound';
  else if (maxUtil < PIPE_LOW_PCT && gap !== null && gap > GAP_NEGLIGIBLE_US) derived = 'launch_bound';
  else if (maxUtil < PIPE_LOW_PCT) derived = 'latency_bound';

  const table = pipes
    .map((p) => `${p.stage || '?'}.${p.pipe || '?'}=${Number(p.utilization_pct).toFixed(1)}%`)
    .join(' ');
  const summary = `Pipe table: ${table}; max=${maxUtil.toFixed(1)}% ` +
    `(${satur.length} pipe(s) at/above ${PIPE_SATURATED_PCT}%); ` +
    `inter-kernel gap median=${gap === null ? 'UNMEASURED' : gap.toFixed(3) + 'us'}; ` +
    `class stated=${rt.class || 'none'} derived=${derived}; ` +
    `${opps} idle-pipe opportunit(y/ies), ${closed} closed axis/axes.`;

  const notes = [];
  if (rt.class && rt.class !== derived) {
    notes.push(`CLASS CONTRADICTS THE TABLE: Analyze stated class="${rt.class}" but the numbers it ` +
      `reported derive "${derived}" (max pipe ${maxUtil.toFixed(1)}%, gap ` +
      `${gap === null ? 'unmeasured' : gap.toFixed(3) + 'us'}). The class decides which lever ` +
      'families are candidates at all, so take the derived one and re-read the rankings against it.');
  }
  if (derived === 'latency_bound') {
    notes.push('LATENCY-BOUND: every pipe is below ' + PIPE_LOW_PCT + '% and the launch gap is ' +
      'negligible, so the machine is stalled on dependences, not on any resource. Two standard ' +
      'levers are dead here and should not be given a lease: raising OCCUPANCY (adding waves adds ' +
      'stalled waves when each wave\'s instruction stream has no independent work, and it spends ' +
      'the registers and LDS that software pipelining needs), and fusing to reclaim LAUNCH ' +
      'OVERHEAD (the gap is already zero). What does pay is making dependency-free work of the ' +
      'NEXT stage issuable in this one\'s shadow — a kernel boundary is a grid-wide barrier plus a ' +
      'pipeline drain, so that work is not unscheduled, it is inexpressible.');
    if (!opps) {
      notes.push('NO IDLE-PIPE OPPORTUNITIES LISTED: the table says the pipes are idle and the ' +
        'artifact names nothing that could be issued into them. That is the one question this ' +
        'analysis existed to answer; without it the directions below cannot have been derived from it.');
    }
  }
  // Directions priced above what their own pipe can pay. Checked here rather than trusted, because
  // expected_speedup is the field most likely to be a wish.
  const overclaimed = [];
  const unpriced = [];
  for (const d of dirs) {
    if (!d || d.specialty === 'deep_explore') continue;
    if (!d.fills_pipe) { unpriced.push(d.id || d.title || '?'); continue; }
    const util = Number.isFinite(Number(d.pipe_util_pct)) ? Number(d.pipe_util_pct) : null;
    const exp = Number(d.expected_speedup);
    if (util === null || !Number.isFinite(exp)) continue;
    const ceiling = 1 / Math.max(util / 100, 1e-6); // filling an idle pipe completely, the optimistic bound
    if (exp > ceiling) {
      overclaimed.push(`${d.id || d.title}: claims ${exp.toFixed(3)}x on ${d.fills_pipe} at ` +
        `${util.toFixed(1)}% utilization, whose perfect-fill bound is ${ceiling.toFixed(3)}x`);
    }
  }
  if (unpriced.length) {
    notes.push(`DIRECTIONS WITH NO PIPE NAMED (${unpriced.length}): ${unpriced.join(', ')}. A ` +
      'direction that cannot say which functional unit it fills has not been derived from the ' +
      'table and its expected_speedup is unbounded by anything. Treat as hypotheses.');
  }
  if (overclaimed.length) {
    notes.push('DIRECTION OVERCLAIMS ITS PIPE: ' + overclaimed.join('; ') +
      '. These are arithmetically impossible against the reported utilization and can be rejected ' +
      'without a run; if the utilization is wrong, that is the finding.');
  }
  // A sized hole no direction claims. Checked only once directions exist; rides_on (another rung
  // absorbs it) is a valid answer, silence is not.
  if (dirs.length) {
    const claimed = new Set();
    for (const d of dirs) {
      if (!d) continue;
      if (d.fills_hole) claimed.add(String(d.fills_hole));
      for (const g of Array.isArray(d.graph_refs) ? d.graph_refs : []) claimed.add(String(g));
    }
    const unowned = (Array.isArray(rt.idle_pipe_opportunities) ? rt.idle_pipe_opportunities : [])
      .filter((o) => o && (Number(o.recoverable_us) > 0 || Number(o.pct_of_e2e) > 0))
      .filter((o) => !o.rides_on && !claimed.has(String(o.id)))
      .map((o) => `${o.id || o.window || '?'} (${Number(o.recoverable_us) > 0 ? Number(o.recoverable_us).toFixed(0) + 'us' : ''}` +
        `${Number(o.pct_of_e2e) > 0 ? (Number(o.recoverable_us) > 0 ? ', ' : '') + Number(o.pct_of_e2e).toFixed(1) + '% of e2e' : ''})`);
    if (unowned.length) {
      notes.push(`SIZED HOLE WITH NO DIRECTION (${unowned.length}): ${unowned.join('; ')}. Analyze ` +
        'located these, measured them and put a number on them, and no direction in this round ' +
        'names one as the thing it fills. A hole nobody claims is not a hole nobody can reach — it ' +
        'is a hole nobody looked at. Either give it a direction, or set `rides_on` to the rung that ' +
        'will absorb it so the claim is recorded and can be checked when that rung reports.');
    }
  }
  const verdict = notes.length ? (overclaimed.length || (rt.class && rt.class !== derived) ? 'INCONSISTENT' : 'ADVISORY') : 'OK';
  return { verdict, summary, class_derived: derived, caveat: notes.join(' ') };
}
// <</REPLAY:pipe_occupancy_gate>>

// <<REPLAY:roadmap_ladder_gate>>
// Did the round plan against Analyze's ladder? Diffs dispatched rungs against it, exposing skipped
// controls and never-proposed fusion rungs.
//
// <<REPLAY:open_rungs>>
// A rung is done when it produced a number, not when it was planned. Outcomes per rung:
//   never_planned  no round has taken it
//   faulted        a direction took it and the code did not run
//   unmeasured     it ran but returned no verified number
//   measured       a verified number exists; only this closes the rung
const RUNG_OUTCOMES = ['never_planned', 'faulted', 'unmeasured', 'measured'];
function rungIdOf(c) {
  return String((c && c.id) || String((c && c.title) || '').trim().split(/\s+/)[0] || '').trim();
}
// tally: Map<rungId, {attempts, last_outcome}>. Returns the entries the next wave still owes,
// strongest-evidence-last so `measured` can never be reintroduced by a later weaker grade.
function openRungs(ladder, tally) {
  const t = tally instanceof Map ? tally : new Map(Object.entries(tally || {}));
  return (Array.isArray(ladder) ? ladder : [])
    .filter((c) => c && (c.id || c.title))
    .map((c) => {
      const id = rungIdOf(c);
      const e = t.get(id) || {};
      const outcome = RUNG_OUTCOMES.includes(e.last_outcome) ? e.last_outcome : 'never_planned';
      return { id, title: c.title || id, gated_on: Array.isArray(c.gated_on) ? c.gated_on : [],
        is_positive_control: !!c.is_positive_control,
        // Carried so the next wave knows what closing this rung would even mean. A prerequisite
        // re-planned as a speed experiment is a prerequisite that gets rejected again.
        step_role: stepRoleOf(c), enables: String(c.enables || '') || undefined,
        mandatory_arms: Array.isArray(c.mandatory_arms) ? c.mandatory_arms : [],
        target_shape: c.target_shape || undefined,
        attempts: Number(e.attempts) || 0, last_outcome: outcome };
    })
    .filter((c) => c.last_outcome !== 'measured');
}
// <</REPLAY:open_rungs>>

// <<REPLAY:direction_contract>>
// Join the planner's direction back to Analyze's rung, so dropped step_role/mandatory_arms do not turn
// a prerequisite into a terminal speed test.
function enrichDirection(direction, ladder) {
  const d = direction || {};
  const idOf = (c) => String((c && (c.id || c.title)) || '').trim().split(/\s+/)[0];
  const rungId = String(d.roadmap_rung || '').trim();
  const rung = (Array.isArray(ladder) ? ladder : []).find((c) => idOf(c) === rungId) || {};
  const pickArray = (a, b) => Array.isArray(a) ? a : (Array.isArray(b) ? b : []);
  return {
    ...d,
    gated_on: pickArray(d.gated_on, rung.gated_on),
    mandatory_arms: pickArray(d.mandatory_arms, rung.mandatory_arms),
    step_role: d.step_role || rung.step_role || 'terminal',
    enables: d.enables || rung.enables || undefined,
    cost_budget_pct: d.cost_budget_pct != null ? d.cost_budget_pct : rung.cost_budget_pct,
    target_shape: d.target_shape || rung.target_shape || undefined,
  };
}

function strictDirectionVerdict(direction, ladder, completedRungs) {
  const d = direction || {};
  const rungId = String(d.roadmap_rung || '').trim();
  if (!rungId || rungId === 'off_ladder') {
    return { pass: false, reason: 'strict autonomy direction must name a ladder rung' };
  }
  const idOf = (c) => String((c && (c.id || c.title)) || '').trim().split(/\s+/)[0];
  const rung = (Array.isArray(ladder) ? ladder : []).find((c) => idOf(c) === rungId);
  if (!rung) return { pass: false, reason: `unknown roadmap rung ${rungId}` };
  const done = new Set(Array.isArray(completedRungs) ? completedRungs
    : (completedRungs instanceof Set ? [...completedRungs] : []));
  const unmet = (Array.isArray(rung.gated_on) ? rung.gated_on : []).filter((g) => !done.has(String(g)));
  if (unmet.length) {
    return { pass: false, reason: `${rungId} prerequisite(s) not completed to spec: ${unmet.join(', ')}` };
  }
  if (String(d.step_role || '') === 'enabling' && !String(d.enables || '').trim()) {
    return { pass: false, reason: `${rungId} is enabling but names no terminal rung` };
  }
  const rungRole = String(rung.step_role || 'terminal');
  if (String(d.step_role || 'terminal') !== rungRole ||
      (rungRole === 'enabling' && String(d.enables || '') !== String(rung.enables || ''))) {
    return { pass: false, reason: `${rungId} changed its Analyze step_role/enables contract` };
  }
  const expected = Array.isArray(rung.mandatory_arms) ? rung.mandatory_arms : [];
  const supplied = new Set(Array.isArray(d.mandatory_arms) ? d.mandatory_arms.map(String) : []);
  const missing = expected.filter((a) => !supplied.has(String(a)));
  if (missing.length) {
    return { pass: false, reason: `${rungId} dropped mandatory arm(s): ${missing.join(', ')}` };
  }
  if (rung.target_shape) {
    const planned = d.target_shape || {};
    const expectedStages = [...new Set((rung.target_shape.stages_fused || []).map(String))].sort();
    const plannedStages = [...new Set((planned.stages_fused || []).map(String))].sort();
    if (Number(planned.launches) !== Number(rung.target_shape.launches) ||
        (!!rung.target_shape.require_overlap && planned.require_overlap !== true) ||
        expectedStages.join('\0') !== plannedStages.join('\0')) {
      return { pass: false, reason: `${rungId} changed its Analyze target_shape` };
    }
  }
  return { pass: true, reason: '' };
}
// <</REPLAY:direction_contract>>

// <<REPLAY:autonomy_transition>>
function finalStatusVerdict(directorStatus, scopedGuardFailure, strictProofFailure) {
  if (directorStatus === 'flagged' || scopedGuardFailure) return 'flagged';
  if (strictProofFailure) return 'autonomy_incomplete';
  return directorStatus || 'unknown';
}

function finalMetricFailure(metric, targetGuards, primary, guards, attribution) {
  const scoped = Array.isArray(targetGuards) && targetGuards.length > 0;
  if ((scoped || metric === 'changed_kernel') && !(Number(primary) > 1.0)) return true;
  if (scoped && (!guards || !guards.complete || !guards.regression_pass)) return true;
  if (metric === 'changed_kernel') {
    const a = attribution || {};
    if (!(a.absolute_to_frozen === true && Number(a.changed_us) > 0 &&
          Number(a.replaced_sum_us) > 0)) return true;
  }
  return false;
}

// <</REPLAY:autonomy_transition>>

// Advisory: never blocks a direction, but a skipped rung must be stated via rung_deviation.
function roadmapLadderGate(ladder, directions, completedRungs) {
  const rungs = (Array.isArray(ladder) ? ladder : []).filter((c) => c && (c.id || c.title));
  const dirs = Array.isArray(directions) ? directions : [];
  const done = new Set(Array.isArray(completedRungs) ? completedRungs : completedRungs instanceof Set ? [...completedRungs] : []);
  if (!rungs.length) {
    return { verdict: 'MISSING', summary: '', planned: [], caveat:
      'LADDER MISSING: Analyze returned no candidate_directions, so there is no recorded ordering ' +
      'for this run and nothing below can be checked against a plan. Each round is then free to ' +
      'start from the profile, which is exactly the state in which the most expensive direction — ' +
      'the one that is conditional on two cheaper ones — is never reached, because nothing is ' +
      'tracking that it was owed.' };
  }
  // `id` is what a plan can quote; fall back to the leading token of the title, which is how these
  // are written in practice ("D2 Per-token ... readiness").
  const idOf = (c) => String(c.id || String(c.title).trim().split(/\s+/)[0] || '').trim();
  const byId = new Map(rungs.map((c) => [idOf(c), c]));

  const planned = [];
  const undeclared = [];
  const offLadder = [];
  const gateViolations = [];
  for (const d of dirs) {
    if (!d) continue;
    const label = d.id || d.title || '?';
    const rung = (d.roadmap_rung || '').trim();
    if (!rung) { undeclared.push(label); continue; }
    if (rung === 'off_ladder') {
      if (!String(d.rung_deviation || '').trim()) offLadder.push(label);
      continue;
    }
    planned.push(rung);
    const c = byId.get(rung);
    if (!c) { offLadder.push(`${label} (names rung "${rung}", which is not on the ladder)`); continue; }
    const gates = Array.isArray(c.gated_on) ? c.gated_on.map(String) : [];
    const unmet = gates.filter((g) => !done.has(g));
    if (unmet.length && !String(d.rung_deviation || '').trim()) {
      gateViolations.push(`${label} takes ${rung}, whose gated_on [${unmet.join(', ')}] ` +
        'has not been dispatched, with no rung_deviation stated');
    }
  }

  const seen = new Set([...done, ...planned]);
  const never = rungs.filter((c) => !seen.has(idOf(c)));
  const controls = never.filter((c) => c.is_positive_control);
  const blocked = rungs.filter((c) => {
    const gates = Array.isArray(c.gated_on) ? c.gated_on.map(String) : [];
    return gates.length && !seen.has(idOf(c)) && gates.some((g) => !seen.has(g));
  });

  const summary = `Ladder: ${rungs.length} rung(s) [${rungs.map(idOf).join(', ')}]; ` +
    `completed to spec [${[...done].join(', ') || 'none'}]; ` +
    `this round [${planned.join(', ') || 'none on ladder'}]; ` +
    `${never.length} never reached.`;

  const notes = [];
  if (undeclared.length) {
    notes.push(`DIRECTIONS WITH NO RUNG NAMED (${undeclared.length}): ${undeclared.join(', ')}. ` +
      'A direction that cannot say which rung it is — including "off_ladder" — has not been ' +
      'planned against the ladder, and the ladder cannot then be used to tell what is still owed.');
  }
  if (offLadder.length) {
    notes.push(`OFF-LADDER WITHOUT A STATED DEVIATION: ${offLadder.join('; ')}. Leaving the ladder ` +
      'is allowed and often right; leaving it silently means the rung it displaced is not recorded ' +
      'anywhere as still open.');
  }
  if (gateViolations.length) {
    notes.push('RUNG TAKEN OUT OF ORDER: ' + gateViolations.join('; ') + '. The prerequisite is ' +
      'usually what makes the result INTERPRETABLE, not merely what makes it likelier to win — a ' +
      'readiness rung measured before its bounding readout can read negative for a reason the ' +
      'bounding readout would have priced in advance.');
  }
  if (controls.length) {
    notes.push('THE LADDER\'S OWN POSITIVE CONTROL HAS NOT BEEN DISPATCHED: ' +
      controls.map(idOf).join(', ') + '. Analyze designated this rung as the change with a ' +
      'predicted direction that proves the harness can detect a win at all. Until it runs, every ' +
      'null below is ambiguous between "no effect" and "no instrument".');
  }
  if (blocked.length) {
    notes.push(`RUNGS STILL BLOCKED BY UNDISPATCHED PREREQUISITES (${blocked.length}): ` +
      blocked.map((c) => `${idOf(c)} gated_on [${(c.gated_on || []).join(', ')}]`).join('; ') +
      '. These cannot be reached by continuing to plan around the ladder. If a gate is being ' +
      'abandoned, say so; a gate that is simply never satisfied silently deletes everything above it.');
  }
  const verdict = notes.length
    ? (gateViolations.length || controls.length ? 'INCONSISTENT' : 'ADVISORY')
    : 'OK';
  return { verdict, summary, planned, caveat: notes.join(' ') };
}
// <</REPLAY:roadmap_ladder_gate>>

// <<REPLAY:mega_candidate_registry>>
// Mega whole-kernel candidates remain independent lineages until verified.
function validMegaCandidateId(value) {
  const id = String(value || '');
  return /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(id) && id !== 'frozen_baseline';
}
// Resumed waves restart round numbers, so `_rN` lineage grows without bound; hash-compact at the cap.
function megaChildCandidateId(parentId, round) {
  const id = `${parentId}_r${round}`;
  if (validMegaCandidateId(id)) return id;
  let h = 5381;
  for (const ch of id) h = (h * 33 + ch.charCodeAt(0)) >>> 0;
  const tag = `_${h.toString(36)}_r${round}`;
  return parentId.slice(0, 64 - tag.length).replace(/[._-]+$/, '') + tag;
}
function validMegaCandidateSourceForId(_id, source) {
  return source === 'search' || source === 'integrated';
}
function normalizeMegaActivation(raw) {
  const a = raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {};
  const switches = Array.isArray(a.switches)
    ? a.switches.map((s) => ({
      switch_name: String(s && s.switch_name || ''),
      switch_value: String(s && s.switch_value || ''),
    })).filter((s) => s.switch_name) : [];
  return {
    mode: String(a.mode || ''),
    switch_name: String(a.switch_name || ''),
    switch_value: String(a.switch_value || ''),
    path_marker: String(a.path_marker || ''),
    marker_how: String(a.marker_how || ''),
    switches,
  };
}
function normalizeContractFailures(raw) {
  return (Array.isArray(raw) ? raw : []).map((failure) => ({
    id: String(failure && failure.id || ''),
    category: String(failure && failure.category || 'semantic'),
    severity: String(failure && failure.severity || 'required'),
    tier: failure && failure.tier === 'surface' ? 'surface' : 'semantic',
    messages: Array.isArray(failure && failure.messages)
      ? failure.messages.map(String) : [],
  })).filter((failure) => failure.id);
}
function normalizeMegaWorkingSnapshot(raw) {
  const w = raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {};
  const failures = normalizeContractFailures(w.contract_failures);
  const structuralConflict = w.structural_verified === true &&
    failures.some((failure) => failure.severity === 'required');
  return {
    head: String(w.head || ''),
    status: String(w.status || ''),
    patch: String(w.patch || ''),
    attempt_id: String(w.attempt_id || ''),
    checkpoint_complete: w.checkpoint_complete === true,
    structural_verified: w.structural_verified === true && !structuralConflict,
    structural_report: String(w.structural_report || ''),
    structural_skill_id: String(w.structural_skill_id || ''),
    structural_candidate_head: String(w.structural_candidate_head || ''),
    structural_candidate_tree_digest: String(w.structural_candidate_tree_digest || ''),
    structural_skill_bundle_sha256: String(w.structural_skill_bundle_sha256 || ''),
    structural_planner_extension_sha256:
      String(w.structural_planner_extension_sha256 || ''),
    structural_contract_revision: String(w.structural_contract_revision || ''),
    structural_contract_sha256: String(w.structural_contract_sha256 || ''),
    evidence_manifest: String(w.evidence_manifest || ''),
    activation: normalizeMegaActivation(w.activation),
    topology_sig: String(w.topology_sig || ''),
    topology: w.topology && typeof w.topology === 'object' && !Array.isArray(w.topology)
      ? { ...w.topology } : {},
    changed_files: Array.isArray(w.changed_files) ? w.changed_files.map(String) : [],
    contract_failures: structuralConflict ? [{
      id: 'structural_state_conflict', category: 'plan', severity: 'required',
      messages: ['working structural pass conflicts with required failures'],
    }] : failures,
    next_blocker: String(w.next_blocker || ''),
    notes: String(w.notes || ''),
  };
}
function normalizeMegaCandidate(raw) {
  const c = raw || {};
  const sourceValid = ['search', 'integrated'].includes(c.source);
  const source = sourceValid ? c.source : 'search';
  const idValid = validMegaCandidateId(c.id);
  const sourceIdValid = validMegaCandidateSourceForId(c.id, source);
  const candidateStates = typeof MEGA_CANDIDATE_STATES === 'undefined'
    ? ['authoring', 'runnable', 'scored', 'finalist', 'rejected', 'parked'] : MEGA_CANDIDATE_STATES;
  const status = (!sourceValid || !idValid || !sourceIdValid) ? 'rejected'
    : (candidateStates.includes(c.status) ? c.status : 'authoring');
  const score = Number(c.absolute_score);
  const launches = Number(c.launches);
  const rawContractFailures = normalizeContractFailures(c.contract_failures);
  const structuralConflict = c.structural_verified === true &&
    rawContractFailures.some((failure) => failure.severity === 'required');
  const contractFailures = structuralConflict ? [{
    id: 'structural_state_conflict', category: 'plan', severity: 'required',
    messages: [`legacy structural pass conflicts with failures: ${
      rawContractFailures.map((failure) => failure.id).join(', ')}`],
  }] : rawContractFailures;
  return {
    id: String(c.id || ''),
    id_valid: idValid,
    source,
    source_valid: sourceValid,
    source_id_valid: sourceIdValid,
    base_id: String(c.base_id || 'frozen_baseline'),
    tree: String(c.tree || ''),
    head: String(c.head || ''),
    working_head: String(c.working_head || ''),
    working_status: String(c.working_status || ''),
    working_snapshot: normalizeMegaWorkingSnapshot(c.working_snapshot),
    patch: String(c.patch || ''),
    status,
    topology_sig: String(c.topology_sig || ''),
    topology: c.topology && typeof c.topology === 'object' && !Array.isArray(c.topology)
      ? { ...c.topology } : {},
    activation: normalizeMegaActivation(c.activation),
    provenance: String(c.provenance || source),
    claim_complete: c.claim_complete === true,
    checkpoint_complete: c.checkpoint_complete === true,
    structural_verified: c.structural_verified === true && !structuralConflict,
    runtime_verified: c.runtime_verified === true,
    score_complete: c.score_complete === true,
    verification_status: String(c.verification_status || ''),
    correctness: String(c.correctness || ''),
    gpu_executed: c.gpu_executed === true,
    activation_on_hardware: String(c.activation_on_hardware || ''),
    structural_report: String(c.structural_report || ''),
    structural_skill_id: String(c.structural_skill_id || ''),
    structural_candidate_head: String(c.structural_candidate_head || ''),
    structural_candidate_tree_digest: String(c.structural_candidate_tree_digest || ''),
    structural_skill_bundle_sha256: String(c.structural_skill_bundle_sha256 || ''),
    structural_planner_extension_sha256:
      String(c.structural_planner_extension_sha256 || ''),
    structural_contract_revision: String(c.structural_contract_revision || ''),
    structural_contract_sha256: String(c.structural_contract_sha256 || ''),
    contract_failures: contractFailures,
    attempt_id: String(c.attempt_id || ''),
    changed_files: Array.isArray(c.changed_files) ? c.changed_files.map(String) : [],
    evidence_manifest: String(c.evidence_manifest || ''),
    target_guard: String(c.target_guard || ''),
    final_tree: String(c.final_tree || ''),
    final_head: String(c.final_head || ''),
    final_attempt_id: String(c.final_attempt_id || ''),
    final_evidence_manifest: String(c.final_evidence_manifest || ''),
    absolute_score: Number.isFinite(score) && score > 0 ? score : null,
    per_case: Array.isArray(c.per_case) ? c.per_case : [],
    paired_readings: Array.isArray(c.paired_readings) ? c.paired_readings : [],
    denominator_mismatch: Array.isArray(c.denominator_mismatch) ? c.denominator_mismatch : [],
    component_breakdown: Array.isArray(c.component_breakdown) ? c.component_breakdown : [],
    hot_path_counters: c.hot_path_counters && typeof c.hot_path_counters === 'object'
      ? c.hot_path_counters : {},
    null_arm_pct: c.null_arm_pct == null ? null : Number(c.null_arm_pct),
    reps: Math.max(0, Number(c.reps || 0)),
    artifact_hash_base: String(c.artifact_hash_base || ''),
    artifact_hash_candidate: String(c.artifact_hash_candidate || ''),
    correctness_pass: c.correctness_pass === true,
    activation_pass: c.activation_pass === true,
    head_pass: c.head_pass === true,
    launch_pass: c.launch_pass === true,
    liveness_pass: c.liveness_pass === true,
    graph_pass: c.graph_pass === true,
    guards_pass: c.guards_pass !== false,
    artifact_distinct: c.artifact_distinct === true,
    measurement_pass: c.measurement_pass === true,
    launches: Number.isInteger(launches) && launches > 0 ? launches : null,
    attempts: Math.max(0, Number(c.attempts || 0)),
    next_blocker: String(c.next_blocker || ''),
    notes: String(c.notes || ''),
  };
}

function upsertMegaCandidate(registry, incoming) {
  const list = Array.isArray(registry) ? registry.map(normalizeMegaCandidate) : [];
  let next = normalizeMegaCandidate(incoming);
  if (!next.id) return list;
  const i = list.findIndex((c) => c.id === next.id);
  if (i < 0) return [...list, next];
  const prev = list[i];
  const priorStructural = megaEffectiveLaneCheckpoint(prev);
  const sameStructuralHead = priorStructural.structural_verified === true &&
    !!priorStructural.structural_candidate_head &&
    next.head === priorStructural.structural_candidate_head &&
    next.head === priorStructural.head;
  const runtimeOnlyExactHeadFailure = sameStructuralHead &&
    !next.structural_verified && next.contract_failures.length === 0;
  if (runtimeOnlyExactHeadFailure) {
    next = normalizeMegaCandidate({
      ...next,
      checkpoint_complete: priorStructural.checkpoint_complete,
      structural_verified: true,
      structural_report: priorStructural.structural_report,
      structural_skill_id: priorStructural.structural_skill_id,
      structural_candidate_head: priorStructural.structural_candidate_head,
      structural_candidate_tree_digest: priorStructural.structural_candidate_tree_digest,
      structural_skill_bundle_sha256: priorStructural.structural_skill_bundle_sha256,
      structural_planner_extension_sha256:
        priorStructural.structural_planner_extension_sha256,
      structural_contract_revision: priorStructural.structural_contract_revision,
      structural_contract_sha256: priorStructural.structural_contract_sha256,
      contract_failures: [],
    });
  }
  if (prev.claim_complete && !next.claim_complete) {
    list[i] = normalizeMegaCandidate({
      ...prev,
      attempts: Math.max(prev.attempts, next.attempts),
      working_head: next.head || next.working_head || prev.working_head,
      working_status: next.status || next.working_status || prev.working_status,
      working_snapshot: {
        head: next.head || next.working_head || prev.working_head,
        status: next.status || next.working_status || prev.working_status,
        patch: next.patch,
        attempt_id: next.attempt_id,
        checkpoint_complete: next.checkpoint_complete,
        structural_verified: next.structural_verified,
        structural_report: next.structural_report,
        structural_skill_id: next.structural_skill_id,
        structural_candidate_head: next.structural_candidate_head,
        structural_candidate_tree_digest: next.structural_candidate_tree_digest,
        structural_skill_bundle_sha256: next.structural_skill_bundle_sha256,
        structural_planner_extension_sha256:
          next.structural_planner_extension_sha256,
        structural_contract_revision: next.structural_contract_revision,
        structural_contract_sha256: next.structural_contract_sha256,
        evidence_manifest: next.evidence_manifest,
        activation: next.activation,
        topology_sig: next.topology_sig,
        topology: next.topology,
        changed_files: next.changed_files,
        contract_failures: next.contract_failures,
        next_blocker: next.next_blocker,
        notes: next.notes,
      },
      next_blocker: next.next_blocker || prev.next_blocker,
      notes: next.notes || prev.notes,
    });
  } else {
    list[i] = normalizeMegaCandidate({
      ...prev,
      ...next,
      claim_complete: next.claim_complete === true,
      attempts: Math.max(prev.attempts, next.attempts),
      working_head: '',
      working_status: '',
      working_snapshot: {},
    });
  }
  return list;
}

function megaRegistryForSearch(registry) {
  return (Array.isArray(registry) ? registry : []).map(normalizeMegaCandidate).map((c) => {
    const effective = megaEffectiveLaneCheckpoint(c);
    return {
      id: c.id,
      source: c.source,
      base_id: c.base_id,
      tree: c.tree,
      head: c.head,
      working_head: c.working_head,
      working_status: c.working_status,
      status: c.status,
      checkpoint_complete: effective.checkpoint_complete === true,
      structural_verified: effective.structural_verified === true,
      runtime_verified: c.runtime_verified,
      score_complete: c.score_complete,
      structural_report: effective.structural_report || '',
      structural_skill_id: effective.structural_skill_id || '',
      structural_candidate_head: effective.structural_candidate_head || '',
      structural_candidate_tree_digest: effective.structural_candidate_tree_digest || '',
      structural_skill_bundle_sha256: effective.structural_skill_bundle_sha256 || '',
      structural_planner_extension_sha256:
        effective.structural_planner_extension_sha256 || '',
      structural_contract_revision: effective.structural_contract_revision || '',
      structural_contract_sha256: effective.structural_contract_sha256 || '',
      contract_failures: effective.contract_failures || [],
      verification_status: c.verification_status,
      correctness: c.correctness,
      gpu_executed: c.gpu_executed,
      activation_on_hardware: c.activation_on_hardware,
      absolute_score: c.absolute_score,
      target_guard: c.target_guard,
      per_case: c.per_case,
      denominator_mismatch: c.denominator_mismatch,
      component_breakdown: c.component_breakdown,
      hot_path_counters: c.hot_path_counters,
      topology_sig: effective.topology_sig || c.topology_sig,
      next_blocker: effective.next_blocker || c.next_blocker,
      notes: effective.notes || c.notes,
      attempts: c.attempts,
    };
  });
}

function megaHistoryForSearch(sourceHistory, registry) {
  const h = sourceHistory || {};
  const searchRegistry = megaRegistryForSearch(registry)
    .filter((c) => c.source === 'search' || c.source === 'integrated');
  return {
    ...h,
    // Global insights are free-form and have no reliable source label. Rebuild the search board from
    // candidate records that belong to the common lifecycle.
    insights: searchRegistry.flatMap((c) =>
      [c.next_blocker, c.notes].filter((text) => String(text || '').trim())),
    rounds: (Array.isArray(h.rounds) ? h.rounds : []).map((round) => {
      const directions = (Array.isArray(round.directions) ? round.directions : [])
        .filter((d) => ['search', 'integrated'].includes(String(d && d.candidate_source || '')));
      const directionIds = new Set(directions.map((d) => String(d && d.id || '')));
      const results = (Array.isArray(round.results) ? round.results : [])
        .filter((r) => directionIds.has(String(r && r.id || '')));
      return { ...round, directions, results };
    }).filter((round) => round.directions.length),
    ledger: searchRegistry.map((c) => ({
      id: c.id,
      direction: c.id,
      candidate_id: c.id,
      candidate_source: c.source,
      expected: null,
      actual: c.absolute_score,
      verdict: c.status,
      lesson: c.next_blocker || c.notes || '',
    })),
  };
}

function megaCandidateHardPass(c) {
  const n = normalizeMegaCandidate(c);
  const configuredTarget = typeof TARGET_GUARDS === 'undefined'
    ? '' : TARGET_GUARDS.join(',');
  const targetReadout = megaTargetsReadout(n.paired_readings,
    String(configuredTarget || n.target_guard).split(','));
  const structuralPass = typeof STRUCTURAL_CONTRACT_BLOCKING === 'undefined' ||
    !STRUCTURAL_CONTRACT_BLOCKING || n.structural_verified;
  return structuralPass &&
    n.claim_complete && n.correctness_pass && n.activation_pass && n.head_pass &&
    n.launch_pass && n.liveness_pass && n.graph_pass && n.guards_pass &&
    n.artifact_distinct && n.id_valid && n.source_valid && n.source_id_valid &&
    n.measurement_pass && String(n.attempt_id) && String(n.evidence_manifest) &&
    n.artifact_hash_base && n.artifact_hash_candidate &&
    n.artifact_hash_base !== n.artifact_hash_candidate &&
    n.reps >= 5 && Number.isFinite(n.null_arm_pct) &&
    targetReadout.count >= 5 && Number(targetReadout.score) > 0 &&
    Math.abs(Number(targetReadout.score) - Number(n.absolute_score)) <= 0.002 &&
    targetReadout.sign_p <= 0.05 &&
    (Number(targetReadout.min) - 1) * 100 > Math.abs(n.null_arm_pct) &&
    Number.isFinite(n.absolute_score) && n.absolute_score > 0;
}

function selectMegaCandidate(registry, tieNoisePct) {
  const eligible = (Array.isArray(registry) ? registry : [])
    .map(normalizeMegaCandidate)
    .filter((c) => (c.status === 'scored' || c.status === 'finalist') &&
      megaCandidateHardPass(c) && c.absolute_score > 1.0)
    .sort((a, b) => b.absolute_score - a.absolute_score);
  if (!eligible.length) return { selected: null, eligible: [], tie_kept_skill: false };
  return { selected: eligible[0], eligible, tie_kept_skill: false };
}

function selectMegaSearchParent(registry) {
  return (Array.isArray(registry) ? registry : [])
    .map(normalizeMegaCandidate)
    .filter((c) => (c.source === 'search' || c.source === 'integrated') &&
      (c.status === 'scored' || c.status === 'finalist') &&
      megaCandidateHardPass(c) && c.absolute_score > 1.0)
    .sort((a, b) => b.absolute_score - a.absolute_score)[0] || null;
}

function megaCalibrationClaimPass(pc, configured) {
  const p = pc || {};
  const c = configured || {};
  const got = Number(p.measured_pct);
  const lo = Number(c.expected_pct_lo), hi = Number(c.expected_pct_hi);
  const sign = Math.sign(lo + hi) || 1;
  const ctrl = Array.isArray(p.control_pairs_pct)
    ? p.control_pairs_pct.map(Number).filter(Number.isFinite) : [];
  const nul = Array.isArray(p.null_pairs_pct)
    ? p.null_pairs_pct.map(Number).filter(Number.isFinite) : [];
  const requiredControlPairs = Math.max(1, Number(c.required_control_pairs || 5));
  const requiredNullPairs = Math.max(1, Number(c.required_null_pairs || 8));
  const nullWorst = nul.length ? Math.max(...nul.map(Math.abs)) : NaN;
  const minEffect = Math.min(Math.abs(lo), Math.abs(hi));
  const maxPlausible = Number.isFinite(Number(c.implausible_pct))
    ? Math.abs(Number(c.implausible_pct)) : Math.max(Math.abs(lo), Math.abs(hi)) * 2;
  return p.claim_complete === true && p.ran === true && p.switch_present === true &&
    String(p.attempt_id || '').length > 0 && String(p.evidence_manifest || '').length > 0 &&
    p.passed === true && Number.isFinite(got) && Math.sign(got) === sign &&
    Math.abs(got) >= minEffect * 0.5 && Math.abs(got) <= maxPlausible &&
    ctrl.length >= requiredControlPairs && ctrl.every((x) => Math.sign(x) === sign) &&
    nul.length >= requiredNullPairs && Number.isFinite(nullWorst) && Math.abs(got) >= 3 * nullWorst;
}

function pairedGuardReadout(rows, guard, bimodalGuards = []) {
  const rawPairs = (Array.isArray(rows) ? rows : []).filter((r) =>
    String(r.guard || '') === String(guard) &&
    Number(r.base) > 0 && Number(r.cand) > 0);
  let pairs = rawPairs;
  let bimodal = false, split = null;
  if ((Array.isArray(bimodalGuards) ? bimodalGuards : [])
      .map(String).includes(String(guard)) && rawPairs.length >= 8) {
    const values = rawPairs.flatMap((r, index) => [
      { value: Number(r.base), arm: 'base', index },
      { value: Number(r.cand), arm: 'cand', index },
    ]).sort((a, b) => a.value - b.value);
    let widest = 0, splitAt = -1;
    for (let i = 0; i + 1 < values.length; i++) {
      const gap = (values[i + 1].value - values[i].value) / values[i].value;
      if (gap > widest) { widest = gap; splitAt = i; }
    }
    if (widest >= 0.03 && splitAt >= 0) {
      const lowArms = new Set(values.slice(0, splitAt + 1).map((x) => x.arm));
      const highArms = new Set(values.slice(splitAt + 1).map((x) => x.arm));
      // Arm-blind state conditioning is valid only when BOTH arms draw both states. Otherwise the
      // gap may simply be a large real effect separating base from candidate.
      if (lowArms.size === 2 && highArms.size === 2) {
        split = (values[splitAt].value + values[splitAt + 1].value) / 2;
        pairs = rawPairs.filter((r) => Number(r.base) <= split && Number(r.cand) <= split);
        bimodal = true;
      }
    }
  }
  const ratios = pairs.map((r) => Number(r.base) / Number(r.cand)).sort((a, b) => a - b);
  const median = ratios.length
    ? (ratios.length % 2 ? ratios[(ratios.length - 1) / 2]
      : (ratios[ratios.length / 2 - 1] + ratios[ratios.length / 2]) / 2)
    : null;
  const favorable = ratios.filter((v) => v > 1.0).length;
  const choose = (n, k) => {
    let out = 1;
    for (let i = 1; i <= k; i++) out = out * (n - k + i) / i;
    return out;
  };
  let signP = 1;
  if (ratios.length) {
    let tail = 0;
    for (let k = favorable; k <= ratios.length; k++) tail += choose(ratios.length, k);
    signP = tail / Math.pow(2, ratios.length);
  }
  return {
    raw_count: rawPairs.length, count: ratios.length, score: median,
    favorable, sign_p: signP, bimodal, split,
  };
}

// Each target passes alone; headline = geomean.
function megaTargetsReadout(rows, guards) {
  const rs = guards.filter(Boolean).map((g) => pairedGuardReadout(rows, g,
    typeof BIMODAL_GUARDS === 'undefined' ? [] : BIMODAL_GUARDS));
  if (rs.length < 2) return { ...(rs[0] || { count: 0, score: null, sign_p: 1 }), min: rs[0] ? rs[0].score : null };
  const s = rs.map((r) => Number(r.score));
  return { count: Math.min(...rs.map((r) => r.count)), sign_p: Math.max(...rs.map((r) => r.sign_p)),
    score: s.every((v) => v > 0) ? Math.exp(s.reduce((a, v) => a + Math.log(v), 0) / s.length) : null,
    min: Math.min(...s) };
}

function megaFinalSelectionVerdict(selection, finalists, opts) {
  const s = selection || {};
  const o = opts || {};
  const list = Array.isArray(finalists) ? finalists.map(normalizeMegaCandidate) : [];
  const source = list.find((c) => c.id === String(s.selected_candidate_id || ''));
  const row = Array.isArray(s.candidates)
    ? s.candidates.find((c) => String(c.candidate_id || '') === String(s.selected_candidate_id || ''))
    : null;
  const reasons = [];
  const returnedIds = Array.isArray(s.candidates)
    ? s.candidates.map((c) => String(c.candidate_id || '')) : [];
  for (const candidate of list) {
    if (returnedIds.filter((id) => id === candidate.id).length !== 1) {
      reasons.push(`finalist ${candidate.id} does not have exactly one evidence row`);
    }
  }
  if (returnedIds.some((id) => !list.some((c) => c.id === id))) {
    reasons.push('selection contains an evidence row outside the supplied finalists');
  }
  const allGuards = [...new Set([...(o.targetGuards || []), ...(o.regressionGuards || [])])];
  const rowFailures = (r) => {
    const fail = [];
    if (!r || r.claim_complete !== true) return ['incomplete evidence row'];
    if (String(r.status || '').toLowerCase() !== 'verified') fail.push('verification status');
    const registered = list.find((c) => c.id === String(r.candidate_id || ''));
    if (!registered) fail.push('candidate is not registered');
    else {
      if (String(r.source || '') !== registered.source) fail.push('source mismatch');
      if (String(r.tree || '') !== registered.tree) fail.push('tree mismatch');
      if (String(r.head || '') !== registered.head) fail.push('head mismatch');
    }
    if (!(Number(r.score) > 1.0)) fail.push('score <= 1.0');
    if (typeof megaFrozenPerCase === 'function' &&
        megaFrozenPerCase(r.per_case, o.frozenBaseline, 0.05).mismatch.length) {
      fail.push('paired base drifted from the frozen baseline');
    }
    if (!String(r.correctness || '').toLowerCase().startsWith('pass')) fail.push('correctness');
    const perCase = new Map((Array.isArray(r.per_case) ? r.per_case : [])
      .map((x) => [String(x.name || ''), Number(x.speedup)]));
    const pairs = Array.isArray(r.paired_readings) ? r.paired_readings : [];
    const nullArm = r.null_arm_pct == null ? null : Number(r.null_arm_pct);
    for (const guard of o.targetGuards || []) {
      const readout = pairedGuardReadout(pairs, guard, BIMODAL_GUARDS);
      const needed = Number((o.requiredPairsByGuard || {})[guard] || o.requiredPairs || 1);
      if (readout.raw_count < needed || readout.count < Math.min(5, needed) ||
          !(Number(readout.score) > 1.0) ||
          readout.sign_p > 0.05 || !Number.isFinite(nullArm) ||
          (Number(readout.score) - 1) * 100 <= Math.abs(nullArm)) {
        fail.push(`target paired result ${guard}`);
      }
      if (readout.bimodal && readout.raw_count < 16) fail.push(`bimodal guard ${guard} needs 16 pairs`);
      if (!(perCase.get(String(guard)) > 1.0) ||
          Math.abs(Number(perCase.get(String(guard))) - Number(readout.score)) >
            Number(o.scoreTolerance || 0.002)) fail.push(`target guard ${guard}`);
    }
    for (const guard of o.regressionGuards || []) {
      const readout = pairedGuardReadout(pairs, guard, BIMODAL_GUARDS);
      const needed = Number((o.requiredPairsByGuard || {})[guard] || o.requiredPairs || 1);
      const fl = typeof RF > 'u' ? 1 : RF[guard] || 1;
      if (readout.raw_count < needed || readout.count < Math.min(5, needed) ||
          !(Number(readout.score) >= fl)) {
        fail.push(`regression paired result ${guard}`);
      }
      if (readout.bimodal && readout.raw_count < 16) fail.push(`bimodal guard ${guard} needs 16 pairs`);
      if (!(perCase.get(String(guard)) >= fl) ||
          Math.abs(Number(perCase.get(String(guard))) - Number(readout.score)) >
            Number(o.scoreTolerance || 0.002)) fail.push(`regression guard ${guard}`);
    }
    const targetScores = (o.targetGuards || []).map((guard) =>
      pairedGuardReadout(pairs, guard, BIMODAL_GUARDS).score)
      .filter((v) => Number.isFinite(Number(v)));
    const recomputedScore = targetScores.length === (o.targetGuards || []).length
      ? Math.exp(targetScores.reduce((sum, value) => sum + Math.log(Number(value)), 0) /
        targetScores.length) : null;
    if (recomputedScore == null || !Number.isFinite(Number(recomputedScore)) ||
        Math.abs(Number(r.score) - Number(recomputedScore)) > Number(o.scoreTolerance || 0.002)) {
      fail.push('headline score does not match paired target readings');
    }
    if (!Number.isFinite(nullArm)) fail.push('null arm');
    if (r.guards_pass !== true) fail.push('guard verdict');
    const worldSize = Number(o.worldSize || r.world_size);
    const marker = String(r.path_marker || '');
    if (!marker || !Number.isInteger(worldSize) || worldSize < 1 ||
        Number(r.path_marker_count) !== worldSize) {
      fail.push('candidate path marker');
    }
    const realizedLaunches = Number(r.launches);
    const baselineLaunches = Number(r.launches_base || o.baselineLaunches);
    const targetLaunches = Number(o.launchTarget || 2);
    // launch_target is an upper bound.
    const launchPass = Number.isInteger(realizedLaunches) && realizedLaunches >= 1 &&
      (o.allowPartialFusion === true || realizedLaunches <= targetLaunches) &&
      realizedLaunches < baselineLaunches;
    if (!launchPass) fail.push('no measured partial/full fusion');
    if (String(r.graph_safe || '').toLowerCase() !== 'pass') fail.push('graph safety');
    const hb = String(r.artifact_hash_base || ''), hc = String(r.artifact_hash_candidate || '');
    if (String(r.artifact_distinct || '').toLowerCase() !== 'yes' || !hb || !hc || hb === hc) {
      fail.push('artifact distinctness');
    }
    const replayRows = Array.isArray(r.replay_results) ? r.replay_results : [];
    for (const guard of allGuards) {
      const replay = replayRows.find((x) => String(x.guard || '') === String(guard));
      const needed = Number(o.requiredReplays || 0);
      if (!replay || Number(replay.count || 0) < needed ||
          String(replay.status || '').toLowerCase() !== 'pass' ||
          String(replay.graph_safe || '').toLowerCase() !== 'pass' ||
          replay.arrival_jitter !== true) fail.push(`replay ${guard}`);
    }
    if (Number(r.liveness_replays || 0) < Number(o.requiredReplays || 0)) fail.push('liveness depth');
    if (String(r.accuracy_metric || '') !== String(o.accuracyMetric || '') ||
        r.accuracy_value == null ||
        !Number.isFinite(Number(r.accuracy_value)) ||
        Number(r.accuracy_value) < 0 ||
        Number(r.accuracy_value) >= Number(o.accuracyThreshold)) fail.push('accuracy');
    if (o.requireOverlap) {
      if (String(r.overlap_measured || '').toLowerCase() !== 'yes' ||
          !(Number(r.overlap_fraction) > 0) ||
          !(Number(r.overlap_cu_fraction) > 0) ||
          r.overlap_base_reading == null || r.overlap_forced_reading == null ||
          Math.abs(Number(r.overlap_base_reading)) > 0.05 ||
          !(Number(r.overlap_forced_reading) > 0.10)) fail.push('controlled overlap');
    }
    if (o.requireAttribution) {
      const a = r.attribution || {};
      if (r.attribution_complete !== true || !(Number(a.changed_us) > 0) ||
          !(Number(a.replaced_sum_us) > 0) ||
          a.residual_ms_base == null || a.residual_ms_cand == null ||
          !Number.isFinite(Number(a.residual_ms_base)) ||
          !Number.isFinite(Number(a.residual_ms_cand)) ||
          !(o.targetGuards || []).includes(String(a.guard || '')) ||
          !String(a.method || '') || a.absolute_to_frozen !== true) fail.push('attribution');
    }
    return fail;
  };
  for (const candidate of list) {
    const candidateRow = (Array.isArray(s.candidates) ? s.candidates : [])
      .find((r) => String(r.candidate_id || '') === candidate.id);
    if (!candidateRow || candidateRow.claim_complete !== true) {
      reasons.push(`finalist ${candidate.id} did not complete independent final measurement`);
    }
  }
  if (s.claim_complete !== true) reasons.push('selection claim is incomplete');
  if (!String(s.attempt_id || '') || !String(s.evidence_manifest || '')) {
    reasons.push('selection evidence manifest is missing');
  }
  if (!source) reasons.push('selected id is not a supplied finalist');
  if (!row || row.claim_complete !== true) reasons.push('selected finalist evidence row is incomplete');
  if (source && String(s.selected_source || '') !== source.source) reasons.push('selected source mismatches registry');
  if (source && row && String(row.source || '') !== source.source) reasons.push('finalist row source mismatches registry');
  if (source && row && String(row.tree || '') !== source.tree) reasons.push('finalist row tree mismatches registry');
  if (source && row && String(row.head || '') !== source.head) reasons.push('finalist row head mismatches registry');
  if (String(s.selected_tree || '') !== String(o.selectedWorkspace || '')) {
    reasons.push('selected tree is not the controlled materialized workspace');
  }
  if (!String(s.selected_head || '') ||
      !source || String(s.materialized_from_head || '') !== source.head) {
    reasons.push('materialized workspace is not bound to the selected candidate head');
  }
  if (!(Number(s.selected_score) > 1.0)) reasons.push('selected score does not beat frozen baseline');
  if (row && Math.abs(Number(row.score) - Number(s.selected_score)) > 1e-9) {
    reasons.push('selected score mismatches finalist evidence row');
  }
  if (row) reasons.push(...rowFailures(row).map((x) => `selected finalist: ${x}`));
  const passedRows = (Array.isArray(s.candidates) ? s.candidates : [])
    .filter((r) => rowFailures(r).length === 0)
    .map((row) => {
      const scores = (o.targetGuards || []).map((guard) =>
        pairedGuardReadout(row.paired_readings, guard, BIMODAL_GUARDS).score);
      return {
        row,
        score: Math.exp(scores.reduce((sum, value) => sum + Math.log(Number(value)), 0) /
          scores.length),
      };
    })
    .sort((a, b) => b.score - a.score);
  if (passedRows.length) {
    const expected = passedRows[0];
    if (String(s.selected_candidate_id || '') !== String(expected.row.candidate_id || '')) {
      reasons.push('selection is not the fastest fully passing finalist under the tie policy');
    }
  } else {
    reasons.push('no finalist evidence row clears the full contract');
  }
  return { pass: reasons.length === 0, reasons, source, row };
}

// The absolute score is frozen-relative. A paired base arm that drifted from the frozen latency (e.g. a
// candidate-tree switch-off path that regressed shared code) is re-scored against the frozen row.
function megaFrozenPerCase(rows, frozen, tol) {
  const f = new Map((Array.isArray(frozen) ? frozen : []).map((r) =>
    [String(r && r.name || ''), Number(r && (r.latency_ms || r.baseline_ms))]));
  const mismatch = [];
  const perCase = (Array.isArray(rows) ? rows : []).map((r) => {
    const ms = f.get(String(r && r.name || '')), opt = Number(r && r.optimized_ms);
    const base = Number(r && r.baseline_ms);
    if (!(ms > 0 && opt > 0 && base > 0) || Math.abs(base - ms) / ms <= tol) return r;
    mismatch.push({ name: r.name, reported_base_ms: base, frozen_ms: ms,
      deviation_pct: Math.round(1000 * (base - ms) / ms) / 10 });
    return { ...r, baseline_ms: ms, speedup: ms / opt, switch_speedup: Number(r.speedup) };
  });
  return { perCase, mismatch };
}

function megaCandidateFromVerification(meta, ver, opts) {
  const o = opts || {};
  const frozenFix = megaFrozenPerCase((ver || {}).per_case, o.frozenBaseline,
    Number(o.denominatorTolerance || 0.05));
  const denomBad = frozenFix.mismatch.length > 0;
  const v = denomBad ? { ...ver, per_case: frozenFix.perCase } : (ver || {});
  const aggregate = denomBad ? NaN
    : Number(v.verified_weighted != null ? v.verified_weighted : v.verified_geomean);
  const score = promotionScore(v, o.promotionMetric || 'operator_e2e',
    o.targetGuards || [], o.regressionGuards || [], aggregate);
  const launches = Number(v.launch_shape && v.launch_shape.launches_cand);
  const baseLaunches = Number(v.launch_shape && v.launch_shape.launches_base);
  const targetLaunches = Number(o.launchTarget || 2);
  // launch_target is an upper bound; partial fusion lifts it.
  const launchPass = o.allowPartialFusion === true
    ? (Number.isInteger(launches) && Number.isInteger(baseLaunches) &&
       launches >= 1 && launches < baseLaunches)
    : (Number.isInteger(launches) && launches >= 1 && launches <= targetLaunches &&
       (!Number.isInteger(baseLaunches) || launches < baseLaunches));
  const accuracy = Array.isArray(v.accuracy_results) ? v.accuracy_results : [];
  const metricAccuracy = accuracy.filter((a) =>
    !o.accuracyMetric || String(a.metric) === String(o.accuracyMetric));
  const requiredAccuracyCases = Array.isArray(o.requiredAccuracyCases)
    ? o.requiredAccuracyCases.map(String) : [];
  const requiredAccuracy = requiredAccuracyCases.length
    ? requiredAccuracyCases.map((name) =>
      metricAccuracy.find((a) => {
        const guard = String(a.guard || '');
        return guard === name || guard.startsWith(`${name}_`);
      })).filter(Boolean)
    : metricAccuracy.filter((a) =>
      !(o.targetGuards || []).length || (o.targetGuards || []).includes(String(a.guard)));
  const accuracyPass = requiredAccuracy.length > 0 &&
    (!requiredAccuracyCases.length || requiredAccuracy.length === requiredAccuracyCases.length) &&
    requiredAccuracy.every((a) => Number.isFinite(Number(a.value)) &&
      Number(a.value) >= 0 && Number(a.value) < Number(o.accuracyThreshold || a.threshold));
  const livenessPass = String(v.liveness || '').toLowerCase() === 'pass';
  const guards = guardContract(v, o.targetGuards || [], o.regressionGuards || [], score);
  const targetGuard = (o.targetGuards || []).join(',');
  const targetReadout = megaTargetsReadout(v.paired_readings, o.targetGuards || []);
  const nullArm = v.null_arm_pct == null ? null : Number(v.null_arm_pct);
  const measurementPass = !!targetGuard &&
    targetReadout.count >= Number(o.requiredPairs || 1) &&
    Number.isFinite(Number(targetReadout.score)) &&
    Math.abs(Number(targetReadout.score) - Number(score)) <= Number(o.scoreTolerance || 0.002) &&
    targetReadout.sign_p <= 0.05 &&
    Number.isFinite(nullArm) &&
    (Number(targetReadout.min) - 1) * 100 > Math.abs(nullArm);
  const verificationStatusPass = String(v.status || '').toLowerCase() === 'verified';
  const verificationFailed = v.claim_complete === true && !verificationStatusPass;
  const record = normalizeMegaCandidate({
    ...(meta || {}),
    claim_complete: v.claim_complete === true,
    attempt_id: v.attempt_id || (meta && meta.attempt_id),
    evidence_manifest: v.evidence_manifest || '',
    target_guard: targetGuard,
    absolute_score: score,
    per_case: v.per_case || [],
    paired_readings: v.paired_readings || [],
    denominator_mismatch: frozenFix.mismatch,
    component_breakdown: v.component_breakdown || [],
    hot_path_counters: v.hot_path_counters || {},
    null_arm_pct: nullArm,
    reps: Number(v.reps || targetReadout.count),
    artifact_hash_base: v.artifact_hash_base || '',
    artifact_hash_candidate: v.artifact_hash_candidate || '',
    launches,
    correctness_pass: verificationStatusPass &&
      accuracyPass && String(v.correctness || '').toLowerCase().startsWith('pass'),
    activation_pass: verificationStatusPass &&
      String(v.activation_confirmed || '').toLowerCase() === 'yes' &&
      String(v.activation_on_hardware || '').toLowerCase() === 'yes',
    head_pass: !!String(meta && meta.head || '') &&
      String(v.candidate_head || '') === String(meta && meta.head || ''),
    launch_pass: launchPass,
    liveness_pass: livenessPass && Number(v.replay_count || 0) >= Number(o.requiredReplays || 30),
    graph_pass: String(v.graph_safe || '').toLowerCase() === 'pass',
    guards_pass: guards.regression_pass !== false,
    artifact_distinct: String(v.artifact_distinct || '').toLowerCase() === 'yes' ||
      (!!v.artifact_hash_base && !!v.artifact_hash_candidate &&
       v.artifact_hash_base !== v.artifact_hash_candidate),
    measurement_pass: verificationStatusPass && measurementPass,
    runtime_verified: verificationStatusPass && accuracyPass &&
      String(v.activation_on_hardware || '').toLowerCase() === 'yes' &&
      launchPass && livenessPass,
    score_complete: verificationStatusPass && measurementPass,
    verification_status: String(v.status || ''),
    correctness: String(v.correctness || ''),
    gpu_executed: String(v.activation_on_hardware || '').toLowerCase() === 'yes',
    activation_on_hardware: String(v.activation_on_hardware || ''),
    next_blocker: (denomBad ? `DENOMINATOR: paired base deviates from the frozen baseline (${
      frozenFix.mismatch.map((m) => `${m.name} ${m.reported_base_ms}ms vs frozen ${m.frozen_ms}ms, ${
        m.deviation_pct}%`).join('; ')}); score re-based on frozen latency. If the base arm was the frozen tree ` +
      'this is machine drift: re-check GPU state and re-measure, do not edit code for it; if it was the ' +
      'candidate switch-off path, shared code regressed: fix that first. ' : '') + (verificationFailed
      ? `independent Verify ${v.status || 'failed'} at HEAD ${
        v.candidate_head || meta && meta.head || '(missing)'}; correctness=${
        v.correctness || 'unknown'}, activation_on_hardware=${
        v.activation_on_hardware || 'unknown'}; evidence=${
        v.evidence_manifest || '(missing)'}; prior_handoff=${
        meta && meta.next_blocker || '(none)'}`
      : String(meta && meta.next_blocker || '')),
    notes: v.claim_complete === true && v.notes ? String(v.notes) :
      String(meta && meta.notes || ''),
  });
  record.status = megaCandidateHardPass(record) ? 'scored'
    : (record.correctness_pass && record.activation_pass && record.launch_pass ? 'runnable' : 'authoring');
  return record;
}
// <</REPLAY:mega_candidate_registry>>

// <<REPLAY:enabling_step>>
   // a prerequisite that costs more than this is a design error
const CHAIN_DEBT_MAX_ROUNDS = 2;

const round2 = (x) => Math.round(Number(x) * 100) / 100;

function stepRoleOf(d) {
  const s = String((d && d.step_role) || '').trim().toLowerCase();
  return s === 'enabling' ? 'enabling' : 'terminal';
}

// The outstanding balance. `debt` is a list of {round, id, enables, cost_pct}.
function chainDebtReport(debt, roundNow, ladderMeasured) {
  const open = (Array.isArray(debt) ? debt : []).filter((e) => e && !ladderMeasured.has(e.enables));
  if (!open.length) return { open: [], overdue: [], caveat: '' };
  const byRung = new Map();
  for (const e of open) {
    const k = e.enables;
    const cur = byRung.get(k) || { enables: k, steps: [], cost_pct: 0, oldest_round: e.round };
    cur.steps.push(e.id);
    cur.cost_pct = round2(cur.cost_pct + (Number(e.cost_pct) || 0));
    cur.oldest_round = Math.min(cur.oldest_round, e.round);
    byRung.set(k, cur);
  }
  const rows = [...byRung.values()];
  const overdue = rows.filter((r) => roundNow - r.oldest_round >= CHAIN_DEBT_MAX_ROUNDS);
  const caveat = 'CHAIN DEBT OUTSTANDING: ' + rows.map((r) =>
    `${r.enables} owes ${r.cost_pct}% from [${r.steps.join(', ')}] since round ${r.oldest_round}`).join('; ') +
    '. These steps were committed on function, not on speed, so the canonical tree is currently ' +
    'that much slower and none of it has been paid back yet. The terminal rung is the only thing ' +
    'that can pay it.' +
    (overdue.length ? ` OVERDUE (${CHAIN_DEBT_MAX_ROUNDS}+ rounds unpaid): ${overdue.map((r) => r.enables).join(', ')}. ` +
      'Plan the terminal rung this round or state what happens to the debt — a chain abandoned ' +
      'half-built leaves the tree slower than it was found, which is the worst of the three outcomes.' : '');
  return { open: rows, overdue, caveat };
}
// <</REPLAY:enabling_step>>

const REQUIRE_TASK_GRAPH = !!A.require_task_graph;
// Overlap caveats, carried to the report. TWO_LAUNCH_REACHED ratchets true once any verified
// candidate meets the launch target.
let TWO_LAUNCH_REACHED = false;
let AUTONOMY_ACCEPTANCE_REACHED = false;
const ACCEPTANCE_CAVEATS = [];
const OVERLAP_CAVEATS = [];
// Launch-shape caveats: launches per candidate and whether the target shape was met.
const LAUNCH_CAVEATS = [];
// Guard caveats: under-sampled bimodal target readings, or waves with only off-target readings.
const GUARD_CAVEATS = [];
// Withdrawn timing readings, so a voided number is still visible in the report.
const OBJECTIVE_CAVEATS = [];
// Attribution caveats and rejections, so "refused for this reason" is distinguishable from "no
// candidate" in the report.
const ATTRIBUTION_CAVEATS = [];
// Chain caveats: enabling steps kept or refused and the outstanding debt, so a half-built fusion is
// not reported as a null result.
const CHAIN_CAVEATS = [];
// Enabling steps committed on function, each {round, id, enables, cost_pct}. Cleared per rung when
// that rung is finally measured.
const CHAIN_DEBT = [];
// Rung ids that have produced a measurement. Kept alongside rungTally because the debt report needs
// the same fact and must not depend on the tally's grading order.
const LADDER_MEASURED = new Set();
// Cumulative speedup and canonical commit pinned at a chain's first enabling step; the terminal
// step is measured against this pin so the chain's own overhead is not counted as a win.
let CHAIN_BASELINE = null;
// Hardware/toolchain facts this wave established, for a human to merge into knowledge/. See
// MEMORY_SCHEMA.knowledge_delta for why this is surfaced rather than written.
const KNOWLEDGE_DELTA = [];
let TASK_GRAPH_CAVEAT = '';
let PIPE_TABLE_CAVEAT = '';
const dispatchedRungs = new Set();
// Separate from the set above and it has to be: that one answers "was this taken", this one answers
// "did taking it produce a number". Only the second can say what the next wave still owes.
const rungTally = new Map();
function recordRungOutcome(id, outcome) {
  const key = String(id || '').trim();
  if (!key) return;
  const e = rungTally.get(key) || { attempts: 0, last_outcome: 'never_planned' };
  // `measured` is absorbing. A rung that produced a number once is closed; a later round that takes
  // it again and faults must not reopen it, or the ladder never terminates.
  if (e.last_outcome !== 'measured') e.last_outcome = outcome;
  rungTally.set(key, e);
}

// Fast-test cache (mega + STATE_DIR): reuse a prior wave's front-matter measurements when the
// validity key matches. Disk I/O goes through an agent. When off, no agent() call is made, so the
// call sequence and resume cache keys are unchanged.
const FAST_TEST_CACHE_DIR = STATE_DIR ? `${STATE_DIR}/fast_test_cache` : '';
async function fastTestCacheLoad(kind /* 'bench' | 'profile' */) {
  if (!MEGA_FAST_TEST) return null;
  const res = await agentT(
    roleAgent('benchmark_engineer', 'fast_test_load',
      `FAST-TEST CACHE LOAD (${kind}). NO GPU, NO lease: read the cache and validate its key, measure ` +
      `nothing. Reconstruct the ${kind === 'bench' ? 'benchmark' : 'mega_analysis'} payload from the ` +
      `cached artifacts ONLY if the stored key matches the key you recompute from disk now.`, {
        CACHE_DIR: FAST_TEST_CACHE_DIR, CACHE_KIND: kind, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR,
        FROZEN_KERNEL_PATH: KERNEL_PATH_ORIG,
        BENCH_HARNESS,
        FAST_TEST_KEY_TOOL: FAST_TEST_KEY_TOOL, CONTROL_JSON: JSON.stringify(POSITIVE_CONTROL),
        GUARDS_JSON: JSON.stringify([...TARGET_GUARDS, ...REGRESSION_GUARDS]),
        TARGET_GUARDS, REGRESSION_GUARDS,
        ...(POSITIVE_CONTROL ? { POSITIVE_CONTROL } : {}),
      }),
    { phase: 'Benchmark', label: `fast_test_load:${kind}`, schema: FAST_TEST_LOAD_SCHEMA,
      timeout_ms: 300000, max_retries: 1 });
  const commandmentReady = kind !== 'bench' ||
    (res && res.commandment_materialized === true &&
     res.commandment_current_identity === true &&
     String(res.bench && res.bench.commandment_path || '') === COMMANDMENT);
  if (!res || res.key_valid !== true || !commandmentReady) {
    log(`Fast-test cache MISS (${kind}): ` +
        `${res ? (!commandmentReady ? 'current EVAL_DIR COMMANDMENT was not materialized' :
          (res.note || (res.cache_present ? 'stored key mismatches current key' : 'no cache on disk'))) :
          'load agent returned nothing'}` +
        ` — running the real ${kind} phase fresh (and re-publishing).`);
    return null;
  }
  log(`Fast-test cache HIT (${kind}): reusing prior on-card artifacts (key ${String(res.cached_key || '').slice(0, 12)}…). ${res.note || ''}`);
  return res;
}
async function fastTestCachePublish() {
  if (!MEGA_FAST_TEST) return;
  const res = await agentT(
    roleAgent('benchmark_engineer', 'fast_test_publish',
      'FAST-TEST CACHE PUBLISH. NO GPU, NO lease: copy THIS wave\'s freshly-measured artifacts into the ' +
      'cache and write the validity key, so a later fast-test wave can reuse them. Copy files, measure ' +
      'nothing.', {
        CACHE_DIR: FAST_TEST_CACHE_DIR, EVAL_DIR, MEGA_ANALYSIS_DIR: `${EVAL_DIR}/mega_analysis`,
        SKILL_DIR: WORKFLOW_DIR, FROZEN_KERNEL_PATH: KERNEL_PATH_ORIG,
        BENCH_HARNESS,
        FAST_TEST_KEY_TOOL: FAST_TEST_KEY_TOOL, CONTROL_JSON: JSON.stringify(POSITIVE_CONTROL),
        GUARDS_JSON: JSON.stringify([...TARGET_GUARDS, ...REGRESSION_GUARDS]),
        TARGET_GUARDS, REGRESSION_GUARDS,
        ...(POSITIVE_CONTROL ? { POSITIVE_CONTROL } : {}),
      }),
    { phase: 'Profile', label: 'fast_test_publish', schema: FAST_TEST_PUBLISH_SCHEMA,
      timeout_ms: 300000, max_retries: 1 });
  if (res && res.published) log(`Fast-test cache PUBLISHED to ${res.cache_dir || FAST_TEST_CACHE_DIR} (key ${String(res.key || '').slice(0, 12)}…, ${(res.files || []).length} files).`);
  else log(`Fast-test cache publish did not complete: ${res ? (res.note || 'no confirmation') : 'agent returned nothing'}.`);
}
const structuralGuardNames = [...new Set([...TARGET_GUARDS, ...REGRESSION_GUARDS])];
const noGpuFrontMatter = MEGA_STRUCTURAL_ONLY || MEGA_ROUTE_ONLY;

// Positive-control gate. Runs BEFORE any direction budget is spent, because the thing it can prove
// false — "this harness can detect a real win" — invalidates every measurement taken after it.
let PC_OVERSHOOT = '';
let PC_UNDERSHOOT = '';
// Whether a control actually ran in this wave (a missing switch means two identical arms).
let PC_RAN = false;
let PC_PASSED = false;
// Rounds with no admissible measurement. Separate from noImprove (no wins vs no readings); both stop
// at MAX_NO_IMPROVE and are never summed. See roundEvidence.
let noEvidence = 0;
// Rounds in which the switched path never reached a card, under every objective including
// working_kernel. Any round that reaches the card (even to crash) resets it. Inert unless
// REQUIRE_HW_ACTIVATION.
let noHardware = 0;
// max_no_hardware (default MAX_NO_IMPROVE): raise it for whole-fusion authoring, whose early rounds
// may crash before launch; a run truly stuck off-hardware still stops.
const MAX_NO_HARDWARE = Math.max(1, parseInt(A.max_no_hardware != null ? A.max_no_hardware : MAX_NO_IMPROVE, 10));
// Why the round loop ended, recorded for the report.
let stopReason = null;
// The blackboard behind `history.insights`. Kept as records rather than strings so an insight can
// carry the round it came from and whether that round produced evidence at all.
let insightBook = [];
// Candidate shelf: verified non-winners kept as offers; absorbedByRound[r] is the file set CANONICAL
// took in round r, which ages shelved patches out.
let shelf = [];
let absorbedByRound = {};
// K historical candidates offered to the integrator per round. Kept small because every offer costs
// verification time on the shared lease; size it from the measured hit rate (0 if offers never combine).
const SHELF_OFFER_K = Number(A.shelf_offer_k) > 0 ? Number(A.shelf_offer_k) : 2;

// <<REPLAY:memory_merge>>
// Cross-round memory is append-and-age, never replaced: an insight the summariser stops repeating
// survives, every insight keeps its origin round, and evictions from a full board are reported.
function normInsight(s) {
  return String(s == null ? '' : s)
    .replace(/^\[r\d+[^\]]*\]\s*/, '')     // strip a provenance tag if the agent echoed one back
    .replace(/\s+/g, ' ').trim().toLowerCase();
}
function mergeInsights(book, incoming, round, roundVoid, max) {
  const MAX = Number.isFinite(max) ? max : 40;
  const out = book.map((e) => Object.assign({}, e));
  const index = new Map(out.map((e, i) => [normInsight(e.text), i]));
  for (const raw of (Array.isArray(incoming) ? incoming : [])) {
    const text = String(raw == null ? '' : raw).replace(/^\[r\d+[^\]]*\]\s*/, '').trim();
    if (!text) continue;
    const k = normInsight(text);
    if (index.has(k)) {
      // Restating an insight refreshes it and is what keeps a still-relevant finding alive under
      // ageing — but it cannot upgrade a finding that came out of a round with no evidence.
      const e = out[index.get(k)];
      e.last_round = round; e.restated = (e.restated || 0) + 1;
      if (!roundVoid) e.void_round = e.void_round && false;
      continue;
    }
    out.push({ text, first_round: round, last_round: round, restated: 0, void_round: !!roundVoid });
    index.set(k, out.length - 1);
  }
  if (out.length <= MAX) return { book: out, evicted: [] };
  // Age out by last-seen round, then by how often the finding kept coming back. Ties break toward
  // the more recent entry.
  const ranked = out.map((e, i) => ({ e, i })).sort((a, b) =>
    (b.e.last_round - a.e.last_round) || ((b.e.restated || 0) - (a.e.restated || 0)) || (b.i - a.i));
  const keep = new Set(ranked.slice(0, MAX).map((x) => x.i));
  return { book: out.filter((_, i) => keep.has(i)), evicted: out.filter((_, i) => !keep.has(i)) };
}
function renderInsights(book) {
  return book.map((e) => `[r${e.first_round}` +
    (e.last_round !== e.first_round ? `-r${e.last_round}` : '') +
    (e.void_round ? ' FROM-VOID-ROUND' : '') + `] ${e.text}`);
}
// <</REPLAY:memory_merge>>

// <<REPLAY:candidate_shelf>>
// The shelf keeps verified non-winners as offers. A patch is a diff against the CANONICAL it was cut
// from, so eligibility is decided by file sets: if nothing absorbed since then touches its files, it
// still applies. An empty file set means an unknown footprint; such entries are held back and named.
const shelfFile = (s) => String(s == null ? '' : s).replace(/^\.\//, '').replace(/^\/+/, '').trim();
const shelfFiles = (c) => [...new Set((Array.isArray(c.touched_files) ? c.touched_files : [])
  .map(shelfFile).filter(Boolean))];

// <</REPLAY:candidate_shelf>>

// <<REPLAY:bimodal_split>>
  // a gap smaller than this is a spread, not two states.

// Credit comes only from explicit target guards; regression guards are vetoes. Uses per_case (the
// verifier's noise-aware result).
function guardContract(result, targetGuards, regressionGuards, fallbackScore) {
  const rows = Array.isArray(result && result.per_case) ? result.per_case : [];
  const byName = new Map(rows.map((r) => [String(r.name || '').trim(), r]));
  const targets = (Array.isArray(targetGuards) ? targetGuards : []).map(String).filter(Boolean);
  const regressions = (Array.isArray(regressionGuards) ? regressionGuards : []).map(String).filter(Boolean);
  const missingTargets = targets.filter((g) => !byName.has(g));
  const missingRegressions = regressions.filter((g) => !byName.has(g));
  const targetValues = targets.map((g) => Number((byName.get(g) || {}).speedup))
    .filter((v) => Number.isFinite(v) && v > 0);
  const score = targets.length
    ? (targetValues.length === targets.length
      ? Math.exp(targetValues.reduce((s, v) => s + Math.log(v), 0) / targetValues.length)
      : 0)
    : Number(fallbackScore || 0);
  const regressed = regressions.filter((g) => {
    const v = Number((byName.get(g) || {}).speedup);
    return Number.isFinite(v) && v < (typeof RF > 'u' ? 1 : RF[g] || 1);
  });
  return {
    score,
    complete: missingTargets.length === 0 && missingRegressions.length === 0,
    regression_pass: missingRegressions.length === 0 && regressed.length === 0,
    missing_targets: missingTargets,
    missing_regressions: missingRegressions,
    regressed,
  };
}

function promotionScore(result, metric, targetGuards, regressionGuards, fallbackScore) {
  if (!result) return 0;
  if (metric === 'changed_kernel') {
    const a = result.attribution || {};
    const changed = Number(a.changed_us), replaced = Number(a.replaced_sum_us);
    return a.absolute_to_frozen === true &&
      Number.isFinite(changed) && changed > 0 && Number.isFinite(replaced) && replaced > 0
      ? replaced / changed : 0;
  }
  return guardContract(result, targetGuards, regressionGuards, fallbackScore).score;
}
// <</REPLAY:bimodal_split>>

const ATTRIBUTION_METRIC = PROMOTION_METRIC === 'legacy' ? 'changed_kernel' : PROMOTION_METRIC;

// A COMPLETE control that failed the gate is re-measured fresh (bounded); the gate never changes.
async function remeasureMegaCalibration(label, prior) {
  if (megaMeasurementCalibration.ready || !POSITIVE_CONTROL ||
      MEGA_STRUCTURAL_ONLY || MEGA_ROUTE_ONLY || megaPcRemeasured >= 2 ||
      !prior || prior.claim_complete !== true) return;
  megaPcRemeasured += 1;
  const r = await agentT(
    roleAgent('benchmark_engineer', 'recalibrate',
      `RE-MEASURE the positive control fresh (attempt ${megaPcRemeasured}/2): the last complete ` +
      `attempt failed the gate. Same gate; never re-gate, drop or reuse pairs.`, {
        WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_SINGLE,
        ANALYSIS: analysis, POSITIVE_CONTROL, PREVIOUS_CONTROL: prior,
        TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC,
        REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
      }),
    { phase: 'Benchmark', label: `mega recalibration ${label}`, schema: MEGA_BENCH_SCHEMA,
      ...(MEGA_PRODUCTION ? { timeout_ms: 7200000, timeout_marker: true, max_retries: 1 } : {}) });
  const pc = r && r.positive_control;
  const ok = megaCalibrationClaimPass(pc, POSITIVE_CONTROL);
  log(`MEGA recalibration ${megaPcRemeasured}/2 (${label}): ` + (pc ? `${pc.measured_pct}%, ` +
    `worst null ${Math.max(...(pc.null_pairs_pct || [NaN]).map(Math.abs))}% -> ` : '') +
    (ok ? 'PASS' : 'FAIL'));
  if (ok) {
    megaMeasurementCalibration = {
      ready: true, candidate_id: 'synthetic_control',
      attempt_id: pc.attempt_id || `recalibrated:${label}`,
      evidence_manifest: pc.evidence_manifest || '',
      note: `re-measured synthetic control passed at ${Number(pc.measured_pct).toFixed(2)}% ` +
        `(attempt ${megaPcRemeasured}; earlier failed attempts kept on disk)`,
    };
  }
}

async function recoverMegaCalibration(label) {
  if (megaMeasurementCalibration.ready || !POSITIVE_CONTROL) return;
  const recovered = await agentT(
    roleAgent('benchmark_engineer', 'recover',
      `RECOVER ONLY the completed positive-control aggregate from ${EVAL_DIR}. Do not run a GPU ` +
      `command or take a lease. Ignore force-emitted/partial claims and return claim_complete:false ` +
      `unless the evidence manifest and every pair are final.`, {
        WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR,
        ANALYSIS: analysis, POSITIVE_CONTROL,
        TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC,
        REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
      }),
    { phase: 'Benchmark', label: `mega calibration recovery ${label}`, schema: MEGA_BENCH_SCHEMA,
      ...(MEGA_PRODUCTION ? { timeout_ms: 300000, max_retries: 1 } : {}) });
  const pc = recovered && recovered.positive_control;
  if (megaCalibrationClaimPass(pc, POSITIVE_CONTROL)) {
    megaMeasurementCalibration = {
      ready: true, candidate_id: 'synthetic_control',
      attempt_id: pc.attempt_id || `recovered:${label}`,
      evidence_manifest: pc.evidence_manifest || '',
      note: `recovered completed synthetic control at ${Number(pc.measured_pct).toFixed(2)}%`,
    };
    log(`MEGA measurement calibration RECOVERED (${label}): ${megaMeasurementCalibration.note}.`);
  } else await remeasureMegaCalibration(label, pc);
}

async function persistMegaCandidateState(currentRound, finalizing) {
  if (!STATE_DIR) return true;
  const sequence = megaStateSequenceBase + Number(currentRound) * 10 + (finalizing ? 9 : 0);
  const generation = `mega:${currentRound}:${megaCandidateRegistry.reduce(
    (m, c) => Math.max(m, Number(c.attempts || 0)), 0)}:${sequence}`;
  const persisted = await agentT(
    roleAgent('tech_lead', 'update_memory',
      'Persist the mega candidate registry exactly; do not benchmark, plan, or modify candidate source.', {
        STATE_DIR, CANONICAL, ROUND: currentRound,
        CUMULATIVE_SPEEDUP: cumulative, BEST_PER_CASE: bestPerCase,
        CANDIDATE_REGISTRY: megaCandidateRegistry,
        MEASUREMENT_CALIBRATION: megaMeasurementCalibration,
        STATE_GENERATION: generation,
        STATE_SEQUENCE: sequence,
        PRIOR_HISTORY: history, OPEN_RUNGS: [], ROADMAP_LADDER: [],
        LADDER_DISPATCHED: [], LADDER_COMPLETED: [],
        SHELF: shelf, ABSORBED_FILES: absorbedByRound,
      }),
    { phase: 'Optimize', label: `mega candidate state r${currentRound}`, schema: MEGA_MEMORY_SCHEMA,
      // Large late-round registries need a real-time hung guard wider than the modeled turn clock.
      ...(MEGA_PRODUCTION ? { timeout_ms: 600000, max_retries: 1 } : {}) });
  const echoOk = (r) => Boolean(r && r.state_written === true &&
    Number(r.state_round) === Number(currentRound) &&
    Number(r.state_sequence) === sequence &&
    String(r.state_generation || '') === generation);
  if (!echoOk(persisted)) {
    // Distinguish a landed write with a lost echo from a genuinely missing write.
    const readState = (labelSuffix) => agentT(
      roleAgent('tech_lead', 'update_memory',
        'READ-ONLY verification: read STATE.json in STATE_DIR and report its state_written (true iff ' +
        'the file exists and parses as JSON), state_round (its last_round), state_sequence, and ' +
        'state_generation EXACTLY as stored. Do NOT write, benchmark, plan, or modify anything.', {
          STATE_DIR, ROUND: currentRound,
          EXPECT_SEQUENCE: sequence, EXPECT_GENERATION: generation,
        }),
      { phase: 'Optimize', label: `mega state verify r${currentRound}${labelSuffix || ''}`, schema: MEGA_MEMORY_SCHEMA,
        ...(MEGA_PRODUCTION ? { timeout_ms: 180000, max_retries: 1 } : {}) });
    let verify = await readState('');
    if (!echoOk(verify)) {
      // Genuinely-missing write. Retry the WRITE once with a larger cap (registry grown by round 4+).
      const retry = await agentT(
        roleAgent('tech_lead', 'update_memory',
          'RETRY persist: the prior attempt did not land STATE.json on disk. Persist the mega ' +
          'candidate registry exactly and WRITE STATE.json now; do not benchmark, plan, or modify ' +
          'candidate source.', {
            STATE_DIR, CANONICAL, ROUND: currentRound,
            CUMULATIVE_SPEEDUP: cumulative, BEST_PER_CASE: bestPerCase,
            CANDIDATE_REGISTRY: megaCandidateRegistry,
            MEASUREMENT_CALIBRATION: megaMeasurementCalibration,
            STATE_GENERATION: generation, STATE_SEQUENCE: sequence,
            PRIOR_HISTORY: history, OPEN_RUNGS: [], ROADMAP_LADDER: [],
            LADDER_DISPATCHED: [], LADDER_COMPLETED: [],
            SHELF: shelf, ABSORBED_FILES: absorbedByRound,
          }),
        { phase: 'Optimize', label: `mega candidate state r${currentRound} retry`, schema: MEGA_MEMORY_SCHEMA,
          ...(MEGA_PRODUCTION ? { timeout_ms: 600000, max_retries: 1 } : {}) });
      if (!echoOk(retry)) verify = await readState(' recheck');
      if (!echoOk(retry) && !echoOk(verify)) {
        // In-memory-only checkpoint is not resumable. Stop before a round advances from stale state (lane git commit stays intact).
        log(`MEGA STATE PERSIST FAILED r${currentRound}: STATE.json did not advance to seq=${sequence} ` +
          `after retry. The invocation must stop; continuing would lose structured candidate progress.`);
        return false;
      }
    }
    log(`MEGA state persist r${currentRound}: primary echo was incomplete, but STATE.json on disk ` +
      `verifies (seq=${sequence}, gen=${generation}); lineage intact, continuing.`);
  }
  return true;
}

function megaCandidateById(id) {
  return megaCandidateRegistry.find((c) => c.id === id) || null;
}

function megaLaneTree(candidateId) {
  if (!validMegaCandidateId(candidateId)) {
    throw new Error(`invalid mega candidate id '${candidateId}'`);
  }
  const prior = megaCandidateById(candidateId);
  if (prior && prior.tree) return prior.tree;
  return STATE_DIR
    ? `${STATE_DIR}/candidates/${candidateId}/tree`
    : `${EVAL_DIR}/candidate_lanes/${candidateId}`;
}

function megaBaseTree(baseId, requesterSource) {
  if (!baseId || baseId === 'frozen_baseline') return `${EVAL_DIR}/baseline`;
  const base = megaCandidateById(baseId);
  return base && base.tree && base.status !== 'rejected' &&
    (!(requesterSource === 'search' || requesterSource === 'integrated') ||
      base.source === 'search' || base.source === 'integrated') &&
    (base.status === 'scored' || base.status === 'finalist') &&
    base.id_valid && base.source_valid && base.source_id_valid
    ? base.tree : `${EVAL_DIR}/baseline`;
}

function megaBaseHead(baseId, requesterSource) {
  if (!baseId || baseId === 'frozen_baseline') return '';
  const base = megaCandidateById(baseId);
  return base &&
    (!(requesterSource === 'search' || requesterSource === 'integrated') ||
      base.source === 'search' || base.source === 'integrated') &&
    (base.status === 'scored' || base.status === 'finalist')
    ? String(base.head || '') : '';
}

function validMegaSearchDirection(direction) {
  const d = direction || {};
  const identity = `${d.candidate_id || d.id || ''} ${d.title || ''}`.toLowerCase();
  // Reject only explicit diagnostic-only work, never generic "control" substrings.
  if (/\b(?:no[-_ ]?payload|diagnostic[-_ ]only|instrument(?:ation)?[-_ ]only|meter(?:ing)?[-_ ]only|control[-_ ]only|profil(?:e|ing)[-_ ]only)\b/.test(identity)) {
    return false;
  }
  return true;
}

function freshMegaDirectionLeak(direction) {
  const d = direction || {};
  const text = [
    d.title, d.prompt, d.fills_hole, d.enables,
  ].map((value) => String(value || '')).join('\n');
  const externalArtifactPath = /(?:^|[\s"'`])\/[^\s"'`]*(?:geak_state|geak_runs)\//i.test(text);
  // A negation ("do not read X, prior lanes/state") covers its clause; a ", continue ..." ends it.
  const asserted = text.replace(/\b(?:no|not|never|without|forbid\w*|avoid\w*)\b[^.;\n]{0,240}/gi,
    (clause) => {
      const cut = clause.search(/,\s*(?:and\s+|then\s+|but\s+)?(?:continue|resume|reconcile|reuse|build on|start from)\b/i);
      return ' ' + (cut >= 0 ? clause.slice(cut) : '');
    });
  const priorArtifactClaim =
    /\b(?:continue|resume|reconcile|existing|prior|stale)\b[\s\S]{0,96}\b(?:candidate|lane|checkpoint|head|state|revision)\b/i
      .test(asserted);
  return externalArtifactPath || priorArtifactClaim;
}

function freshMegaAuthorLeak(result) {
  const r = result || {};
  const text = [r.notes, r.next_blocker].map((value) => String(value || '')).join('\n')
    .replace(/\bcop(?:y|ied|ying)\b[\s\S]{0,64}\bfrozen[_ -]?baseline\b/ig, '')
    .replace(/\b(?:no|not|never|without)\b[^.;\n]{0,160}/gi, ' ');
  // Reproducing the pinned Skill or the current plan is authoring; only prior-candidate reuse leaks.
  return /\b(?:reproduc(?:e|ed|ing)|cop(?:y|ied|ying)|reus(?:e|ed|ing))\b[\s\S]{0,128}\b(?:prior|previous|v\d+|candidate (?:source|lineage))\b/i
    .test(text);
}

function safeVerifyTimeoutRecovery(result) {
  const r = result || {};
  return r.timeout_recovery_safe === true &&
    Number(r.active_gpu_processes || 0) === 0 &&
    r.lane_lock_free === true;
}

function megaWipContinuationDirection(currentRound) {
  const lane = megaCandidateRegistry.find((c) =>
    (c.source === 'search' || c.source === 'integrated') &&
    (c.status === 'authoring' || c.status === 'runnable'));
  if (!lane) return null;
  // Resume from the working checkpoint; the top-level record can hold an older (empty) topology.
  const e = typeof megaEffectiveLaneCheckpoint === 'function' ? megaEffectiveLaneCheckpoint(lane) : lane;
  const wip = { ...lane,
    topology: e.topology && Object.keys(e.topology).length ? e.topology : (lane.topology || {}),
    contract_failures: Array.isArray(e.contract_failures) ? e.contract_failures : [],
    next_blocker: e.next_blocker || lane.next_blocker };
  const sem = wip.contract_failures.filter((f) => f && f.tier !== 'surface');
  const surf = wip.contract_failures.filter((f) => f && f.tier === 'surface');
  const target = analysis && analysis.mega_plan_ir && analysis.mega_plan_ir.target || {};
  const wipLaunches = topologyLaunchCount(wip.topology);
  const reachesTarget = Number.isFinite(wipLaunches) &&
    wipLaunches === Number(target.launch_count);
    return {
    id: `r${currentRound}_continue_${wip.id}`, candidate_id: wip.id,
    candidate_source: wip.source, base_candidate_id: wip.base_id,
    title: `continue candidate ${wip.id}`, specialty: MEGA_DEFAULT_SPECIALTY,
    tree: wip.tree,
    target_topology: {
      launch_count: wipLaunches,
      included_regions: Array.isArray(wip.topology.included_regions)
        ? wip.topology.included_regions
        : (Array.isArray(wip.topology.fused_stages) ? wip.topology.fused_stages : []),
      included_queues: Array.isArray(wip.topology.included_queues)
        ? wip.topology.included_queues : (reachesTarget ? target.required_queues || [] : []),
      capabilities: Array.isArray(wip.topology.capabilities)
        ? wip.topology.capabilities
        : (reachesTarget ? target.required_capabilities || [] : []),
      parameters: wip.topology.parameters || {},
      require_overlap: wip.topology.require_overlap === true,
    },
    contract_failures: wip.contract_failures,
    prompt: (sem.length
      ? `Resolve required semantic contract failures: ${JSON.stringify(sem)}`
      : (wip.next_blocker || 'Continue the first unresolved measured blocker.')) +
      (surf.length ? ` Surface-tier spelling hints (never outrank runtime evidence): ${
        surf.map((f) => f.id).join(', ')}.` : ''),
  };
}

function bindMegaDirectionToWip(currentRound, direction) {
  const continuation = megaWipContinuationDirection(currentRound);
  if (!continuation) return direction;
  const planned = direction || {};
  // WIP identity is controller-owned: a Planner may not rename the lane or rebase an authoring WIP (silent frozen_baseline fallback).
  const ask = String(planned.prompt || '');
  const other = String(planned.candidate_id || '');
  const keepAsk = ask && !(other && other !== continuation.candidate_id && ask.includes(other)) &&
    !/\b(?:restart|recreate|re-create|rebase|from scratch)\b/i.test(ask);
  return {
    ...planned,
    ...continuation,
    id: planned.id || continuation.id,
    title: planned.title || continuation.title,
    specialty: planned.specialty || continuation.specialty,
    ...(planned.roadmap_rung ? { roadmap_rung: planned.roadmap_rung } : {}),
    ...(planned.step_role ? { step_role: planned.step_role } : {}),
    ...(Array.isArray(planned.gated_on) ? { gated_on: planned.gated_on } : {}),
    ...(Array.isArray(planned.focus_files) ? { focus_files: planned.focus_files } : {}),
    ...(planned.target_shape ? { target_shape: planned.target_shape } : {}),
    ...(planned.rewind_head ? { rewind_head: planned.rewind_head,
      rewind_reason: planned.rewind_reason } : {}),
    // Keep the Planner's task (unless it renames the lane/base) and, for the same lane, its topology.
    prompt: keepAsk ? `${ask}\n\nLANE STATE (${continuation.candidate_id}): ${continuation.prompt}`
      : continuation.prompt,
    target_topology: other === continuation.candidate_id && planned.target_topology ||
      continuation.target_topology,
  };
}

async function planMegaCandidateTurn(currentRound, remaining, pool) {
  const searchHistory = megaHistoryForSearch(history, megaCandidateRegistry);
  const plan = await agentT(
    roleAgent('mega_search_lead', 'plan_round',
      'Choose one whole-kernel candidate direction. Expert knowledge guides the common candidate lifecycle; it never creates a reserved lane.', {
        ...(pool ? { GPU_POOL: pool, GPU_MIN_FREE_GIB } : {}),
        EVAL_DIR, ROUND: currentRound, BUDGET_REMAINING: remaining,
        CUMULATIVE_SPEEDUP: cumulative, BASELINE_GEOMEAN_MS,
        SKILL_DIR: WORKFLOW_DIR, PROFILE_SUMMARY: profileSummary, ...GMI, ...MRI,
        CURRENT_BEST_PER_CASE: bestPerCase, HISTORY: searchHistory,
        MEGA_CANDIDATE_REGISTRY: megaRegistryForSearch(megaCandidateRegistry),
        MEASUREMENT_CALIBRATION: megaMeasurementCalibration,
        MEGA_STALL: { rounds: megaStall, limit: MEGA_STALL_LIMIT },
        ...(megaRefusal ? { PREVIOUS_PLAN_REFUSED: megaRefusal.reason } : {}),
        STRUCTURAL_CONTRACT_POLICY: STRUCTURAL_CONTRACT_BLOCKING ? 'blocking' : 'advisory',
        MEGA_PROFILE, CANDIDATE_TIMEOUT_S: MEGA_CANDIDATE_TIMEOUT_S,
        ...(MEGA_STRUCTURAL_ONLY ? {
          STRUCTURAL_ONLY: '1',
          STRUCTURAL_TARGET: 'complete the matched Expert Skill contract and MegaPlanIR target',
        } : {}),
        DEFAULT_BASE_CANDIDATE: (selectMegaSearchParent(megaCandidateRegistry) || {}).id ||
          'frozen_baseline',
        ROADMAP: `${EVAL_DIR}/roadmap.md`, ROADMAP_LADDER: LADDER,
        LADDER_DISPATCHED: [...dispatchedRungs], LADDER_COMPLETED: [...LADDER_MEASURED],
        OPEN_RUNGS: openRungs(LADDER, rungTally),
        ...(analysis && analysis.task_graph
          ? { TASK_GRAPH: analysis.task_graph } : {}),
        ...(analysis && analysis.resource_timeline
          ? { RESOURCE_TIMELINE: analysis.resource_timeline } : {}),
        ...(analysis && analysis.mega_plan_ir
          ? { MEGA_PLAN_IR: analysis.mega_plan_ir } : {}),
        BASELINE_OPERATOR_MAP: analysis && analysis.baseline_operator_map || [],
        ...(USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE ? {
          EXPERT_SKILL_PLANNER_EXTENSION: EXPERT_SKILL_PLANNER_GUIDE_FILE,
          EXPERT_SKILL_ID,
          EXPERT_SKILL_REVISION,
          EXPERT_SKILL_BUNDLE_TOOL,
          EXPERT_SKILL_BUNDLE_SHA256,
          EXPERT_SKILL_PLANNER_EXTENSION_SHA256,
          EXPERT_SKILL_CONTRACT_SHA256,
        } : {}),
        ...(CHAIN_DEBT.length ? {
          CHAIN_DEBT: chainDebtReport(CHAIN_DEBT, currentRound, LADDER_MEASURED).open,
          CHAIN_BASELINE,
        } : {}),
        TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC, LAUNCH_TARGET,
        REQUIRE_OVERLAP, REQUIRE_ATTRIBUTION, REQUIRE_ARTIFACT_DISTINCT,
        REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
        KERNEL_KNOWLEDGE_DIR: '', KK_OPERATOR: '', KK_LANGUAGE: '', KK_REFS: [], ...KB_INPUTS,
      }),
    { phase: 'Optimize', label: `mega:plan r${currentRound}`, schema: MEGA_PLAN_SCHEMA,
      // Late-round plans read a large registry and ran 6-7 min; a 5 min guard nulled them.
      ...(MEGA_PRODUCTION ? { timeout_ms: 900000, max_retries: 1 } : {}) });
  const noPlan = !plan || plan.stop || !Array.isArray(plan.directions) || !plan.directions.length;
  const wipDir = noPlan ? megaWipContinuationDirection(currentRound) : null;
  // A planner that returned nothing never decided to stop: optimize the best scored lane.
  const best = noPlan && !wipDir && !plan ? selectMegaSearchParent(megaCandidateRegistry) : null;
  if (noPlan && !best) return wipDir;
  let raw = best ? { id: `r${currentRound}_child_${best.id}`, candidate_id: best.id,
    candidate_source: best.source, base_candidate_id: best.id, title: `optimize ${best.id}`,
    specialty: MEGA_DEFAULT_SPECIALTY, step_role: 'terminal', focus_files: [],
    roadmap_rung: 'off_ladder', target_topology: best.topology,
    prompt: best.next_blocker || 'Continue the first unresolved measured blocker.' }
    : plan.directions[0];
  const freshLane = !MEGA_RESUME_STATE && megaCandidateRegistry.length === 0;
  if (freshLane && freshMegaDirectionLeak(raw)) {
    const requestedId = String(raw && (raw.candidate_id || raw.id) || '');
    const analyzed = LADDER.find((rung) =>
      String(rung && (rung.candidate_id || rung.id) || '') === requestedId
    ) || LADDER[0];
    if (!analyzed || freshMegaDirectionLeak(analyzed)) {
      log('Mega fresh-state plan rejected: Planner referenced prior/external candidate artifacts ' +
        'and no clean Analyze direction is available.');
      return null;
    }
    log('Mega fresh-state plan ignored a prior/external candidate reference and restored the ' +
      'clean Analyze direction from the frozen baseline.');
    raw = {
      ...analyzed,
      candidate_source: 'search',
      base_candidate_id: 'frozen_baseline',
    };
  }
  // Parking freezes (never deletes) the active lane so a structurally different lane can start.
  const parkId = String(raw.park_candidate_id || '');
  const parked = parkId && String(raw.park_reason || '').trim() && megaCandidateById(parkId);
  const parkedNow = !!parked && ['authoring', 'runnable'].includes(parked.status) &&
    String(raw.candidate_id || '') !== parkId;
  if (parkedNow) {
    megaCandidateRegistry = megaCandidateRegistry.map((c) => c.id !== parkId ? c
      : normalizeMegaCandidate({ ...c, status: 'parked',
        notes: `parked r${currentRound}: ${raw.park_reason}. ${c.notes || ''}` }));
    log(`Mega round ${currentRound}: Planner parked lane ${parkId} (${raw.park_reason}); resumable.`);
  }
  const plannerCandidateId = String(raw.candidate_id || '');
  const bound = parkedNow ? raw : bindMegaDirectionToWip(currentRound, raw);
  if (bound !== raw) {
    if (plannerCandidateId && plannerCandidateId !== bound.candidate_id) {
      log(`Mega round ${currentRound}: Planner requested replacement lane ${plannerCandidateId}; ` +
        `continuing recoverable WIP ${bound.candidate_id} in place instead.`);
    }
    raw = bound;
  }
  if (!validMegaSearchDirection(raw)) {
    log(`Mega search plan rejected diagnostic-only direction ` +
      `${String(raw.candidate_id || raw.id || raw.title || '(unnamed)')}; a search budget unit must ` +
      `author or optimize a selectable whole-kernel candidate.`);
    const continuation = megaWipContinuationDirection(currentRound);
    if (continuation) {
      log(`Mega round ${currentRound}: continuing resumable WIP lane ${continuation.candidate_id} ` +
        `inside this invocation instead of terminating or opening another workflow.`);
      raw = continuation;
    } else {
      raw = {
        id: `search_whole_r${currentRound}`,
        candidate_id: `search_whole_r${currentRound}`,
        candidate_source: 'search',
        base_candidate_id: 'frozen_baseline',
        title: 'autonomous runnable Mega-kernel candidate',
        specialty: MEGA_DEFAULT_SPECIALTY,
        step_role: 'terminal',
        focus_files: [],
        prompt: 'Author a complete runnable operator candidate from the frozen source/task graph. Full ' +
          'or partial fusion is valid if paired rank-max performance beats the frozen baseline. ' +
          'Diagnostics may be temporary inside this turn but cannot be its output.',
      };
    }
  }
  const requestedId = String(raw.candidate_id || raw.id || `search_r${currentRound}`);
  const safeRequestedId = validMegaCandidateId(requestedId)
    ? requestedId : `search_r${currentRound}`;
  const requested = megaCandidateById(safeRequestedId);
  const candidateId = requested && (requested.status === 'scored' || requested.status === 'finalist')
    ? megaChildCandidateId(safeRequestedId, currentRound) : safeRequestedId;
  const requestedBaseId = String(raw.base_candidate_id ||
    (candidateId !== safeRequestedId ? safeRequestedId : '') ||
    ((selectMegaSearchParent(megaCandidateRegistry) || {}).id || 'frozen_baseline'));
  const requestedBase = megaCandidateById(requestedBaseId);
  const baseId = requestedBaseId === 'frozen_baseline' ||
    (requestedBase &&
      (requestedBase.source === 'search' || requestedBase.source === 'integrated') &&
      (requestedBase.status === 'scored' || requestedBase.status === 'finalist'))
    ? requestedBaseId : 'frozen_baseline';
  const targetShape = raw.target_shape || {};
  const planIr = analysis && analysis.mega_plan_ir || {};
  const targetTopology = raw.target_topology || {
    launch_count: Number.isFinite(Number(targetShape.launches))
      ? Number(targetShape.launches)
      : Number(planIr.target && planIr.target.launch_count || LAUNCH_TARGET),
    included_regions: Array.isArray(targetShape.stages_fused)
      ? targetShape.stages_fused.map(String)
      : (Array.isArray(planIr.regions) ? planIr.regions.map((r) => String(r.id || '')).filter(Boolean) : []),
    included_queues: Array.isArray(planIr.queues)
      ? planIr.queues.map((queue) => String(queue.id || '')).filter(Boolean) : [],
    capabilities: [],
    require_overlap: targetShape.require_overlap === true || REQUIRE_OVERLAP,
  };
  return {
    ...enrichDirection(raw, LADDER),
    id: raw.id || candidateId,
    candidate_id: candidateId,
    candidate_source: raw.candidate_source === 'integrated' ? 'integrated' : 'search',
    base_candidate_id: baseId,
    tree: megaLaneTree(candidateId),
    // The topology descriptor is the lowerable whole-kernel contract shared by Author and Verify.
    target_topology: targetTopology,
    topology_inferred: !raw.target_topology,
  };
}

async function runMegaCandidateTurn(currentRound, remaining) {
  const turnStartedMs = megaNowMs();
  const dispatchDeadlineMs = WORKFLOW_STARTED_MS +
    (MEGA_TIME_BUDGET_S - MEGA_FINAL_RESERVE_S) * 1000;
  if (!megaMeasurementCalibration.ready &&
      (currentRound === 2 || (currentRound > 2 && currentRound % 3 === 1))) {
    await recoverMegaCalibration(`before-round-${currentRound}`);
  }
  const pool = (MEGA_STRUCTURAL_ONLY || MEGA_ROUTE_ONLY)
    ? null : await samplePool(currentRound, MEGA_PRODUCTION ? 120000 : 0);
  const d = await planMegaCandidateTurn(currentRound, remaining, pool);
  if (!d) return { stop: true, reason: 'no mega candidate direction was planned' };
  if (analysis && analysis.resource_timeline) {
    const pg = pipeOccupancyGate(analysis.resource_timeline, [d]);
    if (pg.summary) log(`Mega round ${currentRound}: ${pg.summary}`);
    if (pg.caveat) log(`Mega round ${currentRound}: ${pg.caveat}`);
  }
  const lg = roadmapLadderGate(LADDER, [d], LADDER_MEASURED);
  log(`Mega round ${currentRound}: ${lg.summary}`);
  if (lg.caveat) log(`Mega round ${currentRound}: ${lg.caveat}`);
  const topologyVerdict = megaTopologyVerdict(
    d, analysis && analysis.mega_plan_ir, ALLOW_PARTIAL_FUSION);
  if (!topologyVerdict.pass) {
    log(`Mega round ${currentRound}: TOPOLOGY CONTRACT: ${topologyVerdict.reason}`);
    return {
      refused: true, stop: MEGA_ROUTE_ONLY,
      reason: `Mega direction refused before authoring: ${topologyVerdict.reason}`,
    };
  }
  if (MEGA_ROUTE_ONLY) {
    return {
      stop: true,
      direction: d,
      reason: `route-only PASS: candidate=${d.candidate_id}, rung=${
        d.roadmap_rung || d.id}, launch_count=${topologyLaunchCount(d.target_topology)}, ` +
        `target=${Number(analysis && analysis.mega_plan_ir &&
          analysis.mega_plan_ir.target &&
          analysis.mega_plan_ir.target.launch_count)}; no candidate source or GPU was touched`,
    };
  }
  if (STRICT_AUTONOMY) {
    const strict = strictDirectionVerdict(d, LADDER, LADDER_MEASURED);
    if (!strict.pass) {
      return { refused: true, stop: MEGA_ROUTE_ONLY, reason:
        `strict Mega direction refused before authoring: ${strict.reason}` };
    }
  }
  for (const rung of lg.planned) {
    dispatchedRungs.add(rung);
    const tally = rungTally.get(rung) || { attempts: 0, last_outcome: 'never_planned' };
    tally.attempts += 1;
    if (tally.last_outcome !== 'measured') tally.last_outcome = 'unmeasured';
    rungTally.set(rung, tally);
  }
  const candidateId = d.candidate_id;
  const source = d.candidate_source;
  if (!['search', 'integrated'].includes(source)) {
    return { stop: true, reason: `invalid Mega candidate source '${String(source)}'; Expert Skills ` +
      'must use the common search/integrated lifecycle' };
  }
  const existing = megaCandidateById(candidateId);
  if (existing && existing.status === 'parked') {
    megaCandidateRegistry = megaCandidateRegistry.map((c) => c.id !== candidateId ? c
      : normalizeMegaCandidate({ ...c, status: 'authoring' }));
    log(`Mega round ${currentRound}: resuming parked lane ${candidateId}.`);
  }
  megaAdvanceMs(MEGA_PREP_MODEL_MS);
  const dispatchRemainingS = (dispatchDeadlineMs - megaNowMs()) / 1000;
  if (MEGA_PRODUCTION && dispatchRemainingS < 900) {
    return {
      stop: true,
      reason: `only ${Math.max(0, Math.round(dispatchRemainingS))}s remain before the production ` +
        `final-validation reserve; no new candidate attempt was started`,
    };
  }
  const requestedTurnS = MEGA_CANDIDATE_TIMEOUT_S;
  const turnDeadlineMs = MEGA_PRODUCTION
    ? Math.min(turnStartedMs + requestedTurnS * 1000, dispatchDeadlineMs)
    : turnStartedMs + requestedTurnS * 1000;
  const turnBudgetS = (turnDeadlineMs - turnStartedMs) / 1000;
  const prepElapsedS = (megaNowMs() - turnStartedMs) / 1000;
  const availableAfterPrepS = (turnDeadlineMs - megaNowMs()) / 1000;
  if (MEGA_PRODUCTION && availableAfterPrepS < 900) {
    return {
      stop: true,
      reason: `candidate preparation consumed ${Math.round(prepElapsedS)}s; only ` +
        `${Math.max(0, Math.round(availableAfterPrepS))}s remain in the shared turn budget, so no ` +
        `Engineer was started`,
    };
  }
  const gpuCandidateDevelopment = !MEGA_STRUCTURAL_ONLY &&
    EXPERT_SKILL_USAGE === 'candidate_validation';
  const engineerBudgetS = MEGA_PRODUCTION
    ? Math.max(240, Math.min(
      Math.floor(turnBudgetS * (gpuCandidateDevelopment ? 0.85 : 0.60)),
      Math.floor(availableAfterPrepS - Math.max(360, 60 + MEGA_POST_AUTHOR_RESERVE_S))))
    : MEGA_CANDIDATE_TIMEOUT_S;
  const commandBudgetS = Math.max(180, engineerBudgetS - 60);

  const baseCandidateId = (existing && existing.base_id) || d.base_candidate_id || 'frozen_baseline';
  const attempts = (existing ? existing.attempts : 0) + 1;
  const attemptId = `${candidateId}:r${currentRound}:a${attempts}`;
  const tree = d.tree || megaLaneTree(candidateId);
  const laneManifest = tree.replace(/\/tree$/, '') + '/lane.json';
  const laneLock = tree.replace(/\/tree$/, '') + '/author.lock';
  const baseTree = megaBaseTree(baseCandidateId, source);
  const baseHead = megaBaseHead(baseCandidateId, source);
  const outDir = `${EVAL_DIR}/round_${currentRound}/candidate_${candidateId}`;
  const persistentJitCache = STATE_DIR
    ? `${STATE_DIR}/jit_cache/${candidateId}` : `${outDir}/jit_cache`;
  const effectivePrior = existing ? megaEffectiveLaneCheckpoint(existing) : null;
  const priorForAgent = existing ? {
    ...existing,
    ...effectivePrior,
    ...(effectivePrior.head !== existing.head ? {
      verification_status: '', correctness: 'pending', gpu_executed: false,
      activation_on_hardware: 'unknown', runtime_verified: false,
      score_complete: false, absolute_score: null,
    } : {}),
  } : null;
  const candidateClaimsSkillTarget =
    candidateClaimsPlanTarget(d, analysis && analysis.mega_plan_ir);
  const authorPreflightRequired = authoringContractNeedsPreflight(
    d, analysis && analysis.mega_plan_ir, existing, STRUCTURAL_CONTRACT_BLOCKING, {
      skillId: EXPERT_SKILL_ID,
      revision: EXPERT_SKILL_REVISION,
      bundle: EXPERT_SKILL_BUNDLE_SHA256,
      planner: EXPERT_SKILL_PLANNER_EXTENSION_SHA256,
      contract: EXPERT_SKILL_CONTRACT_SHA256,
    }
  );
  const stagedGpuAuthoring = authorPreflightRequired && gpuCandidateDevelopment;
  if (authorPreflightRequired) {
    log(`Mega round ${currentRound}: full-target candidate ${candidateId} has no current exact-HEAD ` +
      `structural pass; ${stagedGpuAuthoring
        ? 'staged GPU construction/JIT smoke is allowed, but promotion remains blocked'
        : 'authoring is GPU-forbidden until independent contract verification'}.`);
  }
  let eng = null;
  let laneWriterTimedOut = false;
  const authorGpuProhibited = MEGA_STRUCTURAL_ONLY ||
    (authorPreflightRequired && !stagedGpuAuthoring);
  const role = 'engineer';
  const roleFile = 'engineer.md';
    eng = await agentT(
      roleAgent(role, 'optimize',
        `Advance candidate lane ${candidateId}; never edit or replace another candidate lane.`, {
          CANDIDATE_ID: candidateId, CANDIDATE_SOURCE: source,
          BASE_CANDIDATE_ID: baseCandidateId, BASE_TREE: baseTree, BASE_HEAD: baseHead,
          CANDIDATE_TREE: tree, OUTPUT_DIR: outDir, ATTEMPT_ID: attemptId,
          CANDIDATE_PYTHONPATH: tree,
          CANDIDATE_IMPORT_MODULES,
          LANE_MANIFEST: laneManifest, LANE_LOCK: laneLock,
          CANDIDATE_TIMEOUT_S: commandBudgetS,
          PERSISTENT_JIT_CACHE_DIR: persistentJitCache,
          SPECIALTY: d.specialty || MEGA_DEFAULT_SPECIALTY, DIRECTION: d,
          KERNEL_PATH: tree, OP_SPEC, TASK_DIR: KERNEL_PATH_ORIG, COMMANDMENT,
          GPU_ID: authorGpuProhibited
            ? (MEGA_STRUCTURAL_ONLY
              ? 'STRUCTURAL_ONLY_NO_GPU' : 'STRUCTURAL_PREFLIGHT_REQUIRED_NO_GPU')
            : GPU_SINGLE,
          GPUS_PER_JOB: authorGpuProhibited
            ? '0' : String(GPU_RESOURCE.gpusPerJob),
          AUTHOR_GPU_MODE: authorGpuProhibited ? 'forbidden' : 'bounded_smoke',
          ...(MEASURE_FIRST_TEXT(megaStall) ? { MEASURE_FIRST: String(megaStall) } : {}),
          ...(/^[0-9a-f]{7,40}$/.test(String(d.rewind_head || ''))
            ? { REWIND_TO: d.rewind_head, REWIND_REASON: String(d.rewind_reason || '') } : {}),
          AUTHORING_CONTRACT_PREFLIGHT_REQUIRED:
            authorPreflightRequired ? '1' : '0',
          STAGED_GPU_AUTHORING: stagedGpuAuthoring ? '1' : '0',
          AUTHORING_STRUCTURAL_EVIDENCE_HEAD:
            effectivePrior && effectivePrior.structural_candidate_head || '',
          FROZEN_KERNEL_PATH: KERNEL_PATH_ORIG,
          TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC, LAUNCH_TARGET,
          ...(MEGA_STRUCTURAL_ONLY ? {
            STRUCTURAL_ONLY: '1',
            STRUCTURAL_TARGET: 'complete the Expert Skill target encoded by MegaPlanIR; no GPU commands',
          } : {}),
          SKILL_DIR: WORKFLOW_DIR,
          KERNEL_KNOWLEDGE_DIR: '',
          ...(USE_EXPERT_SKILLS ? {
            EXPERT_SKILL_ID,
            EXPERT_SKILL_REVISION,
            EXPERT_SKILL_RECIPE: EXPERT_SKILL_AUTHOR_GUIDE_FILE,
          } : {}),
          codebase_context: `${EVAL_DIR}/codebase_context.md`,
          profiling_summary: profileSummary ? profileSummary.summary_path : '',
          baseline_per_case: BASELINE_PER_CASE,
          UT_PATH: KERNEL_PATH_ORIG,
          BASELINE_OPERATOR_MAP: analysis && analysis.baseline_operator_map || [],
          INSIGHTS: megaHistoryForSearch(history, megaCandidateRegistry).insights,
          PRIOR_CANDIDATE: priorForAgent, ...GMI, ...MRI,
        }) +
      `\n\n${authorGpuProhibited
        ? `AUTHOR_GPU_MODE=forbidden (${MEGA_STRUCTURAL_ONLY ? 'STRUCTURAL-ONLY RUN'
          : 'AUTHORING CONTRACT PREFLIGHT IS REQUIRED'}): no rocm-smi, torchrun, benchmark, ` +
          `correctness command or GPU runtime import. Repair required failures in category order, ` +
          `commit a coherent checkpoint, and return claim_complete=true with ` +
          `candidate_status=authoring; runtime_verified, gpu_executed, and score_complete remain ` +
          `false separately. `
        : `AUTHOR_GPU_MODE=bounded_smoke: use the supplied EP8 device for bounded construction/JIT/` +
          `correctness/short-perf smokes of the HEAD you are editing, before AND after source edits, ` +
          `repeatedly in this turn. Author smokes are development evidence; independent Verify owns ` +
          `scoring. Revert diagnostic-only probes before committing. ${MEASURE_FIRST_TEXT(megaStall)}`}` +
      (/^[0-9a-f]{7,40}$/.test(String(d.rewind_head || ''))
        ? `REWIND_TO=${d.rewind_head}: after taking the lane lock and before other edits, run ` +
          `\`git tag -f parked_r${currentRound} HEAD && git reset --hard ${d.rewind_head}\` in ` +
          `${tree}; the old HEAD stays recoverable through that tag. ` : '') +
      `Before any source edit, acquire the single-writer lane lock in your persistent shell: ` +
      `mkdir -p "$(dirname "${laneLock}")"; exec 9>"${laneLock}"; flock -n 9, and keep fd 9 open ` +
      `through the final commit/manifest write, then release it with exec 9>&-. Start every background ` +
      `command or timer with 9>&- so no orphan inherits the lock. If the lock is held, return incomplete without editing. ` +
      `Candidate-lane setup is part of this task. If ${tree}/.git does not exist, create ${tree}. ` +
      (baseHead
        ? `Export exact parent commit ${baseHead} from ${baseTree} with git archive into the lane. `
        : `Copy ${baseTree} into it with a tar pipe excluding .git/build/__pycache__/.torch_ext/*.so/*.o. `) +
      `then initialize a fresh git repository and commit "candidate base ${baseCandidateId}". ` +
      `Before any long-running command, atomically write ${laneManifest} with candidate_id/source/` +
      `base_candidate_id/tree/attempt_id so a ` +
      `timed-out first attempt is discoverable on resume. ` +
      `If it exists, continue from its HEAD; never recreate it. Read ${WORKFLOW_DIR}/roles/mega/${roleFile}. ` +
      `Before every candidate Python or torchrun command, export ` +
      `PYTHONPATH="${tree}\${PYTHONPATH:+:\$PYTHONPATH}" and fail closed unless both ` +
      `every module in ${JSON.stringify(CANDIDATE_IMPORT_MODULES)} resolves under ${tree}. ` +
      `Commit WIP to THIS lane after every real advance. Write ${outDir}/candidate_result.json ` +
      `atomically only after its evidence files are final. Return candidate_id/source/base, tree, head, ` +
      `candidate_status, claim_complete, attempt_id, evidence_manifest, patch_file (cumulative from the ` +
      `lane root), correctness, absolute_score, per_case, topology_sig, next_blocker and notes. ` +
      `This Engineer gets ` +
      `${engineerBudgetS}s and every GPU command is bounded by ${commandBudgetS}s. A timeout or ` +
      `force-emit returns claim_complete:false and status authoring; it is not score 0.`,
      { phase: 'Optimize', label: `mega:${source}:${candidateId}`, schema: MEGA_CANDIDATE_SCHEMA,
        timeout_ms: engineerBudgetS * 1000, timeout_marker: true, max_retries: 1 });

    if (eng && eng.__agent_timed_out) {
    megaUnsafeTimeout = eng;
    laneWriterTimedOut = true;
    log(`Mega round ${currentRound}: candidate author ${eng.label || candidateId} exceeded its ` +
      `deadline. Stopping this workflow invocation because the underlying agent cannot be ` +
      `cancelled and may still hold ${laneLock}.`);
      eng = null;
    }
    if (!eng || eng.claim_complete !== true) {
      // A timed-out writer usually exits within minutes; wait for it instead of ending the wave.
      const drainS = laneWriterTimedOut ? 1500 : 0;
      const laneDir = tree.replace(/\/tree$/, '');
      const recovered = await agentT(
        roleAgent('engineer', 'recover',
          `RECOVER ONLY candidate ${candidateId}. First confirm the timed-out writer is quiescent and ` +
          `the lane lock is free. ` + (drainS ? `If not, wait up to ${drainS}s total, in order: ` +
          `\`flock -w <left> ${laneLock} true\`; the same on every ` +
          `\${KERNEL_EXTRACT_LOCK_DIR:-/tmp/team_gpu_locks}/gpu_*.lock; then poll every 30s until no ` +
          `non-zombie process (ps state not Z) has a cmdline or cwd under ${laneDir} or ${outDir}, ` +
          `excluding your own shell. Zombies and stale lease_*.json files are not holders; never ` +
          `use kill -0. Return writer_quiescent=true only if all three held. ` : '') +
          `Read ${outDir}/candidate_result.json and completed ` +
          `measurement aggregates. Accept only a manifest with claim_complete:true and attempt_id. ` +
          `Do not use a partial/older worker result, do not run a GPU command, and do not edit ${tree}. ` +
          `If the writer is still active or the lock is held, return claim_complete:false.`, {
            OUT_DIR: outDir, OUTPUT_DIR: outDir, CANDIDATE_ID: candidateId,
            CANDIDATE_SOURCE: source, BASE_CANDIDATE_ID: baseCandidateId,
            CANDIDATE_TREE: tree, KERNEL_PATH: tree, ATTEMPT_ID: attemptId,
            SKILL_DIR: WORKFLOW_DIR, COMMANDMENT,
          }),
        { phase: 'Optimize', label: `mega:recover:${candidateId}`, schema: MEGA_CANDIDATE_SCHEMA,
          ...(MEGA_PRODUCTION ? { timeout_ms: (180 + drainS) * 1000, max_retries: 1 } : {}) });
    megaAdvanceMs(drainS * 1000);
    // Before the claim branch, else megaUnsafeTimeout survives and skips the finalist.
    if (laneWriterTimedOut && recovered && !recovered.__agent_timed_out &&
        recovered.writer_quiescent === true) {
      laneWriterTimedOut = false;
      megaUnsafeTimeout = null;
      log(`Mega round ${currentRound}: timed-out author of ${candidateId} is quiescent; its ` +
        `committed WIP stays in the lane and the wave continues.`);
    }
    if (recovered && recovered.claim_complete === true) {
      eng = recovered;
      laneWriterTimedOut = false;
    }
  }
  if (!MEGA_RESUME_STATE && !existing && freshMegaAuthorLeak(eng)) {
    return {
      stop: true,
      reason: `fresh candidate Author disclosed reuse of a prior candidate/plan; rejecting ${
        candidateId} before structural Verify`,
    };
  }
  megaChargeMs(engineerBudgetS * 1000);

  const reportedChangedFiles = Array.isArray(eng && eng.changed_files)
    ? eng.changed_files.map(String) : [];
  const preVerifyStatus = 'authoring';
  const candidateActivation = normalizeMegaActivation(
    (eng && eng.activation) || (existing && existing.activation)
  );
  let meta = normalizeMegaCandidate({
    id: candidateId, source, base_id: baseCandidateId, tree,
    status: preVerifyStatus,
    head: (eng && eng.head) || (eng && eng.candidate_head) ||
      (existing && existing.working_head) || (existing && existing.head) || '',
    patch: (eng && eng.patch_file) || (existing && existing.patch) || '',
    topology_sig: (eng && eng.topology_sig) || (existing && existing.topology_sig) || '',
    topology: (eng && eng.topology) || d.target_topology ||
      (existing && existing.topology) || {},
    activation: candidateActivation,
    provenance: (eng && eng.provenance) || (existing && existing.provenance) || source,
    claim_complete: false,
    checkpoint_complete: !!(eng && eng.claim_complete === true),
    structural_verified: false,
    runtime_verified: false,
    score_complete: false,
    structural_report: String(eng && eng.structural_report || ''),
    contract_failures: normalizeContractFailures(eng && eng.contract_failures),
    attempt_id: (eng && eng.attempt_id) || attemptId,
    evidence_manifest: String(eng && eng.evidence_manifest || ''),
    attempts,
    changed_files: reportedChangedFiles,
    next_blocker: (eng && eng.next_blocker) || '',
    notes: (eng && eng.notes) || '',
  });
  megaCandidateRegistry = upsertMegaCandidate(megaCandidateRegistry, meta);
  if (laneWriterTimedOut) {
    await persistMegaCandidateState(currentRound);
    return {
      stop: true,
      reason: `candidate author timeout for ${candidateId}; invocation stopped to preserve single-writer ` +
        `lineage safety`,
    };
  }

  let ver = null;
  let verifierTimedOut = false;
  const turnRemainingS = turnBudgetS - (megaNowMs() - turnStartedMs) / 1000;
  let verifyBudgetS = MEGA_PRODUCTION
    ? Math.max(0, Math.min(Math.floor(turnRemainingS - 60),
      Math.floor((dispatchDeadlineMs - megaNowMs()) / 1000 - 60)))
    : MEGA_CANDIDATE_TIMEOUT_S;
  const expectedHead = String(eng && (eng.head || eng.candidate_head) || '');
  let structural = null;
  let structuralPassThisTurn = false;
  const shouldStructuralVerify = CHECK_EXPERT_SKILL_CONTRACT &&
    eng && eng.claim_complete === true && expectedHead;
  if (shouldStructuralVerify) {
    structural = await agentT(
      roleAgent('verify_engineer', 'verify_structure',
        'GPU-FREE independent post-authoring structure verification. Seal the candidate before ' +
        'reading an optional reference; never edit candidate source and never run a GPU command.', {
          CANDIDATE_ID: candidateId, CANDIDATE_SOURCE: source,
          CANDIDATE_TREE: tree, EXPECTED_HEAD: expectedHead,
          CANDIDATE_IMPORT_MODULES,
          FROZEN_KERNEL_PATH: KERNEL_PATH_ORIG,
          EXPERT_SKILL_ID,
          EXPERT_SKILL_REVISION,
          EXPERT_SKILL_BUNDLE_TOOL,
          EXPERT_SKILL_BUNDLE_SHA256,
          EXPERT_SKILL_PLANNER_EXTENSION: EXPERT_SKILL_PLANNER_EXTENSION_FILE,
          EXPERT_SKILL_PLANNER_EXTENSION_SHA256,
          EXPERT_SKILL_PLAYBOOK: EXPERT_SKILL_PLAYBOOK_FILE,
          EXPERT_SKILL_CONTRACT: EXPERT_SKILL_CONTRACT_FILE,
          EXPERT_SKILL_CONTRACT_SHA256,
          EXPERT_SKILL_VALIDATION: EXPERT_SKILL_VALIDATION_FILE,
          EXPERT_SKILL_REFERENCE_PATH,
          EXPERT_SKILL_CONTRACT_TOOL,
          STRUCTURAL_VERIFY_DIR: `${outDir}/structure`,
          LANE_LOCK: laneLock,
          REFERENCE_WAS_HIDDEN: EXPERT_SKILL_REFERENCE_PATH ? '1' : 'n/a',
          MEGA_PLAN_IR: analysis && analysis.mega_plan_ir || {},
          SKILL_DIR: WORKFLOW_DIR,
        }),
      { phase: 'Verify', label: `mega:structure:${candidateId}`,
        schema: EXPERT_SKILL_CONTRACT_VERIFY_SCHEMA,
        timeout_ms: 600000, max_retries: 1 });
    const structuralIdentity = structuralEvidenceIdentityVerdict(structural, {
      candidateId,
      candidateHead: expectedHead,
      skillId: EXPERT_SKILL_ID,
      skillRevision: EXPERT_SKILL_REVISION,
      contractSha256: EXPERT_SKILL_CONTRACT_SHA256,
      plannerExtensionSha256: EXPERT_SKILL_PLANNER_EXTENSION_SHA256,
      skillBundleSha256: EXPERT_SKILL_BUNDLE_SHA256,
    });
    const requiredIds = (structural && structural.failed_required_checks || []).map(String);
    const planErrors = (structural && structural.plan_consistency_errors || []).map(String);
    const inputErrors = (structural && structural.input_errors || []).map(String);
    const forbiddenTreeReuse = !MEGA_RESUME_STATE && FORBIDDEN_CANDIDATE_TREE_DIGESTS.has(
      String(structural && structural.candidate_tree_digest || '')
    );
    const authoritativeIds = new Set([
      ...requiredIds, ...planErrors.map((error) => error.split(':', 1)[0]),
    ]);
    const structuralFailures = normalizeContractFailures(
      structural && structural.contract_failures
    ).filter((failure) => authoritativeIds.has(failure.id));
    const structuralPass = !!(structural && structural.claim_complete === true &&
      structural.structural_compatible === true &&
      structural.independent_structure_pass === true &&
      structural.capability_eligible === true &&
      structural.plan_consistent === true &&
      structural.provenance_attestation_valid === true &&
      requiredIds.length === 0 && planErrors.length === 0 && inputErrors.length === 0 &&
      !forbiddenTreeReuse &&
      structural.reference_copy_detected !== true &&
      structural.reference_copy_suspected !== true &&
      structuralIdentity.pass);
    structuralPassThisTurn = structuralPass;
    if (!structuralIdentity.pass) {
      structuralFailures.unshift({
        id: 'structural_evidence_identity',
        category: 'plan',
        severity: 'required',
        messages: structuralIdentity.reasons,
      });
    }
    if (forbiddenTreeReuse) {
      structuralFailures.unshift({
        id: 'fresh_candidate_tree_reuse',
        category: 'plan',
        severity: 'required',
        messages: ['candidate tree digest matches a forbidden prior process artifact'],
      });
    }
    meta = normalizeMegaCandidate({
      ...meta,
      structural_verified: structuralPass,
      structural_report: structural && structural.report_path || '',
      structural_skill_id: structuralPass
        ? String(structural && structural.skill_id || '') : '',
      structural_candidate_head: structuralPass
        ? String(structural && structural.candidate_head || '') : '',
      structural_candidate_tree_digest: structuralPass
        ? String(structural && structural.candidate_tree_digest || '') : '',
      structural_skill_bundle_sha256: structuralPass
        ? String(structural && structural.skill_bundle_sha256 || '') : '',
      structural_planner_extension_sha256: structuralPass
        ? String(structural && structural.planner_extension_sha256 || '') : '',
      structural_contract_revision: structuralPass
        ? String(structural && structural.contract_revision || EXPERT_SKILL_REVISION) : '',
      structural_contract_sha256: structuralPass
        ? String(structural && structural.contract_sha256 || '') : '',
      contract_failures: structuralPass
        ? [] : structuralFailures,
      next_blocker: structuralPass
        ? meta.next_blocker
        : (forbiddenTreeReuse
          ? 'fresh candidate tree exactly matches a forbidden prior process artifact'
          : !structuralIdentity.pass
          ? `structural evidence identity failed: ${structuralIdentity.reasons.join('; ')}`
        : (structural && structural.next_blocker) ||
          'independent static structure contract did not pass'),
    });
    megaCandidateRegistry = upsertMegaCandidate(megaCandidateRegistry, meta);
  }
  if (shouldStructuralVerify) megaChargeMs(600000);
  if (MEGA_PRODUCTION) {
    verifyBudgetS = Math.max(0, Math.min(
      verifyBudgetS,
      Math.floor((turnDeadlineMs - megaNowMs()) / 1000 - 60),
      Math.floor((dispatchDeadlineMs - megaNowMs()) / 1000 - 60),
    ));
  }
  const contractBlocksRuntime = STRUCTURAL_CONTRACT_BLOCKING;
  const priorLaneHead = existing
    ? String(existing.working_head || existing.head || '') : '';
  const sourceAdvancedThisTurn = existing
    ? expectedHead !== priorLaneHead : reportedChangedFiles.length > 0;
  const structurallyReadyAuthoring = !!(
    eng && String(eng.candidate_status || '') === 'authoring' &&
    eng.claim_complete === true && eng.build !== false &&
    meta.checkpoint_complete &&
    (!STRUCTURAL_CONTRACT_BLOCKING || (
      structuralPassThisTurn &&
      meta.structural_candidate_head === expectedHead &&
      !!meta.structural_candidate_tree_digest
    ))
  );
  const postAuthoringVerify = structurallyReadyAuthoring &&
    ((sourceAdvancedThisTurn && candidateClaimsSkillTarget) ||
      (!authorPreflightRequired && !sourceAdvancedThisTurn));
  const gpuWaitBudgetS = Math.max(30, Math.min(300, Math.floor(verifyBudgetS / 4)));
  const gpuRunBudgetS = Math.max(60, verifyBudgetS - gpuWaitBudgetS - 60);
  const shouldVerify = eng && eng.claim_complete === true && expectedHead &&
    (!contractBlocksRuntime || meta.structural_verified) &&
    !MEGA_STRUCTURAL_ONLY &&
    (!MEGA_PRODUCTION || verifyBudgetS >= 300) &&
    (postAuthoringVerify ||
      ['runnable', 'scored', 'finalist'].includes(String(eng.candidate_status || '')));
  if (shouldVerify) {
    const verifyInputs = {
          CANDIDATE_ID: candidateId, CANDIDATE_SOURCE: source,
          CANDIDATE_TREE: tree, EXPECTED_HEAD: expectedHead,
      PERSISTENT_JIT_CACHE_DIR: persistentJitCache,
      CANDIDATE_IMPORT_MODULES,
      EXPECTED_STRUCTURAL_TREE_DIGEST: meta.structural_candidate_tree_digest,
      EXPERT_SKILL_CONTRACT_TOOL,
      EXPERT_SKILL_CONTRACT: EXPERT_SKILL_CONTRACT_FILE,
      LANE_LOCK: laneLock,
      GPU_WAIT_TIMEOUT_S: gpuWaitBudgetS,
      GPU_RUN_TIMEOUT_S: gpuRunBudgetS,
          BASE_CANDIDATE_ID: baseCandidateId,
          CANONICAL: baseTree, PATCH: (eng && eng.patch_file) || '',
          VERIFY_DIR: `${outDir}/verify`, ATTEMPT_ID: attemptId,
          VERIFY_TIMEOUT_S: verifyBudgetS,
          GPU_ID: GPU_SINGLE, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT,
          BASELINE_PER_CASE, FROZEN_KERNEL_PATH: KERNEL_PATH_ORIG,
      VERIFY_TIER: 'score',
      MODIFIABLE_FILES: 'WHOLE_CANDIDATE_TREE',
      SPECIALTY: d.specialty || MEGA_DEFAULT_SPECIALTY,
      TARGET_SHAPE: megaShapeFromTopology(d.target_topology),
          TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC,
          REQUIRE_ARTIFACT_DISTINCT: true, REQUIRE_OVERLAP: false,
          REQUIRE_ATTRIBUTION: false, REQUIRED_REPLAYS: 30,
      REQUIRE_GRAPH_CAPTURE: MEGA_REQUIRE_GRAPH_CAPTURE ? '1' : '0',
          REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD, LAUNCH_TARGET,
          ACCURACY_METRIC, ACCURACY_THRESHOLD,
      REQUIRED_ACCURACY_CASES,
      MEGA_PLAN_IR: analysis && analysis.mega_plan_ir || {},
      RESOURCE_TIMELINE: analysis && analysis.resource_timeline || {},
      REQUIRE_RESOURCE_VERIFY: '1',
      SEARCH_ACCEPTS_PARTIAL_FUSION: ALLOW_PARTIAL_FUSION ? '1' : '0',
      ...(USE_EXPERT_SKILLS ? {
        EXPERT_SKILL_ID,
        EXPERT_SKILL_REVISION,
        EXPERT_SKILL_FILE: `${EXPERT_SKILL_DIR}/skill.md`,
        EXPERT_SKILL_PLAYBOOK: EXPERT_SKILL_PLAYBOOK_FILE,
        EXPERT_SKILL_CONTRACT: EXPERT_SKILL_CONTRACT_FILE,
        EXPERT_SKILL_VALIDATION: EXPERT_SKILL_VALIDATION_FILE,
        EXPERT_SKILL_RUNTIME_FILE,
        EXPERT_SKILL_ACCURACY_CASES,
        EXPERT_SKILL_SOURCE_FILES,
      } : {}),
      GRAPH_CONTRACT_TOOL, ...RFI, ...GMI, ...MRI,
      GRAPH_CONTRACT_REPLAYS: 30,
      ACTIVATION: (eng && eng.activation) ? JSON.stringify(eng.activation) : 'UNDECLARED',
    };
    ver = await agentT(
      roleAgent('verify_engineer', 'verify',
        'Independently verify and score this whole-tree mega candidate against the frozen baseline.',
        verifyInputs),
      { phase: 'Verify', label: `mega:verify:${candidateId}`, schema: MEGA_VERIFY_SCHEMA,
        timeout_ms: verifyBudgetS * 1000, timeout_marker: true, max_retries: 1 });
    if (ver && ver.__agent_timed_out) {
      verifierTimedOut = true;
      log(`Mega round ${currentRound}: verify for ${ver.label || candidateId} exceeded its ` +
        `deadline. Checking process quiescence and on-disk evidence before one safe retry.`);
      ver = null;
    }
    if (!ver || ver.claim_complete !== true) {
      const recoveryBudgetS = MEGA_PRODUCTION
        ? Math.max(0, Math.floor(turnBudgetS - (megaNowMs() - turnStartedMs) / 1000 - 60))
        : 300;
      if (recoveryBudgetS >= 30) {
        const recoveredVer = await agentT(
          roleAgent('verify_engineer', 'recover',
            `RECOVER verification for candidate ${candidateId}, attempt ${attemptId}. Do not run a GPU ` +
            `command or edit source. If the prior Verify timed out, check twice at least 10s apart that ` +
            `no matching GPU/torchrun/lease process remains and the lane lock is free. Set ` +
            `timeout_recovery_safe=true only then; report active_gpu_processes and evidence_complete.`, {
              VERIFY_DIR: `${outDir}/verify`, CANDIDATE_ID: candidateId,
              CANDIDATE_SOURCE: source, CANDIDATE_TREE: tree,
              EXPECTED_HEAD: expectedHead, ATTEMPT_ID: attemptId,
              SKILL_DIR: WORKFLOW_DIR, COMMANDMENT,
            }),
          { phase: 'Verify', label: `mega:verify-recover:${candidateId}`, schema: MEGA_VERIFY_SCHEMA,
            timeout_ms: Math.min(120, recoveryBudgetS) * 1000, max_retries: 1 });
        if (recoveredVer && recoveredVer.claim_complete === true) {
          ver = recoveredVer;
          verifierTimedOut = false;
        } else if (verifierTimedOut && safeVerifyTimeoutRecovery(recoveredVer)) {
          const retryBudgetS = Math.max(300, Math.min(1800, MEGA_FINAL_RESERVE_S));
          const retry = await agentT(
            roleAgent('verify_engineer', 'verify',
              'TIMEOUT RECOVERY: verify the same exact HEAD read-only. Run only the earliest unresolved ' +
              'GPU stage (construction/JIT, then the smallest correctness smoke) and return its concrete ' +
              'result; do not run the full score suite unless the smoke is already cached.', {
                ...verifyInputs,
                VERIFY_DIR: `${outDir}/verify_timeout_retry`,
                VERIFY_TIER: 'timeout_recovery_smoke',
                GPU_WAIT_TIMEOUT_S: Math.min(120, gpuWaitBudgetS),
                GPU_RUN_TIMEOUT_S: Math.min(600, gpuRunBudgetS),
                VERIFY_TIMEOUT_S: retryBudgetS,
                REQUIRED_REPLAYS: 1,
                REQUIRED_PAIRS: 1,
                TIMEOUT_RECOVERY_ATTEMPT: '1',
              }),
            { phase: 'Verify', label: `mega:verify-timeout-retry:${candidateId}`,
              schema: MEGA_VERIFY_SCHEMA, timeout_ms: retryBudgetS * 1000,
              timeout_marker: true, max_retries: 1 });
          if (retry && retry.claim_complete === true) {
            ver = retry;
            verifierTimedOut = false;
          } else if (retry && !retry.__agent_timed_out) {
            // Quiescence was confirmed twice and the bounded retry returned: nothing is left running, so the
            // round ends unscored and the wave continues.
            verifierTimedOut = false;
            meta.next_blocker = `independent Verify of HEAD ${expectedHead} timed out before ` +
              `sealing (retry: ${retry.status || 'unscored'}); route a fresh full Verify of this ` +
              `HEAD before new edits`;
            megaCandidateRegistry = upsertMegaCandidate(megaCandidateRegistry, meta);
          } else {
            megaUnsafeTimeout = retry || { label: `mega:verify-timeout-retry:${candidateId}` };
          }
        } else if (verifierTimedOut) {
          megaUnsafeTimeout = { label: `mega:verify:${candidateId}` };
        }
      }
    }
  }
  if (shouldVerify) megaChargeMs(verifyBudgetS * 1000);

  let record = meta;
  if (ver) {
    record = megaCandidateFromVerification(meta, ver, {
      targetGuards: TARGET_GUARDS, regressionGuards: [],
      launchTarget: LAUNCH_TARGET, promotionMetric: PROMOTION_METRIC,
      accuracyMetric: ACCURACY_METRIC, accuracyThreshold: ACCURACY_THRESHOLD,
      requiredAccuracyCases: REQUIRED_ACCURACY_CASES,
      requiredReplays: 30, requiredPairs: REQUIRED_PAIRS,
      allowPartialFusion: ALLOW_PARTIAL_FUSION, frozenBaseline: BASELINE_PER_CASE,
    });
    record.attempts = attempts;
    record.tree = tree;
    if (record.status === 'scored' && record.absolute_score <= 1.0) {
      record.status = 'runnable';
      record.next_blocker = record.next_blocker ||
        `candidate is correct but ${record.absolute_score.toFixed(4)}x does not beat frozen ` +
        `baseline; keep optimizing this lane, never promote it to a finalist`;
    }
    megaCandidateRegistry = upsertMegaCandidate(megaCandidateRegistry, record);
  }
  if (verifierTimedOut) {
    await persistMegaCandidateState(currentRound);
    return {
      stop: true,
      reason: `candidate score verification timeout for ${candidateId}; invocation stopped before any ` +
        `other EP8 work was dispatched`,
    };
  }

  // A score may be recorded before calibration completes, but it cannot update the incumbent or be
  // published until a synthetic control proves the harness sees an effect.
  const selection = megaMeasurementCalibration.ready
    ? selectMegaCandidate(megaCandidateRegistry, MEGA_TIE_NOISE_PCT)
    : { selected: null, eligible: [], tie_kept_skill: false };
  const selected = selection.selected;
  const improved = !!(selected && selected.absolute_score > cumulative * (1 + MIN_IMPROVE));
  if (selected && (improved || !finalWinner)) {
    cumulative = Math.max(cumulative, selected.absolute_score);
    bestPerCase = selected.per_case.length ? selected.per_case : bestPerCase;
    finalWinner = {
      id: selected.id, source: selected.source, geomean: selected.absolute_score,
      per_case: selected.per_case, patch: selected.patch, tree: selected.tree,
      topology_sig: selected.topology_sig, provenance: selected.provenance,
      activation: selected.activation,
    };
  }

  const hasEvidence = !!(ver && ver.claim_complete === true);
  noEvidence = hasEvidence ? 0 : noEvidence + 1;
  noImprove = improved ? 0 : noImprove + 1;
  const reachedHardware = !!(ver &&
    String(ver.activation_on_hardware || '').toLowerCase() === 'yes');
  // Source-only checkpoints do not consume the hardware-failure budget; only a complete candidate that
  // reached Verify counts.
  const hardwareWasDue = !!shouldVerify;
  if (reachedHardware) noHardware = 0;
  else if (hardwareWasDue) noHardware += 1;
  const blockerKey = String(record.next_blocker || '').slice(0, 160);
  megaStall = hasEvidence || improved ||
    (eng && eng.gpu_executed === true && blockerKey !== megaLastBlocker) ? 0 : megaStall + 1;
  megaLastBlocker = blockerKey;
  // Analysis gets the identity-bound evidence of THIS candidate instead of scanning EVAL_DIR.
  if (ANALYSIS_SKILL_ON && ver) profileSummary = { ...profileSummary,
    analysis_result: await runProfileAnalysis({ ...profileSummary, candidate_evidence: {
      candidate_id: candidateId, head: record.head, attempt_id: record.attempt_id,
      target_guard: record.target_guard, frozen_per_case: BASELINE_PER_CASE,
      per_case: record.per_case, paired_readings: record.paired_readings,
      denominator_mismatch: record.denominator_mismatch,
      component_breakdown: record.component_breakdown, hot_path_counters: record.hot_path_counters,
      evidence_manifest: record.evidence_manifest } }, currentRound, 'mega:analyze') };
  history.rounds.push({
    round: currentRound,
    directions: [{ id: d.id, title: d.title, specialty: d.specialty,
      candidate_id: candidateId, candidate_source: source }],
    results: [{ id: d.id, status: ver ? ver.status : (eng ? eng.candidate_status : 'pending'),
      verified: record.absolute_score, candidate_status: record.status }],
    winner: selected ? { source: selected.id, geomean: selected.absolute_score } : null,
    improved, cumulative,
  });
  const verificationFaulted = !!(ver && ver.claim_complete === true &&
    (String(ver.correctness || '').toLowerCase().startsWith('fail') ||
      /fail|fault|crash|error/.test(String(ver.status || '').toLowerCase())));
  const rungOutcome = ver && ver.claim_complete === true &&
      Array.isArray(ver.per_case) && ver.per_case.length
    ? 'measured'
    : (verificationFaulted || eng && (eng.build === false ||
      String(eng.correctness || '').toLowerCase().startsWith('fail')))
      ? 'faulted' : 'unmeasured';
  recordRungOutcome(d.roadmap_rung, rungOutcome);
  if (rungOutcome === 'measured' && d.roadmap_rung) {
    LADDER_MEASURED.add(String(d.roadmap_rung));
  }
  const stateSaved = await persistMegaCandidateState(currentRound);
  if (!stateSaved) {
    return {
      stop: true,
      reason: `candidate state for round ${currentRound} could not be persisted; stopped before ` +
        `dispatching another lane`,
    };
  }
  return { stop: false, direction: d, record, ver, improved, selected };
}

let megaSelection = null;
let megaSelectionContractPassed = false;
const runFinalistBatch = async (batch, label) => {
  if (!batch.length || !megaMeasurementCalibration.ready) return null;
  const batchStartedMs = megaNowMs();
  const remainingS = MEGA_TIME_BUDGET_S -
    (megaNowMs() - WORKFLOW_STARTED_MS) / 1000 - MEGA_CLOSEOUT_RESERVE_S;
  if (remainingS < 600) {
    log(`MEGA finalist skipped: only ${Math.max(0, Math.round(remainingS))}s remain before the ` +
      `closeout reserve; candidate state is preserved for the next production invocation.`);
    return null;
  }
  const finalistTimeoutS = Math.min(MEGA_FINAL_TIMEOUT_S, remainingS);
  const batchDeadlineMs = batchStartedMs + finalistTimeoutS * 1000;
  // We are committed to running this finalist batch; charge its budget to the deterministic
  // clock so a subsequent fallback batch sees the closeout window shrink.
  megaAdvanceMs(finalistTimeoutS * 1000);
  const result = await agentT(
    roleAgent('director', 'select_mega',
      MEGA_PRODUCTION
        ? 'Validate this highest-ranked workflow-authored candidate; do not pay for lower-ranked fallbacks unless it fails.'
        : 'Re-measure the top workflow-authored Mega candidates together; select the fastest correct whole kernel.', {
        EVAL_DIR, FROZEN_KERNEL_PATH: KERNEL_PATH_ORIG,
        BASELINE_TREE: `${EVAL_DIR}/baseline`, COMMANDMENT,
        GPU_ID: GPU_SINGLE, SKILL_DIR: WORKFLOW_DIR,
        CANDIDATES: batch, TOP_K: batch.length, CANDIDATE_IMPORT_MODULES,
        MEGA_PROFILE,
        TIE_NOISE_PCT: MEGA_TIE_NOISE_PCT,
        TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC,
        LAUNCH_TARGET, BASELINE_LAUNCH_COUNT: optionalNumber(A.baseline_launch_count),
        BIMODAL_GUARDS, BASELINE_ACTIVATION,
        REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
        ACCURACY_METRIC, ACCURACY_THRESHOLD,
        REQUIRED_ACCURACY_CASES,
        REQUIRE_OVERLAP, REQUIRE_ATTRIBUTION, REQUIRE_ARTIFACT_DISTINCT,
        REQUIRE_GRAPH_CAPTURE: MEGA_REQUIRE_GRAPH_CAPTURE ? '1' : '0',
        DIRECT_GRAPH_ACCURACY: DIRECT_GRAPH_ACCURACY ? '1' : '0',
        ...(USE_EXPERT_SKILLS ? {
          EXPERT_SKILL_ID,
          EXPERT_SKILL_REVISION,
          EXPERT_SKILL_FILE: `${EXPERT_SKILL_DIR}/skill.md`,
          EXPERT_SKILL_PLAYBOOK: EXPERT_SKILL_PLAYBOOK_FILE,
          EXPERT_SKILL_CONTRACT: EXPERT_SKILL_CONTRACT_FILE,
          EXPERT_SKILL_VALIDATION: EXPERT_SKILL_VALIDATION_FILE,
          EXPERT_SKILL_RUNTIME_FILE,
          EXPERT_SKILL_TARGET_SCORE: EXPERT_SKILL_RECORDED_SCORE,
          EXPERT_SKILL_TARGET_BAND: [
            EXPERT_SKILL_RECORDED_LOW, EXPERT_SKILL_RECORDED_HIGH,
          ],
          EXPERT_SKILL_ACCURACY_CASES,
          EXPERT_SKILL_SOURCE_FILES,
        } : {}),
        GRAPH_CONTRACT_TOOL, ...RFI, ...GMI, ...MRI,
        GRAPH_CONTRACT_REPLAYS: REQUIRED_REPLAYS,
        SELECTED_WORKSPACE: `${EVAL_DIR}/mega_selected`,
      }),
    { phase: 'Validate', label, schema: MEGA_SELECTION_SCHEMA,
      timeout_ms: finalistTimeoutS * 1000, timeout_marker: true, max_retries: 1 });
  if (result && (result.__agent_timed_out || result.claim_complete === true)) return result;
  // Recovery is a read-only reconciliation; model a fixed bounded window on the deterministic clock.
  const recoveryRemainingS = Math.min(300, Math.max(0, Math.floor(finalistTimeoutS)));
  if (recoveryRemainingS < 30) return result;
  const recovered = await agentT(
    roleAgent('director', 'recover_mega',
      `RECOVER ONLY a completed finalist selection from ${EVAL_DIR}. Read the atomic selection ` +
      `evidence and ${EVAL_DIR}/mega_selected; do not run a GPU command, take a lease, edit a ` +
      `candidate, or create a new selection.`, {
        EVAL_DIR, CANDIDATES: batch, SELECTED_WORKSPACE: `${EVAL_DIR}/mega_selected`,
        SKILL_DIR: WORKFLOW_DIR,
      }),
    { phase: 'Validate', label: `${label}:recover`, schema: MEGA_SELECTION_SCHEMA,
      timeout_ms: Math.min(300, recoveryRemainingS) * 1000, max_retries: 1 });
  return recovered && recovered.claim_complete === true ? recovered : result;
};
let finalists = [];

// Mega provenance/topology labels never change the quality bar. Every source must pass the same final
// correctness/guard contract, and the same frozen-baseline score chooses what ships.
let mega_deliverable = null, mega_note = '';
// Set by the phase hooks below, read across phases.
let FINAL_WORKSPACE = null;
let LADDER = null;
let PRIOR_ART = null;
let PRIOR_ART_RECORDED = null;
let SHELF_STATS = null;
let WORKFLOW_REVISION_START = null;
let benchCache = null;
let benchR = null;
let ranked = null;
let reportReady = null;

// ---------------------------------------------------------------------------
// mode=mega phase hooks
// ---------------------------------------------------------------------------
async function megaAfterSetup() {
  if (setup.mega_workspace_from_original !== true || setup.mega_candidate_state_checked !== true) {
    throw new Error(
      'MEGA SETUP REFUSED: ordinary workspace must be seeded from KERNEL_PATH_ORIG. Candidate state ' +
      'must be checked and resumes only inside STATE_DIR/candidates; a legacy STATE_DIR/best must never ' +
      'replace the frozen base.');
  }
  if (STRICT_AUTONOMY && setup.resumed) {
    throw new Error(
      'STRICT AUTONOMY REFUSED: director seeded this run from STATE_DIR/best. A proof run must start ' +
      'from the frozen input tree with an empty/fresh state directory; otherwise inherited code and ' +
      'manual state edits cannot be distinguished from work derived in this run.');
  }
  if (STRICT_AUTONOMY && (setup.workflow_clean !== true ||
      !String(setup.workflow_revision || '').trim())) {
    throw new Error(
      'STRICT AUTONOMY REFUSED: GEAK workflow/knowledge must be a clean, committed revision before ' +
      'launch. Mid-wave or uncommitted instructions make the run non-reproducible.');
  }
  WORKFLOW_REVISION_START = String(setup.workflow_revision || '');
  if (STRICT_AUTONOMY) {
    const roots = new Set(Array.isArray(CONTAINMENT_PREFLIGHT.roots)
      ? CONTAINMENT_PREFLIGHT.roots.map((r) => String(r).replace(/\/+$/, '')) : []);
    const expectedRoots = [KERNEL_PATH_ORIG, EXP_ROOT, STATE_DIR, WORKFLOW_DIR]
      .filter(Boolean).map((r) => String(r).replace(/\/+$/, ''));
    const missingRoots = expectedRoots.filter((r) => !roots.has(r));
    if (CONTAINMENT_PREFLIGHT.content_scan !== 'clean' ||
        CONTAINMENT_PREFLIGHT.skill_address_scan !== 'clean' ||
        String(CONTAINMENT_PREFLIGHT.workflow_revision || '') !== WORKFLOW_REVISION_START ||
        Number(CONTAINMENT_PREFLIGHT.reference_hash_count || 0) !== KNOWN_REFERENCE_HASHES.length ||
        missingRoots.length) {
      throw new Error(
        `STRICT AUTONOMY REFUSED: containment attestation is not bound to this workflow/roots ` +
        `(missing roots: ${missingRoots.join(', ') || 'none'}).`);
    }
  }

  // Containment enforcement after Setup (the startup check only filters paths): run the two content
  // scanners over the materialised eval dir, baseline copy and inherited assets. It is an agent because
  // the workflow script has no shell.
  if (STRICT_AUTONOMY && CONTAINMENT_PREFLIGHT.clean === true) {
    log(`CONTAINMENT PREFLIGHT clean: trusted bootstrap scanned the run tree before agent startup ` +
        `(marker manifest ${String(CONTAINMENT_PREFLIGHT.marker_manifest_sha256 || '').slice(0, 12)}…, ` +
        `${Number(CONTAINMENT_PREFLIGHT.reference_hash_count || 0)} opaque reference hashes).`);
  }
  if (CAPABILITY_EVAL && RUN_TREE_ANCESTOR &&
      !(STRICT_AUTONOMY && CONTAINMENT_PREFLIGHT.clean === true)) {
    const gate = await agentT(
      `Run two containment scanners and report their exit codes verbatim. Do not fix anything, do not ` +
      `read any file they flag, do not summarise the flagged content — reporting the paths is the whole job.\n\n` +
      // Scan every ref tree (the default cap of 200 makes the hit count a lower bound). The full sweep
      // costs about five minutes once, before any lease.
      `1. REF_SCAN_MAX_TREES=100000 bash ${WORKFLOW_DIR}/scripts/reference_leak_sweep.sh ` +
      `--tree ${RUN_TREE_ANCESTOR} --markers ${REFERENCE_LEAK_MARKER_FILE || '(missing-marker-file)'}\n` +
      `   If its output still contains a REF_SCAN_MAX_TREES coverage NOTE, the scan is PARTIAL: say so\n` +
      `   in your note and return verdict "unknown" rather than clean, whatever the exit code was.\n` +
      `2. bash ${WORKFLOW_DIR}/scripts/skill_address_scan.sh --skills-dir ${EXPERT_SKILLS_DIR || '(skip: expert skills off)'} ` +
      `--scan-root ${RUN_TREE_ANCESTOR}\n\n` +
      `If a script is missing or its --skills-dir is empty, record that step as "skipped" with the reason ` +
      `and do NOT call it clean. "The scanner did not run" and "the scanner found nothing" are different ` +
      `results and this run has already been damaged twice by conflating them.`,
      { phase: 'Setup', label: 'containment gate', schema: CONTAINMENT_GATE_SCHEMA });
    if (!gate || gate.verdict !== 'clean') {
      const verdict = gate ? gate.verdict : 'unknown';
      if (STRICT_AUTONOMY || verdict === 'leak') {
        throw new Error(
          `CONTAINMENT GATE FAILED (${verdict}): strict/capability evidence requires an explicit clean ` +
          `scan of ${RUN_TREE_ANCESTOR} before budget is spent.\n` +
          `${gate ? (gate.findings || []).join('\n') : 'gate agent returned nothing'}\n` +
          `${gate ? gate.note || '' : ''}`);
      }
      log(`CONTAINMENT GATE ${verdict}: scanner did not establish clean. ${gate ? gate.note || '' : ''}`);
    } else log(`CONTAINMENT GATE clean: ${gate.note || ''}`);
  }
  log(`MEGA unified mode: Expert Skills are ${USE_EXPERT_SKILLS ? 'enabled as advisory context' :
    'disabled'}; all candidates use the same autonomous lifecycle.`);
}

async function megaAnalyze() {
  analysis = await agentT(
    roleAgent('mega_search_lead',
      'analyze', 'Analyze the kernel and write the roadmap.', {
      WORKSPACE: CANONICAL, EVAL_DIR, TASK, SKILL_DIR: WORKFLOW_DIR,
      KERNEL_KNOWLEDGE_DIR: '',
      GPUS_PER_JOB: String(GPU_RESOURCE.gpusPerJob),
      ...(A.require_task_graph ? { REQUIRE_TASK_GRAPH: '1' } : {}),
      ...(USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE ? {
        EXPERT_SKILL_PLANNER_EXTENSION: EXPERT_SKILL_PLANNER_EXTENSION_FILE,
        EXPERT_SKILL_ID,
        EXPERT_SKILL_REVISION,
        EXPERT_SKILL_BUNDLE_TOOL,
        EXPERT_SKILL_BUNDLE_SHA256,
        EXPERT_SKILL_PLANNER_EXTENSION_SHA256,
        EXPERT_SKILL_CONTRACT_SHA256,
        EXPERT_SKILL_VALIDATION_STATUS,
        EXPERT_SKILL_USAGE,
      } : {}),
      ...(CAPABILITY_EVAL ? { CAPABILITY_EVAL: '1' } : {}),
      ...(STRICT_AUTONOMY ? {
        STRICT_AUTONOMY: '1', TARGET_GUARDS, REGRESSION_GUARDS,
        PROMOTION_METRIC, LAUNCH_TARGET: Number(A.launch_target || 2),
        REQUIRE_OVERLAP, REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
      } : {}),
      ...({}),
    }),
    { phase: 'Analyze', label: 'tech_lead:analyze',
      schema: MEGA_ANALYZE_SCHEMA,
      ...(MEGA_PRODUCTION
        // 30 min: the task-graph analysis (four docs + source + DAG) took 13.5 min fresh and was cut
        // AT StructuredOutput at 15 min on a resumed wave. One-time; not drawn from MEGA_CLOCK_MS.
        ? { timeout_ms: 1800000, timeout_marker: true, max_retries: 1 } : {}) });
  if (analysis && analysis.__agent_timed_out) {
    return { exit: {
      mode: MODE, mega_deliverable: false, eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
      final_speedup: 0, final_geomean: 0, validation_status: 'agent_timeout',
      reason: 'Analyze exceeded the production timeout; no second Analyze or GPU phase was started',
    } };
  }

  {
    const d = analyzeResumeDegenerate(
      INCREMENTAL, analysis, !!A.require_task_graph, true,
      USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE
        ? EXPERT_SKILL_REVISION : '',
      USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE
        ? EXPERT_SKILL_ID : '',
      USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE
        ? EXPERT_SKILL_PLANNER_EXTENSION_SHA256 : '',
      USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE
        ? EXPERT_SKILL_BUNDLE_SHA256 : '',
      USE_EXPERT_SKILLS ? EXPERT_SKILL_VALIDATION_STATUS : '',
      USE_EXPERT_SKILLS ? EXPERT_SKILL_USAGE : '');
    if (d.retry) {
      log(d.reason);
      // Identical to the call above except that RESUME_INPUT is absent — that omission IS the fix.
      const full = await agentT(
        roleAgent('mega_search_lead',
          'analyze', 'Analyze the kernel and write the roadmap.', {
          WORKSPACE: CANONICAL, EVAL_DIR, TASK, SKILL_DIR: WORKFLOW_DIR,
          KERNEL_KNOWLEDGE_DIR: '',
          GPUS_PER_JOB: String(GPU_RESOURCE.gpusPerJob),
          ...(A.require_task_graph ? { REQUIRE_TASK_GRAPH: '1' } : {}),
          ...(USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE ? {
            EXPERT_SKILL_PLANNER_EXTENSION: EXPERT_SKILL_PLANNER_EXTENSION_FILE,
            EXPERT_SKILL_ID,
            EXPERT_SKILL_REVISION,
            EXPERT_SKILL_BUNDLE_TOOL,
            EXPERT_SKILL_BUNDLE_SHA256,
            EXPERT_SKILL_PLANNER_EXTENSION_SHA256,
            EXPERT_SKILL_CONTRACT_SHA256,
            EXPERT_SKILL_VALIDATION_STATUS,
            EXPERT_SKILL_USAGE,
          } : {}),
          ...(CAPABILITY_EVAL ? { CAPABILITY_EVAL: '1' } : {}),
          ...(STRICT_AUTONOMY ? {
            STRICT_AUTONOMY: '1', TARGET_GUARDS, REGRESSION_GUARDS,
            PROMOTION_METRIC, LAUNCH_TARGET: Number(A.launch_target || 2),
            REQUIRE_OVERLAP, REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
          } : {}),
          // Kept on the recovery call too: the full analysis re-derives the ladder from the source,
          // but the rungs a prior wave already spent, and the ones it left owed, exist only here.
        }),
        { phase: 'Analyze', label: 'tech_lead:analyze:full',
          schema: MEGA_ANALYZE_SCHEMA,
          ...(MEGA_PRODUCTION
            // 900s: same rationale as the primary analyze call site above.
            ? { timeout_ms: 900000, timeout_marker: true, max_retries: 1 } : {}) });
      if (full && full.__agent_timed_out) {
        return { exit: {
          mode: MODE, mega_deliverable: false, eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
          final_speedup: 0, final_geomean: 0, validation_status: 'agent_timeout',
          reason: 'full Analyze fallback exceeded the production timeout; no GPU phase was started',
        } };
      }
      const got = (full && Array.isArray(full.candidate_directions) ? full.candidate_directions : [])
        .filter((c) => c && (c.id || c.title));
      const fullContract = analyzeResumeDegenerate(false, full,
        !!A.require_task_graph, true,
        USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE
          ? EXPERT_SKILL_REVISION : '',
        USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE
          ? EXPERT_SKILL_ID : '',
        USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE
          ? EXPERT_SKILL_PLANNER_EXTENSION_SHA256 : '',
        USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE
          ? EXPERT_SKILL_BUNDLE_SHA256 : '',
        USE_EXPERT_SKILLS ? EXPERT_SKILL_VALIDATION_STATUS : '',
        USE_EXPERT_SKILLS ? EXPERT_SKILL_USAGE : '');
      if (got.length && !fullContract.retry) {
        log(`ANALYZE RE-RUN recovered a ladder of ${got.length} rung(s). Using the full analysis.`);
        analysis = full;
      } else {
        log(`ANALYZE RE-RUN still missed its structured contract: ${fullContract.reason}`);
        if (full) analysis = full;
        return { exit: {
          mode: MODE, mega_deliverable: false, eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
          final_speedup: 0, final_geomean: 0,
          validation_status: 'analyze_contract_incomplete',
          reason: fullContract.reason,
        } };
      
      }
    }
  }
  analysis = applyBaselineOperatorMap(analysis);
  if (REQUIRE_TASK_GRAPH) {
    const g = taskGraphGate(analysis && analysis.task_graph);
    if (g.summary) log(g.summary);
    TASK_GRAPH_CAVEAT = g.caveat;
    if (g.caveat) log(g.caveat);
    // Print both the graph and the pipe table and carry both to the report; they fail independently.
    const pg = pipeOccupancyGate(analysis && analysis.resource_timeline, []);
    if (pg.summary) log(pg.summary);
    PIPE_TABLE_CAVEAT = pg.caveat;
    if (pg.caveat) log(pg.caveat);
  }

  // Analyze's ladder and what has been taken off it, threaded into every plan_round call.
  LADDER = (analysis && Array.isArray(analysis.candidate_directions)) ? analysis.candidate_directions : [];
  if (STRICT_AUTONOMY) {
    if (!LADDER.length) {
      throw new Error(
        'STRICT AUTONOMY REFUSED: Analyze returned no candidate_directions. A proof run with no ' +
        'machine-readable ladder cannot show that the two-launch terminal was derived.');
    }
    const terminalTargets = LADDER.filter((c) => c && c.target_shape &&
      Number(c.target_shape.launches) <= LAUNCH_TARGET);
    if (!terminalTargets.length) {
      throw new Error(
        `STRICT AUTONOMY REFUSED: no ladder rung declares target_shape.launches <= ${LAUNCH_TARGET}. ` +
        'A prose promise to fuse the operator cannot close the machine-tracked acceptance shape.');
    }
    const incompleteTargets = terminalTargets.filter((c) =>
      !Array.isArray(c.mandatory_arms) || c.mandatory_arms.length === 0 ||
      !Array.isArray(c.target_shape.stages_fused) || c.target_shape.stages_fused.length === 0 ||
      (REQUIRE_OVERLAP && c.target_shape.require_overlap !== true));
    if (incompleteTargets.length) {
      throw new Error(
        'STRICT AUTONOMY REFUSED: terminal target rung(s) must declare mandatory_arms and the required ' +
        `overlap shape: ${incompleteTargets.map((c) => c.id || c.title).join(', ')}`);
    }
  }
  {
    const lg0 = roadmapLadderGate(LADDER, [], new Set());
    if (lg0.summary) log(lg0.summary);
    if (lg0.caveat) log(lg0.caveat);
  }
  // Under working_kernel an empty ladder disables rung, chain-debt and forcing tracking. Not fatal
  // (analyze_resume_fallback may re-request directions), but escalated to the report.
  if (WORKING_KERNEL && !LADDER.length) {
    const msg = 'LADDER EMPTY under objective=working_kernel: Analyze returned no candidate_directions, ' +
      'so there is no staged path to a fused terminal rung. The chain-debt and terminal-forcing ' +
      'machinery has nothing to act on and every round will start from the profile. Re-run Analyze so ' +
      'it emits the fusion ladder, or this wave cannot converge on the two-launch shape by construction.';
    log(msg);
    CHAIN_CAVEATS.push(`setup: ${msg}`);
  }

  // Prior art: an existing implementation is a measurement, not a round; one missing from the tree is a
  // ceiling on the run. A missing prior_art key (unknown) is different from [] (searched, none found).
  PRIOR_ART_RECORDED = !!(analysis && Array.isArray(analysis.prior_art));
  PRIOR_ART = PRIOR_ART_RECORDED ? analysis.prior_art : [];
  if (!PRIOR_ART_RECORDED) {
    log(`PRIOR ART UNRECORDED: analysis.json has no "prior_art" key, so the step-4d sweep is not on ` +
        `the record — this is NOT the same as finding none. Every later statement about what does or ` +
        `does not exist in the baseline is UNSOURCED and must be read as such.`);
  } else if (PRIOR_ART.length === 0) {
    log(`PRIOR ART: none — the sweep ran and reported an empty result.`);
  }
  for (const pa of PRIOR_ART) {
    const where = CAPABILITY_EVAL ? 'HIDDEN_REFERENCE' : (pa.implemented_at || '?');
    const eff = pa.measured_effect ? ` (measured ${pa.measured_effect})` : '';
    if (pa.in_baseline === false && CAPABILITY_EVAL) {
      // Same finding, opposite instruction. Here the absent implementation is the ANSWER KEY: it
      // bounds what a successful run should reach, and copying it makes the run unreadable.
      log(`PRIOR ART NOT IN BASELINE (capability eval): "${pa.direction}" exists at ${where}${eff} but ` +
          `is ABSENT from the tree being optimized. Treat it as the answer key, NOT as a source: it is ` +
          `the bar this run must reach on its own. Do not port it, and do not surface its location to ` +
          `engineers.`);
    } else if (pa.in_baseline === false) {
      log(`PRIOR ART NOT IN BASELINE: "${pa.direction}" is already implemented at ${where}${eff} but ` +
          `is ABSENT from the tree being optimized. Port or measure it before re-deriving it; if the ` +
          `intent was to optimize the tree that HAS it, this run is pointed at the wrong target.`);
    } else {
      log(`PRIOR ART: "${pa.direction}" already implemented at ${where}${eff}; ` +
          `enable via ${pa.how_to_enable || '?'}. Measure it, do not re-derive it.`);
    }
    // in_baseline must carry evidence.
    if (pa.in_baseline != null && !pa.evidence) {
      log(`PRIOR ART UNEVIDENCED: "${pa.direction}" asserts in_baseline=${pa.in_baseline} with no ` +
          `"evidence" field naming the check that established it. Treat the flag as unverified.`);
    }
  }

  // Leak scrub under capability_eval: flag engineer-visible artifacts that mention a known reference
  // path. It warns instead of aborting; the provenance gate in Verify is the enforcing half.
  if (CAPABILITY_EVAL && KNOWN_REFERENCE_PATHS.length) {
    const needles = KNOWN_REFERENCE_PATHS.flatMap((p) => {
      const base = p.replace(/\/+$/, '').split('/').pop();
      return [p.replace(/\/+$/, ''), ...(base && base.length > 6 ? [base] : [])];
    });
    // Workflow scripts have no fs, so this file-content check cannot run here; say so and point to the
    // sweep script that does walk the tree.
    if (typeof fs === 'undefined') {
      log('CONTAINMENT CHECK INERT: the in-workflow reference-path scan of EVAL_DIR cannot run — ' +
          'workflow scripts have no fs. This is not a clean result, it is no result. Run ' +
          'scripts/reference_leak_sweep.sh --tree <run tree> and scripts/skill_address_scan.sh ' +
          'out of band; only those actually read the tree.');
    }
    for (const f of (typeof fs === 'undefined' ? [] : ['roadmap.md', 'codebase_context.md', 'analysis.json'])) {
      let txt = '';
      try { txt = fs.readFileSync(`${EVAL_DIR}/${f}`, 'utf8'); } catch { continue; }
      // analysis.json legitimately carries locations in prior_art[].implemented_at (that field is for
      // the orchestrator and the human, and is not forwarded to engineers). Scan the rest of it only.
      if (f === 'analysis.json') {
        try {
          const j = JSON.parse(txt); delete j.prior_art; txt = JSON.stringify(j);
        } catch { /* unparseable -> scan verbatim, a false positive here is cheap */ }
      }
      const hit = needles.find((n) => txt.includes(n));
      if (hit) {
        log(`LEAK WARNING: ${f} mentions the reference location "${hit}" while capability_eval is on. ` +
            `Engineers read this file. Prior art must reach them as MECHANISM, never as an address — ` +
            `a path in the roadmap is how the last run ended up with a byte-identical copy of the ` +
            `answer. Verify's provenance check will reject any patch that imports from it.`);
      }
    }
  }
}

async function megaBenchmark() {
  const structuralBench = {
    baseline_per_case: structuralGuardNames.map((name) => ({
      name, latency_ms: 1, baseline_ms: 1, speedup: 0,
      role: TARGET_GUARDS.includes(name) ? 'target' : 'regression_guard',
      runs_ms: [], drift_pct: 0, params: 'structural-only placeholder',
      metric: PROMOTION_METRIC,
    })),
    baseline_geomean_ms: 1,
    num_test_cases: structuralGuardNames.length,
    reliable: false,
    positive_control: { claim_complete: false },
    notes: 'STRUCTURAL_ONLY: no GPU benchmark.',
  };
  benchCache = noGpuFrontMatter ? null : await fastTestCacheLoad('bench');
  const benchFirst = noGpuFrontMatter ? structuralBench : benchCache ? benchCache.bench : await agentT(
    roleAgent('benchmark_engineer', 'setup', 'Build the COMMANDMENT and record a reliable baseline.', {
      WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_SINGLE,
      ANALYSIS: analysis,
      TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC,
      STRICT_AUTONOMY, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD, ...RFI, ...GMI, ...MRI,
      ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
      ...(WORKLOAD_SPEC_PATH ? { WORKLOAD_SPEC_PATH } : {}),
      ...(WORKLOAD_SPEC ? { WORKLOAD_SPEC } : {}),
      ...(POSITIVE_CONTROL ? { POSITIVE_CONTROL } : {}),
    }),
    { phase: 'Benchmark', label: 'benchmark_engineer', schema: MEGA_BENCH_SCHEMA,
      ...(MEGA_PRODUCTION
        // Baseline/control setup has a separate production window and can recover artifacts on timeout.
        ? { timeout_ms: 10800000, timeout_marker: true, max_retries: 1 } : {}) });
  // A bench agent that dies usually failed to report, not to measure. Before giving up, ask one agent
  // to recover the results already on disk (recover only, never re-measure).
  benchR = benchFirst;
  // A timed-out setup agent usually left its baseline/control on disk: fall through to the GPU-free
  // RECOVERY below; only an empty disk makes the timeout terminal.
  const benchTimedOut = !!(benchR && benchR.__agent_timed_out);
  if (benchTimedOut) {
    log('Benchmark setup hit its full timeout window without returning. Its measurements are usually ' +
        'already on disk — attempting RECOVERY from EVAL_DIR before treating this as a terminal timeout.');
    benchR = null; // a timeout marker carries no baseline_per_case; force the recovery path below
  }
  if (!benchR || !benchR.baseline_per_case) {
    log('Benchmark setup returned nothing. Attempting RECOVERY from EVAL_DIR before aborting — an ' +
        'agent that died mid-phase usually left its measurements on disk.');
    benchR = await agentT(
      roleAgent('benchmark_engineer', 'recover',
        'The previous benchmark_engineer for this EVAL_DIR did not return. RECOVER ONLY: read ' +
        `${EVAL_DIR} — baseline_timing.json, COMMANDMENT.md, setup_ab_*.json, ab_logs/ — and ` +
        'reconstruct the return JSON from what is already there. Do NOT run any GPU command and do ' +
        'NOT take a lease: this step exists to avoid re-measuring, so a fresh measurement here is a ' +
        'failure, not a fallback. Write baseline_timing.json / COMMANDMENT.md if measurements exist ' +
        'but the file does not. A setup_ab_control*.json with claim_complete:true IS the positive ' +
        'control result — report its measured median and null arm rather than re-running it. If the ' +
        'measurements genuinely are not on disk, return baseline_per_case: [] and say so in notes. ' +
        'The timed-out agent may still be writing: if setup_ab_* files changed in the last 10 min, ' +
        'poll up to 12 min for baseline_timing.json before concluding.',
        { WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, ANALYSIS: analysis,
          TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC,
          STRICT_AUTONOMY, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
          ...(POSITIVE_CONTROL ? { POSITIVE_CONTROL } : {}) }),
      { phase: 'Benchmark', label: 'benchmark_recover', schema: MEGA_BENCH_SCHEMA,
        ...(MEGA_PRODUCTION ? { timeout_ms: 1200000, max_retries: 1 } : {}) });
    if (!benchR || !Array.isArray(benchR.baseline_per_case) || !benchR.baseline_per_case.length) {
      if (benchTimedOut) {
        // Nothing recoverable: return an explicit timeout status rather than throwing.
        return { exit: {
          mode: MODE, mega_deliverable: false, eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
          final_speedup: 0, final_geomean: 0, validation_status: 'agent_timeout',
          reason: 'benchmark/calibration agent exceeded the production timeout; recovery from EVAL_DIR ' +
                  'found no baseline on disk',
        } };
      }
      throw new Error('Benchmark setup failed: no baseline recorded, and none recoverable from EVAL_DIR');
    }
    log(`Benchmark RECOVERED from disk: ${benchR.baseline_per_case.length} cases. ${benchR.notes || ''}`);
  }
  bench = benchR;
}

async function megaCalibrate() {
  if (POSITIVE_CONTROL) {
    const pc = benchR.positive_control || {};
    const pcComplete = pc.claim_complete === true;
    if (!pcComplete) {
      log(`Positive control PENDING: claim_complete is not true` +
        `${pc.attempt_id ? ` for attempt ${pc.attempt_id}` : ''}. A force-emitted or partial 0.00% ` +
        `is not a failed instrument. Candidate authoring may continue, but mega performance ranking ` +
        `remains disabled until a completed calibration manifest is recovered.`);
    } else {
    const lo = Number(POSITIVE_CONTROL.expected_pct_lo);
    const hi = Number(POSITIVE_CONTROL.expected_pct_hi);
    const got = Number(pc.measured_pct);
    // The bounds are asymmetric: below lo the instrument cannot see the effect (abort); above hi it
    // over-reads (pass with overshoot); absurd or wrong sign measures something else (abort). Computed on
    // magnitude plus expected sign, so a control may be a deliberate slowdown (benchmark_engineer.md 5b).
    // <<REPLAY:pc_gate>>  scripts/replay_runs.js lifts this region verbatim; keep it pure (reads only
    // lo, hi, got, pc, POSITIVE_CONTROL; no log, throw or outside state).
    const mLo = Math.min(Math.abs(lo), Math.abs(hi));       // smallest effect the loop must resolve
    const mHi = Math.max(Math.abs(lo), Math.abs(hi));
    const wantSign = Math.sign(lo + hi) || 1;               // +1 = control is faster, -1 = slower
    const mGot = Math.abs(got);
    const ABSURD = Number.isFinite(Number(POSITIVE_CONTROL.implausible_pct))
      ? Math.abs(Number(POSITIVE_CONTROL.implausible_pct)) : (Number.isFinite(mHi) ? 2 * mHi : NaN);
    const nullArm = Number(pc.null_arm_pct);
    // An overshoot is only benign if the null arm is quiet. If the byte-identical arm is ALSO reading
    // high, the interleave is drifting and the overshoot is the drift, not the effect.
    const nullQuiet = Number.isFinite(nullArm) && Math.abs(nullArm) <= mLo / 2;
    // Pre-flight: switch_present=false means both arms ran the same code, so the control did not run
    // whatever it reported. Undefined keeps the old behaviour (historical replays unchanged).
    const switchAbsent = pc.switch_present === false;
    const ran = pc.ran !== false && !switchAbsent &&
      Number.isFinite(got) && Number.isFinite(lo) && Number.isFinite(hi);

    // magnitude=constructed marks a synthetic control whose band is a target. An undershoot is admissible
    // only if the reading is an effect: above a floor, at least RESOLVE_K x the worst null pair, and sign-
    // unanimous. A recorded control gets no such latitude.
    const constructed = String(POSITIVE_CONTROL.magnitude || 'recorded') === 'constructed';
    const UNDERSHOOT_FRAC = 0.5;   // below half the target, assume the injection did not take
    const RESOLVE_K = 3;           // effect must beat the worst null pair by this factor
    const nullPairs = Array.isArray(pc.null_pairs_pct) ? pc.null_pairs_pct.map(Number).filter(Number.isFinite) : [];
    const nullWorst = nullPairs.length ? Math.max(...nullPairs.map(Math.abs))
      : (Number.isFinite(nullArm) ? Math.abs(nullArm) : NaN);
    const ctrlPairs = Array.isArray(pc.control_pairs_pct) ? pc.control_pairs_pct.map(Number).filter(Number.isFinite) : [];
    const signUnanimous = ctrlPairs.length >= 3 && ctrlPairs.every((d) => Math.sign(d) === wantSign);
    const resolvedByScale = Number.isFinite(nullWorst) && mGot >= RESOLVE_K * nullWorst;

    // For a bimodal null, a second distribution-free route: effect and null pairs do not overlap by
    // magnitude. Both routes must still clear mLo * UNDERSHOOT_FRAC.
    const absCtrl = ctrlPairs.map(Math.abs);
    const absNull = nullPairs.map(Math.abs);
    const resolvedBySeparation = absCtrl.length >= 5 && absNull.length >= 5 &&
      Math.min(...absCtrl) > Math.max(...absNull);
    const resolved = resolvedByScale || resolvedBySeparation;
    const undershoot = ran && constructed && Math.sign(got) === wantSign &&
      mGot < mLo && mGot >= mLo * UNDERSHOOT_FRAC && resolved && signUnanimous;

    // Wrong sign is an insensitivity failure, not an overshoot: the loop did not see the change it was
    // handed, whatever else it saw.
    const tooSmall = ran && (Math.sign(got) !== wantSign || (mGot < mLo && !undershoot));
    const absurd = ran && Number.isFinite(ABSURD) && mGot > ABSURD;
    const overshoot = ran && mGot > mHi && !absurd;
    const ok = ran && !tooSmall && !absurd && (!overshoot || nullQuiet);
    // <</REPLAY:pc_gate>>
    PC_RAN = ran;
    PC_PASSED = ok && (megaCalibrationClaimPass({ ...pc, passed: ok }, POSITIVE_CONTROL));
    if (switchAbsent) {
      log(`POSITIVE CONTROL SWITCH ABSENT: "${POSITIVE_CONTROL.name || 'unnamed'}" names a knob that ` +
          `is not in the tree it was run against, so both arms executed identical code and the ` +
          `${Number.isFinite(got) ? got.toFixed(2) + '%' : 'reported'} reading is a base-vs-base null. ` +
          `This is a LAUNCH-ARGS defect, not an instrument failure — fix \`positive_control.how\` to ` +
          `name a switch that exists, or have the benchmark engineer construct a synthetic control ` +
          `(benchmark_engineer.md 5b) and declare magnitude:'constructed'.`);
    }
    log(`Positive control "${POSITIVE_CONTROL.name || 'unnamed'}": ` +
        `measured ${Number.isFinite(got) ? got.toFixed(2) + '%' : 'NOT RUN'}, ` +
        `expected ${lo}..${hi}% (absurd above ${Number.isFinite(ABSURD) ? ABSURD.toFixed(2) + '% in magnitude' : '?'}) ` +
        `-> ${ok ? (overshoot ? 'PASS (OVERSHOOT)' : undershoot ? 'PASS (UNDERSHOOT)' : 'PASS') : 'FAIL'}` +
        (Number.isFinite(nullArm) ? ` (null arm ${nullArm.toFixed(2)}%)` : ' (null arm UNREPORTED)'));
    if (ok && undershoot) {
      // An undershoot may proceed, but every reported number carries the caveat.
      PC_UNDERSHOOT = `the positive control read ${got.toFixed(2)}% for a CONSTRUCTED effect targeted at ` +
        `${lo}..${hi}%, with a ${Number.isFinite(nullArm) ? nullArm.toFixed(2) : '?'}% null arm ` +
        `(worst null pair ${Number.isFinite(nullWorst) ? nullWorst.toFixed(2) : '?'}pp) and ` +
        `${ctrlPairs.length}/${ctrlPairs.length} pairs agreeing in sign. The effect is ` +
        `${Number.isFinite(nullWorst) && nullWorst > 0 ? (mGot / nullWorst).toFixed(1) + 'x' : '>>'} the worst ` +
        `null pair, so the loop resolves it; the shortfall against the target is a sizing miss in the ` +
        `injected knob, which was never measured before this run. Treat every effect below as possibly ` +
        `reading LOW by roughly ${(mLo - mGot).toFixed(2)}pp at this size — a marginal win may be real, ` +
        `and a marginal loss may be worse than it looks.`;
      log(`POSITIVE CONTROL UNDERSHOOT: ${PC_UNDERSHOOT}`);
    }
    if (ok && overshoot) {
      // Not a failure, but it must not vanish either: it is a standing caveat on every number the run
      // goes on to report, and the report has to say so.
      PC_OVERSHOOT = `the positive control read ${got.toFixed(2)}% for a change recorded at ` +
        `${lo}..${hi}%, with a ${nullArm.toFixed(2)}% null arm. The harness resolves the known effect ` +
        `with the right sign and clean separation, so it is sensitive enough to trust; but it reads ` +
        `HIGH by roughly ${(mGot - mHi).toFixed(2)}pp at this effect size, so treat every speedup below ` +
        `as possibly carrying the same upward scale bias.`;
      log(`POSITIVE CONTROL OVERSHOOT: ${PC_OVERSHOOT}`);
    }
    if (!ok) {
      const why = `Positive control FAILED: a change with a known effect of ${lo}..${hi}% measured ` +
        `${Number.isFinite(got) ? got.toFixed(2) + '%' : 'nothing (control did not run)'}. ` +
        (absurd
          ? `That is beyond the ${ABSURD.toFixed(2)}% plausibility ceiling — at that size the harness ` +
            `is not measuring this change, it is measuring something else (wrong arm, wrong guard, ` +
            `wrong binary). `
          : overshoot
            ? `It overshot the expected ceiling AND the null arm is ` +
              `${Number.isFinite(nullArm) ? Math.abs(nullArm).toFixed(2) + '%, too large relative to the ' + mLo + '% effect' : 'UNREPORTED'}, ` +
              `so the excess cannot be told apart from interleave drift. `
            : (Math.sign(got) !== wantSign && Number.isFinite(got)
                ? `It came back with the WRONG SIGN, so the loop is not tracking the change it was ` +
                  `handed — a wiring error (arms swapped, wrong guard, wrong binary) before it is a ` +
                  `sensitivity one. `
                : ``) +
              `The measurement loop cannot resolve an effect of the size this run is looking for, so ` +
              `every number it reports — including 1.000x — is uninterpretable. Fix the harness, not ` +
              `the kernel. `) +
        // Say which undershoot condition failed, in the reader's numbers; the two causes need opposite fixes.
        (tooSmall && Math.sign(got) === wantSign && mGot < mLo
          ? `This is an UNDER-read, so before touching the harness, establish which of the two it is. ` +
            (!constructed
              ? `This control is declared \`magnitude: 'recorded'\`, meaning ${lo}..${hi}% is an effect ` +
                `someone has actually measured — so reading ${mGot.toFixed(2)}% IS the instrument's fault. ` +
                `If instead that band was a target you aimed a synthetic knob at and never measured, the ` +
                `control is \`magnitude: 'constructed'\` and should be declared as such; do NOT relabel a ` +
                `recorded effect to get past this gate, that falsifies the experiment. `
              : mGot < mLo * UNDERSHOOT_FRAC
                ? `It read under half the target (${(mLo * UNDERSHOOT_FRAC).toFixed(2)}%), which is the ` +
                  `signature of an injection that never took effect — arms sharing a JIT cache entry ` +
                  `(knowledge/jit_arm_isolation.md), the env var unread, the gated code not on the hot ` +
                  `path. Verify the two arms have distinct disk_keys before blaming the loop. `
                : !signUnanimous
                  ? `Its ${ctrlPairs.length} pair(s) do not all agree in sign, so the median is coming out ` +
                    `of a disagreement rather than an effect — that IS a resolution failure. `
                  : !resolved
                    ? `It is only ${Number.isFinite(nullWorst) && nullWorst > 0 ? (mGot / nullWorst).toFixed(1) + 'x' : '?'} ` +
                      `the worst null pair (${Number.isFinite(nullWorst) ? nullWorst.toFixed(2) : '?'}pp), under the ` +
                      `${RESOLVE_K}x an effect must clear to be told apart from the interleave, and its ` +
                      `pairs do not separate from the null's either ` +
                      `(smallest control pair ${absCtrl.length ? Math.min(...absCtrl).toFixed(2) : '?'}pp ` +
                      `vs largest null pair ${absNull.length ? Math.max(...absNull).toFixed(2) : '?'}pp` +
                      `${absCtrl.length >= 5 && absNull.length >= 5 ? '' : '; separation needs >=5 pairs on BOTH arms'}), ` +
                      `so neither route to resolution is open. Quiet the interleave or raise the injected ` +
                      `cost. If the null is fat-tailed rather than noisy -- a few readings well above an ` +
                      `otherwise tight cluster -- collect more null pairs and check whether the two arms ` +
                      `separate; that route does not care about the tail. `
                    : ``)
          : ``) +
        `Note from benchmark engineer: ${pc.note || '(none)'}`;
      if (PC_ABORT) throw new Error(why);
      log(`WARNING: ${why}`);
    }
    }
  }

  if (PC_PASSED) {
    const pc = benchR.positive_control || {};
    megaMeasurementCalibration = {
      ready: true,
      candidate_id: 'synthetic_control',
      attempt_id: pc.attempt_id || 'benchmark-control',
      evidence_manifest: pc.evidence_manifest || '',
      note: `independent synthetic control passed at ${Number(pc.measured_pct).toFixed(2)}%`,
    };
  }
  if (!PC_PASSED) {
    await remeasureMegaCalibration('after-setup', benchR.positive_control);
  }
}

async function megaProfile() {
  // Before a candidate exists, use only the caller's measured baseline table and declared launch
  // shape. Candidate-only overlap/path metrics are established by score/finalist verification.
  const targetBaseline = BASELINE_PER_CASE.find((row) =>
    TARGET_GUARDS.includes(String(row && row.name || ''))) || BASELINE_PER_CASE[0] || {};
  profileSummary = {
    bottleneck: 'unknown',
    profiler_used: 'benchmark-only',
    device: String(A.device_description || 'task-declared accelerator'),
    dispatch_count: Number(A.baseline_launch_count ||
      targetBaseline.launch_count || targetBaseline.dispatch_count || 0),
    key_metrics: {},
    top_kernels: [],
    top_opportunities: [
      'derive search directions from the frozen-source task graph and rank-max baseline; ' +
      'do not infer fused overlap before a fused candidate exists',
    ],
    rank_max_ms: Number(targetBaseline.latency_ms || targetBaseline.baseline_ms || 0),
    path_marker: '',
    summary_path: '',
    shift_note: 'pre-candidate baseline only',
  };
  if (GMI.MEGA_MEASUREMENT_GUIDE && !noGpuFrontMatter) {
    const mp = await agentT(roleAgent('profile_engineer', 'mega_baseline',
      'Measure the frozen baseline per stage: follow MEGA_MEASUREMENT_GUIDE §1, not the role file.', {
        WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_SINGLE,
        COMMANDMENT, BASELINE_PER_CASE, TARGET_GUARDS, REGRESSION_GUARDS, ...GMI, ...MRI,
      }), { phase: 'Profile', label: 'profile_engineer:mega_baseline', schema: PROFILE_SCHEMA,
      timeout_ms: 3600000, max_retries: 1 });
    // Failure or timeout keeps the stub; it never ends the wave.
    if (mp && mp.bottleneck) profileSummary = { ...profileSummary, ...mp };
    if (mp && pipeOccupancyGate(mp.resource_timeline, []).verdict !== 'MISSING') {
      analysis = { ...analysis, resource_timeline: { ...(analysis && analysis.resource_timeline), ...mp.resource_timeline } };
    }
  }
  if (profileSummary && profileSummary.__agent_timed_out) {
    return { exit: {
      mode: MODE, mega_deliverable: false, eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
      final_speedup: 0, final_geomean: 0, validation_status: 'agent_timeout',
      reason: 'baseline profile exceeded the production timeout; no candidate GPU work was started',
    } };
  }
  const baselineAnalysisResult = await runProfileAnalysis(
    profileSummary, 0, 'analysis_engineer:baseline');
  if (profileSummary) {
    profileSummary = {
      ...profileSummary,
      analysis_result: baselineAnalysisResult,
    };
  }

  // Publish this wave's front-matter measurements to the fast-test cache when any stage ran fresh.
  // No agent() call when MEGA_FAST_TEST is off.
  if (!noGpuFrontMatter && MEGA_FAST_TEST && !benchCache) await fastTestCachePublish();
}

async function megaLoopInit() {
  // The evidence K gets resized from. See the report dispatch for why it is reported even when it is
  // all zeros.
  SHELF_STATS = { k: SHELF_OFFER_K, shelved: 0, withheld_unknown_footprint: 0,
    rounds_offered: 0, offers: 0, hits: 0, hit_ids: [] };

  // Resume: restore cumulative speedup and insight/ledger history from the prior wave. No-op on a fresh
  // run.
  if (setup.resumed && setup.prior_state) {
    const ps = setup.prior_state;
    if (Number.isFinite(ps.cumulative) && ps.cumulative > cumulative) cumulative = ps.cumulative;
    if (Array.isArray(ps.insights)) {
      insightBook = mergeInsights(insightBook, ps.insights, 0, false).book;
      history.insights = renderInsights(insightBook);
    }
    if (Array.isArray(ps.ledger)) history.ledger = ps.ledger;
    if (ps.bottleneck_now) history.bottleneck_now = ps.bottleneck_now;
    if (Array.isArray(ps.best_per_case) && ps.best_per_case.length) bestPerCase = ps.best_per_case;
    if (Array.isArray(ps.shelf)) {
      shelf = ps.shelf.filter((e) => e && e.id && !e.absorbed)
        .map((e) => ({ ...e, base_round: 0, files: shelfFiles(e),
          footprint: shelfFiles(e).length ? 'known' : 'unknown' }));
    }
    if (ps.absorbed_files && typeof ps.absorbed_files === 'object') {
      const prior = [];
      for (const k of Object.keys(ps.absorbed_files)) for (const f of (ps.absorbed_files[k] || [])) prior.push(f);
      if (prior.length) absorbedByRound[0.5] = [...new Set(prior.map(shelfFile))];
    }
    const megaRestoreRejected = [];
    if (Array.isArray(ps.candidate_registry)) {
      for (const c of ps.candidate_registry) {
        const n = normalizeMegaCandidate(c);
        if (!n.id_valid || !n.source_valid || !n.source_id_valid) {
          log(`MEGA candidate state rejected: id=${n.id || '(empty)'} source=${String(c && c.source || '')}.`);
          megaRestoreRejected.push(n.id || '(empty)');
          continue;
        }
        const staleIdentityReasons = [];
        if (CHECK_EXPERT_SKILL_CONTRACT && n.structural_verified) {
          if (n.structural_skill_id !== EXPERT_SKILL_ID) {
            staleIdentityReasons.push('Skill id changed or was not recorded');
          }
          if (n.structural_contract_revision !== EXPERT_SKILL_REVISION) {
            staleIdentityReasons.push('Skill revision changed');
          }
          if (EXPERT_SKILL_CONTRACT_SHA256 &&
              n.structural_contract_sha256 !== EXPERT_SKILL_CONTRACT_SHA256) {
            staleIdentityReasons.push('contract digest changed');
          }
          if (EXPERT_SKILL_PLANNER_EXTENSION_SHA256 &&
              n.structural_planner_extension_sha256 !==
                EXPERT_SKILL_PLANNER_EXTENSION_SHA256) {
            staleIdentityReasons.push('Planner Extension digest changed');
          }
          if (EXPERT_SKILL_BUNDLE_SHA256 &&
              n.structural_skill_bundle_sha256 !== EXPERT_SKILL_BUNDLE_SHA256) {
            staleIdentityReasons.push('Skill bundle digest changed');
          }
          if (!n.structural_candidate_head || n.structural_candidate_head !== n.head) {
            staleIdentityReasons.push('candidate HEAD changed or was not recorded');
          }
          if (!n.structural_candidate_tree_digest) {
            staleIdentityReasons.push('candidate tree digest was not recorded');
          }
        }
        const staleContract = staleIdentityReasons.length > 0;
        if (staleContract) {
          log(`MEGA candidate ${n.id}: invalidating structural evidence identity: ` +
            `${staleIdentityReasons.join('; ')}.`);
        }
        megaCandidateRegistry = upsertMegaCandidate(megaCandidateRegistry, {
          ...n,
          ...(staleContract ? {
            structural_verified: false,
            structural_report: '',
            structural_skill_id: '',
            structural_candidate_head: '',
            structural_candidate_tree_digest: '',
            structural_skill_bundle_sha256: '',
            structural_planner_extension_sha256: '',
            structural_contract_revision: '',
            structural_contract_sha256: '',
            contract_failures: [{
              id: 'structural_evidence_identity_changed',
              category: 'plan',
              severity: 'required',
              messages: staleIdentityReasons,
            }],
            next_blocker: `re-run Expert Skill bundle ${EXPERT_SKILL_REVISION} contract on exact ` +
              `HEAD ${n.head || n.working_head || '(missing)'} before device verification`,
          } : {}),
          // State may carry evidence and a head, never authority to redirect the lane outside its root.
          tree: megaLaneTree(n.id),
        });
      }
    }
    // A resume whose rows were compacted past validity (`source` dropped) silently ran fresh.
    if (megaRestoreRejected.length) {
      throw new Error(`MEGA RESUME REFUSED: setup returned invalid candidate_registry rows ` +
        `(${megaRestoreRejected.join(', ')}); rows must keep id/source/base_id/tree/head/status.`);
    }
    if (Number.isFinite(Number(ps.state_sequence))) {
      megaStateSequenceBase = Math.max(0, Number(ps.state_sequence));
    }
    // Calibration is intentionally NOT restored as authority. Every wave runs/recover-checks its own
    // control; prior calibration remains in STATE for audit only and cannot license fresh scores.
    log(`RESUMED from STATE_DIR: cumulative=${cumulative.toFixed(3)}x, ${history.insights.length} insights, ${history.ledger.length} ledger entries carried forward.`);
    if (shelf.length) log(`  shelf: ${shelf.length} verified non-winner(s) carried forward, ` +
      `${shelf.filter((e) => e.footprint === 'known').length} with a known file footprint.`);
    log(`  mega candidates: ${megaCandidateRegistry.length}, calibration=${megaMeasurementCalibration.ready ? 'ready' : 'pending'}.`);
  }
}

// One candidate turn of the Optimize loop. {stop} ends the loop, {exit} ends the run.
async function megaRound(remaining) {
  const elapsedS = (megaNowMs() - WORKFLOW_STARTED_MS) / 1000;
  if (elapsedS >= MEGA_TIME_BUDGET_S - MEGA_FINAL_RESERVE_S) {
    stopReason = `${MEGA_PROFILE} dispatch deadline reached after ${Math.round(elapsedS / 60)} minutes; ` +
      `preserving ${Math.round(MEGA_FINAL_RESERVE_S / 60)} minutes for finalist validation.`;
    return { stop: true };
  }
  const turn = await runMegaCandidateTurn(round, remaining);
  if (turn.refused && !turn.stop) {
    const count = megaRefusal ? megaRefusal.count + 1 : 1;
    megaRefusal = { reason: turn.reason, count };
    dispatched += 1;
    megaStall += 1;
    log(`Mega round ${round}: ${turn.reason} (refusal ${count}/3; re-planning).`);
    if (count >= 3) {
      stopReason = `${count} consecutive refused directions; last: ${turn.reason}`;
      return { stop: true };
    }
    return {};
  }
  megaRefusal = null;
  if (turn.stop) {
    if (MEGA_ROUTE_ONLY) {
      const d = turn.direction || {};
      return { exit: {
        mode: MODE, mega_route_only: true,
        route_pass: String(turn.reason || '').startsWith('route-only PASS:'),
        eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
        candidate_id: d.candidate_id || '', roadmap_rung: d.roadmap_rung || d.id || '',
        launch_count: topologyLaunchCount(d.target_topology),
        target_launch_count: Number(analysis && analysis.mega_plan_ir &&
          analysis.mega_plan_ir.target && analysis.mega_plan_ir.target.launch_count),
        validation_status: 'route_only', stop_reason: turn.reason,
      } };
    }
    stopReason = turn.reason;
    return { stop: true };
  }
  dispatched += 1;
  log(`Mega round ${round}: candidate=${turn.record.id}[${turn.record.source}:${turn.record.status}], ` +
    `score=${Number.isFinite(turn.record.absolute_score) ? turn.record.absolute_score.toFixed(4) + 'x' : 'pending'}, ` +
    `selected=${turn.selected ? turn.selected.id : 'none'}, calibration=` +
    `${megaMeasurementCalibration.ready ? 'ready' : 'pending'}, budget=${dispatched}/${BUDGET}.`);
  if (MEGA_STRUCTURAL_ONLY && turn.record.structural_verified) {
    stopReason = `structural-only acceptance reached by ${turn.record.id}: source checkpoint and ` +
      `MegaPlanIR consistency passed; hardware/accuracy/liveness/performance remain unverified.`;
    return { stop: true };
  }
  const searchAttempts = megaCandidateRegistry
    .filter((c) => c.source === 'search' || c.source === 'integrated')
    .reduce((sum, c) => sum + Number(c.attempts || 0), 0);
  const expertTargetLaunches = Number(
    analysis && analysis.mega_plan_ir && analysis.mega_plan_ir.target &&
      analysis.mega_plan_ir.target.launch_count
  );
  const expertTargetConfigured = USE_EXPERT_SKILLS &&
    Number.isFinite(EXPERT_SKILL_RECORDED_LOW) &&
    Number.isFinite(expertTargetLaunches);
  const expertTargetReached = !expertTargetConfigured || (
    turn.selected &&
    Number(turn.selected.absolute_score) >= Number(EXPERT_SKILL_RECORDED_LOW) &&
    Number(turn.selected.launches) === expertTargetLaunches
  );
  if (MEGA_STOP_ON_DELIVERABLE && turn.selected &&
      Number(turn.selected.absolute_score) > 1.0 &&
      expertTargetReached &&
      searchAttempts >= MEGA_MIN_SEARCH_ATTEMPTS) {
    stopReason = `production early-stop after ${round} round(s): candidate ${turn.selected.id} ` +
      `has calibrated ${Number(turn.selected.absolute_score).toFixed(4)}x speedup over frozen ` +
      `baseline${expertTargetConfigured ? `, reached Expert Skill target ` +
        `${Number(EXPERT_SKILL_RECORDED_LOW).toFixed(4)}x at ${expertTargetLaunches} launches` : ''}, ` +
      `and ${searchAttempts} search attempt(s) have run; reserve the remaining ` +
      `${BUDGET - dispatched} direction(s) for finalist validation instead of exhaustive search.`;
    return { stop: true };
  }
  if (REQUIRE_HW_ACTIVATION && noHardware >= MAX_NO_HARDWARE) {
    stopReason = `${noHardware} consecutive portfolio attempts did not execute the candidate path ` +
      `on hardware; stopping before another unmeasured direction. Lane WIP remains resumable.`;
    log(`Mega hardware stop: ${stopReason}`);
    return { stop: true };
  }
  if (MEGA_STALL_LIMIT && megaStall >= 2 * MEGA_STALL_LIMIT) {
    stopReason = `${megaStall} consecutive rounds produced no new runtime evidence (independent ` +
      'Verify, better score, or a changed on-card blocker); lane WIP remains resumable.';
    log(`Mega stall stop: ${stopReason}`);
    return { stop: true };
  }
  return {};
}

async function megaAfterLoop() {

  // Record which stop fired, in loop order: the budget counts directions, not GPU leases, so the
  // artifacts alone cannot tell "budget spent" from "hardware left".
  if (!stopReason) {
    const left = BUDGET - dispatched;
    if (dispatched >= BUDGET) {
      stopReason = `the direction budget was exhausted (${dispatched}/${BUDGET} charged over ${round} round(s)). ` +
        'The budget counts directions, not GPU leases; a round of directions that never reached the ' +
        'hardware costs the same as a round that did.';
    } else if (!WORKING_KERNEL && noEvidence >= MAX_NO_IMPROVE) {
      stopReason = `${noEvidence} consecutive round(s) produced no admissible measurement, which is the ` +
        `max_no_improve cap (${MAX_NO_IMPROVE}). ${left} of ${BUDGET} budget unit(s) were left unspent. ` +
        'This is an instrument failure, not an exhausted search: the next wave should fix the lease ' +
        'or the activation path before it plans another direction.';
    } else if (!WORKING_KERNEL && noImprove >= MAX_NO_IMPROVE) {
      stopReason = `${noImprove} round(s) in a row measured a candidate and none beat the current best by ` +
        `more than ${(MIN_IMPROVE * 100).toFixed(0)}%, which is the max_no_improve cap (${MAX_NO_IMPROVE}). ` +
        `${left} of ${BUDGET} budget unit(s) were left unspent.`;
    } else {
      stopReason = `the loop ended after ${round} round(s) with ${left} of ${BUDGET} budget unit(s) unspent ` +
        'and no stop condition recorded.';
    }
  }
  log(`Rounds ended: ${stopReason}`);

  if (megaUnsafeTimeout) {
    log(`MEGA SAFE STOP: ${megaUnsafeTimeout.label || 'agent'} may still be running after the ` +
      `orchestration timeout. No recovery, fallback, finalist GPU work, report agent or validation ` +
      `agent will be started in this invocation.`);
    return { exit: {
      mode: MODE,
      mega_deliverable: false,
      mega_selected_candidate: '',
      mega_selected_source: '',
      mega_candidate_registry: megaCandidateRegistry,
      mega_note: 'agent timeout: candidate lane WIP is preserved, but no overlapping work was launched',
      eval_dir: EVAL_DIR,
      kernel_name: KERNEL_NAME,
      final_speedup: 0,
      final_geomean: 0,
      validation_status: 'agent_timeout',
      rounds: round,
      budget_used: dispatched,
      budget_total: BUDGET,
      stop_reason: stopReason,
      report_path: '',
      final_patch: '',
    } };
  }
  FINAL_WORKSPACE = CANONICAL;
  if (!megaMeasurementCalibration.ready) await recoverMegaCalibration('before-final-selection');
  const provisional = selectMegaCandidate(megaCandidateRegistry, MEGA_TIE_NOISE_PCT);
  const orderedEligible = provisional.selected
    ? [provisional.selected, ...provisional.eligible.filter((c) => c.id !== provisional.selected.id)]
    : provisional.eligible;
  ranked = orderedEligible.slice(0,
    MEGA_PRODUCTION ? MEGA_FINAL_FALLBACK_K : MEGA_FINAL_TOP_K);
  const finalistOrder = ranked;
  let selectionVerdict = megaFinalSelectionVerdict(null, [], {
    selectedWorkspace: `${EVAL_DIR}/mega_selected`,
  });
  if (MEGA_PRODUCTION) {
    for (let i = 0; i < finalistOrder.length; i++) {
      const batch = [finalistOrder[i]];
      const attempt = await runFinalistBatch(batch, `director:mega-finalist-${i + 1}`);
      if (attempt && attempt.__agent_timed_out) {
        megaUnsafeTimeout = attempt;
        selectionVerdict = megaFinalSelectionVerdict(null, batch, {
          selectedWorkspace: `${EVAL_DIR}/mega_selected`,
        });
        finalists = batch;
        break;
      }
      const verdict = megaFinalSelectionVerdict(attempt, batch, {
        selectedWorkspace: `${EVAL_DIR}/mega_selected`,
        launchTarget: LAUNCH_TARGET, worldSize: GPU_RESOURCE.gpusPerJob,
        allowPartialFusion: ALLOW_PARTIAL_FUSION,
        baselineLaunches: optionalNumber(A.baseline_launch_count),
        frozenBaseline: BASELINE_PER_CASE,
        targetGuards: TARGET_GUARDS, regressionGuards: REGRESSION_GUARDS,
        requiredReplays: REQUIRED_REPLAYS, requiredPairs: REQUIRED_PAIRS,
        requiredPairsByGuard: REQUIRED_PAIRS_BY_GUARD,
        accuracyMetric: ACCURACY_METRIC, accuracyThreshold: ACCURACY_THRESHOLD,
        requireOverlap: REQUIRE_OVERLAP, requireAttribution: REQUIRE_ATTRIBUTION,
        tieNoisePct: MEGA_TIE_NOISE_PCT,
      });
      finalists = batch;
      selectionVerdict = verdict;
      if (verdict.pass) {
        const identity = await agentT(
          roleAgent('director', 'check_mega_identity',
            'Independently compare the selected materialized tree with the exact source candidate head; do not run GPU work.', {
              EVAL_DIR, SKILL_DIR: WORKFLOW_DIR,
              SELECTED_CANDIDATE_ID: finalistOrder[i].id,
              SELECTED_SOURCE_TREE: finalistOrder[i].tree,
              SELECTED_SOURCE_HEAD: finalistOrder[i].head,
              SELECTED_WORKSPACE: `${EVAL_DIR}/mega_selected`,
              SELECTED_MATERIALIZED_HEAD: attempt.selected_head,
            }),
          { phase: 'Validate', label: `director:mega-identity-${i + 1}`,
            schema: MEGA_IDENTITY_SCHEMA, timeout_ms: 60000,
            timeout_marker: true, max_retries: 1 });
        if (identity && identity.__agent_timed_out) {
          megaUnsafeTimeout = identity;
          selectionVerdict = { pass: false, reasons: ['independent materialization identity timed out'] };
          break;
        }
        if (!(identity && identity.passed === true &&
            identity.selected_candidate_id === finalistOrder[i].id &&
            identity.source_head === finalistOrder[i].head &&
            identity.materialized_head === attempt.selected_head)) {
          log(`MEGA production finalist ${finalistOrder[i].id} materialization identity failed. ` +
            `${i + 1 < finalistOrder.length ? 'Trying the next scored fallback.' : 'No fallback remains.'}`);
          selectionVerdict = { pass: false, reasons: ['independent materialization identity failed'] };
          continue;
        }
        megaSelection = attempt;
        break;
      }
      log(`MEGA production finalist ${finalistOrder[i].id} failed: ${verdict.reasons.join('; ')}. ` +
        `${i + 1 < finalistOrder.length ? 'Trying the next scored fallback.' : 'No fallback remains.'}`);
    }
  } else {
    finalists = finalistOrder;
    megaSelection = await runFinalistBatch(finalists, 'director:mega-finalist-selection');
    if (megaSelection && megaSelection.__agent_timed_out) {
      megaUnsafeTimeout = megaSelection;
      megaSelection = null;
    }
    selectionVerdict = megaFinalSelectionVerdict(megaSelection, finalists, {
      selectedWorkspace: `${EVAL_DIR}/mega_selected`,
      launchTarget: LAUNCH_TARGET, worldSize: GPU_RESOURCE.gpusPerJob,
      allowPartialFusion: ALLOW_PARTIAL_FUSION,
      baselineLaunches: optionalNumber(A.baseline_launch_count),
      frozenBaseline: BASELINE_PER_CASE,
      targetGuards: TARGET_GUARDS, regressionGuards: REGRESSION_GUARDS,
      requiredReplays: REQUIRED_REPLAYS, requiredPairs: REQUIRED_PAIRS,
      requiredPairsByGuard: REQUIRED_PAIRS_BY_GUARD,
      accuracyMetric: ACCURACY_METRIC, accuracyThreshold: ACCURACY_THRESHOLD,
      requireOverlap: REQUIRE_OVERLAP, requireAttribution: REQUIRE_ATTRIBUTION,
      tieNoisePct: MEGA_TIE_NOISE_PCT,
    });
  }
  if (selectionVerdict.pass) {
    megaSelectionContractPassed = true;
    FINAL_WORKSPACE = megaSelection.selected_tree;
    const selectedRecord = selectionVerdict.source;
    const finalRow = selectionVerdict.row;
    const finalTargetReadout = megaTargetsReadout(finalRow.paired_readings, TARGET_GUARDS);
    selectedRecord.status = 'finalist';
    selectedRecord.absolute_score = Number(finalTargetReadout.score);
    selectedRecord.per_case = Array.isArray(finalRow.per_case) ? finalRow.per_case : [];
    selectedRecord.claim_complete = true;
    selectedRecord.attempt_id = String(megaSelection.attempt_id || '');
    selectedRecord.evidence_manifest = String(megaSelection.evidence_manifest || '');
    selectedRecord.paired_readings = Array.isArray(finalRow.paired_readings)
      ? finalRow.paired_readings : [];
    selectedRecord.null_arm_pct = Number(finalRow.null_arm_pct);
    selectedRecord.reps = finalTargetReadout.count;
    selectedRecord.artifact_hash_base = String(finalRow.artifact_hash_base || '');
    selectedRecord.artifact_hash_candidate = String(finalRow.artifact_hash_candidate || '');
    selectedRecord.correctness_pass = true;
    selectedRecord.activation_pass = true;
    selectedRecord.head_pass = true;
    selectedRecord.launch_pass = true;
    selectedRecord.liveness_pass = true;
    selectedRecord.graph_pass = true;
    selectedRecord.guards_pass = true;
    selectedRecord.artifact_distinct = true;
    selectedRecord.measurement_pass = true;
    selectedRecord.final_tree = megaSelection.selected_tree;
    selectedRecord.final_head = String(megaSelection.selected_head || '');
    selectedRecord.final_attempt_id = String(megaSelection.attempt_id || '');
    selectedRecord.final_evidence_manifest = String(megaSelection.evidence_manifest || '');
    megaCandidateRegistry = upsertMegaCandidate(megaCandidateRegistry, selectedRecord);
    cumulative = Number(megaSelection.selected_score) || cumulative;
    finalWinner = {
      id: megaSelection.selected_candidate_id,
      source: megaSelection.selected_source,
      geomean: cumulative,
      tree: FINAL_WORKSPACE,
      source_tree: selectedRecord.tree,
      source_head: selectedRecord.head,
      materialized_head: megaSelection.selected_head,
      patch: megaSelection.selected_patch || '',
      provenance: selectedRecord ? selectedRecord.provenance : megaSelection.selected_source,
      topology_sig: selectedRecord ? selectedRecord.topology_sig : '',
      activation: selectedRecord ? selectedRecord.activation : {},
      per_case: selectedRecord.per_case,
    };
    log(`MEGA FINALIST SELECTED: ${finalWinner.id}[${finalWinner.source}] ${cumulative.toFixed(4)}x.`);
  } else {
    const why = !megaMeasurementCalibration.ready
      ? 'measurement calibration is incomplete'
      : (!finalistOrder.length
        ? 'no workflow-authored candidate passed the score-tier contract'
        : `final selection contract failed: ${selectionVerdict.reasons.join('; ')}`);
    ACCEPTANCE_CAVEATS.push(`Mega finalist selection unavailable: ${why}.`);
    log(`MEGA FINALIST SELECTION INCOMPLETE: ${why}; no performance result will be presented as a deliverable.`);
    finalWinner = null;
  }

  if (megaUnsafeTimeout) {
    log(`MEGA SAFE STOP: finalist agent ${megaUnsafeTimeout.label || ''} timed out; no fallback or ` +
      `second validation starts while that agent may still own resources.`);
    return { exit: {
      mode: MODE, mega_deliverable: false, mega_selected_candidate: '',
      mega_selected_source: '', mega_candidate_registry: megaCandidateRegistry,
      mega_note: 'finalist timeout: no overlapping fallback was launched',
      eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME, final_speedup: 0, final_geomean: 0,
      validation_status: 'agent_timeout', rounds: round, budget_used: dispatched,
      budget_total: BUDGET, stop_reason: 'finalist agent timeout', report_path: '', final_patch: '',
    } };
  }
}

async function megaReport() {
  return await agentT(
    roleAgent('tech_lead', 'report', 'Write the final report and the cumulative final patch.', {
      EVAL_DIR, WORKSPACE: FINAL_WORKSPACE, SKILL_DIR: WORKFLOW_DIR,
      HISTORY: history, FINAL_WINNER: finalWinner, BASELINE_PER_CASE,
      ...({
        MEGA_CANDIDATE_REGISTRY: megaCandidateRegistry,
        MEGA_SELECTION: megaSelection || {},
      }),
      STRICT_AUTONOMY, PROMOTION_METRIC, TARGET_GUARDS, REGRESSION_GUARDS,
      LAUNCH_TARGET, REQUIRE_OVERLAP, REQUIRE_ATTRIBUTION, REQUIRE_ARTIFACT_DISTINCT,
      REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
      ACCURACY_METRIC, ACCURACY_THRESHOLD, AUTONOMY_ACCEPTANCE_REACHED,
      ...(ACCEPTANCE_CAVEATS.length ? { ACCEPTANCE_CAVEATS } : {}),
      // Why the loop ended, for the next wave's starting conditions.
      STOP_REASON: stopReason,
      BASELINE_GEOMEAN_MS, CUMULATIVE_SPEEDUP: cumulative,
      PROFILE_SUMMARY: profileSummary,
      // The report's provenance paragraph must cite this, not recollection. UNRECORDED means the
      // report may not assert anything about what is or is not in the baseline.
      PRIOR_ART_SWEEP: PRIOR_ART_RECORDED ? JSON.stringify(PRIOR_ART) : 'UNRECORDED',
      // A control that passed but read HIGH is a standing caveat on every number below it. It is
      // supplied here so the Measurement-confidence section states it instead of quietly dropping it.
      ...(PC_OVERSHOOT ? { POSITIVE_CONTROL_OVERSHOOT: PC_OVERSHOOT } : {}),
      ...(PC_UNDERSHOOT ? { POSITIVE_CONTROL_UNDERSHOOT: PC_UNDERSHOOT } : {}),
      // Shelf activity this wave (offered vs composed), reported once anything was shelved; K is sized from it.
      ...(SHELF_STATS.rounds_offered || SHELF_STATS.shelved ? { SHELF_ACTIVITY: SHELF_STATS } : {}),
      ...(OVERLAP_CAVEATS.length ? { OVERLAP_CAVEATS } : {}),
      // Launch-shape caveats, present only when some candidate missed the target or did not report;
      // TWO_LAUNCH_REACHED says whether any verified candidate hit it.
      ...(LAUNCH_CAVEATS.length ? { LAUNCH_CAVEATS, TWO_LAUNCH_REACHED } : {}),
      // Guard-scope caveats: an under-sampled bimodal target route, or a win read only off a skew rail.
      // Present only when something was off, so a wave measured cleanly on target is unchanged.
      ...(GUARD_CAVEATS.length ? { GUARD_CAVEATS } : {}),
      ...(ATTRIBUTION_CAVEATS.length ? { ATTRIBUTION_CAVEATS } : {}),
      // A half-built fusion: committed prerequisites whose chain never closed, so the cumulative number is
      // below the start for a reason the number does not show.
      ...(CHAIN_CAVEATS.length ? { CHAIN_CAVEATS } : {}),
      ...(CHAIN_DEBT.length ? {
        CHAIN_DEBT_FINAL: chainDebtReport(CHAIN_DEBT, round, LADDER_MEASURED).open,
        CHAIN_BASELINE,
        CHAIN_NOTE:
          'Enabling steps are committed on FUNCTION (builds, path taken, correct, no deadlock), not ' +
          'on speed, because the producer half of a fusion cannot be faster on its own. Any rung ' +
          'still listed in CHAIN_DEBT_FINAL is a fusion that was started and not closed: state the ' +
          'cost it left in the tree, and state that the speedup for that chain is NOT YET MEASURED ' +
          'rather than reporting the degraded number as the result. CHAIN_BASELINE is what the ' +
          'terminal step must be compared against — the canonical tree already contains the ' +
          'overhead, so measuring against the canonical would credit the chain with removing its own ' +
          'cost.',
      } : {}),
      // Only on a non-default objective: under working_kernel the wave's timings are withdrawn and the
      // deliverable is whether an artifact ran.
      ...(WORKING_KERNEL ? { OBJECTIVE, OBJECTIVE_NOTE:
        'This wave was run for a WORKING FUSED ARTIFACT, not for speed. One direction per round, the ' +
        'commit gate was "does it run", and the no-improve stop was disabled because a debug round is ' +
        'non-improving by construction. Do NOT read a speedup out of this wave' +
        (PC_RAN ? '.' : ' — it ran no positive control, so every timing reading in it is void.') } : {}),
      ...(OBJECTIVE_CAVEATS.length ? { OBJECTIVE_CAVEATS } : {}),
      // Knowledge deltas in the report, split into generalizing (card edit owed) and not, each with its
      // evidence.
      ...(KNOWLEDGE_DELTA.length ? { KNOWLEDGE_DELTA, KNOWLEDGE_DELTA_NOTE:
        'Facts this wave established about the HARDWARE/TOOLCHAIN, not about this kernel. Give them ' +
        'their own section titled "Knowledge deltas — merge into knowledge/". List the generalizing ' +
        'ones first with their target card and their evidence verbatim; list the operator-specific ' +
        'ones under a separate heading and say plainly that they are not card material. This workflow ' +
        'does not edit knowledge/ itself.' } : {}),
      ...(TASK_GRAPH_CAVEAT ? { TASK_GRAPH_CAVEAT } : {}),
      ...(PIPE_TABLE_CAVEAT ? { PIPE_TABLE_CAVEAT } : {}),
      // Recomputed against the final dispatch record: which rungs the wave never reached.
      ...(LADDER.length ? { ROADMAP_LADDER_FINAL: JSON.stringify(
        roadmapLadderGate(LADDER, [], LADDER_MEASURED)).slice(0, 3000) } : {}),
      ...(REQUIRE_TASK_GRAPH && analysis && analysis.task_graph
        ? { TASK_GRAPH: JSON.stringify(analysis.task_graph).slice(0, 4000) } : {}),
      ...(REQUIRE_TASK_GRAPH && analysis && analysis.resource_timeline
        ? { RESOURCE_TIMELINE: JSON.stringify(analysis.resource_timeline).slice(0, 3000) } : {}),
    }),
    { phase: 'Report', label: 'tech_lead:report', schema: REPORT_SCHEMA,
      ...(MEGA_PRODUCTION
        ? { timeout_ms: 300000, timeout_marker: true, max_retries: 1 } : {}) });
}

async function megaAfterReport() {
  reportReady = !!(report && !report.__agent_timed_out &&
    String(report.report_path || '') && String(report.final_patch || ''));
  if (report && report.__agent_timed_out) megaUnsafeTimeout = report;
  if (megaUnsafeTimeout) {
    return { exit: {
      mode: MODE, mega_deliverable: false, mega_selected_candidate: '',
      mega_selected_source: '', mega_candidate_registry: megaCandidateRegistry,
      mega_note: 'report timeout: no identity validation or state writer was started afterward',
      eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME, final_speedup: 0, final_geomean: 0,
      validation_status: 'agent_timeout', rounds: round, budget_used: dispatched,
      budget_total: BUDGET, stop_reason: 'report agent timeout', report_path: '', final_patch: '',
    } };
  }
}

async function megaValidate() {
  const shouldRunFinalValidation = !(MEGA_PRODUCTION) ||
    (reportReady && !!finalWinner && megaSelectionContractPassed);
  return !shouldRunFinalValidation ? null : await agentT(
    roleAgent('director', 'validate', 'Independently validate the final patch vs the TRUE baseline.', {
      KERNEL_PATH_ORIG, EVAL_DIR, WORKSPACE: FINAL_WORKSPACE, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_SINGLE,
      APPLY_TO_ORIGINAL, COMMANDMENT,
      FINAL_PATCH: reportReady ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
      TECH_LEAD_REPORTED_GEOMEAN: report ? report.final_speedup_geomean : cumulative,
      STRICT_AUTONOMY, PROMOTION_METRIC, TARGET_GUARDS, REGRESSION_GUARDS,
      LAUNCH_TARGET, REQUIRE_OVERLAP, REQUIRE_ATTRIBUTION, REQUIRE_ARTIFACT_DISTINCT,
      REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
      ACCURACY_METRIC, ACCURACY_THRESHOLD, AUTONOMY_ACCEPTANCE_REACHED,
      WORKFLOW_REVISION_START,
      ...(finalWinner ? {
        VALIDATION_TIER: MEGA_PRODUCTION ? 'identity_only' : 'full',
        MEGA_SELECTION_EVIDENCE: megaSelection,
        SELECTED_SOURCE_TREE: finalWinner.source_tree,
        SELECTED_SOURCE_HEAD: finalWinner.source_head,
        SELECTED_MATERIALIZED_HEAD: finalWinner.materialized_head,
      } : {}),
      ...(STRICT_AUTONOMY && KNOWN_REFERENCE_HASHES.length
        ? { KNOWN_REFERENCE_HASHES } : {}),
      ...(HAS_WORKLOAD && report && report.final_speedup_weighted != null
          ? { TECH_LEAD_REPORTED_WEIGHTED: report.final_speedup_weighted } : {}),
      BASELINE_TIMING: BASELINE_PER_CASE,
    }),
    { phase: 'Validate', label: 'director:validate', schema: MEGA_VALIDATE_SCHEMA,
      ...(MEGA_PRODUCTION
        ? { timeout_ms: 300000, timeout_marker: true, max_retries: 1 } : {}) });
}

async function megaFinish() {
  if (validation && validation.__agent_timed_out) {
    return {
      mode: MODE, mega_deliverable: false, mega_selected_candidate: '',
      mega_selected_source: '', mega_candidate_registry: megaCandidateRegistry,
      mega_note: 'identity validation timeout: no state writer was started afterward',
      eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME, final_speedup: 0, final_geomean: 0,
      validation_status: 'agent_timeout', rounds: round, budget_used: dispatched,
      budget_total: BUDGET, stop_reason: 'identity validation agent timeout',
      report_path: reportReady ? report.report_path : '', final_patch: reportReady ? report.final_patch : '',
    };
  }
  const aggregatePrimary = HAS_WORKLOAD && Number.isFinite(finalWeighted) ? finalWeighted : finalGeomean;
  const finalPrimary = (TARGET_GUARDS.length || PROMOTION_METRIC === 'changed_kernel')
    ? promotionScore(validation || {}, PROMOTION_METRIC, TARGET_GUARDS,
      REGRESSION_GUARDS, aggregatePrimary)
    : aggregatePrimary;
  const finalGuards = guardContract(validation || {}, TARGET_GUARDS, REGRESSION_GUARDS, aggregatePrimary);
  const directorStatus = validation ? validation.validation_status : 'unknown';
  const finalAttribution = (validation && validation.attribution) || {};
  const scopedGuardFailure = finalMetricFailure(
    PROMOTION_METRIC, TARGET_GUARDS, finalPrimary, finalGuards, finalAttribution);
  const strictProofFailure = STRICT_AUTONOMY &&
    (!AUTONOMY_ACCEPTANCE_REACHED || !(validation && validation.autonomy_acceptance_confirmed === true) ||
     !(validation && validation.workflow_unchanged === true));
  // Correctness/apply failures outrank the more specific autonomy label; never soften `flagged` into
  // `autonomy_incomplete`. Scoped regression guards are vetoes in every mode, not only strict mode.
  const finalValidationStatus = finalStatusVerdict(directorStatus, scopedGuardFailure, strictProofFailure);
  // Surface leaked lease jobs in the completion line: they keep running after the report and contend
  // with the measurements just accepted.
  if (validation && validation.orphan_leases_swept > 0) {
    log(`WARNING: ${validation.orphan_leases_swept} orphaned GPU lease(s) were still alive at closeout ` +
        `and have been killed. Some direction backgrounded a lease job — its measurement is NOT in ` +
        `the report, and it may have contended with the validated numbers.`);
  }
  const correctnessClean = directorStatus === 'accepted' &&
    reportReady &&
    String(validation && validation.correctness || '').toLowerCase().startsWith('pass') &&
    validation && validation.materialization_matches === true &&
    validation.final_patch_verified === true &&
    (APPLY_TO_ORIGINAL !== 'true' ||
      String(validation.applied_to_original || '').toLowerCase() === 'true') &&
    String(validation.selected_source_head || '') === String(finalWinner && finalWinner.source_head || '') &&
    String(validation.selected_materialized_head || '') ===
      String(finalWinner && finalWinner.materialized_head || '') &&
    !strictProofFailure &&
    finalGuards && finalGuards.regression_pass !== false;
  mega_deliverable = !!(megaSelection && megaSelection.claim_complete === true &&
    megaSelectionContractPassed && finalWinner && correctnessClean && Number.isFinite(finalPrimary) &&
    finalPrimary > 1.0 && !scopedGuardFailure);
  mega_note = mega_deliverable
    ? `mega: selected ${finalWinner.id}[${finalWinner.source}] at ${finalPrimary.toFixed(3)}x vs ` +
      `the frozen baseline; all finalist gates passed.`
    : `mega: no fully verified finalist cleared the common correctness/guard/performance contract; ` +
      `workflow-authored candidate WIP remains recorded, but no result is presented as shipped.`;
  if (!mega_deliverable && finalWinner) {
    const rejectedFinal = megaCandidateById(finalWinner.id);
    if (rejectedFinal) {
      rejectedFinal.status = 'runnable';
      rejectedFinal.next_blocker =
        `independent final validation failed (${finalValidationStatus}); re-establish materialization, ` +
        `correctness and full guard evidence before finalist selection`;
      rejectedFinal.notes = `${rejectedFinal.notes || ''} FINAL VALIDATION: ${mega_note}`.trim();
      megaCandidateRegistry = upsertMegaCandidate(megaCandidateRegistry, rejectedFinal);
    }
    const fallback = selectMegaCandidate(megaCandidateRegistry, MEGA_TIE_NOISE_PCT).selected;
    cumulative = fallback ? fallback.absolute_score : 1.0;
    finalWinner = null;
  }
  const finalStateSaved = await persistMegaCandidateState(round, true);
  if (!finalStateSaved) {
    mega_deliverable = false;
    finalWinner = null;
    mega_note = 'mega: final candidate state could not be persisted; no result is presented as shipped.';
  }
  log(mega_note);

  log(`COMPLETE. ${KERNEL_NAME}: verified ${TARGET_GUARDS.length ? `target-guard ${PROMOTION_METRIC}` :
      (HAS_WORKLOAD ? 'time-weighted' : 'geomean')} ${finalPrimary ? finalPrimary.toFixed(2) : '?'}x` +
      `${HAS_WORKLOAD && Number.isFinite(finalGeomean) ? ` (unweighted geomean ${finalGeomean.toFixed(2)}x)` : ''}` +
      ` (status ${finalValidationStatus}${`, mega_deliverable=${mega_deliverable}, selected=${finalWinner ? finalWinner.id : 'none'}`}). Results in ${EVAL_DIR}`);

  return {
    mode: MODE,
    ...({
      mega_deliverable,
      mega_selected_candidate: finalWinner ? finalWinner.id : '',
      mega_selected_source: finalWinner ? finalWinner.source : '',
      mega_candidate_registry: megaCandidateRegistry,
      mega_note,
    }),
    target_language: undefined,
    authored: undefined,
    eval_dir: EVAL_DIR,
    kernel_name: KERNEL_NAME,
    workload_aligned: HAS_WORKLOAD,
    final_speedup: finalPrimary,                 // PRIMARY metric (weighted when workload-aligned)
    final_weighted: finalWeighted,
    final_geomean: finalGeomean,
    final_arithmetic: validation ? validation.director_verified_speedup_arithmetic : null,
    tech_lead_reported_geomean: report ? report.final_speedup_geomean : cumulative,
    validation_status: finalValidationStatus,
    autonomy_acceptance_reached: AUTONOMY_ACCEPTANCE_REACHED,
    autonomy_acceptance_confirmed: !!(validation && validation.autonomy_acceptance_confirmed === true &&
      validation.workflow_unchanged === true && directorStatus !== 'flagged' &&
      !scopedGuardFailure && !strictProofFailure),
    acceptance_caveats: ACCEPTANCE_CAVEATS,
    rounds: report ? report.rounds : round,
    budget_used: dispatched,
    budget_total: BUDGET,
    stop_reason: stopReason,
    report_path: report ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`,
    final_patch: report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
  };
}
// ===========================================================================
// end of mode=mega backend
// ===========================================================================

// ===========================================================================
// PHASE: Setup
// ===========================================================================
phase('Setup');
const setup = await agentT(
  MEGA ? roleAgent('director', 'setup', 'Build the isolated evaluation environment.', {
    KERNEL_PATH_ORIG, EXP_ROOT, EVAL_DIR_OVERRIDE, KERNEL_NAME_HINT, TASK, SKILL_DIR: WORKFLOW_DIR,
    MODE, TARGET_LANGUAGE, OP_SPEC,
    ...(STRICT_AUTONOMY ? { STRICT_AUTONOMY: '1' } : {}),
    ...(STATE_DIR ? { STATE_DIR, MEGA_RESUME_STATE } : {}),
  }) : roleAgent('director', 'setup', 'Build the isolated evaluation environment.', {
    KERNEL_PATH_ORIG, EXP_ROOT, EVAL_DIR_OVERRIDE, KERNEL_NAME_HINT, TASK,
    WORKFLOW_DIR, SKILL_DIR: WORKFLOW_DIR,
    MODE, TARGET_LANGUAGE, OP_SPEC, EXPECTED_GFX, EXPECTED_TARGET,
    EXPECTED_DEVICE_NAME, EXPECTED_PHYSICAL_CU_COUNT,
    ...(STATE_DIR ? { STATE_DIR } : {}),
  }),
  { phase: 'Setup', label: 'director:setup', schema: MEGA ? MEGA_SETUP_SCHEMA : SETUP_SCHEMA,
    // 5min cut the mega director off while reproducing a grown prior_state; 15min cut an 18-row resume.
    ...(MEGA && MEGA_PRODUCTION ? { timeout_ms: 1800000, max_retries: 1 } : {}) });
if (!setup || !setup.eval_dir) throw new Error('Setup failed: director did not return an eval_dir');
const EVAL_DIR = setup.eval_dir;
const CANONICAL = setup.workspace;       // canonical current-best workspace (advances each round)
const GFX = String(setup.device_gfx || '').trim().toLowerCase();
const DEVICE_TARGET = String(setup.device_target || '').trim().toLowerCase();
const SETUP_DEVICE_NAME = String(setup.device_name || '').trim();
const SETUP_PHYSICAL_CU_COUNT = Number(setup.physical_cu_count);
if (!MEGA && !GFX) {
  throw new Error('Setup failed: director did not return device_gfx (required; refuse CDNA knowledge without an ISA)');
}
if (!MEGA && !DEVICE_TARGET) {
  throw new Error('Setup failed: director did not return device_target (r9700 | r9600d | unknown)');
}
if (!MEGA && (!Number.isFinite(SETUP_PHYSICAL_CU_COUNT) || SETUP_PHYSICAL_CU_COUNT <= 0)) {
  throw new Error('Setup failed: director did not return a positive physical_cu_count');
}
if (!MEGA && GFX === 'gfx1200') {
  throw new Error('Detected gfx1200, which is not validated by this workflow; refusing R9700 guidance');
}
if (!MEGA && EXPECTED_GFX && GFX !== EXPECTED_GFX) {
  throw new Error(`GPU architecture mismatch: expected ${EXPECTED_GFX}, detected ${GFX || 'unknown'}`);
}
if (!MEGA && EXPECTED_TARGET && DEVICE_TARGET !== EXPECTED_TARGET) {
  throw new Error(`GPU product mismatch: expected ${EXPECTED_TARGET}, detected ${DEVICE_TARGET || 'unknown'}`);
}
const normalizeDeviceName = (name) => String(name || '').trim().replace(/\s+/g, ' ').toLowerCase();
if (EXPECTED_DEVICE_NAME &&
    normalizeDeviceName(SETUP_DEVICE_NAME) !== normalizeDeviceName(EXPECTED_DEVICE_NAME)) {
  log(`Setup identity telemetry differs from the deterministic probe: device_name expected ` +
      `${EXPECTED_DEVICE_NAME}, Director returned ${SETUP_DEVICE_NAME || 'unknown'}; using expected value.`);
}
if (EXPECTED_PHYSICAL_CU_COUNT > 0 &&
    SETUP_PHYSICAL_CU_COUNT !== EXPECTED_PHYSICAL_CU_COUNT) {
  log(`Setup identity telemetry differs from the deterministic probe: physical_cu_count expected ` +
      `${EXPECTED_PHYSICAL_CU_COUNT}, Director returned ${SETUP_PHYSICAL_CU_COUNT}; using expected value.`);
}
const DEVICE_NAME = EXPECTED_DEVICE_NAME || SETUP_DEVICE_NAME;
const PHYSICAL_CU_COUNT = EXPECTED_PHYSICAL_CU_COUNT > 0
  ? EXPECTED_PHYSICAL_CU_COUNT : SETUP_PHYSICAL_CU_COUNT;
// Products with a `product:` section in e2e_workflow/knowledge/analysis_skills/roofline/peaks.md.
const CALIBRATED_PRODUCTS = ['r9700', 'r9600d'];
const ROOFLINE_STATUS = (CALIBRATED_PRODUCTS.includes(DEVICE_TARGET) && GFX === 'gfx1201')
  ? `calibrated-${DEVICE_TARGET}`
  : (/^gfx120/.test(GFX) ? 'unknown-device-not-r9700' : 'non-rdna');
const RDNA4_ISOLATE = GFX === 'gfx1201';
if (RDNA4_ISOLATE) {
  // Until the perf/learned stores implement metadata-enforced gens filtering, fail closed:
  // R9700 / gfx1201 receive the dedicated in-tree hardware guide but no CDNA-biased external cards.
  USE_EXPERT_SKILLS = false;
  KERNEL_KNOWLEDGE_DIR = '';
  USE_LEARNED_READ = false;
  WARM_START_ON = false;
}
if (!MEGA && !GPU_RESOURCE.explicit && setup.gpu_requirement) {
  applyGpuResource(resolveGpuRequest(A, GPU_LIST, setup.gpu_requirement));
  if (GPU_RESOURCE.warning) log(`GPU requirement WARNING: ${GPU_RESOURCE.warning}`);
  if (GPU_JOB) {
    log(`GPU requirement: ${GPU_RESOURCE.gpusPerJob} GPUs per job (${setup.gpu_requirement.kind}: ` +
      `${setup.gpu_requirement.evidence || 'no evidence'}) -> ${GPU_JOB}`);
  }
}
// `_task` is the e2e head's DIRECTORY suffix, and the director returns the basename verbatim, so it
// would otherwise ride into the session id and the stored record. The canonical id is folded store-
// side (experience_store.remote_identity); this keeps the rest of the run calling it one name.
const KERNEL_NAME = MEGA ? setup.kernel_name
  : String(setup.kernel_name || '').replace(/_task$/, '') || setup.kernel_name;
const COMMANDMENT = `${EVAL_DIR}/COMMANDMENT.md`;
log(`Setup done. EVAL_DIR=${EVAL_DIR}`);
if (MEGA) {
  const r = await megaAfterSetup();
  if (r) return r.exit;
}

// ---------------------------------------------------------------------------
// Enforce a FROZEN REAL-ONLINE BASELINE in BOTH modes (author AND same-language
// optimize). The immutable unittest times + parity-checks the candidate against
// baseline_src/ / meta.baseline_callable (the live online kernel); if neither
// exists it would silently fall back to timing kernel_src/ against itself — the
// "optimized-HIP vs naive-HIP = fake 15.7×" bug this harness exists to prevent.
// The script has no FS access, so we trust the director's structured verdict
// (it copied baseline_src/ + confirmed the callable). Missing -> abort/re-extract.
// ---------------------------------------------------------------------------
const hasBaseline = setup.baseline_frozen === true ||
  (typeof setup.baseline_callable === 'string' && setup.baseline_callable.trim().length > 0);
if (!hasBaseline) {
  const reason = `no frozen baseline (baseline_src/ or meta.baseline_callable) for ${KERNEL_NAME} — ` +
    `re-extract; refusing to time the candidate against kernel_src/ (fake-win risk)`;
  log(`Setup ABORT: ${reason}`);
  return {
    mode: MODE, authored: false, target_language: TARGET_LANGUAGE,
    eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
    final_geomean: 0, final_patch: '', validation_status: 'no_baseline', reason,
  };
}

// ===========================================================================
// PHASE: Author (mode=author only) — write a fresh from-scratch impl as the
// optimize loop's CODE SEED. On success, HEAD of CANONICAL becomes that seed
// (what the optimize loop diffs its edits against) and the rest of the pipeline
// (Analyze/Benchmark/Profile/optimize loop) runs UNCHANGED on it. The SPEEDUP
// denominator is NEVER the seed — it is the frozen REAL ONLINE kernel in
// baseline_src/ (meta.baseline_callable), regardless of TARGET_LANGUAGE.
// On failure (no correct seed), abort early with a structured result so the
// e2e caller drops this language.
// ===========================================================================
if (MODE === 'author') {
  phase('Author');
  const authored = await agentT(
    roleAgent('author_engineer', 'author', 'Write the simplest correct baseline in the target language.', {
      TARGET_LANGUAGE, OP_SPEC, WORKSPACE: CANONICAL, TASK_DIR: KERNEL_PATH_ORIG,
      GPU_ID: GPU_POOL, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, KERNEL_KNOWLEDGE_DIR,
      // The author engineer reads the learned index too, so the switch has to reach it. It is the
      // second reader; a switch that covers one of two readers is not a switch.
      LEARNED_KB: USE_LEARNED_READ ? 'on' : 'off',
    }),
    { phase: 'Author', label: `author:${TARGET_LANGUAGE}`, schema: AUTHOR_SCHEMA });
  if (!authored || !authored.authored || !says(authored.correctness, 'pass')) {
    log(`Author mode FAILED for ${TARGET_LANGUAGE}: ${authored ? authored.notes || authored.correctness : 'no result'}. Aborting (no seed to optimize).`);
    return {
      mode: 'author', authored: false, target_language: TARGET_LANGUAGE,
      eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
      final_geomean: 0, final_patch: '', validation_status: 'author_failed',
      reason: authored ? authored.notes || 'author produced no correct baseline' : 'author returned nothing',
    };
  }
  log(`Author mode: ${TARGET_LANGUAGE} seed written (correct, seed ${authored.baseline_ms || '?'} ms; denominator = frozen online kernel). Optimizing it now.`);
}

// ===========================================================================
// PHASE: Analyze + Roadmap (TechLead)
// ===========================================================================
phase('Analyze');
let analysis;
if (MEGA) {
  const r = await megaAnalyze();
  if (r) return r.exit;
} else analysis = await agentT(
  roleAgent('tech_lead', 'analyze', 'Analyze the kernel and write the roadmap.', {
    WORKSPACE: CANONICAL, EVAL_DIR, TASK, SKILL_DIR: WORKFLOW_DIR,
    KERNEL_KNOWLEDGE_DIR, GFX, DEVICE_TARGET, PHYSICAL_CU_COUNT, ROOFLINE_STATUS,
    ...RESUME_INPUT,
  }),
  { phase: 'Analyze', label: 'tech_lead:analyze', schema: ANALYZE_SCHEMA });
log(`Analyze done. kernel_type=${analysis ? analysis.kernel_type : '?'}`);

// perf_knowledge pointers resolved by the TechLead in analyze (REFERENCE ONLY; threaded to the
// planner + engineers so they read focused op/language cards instead of the whole base). Empty when
// no operator card applies (e.g. point-cloud HIP ops) or KERNEL_KNOWLEDGE_DIR is unset → no change.
const KK_OPERATOR = (analysis && analysis.kk_operator) || '';
const KK_LANGUAGE = (analysis && analysis.kk_language) || '';
const KK_REFS = (analysis && Array.isArray(analysis.kk_refs)) ? analysis.kk_refs : [];

// ===========================================================================
// PHASE: Benchmark setup (Benchmark Engineer)
// ===========================================================================
phase('Benchmark');
let bench;
if (MEGA) {
  const r = await megaBenchmark();
  if (r) return r.exit;
} else bench = await agentT(
  roleAgent('benchmark_engineer', 'setup', 'Build the COMMANDMENT and record a reliable baseline.', {
    WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL,
    ANALYSIS: analysis,
    ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
    ...(WORKLOAD_SPEC_PATH ? { WORKLOAD_SPEC_PATH } : {}),
    ...(WORKLOAD_SPEC ? { WORKLOAD_SPEC } : {}),
  }),
  { phase: 'Benchmark', label: 'benchmark_engineer', schema: BENCH_SCHEMA });
if (!bench || !bench.baseline_per_case) throw new Error('Benchmark setup failed: no baseline recorded');
const BASELINE_PER_CASE = bench.baseline_per_case;
const BASELINE_GEOMEAN_MS = bench.baseline_geomean_ms;
log(`Benchmark done. ${bench.num_test_cases || BASELINE_PER_CASE.length} cases, baseline geomean ${BASELINE_GEOMEAN_MS} ms, reliable=${bench.reliable}`);
if (MEGA) {
  const r = await megaCalibrate();
  if (r) return r.exit;
}

// ===========================================================================
// PHASE: Baseline profile (Profile Engineer)
// ===========================================================================
phase('Profile');
let profileSummary;
if (MEGA) {
  const r = await megaProfile();
  if (r) return r.exit;
} else profileSummary = await agentT(
  roleAgent('profile_engineer', 'baseline', 'Profile the baseline and classify the bottleneck.', {
    WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL, ROUND: 0,
    COMMANDMENT, GFX, DEVICE_TARGET, PHYSICAL_CU_COUNT, ROOFLINE_STATUS,
    ...RESUME_INPUT,
  }),
  { phase: 'Profile', label: 'profile_engineer:baseline', schema: PROFILE_SCHEMA });
log(`Baseline bottleneck: ${profileSummary ? profileSummary.bottleneck : '?'} (dispatch_count=${profileSummary ? profileSummary.dispatch_count : '?'})`);

// ===========================================================================
// PHASE: Research (Deep Research Agent — OPT-IN via args.dra_enabled)
// Runs AFTER Profile / BEFORE the optimize loop: profile + COMMANDMENT + analysis exist by now, so
// the researcher has the facts it needs. It produces EVAL_DIR/deep_search_brief.md (compact, ranked
// directions) which the TechLead's plan_round seeds directions from. The per-question research is
// fanned out with parallel() and EVERY research agent is wrapped in the agentT() hang-guard, so a
// hung research agent resolves to null and the parallel round-barrier still proceeds (it never wedges
// the run — the known v4 failure mode the hang-guard was built for). DEFAULT OFF → no behavior change.
// ===========================================================================
let researchBriefPath = '';   // EVAL_DIR/deep_search_brief.md when the DRA produced one; '' otherwise
if (DRA_ENABLED && !MEGA) {
  phase('Research');
  const RESEARCH_DIR = `${EVAL_DIR}/research`;

  // --- Stage 0 + 1/2: extract facts, generate + rank research questions (one agent) -------------
  const plan = await agentT(
    roleAgent('researcher', 'research_plan', 'Extract facts and propose ranked research questions.', {
      WORKSPACE: CANONICAL, EVAL_DIR, RESEARCH_DIR, COMMANDMENT, SKILL_DIR: WORKFLOW_DIR,
      ANALYSIS_JSON: `${EVAL_DIR}/analysis.json`, CODEBASE_CONTEXT: `${EVAL_DIR}/codebase_context.md`,
      PROFILING_SUMMARY: profileSummary ? profileSummary.summary_path : '',
      BOTTLENECK: profileSummary ? profileSummary.bottleneck : 'unknown',
      BASELINE_PER_CASE, MAX_QUESTIONS: DRA_MAX_QUESTIONS, TASK,
      KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE, KK_REFS,
    }),
    { phase: 'Research', label: 'researcher:plan', schema: RESEARCH_PLAN_SCHEMA });

  const facts = (plan && plan.facts) || {};
  let questions = (plan && Array.isArray(plan.questions) ? plan.questions : [])
    .slice(0, DRA_MAX_QUESTIONS)
    .map((q, i) => ({ ...q, id: q.id || `q${i}` }));
  log(`Research: ${questions.length} question(s) planned, bottleneck=${facts.bottleneck_type || '?'}`);

  // --- Stages 3/4: research each question IN PARALLEL (native WebSearch/WebFetch), hang-guarded ---
  // parallel() takes an array of zero-arg async thunks and runs them concurrently; each thunk's
  // agentT() bounds the research agent so a hang resolves null instead of blocking the barrier.
  const researchQuestion = (q, stageLabel) => agentT(
    roleAgent('researcher', 'research_question',
      'Research THIS ONE question on the live web and synthesize one judgment.', {
        QUESTION: q, FACTS: facts, RESEARCH_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR,
        ANSWER_OUT: `${RESEARCH_DIR}/answers/${q.id}.json`,
        CODEBASE_CONTEXT: `${EVAL_DIR}/codebase_context.md`,
        PROFILING_SUMMARY: profileSummary ? profileSummary.summary_path : '',
        KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE, KK_REFS,
      }),
    { phase: 'Research', label: `researcher:q ${q.id}${stageLabel ? ' ' + stageLabel : ''}`, schema: RESEARCH_QUESTION_SCHEMA });

  let answers = (await parallel(questions.map((q) => () => researchQuestion(q))))
    .filter(Boolean);
  log(`Research: ${answers.length}/${questions.length} first-pass answers returned`);

  // --- Stages 5/6 (optional): blindspot critique + a second parallel research wave --------------
  if (DRA_BLINDSPOT && answers.length) {
    const crit = await agentT(
      roleAgent('researcher', 'research_blindspot', 'Critique the research and surface new blindspots.', {
        FACTS: facts, RESEARCH_DIR, ANSWERS: answers, MAX_BLINDSPOTS: DRA_MAX_BLINDSPOTS,
        SKILL_DIR: WORKFLOW_DIR,
      }),
      { phase: 'Research', label: 'researcher:blindspot', schema: RESEARCH_BLINDSPOT_SCHEMA });
    const followups = (crit && Array.isArray(crit.blindspots) ? crit.blindspots : [])
      .filter(b => b && b.follow_up_question)
      .slice(0, DRA_MAX_BLINDSPOTS)
      .map((b, i) => ({ id: `b${i}`, question: b.follow_up_question, mode: 'bottleneck',
        tests_hypothesis: '', search_queries: [] }));
    if (followups.length) {
      const more = (await parallel(followups.map((q) => () => researchQuestion(q, 'blindspot'))))
        .filter(Boolean);
      answers = answers.concat(more);
      log(`Research: blindspot pass added ${more.length} answer(s) from ${followups.length} follow-up(s)`);
    }
  }

  // --- Stage 7: synthesize the ranked-directions portfolio + the compact planner brief -----------
  const synth = await agentT(
    roleAgent('researcher', 'research_synthesize',
      'Synthesize the ranked portfolio of optimization directions and the compact brief.', {
        FACTS: facts, RESEARCH_DIR, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR,
        BRIEF_OUT: `${EVAL_DIR}/deep_search_brief.md`, FULL_OUT: `${EVAL_DIR}/deep_search.md`,
        JSON_OUT: `${EVAL_DIR}/deep_search.json`,
      }),
    { phase: 'Research', label: 'researcher:synthesize', schema: RESEARCH_SCHEMA });

  if (synth && synth.brief_path) {
    researchBriefPath = synth.brief_path;
    log(`Research done. ${synth.num_directions || (synth.directions ? synth.directions.length : 0)} ranked direction(s) → ${researchBriefPath}`);
  } else {
    log('Research produced no brief (degraded) — plan_round proceeds without a DRA brief.');
  }
}

// ===========================================================================
// PHASE: Optimization loop (budget-controlled)
// ===========================================================================
let dispatched = 0;          // counts ONLY optimization-direction engineers (the budget)
let round = 0;
let cumulative = 1.0;        // best verified geomean speedup vs the TRUE baseline
let bestSeen = 0;            // best verified geomean of any candidate, committed or not
let noImprove = 0;
let bestPerCase = BASELINE_PER_CASE;
let finalWinner = null;      // {geomean, arithmetic, per_case, patch, source} — also set by a warm-start adopt
let roundsCommitted = 0;     // rounds this run actually landed; a warm-start adopt is NOT one of them
const history = { insights: [], ledger: [], rounds: [], bottleneck_now: profileSummary ? profileSummary.bottleneck : 'unknown', suggest_next: '' };
// One row per KB-seeded direction, joined against what the verifier measured (see the push site in
// the round loop). Fed to update_experience and returned to the caller: a driver aggregating these
// is how anyone notices the KB has been cited fifty times and never once carried a round.
const citations = [];

// DEEP-MODE resume: restore cumulative speedup + insight/ledger history from the prior wave so this
// continuation builds ON the cumulative best (canonical was already seeded from STATE_DIR/best by the
// director) and does not re-explore dead directions. No-op on a fresh run (prior_state undefined).
// mode=mega restores its candidate registry instead (megaLoopInit).
if (!MEGA && setup.resumed && setup.prior_state) {
  const ps = setup.prior_state;
  if (Number.isFinite(ps.cumulative) && ps.cumulative > cumulative) cumulative = ps.cumulative;
  if (Array.isArray(ps.insights)) history.insights = ps.insights;
  if (Array.isArray(ps.ledger)) history.ledger = ps.ledger;
  if (ps.bottleneck_now) history.bottleneck_now = ps.bottleneck_now;
  if (Array.isArray(ps.best_per_case) && ps.best_per_case.length) bestPerCase = ps.best_per_case;
  log(`RESUMED from STATE_DIR: cumulative=${cumulative.toFixed(3)}x, ${history.insights.length} insights, ${history.ledger.length} ledger entries carried forward.`);
}

// ===========================================================================
// PHASE: WarmStart. The recorded speedup only RANKS; adoption is decided by a fresh
// measurement through the verify gate here. gfx comes from the baseline profile's
// on-box `device` string (no extra probe).
// ===========================================================================
// One benched candidate -> one of kb/attest.py's four outcomes. `inapplicable` is a verdict on the
// PAIRING, excluded from the retire arithmetic — both "this workspace lacks those files" and "it
// won but could not be committed here" belong there. A patch that ran and gave a WRONG ANSWER is
// `failed`: it applied, it ran, and it did not deliver what the record promised.
const kbVerdict = (ver, sp) => {
  if (!ver) return 'not_reproduced';
  if (says(ver.correctness, 'fail')) return 'failed';
  if (says(ver.status, 'apply_failed')) {
    return /outside_editable_set/i.test(String(ver.notes || '')) ? 'inapplicable' : 'not_reproduced';
  }
  if (!says(ver.status, 'verified')) return 'not_reproduced';
  if (!(sp > 1.0)) return 'failed';              // applied, ran, correct, just did not go faster
  return says(ver.correctness, 'pass') ? 'inapplicable' : 'not_reproduced';   // fast but never adopted
};
// mode=mega seeds no global score from authoring WIP: each candidate lane advances on its own and
// cumulative moves only when a fully scored candidate wins. megaLoopInit restores its registry.
if (MEGA) {
  const r = await megaLoopInit();
  if (r) return r.exit;
}
let warm_start = { adopted: false, read_reason: WARM_START_ON ? 'read' : 'disabled', candidates: [] };
let skipLoop = false;
if (!MEGA && WARM_START_ON && !setup.resumed && KB_ROOT_OK) {
  phase('WarmStart');
  if (!GFX) {
    warm_start.read_reason = 'missing_arch';
    log('[kb] warm-start skipped: no gfx detected from the baseline profile device string.');
  } else {
    // The two planes take a different root flag and a different name→page rule (the store addresses
    // by canonical id, so there is nothing to match fuzzily), and print the same JSON.
    const localResolveCmd = KB_MODE === 'store'
      ? `resolve-remote --plane local --store ${JSON.stringify(KB_STORE_DIR)}${KB_VERSION_FLAG}`
      : `resolve --root ${JSON.stringify(KB_ARTIFACTS_DIR)} --match ${WARM_START_MATCH}`;
    // Bring the local store level with the tree before reading it, scoped to this kernel and arch
    // so it costs a stat walk over one page. Writes file both schemes, so this is usually a no-op;
    // what it catches is the backlog and anything a curation pass imported by hand — without it,
    // reading by key looks like a cold start on the kernels with the longest history. Idempotent
    // and non-destructive (cmd_sync_local): the session id is a digest of the patch, and the write
    // carries the store's ledger, retraction and reproduction count forward.
    //
    // `|| true`: a sync that cannot run must degrade to a thinner page, never a failed warm start.
    // Its summary goes to a file rather than /dev/null because its failures are otherwise
    // indistinguishable from an empty KB — `skipped.unreadable` above all, which is what a tree
    // written by a root container looks like to everyone else.
    const syncCmd = KB_MODE === 'store' && KB_ARTIFACTS_DIR
      ? `python3 ${JSON.stringify(EXPERIENCE_STORE)} sync-local --root ${JSON.stringify(KB_ARTIFACTS_DIR)} \\
  --store ${JSON.stringify(KB_STORE_DIR)} --kernel-name ${JSON.stringify(KERNEL_NAME)} --gfx ${GFX} \\
  > ${JSON.stringify(EVAL_DIR + '/kb_sync.json')} 2>&1 || true\n`
      : '';
    const commonArgs =
      `--kernel-name ${JSON.stringify(KERNEL_NAME)} --language ${JSON.stringify(TARGET_LANGUAGE)} \\\n` +
      `  --gfx ${GFX} --top-n 3 --min-speedup ${WARM_START_MIN_SPEEDUP} \\\n` +
      `  --refs-dir ${JSON.stringify(EVAL_DIR + '/kb_references')}`;
    // Deliberately NO `--precision` on the read, though the write below states it. The filter earns
    // its keep on the tuned lane, where a table's destination filename encodes its dtype and a
    // mismatched one is simply unusable. A PATCH is source: a Triton kernel proven at bf16 is
    // usually the same code that helps at fp8, so narrowing here would turn a thin page into an
    // empty one and cost a cold start — the exact hours this read exists to avoid. Recording the
    // dtype without filtering on it keeps that call reversible once the pages are thick enough to
    // show whether cross-dtype patches actually carry.
    // Remote first, local curated tree as the fallback. The service is the shared plane and should
    // win when it has an answer, but it is still filling up, while `kb_artifacts/` holds a hand-
    // curated history (retired entries, one entry per direction) that a thin remote page must not
    // shadow. An empty remote answer is indistinguishable from a 404 on this scheme, so "no
    // candidates" — not "no error" — is what triggers the second read. Both reads are seconds and
    // no GPU; the thing they protect against is a cold start that costs hours.
    //
    // The e2e lane pushed this branch into cmd_resolve via `read_planes`; this one stays in bash
    // because its fallback is a DIFFERENT SUBCOMMAND — off the store scheme `localResolveCmd` reads
    // the curated tree, not a second plane of the same store, and no plane list expresses that.
    const resolveScript = KB_REMOTE === 'off'
      ? `${syncCmd}python3 ${JSON.stringify(EXPERIENCE_STORE)} ${localResolveCmd} \\\n  ${commonArgs}`
      : `${syncCmd}${KB_ENV_PRELUDE}
REMOTE_OUT=''
if [ -n "$KB_STORE_TOKEN" ]; then
  REMOTE_OUT=$(python3 ${JSON.stringify(EXPERIENCE_STORE)} resolve-remote --plane remote${KB_VERSION_FLAG} \\
  ${commonArgs} 2>/dev/null || true)
  if printf '%s' "$REMOTE_OUT" | python3 -c 'import json,sys; sys.exit(0 if (json.load(sys.stdin).get("candidates") or []) else 1)' 2>/dev/null; then
    printf '%s\\n' "$REMOTE_OUT"; exit 0
  fi
fi
python3 ${JSON.stringify(EXPERIENCE_STORE)} ${localResolveCmd} \\
  ${commonArgs}`;
    const resolved = await agentT(
      `You are the warm-start resolver. Run EXACTLY this ` +
      `${KB_REMOTE === 'off' && !syncCmd ? 'command' : 'script'} ` +
      `and return its single-line JSON stdout verbatim as StructuredOutput — do not add, drop, reorder, ` +
      `or reinterpret any field. ` +
      // Saying the sync line is not the answer keeps the agent from relaying its summary, or from
      // treating its `|| true` as an error worth retrying.
      (syncCmd ? `Its first line brings the on-disk knowledge base up to date and prints nothing ` +
        `(its summary is redirected to a file); the JSON you return is the LAST command's. ` : '') +
      (KB_REMOTE === 'off' ? '' :
        `It tries the shared KB Store service first and falls back to the on-disk knowledge base by ` +
        `itself; run it as one script, do not split it into separate commands. `) +
      `An empty candidate list is a VALID answer meaning "nothing recorded for this kernel yet". Do not ` +
      `retry with different flags and do not invent candidates:
\`\`\`bash
${resolveScript}
\`\`\``,
      { phase: 'WarmStart', label: 'warm_start:resolve', schema: WARMSTART_RESOLVE_SCHEMA }) || {};
    warm_start.read_reason = resolved.read_reason || 'read';
    warm_start.slug = resolved.slug || '';
    warm_start.match_tier = resolved.match_tier || '';
    warm_start.filtered = resolved.filtered || null;
    const cands = Array.isArray(resolved.candidates) ? resolved.candidates : [];
    warm_start.candidates = cands.map(c => ({
      rank: c.rank, slug: c.slug, speedup: c.speedup, direction: c.direction || '', status: 'read',
      session_id: c.session_id || '', exp_dir: c.exp_dir || '',
    }));
    const f = resolved.filtered || {};
    // Which plane answered — read off the answer, not inferred from the mode. Both keyed reads
    // emit `canonical_id`, so only `read_plane` (kb/plane.py:read_planes) separates service from
    // disk. Not cosmetic: it picks the flags the attest step files this run's verdict with. No
    // canonical id means neither keyed read answered — on `store` mode that is an empty page
    // (`none`), since the tree is only consulted under `--kb_mode local`.
    warm_start.plane = resolved.read_plane === 'remote' ? 'remote'
      : resolved.canonical_id ? 'store'
        : KB_MODE === 'store' ? 'none' : 'local';
    if (resolved.canonical_id) log(`[kb] plane=${warm_start.plane} key=${resolved.canonical_id}`);
    else if (KB_REMOTE !== 'off') log(`[kb] plane=${warm_start.plane} (service had no candidates, ` +
      `or no credentials)`);
    log(`[kb] experience read: slug=${resolved.slug || '?'} (${resolved.match_tier || 'exact'} match of ` +
      `${resolved.requested_slug || KERNEL_NAME}) reason=${warm_start.read_reason} ` +
      `candidates=${cands.length}${f.total ? ` of ${f.total} recorded [${f.retired || 0} retired, ` +
      `${f.below_min_speedup || 0} below ${WARM_START_MIN_SPEEDUP}x, ` +
      `${f.same_direction_collapsed || 0} same-direction]` : ''}` +
      // Every count on this line is over the rows FETCHED; a saturated scan means the page held
      // more.
      `${f.scan_saturated ? ` (SCAN SATURATED at ${f.scanned} — the page holds more)` : ''}`);
    const otherLangs = Array.isArray(resolved.other_language_pages) ? resolved.other_language_pages : [];
    if (!cands.length && otherLangs.length) {   // wrong target_language, not an empty store
      log(`[kb] NOTE: no ${TARGET_LANGUAGE} page, but the store holds ${otherLangs.join(', ')} — ` +
        `check the target_language this lane was invoked with.`);
    }
    // reference-only: prose is already mirrored to EVAL_DIR/kb_references by the resolver; do not apply.
    if (!WARM_START_REF_ONLY) {
      const editableCsv = ((analysis && analysis.modifiable_files) || []).join(',');
      for (const c of cands) {                              // already rank-ordered (fastest first)
        const rec = warm_start.candidates.find(x => x.rank === c.rank);
        const remapOut = `${EVAL_DIR}/warm_start/cand_${c.rank}/remapped.diff`;
        const ver = await agentT(
          roleAgent('verify_engineer', 'verify',
            'Validate a HISTORICAL warm-start patch — the SAME gate as any round candidate. The patch was ' +
            'won in the workspace that produced it, whose paths need not exist here, so FIRST run EXACTLY:\n' +
            '```bash\n' +
            `mkdir -p ${JSON.stringify(`${EVAL_DIR}/warm_start/cand_${c.rank}`)}\n` +
            `python3 ${JSON.stringify(EXPERIENCE_STORE)} remap --patch ${JSON.stringify(c.patch_path)} \\\n` +
            `  --out ${JSON.stringify(remapOut)} --editable ${JSON.stringify(editableCsv)} \\\n` +
            `  --workspace ${JSON.stringify(CANONICAL)}\n` +
            '```\n' +
            'It prints one line of JSON. remapped=true: apply the REMAPPED patch (its paths were rewritten ' +
            'onto this workspace). reason "no_change_needed": apply the original PATCH. Any other reason ' +
            '(notably "unmapped_paths" — the patch touches files this workspace does not have): return ' +
            'status:"apply_failed", notes "patch_outside_editable_set", and DO NOT apply or benchmark. ' +
            'Apply with `git apply "$P" || git apply --3way "$P"`. On ANY failure restore the working copy ' +
            'fully (git checkout -- . and delete untracked files) before returning so the next candidate ' +
            'starts from a clean tree.', {
            CANONICAL, PATCH: c.patch_path, REMAP_OUT: remapOut,
            VERIFY_DIR: `${EVAL_DIR}/warm_start/cand_${c.rank}`,
            EDITABLE_SET: (analysis && analysis.modifiable_files) || [],
            GPU_ID: GPU_SINGLE, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
            ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
            ...(REQUIRE_GRAPH_CAPTURE ? { REQUIRE_GRAPH_CAPTURE: '1' } : {}),
          }),
          { phase: 'WarmStart', label: `warm_start:verify c${c.rank}`, schema: VERIFY_SCHEMA });
        const sp = primSpeedup(ver);
        if (ver && says(ver.status, 'verified') && says(ver.correctness, 'pass') && sp > 1.0) {
          const adopt = await agentT(
            `You are the TechLead adopting a validated warm-start patch into the canonical workspace.
\`\`\`bash
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
cd ${CANONICAL}
git checkout -- .
# the remapped patch when verify produced one (its paths fit THIS workspace), else the stored one
P=${JSON.stringify(remapOut)}; [ -f "$P" ] || P=${JSON.stringify(c.patch_path)}
git apply "$P" || git apply --3way "$P"
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "warm-start adopt: ${c.slug} (${sp.toFixed(2)}x)"
git --no-pager diff "$(git rev-list --max-parents=0 HEAD)..HEAD" > ${EVAL_DIR}/current_best.diff
\`\`\`
If BOTH applies fail, apply manually to match intent, then add -A + commit and RE-RUN the COMMANDMENT
correctness check; only report committed=true if it still passes. Return JSON {committed, current_best_diff, note}.`,
            { phase: 'WarmStart', label: `warm_start:adopt c${c.rank}`, schema: COMMIT_SCHEMA });
          if (adopt && adopt.committed) {
            // The optimize loop now builds ON this patch: cumulative starts at the adopted Nx, so this
            // run's own rounds only earn the delta above it (split out as incremental_speedup below).
            cumulative = sp;
            bestPerCase = (ver.per_case && ver.per_case.length) ? ver.per_case : bestPerCase;
            finalWinner = { source: `warm_start:${c.slug}`, geomean: sp,
              arithmetic: ver.verified_arithmetic || sp, per_case: bestPerCase, patch: c.patch_path };
            if (bestSeen < sp) bestSeen = sp;
            warm_start.adopted = true;
            warm_start.adopted_speedup = sp;
            warm_start.slug = c.slug;
            // Carried into the write-back: the idea this run started from, and the entry it
            // descends from — lineage a later curation pass cannot recover from the diff alone.
            warm_start.direction = c.direction || '';
            warm_start.exp_dir = c.exp_dir || '';
            if (rec) { rec.status = 'adopted'; rec.outcome = 'validated'; rec.verified_speedup = sp; }
            log(`[kb] warm-start ADOPTED ${c.slug} @ ${sp.toFixed(2)}x — optimizing from the patched state.`);
            profileSummary = await agentT(
              roleAgent('profile_engineer', 'reprofile',
                'Re-profile the adopted warm-start state and classify the new bottleneck.', {
                WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_SINGLE, ROUND: 0,
                COMMANDMENT, PREVIOUS_METRICS: profileSummary,
              }),
              { phase: 'WarmStart', label: 'warm_start:reprofile', schema: PROFILE_SCHEMA }) || profileSummary;
            if (history) history.bottleneck_now = profileSummary ? profileSummary.bottleneck : history.bottleneck_now;
            if (WARM_START_RETURN_AFTER) { skipLoop = true; warm_start.returned_after_read_kb = true; }
            break;
          }
        }
        if (rec && rec.status !== 'adopted') {
          rec.status = (ver && ver.status) ? `rejected:${ver.status}` : 'rejected:apply_failed';
          rec.outcome = kbVerdict(ver, sp);
          rec.verified_speedup = sp || 0;
        }
        log(`[kb] warm-start candidate c${c.rank} ${rec ? rec.status : 'rejected'} (${sp ? sp.toFixed(2) + 'x' : 'no measure'}).`);
      }
      // Record what THIS box saw, so the next one reads a ledger and not just a speedup. Only
      // candidates carrying an `outcome` reached the GPU — the loop breaks after adopting, and
      // counting an unbenched record would enter an attempt that never happened.
      const benched = warm_start.candidates.filter(x => x.outcome && (x.session_id || x.exp_dir));
      if (benched.length) {
        const planeFlags = warm_start.plane === 'remote' ? '--plane remote'
          : warm_start.plane === 'store' ? `--plane local --store ${JSON.stringify(KB_STORE_DIR)}` : '';
        const cmds = benched.map(x =>
          `python3 ${JSON.stringify(EXPERIENCE_STORE)} attest ${planeFlags} ` +
          (x.exp_dir ? `--exp-dir ${JSON.stringify(x.exp_dir)} ` : '') +
          (planeFlags && x.session_id ? `--session-id ${JSON.stringify(x.session_id)} ` : '') +
          `--kernel-name ${JSON.stringify(KERNEL_NAME)} --language ${JSON.stringify(TARGET_LANGUAGE)} ` +
          `--gfx ${GFX}${KB_VERSION_FLAG} --outcome ${x.outcome} ` +
          `--measured-speedup ${Number(x.verified_speedup) || 0} ` +
          `--note ${JSON.stringify(`kernel_lane warm-start c${x.rank}: ${x.status}`.slice(0, 200))} ` +
          `--measured-by ${JSON.stringify('kernel_lane:' + GFX)} --apply || true`);
        await agentT(
          `You are the kernel knowledge-base attestor. Run EXACTLY these commands in order and ` +
          `return {"ran": <how many you ran>, "note": "<anything that failed>"}. Each records what ` +
          `this box saw when it benched a stored patch. Do NOT edit them, do NOT add or drop any, ` +
          `and do NOT retry a failure — a repeat would double-count the attempt.\n` +
          '```bash\n' + KB_ENV_PRELUDE + cmds.join('\n') + '\n```',
          { phase: 'WarmStart', label: 'kb:attest',
            schema: obj({ ran: { type: 'number' }, note: { type: 'string' } }, []) });
        log(`[kb] attested ${benched.length} benched candidate(s): ` +
          benched.map(x => `c${x.rank}=${x.outcome}`).join(' '));
      }
    }
  }
}

let plannerStopReason = '';
// mode=mega ends on the budget, its modeled clock or a stop rule in megaRound, not on no-improve rounds.
while (!skipLoop && dispatched < BUDGET && (MEGA || noImprove < MAX_NO_IMPROVE)) {
  // --- (0) HARD STOP: no new round may START past the deadline ----------
  // Checked BEFORE round++ so an expired check does not inflate the round count. The in-flight round
  // is never interrupted — a killed round leaves a half-verified patch and no report.
  let left = await secondsLeft(`pre-r${round + 1}`);
  if (left <= 0) {
    deadlineHit = true;
    log(`WALL-CLOCK DEADLINE reached after ${round} round(s) (budget used ${dispatched}/${BUDGET}) — ` +
        `starting no further rounds; going straight to report + validation.`);
    break;
  }
  if (DEADLINE_EPOCH) log(`[deadline] ${(left / 60).toFixed(0)} min remain before the hard stop.`);

  round++;
  const remaining = BUDGET - dispatched;
  phase('Optimize');
  if (MEGA) {
    const step = (await megaRound(remaining)) || {};
    if (step.exit) return step.exit;
    if (step.stop) break;
    continue;
  }

  // --- (a) Plan the round (TechLead) ------------------------------------
  const planInputs = (forbidStop) => ({
    EVAL_DIR, ROUND: round, BUDGET_REMAINING: remaining, CUMULATIVE_SPEEDUP: cumulative,
    BASELINE_GEOMEAN_MS, SKILL_DIR: WORKFLOW_DIR, PROFILE_SUMMARY: profileSummary,
    CURRENT_BEST_PER_CASE: bestPerCase, HISTORY: history,
    KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE, KK_REFS,
    GFX, DEVICE_TARGET, PHYSICAL_CU_COUNT, ROOFLINE_STATUS,
    ...KB_INPUTS,
    // DRA brief (REFERENCE), from main. plan_round reads it and seeds directions[] from the ranked
    // DRA directions — see tech_lead.md plan_round. Spread conditionally, so when dra_enabled was off
    // (or the research degraded) the key is absent and the prompt is byte-identical to a build
    // without the feature. Threaded through `planInputs` rather than the call site so the forced
    // re-plan below gets it too: a replan that silently loses the research brief would plan against
    // a different input set than the round it is replacing.
    ...(researchBriefPath ? { DEEP_SEARCH_BRIEF: researchBriefPath } : {}),
    // Stated UNCONDITIONALLY, in both directions. `use_learned_kb=false` used to omit the budget and
    // nothing else, while roles/tech_lead.md still listed knowledge/learned/INDEX.md among the files
    // to read — so the off switch removed the budget that CONSTRAINS the KB and left the instruction
    // that USES it, i.e. the arm it exists to produce (a clean KB-off control) was never clean. The
    // role now gates on this value, so it has to be present whichever way it points.
    LEARNED_KB: USE_LEARNED_READ ? 'on' : 'off',
    ...(USE_LEARNED_READ ? { LEARNED_KB_BUDGET:
      `At most ${KB_DIR_CAP} of this round's directions may draw on learned cards, and you must cite ` +
      `the cards you used in that direction's \`learned_refs\`. ` +
      (KB_COLD_DIRECTION ? `At least ONE direction must be planned from the profile alone, with an ` +
        `empty \`learned_refs\` — it is this round's control, and without it a round cannot tell you ` +
        `whether the cards helped or merely crowded out what you would have tried anyway. ` : '') +
      `Read as much of the index as you like; the budget is on how much of the ROUND the KB steers, ` +
      `not on what you may look at.` } : {}),
    // Both keys absent unless a deadline was passed => byte-identical prompt when OFF.
    ...(DEADLINE_EPOCH ? { MINUTES_REMAINING: Math.round(left / 60) } : {}),
    ...(forbidStop ? { STOP_NOT_PERMITTED:
      `You returned stop=true with ${Math.round(left / 60)} min of the wall-clock window still ` +
      `unspent. This run is a FIXED-WINDOW measurement: ending early is an invalid data point no ` +
      `matter how good the number or how well argued the rationale, and your role's own convergence ` +
      `gates (launch floor closed, gain under the promotion bar, roofline %, "only a different tech ` +
      `stack is left") do NOT authorise stopping here. Plan at least one direction. Switch lever ` +
      `rather than re-running an exhausted one — algorithm / memory / compute / host_runtime, or a ` +
      `dedicated deep_explore ground-up rewrite. If every direction really is exhausted, spend the ` +
      `round RE-MEASURING the committed best to tighten its error bars and say so. Sizing: keep the ` +
      `round finishable inside the remaining ${Math.round(left / 60)} min.` } : {}),
  });

  let plan = await agentT(
    roleAgent('tech_lead', 'plan_round', 'Decide this round\'s orthogonal directions (or stop).',
      planInputs(false)),
    { phase: 'Optimize', label: `tech_lead:plan r${round}`, schema: PLAN_SCHEMA });

  // NO EARLY STOP: keep refusing while the window is materially unspent. This used to refuse ONCE,
  // on the reasoning that a role repeating "stop" has run out of ideas. Measured at a 2h budget that
  // cost ~3 min (write_req_to_token_pool_triton stopped at 1.95h/2.00h) and looked harmless; the cost
  // is proportional to the budget, so at 8h the same one-shot escape hands back hours. The caller's
  // rule is "must keep optimizing until the budget is spent", so the refusal has to be as long-lived
  // as the window. Bounded by MAX_FORCED_REPLANS so a role that cannot produce a direction at all
  // degrades to stopping instead of spinning the window away on planning.
  while (DEADLINE_EPOCH && left > NO_STOP_S && forcedReplans < MAX_FORCED_REPLANS &&
         (!plan || plan.stop || !plan.directions || plan.directions.length === 0)) {
    forcedReplans++;
    log(`Round ${round}: stop=true with ${(left / 60).toFixed(0)} min left — refusing (#${forcedReplans}), re-planning.`);
    plan = await agentT(
      roleAgent('tech_lead', 'plan_round', 'Your stop was refused: plan at least one direction.',
        planInputs(true)),
      { phase: 'Optimize', label: `tech_lead:replan r${round}#${forcedReplans}`, schema: PLAN_SCHEMA });
    left = await secondsLeft(`replan-r${round}`);
  }

  if (plan && Array.isArray(plan.directions) && ROOFLINE_STATUS.startsWith('unknown')) {
    const before = plan.directions.length;
    plan.directions = plan.directions.filter((d) =>
      !/roofline|peak\s*(?:flops|bandwidth)|%\s*of\s*peak/i.test(
        `${d && d.title || ''} ${d && d.prompt || ''} ${d && d.why || ''}`));
    if (plan.directions.length !== before) {
      log(`Round ${round}: removed ${before - plan.directions.length} direction(s) that relied on unknown peaks.`);
    }
    if (before > 0 && plan.directions.length === 0) {
      plannerStopReason = 'all_directions_filtered_unknown_peaks';
    }
  }

  if (!plan || plan.stop || !plan.directions || plan.directions.length === 0) {
    if (plannerStopReason === 'all_directions_filtered_unknown_peaks') {
      log(`Round ${round}: harness stopped after all directions were filtered because calibrated peaks are unknown.`);
    } else {
      log(`Round ${round}: TechLead chose to stop. ${plan ? plan.reasoning || '' : ''}`);
    }
    break;
  }

  let directions = plan.directions.slice(0, remaining).map((d, i) => ({
    ...d,
    idx: i,
    id: d.id || `r${round}_d${i}`,
    gpu_id: GPU_MODE === 'pin' && !GPU_JOB ? GPU_LIST[i % GPU_LIST.length] : GPU_POOL,
    out_dir: `${EVAL_DIR}/round_${round}/engineer_${i}`,
  }));
  // deep_explore is a DEDICATED-ROUND, heavyweight mandate: if the plan includes one, run ONLY it this
  // round (its broad ground-up rewrite touches many files and can't be merged with specialist patches),
  // and charge DEEP_COST against the budget. Otherwise each specialist direction costs 1.
  const deepDir = directions.find(d => d.specialty === 'deep_explore');
  if (deepDir) directions = [deepDir];
  const roundCost = directions.reduce((s, d) => s + (d.specialty === 'deep_explore' ? DEEP_COST : 1), 0);
  dispatched += roundCost;
  log(`Round ${round}: ${directions.length} direction(s) [${directions.map(d => d.specialty).join(', ')}], cost ${roundCost}, budget ${dispatched}/${BUDGET}`);

  // --- (b,c) Optimize -> Verify, pipelined per direction ----------------
  const results = await pipeline(
    directions,
    (d) => {
      const isDeep = d.specialty === 'deep_explore';
      // deep_explore reads its own role (broad authority + own iteration loop); specialists read engineer.md.
      const readLine = isDeep
        ? `Then Read ${WORKFLOW_DIR}/roles/deep_engineer.md and the knowledge files under ${WORKFLOW_DIR}/knowledge/ ` +
          (USE_LEARNED_READ ? '' : `— EXCLUDING knowledge/learned/, which is off for this run: do not open ` +
            `INDEX.md or any card there. `) +
          `(you have broad authority — combine algorithm + memory + compute + host_runtime levers in one ` +
          `coherent rewrite), and follow them. You MAY edit ANY modifiable source (kernel + Python wrapper ` +
          `+ C++ binding), not just focus_files. Run your OWN multi-iteration measure→(self-)profile→rewrite ` +
          `loop and push to the TARGET; keep the best correct version.`
        : `Then Read ${WORKFLOW_DIR}/roles/engineer.md and ${WORKFLOW_DIR}/knowledge/self_monitoring.md and the ` +
          `knowledge files for your specialty, and follow them.`;
      return agentT(
      `You are Engineer ${d.id} (specialty=${d.specialty}) for round ${round}.
First create YOUR private workspace, then optimize.
\`\`\`bash
# Issue #429: ALWAYS use materialize_workspace.sh — do NOT inline tar/cp. Agents that omitted
# --exclude='*.so' previously copied multi-GiB aiter/jit/*.so into every engineer/verify clone.
# The script excludes nested *.so/*.o and aiter/jit, preserves symlinks (never -h), and may
# share an immutable aiter tree via --link-aiter under EVAL_DIR/_shared. Candidate builds still
# use per-workspace .torch_ext via gpu_lock.sh (isolation unchanged).
mkdir -p ${d.out_dir}
bash ${WORKFLOW_DIR}/scripts/materialize_workspace.sh \\
  --src ${CANONICAL} --dst ${d.out_dir}/workspace \\
  --shared-root ${EVAL_DIR}/_shared --link-aiter
\`\`\`
${readLine} If KK_OPERATOR is non-empty, also consult the operator/language SOTA cards under
KERNEL_KNOWLEDGE_DIR per your role's "operator/language SOTA knowledge (REFERENCE ONLY)" section
(facts/how-to only; measure everything; never go below baseline).
Save best_patch.diff via \`cd <KERNEL_PATH> && git diff > ${d.out_dir}/best_patch.diff\` when geomean>${CANDIDATE_FLOOR_TXT}.

## Inputs
${cfg({
        SPECIALTY: d.specialty,
        DIRECTION: { id: d.id, title: d.title, focus_files: d.focus_files || [], expected_speedup: d.expected_speedup, prompt: d.prompt },
        ...(isDeep ? { TARGET: d.expected_speedup ? `reach ${d.expected_speedup}x (or ~90% of the roofline ceiling), whichever is the harder bar` : 'reach ~90% of the roofline ceiling' } : {}),
        KERNEL_PATH: `${d.out_dir}/workspace`,
        OUTPUT_DIR: d.out_dir,
        CANONICAL, GPU_ID: d.gpu_id, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT,
        codebase_context: `${EVAL_DIR}/codebase_context.md`,
        profiling_summary: profileSummary ? profileSummary.summary_path : '',
        baseline_per_case: BASELINE_PER_CASE,
        INSIGHTS: history.insights,
        KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE,
        KK_REFS: (d.kk_refs && d.kk_refs.length ? d.kk_refs : KK_REFS),
        GFX, DEVICE_TARGET, PHYSICAL_CU_COUNT, ROOFLINE_STATUS,
        ...KB_INPUTS,
      })}

Return ONLY the worker_result.json structure as StructuredOutput.` +
      // Built inline, not via roleAgent(), so the injection has to be appended here too.
      expertSkillsBlock(isDeep ? 'deep_engineer' : 'engineer'),
      { phase: 'Optimize', label: `${isDeep ? 'deep' : 'eng'} ${d.id}:${d.specialty}`, schema: ENG_SCHEMA }
    ).then((eng) => ({ d, eng }));
    },

    (prev) => {
      const { d, eng } = prev;
      const patch = `${d.out_dir}/best_patch.diff`;
      // Harvest is MEASUREMENT-anchored, not return-value-anchored. An engineer only writes
      // best_patch.diff when it beat the floor (the Optimize prompt: "Save best_patch.diff ... when
      // geomean>CANDIDATE_FLOOR"), so a lost/failed StructuredOutput does NOT imply there is no winning patch: an
      // engineer that died, timed out, or mis-returned can still have left an applies-clean above-floor diff
      // on disk (observed in a bake-off: a 1.56x Triton patch was silently dropped because its
      // worker_result.json/StructuredOutput never came back). The only return we can TRUST to suppress
      // a patch is a clean below-floor one — the engineer ran, measured, honestly reported <=the floor, and
      // therefore wrote no patch. In every other case (null/failed return, OR a claimed above-floor one) a patch
      // MIGHT be on disk, so we hand the path to verify and let the oracle be the source of truth. We
      // cannot stat the file from the workflow sandbox, so the "is there actually a patch" decision is
      // delegated to verify, which returns apply_failed on an absent/empty patch — dropped by the
      // `verified` filter below, i.e. the same outcome as skipping, but with no false loss.
      // An invalid/missing source verdict must reach Verify, even after an empty-patch return.
      const trustworthyBelowBaseline = eng && eng.status !== 'failed' && !(primSpeedup(eng) > CANDIDATE_FLOOR) &&
        eng.status !== 'invalid_measurement' && eng.measurement_valid === true;
      if (trustworthyBelowBaseline) {
        return { d, eng, ver: null };
      }
      const recovered = !eng || eng.status === 'failed';
      if (recovered) log(`Round ${round} dir ${d.id}: engineer return ${eng ? 'failed' : 'missing'} — ` +
        `sending best_patch.diff to verify anyway (oracle decides).`);
      return agentT(
        roleAgent('verify_engineer', 'verify', 'Independently re-measure this candidate patch.', {
          CANONICAL, PATCH: patch, VERIFY_DIR: `${d.out_dir}/verify`,
          ENGINEER_WORKSPACE: `${d.out_dir}/workspace`,
          EVAL_DIR, WORKFLOW_DIR, GPU_ID: d.gpu_id, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
          ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
          ...(REQUIRE_GRAPH_CAPTURE ? { REQUIRE_GRAPH_CAPTURE: '1' } : {}),
        }),
        { phase: 'Verify', label: `verify ${d.id}${recovered ? ' (recovered)' : ''}`, schema: VERIFY_SCHEMA }
      ).then((ver) => ({ d, eng, ver, patch }));
    }
  );

  const clean = results.filter(Boolean);
  const verified = clean.filter(r => r.ver && says(r.ver.status, 'verified') &&
    says(r.ver.correctness, 'pass') && primSpeedup(r.ver) > CANDIDATE_FLOOR);

  // --- (d) Build candidate list; integrate if >=2 verified --------------
  // `geomean` here is the PRIMARY metric used for sorting/gating/cumulative: the time-weighted
  // ratio-of-sums when a workload spec is active, else the unweighted geomean (unchanged behavior).
  // The raw unweighted geomean is retained separately for the report.
  let candidates = verified.map(r => ({
    source: `engineer ${r.d.id}`, id: r.d.id, title: r.d.title, specialty: r.d.specialty,
    geomean: primSpeedup(r.ver), geomean_unweighted: r.ver.verified_geomean,
    weighted: r.ver.verified_weighted != null ? r.ver.verified_weighted : null,
    arithmetic: r.ver.verified_arithmetic || r.ver.verified_geomean,
    per_case: r.ver.per_case || [], patch: r.patch,
  }));

  let integrate = null;
  if (verified.length >= 2) {
    phase('Merge');
    integrate = await agentT(
      roleAgent('integrator', 'integrate', 'Combine this round\'s verified patches into one best implementation.', {
        CANONICAL, INTEGRATE_DIR: `${EVAL_DIR}/round_${round}/integrate`,
        EVAL_DIR, WORKFLOW_DIR, GPU_ID: GPU_POOL, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
        BEST_INDIVIDUAL: Math.max(...candidates.map(c => c.geomean)),
        PATCHES: verified.map(r => ({ id: r.d.id, specialty: r.d.specialty, title: r.d.title,
          strategy: r.eng ? r.eng.strategy : '', verified_geomean: r.ver.verified_geomean,
          files: r.d.focus_files || [], patch: r.patch })),
        INSIGHTS: history.insights,
      }),
      { phase: 'Merge', label: `integrate r${round}`, schema: INTEGRATE_SCHEMA });
    const integPrim = integrate && integrate.best ? primSpeedup({
      verified_weighted: integrate.best.weighted, verified_geomean: integrate.best.geomean,
    }) : 0;
    if (integrate && integrate.conclusion === 'improved' && integrate.best &&
      integPrim > Math.max(...candidates.map(c => c.geomean))) {
      candidates.push({
        source: 'integrated', id: `r${round}_integrated`, title: 'integrated', specialty: 'integrate',
        geomean: integPrim, geomean_unweighted: integrate.best.geomean,
        weighted: integrate.best.weighted != null ? integrate.best.weighted : null,
        arithmetic: integrate.best.arithmetic || integrate.best.geomean,
        per_case: integrate.best.per_case || [], patch: integrate.best.patch_file,
      });
    }
  }

  candidates.sort((a, b) => b.geomean - a.geomean);
  const winner = candidates[0] || null;
  const improved = !!(winner && winner.geomean > cumulative * (1 + MIN_IMPROVE));
  // Separate question from `improved`: is the SEARCH advancing, not did it beat the incumbent. The
  // `bestSeen > 0` guard keeps round 1 deciding on `improved` alone; from then on bestSeen >= cumulative
  // at the default floor, so at the default PROGRESS_DELTA this implies `improved` and changes nothing.
  // A round with NO candidate is never progress, so a dead round still counts against MAX_NO_IMPROVE.
  const madeProgress = !!(winner && bestSeen > 0 && winner.geomean > bestSeen * (1 + PROGRESS_DELTA));

  // --- (e) Commit the winner into the canonical workspace ---------------
  if (improved) {
    await agentT(
      `You are the TechLead committing round ${round}'s winning patch into the canonical workspace.
\`\`\`bash
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
cd ${CANONICAL}
git checkout -- .
# Try a plain apply first, then a 3-way apply (auto-reconciles context-line drift against the blobs)
# before falling back to a manual reconstruction. --3way resolves most "patch does not apply" cases
# that are just context offsets, so the manual path is only hit on a genuine semantic conflict.
git apply ${winner.patch} || git apply --3way ${winner.patch}
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "round ${round} winner: ${winner.source} (${winner.geomean.toFixed(2)}x)"
git --no-pager diff "$(git rev-list --max-parents=0 HEAD)..HEAD" > ${EVAL_DIR}/current_best.diff
\`\`\`
If BOTH \`git apply\` and \`git apply --3way\` fail, inspect the patch and apply it manually (edit the
files to match the patch's intent), then \`add -A\` + commit. The applied source is NOT guaranteed to
match the patch verbatim after a hand-merge, so after committing, RE-RUN the COMMANDMENT correctness
check (cd ${CANONICAL} && the COMMANDMENT CORRECTNESS cmd via gpu_lock); only report committed=true if
it still passes. (When a clean \`git apply\`/\`--3way\` succeeds, correctness was already verified and a
re-check is not required.) Return JSON {committed, current_best_diff, note}.`,
      { phase: 'Merge', label: `commit r${round}`, schema: COMMIT_SCHEMA });
    cumulative = winner.geomean;
    bestPerCase = winner.per_case && winner.per_case.length ? winner.per_case : bestPerCase;
    finalWinner = winner;
    roundsCommitted += 1;

    // --- (f) Re-profile the new best ------------------------------------
    profileSummary = await agentT(
      roleAgent('profile_engineer', 'reprofile', 'Re-profile the new best and explain the bottleneck shift.', {
        WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL, ROUND: round,
        COMMANDMENT, PREVIOUS_METRICS: profileSummary,
      }),
      { phase: 'Optimize', label: `reprofile r${round}`, schema: PROFILE_SCHEMA });
  }

  if (winner && winner.geomean > bestSeen) bestSeen = winner.geomean;
  if (madeProgress || improved) { noImprove = 0; } else { noImprove++; }

  // --- update cross-round memory (insight blackboard + hypothesis ledger)
  const mem = await agentT(
    roleAgent('tech_lead', 'update_memory', 'Distill durable insights + update the hypothesis ledger.', {
      EVAL_DIR, ROUND: round, SKILL_DIR: WORKFLOW_DIR,
      ROUND_RESULTS: clean.map(r => ({ id: r.d.id, title: r.d.title, specialty: r.d.specialty,
        expected: r.d.expected_speedup, claimed: r.eng ? r.eng.speedup_geomean : 0,
        verified: r.ver ? r.ver.verified_geomean : 0, status: r.ver ? r.ver.status : (r.eng ? r.eng.status : 'none'),
        notes: r.eng ? r.eng.notes : '' })),
      INTEGRATE: integrate, WINNER: winner ? { source: winner.source, geomean: winner.geomean } : null,
      IMPROVED: improved, REPROFILE_SHIFT: profileSummary ? profileSummary.shift_note : '',
      PRIOR_HISTORY: history,
      ...(STATE_DIR ? { STATE_DIR, CANONICAL, CUMULATIVE_SPEEDUP: cumulative, BEST_PER_CASE: bestPerCase } : {}),
      ...(SHARED_KB ? { SHARED_KB, TARGET_LANGUAGE } : {}),
    }),
    { phase: 'Optimize', label: `tech_lead:memory r${round}`, schema: MEMORY_SCHEMA });
  if (mem) {
    if (mem.insights) history.insights = mem.insights;
    if (mem.ledger) history.ledger = history.ledger.concat(mem.ledger);
    if (mem.bottleneck_now) history.bottleneck_now = mem.bottleneck_now;
    if (mem.suggest_next) history.suggest_next = mem.suggest_next;
  }
  // Enforce the round's KB budget. Honest about what this can and cannot do: stripping a citation
  // does NOT un-influence a direction that was already planned with a card in mind — only the prompt
  // above can do that. What the strip buys is OBSERVABILITY: `kbCapBound` counts the rounds where the
  // planner overran its budget, so "the KB quietly took over planning" becomes a number in the run's
  // return value instead of something you infer from a speedup that got worse.
  if (USE_LEARNED_READ) {
    let seeded = 0; const stripped = [];
    for (const d of directions) {
      if (!d.learned_refs || !d.learned_refs.length) continue;
      if (seeded < KB_DIR_CAP) { seeded++; continue; }
      stripped.push(d.id); d.learned_refs = [];
    }
    // A round where EVERY direction is KB-seeded has no control in it at all.
    if (KB_COLD_DIRECTION && directions.length && directions.every(d => (d.learned_refs || []).length)) {
      const last = directions[directions.length - 1];
      stripped.push(`${last.id}(forced cold)`); last.learned_refs = [];
    }
    if (stripped.length) {
      kbCapBound++;
      log(`Round ${round}: KB budget ${KB_DIR_CAP} exceeded — un-cited ${stripped.join(', ')} ` +
          `(the directions still run). Rounds where the cap bound so far: ${kbCapBound}.`);
    }
  }

  // Join the planner's declared citations against what the verifier independently measured. Derived
  // here rather than self-reported: the planner said which card seeded which direction, the verifier
  // re-measured that direction without knowing, and their join is a fact neither one could fake.
  // `became_winner` and not `verified > 1.0` is the standing test — verified_geomean is measured
  // against the FROZEN baseline, so once a kernel sits at 2.5x cumulative every non-regressing
  // direction clears 1.0 and a card would accrue credit for advancing nothing.
  for (const r of clean) {
    for (const cardRef of (r.d.learned_refs || [])) {
      citations.push({
        card: cardRef, round, direction: r.d.id, specialty: r.d.specialty,
        // null, NOT 0, when the verifier produced nothing. `kb.py` scores <=1.0 as a real loss, so
        // encoding "no evidence" as 0 charged a card for a verifier that crashed or timed out.
        cited_then_verified: r.ver && Number.isFinite(r.ver.verified_geomean)
          ? r.ver.verified_geomean : null,
        // `winner` is only the best candidate OF THIS ROUND; `improved` is whether it beat the
        // incumbent by MIN_IMPROVE. Crediting on `winner` alone let a 1.5x round winner confirm a
        // card while the committed best already sat at 2x — a confirmation for advancing nothing,
        // which is precisely what the three-state scoring was written to avoid. Require both.
        became_winner: !!(winner && winner.id === r.d.id && improved),
      });
    }
  }
  history.rounds.push({
    round,
    directions: directions.map(d => ({ id: d.id, title: d.title, specialty: d.specialty,
      learned_refs: d.learned_refs || [] })),
    results: clean.map(r => ({ id: r.d.id, claimed: r.eng ? r.eng.speedup_geomean : 0,
      verified: r.ver ? r.ver.verified_geomean : 0, status: r.ver ? r.ver.status : (r.eng ? r.eng.status : 'none') })),
    integrate: integrate ? { conclusion: integrate.conclusion, geomean: integrate.best ? integrate.best.geomean : 0 } : null,
    winner: winner ? { source: winner.source, geomean: winner.geomean } : null,
    improved, cumulative,
  });
  log(`Round ${round} done. winner=${winner ? winner.source + ' ' + winner.geomean.toFixed(2) + 'x' : 'none'}, cumulative=${cumulative.toFixed(2)}x, noImprove=${noImprove}`);

  // Issue #429: reclaim superseded heavy copies after each round, then CONTINUE optimizing.
  // Disk pressure must never abort the loop — reclaim (and under pressure, --force-heavy) only.
  // Agents (not this JS runtime) execute the bash; the prompt below is also mirrored into the
  // next round's tech_lead context via STORAGE_NOTE so it is not dropped on a dry orchestrator.
  const reclaimCmd =
    `bash ${WORKFLOW_DIR}/scripts/reclaim_eval_artifacts.sh --eval-dir ${EVAL_DIR} --keep-round ${round}` +
    `; bytes=$(du -sb ${EVAL_DIR} 2>/dev/null | awk '{print $1}');` +
    ` soft=\${GEAK_EVAL_SOFT_BYTES:-34359738368}; hard=\${GEAK_EVAL_HARD_BYTES:-68719476736};` +
    ` if [ -n "$bytes" ] && [ "$bytes" -gt "$hard" ]; then` +
    `   bash ${WORKFLOW_DIR}/scripts/reclaim_eval_artifacts.sh --eval-dir ${EVAL_DIR} --keep-round ${round} --force-heavy;` +
    ` elif [ -n "$bytes" ] && [ "$bytes" -gt "$soft" ]; then` +
    `   bash ${WORKFLOW_DIR}/scripts/reclaim_eval_artifacts.sh --eval-dir ${EVAL_DIR} --keep-round ${round};` +
    ` fi` +
    `; echo STORAGE_RECLAIM_DONE round=${round}`;
  if (ABL_B6) {
    ABL_PENDING.push(reclaimCmd);
    log(`  [B6] storage reclaim r${round} folded into the next agent — no dedicated dispatch.`);
  } else
  await agentT(
    `Storage reclaim after round ${round} (issue #429). Run EXACTLY this bash, then return ` +
    `{ok:true, note:"reclaimed"}. Do NOT stop optimizing — reclaim frees disk so later rounds can run.\n` +
    `\`\`\`bash\n${reclaimCmd}\n\`\`\``,
    { phase: 'Optimize', label: `storage:reclaim r${round}`,
      schema: { type: 'object', additionalProperties: true,
        properties: { ok: { type: 'boolean' }, note: { type: 'string' } },
        required: ['ok'] } });
}

if (MEGA) {
  const r = await megaAfterLoop();
  if (r) return r.exit;
}

// ===========================================================================
// PHASE: Final report (TechLead)
// ===========================================================================
phase('Report');
const report = MEGA ? await megaReport() : await agentT(
  roleAgent('tech_lead', 'report', 'Write the final report and the cumulative final patch.', {
    EVAL_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR,
    HISTORY: history, FINAL_WINNER: finalWinner, BASELINE_PER_CASE,
    BASELINE_GEOMEAN_MS, CUMULATIVE_SPEEDUP: cumulative,
  }),
  { phase: 'Report', label: 'tech_lead:report', schema: REPORT_SCHEMA });
if (MEGA) {
  const r = await megaAfterReport();
  if (r) return r.exit;
}

// ===========================================================================
// PHASE: Director validation + arbitration
// ===========================================================================
phase('Validate');
const validation = MEGA ? await megaValidate() : await agentT(
  roleAgent('director', 'validate', 'Independently validate the final patch vs the TRUE baseline.', {
    KERNEL_PATH_ORIG, EVAL_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL,
    APPLY_TO_ORIGINAL, COMMANDMENT, FROZEN_ORACLE: FROZEN_ORACLE ? 'true' : 'false',
    FINAL_PATCH: report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
    TECH_LEAD_REPORTED_GEOMEAN: report ? report.final_speedup_geomean : cumulative,
    ...(HAS_WORKLOAD && report && report.final_speedup_weighted != null
        ? { TECH_LEAD_REPORTED_WEIGHTED: report.final_speedup_weighted } : {}),
    BASELINE_TIMING: BASELINE_PER_CASE,
  }),
  { phase: 'Validate', label: 'director:validate', schema: VALIDATE_SCHEMA });

const finalGeomean = validation ? validation.director_verified_speedup_geomean : cumulative;
// PRIMARY headline: the time-weighted speedup when workload-aligned, else the geomean (unchanged).
const finalWeighted = validation && validation.director_verified_speedup_weighted != null
  ? validation.director_verified_speedup_weighted : null;
if (MEGA) return await megaFinish();
const finalPrimary = HAS_WORKLOAD && Number.isFinite(finalWeighted) ? finalWeighted : finalGeomean;
log(`COMPLETE. ${KERNEL_NAME}: verified ${HAS_WORKLOAD ? 'time-weighted' : 'geomean'} ${finalPrimary ? finalPrimary.toFixed(2) : '?'}x` +
    `${HAS_WORKLOAD && Number.isFinite(finalGeomean) ? ` (unweighted geomean ${finalGeomean.toFixed(2)}x)` : ''}` +
    ` (status ${validation ? validation.validation_status : '?'}). Results in ${EVAL_DIR}`);

// ===========================================================================
// PHASE: UpdateExperience — distill ONE reusable card into this workflow's knowledge/learned/
// (see knowledge/learned/README.md). Runs ONCE, at the very end: never per round, because only
// the final director-verified speedup is evidence a card may cite. `validation` is required for
// the same reason — without it `finalPrimary` falls back to the TechLead's own unverified
// `cumulative`. ADD-only, so a skipped or failing step leaves the run byte-neutral.
// ===========================================================================
let learned_card = null;
// Two runs must never become a card, and neither is a failure. A run that shared its GPU measured
// contention, not the kernel. A run on a HELD-OUT kernel is the instrument for measuring whether the
// KB helps at all — distil it and the next A/B over that kernel reads back its own answer and looks
// spectacular. Defaults are permissive, so the CALLER is the one who has to declare a busy box.
const BOX_QUIET = String(A.box_quiet != null ? A.box_quiet : 'true') === 'true';
const HELD_OUT = String(A.held_out != null ? A.held_out : 'false') === 'true';
const kbGate = !BOX_QUIET ? 'the box was contended — the number measured neighbours, not the kernel'
  : HELD_OUT ? 'this kernel is in the held-out split and is the measuring instrument'
  : '';
if (kbGate) log(`[kb] not distilling: ${kbGate}.`);
// `validation.validation_status === 'accepted'` and not merely "a validation object exists": a
// `flagged` result means the director found a reason not to believe the number (patch install
// no-op, correctness fail, contended box), and curating from it teaches the next run a lesson this
// run did not earn. Reported in review of #411.
const kbAccepted = String((validation && validation.validation_status) || '').toLowerCase() === 'accepted';
if (!kbGate && UPDATE_EXPERIENCE_ON && kbAccepted && Number.isFinite(finalPrimary) && finalPrimary > 1.0) {
  try {
    learned_card = await agentT(
      roleAgent('update_experience', 'Validate',
        'Curate one distilled learned card from this lane\'s verified win (ADD-only, measured evidence, ' +
        'ratios not wall-clock; record the pitfalls hit; total-then-per-direction for a stacked win).', {
          SCOPE: 'lane', LEARNED_DIR, SKILL_DIR: WORKFLOW_DIR, EVAL_DIR,
          PERF_KNOWLEDGE_DIR: KERNEL_KNOWLEDGE_DIR,
          CURATION_ISOLATE: RDNA4_ISOLATE ? 'true' : 'false',
          REQUIRED_PLATFORMS: RDNA4_ISOLATE ? ['gfx1201'] : [],
          WINNER: {
            kernel: KERNEL_NAME, language: TARGET_LANGUAGE, mode: MODE, gfx: GFX,
            device_target: DEVICE_TARGET, physical_cu_count: PHYSICAL_CU_COUNT,
            kernel_class: (analysis && analysis.kernel_type) || '',
            speedup: finalPrimary, validation_status: validation ? validation.validation_status : '',
            bottleneck: profileSummary ? profileSummary.bottleneck : '',
          },
          // `rounds[]` (per-round directions + per-candidate claimed/verified/status + running
          // `cumulative`) is what makes the card's `stack:` attribution and `pitfall:` lines derivable.
          HISTORY: history,
          // What this run's plan cited and what the verifier then measured. CONTEXT ONLY — the
          // counters are applied by `kb.py drain` from the ledger filed below, never by this agent.
          // The comment that used to sit here said the curator merges them, which stopped being
          // true when that arithmetic moved into code.
          CITATIONS: citations,
          PROFILE: profileSummary,
          REPORT_PATH: report && report.report_path ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`,
        }),
      { phase: 'Validate', label: 'update_experience', schema: UPDATE_EXPERIENCE_SCHEMA });
    if (learned_card && learned_card.card_path) {
      log(`[kb] learned card ${learned_card.action || 'written'}: ${learned_card.card_path}`);
    }
  } catch (e) {
    log(`[kb] update_experience skipped: ${e && e.message ? e.message : e}`);
  }
}

// FILE THE CITATION LEDGER — OUTSIDE the curation gate, and deliberately so.
// This is the loss half of the loop. It first shipped INSIDE the `finalPrimary > 1.0` branch above,
// which is the exact bias it exists to remove: the runs that produce losses are the ones that did
// not beat the baseline, were flagged, or ran on a contended box, and every one of them was
// filtered out before it could file anything. Selecting successful runs into the feedback data is
// worse than no feedback, because the resulting counters look earned. Reported in review of #411.
// Curation (writing a NEW card) still requires a verified win — a run with nothing to teach must
// not teach — but the ledger of what the cards ALREADY in the tree did is owed for every run.
// `held_out` is the one exception: that split is the measuring instrument, so it neither writes
// cards nor votes on them.
// A workflow script has no filesystem, hence the one-line agent; the JSON is assembled by
// `kb.py cite` rather than by the agent, because a proposal hand-written by a model is one more
// place for the schema to drift.
if (citations.length && !HELD_OUT) {
  try {
    await agentT(
      `Run EXACTLY this command and nothing else. Do NOT edit any file.
\`\`\`bash
cat <<'CITEJSON' | python3 ${WORKFLOW_DIR}/scripts/kb.py --kb-dir ${LEARNED_DIR} cite \\
--run-id ${JSON.stringify(KERNEL_NAME)} \\
--kernel ${JSON.stringify(KERNEL_NAME)} --citations -
${JSON.stringify(citations)}
CITEJSON
\`\`\`
Return {"filed": <the "citations" number the command printed, or 0>}.`,
      { phase: 'Validate', label: 'kb:cite', effort: 'low',
        schema: { type: 'object', properties: { filed: { type: 'number' } },
                  required: ['filed'], additionalProperties: true } });
    log(`[kb] citation ledger filed: ${citations.length} row(s) for the next drain`);
  } catch (e) {
    log(`[kb] citation ledger NOT filed: ${e && e.message ? e.message : e}`);
  }
}

// ===========================================================================
// Write this run's outcome back — the producer half of the loop. One call files all three: the
// kb_artifacts/ tree (which the remote payload is derived FROM, and which a curation pass edits),
// the on-disk store the read addresses by key, and the service when credentialed.
// The script applies its own gate and never fails the run; the `finalPrimary > 1.0`
// pre-check just avoids spending an agent on a run that cannot pass the gate anyway.
// ===========================================================================
let kb_written = null;
if (KB_WRITE_OK && GFX && Number.isFinite(finalPrimary) && finalPrimary > 1.0) {
  const kernelClass = (analysis && analysis.kernel_type) || 'unknown';
  const finalPatch = report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`;
  const reportPath = report && report.report_path ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`;
  // What the store cannot infer from a diff but needs to curate this entry later. `direction` is the
  // IDEA, and only a warm start supplies a real one: a round title is free-form per run, so kebabbing
  // it would mint a unique label every time and group nothing. Unlabeled is honest — resolve treats
  // such an entry as its own direction, and a curation pass assigns the shared label later.
  const winnerDirection = kebab(warm_start.direction || '');
  const metricKind = HAS_WORKLOAD ? 'time_weighted' : 'geomean';
  // Commas separate the list, so a case name may not contain one.
  const caseNames = (bestPerCase || []).map(c => c && String(c.name || '').replace(/,/g, ';'))
    .filter(Boolean).join(',');
  // write-remote runs the directory write first and files the same entry in the store, so the two
  // planes cannot drift; its extra `remote` field rides along under the schema's open object.
  // `--plane both` extends that ordering one hop further: directory tree, then local store, then the
  // service. Unlike the read there is no branch on credentials here, because `_open_plane` already
  // makes the right call — a `both` without them degrades to the local planes and reports
  // `remote_unavailable` rather than failing, which is the behaviour we would have written by hand.
  const remoteWriteOn = KB_REMOTE !== 'off' && !!KB_STORE_DIR;
  const writeCmd = remoteWriteOn
    ? `write-remote --plane both --store ${JSON.stringify(KB_STORE_DIR)}${KB_VERSION_FLAG}`
    : KB_MODE === 'store'
      ? `write-remote --plane local --store ${JSON.stringify(KB_STORE_DIR)}${KB_VERSION_FLAG}`
      : 'write';
  kb_written = await agentT(
    `You are the experience writer. Run EXACTLY this command (it applies its own gates and prints a ` +
    `single-line JSON) and return that JSON verbatim as StructuredOutput. If the command errors, return ` +
    `{"written": false, "reason": "io_error"}` +
    (remoteWriteOn
      ? ` — do NOT retry it and do NOT adjust its arguments. It may reach the shared KB Store service, ` +
        `and every write that service accepts is PERMANENT (it exposes no delete), so a second attempt ` +
        `creates a second permanent record instead of fixing the first.`
      : `.`) + `
\`\`\`bash
${remoteWriteOn ? KB_ENV_PRELUDE + '\n' : ''}python3 ${JSON.stringify(EXPERIENCE_STORE)} ${writeCmd} --root ${JSON.stringify(KB_ARTIFACTS_DIR)} \\
  --kernel-name ${JSON.stringify(KERNEL_NAME)} --language ${JSON.stringify(TARGET_LANGUAGE)} \\
  --gfx ${GFX} --kernel-class ${JSON.stringify(kernelClass)} \\
  --speedup ${finalPrimary} --baseline-wall-ms ${BASELINE_GEOMEAN_MS} \\
  --patch ${JSON.stringify(finalPatch)} --eval-dir ${JSON.stringify(EVAL_DIR)} \\
  --report ${JSON.stringify(reportPath)} --metric-kind ${metricKind} \\
  --direction ${JSON.stringify(winnerDirection)} --case-names ${JSON.stringify(caseNames)}\
${OP_SPEC.dtype ? ` \\\n  --precision ${JSON.stringify(String(OP_SPEC.dtype))}` : ''}\
${warm_start.exp_dir ? ` \\\n  --parent ${JSON.stringify(warm_start.exp_dir)}` : ''}
\`\`\``,
    { phase: 'Validate', label: 'kb:write', schema: WARMSTART_WRITE_SCHEMA });
  const remoteWrite = (kb_written && kb_written.remote) || null;
  if (remoteWrite) {
    log(remoteWrite.written
      // `replaced` distinguishes the two write outcomes the store has: a NEW patch appends a session
      // under the same key, the SAME patch remeasured lands back on its own content-addressed one.
      ? `[kb] plane=store ${remoteWrite.replaced ? 'updated' : 'appended'} ` +
        `${remoteWrite.canonical_id}/${remoteWrite.session_id}${remoteWrite.champion ? ' (champion)' : ''}`
      : `[kb] plane=store not written: ${remoteWrite.reason || 'unknown'}`);
  }
  log(kb_written && kb_written.written
    ? `[kb] experience written: ${kb_written.slug} (speedup ${finalPrimary.toFixed(2)}, direction ` +
      `${winnerDirection || 'unlabeled'})`
    : kb_written && kb_written.reason === 'duplicate_impl'
      // Not a failure — the expected outcome of adopting a warm start and not beating it.
      ? `[kb] experience already known — counted as reproduction #${kb_written.reproductions || '?'} of ` +
        `${kb_written.reproduced || kb_written.dir}`
      : `[kb] experience not written: ${kb_written ? kb_written.reason : 'writer returned nothing'}`);
}

// finalPrimary is the total vs the pristine baseline; when a warm-start patch was adopted, split out
// the delta ABOVE it so a KB-derived gain is never reported as this run's own work.
//
// With no round committed after the adopt, the workspace still holds exactly the adopted code, and
// the two numbers are two measurements of the SAME thing taken at different times — the ratio is
// bench noise (~2% on this box; `_gemm_a16_w16_kernel` reported 0.98 and read as a regression).
// Report 1.0 and say why, rather than dressing the noise up as a delta.
const noRoundsAfterAdopt = !!(warm_start.adopted && roundsCommitted === 0);
const incrementalSpeedup = warm_start.adopted && warm_start.adopted_speedup
  ? (noRoundsAfterAdopt ? 1.0
     : Number.isFinite(finalPrimary) ? finalPrimary / warm_start.adopted_speedup : null)
  : finalPrimary;

return {
  mode: MODE,
  // `instance` = this lane's eval dir: a stable identity so the parser dedupes a timeline reached
  // twice (parent-merge + glob) without collapsing two distinct lanes that share a shape.
  llm_timeline: LLM_STATS ? { ...LLM_TL, instance: EVAL_DIR } : undefined,
  target_language: MODE === 'author' ? TARGET_LANGUAGE : undefined,
  authored: MODE === 'author' ? true : undefined,
  eval_dir: EVAL_DIR,
  kernel_name: KERNEL_NAME,
  workload_aligned: HAS_WORKLOAD,
  final_speedup: finalPrimary,                 // PRIMARY metric (weighted when workload-aligned)
  final_weighted: finalWeighted,
  final_geomean: finalGeomean,
  final_arithmetic: validation ? validation.director_verified_speedup_arithmetic : null,
  tech_lead_reported_geomean: report ? report.final_speedup_geomean : cumulative,
  validation_status: validation ? validation.validation_status : 'unknown',
  // What KIND of number `final_speedup` is, so bake-off ranking and campaign summaries don't read a
  // host_bound/unprimed float as a clean device-time win. 'not_applicable' = this lane's own baseline
  // ratio, no frozen-oracle receipt behind it.
  timing_basis: validation ? validation.timing_basis || 'unknown' : 'unknown',
  rounds: report ? report.rounds : round,
  budget_used: dispatched,
  budget_total: BUDGET,
  report_path: report ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`,
  final_patch: report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
  // null when the step was off, or the run earned nothing worth a card.
  // Which condition actually ended the loop. RECORDED, not inferred: on the previous campaign this
  // existed only in the driver's return value, which the caller sees after ALL its kernels finish —
  // so a parent-process death erased it for every kernel already done, and the realised window had
  // to be reconstructed from queue timestamps. It is the whole point of an enforced window that you
  // can tell a run that used its budget from one that gave up.
  stopped_by: deadlineHit ? 'deadline'
    : (dispatched >= BUDGET) ? 'budget'
    : (noImprove >= MAX_NO_IMPROVE) ? 'no_improve'
    : (plannerStopReason || 'tech_lead_stop'),
  deadline_hit: deadlineHit,
  forced_replans: forcedReplans,
  // What the plan cited and whether it carried its round. Returned ALWAYS, including when nothing was
  // cited (an empty array is the finding: the KB was read and nothing in it was worth acting on).
  learned_citations: citations,
  // Rounds where the planner overran the KB budget. A KB that binds every round is one that has taken
  // over planning, which is the regression this budget exists to catch.
  kb_cap_bound_rounds: kbCapBound,
  // The monoculture canary, and it costs no GPU time: how many DISTINCT (specialty, focus_files)
  // directions this run explored vs how many it issued. A KB that helps raises the verified speedup;
  // a KB that cages lowers this without raising that. Reported on KB-less runs too — those are the
  // baseline it has to be read against.
  direction_entropy: (() => {
    const seen = new Set(); let n = 0;
    for (const r of history.rounds) for (const d of (r.directions || [])) {
      n++; seen.add(`${d.specialty}|${(d.focus_files || []).slice().sort().join(',')}`);
    }
    return n ? { distinct: seen.size, issued: n, ratio: +(seen.size / n).toFixed(3) } : null;
  })(),
  learned_card: learned_card && learned_card.card_path
    ? { action: learned_card.action || '', card_path: learned_card.card_path, key: learned_card.key || '' }
    : null,
  // Warm-start (local experience KB) outcome. adopted=false + read_reason on a cold start.
  warm_start: {
    adopted: warm_start.adopted,
    read_reason: warm_start.read_reason,
    slug: warm_start.slug || '',
    // How the store was reached and what it filtered out: tells a genuine cold start from a
    // name/language mismatch that only looks like an empty KB.
    match_tier: warm_start.match_tier || '',
    filtered: warm_start.filtered || null,
    adopted_speedup: warm_start.adopted ? warm_start.adopted_speedup : null,
    total_speedup: Number.isFinite(finalPrimary) ? finalPrimary : null,
    incremental_speedup: incrementalSpeedup,
    // true = the run adopted a stored patch and never improved on it; incremental is 1.0 by
    // definition, not by measurement.
    no_rounds_after_adopt: noRoundsAfterAdopt,
    rounds_committed: roundsCommitted,
    incremental_improved: !!(warm_start.adopted && Number.isFinite(incrementalSpeedup) && incrementalSpeedup > 1 + MIN_IMPROVE),
    returned_after_read_kb: !!warm_start.returned_after_read_kb,
    candidates: warm_start.candidates,
  },
  kb_written: kb_written && kb_written.written ? { slug: kb_written.slug, dir: kb_written.dir } : null,
};

