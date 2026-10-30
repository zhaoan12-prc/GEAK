export const meta = {
  name: 'kernel-fusion',
  description: 'KernelFusion stage, run BEFORE the GEAK e2e workflow: capture a serving trace (GEAK profiler), pick the Top-K fusion patterns with a user-supplied analysis prompt, author each fused kernel with the GEAK kernel layer, and apply the winners back through the GEAK e2e integrator A/B. Returns the accepted fused overlay, which the e2e workflow then uses as its baseline.',
  whenToUse: 'Started by interface/run_e2e.py when kernel fusion is switched on (handoff "kernel_fusion", GEAK_KERNEL_FUSION=1 or --kernel-fusion). Takes the same args as e2e_workflow.js plus kf_dir, kf_prompt and the optional kf_* knobs documented in kernel_fusion/README.md.',
  phases: [
    { title: 'Setup', detail: 'e2e Director builds the KF eval dir + baseline throughput (roles/director.md PHASE=setup)' },
    { title: 'Trace', detail: 'Profiler captures a warm serving trace (roles/profiler.md), or kf_trace_dir supplies one' },
    { title: 'TopK', detail: 'fusion analyst: kf_trace_digest.py + the user prompt -> ranked Top-K fusion candidates (topk.json)' },
    { title: 'Author', detail: 'per candidate: kernel_extractor builds the region unittest, kernel_lane.js mode=author writes the fused kernel' },
    { title: 'ApplyBack', detail: 'per authored kernel, serially: e2e_integrator overlay + ref/cand A/B; accepted overlays compound' },
  ],
};

// ---------------------------------------------------------------------------
// Args. Everything a GEAK role needs comes in under the SAME names e2e_workflow.js reads, because
// run_e2e.py builds both arg sets with one map_args(); only the kf_* keys are new.
// ---------------------------------------------------------------------------
const A = args || {};
const WORKFLOW_DIR = String(A.workflow_dir || '').replace(/\/+$/, '');   // e2e_workflow/
if (!WORKFLOW_DIR) throw new Error('args.workflow_dir is required (the e2e_workflow dir)');
const KF_DIR = String(A.kf_dir || '').replace(/\/+$/, '');
if (!KF_DIR) throw new Error('args.kf_dir is required (the KernelFusion eval dir)');
const KF_HOME = `${WORKFLOW_DIR}/kernel_fusion`;
const KERNEL_WF_DIR = String(A.kernel_workflow_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/kernel_workflow')).replace(/\/+$/, '');
const KERNEL_WF_SCRIPT = `${KERNEL_WF_DIR}/kernel_lane.js`;

const MODEL_PATH = String(A.model_path || '');
const LAUNCH_SCRIPT = String(A.launch_script || '');
const BACKEND = String(A.backend || 'sglang');
const GPU_IDS = String(A.gpu_ids != null ? A.gpu_ids : '0');
const GPU_LIST = GPU_IDS.split(',').map((s) => s.trim()).filter(Boolean);
const SERVING_TP = parseInt(A.tp != null ? A.tp : 1, 10);
const SERVING_GPU = String(A.serving_gpu != null ? A.serving_gpu
  : GPU_LIST.slice(0, Math.max(1, SERVING_TP)).join(',') || '0');
const WORKLOAD = { isl: parseInt(A.isl != null ? A.isl : 1024, 10), osl: parseInt(A.osl != null ? A.osl : 1024, 10),
  conc: parseInt(A.conc != null ? A.conc : 64, 10), shape_provenance: String(A.workload_shape_provenance || 'handoff_workload') };
const INIT_FLAGS = String(A.initial_extra_server_args || '');
const INIT_ENV = String(A.initial_extra_env || '');
const INIT_OVERLAY = String(A.initial_overlay_pythonpath || '');
const EFFECTIVE_CONFIG_DIGEST = String(A.effective_config_digest || '');
const MEASUREMENT_MODE = String(A.measurement_mode || 'warm_server');
const PARITY_REPLICAS = parseInt(A.parity_replicas != null ? A.parity_replicas : 1, 10);
const SEARCH_REPLICAS = parseInt(A.search_replicas != null ? A.search_replicas : 1, 10);
const NOISE_BAND_DEFAULT = parseFloat(A.noise_band_pct != null ? A.noise_band_pct : 0.5);
const AB_FINISH_RETRIES = parseInt(A.ab_finish_retries != null ? A.ab_finish_retries : 2, 10);

// KernelFusion knobs.
const PROMPT = String(A.kf_prompt || `${KF_HOME}/default_prompt.md`);
const TRACE_DIR_IN = String(A.kf_trace_dir || '');          // existing traces: skip the capture
const WORKLOAD_SUMMARY = String(A.kf_workload_summary || ''); // per-point weights, when the caller has them
const SOURCE_DIRS = String(A.kf_source_dirs || '');           // framework source, when not importable here
const STEPS = new Set(String(A.kf_steps || 'trace,topk,author,apply').split(',').map((s) => s.trim()).filter(Boolean));
const want = (s) => STEPS.has(s);
const AUTHOR_K = parseInt(A.kf_author_k != null ? A.kf_author_k : 3, 10);
const KERNEL_BUDGET = parseInt(A.kf_kernel_budget != null ? A.kf_kernel_budget : 4, 10);
const AGENT_TIMEOUT_MS = parseInt(A.agent_timeout_ms != null ? A.agent_timeout_ms : 7200000, 10);

// Stop STARTING author/apply work once 85% of the stage budget is gone; in-flight work finishes and the
// stage returns whatever was accepted. Inert without time_budget_s.
const TIME_BUDGET_MS = A.time_budget_s ? parseInt(A.time_budget_s, 10) * 1000 : null;
let DEADLINE_HIT = false;
if (TIME_BUDGET_MS && typeof setTimeout === 'function') {
  setTimeout(() => { DEADLINE_HIT = true; log('[kf] 85% of the stage budget spent: no new author/apply work starts.'); },
    Math.floor(TIME_BUDGET_MS * 0.85));
}

// Same deployment requirement e2e_workflow.js injects into every kernel task (derived the same way).
const GRAPH_REQ = ((BACKEND === 'vllm' || BACKEND === 'sglang') && !/enforce[-_]eager/i.test(INIT_FLAGS))
  ? ' DEPLOYMENT REQUIREMENT: the server captures decode into a CUDA/HIP graph, so the fused kernel runs inside ' +
    'that graph. It must be graph-capture-safe (no .item()/.cpu()/.tolist()/synchronize(), no Python branch on a ' +
    'GPU scalar, shape-stable, weight prep cached once by data_ptr) and its speedup must be intrinsic (fewer ' +
    'launches / less HBM traffic inside the captured region), never a per-call graph-replay wrapper.'
  : '';

// ---------------------------------------------------------------------------
// Agent plumbing: the e2e_workflow.js roleAgent/safeAgent contract, trimmed to what this stage uses.
// ---------------------------------------------------------------------------
const obj = (props, required) => ({ type: 'object', properties: props, required: required || [], additionalProperties: true });
const arrObj = { type: 'array', items: { type: 'object', additionalProperties: true } };
const arrStr = { type: 'array', items: { type: 'string' } };
const cfg = (o) => Object.entries(o).map(([k, v]) => `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');

const PROCESS_SAFETY = `## PROCESS SAFETY (a violation can kill the caller's orchestrator, failing the whole task)
NEVER run global or pattern-matched process cleanup (no \`pkill -f\`, \`killall\`, \`ps | grep | xargs kill\`,
\`kill -- -PGID\` for a group you did not create). Manage ONLY processes you started, by the pid you captured.
For any server you launch yourself: \`source ${WORKFLOW_DIR}/scripts/server_teardown.sh; trap server_teardown EXIT\`,
launch with \`\${SERVER_LAUNCH_PREFIX:-} <server> &\`, then \`server_record_identity "$!"\`.

`;

function roleAgent(role, phase, intro, inputs) {
  const inall = { BACKEND, SERVING_TP, SERVING_GPU, MEASUREMENT_MODE, PARITY_REPLICAS, SEARCH_REPLICAS,
    EFFECTIVE_CONFIG_DIGEST, ...inputs };
  return `You are the ${role}. PHASE=${phase}.
First Read ${WORKFLOW_DIR}/roles/${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${WORKFLOW_DIR}/knowledge/.
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}
You are running inside the GEAK KernelFusion stage: EVAL_DIR is the KernelFusion eval dir, and everything
the role file says about EVAL_DIR applies to it unchanged.

## SERVING CONFIG INVARIANT
Every e2e serving launch (baseline, profile, integrate ref/cand) uses BACKEND=${BACKEND} TP=${SERVING_TP}
GPU=${SERVING_GPU} in the bench_e2e.sh env. GPU_IDS=${GPU_IDS} is only for single-GPU isolated work.

## Inputs
${cfg(inall)}

Return ONLY the structured JSON the role file specifies.`;
}

function bounded(prompt, opts) {
  const p = agent(PROCESS_SAFETY + prompt, opts);
  if (typeof setTimeout !== 'function' || !(AGENT_TIMEOUT_MS > 0)) return p;
  let to;
  const guard = new Promise((resolve) => { to = setTimeout(() => resolve(null), AGENT_TIMEOUT_MS); });
  return Promise.race([p.then((r) => { clearTimeout(to); return r; }, (e) => { clearTimeout(to); throw e; }), guard]);
}

async function safeAgent(prompt, opts, tries = 3) {
  for (let i = 0; i < tries; i++) {
    try {
      const r = await bounded(prompt, opts);
      if (r) return r;
    } catch (e) { log(`agent[${opts.label}] attempt ${i + 1}/${tries}: ${String(e).slice(0, 160)}`); }
  }
  log(`agent[${opts.label}] gave no result after ${tries} tries.`);
  return null;
}

const SETUP_SCHEMA = obj({ eval_dir: { type: 'string' }, baseline_throughput_tok_s: { type: 'number' },
  noise_band_pct: { type: 'number' }, server_flags: { type: 'object', additionalProperties: true },
  server_env: { type: 'string' }, notes: { type: 'string' } }, ['eval_dir', 'baseline_throughput_tok_s']);
const PROFILE_SCHEMA = obj({ profile_topN_json: { type: 'string' }, top_kernels: arrObj, notes: { type: 'string' } },
  ['profile_topN_json']);
const TOPK_SCHEMA = obj({
  topk_json: { type: 'string' }, report_path: { type: 'string' }, digest_md: { type: 'string' },
  candidates: arrObj, recommended: arrStr, notes: { type: 'string' },
}, ['topk_json', 'candidates', 'recommended']);
const EXTRACT_SCHEMA = obj({ short_name: { type: 'string' }, editable: { type: 'boolean' }, task_dir: { type: 'string' },
  target_callable: { type: 'string' }, unittest_smoke: { type: 'string' }, candidate_bind: { type: 'object', additionalProperties: true },
  baseline_frozen: { type: 'boolean' },
  notes: { type: 'string' } }, ['editable', 'task_dir', 'unittest_smoke']);
const INTEGRATE_SCHEMA = obj({ short_name: { type: 'string' }, e2e_throughput_tok_s: { type: 'number' },
  e2e_delta_pct: { type: 'number' }, ref_med: { type: 'number' }, cand_med: { type: 'number' },
  ab_complete: { type: 'boolean' }, output_parity: { type: 'string' }, parity_kind: { type: 'string' },
  gate: { type: 'string', enum: ['accepted', 'stack', 'rejected', 'incomplete'] },
  accepted_overlay: { type: 'string' }, reason: { type: 'string' } }, ['gate', 'e2e_throughput_tok_s']);

// ---------------------------------------------------------------------------
// Setup (GEAK director) -- needed whenever this stage touches a server or a GPU.
// ---------------------------------------------------------------------------
const needsServer = (want('trace') && !TRACE_DIR_IN) || want('author') || want('apply');
let EVAL_DIR = KF_DIR, baseTput = 0, noiseBand = NOISE_BAND_DEFAULT;
if (needsServer) {
  phase('Setup');
  const setup = await safeAgent(roleAgent('director', 'setup',
    'Build the isolated KernelFusion eval dir (use EVAL_DIR_OVERRIDE exactly) and record the baseline throughput.', {
      LAUNCH_SCRIPT, MODEL_PATH, EXP_ROOT: KF_DIR.replace(/\/[^/]*$/, ''), EVAL_DIR_OVERRIDE: KF_DIR,
      MODEL_NAME_HINT: MODEL_PATH.replace(/\/+$/, '').split('/').pop(), TASK: '', GPU_IDS, WORKLOAD,
      INIT_FLAGS, INIT_ENV, INIT_BASE_OVERLAY: INIT_OVERLAY, MEASUREMENT_PURPOSE: 'parity',
      REPLICAS: PARITY_REPLICAS, SKILL_DIR: WORKFLOW_DIR,
    }), { phase: 'Setup', label: 'kf:director:setup', schema: SETUP_SCHEMA });
  if (!setup || !setup.eval_dir) {
    return { eval_dir: KF_DIR, kf_dir: KF_DIR, status: 'error', reason: 'setup failed', accepted: [], final_overlay: INIT_OVERLAY };
  }
  EVAL_DIR = setup.eval_dir;
  baseTput = setup.baseline_throughput_tok_s || 0;
  noiseBand = setup.noise_band_pct || NOISE_BAND_DEFAULT;
  log(`[kf] setup: baseline ${baseTput} tok/s, noise band ${noiseBand}%`);
}

// ---------------------------------------------------------------------------
// Trace (GEAK profiler, unchanged) -- or the caller's traces.
// ---------------------------------------------------------------------------
let TRACE_DIR = TRACE_DIR_IN;
let profileTopN = '';
if (want('trace') && !TRACE_DIR_IN) {
  phase('Trace');
  const prof = await safeAgent(roleAgent('profiler', 'baseline',
    'Capture a warm serving trace and emit the standardized Top-N. Keep the raw per-rank trace files on disk.', {
      EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, ROUND: 0,
      OVERLAY_PYTHONPATH: INIT_OVERLAY, EXTRA_SERVER_ARGS: INIT_FLAGS, EXTRA_ENV: INIT_ENV, SKILL_DIR: WORKFLOW_DIR,
    }), { phase: 'Trace', label: 'kf:profiler', schema: PROFILE_SCHEMA });
  profileTopN = (prof && prof.profile_topN_json) || '';
  TRACE_DIR = `${EVAL_DIR}/profile`;
}

// ---------------------------------------------------------------------------
// TopK: deterministic digest + the user's analysis prompt.
// ---------------------------------------------------------------------------
let topk = null;
if (want('topk')) {
  phase('TopK');
  if (!TRACE_DIR) {
    return { eval_dir: KF_DIR, kf_dir: KF_DIR, status: 'error', reason: 'no trace: pass kf_trace_dir or enable the trace step',
      accepted: [], final_overlay: INIT_OVERLAY };
  }
  const OUT = `${EVAL_DIR}/analysis`;
  topk = await safeAgent(`You are the GEAK KernelFusion analyst. You choose which kernel-fusion patterns to implement.

## Your method is the user's prompt
Read ${PROMPT} in full. It is the AUTHORITATIVE analysis method, ranking rules and deliverable list for this
task: follow it step by step. It may name a container, a working directory or a materials/ layout from the
setting it was written in; those are NOT this environment. Map them onto the inputs below instead:
- its traces / sampling points            -> the trace files under TRACE_DIR (any depth; one dir = one point)
- its workload summary / weights           -> WORKLOAD_SUMMARY when given; otherwise derive the weights from the
                                              baseline bench + server log under EVAL_DIR and say so
- its deploy/server args + env             -> SERVER_FLAGS / SERVER_ENV / LAUNCH_SCRIPT (and EVAL_DIR/baseline/server.log)
- its model config                         -> MODEL_PATH/config.json
- framework + operator-library source      -> ${SOURCE_DIRS ? `SOURCE_DIRS (${SOURCE_DIRS})` : `the installed packages (python3 -c "import ${BACKEND}; print(${BACKEND}.__file__)", aiter, ...)`}
- its out/ directory                       -> ${OUT}/
Where the prompt says not to write kernels, that is right for you too: the NEXT step of this stage authors
your recommended candidates with the GEAK kernel layer and A/B-tests them on the live server.

## Do this first (deterministic, reproducible)
    python3 ${KF_HOME}/kf_trace_digest.py --trace-dir "${TRACE_DIR}" --out "${OUT}/digest"
    # optional per-kernel Top-N of any one trace file:  python3 ${WORKFLOW_DIR}/scripts/parse_profile.py --torch-trace <file> --top 40 --out ${OUT}/topn_<point>
${OUT}/digest/digest.md has, per trace and per step kind: steps, busy/span/idle per step, kernels/step, short-kernel
share, the per-layer repeating block, the top kernels, and seq_*.tsv (the median step's full ordered kernel
sequence with gaps and, for eager launches, the launching cpu_op + framework frame). Use these numbers; quote
them in your report. Graph-replayed steps have no launch frames: map those kernels to source by name.

## Machine-readable result (in addition to everything the prompt asks for)
Write ${OUT}/topk.json: {"candidates": [...], "recommended": [ids in the order to implement]}, every candidate:
  id (short, [A-Za-z0-9_]), rank, level ("L1".."L4"), region (one line), kernels (device kernel names, in order),
  phases (step kinds it runs in), occurrences {per_layer, per_step, layers}, m_range,
  seam {target_callable: "pkg.module:Class.method or pkg.module:function" -- ONE Python callable whose body
        covers the whole region, so the region can be rebound to a fused implementation; source_file; lines},
  inputs_outputs (tensors that cross the region boundary, with shapes/dtypes), launches_before, launches_after,
  region_pct_device_time (weighted), est_gain_pct_upper, est_gain_pct_expected (weighted e2e %),
  difficulty (1-5), bitwise_exact (bool), exclusive_with [ids], language ("triton" unless you have a reason),
  fused_design (how the fused kernel works: tiling, what stays in registers/LDS, what it no longer writes to HBM).
"recommended" = the non-conflicting set the prompt's ranking recommends, best first. A candidate whose region
cannot be reached through one Python seam can still be ranked, but must not be recommended.

## Inputs
${cfg({ EVAL_DIR, TRACE_DIR, WORKLOAD_SUMMARY: WORKLOAD_SUMMARY || '(none)', PROFILE_TOPN: profileTopN || '(none)',
    MODEL_PATH, BACKEND, SERVING_TP, SERVER_FLAGS: INIT_FLAGS, SERVER_ENV: INIT_ENV, LAUNCH_SCRIPT: LAUNCH_SCRIPT || '(none)',
    WORKLOAD, OVERLAY: INIT_OVERLAY || '(none)' })}

Return JSON: {"topk_json": "${OUT}/topk.json", "report_path": "<main report the prompt asks for>",
"digest_md": "${OUT}/digest/digest.md", "candidates": <topk.json candidates>, "recommended": <ids>, "notes": "..."}`,
  { phase: 'TopK', label: 'kf:analyst', schema: TOPK_SCHEMA });
  log(`[kf] topk: ${topk ? topk.candidates.length : 0} candidates, recommended ${topk ? JSON.stringify(topk.recommended) : '[]'}`);
}

const result = {
  eval_dir: KF_DIR, kf_dir: KF_DIR, kf_eval_dir: EVAL_DIR, trace_dir: TRACE_DIR, prompt: PROMPT,
  topk_json: topk ? topk.topk_json : '', analysis_report: topk ? (topk.report_path || '') : '',
  baseline_throughput_tok_s: baseTput, final_throughput_tok_s: baseTput, throughput_speedup: 1,
  final_overlay: INIT_OVERLAY, final_flags: INIT_FLAGS, final_env: INIT_ENV,
  accepted: [], rejected: [], status: 'analysis_only',
};
if (!topk || !(want('author') || want('apply'))) {
  if (!topk && want('topk')) { result.status = 'error'; result.reason = 'analyst returned nothing'; }
  return result;
}

// ---------------------------------------------------------------------------
// Author: extract the region unittest (GEAK kernel_extractor) and author the fused kernel (kernel_lane).
// ---------------------------------------------------------------------------
phase('Author');
const byId = new Map((topk.candidates || []).map((c) => [String(c.id), c]));
const picks = (topk.recommended || []).map((id) => byId.get(String(id))).filter(Boolean).slice(0, AUTHOR_K);
log(`[kf] authoring ${picks.length} candidate(s): ${picks.map((c) => c.id).join(', ')}`);
const authored = await parallel(picks.map((c, i) => async () => {
  if (DEADLINE_HIT) return { c, skip: 'stage deadline' };
  const gpu = GPU_LIST[i % GPU_LIST.length];
  const short = String(c.id).replace(/[^A-Za-z0-9_]+/g, '_');
  const ext = await safeAgent(roleAgent('kernel_extractor', 'extract',
    'Build the immutable unittest task dir for a FUSION REGION, not a single kernel. The baseline leg is the ' +
    'live, unfused region reached through ONE Python seam (KERNEL.target_callable, a function whose body launches ' +
    'every kernel in KERNEL.device_kernels); candidate_bind rebinds that seam, so the candidate leg runs the fused ' +
    'implementation from kernel_src/. Record I/O at the seam boundary (KERNEL.inputs_outputs) for the oracle. If ' +
    'the given seam does not cover the region, use the narrowest enclosing callable that does and say so in notes; ' +
    'if none exists return editable:false with notes "no_rebind_seam".', {
      EVAL_DIR, MODEL_PATH, GPU_ID: gpu, WORKLOAD, SKILL_DIR: WORKFLOW_DIR,
      KERNEL: { short_name: short, classification: 'fusion', fusion_level: c.level, region: c.region,
        device_kernel: (c.kernels || [])[0] || '', device_kernels: c.kernels || [],
        target_callable: (c.seam || {}).target_callable || '', source_hint: (c.seam || {}).source_file || '',
        seam_candidates: [(c.seam || {}).target_callable].filter(Boolean), inputs_outputs: c.inputs_outputs || '',
        phases: c.phases || [], m_range: c.m_range || '', candidate_backends: [c.language || 'triton'] },
      CURRENT_OVERLAY: INIT_OVERLAY, CURRENT_FLAGS: INIT_FLAGS, CURRENT_ENV: INIT_ENV,
    }), { phase: 'Author', label: `kf:extract ${short}`, schema: EXTRACT_SCHEMA });
  // Same guard as e2e extractWithBaseline: without a frozen live baseline the unittest would time the
  // fused kernel against its own scaffold.
  if (!ext || ext.editable === false || ext.unittest_smoke !== 'pass' || !ext.task_dir || ext.baseline_frozen === false) {
    return { c, skip: `extraction failed (${ext ? ext.notes || ext.unittest_smoke : 'none'})` };
  }
  const lang = c.language || 'triton';
  let kl;
  try {
    kl = await workflow({ scriptPath: KERNEL_WF_SCRIPT }, {
      use_learned_kb: 'false', kernel_path: ext.task_dir, workflow_dir: KERNEL_WF_DIR,
      mode: 'author', target_language: lang,
      op_spec: { op_kind: 'fusion', fusion_level: c.level, kernels: c.kernels || [], m_range: c.m_range || '', cuda_graph_safe: true },
      budget: KERNEL_BUDGET, gpu_ids: gpu, exp_root: `${EVAL_DIR}/kernels/_exp`, apply_to_original: 'false',
      task: `Author ONE fused ${lang} kernel (fusion ${c.level}) that replaces this whole region: ${c.region}. ` +
        `It currently launches ${c.launches_before || (c.kernels || []).length} kernels [${(c.kernels || []).join(' -> ')}]; ` +
        `target ${c.launches_after || 1}. Design from the analysis: ${c.fused_design || '(none given)'}. ` +
        `Correctness is the immutable oracle at the region boundary${c.bitwise_exact ? ' and must be bit-exact' : ''}. ` +
        `Beat the live unfused region on the recorded serving shapes.` + GRAPH_REQ,
    });
  } catch (e) { kl = { validation_status: 'error', reason: String(e) }; }
  return { c, ext, kl, lang };
}));

// ---------------------------------------------------------------------------
// ApplyBack: GEAK e2e_integrator, serially, compounding accepted overlays.
// ---------------------------------------------------------------------------
phase('ApplyBack');
let curOverlay = INIT_OVERLAY, curTput = baseTput, speedup = 1;
for (const r of authored) {
  if (!r) continue;
  const { c } = r;
  const iso = r.kl ? (r.kl.final_weighted != null ? r.kl.final_weighted : r.kl.final_geomean) : 0;
  if (r.skip || !r.kl || r.kl.authored === false || !(iso > 1.0) || !r.kl.final_patch) {
    const why = r.skip || `no isolated win (${iso || 0}x, ${r.kl ? r.kl.reason || r.kl.validation_status || '' : 'none'})`;
    log(`[kf] ${c.id}: ${why}`);
    result.rejected.push({ id: c.id, stage: r.skip ? 'extract' : 'author', reason: why, isolated_speedup: iso || 0 });
    continue;
  }
  if (!want('apply') || DEADLINE_HIT) {
    result.rejected.push({ id: c.id, stage: 'apply', reason: want('apply') ? 'stage deadline' : 'apply step disabled', isolated_speedup: iso });
    continue;
  }
  const inputs = {
    EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, NOISE_BAND_PCT: noiseBand,
    MEASUREMENT_PURPOSE: 'search', REPLICAS: SEARCH_REPLICAS, SKILL_DIR: WORKFLOW_DIR,
    KERNEL_RESULT: { short_name: r.ext.short_name || c.id, task_dir: r.ext.task_dir, op_kind: 'fusion',
      winner_kind: 'authored', winner_backend: r.lang, authored_language: r.lang,
      authored_kernel_eval_dir: r.kl.eval_dir || '', target_callable: r.ext.target_callable || (c.seam || {}).target_callable || '',
      code_patch: r.kl.final_patch, final_patch: r.kl.final_patch, verified_isolated_speedup: iso,
      pct_gpu_time: c.region_pct_device_time || 0, live_call_seam: (c.seam || {}).target_callable || '',
      parity_note: c.bitwise_exact ? 'expected_exact' : 'expected_close' },
    CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: INIT_FLAGS, CURRENT_ENV: INIT_ENV, CURRENT_THROUGHPUT: curTput,
  };
  let integ = await safeAgent(roleAgent('e2e_integrator', 'integrate',
    'Overlay this fused kernel back onto the live server and gate it on the e2e A/B.', inputs),
    { phase: 'ApplyBack', label: `kf:integrate ${c.id}`, schema: INTEGRATE_SCHEMA });
  for (let t = 1; t <= AB_FINISH_RETRIES && !(integ && integ.gate !== 'incomplete' && integ.ab_complete !== false); t++) {
    integ = await safeAgent(roleAgent('e2e_integrator', 'integrate',
      'FINISH this e2e A/B: reuse a reference leg already on disk if usable, run the missing leg, then return the gate.',
      { ...inputs, RESUME_AB: true }), { phase: 'ApplyBack', label: `kf:integrate ${c.id} (finish ${t})`, schema: INTEGRATE_SCHEMA });
  }
  const rec = { id: c.id, level: c.level, isolated_speedup: iso, gate: integ ? integ.gate : 'incomplete',
    e2e_delta_pct: integ ? integ.e2e_delta_pct : null, ref_med: integ ? integ.ref_med : null,
    cand_med: integ ? integ.cand_med : null, parity: integ ? integ.output_parity : '', reason: integ ? integ.reason || '' : 'no integrator result' };
  // Only a clear win banks here ('accepted' = past the noise band with non-overlapping legs). A 'stack'
  // verdict needs the e2e workflow's final combined validation to mean anything, and this stage has none.
  if (integ && integ.gate === 'accepted' && integ.ab_complete !== false && integ.accepted_overlay) {
    curOverlay = integ.accepted_overlay;
    curTput = integ.e2e_throughput_tok_s;
    // Each gate is its own same-protocol ref/cand pair; the stage gain is their product.
    if (integ.ref_med > 0 && integ.cand_med > 0) speedup *= integ.cand_med / integ.ref_med;
    else speedup *= 1 + (integ.e2e_delta_pct || 0) / 100;
    result.accepted.push({ ...rec, overlay: curOverlay });
    log(`[kf] ${c.id}: ACCEPTED ${rec.e2e_delta_pct}% -> ${curTput} tok/s`);
  } else {
    result.rejected.push({ ...rec, stage: 'apply' });
    log(`[kf] ${c.id}: not applied (${rec.gate}: ${rec.reason})`);
  }
}

result.final_overlay = curOverlay;
result.final_throughput_tok_s = curTput;
result.throughput_speedup = speedup;
result.status = result.accepted.length ? 'accepted' : 'no_gain';
return result;
