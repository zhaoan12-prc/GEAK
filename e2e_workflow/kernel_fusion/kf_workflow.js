export const meta = {
  name: 'kernel-fusion',
  description: 'KernelFusion stage, run BEFORE the GEAK e2e workflow: capture a serving trace (GEAK profiler), pick the Top-K fusion patterns with a user-supplied analysis prompt, build a region UT and have an agent write each fused kernel against it, and apply the winners back through the GEAK e2e integrator A/B. Returns the accepted fused overlay, which the e2e workflow then uses as its baseline.',
  whenToUse: 'Started by interface/run_e2e.py before the e2e workflow unless switched off (handoff "kernel_fusion": false, GEAK_KERNEL_FUSION=0 or --no-kernel-fusion). Takes the same args as e2e_workflow.js plus kf_dir, kf_prompt and the optional kf_* knobs documented in kernel_fusion/README.md.',
  phases: [
    { title: 'Setup', detail: 'e2e Director builds the KF eval dir + baseline throughput, one fresh server and one timed round, no warmup round (roles/director.md PHASE=setup)' },
    { title: 'Trace', detail: 'AgentX: kernel-extract skill sampling capture (points from the baseline log, one probe each); else roles/profiler.md; or kf_trace_dir' },
    { title: 'TopK', detail: 'fusion analyst: kf_trace_digest.py + the user prompt -> ranked Top-K fusion candidates (topk.json)' },
    { title: 'Author', detail: 'a triage agent picks the Top-K patterns worth authoring (no cap, no gain floor); each: region UT (AgentX: kernel-extract skill prepare; else roles/kernel_extractor.md), then one author agent writes the fused kernel against that UT; with kf_l3_mega a multi-rank stage (L3: EP MoE, collective + compute) goes to GEAK mode=mega instead' },
    { title: 'ApplyBack', detail: 'a select agent picks from the UT results which kernels go on; they go in as one combined overlay on one e2e_integrator A/B (gsm8k on the whole stack first); only if the stack fails accuracy is each kernel checked alone and the A/B re-run without the failing ones' },
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
// Operator constraints for every agent of this stage (e.g. paths it must not read).
const KF_TASK = String(A.kf_task || '');
// Simulated speculative acceptance (run_e2e.py passes the names it found); same rule and wording as e2e_workflow.js.
const CORRECTNESS_UNSET_ENV = String(A.correctness_unset_env || '').split(',')
  .map((s) => s.trim()).filter((s) => /^[A-Za-z_][A-Za-z0-9_]*$/.test(s));
const CORRECTNESS_PROBE_ROLES = new Set(['e2e_integrator', 'director']);
function correctnessEnvBlock(role) {
  if (!CORRECTNESS_UNSET_ENV.length || !CORRECTNESS_PROBE_ROLES.has(role)) return '';
  const unset = CORRECTNESS_UNSET_ENV.map((v) => `-u ${v}`).join(' ');
  return `\n\n## CORRECTNESS PROBES RUN WITHOUT SIMULATED ACCEPTANCE\n` +
    `This stack serves with simulated speculative-decoding acceptance (${CORRECTNESS_UNSET_ENV.join(', ')}). ` +
    `It keeps throughput comparable, but what it serves is neither deterministic nor the model's own ` +
    `output: the unmodified reference does not reproduce itself, so a greedy-parity or task-accuracy probe ` +
    `run under it measures the simulator, not the change. For EVERY correctness probe, launch BOTH servers ` +
    `(the TRUE baseline and the candidate) with these variables removed, from the process env ` +
    `(\`env ${unset} ...\`) and from EXTRA_ENV, and change nothing else: same flags, same overlay, same ` +
    `TP/GPU. Throughput legs keep them, because those legs ARE the measurement, so a correctness server ` +
    `is never one of the throughput servers. Say in your notes that the probe ran with simulated ` +
    `acceptance off.`;
}

const INIT_OVERLAY = String(A.initial_overlay_pythonpath || '');
const EFFECTIVE_CONFIG_DIGEST = String(A.effective_config_digest || '');
const MEASUREMENT_MODE = String(A.measurement_mode || 'warm_server');
const PARITY_REPLICAS = parseInt(A.parity_replicas != null ? A.parity_replicas : 1, 10);
const SEARCH_REPLICAS = parseInt(A.search_replicas != null ? A.search_replicas : 1, 10);
const NOISE_BAND_DEFAULT = parseFloat(A.noise_band_pct != null ? A.noise_band_pct : 0.5);
// The combined A/B hands the fused overlay on only when its gain clears the noise band: the larger of the measured
// band (setup) and this floor. With one replica per side a gain inside it is indistinguishable from noise.
const KF_MIN_GAIN_PCT = parseFloat(A.kf_min_gain_pct != null ? A.kf_min_gain_pct : 0.5);
const AB_FINISH_RETRIES = parseInt(A.ab_finish_retries != null ? A.ab_finish_retries : 2, 10);

// KernelFusion knobs.
const PROMPT = String(A.kf_prompt || `${WORKFLOW_DIR}/knowledge/kernel_fusion_skills/fusion-analysis/SKILL.md`);
// The fusion levels the author/apply steps implement; the analysis may recommend only these (the skill ranks all).
const IMPLEMENTABLE_LEVELS = String(A.kf_implementable_levels || 'L1,L2,L3');
// Patterns inside those levels the author/apply steps do not implement yet; the analysis ranks them but does not
// recommend them, and the triage does not author them.
const UNSUPPORTED_PATTERNS = String(A.kf_unsupported_patterns || 'L3 MegaKernel (one persistent kernel that absorbs a heavy expert/GEMM or attention stage together with its dispatch, combine or communication, e.g. MegaMoE); L4 (whole-layer MonoKernel)');
const PROMPTS = PROMPT.split(',').map((p) => p.trim()).filter(Boolean);   // several files drive one analysis
const TRACE_DIR_IN = String(A.kf_trace_dir || '');          // existing traces: skip the capture
const WORKLOAD_SUMMARY = String(A.kf_workload_summary || ''); // per-point weights, when the caller has them
// Finished UT-prepare / author results of an earlier run of this stage, by agent label (kf_stage reads them
// from that run's workflow journal): those agents are not re-run.
const REUSED = (A.kf_reuse && typeof A.kf_reuse === 'object') ? A.kf_reuse : {};
const TOPK_IN = String(A.kf_topk_json || '');                 // a finished analysis: reused, not re-run
const BASELINE_DIR_IN = String(A.kf_baseline_dir || '');     // a measured baseline of this same stack: reused, not re-measured
const SOURCE_DIRS = String(A.kf_source_dirs || '');           // framework source, when not importable here
const STEPS = new Set(String(A.kf_steps || 'trace,topk,author,apply').split(',').map((s) => s.trim()).filter(Boolean));
const want = (s) => STEPS.has(s);
// A fused kernel reorders floating-point work (a fused reduction, fp32 kept in registers), so like a quant kernel
// it is held to TASK ACCURACY, not byte-exact greedy parity: the GEAK integrator's existing ACCURACY_GATE switch.
const KF_ACCURACY_GATE = String(A.accuracy_gate || 'gsm8k').trim();
const KF_ACCURACY_INPUTS = KF_ACCURACY_GATE !== 'none'
  ? { ACCURACY_GATE: KF_ACCURACY_GATE, ACCURACY_LIMIT: parseInt(A.accuracy_limit != null ? A.accuracy_limit : 200, 10),
      ACCURACY_TOL: parseFloat(A.accuracy_tol != null ? A.accuracy_tol : 0.01),
      GSM8K_EVAL_SCRIPT: `${WORKFLOW_DIR}/scripts/gsm8k_eval.py` }
  : {};
// A 200-question gsm8k draw moves by ~1 point run to run, as much as the tolerance. When the stack misses the gate on
// the sampled subset, the full test set (1319 questions) decides before any per-kernel screen; the screen uses it too.
const ACCURACY_CONFIRM_LIMIT = parseInt(A.kf_accuracy_confirm_limit != null ? A.kf_accuracy_confirm_limit : 1319, 10);
const VALIDATION_MODE = String(A.kf_validation_mode || 'isolated_server');   // fresh servers: boot-to-boot noise
const VALIDATION_REPLICAS = parseInt(A.kf_validation_replicas != null ? A.kf_validation_replicas : 1, 10);   // ONE A/B
// The stage's own baseline: one fresh server, one timed round -- no discarded full warmup round.
const BASELINE_MODE = String(A.kf_baseline_mode || 'isolated_server');
// AgentX trace replays use the kernel-extract skill for both trace capture and UT preparation, as
// e2e_workflow.js does (its KERNEL_EXTRACT_ON); other workloads keep the profiler / kernel_extractor roles.
const WORKLOAD_SPEC = A.workload_spec && typeof A.workload_spec === 'object' ? A.workload_spec : {};
const IS_AGENTX = String(A.workload_kind || WORKLOAD_SPEC.kind || '') === 'agentx_trace_replay';
const KERNEL_EXTRACT_DIR = `${WORKFLOW_DIR}/knowledge/kernel_extraction_skills/kernel-extract`;
const CAPTURE_PERSIST_POLICY = String(A.capture_persist_policy || 'structure_only');
const CAPTURE_STORAGE_ENV = `CAPTURE_BYTE_BUDGET=${A.capture_byte_budget || '8GiB'} ` +
  `CAPTURE_CASE_BYTE_LIMIT=${A.capture_case_byte_limit || '2GiB'} CAPTURE_PERSIST_POLICY=${CAPTURE_PERSIST_POLICY}`;
const AUTHOR_K = parseInt(A.kf_author_k != null ? A.kf_author_k : 0, 10);   // 0 = no cap: the triage agent decides
const AGENT_TIMEOUT_MS = parseInt(A.agent_timeout_ms != null ? A.agent_timeout_ms : 7200000, 10);

// L3 stage fusion (opt-in). A multi-rank stage -- the kernels of one functional stage that include cross-GPU
// communication (EP MoE dispatch -> expert GEMMs -> combine; an all-reduce / all-gather / reduce-scatter with
// the compute around it) -- cannot be authored on one card. With kf_l3_mega on, the analyst may rank it as an
// implementable L3 candidate, and Author hands it to GEAK mode=mega (kernel_workflow/kernel_mega.js, one
// nested workflow) on the serving GPUs instead of an author agent. Off: nothing below changes a prompt.
const jsonArg = (v) => { if (v && typeof v === 'object') return v;
  try { const o = JSON.parse(String(v || '')); return o && typeof o === 'object' ? o : null; } catch (e) { return null; } };
const L3_ON = String(A.kf_l3_mega != null ? A.kf_l3_mega : 'false').trim().toLowerCase() === 'true';
// Kernel language. Languages differ in speed on the same GPU, and nothing here races them, so the language can be
// pinned: kf_kernel_language = a language (e.g. flydsl), or "auto" for the GPU's default below. Unset, L1/L2/L3
// stage-fusion kernels keep the analyst's choice (triton unless it has a reason) and a mode=mega search follows
// its GPU's default, else the language of the source it starts from.
const LANGUAGE_BY_GFX = { gfx950: 'flydsl' };
const KF_GFX = String(A.expected_gfx || A.gfx || '').trim().toLowerCase();
const LANG_ARG = String(A.kf_kernel_language || '').trim().toLowerCase();
const KERNEL_LANGUAGE = LANG_ARG === 'auto' ? (LANGUAGE_BY_GFX[KF_GFX] || '') : LANG_ARG;
const L3_LANGUAGE = KERNEL_LANGUAGE || LANGUAGE_BY_GFX[KF_GFX] || '';
const L3_GEAK_DIR = String(A.kf_l3_geak_dir || `${WORKFLOW_DIR.replace(/\/[^/]*$/, '')}/kernel_workflow`).replace(/\/+$/, '');
const L3_MAX = Math.max(1, parseInt(A.kf_l3_max != null ? A.kf_l3_max : 1, 10) || 1);
const L3_BUDGET_S = parseInt(A.kf_l3_budget_s != null ? A.kf_l3_budget_s : 8 * 3600, 10);
const L3_ROUNDS = parseInt(A.kf_l3_rounds != null ? A.kf_l3_rounds : 12, 10);
const L3_MEGA_ARGS = jsonArg(A.kf_l3_mega_args) || {};    // merged last into the mega args
const L3_OPERATOR = jsonArg(A.kf_l3_operator);             // known operator facts: prepare verifies them
// A finished mode=mega run of this stage (its state_dir, eval dir or a mega_result.json): packaged in minutes
// instead of a new search. Without it the stage searches, which takes hours beyond the stage's usual ~7 h.
const L3_MEGA_RUN = String(A.kf_l3_mega_run || '').replace(/\/+$/, '');
const L3_EP_DEFAULT = parseInt(A.ep != null ? A.ep : SERVING_TP, 10) || SERVING_TP;
// Stage kinds L3 hands to mode=mega. ep_moe (dispatch -> expert GEMMs -> combine) is the validated one and the
// default; collective (a TP all-reduce / all-gather / reduce-scatter with its compute) is wired but opt-in
// (kf_l3_stage_kinds) until mode=mega has been measured on it.
const L3_KINDS = new Set(String(A.kf_l3_stage_kinds || 'ep_moe').split(',').map((s) => s.trim()).filter(Boolean));
const EP_MOE_RE = new RegExp('megamoe|ep_combine|ep_dispatch|deepep|mori_|all_?to_?all|a2a_' +
  (A.kf_l3_kernel_regex ? `|${A.kf_l3_kernel_regex}` : ''), 'i');
const COLLECTIVE_RE = /all_?reduce|cross_device_reduce|custom_ar|quickreduce|all_?gather|reduce_?scatter|rccl|nccl/i;
// MoE expert compute (expert GEMMs, not routing alone): on an expert-parallel deployment (args.ep > 1) a stage of these
// is an EP MoE stage even when its dispatch/combine kernels carry no library name (a from-scratch Triton path).
const MOE_KERNEL_RE = /fused_moe|moe_gemm|ck_moe|fmoe|moe_stage|moe_mlp|grouped_gemm|experts?_(gemm|mlp|ffn)/i;
const EP_DEPLOYED = (parseInt(A.ep, 10) || 1) > 1;
const hasKernel = (c, re) => (c.kernels || []).some((k) => re.test(String(k)));
// The kind is read from the stage's own kernels, so a candidate is never taken on its label alone.
const stageKind = (c) => ((hasKernel(c, EP_MOE_RE) || (EP_DEPLOYED && hasKernel(c, MOE_KERNEL_RE))) ? 'ep_moe'
  : (hasKernel(c, COLLECTIVE_RE) ? 'collective' : ''));
// Mode=mega takes the MegaKernel form only (it rewrites the heavy expert/GEMM work). A light L3 stage fusion over
// the same kernels (router + metadata + dispatch preparation) stays with the author agents.
// Evidence is again the stage's own kernels: expert GEMMs for an EP MoE, a GEMM for a collective stage.
const isMegaKernel = (c, kind) => (kind === 'ep_moe'
  ? hasKernel(c, MOE_KERNEL_RE) || hasKernel(c, /megamoe_(stage|gemm)/i)
  : hasKernel(c, /gemm|matmul/i));
const isMegaStage = (c) => L3_ON && !!c && String(c.level || '').toUpperCase() === 'L3' && L3_KINDS.has(stageKind(c)) &&
  isMegaKernel(c, stageKind(c));
const L3_COLLECTIVE = L3_KINDS.has('collective');
// The analysis scope with L3 on: L3 is implementable, and the MegaKernel mode=mega implements leaves the
// unsupported patterns (a caller's own kf_unsupported_patterns is kept as given).
const L3_LEVELS = L3_ON && !/\bL3\b/i.test(IMPLEMENTABLE_LEVELS) ? `${IMPLEMENTABLE_LEVELS},L3` : IMPLEMENTABLE_LEVELS;
const L3_UNSUPPORTED = L3_ON && A.kf_unsupported_patterns == null
  ? `L3 MegaKernel other than the ${L3_COLLECTIVE ? 'expert-parallel MoE (MegaMoE) and TP-collective stages, e.g. an attention ' +
      'MegaKernel' : 'expert-parallel MoE (MegaMoE) pattern, e.g. a TP all-reduce overlapped with its GEMM or an attention ' +
      'MegaKernel'}, which GEAK mode=mega does not implement here; L4 (whole-layer MonoKernel)`
  : UNSUPPORTED_PATTERNS;
// Search-policy defaults per stage kind (kernel_fusion/l3_mega_defaults.json, passed in by kf_stage.py).
const L3_DEFAULTS = jsonArg(A.kf_l3_defaults) || {};
// The workflow owns where mega runs, on which GPUs and for how long; l3_mega_args cannot move them
// (l3_operator / l3_rounds / l3_budget_s are the knobs for those).
const L3_OWNED_KEYS = ['mode', 'workflow_dir', 'kernel_path', 'benchmark_harness', 'gpu_ids', 'gpus_per_job',
  'job_gpu_ids', 'exp_root', 'state_dir', 'budget', 'mega_time_budget_s'];
const L3_OVERRIDES = Object.fromEntries(Object.entries(L3_MEGA_ARGS).filter(([k]) => !L3_OWNED_KEYS.includes(k)));
const L3_IGNORED = Object.keys(L3_MEGA_ARGS).filter((k) => L3_OWNED_KEYS.includes(k));
if (L3_ON && L3_IGNORED.length) {
  log(`[kf] L3: l3_mega_args keys the workflow owns are ignored: ${L3_IGNORED.join(', ')}`);
}

// Stop STARTING author/apply work once 85% of the stage budget is gone; in-flight work finishes and the
// stage returns whatever was accepted. Inert without time_budget_s.
const TIME_BUDGET_MS = A.time_budget_s ? parseInt(A.time_budget_s, 10) * 1000 : null;
let DEADLINE_HIT = false;
if (TIME_BUDGET_MS && typeof setTimeout === 'function') {
  setTimeout(() => { DEADLINE_HIT = true; log('[kf] 85% of the stage budget spent: no new author/apply work starts.'); },
    Math.floor(TIME_BUDGET_MS * 0.85));
}
// kf_stage stops the stage at 1.1 x budget + 900 s. A combined A/B (stack gsm8k, a reference leg and a candidate
// leg) starts only while that hard limit still leaves AB_NEEDS_S; a re-run after a failed stack also needs the
// per-kernel screen. An A/B cut off mid-leg measures nothing and leaves detached legs on the serving GPUs.
const AB_NEEDS_S = parseInt(A.kf_ab_needs_s != null ? A.kf_ab_needs_s : 5400, 10);
const SCREEN_NEEDS_S = parseInt(A.kf_screen_needs_s != null ? A.kf_screen_needs_s : 1800, 10);
let AB_LATE = false, RERUN_LATE = false;
if (TIME_BUDGET_MS && typeof setTimeout === 'function') {
  const hardMs = TIME_BUDGET_MS * 1.1 + 900000;
  setTimeout(() => { AB_LATE = true; }, Math.max(0, Math.floor(hardMs - AB_NEEDS_S * 1000)));
  setTimeout(() => { RERUN_LATE = true; }, Math.max(0, Math.floor(hardMs - (AB_NEEDS_S + SCREEN_NEEDS_S) * 1000)));
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

// Same text as e2e_workflow.js PROCESS_SAFETY (its final-phase section does not apply to this stage).
const PROCESS_SAFETY = `## PROCESS SAFETY (a violation can kill the caller's orchestrator, failing the whole task)
This container's PID 1 is the CALLER's orchestrator process, not yours, and it is NOT restartable.
NEVER run global or pattern-matched process cleanup: no \`pkill -f\` / \`pgrep -f ... | xargs kill\` /
\`killall\` / \`ps aux | grep ... | xargs kill\`, and never \`kill -- -PGID\` for a group you did not
create. A pattern as innocent as \`-f vllm\` matches the orchestrator's own command line and TERMs it.
Manage ONLY processes you started, by the pid you captured at launch. When a script of yours launches a
server, do NOT hand-roll the shutdown — source the shared teardown contract, which verifies process
identity (pid, pgid, /proc start time) before signalling anything:
    source ${WORKFLOW_DIR}/scripts/server_teardown.sh
    trap server_teardown EXIT
    \${SERVER_LAUNCH_PREFIX:-} <launch server> &   # own session => pgid == pid
    SERVER_PID=\$!; server_record_identity "\$SERVER_PID"

## LONG MEASUREMENTS OUTLIVE YOUR TOOL CALL — DETACH THEM FROM THE SHELL MANAGER
A serving measurement is longer than any bound your Bash tool can give it. One canonical trace-replay
leg is a 3600s window PLUS warmup and finalization (~70min wall), and a cold server boot alone is ~9min.
Your foreground bound is BASH_MAX_TIMEOUT_MS, and your tool's OWN background mode is worse: a shell the
tool manages for you is reaped on the manager's schedule, and the reap is a GROUP kill that takes the
server down with the client — you get a truncated log, an unloaded GPU, and no summary, from a leg that
was measuring perfectly. Do NOT run a bench through your tool's background/\`run_in_background\` mode.
Launch it so it is orphaned to init, OUTSIDE the manager's reach, then poll its log:
    cd "\$EVAL_DIR" && setsid nohup bash run_bench.sh > run_bench.out 2>&1 < /dev/null &
    # returns immediately; the leg's parent becomes PID 1 and no tool-call bound applies
Then poll in SEPARATE short calls (\`sleep 300; tail -5 run_bench.out\`) until the summary lands. Before
you conclude a leg died, check whether it is still alive (\`ps -o pid,stat,etime -p <pid>\`) and whether
the GPUs still hold weights — a reaped wrapper with a live leg underneath looks identical to a crash.

## INSIDE THIS WORKFLOW, ENDING YOUR TURN IS YOUR FINAL ANSWER
Ending your turn to wait for a background task or notification forces your structured result at once:
you are finalized mid-job, and the server and captures you are waiting on are lost with you. Wait only in
the foreground, in calls of at most ~9 minutes each (a loop of short sleeps + a check), re-issued until the
job is done. Profiler traces with Python stacks of a large TP model take many minutes to flush: keep polling
until every rank's file is complete, and only then tear the server down.

## THE GEAK REPO IS READ-ONLY
Never edit, add or delete anything under WORKFLOW_DIR's repository (scripts, skills, roles, tools). When a
GEAK tool does not handle this model (an unparsed log format, a missing case), work around it with your own
script under EVAL_DIR and say so in notes.

`;

function roleAgent(role, phase, intro, inputs) {
  const inall = { BACKEND, SERVING_TP, SERVING_GPU, MEASUREMENT_MODE, PARITY_REPLICAS, SEARCH_REPLICAS,
    EFFECTIVE_CONFIG_DIGEST,
    ...(correctnessEnvBlock(role) ? { CORRECTNESS_UNSET_ENV: CORRECTNESS_UNSET_ENV.join(' ') } : {}), ...inputs };
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

${KF_TASK ? `\n## Operator constraints\n${KF_TASK}\n` : ''}
Return ONLY the structured JSON the role file specifies.` + correctnessEnvBlock(role);
}

// The kernel-extract skill, invoked the way e2e_workflow.js roleAgent does for kernel_extractor on AgentX.
function skillAgent(phase, intro, inputs) {
  const inall = { BACKEND, SERVING_TP, SERVING_GPU, MEASUREMENT_MODE, PARITY_REPLICAS, SEARCH_REPLICAS,
    EFFECTIVE_CONFIG_DIGEST, ...inputs,
    SKILL_ROOT: KERNEL_EXTRACT_DIR, SHARED_TOOLS: `${WORKFLOW_DIR}/scripts`, WORKFLOW_DIR,
    PARSE_PROFILE_STRICT_PHASES: '1', KERNEL_ENV_SKIP_ENUM_REAP: '1', CAPTURE_PERSIST_POLICY, CAPTURE_STORAGE_ENV };
  return `You are the kernel_extractor. PHASE=${phase}.
First Read ${KERNEL_EXTRACT_DIR}/SKILL.md and ${KERNEL_EXTRACT_DIR}/references/workflow-handoff.md.
Use the complete extraction procedure and its shared and skill-specific tools. The handoff reference supplies only workflow inputs and output packaging.
Do not load roles/kernel_extractor.md; it is the legacy procedure for other workloads.
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}
You are running inside the GEAK KernelFusion stage: EVAL_DIR is the KernelFusion eval dir.
Every serving launch uses BACKEND=${BACKEND} TP=${SERVING_TP} GPU=${SERVING_GPU}.

## Inputs
${cfg(inall)}
${KF_TASK ? `\n## Operator constraints\n${KF_TASK}\n` : ''}
Return ONLY the structured JSON requested above.`;
}

// The guard only stops WAITING: the agent keeps running, and safeAgent then starts a second one on the same
// job. Steps that own long serving legs (the screen, the combined A/B) therefore get APPLY_TIMEOUT_MS.
const APPLY_TIMEOUT_MS = parseInt(A.kf_apply_timeout_ms != null ? A.kf_apply_timeout_ms : 6 * 3600 * 1000, 10);
function bounded(prompt, opts) {
  const { timeoutMs, ...agentOpts } = opts;
  const limit = timeoutMs != null ? timeoutMs : AGENT_TIMEOUT_MS;
  const p = agent(PROCESS_SAFETY + prompt, agentOpts);
  if (typeof setTimeout !== 'function' || !(limit > 0)) return p;
  let to;
  const guard = new Promise((resolve) => { to = setTimeout(() => resolve(null), limit); });
  return Promise.race([p.then((r) => { clearTimeout(to); return r; }, (e) => { clearTimeout(to); throw e; }), guard]);
}

async function safeAgent(prompt, opts, tries = 3) {
  if (REUSED[opts.label]) { log(`[kf] ${opts.label}: reused`); return REUSED[opts.label]; }
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
const BASELINE_WAIT_SCHEMA = obj({ status: { type: 'string', enum: ['done', 'running', 'dead'] },
  baseline_throughput_tok_s: { type: 'number' }, notes: { type: 'string' } }, ['status']);
const SKILL_TRACE_SCHEMA = obj({ trace_dir: { type: 'string' }, workload_json: { type: 'string' },
  sampling_points_json: { type: 'string' }, ranking_json: { type: 'string' }, points: arrObj, notes: { type: 'string' } },
  ['trace_dir', 'workload_json', 'points']);
const AGENTX_PREPARATION_SCHEMA = obj({ delivery_path: { type: 'string' },
  prepared: obj({ head_candidates: arrObj, kernel_candidates: arrObj, failed_tasks: arrObj, deferred_tasks: arrObj },
    ['head_candidates', 'kernel_candidates']) }, ['delivery_path', 'prepared']);
// Typed where downstream code reads a field: a seam given as prose or a bare string, or ids that are not
// strings, are rejected here so the analyst rewrites them, instead of silently dropping every pick later.
const CANDIDATE_SCHEMA = obj({
  id: { type: 'string' },
  seam: obj({ target_callable: { type: 'string' }, source_file: { type: 'string' } }),
  exclusive_with: arrStr,
  fallback: { type: ['string', 'null'] },
  kernels: arrStr, phases: arrStr,
}, ['id']);
const TOPK_SCHEMA = obj({
  topk_json: { type: 'string' }, report_path: { type: 'string' }, digest_md: { type: 'string' },
  candidates: { type: 'array', items: CANDIDATE_SCHEMA }, recommended: arrStr, notes: { type: 'string' },
  report_md: { type: 'string' }, model_notes_md: { type: 'string' },
}, ['topk_json', 'candidates', 'recommended', 'report_md']);
const EXTRACT_SCHEMA = obj({ short_name: { type: 'string' }, editable: { type: 'boolean' }, task_dir: { type: 'string' },
  target_callable: { type: 'string' }, unittest_smoke: { type: 'string' }, candidate_bind: { type: 'object', additionalProperties: true },
  baseline_frozen: { type: 'boolean' },
  notes: { type: 'string' } }, ['editable', 'task_dir', 'unittest_smoke']);
const SCREEN_SCHEMA = obj({ members: { type: 'array', items: obj({ short_name: { type: 'string' },
  correctness: { type: 'string' }, engaged: { type: 'boolean' }, reason: { type: 'string' } }, ['short_name', 'correctness']) },
  notes: { type: 'string' } }, ['members']);
const PICK_SCHEMA = obj({ selected: { type: 'array', items: obj({ id: { type: 'string' }, reason: { type: 'string' } }, ['id']) },
  skipped: { type: 'array', items: obj({ id: { type: 'string' }, reason: { type: 'string' } }, ['id']) },
  notes: { type: 'string' } }, ['selected']);
const AUTHOR_SCHEMA = obj({ final_patch: { type: 'string' }, work_dir: { type: 'string' },
  overlay_dir: { type: 'string' }, banner: { type: 'string' },
  isolated_speedup: { type: 'number' }, correctness: { type: 'string' }, launches_after: { type: 'number' },
  ut_command: { type: 'string' }, notes: { type: 'string' } }, ['final_patch', 'isolated_speedup', 'correctness']);
const VALIDATE_SCHEMA = obj({ validation_status: { type: 'string' }, throughput_speedup: { type: 'number' },
  director_verified_throughput_tok_s: { type: 'number' }, output_parity: { type: 'string' }, notes: { type: 'string' } },
  ['validation_status']);
const INTEGRATE_SCHEMA = obj({ short_name: { type: 'string' }, e2e_throughput_tok_s: { type: 'number' },
  e2e_delta_pct: { type: 'number' }, ref_med: { type: 'number' }, cand_med: { type: 'number' },
  ab_complete: { type: 'boolean' }, output_parity: { type: 'string' }, parity_kind: { type: 'string' },
  gate: { type: 'string', enum: ['accepted', 'stack', 'rejected', 'incomplete'] },
  accepted_overlay: { type: 'string' }, reason: { type: 'string' } }, ['gate', 'e2e_throughput_tok_s']);

// A director that follows the detach rule may return while its baseline leg is still running, with no
// number yet. Waiting is then a job of its own: an agent that starts nothing and only polls, re-issued
// until the summary lands or the leg is gone.
async function awaitBaseline(dir) {
  for (let i = 1; i <= 6 && !DEADLINE_HIT; i++) {
    const w = await safeAgent(`You wait for one measurement and start nothing. A baseline leg (bench_e2e.sh) was ` +
      `launched detached under ${dir} (see ${dir}/logs, ${dir}/*.out and ${dir}/baseline). Poll in separate ` +
      `foreground calls of at most ~9 minutes each (a loop of sleep + check), for up to ~100 minutes in total, until ` +
      `${dir}/baseline/bench_summary.json exists or the leg is no longer alive (its pid / process group, and whether ` +
      `the GPUs still hold weights). Return {status: "done" with baseline_throughput_tok_s = that summary's ` +
      `throughput_tok_s_median, or "running" if it is still going when you stop, or "dead" if it exited without a ` +
      `summary, notes}.`, { phase: 'Setup', label: `kf:baseline wait ${i}`, schema: BASELINE_WAIT_SCHEMA });
    if (w && w.status === 'done' && w.baseline_throughput_tok_s > 0) return w.baseline_throughput_tok_s;
    if (w && w.status === 'dead') return 0;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Setup (GEAK director) -- needed whenever this stage touches a server or a GPU.
// ---------------------------------------------------------------------------
const needsServer = (want('trace') && !TRACE_DIR_IN) || want('author') || want('apply');
let EVAL_DIR = KF_DIR, baseTput = 0, noiseBand = NOISE_BAND_DEFAULT;
if (needsServer) {
  phase('Setup');
  const setup = await safeAgent(roleAgent('director', 'setup',
    'Build the isolated KernelFusion eval dir (use EVAL_DIR_OVERRIDE exactly) and record the baseline throughput.' +
    (BASELINE_DIR_IN ? ` The baseline of this exact stack (same flags, env, overlay, workload) was already measured in ` +
      `${BASELINE_DIR_IN}: do every setup step EXCEPT the baseline measurement -- launch no server and no bench; copy ` +
      `that dir to EVAL_DIR/baseline (cp -a) and return its bench_summary.json throughput_tok_s_median as ` +
      'baseline_throughput_tok_s.' : ''), {
      LAUNCH_SCRIPT, MODEL_PATH, EXP_ROOT: KF_DIR.replace(/\/[^/]*$/, ''), EVAL_DIR_OVERRIDE: KF_DIR,
      MODEL_NAME_HINT: MODEL_PATH.replace(/\/+$/, '').split('/').pop(), TASK: '', GPU_IDS, WORKLOAD,
      INIT_FLAGS, INIT_ENV, INIT_BASE_OVERLAY: INIT_OVERLAY, MEASUREMENT_PURPOSE: 'parity',
      MEASUREMENT_MODE: BASELINE_MODE, REPLICAS: 1, SKILL_DIR: WORKFLOW_DIR,
    }), { phase: 'Setup', label: 'kf:director:setup', schema: SETUP_SCHEMA });
  if (!setup || !setup.eval_dir) {
    return { eval_dir: KF_DIR, kf_dir: KF_DIR, status: 'error', reason: 'setup failed', accepted: [], final_overlay: INIT_OVERLAY };
  }
  EVAL_DIR = setup.eval_dir;
  baseTput = setup.baseline_throughput_tok_s || 0;
  if (!(baseTput > 0)) baseTput = await awaitBaseline(EVAL_DIR);
  if (!(baseTput > 0) && want('trace') && !TRACE_DIR_IN) {
    // The trace step plans its sampling points from this baseline's serving log: without it there is
    // nothing to sample from. One more attempt, then the stage stops (the e2e workflow still runs).
    const again = await safeAgent(roleAgent('director', 'setup',
      `The eval dir ${EVAL_DIR} exists but its baseline was not measured. Reuse the dir and measure the baseline ` +
      '(step 5 of PHASE=setup) to completion.', {
        LAUNCH_SCRIPT, MODEL_PATH, EXP_ROOT: KF_DIR.replace(/\/[^/]*$/, ''), EVAL_DIR_OVERRIDE: EVAL_DIR,
        MODEL_NAME_HINT: MODEL_PATH.replace(/\/+$/, '').split('/').pop(), TASK: '', GPU_IDS, WORKLOAD,
        INIT_FLAGS, INIT_ENV, INIT_BASE_OVERLAY: INIT_OVERLAY, MEASUREMENT_PURPOSE: 'parity',
        MEASUREMENT_MODE: BASELINE_MODE, REPLICAS: 1, SKILL_DIR: WORKFLOW_DIR,
      }), { phase: 'Setup', label: 'kf:director:setup (baseline)', schema: SETUP_SCHEMA });
    baseTput = (again && again.baseline_throughput_tok_s) || 0;
    if (!(baseTput > 0)) baseTput = await awaitBaseline(EVAL_DIR);
    if (!(baseTput > 0)) {
      return { eval_dir: KF_DIR, kf_dir: KF_DIR, status: 'error', reason: 'baseline not measured (twice)',
        accepted: [], final_overlay: INIT_OVERLAY };
    }
  }
  noiseBand = setup.noise_band_pct || NOISE_BAND_DEFAULT;
  log(`[kf] setup: baseline ${baseTput} tok/s, noise band ${noiseBand}%`);
}

// ---------------------------------------------------------------------------
// Trace: on AgentX the kernel-extract skill's workload measurement and sampling capture (stages 1-2:
// sampling points planned from the baseline log, one bounded probe per point); otherwise the GEAK
// profiler. Or the caller's traces.
// ---------------------------------------------------------------------------
let TRACE_DIR = TRACE_DIR_IN;
let profileTopN = '';
let workloadJson = WORKLOAD_SUMMARY;
let rankingJson = '';
const skillInputs = () => ({
  EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, TASK: '',
  BASELINE_LOG: `${EVAL_DIR}/baseline/server.log`, BASELINE_BENCH_DIR: `${EVAL_DIR}/baseline`,
  BASELINE_SUMMARY: `${EVAL_DIR}/baseline/bench_summary.json`, EVIDENCE_DIR: `${EVAL_DIR}/kernel_extraction/round_kf`,
  BASELINE_THROUGHPUT: baseTput, CURRENT_FLAGS: INIT_FLAGS, CURRENT_ENV: INIT_ENV, CURRENT_OVERLAY: INIT_OVERLAY,
  PREVIOUS_WORKLOAD_JSON: workloadJson, PREVIOUS_RANKING_JSON: rankingJson, SKILL_DIR: WORKFLOW_DIR,
});
if (want('trace') && !TRACE_DIR_IN && IS_AGENTX) {
  phase('Trace');
  const traceIntro = 'Run ONLY skill stage 1 and the workload-measurement and capture part of stage 2 for the ' +
    'completed baseline in the inputs: observed_workload.py -> plan_sampling_points.py -> serving_probe.py once per ' +
    'point (traces with record_shapes and Python stacks, steady state, verified) -> coverage_ledger.py, then write ' +
    'kernel_workload.json with each point\'s weight. Do NOT select targets, declare tasks, extract source or author UTs: ' +
    'a fusion analysis consumes these traces next. Return {trace_dir: the directory holding one sub-directory of ' +
    'per-rank traces per sampling point, workload_json, sampling_points_json, ranking_json, points: [{point_id, phase, ' +
    'trace_files, verified}], notes}.';
  let tr = await safeAgent(skillAgent('trace', traceIntro, skillInputs()),
    { phase: 'Trace', label: 'kf:kernel-extract trace', schema: SKILL_TRACE_SCHEMA });
  const verified = (t) => t && t.trace_dir && (t.points || []).some((p) => p && p.verified !== false);
  if (!verified(tr)) {
    log('[kf] sampling capture returned no verified trace; retrying once.');
    tr = await safeAgent(skillAgent('trace', 'A previous attempt returned no verified trace. Reuse what it saved, ' +
      'recapture each failed point once, and wait for the captures to finish. ' + traceIntro, skillInputs()),
      { phase: 'Trace', label: 'kf:kernel-extract trace (retry)', schema: SKILL_TRACE_SCHEMA });
  }
  if (!verified(tr)) {
    return { eval_dir: KF_DIR, kf_dir: KF_DIR, status: 'error', reason: 'sampling capture produced no verified trace (twice)',
      accepted: [], final_overlay: INIT_OVERLAY };
  }
  TRACE_DIR = tr.trace_dir;
  workloadJson = WORKLOAD_SUMMARY || tr.workload_json;
  rankingJson = tr.ranking_json || '';
} else if (want('trace') && !TRACE_DIR_IN) {
  phase('Trace');
  let prof = await safeAgent(roleAgent('profiler', 'baseline',
    'Capture a warm serving trace and emit the standardized Top-N. Keep the raw per-rank trace files on disk.', {
      EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, ROUND: 0,
      OVERLAY_PYTHONPATH: INIT_OVERLAY, EXTRA_SERVER_ARGS: INIT_FLAGS, EXTRA_ENV: INIT_ENV, SKILL_DIR: WORKFLOW_DIR,
    }), { phase: 'Trace', label: 'kf:profiler', schema: PROFILE_SCHEMA });
  let got = prof && prof.profile_topN_json && (prof.top_kernels || []).length;
  if (!got) {
    // A profiler that returned without a trace (killed capture, unfinished wait) must not hand TopK an
    // empty dir: retry once, then stop the stage -- the e2e workflow still runs on the original stack.
    log('[kf] profiler returned no Top-N; retrying the capture once.');
    prof = await safeAgent(roleAgent('profiler', 'baseline',
      'A previous attempt returned before its trace existed. Capture the warm serving trace again, wait for it ' +
      'to finish, and emit the standardized Top-N. Keep the raw per-rank trace files on disk.', {
        EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, ROUND: 0,
        OVERLAY_PYTHONPATH: INIT_OVERLAY, EXTRA_SERVER_ARGS: INIT_FLAGS, EXTRA_ENV: INIT_ENV, SKILL_DIR: WORKFLOW_DIR,
      }), { phase: 'Trace', label: 'kf:profiler (retry)', schema: PROFILE_SCHEMA });
    got = prof && prof.profile_topN_json && (prof.top_kernels || []).length;
  }
  if (!got) {
    return { eval_dir: KF_DIR, kf_dir: KF_DIR, status: 'error', reason: 'trace capture produced no Top-N (twice)',
      accepted: [], final_overlay: INIT_OVERLAY };
  }
  profileTopN = prof.profile_topN_json;
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
  topk = TOPK_IN ? { topk_json: TOPK_IN, candidates: (A.kf_topk || {}).candidates || [],
    recommended: (A.kf_topk || {}).recommended || [], report_md: '', notes: `reused ${TOPK_IN}` }
  : await safeAgent(`You are the GEAK KernelFusion analyst. You choose which kernel-fusion patterns to implement.

## Your method is the user's prompt
Read every one of these files in full, and the reference files a SKILL.md among them links: ${PROMPTS.join(', ')}.
Together they are the AUTHORITATIVE analysis
method, ranking rules, report content, deployment and workload description for this task: follow them step
by step. They are NOT authoritative on output files or formats: where a file names output paths, an out/ layout
or a machine-readable candidate format (field names, a per-candidate "recommended" flag, a seam written as
prose), the sections "Machine-readable result" and "Return JSON" below replace it. A file may describe a whole run (launching servers, running GEAK, where outputs go): take from it the
deployment, workload, metric and constraints, but your job is only the analysis below. They may also name a
container, a working directory or a materials/ layout from the setting they were written in; those are NOT
this environment. Map them onto the inputs below instead:
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
them in your report. Its "Live model code" table lists the module classes this server actually runs and the file
of each forward (from the eager steps' Python stacks). Graph-replayed steps have no launch frames but run those same
classes: map their kernels to source by name WITHIN those files -- the framework may ship other implementations of the
same layers that this server never loads, and a seam there is never called.

This environment does not let you write .md report files: return the full report the prompt asks for as markdown
in report_md, and the model / deployment notes in model_notes_md; the stage writes them to ${OUT}/REPORT.md and
${OUT}/MODEL_NOTES.md.

## Machine-readable result (in addition to everything the prompt asks for)
Write ${OUT}/topk.json: {"candidates": [...], "recommended": [ids in the order to implement]}, every candidate:
  id (short, [A-Za-z0-9_]), rank, level ("L1".."L4"), region (one line), kernels (device kernel names, in order),
  phases (step kinds it runs in), occurrences {per_layer, per_step, layers}, m_range,
  seam (an OBJECT, never a bare string) {target_callable: "pkg.module:Class.method or pkg.module:function" -- ONE Python callable whose body
        covers the whole region, so the region can be rebound to a fused implementation -- the one the LIVE server
        runs: a model can have several implementations of the same layer (generic vs platform-specific modules),
        so confirm the module from the trace's launch frames, not from a name match; source_file; lines},
  inputs_outputs (tensors that cross the region boundary, with shapes/dtypes), launches_before, launches_after,
  region_pct_device_time (weighted), est_gain_pct_upper, est_gain_pct_expected (weighted e2e %),
  difficulty (1-5), bitwise_exact (bool), exclusive_with [ids of conflicting candidates], language (${KERNEL_LANGUAGE ? `"${KERNEL_LANGUAGE}": this run's kernels are written in it` : '"triton" unless you have a reason'}),
  fallback (id of the smaller-scope candidate to implement instead if this one fails, or null),
  fused_design (how the fused kernel works: tiling, what stays in registers/LDS, what it no longer writes to HBM).
"recommended" = a list of candidate ids (not a per-candidate flag): the non-conflicting set the prompt's ranking
recommends, best first. A candidate whose region
cannot be reached through one Python seam can still be ranked, but must not be recommended.
${L3_ON ? `
## L3 stage fusion through GEAK mode=mega (on in this run)
${L3_COLLECTIVE
    ? `A MULTI-RANK stage IS implementable in this run: the kernels of one functional stage that include cross-GPU
communication, e.g. an expert-parallel MoE (dispatch -> expert GEMMs -> combine over the EP group), or an
all-reduce / all-gather / reduce-scatter together with the compute it feeds or follows (GEMM + reduce, reduce +
norm + quant, attention output + all-reduce).`
    : `The expert-parallel MoE MegaKernel (the MegaMoE pattern) IS implementable in this run, one stage over the EP group.
Its region is everything the live path launches between the MoE layer's input and its output: routing (gate GEMM,
top-k, sorting / routing metadata), the activation quant before dispatch, dispatch, expert GEMM1 + activation +
GEMM2 (with their quant), combine and the weighted reduction, and the shared expert(s) when the layer computes them
there (fused into the routed experts, or alongside them). List every one of these kernels the trace shows as ONE
candidate: mode=mega decides how much of it to fuse (it may keep some launches), so do not split it into smaller
L3 candidates. Other multi-rank stages (a TP collective with its compute, attention) stay not implemented as MegaKernels.`} GEAK mode=mega searches a fused multi-rank kernel for it on the
serving GPUs, so UNSUPPORTED_PATTERNS no longer lists it: rank it as a candidate with
level "L3", implementable_by "mega", kernels = the stage's device kernels in order (communication kernels
included), seam.target_callable = the ONE Python callable that runs the whole stage on the served path,
exclusive_with = every L1/L2 candidate whose region lies inside it, and the same gain fields as the others. It may
be recommended together with the candidates it does not conflict with. Give implementable_by "mega" to this
MegaKernel only: an L3 stage fusion that rewrites no expert GEMM (e.g. router + metadata + dispatch preparation)
is authored on one GPU as usual.
` : ''}
## Inputs
${cfg({ EVAL_DIR, TRACE_DIR, WORKLOAD_SUMMARY: workloadJson || '(none)', PROFILE_TOPN: profileTopN || '(none)',
    MODEL_PATH, BACKEND, SERVING_TP, SERVER_FLAGS: INIT_FLAGS, SERVER_ENV: INIT_ENV, LAUNCH_SCRIPT: LAUNCH_SCRIPT || '(none)',
    WORKLOAD, OVERLAY: INIT_OVERLAY || '(none)', IMPLEMENTABLE_LEVELS: L3_LEVELS, UNSUPPORTED_PATTERNS: L3_UNSUPPORTED })}

${KF_TASK ? `## Operator constraints\n${KF_TASK}\n\n` : ''}Return JSON: {"topk_json": "${OUT}/topk.json", "report_path": "${OUT}/REPORT.md", "report_md": "<full report>",
"model_notes_md": "<model notes>", "digest_md": "${OUT}/digest/digest.md", "candidates": <topk.json candidates>,
"recommended": <ids>, "notes": "..."}`,
  { phase: 'TopK', label: 'kf:analyst', schema: TOPK_SCHEMA });
  log(`[kf] topk: ${topk ? topk.candidates.length : 0} candidates, recommended ${topk ? JSON.stringify(topk.recommended) : '[]'}`);
}

const result = {
  eval_dir: KF_DIR, kf_dir: KF_DIR, kf_eval_dir: EVAL_DIR, trace_dir: TRACE_DIR, prompt: PROMPT,
  topk_json: topk ? topk.topk_json : '', analysis_report: topk ? `${EVAL_DIR}/analysis/REPORT.md` : '',
  analysis_files: topk ? { [`${EVAL_DIR}/analysis/REPORT.md`]: topk.report_md || '',
    [`${EVAL_DIR}/analysis/MODEL_NOTES.md`]: topk.model_notes_md || '' } : {},
  baseline_throughput_tok_s: baseTput, final_throughput_tok_s: baseTput, throughput_speedup: 1,
  final_overlay: INIT_OVERLAY, final_flags: INIT_FLAGS, final_env: INIT_ENV,
  accepted: [], rejected: [], status: 'analysis_only',
};
if (!topk || !(want('author') || want('apply'))) {
  if (!topk && want('topk')) { result.status = 'error'; result.reason = 'analyst returned nothing'; }
  return result;
}
// ---------------------------------------------------------------------------
// Author: an agent reads the whole Top-K and decides, by its own judgement, which patterns are worth a
// fused kernel and a region UT -- no cap on how many, no gain threshold. Each chosen pattern: region
// unittest (GEAK kernel-extract skill / kernel_extractor) -> fused kernel (one author agent).
// ---------------------------------------------------------------------------
phase('Author');
// The analyst's structured result may give a seam as the bare "pkg.mod:fn" (topk.json on disk keeps the object):
// normalise it once here, or every pattern the triage picks is dropped for having no seam.
topk.candidates = (topk.candidates || []).map((c) =>
  (c && typeof c.seam === 'string' ? { ...c, seam: { target_callable: c.seam } } : c));
const byId = new Map(topk.candidates.map((c) => [String(c.id), c]));
const seamOf = (c) => (c && c.seam && c.seam.target_callable) || '';
const triage = await safeAgent(`You decide which fusion patterns of a Top-K analysis go on to the next step: writing a
fused kernel and a region unit test for each. Judge each candidate yourself from the analysis (its numbers, the
report's reasoning, the trace digest, the source) -- there is no limit on how many you pick and no minimum gain;
pick what you judge worth the effort and say why, and say why for each one you leave out.
The one hard line: a candidate is authored by rebinding ONE Python callable that covers its whole region
(seam.target_callable) and is the one the live server actually runs (check the trace's launch frames: a model can
have several implementations of the same layer); one without such a seam cannot be authored. Implementable scope:
levels ${L3_LEVELS}, except ${L3_UNSUPPORTED}; a candidate outside it is not authored (a level alone
is no reason to skip a candidate inside it). Mutually exclusive candidates (exclusive_with)
may both be picked: which of them goes into the server is decided after their unit tests.${L3_ON ? `
An L3 MegaKernel candidate with implementable_by "mega" (${L3_COLLECTIVE ? 'an EP MoE or TP-collective stage' : 'the expert-parallel MoE stage'}) is not written by an
author agent: GEAK mode=mega searches it on all serving GPUs, which takes hours, and at most ${L3_MAX} such stage(s)
run. Pick it when its weighted gain is worth that; its exclusive L1/L2/L3 candidates may be picked too.` : ''}
Read ${topk.topk_json || '(the candidates below)'}${topk.report_md ? ' and the report below' : ''}; read anything else you need.

## Candidates
${JSON.stringify(topk.candidates || [])}

## Analyst's recommendation
${JSON.stringify(topk.recommended || [])}
${topk.report_md ? `\n## Report\n${topk.report_md}\n` : ''}
${KF_TASK ? `## Operator constraints\n${KF_TASK}\n\n` : ''}Return JSON {"selected": [{"id", "reason"}], "skipped": [{"id", "reason"}]}.`,
  { phase: 'Author', label: 'kf:triage', schema: PICK_SCHEMA });
let jobs = ((triage && triage.selected) ? triage.selected.map((x) => x && x.id) : (topk.recommended || []))
  .map((id) => byId.get(String(id))).filter((c, i, a) => c && seamOf(c) && a.indexOf(c) === i);
if (AUTHOR_K > 0) jobs = jobs.slice(0, AUTHOR_K);
// L3 stages go to mode=mega, serially and after every author agent: mega needs all the serving GPUs.
const megaJobs = jobs.filter(isMegaStage);
jobs = jobs.filter((c) => !isMegaStage(c));
for (const c of megaJobs.slice(L3_MAX)) {
  result.rejected.push({ id: String(c.id), stage: 'triage', reason: `L3 stage past kf_l3_max=${L3_MAX}` });
}
megaJobs.splice(L3_MAX);
if (L3_ON) result.l3 = megaJobs.map((c) => ({ id: String(c.id), status: 'pending', reason: '' }));
for (const x of (triage && triage.skipped) || []) {
  if (x && byId.has(String(x.id))) result.rejected.push({ id: String(x.id), stage: 'triage', reason: String(x.reason || '').slice(0, 600) });
}
log(`[kf] triage: authoring ${jobs.length} pattern(s): ${jobs.map((c) => c.id).join(', ')}` +
    (megaJobs.length ? `; L3 through mode=mega: ${megaJobs.map((c) => c.id).join(', ')}` : ''));
// Picks that all vanish here are a broken hand-off, not a measured "no gain": say so.
const picked = ((triage && triage.selected) || []).length;
if (picked && !jobs.length && !megaJobs.length) {
  result.status = 'error';
  result.reason = `triage selected ${picked} pattern(s) but none has a usable seam.target_callable`;
  return result;
}

const shortOf = (c) => String(c.id).replace(/[^A-Za-z0-9_]+/g, '_');

// AgentX: the kernel-extract skill prepares the fusion UTs (PHASE=prepare), ONE invocation PER REGION, in
// parallel, each on its own GPU and evidence dir with the whole stage budget, reusing the trace step's
// evidence. A region whose UT does not pass gets one more attempt that starts from the first one's notes.
// Same gate as e2e_workflow.js preparedUtDefect: a UT without a passing smoke and a frozen baseline is unusable.
let preparedById = new Map();
// A fusion UT must let the candidate REPLACE the region: the per-kernel UT contract (every member kernel
// fires, the split is pinned, the timer launches the members directly) makes any fusion fail its gates.
const FUSION_UT_CONTRACT = 'FUSION UT CONTRACT -- this task exists so that the next step can REPLACE the whole ' +
  'region with fewer (usually one) kernel launches, so the UT must not pin the current kernel structure: ' +
  '(1) kernel_src/ holds an editable copy of the seam callable (the host-side function whose body launches the ' +
  'region) together with the kernels it launches; the case runner and the timer call that seam callable from ' +
  'kernel_src with the recorded inputs -- never the member kernels directly -- so a candidate may change how many ' +
  'kernels the region launches, their grid/split and their intermediates. ' +
  '(2) Correctness compares only the region\'s outputs at the seam boundary (and any declared in-place mutation) ' +
  'with the frozen live baseline; intermediates between member kernels are not checked. ' +
  '(3) Execution/binding proof that every member kernel fires is required for the BASELINE leg only; for the ' +
  'candidate leg require only that the kernel_src seam implementation ran, never that the original member ' +
  'kernels still launch or keep their count/split. ' +
  '(4) The negative control mutates the OUTPUT of the kernel_src seam implementation (e.g. perturb what the host ' +
  'function returns or writes), so it still triggers after the members are fused away. ' +
  '(5) Time one call of the whole seam per case (launch overhead included) -- the win of a fusion is fewer ' +
  'launches and fewer HBM round-trips. Record "fusion_contract": true in meta.json. ';
const PREPARE_INTRO = 'Explicit user scope (overrides the default selection): build exactly ONE task for the single ' +
  'entry of FUSION_SCOPE, named by its `task`. The entry is a FUSION REGION: declare its device_kernels as one joint ' +
  'boundary (joint_ut.phases) owned by that task, reached through the entry\'s seam (or the narrowest enclosing live ' +
  'callable; record why). The candidate leg will rebind that callable to one fused implementation, so the oracle is ' +
  'taken at the region boundary. ' + FUSION_UT_CONTRACT + 'Reuse the trace step\'s evidence (PREVIOUS_WORKLOAD_JSON / PREVIOUS_RANKING_JSON and ' +
  'the traces under TRACE_DIR); do not launch a new full replay. This region is your ONLY job and the time in ' +
  'REMAINING_MIN is all yours: do not defer or stop early for budget. When an input is hard to reconstruct (quantized ' +
  'checkpoint operands, split/partial buffers, library-op wrappers), build it the way the live path does -- load the ' +
  'real tensors from MODEL_PATH, call the live producer, or bind below the wrapper -- and verify by running it. ' +
  'Other regions are prepared in parallel on the other GPUs: use only GPU_ID for UT work and write only under ' +
  'EVIDENCE_DIR and your task dir. Complete source extraction, UT authoring and GPU validation, keep a failure in ' +
  'the delivery with its concrete blocker, pack with `agentx_kernel_preparation.mjs --emit-prepared` and return ONLY ' +
  '{delivery_path, prepared} (prepared = that command\'s stdout VERBATIM). Do not optimize or integrate.';
async function prepareOne(c, i, previous) {
  const t = shortOf(c);
  const scope = [{ task: t, fusion_level: c.level, region: c.region, device_kernels: c.kernels || [],
    phases: c.phases || [], seam: c.seam || {}, inputs_outputs: c.inputs_outputs || '', m_range: c.m_range || '' }];
  const intro = previous
    ? `A previous attempt at this region did not produce a passing UT. Its outcome: ${previous}. Reuse what it saved ` +
      'under EVIDENCE_DIR, fix that blocker (do not just restate it), and finish. ' + PREPARE_INTRO
    : PREPARE_INTRO;
  const prep = await safeAgent(skillAgent('prepare', intro, { ...skillInputs(), GPU_ID: GPU_LIST[i % GPU_LIST.length],
    EVIDENCE_DIR: `${EVAL_DIR}/kernel_extraction/round_kf_${t}`, TRACE_DIR, FUSION_SCOPE: scope,
    // one agent call is capped at AGENT_TIMEOUT_MS (workflow scripts have no clock)
    SOURCE_DELIVERY_REQUIRED: true, REMAINING_MIN: AGENT_TIMEOUT_MS > 0 ? Math.floor(AGENT_TIMEOUT_MS / 60000) - 10 : 240 }),
    { phase: 'Author', label: `kf:kernel-extract prepare ${t}${previous ? ' (retry)' : ''}`, schema: AGENTX_PREPARATION_SCHEMA });
  const p = (prep && prep.prepared) || {};
  const all = [...(p.head_candidates || []), ...(p.kernel_candidates || [])];
  for (const k of all) {
    // The packer (agentx_kernel_preparation.mjs --emit-prepared) nests the UT under prepared_ut; a delivery
    // assembled by hand when the packer cannot run puts the same result (and phase) on the entry itself.
    const ut = (k && k.prepared_ut) || (k && k.result ? { phase: k.phase, result: k.result } : null);
    const r = ut && ut.result;
    if (!r || typeof r.task_dir !== 'string' || !r.task_dir.trim()) continue;
    const ok = r.editable !== false && r.baseline_frozen === true &&
      (ut.phase === 'extract' ? r.unittest_smoke : r.smoke || r.unittest_smoke) === 'pass';
    if (ok || all.length === 1) return { ...r, ok };
  }
  const why = [...(p.failed_tasks || []), ...(p.deferred_tasks || [])].map((f) => f && (f.reason || f.notes || f.status))
    .filter(Boolean).join('; ');
  return { ok: false, notes: why || (prep ? 'no task in the delivery' : 'no result') };
}
if (IS_AGENTX && jobs.length && !DEADLINE_HIT) {
  const res = await parallel(jobs.map((c, i) => async () => {
    let r = await prepareOne(c, i, null);
    if (!(r && r.ok) && !DEADLINE_HIT) r = await prepareOne(c, i, String((r && r.notes) || 'no result').slice(0, 1500));
    return [shortOf(c), r];
  }));
  for (const x of res) if (x) preparedById.set(x[0], x[1]);
  log(`[kf] kernel-extract prepared ${[...preparedById.values()].filter((r) => r && r.ok).length}/${jobs.length} fusion UT(s)`);
}

async function authorOne(c, i) {
  if (DEADLINE_HIT) return { c, skip: 'stage deadline' };
  const gpu = GPU_LIST[i % GPU_LIST.length];
  const short = shortOf(c);
  const ext = IS_AGENTX ? (preparedById.get(short) || null) : await safeAgent(roleAgent('kernel_extractor', 'extract',
    'Build the immutable unittest task dir for a FUSION REGION, not a single kernel. The baseline leg is the ' +
    'live, unfused region reached through ONE Python seam (KERNEL.target_callable, a function whose body launches ' +
    'every kernel in KERNEL.device_kernels); candidate_bind rebinds that seam, so the candidate leg runs the fused ' +
    'implementation from kernel_src/. Record I/O at the seam boundary (KERNEL.inputs_outputs) for the oracle. If ' +
    'the given seam does not cover the region, use the narrowest enclosing callable that does and say so in notes; ' +
    'if none exists return editable:false with notes "no_rebind_seam". ' + FUSION_UT_CONTRACT, {
      EVAL_DIR, MODEL_PATH, GPU_ID: gpu, WORKLOAD, SKILL_DIR: WORKFLOW_DIR,
      KERNEL: { short_name: short, classification: 'fusion', fusion_level: c.level, region: c.region,
        device_kernel: (c.kernels || [])[0] || '', device_kernels: c.kernels || [],
        target_callable: seamOf(c), source_hint: (c.seam || {}).source_file || '',
        seam_candidates: [seamOf(c)].filter(Boolean), inputs_outputs: c.inputs_outputs || '',
        phases: c.phases || [], m_range: c.m_range || '', candidate_backends: [KERNEL_LANGUAGE || c.language || 'triton'] },
      CURRENT_OVERLAY: INIT_OVERLAY, CURRENT_FLAGS: INIT_FLAGS, CURRENT_ENV: INIT_ENV,
    }), { phase: 'Author', label: `kf:extract ${short}`, schema: EXTRACT_SCHEMA });
  // Same guard as e2e extractWithBaseline: without a frozen live baseline the unittest would time the
  // fused kernel against its own scaffold.
  const extOk = IS_AGENTX ? ext && ext.ok
    : ext && ext.editable !== false && ext.unittest_smoke === 'pass' && ext.task_dir && ext.baseline_frozen !== false;
  if (!extOk) {
    return { c, skip: `extraction failed (${ext ? ext.notes || ext.unittest_smoke : 'none'})` };
  }
  const lang = KERNEL_LANGUAGE || c.language || 'triton';
  // The fused kernel is written by one agent from a plain instruction (not the GEAK kernel layer): the region,
  // the UT it must pass and the goal. It works on a copy of the task dir and hands back a diff for apply back.
  const work = `${EVAL_DIR}/kernels/${short}`;
  const au = await safeAgent(`You write ONE fused ${lang} kernel that replaces a whole kernel region of a live LLM serving stack.

## The region
${cfg({ id: c.id, level: c.level, region: c.region, kernels_in_order: c.kernels || [], phases: c.phases || [],
    m_range: c.m_range || '', seam: c.seam || {}, inputs_outputs: c.inputs_outputs || '',
    launches_before: c.launches_before || (c.kernels || []).length, launches_after: c.launches_after || 1,
    suggested_design: c.fused_design || '', bitwise_exact_expected: !!c.bitwise_exact })}

## The unit test
TASK_DIR=${ext.task_dir} is a prepared, GPU-validated unit test of this region at its seam
(${ext.target_callable || seamOf(c)}): kernel_src/ holds the editable copy of the seam callable and the kernels it
launches, cases.py the recorded serving cases, unittest.py the correctness check against the frozen live baseline
and the timing. Read its meta.json / README / run/ to see how to run it.

## Do
cp -a TASK_DIR to WORK_DIR=${work}/work and work only there (never edit TASK_DIR). Use GPU ${gpu} only. Design,
iterate and measure as you see fit, and deliver three things:
1. The fused kernel: a self-contained module under ${work}/kernel/ that replaces the region (fewer launches, usually
   one), with the region's outputs unchanged${c.bitwise_exact ? ' (bit-exact)' : ''}.
2. Its unit test: the fused version against the stock region on the recorded serving shapes (the prepared UT in
   TASK_DIR, with kernel_src/ pointed at your kernel, or your own test calling the live stock functions), checking
   every output of the region, and timing both sides with CUDA/HIP-graph replay (as the server runs decode).
   Time it on the shapes the server actually runs: the prepared UT's cases and weights may not match serving, so
   check them against the served distribution (WORKLOAD_SUMMARY=${workloadJson || '(none: derive it from the baseline server log)'}
   and the region's phases / m_range), add every served shape with real weight that the UT lacks, and weight the
   speedup by each shape's served share. Report the per-shape times.
3. The overlay that puts it in the server: ${work}/overlay/sitecustomize.py (the dir goes first on the server's
   PYTHONPATH). sitecustomize runs in EVERY Python process the server spawns (each rank, tokenizer, detokenizer,
   tools): at its top level import only the standard library -- torch / triton / the op libraries and your kernel
   module are imported inside the hook, when it fires. Patch the seam lazily -- after its module is first imported,
   never by importing it eagerly -- fall
   back to the stock path whenever the inputs do not match what the kernel supports, and print
   "[kf-${short}] ENGAGED" to stderr once, the first time the patched path actually runs. It must survive graph
   capture.${GRAPH_REQ}

## Return
Write the diff of WORK_DIR against TASK_DIR (diff -ruN) plus the kernel and overlay files to
${work}/final_patch.diff and return JSON {"final_patch": "${work}/final_patch.diff" (deliver your best correct kernel
whether or not it beats stock -- the next step judges it) or "" if you have no correct kernel, "work_dir": "${work}/work", "overlay_dir": "${work}/overlay", "banner": "[kf-${short}] ENGAGED",
"isolated_speedup": <stock region time / fused time, graph replay, weighted by served share>, "correctness":
"pass" | "fail", "launches_after": <kernel launches of the fused region>, "ut_command": "<how you ran the test>",
"notes": "<design and per-case measurements>"}.`
    + (KF_TASK ? `\n\n## Operator constraints\n${KF_TASK}` : ''),
    { phase: 'Author', label: `kf:author ${short}`, schema: AUTHOR_SCHEMA });
  const kl = au ? { eval_dir: work, final_patch: au.correctness === 'pass' ? au.final_patch || '' : '',
    overlay_dir: au.overlay_dir || '', banner: au.banner || '',
    final_geomean: au.isolated_speedup || 0, validation_status: au.correctness, reason: au.notes || '' }
    : { validation_status: 'error', reason: 'author agent returned nothing' };
  const iso = kl ? (kl.final_weighted != null ? kl.final_weighted : kl.final_geomean) : 0;
  return { c, ext, kl, lang, iso };
}
// ---- L3: one multi-rank stage through GEAK mode=mega ----------------------------------------------------
// prepare (the served operator, its frozen source tree, a multi-rank UT) -> kernel_mega.js on the serving
// GPUs -> package the delivered candidate as an overlay of the same shape an author agent returns, so select,
// the stack A/B and the correctness screen treat it like any other fused kernel. With kf_l3_mega_run only
// package runs. Instructions: kernel_fusion/l3_mega.md.
const L3_DOC = `${KF_HOME}/l3_mega.md`;
const L3_PREP_SCHEMA = obj({ status: { type: 'string', enum: ['ready', 'blocked'] },
  kernel_path: { type: 'string' }, benchmark_harness: { type: 'string' }, ep_size: { type: 'number' },
  package: { type: 'string' }, call_site: { type: 'string' }, cases: arrObj,
  mega_args: { type: 'object', additionalProperties: true }, reason: { type: 'string' } }, ['status']);
function megaSummary(ret) {
  if (!ret || typeof ret !== 'object') return null;
  const sel = String(ret.mega_selected_candidate || '');
  const row = (Array.isArray(ret.mega_candidate_registry) ? ret.mega_candidate_registry : [])
    .find((x) => x && String(x.id || '') === sel) || null;
  return { deliverable: ret.mega_deliverable === true, selected_candidate: sel,
    selected: row ? { id: row.id, tree: row.tree, final_tree: row.final_tree, patch: row.patch,
      activation: row.activation, absolute_score: row.absolute_score, per_case: row.per_case } : null,
    final_speedup: ret.final_speedup != null && Number.isFinite(Number(ret.final_speedup)) ? Number(ret.final_speedup) : null,
    eval_dir: String(ret.eval_dir || ''), report_path: String(ret.report_path || ''),
    final_patch: String(ret.final_patch || ''), note: String(ret.mega_note || '') };
}
async function megaOne(c) {
  const short = shortOf(c);
  const work = `${EVAL_DIR}/kernels/${short}`;
  const row = (result.l3 || []).find((x) => x.id === String(c.id)) || {};
  const stop = (status, why) => {
    Object.assign(row, { status, reason: String(why || '').slice(0, 600) });
    log(`[kf] L3 ${c.id}: ${status} (${row.reason})`);
    return { c, skip: `L3 ${status}: ${row.reason}`, stage: 'l3' };
  };
  if (DEADLINE_HIT) return stop('skipped', 'stage deadline');
  const stage = { id: c.id, region: c.region, kernels: c.kernels || [], phases: c.phases || [], seam: c.seam || {},
    m_range: c.m_range || '', occurrences: c.occurrences || {}, exclusive_with: c.exclusive_with || [] };
  const pkgLabel = `kf:l3 package ${short}`;
  let prep = null, mega = null;
  if (!L3_MEGA_RUN && !REUSED[pkgLabel]) {
    const prepare = (prior) => safeAgent(`You prepare one multi-rank stage for GEAK mode=mega.
Read ${L3_DOC} and follow PHASE=prepare.${prior ? ` RETRY: the previous prepare returned blocked (${prior}); read what it left
under L3_DIR, fix that cause and try once more.` : ''}

## Inputs
${cfg({ EVAL_DIR, L3_STAGE: stage, L3_DIR: work, MODEL_PATH, BACKEND, SERVING_TP, SERVING_GPU, EP_DEFAULT: L3_EP_DEFAULT,
    ...(L3_LANGUAGE ? { KERNEL_LANGUAGE: L3_LANGUAGE } : {}),
    SERVER_FLAGS: INIT_FLAGS, SERVER_ENV: INIT_ENV, CURRENT_OVERLAY: INIT_OVERLAY || '(none)', TRACE_DIR,
    WORKLOAD_SUMMARY: workloadJson || '(none)', GEAK_KERNEL_WORKFLOW_DIR: L3_GEAK_DIR,
    ...(L3_OPERATOR ? { OPERATOR_HINT: L3_OPERATOR } : {}) })}
${KF_TASK ? `\n## Operator constraints\n${KF_TASK}\n` : ''}
Return JSON {status: "ready"|"blocked", kernel_path, benchmark_harness, ep_size, package, call_site, cases, mega_args, reason}.`,
      { phase: 'Author', label: `kf:l3 prepare ${short}${prior ? ' (retry)' : ''}`, schema: L3_PREP_SCHEMA });
    const ok = (p) => !!(p && p.status === 'ready' && p.kernel_path && p.benchmark_harness);
    prep = await prepare('');
    if (!ok(prep) && !DEADLINE_HIT) prep = await prepare(String((prep && prep.reason) || 'no result').slice(0, 800));
    if (!ok(prep)) return stop('prepare_blocked', (prep && prep.reason) || 'no result');
    // policy defaults -> stage-kind defaults -> prepare's case-level args -> workflow-owned -> caller overrides
    const kind = stageKind(c);
    const megaArgs = {
      ...(L3_DEFAULTS.common || {}), ...(L3_DEFAULTS[kind] || {}),
      ...(prep.mega_args && typeof prep.mega_args === 'object' ? prep.mega_args : {}),
      mode: 'mega', workflow_dir: L3_GEAK_DIR, kernel_path: prep.kernel_path, benchmark_harness: prep.benchmark_harness,
      gpu_ids: SERVING_GPU, gpus_per_job: Math.min(SERVING_GPU.split(',').length,
        Number(prep.ep_size) > 0 ? Number(prep.ep_size) : L3_EP_DEFAULT),
      exp_root: `${work}/mega_runs`, state_dir: `${work}/mega_state`,
      budget: L3_ROUNDS, mega_time_budget_s: L3_BUDGET_S, ...L3_OVERRIDES,
    };
    if (L3_LANGUAGE) {
      megaArgs.task = `${megaArgs.task || ''}${megaArgs.task ? ' ' : ''}KERNEL LANGUAGE: write every candidate's fused ` +
        `kernel in ${L3_LANGUAGE}; a candidate in another language does not count.`;
    }
    Object.assign(row, { stage_kind: kind, ...(L3_LANGUAGE ? { kernel_language: L3_LANGUAGE } : {}), mega_args: { kernel_path: megaArgs.kernel_path,
      benchmark_harness: megaArgs.benchmark_harness, gpus_per_job: megaArgs.gpus_per_job, budget: megaArgs.budget,
      mega_time_budget_s: megaArgs.mega_time_budget_s, state_dir: megaArgs.state_dir },
      ...(L3_IGNORED.length ? { ignored_mega_args: L3_IGNORED } : {}) });
    log(`[kf] L3 ${c.id}: GEAK mode=mega on GPU ${megaArgs.gpu_ids} (gpus_per_job ${megaArgs.gpus_per_job}), ` +
        `${megaArgs.budget} rounds, ${Math.round(megaArgs.mega_time_budget_s / 60)} min`);
    try {
      mega = megaSummary(await workflow({ scriptPath: `${L3_GEAK_DIR}/kernel_mega.js` }, megaArgs));
    } catch (e) {
      return stop('mega_error', (e && e.message) || String(e));
    }
    row.mega = mega;
    if (!(mega && mega.deliverable)) return stop('mega_no_deliverable', (mega && mega.note) || 'mega returned nothing');
  }
  const pkg = await safeAgent(`You package a delivered GEAK mode=mega candidate as a KernelFusion overlay.
Read ${L3_DOC} and follow PHASE=package.

## Inputs
${cfg({ EVAL_DIR, L3_STAGE: stage, L3_DIR: work, MODEL_PATH, BACKEND, SERVING_TP, SERVING_GPU, EP_DEFAULT: L3_EP_DEFAULT,
    SERVER_FLAGS: INIT_FLAGS, SERVER_ENV: INIT_ENV, CURRENT_OVERLAY: INIT_OVERLAY || '(none)',
    OVERLAY_DIR: `${work}/overlay`, BANNER: `[kf-${short}] ENGAGED`, GEAK_KERNEL_WORKFLOW_DIR: L3_GEAK_DIR,
    ...(mega ? { MEGA_RESULT: mega, PREPARED: prep } : { MEGA_RUN: L3_MEGA_RUN }),
    ...(L3_OPERATOR ? { OPERATOR_HINT: L3_OPERATOR } : {}) })}${GRAPH_REQ ? `\n\n${GRAPH_REQ.trim()}` : ''}
${KF_TASK ? `\n## Operator constraints\n${KF_TASK}\n` : ''}
Return JSON {"final_patch": "${work}/final_patch.diff" or "" if there is no correct candidate, "work_dir", "overlay_dir":
"${work}/overlay", "banner": "[kf-${short}] ENGAGED", "isolated_speedup": <mega's verified speedup on the target case,
rank-max graph replay>, "correctness": "pass"|"fail", "launches_after", "ut_command", "notes"}.`,
    { phase: 'Author', label: pkgLabel, schema: AUTHOR_SCHEMA });
  if (!(pkg && pkg.correctness === 'pass' && pkg.final_patch && pkg.overlay_dir)) {
    return stop('package_blocked', (pkg && pkg.notes) || 'no result');
  }
  const iso = Number(pkg.isolated_speedup) || (mega && mega.final_speedup) || 0;
  Object.assign(row, { status: 'packaged', reason: '', overlay_dir: pkg.overlay_dir, isolated_speedup: iso,
    source: mega ? 'search' : (L3_MEGA_RUN ? `mega_run ${L3_MEGA_RUN}` : 'reused') });
  return { c, lang: 'mega', iso,
    ext: { short_name: short, task_dir: (prep && prep.kernel_path) || work,
      target_callable: (prep && prep.call_site) || seamOf(c) },
    kl: { eval_dir: (mega && mega.eval_dir) || work, final_patch: pkg.final_patch, overlay_dir: pkg.overlay_dir,
      banner: pkg.banner || `[kf-${short}] ENGAGED`, final_geomean: iso, validation_status: 'pass',
      reason: pkg.notes || '' } };
}
const authored = await parallel(jobs.map((c, i) => () => authorOne(c, i)));
for (const c of megaJobs) authored.push(await megaOne(c));
const authoredById = new Map(authored.filter(Boolean).map((r) => [String(r.c.id), r]));
// A kernel can go on only if it exists, passed its UT and has an overlay: everything else is the selector's call.
const usable = (r) => r && !r.skip && r.kl && r.kl.authored !== false && r.kl.final_patch && r.kl.overlay_dir;
for (const r of authored) {
  if (r && !usable(r)) {
    const why = r.skip || `no correct kernel (${r.kl ? r.kl.reason || r.kl.validation_status || '' : 'none'})`;
    log(`[kf] ${r.c.id}: ${why}`);
    result.rejected.push({ id: r.c.id, stage: r.stage || (r.skip ? 'extract' : 'author'), reason: why, isolated_speedup: r.iso || 0 });
  }
}
const ready = authored.filter(usable);

// ---------------------------------------------------------------------------
// ApplyBack: an agent reads the UT results and decides which fused kernels go on to the accuracy check and
// the e2e A/B (no gain threshold, its judgement). They go in at once as ONE combined overlay on ONE
// e2e_integrator A/B (task accuracy on the whole stack first, then reference vs the stack). Only when the
// stack fails accuracy/engagement is each kernel checked alone, and the A/B re-run without the failing ones.
// ---------------------------------------------------------------------------
phase('ApplyBack');
const utView = (r) => ({ id: r.c.id, level: r.c.level, region: r.c.region, phases: r.c.phases || [],
  seam: seamOf(r.c), exclusive_with: r.c.exclusive_with || [], region_pct_device_time: r.c.region_pct_device_time || 0,
  est_gain_pct_expected: r.c.est_gain_pct_expected || 0, ut_correctness: r.kl.validation_status,
  isolated_speedup: r.iso, launches_before: r.c.launches_before || (r.c.kernels || []).length,
  final_patch: r.kl.final_patch, author_notes: String(r.kl.reason || '').slice(0, 3000),
  ...(r.lang === 'mega' ? { authored_by: 'GEAK mode=mega: multi-rank stage, speedup is rank-max graph replay vs its frozen baseline' } : {}) });
let winners = [];
if (ready.length && want('apply') && !DEADLINE_HIT) {
  const sel = await safeAgent(`You decide which fused kernels go on to the e2e accuracy check and the e2e A/B of a
live LLM serving stack. Each one below passed its region unit test; the A/B puts all you pick into the server
together. Judge from the UT results (speedup on the served shapes, per-shape notes, how much of the step the region
is, which phases it runs in) and anything else you want to read -- no minimum speedup, no limit on how many; say
why for each one you pick or drop.
The one hard line: kernels that rebind the same callable or whose regions overlap (exclusive_with) cannot go in
together -- pick at most one of them.

## UT results
${JSON.stringify(ready.map(utView))}
${KF_TASK ? `\n## Operator constraints\n${KF_TASK}\n` : ''}
Return JSON {"selected": [{"id", "reason"}], "skipped": [{"id", "reason"}]}.`,
    { phase: 'ApplyBack', label: 'kf:select', schema: PICK_SCHEMA });
  const readyById = new Map(ready.map((r) => [String(r.c.id), r]));
  if (sel && sel.selected) {
    winners = sel.selected.map((x) => x && readyById.get(String(x.id))).filter((r, i, a) => r && a.indexOf(r) === i);
  } else {
    winners = ready.filter((r) => r.iso > 1);
  }
  // the hard line, enforced: one kernel per seam / overlapping region, first pick wins
  winners = winners.filter((r, i) => !winners.slice(0, i).some((q) => seamOf(q.c) === seamOf(r.c) ||
    (q.c.exclusive_with || []).map(String).includes(String(r.c.id)) || (r.c.exclusive_with || []).map(String).includes(String(q.c.id))));
  const why = new Map(((sel && sel.skipped) || []).map((x) => [String(x && x.id), String((x && x.reason) || '')]));
  for (const r of ready) {
    if (!winners.includes(r)) result.rejected.push({ id: r.c.id, stage: 'select', isolated_speedup: r.iso,
      reason: (why.get(String(r.c.id)) || 'not selected').slice(0, 600) });
    const l3row = r.lang === 'mega' && !winners.includes(r) && (result.l3 || []).find((x) => x.id === String(r.c.id));
    if (l3row) Object.assign(l3row, { status: 'not_selected', reason: (why.get(String(r.c.id)) || '').slice(0, 600) });
  }
  log(`[kf] select: ${winners.map((r) => `${r.c.id} (${r.iso}x)`).join(', ') || 'none'}`);
}
const kernelResult = (r) => ({ short_name: r.ext.short_name || r.c.id, task_dir: r.ext.task_dir, op_kind: 'fusion',
  winner_kind: 'authored', winner_backend: r.lang, authored_language: r.lang, authored_kernel_eval_dir: r.kl.eval_dir || '',
  target_callable: r.ext.target_callable || seamOf(r.c), live_call_seam: seamOf(r.c),
  code_patch: r.kl.final_patch, final_patch: r.kl.final_patch, verified_isolated_speedup: r.iso,
  overlay_dir: r.kl.overlay_dir || '', engaged_banner: r.kl.banner || '', pct_gpu_time: r.c.region_pct_device_time || 0,
  parity_note: r.c.bitwise_exact ? 'expected_exact' : 'expected_close' });
// Only when the whole stack fails accuracy / engagement: each kernel alone on the live server, correctness only.
async function screen(list, label) {
  const sc = await safeAgent(roleAgent('e2e_integrator', 'integrate',
    `CORRECTNESS SCREEN ONLY -- no throughput legs. The combined stack of these kernels failed its accuracy or ` +
    `engagement check; find which. For EACH entry of KERNEL_RESULTS on its own, build a CAND overlay ` +
    `${EVAL_DIR}/overlay/screen_<short_name> = CURRENT_OVERLAY plus that entry's ready overlay_dir (its sitecustomize.py ` +
    'patches one seam lazily), and run your output-correctness check of that CAND against the reference with the same ' +
    'probes, the same correctness gate (the ACCURACY GATE of your role when ACCURACY_GATE is in your inputs) and the ' +
    'same correctness env (one reference measurement shared by all entries; reuse the stack A/B\'s reference result ' +
    'on disk if it has one); confirm the entry\'s engaged_banner in the CAND server log. ' +
    'Return {members: [{short_name, correctness: "pass"|"fail", engaged, reason}]}.',
    { EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, SKILL_DIR: WORKFLOW_DIR, MEASUREMENT_PURPOSE: 'parity',
      ...KF_ACCURACY_INPUTS,
      // the stack already missed on the sampled subset: each kernel is judged on the full test set
      ...(KF_ACCURACY_GATE === 'gsm8k' ? { ACCURACY_LIMIT: ACCURACY_CONFIRM_LIMIT } : {}),
      KERNEL_RESULTS: list.map(kernelResult), CURRENT_OVERLAY: INIT_OVERLAY, CURRENT_FLAGS: INIT_FLAGS, CURRENT_ENV: INIT_ENV }),
    { phase: 'ApplyBack', label, schema: SCREEN_SCHEMA, timeoutMs: APPLY_TIMEOUT_MS });
  if (!sc) return null;
  const verdict = new Map((sc.members || []).map((m) => [String(m.short_name), m]));
  const failed = [];
  for (const r of list) {
    const m = verdict.get(String(r.ext.short_name || r.c.id)) || verdict.get(String(r.c.id));
    if (!m || m.correctness !== 'pass' || m.engaged === false) {
      failed.push(r);
      result.rejected.push({ id: r.c.id, stage: 'screen', isolated_speedup: r.iso,
        reason: m ? `e2e correctness ${m.correctness}${m.engaged === false ? ', not engaged' : ''}: ${m.reason || ''}`.slice(0, 600)
          : 'no screen verdict' });
    }
  }
  return failed;
}
if (!want('apply') || !winners.length || DEADLINE_HIT || AB_LATE) {
  for (const r of winners) result.rejected.push({ id: r.c.id, stage: 'apply', isolated_speedup: r.iso,
    reason: !want('apply') ? 'apply step disabled'
      : AB_LATE && !DEADLINE_HIT ? `stage deadline: less than ${AB_NEEDS_S}s left for the combined A/B` : 'stage deadline' });
  result.status = 'no_gain';
  return result;
}
const stackName = 'kf_stack';
const CORRECTNESS_FAIL = /^correctness/i;
async function stackAB(members, suffix) {
  const inputs = {
    EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, NOISE_BAND_PCT: noiseBand,
    MEASUREMENT_MODE: VALIDATION_MODE, MEASUREMENT_PURPOSE: 'validation', REPLICAS: VALIDATION_REPLICAS, SKILL_DIR: WORKFLOW_DIR,
    ...KF_ACCURACY_INPUTS,
    KERNEL_RESULT: { short_name: stackName, op_kind: 'fusion', winner_kind: 'authored',
      verified_isolated_speedup: Math.max(...members.map((r) => r.iso)), parity_note: 'expected_close' },
    KERNEL_RESULTS: members.map(kernelResult),
    CURRENT_OVERLAY: INIT_OVERLAY, CURRENT_FLAGS: INIT_FLAGS, CURRENT_ENV: INIT_ENV, CURRENT_THROUGHPUT: baseTput,
  };
  const intro = `Apply ALL ${members.length} fused kernels in KERNEL_RESULTS at once and gate the whole set on ONE e2e A/B ` +
    '(KERNEL_RESULT names the set). Each entry\'s overlay_dir is a READY overlay (its sitecustomize.py patches one ' +
    'seam lazily and falls back to the stock path; the kernel module sits beside it). Build ONE CAND overlay ' +
    `${EVAL_DIR}/overlay/cand_${stackName} = CURRENT_OVERLAY plus every entry's files -- rename each ` +
    'sitecustomize.py and write one sitecustomize.py that imports all of them -- do not re-materialize kernels from ' +
    'patches. Run the correctness checks once on that stack and confirm EVERY entry\'s engaged_banner in the CAND ' +
    'server log.' + (KF_ACCURACY_GATE === 'gsm8k'
      ? ` ACCURACY: score gsm8k with --limit ACCURACY_LIMIT on the reference and the CAND. If the CAND misses the gate ` +
        `there, do not reject yet: on the same two correctness servers score BOTH again with --limit ` +
        `${ACCURACY_CONFIRM_LIMIT} (the full test set), and decide the accuracy gate on that result alone; report all ` +
        'scores.' : '') +
    ' If the stack fails the accuracy gate or an engagement check, skip the throughput legs and return gate ' +
    '"rejected" with a reason that starts with "correctness:" (name the kernels that did not engage). Otherwise ONE ' +
    'reference-vs-CAND throughput A/B. Return the gate for the whole set; accepted_overlay = the CAND overlay when it ' +
    'is accepted or stack.';
  const label = `kf:integrate ${stackName}${suffix}`;
  let integ = await safeAgent(roleAgent('e2e_integrator', 'integrate', intro, inputs),
    { phase: 'ApplyBack', label, schema: INTEGRATE_SCHEMA, timeoutMs: APPLY_TIMEOUT_MS });
  // finish only an A/B that is genuinely unfinished; a correctness reject is final (it skips the timed legs on purpose)
  for (let t = 1; t <= AB_FINISH_RETRIES && (!integ || integ.gate === 'incomplete'); t++) {
    integ = await safeAgent(roleAgent('e2e_integrator', 'integrate',
      'FINISH this e2e A/B of the combined fused stack: reuse a reference leg already on disk if usable, run the ' +
      'missing leg, then return the gate. ' + intro, { ...inputs, RESUME_AB: true }),
      { phase: 'ApplyBack', label: `${label} (finish ${t})`, schema: INTEGRATE_SCHEMA, timeoutMs: APPLY_TIMEOUT_MS });
  }
  return integ;
}
let integ = await stackAB(winners, '');
const correctnessFail = (x) => x && x.gate === 'rejected' &&
  (CORRECTNESS_FAIL.test(String(x.reason || '')) || /fail/i.test(String(x.output_parity || '')));
if (correctnessFail(integ) && winners.length > 1 && !DEADLINE_HIT && !RERUN_LATE) {
  const failed = await screen(winners, 'kf:integrate screen');
  if (failed && failed.length && failed.length < winners.length) {
    winners = winners.filter((r) => !failed.includes(r));
    log(`[kf] correctness screen: ${failed.map((r) => r.c.id).join(', ')} fail; A/B again with ${winners.map((r) => r.c.id).join(', ')}`);
    integ = await stackAB(winners, ' (screened)');
  }
}
// A win is a complete A/B whose gain clears the noise band (and the stage's minimum gain).
const gainPctOf = (x) => (x && x.ref_med > 0 && x.cand_med > 0 ? 100 * (x.cand_med / x.ref_med - 1) : null);
const wonOf = (x) => !!(x && (x.gate === 'accepted' || x.gate === 'stack') && x.ab_complete !== false &&
  x.accepted_overlay && gainPctOf(x) != null && gainPctOf(x) > Math.max(Number(noiseBand) || 0, KF_MIN_GAIN_PCT));
// A stack holding an L3 kernel that did not win: gate the L1/L2 kernels alone once (with the ones only the L3
// kernel had excluded), so an L3 kernel that loses end to end does not cost them their A/B.
const l3Winners = winners.filter((r) => r.lang === 'mega');
if (!wonOf(integ) && l3Winners.length && !DEADLINE_HIT) {
  const l3Ids = new Set(l3Winners.map((r) => String(r.c.id)));
  const clashesL3 = (r) => l3Winners.some((q) => seamOf(q.c) === seamOf(r.c) ||
    (q.c.exclusive_with || []).map(String).includes(String(r.c.id)) ||
    (r.c.exclusive_with || []).map(String).some((x) => l3Ids.has(x)));
  let rest = winners.filter((r) => r.lang !== 'mega')
    .concat(ready.filter((r) => r.lang !== 'mega' && !winners.includes(r) && clashesL3(r)));
  rest = rest.filter((r, i) => !rest.slice(0, i).some((q) => seamOf(q.c) === seamOf(r.c) ||
    (q.c.exclusive_with || []).map(String).includes(String(r.c.id)) || (r.c.exclusive_with || []).map(String).includes(String(q.c.id))));
  const why = `combined A/B with L3: ${integ ? `${integ.gate} ${integ.e2e_delta_pct}% ${integ.reason || ''}` : 'no result'}`.slice(0, 600);
  for (const r of l3Winners) {
    result.rejected.push({ id: r.c.id, level: r.c.level, stage: 'apply', isolated_speedup: r.iso, reason: why });
    const row = (result.l3 || []).find((x) => x.id === String(r.c.id));
    if (row) Object.assign(row, { status: 'ab_rejected', reason: why });
  }
  if (rest.length) {
    const back = new Set(rest.filter((r) => !winners.includes(r)).map((r) => String(r.c.id)));
    result.rejected = result.rejected.filter((x) => !(x.stage === 'select' && back.has(String(x.id))));
    log(`[kf] ${why}; A/B again without L3: ${rest.map((r) => r.c.id).join(', ')}`);
    winners = rest;
    integ = await stackAB(winners, ' (without L3)');
  } else {
    winners = [];
  }
}
const ab = { members: winners.map((r) => r.c.id), gate: integ ? integ.gate : 'incomplete',
  e2e_delta_pct: integ ? integ.e2e_delta_pct : null, ref_med: integ ? integ.ref_med : null,
  cand_med: integ ? integ.cand_med : null, parity: integ ? integ.output_parity : '',
  reason: integ ? integ.reason || '' : 'no integrator result' };
result.validation = ab;
const gainBand = Math.max(Number(noiseBand) || 0, KF_MIN_GAIN_PCT);
const gainPct = gainPctOf(integ);
const won = wonOf(integ);
ab.noise_band_pct = gainBand;
if (integ && !won && gainPct != null && gainPct > 0 && (integ.gate === 'accepted' || integ.gate === 'stack')) {
  ab.reason = `gain ${gainPct.toFixed(2)}% is within the noise band ${gainBand}%; ${ab.reason}`;
}
for (const r of winners) {
  const rec = { id: r.c.id, level: r.c.level, isolated_speedup: r.iso, gate: ab.gate, e2e_delta_pct: ab.e2e_delta_pct };
  const l3row = r.lang === 'mega' && (result.l3 || []).find((x) => x.id === String(r.c.id));
  if (l3row) Object.assign(l3row, won ? { status: 'applied' } : { status: 'ab_rejected', reason: `combined A/B: ${ab.gate}` });
  if (won) result.accepted.push({ ...rec, overlay: integ.accepted_overlay });
  else result.rejected.push({ ...rec, stage: 'apply', reason: `combined A/B: ${ab.gate} ${ab.e2e_delta_pct}% ${ab.reason}`.slice(0, 600) });
}
if (won) {
  result.final_overlay = integ.accepted_overlay;
  result.final_throughput_tok_s = baseTput * integ.cand_med / integ.ref_med;
  result.throughput_speedup = integ.cand_med / integ.ref_med;
  result.status = 'accepted';
  log(`[kf] combined stack ${ab.members.join('+')}: ${ab.gate}, ${ab.e2e_delta_pct}%`);
  return result;
}
log(`[kf] combined stack ${ab.members.join('+')} not accepted (${ab.gate}, ${ab.e2e_delta_pct}%); main workflow keeps the original stack.`);
result.status = 'no_gain';
return result;
