// Runs kf_workflow.js against scripted agent / nested-workflow answers (no GPU, no LLM).
//   node kf_workflow_mockrun.mjs <scenario> <workflow_dir>
// Prints one JSON line: {result, calls: [{label, prompt_head}], workflows: [args]}.
import { readFile } from 'node:fs/promises';

const [scenario, WF_DIR] = process.argv.slice(2);
const KF = '/tmp/kf_mock/kf';
const calls = [];
const workflows = [];

const CANDS = [
  { id: 'c1', rank: 1, level: 'L1', region: 'rmsnorm chain', kernels: ['pow', 'mean', 'mul'],
    seam: { target_callable: 'pkg.layers:RMSNorm.forward' }, region_pct_device_time: 4, bitwise_exact: true },
  { id: 'c2', rank: 2, level: 'L2', region: 'gemm+quant', kernels: ['gemm', 'quant'],
    seam: { target_callable: 'pkg.linear:apply' }, region_pct_device_time: 6 },
  { id: 'c3', rank: 3, level: 'L3', region: 'moe stage', kernels: ['a', 'b'], seam: {} },
];

const answers = {
  'kf:director:setup': () => (scenario === 'setup_fail' ? null : { eval_dir: KF, baseline_throughput_tok_s: 100, noise_band_pct: 0.5 }),
  'kf:profiler': () => ({ profile_topN_json: `${KF}/profile/round_0/profile_topN.json`, top_kernels: [] }),
  'kf:analyst': () => ({ topk_json: `${KF}/analysis/topk.json`, report_path: `${KF}/analysis/report.md`,
    candidates: CANDS, recommended: ['c1', 'c2'] }),
  'kf:extract c1': () => ({ editable: true, task_dir: `${KF}/kernels/c1_task`, unittest_smoke: 'pass', target_callable: 'pkg.layers:RMSNorm.forward', short_name: 'c1',
    baseline_frozen: scenario !== 'unfrozen' }),
  'kf:extract c2': () => ({ editable: true, task_dir: `${KF}/kernels/c2_task`, unittest_smoke: 'pass', short_name: 'c2' }),
  'kf:integrate c1': () => ({
    accept: { gate: 'accepted', ab_complete: true, e2e_throughput_tok_s: 103, ref_med: 100, cand_med: 103, e2e_delta_pct: 3, accepted_overlay: `${KF}/overlay/cand_c1` },
    stack: { gate: 'stack', ab_complete: true, e2e_throughput_tok_s: 100.2, ref_med: 100, cand_med: 100.2, accepted_overlay: `${KF}/overlay/cand_c1` },
    retry: { gate: 'incomplete', ab_complete: false, e2e_throughput_tok_s: 0, ref_med: 100 },
  }[scenario] || { gate: 'rejected', e2e_throughput_tok_s: 99, reason: 'parity_regression' }),
  'kf:integrate c1 (finish 1)': () => ({ gate: 'accepted', ab_complete: true, e2e_throughput_tok_s: 102, ref_med: 100, cand_med: 102, accepted_overlay: `${KF}/overlay/cand_c1` }),
};

async function agent(prompt, opts) {
  calls.push({ label: opts.label, phase: opts.phase, prompt });
  const f = answers[opts.label];
  if (!f) throw new Error(`unscripted agent ${opts.label}`);
  return f();
}
async function workflow(ref, a) {
  workflows.push({ scriptPath: ref.scriptPath, ...a });
  return a.kernel_path.includes('c1')
    ? { eval_dir: `${KF}/kernels/_exp/c1`, final_patch: 'diff --git a b', final_geomean: 1.4, validation_status: 'validated' }
    : { eval_dir: `${KF}/kernels/_exp/c2`, final_patch: '', final_geomean: 0.95, validation_status: 'no_gain' };
}
const parallel = (ts) => Promise.all(ts.map((t) => Promise.resolve().then(t).catch(() => null)));
const pipeline = null;
const phase = () => {};
const log = () => {};

const args = { workflow_dir: WF_DIR, kf_dir: KF, model_path: '/models/M', gpu_ids: '0,1', tp: 2,
  kf_prompt: '/prompts/p.txt', agent_timeout_ms: 0 };
if (scenario === 'analysis_only') Object.assign(args, { kf_steps: 'topk', kf_trace_dir: '/traces/x' });

const src = (await readFile(`${WF_DIR}/kernel_fusion/kf_workflow.js`, 'utf8'))
  .replace(/^[ \t]*export\s+(?=(const|let|var|function|class|async|default)\b)/gm, '');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const fn = new AsyncFunction('agent', 'parallel', 'pipeline', 'phase', 'log', 'workflow', 'args', 'budget', src);
const result = await fn(agent, parallel, pipeline, phase, log, workflow, args, {});
process.stdout.write(JSON.stringify({ result, calls, workflows }) + '\n');
