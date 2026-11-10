#!/usr/bin/env node
// A step that ENABLES a fusion is not a step that speeds one up.
//
// The round filter is `primSpeedup(ver) > 1.0`. For a fusion built in stages that is not a quality
// bar, it is a structural block: the producer half adds completion signalling and a second buffer,
// has no consumer yet, and can only measure its own overhead. It is supposed to be slower. Under a
// speed-only filter it is discarded, never enters the canonical tree, and the consumer half is then
// written against a tree where the producer half does not exist -- so it is never written at all.
// That is how this project reached a half-fused kernel and stopped there.
//
// Each assertion below is a way the fix could be wrong in a way nobody would notice:
//   - `enabling` becomes a way to commit anything, including a step that does not work
//   - a prerequisite is committed but its cost vanishes, so the tree silently gets slower
//   - the debt is tracked but never comes due, so a chain can be abandoned half-built in silence
//   - the terminal step is measured against the degraded tree and credited with removing the
//     overhead its own prerequisites installed
'use strict';

const fs = require('fs');
const path = require('path');

const WF = path.resolve(__dirname, '..');
const src = require('./mega_source')();

let failures = 0;
const ok = (cond, msg) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg); failures++; }
};

const m = src.match(/\/\/ <<REPLAY:enabling_step>>([\s\S]*?)\/\/ <<\/REPLAY:enabling_step>>/);
if (!m) { console.error('  FAIL: no <<REPLAY:enabling_step>> region — nothing to test'); process.exit(1); }
const { stepRoleOf, chainDebtReport, CHAIN_DEBT_MAX_ROUNDS } =
  // eslint-disable-next-line no-new-func
  new Function(`${m[1]}\nreturn { stepRoleOf, chainDebtReport, CHAIN_DEBT_MAX_ROUNDS };`)();

console.log('\n# the default does not change');
{
  ok(stepRoleOf({}) === 'terminal' && stepRoleOf(null) === 'terminal',
     'a direction that says nothing is terminal — the speed bar stays the default for everything else');
  ok(stepRoleOf({ step_role: 'ENABLING' }) === 'enabling',
     'the role is read case-insensitively; a direction is not silently demoted by capitalisation');
  ok(stepRoleOf({ step_role: 'prerequisite' }) === 'terminal',
     'an unrecognised role falls back to the STRICTER bar, not the looser one');
}

console.log('\n# the debt has to come due');
{
  const debt = [{ round: 1, id: 'r1_d0', enables: 'D2', cost_pct: 2.0 },
                { round: 2, id: 'r2_d1', enables: 'D2', cost_pct: 1.5 }];
  const open = chainDebtReport(debt, 2, new Set());
  ok(open.open.length === 1 && open.open[0].cost_pct === 3.5,
     'two prerequisites for the same rung accumulate into one balance of 3.5%');
  ok(/D2 owes 3.5%/.test(open.caveat) && /round 1/.test(open.caveat),
     'and the caveat names the rung, the amount, and how long it has been owed');
  ok(chainDebtReport(debt, 2, new Set(['D2'])).caveat === '',
     'once the terminal rung is MEASURED the balance clears — measured, not won: a chain that closed and lost still closed');
}
{
  const debt = [{ round: 1, id: 'r1_d0', enables: 'D2', cost_pct: 2.0 }];
  ok(chainDebtReport(debt, 1, new Set()).overdue.length === 0,
     'a chain opened this round is not yet overdue');
  ok(chainDebtReport(debt, 1 + CHAIN_DEBT_MAX_ROUNDS, new Set()).overdue.length === 1,
     `and it is overdue after ${CHAIN_DEBT_MAX_ROUNDS} rounds — a chain abandoned half-built leaves the tree slower with nothing to show`);
  ok(/OVERDUE/.test(chainDebtReport(debt, 9, new Set()).caveat),
     'which is said in words, not only in a field nobody reads');
}
{
  ok(chainDebtReport([], 3, new Set()).caveat === '' && chainDebtReport(null, 3, new Set()).open.length === 0,
     'no debt produces no noise, so the caveat means something when it appears');
}

console.log('\n# wired into the round');
{
  ok(/CHAIN_DEBT: chainDebtReport\(CHAIN_DEBT, (currentRound|round), LADDER_MEASURED\)\.open/.test(src),
     'the balance reaches the planner, which is the only phase that can decide to close the chain');
  ok(/CHAIN_CAVEATS\.length \? \{ CHAIN_CAVEATS \}/.test(src),
     'and the report, so a wave that stopped half-fused does not read as a failed search');
}

console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exit(failures ? 1 : 0);
