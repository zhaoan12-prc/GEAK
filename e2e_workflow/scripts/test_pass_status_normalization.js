#!/usr/bin/env node
// Regression guard for extractor PASS-status compatibility (no GPU or model required).
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const WORKFLOW = path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js');
const src = fs.readFileSync(WORKFLOW, 'utf8');
const start = src.indexOf('function isPassStatus');
const end = src.indexOf('// A FROZEN baseline', start);

if (start < 0 || end <= start) {
  console.error('FAIL: isPassStatus block not found in e2e_workflow.js');
  process.exit(1);
}

const isPassStatus = new Function(
  `${src.slice(start, end)}\nreturn isPassStatus;`,
)();

const accepted = [
  'pass', ' PASS ', 'passed', 'RESULT: PASS',
  { result: 'PASS' }, { status: 'passed' },
  { result: { status: 'PASS' } },
];
const rejected = [
  null, undefined, false, true, 1, '', 'fail', '1 failed, 9 passed',
  { result: 'FAIL' }, { status: 'unknown' }, {},
];

let failures = 0;
for (const value of accepted) {
  if (!isPassStatus(value)) {
    console.error('FAIL: should accept', JSON.stringify(value));
    failures++;
  }
}
for (const value of rejected) {
  if (isPassStatus(value)) {
    console.error('FAIL: should reject', JSON.stringify(value));
    failures++;
  }
}

if (failures) process.exit(1);
console.log(`PASS: ${accepted.length} accepted and ${rejected.length} rejected status forms`);
