// The mode=mega backend section of kernel_lane.js (its args, helpers, run state and phase hooks). The mega
// suites read this, not the whole file, so a pattern can only match mega code; the lane phases that call
// the hooks are in the rest of the file (`.full()`).
'use strict';
const fs = require('fs');
const path = require('path');

const full = () => fs.readFileSync(path.join(__dirname, '..', 'kernel_lane.js'), 'utf8');
module.exports = () => {
  const src = full();
  const i = src.indexOf('\n// mode=mega backend:');
  const j = src.indexOf('\n// end of mode=mega backend\n');
  if (i < 0 || j < i) throw new Error('mode=mega backend section not found in kernel_lane.js');
  return src.slice(i + 1, j);
};
module.exports.full = full;
