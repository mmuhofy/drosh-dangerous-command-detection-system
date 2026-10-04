/* Locate which dense feature drifts between Python and JavaScript.
 *
 * golden-vectors.json holds Python's scores; scoring in JS and diffing tells us
 * *that* something is wrong, but not what. This compares the two dense
 * implementations feature by feature on a handful of commands, so the offending
 * entry is named rather than guessed at.
 *
 * Requires the Python dense vector to be dumped alongside: see
 * tools/dump_dense.py, which writes tools/dense_reference.json.
 *
 *   node tools/diagnose_parity.js
 */
globalThis.window = globalThis;
const fs = require("fs");
const vm = require("vm");

vm.runInThisContext(fs.readFileSync("html/command-risk/risk-model.js", "utf8"));
vm.runInThisContext(fs.readFileSync("html/command-risk/risk-scoring.js", "utf8"));

const R = globalThis.DroshRisk;
const model = R.buildModel(globalThis.DROSH_RISK_MODEL);
const reference = JSON.parse(fs.readFileSync("tools/dense_reference.json", "utf8"));

const names = R.LABELS ? null : null;
let worstCount = 0;

for (const entry of reference) {
  const norm = R.normalize(entry.command);
  const js = R.denseFeatures(norm);
  const py = entry.dense;

  const diffs = [];
  for (let i = 0; i < Math.max(js.length, py.length); i++) {
    const a = js[i];
    const b = py[i];
    if (Math.abs(a - b) > 1e-9) diffs.push({ i, js: a, py: b });
  }

  const label = diffs.length ? "DIFF" : "ok  ";
  console.log(`${label} ${String(entry.command).slice(0, 44).padEnd(46)} ${diffs.length} feature(s)`);
  for (const d of diffs) {
    const name = globalThis.__DENSE_NAMES ? globalThis.__DENSE_NAMES[d.i] : `#${d.i}`;
    console.log(`       [${d.i}] js=${d.js} py=${d.py}`);
    worstCount++;
  }
}
console.log(`\ntotal differing features: ${worstCount}`);
