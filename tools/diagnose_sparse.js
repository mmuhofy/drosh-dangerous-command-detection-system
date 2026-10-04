/* Find the n-grams JS misses for a single command.
 *
 * The dense block now agrees exactly with features.py, so a remaining score
 * divergence against golden-vectors.json has to be in the sparse sum: some
 * n-grams Python folded into the vocabulary have no entry on the JS side.
 *
 * A score difference says "something is wrong" but not "what". This prints the
 * windowed set next to the lookup result and the coefficient contribution, so
 * the missing entries are visible by name.
 *
 *   node tools/diagnose_sparse.js "sed --interactive '' 's/foo/bar/g' src/*.js"
 */
globalThis.window = globalThis;
const fs = require("fs");
const vm = require("vm");

vm.runInThisContext(fs.readFileSync("html/command-risk/risk-model.js", "utf8"));
vm.runInThisContext(fs.readFileSync("html/command-risk/risk-scoring.js", "utf8"));

const R = globalThis.DroshRisk;
const model = R.buildModel(globalThis.DROSH_RISK_MODEL);

// Must match train.VIEW_JOIN. Written as String.fromCharCode because a literal
// NUL cannot survive being pasted through a shell heredoc.
const VIEW_JOIN = String.fromCharCode(0);

const command = process.argv[2] || "ls -la";
const norm = R.normalize(command);
const joined = norm.views.join(VIEW_JOIN);
const grams = R.windowNgrams(joined, model.meta.ngramMin, model.meta.ngramMax);

let hits = 0;
let missing = 0;
let total = 0;
const missingList = [];
const contributing = [];

for (const gram of grams) {
  const index = model.vocabIndex[gram];
  if (index === undefined) {
    missing++;
    if (missingList.length < 40) missingList.push(gram);
  } else {
    hits++;
    total += model.coef[index];
    contributing.push({ gram, coef: model.coef[index] });
  }
}

console.log(`command      : ${command}`);
console.log(`views        : ${JSON.stringify(norm.views)}`);
console.log(`grams        : ${grams.size}   hits: ${hits}   missing: ${missing}`);
console.log(`sparse total : ${total.toFixed(6)}`);

console.log(`\nmissing grams (no coefficient, so no contribution):`);
for (const gram of missingList) {
  const hasNul = gram.indexOf(VIEW_JOIN) !== -1;
  console.log(`   ${JSON.stringify(gram)}${hasNul ? "   <- spans the view join" : ""}`);
}

console.log(`\nlargest contributors:`);
contributing
  .sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef))
  .slice(0, 10)
  .forEach((entry) => {
    console.log(`   ${entry.coef.toFixed(5).padStart(10)}  ${JSON.stringify(entry.gram)}`);
  });