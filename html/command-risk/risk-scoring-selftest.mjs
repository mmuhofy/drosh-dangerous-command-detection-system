/* Self-test for risk-scoring.js.
 *
 * These are the same 13 cases asserted by
 * `python -m drosh_ml.normalize` — the tables below are transcribed from
 * _self_test() in ml/src/drosh_ml/normalize.py. If you change a rule there,
 * change it here, and run `scripts/sync-js.sh`.
 *
 * Cross-language parity is a separate, stronger check: ml/tests/test_parity.py
 * runs both implementations over a shared corpus and asserts byte equality.
 * This file only guarantees that the JS side satisfies the contract on its own,
 * and it runs with no Python and no npm install.
 *
 * Usage: node html/command-risk/risk-scoring-selftest.mjs
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import vm from "node:vm";

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(here, "risk-scoring.js"), "utf8");
vm.runInThisContext(source, { filename: "risk-scoring.js" });

const { normalize, presenceNgrams } = globalThis.DroshRisk;

const CASES = [
  ["plain", "rm -rf /", "rm -rf /"],
  ["tab+spaces", "rm\t\t-rf   /", "rm -rf /"],
  ["newline segments", "ls\nrm -rf /", "ls ; rm -rf /"],
  ["ansi colour", "\x1b[31mrm -rf /\x1b[0m", "rm -rf /"],
  ["ansi 256", "\x1b[38;5;196mrm -rf /", "rm -rf /"],
  ["osc title", "\x1b]0;title\x07rm -rf /", "rm -rf /"],
  ["uppercase", "RM -RF /", "rm -rf /"],
  ["turkish", "ĞÜNYE", "gunye"],
  ["turkish dotted I", "İndirilenler", "indirilenler"],
  ["accented", "café", "cafe"],
  ["nbsp", "rm -rf /", "rm -rf /"],
  ["quoted", 'r"m" -rf /', 'r"m" -rf /'],
  ["empty", "   ", ""],
];

let failures = 0;

const pad = (value, width) => String(value).padEnd(width);

console.log(`${pad("case", 22)}${pad("expected", 24)}${pad("actual", 24)}ok`);
console.log("-".repeat(78));

for (const [name, raw, expected] of CASES) {
  const actual = normalize(raw).text;
  const ok = actual === expected;
  if (!ok) failures++;
  console.log(
    `${pad(name, 22)}${pad(JSON.stringify(expected), 24)}${pad(JSON.stringify(actual), 24)}${ok ? "yes" : "NO"}`,
  );
}

const plain = presenceNgrams(normalize("rm -rf /").views);
const quoted = presenceNgrams(normalize('r"m" -rf /').views);
let shared = 0;
for (const gram of plain) if (quoted.has(gram)) shared++;
const overlap = shared / Math.max(1, plain.size);

console.log(`\nquote-obfuscation n-gram overlap with plain form: ${(overlap * 100).toFixed(1)}%`);

if (overlap < 1) {
  console.log("expected the unquoted view to recover the plain n-gram set exactly");
  failures++;
}

console.log(`total failures: ${failures}`);
process.exit(failures === 0 ? 0 : 1);