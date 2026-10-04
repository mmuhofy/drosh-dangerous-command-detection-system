/* Parity runner — used by ml/tests/test_parity.py.
 *
 * Reads a JSON array of raw command strings on stdin, writes a JSON array of
 * normalisation results on stdout. Deliberately tiny and dependency-free: the
 * point is to invoke the *real* html/command-risk/risk-scoring.js, not a
 * reimplementation of it, because a reimplementation would test nothing.
 *
 * Usage: echo '["rm -rf /"]' | node html/command-risk/parity-runner.mjs
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import vm from "node:vm";

const here = dirname(fileURLToPath(import.meta.url));
vm.runInThisContext(readFileSync(join(here, "risk-scoring.js"), "utf8"), {
  filename: "risk-scoring.js",
});

const { normalize, presenceNgrams } = globalThis.DroshRisk;

const raw = JSON.parse(readFileSync(0, "utf8"));

const out = raw.map((input) => {
  const result = normalize(input);
  return {
    text: result.text,
    unquoted: result.unquoted,
    segments: result.segments,
    droppedNonAscii: result.droppedNonAscii,
    ngramCount: presenceNgrams(result.views).size,
    // Sorted so the comparison in Python is order-independent.
    ngrams: [...presenceNgrams(result.views)].sort(),
  };
});

process.stdout.write(JSON.stringify(out));