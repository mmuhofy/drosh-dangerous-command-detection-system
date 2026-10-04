/* Drosh — command risk scoring runtime.
 *
 * THE JAVASCRIPT TWIN OF ml/src/drosh_ml/normalize.py
 * ===========================��========================
 *
 * Every rule below has a numbered counterpart in the Python module, and the two
 * must stay in lockstep. If they drift, the model evaluated in Python is not
 * the model running on the device, and no amount of accuracy in the training
 * report will catch it. The mechanism that catches it is golden-vectors.json:
 * export.py records Python's scores for 300 commands, and runSelfTest() below
 * re-computes them here and reports the largest absolute difference.
 *
 * Why normalise at all
 * --------------------
 * A user typing `rm -rf` may produce any of:
 *
 *     rm -rf /
 *     rm     -rf   /
 *     r"m" -rf /
 *     Rm -Rf /
 *     <ESC>[31mrm -rf /<ESC>[0m
 *
 * Folding all of those onto a small set of canonical strings is what lets a
 * model trained on synthetic text generalise to real input.
 *
 * The one design decision that matters most here: **the output is pure ASCII**.
 * Once everything outside 0x20-0x7E has been removed, JavaScript UTF-16 code
 * units, Python code points and bytes are all the same unit, so the n-gram
 * windowing in presenceNgrams() cannot silently disagree with the trainer.
 * Without that guarantee, a single Turkish "ğ" (one Python code point, one JS
 * surrogate half) would shift every n-gram position after it.
 *
 * Pipeline order — do not reorder without reading the Python docstring:
 *   1. stripAnsi      escape sequences first; deleting control characters first
 *                     would leave the printable tail "31m" of ESC[31m as text.
 *   2. segment split  on [;|&] runs and newlines, so a single dangerous line in
 *                     a multi-line paste is still visible to dense features.
 *   3. per segment    whitespace controls to space, other controls deleted,
 *                     folded to ASCII, lowercased, whitespace collapsed.
 *   4. reassemble     joined with " ; " — a distinctive separator, so n-grams
 *                     spanning a boundary differ from those inside a command.
 *
 * Two views feed the n-gram extractor:
 *   text      the reassembled string, quotes intact
 *   unquoted  the same with ' and " removed
 *
 * The second view exists so quote obfuscation (r"m" -rf /) lands on the same
 * n-gram set as the plain form. Presence semantics are used, not counts — see
 * presenceNgrams() for why.
 */

(function (global) {
  "use strict";

  var NGRAM_MIN = 2;
  var NGRAM_MAX = 5;
  var SEGMENT_SEP = " ; ";

  // --- regexes: line-for-line mirrors of the Python originals ---------------

  var ANSI_RE =
    /\x1b(?:\[[0-?]*[ -\/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[PX^_][^\x1b]*(?:\x1b\\)|[@-Z\x5c-\x5f]|[ -\/]*[0-~])/g;

  var SEGMENT_SPLIT_RE = /[;|&\r\n]+/;
  var WS_CONTROL_RE = /[\t\v\f\r\n]/g;
  var CONTROL_RE = /[\x00-\x1f\x7f]/g;
  var SPACE_RUN_RE = / {2,}/g;
  var SPACE_TRIM_RE = /^ +| +$/g;
  var UPPER_RE = /[A-Z]/g;
  // Mn | Mc | Me — the same set as Python's category(ch).startswith("M").
  var COMBINING_MARK_RE = /\p{Mn}|\p{Mc}|\p{Me}/gu;

  // >>> BEGIN GENERATED FOLD TABLE — run scripts/sync-js.sh after editing the
  //     folding table in ml/src/drosh_ml/normalize.py
    // GENERATED from ml/src/drosh_ml/normalize.py — do not edit by hand.
    // 218 codepoints. Regenerate with:
    //   python -m drosh_ml.normalize --emit-js
    const FOLD = new Map([
      [0x00A0, " "],
      [0x00AB, "\""],
      [0x00B7, "*"],
      [0x00BB, "\""],
      [0x00C0, "A"],
      [0x00C1, "A"],
      [0x00C2, "A"],
      [0x00C3, "A"],
      [0x00C4, "A"],
      [0x00C5, "A"],
      [0x00C6, "AE"],
      [0x00C7, "C"],
      [0x00C8, "E"],
      [0x00C9, "E"],
      [0x00CA, "E"],
      [0x00CB, "E"],
      [0x00CC, "I"],
      [0x00CD, "I"],
      [0x00CE, "I"],
      [0x00CF, "I"],
      [0x00D0, "D"],
      [0x00D1, "N"],
      [0x00D2, "O"],
      [0x00D3, "O"],
      [0x00D4, "O"],
      [0x00D5, "O"],
      [0x00D6, "O"],
      [0x00D7, "x"],
      [0x00D8, "O"],
      [0x00D9, "U"],
      [0x00DA, "U"],
      [0x00DB, "U"],
      [0x00DC, "U"],
      [0x00DD, "Y"],
      [0x00DE, "TH"],
      [0x00E0, "a"],
      [0x00E1, "a"],
      [0x00E2, "a"],
      [0x00E3, "a"],
      [0x00E4, "a"],
      [0x00E5, "a"],
      [0x00E6, "ae"],
      [0x00E7, "c"],
      [0x00E8, "e"],
      [0x00E9, "e"],
      [0x00EA, "e"],
      [0x00EB, "e"],
      [0x00EC, "i"],
      [0x00ED, "i"],
      [0x00EE, "i"],
      [0x00EF, "i"],
      [0x00F0, "d"],
      [0x00F1, "n"],
      [0x00F2, "o"],
      [0x00F3, "o"],
      [0x00F4, "o"],
      [0x00F5, "o"],
      [0x00F6, "o"],
      [0x00F7, "/"],
      [0x00F8, "o"],
      [0x00F9, "u"],
      [0x00FA, "u"],
      [0x00FB, "u"],
      [0x00FC, "u"],
      [0x00FD, "y"],
      [0x00FE, "th"],
      [0x00FF, "y"],
      [0x0104, "C"],
      [0x0106, "C"],
      [0x0108, "C"],
      [0x010A, "C"],
      [0x010B, "c"],
      [0x010C, "C"],
      [0x010D, "c"],
      [0x010E, "D"],
      [0x010F, "d"],
      [0x0110, "D"],
      [0x0111, "d"],
      [0x0112, "E"],
      [0x0113, "e"],
      [0x0116, "E"],
      [0x0117, "e"],
      [0x0118, "E"],
      [0x0119, "e"],
      [0x011A, "E"],
      [0x011B, "e"],
      [0x011C, "G"],
      [0x011D, "g"],
      [0x011E, "G"],
      [0x011F, "g"],
      [0x0120, "G"],
      [0x0121, "g"],
      [0x0122, "G"],
      [0x0123, "g"],
      [0x0124, "H"],
      [0x0125, "h"],
      [0x0126, "H"],
      [0x0127, "h"],
      [0x0130, "i"],
      [0x0131, "i"],
      [0x0134, "J"],
      [0x0135, "j"],
      [0x0136, "K"],
      [0x0137, "k"],
      [0x0139, "L"],
      [0x013A, "l"],
      [0x013B, "L"],
      [0x013C, "l"],
      [0x013D, "L"],
      [0x013E, "l"],
      [0x013F, "L"],
      [0x0140, "l"],
      [0x0141, "L"],
      [0x0142, "l"],
      [0x0143, "N"],
      [0x0144, "n"],
      [0x0145, "N"],
      [0x0146, "n"],
      [0x0147, "N"],
      [0x0148, "n"],
      [0x0152, "OE"],
      [0x0153, "oe"],
      [0x0154, "R"],
      [0x0155, "r"],
      [0x0156, "R"],
      [0x0157, "r"],
      [0x0158, "R"],
      [0x0159, "r"],
      [0x015A, "S"],
      [0x015B, "s"],
      [0x015C, "S"],
      [0x015D, "s"],
      [0x015E, "S"],
      [0x015F, "s"],
      [0x0162, "T"],
      [0x0163, "t"],
      [0x0164, "T"],
      [0x0165, "t"],
      [0x0168, "U"],
      [0x0169, "u"],
      [0x016A, "U"],
      [0x016B, "u"],
      [0x016C, "U"],
      [0x016D, "u"],
      [0x016E, "U"],
      [0x016F, "u"],
      [0x0170, "U"],
      [0x0171, "u"],
      [0x0172, "U"],
      [0x0173, "u"],
      [0x0174, "W"],
      [0x0175, "w"],
      [0x0176, "Y"],
      [0x0177, "y"],
      [0x0178, "Y"],
      [0x0179, "Z"],
      [0x017A, "z"],
      [0x017B, "Z"],
      [0x017C, "z"],
      [0x017D, "Z"],
      [0x017E, "z"],
      [0x017F, "s"],
      [0x0218, "S"],
      [0x0219, "s"],
      [0x021A, "T"],
      [0x021B, "t"],
      [0x0250, "a"],
      [0x0251, "a"],
      [0x0259, "e"],
      [0x025B, "e"],
      [0x025C, "e"],
      [0x0269, "i"],
      [0x026A, "i"],
      [0x026F, "u"],
      [0x028F, "y"],
      [0x02BC, "'"],
      [0x1680, " "],
      [0x2000, " "],
      [0x2001, " "],
      [0x2002, " "],
      [0x2003, " "],
      [0x2004, " "],
      [0x2005, " "],
      [0x2006, " "],
      [0x2007, " "],
      [0x2008, " "],
      [0x2009, " "],
      [0x200A, " "],
      [0x200B, ""],
      [0x200C, ""],
      [0x200D, ""],
      [0x2010, "-"],
      [0x2011, "-"],
      [0x2012, "-"],
      [0x2013, "-"],
      [0x2014, "-"],
      [0x2018, "'"],
      [0x2019, "'"],
      [0x201A, "'"],
      [0x201B, "'"],
      [0x201C, "\""],
      [0x201D, "\""],
      [0x201E, "\""],
      [0x201F, "\""],
      [0x2022, "*"],
      [0x2026, "..."],
      [0x202F, " "],
      [0x2032, "'"],
      [0x2033, "\""],
      [0x205F, " "],
      [0x2212, "-"],
      [0xFB00, "ff"],
      [0xFB01, "fi"],
      [0xFB02, "fl"],
      [0xFB03, "ffi"],
      [0xFB04, "ffl"],
      [0xFEFF, ""],
      [0xFFFD, ""],
    ]);
  // <<< END GENERATED FOLD TABLE

  // --- step 1 ---------------------------------------------------------------

  function stripAnsi(value) {
    return value.replace(ANSI_RE, "");
  }

  // --- step 3 ---------------------------------------------------------------

  function foldAscii(value) {
    var decomposed = value.normalize("NFKD");
    var withoutMarks = decomposed.replace(COMBINING_MARK_RE, "");

    var kept = "";
    var dropped = 0;

    // Iterating with for..of walks code points, matching Python's str iteration.
    for (var ch of withoutMarks) {
      var code = ch.codePointAt(0);
      var mapped = FOLD.get(code);
      var out = mapped !== undefined ? mapped : ch;
      for (var i = 0; i < out.length; i++) {
        var outCode = out.charCodeAt(i);
        if (outCode >= 0x20 && outCode <= 0x7e) {
          kept += out[i];
        } else {
          dropped++;
        }
      }
    }

    return { text: kept, dropped: dropped };
  }

  function normaliseSegment(value) {
    var spaced = value.replace(WS_CONTROL_RE, " ");
    var stripped = spaced.replace(CONTROL_RE, "");
    var folded = foldAscii(stripped);
    var lowered = folded.text.replace(UPPER_RE, function (c) {
      return String.fromCharCode(c.charCodeAt(0) + 32);
    });
    return lowered.replace(SPACE_RUN_RE, " ").replace(SPACE_TRIM_RE, "");
  }

  function normalize(raw) {
    var withoutEscapes = stripAnsi(String(raw));
    var rawFold = foldAscii(withoutEscapes);

    var segments = withoutEscapes
      .split(SEGMENT_SPLIT_RE)
      .map(normaliseSegment)
      .filter(function (segment) {
        return segment.length > 0;
      });

    var text = segments.join(SEGMENT_SEP);
    var unquoted = text.split("'").join("").split('"').join("");

    return {
      raw: String(raw),
      text: text,
      unquoted: unquoted,
      segments: segments,
      droppedNonAscii: rawFold.dropped,
      views: [text, unquoted],
    };
  }

  // --- steps 5-6 ------------------------------------------------------------

  /* Presence semantics, not counts. Deliberate, not a simplification:
   *
   *   rm -rf / rm -rf / rm -rf /
   * must score identically to `rm -rf /`. With counts the score grows
   * superlinearly, and someone who cannot read the model can still inflate or
   * deflate it by repetition.
   *
   * It also means no length normalisation is required, so sum(w_j * x_j) folds
   * exactly into one flat coefficient array — which is what makes the exported
   * model a single float array with no vocabulary object at inference time
   * beyond the n-gram -> index lookup.
   */
  function presenceNgrams(views) {
    var found = new Set();
    for (var v = 0; v < views.length; v++) {
      var view = views[v];
      var length = view.length;
      for (var size = NGRAM_MIN; size <= NGRAM_MAX; size++) {
        if (length < size) break;
        for (var start = 0; start + size <= length; start++) {
          found.add(view.slice(start, start + size));
        }
      }
    }
    return found;
  }

  // --- scoring --------------------------------------------------------------
  //
  // Implemented in export.py's phase (step 4). Declared here so index.html can
  // feature-detect it and show "model not loaded" rather than throwing.

  var api = {
    // Must match drosh_ml.NORMALIZE_VERSION. The exporter stamps this value
    // into the model; the loader refuses a model whose normalisation differs,
    // because a model scored against other rules produces meaningless numbers.
    NORMALIZE_VERSION: "1",
    NGRAM_MIN: NGRAM_MIN,
    NGRAM_MAX: NGRAM_MAX,
    SEGMENT_SEP: SEGMENT_SEP,
    stripAnsi: stripAnsi,
    foldAscii: foldAscii,
    normalize: normalize,
    presenceNgrams: presenceNgrams,
    score: null,
    runSelfTest: null,
  };

  global.DroshRisk = api;
})(typeof globalThis !== "undefined" ? globalThis : this);