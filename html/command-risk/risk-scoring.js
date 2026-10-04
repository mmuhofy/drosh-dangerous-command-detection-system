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
  // Canonical form of each shell operator. | and ; are deliberately distinct:
  // collapsing them made `curl x | sh` identical to `curl x ; sh`, which
  // destroyed the pipe-to-shell signal.
  var OPERATORS = {
    nl: " ; ",
    pipe2: " || ",
    pipe: " | ",
    and2: " && ",
    and: " & ",
  };

  // --- regexes: line-for-line mirrors of the Python originals ---------------

  var ANSI_RE =
    /\x1b(?:\[[0-?]*[ -\/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[PX^_][^\x1b]*(?:\x1b\\)|[@-Z\x5c-\x5f]|[ -\/]*[0-~])/g;

  var SEGMENT_SPLIT_RE = /([;|&]+|[\r\n]+)/;
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

  /* Break an operator run into canonical, individually-spaced operators.
     Greedy longest-match on the two-character forms first, so "&&" and "||"
     survive intact, and each remaining single character becomes its own token.
     Mirrors _split_operators() in ml/src/drosh_ml/normalize.py. */
  function splitOperators(token) {
    var out = [];
    var i = 0;
    var n = token.length;
    while (i < n) {
      var ch = token[i];
      if (ch === "\n" || ch === "\r" || ch === ";") {
        out.push(OPERATORS.nl);
        i += 1;
      } else if (ch === "|" && token[i + 1] === "|") {
        out.push(OPERATORS.pipe2);
        i += 2;
      } else if (ch === "&" && token[i + 1] === "&") {
        out.push(OPERATORS.and2);
        i += 2;
      } else if (ch === "|") {
        out.push(OPERATORS.pipe);
        i += 1;
      } else if (ch === "&") {
        out.push(OPERATORS.and);
        i += 1;
      } else {
        out.push(" " + ch + " ");
        i += 1;
      }
    }
    return out;
  }

  function normalize(raw) {
    var withoutEscapes = stripAnsi(String(raw));
    var rawFold = foldAscii(withoutEscapes);

    // Split into alternating content/operator runs; odd indices are operators.
    var pieces = withoutEscapes.split(SEGMENT_SPLIT_RE);

    var segments = [];
    var rebuilt = [];
    for (var i = 0; i < pieces.length; i++) {
      if (i % 2 === 1) {
        rebuilt = rebuilt.concat(splitOperators(pieces[i]));
        continue;
      }
      var segment = normaliseSegment(pieces[i]);
      if (segment.length > 0) {
        segments.push(segment);
        rebuilt.push(segment);
      }
    }

    var text = rebuilt
      .join(" ")
      .replace(SPACE_RUN_RE, " ")
      .replace(SPACE_TRIM_RE, "");
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

  /* Split a MODEL_JS blob into lookup structures.
   *
   * The vocabulary is shipped as one newline-delimited string rather than an
   * object literal, because 55k entries as {"gram": i} is roughly 40% larger
   * once the JSON quoting is counted, and this file is downloaded by a phone.
   * Split once at load; the array is then indexed by position. */
  function buildModel(blob) {
    /* The vocabulary arrives base64-encoded because it contains the view-join
     * NUL, and a raw 0x00 byte is not legal inside a JS string literal. It also
     * compresses far better than escaped text would.
     *
     * One atob + one UTF-8 decode at startup, ~55k entries, negligible. */
    var bytes = atob(blob.vocabB64);
    var bytes8 = new Uint8Array(bytes.length);
    for (var b = 0; b < bytes.length; b++) bytes8[b] = bytes.charCodeAt(b);
    var vocab = new TextDecoder("utf-8").decode(bytes8).split("\n");

    var vocabIndex = Object.create(null);
    for (var i = 0; i < vocab.length; i++) vocabIndex[vocab[i]] = i;

    return {
      meta: blob,
      vocab: vocab,
      vocabIndex: vocabIndex,
      coef: blob.coef,
      denseCoef: blob.denseCoef,
      denseMean: blob.denseMean,
      denseStd: blob.denseStd,
      bias: blob.bias,
      thresholds: blob.thresholds,
    };
  }

  /* Character n-grams of one already-joined string, presence semantics.
   * Mirrors window_ngrams() in ml/src/drosh_ml/train.py. */
  function windowNgrams(joined, min, max) {
    var found = new Set();
    var length = joined.length;
    for (var size = min; size <= max; size++) {
      if (length < size) break;
      for (var start = 0; start + size <= length; start++) {
        found.add(joined.slice(start, start + size));
      }
    }
    return found;
  }

  /* Dense features — a line-for-line mirror of
   * ml/src/drosh_ml/features.py::dense_features.
   *
   * This is the riskiest function in the file: the coefficients were fitted
   * against Python's output for these 38 numbers. The golden-vector fixture is
   * what proves the two agree; without it a silent drift here would only show
   * up as subtly wrong scores.
   *
   * Returns a plain array in DENSE_NAMES order. */
  var DENSE_NAMES = [
    "segment_count", "has_sudo", "has_su", "has_doas", "rm_recursive_force",
    "rm_any_recursive", "delete_target_system_path", "delete_target_home",
    "delete_target_wildcard", "delete_target_relative", "delete_target_disposable",
    "has_dd", "dd_to_block_device", "has_mkfs", "has_fs_table_tool", "has_fork_bomb",
    "has_chmod_777", "chmod_recursive", "has_chown_recursive_root", "has_curl_or_wget",
    "curl_pipe_to_shell", "has_base64_decode", "has_eval", "has_history_tamper",
    "touches_ssh_keys", "has_shutdown", "has_kill_broad", "has_git_destructive",
    "has_docker_prune", "has_kubectl_delete", "has_package_remove",
    "has_truncate_or_log_empty", "subshell_depth", "backtick_count",
    "quote_nesting", "flag_density", "path_depth", "mentions_only",
  ];

  function denseFeatures(norm) {
    var text = norm.text;
    var joined = norm.segments.join(" ");
    var bin = function (b) {
      return b ? 1.0 : 0.0;
    };
    var hit = function (re) {
      return re.test(text);
    };

    var SYSTEM = /(?:^|[\s;&|(])\/(?:etc|usr|var|bin|sbin|boot|lib|opt|srv|root|home|sys|proc|dev|sdcard|storage|data)(?:\/|\b)/;
    var HOME = /(?:^|[\s;&|(])(~|\$\{?HOME\}?|\$\{?PWD\}?|\$\(pwd\))(?:\/|\b|$)/;
    var WILDCARD = /(?:^|[\s;&|(])\*(?:\.\*)?\*?(?:\/|$)/;
    var RELATIVE = /(?:^|[\s;&|(])\.|\s\./;
    var DISPOSABLE = /node_modules|\bbuild\/?\b|\bdist\/?\b|\btarget\/?\b|\.gradle|\.venv|\bvenv\/?\b|__pycache__|\.pytest_cache|\.mypy_cache|\.next|\.nuxt|\.cache|vendor\/?\b|pods\/?\b|deriveddata|\.terraform|\bcoverage\b|\.tox|\.parcel-cache|\.turbo|\*\.o\b|\*\.class\b|\*\.pyc\b|\.ds_store|\*\.log\b|\*\.tmp\b|\*\.bak\b/;
    var BLOCKDEV = /\/dev\/(?:sd[a-z]|nvme\d|mmcblk|hd[a-z])/;
    var FSTABLE = /\b(?:fdisk|parted|sgdisk|gparted|blkdiscard|wipefs)\b/;
    var RMSEG = /\brm\b/;

    var hasRmRf = /\brm\b[^;|&]*\s-{1,2}[a-z]*r[a-z]*f\b|\brm\s+-[a-z]*f[a-z]*r\b/.test(text);
    var rmRecursive = /\brm\b[^;|&]*\s-{1,2}(?:r\b|recursive)/.test(text);

    var rmSegment = joined;
    for (var i = 0; i < norm.segments.length; i++) {
      if (RMSEG.test(norm.segments[i])) {
        rmSegment = norm.segments[i];
        break;
      }
    }

    var pipeToShell =
      /(?:curl|wget|fetch)[^;|&]*\|\s*(?:sudo\s+)?(?:ba|z|k|da)?sh\b/.test(text) ||
      /\|\s*(?:sudo\s+)?(?:python3?|perl|ruby|node|php)\b/.test(text);

    var subshellDepth = countOf(text, "$(") + countOf(text, "${");
    var backticks = countOf(text, "`");
    var quotes = countOf(text, '"') + countOf(text, "'");

    var flags = (text.match(/(?:^|\s)-{1,2}[a-zA-Z][\w-]*/g) || []).length;
    var words = Math.max(1, text.split(/\s+/).filter(Boolean).length);

    var paths = text.match(/(?:\/|~|\$\{?HOME\}?)\/[\w./-]*/g) || [];
    var pathDepth = 0;
    for (var j = 0; j < paths.length; j++) {
      var d = countOf(paths[j], "/");
      if (d > pathDepth) pathDepth = d;
    }

    var hasDangerWord = /\brm\b|\bdd\b|mkfs|chmod 777|fork bomb|:\{/.test(text);
    var hasDangerVerb = /\brm\s+-|^\s*dd\b|\bmkfs\b|chmod|chown|shutdown|reboot|kill/.test(text);

    return [
      norm.segments.length,
      bin(/(?:^|\s)sudo\b/.test(text)),
      bin(/(?:^|\s)su\s+-|\bsu\s+c\b/.test(text)),
      bin(/(?:^|\s)doas\b/.test(text)),
      bin(hasRmRf),
      bin(rmRecursive),
      bin(SYSTEM.test(rmSegment)),
      bin(HOME.test(rmSegment)),
      bin(WILDCARD.test(rmSegment)),
      bin(RELATIVE.test(rmSegment)),
      bin(DISPOSABLE.test(rmSegment)),
      bin(/\bdd\b/.test(text)),
      bin(/\bdd\b/.test(text) && BLOCKDEV.test(text)),
      bin(/\bmkfs\b|\bmke2fs\b|\bmkswap\b/.test(text)),
      bin(FSTABLE.test(text)),
      bin(/:\s*\(\s*\)\s*\{/.test(text)),
      bin(/chmod[^;|&]*777/.test(text)),
      bin(/chmod\s+-R\b|chmod\s+-r\b/.test(text)),
      bin(/chown\s+-R\b[^;|&]*(?:root|0:0)/.test(text)),
      bin(/\b(?:curl|wget|fetch)\b/.test(text)),
      bin(pipeToShell),
      bin(/base64\s+(?:-d|--decode|-D)/.test(text)),
      bin(/(?:^|\s)eval\b/.test(text)),
      bin(/history\s+-c|unset\s+HIST|>\s*~?\/?\.bash_history/.test(text)),
      bin(/\.ssh\/|authorized_keys|id_rsa/.test(text)),
      bin(/\b(?:shutdown|reboot|poweroff|halt|init\s+[06])\b/.test(text)),
      bin(/kill\s+-9\s+-1|killall|pkill/.test(text)),
      bin(/git\s+(reset\s+--hard|clean\s+-f|push\s+-[a-z]*f|--force)/.test(text)),
      bin(/docker\s+(system|image|volume|container)\s+prune/.test(text)),
      bin(/kubectl\s+delete/.test(text)),
      bin(/(?:apt|apt-get|yum|dnf)\s+(remove|purge|autoremove)|(?:pip|npm|gem)\s+uninstall/.test(text)),
      bin(/truncate\s+-s|>\s*[\w./-]*\.log/.test(text)),
      subshellDepth,
      backticks,
      Math.floor(quotes / 2),
      flags / words,
      Math.min(pathDepth, 8),
      bin(hasDangerWord && !hasDangerVerb),
    ];
  }

  function countOf(haystack, needle) {
    if (!needle) return 0;
    var count = 0;
    var index = haystack.indexOf(needle);
    while (index !== -1) {
      count++;
      index = haystack.indexOf(needle, index + needle.length);
    }
    return count;
  }

  /* Score one normalised command.
   *
   * Returns { score, risk, reasons, featureCount }, where risk is one of
   * safe/risky/destructive and reasons lists the strongest contributors, so the
   * prototype can explain itself instead of being a black box.
   *
   * reasons carries the model weight, not the raw value: a token is only worth
   * showing because the model found it informative. */
  function score(model, norm) {
    if (!model) return null;

    var joined = norm.views.join("\u0000");
    var grams = windowNgrams(joined, model.meta.ngramMin, model.meta.ngramMax);

    var total = model.bias;
    var reasons = [];

    grams.forEach(function (gram) {
      var index = model.vocabIndex[gram];
      if (index === undefined) return;
      var weight = model.coef[index];
      total += weight;
      if (Math.abs(weight) > 0.01) {
        reasons.push({ feature: gram, weight: weight, description: "" });
      }
    });

    var dense = denseFeatures(norm);
    for (var i = 0; i < dense.length; i++) {
      var z = (dense[i] - model.denseMean[i]) / model.denseStd[i];
      var contribution = model.denseCoef[i] * z;
      total += contribution;
      if (Math.abs(contribution) > 0.01) {
        reasons.push({
          feature: DENSE_NAMES[i] || "dense[" + i + "]",
          weight: contribution,
          description: "",
        });
      }
    }

    var risk2 = 2.0 / (1.0 + Math.exp(-total));
    var key =
      risk2 >= model.thresholds.block
        ? "destructive"
        : risk2 >= model.thresholds.warn
          ? "risky"
          : "safe";

    reasons.sort(function (a, b) {
      return Math.abs(b.weight) - Math.abs(a.weight);
    });

    return {
      score: risk2,
      risk: { key: key, label: LABELS[key] },
      reasons: reasons,
      featureCount: grams.size,
    };
  }

  var LABELS = { safe: "Guvenli", risky: "Supheli", destructive: "Tehlikeli" };


  var api = {
    // Must match drosh_ml.NORMALIZE_VERSION. The exporter stamps this value
    // into the model; the loader refuses a model whose normalisation differs,
    // because a model scored against other rules produces meaningless numbers.
    NORMALIZE_VERSION: "1",
    NGRAM_MIN: NGRAM_MIN,
    NGRAM_MAX: NGRAM_MAX,
    splitOperators: splitOperators,
    stripAnsi: stripAnsi,
    foldAscii: foldAscii,
    normalize: normalize,
    presenceNgrams: presenceNgrams,
    buildModel: buildModel,
    score: score,
    denseFeatures: denseFeatures,
    windowNgrams: windowNgrams,
    LABELS: LABELS,
  };

  global.DroshRisk = api;
})(typeof globalThis !== "undefined" ? globalThis : this);