#!/usr/bin/env python3
"""
Build a CSD(csd-en.txt 40-symbol) <-> espeak/IPA phoneme transducer, data-driven
from the CSD English corpus itself rather than hand-guessed.

Method: CSD's own txt/ files are already word-space-separated,
phoneme-underscore-separated (e.g. "r_ou r_ou r_ou y_oo_r b_ou_t" = 5 words).
The lyric/ files give the plain-English words in the same order (ignoring
punctuation/casing). For each word we run espeak-ng in per-phoneme IPA mode
and, where the CSD token count for that word matches espeak's phone count,
zip them position-wise into a co-occurrence table. Majority vote per CSD
symbol becomes the canonical mapping; full distributions are kept so
ambiguous/rare mappings are visible rather than hidden.

Usage: run inside the ctc-align venv (source env.sh) with espeak-ng on PATH.
"""
import csv as csvmod
import glob
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

CSD_ROOT = "/var/lib/symthaea/training-runs/diffsinger/CSD_extracted/CSD/english"
OUT_DIR = "/var/lib/symthaea/training-runs/ctc-align"

STRESS_MARKS = "ˈˌ"  # primary / secondary stress
TIE_BAR = "‍"  # zero-width joiner used by --ipa=3, not --sep mode
WORD_RE = re.compile(r"[A-Za-z']+")


def espeak_phones_flat(text: str) -> list:
    """Return the FLAT phoneme-token list for a whole line via espeak-ng
    --ipa --sep=_ (word breaks collapsed away -- CSD's own space-separated
    tokens are per-NOTE, not per-English-word, so word-level alignment is
    the wrong granularity; see build script docstring)."""
    out = subprocess.run(
        ["espeak-ng", "-v", "en-us", "-q", "--ipa", "--sep=_", text],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    toks = [t.strip(STRESS_MARKS + TIE_BAR) for t in out.replace(" ", "_").split("_")]
    return [t for t in toks if t]


def main():
    txt_files = sorted(glob.glob(os.path.join(CSD_ROOT, "txt", "*.txt")))
    cooc = defaultdict(Counter)
    matched_words = 0
    mismatched_words = 0
    mismatch_examples = []

    for txt_path in txt_files:
        name = os.path.splitext(os.path.basename(txt_path))[0]
        lyric_path = os.path.join(CSD_ROOT, "lyric", f"{name}.txt")
        if not os.path.exists(lyric_path):
            continue
        with open(txt_path) as fh:
            csd_lines = [l.strip() for l in fh if l.strip()]
        with open(lyric_path) as fh:
            lyric_lines = [l.strip() for l in fh if l.strip()]
        if len(csd_lines) != len(lyric_lines):
            print(f"WARN {name}: line count mismatch csd={len(csd_lines)} lyric={len(lyric_lines)}", file=sys.stderr)
            continue
        for csd_line, lyric_line in zip(csd_lines, lyric_lines):
            # CSD's space-separated tokens are per-NOTE, not per-English-word
            # (e.g. "merrily" -> "m_e_r l_ii", two CSD tokens for one word) --
            # so align at the flattened whole-line phone sequence, not per word.
            csd_phones_line = [p for tok in csd_line.split(" ") for p in tok.split("_")]
            esp_phones_line = espeak_phones_flat(lyric_line)
            if len(csd_phones_line) == len(esp_phones_line):
                matched_words += len(csd_phones_line)
                for cp, ep in zip(csd_phones_line, esp_phones_line):
                    cooc[cp][ep] += 1
            else:
                mismatched_words += len(csd_phones_line)
                if len(mismatch_examples) < 8:
                    mismatch_examples.append((name, lyric_line, csd_line,
                                               len(csd_phones_line), len(esp_phones_line)))

    print(f"matched words: {matched_words}, mismatched/skipped: {mismatched_words}", file=sys.stderr)
    for ex in mismatch_examples:
        print("  mismatch example:", ex, file=sys.stderr)

    # Load canonical CSD symbol list to report coverage.
    csd_symbols = []
    with open("/var/lib/symthaea/training-runs/diffsinger/DiffSinger/dictionaries/csd-en.txt") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                csd_symbols.append(parts[1])

    table = {}
    for sym in csd_symbols:
        counts = cooc.get(sym, Counter())
        total = sum(counts.values())
        if total == 0:
            table[sym] = {"best": None, "total": 0, "distribution": {}}
            continue
        best, best_n = counts.most_common(1)[0]
        table[sym] = {
            "best": best,
            "confidence": round(best_n / total, 3),
            "total": total,
            "distribution": dict(counts.most_common()),
        }

    with open(os.path.join(OUT_DIR, "csd_espeak_cooccurrence.json"), "w") as fh:
        json.dump(table, fh, indent=2, ensure_ascii=False)

    print("\n=== CSD -> espeak best-match table ===")
    for sym in csd_symbols:
        e = table[sym]
        if e["total"] == 0:
            print(f"  {sym:5s} -> NO DATA")
        else:
            print(f"  {sym:5s} -> {e['best']:5s}  (conf={e['confidence']:.2f}, n={e['total']}, dist={e['distribution']})")


if __name__ == "__main__":
    main()
