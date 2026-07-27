#!/usr/bin/env python3
"""Tests for phoneme_transducer.py. Run: python3 test_phoneme_transducer.py"""
import json
import sys

from phoneme_transducer import (
    CSD_TO_ESPEAK, CSD_PHONETIC_SYMBOLS, canonical_target,
    acceptable_targets, build_expected_sequence,
)

VOCAB_PATH = ("/var/lib/symthaea/training-runs/ctc-align/hf-cache/hub/"
              "models--facebook--wav2vec2-lv-60-espeak-cv-ft/snapshots/"
              "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4/vocab.json")

failures = []


def check(name, cond):
    if not cond:
        failures.append(name)
        print(f"FAIL: {name}")
    else:
        print(f"ok:   {name}")


def main():
    # 1. Every csd-en.txt symbol (read straight from the dictionary file, not
    #    a hardcoded copy) has a transducer entry.
    dict_symbols = []
    with open("/var/lib/symthaea/training-runs/diffsinger/DiffSinger/dictionaries/csd-en.txt") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                dict_symbols.append(parts[1])
    check("all csd-en.txt symbols covered",
          all(s in CSD_TO_ESPEAK for s in dict_symbols))
    check("no extra/stale symbols in transducer beyond dict+AP+SP",
          set(CSD_TO_ESPEAK) == set(dict_symbols) | {"AP", "SP"})

    # 2. Every canonical target (and every declared variant) is a real token
    #    in the actual wav2vec2-espeak model vocabulary -- catches typos and
    #    IPA characters the model was never trained to emit.
    with open(VOCAB_PATH) as fh:
        vocab = set(json.load(fh).keys())
    bad = []
    for sym, entry in CSD_TO_ESPEAK.items():
        if entry["canonical"] is not None and entry["canonical"] not in vocab:
            bad.append((sym, "canonical", entry["canonical"]))
        for v in entry["variants"]:
            if v not in vocab:
                bad.append((sym, "variant", v))
    check(f"all canonical+variant phones exist in model vocab (bad={bad})", not bad)

    # 3. AP/SP have no espeak target (they're silence, handled separately).
    check("AP has no phonetic target", canonical_target("AP") is None)
    check("SP has no phonetic target", canonical_target("SP") is None)

    # 4. build_expected_sequence drops AP/SP and keeps phonetic content in order.
    seq = build_expected_sequence(["d", "SP", "ao", "AP", "n", "t"])
    check(f"build_expected_sequence drops silence tokens (got {seq})",
          seq == ["d", "ʌ", "n", "t"])

    # 5. acceptable_targets always includes the canonical target itself.
    check("acceptable_targets superset of canonical",
          all(canonical_target(s) in acceptable_targets(s)
              for s in CSD_PHONETIC_SYMBOLS))

    # 6. Spot-check a few unambiguous, high-confidence mappings against
    #    hand-known English phonology (not just internal consistency).
    spot = {"b": "b", "s": "s", "t": "t", "sh": "ʃ", "ch": "tʃ",
            "ng": "ŋ", "ou": "oʊ", "ai": "aɪ", "au": "aʊ", "j": "dʒ"}
    for sym, expected in spot.items():
        check(f"{sym} -> {expected}", canonical_target(sym) == expected)

    print(f"\n{len(failures)} failing checks" if failures else "\nall checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
