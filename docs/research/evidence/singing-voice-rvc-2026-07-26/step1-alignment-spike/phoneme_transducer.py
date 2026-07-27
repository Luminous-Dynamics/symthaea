#!/usr/bin/env python3
"""
CSD (csd-en.txt, 40 symbols + AP/SP) <-> wav2vec2-lv-60-espeak-cv-ft IPA
phoneme transducer for the singing-voice alignment spike.

Version: 1 (2026-07-27). Built from data (see build_transducer_from_corpus.py,
csd_espeak_cooccurrence.json for full co-occurrence distributions across the
CSD English corpus, 16,363 position-matched phone pairs) with a small number
of documented manual overrides where the raw majority vote was noisy or
phonetically implausible for a low-sample symbol.

Each CSD symbol maps to a canonical target phone (used to build the expected
token sequence for forced alignment) plus a set of ACCEPTABLE variant phones
observed in the real corpus (used for scoring/QA -- singing pronunciation
genuinely varies, e.g. American-English intervocalic flapping realizes /t/ as
[ɾ], and unstressed vowels commonly reduce). Confidence is reported from the
corpus statistics, not invented.
"""

# canonical: the single target phone used to build expected sequences for
#            forced alignment.
# variants:  other phones actually observed aligned to this symbol in the
#            corpus (>=2 occurrences), kept for QA/scoring, not substitution.
# confidence: fraction of corpus occurrences matching `canonical` (or the
#            phonetically-justified target, for manually overridden entries).
# note: free-text rationale, especially for anything ambiguous or overridden.
CSD_TO_ESPEAK = {
    "a":   {"canonical": "ɑː", "variants": {"ɔ", "ʌ", "æ", "ə"}, "confidence": 0.70,
            "note": "open back vowel; real sung-pronunciation variation (e.g. 'but'->ʌ, 'all'->ɔ)"},
    "ae":  {"canonical": "æ",  "variants": {"ɐ"}, "confidence": 0.89, "note": None},
    "ai":  {"canonical": "aɪ", "variants": {"aɪə"}, "confidence": 0.99, "note": None},
    "ao":  {"canonical": "ʌ",  "variants": {"ɔ", "ə", "ɔː", "ɐ"}, "confidence": 0.90, "note": None},
    "au":  {"canonical": "aʊ", "variants": set(), "confidence": 0.99, "note": None},
    "b":   {"canonical": "b",  "variants": set(), "confidence": 0.99, "note": None},
    "ch":  {"canonical": "tʃ", "variants": set(), "confidence": 0.88, "note": None},
    "d":   {"canonical": "d",  "variants": set(), "confidence": 0.97, "note": None},
    "dh":  {"canonical": "ð",  "variants": set(), "confidence": 1.00, "note": None},
    "e":   {"canonical": "ɛ",  "variants": set(), "confidence": 0.98, "note": None},
    "ei":  {"canonical": "eɪ", "variants": set(), "confidence": 0.98, "note": None},
    "eo":  {"canonical": "ə",  "variants": {"ɐ", "æ", "ɪ", "ɚ", "ɜː"}, "confidence": 0.48,
            "note": "CSD's generic reduced/schwa-family vowel -- genuinely ambiguous, "
                    "treat as a class (a, ɐ, ɚ, ɜː all acceptable), not a single target"},
    "er":  {"canonical": "ɚ",  "variants": {"ɜː", "ɔːɹ"}, "confidence": 0.69,
            "note": "rhotic vowel; r-coloring quality varies"},
    "f":   {"canonical": "f",  "variants": set(), "confidence": 1.00, "note": None},
    "g":   {"canonical": "ɡ",  "variants": set(), "confidence": 1.00, "note": None},
    "h":   {"canonical": "h",  "variants": set(), "confidence": 0.91, "note": None},
    "i":   {"canonical": "ɪ",  "variants": {"ᵻ"}, "confidence": 0.96, "note": None},
    "ii":  {"canonical": "iː", "variants": {"i"}, "confidence": 0.45,
            "note": "raw espeak-ng CLI also emits tie-barred 'iːʲ'/'iʲ'/'ɪʲ' variants for "
                    "glide contexts, but NONE of those exist in the model's own vocab.json "
                    "(it was trained via phonemizer, not this CLI) -- they can never actually "
                    "appear in model output, so they're excluded here rather than listed as "
                    "acceptable; real confidence for the iː/i family once merged is ~0.90+"},
    "j":   {"canonical": "dʒ", "variants": set(), "confidence": 0.97, "note": None},
    "k":   {"canonical": "k",  "variants": set(), "confidence": 1.00, "note": None},
    "l":   {"canonical": "l",  "variants": set(), "confidence": 1.00, "note": None},
    "m":   {"canonical": "m",  "variants": set(), "confidence": 1.00, "note": None},
    "n":   {"canonical": "n",  "variants": {"d"}, "confidence": 0.96, "note": None},
    "ng":  {"canonical": "ŋ",  "variants": set(), "confidence": 0.99, "note": None},
    "oi":  {"canonical": "ɔɪ", "variants": {"ɪ"}, "confidence": 0.28,
            "note": "MANUAL OVERRIDE: raw majority vote picked 'ɪ' (n=24) over 'ɔɪ' (n=10) "
                    "at low total sample (36) -- phonologically 'oi' is unambiguously the "
                    "diphthong /ɔɪ/ (boy/toy); treating raw plurality as ground truth here "
                    "would encode noise, most likely from melisma/held-note decomposition"},
    "oo":  {"canonical": "ɔː", "variants": {"ɔ", "ɚ", "ɑː"}, "confidence": 0.54,
            "note": "back rounded vowel, real openness variation across tokens"},
    "ou":  {"canonical": "oʊ", "variants": set(), "confidence": 1.00, "note": None},
    "p":   {"canonical": "p",  "variants": {"i"}, "confidence": 0.90, "note": None},
    "r":   {"canonical": "ɹ",  "variants": {"ɛɹ", "ɔːɹ", "ɪɹ"}, "confidence": 0.94, "note": None},
    "s":   {"canonical": "s",  "variants": {"z"}, "confidence": 0.99, "note": None},
    "sh":  {"canonical": "ʃ",  "variants": set(), "confidence": 1.00, "note": None},
    "t":   {"canonical": "t",  "variants": {"ɾ"}, "confidence": 0.93,
            "note": "American-English intervocalic flapping is expected, not an error"},
    "th":  {"canonical": "θ",  "variants": set(), "confidence": 0.92, "note": None},
    "u":   {"canonical": "ʊ",  "variants": {"ʊɹ"}, "confidence": 0.89, "note": None},
    "uu":  {"canonical": "uː", "variants": {"ə", "ʌ", "ʊɹ", "ʊ"}, "confidence": 0.68,
            "note": "reduces toward schwa/ʌ in unstressed/rapid contexts"},
    "v":   {"canonical": "v",  "variants": set(), "confidence": 1.00, "note": None},
    "w":   {"canonical": "w",  "variants": set(), "confidence": 1.00, "note": None},
    "y":   {"canonical": "j",  "variants": {"uː"}, "confidence": 0.88, "note": None},
    "z":   {"canonical": "z",  "variants": set(), "confidence": 0.99, "note": None},
    "zh":  {"canonical": "ʒ",  "variants": set(), "confidence": 1.00,
            "note": "only 2 corpus occurrences -- phonologically unambiguous, low n"},
    # Silence/breath tokens introduced by convert_csd.py; no espeak counterpart.
    "AP":  {"canonical": None, "variants": set(), "confidence": None, "note": "breath -- non-speech"},
    "SP":  {"canonical": None, "variants": set(), "confidence": None, "note": "pause -- non-speech"},
}

# The 40 phonetic (non-silence) CSD symbols, in csd-en.txt order.
CSD_PHONETIC_SYMBOLS = [s for s in CSD_TO_ESPEAK if s not in ("AP", "SP")]

# Diacritic-normalized "families" -- used only for QA/reporting so that
# palatalization/length-mark tokenizer variants don't read as confusable
# with an unrelated phoneme.
IPA_FAMILY = {
    "iː": "i", "i": "i", "iːʲ": "i", "iʲ": "i", "ɪʲ": "ɪ",
}


def canonical_target(csd_symbol: str) -> str:
    """The single espeak/IPA phone to use when building an expected token
    sequence for forced alignment. Raises for unknown symbols; returns None
    for AP/SP (caller should treat as a silence/non-speech span)."""
    return CSD_TO_ESPEAK[csd_symbol]["canonical"]


def acceptable_targets(csd_symbol: str) -> set:
    """canonical + documented real-corpus variants -- used to score whether
    an aligned phone is a *plausible* realization, not just an exact hit."""
    entry = CSD_TO_ESPEAK[csd_symbol]
    if entry["canonical"] is None:
        return set()
    return {entry["canonical"]} | entry["variants"]


def build_expected_sequence(csd_phones: list) -> list:
    """csd_seq (list of CSD symbols, e.g. ['d','ao']) -> list of espeak IPA
    tokens for forced alignment. AP/SP are dropped (the aligner is only
    given phonetic content; silence is inferred from what's left over)."""
    seq = []
    for sym in csd_phones:
        tgt = canonical_target(sym)
        if tgt is not None:
            seq.append(tgt)
    return seq
