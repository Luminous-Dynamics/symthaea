"""Transducer: misaki (Kokoro's G2P) phoneme characters -> the phone
inventory of facebook/wav2vec2-lv-60-espeak-cv-ft.

Verified character inventory: extracted directly from misaki's own output
across all 10 Gate-2 test phrases (see native-duration-check evidence),
not assumed from external IPA documentation. Every character below marked
"(verified)" actually occurred in that inventory; characters marked
"(untested)" are educated-guess additions for misaki characters seen
elsewhere in this project's codebase (VOWEL_CHARS/SONORANT_CONSONANT_CHARS
etc. in 03v8_exit_crossfade_ablation.py) but not exercised by this
specific 10-phrase set -- flagged explicitly, not silently assumed good.

Diphthong codes: misaki uses single capital letters for diphthongs
(A=face, I=price, O=goat, W=mouth, Y=choice) -- these map to the
espeak-cv-ft model's own 2-character diphthong tokens, which are single
vocab entries (not two characters to align separately).
"""

MISAKI_TO_ESPEAK = {
    # --- vowels (verified against this phrase set unless noted) ---
    "ə": "ə", "ɜ": "ɜ", "ʌ": "ʌ",
    "I": "aɪ",   # diphthong (verified: "by", "light")
    "A": "eɪ",   # diphthong (verified: "strangely")
    "O": "oʊ",   # diphthong (untested this phrase set, present in codebase's VOWEL_CHARS)
    "W": "aʊ",   # diphthong (untested)
    "Y": "ɔɪ",   # diphthong (untested)
    "ᵻ": "ᵻ",     # (untested this phrase set)
    "æ": "æ", "i": "i", "ɔ": "ɔ", "ɪ": "ɪ", "ɐ": "ɐ",
    "u": "u", "ʊ": "ʊ", "ɑ": "ɑ", "ɛ": "ɛ", "e": "e",
    "ɚ": "ɚ",     # (untested)
    "ɝ": "ɜː",   # (untested; ɝ absent from model vocab, nearest stressed r-colored/central vowel)
    "ᵊ": "ə",     # (untested; rare misaki elision marker, fallback to schwa)

    # --- sonorant consonants (verified unless noted) ---
    "m": "m", "n": "n", "ŋ": "ŋ", "l": "l", "ɹ": "ɹ",
    "r": "r",    # (untested this phrase set)
    "w": "w", "j": "j",

    # --- voiceless obstruents (verified unless noted) ---
    "p": "p", "t": "t", "k": "k", "f": "f", "s": "s", "ʃ": "ʃ",
    "θ": "θ",    # (untested)
    "h": "h",    # (untested)
    "ʧ": "tʃ",   # affricate (untested this phrase set; "ʤ" IS verified, "ʧ" is its voiceless counterpart)

    # --- voiced obstruents (verified unless noted) ---
    "b": "b", "d": "d", "v": "v", "ð": "ð", "z": "z",
    "ɡ": "ɡ",    # verified: misaki emits IPA U+0261, matching the model's own token exactly (no g->ɡ conversion needed)
    "ʒ": "ʒ",    # (untested)
    "ʤ": "dʒ",   # affricate (verified: "strangely")

    # --- markers: not phones, dropped from the CTC target sequence ---
    " ": None, "ˈ": None, "ˌ": None,
}


def transduce(ps: str):
    """Convert a misaki phoneme string to a list of (misaki_char,
    misaki_char_index, espeak_token) triples, skipping markers (spaces,
    stress). misaki_char_index preserves the position in the ORIGINAL ps
    string so results can be re-joined with pred_dur (which is indexed by
    that same position)."""
    out = []
    unknown = set()
    for i, c in enumerate(ps):
        if c not in MISAKI_TO_ESPEAK:
            unknown.add(c)
            continue
        tok = MISAKI_TO_ESPEAK[c]
        if tok is None:
            continue
        out.append((c, i, tok))
    return out, unknown
