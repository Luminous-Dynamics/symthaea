#!/usr/bin/env python3
"""Gate 0: phoneme/duration table audit for en001a (the ABC song).

Recomputes EXACTLY what convert_csd.py produced (same split_syllable_duration
heuristic) against the real CSD ground-truth note timings, for two regions:
  1. Closing phrase: "Now I know my ABC, won't you sing along with me"
  2. Representative alphabet segment: "H I J K L M N O P" (includes the
     user-flagged letters M, N)

This is the exact planned duration data that was fed to DiffSinger at both
training and inference time (build_ds_file.py reused convert_csd.py's
ph_seq/ph_dur directly, not an independent re-alignment) -- so any
consonant-starvation bug found here was baked into every render, not
introduced at inference.
"""
import csv

SAMPLE_RATE = 44100
HOP_SIZE = 512
FRAME_MS = 1000.0 * HOP_SIZE / SAMPLE_RATE  # ~11.61 ms

VOWELS = {
    "a", "ae", "ai", "ao", "au", "e", "ei", "eo", "er",
    "i", "ii", "oi", "oo", "ou", "u", "uu",
}
CONSONANT_DUR = 0.07  # seconds -- convert_csd.py's fixed heuristic

STOPS = {"b", "d", "g", "k", "p", "t"}
FRICATIVES = {"f", "v", "s", "z", "sh", "zh", "th", "dh", "h"}
NASALS = {"m", "n", "ng"}
LIQUIDS = {"l", "r"}
GLIDES = {"w", "y"}
AFFRICATES = {"ch", "j"}

def phoneme_class(p):
    if p in VOWELS:
        return "vowel"
    if p in STOPS:
        return "stop"
    if p in FRICATIVES:
        return "fricative"
    if p in NASALS:
        return "nasal"
    if p in LIQUIDS:
        return "liquid"
    if p in GLIDES:
        return "glide"
    if p in AFFRICATES:
        return "affricate"
    return "other"


def split_syllable_duration(tokens, total_dur):
    n_cons = sum(1 for t in tokens if t not in VOWELS)
    cons_total = min(CONSONANT_DUR * n_cons, total_dur * 0.8)
    per_cons = cons_total / n_cons if n_cons else 0.0
    remaining = total_dur - cons_total
    n_vowels = sum(1 for t in tokens if t in VOWELS)
    per_vowel = remaining / n_vowels if n_vowels else remaining / len(tokens)
    durs = []
    for t in tokens:
        durs.append(per_vowel if (t in VOWELS or n_vowels == 0) else per_cons)
    drift = total_dur - sum(durs)
    durs[-1] += drift
    return durs


# Letter -> syllable(s) mapping for the ABC song, in the order they appear
# (letter names as pronounced, cross-referenced against the CSD syllable
# tokens and the real lyric text).
LETTER_NAMES = {
    "ei": "A", "b_ii": "B", "s_ii": "C", "d_ii": "D", "ii": "E",
    "e_f": "F", "j_ii": "G", "ei_ch": "H", "ai": "I", "j_ei": "J",
    "k_ei": "K", "e_l": "L", "e_m": "M", "e_n": "N", "ou": "O",
    "p_ii": "P", "k_y_uu": "Q", "a_r": "R", "e_s": "S", "t_ii": "T",
    "y_uu": "U", "v_ii": "V", "d_ao": "W(1/3)", "b_eo_l": "W(2/3)",
    "e_k_s": "X", "w_ai": "Y(1/2)", "eo_n_d": "Y(2/2)-and", "z_ii": "Z",
}

WORD_LABELS = {
    "n_au": "Now", "n_ou": "know", "m_ai": "my",
    "n_e_k_s_t": "Next", "t_ai_m": "time", "w_ou_n_t": "won't",
    "s_i_ng": "sing", "w_i_dh": "with", "m_ii": "me",
    "l_ou_ng": "along",
}


def load_csd_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def audit_region(rows, label):
    print(f"\n{'='*100}\nREGION: {label}\n{'='*100}")
    header = (f"{'word':<10}{'syll':<10}{'phon':<6}{'class':<11}"
              f"{'note':<6}{'note_dur_ms':<13}{'ph_start_ms':<13}"
              f"{'ph_end_ms':<12}{'ph_dur_ms':<11}{'frames':<8}{'flags'}")
    print(header)
    print("-" * len(header))

    prev_end = None
    flags_found = []
    for i, row in enumerate(rows):
        start = float(row["start"])
        end = float(row["end"])
        pitch = int(row["pitch"])
        syllable = row["syllable"]
        tokens = syllable.split("_")
        note_dur = end - start
        note_dur_ms = note_dur * 1000

        word = WORD_LABELS.get(syllable, LETTER_NAMES.get(syllable, "?"))

        pause_before = None
        if prev_end is not None and start > prev_end + 1e-4:
            pause_before = (start - prev_end) * 1000

        durs = split_syllable_duration(tokens, note_dur)
        t_cursor = start
        for tok, d in zip(tokens, durs):
            ph_start_ms = t_cursor * 1000
            ph_end_ms = (t_cursor + d) * 1000
            dur_ms = d * 1000
            frames = dur_ms / FRAME_MS
            cls = phoneme_class(tok)

            flags = []
            if cls != "vowel" and dur_ms < 30:
                flags.append("NEAR_ZERO_CONSONANT")
            elif cls != "vowel" and dur_ms < 50:
                flags.append("short_consonant")
            if cls == "vowel" and dur_ms > note_dur_ms * 0.9 and len(tokens) > 1:
                flags.append("vowel_dominates_note")
            if len(tokens) >= 3:
                flags.append(f"compressed_syllable(n={len(tokens)})")
            if frames < 1.0:
                flags.append("SUB_FRAME")

            flag_str = ",".join(flags)
            if flags:
                flags_found.append((word, syllable, tok, flag_str))

            print(f"{word:<10}{syllable:<10}{tok:<6}{cls:<11}{pitch:<6}"
                  f"{note_dur_ms:<13.1f}{ph_start_ms:<13.1f}{ph_end_ms:<12.1f}"
                  f"{dur_ms:<11.1f}{frames:<8.2f}{flag_str}")
            t_cursor += d

        pause_str = f"{pause_before:.1f}ms" if pause_before else "NONE (no SP gap)"
        print(f"    [word-boundary pause before this syllable: {pause_str}]")
        prev_end = end

    print(f"\n--- SUMMARY for {label} ---")
    if flags_found:
        for word, syll, tok, flag_str in flags_found:
            print(f"  FLAG: word={word!r} syllable={syll!r} phoneme={tok!r} -> {flag_str}")
    else:
        print("  No flags raised.")


if __name__ == "__main__":
    rows = load_csd_rows(
        "/var/lib/symthaea/training-runs/diffsinger/CSD_extracted/CSD/english/csv/en001a.csv"
    )
    # Closing phrase: rows 74-87 (1-indexed incl header) -> list index 72-85
    closing = rows[72:86]
    audit_region(closing, "Closing phrase: 'Now I know my ABC, won't you sing along with me'")

    # Alphabet segment H I J K L M N O P: rows 9-17 -> list index 7-15
    alphabet = rows[7:16]
    audit_region(alphabet, "Alphabet segment: H I J K L M N O P")
