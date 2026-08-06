#!/usr/bin/env python3
"""Memorization/overfit gate dataset: ONE real phrase ("won't you sing along
with me", en001a rows 80-86, 60.0000-63.8813s), used as literally the only
training example, per Gate 3 methodology proposed 2026-07-26 (external
review of the Gate 0-2 results): before assuming "under-trained" explains
Gate 2's finding, test whether the architecture/pipeline can even memorize
one clean example to intelligibility at all.

Writes two copies (train + val, identical content) so the binarizer's
test_prefixes split has a non-empty validation set without needing a
second real recording -- this is a deliberate overfit test, train==val by
design.
"""
import csv
import os
import subprocess

DS = "/var/lib/symthaea/training-runs/diffsinger"
CSD_CSV = f"{DS}/CSD_extracted/CSD/english/csv/en001a.csv"
CSD_WAV = f"{DS}/CSD_extracted/CSD/english/wav/en001a.wav"
OUT_DIR = f"{DS}/raw/overfit-01"

T_START, T_END = 60.0000, 63.8813
TAIL_SP_DUR = 0.40  # real trailing audio after "me" ends -- needed because
                     # the binarizer's dictionary-coverage check requires
                     # at least one SP (silence) phoneme in the training
                     # data, and this phrase has none mid-clip (only a
                     # leading AP breath).

VOWELS = {
    "a", "ae", "ai", "ao", "au", "e", "ei", "eo", "er",
    "i", "ii", "oi", "oo", "ou", "u", "uu",
}
CONSONANT_DUR = 0.07  # identical heuristic to convert_csd.py -- isolates
                       # capacity/data-scale, not duration-heuristic quality


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


def build():
    os.makedirs(f"{OUT_DIR}/wavs", exist_ok=True)

    with open(CSD_CSV) as fh:
        rows = list(csv.DictReader(fh))
    region = [r for r in rows if T_START - 1e-4 <= float(r["start"]) < T_END]
    assert region, "no rows found in the target region"

    ph_seq, ph_dur = [], []
    prev_end = T_START
    for r in region:
        start, end = float(r["start"]), float(r["end"])
        if start > prev_end + 1e-4:
            ph_seq.append("AP")
            ph_dur.append(start - prev_end)
        tokens = r["syllable"].split("_")
        durs = split_syllable_duration(tokens, end - start)
        ph_seq.extend(tokens)
        ph_dur.extend(durs)
        prev_end = end

    # Real trailing audio after "me" ends -- gives the binarizer's
    # dictionary-coverage check at least one SP occurrence.
    ph_seq.append("SP")
    ph_dur.append(TAIL_SP_DUR)
    clip_end = T_END + TAIL_SP_DUR

    print(f"region: {len(region)} syllables, {len(ph_seq)} phonemes, "
          f"{sum(ph_dur):.3f}s (clip length {clip_end - T_START:.3f}s)")

    # Slice audio: ffmpeg, mono 44.1kHz, matching convert_csd.py's convention.
    for name in ("overfit01", "overfit01_val"):
        dst = f"{OUT_DIR}/wavs/{name}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", CSD_WAV, "-ss", str(T_START), "-to", str(clip_end),
             "-ac", "1", "-ar", "44100", dst],
            check=True,
        )
        print(f"wrote {dst}")

    rows_out = [
        {"name": name, "ph_seq": " ".join(ph_seq), "ph_dur": " ".join(f"{d:.6f}" for d in ph_dur)}
        for name in ("overfit01", "overfit01_val")
    ]
    csv_out = f"{OUT_DIR}/transcriptions.csv"
    with open(csv_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "ph_seq", "ph_dur"])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"wrote {csv_out}")


if __name__ == "__main__":
    build()
