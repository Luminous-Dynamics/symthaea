#!/usr/bin/env python3
"""Gate 4: generalization test dataset. Trains on nearly the whole
en001a song (2.4s-59.2s, everything except the held-out target phrase)
and holds out "won't you sing along with me" (60.0-63.8813s, the exact
same phrase used in Gates 2-3) as a genuinely unseen eval target -- no
overlap with training. Tests whether the model can produce an
intelligible phrase it never trained on, given real content diversity
(the whole song's alphabet + "next time" verses) rather than the single
memorized phrase from Gate 3.
"""
import csv
import os
import subprocess

VOWELS = {
    "a", "ae", "ai", "ao", "au", "e", "ei", "eo", "er",
    "i", "ii", "oi", "oo", "ou", "u", "uu",
}
CONSONANT_DUR = 0.07  # same heuristic as convert_csd.py / Gate 3

DS = "/var/lib/symthaea/training-runs/diffsinger"
CSD_CSV = f"{DS}/CSD_extracted/CSD/english/csv/en001a.csv"
CSD_WAV = f"{DS}/CSD_extracted/CSD/english/wav/en001a.wav"
OUT_DIR = f"{DS}/raw/generalize-01"

HELD_OUT_START = 60.0000  # exact Gate 2/3 boundary -- nothing at/after
                           # this timestamp is used in training.


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


def build_ph_seq(rows, clip_start):
    ph_seq, ph_dur = [], []
    prev_end = clip_start
    first_gap = True
    for r in rows:
        start, end = float(r["start"]), float(r["end"])
        if start > prev_end + 1e-4:
            ph_seq.append("AP" if first_gap else "SP")
            ph_dur.append(start - prev_end)
            first_gap = False
        tokens = r["syllable"].split("_")
        durs = split_syllable_duration(tokens, end - start)
        ph_seq.extend(tokens)
        ph_dur.extend(durs)
        prev_end = end
    return ph_seq, ph_dur, prev_end


def slice_wav(src, dst, t_start, t_end):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-ss", str(t_start), "-to", str(t_end),
         "-ac", "1", "-ar", "44100", dst],
        check=True,
    )


def build():
    os.makedirs(f"{OUT_DIR}/wavs", exist_ok=True)
    with open(CSD_CSV) as fh:
        rows = list(csv.DictReader(fh))

    train_rows = [r for r in rows if float(r["start"]) < HELD_OUT_START]
    held_out_rows = [r for r in rows if float(r["start"]) >= HELD_OUT_START]
    print(f"train: {len(train_rows)} syllables, held-out: {len(held_out_rows)} syllables")

    # Training clip: from t=0 to the end of the last pre-held-out syllable,
    # plus a short trailing SP so it ends cleanly before the held-out region.
    train_end = float(train_rows[-1]["end"]) + 0.3
    train_ph_seq, train_ph_dur, prev_end = build_ph_seq(train_rows, 0.0)
    train_ph_seq.append("SP")
    train_ph_dur.append(train_end - prev_end)
    slice_wav(CSD_WAV, f"{OUT_DIR}/wavs/train01.wav", 0.0, train_end)
    print(f"train clip: 0.0-{train_end:.3f}s, {len(train_ph_seq)} phonemes")

    # Held-out target: identical to Gate 2/3's phrase.
    held_out_end = 63.8813 + 0.40
    held_ph_seq, held_ph_dur, prev_end2 = build_ph_seq(held_out_rows, HELD_OUT_START)
    held_ph_seq.append("SP")
    held_ph_dur.append(held_out_end - prev_end2)
    slice_wav(CSD_WAV, f"{OUT_DIR}/wavs/heldout01.wav", HELD_OUT_START, held_out_end)
    print(f"held-out clip: {HELD_OUT_START}-{held_out_end:.3f}s, {len(held_ph_seq)} phonemes")

    rows_out = [
        {"name": "train01", "ph_seq": " ".join(train_ph_seq), "ph_dur": " ".join(f"{d:.6f}" for d in train_ph_dur)},
        {"name": "heldout01", "ph_seq": " ".join(held_ph_seq), "ph_dur": " ".join(f"{d:.6f}" for d in held_ph_dur)},
    ]
    csv_out = f"{OUT_DIR}/transcriptions.csv"
    with open(csv_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "ph_seq", "ph_dur"])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"wrote {csv_out}")

    used_phonemes = sorted(set(train_ph_seq) - {"AP", "SP"})
    print(f"\nphonemes used in TRAINING ({len(used_phonemes)}): {used_phonemes}")
    held_only = sorted((set(held_ph_seq) - {"AP", "SP"}) - set(train_ph_seq))
    print(f"phonemes in HELD-OUT but NOT in training: {held_only}")


if __name__ == "__main__":
    build()
