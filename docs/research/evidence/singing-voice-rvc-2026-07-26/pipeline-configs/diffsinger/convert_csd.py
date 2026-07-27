#!/usr/bin/env python3
"""
Convert CSD (Children's Song Dataset, KAIST, CC BY-NC-SA 4.0 -- research-only,
see SING-17 notes) English subset into an OpenVPI DiffSinger "raw dataset":
    <out>/wavs/<name>.wav   (mono, 44.1kHz)
    <out>/transcriptions.csv  (name, ph_seq, ph_dur)

Scope: acoustic-model training only (per docs/BestPractices.md, that needs
just name/ph_seq/ph_dur -- F0 is extracted from the ground-truth recordings
by the binarizer itself, not supplied here). Duration/pitch/variance
predictors are out of scope, matching the same "Symthaea supplies explicit
control" choice already made on the inference side (SING-15/16).

CSD's syllable column packs a syllable's phonemes into one underscore-
joined token (e.g. "b_ii"), each already time-aligned to [start,end] as
a whole. We split the syllable-level duration across its sub-phonemes:
fixed short duration per consonant, remainder to vowel token(s).
"""
import csv
import glob
import os
import subprocess
import sys

VOWELS = {
    "a", "ae", "ai", "ao", "au", "e", "ei", "eo", "er",
    "i", "ii", "oi", "oo", "ou", "u", "uu",
}
CONSONANT_DUR = 0.07  # seconds, nominal


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
    # Fix any rounding drift onto the last token.
    drift = total_dur - sum(durs)
    durs[-1] += drift
    return durs


def convert(csd_english_dir: str, out_dir: str):
    wav_out = os.path.join(out_dir, "wavs")
    os.makedirs(wav_out, exist_ok=True)
    rows = []
    csv_files = sorted(glob.glob(os.path.join(csd_english_dir, "csv", "*.csv")))
    for csv_path in csv_files:
        name = os.path.splitext(os.path.basename(csv_path))[0]
        src_wav = os.path.join(csd_english_dir, "wav", f"{name}.wav")
        if not os.path.exists(src_wav):
            print(f"WARN: missing wav for {name}, skipping", file=sys.stderr)
            continue
        dst_wav = os.path.join(wav_out, f"{name}.wav")
        # Convert to mono 44.1kHz via ffmpeg (source is stereo per SING-17
        # inspection). -y overwrite, quiet.
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src_wav,
             "-ac", "1", "-ar", "44100", dst_wav],
            check=True,
        )

        ph_seq = []
        ph_dur = []
        prev_end = 0.0
        first_gap = True
        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            for entry in reader:
                start = float(entry["start"])
                end = float(entry["end"])
                syllable = entry["syllable"]
                if start > prev_end + 1e-4:
                    # Gap between notes: an explicit pause phoneme. The
                    # very first gap (lead-in before singing starts) is a
                    # breath -- AP, DiffSinger's convention -- everything
                    # after is plain silence -- SP.
                    ph_seq.append("AP" if first_gap else "SP")
                    ph_dur.append(start - prev_end)
                    first_gap = False
                tokens = syllable.split("_")
                durs = split_syllable_duration(tokens, end - start)
                ph_seq.extend(tokens)
                ph_dur.extend(durs)
                prev_end = end

        rows.append({
            "name": name,
            "ph_seq": " ".join(ph_seq),
            "ph_dur": " ".join(f"{d:.6f}" for d in ph_dur),
        })
        print(f"converted {name}: {len(ph_seq)} phonemes, "
              f"{sum(ph_dur):.1f}s")

    csv_out = os.path.join(out_dir, "transcriptions.csv")
    with open(csv_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "ph_seq", "ph_dur"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} entries to {csv_out}")


if __name__ == "__main__":
    csd_dir = sys.argv[1] if len(sys.argv) > 1 else "CSD_extracted/CSD/english"
    out = sys.argv[2] if len(sys.argv) > 2 else "raw/csd-en"
    convert(csd_dir, out)
