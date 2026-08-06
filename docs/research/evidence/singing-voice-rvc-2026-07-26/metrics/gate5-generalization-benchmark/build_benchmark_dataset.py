#!/usr/bin/env python3
"""Gate 5: generalization benchmark dataset.

Training: en001a (minus the Gate 3/4 held-out target phrase, for
continuity) + en002a, en003a, en004a in FULL (real, varied multi-song
content, ~330s combined).

Held-out (never in training):
  - the exact en001a "won't you sing along with me" phrase (continuity
    with Gates 2-4)
  - 8 phrases from en005a, an ENTIRELY unseen song, chosen for phonetic
    diversity, not at random: one deliberately simple/short phrase
    ("chirp chirp chirp"), one deliberately consonant-cluster-heavy
    phrase ("and a windy spring time day" -- spr/nd/nt clusters), plus 6
    more spanning different word/syllable counts.
"""
import csv
import os
import subprocess

VOWELS = {
    "a", "ae", "ai", "ao", "au", "e", "ei", "eo", "er",
    "i", "ii", "oi", "oo", "ou", "u", "uu",
}
CONSONANT_DUR = 0.07

DS = "/var/lib/symthaea/training-runs/diffsinger"
CSD_DIR = f"{DS}/CSD_extracted/CSD/english"
OUT_DIR = f"{DS}/raw/benchmark-01"

EN001A_HELDOUT_START = 60.0000  # same boundary as Gates 2-4


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


def load_rows(name):
    with open(f"{CSD_DIR}/csv/{name}.csv") as fh:
        return list(csv.DictReader(fh))


# (name, row_slice) -- indices into en005a's csv, chosen for phonetic
# diversity per Gate 5's design (not random).
EN005A_HELDOUT = {
    "simple_chirp": (43, 46),          # "chirp chirp chirp" -- SIMPLE control
    "cluster_windyspring": (27, 34),   # "and a windy spring time day" -- CLUSTER control (spr/nd/nt)
    "butterfly": (0, 3),               # "butterfly"
    "comeflyover": (6, 13),            # "come and fly and over here"
    "yellowwait": (13, 17),            # "yellow and wait"
    "petalssmile": (34, 37),           # "petals smile"
    "singsong": (46, 53),              # "sing a song and dance along"
    "comedanceover": (22, 27),         # "come and dance and over here"
}


def build():
    os.makedirs(f"{OUT_DIR}/wavs", exist_ok=True)
    rows_out = []

    # --- Training row 1: en001a minus the held-out target ---
    en001a_rows = load_rows("en001a")
    train_rows = [r for r in en001a_rows if float(r["start"]) < EN001A_HELDOUT_START]
    held_rows = [r for r in en001a_rows if float(r["start"]) >= EN001A_HELDOUT_START]
    train_end = float(train_rows[-1]["end"]) + 0.3
    ph_seq, ph_dur, prev_end = build_ph_seq(train_rows, 0.0)
    ph_seq.append("SP"); ph_dur.append(train_end - prev_end)
    slice_wav(f"{CSD_DIR}/wav/en001a.wav", f"{OUT_DIR}/wavs/train_en001a.wav", 0.0, train_end)
    rows_out.append({"name": "train_en001a", "ph_seq": " ".join(ph_seq), "ph_dur": " ".join(f"{d:.6f}" for d in ph_dur)})
    print(f"train_en001a: 0.0-{train_end:.2f}s, {len(ph_seq)} phonemes")

    held_out_end = 63.8813 + 0.40
    hph_seq, hph_dur, hprev = build_ph_seq(held_rows, EN001A_HELDOUT_START)
    hph_seq.append("SP"); hph_dur.append(held_out_end - hprev)
    slice_wav(f"{CSD_DIR}/wav/en001a.wav", f"{OUT_DIR}/wavs/heldout_wontyou.wav", EN001A_HELDOUT_START, held_out_end)
    rows_out.append({"name": "heldout_wontyou", "ph_seq": " ".join(hph_seq), "ph_dur": " ".join(f"{d:.6f}" for d in hph_dur)})
    print(f"heldout_wontyou: {EN001A_HELDOUT_START}-{held_out_end:.2f}s, {len(hph_seq)} phonemes")

    # --- Training rows 2-4: en002a, en003a, en004a, FULL ---
    train_phonemes_used = set(ph_seq) - {"AP", "SP"}
    for song in ("en002a", "en003a", "en004a"):
        song_rows = load_rows(song)
        song_end = float(song_rows[-1]["end"]) + 0.3
        sph_seq, sph_dur, sprev = build_ph_seq(song_rows, 0.0)
        sph_seq.append("SP"); sph_dur.append(song_end - sprev)
        slice_wav(f"{CSD_DIR}/wav/{song}.wav", f"{OUT_DIR}/wavs/train_{song}.wav", 0.0, song_end)
        rows_out.append({"name": f"train_{song}", "ph_seq": " ".join(sph_seq), "ph_dur": " ".join(f"{d:.6f}" for d in sph_dur)})
        train_phonemes_used |= (set(sph_seq) - {"AP", "SP"})
        print(f"train_{song}: 0.0-{song_end:.2f}s, {len(sph_seq)} phonemes")

    # --- Held-out phrases from en005a (entirely unseen song) ---
    en005a_rows = load_rows("en005a")
    for tag, (i0, i1) in EN005A_HELDOUT.items():
        phrase_rows = en005a_rows[i0:i1]
        t_start = float(phrase_rows[0]["start"])
        t_end = float(phrase_rows[-1]["end"]) + 0.30
        pph_seq, pph_dur, pprev = build_ph_seq(phrase_rows, t_start)
        pph_seq.append("SP"); pph_dur.append(t_end - pprev)
        name = f"heldout_{tag}"
        slice_wav(f"{CSD_DIR}/wav/en005a.wav", f"{OUT_DIR}/wavs/{name}.wav", t_start, t_end)
        rows_out.append({"name": name, "ph_seq": " ".join(pph_seq), "ph_dur": " ".join(f"{d:.6f}" for d in pph_dur)})
        missing = (set(pph_seq) - {"AP", "SP"}) - train_phonemes_used
        print(f"{name}: {t_start:.2f}-{t_end:.2f}s, {len(pph_seq)} phonemes"
              + (f"  ** phonemes NOT in training: {sorted(missing)} **" if missing else ""))

    csv_out = f"{OUT_DIR}/transcriptions.csv"
    with open(csv_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "ph_seq", "ph_dur"])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nwrote {csv_out}: {len(rows_out)} entries ({sum(1 for r in rows_out if r['name'].startswith('train'))} train, {sum(1 for r in rows_out if r['name'].startswith('heldout'))} held-out)")
    print(f"\nTotal training phonemes used ({len(train_phonemes_used)}): {sorted(train_phonemes_used)}")


if __name__ == "__main__":
    build()
