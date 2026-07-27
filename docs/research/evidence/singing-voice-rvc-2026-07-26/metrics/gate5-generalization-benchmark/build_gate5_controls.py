#!/usr/bin/env python3
"""Gate 5 controls (built for inference only, no retraining needed --
evaluated against the trained checkpoint):

Control 1 (seen-lyrics, unseen-melody): "angels we have heard on high"
(en002a rows 0-6, ph_seq/ph_dur real and literally present in training)
but with F0 transposed up a perfect fifth (x2^(7/12)) -- same linguistic
content, a melody the model never saw. Tests whether the model separates
linguistic knowledge from memorized musical realization.

Control 2 (unseen-lyrics, simple-melody): heldout_cluster_windyspring's
real phonemes ("and a windy spring time day", the consonant-cluster
held-out phrase, NEVER in training) but with GENEROUS durations (150ms
cons/350ms vowel, same policy as Gate 1's ladder) and a SMOOTHED pitch
contour (per-syllable constant, using each syllable's mean real pitch,
removing within-syllable jumps) instead of the original fine-grained
real F0 curve. Tests linguistic generalization without difficult
musical timing.
"""
import csv
import json
import sys

import numpy as np
import parselmouth

DS = "/var/lib/symthaea/training-runs/diffsinger"
HOP_SIZE = 512
SAMPLE_RATE = 44100
F0_TIMESTEP = HOP_SIZE / SAMPLE_RATE
F0_MIN, F0_MAX = 65, 1100
VOWELS = {
    "a", "ae", "ai", "ao", "au", "e", "ei", "eo", "er",
    "i", "ii", "oi", "oo", "ou", "u", "uu",
}
CONS_DUR, VOWEL_DUR = 0.15, 0.35


def real_f0_curve(wav_path, t_start, t_end):
    sound = parselmouth.Sound(wav_path).extract_part(from_time=t_start, to_time=t_end, preserve_times=False)
    pitch = sound.to_pitch_ac(time_step=F0_TIMESTEP, pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
    f0 = pitch.selected_array["frequency"]
    voiced = f0 > 0
    if voiced.any():
        idx = np.where(voiced, np.arange(len(f0)), 0)
        np.maximum.accumulate(idx, out=idx)
        filled = f0[idx]
        first_voiced = np.argmax(voiced)
        filled[:first_voiced] = f0[voiced][0]
    else:
        filled = np.full_like(f0, 220.0)
    return filled


def write_ds(path, name, ph_seq, ph_dur, f0_seq):
    entry = {
        "offset": 0.0, "text": name,
        "ph_seq": " ".join(ph_seq),
        "ph_dur": " ".join(f"{d:.6f}" for d in ph_dur),
        "f0_seq": " ".join(f"{f:.1f}" for f in f0_seq),
        "f0_timestep": str(F0_TIMESTEP),
    }
    with open(path, "w") as fh:
        json.dump([entry], fh)
    print(f"wrote {path}: {len(ph_seq)} phonemes, {len(f0_seq)} f0 frames")


def control1_seen_lyrics_unseen_melody():
    with open(f"{DS}/CSD_extracted/CSD/english/csv/en002a.csv") as fh:
        rows = list(csv.DictReader(fh))[:7]  # "angels we have heard on high"
    t_start, t_end = float(rows[0]["start"]), float(rows[-1]["end"]) + 0.3

    ph_seq, ph_dur = [], []
    prev_end = t_start
    for r in rows:
        start, end = float(r["start"]), float(r["end"])
        tokens = r["syllable"].split("_")
        n_cons = sum(1 for t in tokens if t not in VOWELS)
        cons_total = min(0.07 * n_cons, (end - start) * 0.8)
        per_cons = cons_total / n_cons if n_cons else 0.0
        n_vowels = sum(1 for t in tokens if t in VOWELS)
        per_vowel = ((end - start) - cons_total) / n_vowels if n_vowels else (end - start) / len(tokens)
        for t in tokens:
            ph_seq.append(t)
            ph_dur.append(per_vowel if (t in VOWELS or n_vowels == 0) else per_cons)
        prev_end = end
    ph_seq.append("SP"); ph_dur.append(t_end - prev_end)

    real_f0 = real_f0_curve(f"{DS}/CSD_extracted/CSD/english/wav/en002a.wav", t_start, t_end)
    n_frames = int(round(sum(ph_dur) / F0_TIMESTEP))
    if len(real_f0) < n_frames:
        real_f0 = np.pad(real_f0, (0, n_frames - len(real_f0)), mode="edge")
    real_f0 = real_f0[:n_frames]
    transposed_f0 = real_f0 * (2 ** (7 / 12))  # +7 semitones, perfect fifth up

    write_ds(f"{DS}/gate5_control1_seenlyrics_unseenmelody.ds", "control1", ph_seq, ph_dur, transposed_f0)


def control2_unseen_lyrics_simple_melody():
    heldout = json.load(open(f"{DS}/benchmark_ds/heldout_cluster_windyspring.ds"))[0]
    ph_seq = heldout["ph_seq"].split()
    orig_dur = [float(x) for x in heldout["ph_dur"].split()]
    orig_f0 = [float(x) for x in heldout["f0_seq"].split()]
    orig_timestep = float(heldout["f0_timestep"])

    # Generous durations, same policy as Gate 1's ladder.
    new_dur = [VOWEL_DUR if t in VOWELS else (0.25 if t in ("AP", "SP") else CONS_DUR) for t in ph_seq]

    # Smoothed pitch: one constant value per phoneme, using the mean of
    # the ORIGINAL curve over that phoneme's original time window
    # (removes within-phoneme jitter/transitions while keeping the real
    # note-to-note melodic shape, unlike Gate 1 v1's single flat pitch).
    orig_bounds = np.cumsum([0.0] + orig_dur)
    smoothed_per_phoneme = []
    for i in range(len(ph_seq)):
        t0, t1 = orig_bounds[i], orig_bounds[i + 1]
        f0_i0 = int(t0 / orig_timestep)
        f0_i1 = max(f0_i0 + 1, int(t1 / orig_timestep))
        window = orig_f0[f0_i0:f0_i1] if f0_i0 < len(orig_f0) else [orig_f0[-1]]
        smoothed_per_phoneme.append(float(np.mean(window)) if window else orig_f0[-1])

    n_frames = int(round(sum(new_dur) / F0_TIMESTEP))
    new_bounds = np.cumsum([0.0] + new_dur)
    new_f0 = []
    ph_idx = 0
    for frame in range(n_frames):
        t = frame * F0_TIMESTEP
        while ph_idx < len(new_bounds) - 2 and t >= new_bounds[ph_idx + 1]:
            ph_idx += 1
        new_f0.append(smoothed_per_phoneme[ph_idx])

    write_ds(f"{DS}/gate5_control2_unseenlyrics_simplemelody.ds", "control2", ph_seq, new_dur, new_f0)


if __name__ == "__main__":
    control1_seen_lyrics_unseen_melody()
    control2_unseen_lyrics_simple_melody()
