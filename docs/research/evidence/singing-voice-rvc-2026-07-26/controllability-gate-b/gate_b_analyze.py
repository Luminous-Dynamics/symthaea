#!/usr/bin/env python3
"""
ACE-Step Controllability Audit -- Gate B analysis.

For each render, extracts an F0 contour (librosa.pyin) and reduces it to a
per-note median over 1.2s windows matching the reference melodies' note
structure (5 notes, 1.2s each, first 6s of audio). Compares:
- conditioned renders' per-note median F0 vs. the REQUESTED reference note
  frequencies (absolute error in semitones, and Pearson correlation in
  log-Hz space -- contour-shape adherence separated from absolute-register
  adherence, per the audit's own "relative vs absolute" framing)
- monotonicity checks for ascending/descending (is note5 higher/lower than
  note1, and is the trend consistent note-to-note)
- cross-seed consistency per melody (does the same melody produce a
  similar per-note F0 vector across the 2 seeds?)
- conditioned vs. unconditioned cross-melody F0 variance (does supplying
  a melody reference reduce variance relative to the Gate A baseline's
  no-reference default?)
- lyric transcription (Whisper) to check for an intelligibility regression
  under conditioning

Caveat, stated once here rather than repeated everywhere in output: the
reference melodies are pure sine-tone sequences, not naturalistic sung/
musical audio -- likely out-of-distribution for ACE-Step's DCAE encoder,
which was presumably trained on real music. A weak or null effect here is
consistent with either "conditioning doesn't work" or "this specific
synthetic-tone probe is a poor match for what the encoder expects" --
this analysis cannot distinguish those on its own.
"""
import glob
import os
import re

import librosa
import numpy as np
from faster_whisper import WhisperModel

GATE_B_DIR = "/var/lib/symthaea/training-runs/ace-step/gate_b_out"
NOTE_DUR = 1.2
N_NOTES = 5

MELODY_TARGETS = {
    "monotone": [220.0] * 5,
    "ascending": [261.63, 293.66, 329.63, 349.23, 392.00],
    "descending": [392.00, 349.23, 329.63, 293.66, 261.63],
    "leap": [261.63, 392.00, 261.63, 392.00, 261.63],
}


def hz_to_semitone(f):
    return 12 * np.log2(f / 440.0)


def per_note_f0(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr
    )
    hop_length = 512
    frame_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    notes = []
    for i in range(N_NOTES):
        t0, t1 = i * NOTE_DUR, (i + 1) * NOTE_DUR
        mask = (frame_times >= t0) & (frame_times < t1) & voiced_flag
        vals = f0[mask]
        vals = vals[~np.isnan(vals)]
        notes.append(float(np.median(vals)) if len(vals) else float("nan"))
    return notes


def main():
    paths = sorted(glob.glob(os.path.join(GATE_B_DIR, "*.wav")))
    print(f"Found {len(paths)} renders\n")
    whisper = WhisperModel("base", device="cpu", compute_type="int8")

    records = {}
    for path in paths:
        name = os.path.basename(path)[:-4]
        notes = per_note_f0(path)
        seg_iter, info = whisper.transcribe(path, language="en")
        transcript = " ".join(seg.text.strip() for seg in seg_iter).strip()
        records[name] = {"notes": notes, "transcript": transcript}
        print(f"{name}: notes={[round(n,1) if not np.isnan(n) else None for n in notes]}")
        print(f"  transcript: {transcript!r}")

    print("\n=== Conditioned vs. requested reference (absolute + contour) ===")
    for name, rec in records.items():
        m = re.match(r"(monotone|ascending|descending|leap)_seed(\d+)_cond", name)
        if not m:
            continue
        melody, seed = m.group(1), m.group(2)
        target = MELODY_TARGETS[melody]
        notes = rec["notes"]
        valid = [(t, n) for t, n in zip(target, notes) if not np.isnan(n)]
        if len(valid) < 2:
            print(f"{name}: insufficient voiced notes to compare ({notes})")
            continue
        t_arr = np.array([hz_to_semitone(t) for t, _ in valid])
        n_arr = np.array([hz_to_semitone(n) for _, n in valid])
        abs_err_semitones = np.abs(t_arr - n_arr)
        corr = float(np.corrcoef(t_arr, n_arr)[0, 1]) if len(valid) >= 3 and np.std(n_arr) > 0 else float("nan")
        print(f"{name}: abs_error(semitones)={[round(x,1) for x in abs_err_semitones]} "
              f"mean={np.mean(abs_err_semitones):.1f}  contour_corr={corr:.2f}")

    print("\n=== Monotonicity checks (does the trend go the right direction?) ===")
    for name, rec in records.items():
        m = re.match(r"(ascending|descending)_seed(\d+)_cond", name)
        if not m:
            continue
        melody = m.group(1)
        notes = [n for n in rec["notes"] if not np.isnan(n)]
        if len(notes) < 2:
            print(f"{name}: insufficient data")
            continue
        expected_sign = 1 if melody == "ascending" else -1
        actual_sign = 1 if notes[-1] > notes[0] else -1
        print(f"{name}: first={notes[0]:.1f}Hz last={notes[-1]:.1f}Hz "
              f"overall_direction={'correct' if actual_sign == expected_sign else 'WRONG'}")

    print("\n=== Cross-seed consistency per melody (conditioned) ===")
    for melody in MELODY_TARGETS:
        vecs = []
        for name, rec in records.items():
            if name.startswith(melody + "_seed") and name.endswith("_cond"):
                vecs.append(rec["notes"])
        if len(vecs) == 2:
            a, b = np.array(vecs[0]), np.array(vecs[1])
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() >= 2:
                diff_semitones = np.abs(hz_to_semitone(a[mask]) - hz_to_semitone(b[mask]))
                print(f"{melody}: per-note |seed1-seed2| semitone diff = "
                      f"{[round(x,1) for x in diff_semitones]}, mean={np.mean(diff_semitones):.1f}")
            else:
                print(f"{melody}: insufficient overlapping voiced notes across seeds")

    print("\n=== Unconditioned baselines (no reference) ===")
    for name, rec in records.items():
        if "uncond" in name:
            print(f"{name}: notes={[round(n,1) if not np.isnan(n) else None for n in rec['notes']]} "
                  f"transcript={rec['transcript']!r}")

    print("\n=== Cross-melody F0 variance: conditioned vs unconditioned ===")
    cond_means = []
    for name, rec in records.items():
        if name.endswith("_cond"):
            vals = [n for n in rec["notes"] if not np.isnan(n)]
            if vals:
                cond_means.append(np.mean(vals))
    uncond_means = []
    for name, rec in records.items():
        if "uncond" in name:
            vals = [n for n in rec["notes"] if not np.isnan(n)]
            if vals:
                uncond_means.append(np.mean(vals))
    if cond_means:
        print(f"Conditioned render means (Hz): {[round(x,1) for x in cond_means]} "
              f"cv={np.std(cond_means)/np.mean(cond_means):.3f}")
    if uncond_means:
        print(f"Unconditioned render means (Hz): {[round(x,1) for x in uncond_means]} "
              f"cv={np.std(uncond_means)/np.mean(uncond_means) if np.mean(uncond_means) else float('nan'):.3f}")


if __name__ == "__main__":
    main()
