#!/usr/bin/env python3
"""F0 per-note analysis of v1's audio2audio cover-comparison renders,
same methodology as Gate B's original vocal-reference test."""
import glob
import os
import re

import librosa
import numpy as np

TARGETS = {
    "ascending_vocal": [261.63, 293.66, 329.63, 349.23, 392.00],
    "leap_vocal": [261.63, 392.00, 261.63, 392.00, 261.63],
}


def hz_to_semitone(f):
    return 12 * np.log2(f / 440.0)


def per_note_f0(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr)
    hop_length = 512
    frame_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    notes = []
    for i in range(5):
        t0, t1 = i * 1.2, (i + 1) * 1.2
        mask = (frame_times >= t0) & (frame_times < t1) & voiced_flag
        vals = f0[mask]
        vals = vals[~np.isnan(vals)]
        notes.append(float(np.median(vals)) if len(vals) else float("nan"))
    return notes


def main():
    paths = sorted(glob.glob("/var/lib/symthaea/training-runs/ace-step/cover_compare_v1_out/*.wav"))
    for path in paths:
        name = os.path.basename(path)[:-4]
        m = re.match(r"(ascending_vocal|leap_vocal)_", name)
        if not m:
            continue  # skip uncond
        melody = m.group(1)
        target = TARGETS[melody]
        notes = per_note_f0(path)
        valid = [(t, n) for t, n in zip(target, notes) if not np.isnan(n)]
        if len(valid) >= 2:
            t_arr = np.array([hz_to_semitone(t) for t, _ in valid])
            n_arr = np.array([hz_to_semitone(n) for _, n in valid])
            abs_err = np.abs(t_arr - n_arr)
            corr = float(np.corrcoef(t_arr, n_arr)[0, 1]) if np.std(n_arr) > 0 else float("nan")
            print(f"{name}: notes={[round(x,1) if not np.isnan(x) else None for x in notes]} "
                  f"mean_abs_err={np.mean(abs_err):.1f}st corr={corr:.2f}")
        else:
            print(f"{name}: notes={notes} (insufficient voiced)")


if __name__ == "__main__":
    main()
