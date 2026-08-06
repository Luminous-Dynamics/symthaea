#!/usr/bin/env python3
import glob
import os
import re

import librosa
import numpy as np
from faster_whisper import WhisperModel

GATE_B_DIR = "/var/lib/symthaea/training-runs/ace-step/gate_b_out"
TARGETS = {
    "ascending_vocal": [261.63, 293.66, 329.63, 349.23, 392.00],
    "leap_vocal": [261.63, 392.00, 261.63, 392.00, 261.63],
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
    for i in range(5):
        t0, t1 = i * 1.2, (i + 1) * 1.2
        mask = (frame_times >= t0) & (frame_times < t1) & voiced_flag
        vals = f0[mask]
        vals = vals[~np.isnan(vals)]
        notes.append(float(np.median(vals)) if len(vals) else float("nan"))
    return notes


def main():
    paths = sorted(glob.glob(os.path.join(GATE_B_DIR, "*_vocal_seed*.wav")))
    print(f"Found {len(paths)} renders\n")
    whisper = WhisperModel("base", device="cpu", compute_type="int8")

    for path in paths:
        name = os.path.basename(path)[:-4]
        m = re.match(r"(ascending_vocal|leap_vocal)_seed(\d+)_strength([\d.]+)", name)
        melody, seed, strength = m.group(1), m.group(2), m.group(3)
        notes = per_note_f0(path)
        seg_iter, info = whisper.transcribe(path, language="en")
        transcript = " ".join(seg.text.strip() for seg in seg_iter).strip()
        target = TARGETS[melody]
        valid = [(t, n) for t, n in zip(target, notes) if not np.isnan(n)]
        if len(valid) >= 2:
            t_arr = np.array([hz_to_semitone(t) for t, _ in valid])
            n_arr = np.array([hz_to_semitone(n) for _, n in valid])
            abs_err = np.abs(t_arr - n_arr)
            corr = float(np.corrcoef(t_arr, n_arr)[0, 1]) if np.std(n_arr) > 0 else float("nan")
            err_str = f"mean_abs_err={np.mean(abs_err):.1f}st corr={corr:.2f}"
        else:
            err_str = "insufficient voiced notes"
        print(f"{name}: transcript={transcript!r}")
        print(f"  notes={[round(n,1) if not np.isnan(n) else None for n in notes]} "
              f"target={target}  {err_str}\n")


if __name__ == "__main__":
    main()
