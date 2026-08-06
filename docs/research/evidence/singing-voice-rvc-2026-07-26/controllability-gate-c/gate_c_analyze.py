#!/usr/bin/env python3
"""
Gate C analysis: does a prompt-based tempo descriptor ("slow tempo, 60
bpm" vs "fast tempo, 150 bpm" vs no descriptor) have any measurable
effect on pacing? Measures per render: voiced duration (librosa.pyin
voiced-frame count), onset of first voiced frame, and phrase-repetition
count + total word count (via Whisper transcript) as a word-rate proxy.
"""
import glob
import os
import re

import librosa
import numpy as np
from faster_whisper import WhisperModel

GATE_C_DIR = "/var/lib/symthaea/training-runs/ace-step/gate_c_out"


def voice_activity(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr)
    hop_length = 512
    frame_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    voiced_idx = np.where(voiced_flag)[0]
    onset = float(frame_times[voiced_idx[0]]) if len(voiced_idx) else float("nan")
    voiced_dur = float(len(voiced_idx) * hop_length / sr)
    return onset, voiced_dur


def main():
    paths = sorted(glob.glob(os.path.join(GATE_C_DIR, "*.wav")))
    whisper = WhisperModel("base", device="cpu", compute_type="int8")

    results = {}
    for path in paths:
        name = os.path.basename(path)[:-4]
        onset, voiced_dur = voice_activity(path)
        seg_iter, info = whisper.transcribe(path, language="en")
        transcript = " ".join(seg.text.strip() for seg in seg_iter).strip()
        n_words = len(re.findall(r"[A-Za-z']+", transcript))
        n_reps = len(re.findall(r"sing along", transcript, re.IGNORECASE))
        results[name] = {"onset": onset, "voiced_dur": voiced_dur, "n_words": n_words,
                          "n_reps": n_reps, "transcript": transcript}
        print(f"{name}: onset={onset:.2f}s voiced_dur={voiced_dur:.2f}s "
              f"n_words={n_words} n_reps={n_reps} transcript={transcript!r}")

    print("\n=== Per-condition summary (mean across 3 seeds) ===")
    for cond in ["baseline", "slow", "fast"]:
        onsets = [r["onset"] for n, r in results.items() if n.startswith(cond + "_") and not np.isnan(r["onset"])]
        durs = [r["voiced_dur"] for n, r in results.items() if n.startswith(cond + "_")]
        words = [r["n_words"] for n, r in results.items() if n.startswith(cond + "_")]
        reps = [r["n_reps"] for n, r in results.items() if n.startswith(cond + "_")]
        print(f"{cond}: mean_onset={np.mean(onsets):.2f}s mean_voiced_dur={np.mean(durs):.2f}s "
              f"mean_words={np.mean(words):.1f} mean_reps={np.mean(reps):.1f} "
              f"mean_words_per_voiced_sec={np.mean(words)/np.mean(durs) if np.mean(durs) else float('nan'):.2f}")


if __name__ == "__main__":
    main()
