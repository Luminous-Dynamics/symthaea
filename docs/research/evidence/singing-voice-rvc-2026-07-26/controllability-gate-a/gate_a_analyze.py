#!/usr/bin/env python3
"""
ACE-Step Controllability Audit -- Gate A analysis.

For each seed's render, measures:
- transcription (faster-whisper) vs. target lyrics
- F0 contour (librosa.pyin) -- mean/std/range, for melody-variation proxy
- a timbre-consistency PROXY (MFCC mean vector cosine similarity across
  seeds) -- explicitly not a validated speaker-verification metric, just
  a cheap, transparent stand-in given no speaker-embedding model was
  installed for this bounded audit
- crude phrase-timing proxy: onset envelope's first strong onset (voice
  activity start) and total voiced duration via the F0 track's non-NaN span
"""
import glob
import os

import librosa
import numpy as np
from faster_whisper import WhisperModel

GATE_A_DIR = "/var/lib/symthaea/training-runs/ace-step/gate_a_out"
TARGET = "won't you sing along with me"


def load_mono(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr


def analyze_one(path):
    y, sr = load_mono(path)
    duration = len(y) / sr

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr
    )
    voiced_f0 = f0[voiced_flag]
    f0_mean = float(np.nanmean(voiced_f0)) if len(voiced_f0) else float("nan")
    f0_std = float(np.nanstd(voiced_f0)) if len(voiced_f0) else float("nan")
    f0_range = float(np.nanmax(voiced_f0) - np.nanmin(voiced_f0)) if len(voiced_f0) else float("nan")

    # first frame with voiced pitch -> crude "voice activity onset" proxy
    voiced_idx = np.where(voiced_flag)[0]
    hop_length = 512  # librosa.pyin default
    onset_time = float(voiced_idx[0] * hop_length / sr) if len(voiced_idx) else float("nan")
    voiced_duration = float(len(voiced_idx) * hop_length / sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean_vec = mfcc.mean(axis=1)

    return {
        "path": path,
        "duration": duration,
        "f0_mean_hz": f0_mean,
        "f0_std_hz": f0_std,
        "f0_range_hz": f0_range,
        "onset_time_s": onset_time,
        "voiced_duration_s": voiced_duration,
        "mfcc_mean_vec": mfcc_mean_vec,
    }


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    paths = sorted(glob.glob(os.path.join(GATE_A_DIR, "seed_*.wav")))
    print(f"Found {len(paths)} renders: {[os.path.basename(p) for p in paths]}\n")

    whisper = WhisperModel("base", device="cpu", compute_type="int8")

    results = []
    for path in paths:
        seg_iter, info = whisper.transcribe(path, language="en")
        transcript = " ".join(seg.text.strip() for seg in seg_iter)
        analysis = analyze_one(path)
        analysis["transcript"] = transcript.strip()
        results.append(analysis)
        print(f"{os.path.basename(path)}:")
        print(f"  transcript: {analysis['transcript']!r}")
        print(f"  duration={analysis['duration']:.2f}s  "
              f"f0_mean={analysis['f0_mean_hz']:.1f}Hz  f0_std={analysis['f0_std_hz']:.1f}Hz  "
              f"f0_range={analysis['f0_range_hz']:.1f}Hz")
        print(f"  onset={analysis['onset_time_s']:.2f}s  voiced_dur={analysis['voiced_duration_s']:.2f}s\n")

    # Cross-seed comparisons
    print("=== Cross-seed summary ===")
    exact_matches = sum(
        1 for r in results if TARGET in r["transcript"].lower().rstrip(".!?"))
    print(f"Transcripts containing target phrase verbatim: {exact_matches}/{len(results)}")

    f0_means = [r["f0_mean_hz"] for r in results if not np.isnan(r["f0_mean_hz"])]
    print(f"F0 mean across seeds: {[round(x, 1) for x in f0_means]} "
          f"(range {max(f0_means)-min(f0_means):.1f}Hz, "
          f"cv={np.std(f0_means)/np.mean(f0_means):.3f})" if f0_means else "no F0 detected in any seed")

    onsets = [r["onset_time_s"] for r in results if not np.isnan(r["onset_time_s"])]
    print(f"Onset time across seeds: {[round(x, 2) for x in onsets]} "
          f"(range {max(onsets)-min(onsets):.2f}s)" if onsets else "no onsets detected")

    # Pairwise MFCC cosine similarity (timbre-consistency proxy)
    print("\nPairwise MFCC cosine similarity (timbre-consistency proxy, NOT a validated speaker metric):")
    n = len(results)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine(results[i]["mfcc_mean_vec"], results[j]["mfcc_mean_vec"])
            sims.append(sim)
            print(f"  {os.path.basename(results[i]['path'])} vs {os.path.basename(results[j]['path'])}: {sim:.4f}")
    if sims:
        print(f"  mean={np.mean(sims):.4f} min={np.min(sims):.4f} max={np.max(sims):.4f}")


if __name__ == "__main__":
    main()
