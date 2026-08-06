#!/usr/bin/env python3
"""Evaluate v6 (voiced/voiceless obstruent split) against: the existing
4-arm ablation results (for consonant_clusters, direct comparison) and
spoken references (for the two new phrases, fricative_heavy and
phrase_final_stops). Both F0 estimators reported throughout.
"""
import re
import numpy as np
import pyworld as pw
import soundfile as sf
from faster_whisper import WhisperModel
from pathlib import Path

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def word_error_rate(ref, hyp):
    ref_words, hyp_words = ref.split(), hyp.split()
    n, m = len(ref_words), len(hyp_words)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return dp[n, m] / max(1, n)


def stats(path):
    x, fs = sf.read(str(path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    f0_h, _ = pw.harvest(x, fs, frame_period=5.0)
    f0_d, _ = pw.dio(x, fs, frame_period=5.0)
    win, hop = 1024, 256
    centroids = []
    for i in range(0, len(x) - win, hop):
        frame = x[i:i + win] * np.hanning(win)
        spec = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(win, 1 / fs)
        if spec.sum() > 1e-9:
            centroids.append(np.sum(freqs * spec) / spec.sum())
    return {
        "voiced_h": round(float(np.mean(f0_h > 0)), 3),
        "voiced_d": round(float(np.mean(f0_d > 0)), 3),
        "centroid": round(float(np.mean(centroids)), 0),
    }


whisper = WhisperModel("base", device="cpu", compute_type="int8")

print("=== consonant_clusters: v6 vs. existing ablation arms ===")
target = normalize("strong streams splashed strangely")
for label, path in [
    ("spoken", BASE / "gate2_audio/consonant_clusters_spoken.wav"),
    ("A_v3", BASE / "gate2_audio/consonant_clusters_ablation_A_v3.wav"),
    ("B_mask_only", BASE / "gate2_audio/consonant_clusters_ablation_B_mask_only.wav"),
    ("v6_voiced_split", BASE / "gate2_audio/consonant_clusters_sung_v6.wav"),
]:
    s = stats(path)
    if label != "spoken":
        segments, _info = whisper.transcribe(str(path), language="en")
        hyp = " ".join(seg.text for seg in segments).strip()
        wer = round(word_error_rate(target, normalize(hyp)), 3)
        print(f"  {label:16s} WER={wer:.3f}  {s}  hyp=\"{hyp}\"")
    else:
        print(f"  {label:16s} (reference)   {s}")

print("\n=== fricative_heavy / phrase_final_stops: v6 vs. spoken ===")
for pid, target_text in [
    ("fricative_heavy", "she sells seashells by the seashore"),
    ("phrase_final_stops", "turn off the light and lock it"),
]:
    target = normalize(target_text)
    print(f"\n{pid}:")
    for label, suffix in [("spoken", "spoken"), ("v6", "sung_v6")]:
        path = BASE / "gate2_audio" / f"{pid}_{suffix}.wav"
        s = stats(path)
        if label == "spoken":
            print(f"  {label:8s} (reference)   {s}")
        else:
            segments, _info = whisper.transcribe(str(path), language="en")
            hyp = " ".join(seg.text for seg in segments).strip()
            wer = round(word_error_rate(target, normalize(hyp)), 3)
            print(f"  {label:8s} WER={wer:.3f}  {s}  hyp=\"{hyp}\"")

print("\n=== hello_world negative control: v6 ===")
target = normalize("hello world")
for label, suffix in [("spoken", "spoken"), ("v3", "sung_v3"), ("v6", "sung_v6")]:
    path = BASE / "audio" / f"hello_world_{suffix}.wav"
    s = stats(path)
    if label == "spoken":
        print(f"  {label:8s} (reference)   {s}")
    else:
        segments, _info = whisper.transcribe(str(path), language="en")
        hyp = " ".join(seg.text for seg in segments).strip()
        wer = round(word_error_rate(target, normalize(hyp)), 3)
        print(f"  {label:8s} WER={wer:.3f}  {s}  hyp=\"{hyp}\"")
