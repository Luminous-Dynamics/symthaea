#!/usr/bin/env python3
"""Evaluate the 4-arm ablation with BOTH harvest and dio F0 estimators
(the v4 correction found these can disagree on transient/consonant-heavy
material -- report both rather than trusting either alone), plus WER.
"""
import re
import numpy as np
import pyworld as pw
import soundfile as sf
from faster_whisper import WhisperModel
from pathlib import Path

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
ARMS = ["A_v3", "B_mask_only", "C_duration_only", "D_combined"]


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
    centroids, zcrs = [], []
    for i in range(0, len(x) - win, hop):
        frame = x[i:i + win] * np.hanning(win)
        spec = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(win, 1 / fs)
        if spec.sum() > 1e-9:
            centroids.append(np.sum(freqs * spec) / spec.sum())
        zc = np.mean(np.abs(np.diff(np.sign(x[i:i + win]))) > 0)
        zcrs.append(zc)
    return {
        "voiced_harvest": round(float(np.mean(f0_h > 0)), 3),
        "voiced_dio": round(float(np.mean(f0_d > 0)), 3),
        "centroid_hz": round(float(np.mean(centroids)), 0),
        "zcr": round(float(np.mean(zcrs)), 4),
    }


whisper = WhisperModel("base", device="cpu", compute_type="int8")

targets = {"consonant_clusters": "strong streams splashed strangely", "hello_world": "hello world"}
dirs = {"consonant_clusters": "gate2_audio", "hello_world": "audio"}

for pid, target_text in targets.items():
    audio_dir = BASE / dirs[pid]
    print(f"\n=== {pid} (target: \"{target_text}\") ===")
    spoken = stats(audio_dir / f"{pid}_spoken.wav")
    print(f"  spoken (reference): {spoken}")
    for arm in ARMS:
        wav_path = audio_dir / f"{pid}_ablation_{arm}.wav"
        s = stats(wav_path)
        segments, _info = whisper.transcribe(str(wav_path), language="en")
        hyp = " ".join(seg.text for seg in segments).strip()
        wer = round(word_error_rate(normalize(target_text), normalize(hyp)), 3)
        print(f"  {arm:16s} WER={wer:.3f}  {s}  hyp=\"{hyp}\"")
