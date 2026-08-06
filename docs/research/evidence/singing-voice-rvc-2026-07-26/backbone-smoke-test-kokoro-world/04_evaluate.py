#!/usr/bin/env python3
"""Stage 4: intelligibility (Whisper WER, same method as every other gate
in this arc) + a melody-tracking check (does the resynthesized F0 actually
follow the target note sequence, not just wander).
"""
import json
import re
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf
from faster_whisper import WhisperModel

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
config = json.loads((BASE / "config.json").read_text())


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def word_error_rate(ref, hyp):
    ref_words = ref.split()
    hyp_words = hyp.split()
    n, m = len(ref_words), len(hyp_words)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return dp[n, m] / max(1, n)


def melody_tracking_score(wav_path, target_hz_sequence):
    x, fs = sf.read(str(wav_path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    f0, t = pw.harvest(x, fs, frame_period=5.0)
    voiced = f0 > 0
    if voiced.sum() < 5:
        return None, 0.0
    # Coarse check: does the voiced-frame median F0 sit near ANY target
    # note (in log-Hz / cents), and does the overall voiced F0 range span
    # a comparable range to the target melody's range (not collapsed to
    # one flat pitch, not wildly outside it)?
    voiced_f0 = f0[voiced]
    log_f0 = np.log2(voiced_f0)
    log_targets = np.log2(np.array(target_hz_sequence))
    # nearest-target absolute cents error per voiced frame
    cents_err = np.min(
        np.abs(log_f0[:, None] - log_targets[None, :]) * 1200.0, axis=1
    )
    median_cents_err = float(np.median(cents_err))
    frac_within_50c = float(np.mean(cents_err < 50.0))
    target_range_semitones = (log_targets.max() - log_targets.min()) * 12.0
    observed_range_semitones = (log_f0.max() - log_f0.min()) * 12.0
    return {
        "median_cents_error_to_nearest_target_note": round(median_cents_err, 1),
        "fraction_frames_within_50_cents_of_a_target_note": round(frac_within_50c, 3),
        "target_melody_range_semitones": round(target_range_semitones, 2),
        "observed_range_semitones": round(observed_range_semitones, 2),
    }, median_cents_err


whisper = WhisperModel("base", device="cpu", compute_type="int8")

results = []
for phrase in config["phrases"]:
    target_text = normalize(phrase["text"])
    row = {"id": phrase["id"], "target": target_text}
    for kind in ("spoken", "sung"):
        wav_path = BASE / "audio" / f"{phrase['id']}_{kind}.wav"
        segments, _info = whisper.transcribe(str(wav_path), language="en")
        hyp = " ".join(s.text for s in segments).strip()
        hyp_norm = normalize(hyp)
        wer = word_error_rate(target_text, hyp_norm)
        row[f"{kind}_hypothesis"] = hyp
        row[f"{kind}_wer"] = round(wer, 3)
    melody_stats, _ = melody_tracking_score(
        BASE / "audio" / f"{phrase['id']}_sung.wav", phrase["melody_hz"]
    )
    row["melody_tracking"] = melody_stats
    results.append(row)

out_path = BASE / "results.json"
out_path.write_text(json.dumps(results, indent=2))

print("\n=== Results ===")
for r in results:
    print(f"\n{r['id']}  (target: \"{r['target']}\")")
    print(f"  spoken WER: {r['spoken_wer']}   hyp: \"{r['spoken_hypothesis']}\"")
    print(f"  sung   WER: {r['sung_wer']}   hyp: \"{r['sung_hypothesis']}\"")
    print(f"  melody tracking: {r['melody_tracking']}")

overall_spoken_wer = float(np.mean([r["spoken_wer"] for r in results]))
overall_sung_wer = float(np.mean([r["sung_wer"] for r in results]))
print(f"\nOverall spoken WER: {overall_spoken_wer:.3f}")
print(f"Overall sung   WER: {overall_sung_wer:.3f}")
print(f"\nWrote {out_path}")
