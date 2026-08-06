#!/usr/bin/env python3
"""WER for v13 (continuous-trajectory WORLD, world-only and +Vocos) on the
3 target phrases, same method as gate2_04_evaluate.py / 18_vocos_wer_evaluate.py,
for direct comparison against Arm B / v12 baselines on the SAME 3 phrases
(not the full 10-phrase set, for a fair apples-to-apples comparison here).
Run in the ace-step venv (faster_whisper).
"""
import json
import re
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
V10_DIR = Path("/srv/luminous-dynamics/symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v10_4arm_matrix_full10")
V12_DIR = Path("/srv/luminous-dynamics/symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v12_vocos_resynth")
V13_DIR = Path("/srv/luminous-dynamics/symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v13_continuous_trajectory")

config = json.loads((BASE / "gate2_config.json").read_text())
TARGET_TEXT = {p["id"]: p["text"] for p in config["phrases"]}
PHRASES = ["positive_control", "fricative_heavy", "long_sustained_vowels"]


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


whisper = WhisperModel("base", device="cpu", compute_type="int8")

CONDITIONS = {
    "Arm B (v10)": lambda p: V10_DIR / f"{p}_sung_v10full_b.wav",
    "v12 (Arm B + Vocos)": lambda p: V12_DIR / f"{p}_sung_v12_vocos.wav",
    "v13 world-only": lambda p: V13_DIR / f"{p}_sung_v13_world_only.wav",
    "v13 + Vocos": lambda p: V13_DIR / f"{p}_sung_v13_vocos.wav",
}

results = {}
for label, path_fn in CONDITIONS.items():
    wers = []
    print(f"\n=== {label} ===")
    for phrase in PHRASES:
        target = normalize(TARGET_TEXT[phrase])
        p = path_fn(phrase)
        if not p.exists():
            print(f"  {phrase:24s} MISSING: {p}")
            continue
        segments, _info = whisper.transcribe(str(p), language="en")
        hyp = " ".join(s.text for s in segments).strip()
        wer = word_error_rate(target, normalize(hyp))
        wers.append(wer)
        print(f"  {phrase:24s} WER={wer:.3f}  hyp=\"{hyp}\"")
    mean_wer = sum(wers) / len(wers) if wers else float("nan")
    results[label] = {"per_phrase_wer": wers, "mean_wer": round(mean_wer, 3)}
    print(f"  mean WER: {mean_wer:.3f}")

(BASE / "v13_wer_results.json").write_text(json.dumps(results, indent=2))
print("\n=== Summary (3-phrase subset: positive_control, fricative_heavy, long_sustained_vowels) ===")
for label, r in results.items():
    print(f"  {label:24s} mean WER = {r['mean_wer']}")
