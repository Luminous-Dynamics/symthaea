#!/usr/bin/env python3
"""WER for the v12 Vocos-resynthesized renders (Step 4 sanity check), same
method as gate2_04_evaluate.py / the v10full arm evaluators, for direct
comparison against Arm B's baseline WER (0.284 mean, v10full_wer_results.json).
"""
import json
import re
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
AUDIO_DIR = Path("/srv/luminous-dynamics/symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v12_vocos_resynth")
config = json.loads((BASE / "gate2_config.json").read_text())


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

results = []
for phrase in config["phrases"]:
    target = normalize(phrase["text"])
    sung_path = AUDIO_DIR / f"{phrase['id']}_sung_v12_vocos.wav"
    if not sung_path.exists():
        results.append({"id": phrase["id"], "target": target, "status": "SKIPPED (no render)"})
        continue
    segments, _info = whisper.transcribe(str(sung_path), language="en")
    hyp = " ".join(s.text for s in segments).strip()
    wer = word_error_rate(target, normalize(hyp))
    results.append({"id": phrase["id"], "target": target, "wer": round(wer, 3), "hypothesis": hyp})

(BASE / "v12_vocos_wer_results.json").write_text(json.dumps(results, indent=2))

print("\n=== v12: Arm B baseline resynthesized through Vocos (charactr/vocos-mel-24khz) ===\n")
for r in results:
    if "wer" in r:
        print(f"{r['id']:22s} WER={r['wer']:.3f}  hyp=\"{r['hypothesis']}\"")
    else:
        print(f"{r['id']:22s} {r['status']}")

valid = [r for r in results if "wer" in r]
mean_wer = sum(r["wer"] for r in valid) / len(valid) if valid else float("nan")
print(f"\nmean WER (v12 vocos-resynth): {mean_wer:.3f}")
print("mean WER (Arm B baseline, v10full): 0.284  <- for comparison")
