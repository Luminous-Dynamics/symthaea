#!/usr/bin/env python3
"""Gate 2 evaluation: Whisper WER on the Gate D phrase set, same method
throughout this arc, for direct comparison against ACE-Step v1's Gate D
capability-boundary map.
"""
import json
import re
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
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
    sung_path = BASE / "gate2_audio" / f"{phrase['id']}_sung.wav"
    if not sung_path.exists():
        results.append({"id": phrase["id"], "target": target, "status": "SKIPPED (no render)"})
        continue
    segments, _info = whisper.transcribe(str(sung_path), language="en")
    hyp = " ".join(s.text for s in segments).strip()
    wer = word_error_rate(target, normalize(hyp))
    results.append({"id": phrase["id"], "target": target, "wer": round(wer, 3), "hypothesis": hyp})

(BASE / "gate2_results.json").write_text(json.dumps(results, indent=2))

print("\n=== Gate 2: Kokoro+WORLD-vocoder on the Gate D phrase set ===\n")
for r in results:
    if "wer" in r:
        print(f"{r['id']:22s} WER={r['wer']:.3f}  target=\"{r['target']}\"")
        print(f"{'':22s}  hyp=\"{r['hypothesis']}\"")
    else:
        print(f"{r['id']:22s} {r['status']}")

valid = [r for r in results if "wer" in r]
overall = np.mean([r["wer"] for r in valid])
print(f"\nOverall WER ({len(valid)}/{len(results)} phrases rendered): {overall:.3f}")
