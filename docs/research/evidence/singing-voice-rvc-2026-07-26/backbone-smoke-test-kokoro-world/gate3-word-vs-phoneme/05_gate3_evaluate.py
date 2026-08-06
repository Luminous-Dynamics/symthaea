#!/usr/bin/env python3
"""Gate 3 evaluation: word-level vs. phoneme-level retiming, same
click/loudness fixes held constant in both -- isolates the timing-method
variable per the reviewer's Gate 3 spec.
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


def melody_tracking(wav_path, target_hz_sequence):
    x, fs = sf.read(str(wav_path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    f0, _t = pw.harvest(x, fs, frame_period=5.0)
    voiced = f0 > 0
    if voiced.sum() < 5:
        return None
    log_f0 = np.log2(f0[voiced])
    log_targets = np.log2(np.array(target_hz_sequence))
    cents_err = np.min(np.abs(log_f0[:, None] - log_targets[None, :]) * 1200.0, axis=1)
    return {
        "median_cents_err": round(float(np.median(cents_err)), 1),
        "frac_within_50c": round(float(np.mean(cents_err < 50.0)), 3),
    }


def max_click(wav_path):
    x, fs = sf.read(str(wav_path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    diffs = np.abs(np.diff(x.astype(np.float64)))
    return round(float(diffs.max()), 3)


whisper = WhisperModel("base", device="cpu", compute_type="int8")

rows = []
for phrase in config["phrases"]:
    target_text = normalize(phrase["text"])
    row = {"id": phrase["id"], "target": target_text}
    for variant, suffix in (("phoneme_level", "sung"), ("word_level", "sung_wordlevel")):
        wav_path = BASE / "audio" / f"{phrase['id']}_{suffix}.wav"
        segments, _info = whisper.transcribe(str(wav_path), language="en")
        hyp = " ".join(s.text for s in segments).strip()
        wer = word_error_rate(target_text, normalize(hyp))
        melody = melody_tracking(wav_path, phrase["melody_hz"])
        click = max_click(wav_path)
        row[variant] = {"wer": round(wer, 3), "hyp": hyp, "melody": melody, "max_click": click}
    rows.append(row)

out_path = BASE / "gate3_results.json"
out_path.write_text(json.dumps(rows, indent=2))

print("\n=== Gate 3: word-level vs. phoneme-level (same click/loudness fixes) ===")
for r in rows:
    print(f"\n{r['id']}  (target: \"{r['target']}\")")
    for variant in ("word_level", "phoneme_level"):
        v = r[variant]
        print(f"  {variant:14s} WER={v['wer']:.3f}  melody={v['melody']}  "
              f"max_click={v['max_click']}  hyp=\"{v['hyp']}\"")

wl_wer = np.mean([r["word_level"]["wer"] for r in rows])
pl_wer = np.mean([r["phoneme_level"]["wer"] for r in rows])
wl_cents = np.mean([r["word_level"]["melody"]["median_cents_err"] for r in rows])
pl_cents = np.mean([r["phoneme_level"]["melody"]["median_cents_err"] for r in rows])
print(f"\nOverall word-level:    WER={wl_wer:.3f}  mean median-cents-err={wl_cents:.2f}")
print(f"Overall phoneme-level: WER={pl_wer:.3f}  mean median-cents-err={pl_cents:.2f}")
print(f"\nWrote {out_path}")
