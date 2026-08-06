#!/usr/bin/env python3
"""Evaluate the exit-crossfade ablation (Arms A/B/C) against the
reviewer's pre-registered acceptance criteria:
  - at least double high-band retention (exit_hb/core_hb) vs. Arm A's
    baseline ratio (~0.135);
  - improve centroid retention (exit_centroid/core_centroid) vs. Arm A's
    baseline (~44%);
  - preserve perfect transcription (WER 0.0 on all 3 phrases);
  - not increase max boundary discontinuity by more than 20% vs Arm A.
"""
import json
import re
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from pathlib import Path

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
ARMS = ["A", "B", "C"]
PHRASES = {
    "consonant_clusters": "strong streams splashed strangely",
    "fricative_heavy": "she sells seashells by the seashore",
    "phrase_final_stops": "turn off the light and lock it",
}


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def word_error_rate(ref, hyp):
    rw, hw = ref.split(), hyp.split()
    n, m = len(rw), len(hw)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if rw[i - 1] == hw[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return dp[n, m] / max(1, n)


def centroid(x, fs):
    if len(x) < 8:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    return float(np.sum(freqs * spec) / spec.sum()) if spec.sum() > 1e-9 else 0.0


def high_band_frac(x, fs, lo=4000, hi=10000):
    if len(x) < 8:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    mask = (freqs >= lo) & (freqs <= hi)
    tot = np.sum(spec**2) + 1e-12
    return float(np.sum(spec[mask] ** 2) / tot)


whisper = WhisperModel("base", device="cpu", compute_type="int8")

results = {}
for arm in ARMS:
    core_cents, exit_cents, core_hbs, exit_hbs, max_clicks, wers = [], [], [], [], [], []
    for pid, target_text in PHRASES.items():
        wav_path = BASE / "gate2_audio" / f"{pid}_sung_v8_{arm}.wav"
        lineage_path = BASE / "gate2_audio" / f"{pid}_sung_v8_{arm}_lineage.json"
        x, fs = sf.read(str(wav_path))
        if x.ndim > 1:
            x = x.mean(axis=1)
        lineage = json.loads(lineage_path.read_text())
        for g in lineage["groups"]:
            if g["method"] != "raw":
                continue
            core = x[g["core_start"]:g["core_end"]]
            core_cents.append(centroid(core, fs))
            core_hbs.append(high_band_frac(core, fs))
            if g["exit_fade"]:
                ef = x[g["exit_fade"][0]:g["exit_fade"][1]]
                exit_cents.append(centroid(ef, fs))
                exit_hbs.append(high_band_frac(ef, fs))

        diffs = np.abs(np.diff(x))
        max_clicks.append(float(diffs.max()))

        segments, _info = whisper.transcribe(str(wav_path), language="en")
        hyp = " ".join(s.text for s in segments).strip()
        wers.append(word_error_rate(normalize(target_text), normalize(hyp)))

    results[arm] = {
        "mean_core_centroid": np.mean(core_cents),
        "mean_exit_centroid": np.mean(exit_cents) if exit_cents else float("nan"),
        "mean_core_hb": np.mean(core_hbs),
        "mean_exit_hb": np.mean(exit_hbs) if exit_hbs else float("nan"),
        "mean_max_click": np.mean(max_clicks),
        "mean_wer": np.mean(wers),
    }

print("\n=== Exit-crossfade ablation: Arms A/B/C ===\n")
for arm in ARMS:
    r = results[arm]
    centroid_ratio = r["mean_exit_centroid"] / r["mean_core_centroid"]
    hb_ratio = r["mean_exit_hb"] / r["mean_core_hb"]
    print(f"Arm {arm}:")
    print(f"  core centroid={r['mean_core_centroid']:.0f}Hz  exit centroid={r['mean_exit_centroid']:.0f}Hz  "
          f"ratio={centroid_ratio:.3f}")
    print(f"  core high-band={r['mean_core_hb']:.4f}  exit high-band={r['mean_exit_hb']:.4f}  "
          f"ratio={hb_ratio:.4f}")
    print(f"  mean max click={r['mean_max_click']:.3f}  mean WER={r['mean_wer']:.3f}")
    print()

a = results["A"]
a_centroid_ratio = a["mean_exit_centroid"] / a["mean_core_centroid"]
a_hb_ratio = a["mean_exit_hb"] / a["mean_core_hb"]
print("=== Pre-registered acceptance check (vs Arm A baseline) ===")
for arm in ["B", "C"]:
    r = results[arm]
    c_ratio = r["mean_exit_centroid"] / r["mean_core_centroid"]
    h_ratio = r["mean_exit_hb"] / r["mean_core_hb"]
    hb_doubled = h_ratio >= 2 * a_hb_ratio
    centroid_improved = c_ratio > a_centroid_ratio
    wer_perfect = r["mean_wer"] == 0.0
    click_ok = r["mean_max_click"] <= a["mean_max_click"] * 1.2
    print(f"Arm {arm}: hb_ratio={h_ratio:.4f} (need >= {2*a_hb_ratio:.4f}) -> {hb_doubled}; "
          f"centroid_ratio={c_ratio:.3f} (need > {a_centroid_ratio:.3f}) -> {centroid_improved}; "
          f"WER={r['mean_wer']:.3f} -> {wer_perfect}; "
          f"click={r['mean_max_click']:.3f} (need <= {a['mean_max_click']*1.2:.3f}) -> {click_ok}")
    print(f"  ACCEPT: {hb_doubled and centroid_improved and wer_perfect and click_ok}")
