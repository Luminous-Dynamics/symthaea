#!/usr/bin/env python3
"""Evaluate v3 (syllable + gestures) against v2 (phoneme-level, word
granularity) -- same phrases, WER + click, plus a v3-specific check:
recompute the syllable-level target sequence to measure melody tracking
against the FINER target (since v3's melody is per-syllable, the old
per-word target array from config.json no longer matches v3's actual
note assignment).
"""
import json
import re
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf
from faster_whisper import WhisperModel
from kokoro import KPipeline

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
config = json.loads((BASE / "config.json").read_text())

VOWEL_CHARS = set("əɜʌIᵻæiɔɪɐuʊɑɛeAOWYɚɝᵊ")
STRESS_MARKS = "ˈˌ"
NOTES_CYCLE = [261.63, 293.66, 329.63, 392.00, 440.00]


def strip_stress(ps):
    return "".join(c for c in ps if c not in STRESS_MARKS)


def classify(ps):
    return [(c, c in VOWEL_CHARS) for c in strip_stress(ps)]


def syllabify(phonemes):
    if not phonemes:
        return []
    vowel_idx = [i for i, (_, v) in enumerate(phonemes) if v]
    if not vowel_idx:
        return [phonemes]
    syllables, start = [], 0
    for k, vi in enumerate(vowel_idx):
        end = vi + 1 if k + 1 < len(vowel_idx) else len(phonemes)
        syllables.append(phonemes[start:end])
        start = end
    return syllables


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


def melody_tracking(wav_path, target_hz_seq):
    x, fs = sf.read(str(wav_path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    f0, _t = pw.harvest(x, fs, frame_period=5.0)
    voiced = f0[f0 > 0]
    if len(voiced) < 5:
        return None
    log_f0 = np.log2(voiced)
    log_targets = np.log2(np.array(target_hz_seq))
    cents_err = np.min(np.abs(log_f0[:, None] - log_targets[None, :]) * 1200.0, axis=1)
    return {"median_cents_err": round(float(np.median(cents_err)), 1),
            "frac_within_50c": round(float(np.mean(cents_err < 50.0)), 3)}


def max_click(wav_path):
    x, fs = sf.read(str(wav_path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    diffs = np.abs(np.diff(x.astype(np.float64)))
    return round(float(diffs.max()), 3)


pipeline = KPipeline(lang_code="a")
whisper = WhisperModel("base", device="cpu", compute_type="int8")

note_idx = 0
rows = []
for phrase in config["phrases"]:
    target_text = normalize(phrase["text"])
    ps_full = " ".join(ps for _gs, ps, _audio in pipeline(phrase["text"], voice="af_heart"))
    n_syllables_total = sum(len(syllabify(classify(w))) for w in ps_full.split())
    syll_melody = [NOTES_CYCLE[(note_idx + i) % len(NOTES_CYCLE)] for i in range(n_syllables_total)]
    note_idx += n_syllables_total

    row = {"id": phrase["id"], "target": target_text, "n_syllables": n_syllables_total}
    for variant, suffix, targets in (
        ("v2_phoneme_word", "sung", phrase["melody_hz"]),
        ("v3_syllable_gesture", "sung_v3", syll_melody),
    ):
        wav_path = BASE / "audio" / f"{phrase['id']}_{suffix}.wav"
        segments, _info = whisper.transcribe(str(wav_path), language="en")
        hyp = " ".join(s.text for s in segments).strip()
        wer = word_error_rate(target_text, normalize(hyp))
        melody = melody_tracking(wav_path, targets)
        click = max_click(wav_path)
        row[variant] = {"wer": round(wer, 3), "hyp": hyp, "melody": melody, "max_click": click}
    rows.append(row)

(BASE / "v3_results.json").write_text(json.dumps(rows, indent=2))

print("\n=== v2 (phoneme, word-granularity) vs v3 (syllable + gestures) ===")
for r in rows:
    print(f"\n{r['id']} ({r['n_syllables']} syllables, target: \"{r['target']}\")")
    for variant in ("v2_phoneme_word", "v3_syllable_gesture"):
        v = r[variant]
        print(f"  {variant:20s} WER={v['wer']:.3f}  melody={v['melody']}  "
              f"max_click={v['max_click']}  hyp=\"{v['hyp']}\"")

v2_wer = np.mean([r["v2_phoneme_word"]["wer"] for r in rows])
v3_wer = np.mean([r["v3_syllable_gesture"]["wer"] for r in rows])
print(f"\nOverall v2: WER={v2_wer:.3f}   Overall v3: WER={v3_wer:.3f}")
