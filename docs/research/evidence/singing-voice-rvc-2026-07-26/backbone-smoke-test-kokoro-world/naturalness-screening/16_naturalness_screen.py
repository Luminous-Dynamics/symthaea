#!/usr/bin/env python3
"""Step 1 of the naturalness-improvement plan: no-reference perceptual-
quality screening via DNSMOS (torchmetrics) and UTMOS (SpeechMOS),
alongside the spoken-reference calibration anchor, per
`/home/tstoltz/.claude/plans/synthetic-tumbling-raccoon.md`.

Both predictors are speech-trained, not singing-validated -- reported
honestly as a screening proxy, not ground truth. Purpose: get a fast,
cheap signal for whether a future change (e.g. pitch micro-naturalization)
moves perceived naturalness at all, before spending more of the user's
own listening time.
"""
import json
from pathlib import Path

import soundfile as sf
import torch
from torchmetrics.functional.audio.dnsmos import (
    deep_noise_suppression_mean_opinion_score as dnsmos,
)

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
AUDIO_DIR = BASE / "gate2_audio"

PHRASE_IDS = [
    "positive_control", "conversational", "repeated_syllables",
    "rapid_letter_names", "phrase_final_stops", "fricative_heavy",
    "consonant_clusters", "long_sustained_vowels", "short_unstressed",
    "semantically_unusual",
]
ARMS = ["a", "b", "c", "d"]


def load(path):
    y, fs = sf.read(str(path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    return torch.tensor(y, dtype=torch.float32), fs


def score_dnsmos(y, fs):
    s = dnsmos(y, fs, personalized=False, device="cpu")
    # order: p808_mos, sig, bak, ovr
    return {"p808_mos": float(s[0]), "sig": float(s[1]), "bak": float(s[2]), "ovr": float(s[3])}


def main():
    utmos_predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    utmos_predictor.eval()

    results = {}

    print("=== Spoken references (calibration anchor) ===")
    for pid in PHRASE_IDS:
        p = AUDIO_DIR / f"{pid}_spoken.wav"
        if not p.exists():
            continue
        y, fs = load(p)
        d = score_dnsmos(y, fs)
        with torch.no_grad():
            u = float(utmos_predictor(y.unsqueeze(0), fs))
        results.setdefault(pid, {})["spoken"] = {"dnsmos": d, "utmos": u}
        print(f"{pid:22s} spoken   dnsmos_ovr={d['ovr']:.3f}  utmos={u:.3f}")

    print()
    print("=== v10 4-arm-matrix renders (full 10-phrase set) ===")
    for arm in ARMS:
        for pid in PHRASE_IDS:
            p = AUDIO_DIR / f"{pid}_sung_v10full_{arm}.wav"
            if not p.exists():
                continue
            y, fs = load(p)
            d = score_dnsmos(y, fs)
            with torch.no_grad():
                u = float(utmos_predictor(y.unsqueeze(0), fs))
            results.setdefault(pid, {})[arm] = {"dnsmos": d, "utmos": u}
            print(f"{pid:22s} arm={arm.upper()}  dnsmos_ovr={d['ovr']:.3f}  utmos={u:.3f}")

    print()
    print("=== Summary: mean scores ===")
    import numpy as np
    for key in ["spoken"] + ARMS:
        dnsmos_ovr_vals = [results[pid][key]["dnsmos"]["ovr"] for pid in PHRASE_IDS if key in results.get(pid, {})]
        utmos_vals = [results[pid][key]["utmos"] for pid in PHRASE_IDS if key in results.get(pid, {})]
        if not dnsmos_ovr_vals:
            continue
        print(f"{key:10s} n={len(dnsmos_ovr_vals):2d}  mean_dnsmos_ovr={np.mean(dnsmos_ovr_vals):.3f}  mean_utmos={np.mean(utmos_vals):.3f}")

    out_path = BASE / "naturalness_screen_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
