#!/usr/bin/env python3
"""DNSMOS/UTMOS naturalness screen for the v12 Vocos-resynthesized renders
(Step 4), same method as 16_naturalness_screen.py, for direct comparison
against the Arm B baseline (1.784 DNSMOS / 1.881 UTMOS), the naturalized-B
attempt (1.852 / 1.913), and the spoken reference (3.319 / 4.373).
"""
import json
from pathlib import Path

import soundfile as sf
import torch
from torchmetrics.functional.audio.dnsmos import (
    deep_noise_suppression_mean_opinion_score as dnsmos,
)

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
VOCOS_DIR = Path("/srv/luminous-dynamics/symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v12_vocos_resynth")

PHRASE_IDS = [
    "positive_control", "conversational", "repeated_syllables",
    "rapid_letter_names", "phrase_final_stops", "fricative_heavy",
    "consonant_clusters", "long_sustained_vowels", "short_unstressed",
    "semantically_unusual",
]


def load(path):
    y, fs = sf.read(str(path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    return torch.tensor(y, dtype=torch.float32), fs


def score_dnsmos(y, fs):
    s = dnsmos(y, fs, personalized=False, device="cpu")
    return {"p808_mos": float(s[0]), "sig": float(s[1]), "bak": float(s[2]), "ovr": float(s[3])}


def main():
    utmos_predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    utmos_predictor.eval()

    results = {}
    for pid in PHRASE_IDS:
        p = VOCOS_DIR / f"{pid}_sung_v12_vocos.wav"
        if not p.exists():
            print(f"  MISSING: {p}")
            continue
        y, fs = load(p)
        d = score_dnsmos(y, fs)
        with torch.no_grad():
            u = float(utmos_predictor(y.unsqueeze(0), fs))
        results[pid] = {"dnsmos": d, "utmos": u}
        print(f"{pid:22s} dnsmos_ovr={d['ovr']:.3f}  utmos={u:.3f}")

    import numpy as np
    dnsmos_vals = [r["dnsmos"]["ovr"] for r in results.values()]
    utmos_vals = [r["utmos"] for r in results.values()]
    mean_dnsmos = float(np.mean(dnsmos_vals)) if dnsmos_vals else float("nan")
    mean_utmos = float(np.mean(utmos_vals)) if utmos_vals else float("nan")

    print(f"\nmean dnsmos_ovr (v12 vocos-resynth): {mean_dnsmos:.3f}  vs Arm B baseline: 1.784  vs naturalized-B: 1.852  vs spoken: 3.319")
    print(f"mean utmos (v12 vocos-resynth):      {mean_utmos:.3f}  vs Arm B baseline: 1.881  vs naturalized-B: 1.913  vs spoken: 4.373")

    out_path = BASE / "v12_vocos_naturalness_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
