#!/usr/bin/env python3
"""Real audio-comparison analysis for the singing-voice-rvc-2026-07-26
evidence bundle. Run with the RVC venv's Python (has parselmouth +
soundfile): /var/lib/symthaea/training-runs/voice-conversion/rvc-venv/bin/python3

Methodology (see methodology.md for full detail):
  - F0 extracted via Praat autocorrelation (parselmouth Sound.to_pitch_ac),
    20ms time step, 65-1100 Hz search range, no manual octave correction.
  - F0 correlation is a direct Pearson correlation over frames voiced in
    BOTH signals (no DTW/time-alignment -- a stricter, lower-bound measure
    since RVC's synthesis path can introduce small timing drift).
  - Silence fraction: RMS computed per 2048-sample frame (512 hop, ~46ms/12ms
    at 44.1kHz), fraction of frames with 20*log10(rms) < -50 dB.
  - This is a real, run script, not a black-box number -- rerun to verify.
"""
import json
import sys

import numpy as np
import parselmouth
import soundfile as sf

BASE = "/srv/luminous-dynamics/symthaea/audio_output/diffsinger_csd_poc_2026-07-25/"


def f0_curve(path, timestep=0.02):
    snd = parselmouth.Sound(path)
    pitch = snd.to_pitch_ac(time_step=timestep, pitch_floor=65, pitch_ceiling=1100)
    return pitch.selected_array["frequency"]


def silence_frac(path, thresh_db=-50, frame=2048, hop=512):
    a, sr = sf.read(path)
    if a.ndim > 1:
        a = a.mean(axis=1)
    n_frames = 1 + (len(a) - frame) // hop
    below = 0
    for i in range(n_frames):
        seg = a[i * hop : i * hop + frame]
        rms = np.sqrt(np.mean(seg**2)) + 1e-12
        db = 20 * np.log10(rms)
        if db < thresh_db:
            below += 1
    return below / n_frames if n_frames else 0.0


def f0_correlation(src, cmp):
    f0_a = f0_curve(src)
    f0_b = f0_curve(cmp)
    n = min(len(f0_a), len(f0_b))
    f0_a, f0_b = f0_a[:n], f0_b[:n]
    voiced = (f0_a > 0) & (f0_b > 0)
    if voiced.sum() < 10:
        return None, int(voiced.sum())
    return float(np.corrcoef(f0_a[voiced], f0_b[voiced])[0, 1]), int(voiced.sum())


def analyze_pair(name, src, cmp):
    corr, n_voiced = f0_correlation(src, cmp)
    return {
        "pair": name,
        "source_file": src.replace(BASE, ""),
        "compare_file": cmp.replace(BASE, ""),
        "f0_correlation_pearson_no_dtw": corr,
        "f0_co_voiced_frames": n_voiced,
        "silence_fraction_below_-50dB": {
            "source": round(silence_frac(src), 4),
            "compare": round(silence_frac(cmp), 4),
        },
    }


if __name__ == "__main__":
    results = []
    # 12s-clip checkpoints: ep50 and ep75 were both converted from the same
    # trimmed 12s source (en001a_clip12s_ORIGINAL_diffsinger.wav).
    src12 = BASE + "en001a_clip12s_ORIGINAL_diffsinger.wav"
    results.append(analyze_pair("ep50_vs_source_12s", src12, BASE + "en001a_af_heart_ep50_12s.wav"))
    results.append(analyze_pair("ep75_vs_source_12s", src12, BASE + "en001a_af_heart_ep75_12s.wav"))
    # Full 64s clip: final (epoch 200) checkpoint vs the full DiffSinger source.
    src64 = BASE + "en001a-step2000-final.wav"
    results.append(analyze_pair("final_ep200_vs_source_64s", src64, BASE + "en001a_af_heart_FINAL_ep200.wav"))

    out = {"methodology_ref": "methodology.md", "results": results}
    print(json.dumps(out, indent=2))
    with open("audio-comparison.json", "w") as f:
        json.dump(out, f, indent=2)
