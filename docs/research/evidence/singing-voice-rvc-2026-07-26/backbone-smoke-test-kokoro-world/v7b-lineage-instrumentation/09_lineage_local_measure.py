#!/usr/bin/env python3
"""Per-phoneme-span localized measurement, using v7b's exact frame
lineage (ground truth from the renderer's own bookkeeping, not an
inferred external mapping) to measure three regions SEPARATELY for
every raw (waveform-preserved voiceless-obstruent) group, per the
reviewer's request:

  1. Core interior (excluding crossfade) vs. its exact source region --
     "was the original consonant actually preserved?"
  2. Entry crossfade -- energy loss / high-frequency attenuation vs. the
     group's own pre-crossfade core.
  3. Exit crossfade -- same, at the other boundary.
"""
import json
import numpy as np
import soundfile as sf
from pathlib import Path

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")


def band_energy(x, fs, lo, hi):
    if len(x) < 8:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    mask = (freqs >= lo) & (freqs <= hi)
    total = np.sum(spec**2) + 1e-12
    return float(np.sum(spec[mask] ** 2) / total)


def centroid(x, fs):
    if len(x) < 8:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    if spec.sum() < 1e-9:
        return 0.0
    return float(np.sum(freqs * spec) / spec.sum())


def zcr(x):
    if len(x) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(np.sign(x))) > 0))


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0


def measure_region(x, fs):
    return {
        "rms": round(rms(x), 5),
        "centroid_hz": round(centroid(x, fs), 0),
        "zcr": round(zcr(x), 4),
        "high_band_frac_4_10k": round(band_energy(x, fs, 4000, 10000), 4),
        "n_samples": len(x),
    }


def analyze_phrase(pid, spoken_path, output_path, lineage_path):
    x_source, fs_s = sf.read(str(spoken_path))
    if x_source.ndim > 1:
        x_source = x_source.mean(axis=1)
    x_output, fs_o = sf.read(str(output_path))
    if x_output.ndim > 1:
        x_output = x_output.mean(axis=1)
    lineage = json.loads(Path(lineage_path).read_text())
    fs = lineage["sample_rate"]

    print(f"\n=== {pid} ===")
    for g in lineage["groups"]:
        if g["method"] != "raw":
            continue
        core = x_output[g["core_start"]:g["core_end"]]
        src = x_source[g["source_start_sample"]:g["source_end_sample"]]
        core_stats = measure_region(core, fs)
        src_stats = measure_region(src, fs)
        centroid_retention = (core_stats["centroid_hz"] / src_stats["centroid_hz"] * 100
                               if src_stats["centroid_hz"] > 0 else float("nan"))
        zcr_retention = (core_stats["zcr"] / src_stats["zcr"] * 100
                          if src_stats["zcr"] > 0 else float("nan"))
        print(f"  word={g['word_text']!r} phonemes={g['phonemes']!r}")
        print(f"    source (n={src_stats['n_samples']:4d}): {src_stats}")
        print(f"    core   (n={core_stats['n_samples']:4d}): {core_stats}")
        print(f"    RETENTION: centroid={centroid_retention:.0f}%  zcr={zcr_retention:.0f}%")

        if g["entry_fade"] is not None:
            ef = x_output[g["entry_fade"][0]:g["entry_fade"][1]]
            print(f"    entry_fade: {measure_region(ef, fs)}")
        if g["exit_fade"] is not None:
            xf = x_output[g["exit_fade"][0]:g["exit_fade"][1]]
            print(f"    exit_fade:  {measure_region(xf, fs)}")


if __name__ == "__main__":
    for pid in ["consonant_clusters", "fricative_heavy", "phrase_final_stops"]:
        analyze_phrase(
            pid,
            BASE / "gate2_audio" / f"{pid}_spoken.wav",
            BASE / "gate2_audio" / f"{pid}_sung_v7b.wav",
            BASE / "gate2_audio" / f"{pid}_sung_v7b_lineage.json",
        )
