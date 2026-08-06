#!/usr/bin/env python3
"""Pre-registered analytics gates for v13 (continuous-trajectory WORLD +
Vocos) vs. the Arm B / v12 baselines, on the same 3 phrases
(positive_control, fricative_heavy, long_sustained_vowels), per the
reviewer's plan (2026-07-29): F0/note accuracy, unintended silence
duration, RMS drop across word boundaries, spectral-envelope and
aperiodicity discontinuity across joins, and clipping/waveform stability.
WER is scored separately (25_v13_wer_evaluate.py, needs faster_whisper).

Run in the voice-conversion venv (pyworld/soundfile/numpy).
"""
import json
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
V10_DIR = Path("/srv/luminous-dynamics/symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v10_4arm_matrix_full10")
V12_DIR = Path("/srv/luminous-dynamics/symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v12_vocos_resynth")
V13_DIR = Path("/srv/luminous-dynamics/symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v13_continuous_trajectory")

FRAME_PERIOD_MS = 5.0
SILENCE_WINDOW_MS = 20.0
SILENCE_REL_DB = -30.0  # relative to the phrase's own peak
BOUNDARY_WINDOW_MS = 20.0

v13_results = {r["phrase"]: r for r in json.loads((BASE / "v13_continuous_trajectory_results.json").read_text())}


def load(path):
    y, fs = sf.read(str(path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float64), fs


def sanity(y):
    n_nan = int(np.isnan(y).sum())
    n_inf = int(np.isinf(y).sum())
    peak = float(np.max(np.abs(y))) if y.size else float("nan")
    clipped = int(np.sum(np.abs(y) >= 0.999))
    return {"n_nan": n_nan, "n_inf": n_inf, "peak": round(peak, 4), "clipped_samples": clipped}


def silence_duration_s(y, fs):
    win = max(1, int(round(SILENCE_WINDOW_MS / 1000.0 * fs)))
    peak = np.max(np.abs(y)) + 1e-12
    thresh = peak * (10 ** (SILENCE_REL_DB / 20.0))
    n_windows = len(y) // win
    silent_windows = 0
    for i in range(n_windows):
        seg = y[i * win:(i + 1) * win]
        if np.sqrt(np.mean(seg ** 2)) < thresh:
            silent_windows += 1
    return round(silent_windows * win / fs, 4), n_windows


def auto_detect_gap_centers(y, fs, min_gap_ms=25.0):
    """For Arm B / v12 (fixed 60ms silence gaps between words): find
    contiguous silent runs and return their center sample index -- the
    equivalent of v13's recorded word-join boundaries, for a fair
    boundary-RMS-drop / envelope-discontinuity comparison."""
    win = max(1, int(round(SILENCE_WINDOW_MS / 1000.0 * fs)))
    peak = np.max(np.abs(y)) + 1e-12
    thresh = peak * (10 ** (SILENCE_REL_DB / 20.0))
    n_windows = len(y) // win
    is_silent = [np.sqrt(np.mean(y[i * win:(i + 1) * win] ** 2)) < thresh for i in range(n_windows)]
    centers = []
    i = 0
    min_windows = max(1, int(round(min_gap_ms / SILENCE_WINDOW_MS)))
    while i < n_windows:
        if is_silent[i]:
            j = i
            while j < n_windows and is_silent[j]:
                j += 1
            if j - i >= min_windows:
                centers.append(((i + j) // 2) * win)
            i = j
        else:
            i += 1
    return centers


def rms_db(seg):
    r = np.sqrt(np.mean(seg ** 2)) + 1e-12
    return 20 * np.log10(r)


def boundary_rms_drop(y, fs, boundary_samples):
    win = max(1, int(round(BOUNDARY_WINDOW_MS / 1000.0 * fs)))
    interior_rms = []
    boundary_rms = []
    for i in range(0, len(y) - win, win):
        interior_rms.append(rms_db(y[i:i + win]))
    for b in boundary_samples:
        s = max(0, b - win // 2)
        e = min(len(y), s + win)
        if e > s:
            boundary_rms.append(rms_db(y[s:e]))
    if not interior_rms or not boundary_rms:
        return None
    median_interior = float(np.median(interior_rms))
    median_boundary = float(np.median(boundary_rms))
    return {
        "median_interior_rms_db": round(median_interior, 2),
        "median_boundary_rms_db": round(median_boundary, 2),
        "boundary_drop_db": round(median_interior - median_boundary, 2),
    }


def envelope_discontinuity(y, fs, boundary_samples):
    """Re-derive sp/ap from the OUTPUT waveform itself (what's actually
    in the audio, not internal synthesis arrays) and compare frame-to-
    frame log-sp / ap distance AT each boundary vs the typical (median)
    adjacent-frame distance elsewhere in the phrase."""
    f0, t = pw.harvest(y, fs, frame_period=FRAME_PERIOD_MS)
    sp = pw.cheaptrick(y, f0, t, fs)
    ap = pw.d4c(y, f0, t, fs)
    log_sp = np.log(np.clip(sp, 1e-10, None))

    def frame_dist(arr, i, j):
        if i < 0 or j >= arr.shape[0]:
            return None
        return float(np.linalg.norm(arr[j] - arr[i]))

    all_sp_d = [frame_dist(log_sp, i, i + 1) for i in range(log_sp.shape[0] - 1)]
    all_ap_d = [frame_dist(ap, i, i + 1) for i in range(ap.shape[0] - 1)]
    median_sp_d = float(np.median(all_sp_d))
    median_ap_d = float(np.median(all_ap_d))

    frame_dt = FRAME_PERIOD_MS / 1000.0
    b_sp_d, b_ap_d = [], []
    for b_sample in boundary_samples:
        f = int(round(b_sample / fs / frame_dt))
        d_sp = frame_dist(log_sp, f - 1, f)
        d_ap = frame_dist(ap, f - 1, f)
        if d_sp is not None:
            b_sp_d.append(d_sp)
        if d_ap is not None:
            b_ap_d.append(d_ap)

    if not b_sp_d:
        return None
    return {
        "median_interior_sp_frame_dist": round(median_sp_d, 4),
        "median_boundary_sp_frame_dist": round(float(np.median(b_sp_d)), 4),
        "sp_discontinuity_ratio": round(float(np.median(b_sp_d)) / max(median_sp_d, 1e-9), 3),
        "median_interior_ap_frame_dist": round(median_ap_d, 4),
        "median_boundary_ap_frame_dist": round(float(np.median(b_ap_d)), 4),
        "ap_discontinuity_ratio": round(float(np.median(b_ap_d)) / max(median_ap_d, 1e-9), 3),
    }


def melody_tracking_score(y, fs, target_hz_sequence):
    f0, t = pw.harvest(y, fs, frame_period=FRAME_PERIOD_MS)
    voiced = f0[f0 > 0]
    if len(voiced) == 0:
        return {"median_cents_error_to_nearest_target_note": None, "fraction_frames_within_50_cents": None}
    log_f0 = np.log2(voiced)
    log_targets = np.log2(np.array(target_hz_sequence))
    cents_err = np.min(np.abs(log_f0[:, None] - log_targets[None, :]) * 1200.0, axis=1)
    return {
        "median_cents_error_to_nearest_target_note": round(float(np.median(cents_err)), 1),
        "fraction_frames_within_50_cents": round(float(np.mean(cents_err < 50.0)), 3),
    }


def analyze_one(label, path, target_hz_sequence, boundary_samples=None):
    y, fs = load(path)
    s = sanity(y)
    sil_s, n_windows = silence_duration_s(y, fs)
    if boundary_samples is None:
        boundary_samples = auto_detect_gap_centers(y, fs)
    rms = boundary_rms_drop(y, fs, boundary_samples)
    env = envelope_discontinuity(y, fs, boundary_samples)
    melody = melody_tracking_score(y, fs, target_hz_sequence)
    return {
        "label": label, "path": str(path), "duration_s": round(len(y) / fs, 3),
        "n_boundaries_measured": len(boundary_samples),
        "sanity": s, "silence_duration_s": sil_s,
        "boundary_rms": rms, "envelope_discontinuity": env, "melody": melody,
    }


def main():
    all_results = {}
    for phrase, r in v13_results.items():
        target_hz = r["target_hz_sequence"]
        v13_boundaries = r["boundary_sample_frames"]

        entries = []
        armb_path = V10_DIR / f"{phrase}_sung_v10full_b.wav"
        if armb_path.exists():
            entries.append(analyze_one("Arm B (v10, per-word + 60ms gaps)", armb_path, target_hz))
        v12_path = V12_DIR / f"{phrase}_sung_v12_vocos.wav"
        if v12_path.exists():
            entries.append(analyze_one("v12 (Arm B + Vocos resynth)", v12_path, target_hz))
        v13w_path = V13_DIR / f"{phrase}_sung_v13_world_only.wav"
        if v13w_path.exists():
            entries.append(analyze_one("v13 world-only (continuous trajectory, no Vocos)", v13w_path, target_hz, v13_boundaries))
        v13v_path = V13_DIR / f"{phrase}_sung_v13_vocos.wav"
        if v13v_path.exists():
            entries.append(analyze_one("v13 + Vocos (continuous trajectory)", v13v_path, target_hz, v13_boundaries))

        all_results[phrase] = entries

        print(f"\n=== {phrase} ===")
        for e in entries:
            print(f"  {e['label']}")
            print(f"    sanity: {e['sanity']}")
            print(f"    silence_duration_s: {e['silence_duration_s']}")
            if e["boundary_rms"]:
                print(f"    boundary_rms: {e['boundary_rms']}")
            if e["envelope_discontinuity"]:
                print(f"    envelope_discontinuity: {e['envelope_discontinuity']}")
            print(f"    melody: {e['melody']}")

    out_path = BASE / "v13_boundary_analytics_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
