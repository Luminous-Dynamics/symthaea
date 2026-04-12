#!/usr/bin/env python3
"""Symthaea Audio Quality Analysis — MIR-based evaluation of generated compositions.

Uses librosa for scientifically grounded audio features:
- Spectral: centroid, bandwidth, rolloff, flatness
- Rhythm: onset detection, beat tracking, tempo estimation
- Tonal: chroma features, key estimation
- Dynamics: RMS, peak, crest factor, silence ratio, loudness contour

Usage:
    nix-shell -p "python3.withPackages(ps: with ps; [numpy scipy librosa soundfile])" \
        --run "python3 scripts/analyze_audio.py audio_output/demo_*.wav"
"""

import sys
import os
import numpy as np
import librosa
import soundfile as sf


def analyze(path: str) -> dict:
    """Full MIR analysis of a WAV file."""
    y, sr = sf.read(path)

    # Mono mix if stereo
    if y.ndim > 1:
        channels = y.shape[1]
        y_mono = y.mean(axis=1)
    else:
        channels = 1
        y_mono = y

    duration = len(y_mono) / sr

    # ── Dynamics ──────────────────────────────────────────────────────
    peak = float(np.max(np.abs(y_mono)))
    rms_arr = librosa.feature.rms(y=y_mono, frame_length=2048, hop_length=512)[0]
    rms_mean = float(np.mean(rms_arr))
    rms_std = float(np.std(rms_arr))

    peak_db = 20 * np.log10(peak) if peak > 0 else -100
    rms_db = 20 * np.log10(rms_mean) if rms_mean > 0 else -100
    crest_db = peak_db - rms_db

    silence_thresh = 0.001
    silence_ratio = float(np.sum(np.abs(y_mono) < silence_thresh) / len(y_mono))
    clipped = int(np.sum(np.abs(y_mono) >= 0.999))

    # Dynamic range: difference between loud and quiet passages
    rms_db_arr = 20 * np.log10(rms_arr + 1e-10)
    dynamic_range = float(np.percentile(rms_db_arr, 95) - np.percentile(rms_db_arr, 5))

    # ── Spectral ─────────────────────────────────────────────────────
    S = np.abs(librosa.stft(y_mono))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]

    centroid_mean = float(np.mean(centroid))
    bandwidth_mean = float(np.mean(bandwidth))
    rolloff_mean = float(np.mean(rolloff))
    flatness_mean = float(np.mean(flatness))

    # ── Rhythm ───────────────────────────────────────────────────────
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
    onsets = librosa.onset.onset_detect(y=y_mono, sr=sr, onset_envelope=onset_env)
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    n_onsets = len(onset_times)

    # Beat tracking
    tempo_est, beats = librosa.beat.beat_track(y=y_mono, sr=sr, onset_envelope=onset_env)
    beat_times = librosa.frames_to_time(beats, sr=sr)

    # Onset regularity (CV of inter-onset intervals)
    if len(onset_times) > 2:
        iois = np.diff(onset_times)
        ioi_mean = float(np.mean(iois))
        ioi_cv = float(np.std(iois) / np.mean(iois)) if np.mean(iois) > 0 else 999
    else:
        ioi_mean = 0
        ioi_cv = 999

    # ── Tonal ────────────────────────────────────────────────────────
    chroma = librosa.feature.chroma_cqt(y=y_mono, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    key_idx = int(np.argmax(chroma_mean))
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    estimated_key = key_names[key_idx]
    key_strength = float(chroma_mean[key_idx])

    # Harmonic-to-noise ratio proxy: harmonic vs percussive energy
    y_harm, y_perc = librosa.effects.hpss(y_mono)
    harm_energy = float(np.sum(y_harm ** 2))
    perc_energy = float(np.sum(y_perc ** 2))
    hnr = 10 * np.log10(harm_energy / (perc_energy + 1e-10))

    # ── Zero crossing rate ───────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y_mono)[0]
    zcr_mean = float(np.mean(zcr)) * sr

    # ── Quality scores (0-1) ─────────────────────────────────────────
    # These are interpretive scores derived from the raw features

    # Dynamics score: good music has moderate dynamic range (not flat, not clipping)
    dynamics_score = min(dynamic_range / 30.0, 1.0) * (1.0 if clipped == 0 else 0.5)

    # Brightness score: centroid between 500-3000 Hz is typical for music
    brightness_score = 1.0 - abs(centroid_mean - 1500) / 3000
    brightness_score = max(0, min(1, brightness_score))

    # Rhythm score: based on onset regularity and density
    onset_density = n_onsets / duration
    rhythm_density_score = min(onset_density / 4.0, 1.0)  # 4 onsets/sec = good
    rhythm_regularity_score = max(0, 1.0 - ioi_cv) if ioi_cv < 10 else 0
    rhythm_score = 0.5 * rhythm_density_score + 0.5 * rhythm_regularity_score

    # Tonal score: strong key center = more musical
    tonal_score = min(key_strength / 0.3, 1.0)

    # Harmonic score: harmonic content should dominate
    harmonic_score = min(max(hnr, 0) / 20.0, 1.0)

    # Composite
    composite = (0.25 * dynamics_score + 0.20 * brightness_score +
                 0.25 * rhythm_score + 0.15 * tonal_score + 0.15 * harmonic_score)

    return {
        'name': os.path.basename(path).replace('.wav', '').replace('demo_', ''),
        'duration': duration,
        'channels': channels,
        'sr': sr,
        # Dynamics
        'peak_db': peak_db,
        'rms_db': rms_db,
        'crest_db': crest_db,
        'silence_pct': silence_ratio * 100,
        'clipped': clipped,
        'dynamic_range': dynamic_range,
        'rms_std': rms_std,
        # Spectral
        'centroid': centroid_mean,
        'bandwidth': bandwidth_mean,
        'rolloff': rolloff_mean,
        'flatness': flatness_mean,
        'zcr': zcr_mean,
        # Rhythm
        'tempo': float(np.mean(tempo_est)) if hasattr(tempo_est, '__len__') else float(tempo_est),
        'n_onsets': n_onsets,
        'onset_density': n_onsets / duration,
        'ioi_mean': ioi_mean,
        'ioi_cv': ioi_cv,
        # Tonal
        'key': estimated_key,
        'key_strength': key_strength,
        'hnr': hnr,
        # Scores
        'dynamics_score': dynamics_score,
        'brightness_score': brightness_score,
        'rhythm_score': rhythm_score,
        'tonal_score': tonal_score,
        'harmonic_score': harmonic_score,
        'composite': composite,
    }


def print_report(results: list[dict]):
    print("\n" + "=" * 90)
    print("  Symthaea Audio Quality Report (librosa MIR analysis)")
    print("=" * 90)

    # Summary table
    print(f"\n  {'Composition':15s} | {'Peak':>6s} | {'RMS':>7s} | {'Silence':>7s} | {'Tempo':>5s} | {'Onsets':>6s} | {'Key':>4s} | {'HNR':>5s} | {'Score':>5s}")
    print("  " + "-" * 82)
    for r in results:
        print(f"  {r['name']:15s} | {r['peak_db']:5.1f}dB | {r['rms_db']:6.1f}dB | {r['silence_pct']:5.1f}% | {r['tempo']:5.1f} | {r['n_onsets']:6d} | {r['key']:>4s} | {r['hnr']:4.1f}dB | {r['composite']:.3f}")

    print(f"\n  Detailed Scores:")
    print(f"  {'Composition':15s} | {'Dynamics':>8s} | {'Bright':>6s} | {'Rhythm':>6s} | {'Tonal':>5s} | {'Harmonic':>8s} | {'Composite':>9s}")
    print("  " + "-" * 72)
    for r in results:
        print(f"  {r['name']:15s} | {r['dynamics_score']:8.3f} | {r['brightness_score']:6.3f} | {r['rhythm_score']:6.3f} | {r['tonal_score']:5.3f} | {r['harmonic_score']:8.3f} | {r['composite']:9.3f}")

    # Spectral detail
    print(f"\n  Spectral Features:")
    print(f"  {'Composition':15s} | {'Centroid':>8s} | {'Bandwidth':>9s} | {'Rolloff':>7s} | {'Flatness':>8s} | {'ZCR':>6s}")
    print("  " + "-" * 68)
    for r in results:
        print(f"  {r['name']:15s} | {r['centroid']:7.0f}Hz | {r['bandwidth']:8.0f}Hz | {r['rolloff']:6.0f}Hz | {r['flatness']:8.4f} | {r['zcr']:5.0f}Hz")

    # Rhythm detail
    print(f"\n  Rhythm Features:")
    print(f"  {'Composition':15s} | {'Tempo':>6s} | {'Onsets':>6s} | {'Density':>7s} | {'IOI mean':>8s} | {'IOI CV':>6s}")
    print("  " + "-" * 58)
    for r in results:
        print(f"  {r['name']:15s} | {r['tempo']:5.1f} | {r['n_onsets']:6d} | {r['onset_density']:6.2f}/s | {r['ioi_mean']:7.3f}s | {r['ioi_cv']:6.3f}")

    # Diagnosis
    print("\n  Diagnosis:")
    for r in results:
        issues = []
        if r['silence_pct'] > 50:
            issues.append(f"HIGH SILENCE ({r['silence_pct']:.0f}%)")
        if r['n_onsets'] < 5:
            issues.append(f"VERY SPARSE ({r['n_onsets']} onsets)")
        if r['clipped'] > 0:
            issues.append(f"CLIPPING ({r['clipped']} samples)")
        if r['rms_db'] < -40:
            issues.append(f"VERY QUIET (RMS {r['rms_db']:.1f}dB)")
        if r['ioi_cv'] > 1.0:
            issues.append("IRREGULAR RHYTHM")
        if r['hnr'] < 5:
            issues.append(f"LOW HARMONICITY (HNR {r['hnr']:.1f}dB)")
        if r['dynamic_range'] < 3:
            issues.append("FLAT DYNAMICS")

        status = " | ".join(issues) if issues else "OK"
        print(f"  {r['name']:15s}: {status}")

    # Mean composite
    mean_comp = np.mean([r['composite'] for r in results])
    print(f"\n  Mean Audio Quality Score: {mean_comp:.3f}")
    print()


if __name__ == '__main__':
    paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not paths:
        d = "audio_output"
        if os.path.isdir(d):
            paths = sorted([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.wav')])
    if not paths:
        print("Usage: python3 analyze_audio.py <wav_files...>")
        sys.exit(1)

    results = [analyze(p) for p in paths]
    print_report(results)
