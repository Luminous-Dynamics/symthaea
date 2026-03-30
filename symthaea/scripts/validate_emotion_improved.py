#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Improved Emotion Validation with Multiple Enhancements

Improvements over baseline:
1. Multiple frontal channel pairs (F3/F4, F7/F8, Fp1/Fp2)
2. Theta/Beta ratio for arousal (more reliable than Beta/Alpha)
3. Individual alpha frequency detection
4. Artifact rejection (amplitude threshold)
5. Longer baseline period
6. Time-windowed analysis within epochs
"""

import json
import struct
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# EGI 128 to 10-20 mapping
EGI = {
    'F3': 23, 'F4': 123, 'F7': 21, 'F8': 8,
    'Fz': 10, 'Fp1': 17, 'Fp2': 14, 'AF3': 27, 'AF4': 122,
    'C3': 35, 'C4': 103, 'Cz': 64,
    'P3': 51, 'P4': 91, 'Pz': 61,
    'O1': 69, 'O2': 82, 'T7': 44, 'T8': 107
}


def load_binary_eeg(path):
    with open(path, 'rb') as f:
        n_ch = struct.unpack('<I', f.read(4))[0]
        n_samp = struct.unpack('<I', f.read(4))[0]
        srate = struct.unpack('<d', f.read(8))[0]
        data = np.fromfile(f, dtype=np.float32).reshape((n_ch, n_samp))
    return data, srate


def parse_events(path):
    events = []
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4 and parts[2] == 'stm':
                onset_ms = float(parts[0])
                label = parts[3]
                stim = label.split('_')[0]
                trial = int(label.split('_')[1]) if '_' in label else 0
                events.append({
                    'onset_samples': int(onset_ms * 250 / 1000),
                    'stimulus': stim,
                    'trial': trial
                })
    return events


def parse_ratings(path):
    ratings = {}
    with open(path) as f:
        header = f.readline().strip().split('\t')
        vi, ai = header.index('valence'), header.index('arousal')
        for line in f:
            p = line.strip().split('\t')
            if len(p) > max(vi, ai):
                stim = p[0].replace('.mp4', '').replace('.m4v', '')
                if 'neutral' in stim:
                    stim = stim.replace('neutral_', 'neutral')
                else:
                    stim = stim.split('.')[0]
                try:
                    ratings[stim] = {'valence': float(p[vi]), 'arousal': float(p[ai])}
                except:
                    pass
    return ratings


def band_power(data, srate, fmin, fmax):
    """Welch PSD with proper parameters."""
    nperseg = min(int(2 * srate), len(data))
    if nperseg < 4:
        return 0.0
    freqs, psd = signal.welch(data, srate, nperseg=nperseg, noverlap=nperseg//2)
    idx = (freqs >= fmin) & (freqs <= fmax)
    if idx.sum() < 2:
        return 0.0
    return np.trapz(psd[idx], freqs[idx])


def find_individual_alpha_peak(data, srate):
    """Find individual alpha frequency (IAF) - typically 8-13 Hz."""
    nperseg = min(int(4 * srate), len(data))
    if nperseg < 4:
        return 10.0  # Default
    freqs, psd = signal.welch(data, srate, nperseg=nperseg)
    alpha_idx = (freqs >= 7) & (freqs <= 14)
    if alpha_idx.sum() < 2:
        return 10.0
    alpha_freqs = freqs[alpha_idx]
    alpha_psd = psd[alpha_idx]
    peak_idx = np.argmax(alpha_psd)
    return alpha_freqs[peak_idx]


def reject_artifacts(epoch, threshold_uv=150):
    """Check if epoch has artifacts (>150 µV peak-to-peak)."""
    # Convert to microvolts (assuming data is in volts)
    epoch_uv = epoch * 1e6
    ptp = np.ptp(epoch_uv, axis=1)
    return np.any(ptp > threshold_uv)


def compute_multi_channel_fai(epoch, srate, egi):
    """
    Compute FAI using multiple frontal channel pairs.
    Average of (F4-F3), (F8-F7), (Fp2-Fp1) asymmetries.
    """
    pairs = [('F3', 'F4'), ('F7', 'F8'), ('Fp1', 'Fp2')]
    fais = []

    for left, right in pairs:
        if left not in egi or right not in egi:
            continue

        left_data = epoch[egi[left], :]
        right_data = epoch[egi[right], :]

        # Use individual alpha frequency
        iaf = find_individual_alpha_peak(left_data, srate)
        alpha_low, alpha_high = iaf - 2, iaf + 2

        left_alpha = band_power(left_data, srate, alpha_low, alpha_high)
        right_alpha = band_power(right_data, srate, alpha_low, alpha_high)

        if left_alpha > 0 and right_alpha > 0:
            fai = np.log(right_alpha) - np.log(left_alpha)
            fais.append(fai)

    return np.mean(fais) if fais else 0.0


def compute_theta_beta_ratio(epoch, srate, egi):
    """
    Compute frontal theta/beta ratio for arousal.
    Higher theta/beta = lower arousal (more relaxed)
    Lower theta/beta = higher arousal (more alert)
    """
    # Use Fz and surrounding channels
    channels = ['Fz', 'F3', 'F4']
    thetas, betas = [], []

    for ch in channels:
        if ch not in egi:
            continue
        data = epoch[egi[ch], :]
        theta = band_power(data, srate, 4, 8)
        beta = band_power(data, srate, 13, 30)
        thetas.append(theta)
        betas.append(beta)

    avg_theta = np.mean(thetas) if thetas else 0
    avg_beta = np.mean(betas) if betas else 1

    if avg_beta > 0:
        return avg_theta / avg_beta
    return 1.0


def compute_arousal_beta_gamma(epoch, srate, egi):
    """
    Alternative arousal: beta + low gamma power.
    High beta/gamma = high arousal.
    """
    channels = ['Fz', 'Cz', 'Pz']
    powers = []

    for ch in channels:
        if ch not in egi:
            continue
        data = epoch[egi[ch], :]
        beta = band_power(data, srate, 15, 25)
        gamma = band_power(data, srate, 25, 40)
        powers.append(beta + 0.5 * gamma)

    return np.mean(powers) if powers else 0


def analyze_epoch_improved(epoch, srate, egi):
    """Improved epoch analysis with multiple metrics."""

    # Check for artifacts
    has_artifact = reject_artifacts(epoch)

    # Multi-channel FAI
    fai = compute_multi_channel_fai(epoch, srate, egi)

    # Theta/Beta ratio (inverse arousal)
    tbr = compute_theta_beta_ratio(epoch, srate, egi)

    # Beta+Gamma arousal
    bg_arousal = compute_arousal_beta_gamma(epoch, srate, egi)

    # Traditional Beta/Alpha for comparison
    fz_data = epoch[egi.get('Fz', 10), :]
    beta = band_power(fz_data, srate, 13, 30)
    alpha = band_power(fz_data, srate, 8, 13)
    ba_ratio = beta / alpha if alpha > 0 else 1.0

    # Convert to 1-9 scale
    # FAI: -0.5 to +0.5 -> 1 to 9
    pred_valence = 5 + fai * 8
    pred_valence = np.clip(pred_valence, 1, 9)

    # Arousal from inverse TBR: low TBR = high arousal
    # TBR typically 0.5-3.0, map inversely
    pred_arousal_tbr = 9 - (tbr - 0.5) * 3
    pred_arousal_tbr = np.clip(pred_arousal_tbr, 1, 9)

    # Arousal from B/A ratio: 0.5-2.0 -> 1-9
    pred_arousal_ba = 1 + (ba_ratio - 0.5) * 5
    pred_arousal_ba = np.clip(pred_arousal_ba, 1, 9)

    # Ensemble arousal (average of methods)
    pred_arousal = (pred_arousal_tbr + pred_arousal_ba) / 2

    return {
        'fai': fai,
        'tbr': tbr,
        'ba_ratio': ba_ratio,
        'bg_arousal': bg_arousal,
        'pred_valence': pred_valence,
        'pred_arousal': pred_arousal,
        'pred_arousal_tbr': pred_arousal_tbr,
        'pred_arousal_ba': pred_arousal_ba,
        'has_artifact': has_artifact
    }


def analyze_with_time_windows(epoch, srate, egi, window_s=3.0, step_s=1.0):
    """Analyze epoch with sliding time windows, return best prediction."""
    window_samples = int(window_s * srate)
    step_samples = int(step_s * srate)

    results = []
    for start in range(0, epoch.shape[1] - window_samples, step_samples):
        window = epoch[:, start:start + window_samples]
        r = analyze_epoch_improved(window, srate, egi)
        if not r['has_artifact']:
            results.append(r)

    if not results:
        return analyze_epoch_improved(epoch, srate, egi)

    # Return result with median FAI (robust to outliers)
    fais = [r['fai'] for r in results]
    median_idx = np.argsort(fais)[len(fais)//2]
    return results[median_idx]


def main():
    print("╔" + "═" * 66 + "╗")
    print("║" + " IMPROVED DENS Validation with Multi-Channel Analysis".center(66) + "║")
    print("╚" + "═" * 66 + "╝")
    print()

    base = Path("data/ds003751")
    subj = "sub-mit003"

    # Load data
    data, srate = load_binary_eeg(base / f"{subj}_eeg.bin")
    print(f"✓ Loaded {data.shape[0]} ch × {data.shape[1]} samples @ {srate} Hz")

    # Parse events and ratings
    events = parse_events(base / subj / "eeg" / f"{subj}_task-emotion_events.tsv")
    ratings = parse_ratings(base / subj / "beh" / f"{subj}_task-Emotion_beh.tsv")
    print(f"✓ {len(events)} stimuli, {len(ratings)} ratings")
    print()

    # Analysis parameters
    pre_s, post_s = 2.0, 15.0
    baseline_s = 2.0

    print("═" * 70)
    print(f"{'Stim':<10} {'FAI':>7} {'TBR':>6} {'B/A':>6} │ {'PredV':>6} {'PredA':>6} │ {'TrueV':>6} {'TrueA':>6}")
    print("─" * 70)

    results = []
    for ev in events:
        onset = ev['onset_samples']
        pre, post = int(pre_s * srate), int(post_s * srate)
        base_samp = int(baseline_s * srate)

        if onset - pre < 0 or onset + post > data.shape[1]:
            continue

        epoch = data[:, onset-pre:onset+post]
        # Baseline correction
        baseline = epoch[:, :base_samp].mean(axis=1, keepdims=True)
        epoch = epoch - baseline

        # Analyze with time windows (skip baseline)
        analysis_epoch = epoch[:, base_samp:]
        r = analyze_with_time_windows(analysis_epoch, srate, EGI)

        # Get ground truth
        stim = ev['stimulus']
        gt_v, gt_a = np.nan, np.nan
        for key in ratings:
            if stim in key or key in stim or stim.replace('neutral', 'neutral_') in key:
                gt_v = ratings[key]['valence']
                gt_a = ratings[key]['arousal']
                break

        results.append({
            'stim': stim, **r, 'gt_valence': gt_v, 'gt_arousal': gt_a
        })

        art_flag = "⚠" if r['has_artifact'] else " "
        print(f"{stim:<10} {r['fai']:>+7.3f} {r['tbr']:>6.2f} {r['ba_ratio']:>6.2f} │ "
              f"{r['pred_valence']:>6.2f} {r['pred_arousal']:>6.2f} │ "
              f"{gt_v:>6.1f} {gt_a:>6.1f} {art_flag}")

    print("═" * 70)
    print()

    # Statistics
    pv = np.array([r['pred_valence'] for r in results])
    pa = np.array([r['pred_arousal'] for r in results])
    gv = np.array([r['gt_valence'] for r in results])
    ga = np.array([r['gt_arousal'] for r in results])

    valid = ~np.isnan(gv)

    print("╔" + "═" * 66 + "╗")
    print("║" + " STATISTICAL RESULTS".center(66) + "║")
    print("╚" + "═" * 66 + "╝")
    print()

    if valid.sum() >= 3:
        # Valence
        r_v, p_v = pearsonr(pv[valid], gv[valid])
        rho_v, _ = spearmanr(pv[valid], gv[valid])
        acc_v = ((pv[valid] > 5) == (gv[valid] > 5)).mean() * 100

        print("VALENCE (Multi-Channel FAI)")
        print(f"  Pearson r  = {r_v:+.3f} (p = {p_v:.4f})")
        print(f"  Spearman ρ = {rho_v:+.3f}")
        print(f"  Binary Acc = {acc_v:.1f}%")
        print()

        # Arousal
        r_a, p_a = pearsonr(pa[valid], ga[valid])
        rho_a, _ = spearmanr(pa[valid], ga[valid])
        acc_a = ((pa[valid] > 5) == (ga[valid] > 5)).mean() * 100

        print("AROUSAL (Ensemble: TBR + B/A)")
        print(f"  Pearson r  = {r_a:+.3f} (p = {p_a:.4f})")
        print(f"  Spearman ρ = {rho_a:+.3f}")
        print(f"  Binary Acc = {acc_a:.1f}%")
        print()

        # Also test individual arousal metrics
        pa_tbr = np.array([r['pred_arousal_tbr'] for r in results])
        pa_ba = np.array([r['pred_arousal_ba'] for r in results])

        r_tbr, _ = pearsonr(pa_tbr[valid], ga[valid])
        r_ba, _ = pearsonr(pa_ba[valid], ga[valid])
        print("AROUSAL (Individual Metrics)")
        print(f"  TBR method: r = {r_tbr:+.3f}")
        print(f"  B/A method: r = {r_ba:+.3f}")
        print()

        # Quadrant
        pred_q = (pv[valid] > 5).astype(int) * 2 + (pa[valid] > 5).astype(int)
        true_q = (gv[valid] > 5).astype(int) * 2 + (ga[valid] > 5).astype(int)
        quad_acc = (pred_q == true_q).mean() * 100

        print("QUADRANT (4-way)")
        print(f"  Accuracy = {quad_acc:.1f}% (chance = 25%)")
        print()

        # Summary
        print("═" * 68)
        sig_v = "✓ p<0.05" if p_v < 0.05 else "p≥0.05"
        sig_a = "✓ p<0.05" if p_a < 0.05 else "p≥0.05"
        print(f"SUMMARY: Valence r={r_v:+.3f} ({sig_v}), Arousal r={r_a:+.3f} ({sig_a})")
        print(f"         Binary: Valence {acc_v:.0f}%, Arousal {acc_a:.0f}%, Quadrant {quad_acc:.0f}%")

        # Improvement over baseline
        print()
        print("Comparison to Baseline (single-channel, no windowing):")
        print(f"  Valence: {r_v:+.3f} vs +0.269 (baseline)")
        print(f"  Arousal: {r_a:+.3f} vs -0.389 (baseline)")

        improved = (r_v > 0.269) or (r_a > -0.389)
        if improved:
            print()
            print("╔" + "═" * 66 + "╗")
            print("║" + " ✅ IMPROVEMENT ACHIEVED".center(66) + "║")
            print("╚" + "═" * 66 + "╝")


if __name__ == '__main__':
    main()
