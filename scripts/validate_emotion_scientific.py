#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Scientific Validation of EmotionSentinel on DENS Dataset

This script performs rigorous validation with:
- Proper FFT-based spectral analysis
- Stimulus-locked epoch extraction
- Baseline correction
- Correlation with ground truth
- Statistical significance testing

Dataset: DENS (ds003751) from OpenNeuro
"""

import json
import struct
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# EGI 128 to 10-20 channel mapping (0-indexed)
EGI_MAPPING = {
    'F3': 23,    # E24
    'F4': 123,   # E124
    'F7': 21,    # E22
    'F8': 8,     # E9
    'Fz': 10,    # E11
    'Fp1': 17,   # E18
    'Fp2': 14,   # E15
    'C3': 35,    # E36
    'C4': 103,   # E104
    'Cz': 64,    # E65
    'P3': 51,    # E52
    'P4': 91,    # E92
    'Pz': 61,    # E62
    'O1': 69,    # E70
    'O2': 82,    # E83
}


def load_binary_eeg(path):
    """Load EEG data from our simple binary format."""
    with open(path, 'rb') as f:
        n_channels = struct.unpack('<I', f.read(4))[0]
        n_samples = struct.unpack('<I', f.read(4))[0]
        srate = struct.unpack('<d', f.read(8))[0]
        data = np.fromfile(f, dtype=np.float32)
        data = data.reshape((n_channels, n_samples))
    return data, srate, n_channels, n_samples


def parse_events(events_path):
    """Parse BIDS events file for stimulus onsets."""
    events = []
    with open(events_path) as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                onset_ms = float(parts[0])
                trial_type = parts[2]
                label = parts[3]
                if trial_type == 'stm':
                    # Parse stimulus name (e.g., "12_2" -> "12.m4v" or "12.mp4")
                    stim_name = label.split('_')[0]
                    trial_num = int(label.split('_')[1]) if '_' in label else 0
                    events.append({
                        'onset_ms': onset_ms,
                        'onset_samples': int(onset_ms * 250 / 1000),  # 250 Hz
                        'stimulus': stim_name,
                        'trial': trial_num,
                        'label': label
                    })
    return events


def parse_behavioral(beh_path):
    """Parse behavioral ratings file."""
    ratings = {}
    with open(beh_path) as f:
        header = f.readline().strip().split('\t')
        val_idx = header.index('valence')
        aro_idx = header.index('arousal')

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > max(val_idx, aro_idx):
                stim_name = parts[0].replace('.mp4', '').replace('.m4v', '')
                # Handle neutral stimuli
                if 'neutral' in stim_name:
                    stim_name = stim_name.split('_')[0] + '_' + stim_name.split('_')[1]
                else:
                    stim_name = stim_name.split('.')[0]

                try:
                    valence = float(parts[val_idx])
                    arousal = float(parts[aro_idx])
                    ratings[stim_name] = {'valence': valence, 'arousal': arousal}
                except:
                    pass
    return ratings


def compute_band_power(data, srate, fmin, fmax):
    """
    Compute band power using Welch's method (proper FFT).

    Parameters:
    - data: 1D array of EEG samples
    - srate: sampling rate in Hz
    - fmin, fmax: frequency band limits

    Returns:
    - Band power in µV²
    """
    # Welch's method with 2-second windows, 50% overlap
    nperseg = int(2 * srate)  # 2-second windows
    freqs, psd = signal.welch(data, srate, nperseg=nperseg, noverlap=nperseg//2)

    # Find frequency band indices
    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)

    # Integrate power in band (trapezoidal rule)
    band_power = np.trapz(psd[idx_band], freqs[idx_band])

    return band_power


def compute_fai(f3_alpha, f4_alpha):
    """
    Compute Frontal Asymmetry Index.
    FAI = ln(F4_alpha) - ln(F3_alpha)

    Positive FAI = approach motivation (positive valence)
    Negative FAI = withdrawal motivation (negative valence)
    """
    if f3_alpha <= 0 or f4_alpha <= 0:
        return 0.0
    return np.log(f4_alpha) - np.log(f3_alpha)


def compute_arousal_index(beta, alpha):
    """
    Compute arousal from beta/alpha ratio.
    Higher ratio = higher arousal.
    """
    if alpha <= 0:
        return 0.0
    return beta / alpha


def extract_epoch(data, onset_sample, pre_samples, post_samples):
    """Extract an epoch around a stimulus onset."""
    start = max(0, onset_sample - pre_samples)
    end = min(data.shape[1], onset_sample + post_samples)
    return data[:, start:end]


def baseline_correct(epoch_data, baseline_samples):
    """Subtract baseline (first N samples) from epoch."""
    baseline = epoch_data[:, :baseline_samples].mean(axis=1, keepdims=True)
    return epoch_data - baseline


def analyze_epoch(epoch_data, srate, mapping):
    """
    Analyze a single epoch for valence and arousal.

    Returns:
    - fai: Frontal Asymmetry Index (valence indicator)
    - arousal_ratio: Beta/Alpha ratio (arousal indicator)
    - predicted_valence: Normalized to 1-9 scale
    - predicted_arousal: Normalized to 1-9 scale
    """
    # Get channel data
    f3_data = epoch_data[mapping['F3'], :]
    f4_data = epoch_data[mapping['F4'], :]
    fz_data = epoch_data[mapping['Fz'], :]

    # Compute alpha power (8-13 Hz)
    f3_alpha = compute_band_power(f3_data, srate, 8, 13)
    f4_alpha = compute_band_power(f4_data, srate, 8, 13)
    fz_alpha = compute_band_power(fz_data, srate, 8, 13)

    # Compute beta power (13-30 Hz)
    fz_beta = compute_band_power(fz_data, srate, 13, 30)

    # Compute indices
    fai = compute_fai(f3_alpha, f4_alpha)
    arousal_ratio = compute_arousal_index(fz_beta, fz_alpha)

    # Convert to 1-9 scale for comparison with ground truth
    # FAI typically ranges from -0.5 to +0.5
    # Map to 1-9: FAI=0 -> 5, FAI=-0.5 -> 1, FAI=+0.5 -> 9
    predicted_valence = 5 + fai * 8  # Scale factor of 8
    predicted_valence = np.clip(predicted_valence, 1, 9)

    # Arousal ratio typically 0.5-2.0 during tasks
    # Map to 1-9: ratio=0.5 -> 1, ratio=1.25 -> 5, ratio=2.0 -> 9
    predicted_arousal = 1 + (arousal_ratio - 0.5) * (8 / 1.5)
    predicted_arousal = np.clip(predicted_arousal, 1, 9)

    return {
        'fai': fai,
        'f3_alpha': f3_alpha,
        'f4_alpha': f4_alpha,
        'arousal_ratio': arousal_ratio,
        'fz_alpha': fz_alpha,
        'fz_beta': fz_beta,
        'predicted_valence': predicted_valence,
        'predicted_arousal': predicted_arousal
    }


def main():
    print("╔" + "═" * 66 + "╗")
    print("║" + " " * 10 + "DENS Scientific Validation" + " " * 30 + "║")
    print("║" + " " * 10 + "Frontal Asymmetry & Arousal Analysis" + " " * 19 + "║")
    print("╚" + "═" * 66 + "╝")
    print()

    # Paths
    base_path = Path("data/ds003751")
    subject = "sub-mit003"

    # Check for full data file
    full_data_path = base_path / f"{subject}_eeg.bin"
    if not full_data_path.exists():
        print(f"⚠ Full data file not found: {full_data_path}")
        print("  Please run: python scripts/extract_eeglab.py \\")
        print(f"    {base_path}/{subject}/eeg/{subject}_task-Emotion_eeg.set \\")
        print(f"    {base_path}/{subject}_eeg")
        return

    # Load data
    print(f"Loading EEG data from {full_data_path}...")
    data, srate, n_channels, n_samples = load_binary_eeg(full_data_path)
    duration_s = n_samples / srate
    print(f"✓ Loaded {n_channels} channels × {n_samples} samples @ {srate} Hz")
    print(f"  Duration: {duration_s:.1f} seconds ({duration_s/60:.1f} minutes)")
    print()

    # Parse events
    events_path = base_path / subject / "eeg" / f"{subject}_task-emotion_events.tsv"
    events = parse_events(events_path)
    print(f"✓ Parsed {len(events)} stimulus events")

    # Parse behavioral ratings (ground truth)
    beh_path = base_path / subject / "beh" / f"{subject}_task-Emotion_beh.tsv"
    if not beh_path.exists():
        # Try alternate location
        beh_path = Path(f"data/{subject}/beh/{subject}_task-Emotion_beh.tsv")

    ratings = parse_behavioral(beh_path)
    print(f"✓ Parsed {len(ratings)} behavioral ratings")
    print()

    # Analysis parameters
    pre_stim_s = 2.0    # 2 seconds before stimulus
    post_stim_s = 15.0  # 15 seconds after (typical video duration)
    baseline_s = 1.0    # 1 second baseline

    pre_samples = int(pre_stim_s * srate)
    post_samples = int(post_stim_s * srate)
    baseline_samples = int(baseline_s * srate)

    print("═" * 68)
    print("EPOCH ANALYSIS")
    print("═" * 68)
    print(f"{'Stimulus':<15} {'FAI':>8} {'β/α':>8} {'Pred V':>8} {'Pred A':>8} │ {'True V':>7} {'True A':>7}")
    print("─" * 68)

    results = []

    for event in events:
        stim = event['stimulus']
        onset = event['onset_samples']

        # Skip if epoch would go out of bounds
        if onset - pre_samples < 0 or onset + post_samples > n_samples:
            continue

        # Extract and baseline-correct epoch
        epoch = extract_epoch(data, onset, pre_samples, post_samples)
        epoch = baseline_correct(epoch, baseline_samples)

        # Analyze epoch (skip baseline period for main analysis)
        analysis_epoch = epoch[:, baseline_samples:]
        analysis = analyze_epoch(analysis_epoch, srate, EGI_MAPPING)

        # Get ground truth
        # Match stimulus name to behavioral ratings
        gt_key = None
        for key in ratings:
            if stim in key or key in stim:
                gt_key = key
                break

        if gt_key:
            gt = ratings[gt_key]
            gt_valence = gt['valence']
            gt_arousal = gt['arousal']
        else:
            gt_valence = np.nan
            gt_arousal = np.nan

        results.append({
            'stimulus': stim,
            'trial': event['trial'],
            **analysis,
            'gt_valence': gt_valence,
            'gt_arousal': gt_arousal
        })

        # Print row
        gt_v_str = f"{gt_valence:.1f}" if not np.isnan(gt_valence) else "n/a"
        gt_a_str = f"{gt_arousal:.1f}" if not np.isnan(gt_arousal) else "n/a"
        print(f"{stim:<15} {analysis['fai']:>+8.4f} {analysis['arousal_ratio']:>8.3f} "
              f"{analysis['predicted_valence']:>8.2f} {analysis['predicted_arousal']:>8.2f} │ "
              f"{gt_v_str:>7} {gt_a_str:>7}")

    print("═" * 68)
    print()

    # Compute correlations
    if results:
        pred_valence = np.array([r['predicted_valence'] for r in results])
        pred_arousal = np.array([r['predicted_arousal'] for r in results])
        gt_valence = np.array([r['gt_valence'] for r in results])
        gt_arousal = np.array([r['gt_arousal'] for r in results])

        # Filter out NaN values
        valid_v = ~np.isnan(gt_valence)
        valid_a = ~np.isnan(gt_arousal)

        print("╔" + "═" * 66 + "╗")
        print("║" + " " * 20 + "STATISTICAL RESULTS" + " " * 27 + "║")
        print("╚" + "═" * 66 + "╝")
        print()

        if valid_v.sum() >= 3:
            # Pearson correlation
            r_val, p_val = pearsonr(pred_valence[valid_v], gt_valence[valid_v])
            rho_val, p_rho = spearmanr(pred_valence[valid_v], gt_valence[valid_v])

            print(f"VALENCE (FAI → Subjective Rating)")
            print(f"  Pearson r  = {r_val:+.3f} (p = {p_val:.4f})")
            print(f"  Spearman ρ = {rho_val:+.3f} (p = {p_rho:.4f})")

            # Classification accuracy (positive vs negative)
            pred_pos = pred_valence[valid_v] > 5
            gt_pos = gt_valence[valid_v] > 5
            valence_acc = (pred_pos == gt_pos).mean() * 100
            print(f"  Binary accuracy (pos/neg): {valence_acc:.1f}%")
            print()

        if valid_a.sum() >= 3:
            r_aro, p_aro = pearsonr(pred_arousal[valid_a], gt_arousal[valid_a])
            rho_aro, p_rho_a = spearmanr(pred_arousal[valid_a], gt_arousal[valid_a])

            print(f"AROUSAL (β/α → Subjective Rating)")
            print(f"  Pearson r  = {r_aro:+.3f} (p = {p_aro:.4f})")
            print(f"  Spearman ρ = {rho_aro:+.3f} (p = {p_rho_a:.4f})")

            # Classification accuracy (high vs low)
            pred_high = pred_arousal[valid_a] > 5
            gt_high = gt_arousal[valid_a] > 5
            arousal_acc = (pred_high == gt_high).mean() * 100
            print(f"  Binary accuracy (high/low): {arousal_acc:.1f}%")
            print()

        # Overall quadrant accuracy
        if valid_v.sum() >= 3 and valid_a.sum() >= 3:
            pred_quad = (pred_valence[valid_v] > 5).astype(int) * 2 + (pred_arousal[valid_a] > 5).astype(int)
            gt_quad = (gt_valence[valid_v] > 5).astype(int) * 2 + (gt_arousal[valid_a] > 5).astype(int)
            quad_acc = (pred_quad == gt_quad).mean() * 100

            print(f"QUADRANT ACCURACY (HVHA/HVLA/LVHA/LVLA)")
            print(f"  4-way classification: {quad_acc:.1f}%")
            print(f"  Chance level: 25%")
            print()

        # Summary
        print("═" * 68)
        print("SUMMARY")
        print("═" * 68)
        print(f"  Epochs analyzed: {len(results)}")
        print(f"  With ground truth: {valid_v.sum()}")

        if valid_v.sum() >= 3:
            sig_val = "✓ significant" if p_val < 0.05 else "✗ not significant"
            sig_aro = "✓ significant" if p_aro < 0.05 else "✗ not significant"
            print(f"  Valence correlation: r={r_val:+.3f} ({sig_val})")
            print(f"  Arousal correlation: r={r_aro:+.3f} ({sig_aro})")
            print(f"  Valence binary acc:  {valence_acc:.1f}%")
            print(f"  Arousal binary acc:  {arousal_acc:.1f}%")

        print()
        print("╔" + "═" * 66 + "╗")
        if valid_v.sum() >= 3 and (p_val < 0.05 or p_aro < 0.05):
            print("║" + " " * 15 + "✅ VALIDATION SUCCESSFUL" + " " * 26 + "║")
        else:
            print("║" + " " * 15 + "📊 VALIDATION COMPLETE" + " " * 28 + "║")
        print("╚" + "═" * 66 + "╝")


if __name__ == '__main__':
    main()
