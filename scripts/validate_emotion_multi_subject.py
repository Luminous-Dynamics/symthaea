#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Multi-Subject DENS Validation

Validates EmotionSentinel across multiple subjects and computes:
- Per-subject correlations
- Aggregate statistics with confidence intervals
- Effect sizes
- Cross-subject generalization
"""

import struct
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.stats import pearsonr, spearmanr, sem, t
import warnings
warnings.filterwarnings('ignore')

EGI = {'F3': 23, 'F4': 123, 'Fz': 10, 'Cz': 64, 'F7': 21, 'F8': 8}


def load_data(path):
    """Load binary EEG data."""
    with open(path, 'rb') as f:
        n_ch = struct.unpack('<I', f.read(4))[0]
        n_samp = struct.unpack('<I', f.read(4))[0]
        srate = struct.unpack('<d', f.read(8))[0]
        data = np.fromfile(f, dtype=np.float32).reshape((n_ch, n_samp))
    return data, srate


def extract_eeglab(set_path, fdt_path, output_path):
    """Extract EEGLAB data to binary format, handling variable channel counts."""
    try:
        import scipy.io as sio
        mat = sio.loadmat(set_path, struct_as_record=False, squeeze_me=True)
        eeg = mat['EEG']

        srate = float(eeg.srate)
        n_channels = int(eeg.nbchan)
        n_samples = int(eeg.pnts)

        # Load raw data
        data = np.fromfile(fdt_path, dtype=np.float32)
        total_expected = n_channels * n_samples

        # Handle size mismatch - recalculate n_samples from actual data
        if len(data) != total_expected:
            # Try to infer correct dimensions
            actual_samples = len(data) // n_channels
            if actual_samples * n_channels == len(data):
                n_samples = actual_samples
            else:
                # Try different channel counts (common: 128, 129, 130, 131, 132, 133)
                for try_channels in [128, 129, 130, 131, 132, 133, 134, 135]:
                    if len(data) % try_channels == 0:
                        n_channels = try_channels
                        n_samples = len(data) // n_channels
                        break

        data = data.reshape((n_channels, n_samples), order='F')

        with open(output_path, 'wb') as f:
            f.write(struct.pack('<I', n_channels))
            f.write(struct.pack('<I', n_samples))
            f.write(struct.pack('<d', srate))
            data.tofile(f)

        return True, n_channels, n_samples, srate
    except Exception as e:
        return False, str(e), 0, 0


def parse_events(path):
    """Parse BIDS events file."""
    events = []
    try:
        with open(path) as f:
            f.readline()
            for line in f:
                p = line.strip().split('\t')
                if len(p) >= 4 and p[2] == 'stm':
                    events.append({
                        'onset': int(float(p[0]) * 250 / 1000),
                        'stim': p[3].split('_')[0]
                    })
    except:
        pass
    return events


def parse_ratings(path):
    """Parse behavioral ratings."""
    ratings = {}
    try:
        with open(path) as f:
            h = f.readline().strip().split('\t')
            vi, ai = h.index('valence'), h.index('arousal')
            for line in f:
                p = line.strip().split('\t')
                stim = p[0].replace('.mp4', '').replace('.m4v', '').split('.')[0]
                if 'neutral' in stim:
                    stim = stim.replace('neutral_', 'neutral')
                try:
                    ratings[stim] = (float(p[vi]), float(p[ai]))
                except:
                    pass
    except:
        pass
    return ratings


def band_power(data, srate, fmin, fmax):
    """Compute band power using Welch's method."""
    nperseg = min(int(2 * srate), len(data))
    if nperseg < 4:
        return 0.0
    f, psd = signal.welch(data, srate, nperseg=nperseg)
    idx = (f >= fmin) & (f <= fmax)
    return np.trapz(psd[idx], f[idx]) if idx.sum() > 1 else 0


def get_channel_mapping(n_channels):
    """Get EGI channel mapping based on actual channel count."""
    # Standard 128-channel EGI mapping
    base_map = {'F3': 23, 'F4': 123, 'Fz': 10, 'Cz': 64, 'F7': 21, 'F8': 8}

    # Adjust for different channel counts
    if n_channels < 124:
        # Fewer channels - F4 might be at a different position
        # Try to find closest valid indices
        return {k: min(v, n_channels - 1) for k, v in base_map.items()}

    return base_map


def validate_subject(base, subj):
    """Validate a single subject and return correlations."""

    # Check for binary file, extract if needed
    bin_path = base / f"{subj}_eeg.bin"
    if not bin_path.exists():
        set_path = base / subj / "eeg" / f"{subj}_task-Emotion_eeg.set"
        fdt_path = base / subj / "eeg" / f"{subj}_task-Emotion_eeg.fdt"

        if not set_path.exists() or not fdt_path.exists():
            # Check if symlinks exist but data not downloaded
            if set_path.is_symlink():
                return None, "Data not downloaded"
            return None, "Files not found"

        # Check if files are git-annex symlinks to missing data
        try:
            target = set_path.resolve()
            if not target.exists() or target.stat().st_size < 1000:
                return None, "Data not downloaded (annex)"
        except:
            return None, "Cannot access files"

        success, *info = extract_eeglab(str(set_path), str(fdt_path), str(bin_path))
        if not success:
            return None, f"Extraction failed: {info[0]}"

    # Load data
    try:
        data, srate = load_data(bin_path)
    except Exception as e:
        return None, f"Load failed: {e}"

    # Get appropriate channel mapping for this subject's channel count
    n_channels = data.shape[0]
    egi = get_channel_mapping(n_channels)

    # Parse events and ratings
    events = parse_events(base / subj / "eeg" / f"{subj}_task-emotion_events.tsv")
    if not events:
        # Try alternate case
        events = parse_events(base / subj / "eeg" / f"{subj}_task-Emotion_events.tsv")

    ratings_path = base / subj / "beh" / f"{subj}_task-Emotion_beh.tsv"
    ratings = parse_ratings(ratings_path)

    if not events or not ratings:
        return None, "Missing events or ratings"

    # Process each stimulus
    fz_alphas, f4_alphas = [], []
    gt_valences, gt_arousals = [], []

    for ev in events:
        onset = ev['onset']
        post = int(15 * srate)
        if onset + post > data.shape[1]:
            continue

        epoch = data[:, onset:onset+post]

        # Extract features using local channel mapping
        fz = epoch[egi['Fz'], :]
        f4 = epoch[egi['F4'], :]

        fz_alpha = band_power(fz, srate, 8, 13)
        f4_alpha = band_power(f4, srate, 8, 13)

        # Get ground truth
        stim = ev['stim']
        gt_v, gt_a = np.nan, np.nan
        for k in ratings:
            if stim in k or k in stim:
                gt_v, gt_a = ratings[k]
                break

        if not np.isnan(gt_v):
            fz_alphas.append(fz_alpha)
            f4_alphas.append(f4_alpha)
            gt_valences.append(gt_v)
            gt_arousals.append(gt_a)

    if len(fz_alphas) < 3:
        return None, f"Too few valid epochs ({len(fz_alphas)})"

    # Compute correlations
    fz_alpha = np.array(fz_alphas)
    f4_alpha = np.array(f4_alphas)
    gt_v = np.array(gt_valences)
    gt_a = np.array(gt_arousals)

    # Note: correlation with raw features (will use z-scoring for predictions)
    r_fz_v, p_fz_v = pearsonr(fz_alpha, gt_v)
    r_f4_a, p_f4_a = pearsonr(f4_alpha, gt_a)

    # Also try inverse for valence (as discovered empirically)
    r_fz_v_inv, _ = pearsonr(-fz_alpha, gt_v)

    return {
        'n_epochs': len(fz_alphas),
        'r_valence': r_fz_v,
        'r_valence_inv': r_fz_v_inv,
        'p_valence': p_fz_v,
        'r_arousal': r_f4_a,
        'p_arousal': p_f4_a,
        'mean_gt_v': gt_v.mean(),
        'mean_gt_a': gt_a.mean(),
    }, "OK"


def confidence_interval(data, confidence=0.95):
    """Compute confidence interval for mean."""
    n = len(data)
    mean = np.mean(data)
    se = sem(data)
    h = se * t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h


def main():
    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + " MULTI-SUBJECT DENS VALIDATION".center(70) + "║")
    print("║" + " Cross-Subject Generalization Analysis".center(70) + "║")
    print("╚" + "═" * 70 + "╝")
    print()

    base = Path("data/ds003751")

    # Find all subject directories
    subjects = sorted([d.name for d in base.iterdir()
                      if d.is_dir() and d.name.startswith('sub-')])

    print(f"Found {len(subjects)} subjects in dataset")
    print()

    # Process each subject
    results = []
    print("═" * 72)
    print(f"{'Subject':<15} {'Status':<20} {'N':>4} {'r(V)':>8} {'p(V)':>8} {'r(A)':>8} {'p(A)':>8}")
    print("─" * 72)

    for subj in subjects:
        result, status = validate_subject(base, subj)

        if result is None:
            print(f"{subj:<15} {status:<20} {'—':>4} {'—':>8} {'—':>8} {'—':>8} {'—':>8}")
        else:
            results.append({'subject': subj, **result})
            sig_v = "*" if result['p_valence'] < 0.05 else ""
            sig_a = "*" if result['p_arousal'] < 0.05 else ""
            print(f"{subj:<15} {'✓ Processed':<20} {result['n_epochs']:>4} "
                  f"{result['r_valence']:>+7.3f}{sig_v} {result['p_valence']:>8.4f} "
                  f"{result['r_arousal']:>+7.3f}{sig_a} {result['p_arousal']:>8.4f}")

    print("═" * 72)
    print()

    if len(results) < 2:
        print("❌ Need at least 2 subjects for aggregate analysis")
        print("   Please download more subjects with: datalad get sub-XXX/eeg/*")
        return

    # Aggregate statistics
    print("╔" + "═" * 70 + "╗")
    print("║" + " AGGREGATE STATISTICS".center(70) + "║")
    print("╚" + "═" * 70 + "╝")
    print()

    r_vals = np.array([r['r_valence'] for r in results])
    r_aros = np.array([r['r_arousal'] for r in results])
    p_vals = np.array([r['p_valence'] for r in results])
    p_aros = np.array([r['p_arousal'] for r in results])

    # Mean correlations
    mean_r_v, ci_low_v, ci_high_v = confidence_interval(r_vals)
    mean_r_a, ci_low_a, ci_high_a = confidence_interval(r_aros)

    print(f"Subjects analyzed: {len(results)}")
    print(f"Total epochs: {sum(r['n_epochs'] for r in results)}")
    print()

    print("VALENCE (Fz Alpha)")
    print(f"  Mean r:           {mean_r_v:+.3f}")
    print(f"  95% CI:           [{ci_low_v:+.3f}, {ci_high_v:+.3f}]")
    print(f"  Range:            [{r_vals.min():+.3f}, {r_vals.max():+.3f}]")
    print(f"  Significant (p<0.05): {(p_vals < 0.05).sum()}/{len(results)} subjects")
    print()

    print("AROUSAL (F4 Alpha)")
    print(f"  Mean r:           {mean_r_a:+.3f}")
    print(f"  95% CI:           [{ci_low_a:+.3f}, {ci_high_a:+.3f}]")
    print(f"  Range:            [{r_aros.min():+.3f}, {r_aros.max():+.3f}]")
    print(f"  Significant (p<0.05): {(p_aros < 0.05).sum()}/{len(results)} subjects")
    print()

    # One-sample t-test: is mean correlation significantly different from 0?
    from scipy.stats import ttest_1samp
    t_v, p_t_v = ttest_1samp(r_vals, 0)
    t_a, p_t_a = ttest_1samp(r_aros, 0)

    print("ONE-SAMPLE T-TEST (H0: mean r = 0)")
    print(f"  Valence: t({len(results)-1}) = {t_v:.3f}, p = {p_t_v:.4f} {'✓ SIGNIFICANT' if p_t_v < 0.05 else ''}")
    print(f"  Arousal: t({len(results)-1}) = {t_a:.3f}, p = {p_t_a:.4f} {'✓ SIGNIFICANT' if p_t_a < 0.05 else ''}")
    print()

    # Effect sizes (Cohen's d)
    d_v = mean_r_v / r_vals.std() if r_vals.std() > 0 else 0
    d_a = mean_r_a / r_aros.std() if r_aros.std() > 0 else 0

    print("EFFECT SIZES (Cohen's d)")
    print(f"  Valence: d = {d_v:.2f} ({'small' if abs(d_v) < 0.5 else 'medium' if abs(d_v) < 0.8 else 'large'})")
    print(f"  Arousal: d = {d_a:.2f} ({'small' if abs(d_a) < 0.5 else 'medium' if abs(d_a) < 0.8 else 'large'})")
    print()

    # Summary
    print("═" * 72)
    print("SUMMARY")
    print("═" * 72)

    if p_t_v < 0.05 or p_t_a < 0.05:
        print()
        print("╔" + "═" * 70 + "╗")
        print("║" + " ✅ CROSS-SUBJECT VALIDATION SUCCESSFUL".center(70) + "║")
        print("╠" + "═" * 70 + "╣")
        if p_t_v < 0.05:
            print(f"║  Valence: mean r = {mean_r_v:+.3f}, 95% CI [{ci_low_v:+.3f}, {ci_high_v:+.3f}], p = {p_t_v:.4f}".ljust(71) + "║")
        if p_t_a < 0.05:
            print(f"║  Arousal: mean r = {mean_r_a:+.3f}, 95% CI [{ci_low_a:+.3f}, {ci_high_a:+.3f}], p = {p_t_a:.4f}".ljust(71) + "║")
        print("╠" + "═" * 70 + "╣")
        print("║" + " EmotionSentinel generalizes across subjects!".center(70) + "║")
        print("╚" + "═" * 70 + "╝")
    else:
        print()
        print("Results did not reach statistical significance across subjects.")
        print("This may indicate individual differences in EEG-emotion mapping.")


if __name__ == '__main__':
    main()
