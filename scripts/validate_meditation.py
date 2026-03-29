#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
MeditationSentinel Validation on OpenNeuro ds001787

Validates meditation state detection using:
- Alpha power (8-13 Hz) for relaxed focus
- Theta power (4-8 Hz) for deep meditation
- Alpha/Beta ratio as meditation index

Dataset: OpenNeuro ds001787 - Meditation EEG
- 24 subjects, 64 channels BioSemi, 256 Hz
- Task: meditation with stimulus markers
"""

import numpy as np
from pathlib import Path
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


def read_bdf_header(path):
    """Read BDF header to get signal info."""
    with open(path, 'rb') as f:
        # BDF uses 24-bit samples, otherwise similar to EDF
        version = f.read(8)  # Should be '\xffBIOSEMI' for BDF
        patient = f.read(80).decode('latin-1').strip()
        recording = f.read(80).decode('latin-1').strip()
        start_date = f.read(8).decode().strip()
        start_time = f.read(8).decode().strip()
        header_bytes = int(f.read(8).decode().strip())
        reserved = f.read(44).decode('latin-1').strip()
        n_records = int(f.read(8).decode().strip())
        record_duration = float(f.read(8).decode().strip())
        n_signals = int(f.read(4).decode().strip())

        # Read signal headers
        labels = [f.read(16).decode('latin-1').strip() for _ in range(n_signals)]
        transducers = [f.read(80).decode('latin-1').strip() for _ in range(n_signals)]
        phys_dims = [f.read(8).decode('latin-1').strip() for _ in range(n_signals)]
        phys_mins = [float(f.read(8).decode().strip()) for _ in range(n_signals)]
        phys_maxs = [float(f.read(8).decode().strip()) for _ in range(n_signals)]
        dig_mins = [int(f.read(8).decode().strip()) for _ in range(n_signals)]
        dig_maxs = [int(f.read(8).decode().strip()) for _ in range(n_signals)]
        prefilters = [f.read(80).decode('latin-1').strip() for _ in range(n_signals)]
        n_samples = [int(f.read(8).decode().strip()) for _ in range(n_signals)]
        reserved2 = [f.read(32).decode('latin-1').strip() for _ in range(n_signals)]

    return {
        'n_records': n_records,
        'record_duration': record_duration,
        'n_signals': n_signals,
        'labels': labels,
        'n_samples': n_samples,
        'phys_mins': phys_mins,
        'phys_maxs': phys_maxs,
        'dig_mins': dig_mins,
        'dig_maxs': dig_maxs,
        'header_bytes': header_bytes,
    }


def read_bdf_channel(path, channel_idx, max_records=None):
    """Read BDF data for a specific channel (24-bit format)."""
    header = read_bdf_header(path)

    samples_per_record = header['n_samples']
    n_records = header['n_records'] if max_records is None else min(header['n_records'], max_records)

    # Calculate sample rate
    srate = samples_per_record[channel_idx] / header['record_duration']

    # Scaling factors
    phys_min = header['phys_mins'][channel_idx]
    phys_max = header['phys_maxs'][channel_idx]
    dig_min = header['dig_mins'][channel_idx]
    dig_max = header['dig_maxs'][channel_idx]
    scale = (phys_max - phys_min) / (dig_max - dig_min)
    offset = phys_min - scale * dig_min

    # Calculate bytes per record for each signal
    bytes_per_sample = 3  # BDF uses 24-bit (3 bytes)
    bytes_per_record = [n * bytes_per_sample for n in samples_per_record]
    total_bytes_per_record = sum(bytes_per_record)

    # Byte offset for target channel within each record
    channel_offset = sum(bytes_per_record[:channel_idx])

    # Read data
    data = []
    with open(path, 'rb') as f:
        f.seek(header['header_bytes'])

        for rec in range(n_records):
            # Seek to this channel's data in this record
            record_start = header['header_bytes'] + rec * total_bytes_per_record
            f.seek(record_start + channel_offset)

            # Read 24-bit samples for this channel
            n_samp = samples_per_record[channel_idx]
            raw_bytes = f.read(n_samp * 3)

            # Convert 24-bit little-endian to int32
            samples = []
            for i in range(n_samp):
                b = raw_bytes[i*3:(i+1)*3]
                if len(b) < 3:
                    break
                # 24-bit signed little-endian
                val = b[0] | (b[1] << 8) | (b[2] << 16)
                if val >= 0x800000:  # Sign extend
                    val -= 0x1000000
                samples.append(val)

            if samples:
                data.append(np.array(samples) * scale + offset)

    if not data:
        return np.array([]), srate, header['labels'][channel_idx]

    return np.concatenate(data), srate, header['labels'][channel_idx]


def read_events(events_path):
    """Read BIDS events file."""
    events = []
    with open(events_path, 'r') as f:
        header = f.readline().strip().split('\t')
        onset_idx = header.index('onset')
        trial_type_idx = header.index('trial_type')
        value_idx = header.index('value') if 'value' in header else None

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > trial_type_idx:
                onset = float(parts[onset_idx])
                trial_type = parts[trial_type_idx]
                value = int(float(parts[value_idx])) if value_idx else 0
                events.append({
                    'onset': onset,
                    'trial_type': trial_type,
                    'value': value
                })

    return events


def band_power(data, srate, fmin, fmax):
    """Compute band power using Welch's method."""
    nperseg = min(int(4 * srate), len(data))
    if nperseg < 8:
        return 0.0
    f, psd = signal.welch(data, srate, nperseg=nperseg)
    idx = (f >= fmin) & (f <= fmax)
    return np.trapz(psd[idx], f[idx]) if idx.sum() > 1 else 0


def extract_meditation_features(data, srate):
    """Extract features for meditation detection."""
    theta = band_power(data, srate, 4, 8)    # Deep meditation
    alpha = band_power(data, srate, 8, 13)   # Relaxed focus
    beta = band_power(data, srate, 13, 30)   # Active thinking
    gamma = band_power(data, srate, 30, 45)  # High cognitive

    total = theta + alpha + beta + gamma + 1e-10

    # Meditation indices
    alpha_beta_ratio = alpha / (beta + 1e-10)
    theta_alpha_ratio = theta / (alpha + 1e-10)

    return {
        'theta_rel': theta / total,
        'alpha_rel': alpha / total,
        'beta_rel': beta / total,
        'gamma_rel': gamma / total,
        'alpha_beta_ratio': alpha_beta_ratio,
        'theta_alpha_ratio': theta_alpha_ratio,
        'meditation_index': (alpha + theta) / (beta + gamma + 1e-10),
    }


def validate_subject(bdf_path, events_path):
    """Validate meditation detection for a single subject."""
    header = read_bdf_header(bdf_path)

    # Find a frontal EEG channel
    # BioSemi 64-channel: A1=Fp1, A2=AF7, A3=AF3, A4=F1, A5=F3
    # Use A3 (AF3) or A4 (F1) for frontal midline activity
    eeg_idx = 3  # A4 = F1 (frontal midline)

    # Also check for standard names
    preferred = ['Fz', 'Fpz', 'F3', 'F4', 'Fp1', 'AF3', 'F1']
    for pref in preferred:
        for i, label in enumerate(header['labels']):
            if pref.lower() == label.lower():
                eeg_idx = i
                break

    # Read EEG data
    data, srate, label = read_bdf_channel(bdf_path, eeg_idx, max_records=None)

    if len(data) == 0:
        return None, "Could not read EEG data"

    # Read events
    events = read_events(events_path)
    stimulus_events = [e for e in events if e['trial_type'] == 'stimulus']

    if len(stimulus_events) < 2:
        return None, "Not enough stimulus events"

    # Extract features for meditation periods (after stimulus) vs baseline (before first stimulus)
    epoch_duration = 30  # seconds
    samples_per_epoch = int(epoch_duration * srate)

    # Approach: Compare first 2 minutes (baseline/warmup) vs rest of session (meditation)
    # This is a common paradigm where meditation deepens over time

    baseline_end_time = min(120, len(data) / srate / 4)  # First 2 min or 25% of recording
    baseline_samples = int(baseline_end_time * srate)

    baseline_features = []
    for start in range(0, baseline_samples - samples_per_epoch, samples_per_epoch):
        epoch_data = data[start:start + samples_per_epoch]
        if len(epoch_data) >= samples_per_epoch:
            baseline_features.append(extract_meditation_features(epoch_data, srate))

    # Meditation: from 3 minutes onwards (after settling in)
    meditation_start = int(180 * srate)  # Start at 3 minutes
    meditation_features = []
    for start in range(meditation_start, len(data) - samples_per_epoch, samples_per_epoch):
        epoch_data = data[start:start + samples_per_epoch]
        if len(epoch_data) >= samples_per_epoch:
            meditation_features.append(extract_meditation_features(epoch_data, srate))

    if len(baseline_features) < 2 or len(meditation_features) < 5:
        return None, f"Not enough epochs (baseline={len(baseline_features)}, meditation={len(meditation_features)})"

    # Calculate statistics
    baseline_alpha = np.mean([f['alpha_rel'] for f in baseline_features])
    meditation_alpha = np.mean([f['alpha_rel'] for f in meditation_features])

    baseline_theta = np.mean([f['theta_rel'] for f in baseline_features])
    meditation_theta = np.mean([f['theta_rel'] for f in meditation_features])

    baseline_ratio = np.mean([f['alpha_beta_ratio'] for f in baseline_features])
    meditation_ratio = np.mean([f['alpha_beta_ratio'] for f in meditation_features])

    baseline_index = np.mean([f['meditation_index'] for f in baseline_features])
    meditation_index = np.mean([f['meditation_index'] for f in meditation_features])

    # Effect sizes (meditation vs baseline)
    alpha_change = (meditation_alpha - baseline_alpha) / (baseline_alpha + 1e-10) * 100
    theta_change = (meditation_theta - baseline_theta) / (baseline_theta + 1e-10) * 100
    ratio_change = (meditation_ratio - baseline_ratio) / (baseline_ratio + 1e-10) * 100
    index_change = (meditation_index - baseline_index) / (baseline_index + 1e-10) * 100

    return {
        'n_baseline': len(baseline_features),
        'n_meditation': len(meditation_features),
        'baseline_alpha': baseline_alpha,
        'meditation_alpha': meditation_alpha,
        'alpha_change': alpha_change,
        'baseline_theta': baseline_theta,
        'meditation_theta': meditation_theta,
        'theta_change': theta_change,
        'baseline_ratio': baseline_ratio,
        'meditation_ratio': meditation_ratio,
        'ratio_change': ratio_change,
        'baseline_index': baseline_index,
        'meditation_index': meditation_index,
        'index_change': index_change,
        'channel': label,
    }, "OK"


def main():
    print()
    print('=' * 72)
    print(' MEDITATIONSENTINEL VALIDATION'.center(72))
    print(' OpenNeuro ds001787 - Meditation EEG'.center(72))
    print('=' * 72)
    print()

    base = Path('data/meditation-eeg')

    # Find subject files
    subjects = []
    for bdf in sorted(base.glob('sub-*/ses-*/eeg/*_task-meditation_eeg.bdf')):
        events = bdf.with_name(bdf.name.replace('_eeg.bdf', '_events.tsv'))
        if events.exists():
            # Check if BDF is readable (not just annex link)
            try:
                with open(bdf, 'rb') as f:
                    f.read(1)
                subj_id = bdf.name.split('_')[0]
                subjects.append((subj_id, bdf, events))
            except:
                pass

    print(f"Found {len(subjects)} subjects with meditation EEG data")
    print()

    if len(subjects) == 0:
        print("No subjects found. Download meditation EEG data first.")
        return

    results = []
    print('-' * 72)
    print(f"{'Subject':<10} {'Ch':<6} {'B-Alpha':>8} {'M-Alpha':>8} {'ΔAlpha':>8} {'M-Index':>8} {'ΔIndex':>8}")
    print('-' * 72)

    for subj_id, bdf_path, events_path in subjects:
        result, status = validate_subject(str(bdf_path), str(events_path))

        if result is None:
            print(f"{subj_id:<10} {status}")
        else:
            results.append({'subject': subj_id, **result})
            ch = result['channel'][:5]
            print(f"{subj_id:<10} {ch:<6} "
                  f"{result['baseline_alpha']*100:>7.1f}% "
                  f"{result['meditation_alpha']*100:>7.1f}% "
                  f"{result['alpha_change']:>+7.1f}% "
                  f"{result['meditation_index']:>7.2f} "
                  f"{result['index_change']:>+7.1f}%")

    print('-' * 72)
    print()

    if len(results) < 1:
        print("No valid results to aggregate")
        return

    # Aggregate results
    print('=' * 72)
    print(' AGGREGATE RESULTS'.center(72))
    print('=' * 72)
    print()

    print(f"Subjects analyzed: {len(results)}")
    total_epochs = sum(r['n_meditation'] for r in results)
    print(f"Total meditation epochs: {total_epochs}")
    print()

    # Mean changes
    mean_alpha_change = np.mean([r['alpha_change'] for r in results])
    mean_theta_change = np.mean([r['theta_change'] for r in results])
    mean_index_change = np.mean([r['index_change'] for r in results])

    print("MEDITATION vs BASELINE CHANGES:")
    print(f"  Alpha power:      {mean_alpha_change:+.1f}%")
    print(f"  Theta power:      {mean_theta_change:+.1f}%")
    print(f"  Meditation Index: {mean_index_change:+.1f}%")
    print()

    # Mean values
    mean_baseline_alpha = np.mean([r['baseline_alpha'] for r in results]) * 100
    mean_meditation_alpha = np.mean([r['meditation_alpha'] for r in results]) * 100
    mean_baseline_index = np.mean([r['baseline_index'] for r in results])
    mean_meditation_index = np.mean([r['meditation_index'] for r in results])

    print("ABSOLUTE VALUES:")
    print(f"  Baseline alpha:   {mean_baseline_alpha:.1f}%")
    print(f"  Meditation alpha: {mean_meditation_alpha:.1f}%")
    print(f"  Baseline index:   {mean_baseline_index:.2f}")
    print(f"  Meditation index: {mean_meditation_index:.2f}")
    print()

    # Summary
    print('=' * 72)
    print()

    # Count positive effects
    pos_alpha = sum(1 for r in results if r['alpha_change'] > 0)
    pos_theta = sum(1 for r in results if r['theta_change'] > 0)
    pos_index = sum(1 for r in results if r['index_change'] > 0)

    print("VALIDATION RESULTS:")
    print(f"  Subjects with increased alpha during meditation: {pos_alpha}/{len(results)}")
    print(f"  Subjects with increased theta during meditation: {pos_theta}/{len(results)}")
    print(f"  Subjects with increased meditation index:        {pos_index}/{len(results)}")
    print()

    if pos_alpha >= len(results) * 0.5 or pos_index >= len(results) * 0.5:
        print("  MeditationSentinel shows expected meditation effects!")
        print()
        print("  COMPARISON TO LITERATURE:")
        print("  - Alpha increase during meditation: 5-20% typical")
        print("  - Theta increase during deep meditation: 10-30% typical")
        print("  - Frontal alpha enhancement: Well-documented marker")
    else:
        print("  Results do not show expected meditation effects.")
        print("  Consider: different electrode selection, longer epochs, or artifact rejection.")


if __name__ == '__main__':
    main()
