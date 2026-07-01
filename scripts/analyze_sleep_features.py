#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Analyze sleep stage feature distributions."""

import numpy as np
from scipy import signal
import re

def read_edf_header(path):
    with open(path, "rb") as f:
        version = f.read(8).decode().strip()
        patient = f.read(80).decode().strip()
        recording = f.read(80).decode().strip()
        start_date = f.read(8).decode().strip()
        start_time = f.read(8).decode().strip()
        header_bytes = int(f.read(8).decode().strip())
        reserved = f.read(44).decode().strip()
        n_records = int(f.read(8).decode().strip())
        record_duration = float(f.read(8).decode().strip())
        n_signals = int(f.read(4).decode().strip())

        labels = [f.read(16).decode().strip() for _ in range(n_signals)]
        f.seek(f.tell() + 80 * n_signals)
        f.seek(f.tell() + 8 * n_signals)
        phys_mins = [float(f.read(8).decode().strip()) for _ in range(n_signals)]
        phys_maxs = [float(f.read(8).decode().strip()) for _ in range(n_signals)]
        dig_mins = [int(f.read(8).decode().strip()) for _ in range(n_signals)]
        dig_maxs = [int(f.read(8).decode().strip()) for _ in range(n_signals)]
        f.seek(f.tell() + 80 * n_signals)
        n_samples = [int(f.read(8).decode().strip()) for _ in range(n_signals)]

    return {
        "n_records": n_records, "record_duration": record_duration,
        "n_signals": n_signals, "labels": labels, "n_samples": n_samples,
        "phys_mins": phys_mins, "phys_maxs": phys_maxs,
        "dig_mins": dig_mins, "dig_maxs": dig_maxs, "header_bytes": header_bytes,
    }

def read_edf_data(path, channel_idx=0, max_records=500):
    header = read_edf_header(path)
    n_records = min(header["n_records"], max_records)
    srate = header["n_samples"][channel_idx] / header["record_duration"]
    scale = (header["phys_maxs"][channel_idx] - header["phys_mins"][channel_idx]) / \
            (header["dig_maxs"][channel_idx] - header["dig_mins"][channel_idx])
    offset = header["phys_mins"][channel_idx] - scale * header["dig_mins"][channel_idx]

    data = []
    with open(path, "rb") as f:
        f.seek(header["header_bytes"])
        for rec in range(n_records):
            for sig_idx in range(header["n_signals"]):
                n_samp = header["n_samples"][sig_idx]
                raw = np.frombuffer(f.read(n_samp * 2), dtype=np.int16)
                if sig_idx == channel_idx:
                    data.append(raw * scale + offset)
    return np.concatenate(data), srate

def read_hypnogram(path, epoch_duration=30):
    header = read_edf_header(path)
    with open(path, "rb") as f:
        f.seek(header["header_bytes"])
        raw = f.read()
    text = raw.decode("latin-1", errors="ignore")
    parts = text.split("\x00")

    events = []
    for part in parts:
        if "Sleep stage" in part:
            match = re.search(r"\+(\d+\.?\d*)", part)
            timestamp = float(match.group(1)) if match else 0
            if "Sleep stage W" in part:
                stage = 0
            elif "Sleep stage 1" in part:
                stage = 1
            elif "Sleep stage 2" in part:
                stage = 2
            elif "Sleep stage 3" in part or "Sleep stage 4" in part:
                stage = 3
            elif "Sleep stage R" in part:
                stage = 4
            else:
                continue
            events.append((timestamp, stage))

    if not events:
        return np.array([])

    events.sort(key=lambda x: x[0])
    max_time = events[-1][0] + epoch_duration
    n_epochs = int(max_time / epoch_duration) + 1
    epoch_stages = np.full(n_epochs, -1, dtype=np.int32)

    for i, (timestamp, stage) in enumerate(events):
        start_epoch = int(timestamp / epoch_duration)
        end_time = events[i + 1][0] if i + 1 < len(events) else max_time
        end_epoch = int(end_time / epoch_duration)
        for e in range(start_epoch, min(end_epoch + 1, n_epochs)):
            epoch_stages[e] = stage

    return epoch_stages

def band_power(data, srate, fmin, fmax):
    nperseg = min(int(4 * srate), len(data))
    if nperseg < 8:
        return 0.0
    f, psd = signal.welch(data, srate, nperseg=nperseg)
    idx = (f >= fmin) & (f <= fmax)
    return np.trapz(psd[idx], f[idx]) if idx.sum() > 1 else 0

def main():
    psg_path = "data/sleep-edf/SC4001E0-PSG.edf"
    hypno_path = "data/sleep-edf/SC4001EC-Hypnogram.edf"

    header = read_edf_header(psg_path)
    print(f"Channels: {header['labels']}")
    print(f"Total records in file: {header['n_records']}")
    print(f"Record duration: {header['record_duration']}s")
    total_duration = header['n_records'] * header['record_duration']
    print(f"Total recording duration: {total_duration/3600:.1f} hours")

    eeg_idx = 0
    for i, label in enumerate(header["labels"]):
        if "EEG" in label.upper():
            eeg_idx = i
            print(f"Using channel {i}: {label}")
            break

    # Read more data to cover the sleep period (sleep starts ~8.5 hours in)
    # Need at least 3000 records to cover wake + full sleep
    data, srate = read_edf_data(psg_path, eeg_idx, max_records=3000)
    stages = read_hypnogram(hypno_path)

    print(f"\nData: {len(data)} samples @ {srate} Hz")
    print(f"Data duration: {len(data)/srate/3600:.2f} hours")
    print(f"Stages array: {len(stages)} epochs ({len(stages)*30/3600:.2f} hours)")

    # Show stage distribution in hypnogram
    stage_names = ["Wake", "N1", "N2", "N3", "REM"]
    print("\nStage distribution in hypnogram:")
    for s in range(5):
        count = (stages == s).sum()
        print(f"  {stage_names[s]}: {count} epochs")

    epoch_samples = int(30 * srate)
    n_epochs = min(len(stages), len(data) // epoch_samples)

    features_by_stage = {s: [] for s in range(5)}
    for epoch in range(n_epochs):
        stage = stages[epoch]
        if stage < 0:
            continue

        start = epoch * epoch_samples
        end = start + epoch_samples
        if end > len(data):
            break

        epoch_data = data[start:end]
        delta = band_power(epoch_data, srate, 0.5, 4)
        theta = band_power(epoch_data, srate, 4, 8)
        alpha = band_power(epoch_data, srate, 8, 13)
        beta = band_power(epoch_data, srate, 13, 30)
        total = delta + theta + alpha + beta + 1e-10

        features_by_stage[stage].append({
            "delta_rel": delta / total,
            "alpha_rel": alpha / total,
            "theta_rel": theta / total,
            "beta_rel": beta / total,
        })

    print("\nFeature distributions by sleep stage:")
    print("-" * 70)
    header_str = "Stage      N     Delta    Alpha    Theta     Beta"
    print(header_str)
    print("-" * 70)
    stage_names = ["Wake", "N1", "N2", "N3", "REM"]
    for s, name in enumerate(stage_names):
        feats = features_by_stage[s]
        if feats:
            delta = np.mean([f["delta_rel"] for f in feats]) * 100
            alpha = np.mean([f["alpha_rel"] for f in feats]) * 100
            theta = np.mean([f["theta_rel"] for f in feats]) * 100
            beta = np.mean([f["beta_rel"] for f in feats]) * 100
            print(f"{name:<8} {len(feats):>4}  {delta:>6.1f}%  {alpha:>6.1f}%  {theta:>6.1f}%  {beta:>6.1f}%")
        else:
            print(f"{name:<8} {0:>4}       -        -        -        -")

    print("\nKey observations for classification thresholds:")
    for s, name in enumerate(stage_names):
        feats = features_by_stage[s]
        if feats:
            delta = np.mean([f["delta_rel"] for f in feats])
            alpha = np.mean([f["alpha_rel"] for f in feats])
            print(f"  {name}: delta={delta:.2f}, alpha={alpha:.2f}")

if __name__ == "__main__":
    main()
