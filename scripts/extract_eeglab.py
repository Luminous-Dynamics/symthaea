#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Extract EEG data from EEGLAB .set/.fdt format to simple binary format.

DENS Dataset (ds003751) Format:
- .set file: MATLAB v5 format with header info
- .fdt file: Binary float32 data (channels x samples)

Output:
- .bin file: Simple binary format readable by Rust
- .json file: Metadata (channels, sampling rate, events)
"""

import sys
import os
import json
import struct
import numpy as np

def load_set_file(set_path):
    """Load EEGLAB .set file header using scipy."""
    try:
        import scipy.io as sio
        mat = sio.loadmat(set_path, struct_as_record=False, squeeze_me=True)
        eeg = mat['EEG']
        return eeg
    except ImportError:
        print("scipy not installed, trying mne...")
        return None
    except Exception as e:
        print(f"scipy failed: {e}, trying mne...")
        return None

def load_with_mne(set_path):
    """Load using MNE-Python."""
    try:
        import mne
        raw = mne.io.read_raw_eeglab(set_path, preload=True)
        return raw
    except ImportError:
        print("mne not installed")
        return None
    except Exception as e:
        print(f"mne failed: {e}")
        return None

def extract_eeglab_data(set_path, fdt_path, output_prefix):
    """Extract data and save in Rust-friendly format."""

    # Try scipy first
    eeg = load_set_file(set_path)

    if eeg is not None:
        # Extract metadata from scipy load
        srate = float(eeg.srate)
        n_channels = int(eeg.nbchan)
        n_samples = int(eeg.pnts)

        # Get channel names
        chanlocs = eeg.chanlocs
        if hasattr(chanlocs, '__iter__'):
            channel_names = [ch.labels if hasattr(ch, 'labels') else f'Ch{i}'
                           for i, ch in enumerate(chanlocs)]
        else:
            channel_names = [f'Ch{i}' for i in range(n_channels)]

        # Load data from .fdt
        data = np.fromfile(fdt_path, dtype=np.float32)
        data = data.reshape((n_channels, n_samples), order='F')  # Fortran order for MATLAB

        # Get events
        events = []
        if hasattr(eeg, 'event') and eeg.event is not None:
            evts = eeg.event
            if not hasattr(evts, '__iter__'):
                evts = [evts]
            for ev in evts:
                event = {
                    'latency': float(ev.latency) if hasattr(ev, 'latency') else 0,
                    'type': str(ev.type) if hasattr(ev, 'type') else ''
                }
                events.append(event)

    else:
        # Try MNE
        raw = load_with_mne(set_path)
        if raw is None:
            print("ERROR: Could not load file with scipy or mne")
            sys.exit(1)

        srate = raw.info['sfreq']
        n_channels = len(raw.ch_names)
        n_samples = len(raw.times)
        channel_names = raw.ch_names
        data = raw.get_data().astype(np.float32)

        # Get annotations as events
        events = []
        if raw.annotations:
            for ann in raw.annotations:
                events.append({
                    'latency': float(ann['onset'] * srate),
                    'type': ann['description']
                })

    # Save metadata
    metadata = {
        'sampling_rate': srate,
        'n_channels': n_channels,
        'n_samples': n_samples,
        'channel_names': channel_names,
        'events': events[:100]  # Limit events to keep JSON reasonable
    }

    with open(output_prefix + '.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {output_prefix}.json")
    print(f"  Sampling rate: {srate} Hz")
    print(f"  Channels: {n_channels}")
    print(f"  Samples: {n_samples}")
    print(f"  Duration: {n_samples/srate:.1f} seconds")
    print(f"  Events: {len(events)}")

    # Save data in simple binary format: header + data
    # Header: 4 bytes n_channels, 4 bytes n_samples, 8 bytes srate
    with open(output_prefix + '.bin', 'wb') as f:
        f.write(struct.pack('<I', n_channels))
        f.write(struct.pack('<I', n_samples))
        f.write(struct.pack('<d', srate))
        data.tofile(f)

    print(f"Data saved to {output_prefix}.bin ({os.path.getsize(output_prefix + '.bin') / 1e6:.1f} MB)")

    # Also create a small test segment (first 60 seconds)
    test_samples = int(srate * 60)
    test_data = data[:, :test_samples]
    with open(output_prefix + '_test.bin', 'wb') as f:
        f.write(struct.pack('<I', n_channels))
        f.write(struct.pack('<I', test_samples))
        f.write(struct.pack('<d', srate))
        test_data.tofile(f)

    print(f"Test segment saved to {output_prefix}_test.bin (60s)")

    return metadata

def main():
    if len(sys.argv) < 3:
        print("Usage: extract_eeglab.py <set_file> <output_prefix>")
        print("  Automatically finds .fdt file based on .set location")
        sys.exit(1)

    set_path = sys.argv[1]
    output_prefix = sys.argv[2]

    # Find .fdt file
    fdt_path = set_path.replace('.set', '.fdt')
    if not os.path.exists(fdt_path):
        # Try with capital letters (DENS uses mixed case)
        base = os.path.dirname(set_path)
        name = os.path.basename(set_path).replace('.set', '').replace('.SET', '')
        for ext in ['.fdt', '.FDT', '.Fdt']:
            candidate = os.path.join(base, name + ext)
            if os.path.exists(candidate):
                fdt_path = candidate
                break

    if not os.path.exists(set_path):
        print(f"ERROR: .set file not found: {set_path}")
        sys.exit(1)

    if not os.path.exists(fdt_path):
        print(f"ERROR: .fdt file not found: {fdt_path}")
        sys.exit(1)

    print(f"Loading: {set_path}")
    print(f"Data:    {fdt_path}")

    extract_eeglab_data(set_path, fdt_path, output_prefix)

if __name__ == '__main__':
    main()
