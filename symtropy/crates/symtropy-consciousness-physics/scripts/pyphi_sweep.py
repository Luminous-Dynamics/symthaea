#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
PyPhi correlation sweep: compute exact IIT Phi for multiple TPMs
and compare against our engine's softmin approximation.

Reads TPMs from stdin (one JSON per line), returns results on stdout.
Computes Pearson r² at the end.

Usage: cat tpms.jsonl | python3 pyphi_sweep.py
"""

import json
import sys
import collections
import collections.abc

# Python 3.10+ compatibility shim for PyPhi
for attr in ('Iterable', 'Mapping', 'MutableMapping', 'Sequence', 'MutableSequence'):
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))

import os
os.environ['PYPHI_WELCOME_OFF'] = '1'

try:
    import pyphi
    import numpy as np
except ImportError:
    print('{"error": "pyphi or numpy not installed"}')
    sys.exit(1)


def compute_phi(tpm_data, state):
    """Compute exact IIT Phi via PyPhi."""
    tpm = np.array(tpm_data, dtype=float)
    tpm = np.clip(tpm, 0.0, 1.0)
    state_tuple = tuple(int(s) for s in state)
    n = len(state_tuple)

    if tpm.shape != (2**n, n):
        return None, f"TPM shape {tpm.shape} != expected ({2**n}, {n})"

    cm = np.ones((n, n), dtype=int)
    network = pyphi.Network(tpm, cm=cm)
    subsystem = pyphi.Subsystem(network, state_tuple)
    sia = pyphi.compute.sia(subsystem)
    return float(sia.phi), None


def main():
    results = []

    # Read JSON lines from stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        engine_phi = req.get('engine_phi', 0.0)
        tpm = req.get('tpm', [])
        state = req.get('state', [])
        label = req.get('label', f'run_{len(results)}')

        pyphi_phi, err = compute_phi(tpm, state)

        result = {
            'label': label,
            'engine_phi': engine_phi,
            'pyphi_phi': pyphi_phi,
            'error': err,
        }
        results.append(result)
        print(json.dumps(result), flush=True)

    # Compute correlation if we have enough valid points
    valid = [(r['engine_phi'], r['pyphi_phi']) for r in results if r['pyphi_phi'] is not None]
    if len(valid) >= 3:
        engine_vals = np.array([v[0] for v in valid])
        pyphi_vals = np.array([v[1] for v in valid])

        # Pearson correlation
        if np.std(engine_vals) > 1e-10 and np.std(pyphi_vals) > 1e-10:
            r = np.corrcoef(engine_vals, pyphi_vals)[0, 1]
            r_squared = r ** 2
        else:
            r, r_squared = 0.0, 0.0

        summary = {
            'summary': True,
            'n_valid': len(valid),
            'n_total': len(results),
            'pearson_r': float(r),
            'r_squared': float(r_squared),
            'engine_mean': float(np.mean(engine_vals)),
            'pyphi_mean': float(np.mean(pyphi_vals)),
            'engine_std': float(np.std(engine_vals)),
            'pyphi_std': float(np.std(pyphi_vals)),
        }
        print(json.dumps(summary), flush=True)
    else:
        print(json.dumps({'summary': True, 'error': f'Only {len(valid)} valid points, need >= 3'}))


if __name__ == '__main__':
    main()
