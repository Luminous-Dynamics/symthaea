#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
PyPhi ground-truth IIT Phi computation for Symtropy validation.

Takes a JSON request on stdin, computes IIT integrated information via PyPhi,
and returns a JSON response on stdout.

Input JSON:
{
    "tpm": [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], ...],  # 2^N x N state-by-node TPM
    "state": [1, 0, 1],                                 # current binary state
    "cm": [[1,1,0], [1,1,1], [0,1,1]],                 # connectivity matrix (optional)
    "node_labels": ["A0", "A1", "A2"]                   # labels (optional)
}

Output JSON:
{
    "phi": 0.3125,
    "num_concepts": 3,
    "major_complex_size": 3,
    "error": null
}

Requirements: pyphi >= 1.2  (pip install pyphi)
Run via:  echo '{"tpm": ..., "state": ...}' | python3 pyphi_compute.py
"""

import json
import sys
import collections
import collections.abc

# Compatibility shim for PyPhi on Python 3.10+
# PyPhi uses collections.Iterable which was removed in 3.10
for attr in ('Iterable', 'Mapping', 'MutableMapping', 'Sequence', 'MutableSequence'):
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))

def compute_phi(request: dict) -> dict:
    """Compute IIT Phi using PyPhi."""
    try:
        import pyphi
    except ImportError:
        return {
            "phi": None,
            "num_concepts": 0,
            "major_complex_size": 0,
            "error": "pyphi not installed. Run: pip install pyphi",
        }

    try:
        import numpy as np

        tpm = np.array(request["tpm"], dtype=float)
        state = tuple(int(s) for s in request["state"])
        n = len(state)

        # Connectivity matrix: fully connected by default
        if "cm" in request and request["cm"] is not None:
            cm = np.array(request["cm"], dtype=int)
        else:
            cm = np.ones((n, n), dtype=int)

        # Node labels
        labels = request.get("node_labels") or [f"n{i}" for i in range(n)]

        # Validate TPM shape: should be 2^N x N (state-by-node format)
        expected_rows = 2 ** n
        if tpm.shape != (expected_rows, n):
            return {
                "phi": None,
                "num_concepts": 0,
                "major_complex_size": 0,
                "error": f"TPM shape {tpm.shape} != expected ({expected_rows}, {n})",
            }

        # Clamp probabilities to [0, 1]
        tpm = np.clip(tpm, 0.0, 1.0)

        # Create PyPhi network and subsystem
        network = pyphi.Network(tpm, cm=cm, node_labels=labels)
        subsystem = pyphi.Subsystem(network, state)

        # Compute Big Phi (IIT 3.0/4.0)
        sia = pyphi.compute.sia(subsystem)

        return {
            "phi": float(sia.phi),
            "num_concepts": len(sia.ces) if hasattr(sia, "ces") else 0,
            "major_complex_size": n,
            "error": None,
        }

    except Exception as e:
        return {
            "phi": None,
            "num_concepts": 0,
            "major_complex_size": 0,
            "error": str(e),
        }


if __name__ == "__main__":
    request = json.loads(sys.stdin.read())
    result = compute_phi(request)
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
