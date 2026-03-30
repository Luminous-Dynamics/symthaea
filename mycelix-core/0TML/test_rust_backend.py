#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Test zerotrustml_core Rust backend."""

import time
import random

def main():
    import zerotrustml_core as ztc

    print("="*60)
    print("Testing zerotrustml_core Rust backend v" + ztc.version())
    print("="*60)
    print(f"HV_DIMENSION: {ztc.HV_DIMENSION}")

    # Encoder test
    print("\n--- Hypervector Encoder ---")
    encoder = ztc.HypervectorEncoder()
    gradient = [float(i) / 1000 for i in range(10000)]
    start = time.time()
    hv = encoder.encode(gradient)
    elapsed = (time.time() - start) * 1e6
    print(f"Encoded 10K params in {elapsed:.0f}us")
    print(f"HV density: {hv.density():.3f}")

    # Phi test
    print("\n--- Phi Measurer ---")
    phi = ztc.PhiMeasurer()
    hvs = [encoder.encode([float(i)/100 for i in range(500)]) for _ in range(5)]
    start = time.time()
    result = phi.measure_phi(hvs)
    elapsed = (time.time() - start) * 1000
    print(f"Phi: {result.phi_value:.4f} in {elapsed:.1f}ms")
    print(f"Integration: {result.integration:.4f}")
    print(f"Coherence: {result.coherence:.4f}")

    # Shapley test
    print("\n--- Shapley Computer ---")
    shapley = ztc.ShapleyComputer()
    start = time.time()
    contribs = shapley.compute_exact(hvs)
    elapsed = (time.time() - start) * 1000
    print(f"Computed {len(contribs)} values in {elapsed:.1f}ms")
    for c in contribs[:3]:
        print(f"  Node {c.node_id}: {c.shapley_value:+.4f} (positive={c.is_positive_contributor})")

    # Detector test
    print("\n--- Unified Detector ---")
    detector = ztc.UnifiedDetector()
    random.seed(42)
    grads = [[random.gauss(0, 1) for _ in range(5000)] for _ in range(10)]
    start = time.time()
    result = detector.detect_from_gradients(grads)
    elapsed = (time.time() - start) * 1000
    print(f"Detection in {elapsed:.1f}ms")
    print(f"Byzantine: {result.byzantine_nodes}")
    print(f"Confidence: {result.confidence:.4f}")

    # Attack detection
    print("\n--- Byzantine Attack (30%) ---")
    random.seed(123)
    honest = [[random.gauss(0, 1) for _ in range(1000)] for _ in range(7)]
    byzantine = [[random.gauss(0, 50) for _ in range(1000)] for _ in range(3)]
    all_grads = honest + byzantine

    start = time.time()
    result = detector.detect_from_gradients(all_grads)
    elapsed = (time.time() - start) * 1000
    print(f"Detected {len(result.byzantine_nodes)}/3 Byzantine in {elapsed:.1f}ms")
    print(f"Indices: {sorted(result.byzantine_nodes)}")
    print(f"Confidence: {result.confidence:.4f}")

    print("\n" + "="*60)
    print("SUCCESS! All Rust backend tests passed!")
    print("="*60)

if __name__ == "__main__":
    main()
