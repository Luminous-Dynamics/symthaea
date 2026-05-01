#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Neural Bridge End-to-End Test

Tests the complete pipeline:
1. Get embedding from Ollama (embeddinggemma)
2. Project to HDC space using trained probe
3. Match against known concept targets

This demonstrates "LLM as Semantic Sensor" working in practice.
"""

import json
import urllib.request
from pathlib import Path

import numpy as np


def get_ollama_embedding(text: str, model: str = "embeddinggemma:300m") -> np.ndarray:
    """Get embedding from Ollama."""
    data = json.dumps({"model": model, "input": text}).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/embed",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return np.array(result["embeddings"][0], dtype=np.float32)


def generate_target_hdc(concept_name: str, seed_base: int = 42) -> np.ndarray:
    """Generate target HDC vector (matches Python train_probe.py)."""
    seed = seed_base
    for c in concept_name:
        seed = ((seed * 31) + ord(c)) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    return rng.choice([-1, 1], size=16384).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    print("=" * 65)
    print("  Neural Bridge End-to-End Test")
    print("  LLM as Semantic Sensor -> HDC Space")
    print("=" * 65)
    print()

    # Load trained probe
    weights_path = Path("models/neural_bridge/probe_weights_embeddinggemma.npy")
    print(f"Loading trained probe from {weights_path}...")
    weights = np.load(weights_path)
    print(f"  Probe shape: {weights.shape} (output_dim × input_dim)")
    print()

    # Load known concepts
    data_path = Path("data/neural_bridge/concept_activations_embeddinggemma.npz")
    data = np.load(data_path, allow_pickle=True)
    known_names = list(data["names"])
    print(f"Loaded {len(known_names)} known concepts")
    print()

    # Test with NEW sentences (not in training data)
    test_cases = [
        ("NixOS flakes are a way to manage dependencies declaratively.", "Nix_Flake"),
        ("Justice requires fair treatment under the law.", "Justice"),
        ("Consciousness arises from complex neural activity.", "Consciousness"),
        (
            "Mitochondria generate ATP through oxidative phosphorylation.",
            "Mitochondria",
        ),
        ("Democracy enables citizens to vote for their leaders.", "Democracy"),
        ("Evolution shapes species through natural selection.", "Evolution"),
        ("A nix derivation is a build recipe for a package.", "Nix_Derivation"),
        ("Gravity attracts objects with mass toward each other.", "Gravity"),
    ]

    print("Testing Neural Bridge with NEW sentences:")
    print("-" * 65)

    correct = 0
    total = len(test_cases)

    for sentence, expected_concept in test_cases:
        # Get embedding from Ollama
        embedding = get_ollama_embedding(sentence)

        # Project to HDC space using trained probe
        hdc_projected = embedding @ weights.T
        hdc_bipolar = np.sign(hdc_projected)

        # Get target HDC for expected concept
        target_hdc = generate_target_hdc(expected_concept)

        # Compute similarity to expected
        sim_to_target = cosine_similarity(hdc_bipolar, target_hdc)

        # Find closest known concept
        best_concept = None
        best_sim = -1
        for name in known_names:
            concept_hdc = generate_target_hdc(name)
            sim = cosine_similarity(hdc_bipolar, concept_hdc)
            if sim > best_sim:
                best_sim = sim
                best_concept = name

        match = best_concept == expected_concept
        symbol = "✓" if match else "✗"
        if match:
            correct += 1

        # Truncate sentence for display
        display_sentence = sentence[:50] + "..." if len(sentence) > 50 else sentence
        print(f'{symbol} "{display_sentence}"')
        print(f"    Expected: {expected_concept:20s} (sim={sim_to_target:+.4f})")
        print(f"    Matched:  {best_concept:20s} (sim={best_sim:+.4f})")
        print()

    print("=" * 65)
    print(f"  Results: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
