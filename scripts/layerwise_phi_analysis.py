#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Layer-wise Φ Trajectory Analysis

Tests the "phenomenal corridor" hypothesis across all layers of XLM-RoBERTa.
Computes a proxy for topological unity at each layer for phenomenal vs functional concepts.

Usage:
    nix-shell -p python3 python3Packages.torch python3Packages.transformers python3Packages.numpy --run \
        "python3 scripts/layerwise_phi_analysis.py"
"""

import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Concept corpus
PHENOMENAL_CONCEPTS = [
    "The vivid experience of seeing red",
    "What it is like to taste chocolate",
    "The subjective feeling of pain",
    "The unified field of conscious awareness",
    "The felt quality of sadness",
    "Qualia of the color blue",
    "The hard problem of consciousness",
    "First-person subjective experience",
    "The redness of red as experienced",
    "Phenomenal consciousness itself",
]

FUNCTIONAL_CONCEPTS = [
    "The recursive algorithm terminates",
    "Matrix multiplication computes dot products",
    "Memory is allocated on the heap",
    "The compiler optimizes bytecode",
    "Hash tables provide O(1) lookup",
    "TCP ensures reliable delivery",
    "The function returns an integer",
    "Binary search has logarithmic complexity",
    "The database indexes the primary key",
    "Garbage collection frees unused memory",
]


def extract_all_layers(
    model, tokenizer, text: str, device: str = "cpu"
) -> List[np.ndarray]:
    """Extract mean-pooled activations from all layers."""
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden) tensors

    # Mean pooling with attention mask
    attention_mask = inputs["attention_mask"].unsqueeze(-1).float()

    layer_activations = []
    for layer_output in hidden_states:
        # Mean pool
        masked = layer_output * attention_mask
        summed = masked.sum(dim=1)
        count = attention_mask.sum(dim=1)
        pooled = (summed / count).squeeze(0).cpu().numpy()
        layer_activations.append(pooled)

    return layer_activations


def compute_unity_proxy(activation: np.ndarray) -> float:
    """
    Compute a proxy for topological unity based on activation statistics.

    Higher values indicate more "unified" representation.
    Uses normalized L2 norm and inverse coefficient of variation.
    """
    norm = np.linalg.norm(activation)
    mean = np.mean(np.abs(activation))
    std = np.std(activation)

    # Inverse coefficient of variation (higher = more uniform activation)
    if std > 0:
        inv_cv = mean / std
    else:
        inv_cv = float("inf")

    # Combine norm and uniformity
    unity = norm * min(inv_cv, 10.0) / 100.0  # Scale to reasonable range
    return float(unity)


def compute_discrimination(
    phen_activations: List[np.ndarray], func_activations: List[np.ndarray]
) -> float:
    """
    Compute how well this layer discriminates phenomenal from functional.

    Uses cosine similarity between class centroids and within-class variance.
    """
    phen_centroid = np.mean(phen_activations, axis=0)
    func_centroid = np.mean(func_activations, axis=0)

    # Distance between centroids
    centroid_dist = np.linalg.norm(phen_centroid - func_centroid)

    # Within-class variance
    phen_var = np.mean([np.linalg.norm(a - phen_centroid) for a in phen_activations])
    func_var = np.mean([np.linalg.norm(a - func_centroid) for a in func_activations])
    avg_var = (phen_var + func_var) / 2

    # Discrimination score (like Fisher's criterion)
    if avg_var > 0:
        discrimination = centroid_dist / avg_var
    else:
        discrimination = 0.0

    return float(discrimination)


def main():
    print("\n" + "=" * 70)
    print("   LAYER-WISE Φ TRAJECTORY ANALYSIS")
    print("   Testing Phenomenal Corridor Hypothesis in XLM-RoBERTa")
    print("=" * 70 + "\n")

    # Load model
    print("Loading XLM-RoBERTa-base...")
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")
    print(f"Model has {model.config.num_hidden_layers} transformer layers\n")

    # Extract activations for all concepts
    print("Extracting activations for phenomenal concepts...")
    phen_layer_activations = [[] for _ in range(model.config.num_hidden_layers + 1)]
    for text in PHENOMENAL_CONCEPTS:
        layers = extract_all_layers(model, tokenizer, text, device)
        for i, act in enumerate(layers):
            phen_layer_activations[i].append(act)

    print("Extracting activations for functional concepts...")
    func_layer_activations = [[] for _ in range(model.config.num_hidden_layers + 1)]
    for text in FUNCTIONAL_CONCEPTS:
        layers = extract_all_layers(model, tokenizer, text, device)
        for i, act in enumerate(layers):
            func_layer_activations[i].append(act)

    # Analyze each layer
    print("\n" + "=" * 70)
    print("   LAYER-BY-LAYER ANALYSIS")
    print("=" * 70 + "\n")

    print(
        f"{'Layer':<8} {'Phen Unity':<12} {'Func Unity':<12} {'Diff':<10} {'Discrim':<10}"
    )
    print("-" * 52)

    results = []
    for layer_idx in range(model.config.num_hidden_layers + 1):
        phen_acts = phen_layer_activations[layer_idx]
        func_acts = func_layer_activations[layer_idx]

        # Compute unity for each class
        phen_unity = np.mean([compute_unity_proxy(a) for a in phen_acts])
        func_unity = np.mean([compute_unity_proxy(a) for a in func_acts])
        unity_diff = phen_unity - func_unity

        # Compute discrimination
        discrim = compute_discrimination(phen_acts, func_acts)

        layer_name = "Embed" if layer_idx == 0 else f"L{layer_idx}"
        print(
            f"{layer_name:<8} {phen_unity:<12.4f} {func_unity:<12.4f} {unity_diff:+<10.4f} {discrim:<10.4f}"
        )

        results.append(
            {
                "layer": layer_idx,
                "phen_unity": phen_unity,
                "func_unity": func_unity,
                "unity_diff": unity_diff,
                "discrimination": discrim,
                "depth_pct": layer_idx / model.config.num_hidden_layers * 100,
            }
        )

    # Find peak discrimination
    best_layer = max(results[1:], key=lambda x: x["discrimination"])  # Skip embedding
    corridor_layer = int(model.config.num_hidden_layers * 0.92)

    print("\n" + "=" * 70)
    print("   PHENOMENAL CORRIDOR ANALYSIS")
    print("=" * 70 + "\n")

    print(
        f"Best discriminating layer: L{best_layer['layer']} (discrimination = {best_layer['discrimination']:.4f})"
    )
    print(f"Predicted corridor layer (~92%): L{corridor_layer}")
    print(
        f"Layer {corridor_layer} discrimination: {results[corridor_layer]['discrimination']:.4f}"
    )

    # Check if best layer is near 92%
    best_depth = best_layer["layer"] / model.config.num_hidden_layers * 100
    print(f"\nBest layer depth: {best_depth:.1f}%")

    if abs(best_depth - 92) < 10:
        print("\n✓ PHENOMENAL CORRIDOR HYPOTHESIS SUPPORTED")
        print(f"  Peak discrimination occurs near 92% depth ({best_depth:.1f}%)")
    else:
        print("\n◐ PARTIAL SUPPORT")
        print(f"  Peak at {best_depth:.1f}%, expected ~92%")

    # Save results
    output_path = "data/layerwise_phi_trajectory.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "n_layers": model.config.num_hidden_layers,
                "n_phenomenal": len(PHENOMENAL_CONCEPTS),
                "n_functional": len(FUNCTIONAL_CONCEPTS),
                "results": results,
                "best_layer": best_layer["layer"],
                "corridor_layer": corridor_layer,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
