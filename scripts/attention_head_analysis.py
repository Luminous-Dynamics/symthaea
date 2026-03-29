#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Attention Head Analysis for Phenomenal Concepts

Identifies which specific attention heads show the strongest
phenomenal/functional discrimination. This is the first step
toward mechanistic interpretability.

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers])" \
        --run "python3 scripts/attention_head_analysis.py"
"""

import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

PHENOMENAL_CONCEPTS = [
    "The vivid experience of seeing red",
    "What it is like to taste chocolate",
    "The subjective feeling of pain",
    "The unified field of conscious awareness",
    "The felt quality of sadness",
    "Qualia of the color blue",
    "First-person subjective experience",
    "The redness of red as experienced",
    "Phenomenal consciousness itself",
    "The hard problem of consciousness",
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


def compute_head_entropy(attn_head: np.ndarray) -> float:
    """Compute entropy for a single attention head pattern."""
    flat = attn_head.flatten() + 1e-10
    flat = flat / flat.sum()
    return float(-np.sum(flat * np.log2(flat)))


def extract_head_patterns(
    model, tokenizer, text: str, device: str = "cpu"
) -> np.ndarray:
    """Extract attention patterns for all heads at all layers.

    Returns: (n_layers, n_heads, seq_len, seq_len) array
    """
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Stack all layers: (n_layers, batch, heads, seq, seq)
    attentions = torch.stack(outputs.attentions, dim=0)
    # Remove batch dim: (n_layers, heads, seq, seq)
    attentions = attentions.squeeze(1).cpu().numpy()

    return attentions


def analyze_head_discrimination(
    model,
    tokenizer,
    phen_concepts: List[str],
    func_concepts: List[str],
    device: str = "cpu",
) -> Dict:
    """Analyze which heads best discriminate phenomenal from functional."""

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    # Collect entropy for each head for each concept
    phen_entropies = np.zeros((len(phen_concepts), n_layers, n_heads))
    func_entropies = np.zeros((len(func_concepts), n_layers, n_heads))

    print("Extracting phenomenal concept attention patterns...")
    for i, text in enumerate(phen_concepts):
        attns = extract_head_patterns(model, tokenizer, text, device)
        for layer in range(n_layers):
            for head in range(n_heads):
                phen_entropies[i, layer, head] = compute_head_entropy(
                    attns[layer, head]
                )

    print("Extracting functional concept attention patterns...")
    for i, text in enumerate(func_concepts):
        attns = extract_head_patterns(model, tokenizer, text, device)
        for layer in range(n_layers):
            for head in range(n_heads):
                func_entropies[i, layer, head] = compute_head_entropy(
                    attns[layer, head]
                )

    # Compute mean entropy per head per class
    phen_mean = np.mean(phen_entropies, axis=0)  # (n_layers, n_heads)
    func_mean = np.mean(func_entropies, axis=0)
    phen_std = np.std(phen_entropies, axis=0)
    func_std = np.std(func_entropies, axis=0)

    # Compute effect size (Cohen's d) for each head
    pooled_std = np.sqrt((phen_std**2 + func_std**2) / 2)
    pooled_std[pooled_std == 0] = 1  # Avoid division by zero
    effect_sizes = (
        phen_mean - func_mean
    ) / pooled_std  # Negative = phen has lower entropy

    return {
        "phen_mean": phen_mean,
        "func_mean": func_mean,
        "effect_sizes": effect_sizes,
        "n_layers": n_layers,
        "n_heads": n_heads,
    }


def main():
    print("\n" + "=" * 70)
    print("   ATTENTION HEAD ANALYSIS")
    print("   Identifying phenomenal-discriminating heads")
    print("=" * 70 + "\n")

    model_name = "bert-base-uncased"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")
    print(
        f"Layers: {model.config.num_hidden_layers}, Heads: {model.config.num_attention_heads}\n"
    )

    # Analyze
    results = analyze_head_discrimination(
        model, tokenizer, PHENOMENAL_CONCEPTS, FUNCTIONAL_CONCEPTS, device
    )

    effect_sizes = results["effect_sizes"]
    n_layers = results["n_layers"]
    n_heads = results["n_heads"]

    # Find top discriminating heads (most negative = phenomenal has lower entropy)
    flat_effects = effect_sizes.flatten()
    sorted_indices = np.argsort(flat_effects)  # Ascending (most negative first)

    print("\n" + "=" * 70)
    print("   TOP 10 PHENOMENAL-DISCRIMINATING HEADS")
    print("   (Lower entropy for phenomenal = more concentrated attention)")
    print("=" * 70 + "\n")

    print(
        f"{'Rank':<6} {'Layer':<8} {'Head':<8} {'Effect d':<12} {'Phen Entropy':<14} {'Func Entropy':<14}"
    )
    print("-" * 62)

    top_heads = []
    for rank, flat_idx in enumerate(sorted_indices[:10]):
        layer = flat_idx // n_heads
        head = flat_idx % n_heads
        d = effect_sizes[layer, head]
        pe = results["phen_mean"][layer, head]
        fe = results["func_mean"][layer, head]
        print(
            f"{rank+1:<6} L{layer+1:<7} H{head+1:<7} {d:+.3f}        {pe:<14.3f} {fe:<14.3f}"
        )
        top_heads.append({"layer": layer + 1, "head": head + 1, "effect_d": float(d)})

    # Also show bottom 10 (functional has lower entropy)
    print("\n" + "=" * 70)
    print("   TOP 10 FUNCTIONAL-DISCRIMINATING HEADS")
    print("   (Lower entropy for functional)")
    print("=" * 70 + "\n")

    print(
        f"{'Rank':<6} {'Layer':<8} {'Head':<8} {'Effect d':<12} {'Phen Entropy':<14} {'Func Entropy':<14}"
    )
    print("-" * 62)

    for rank, flat_idx in enumerate(sorted_indices[-10:][::-1]):
        layer = flat_idx // n_heads
        head = flat_idx % n_heads
        d = effect_sizes[layer, head]
        pe = results["phen_mean"][layer, head]
        fe = results["func_mean"][layer, head]
        print(
            f"{rank+1:<6} L{layer+1:<7} H{head+1:<7} {d:+.3f}        {pe:<14.3f} {fe:<14.3f}"
        )

    # Layer-wise summary
    print("\n" + "=" * 70)
    print("   LAYER-WISE SUMMARY")
    print("=" * 70 + "\n")

    layer_means = np.mean(effect_sizes, axis=1)
    print(f"{'Layer':<8} {'Mean Effect d':<14} {'Best Head':<12} {'Best d':<10}")
    print("-" * 44)

    for layer in range(n_layers):
        best_head = np.argmin(effect_sizes[layer])
        best_d = effect_sizes[layer, best_head]
        print(
            f"L{layer+1:<7} {layer_means[layer]:+.3f}          H{best_head+1:<11} {best_d:+.3f}"
        )

    # Find the best layer
    best_layer = np.argmin(layer_means)
    print(
        f"\nBest layer overall: L{best_layer + 1} (mean d = {layer_means[best_layer]:+.3f})"
    )

    # Key finding
    print("\n" + "=" * 70)
    print("   KEY FINDING: PHENOMENAL CIRCUIT CANDIDATES")
    print("=" * 70 + "\n")

    # Heads with d < -0.5
    strong_heads = []
    for layer in range(n_layers):
        for head in range(n_heads):
            if effect_sizes[layer, head] < -0.5:
                strong_heads.append((layer + 1, head + 1, effect_sizes[layer, head]))

    if strong_heads:
        print(
            f"  Found {len(strong_heads)} heads with strong phenomenal effect (d < -0.5):"
        )
        for layer, head, d in sorted(strong_heads, key=lambda x: x[2]):
            print(f"    L{layer}.H{head}: d = {d:.3f}")
        print("\n  These heads form the 'phenomenal circuit' - candidates for")
        print("  mechanistic interpretability and causal intervention.")
    else:
        print("  No heads with d < -0.5 found.")
        print(f"  Strongest effect: d = {np.min(effect_sizes):.3f}")

    # Save results
    output = {
        "model": model_name,
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "effect_sizes": effect_sizes.tolist(),
        "layer_mean_effects": [float(x) for x in layer_means.tolist()],
        "top_phenomenal_heads": top_heads,
        "best_layer": int(best_layer + 1),
        "strong_phenomenal_heads": [
            {"layer": int(l), "head": int(h), "effect_d": float(d)}
            for l, h, d in strong_heads
        ],
    }

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/attention_head_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
