#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Attention Entropy Analysis for Phenomenal Concepts

Investigates whether phenomenal concepts exhibit different attention patterns
than functional concepts. Measures:
1. Attention entropy (spread of attention)
2. Attention concentration (max attention value)
3. Self-attention vs cross-attention patterns

Hypothesis: Phenomenal concepts may show more distributed (higher entropy)
attention patterns, reflecting their integrative/unified nature.

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers])" \
        --run "python3 scripts/attention_entropy_analysis.py"
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


def compute_attention_entropy(attention_weights: np.ndarray) -> float:
    """Compute entropy of attention distribution (higher = more spread out)."""
    # Flatten and add small epsilon to avoid log(0)
    flat = attention_weights.flatten()
    flat = flat + 1e-10
    flat = flat / flat.sum()  # Normalize

    # Compute entropy
    entropy = -np.sum(flat * np.log2(flat))
    return float(entropy)


def compute_attention_concentration(attention_weights: np.ndarray) -> float:
    """Compute max attention value (higher = more concentrated)."""
    return float(np.max(attention_weights))


def extract_attention_stats(model, tokenizer, text: str, device: str = "cpu") -> Dict:
    """Extract attention statistics for a text."""
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # attentions is tuple of (batch, heads, seq, seq) per layer
    attentions = outputs.attentions

    n_layers = len(attentions)
    results = {
        "text": text,
        "n_tokens": inputs["input_ids"].shape[1],
        "layer_entropy": [],
        "layer_concentration": [],
        "layer_self_attention": [],  # Diagonal attention (token attending to itself)
    }

    for layer_idx, attn in enumerate(attentions):
        # Average over heads and batch
        attn_np = attn.squeeze(0).mean(dim=0).cpu().numpy()  # (seq, seq)

        entropy = compute_attention_entropy(attn_np)
        concentration = compute_attention_concentration(attn_np)

        # Self-attention: diagonal elements
        diag = np.diag(attn_np)
        self_attn = float(np.mean(diag))

        results["layer_entropy"].append(entropy)
        results["layer_concentration"].append(concentration)
        results["layer_self_attention"].append(self_attn)

    return results


def analyze_concept_class(
    model, tokenizer, concepts: List[str], device: str = "cpu"
) -> Dict:
    """Analyze attention patterns for a class of concepts."""
    all_entropy = []
    all_concentration = []
    all_self_attn = []

    for text in concepts:
        stats = extract_attention_stats(model, tokenizer, text, device)
        all_entropy.append(stats["layer_entropy"])
        all_concentration.append(stats["layer_concentration"])
        all_self_attn.append(stats["layer_self_attention"])

    # Convert to numpy and compute means per layer
    entropy_array = np.array(all_entropy)  # (n_concepts, n_layers)
    conc_array = np.array(all_concentration)
    self_array = np.array(all_self_attn)

    return {
        "entropy_mean": np.mean(entropy_array, axis=0).tolist(),
        "entropy_std": np.std(entropy_array, axis=0).tolist(),
        "concentration_mean": np.mean(conc_array, axis=0).tolist(),
        "concentration_std": np.std(conc_array, axis=0).tolist(),
        "self_attention_mean": np.mean(self_array, axis=0).tolist(),
        "self_attention_std": np.std(self_array, axis=0).tolist(),
    }


def main():
    print("\n" + "=" * 70)
    print("   ATTENTION ENTROPY ANALYSIS")
    print("   Testing attention pattern differences for phenomenal concepts")
    print("=" * 70 + "\n")

    model_name = "bert-base-uncased"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")
    print(f"Layers: {model.config.num_hidden_layers}\n")

    # Analyze both classes
    print("Analyzing phenomenal concepts...")
    phen_stats = analyze_concept_class(model, tokenizer, PHENOMENAL_CONCEPTS, device)

    print("Analyzing functional concepts...")
    func_stats = analyze_concept_class(model, tokenizer, FUNCTIONAL_CONCEPTS, device)

    # Summary
    print("\n" + "=" * 70)
    print("   LAYER-BY-LAYER COMPARISON")
    print("=" * 70 + "\n")

    n_layers = len(phen_stats["entropy_mean"])

    print(
        f"{'Layer':<8} {'Phen Entropy':<14} {'Func Entropy':<14} {'Diff':<10} {'Effect':<8}"
    )
    print("-" * 54)

    entropy_diffs = []
    for i in range(n_layers):
        pe = phen_stats["entropy_mean"][i]
        fe = func_stats["entropy_mean"][i]
        diff = pe - fe
        entropy_diffs.append(diff)

        # Simple effect size
        ps = phen_stats["entropy_std"][i]
        fs = func_stats["entropy_std"][i]
        pooled_std = np.sqrt((ps**2 + fs**2) / 2) if ps + fs > 0 else 1
        d = diff / pooled_std if pooled_std > 0 else 0

        print(f"L{i+1:<7} {pe:<14.4f} {fe:<14.4f} {diff:+<10.4f} {d:+.2f}")

    # Find peak difference
    best_layer = np.argmax(np.abs(entropy_diffs))
    best_diff = entropy_diffs[best_layer]

    print("\n" + "=" * 70)
    print("   KEY FINDING")
    print("=" * 70 + "\n")

    if best_diff > 0:
        print(f"  Phenomenal concepts show HIGHER attention entropy")
        print(f"  (more distributed attention)")
    else:
        print(f"  Phenomenal concepts show LOWER attention entropy")
        print(f"  (more concentrated attention)")

    print(f"  Peak difference at Layer {best_layer + 1}: {best_diff:+.4f}")

    # Also check concentration
    conc_diffs = [
        phen_stats["concentration_mean"][i] - func_stats["concentration_mean"][i]
        for i in range(n_layers)
    ]
    best_conc_layer = np.argmax(np.abs(conc_diffs))
    best_conc_diff = conc_diffs[best_conc_layer]

    print(
        f"\n  Concentration difference at Layer {best_conc_layer + 1}: {best_conc_diff:+.4f}"
    )

    if best_conc_diff < 0:
        print(f"  Phenomenal concepts have LOWER max attention")
        print(f"  (consistent with more distributed processing)")

    # Save results
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "phenomenal": phen_stats,
        "functional": func_stats,
        "entropy_diffs": entropy_diffs,
        "concentration_diffs": conc_diffs,
        "best_entropy_layer": int(best_layer + 1),
        "best_entropy_diff": float(best_diff),
    }

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/attention_entropy_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
