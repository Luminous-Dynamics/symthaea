#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Attention-Topology Correlation Analysis

Tests whether attention entropy correlates with topological unity.
Hypothesis: Concepts with lower attention entropy (more concentrated attention)
may have higher topological unity (more integrated representation).

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers scipy])" \
        --run "python3 scripts/attention_topology_correlation.py"
"""

import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import stats
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
    """Compute entropy of attention distribution."""
    flat = attention_weights.flatten() + 1e-10
    flat = flat / flat.sum()
    return float(-np.sum(flat * np.log2(flat)))


def compute_unity_proxy(activation: np.ndarray) -> float:
    """Compute unity proxy from activation statistics."""
    norm = np.linalg.norm(activation)
    mean = np.mean(np.abs(activation))
    std = np.std(activation)
    if std > 0:
        inv_cv = mean / std
    else:
        inv_cv = float("inf")
    unity = norm * min(inv_cv, 10.0) / 100.0
    return float(unity)


def extract_both_metrics(
    model, tokenizer, text: str, layer_idx: int, device: str = "cpu"
) -> Tuple[float, float]:
    """Extract both attention entropy and unity proxy for a specific layer."""
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

    # Attention entropy for the specified layer
    attn = outputs.attentions[layer_idx]
    attn_np = attn.squeeze(0).mean(dim=0).cpu().numpy()
    entropy = compute_attention_entropy(attn_np)

    # Unity from hidden state at the specified layer
    hidden = outputs.hidden_states[
        layer_idx + 1
    ]  # +1 because hidden_states[0] is embeddings
    attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
    masked = hidden * attention_mask
    summed = masked.sum(dim=1)
    count = attention_mask.sum(dim=1)
    pooled = (summed / count).squeeze(0).cpu().numpy()
    unity = compute_unity_proxy(pooled)

    return entropy, unity


def main():
    print("\n" + "=" * 70)
    print("   ATTENTION-TOPOLOGY CORRELATION ANALYSIS")
    print("   Testing if attention entropy predicts topological unity")
    print("=" * 70 + "\n")

    model_name = "bert-base-uncased"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")

    # Test at the phenomenal corridor layer (L11 for BERT-base, 0-indexed as 10)
    target_layer = 10  # Layer 11 (0-indexed)
    print(f"Analyzing at Layer {target_layer + 1} (phenomenal corridor)\n")

    # Collect data
    all_data = []

    print("Extracting metrics for phenomenal concepts...")
    for text in PHENOMENAL_CONCEPTS:
        entropy, unity = extract_both_metrics(
            model, tokenizer, text, target_layer, device
        )
        all_data.append(
            {
                "text": text[:40],
                "class": "phenomenal",
                "entropy": entropy,
                "unity": unity,
            }
        )
        print(f"  {text[:30]:30s} entropy={entropy:.3f} unity={unity:.4f}")

    print("\nExtracting metrics for functional concepts...")
    for text in FUNCTIONAL_CONCEPTS:
        entropy, unity = extract_both_metrics(
            model, tokenizer, text, target_layer, device
        )
        all_data.append(
            {
                "text": text[:40],
                "class": "functional",
                "entropy": entropy,
                "unity": unity,
            }
        )
        print(f"  {text[:30]:30s} entropy={entropy:.3f} unity={unity:.4f}")

    # Compute correlations
    entropies = np.array([d["entropy"] for d in all_data])
    unities = np.array([d["unity"] for d in all_data])
    classes = np.array([1 if d["class"] == "phenomenal" else 0 for d in all_data])

    # Overall correlation
    r_overall, p_overall = stats.pearsonr(entropies, unities)

    # Within-class correlations
    phen_mask = classes == 1
    func_mask = classes == 0

    r_phen, p_phen = stats.pearsonr(entropies[phen_mask], unities[phen_mask])
    r_func, p_func = stats.pearsonr(entropies[func_mask], unities[func_mask])

    # Summary
    print("\n" + "=" * 70)
    print("   CORRELATION RESULTS")
    print("=" * 70 + "\n")

    print(
        f"Overall correlation (entropy vs unity):    r = {r_overall:+.3f}  p = {p_overall:.4f}"
    )
    print(
        f"Phenomenal concepts only:                  r = {r_phen:+.3f}  p = {p_phen:.4f}"
    )
    print(
        f"Functional concepts only:                  r = {r_func:+.3f}  p = {p_func:.4f}"
    )

    # Class differences
    phen_entropy_mean = np.mean(entropies[phen_mask])
    func_entropy_mean = np.mean(entropies[func_mask])
    phen_unity_mean = np.mean(unities[phen_mask])
    func_unity_mean = np.mean(unities[func_mask])

    print(f"\nClass means:")
    print(
        f"  Phenomenal:  entropy={phen_entropy_mean:.3f}  unity={phen_unity_mean:.4f}"
    )
    print(
        f"  Functional:  entropy={func_entropy_mean:.3f}  unity={func_unity_mean:.4f}"
    )
    print(
        f"  Difference:  entropy={phen_entropy_mean - func_entropy_mean:+.3f}  unity={phen_unity_mean - func_unity_mean:+.4f}"
    )

    # Key finding
    print("\n" + "=" * 70)
    print("   KEY FINDING")
    print("=" * 70 + "\n")

    if r_overall < -0.3 and p_overall < 0.05:
        print("  NEGATIVE CORRELATION FOUND")
        print(f"  Lower entropy (concentrated attention) → Higher unity")
        print(f"  r = {r_overall:.3f}, p = {p_overall:.4f}")
        print(
            "\n  Interpretation: Focused attention may enable integrated representation"
        )
    elif r_overall > 0.3 and p_overall < 0.05:
        print("  POSITIVE CORRELATION FOUND")
        print(f"  Higher entropy (distributed attention) → Higher unity")
        print(f"  r = {r_overall:.3f}, p = {p_overall:.4f}")
    else:
        print("  NO SIGNIFICANT CORRELATION")
        print(f"  r = {r_overall:.3f}, p = {p_overall:.4f}")
        print("\n  Attention entropy and topological unity appear to be independent")
        print("  The phenomenal effect may have separate mechanisms")

    # Check if phenomenal class shows different pattern
    if np.abs(r_phen - r_func) > 0.3:
        print(f"\n  CLASS-SPECIFIC PATTERN:")
        print(f"  Phenomenal r={r_phen:.3f} vs Functional r={r_func:.3f}")
        print(f"  The attention-topology relationship differs by concept class!")

    # Save results
    results = {
        "model": model_name,
        "target_layer": target_layer + 1,
        "correlations": {
            "overall": {"r": float(r_overall), "p": float(p_overall)},
            "phenomenal": {"r": float(r_phen), "p": float(p_phen)},
            "functional": {"r": float(r_func), "p": float(p_func)},
        },
        "class_means": {
            "phenomenal": {
                "entropy": float(phen_entropy_mean),
                "unity": float(phen_unity_mean),
            },
            "functional": {
                "entropy": float(func_entropy_mean),
                "unity": float(func_unity_mean),
            },
        },
        "concept_data": all_data,
    }

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/attention_topology_correlation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
