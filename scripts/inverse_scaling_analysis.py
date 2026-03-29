#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Inverse Scaling Analysis

Investigates WHY larger models show weaker phenomenal discrimination.
Tests hypotheses:
1. Dimensionality curse (Fisher's criterion less sensitive in high-D)
2. Centroid distance vs variance decomposition
3. Dimensionality-controlled comparison

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers scikit-learn])" \
        --run "python3 scripts/inverse_scaling_analysis.py"
"""

import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
    "The raw sensation of cold",
    "What it is like to hear music",
    "The felt presence of another mind",
    "Subjective quality of seeing green",
    "The experience of time passing",
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
    "Sorting algorithms arrange elements",
    "The CPU executes machine instructions",
    "Graphs represent connected nodes",
    "Linked lists store sequential data",
    "The network protocol handshake completes",
]


def extract_representations(
    model, tokenizer, texts: List[str], layer_idx: int, device: str
) -> np.ndarray:
    """Extract mean-pooled representations from a specific layer."""
    representations = []

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[layer_idx]
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        representations.append(pooled.squeeze(0).cpu().numpy())

    return np.array(representations)


def compute_fisher_components(phen_reps: np.ndarray, func_reps: np.ndarray) -> Dict:
    """Decompose Fisher's criterion into components."""
    # Centroids
    phen_centroid = np.mean(phen_reps, axis=0)
    func_centroid = np.mean(func_reps, axis=0)

    # Between-class: centroid distance
    centroid_distance = np.linalg.norm(phen_centroid - func_centroid)

    # Within-class: average distance to centroid
    phen_within = np.mean([np.linalg.norm(r - phen_centroid) for r in phen_reps])
    func_within = np.mean([np.linalg.norm(r - func_centroid) for r in func_reps])
    avg_within = (phen_within + func_within) / 2

    # Fisher's criterion
    fisher = centroid_distance / avg_within if avg_within > 0 else 0

    # Additional metrics
    phen_spread = np.std([np.linalg.norm(r - phen_centroid) for r in phen_reps])
    func_spread = np.std([np.linalg.norm(r - func_centroid) for r in func_reps])

    # Cosine similarity between centroids
    cos_sim = np.dot(phen_centroid, func_centroid) / (
        np.linalg.norm(phen_centroid) * np.linalg.norm(func_centroid) + 1e-10
    )

    return {
        "centroid_distance": float(centroid_distance),
        "phen_within_variance": float(phen_within),
        "func_within_variance": float(func_within),
        "avg_within_variance": float(avg_within),
        "fisher_criterion": float(fisher),
        "phen_spread_std": float(phen_spread),
        "func_spread_std": float(func_spread),
        "centroid_cosine_similarity": float(cos_sim),
    }


def project_to_dimension(reps: np.ndarray, target_dim: int) -> np.ndarray:
    """Project representations to target dimensionality using PCA."""
    if reps.shape[1] <= target_dim:
        return reps

    # Can't have more components than samples
    max_components = min(target_dim, reps.shape[0] - 1)
    if max_components < 1:
        return reps

    pca = PCA(n_components=max_components, random_state=42)
    return pca.fit_transform(reps)


def analyze_model(model_name: str, device: str) -> Dict:
    """Full analysis for a single model."""
    print(f"\n  Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
    model.eval()
    model = model.to(device)

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    # Find peak layer (use late layers: 80-95% depth)
    peak_layer = int(n_layers * 0.9)

    print(f"    Layers: {n_layers}, Hidden: {hidden_size}, Peak layer: {peak_layer}")

    # Extract representations
    phen_reps = extract_representations(
        model, tokenizer, PHENOMENAL_CONCEPTS, peak_layer, device
    )
    func_reps = extract_representations(
        model, tokenizer, FUNCTIONAL_CONCEPTS, peak_layer, device
    )

    # Original analysis
    original = compute_fisher_components(phen_reps, func_reps)

    # Dimensionality-controlled analysis
    # Project all to same dimension (max 29 with 30 samples)
    all_reps = np.vstack([phen_reps, func_reps])
    n_samples = all_reps.shape[0]

    # Control dimensions: 10, 20, and 25 (all < n_samples)
    all_reps_10 = project_to_dimension(all_reps, 10)
    phen_reps_10 = all_reps_10[: len(phen_reps)]
    func_reps_10 = all_reps_10[len(phen_reps) :]
    controlled_10 = compute_fisher_components(phen_reps_10, func_reps_10)

    all_reps_20 = project_to_dimension(all_reps, 20)
    phen_reps_20 = all_reps_20[: len(phen_reps)]
    func_reps_20 = all_reps_20[len(phen_reps) :]
    controlled_20 = compute_fisher_components(phen_reps_20, func_reps_20)

    all_reps_25 = project_to_dimension(all_reps, 25)
    phen_reps_25 = all_reps_25[: len(phen_reps)]
    func_reps_25 = all_reps_25[len(phen_reps) :]
    controlled_25 = compute_fisher_components(phen_reps_25, func_reps_25)

    # Normalized by sqrt(dimensions) - theoretical correction
    sqrt_correction = original["fisher_criterion"] * np.sqrt(hidden_size / 768)

    # Clean up
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "n_params_millions": n_params,
        "peak_layer": peak_layer,
        "original": original,
        "controlled_10d": controlled_10,
        "controlled_20d": controlled_20,
        "controlled_25d": controlled_25,
        "sqrt_corrected_fisher": float(sqrt_correction),
    }


def main():
    print("\n" + "=" * 70)
    print("   INVERSE SCALING MECHANISM ANALYSIS")
    print("   Why do larger models show weaker phenomenal discrimination?")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    models = [
        "bert-base-uncased",  # 12L, 768D
        "bert-large-uncased",  # 24L, 1024D
        "roberta-base",  # 12L, 768D
        "roberta-large",  # 24L, 1024D
    ]

    results = []
    for model_name in models:
        try:
            result = analyze_model(model_name, device)
            results.append(result)
            print(
                f"    Fisher (original): {result['original']['fisher_criterion']:.4f}"
            )
            print(
                f"    Fisher (256D):     {result['controlled_256d']['fisher_criterion']:.4f}"
            )
        except Exception as e:
            print(f"    Error: {e}")

    # Analysis
    print("\n" + "=" * 70)
    print("   COMPONENT DECOMPOSITION")
    print("=" * 70 + "\n")

    print(
        f"{'Model':<20} {'Dim':<6} {'Centroid Dist':<14} {'Within Var':<12} {'Fisher':<10}"
    )
    print("-" * 62)
    for r in results:
        o = r["original"]
        print(
            f"{r['model']:<20} {r['hidden_size']:<6} {o['centroid_distance']:<14.2f} {o['avg_within_variance']:<12.2f} {o['fisher_criterion']:<10.4f}"
        )

    # Correlation analysis
    print("\n" + "=" * 70)
    print("   SCALING CORRELATIONS")
    print("=" * 70 + "\n")

    dims = [r["hidden_size"] for r in results]
    params = [r["n_params_millions"] for r in results]

    # Original Fisher
    orig_fishers = [r["original"]["fisher_criterion"] for r in results]
    orig_corr_dim = np.corrcoef(dims, orig_fishers)[0, 1]
    orig_corr_params = np.corrcoef(params, orig_fishers)[0, 1]

    # Controlled Fisher (20D - middle ground)
    ctrl_fishers = [r["controlled_20d"]["fisher_criterion"] for r in results]
    ctrl_corr_dim = np.corrcoef(dims, ctrl_fishers)[0, 1]
    ctrl_corr_params = np.corrcoef(params, ctrl_fishers)[0, 1]

    # Centroid distance
    cent_dists = [r["original"]["centroid_distance"] for r in results]
    cent_corr = np.corrcoef(params, cent_dists)[0, 1]

    # Within variance
    within_vars = [r["original"]["avg_within_variance"] for r in results]
    within_corr = np.corrcoef(params, within_vars)[0, 1]

    print(f"  Original Fisher vs params:     r = {orig_corr_params:.3f}")
    print(f"  Original Fisher vs dimension:  r = {orig_corr_dim:.3f}")
    print(f"  Controlled Fisher vs params:   r = {ctrl_corr_params:.3f}")
    print(f"  Controlled Fisher vs dimension: r = {ctrl_corr_dim:.3f}")
    print(f"")
    print(f"  Centroid distance vs params:   r = {cent_corr:.3f}")
    print(f"  Within variance vs params:     r = {within_corr:.3f}")

    # Dimensionality control results
    print("\n" + "=" * 70)
    print("   DIMENSIONALITY CONTROL RESULTS")
    print("=" * 70 + "\n")

    print(f"{'Model':<20} {'Original':<10} {'10D':<10} {'20D':<10} {'25D':<10}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['model']:<20} {r['original']['fisher_criterion']:<10.4f} "
            f"{r['controlled_10d']['fisher_criterion']:<10.4f} "
            f"{r['controlled_20d']['fisher_criterion']:<10.4f} "
            f"{r['controlled_25d']['fisher_criterion']:<10.4f}"
        )

    # Key finding
    print("\n" + "=" * 70)
    print("   KEY FINDING")
    print("=" * 70 + "\n")

    if abs(ctrl_corr_params) < 0.5 and abs(orig_corr_params) > 0.5:
        print("  DIMENSIONALITY IS THE CAUSE")
        print(f"  Original correlation:   r = {orig_corr_params:.3f} (strong inverse)")
        print(f"  Controlled correlation: r = {ctrl_corr_params:.3f} (weak/none)")
        print("")
        print("  After controlling for dimensionality, inverse scaling DISAPPEARS")
        print("  The effect is a MEASUREMENT ARTIFACT of Fisher's criterion in high-D")
        finding = "dimensionality_artifact"
    elif abs(ctrl_corr_params) > 0.5:
        print("  INVERSE SCALING IS REAL")
        print(f"  Original correlation:   r = {orig_corr_params:.3f}")
        print(f"  Controlled correlation: r = {ctrl_corr_params:.3f}")
        print("")
        print("  Even after dimensionality control, inverse scaling PERSISTS")
        print("  Larger models genuinely encode phenomenal content less distinctly")
        finding = "real_inverse_scaling"
    else:
        print("  MIXED RESULTS")
        print(f"  Original correlation:   r = {orig_corr_params:.3f}")
        print(f"  Controlled correlation: r = {ctrl_corr_params:.3f}")
        finding = "mixed"

    # Component analysis
    print("\n" + "-" * 70)
    print("   COMPONENT ANALYSIS")
    print("-" * 70 + "\n")

    if cent_corr > 0.5:
        print(f"  Centroid distance INCREASES with scale (r = {cent_corr:.3f})")
    elif cent_corr < -0.5:
        print(f"  Centroid distance DECREASES with scale (r = {cent_corr:.3f})")
    else:
        print(f"  Centroid distance roughly constant (r = {cent_corr:.3f})")

    if within_corr > 0.5:
        print(f"  Within-class variance INCREASES with scale (r = {within_corr:.3f})")
        print("  --> Larger models have MORE dispersed representations")
    elif within_corr < -0.5:
        print(f"  Within-class variance DECREASES with scale (r = {within_corr:.3f})")
    else:
        print(f"  Within-class variance roughly constant (r = {within_corr:.3f})")

    # Save results
    output = {
        "experiment": "inverse_scaling_mechanism",
        "models": [r["model"] for r in results],
        "results": results,
        "correlations": {
            "original_fisher_vs_params": float(orig_corr_params),
            "original_fisher_vs_dim": float(orig_corr_dim),
            "controlled_fisher_vs_params": float(ctrl_corr_params),
            "controlled_fisher_vs_dim": float(ctrl_corr_dim),
            "centroid_distance_vs_params": float(cent_corr),
            "within_variance_vs_params": float(within_corr),
        },
        "finding": finding,
    }

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/inverse_scaling_mechanism.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
