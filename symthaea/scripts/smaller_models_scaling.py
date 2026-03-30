#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Smaller Models Scaling Analysis

Tests whether inverse scaling law extends to smaller/efficient models:
- TinyBERT, MobileBERT, ALBERT, DistilBERT
- Provides full scaling curve from tiny to large models

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers scikit-learn])" \
        --run "python3 scripts/smaller_models_scaling.py"
"""

import json
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


# Load expanded corpus
def load_corpus():
    corpus_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/expanded_concept_corpus.json"
    try:
        with open(corpus_path) as f:
            corpus = json.load(f)
        return corpus["phenomenal_concepts"], corpus["functional_concepts"]
    except FileNotFoundError:
        # Fallback to inline corpus
        return PHENOMENAL_CONCEPTS, FUNCTIONAL_CONCEPTS


# Fallback corpus (subset)
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


def compute_fisher_criterion(phen_reps: np.ndarray, func_reps: np.ndarray) -> Dict:
    """Compute Fisher's criterion and related metrics."""
    phen_centroid = np.mean(phen_reps, axis=0)
    func_centroid = np.mean(func_reps, axis=0)

    centroid_distance = np.linalg.norm(phen_centroid - func_centroid)

    phen_within = np.mean([np.linalg.norm(r - phen_centroid) for r in phen_reps])
    func_within = np.mean([np.linalg.norm(r - func_centroid) for r in func_reps])
    avg_within = (phen_within + func_within) / 2

    fisher = centroid_distance / avg_within if avg_within > 0 else 0

    cos_sim = np.dot(phen_centroid, func_centroid) / (
        np.linalg.norm(phen_centroid) * np.linalg.norm(func_centroid) + 1e-10
    )

    return {
        "centroid_distance": float(centroid_distance),
        "avg_within_variance": float(avg_within),
        "fisher_criterion": float(fisher),
        "centroid_cosine_similarity": float(cos_sim),
    }


def analyze_model(
    model_name: str, device: str, phen_concepts: List[str], func_concepts: List[str]
) -> Dict:
    """Full analysis for a single model."""
    print(f"\n  Loading {model_name}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
    except Exception as e:
        # Some models need trust_remote_code
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, attn_implementation="eager"
        )

    model.eval()
    model = model.to(device)

    # Get model architecture info
    config = model.config
    n_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layers", 12))
    hidden_size = getattr(config, "hidden_size", getattr(config, "dim", 768))
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    # Peak layer at 90% depth
    peak_layer = int(n_layers * 0.9)

    print(
        f"    Layers: {n_layers}, Hidden: {hidden_size}, Params: {n_params:.1f}M, Peak: {peak_layer}"
    )

    # Extract representations
    phen_reps = extract_representations(
        model, tokenizer, phen_concepts, peak_layer, device
    )
    func_reps = extract_representations(
        model, tokenizer, func_concepts, peak_layer, device
    )

    # Compute metrics
    metrics = compute_fisher_criterion(phen_reps, func_reps)

    # Cleanup
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "n_params_millions": n_params,
        "peak_layer": peak_layer,
        **metrics,
    }


def main():
    print("\n" + "=" * 70)
    print("   SMALLER MODELS SCALING ANALYSIS")
    print("   Testing inverse scaling across model size spectrum")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load corpus
    phen_concepts, func_concepts = load_corpus()
    print(
        f"Corpus: {len(phen_concepts)} phenomenal, {len(func_concepts)} functional concepts"
    )

    # Model spectrum from tiny to large
    models = [
        # Tiny models
        ("google/bert_uncased_L-2_H-128_A-2", "BERT-Tiny"),  # 2L, 128H, ~4M
        ("google/bert_uncased_L-4_H-256_A-4", "BERT-Mini"),  # 4L, 256H, ~11M
        ("google/bert_uncased_L-4_H-512_A-8", "BERT-Small"),  # 4L, 512H, ~29M
        ("google/bert_uncased_L-8_H-512_A-8", "BERT-Medium"),  # 8L, 512H, ~42M
        # Efficient models
        ("huawei-noah/TinyBERT_General_4L_312D", "TinyBERT-4L"),  # 4L, 312H, ~14M
        ("google/mobilebert-uncased", "MobileBERT"),  # 24L, 512H, ~25M
        ("albert-base-v2", "ALBERT-base"),  # 12L, 768H, ~12M
        ("albert-large-v2", "ALBERT-large"),  # 24L, 1024H, ~18M
        ("distilbert-base-uncased", "DistilBERT"),  # 6L, 768H, ~66M
        # Standard models (for reference)
        ("bert-base-uncased", "BERT-base"),  # 12L, 768H, ~110M
        ("bert-large-uncased", "BERT-large"),  # 24L, 1024H, ~340M
    ]

    results = []
    for model_id, display_name in models:
        try:
            result = analyze_model(model_id, device, phen_concepts, func_concepts)
            result["display_name"] = display_name
            results.append(result)
            print(f"    Fisher: {result['fisher_criterion']:.4f}")
        except Exception as e:
            print(f"    Error: {e}")

    if len(results) < 3:
        print("\n  Too few models loaded successfully for analysis")
        return

    # Sort by parameter count
    results.sort(key=lambda x: x["n_params_millions"])

    # Analysis
    print("\n" + "=" * 70)
    print("   SCALING CURVE (sorted by size)")
    print("=" * 70 + "\n")

    print(
        f"{'Model':<20} {'Params (M)':<12} {'Hidden':<8} {'Fisher':<10} {'Cosine':<10}"
    )
    print("-" * 60)
    for r in results:
        print(
            f"{r['display_name']:<20} {r['n_params_millions']:<12.1f} {r['hidden_size']:<8} "
            f"{r['fisher_criterion']:<10.4f} {r['centroid_cosine_similarity']:<10.4f}"
        )

    # Correlation analysis
    print("\n" + "=" * 70)
    print("   SCALING CORRELATIONS")
    print("=" * 70 + "\n")

    params = [r["n_params_millions"] for r in results]
    fishers = [r["fisher_criterion"] for r in results]
    dims = [r["hidden_size"] for r in results]
    layers = [r["n_layers"] for r in results]

    corr_params = np.corrcoef(params, fishers)[0, 1]
    corr_dims = np.corrcoef(dims, fishers)[0, 1]
    corr_layers = np.corrcoef(layers, fishers)[0, 1]

    print(f"  Fisher vs Parameters:     r = {corr_params:.3f}")
    print(f"  Fisher vs Hidden Size:    r = {corr_dims:.3f}")
    print(f"  Fisher vs Layers:         r = {corr_layers:.3f}")

    # Key finding
    print("\n" + "=" * 70)
    print("   KEY FINDING")
    print("=" * 70 + "\n")

    if corr_params < -0.5:
        print("  ✓ INVERSE SCALING CONFIRMED ACROSS FULL SPECTRUM")
        print(f"    Correlation with parameters: r = {corr_params:.3f}")
        print("    Smaller models show STRONGER phenomenal discrimination")
        finding = "inverse_scaling_confirmed"
    elif corr_params > 0.5:
        print("  UNEXPECTED: Positive scaling found")
        print(f"    Correlation with parameters: r = {corr_params:.3f}")
        finding = "positive_scaling"
    else:
        print("  MIXED: No clear scaling relationship")
        print(f"    Correlation with parameters: r = {corr_params:.3f}")
        finding = "no_clear_pattern"

    # Identify optimal model size
    best_idx = np.argmax(fishers)
    worst_idx = np.argmin(fishers)
    print(
        f"\n  Best discrimination:  {results[best_idx]['display_name']} (Fisher = {fishers[best_idx]:.4f})"
    )
    print(
        f"  Worst discrimination: {results[worst_idx]['display_name']} (Fisher = {fishers[worst_idx]:.4f})"
    )

    # Save results
    output = {
        "experiment": "smaller_models_scaling",
        "n_phenomenal_concepts": len(phen_concepts),
        "n_functional_concepts": len(func_concepts),
        "models": [r["model"] for r in results],
        "results": results,
        "correlations": {
            "fisher_vs_params": float(corr_params),
            "fisher_vs_hidden_size": float(corr_dims),
            "fisher_vs_layers": float(corr_layers),
        },
        "finding": finding,
        "best_model": results[best_idx]["display_name"],
        "worst_model": results[worst_idx]["display_name"],
    }

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/smaller_models_scaling.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
