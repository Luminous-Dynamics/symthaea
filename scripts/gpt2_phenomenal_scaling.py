#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
GPT-2 Phenomenal Scaling Analysis

Tests whether the optimal-size phenomenon for phenomenal discrimination
generalizes to decoder-only (autoregressive) architectures.

Models tested:
- GPT-2 small (124M)
- GPT-2 medium (355M)
- GPT-2 large (774M)
- GPT-2 XL (1.5B)

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers])" \
        --run "python3 scripts/gpt2_phenomenal_scaling.py"
"""

import json
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def load_corpus():
    """Load expanded concept corpus."""
    corpus_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/expanded_concept_corpus.json"
    try:
        with open(corpus_path) as f:
            corpus = json.load(f)
        return corpus["phenomenal_concepts"], corpus["functional_concepts"]
    except FileNotFoundError:
        return PHENOMENAL_CONCEPTS, FUNCTIONAL_CONCEPTS


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
    """Extract representations from a specific layer using last token (for decoder models)."""
    representations = []

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[layer_idx]

        # For decoder models, use the last non-padding token
        # Get sequence lengths
        attention_mask = inputs["attention_mask"]
        seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing

        # Extract last token representation for each sequence
        batch_size = hidden.shape[0]
        last_token_reps = []
        for i in range(batch_size):
            last_idx = seq_lengths[i].item()
            last_token_reps.append(hidden[i, last_idx, :].cpu().numpy())

        representations.append(last_token_reps[0])

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

    # Angular separation
    cos_sim = np.dot(phen_centroid, func_centroid) / (
        np.linalg.norm(phen_centroid) * np.linalg.norm(func_centroid) + 1e-10
    )
    angular_sep = 1 - cos_sim

    return {
        "centroid_distance": float(centroid_distance),
        "avg_within_variance": float(avg_within),
        "fisher_criterion": float(fisher),
        "centroid_cosine_similarity": float(cos_sim),
        "angular_separation": float(angular_sep),
    }


def analyze_model(
    model_name: str,
    display_name: str,
    device: str,
    phen_concepts: List[str],
    func_concepts: List[str],
) -> Dict:
    """Full analysis for a single GPT-2 model."""
    print(f"\n  Loading {display_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT-2 doesn't have a pad token by default
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model = model.to(device)

    config = model.config
    n_layers = config.n_layer
    hidden_size = config.n_embd
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
        "display_name": display_name,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "n_params_millions": float(n_params),
        "peak_layer": peak_layer,
        **metrics,
    }


def main():
    print("\n" + "=" * 70)
    print("   GPT-2 PHENOMENAL SCALING ANALYSIS")
    print("   Testing decoder models for optimal-size phenomenon")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    phen_concepts, func_concepts = load_corpus()
    print(
        f"Corpus: {len(phen_concepts)} phenomenal, {len(func_concepts)} functional concepts"
    )

    # GPT-2 model family
    models = [
        ("gpt2", "GPT-2 Small"),  # 12L, 768H, 124M
        ("gpt2-medium", "GPT-2 Medium"),  # 24L, 1024H, 355M
        ("gpt2-large", "GPT-2 Large"),  # 36L, 1280H, 774M
        ("gpt2-xl", "GPT-2 XL"),  # 48L, 1600H, 1.5B
    ]

    results = []
    for model_id, display_name in models:
        try:
            result = analyze_model(
                model_id, display_name, device, phen_concepts, func_concepts
            )
            results.append(result)
            print(f"    Fisher: {result['fisher_criterion']:.4f}")
            print(f"    Angular sep: {result['angular_separation']:.4f}")
        except Exception as e:
            print(f"    Error: {e}")

    if len(results) < 2:
        print("\n  Not enough models for comparison")
        return

    # Sort by parameter count
    results.sort(key=lambda x: x["n_params_millions"])

    # Analysis
    print("\n" + "=" * 70)
    print("   GPT-2 SCALING CURVE")
    print("=" * 70 + "\n")

    print(
        f"{'Model':<15} {'Params (M)':<12} {'Fisher':<10} {'Angular Sep':<12} {'Cosine':<10}"
    )
    print("-" * 60)
    for r in results:
        print(
            f"{r['display_name']:<15} {r['n_params_millions']:<12.1f} "
            f"{r['fisher_criterion']:<10.4f} {r['angular_separation']:<12.4f} "
            f"{r['centroid_cosine_similarity']:<10.4f}"
        )

    # Correlation analysis
    print("\n" + "=" * 70)
    print("   SCALING CORRELATIONS")
    print("=" * 70 + "\n")

    params = [r["n_params_millions"] for r in results]
    fishers = [r["fisher_criterion"] for r in results]
    angular_seps = [r["angular_separation"] for r in results]

    corr_fisher = np.corrcoef(params, fishers)[0, 1]
    corr_angular = np.corrcoef(params, angular_seps)[0, 1]

    print(f"  Fisher vs Parameters:      r = {corr_fisher:.3f}")
    print(f"  Angular Sep vs Parameters: r = {corr_angular:.3f}")

    # Key finding
    print("\n" + "=" * 70)
    print("   KEY FINDING")
    print("=" * 70 + "\n")

    if corr_fisher < -0.5:
        print("  INVERSE SCALING CONFIRMED IN DECODER MODELS")
        print(f"    Correlation: r = {corr_fisher:.3f}")
        finding = "inverse_scaling"
    elif corr_fisher > 0.5:
        print("  POSITIVE SCALING IN DECODER MODELS")
        print(f"    Correlation: r = {corr_fisher:.3f}")
        finding = "positive_scaling"
    else:
        # Check for non-monotonic pattern
        best_idx = np.argmax(fishers)
        if best_idx > 0 and best_idx < len(fishers) - 1:
            print("  NON-MONOTONIC SCALING (OPTIMAL SIZE PHENOMENON)")
            print(
                f"    Best model: {results[best_idx]['display_name']} (Fisher = {fishers[best_idx]:.4f})"
            )
            finding = "non_monotonic"
        else:
            print("  NO CLEAR SCALING PATTERN")
            print(f"    Correlation: r = {corr_fisher:.3f}")
            finding = "unclear"

    best_idx = np.argmax(fishers)
    worst_idx = np.argmin(fishers)
    print(
        f"\n  Best discrimination:  {results[best_idx]['display_name']} (Fisher = {fishers[best_idx]:.4f})"
    )
    print(
        f"  Worst discrimination: {results[worst_idx]['display_name']} (Fisher = {fishers[worst_idx]:.4f})"
    )

    # Compare to encoder findings
    print("\n" + "=" * 70)
    print("   COMPARISON TO ENCODER MODELS")
    print("=" * 70 + "\n")

    print("  Encoder (BERT) optimal: ~110M params (BERT-base)")
    if results:
        decoder_best = results[best_idx]
        print(
            f"  Decoder (GPT-2) best:   ~{decoder_best['n_params_millions']:.0f}M params ({decoder_best['display_name']})"
        )

    # Save results
    output = {
        "experiment": "gpt2_phenomenal_scaling",
        "architecture": "decoder",
        "n_phenomenal_concepts": len(phen_concepts),
        "n_functional_concepts": len(func_concepts),
        "models": [r["model"] for r in results],
        "results": results,
        "correlations": {
            "fisher_vs_params": float(corr_fisher),
            "angular_sep_vs_params": float(corr_angular),
        },
        "finding": finding,
        "best_model": results[best_idx]["display_name"] if results else None,
        "worst_model": results[worst_idx]["display_name"] if results else None,
    }

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/gpt2_phenomenal_scaling.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
