#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Modern Decoder Scaling Analysis

Tests phenomenal discrimination scaling on modern decoder architectures:
- Qwen2 (small models available without auth)
- OPT (Facebook's open decoder)
- Pythia (EleutherAI's decoder suite with multiple sizes)

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers])" \
        --run "python3 scripts/modern_decoder_scaling.py"
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
        # Use subset for faster testing
        return corpus["phenomenal_concepts"][:20], corpus["functional_concepts"][:20]
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


def extract_representations(
    model, tokenizer, texts: List[str], layer_idx: int, device: str
) -> np.ndarray:
    """Extract representations using last token for decoder models."""
    representations = []

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[layer_idx]
        attention_mask = inputs["attention_mask"]
        seq_lengths = attention_mask.sum(dim=1) - 1

        batch_size = hidden.shape[0]
        for i in range(batch_size):
            last_idx = seq_lengths[i].item()
            representations.append(hidden[i, last_idx, :].cpu().numpy())

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
    """Full analysis for a single model."""
    print(f"\n  Loading {display_name}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"    Failed to load: {e}")
        return None

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    model = model.to(device)

    # Get config
    config = model.config
    n_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 12))
    hidden_size = getattr(config, 'hidden_size', getattr(config, 'n_embd', 768))
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    peak_layer = int(n_layers * 0.9)

    print(f"    Layers: {n_layers}, Hidden: {hidden_size}, Params: {n_params:.1f}M, Peak: {peak_layer}")

    # Extract representations
    phen_reps = extract_representations(model, tokenizer, phen_concepts, peak_layer, device)
    func_reps = extract_representations(model, tokenizer, func_concepts, peak_layer, device)

    metrics = compute_fisher_criterion(phen_reps, func_reps)

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
    print("   MODERN DECODER SCALING ANALYSIS")
    print("   Testing Pythia and OPT model families")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    phen_concepts, func_concepts = load_corpus()
    print(f"Corpus: {len(phen_concepts)} phenomenal, {len(func_concepts)} functional concepts")

    # Modern decoder models (publicly available without auth)
    models = [
        # Pythia suite (EleutherAI) - good size range
        ("EleutherAI/pythia-70m", "Pythia-70M"),
        ("EleutherAI/pythia-160m", "Pythia-160M"),
        ("EleutherAI/pythia-410m", "Pythia-410M"),
        ("EleutherAI/pythia-1b", "Pythia-1B"),

        # OPT suite (Meta) - similar to GPT-2
        ("facebook/opt-125m", "OPT-125M"),
        ("facebook/opt-350m", "OPT-350M"),
        ("facebook/opt-1.3b", "OPT-1.3B"),
    ]

    results = []
    for model_id, display_name in models:
        try:
            result = analyze_model(model_id, display_name, device, phen_concepts, func_concepts)
            if result:
                results.append(result)
                print(f"    Fisher: {result['fisher_criterion']:.4f}")
                print(f"    Angular sep: {result['angular_separation']:.4f}")
        except Exception as e:
            print(f"    Error: {e}")

    if len(results) < 2:
        print("\n  Not enough models for comparison")
        return

    # Sort by parameters
    results.sort(key=lambda x: x["n_params_millions"])

    # Analysis
    print("\n" + "=" * 70)
    print("   MODERN DECODER SCALING CURVE")
    print("=" * 70 + "\n")

    print(f"{'Model':<15} {'Params (M)':<12} {'Fisher':<10} {'Angular Sep':<12}")
    print("-" * 50)
    for r in results:
        print(f"{r['display_name']:<15} {r['n_params_millions']:<12.1f} "
              f"{r['fisher_criterion']:<10.4f} {r['angular_separation']:<12.4f}")

    # Correlations
    params = [r["n_params_millions"] for r in results]
    fishers = [r["fisher_criterion"] for r in results]

    corr_fisher = np.corrcoef(params, fishers)[0, 1]

    print("\n" + "=" * 70)
    print("   SCALING CORRELATIONS")
    print("=" * 70 + "\n")
    print(f"  Fisher vs Parameters: r = {corr_fisher:.3f}")

    # Key finding
    print("\n" + "=" * 70)
    print("   KEY FINDING")
    print("=" * 70 + "\n")

    best_idx = np.argmax(fishers)
    worst_idx = np.argmin(fishers)

    if corr_fisher < -0.5:
        print("  INVERSE SCALING CONFIRMED IN MODERN DECODERS")
        finding = "inverse_scaling"
    elif corr_fisher > 0.5:
        print("  POSITIVE SCALING IN MODERN DECODERS")
        finding = "positive_scaling"
    else:
        if best_idx > 0 and best_idx < len(fishers) - 1:
            print("  NON-MONOTONIC SCALING (OPTIMAL SIZE PHENOMENON)")
            finding = "non_monotonic"
        else:
            print("  NO CLEAR PATTERN")
            finding = "unclear"

    print(f"\n  Best:  {results[best_idx]['display_name']} (F={fishers[best_idx]:.4f})")
    print(f"  Worst: {results[worst_idx]['display_name']} (F={fishers[worst_idx]:.4f})")

    # Compare to GPT-2
    print("\n" + "=" * 70)
    print("   COMPARISON TO GPT-2")
    print("=" * 70 + "\n")
    print("  GPT-2 optimal: ~355M (GPT-2 Medium)")
    print(f"  Modern decoder optimal: ~{results[best_idx]['n_params_millions']:.0f}M ({results[best_idx]['display_name']})")

    # Save results
    output = {
        "experiment": "modern_decoder_scaling",
        "architecture": "decoder",
        "models_tested": [r["model"] for r in results],
        "n_phenomenal_concepts": len(phen_concepts),
        "n_functional_concepts": len(func_concepts),
        "results": results,
        "correlations": {"fisher_vs_params": float(corr_fisher)},
        "finding": finding,
        "best_model": results[best_idx]["display_name"],
        "worst_model": results[worst_idx]["display_name"],
    }

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/modern_decoder_scaling.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
