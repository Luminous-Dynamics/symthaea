#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Multi-Architecture Phenomenal Corridor Analysis

Tests the "phenomenal corridor" hypothesis across multiple transformer architectures.
Investigates whether the peak discrimination depth (~92% in BGE-M3) is consistent
across different model sizes and architectures.

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix develop --command python3 scripts/multi_arch_analysis.py

Or for a single model:
    nix develop --command python3 scripts/multi_arch_analysis.py --model bert-base-uncased

Results saved to: data/architecture_depth_map.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Concept corpus (same as original)
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

# Models to test
DEFAULT_MODELS = [
    "bert-base-uncased",
    "bert-large-uncased",
    "roberta-base",
    "distilbert-base-uncased",
]

# Reference results from prior analysis
REFERENCE_RESULTS = {
    "BAAI/bge-m3": {"n_layers": 24, "peak_layer": 22, "peak_depth_pct": 91.67},
    "xlm-roberta-base": {"n_layers": 12, "peak_layer": 6, "peak_depth_pct": 50.0},
}


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
    """
    norm = np.linalg.norm(activation)
    mean = np.mean(np.abs(activation))
    std = np.std(activation)

    if std > 0:
        inv_cv = mean / std
    else:
        inv_cv = float("inf")

    unity = norm * min(inv_cv, 10.0) / 100.0
    return float(unity)


def compute_discrimination(
    phen_activations: List[np.ndarray], func_activations: List[np.ndarray]
) -> float:
    """
    Compute how well this layer discriminates phenomenal from functional.
    Uses Fisher's criterion: distance between centroids / within-class variance.
    """
    phen_centroid = np.mean(phen_activations, axis=0)
    func_centroid = np.mean(func_activations, axis=0)

    # Distance between centroids
    centroid_dist = np.linalg.norm(phen_centroid - func_centroid)

    # Within-class variance
    phen_var = np.mean([np.linalg.norm(a - phen_centroid) for a in phen_activations])
    func_var = np.mean([np.linalg.norm(a - func_centroid) for a in func_activations])
    avg_var = (phen_var + func_var) / 2

    if avg_var > 0:
        discrimination = centroid_dist / avg_var
    else:
        discrimination = 0.0

    return float(discrimination)


def analyze_model(model_name: str, device: str = "cpu") -> Dict:
    """
    Analyze a single model's phenomenal corridor.

    Returns dict with:
        - model: model name
        - n_layers: number of transformer layers
        - hidden_size: hidden dimension
        - layer_results: per-layer analysis
        - peak_layer: layer with best discrimination
        - peak_depth_pct: depth percentage of peak layer
        - peak_discrimination: discrimination score at peak
    """
    print(f"\n{'='*70}")
    print(f"  Analyzing: {model_name}")
    print(f"{'='*70}")

    try:
        print(f"  Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        model = model.to(device)

        n_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        print(f"  Layers: {n_layers}, Hidden size: {hidden_size}")
        print(f"  Device: {device}")

    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return {
            "model": model_name,
            "error": str(e),
            "n_layers": None,
            "peak_layer": None,
            "peak_depth_pct": None,
        }

    # Extract activations for all concepts
    print(f"  Extracting phenomenal concept activations...")
    phen_layer_activations = [[] for _ in range(n_layers + 1)]
    for text in PHENOMENAL_CONCEPTS:
        layers = extract_all_layers(model, tokenizer, text, device)
        for i, act in enumerate(layers):
            phen_layer_activations[i].append(act)

    print(f"  Extracting functional concept activations...")
    func_layer_activations = [[] for _ in range(n_layers + 1)]
    for text in FUNCTIONAL_CONCEPTS:
        layers = extract_all_layers(model, tokenizer, text, device)
        for i, act in enumerate(layers):
            func_layer_activations[i].append(act)

    # Analyze each layer
    print(f"\n  {'Layer':<8} {'Phen Unity':<12} {'Func Unity':<12} {'Discrim':<10}")
    print(f"  {'-'*42}")

    layer_results = []
    for layer_idx in range(n_layers + 1):
        phen_acts = phen_layer_activations[layer_idx]
        func_acts = func_layer_activations[layer_idx]

        phen_unity = np.mean([compute_unity_proxy(a) for a in phen_acts])
        func_unity = np.mean([compute_unity_proxy(a) for a in func_acts])
        unity_diff = phen_unity - func_unity
        discrim = compute_discrimination(phen_acts, func_acts)

        layer_name = "Embed" if layer_idx == 0 else f"L{layer_idx}"
        print(
            f"  {layer_name:<8} {phen_unity:<12.4f} {func_unity:<12.4f} {discrim:<10.4f}"
        )

        layer_results.append(
            {
                "layer": layer_idx,
                "phen_unity": phen_unity,
                "func_unity": func_unity,
                "unity_diff": unity_diff,
                "discrimination": discrim,
                "depth_pct": (layer_idx / n_layers * 100) if n_layers > 0 else 0,
            }
        )

    # Find peak discrimination (skip embedding layer)
    transformer_layers = layer_results[1:] if len(layer_results) > 1 else layer_results
    best_layer = max(transformer_layers, key=lambda x: x["discrimination"])

    peak_depth_pct = (best_layer["layer"] / n_layers * 100) if n_layers > 0 else 0

    print(f"\n  Peak Layer: L{best_layer['layer']} ({peak_depth_pct:.1f}% depth)")
    print(f"  Peak Discrimination: {best_layer['discrimination']:.4f}")

    # Free memory
    del model
    del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "layer_results": layer_results,
        "peak_layer": best_layer["layer"],
        "peak_depth_pct": peak_depth_pct,
        "peak_discrimination": best_layer["discrimination"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multi-architecture phenomenal corridor analysis"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single model to analyze (default: analyze all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/architecture_depth_map.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)",
    )
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 70)
    print("   MULTI-ARCHITECTURE PHENOMENAL CORRIDOR ANALYSIS")
    print("   Testing depth consistency across transformer architectures")
    print("=" * 70)
    print(f"\n  Device: {device}")
    print(f"  Phenomenal concepts: {len(PHENOMENAL_CONCEPTS)}")
    print(f"  Functional concepts: {len(FUNCTIONAL_CONCEPTS)}")

    # Determine which models to test
    if args.model:
        models_to_test = [args.model]
    else:
        models_to_test = DEFAULT_MODELS

    print(f"\n  Models to test: {len(models_to_test)}")
    for m in models_to_test:
        print(f"    - {m}")

    # Run analysis
    results = {}
    start_time = time.time()

    for model_name in models_to_test:
        model_result = analyze_model(model_name, device)
        results[model_name] = model_result

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("   SUMMARY: PHENOMENAL CORRIDOR DEPTH BY ARCHITECTURE")
    print("=" * 70)

    print(
        f"\n  {'Model':<30} {'Layers':<8} {'Peak':<8} {'Depth %':<10} {'Discrim':<10}"
    )
    print(f"  {'-'*66}")

    # Include reference results
    for model_name, ref in REFERENCE_RESULTS.items():
        print(
            f"  {model_name:<30} {ref['n_layers']:<8} L{ref['peak_layer']:<6} {ref['peak_depth_pct']:<10.1f} {'(ref)':<10}"
        )

    # Print new results
    depth_pcts = []
    for model_name, result in results.items():
        if result.get("error"):
            print(f"  {model_name:<30} ERROR: {result['error'][:30]}")
        else:
            discrim_str = f"{result['peak_discrimination']:.4f}"
            print(
                f"  {model_name:<30} {result['n_layers']:<8} L{result['peak_layer']:<6} {result['peak_depth_pct']:<10.1f} {discrim_str:<10}"
            )
            depth_pcts.append(result["peak_depth_pct"])

    # Statistical summary
    if depth_pcts:
        print(f"\n  Depth Statistics (new models only):")
        print(f"    Mean:   {np.mean(depth_pcts):.1f}%")
        print(f"    Std:    {np.std(depth_pcts):.1f}%")
        print(f"    Min:    {np.min(depth_pcts):.1f}%")
        print(f"    Max:    {np.max(depth_pcts):.1f}%")

        # Include reference results in overall analysis
        all_depths = depth_pcts + [
            ref["peak_depth_pct"] for ref in REFERENCE_RESULTS.values()
        ]
        print(f"\n  Depth Statistics (including reference):")
        print(f"    Mean:   {np.mean(all_depths):.1f}%")
        print(f"    Std:    {np.std(all_depths):.1f}%")
        print(f"    Range:  {np.min(all_depths):.1f}% - {np.max(all_depths):.1f}%")

    print(f"\n  Total time: {elapsed:.1f}s")

    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "n_phenomenal_concepts": len(PHENOMENAL_CONCEPTS),
        "n_functional_concepts": len(FUNCTIONAL_CONCEPTS),
        "reference_results": REFERENCE_RESULTS,
        "model_results": results,
        "summary": {
            "models_tested": list(results.keys()),
            "depth_percentages": {
                name: r.get("peak_depth_pct")
                for name, r in results.items()
                if not r.get("error")
            },
            "mean_depth_pct": float(np.mean(depth_pcts)) if depth_pcts else None,
            "std_depth_pct": float(np.std(depth_pcts)) if depth_pcts else None,
        },
    }

    # Ensure output directory exists
    output_path = args.output
    if not output_path.startswith("/"):
        # Relative path - use script's parent directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        output_path = os.path.join(project_dir, output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n  Results saved to: {output_path}")

    # Key finding
    print("\n" + "=" * 70)
    print("   KEY FINDING")
    print("=" * 70)

    if depth_pcts:
        mean_depth = np.mean(all_depths)
        std_depth = np.std(all_depths)

        if std_depth < 15:
            print(f"\n  The phenomenal corridor depth shows MODERATE CONSISTENCY")
            print(
                f"  across architectures (mean={mean_depth:.1f}%, std={std_depth:.1f}%)"
            )
        elif std_depth < 25:
            print(f"\n  The phenomenal corridor depth shows HIGH VARIABILITY")
            print(
                f"  across architectures (mean={mean_depth:.1f}%, std={std_depth:.1f}%)"
            )
        else:
            print(f"\n  The phenomenal corridor depth is ARCHITECTURE-DEPENDENT")
            print(f"  (mean={mean_depth:.1f}%, std={std_depth:.1f}%)")

        # Check if any match the BGE-M3 pattern (~92%)
        high_depth_models = [
            (name, r["peak_depth_pct"])
            for name, r in results.items()
            if r.get("peak_depth_pct") and r["peak_depth_pct"] > 80
        ]
        low_depth_models = [
            (name, r["peak_depth_pct"])
            for name, r in results.items()
            if r.get("peak_depth_pct") and r["peak_depth_pct"] < 60
        ]

        if high_depth_models:
            print(
                f"\n  Models with high peak depth (>80%): {[m[0] for m in high_depth_models]}"
            )
        if low_depth_models:
            print(
                f"  Models with low peak depth (<60%): {[m[0] for m in low_depth_models]}"
            )

    print("\n" + "=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
