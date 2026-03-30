#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
GPT-2 Scaling Analysis

Tests whether phenomenal corridor depth scales with model size.
Compares GPT-2 (12 layers), GPT-2-medium (24 layers), GPT-2-large (36 layers).

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers])" \
        --run "python3 scripts/gpt2_scaling_analysis.py"
"""

import json
from typing import Dict, List

import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer

PHENOMENAL_CONCEPTS = [
    "The vivid experience of seeing red",
    "What it is like to taste chocolate",
    "The subjective feeling of pain",
    "The unified field of conscious awareness",
    "The felt quality of sadness",
]

FUNCTIONAL_CONCEPTS = [
    "The recursive algorithm terminates",
    "Matrix multiplication computes dot products",
    "Memory is allocated on the heap",
    "The compiler optimizes bytecode",
    "Hash tables provide O(1) lookup",
]


def extract_all_layers(
    model, tokenizer, text: str, device: str = "cpu"
) -> List[np.ndarray]:
    """Extract last-token representation from all layers."""
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    layer_activations = []
    for layer_output in hidden_states:
        seq_len = inputs["attention_mask"].sum().item()
        last_hidden = layer_output[0, seq_len - 1, :].cpu().numpy()
        layer_activations.append(last_hidden)

    return layer_activations


def compute_discrimination(
    phen_activations: List[np.ndarray], func_activations: List[np.ndarray]
) -> float:
    """Fisher's criterion discrimination score."""
    phen_centroid = np.mean(phen_activations, axis=0)
    func_centroid = np.mean(func_activations, axis=0)
    centroid_dist = np.linalg.norm(phen_centroid - func_centroid)

    phen_var = np.mean([np.linalg.norm(a - phen_centroid) for a in phen_activations])
    func_var = np.mean([np.linalg.norm(a - func_centroid) for a in func_activations])
    avg_var = (phen_var + func_var) / 2

    if avg_var > 0:
        return float(centroid_dist / avg_var)
    return 0.0


def analyze_model(model_name: str, device: str = "cpu") -> Dict:
    """Analyze phenomenal corridor for a GPT-2 variant."""
    print(f"\n{'='*60}")
    print(f"  Analyzing: {model_name}")
    print(f"{'='*60}")

    print("  Loading model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained(model_name)
    model.eval()
    model = model.to(device)

    n_layers = model.config.n_layer
    hidden_size = model.config.n_embd
    print(f"  Layers: {n_layers}, Hidden: {hidden_size}")

    # Extract activations
    print("  Extracting activations...")
    phen_layer_acts = [[] for _ in range(n_layers + 1)]
    for text in PHENOMENAL_CONCEPTS:
        layers = extract_all_layers(model, tokenizer, text, device)
        for i, act in enumerate(layers):
            phen_layer_acts[i].append(act)

    func_layer_acts = [[] for _ in range(n_layers + 1)]
    for text in FUNCTIONAL_CONCEPTS:
        layers = extract_all_layers(model, tokenizer, text, device)
        for i, act in enumerate(layers):
            func_layer_acts[i].append(act)

    # Compute discrimination per layer
    layer_results = []
    for layer_idx in range(n_layers + 1):
        discrim = compute_discrimination(
            phen_layer_acts[layer_idx], func_layer_acts[layer_idx]
        )
        depth_pct = (layer_idx / n_layers * 100) if n_layers > 0 else 0
        layer_results.append(
            {
                "layer": layer_idx,
                "discrimination": discrim,
                "depth_pct": depth_pct,
            }
        )

    # Find peak
    transformer_layers = layer_results[1:] if len(layer_results) > 1 else layer_results
    best_layer = max(transformer_layers, key=lambda x: x["discrimination"])

    print(
        f"  Peak: L{best_layer['layer']} ({best_layer['depth_pct']:.1f}%) discrim={best_layer['discrimination']:.4f}"
    )

    # Clean up memory
    del model
    del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "peak_layer": best_layer["layer"],
        "peak_depth_pct": best_layer["depth_pct"],
        "peak_discrimination": best_layer["discrimination"],
        "layer_results": layer_results,
    }


def main():
    print("\n" + "=" * 70)
    print("   GPT-2 SCALING ANALYSIS")
    print("   Testing phenomenal corridor across model sizes")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    models = [
        "gpt2",  # 12 layers, 768 hidden, 117M params
        "gpt2-medium",  # 24 layers, 1024 hidden, 345M params
        "gpt2-large",  # 36 layers, 1280 hidden, 762M params
    ]

    results = []
    for model_name in models:
        try:
            result = analyze_model(model_name, device)
            results.append(result)
        except Exception as e:
            print(f"  Error with {model_name}: {e}")
            results.append({"model": model_name, "error": str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("   SCALING RESULTS SUMMARY")
    print("=" * 70 + "\n")

    print(
        f"{'Model':<15} {'Layers':<8} {'Hidden':<8} {'Peak':<8} {'Depth %':<10} {'Discrim':<10}"
    )
    print("-" * 59)

    for r in results:
        if "error" in r:
            print(f"{r['model']:<15} ERROR: {r['error'][:30]}")
        else:
            print(
                f"{r['model']:<15} {r['n_layers']:<8} {r['hidden_size']:<8} L{r['peak_layer']:<6} {r['peak_depth_pct']:<10.1f} {r['peak_discrimination']:<10.4f}"
            )

    # Scaling analysis
    valid_results = [r for r in results if "error" not in r]

    if len(valid_results) >= 2:
        print("\n" + "-" * 70)
        print("   SCALING LAW ANALYSIS")
        print("-" * 70 + "\n")

        layers = [r["n_layers"] for r in valid_results]
        depths = [r["peak_depth_pct"] for r in valid_results]
        discrims = [r["peak_discrimination"] for r in valid_results]

        # Correlation between layers and depth
        if len(set(depths)) > 1:
            corr_depth = np.corrcoef(layers, depths)[0, 1]
            print(f"  Correlation (layers vs depth %): r = {corr_depth:.3f}")
        else:
            print(f"  Depth consistent across scales: {depths[0]:.1f}%")

        # Correlation between layers and discrimination
        corr_discrim = np.corrcoef(layers, discrims)[0, 1]
        print(f"  Correlation (layers vs discrimination): r = {corr_discrim:.3f}")

        print("\n" + "=" * 70)
        print("   KEY FINDING")
        print("=" * 70 + "\n")

        if all(d >= 90 for d in depths):
            print("  CONSISTENT LATE-LAYER CORRIDOR")
            print(f"  All models show peak at ~{np.mean(depths):.0f}% depth")
            print("  Phenomenal corridor scales with model size")
        elif np.std(depths) < 10:
            print("  RELATIVELY CONSISTENT DEPTH")
            print(f"  Mean depth: {np.mean(depths):.1f}% (std={np.std(depths):.1f}%)")
        else:
            print("  VARIABLE DEPTH ACROSS SCALES")
            print(f"  Depths range: {min(depths):.1f}% - {max(depths):.1f}%")

        if corr_discrim > 0.5:
            print("\n  Larger models show STRONGER phenomenal discrimination")
        elif corr_discrim < -0.5:
            print("\n  Larger models show WEAKER phenomenal discrimination")

    # Save results
    output = {
        "models_tested": models,
        "device": device,
        "results": results,
    }

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/gpt2_scaling_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
