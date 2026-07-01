#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
GPT-2 Phenomenal Corridor Analysis

Tests whether the phenomenal corridor exists in decoder-only (autoregressive)
transformer models. GPT-2 uses causal attention (can only attend to past tokens).

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers])" \
        --run "python3 scripts/gpt2_phenomenal_corridor.py"
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


def extract_all_layers(
    model, tokenizer, text: str, device: str = "cpu"
) -> List[np.ndarray]:
    """Extract last-token representation from all layers (causal LM style)."""
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden)

    # For causal LM, use last token representation
    layer_activations = []
    for layer_output in hidden_states:
        # Get last non-padding token
        seq_len = inputs["attention_mask"].sum().item()
        last_hidden = layer_output[0, seq_len - 1, :].cpu().numpy()
        layer_activations.append(last_hidden)

    return layer_activations


def compute_unity_proxy(activation: np.ndarray) -> float:
    """Unity proxy based on activation statistics."""
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


def main():
    print("\n" + "=" * 70)
    print("   GPT-2 PHENOMENAL CORRIDOR ANALYSIS")
    print("   Testing decoder-only (autoregressive) architecture")
    print("=" * 70 + "\n")

    model_name = "gpt2"  # 12 layers, 768 hidden
    print(f"Loading {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")
    print(f"Layers: {model.config.n_layer}, Hidden: {model.config.n_embd}")
    print(f"Architecture: Decoder-only (causal attention)\n")

    n_layers = model.config.n_layer

    # Extract activations
    print("Extracting phenomenal concept activations...")
    phen_layer_activations = [[] for _ in range(n_layers + 1)]
    for text in PHENOMENAL_CONCEPTS:
        layers = extract_all_layers(model, tokenizer, text, device)
        for i, act in enumerate(layers):
            phen_layer_activations[i].append(act)

    print("Extracting functional concept activations...")
    func_layer_activations = [[] for _ in range(n_layers + 1)]
    for text in FUNCTIONAL_CONCEPTS:
        layers = extract_all_layers(model, tokenizer, text, device)
        for i, act in enumerate(layers):
            func_layer_activations[i].append(act)

    # Analyze each layer
    print(
        f"\n{'Layer':<8} {'Phen Unity':<12} {'Func Unity':<12} {'Diff':<10} {'Discrim':<10}"
    )
    print("-" * 52)

    results = []
    for layer_idx in range(n_layers + 1):
        phen_acts = phen_layer_activations[layer_idx]
        func_acts = func_layer_activations[layer_idx]

        phen_unity = np.mean([compute_unity_proxy(a) for a in phen_acts])
        func_unity = np.mean([compute_unity_proxy(a) for a in func_acts])
        unity_diff = phen_unity - func_unity
        discrim = compute_discrimination(phen_acts, func_acts)

        layer_name = "Embed" if layer_idx == 0 else f"L{layer_idx}"
        print(
            f"{layer_name:<8} {phen_unity:<12.4f} {func_unity:<12.4f} {unity_diff:+<10.4f} {discrim:<10.4f}"
        )

        results.append(
            {
                "layer": layer_idx,
                "phen_unity": float(phen_unity),
                "func_unity": float(func_unity),
                "unity_diff": float(unity_diff),
                "discrimination": float(discrim),
                "depth_pct": float(layer_idx / n_layers * 100) if n_layers > 0 else 0,
            }
        )

    # Find peak
    transformer_layers = results[1:] if len(results) > 1 else results
    best_layer = max(transformer_layers, key=lambda x: x["discrimination"])

    print("\n" + "=" * 70)
    print("   GPT-2 RESULTS SUMMARY")
    print("=" * 70 + "\n")

    print(
        f"Peak discrimination layer: L{best_layer['layer']} ({best_layer['depth_pct']:.1f}% depth)"
    )
    print(f"Peak discrimination score: {best_layer['discrimination']:.4f}")

    # Compare to encoder models
    print("\n" + "-" * 70)
    print("   COMPARISON: ENCODER vs DECODER")
    print("-" * 70 + "\n")

    print(f"{'Model':<25} {'Type':<12} {'Layers':<8} {'Peak':<8} {'Depth %':<10}")
    print("-" * 63)
    print(f"{'BERT-base':<25} {'Encoder':<12} {'12':<8} {'L11':<8} {'91.7%':<10}")
    print(
        f"{'GPT-2':<25} {'Decoder':<12} {'12':<8} {'L' + str(best_layer['layer']):<8} {best_layer['depth_pct']:.1f}%"
    )

    # Key finding
    print("\n" + "=" * 70)
    print("   KEY FINDING")
    print("=" * 70 + "\n")

    if best_layer["depth_pct"] > 80:
        print("  PHENOMENAL CORRIDOR EXISTS IN DECODER ARCHITECTURE")
        print(
            f"  GPT-2 shows peak at {best_layer['depth_pct']:.1f}% depth, similar to encoders"
        )
        print("  The late-layer phenomenal effect is ARCHITECTURE-GENERAL")
    elif best_layer["depth_pct"] > 50:
        print("  PARTIAL SUPPORT for decoder phenomenal corridor")
        print(f"  GPT-2 shows peak at {best_layer['depth_pct']:.1f}% depth")
    else:
        print("  NO CLEAR PHENOMENAL CORRIDOR in GPT-2")
        print(
            f"  Peak at {best_layer['depth_pct']:.1f}% depth differs from encoder pattern"
        )
        print("  Decoder architecture may process phenomenal content differently")

    # Save results
    output = {
        "model": model_name,
        "architecture": "decoder-only",
        "n_layers": n_layers,
        "hidden_size": model.config.n_embd,
        "layer_results": results,
        "peak_layer": best_layer["layer"],
        "peak_depth_pct": best_layer["depth_pct"],
        "peak_discrimination": best_layer["discrimination"],
    }

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/gpt2_phenomenal_corridor.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
