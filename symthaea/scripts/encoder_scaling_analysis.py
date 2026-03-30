#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Encoder Scaling Analysis

Tests whether inverse scaling (larger models = weaker discrimination) holds for encoders.
Compares BERT-base vs BERT-large and adds RoBERTa-large.

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers])" \
        --run "python3 scripts/encoder_scaling_analysis.py"
"""

import json
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

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


def extract_peak_layer(
    model, tokenizer, text: str, device: str = "cpu"
) -> List[np.ndarray]:
    """Extract mean-pooled representation from all layers."""
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    layer_activations = []

    for layer_output in hidden_states:
        # Mean pooling
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
        masked = layer_output * attention_mask
        pooled = (
            (masked.sum(dim=1) / attention_mask.sum(dim=1)).squeeze(0).cpu().numpy()
        )
        layer_activations.append(pooled)

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
    """Analyze phenomenal corridor for an encoder model."""
    print(f"\n{'='*60}")
    print(f"  Analyzing: {model_name}")
    print(f"{'='*60}")

    print("  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
    model.eval()
    model = model.to(device)

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())

    print(f"  Layers: {n_layers}, Hidden: {hidden_size}, Params: {n_params/1e6:.1f}M")

    # Extract activations for all layers
    print("  Extracting activations...")
    phen_layer_acts = [[] for _ in range(n_layers + 1)]
    for text in PHENOMENAL_CONCEPTS:
        layers = extract_peak_layer(model, tokenizer, text, device)
        for i, act in enumerate(layers):
            phen_layer_acts[i].append(act)

    func_layer_acts = [[] for _ in range(n_layers + 1)]
    for text in FUNCTIONAL_CONCEPTS:
        layers = extract_peak_layer(model, tokenizer, text, device)
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

    # Find peak (excluding embedding layer 0)
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
        "n_params_millions": n_params / 1e6,
        "peak_layer": best_layer["layer"],
        "peak_depth_pct": best_layer["depth_pct"],
        "peak_discrimination": best_layer["discrimination"],
        "layer_results": layer_results,
    }


def main():
    print("\n" + "=" * 70)
    print("   ENCODER SCALING ANALYSIS")
    print("   Testing inverse scaling law on encoder models")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Test encoder models of different sizes
    models = [
        # BERT family
        "bert-base-uncased",  # 12 layers, 768 hidden, 110M params
        "bert-large-uncased",  # 24 layers, 1024 hidden, 340M params
        # RoBERTa family
        "roberta-base",  # 12 layers, 768 hidden, 125M params
        "roberta-large",  # 24 layers, 1024 hidden, 355M params
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
    print("   ENCODER SCALING RESULTS")
    print("=" * 70 + "\n")

    print(f"{'Model':<20} {'Layers':<8} {'Params':<10} {'Peak':<8} {'Discrim':<10}")
    print("-" * 56)

    for r in results:
        if "error" in r:
            print(f"{r['model']:<20} ERROR: {r['error'][:30]}")
        else:
            print(
                f"{r['model']:<20} {r['n_layers']:<8} {r['n_params_millions']:<10.1f}M L{r['peak_layer']:<6} {r['peak_discrimination']:<10.4f}"
            )

    # Scaling analysis
    valid_results = [r for r in results if "error" not in r]

    if len(valid_results) >= 2:
        print("\n" + "-" * 70)
        print("   SCALING LAW ANALYSIS")
        print("-" * 70 + "\n")

        # BERT family
        bert_results = [
            r
            for r in valid_results
            if "bert-" in r["model"] and "roberta" not in r["model"]
        ]
        if len(bert_results) >= 2:
            bert_params = [r["n_params_millions"] for r in bert_results]
            bert_discrims = [r["peak_discrimination"] for r in bert_results]
            bert_corr = np.corrcoef(bert_params, bert_discrims)[0, 1]
            print(f"  BERT family (n={len(bert_results)}):")
            print(f"    Params: {bert_params}")
            print(f"    Discrimination: {[f'{d:.4f}' for d in bert_discrims]}")
            print(f"    Correlation (params vs discrim): r = {bert_corr:.3f}")

            if bert_corr < -0.5:
                print(f"    --> INVERSE SCALING CONFIRMED for BERT")

        # RoBERTa family
        roberta_results = [r for r in valid_results if "roberta" in r["model"]]
        if len(roberta_results) >= 2:
            roberta_params = [r["n_params_millions"] for r in roberta_results]
            roberta_discrims = [r["peak_discrimination"] for r in roberta_results]
            roberta_corr = np.corrcoef(roberta_params, roberta_discrims)[0, 1]
            print(f"\n  RoBERTa family (n={len(roberta_results)}):")
            print(f"    Params: {roberta_params}")
            print(f"    Discrimination: {[f'{d:.4f}' for d in roberta_discrims]}")
            print(f"    Correlation (params vs discrim): r = {roberta_corr:.3f}")

            if roberta_corr < -0.5:
                print(f"    --> INVERSE SCALING CONFIRMED for RoBERTa")

        # Overall
        all_params = [r["n_params_millions"] for r in valid_results]
        all_discrims = [r["peak_discrimination"] for r in valid_results]
        overall_corr = np.corrcoef(all_params, all_discrims)[0, 1]

        print(f"\n  Overall (n={len(valid_results)}):")
        print(f"    Correlation (params vs discrim): r = {overall_corr:.3f}")

        # Key finding
        print("\n" + "=" * 70)
        print("   KEY FINDING")
        print("=" * 70 + "\n")

        if overall_corr < -0.5:
            print(f"  INVERSE SCALING CONFIRMED FOR ENCODERS (r = {overall_corr:.3f})")
            print("  Larger encoder models show WEAKER phenomenal discrimination")
            print("  This matches the GPT-2 decoder finding (r = -0.88)")
        elif overall_corr > 0.5:
            print(f"  POSITIVE SCALING (r = {overall_corr:.3f})")
            print("  Larger encoder models show STRONGER discrimination")
            print("  This DIFFERS from GPT-2 decoder finding")
        else:
            print(f"  WEAK/NO SCALING RELATIONSHIP (r = {overall_corr:.3f})")
            print("  Model size does not strongly predict discrimination")

    # Save results
    output = {
        "experiment": "encoder_scaling_analysis",
        "models_tested": models,
        "device": device,
        "results": results,
        "analysis": {
            "overall_correlation": (
                float(overall_corr) if len(valid_results) >= 2 else None
            ),
        },
    }

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/encoder_scaling_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
