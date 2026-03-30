#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Causal Head Ablation Experiment

Tests whether L11.H4 (the strongest phenomenal head) is causally necessary
for phenomenal/functional discrimination. Uses head_mask for clean ablation.

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers])" \
        --run "python3 scripts/causal_head_ablation.py"
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


def create_head_mask(
    n_layers: int,
    n_heads: int,
    ablate_layer: int = None,
    ablate_head: int = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Create a head mask that zeros out specific heads."""
    # Shape: (n_layers, n_heads)
    mask = torch.ones(n_layers, n_heads, device=device)

    if ablate_layer is not None and ablate_head is not None:
        mask[ablate_layer, ablate_head] = 0.0

    return mask


def create_layer_mask(
    n_layers: int, n_heads: int, ablate_layer: int, device: str = "cpu"
) -> torch.Tensor:
    """Create a mask that zeros out all heads in a layer."""
    mask = torch.ones(n_layers, n_heads, device=device)
    mask[ablate_layer, :] = 0.0
    return mask


def compute_layer_activation_diff(
    model,
    tokenizer,
    text: str,
    layer_idx: int,
    head_mask: torch.Tensor = None,
    device: str = "cpu",
) -> np.ndarray:
    """Extract layer activations with optional head mask."""
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, head_mask=head_mask)

    # Mean-pooled activations from target layer
    hidden = outputs.hidden_states[layer_idx + 1]
    attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

    return pooled.squeeze(0).cpu().numpy()


def compute_discrimination(
    phen_acts: List[np.ndarray], func_acts: List[np.ndarray]
) -> float:
    """Compute Fisher's criterion discrimination."""
    phen_centroid = np.mean(phen_acts, axis=0)
    func_centroid = np.mean(func_acts, axis=0)

    centroid_dist = np.linalg.norm(phen_centroid - func_centroid)

    phen_var = np.mean([np.linalg.norm(a - phen_centroid) for a in phen_acts])
    func_var = np.mean([np.linalg.norm(a - func_centroid) for a in func_acts])
    avg_var = (phen_var + func_var) / 2

    if avg_var > 0:
        return float(centroid_dist / avg_var)
    return 0.0


def run_analysis(
    model,
    tokenizer,
    layer_idx: int,
    head_mask: torch.Tensor,
    device: str,
    condition: str,
) -> Dict:
    """Run discrimination analysis with optional head mask."""
    phen_acts = []
    func_acts = []

    for text in PHENOMENAL_CONCEPTS:
        act = compute_layer_activation_diff(
            model, tokenizer, text, layer_idx, head_mask, device
        )
        phen_acts.append(act)

    for text in FUNCTIONAL_CONCEPTS:
        act = compute_layer_activation_diff(
            model, tokenizer, text, layer_idx, head_mask, device
        )
        func_acts.append(act)

    discrim = compute_discrimination(phen_acts, func_acts)

    return {
        "condition": condition,
        "discrimination": discrim,
    }


def main():
    print("\n" + "=" * 70)
    print("   CAUSAL HEAD ABLATION EXPERIMENT")
    print("   Testing if L11.H4 is causally necessary")
    print("=" * 70 + "\n")

    model_name = "bert-base-uncased"
    target_layer = 10  # Layer 11 (0-indexed)
    target_head = 3  # Head 4 (0-indexed)

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    print(f"Device: {device}")
    print(f"Model: {n_layers} layers, {n_heads} heads")
    print(f"Target head: Layer {target_layer + 1}, Head {target_head + 1}\n")

    results = []

    # 1. Baseline (no ablation)
    print("Running baseline (no ablation)...")
    baseline = run_analysis(model, tokenizer, target_layer, None, device, "baseline")
    results.append(baseline)
    print(f"  Baseline discrimination: {baseline['discrimination']:.4f}")

    # 2. Ablate target head (L11.H4)
    print(
        f"\nAblating L{target_layer + 1}.H{target_head + 1} (strongest phenomenal head)..."
    )
    mask_target = create_head_mask(n_layers, n_heads, target_layer, target_head, device)
    ablated_target = run_analysis(
        model,
        tokenizer,
        target_layer,
        mask_target,
        device,
        f"ablate_L{target_layer+1}_H{target_head+1}",
    )
    results.append(ablated_target)
    print(f"  After ablation: {ablated_target['discrimination']:.4f}")

    # 3. Ablate other L11 phenomenal heads (H1, H2, H9)
    other_phenomenal_heads = [0, 1, 8]  # H1, H2, H9 (0-indexed)
    print("\nAblating other L11 phenomenal heads...")
    for head in other_phenomenal_heads:
        mask = create_head_mask(n_layers, n_heads, target_layer, head, device)
        result = run_analysis(
            model,
            tokenizer,
            target_layer,
            mask,
            device,
            f"ablate_L{target_layer+1}_H{head+1}",
        )
        results.append(result)
        print(f"  L11.H{head+1}: {result['discrimination']:.4f}")

    # 4. Ablate control heads (non-phenomenal)
    control_heads = [(5, 5), (2, 7), (8, 10)]  # Random heads from other layers
    print("\nAblating control heads...")
    for layer, head in control_heads:
        mask = create_head_mask(n_layers, n_heads, layer, head, device)
        result = run_analysis(
            model, tokenizer, target_layer, mask, device, f"ablate_L{layer+1}_H{head+1}"
        )
        results.append(result)
        print(f"  L{layer+1}.H{head+1}: {result['discrimination']:.4f}")

    # 5. Ablate all of L11
    print("\nAblating ALL L11 heads...")
    mask_all_l11 = create_layer_mask(n_layers, n_heads, target_layer, device)
    ablated_all_l11 = run_analysis(
        model, tokenizer, target_layer, mask_all_l11, device, "ablate_all_L11"
    )
    results.append(ablated_all_l11)
    print(f"  After ablating all L11: {ablated_all_l11['discrimination']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("   ABLATION RESULTS SUMMARY")
    print("=" * 70 + "\n")

    print(f"{'Condition':<30} {'Discrimination':<15} {'Change':<12}")
    print("-" * 57)

    baseline_d = baseline["discrimination"]
    for r in results:
        change = r["discrimination"] - baseline_d
        change_str = f"{change:+.4f}" if r["condition"] != "baseline" else "---"
        print(f"{r['condition']:<30} {r['discrimination']:<15.4f} {change_str:<12}")

    # Key finding
    print("\n" + "=" * 70)
    print("   KEY FINDING")
    print("=" * 70 + "\n")

    target_reduction = baseline_d - ablated_target["discrimination"]
    control_reductions = [
        baseline_d - r["discrimination"]
        for r in results
        if r["condition"].startswith("ablate_L") and "L11" not in r["condition"]
    ]
    mean_control_reduction = np.mean(control_reductions) if control_reductions else 0

    phenomenal_head_reductions = [
        baseline_d - r["discrimination"] for r in results if "L11_H" in r["condition"]
    ]
    mean_phenomenal_reduction = np.mean(phenomenal_head_reductions)

    print(f"  Baseline discrimination: {baseline_d:.4f}")
    print(f"  After L11.H4 ablation: {ablated_target['discrimination']:.4f}")
    print(f"  L11.H4 reduction: {target_reduction:.4f}")
    print(f"  Mean L11 head reduction: {mean_phenomenal_reduction:.4f}")
    print(f"  Mean control reduction: {mean_control_reduction:.4f}")

    if target_reduction > mean_control_reduction * 1.5:
        print("\n  CAUSAL EVIDENCE FOUND")
        print(f"  L11.H4 ablation reduces discrimination more than controls")
        print(f"  ({target_reduction:.4f} vs {mean_control_reduction:.4f})")
    else:
        print("\n  DISTRIBUTED CAUSALITY")
        print(f"  L11.H4 is not uniquely causal")

    all_l11_reduction = baseline_d - ablated_all_l11["discrimination"]
    print(f"\n  Full L11 ablation reduction: {all_l11_reduction:.4f}")

    if all_l11_reduction > target_reduction * 2:
        print("  Phenomenal processing is DISTRIBUTED across L11")
        print("  Multiple heads contribute to the phenomenal effect")

    # Save results
    output = {
        "model": model_name,
        "target_layer": target_layer + 1,
        "target_head": target_head + 1,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "results": results,
        "analysis": {
            "baseline": float(baseline_d),
            "target_ablation": float(ablated_target["discrimination"]),
            "target_reduction": float(target_reduction),
            "mean_control_reduction": float(mean_control_reduction),
            "mean_phenomenal_head_reduction": float(mean_phenomenal_reduction),
            "all_l11_reduction": float(all_l11_reduction),
        },
    }

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/causal_head_ablation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
