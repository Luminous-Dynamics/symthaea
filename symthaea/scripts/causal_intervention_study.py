#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Causal Intervention Study: L11.H4 Ablation

Tests the causal role of identified phenomenal-discriminating attention heads
by ablating them and measuring changes in phenomenal vs functional discrimination.

Key Finding from attention_head_analysis.py:
- L11.H4 has the strongest phenomenal effect (d = -2.629)
- Layer 11 is the "phenomenal layer" (mean d = -0.844)

Experiment Design:
1. Baseline: Measure embedding similarity structure with intact model
2. Ablation: Zero out L11.H4 attention weights
3. Compare: Does ablation reduce phenomenal clustering more than functional?

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers scipy])" \
        --run "python3 scripts/causal_intervention_study.py"
"""

import json
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from scipy.stats import mannwhitneyu, permutation_test


# Same concept sets as attention_head_analysis.py
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
    "The warmth of sunlight on skin",
    "The sweet taste of honey",
    "The smell of fresh rain",
    "The sound of a bell ringing",
    "The texture of silk",
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
    "The stack frame stores local variables",
    "Quicksort partitions the array",
    "The cache reduces memory latency",
    "Type inference deduces annotations",
    "The scheduler allocates CPU time",
]


@dataclass
class InterventionResult:
    """Results from a single intervention experiment."""
    intervention_name: str
    baseline_phen_sim: float
    baseline_func_sim: float
    baseline_cross_sim: float
    ablated_phen_sim: float
    ablated_func_sim: float
    ablated_cross_sim: float
    phen_change: float  # ablated - baseline (negative = reduced clustering)
    func_change: float
    specificity: float  # |phen_change| - |func_change| (positive = targeted effect)


class AttentionHeadAblation(nn.Module):
    """Hook to ablate specific attention heads."""

    def __init__(self, layer: int, head: int, n_heads: int, ablation_type: str = "zero"):
        super().__init__()
        self.layer = layer
        self.head = head
        self.n_heads = n_heads
        self.ablation_type = ablation_type
        self.active = False

    def forward(self, module, inputs, outputs):
        if not self.active:
            return outputs

        # The BertSelfOutput module receives (hidden_states, input_tensor)
        # and returns hidden_states after dense + LayerNorm + residual
        # We intercept the output which is (batch, seq, hidden)

        # Handle both tensor and tuple outputs
        if isinstance(outputs, tuple):
            attn_output = outputs[0]
            rest = outputs[1:]
        else:
            attn_output = outputs
            rest = ()

        # Check shape - should be (batch, seq, hidden)
        if attn_output.dim() == 2:
            # (seq, hidden) - single sample without batch
            attn_output = attn_output.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        hidden_dim = attn_output.shape[-1]
        head_dim = hidden_dim // self.n_heads
        batch_size, seq_len, _ = attn_output.shape

        # Reshape to separate heads
        attn_output = attn_output.view(batch_size, seq_len, self.n_heads, head_dim)

        if self.ablation_type == "zero":
            attn_output = attn_output.clone()
            attn_output[:, :, self.head, :] = 0
        elif self.ablation_type == "mean":
            # Replace with mean of other heads
            attn_output = attn_output.clone()
            other_heads = [h for h in range(self.n_heads) if h != self.head]
            mean_head = attn_output[:, :, other_heads, :].mean(dim=2)
            attn_output[:, :, self.head, :] = mean_head

        # Reshape back
        attn_output = attn_output.view(batch_size, seq_len, hidden_dim)

        if squeeze_back:
            attn_output = attn_output.squeeze(0)

        if rest:
            return (attn_output,) + rest
        else:
            return attn_output


def get_embeddings(model, tokenizer, texts: List[str], device: str) -> np.ndarray:
    """Get [CLS] embeddings for a list of texts."""
    embeddings = []

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embedding[0])

    return np.array(embeddings)


def compute_within_class_similarity(embeddings: np.ndarray) -> float:
    """Compute mean pairwise cosine similarity within a class."""
    n = len(embeddings)
    if n < 2:
        return 0.0

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)

    # Compute all pairwise similarities
    similarities = normalized @ normalized.T

    # Extract upper triangle (excluding diagonal)
    upper_tri = np.triu_indices(n, k=1)
    return float(similarities[upper_tri].mean())


def compute_cross_class_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute mean cosine similarity between two embedding sets."""
    # Normalize
    norm1 = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
    norm2 = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)

    # Compute all pairwise similarities
    similarities = norm1 @ norm2.T
    return float(similarities.mean())


def register_ablation_hook(model, layer: int, head: int) -> Tuple[AttentionHeadAblation, any]:
    """Register an ablation hook on the specified layer/head."""
    n_heads = model.config.num_attention_heads
    ablation = AttentionHeadAblation(layer, head, n_heads)

    # Get the attention output layer
    # BERT structure: encoder.layer[i].attention.output
    target_layer = model.encoder.layer[layer].attention.output
    handle = target_layer.register_forward_hook(ablation.forward)

    return ablation, handle


def run_intervention(
    model,
    tokenizer,
    device: str,
    layer: int,
    head: int,
    name: str
) -> InterventionResult:
    """Run a single intervention experiment."""

    print(f"\n--- Intervention: {name} (L{layer+1}.H{head+1}) ---")

    # Register ablation hook
    ablation, handle = register_ablation_hook(model, layer, head)

    # Baseline (ablation inactive)
    ablation.active = False
    phen_emb_base = get_embeddings(model, tokenizer, PHENOMENAL_CONCEPTS, device)
    func_emb_base = get_embeddings(model, tokenizer, FUNCTIONAL_CONCEPTS, device)

    base_phen_sim = compute_within_class_similarity(phen_emb_base)
    base_func_sim = compute_within_class_similarity(func_emb_base)
    base_cross_sim = compute_cross_class_similarity(phen_emb_base, func_emb_base)

    print(f"  Baseline - Phen sim: {base_phen_sim:.4f}, Func sim: {base_func_sim:.4f}, Cross: {base_cross_sim:.4f}")

    # Ablated (ablation active)
    ablation.active = True
    phen_emb_abl = get_embeddings(model, tokenizer, PHENOMENAL_CONCEPTS, device)
    func_emb_abl = get_embeddings(model, tokenizer, FUNCTIONAL_CONCEPTS, device)

    abl_phen_sim = compute_within_class_similarity(phen_emb_abl)
    abl_func_sim = compute_within_class_similarity(func_emb_abl)
    abl_cross_sim = compute_cross_class_similarity(phen_emb_abl, func_emb_abl)

    print(f"  Ablated  - Phen sim: {abl_phen_sim:.4f}, Func sim: {abl_func_sim:.4f}, Cross: {abl_cross_sim:.4f}")

    # Compute changes
    phen_change = abl_phen_sim - base_phen_sim
    func_change = abl_func_sim - base_func_sim
    specificity = abs(phen_change) - abs(func_change)

    print(f"  Changes  - Phen: {phen_change:+.4f}, Func: {func_change:+.4f}, Specificity: {specificity:+.4f}")

    # Cleanup hook
    handle.remove()

    return InterventionResult(
        intervention_name=name,
        baseline_phen_sim=base_phen_sim,
        baseline_func_sim=base_func_sim,
        baseline_cross_sim=base_cross_sim,
        ablated_phen_sim=abl_phen_sim,
        ablated_func_sim=abl_func_sim,
        ablated_cross_sim=abl_cross_sim,
        phen_change=phen_change,
        func_change=func_change,
        specificity=specificity,
    )


def run_permutation_test(results: List[InterventionResult], n_permutations: int = 10000) -> Dict:
    """Run permutation test to assess significance of phenomenal-specific effects."""

    phen_changes = np.array([r.phen_change for r in results])
    func_changes = np.array([r.func_change for r in results])

    # Observed statistic: mean difference in absolute changes
    observed_diff = np.mean(np.abs(phen_changes)) - np.mean(np.abs(func_changes))

    # Permutation test
    combined = np.concatenate([phen_changes, func_changes])
    n = len(phen_changes)

    count_extreme = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_phen = combined[:n]
        perm_func = combined[n:]
        perm_diff = np.mean(np.abs(perm_phen)) - np.mean(np.abs(perm_func))
        if perm_diff >= observed_diff:
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    return {
        "observed_diff": float(observed_diff),
        "p_value": p_value,
        "n_permutations": n_permutations,
        "significant": p_value < 0.05,
    }


def main():
    print("\n" + "=" * 70)
    print("   CAUSAL INTERVENTION STUDY")
    print("   Testing L11.H4 as phenomenal circuit candidate")
    print("=" * 70)

    model_name = "bert-base-uncased"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"Layers: {n_layers}, Heads: {n_heads}")

    # Define interventions to test
    # Primary target: L11.H4 (strongest phenomenal effect)
    # Controls: random heads, functional-preferring heads
    interventions = [
        # Primary target
        (10, 3, "L11.H4 (Primary Phenomenal)"),  # 0-indexed: layer 10, head 3
        # Other top phenomenal heads
        (10, 0, "L11.H1 (2nd Phenomenal)"),
        (10, 8, "L11.H9 (3rd Phenomenal)"),
        # Control: functional-preferring head
        (11, 6, "L12.H7 (Top Functional)"),
        # Control: early layer head
        (1, 6, "L2.H7 (Early Layer)"),
        # Control: middle layer
        (5, 11, "L6.H12 (Middle Layer)"),
    ]

    print(f"\nRunning {len(interventions)} intervention experiments...")

    results = []
    for layer, head, name in interventions:
        result = run_intervention(model, tokenizer, device, layer, head, name)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("   SUMMARY: CAUSAL EFFECTS")
    print("=" * 70)

    print(f"\n{'Intervention':<30} {'Phen Δ':<12} {'Func Δ':<12} {'Specificity':<12}")
    print("-" * 66)

    for r in results:
        spec_marker = "***" if r.specificity > 0.01 else ""
        print(f"{r.intervention_name:<30} {r.phen_change:+.4f}      {r.func_change:+.4f}      {r.specificity:+.4f} {spec_marker}")

    # Statistical test (if we have enough interventions)
    print("\n" + "=" * 70)
    print("   KEY FINDINGS")
    print("=" * 70)

    # Compare phenomenal vs control heads
    phen_heads = [r for r in results if "Phenomenal" in r.intervention_name]
    ctrl_heads = [r for r in results if "Phenomenal" not in r.intervention_name]

    if phen_heads and ctrl_heads:
        phen_spec = np.mean([r.specificity for r in phen_heads])
        ctrl_spec = np.mean([r.specificity for r in ctrl_heads])

        print(f"\n  Phenomenal heads mean specificity: {phen_spec:+.4f}")
        print(f"  Control heads mean specificity:    {ctrl_spec:+.4f}")
        print(f"  Difference: {phen_spec - ctrl_spec:+.4f}")

        if phen_spec > ctrl_spec:
            print("\n  ✓ Phenomenal heads show MORE targeted effects than controls")
        else:
            print("\n  ✗ No specific phenomenal effect detected")

    # Primary finding about L11.H4
    l11h4 = next((r for r in results if "L11.H4" in r.intervention_name), None)
    if l11h4:
        print(f"\n  PRIMARY TARGET (L11.H4):")
        print(f"    Phenomenal clustering change: {l11h4.phen_change:+.4f}")
        print(f"    Functional clustering change: {l11h4.func_change:+.4f}")
        if l11h4.phen_change < l11h4.func_change:
            print("    → Ablating L11.H4 SPECIFICALLY reduces phenomenal clustering")
        else:
            print("    → Effect is not phenomenal-specific")

    # Save results
    output = {
        "model": model_name,
        "n_concepts": {
            "phenomenal": len(PHENOMENAL_CONCEPTS),
            "functional": len(FUNCTIONAL_CONCEPTS),
        },
        "interventions": [
            {
                "name": r.intervention_name,
                "baseline": {
                    "phenomenal_similarity": r.baseline_phen_sim,
                    "functional_similarity": r.baseline_func_sim,
                    "cross_similarity": r.baseline_cross_sim,
                },
                "ablated": {
                    "phenomenal_similarity": r.ablated_phen_sim,
                    "functional_similarity": r.ablated_func_sim,
                    "cross_similarity": r.ablated_cross_sim,
                },
                "changes": {
                    "phenomenal": r.phen_change,
                    "functional": r.func_change,
                    "specificity": r.specificity,
                },
            }
            for r in results
        ],
        "summary": {
            "phenomenal_heads_mean_specificity": float(phen_spec) if phen_heads else None,
            "control_heads_mean_specificity": float(ctrl_spec) if ctrl_heads else None,
        },
    }

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/causal_intervention_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
