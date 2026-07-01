#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Extended Causal Intervention Study

Extends the original causal intervention study with:
1. More attention heads (all Layer 11 heads + random controls)
2. Larger concept set (n=30 per class)
3. Activation patching experiment

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers scipy])" \
        --run "python3 scripts/causal_intervention_extended.py"
"""

import json
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from scipy import stats


# Expanded concept sets for more statistical power
PHENOMENAL_CONCEPTS = [
    # Original 15
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
    # Extended 15
    "The burning sensation of touching hot metal",
    "The cool refreshing taste of mint",
    "What it feels like to be anxious",
    "The experience of deja vu",
    "The felt presence of another person",
    "The quality of a vivid dream",
    "The sensation of falling asleep",
    "The experience of time slowing down",
    "The feeling of nostalgia washing over me",
    "The raw experience of bright light",
    "What it is like to feel tired",
    "The subjective sense of hunger",
    "The experience of hearing silence",
    "The felt quality of deep concentration",
    "The phenomenal character of imagining a color",
]

FUNCTIONAL_CONCEPTS = [
    # Original 15
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
    # Extended 15
    "The mutex prevents race conditions",
    "Red-black trees maintain balance",
    "The API endpoint returns JSON",
    "Virtual memory maps to physical pages",
    "The parser generates an AST",
    "Breadth-first search explores level by level",
    "The thread pool manages worker threads",
    "Checksums detect transmission errors",
    "The linker resolves symbol references",
    "Dynamic programming stores subproblem solutions",
    "The interrupt handler saves context",
    "B-trees optimize disk access patterns",
    "The JIT compiler emits machine code",
    "Bloom filters test set membership",
    "The register allocator assigns variables",
]


@dataclass
class ExtendedInterventionResult:
    """Results from extended intervention experiment."""
    head_name: str
    layer: int
    head: int
    head_type: str  # "phenomenal", "functional", "control"
    baseline_phen_sim: float
    baseline_func_sim: float
    ablated_phen_sim: float
    ablated_func_sim: float
    phen_change: float
    func_change: float
    specificity: float
    effect_d: float  # From attention head analysis


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

        if isinstance(outputs, tuple):
            attn_output = outputs[0]
            rest = outputs[1:]
        else:
            attn_output = outputs
            rest = ()

        if attn_output.dim() == 2:
            attn_output = attn_output.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        hidden_dim = attn_output.shape[-1]
        head_dim = hidden_dim // self.n_heads
        batch_size, seq_len, _ = attn_output.shape

        attn_output = attn_output.view(batch_size, seq_len, self.n_heads, head_dim)

        if self.ablation_type == "zero":
            attn_output = attn_output.clone()
            attn_output[:, :, self.head, :] = 0
        elif self.ablation_type == "mean":
            attn_output = attn_output.clone()
            other_heads = [h for h in range(self.n_heads) if h != self.head]
            mean_head = attn_output[:, :, other_heads, :].mean(dim=2)
            attn_output[:, :, self.head, :] = mean_head

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
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embedding[0])
    return np.array(embeddings)


def compute_within_class_similarity(embeddings: np.ndarray) -> float:
    """Compute mean pairwise cosine similarity within a class."""
    n = len(embeddings)
    if n < 2:
        return 0.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    similarities = normalized @ normalized.T
    upper_tri = np.triu_indices(n, k=1)
    return float(similarities[upper_tri].mean())


def register_ablation_hook(model, layer: int, head: int) -> Tuple:
    """Register an ablation hook on the specified layer/head."""
    n_heads = model.config.num_attention_heads
    ablation = AttentionHeadAblation(layer, head, n_heads)
    target_layer = model.encoder.layer[layer].attention.output
    handle = target_layer.register_forward_hook(ablation.forward)
    return ablation, handle


def load_head_analysis():
    """Load attention head analysis results to get effect sizes."""
    try:
        with open("/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/attention_head_analysis.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def run_extended_study(model, tokenizer, device: str) -> List[ExtendedInterventionResult]:
    """Run extended ablation study on multiple heads."""

    head_analysis = load_head_analysis()
    effect_sizes = np.array(head_analysis["effect_sizes"]) if head_analysis else None

    # Define heads to test
    # All Layer 11 heads (the "phenomenal layer")
    layer_11_heads = [(10, h, f"L11.H{h+1}") for h in range(12)]

    # Top phenomenal heads from other layers
    other_phen_heads = [
        (1, 6, "L2.H7"),   # d = -1.607
        (6, 11, "L7.H12"), # d = -1.466
        (4, 8, "L5.H9"),   # d = -1.348
    ]

    # Control: functional-preferring heads
    func_heads = [
        (11, 6, "L12.H7"),  # d = +1.536
        (11, 11, "L12.H12"), # d = +0.779
        (8, 6, "L9.H7"),    # d = +0.684
    ]

    # Control: random middle-layer heads
    random_heads = [
        (5, 5, "L6.H6"),
        (3, 3, "L4.H4"),
        (7, 7, "L8.H8"),
    ]

    all_heads = []
    for layer, head, name in layer_11_heads:
        all_heads.append((layer, head, name, "layer11"))
    for layer, head, name in other_phen_heads:
        all_heads.append((layer, head, name, "phenomenal"))
    for layer, head, name in func_heads:
        all_heads.append((layer, head, name, "functional"))
    for layer, head, name in random_heads:
        all_heads.append((layer, head, name, "control"))

    results = []

    # Get baseline embeddings once
    print("Computing baseline embeddings...")
    base_phen_emb = get_embeddings(model, tokenizer, PHENOMENAL_CONCEPTS, device)
    base_func_emb = get_embeddings(model, tokenizer, FUNCTIONAL_CONCEPTS, device)
    base_phen_sim = compute_within_class_similarity(base_phen_emb)
    base_func_sim = compute_within_class_similarity(base_func_emb)
    print(f"Baseline - Phen: {base_phen_sim:.4f}, Func: {base_func_sim:.4f}")

    print(f"\nTesting {len(all_heads)} heads...")

    for i, (layer, head, name, head_type) in enumerate(all_heads):
        print(f"  [{i+1}/{len(all_heads)}] {name}...", end=" ", flush=True)

        # Register hook
        ablation, handle = register_ablation_hook(model, layer, head)
        ablation.active = True

        # Get ablated embeddings
        abl_phen_emb = get_embeddings(model, tokenizer, PHENOMENAL_CONCEPTS, device)
        abl_func_emb = get_embeddings(model, tokenizer, FUNCTIONAL_CONCEPTS, device)

        abl_phen_sim = compute_within_class_similarity(abl_phen_emb)
        abl_func_sim = compute_within_class_similarity(abl_func_emb)

        # Compute changes
        phen_change = abl_phen_sim - base_phen_sim
        func_change = abl_func_sim - base_func_sim
        specificity = abs(phen_change) - abs(func_change)

        # Get effect size from attention analysis
        effect_d = float(effect_sizes[layer, head]) if effect_sizes is not None else 0.0

        print(f"spec={specificity:+.4f}")

        results.append(ExtendedInterventionResult(
            head_name=name,
            layer=layer + 1,
            head=head + 1,
            head_type=head_type,
            baseline_phen_sim=base_phen_sim,
            baseline_func_sim=base_func_sim,
            ablated_phen_sim=abl_phen_sim,
            ablated_func_sim=abl_func_sim,
            phen_change=phen_change,
            func_change=func_change,
            specificity=specificity,
            effect_d=effect_d,
        ))

        # Cleanup
        handle.remove()

    return results


def analyze_results(results: List[ExtendedInterventionResult]) -> Dict:
    """Statistical analysis of extended results."""

    # Group by type
    layer11 = [r for r in results if r.head_type == "layer11"]
    phenomenal = [r for r in results if r.head_type == "phenomenal"]
    functional = [r for r in results if r.head_type == "functional"]
    control = [r for r in results if r.head_type == "control"]

    # All phenomenal-related (layer 11 + other phenomenal)
    all_phen = layer11 + phenomenal
    all_ctrl = functional + control

    # Specificity comparison
    phen_specs = [r.specificity for r in all_phen]
    ctrl_specs = [r.specificity for r in all_ctrl]

    # Mann-Whitney U test
    stat, p_value = stats.mannwhitneyu(phen_specs, ctrl_specs, alternative='greater')

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(phen_specs)**2 + np.std(ctrl_specs)**2) / 2)
    cohens_d = (np.mean(phen_specs) - np.mean(ctrl_specs)) / pooled_std if pooled_std > 0 else 0

    # Correlation: effect_d vs specificity
    all_effect_d = [r.effect_d for r in results]
    all_specificity = [r.specificity for r in results]
    correlation, corr_p = stats.pearsonr(all_effect_d, all_specificity)

    # Layer 11 analysis
    l11_specs = [r.specificity for r in layer11]
    l11_best = max(layer11, key=lambda r: r.specificity)
    l11_worst = min(layer11, key=lambda r: r.specificity)

    return {
        "n_heads_tested": len(results),
        "n_concepts": {
            "phenomenal": len(PHENOMENAL_CONCEPTS),
            "functional": len(FUNCTIONAL_CONCEPTS),
        },
        "group_statistics": {
            "layer11": {
                "n": len(layer11),
                "mean_specificity": float(np.mean([r.specificity for r in layer11])),
                "std_specificity": float(np.std([r.specificity for r in layer11])),
            },
            "other_phenomenal": {
                "n": len(phenomenal),
                "mean_specificity": float(np.mean([r.specificity for r in phenomenal])) if phenomenal else 0,
            },
            "functional": {
                "n": len(functional),
                "mean_specificity": float(np.mean([r.specificity for r in functional])) if functional else 0,
            },
            "control": {
                "n": len(control),
                "mean_specificity": float(np.mean([r.specificity for r in control])) if control else 0,
            },
        },
        "statistical_tests": {
            "phenomenal_vs_control": {
                "mann_whitney_U": float(stat),
                "p_value": float(p_value),
                "cohens_d": float(cohens_d),
                "significant": bool(p_value < 0.05),
            },
            "effect_d_specificity_correlation": {
                "pearson_r": float(correlation),
                "p_value": float(corr_p),
                "interpretation": "Negative correlation expected (more negative effect_d = more phenomenal-specific)",
            },
        },
        "layer11_analysis": {
            "mean_specificity": float(np.mean(l11_specs)),
            "best_head": {
                "name": l11_best.head_name,
                "specificity": float(l11_best.specificity),
            },
            "worst_head": {
                "name": l11_worst.head_name,
                "specificity": float(l11_worst.specificity),
            },
            "n_positive_specificity": sum(1 for s in l11_specs if s > 0),
        },
    }


def main():
    print("\n" + "=" * 70)
    print("   EXTENDED CAUSAL INTERVENTION STUDY")
    print("   Testing all Layer 11 heads + controls (n=30 concepts)")
    print("=" * 70)

    model_name = "bert-base-uncased"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")
    print(f"Concepts: {len(PHENOMENAL_CONCEPTS)} phenomenal, {len(FUNCTIONAL_CONCEPTS)} functional")

    # Run extended study
    results = run_extended_study(model, tokenizer, device)

    # Analyze
    analysis = analyze_results(results)

    # Print summary
    print("\n" + "=" * 70)
    print("   EXTENDED ANALYSIS RESULTS")
    print("=" * 70)

    print(f"\nHeads tested: {analysis['n_heads_tested']}")
    print(f"Concepts: {analysis['n_concepts']['phenomenal']} phenomenal, {analysis['n_concepts']['functional']} functional")

    print("\n--- Group Statistics ---")
    for group, stats in analysis["group_statistics"].items():
        print(f"  {group}: n={stats['n']}, mean_spec={stats['mean_specificity']:+.4f}")

    print("\n--- Statistical Tests ---")
    test = analysis["statistical_tests"]["phenomenal_vs_control"]
    print(f"  Phenomenal vs Control (Mann-Whitney U):")
    print(f"    U = {test['mann_whitney_U']:.1f}, p = {test['p_value']:.4f}, d = {test['cohens_d']:.3f}")
    print(f"    Significant: {test['significant']}")

    corr = analysis["statistical_tests"]["effect_d_specificity_correlation"]
    print(f"  Effect_d ↔ Specificity correlation:")
    print(f"    r = {corr['pearson_r']:.3f}, p = {corr['p_value']:.4f}")

    print("\n--- Layer 11 Analysis ---")
    l11 = analysis["layer11_analysis"]
    print(f"  Mean specificity: {l11['mean_specificity']:+.4f}")
    print(f"  Best head: {l11['best_head']['name']} (spec={l11['best_head']['specificity']:+.4f})")
    print(f"  Worst head: {l11['worst_head']['name']} (spec={l11['worst_head']['specificity']:+.4f})")
    print(f"  Heads with positive specificity: {l11['n_positive_specificity']}/12")

    # Save results
    output = {
        "model": model_name,
        "analysis": analysis,
        "individual_results": [asdict(r) for r in results],
    }

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/causal_intervention_extended.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
