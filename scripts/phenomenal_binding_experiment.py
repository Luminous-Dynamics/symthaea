#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Phenomenal Binding via HDC: Testing H2

Hypothesis H2: HDC binding (⊗) produces representations with higher
topological integration (lower component count, higher unity score)
than bundling (⊕), specifically for concept pairs humans report as
phenomenally unified.

Key insight: Binding (XOR) creates NEW orthogonal vectors from inputs,
while bundling (majority vote) creates superposition. If binding produces
measurably higher unity for unified percepts, we have computational
evidence for binding's role in phenomenal unity.

Experimental Design:
- Create human-validated concept pairs
  - Unified: (red, apple) → "red apple" (perceptually unified)
  - Co-occurring: (red, mailbox) → less unified
- For each pair: compute bind(a,b) and bundle([a,b])
- Measure "unity" via similarity structure and clustering
- 2x2 ANOVA: Operation (Bind/Bundle) × Pair Type (Unified/Separate)

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy scipy])" \
        --run "python3 scripts/phenomenal_binding_experiment.py"
"""

import json
import hashlib
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy import stats


# =============================================================================
# HDC Operations (Binary Hypervectors)
# =============================================================================

DIM = 16384  # Match Symthaea HDC dimension


def random_hv(seed: int) -> np.ndarray:
    """Generate deterministic random binary hypervector from seed."""
    # Use BLAKE3-like hash expansion
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(0, 2, size=DIM, dtype=np.uint8)


def concept_hv(concept: str) -> np.ndarray:
    """Generate deterministic HV for a concept string."""
    # Hash concept to get seed
    h = hashlib.sha256(concept.encode()).hexdigest()
    seed = int(h[:16], 16)
    return random_hv(seed)


def bind_hv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bind two HVs via XOR (creates orthogonal combination)."""
    return np.logical_xor(a, b).astype(np.uint8)


def bundle_hv(vectors: List[np.ndarray]) -> np.ndarray:
    """Bundle HVs via majority vote (creates superposition)."""
    if len(vectors) == 0:
        return np.zeros(DIM, dtype=np.uint8)

    # Sum bits
    counts = np.sum(vectors, axis=0)
    threshold = len(vectors) / 2

    # Majority vote
    return (counts > threshold).astype(np.uint8)


def hamming_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute similarity as 1 - normalized Hamming distance."""
    diff = np.sum(a != b)
    return 1.0 - diff / DIM


def bipolar(hv: np.ndarray) -> np.ndarray:
    """Convert binary {0,1} to bipolar {-1,+1}."""
    return 2 * hv.astype(np.float32) - 1


# =============================================================================
# Unity Metrics
# =============================================================================

def compute_unity_metrics(hv: np.ndarray) -> dict:
    """
    Compute unity-related metrics for a single HV.

    Higher unity = more "unified" representation.
    """
    bp = bipolar(hv)

    # 1. Density (balance between 0s and 1s)
    density = np.mean(hv)
    density_balance = 1 - abs(density - 0.5) * 2  # 1 = perfectly balanced

    # 2. Autocorrelation (self-similarity at different shifts)
    # Higher autocorrelation = more structured/unified
    autocorr = []
    for shift in [16, 64, 256, 1024]:
        shifted = np.roll(bp, shift)
        corr = np.corrcoef(bp, shifted)[0, 1]
        autocorr.append(corr if not np.isnan(corr) else 0)
    avg_autocorr = np.mean(autocorr)

    # 3. Block coherence (do nearby bits agree?)
    block_size = 64
    n_blocks = DIM // block_size
    block_means = [np.mean(hv[i*block_size:(i+1)*block_size]) for i in range(n_blocks)]
    block_variance = np.var(block_means)
    block_coherence = 1 - min(block_variance * 4, 1)  # Normalized

    # 4. Combined unity score
    unity_score = (density_balance * 0.3 +
                   (avg_autocorr + 1) / 2 * 0.3 +
                   block_coherence * 0.4)

    return {
        "density": density,
        "density_balance": density_balance,
        "avg_autocorrelation": avg_autocorr,
        "block_coherence": block_coherence,
        "unity_score": unity_score,
    }


def compute_pair_integration(hv_a: np.ndarray, hv_b: np.ndarray,
                              bound: np.ndarray, bundled: np.ndarray) -> dict:
    """
    Compute integration metrics comparing binding vs bundling.

    Key question: Does binding create more "integrated" representation
    than bundling, especially for phenomenally unified pairs?
    """
    # Similarity structure
    bind_sim_to_a = hamming_similarity(bound, hv_a)
    bind_sim_to_b = hamming_similarity(bound, hv_b)
    bundle_sim_to_a = hamming_similarity(bundled, hv_a)
    bundle_sim_to_b = hamming_similarity(bundled, hv_b)

    # Binding creates orthogonal vector (low sim to inputs)
    # Bundling creates superposition (higher sim to inputs)
    bind_orthogonality = 1 - (bind_sim_to_a + bind_sim_to_b) / 2
    bundle_preservation = (bundle_sim_to_a + bundle_sim_to_b) / 2

    # Unity metrics for each result
    bind_unity = compute_unity_metrics(bound)
    bundle_unity = compute_unity_metrics(bundled)

    # Integration difference (positive = binding more unified)
    integration_diff = bind_unity["unity_score"] - bundle_unity["unity_score"]

    return {
        "bind_sim_to_a": bind_sim_to_a,
        "bind_sim_to_b": bind_sim_to_b,
        "bundle_sim_to_a": bundle_sim_to_a,
        "bundle_sim_to_b": bundle_sim_to_b,
        "bind_orthogonality": bind_orthogonality,
        "bundle_preservation": bundle_preservation,
        "bind_unity": bind_unity["unity_score"],
        "bundle_unity": bundle_unity["unity_score"],
        "integration_diff": integration_diff,
    }


# =============================================================================
# Concept Pairs Dataset
# =============================================================================

@dataclass
class ConceptPair:
    concept_a: str
    concept_b: str
    pair_type: str  # "unified" or "separate"
    description: str


def get_concept_pairs() -> List[ConceptPair]:
    """
    Human-validated concept pairs.

    Unified: Features that humans perceive as a single integrated object
    Separate: Features that co-occur but are perceived separately
    """
    pairs = [
        # === UNIFIED PAIRS (phenomenally bound) ===
        # Visual feature binding
        ConceptPair("red", "apple", "unified", "Red apple - color bound to object"),
        ConceptPair("blue", "sky", "unified", "Blue sky - color bound to space"),
        ConceptPair("loud", "crash", "unified", "Loud crash - volume bound to sound"),
        ConceptPair("sharp", "pain", "unified", "Sharp pain - quality bound to sensation"),
        ConceptPair("sweet", "taste", "unified", "Sweet taste - quality bound to sense"),
        ConceptPair("warm", "sunshine", "unified", "Warm sunshine - temperature bound to source"),
        ConceptPair("fast", "car", "unified", "Fast car - motion bound to object"),
        ConceptPair("bright", "light", "unified", "Bright light - intensity bound to phenomenon"),
        ConceptPair("soft", "fur", "unified", "Soft fur - texture bound to material"),
        ConceptPair("cold", "ice", "unified", "Cold ice - temperature bound to object"),

        # Conceptual binding
        ConceptPair("angry", "face", "unified", "Angry face - emotion bound to expression"),
        ConceptPair("sad", "music", "unified", "Sad music - emotion bound to medium"),
        ConceptPair("beautiful", "sunset", "unified", "Beautiful sunset - aesthetic bound to scene"),
        ConceptPair("peaceful", "garden", "unified", "Peaceful garden - quality bound to place"),
        ConceptPair("mysterious", "forest", "unified", "Mysterious forest - quality bound to place"),

        # === SEPARATE PAIRS (co-occurring but not phenomenally unified) ===
        # Spatial co-occurrence
        ConceptPair("red", "mailbox", "separate", "Red near mailbox - co-located but separate"),
        ConceptPair("loud", "background", "separate", "Loud amid background - not unified"),
        ConceptPair("blue", "fence", "separate", "Blue and fence - co-occurring"),
        ConceptPair("car", "tree", "separate", "Car and tree - near but separate"),
        ConceptPair("music", "conversation", "separate", "Music and talk - simultaneous but separate"),
        ConceptPair("pain", "thought", "separate", "Pain and thought - concurrent but distinct"),
        ConceptPair("taste", "smell", "separate", "Taste and smell - different modalities"),
        ConceptPair("light", "sound", "separate", "Light and sound - different channels"),
        ConceptPair("warmth", "texture", "separate", "Warmth and texture - separate qualia"),
        ConceptPair("color", "motion", "separate", "Color and motion - unbound features"),

        # Temporal co-occurrence
        ConceptPair("memory", "present", "separate", "Memory and present - different times"),
        ConceptPair("anticipation", "sensation", "separate", "Future and now - temporal split"),
        ConceptPair("word", "image", "separate", "Word and image - different codes"),
        ConceptPair("feeling", "reason", "separate", "Feeling and reason - different processes"),
        ConceptPair("self", "other", "separate", "Self and other - fundamental distinction"),
    ]

    return pairs


# =============================================================================
# Experiment
# =============================================================================

def run_experiment():
    """Run the phenomenal binding experiment."""
    print("\n" + "=" * 70)
    print("   PHENOMENAL BINDING EXPERIMENT")
    print("   Testing H2: Binding vs Bundling for Phenomenal Unity")
    print("=" * 70)

    pairs = get_concept_pairs()
    print(f"\nConcept pairs: {len(pairs)} total")
    print(f"  Unified: {sum(1 for p in pairs if p.pair_type == 'unified')}")
    print(f"  Separate: {sum(1 for p in pairs if p.pair_type == 'separate')}")

    results = []

    print("\nProcessing pairs...")
    for pair in pairs:
        # Generate HVs for concepts
        hv_a = concept_hv(pair.concept_a)
        hv_b = concept_hv(pair.concept_b)

        # Apply binding and bundling
        bound = bind_hv(hv_a, hv_b)
        bundled = bundle_hv([hv_a, hv_b])

        # Compute metrics
        metrics = compute_pair_integration(hv_a, hv_b, bound, bundled)

        results.append({
            "concept_a": pair.concept_a,
            "concept_b": pair.concept_b,
            "pair_type": pair.pair_type,
            "description": pair.description,
            **metrics
        })

    # Convert to numpy for analysis
    unified_results = [r for r in results if r["pair_type"] == "unified"]
    separate_results = [r for r in results if r["pair_type"] == "separate"]

    # Extract key metrics
    unified_bind_unity = [r["bind_unity"] for r in unified_results]
    unified_bundle_unity = [r["bundle_unity"] for r in unified_results]
    separate_bind_unity = [r["bind_unity"] for r in separate_results]
    separate_bundle_unity = [r["bundle_unity"] for r in separate_results]

    unified_integration_diff = [r["integration_diff"] for r in unified_results]
    separate_integration_diff = [r["integration_diff"] for r in separate_results]

    # ==========================================================================
    # Statistical Analysis
    # ==========================================================================

    print("\n" + "=" * 70)
    print("   RESULTS")
    print("=" * 70)

    # 1. Main effect of operation (bind vs bundle)
    all_bind_unity = unified_bind_unity + separate_bind_unity
    all_bundle_unity = unified_bundle_unity + separate_bundle_unity

    t_operation, p_operation = stats.ttest_rel(all_bind_unity, all_bundle_unity)

    print(f"\n1. MAIN EFFECT OF OPERATION (Bind vs Bundle)")
    print(f"   Bind mean unity:   {np.mean(all_bind_unity):.4f} ± {np.std(all_bind_unity):.4f}")
    print(f"   Bundle mean unity: {np.mean(all_bundle_unity):.4f} ± {np.std(all_bundle_unity):.4f}")
    print(f"   Paired t-test: t = {t_operation:.3f}, p = {p_operation:.4f}")

    # 2. Main effect of pair type
    unified_overall = [(u + b) / 2 for u, b in zip(unified_bind_unity, unified_bundle_unity)]
    separate_overall = [(u + b) / 2 for u, b in zip(separate_bind_unity, separate_bundle_unity)]

    t_type, p_type = stats.ttest_ind(unified_overall, separate_overall)

    print(f"\n2. MAIN EFFECT OF PAIR TYPE (Unified vs Separate)")
    print(f"   Unified pairs:  {np.mean(unified_overall):.4f} ± {np.std(unified_overall):.4f}")
    print(f"   Separate pairs: {np.mean(separate_overall):.4f} ± {np.std(separate_overall):.4f}")
    print(f"   Independent t-test: t = {t_type:.3f}, p = {p_type:.4f}")

    # 3. INTERACTION: Does binding help unified pairs MORE than separate pairs?
    # Integration difference = bind_unity - bundle_unity
    # H2 predicts: unified pairs should have LARGER positive integration_diff

    t_interaction, p_interaction = stats.ttest_ind(unified_integration_diff, separate_integration_diff)

    print(f"\n3. INTERACTION EFFECT (Key H2 test)")
    print(f"   Unified integration diff:  {np.mean(unified_integration_diff):.4f} ± {np.std(unified_integration_diff):.4f}")
    print(f"   Separate integration diff: {np.mean(separate_integration_diff):.4f} ± {np.std(separate_integration_diff):.4f}")
    print(f"   Independent t-test: t = {t_interaction:.3f}, p = {p_interaction:.4f}")

    # Cohen's d for effect size
    pooled_std = np.sqrt((np.var(unified_integration_diff) + np.var(separate_integration_diff)) / 2)
    cohens_d = (np.mean(unified_integration_diff) - np.mean(separate_integration_diff)) / pooled_std if pooled_std > 0 else 0

    print(f"   Cohen's d: {cohens_d:.3f}")

    # 4. Orthogonality analysis (binding creates more distinct representations)
    unified_orthogonality = np.mean([r["bind_orthogonality"] for r in unified_results])
    separate_orthogonality = np.mean([r["bind_orthogonality"] for r in separate_results])

    print(f"\n4. BINDING ORTHOGONALITY")
    print(f"   Unified pairs:  {unified_orthogonality:.4f}")
    print(f"   Separate pairs: {separate_orthogonality:.4f}")

    # ==========================================================================
    # Key Finding
    # ==========================================================================

    print("\n" + "=" * 70)
    print("   KEY FINDING")
    print("=" * 70)

    if p_interaction < 0.05 and np.mean(unified_integration_diff) > np.mean(separate_integration_diff):
        finding = "H2_SUPPORTED"
        print("\n   ✓ H2 SUPPORTED: Binding produces higher integration")
        print("     specifically for phenomenally unified concept pairs.")
    elif p_operation < 0.05 and np.mean(all_bind_unity) > np.mean(all_bundle_unity):
        finding = "MAIN_EFFECT_ONLY"
        print("\n   ~ PARTIAL SUPPORT: Binding produces higher unity overall,")
        print("     but no specific advantage for unified pairs.")
    else:
        finding = "H2_NOT_SUPPORTED"
        print("\n   ✗ H2 NOT SUPPORTED: No significant binding advantage")
        print("     for phenomenally unified pairs.")

    print(f"\n   Statistical summary:")
    print(f"   - Interaction p-value: {p_interaction:.4f} {'(significant)' if p_interaction < 0.05 else '(not significant)'}")
    print(f"   - Effect size (d): {cohens_d:.3f} {'(large)' if abs(cohens_d) > 0.8 else '(medium)' if abs(cohens_d) > 0.5 else '(small)'}")

    # ==========================================================================
    # Save Results
    # ==========================================================================

    output = {
        "experiment": "phenomenal_binding",
        "hypothesis": "H2: Binding produces higher integration than bundling for unified pairs",
        "n_unified_pairs": len(unified_results),
        "n_separate_pairs": len(separate_results),
        "results": {
            "main_effect_operation": {
                "bind_mean": float(np.mean(all_bind_unity)),
                "bind_std": float(np.std(all_bind_unity)),
                "bundle_mean": float(np.mean(all_bundle_unity)),
                "bundle_std": float(np.std(all_bundle_unity)),
                "t_statistic": float(t_operation),
                "p_value": float(p_operation),
            },
            "main_effect_pair_type": {
                "unified_mean": float(np.mean(unified_overall)),
                "separate_mean": float(np.mean(separate_overall)),
                "t_statistic": float(t_type),
                "p_value": float(p_type),
            },
            "interaction": {
                "unified_integration_diff": float(np.mean(unified_integration_diff)),
                "unified_integration_std": float(np.std(unified_integration_diff)),
                "separate_integration_diff": float(np.mean(separate_integration_diff)),
                "separate_integration_std": float(np.std(separate_integration_diff)),
                "t_statistic": float(t_interaction),
                "p_value": float(p_interaction),
                "cohens_d": float(cohens_d),
            },
            "orthogonality": {
                "unified_mean": float(unified_orthogonality),
                "separate_mean": float(separate_orthogonality),
            },
        },
        "finding": finding,
        "pair_results": results,
    }

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/phenomenal_binding_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")

    return output


if __name__ == "__main__":
    run_experiment()
