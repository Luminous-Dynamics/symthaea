#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mechanistic Circuit Analysis

Investigates WHY larger models show weaker phenomenal discrimination.
Tests multiple mechanistic hypotheses:
1. Superposition hypothesis: Larger models pack more concepts into each dimension
2. Attention diffusion: Larger models spread attention more broadly
3. Layer specialization: Larger models have more diffuse processing across layers
4. Representation geometry: How phenomenal concepts are embedded geometrically

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers scikit-learn])" \
        --run "python3 scripts/mechanistic_circuit_analysis.py"
"""

import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer


def load_corpus():
    """Load expanded concept corpus."""
    corpus_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/expanded_concept_corpus.json"
    try:
        with open(corpus_path) as f:
            corpus = json.load(f)
        return corpus["phenomenal_concepts"][:15], corpus["functional_concepts"][:15]
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


def extract_all_layers(
    model, tokenizer, texts: List[str], device: str
) -> List[np.ndarray]:
    """Extract representations from ALL layers for layer-wise analysis."""
    all_layer_reps = []

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        layer_reps = []
        for hidden in outputs.hidden_states:
            attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            layer_reps.append(pooled.squeeze(0).cpu().numpy())

        all_layer_reps.append(layer_reps)

    # Reorganize: list of layers, each containing all texts
    n_layers = len(all_layer_reps[0])
    return [
        np.array([all_layer_reps[t][l] for t in range(len(texts))])
        for l in range(n_layers)
    ]


def extract_attention_patterns(
    model, tokenizer, texts: List[str], device: str
) -> List[np.ndarray]:
    """Extract attention patterns for attention analysis."""
    all_attentions = []

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # Average across heads and layers
        attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)
        avg_attn = torch.stack([a.mean(dim=1) for a in attentions]).mean(
            dim=0
        )  # (batch, seq, seq)
        all_attentions.append(avg_attn.squeeze(0).cpu().numpy())

    return all_attentions


def compute_fisher_criterion(phen_reps: np.ndarray, func_reps: np.ndarray) -> float:
    """Compute Fisher's criterion."""
    phen_centroid = np.mean(phen_reps, axis=0)
    func_centroid = np.mean(func_reps, axis=0)
    centroid_distance = np.linalg.norm(phen_centroid - func_centroid)
    phen_within = np.mean([np.linalg.norm(r - phen_centroid) for r in phen_reps])
    func_within = np.mean([np.linalg.norm(r - func_centroid) for r in func_reps])
    avg_within = (phen_within + func_within) / 2
    return centroid_distance / avg_within if avg_within > 0 else 0


def analyze_superposition(phen_reps: np.ndarray, func_reps: np.ndarray) -> Dict:
    """
    Superposition hypothesis: More concepts packed per dimension in larger models.
    Measure: Effective dimensionality via PCA explained variance.
    """
    all_reps = np.vstack([phen_reps, func_reps])

    # Compute effective dimensionality (dims needed for 90% variance)
    n_components = min(all_reps.shape[0] - 1, all_reps.shape[1])
    if n_components < 2:
        return {
            "effective_dim_90": float("nan"),
            "variance_concentration": float("nan"),
        }

    pca = PCA(n_components=n_components)
    pca.fit(all_reps)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    effective_dim_90 = np.searchsorted(cumsum, 0.90) + 1

    # Variance concentration: how much variance in top 10 dims
    top10_var = cumsum[min(9, len(cumsum) - 1)]

    return {
        "effective_dim_90": int(effective_dim_90),
        "variance_concentration_top10": float(top10_var),
        "total_dims": all_reps.shape[1],
    }


def analyze_attention_diffusion(attentions: List[np.ndarray]) -> Dict:
    """
    Attention diffusion hypothesis: Larger models spread attention more broadly.
    Measure: Entropy of attention distributions.
    """
    entropies = []
    for attn in attentions:
        # Average across all positions
        avg_attn = attn.mean(axis=0)  # (seq,)
        # Normalize to probability
        avg_attn = avg_attn / (avg_attn.sum() + 1e-10)
        # Entropy
        entropy = -np.sum(avg_attn * np.log(avg_attn + 1e-10))
        entropies.append(entropy)

    return {
        "mean_attention_entropy": float(np.mean(entropies)),
        "std_attention_entropy": float(np.std(entropies)),
    }


def analyze_layer_specialization(layer_reps: List[np.ndarray], phen_count: int) -> Dict:
    """
    Layer specialization hypothesis: Where does discrimination peak?
    Measure: Fisher criterion at each layer.
    """
    layer_fishers = []
    for reps in layer_reps:
        phen = reps[:phen_count]
        func = reps[phen_count:]
        fisher = compute_fisher_criterion(phen, func)
        layer_fishers.append(fisher)

    peak_layer = int(np.argmax(layer_fishers))
    peak_depth = peak_layer / (len(layer_fishers) - 1) if len(layer_fishers) > 1 else 0

    # Width of discrimination curve (layers with > 80% of peak)
    peak_fisher = max(layer_fishers)
    threshold = 0.8 * peak_fisher
    width = sum(1 for f in layer_fishers if f >= threshold)

    return {
        "layer_fishers": [float(f) for f in layer_fishers],
        "peak_layer": peak_layer,
        "peak_depth_ratio": float(peak_depth),
        "peak_fisher": float(peak_fisher),
        "discrimination_width": width,
    }


def analyze_geometry(phen_reps: np.ndarray, func_reps: np.ndarray) -> Dict:
    """
    Geometry hypothesis: How are concepts arranged in representation space?
    Measure: Cluster compactness, inter-cluster distance, isotropy.
    """
    all_reps = np.vstack([phen_reps, func_reps])
    all_centroid = np.mean(all_reps, axis=0)

    # Isotropy: how uniformly distributed are representations?
    # Measure: ratio of min/max eigenvalue of covariance
    cov = np.cov(all_reps.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove near-zero
    isotropy = eigenvalues.min() / eigenvalues.max() if len(eigenvalues) > 0 else 0

    # Within-class compactness (average distance to class centroid)
    phen_centroid = np.mean(phen_reps, axis=0)
    func_centroid = np.mean(func_reps, axis=0)

    phen_compactness = np.mean([np.linalg.norm(r - phen_centroid) for r in phen_reps])
    func_compactness = np.mean([np.linalg.norm(r - func_centroid) for r in func_reps])

    # Angular separation (cosine similarity of centroids from origin)
    phen_dir = phen_centroid / (np.linalg.norm(phen_centroid) + 1e-10)
    func_dir = func_centroid / (np.linalg.norm(func_centroid) + 1e-10)
    angular_sep = 1 - np.dot(phen_dir, func_dir)  # 0 = same direction, 2 = opposite

    return {
        "isotropy": float(isotropy),
        "phen_compactness": float(phen_compactness),
        "func_compactness": float(func_compactness),
        "angular_separation": float(angular_sep),
    }


def analyze_model(
    model_name: str,
    display_name: str,
    device: str,
    phen_concepts: List[str],
    func_concepts: List[str],
) -> Dict:
    """Complete mechanistic analysis for one model."""
    print(f"\n  Loading {display_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
    model.eval()
    model = model.to(device)

    config = model.config
    n_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"    Architecture: {n_layers}L, {hidden_size}H, {n_params:.1f}M params")

    # Extract representations from all layers
    all_concepts = phen_concepts + func_concepts
    layer_reps = extract_all_layers(model, tokenizer, all_concepts, device)

    # Peak layer analysis
    peak_layer = int(n_layers * 0.9)
    phen_reps = layer_reps[peak_layer][: len(phen_concepts)]
    func_reps = layer_reps[peak_layer][len(phen_concepts) :]

    # Extract attention patterns
    phen_attentions = extract_attention_patterns(
        model, tokenizer, phen_concepts, device
    )
    func_attentions = extract_attention_patterns(
        model, tokenizer, func_concepts, device
    )

    # Run analyses
    superposition = analyze_superposition(phen_reps, func_reps)
    phen_attn_analysis = analyze_attention_diffusion(phen_attentions)
    func_attn_analysis = analyze_attention_diffusion(func_attentions)
    layer_spec = analyze_layer_specialization(layer_reps, len(phen_concepts))
    geometry = analyze_geometry(phen_reps, func_reps)

    # Compute overall Fisher
    fisher = compute_fisher_criterion(phen_reps, func_reps)

    # Cleanup
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "display_name": display_name,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "n_params_millions": n_params,
        "fisher_criterion": fisher,
        "superposition": superposition,
        "attention": {
            "phenomenal": phen_attn_analysis,
            "functional": func_attn_analysis,
            "entropy_difference": phen_attn_analysis["mean_attention_entropy"]
            - func_attn_analysis["mean_attention_entropy"],
        },
        "layer_specialization": layer_spec,
        "geometry": geometry,
    }


def main():
    print("\n" + "=" * 70)
    print("   MECHANISTIC CIRCUIT ANALYSIS")
    print("   Why do larger models show weaker phenomenal discrimination?")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    phen_concepts, func_concepts = load_corpus()
    print(
        f"Corpus: {len(phen_concepts)} phenomenal, {len(func_concepts)} functional concepts"
    )

    # Compare base vs large for both BERT and RoBERTa
    models = [
        ("bert-base-uncased", "BERT-base"),
        ("bert-large-uncased", "BERT-large"),
        ("roberta-base", "RoBERTa-base"),
        ("roberta-large", "RoBERTa-large"),
    ]

    results = []
    for model_id, display_name in models:
        try:
            result = analyze_model(
                model_id, display_name, device, phen_concepts, func_concepts
            )
            results.append(result)
            print(f"    Fisher: {result['fisher_criterion']:.4f}")
        except Exception as e:
            print(f"    Error: {e}")

    if len(results) < 2:
        print("\n  Not enough models for comparison")
        return

    # Analysis
    print("\n" + "=" * 70)
    print("   MECHANISTIC COMPARISON: BASE vs LARGE")
    print("=" * 70)

    # Superposition analysis
    print("\n  [1] SUPERPOSITION HYPOTHESIS")
    print("      More concepts packed per dimension in larger models?")
    print("-" * 50)
    for r in results:
        s = r["superposition"]
        print(
            f"  {r['display_name']:<15} Effective dim (90%): {s['effective_dim_90']:<4} "
            f"Top-10 var: {s['variance_concentration_top10']:.2%}"
        )

    # Attention diffusion
    print("\n  [2] ATTENTION DIFFUSION HYPOTHESIS")
    print("      Larger models spread attention more broadly?")
    print("-" * 50)
    for r in results:
        a = r["attention"]
        print(
            f"  {r['display_name']:<15} Phen entropy: {a['phenomenal']['mean_attention_entropy']:.3f}  "
            f"Func entropy: {a['functional']['mean_attention_entropy']:.3f}"
        )

    # Layer specialization
    print("\n  [3] LAYER SPECIALIZATION HYPOTHESIS")
    print("      Larger models have more diffuse processing?")
    print("-" * 50)
    for r in results:
        l = r["layer_specialization"]
        print(
            f"  {r['display_name']:<15} Peak: layer {l['peak_layer']:<2} ({l['peak_depth_ratio']:.0%} depth)  "
            f"Width: {l['discrimination_width']} layers"
        )

    # Geometry analysis
    print("\n  [4] GEOMETRY HYPOTHESIS")
    print("      Representation space structure differs?")
    print("-" * 50)
    for r in results:
        g = r["geometry"]
        print(
            f"  {r['display_name']:<15} Isotropy: {g['isotropy']:.4f}  "
            f"Angular sep: {g['angular_separation']:.4f}"
        )

    # Key findings
    print("\n" + "=" * 70)
    print("   KEY MECHANISTIC FINDINGS")
    print("=" * 70 + "\n")

    # Compare base vs large within each family
    findings = []
    for family in ["BERT", "RoBERTa"]:
        base = [r for r in results if r["display_name"] == f"{family}-base"]
        large = [r for r in results if r["display_name"] == f"{family}-large"]
        if base and large:
            base, large = base[0], large[0]

            # Superposition
            if (
                large["superposition"]["effective_dim_90"]
                > base["superposition"]["effective_dim_90"]
            ):
                findings.append(
                    f"  {family}: Larger model uses MORE effective dimensions"
                )
            else:
                findings.append(
                    f"  {family}: Larger model uses FEWER effective dimensions (more superposition)"
                )

            # Isotropy
            if large["geometry"]["isotropy"] > base["geometry"]["isotropy"]:
                findings.append(f"  {family}: Larger model is MORE isotropic")
            else:
                findings.append(
                    f"  {family}: Larger model is LESS isotropic (more anisotropic)"
                )

            # Angular separation
            if (
                large["geometry"]["angular_separation"]
                < base["geometry"]["angular_separation"]
            ):
                findings.append(
                    f"  {family}: Larger model has LESS angular separation between classes"
                )

    for f in findings:
        print(f)

    # Save results
    output = {
        "experiment": "mechanistic_circuit_analysis",
        "n_phenomenal_concepts": len(phen_concepts),
        "n_functional_concepts": len(func_concepts),
        "results": results,
        "findings": findings,
    }

    # Custom encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/mechanistic_circuit_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
