#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Cross-Model and Linguistic Feature Analysis

Tests if the sensory/procedural topological distinction holds across models
and correlates with linguistic features.

Directions A + C from the research plan:
A: Cross-model validation (Sentence-BERT, BERT, RoBERTa)
C: Linguistic feature analysis (word frequency, concreteness proxies)

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers sentence-transformers scikit-learn])" \
        --run "python3 scripts/cross_model_linguistic_analysis.py"
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import re

import numpy as np

# Try sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("Warning: sentence-transformers not available")

# Try transformers for BERT comparison
try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: torch/transformers not available")


def load_concepts():
    """Load the expanded concept corpora."""
    base_path = Path("/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/consciousness_probe")

    with open(base_path / "phenomenal_concepts_expanded.json") as f:
        phen_data = json.load(f)
    with open(base_path / "functional_concepts_expanded.json") as f:
        func_data = json.load(f)

    return phen_data["concepts"], func_data["concepts"]


def extract_linguistic_features(text: str) -> Dict[str, float]:
    """Extract simple linguistic features from text."""
    words = text.lower().split()

    # Sensory/embodied word lists (approximation)
    sensory_words = {
        "see", "seeing", "red", "blue", "green", "yellow", "color", "bright", "dark",
        "hear", "hearing", "sound", "loud", "quiet", "music", "noise",
        "feel", "feeling", "touch", "soft", "hard", "warm", "cold", "hot", "pain",
        "taste", "sweet", "sour", "bitter", "smell", "aroma", "scent",
        "body", "skin", "hand", "eye", "sensation", "pressure", "texture"
    }

    # Abstract/procedural word lists
    abstract_words = {
        "algorithm", "function", "variable", "compute", "calculate", "process",
        "system", "structure", "data", "memory", "optimize", "complexity",
        "recursive", "binary", "tree", "graph", "hash", "sort", "search",
        "abstract", "concept", "theory", "principle", "model", "equation"
    }

    # First-person words (associated with phenomenal)
    first_person = {"i", "my", "me", "mine", "myself"}

    # Technical words
    technical_suffixes = ["tion", "ment", "ness", "ity", "ing"]

    # Count features
    sensory_count = sum(1 for w in words if w in sensory_words)
    abstract_count = sum(1 for w in words if w in abstract_words)
    first_person_count = sum(1 for w in words if w in first_person)

    # Word length (longer = more technical/abstract)
    avg_word_length = np.mean([len(w) for w in words]) if words else 0

    # Sentence length
    sentence_length = len(words)

    return {
        "sensory_word_ratio": sensory_count / len(words) if words else 0,
        "abstract_word_ratio": abstract_count / len(words) if words else 0,
        "first_person_ratio": first_person_count / len(words) if words else 0,
        "avg_word_length": avg_word_length,
        "sentence_length": sentence_length,
    }


def compute_embedding_metrics(embeddings: np.ndarray) -> Dict[str, float]:
    """Compute embedding-based metrics similar to topological unity."""
    # Norm (magnitude)
    norms = np.linalg.norm(embeddings, axis=1)

    # Isotropy (variance of norms)
    norm_variance = np.var(norms)

    # Effective dimensionality via PCA
    from sklearn.decomposition import PCA
    n_components = min(embeddings.shape[0] - 1, embeddings.shape[1], 50)
    if n_components < 2:
        return {"norm_mean": float(np.mean(norms)), "norm_variance": float(norm_variance)}

    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    effective_dim = np.searchsorted(cumsum, 0.90) + 1

    return {
        "norm_mean": float(np.mean(norms)),
        "norm_variance": float(norm_variance),
        "effective_dim_90": int(effective_dim),
        "top_10_variance": float(cumsum[min(9, len(cumsum)-1)]),
    }


def analyze_with_sbert(phen_concepts: List[Dict], func_concepts: List[Dict], model_name: str = "all-mpnet-base-v2"):
    """Analyze concepts using Sentence-BERT."""
    print(f"\n{'='*60}")
    print(f"   SENTENCE-BERT ANALYSIS: {model_name}")
    print(f"{'='*60}\n")

    model = SentenceTransformer(model_name)

    # Extract texts
    phen_texts = [c["text"] for c in phen_concepts]
    func_texts = [c["text"] for c in func_concepts]

    print(f"Encoding {len(phen_texts)} phenomenal concepts...")
    phen_embeddings = model.encode(phen_texts, show_progress_bar=True)

    print(f"Encoding {len(func_texts)} functional concepts...")
    func_embeddings = model.encode(func_texts, show_progress_bar=True)

    # Compute metrics
    phen_metrics = compute_embedding_metrics(phen_embeddings)
    func_metrics = compute_embedding_metrics(func_embeddings)

    print("\nEmbedding Metrics:")
    print(f"  Phenomenal: norm={phen_metrics['norm_mean']:.4f}, eff_dim={phen_metrics.get('effective_dim_90', 'N/A')}")
    print(f"  Functional: norm={func_metrics['norm_mean']:.4f}, eff_dim={func_metrics.get('effective_dim_90', 'N/A')}")

    # Compute cosine similarity within and between classes
    from sklearn.metrics.pairwise import cosine_similarity

    phen_sim = cosine_similarity(phen_embeddings)
    func_sim = cosine_similarity(func_embeddings)
    cross_sim = cosine_similarity(phen_embeddings, func_embeddings)

    # Extract upper triangle (excluding diagonal)
    phen_within = phen_sim[np.triu_indices(len(phen_sim), k=1)]
    func_within = func_sim[np.triu_indices(len(func_sim), k=1)]

    print("\nWithin-class similarity:")
    print(f"  Phenomenal: {np.mean(phen_within):.4f} (+/- {np.std(phen_within):.4f})")
    print(f"  Functional: {np.mean(func_within):.4f} (+/- {np.std(func_within):.4f})")
    print(f"  Cross-class: {np.mean(cross_sim):.4f} (+/- {np.std(cross_sim):.4f})")

    # Category-level analysis
    print("\nCategory-level analysis (phenomenal):")
    phen_by_cat = {}
    for i, c in enumerate(phen_concepts):
        cat = c["category"]
        if cat not in phen_by_cat:
            phen_by_cat[cat] = []
        phen_by_cat[cat].append(phen_embeddings[i])

    cat_metrics = []
    for cat, embs in sorted(phen_by_cat.items()):
        embs = np.array(embs)
        if len(embs) > 1:
            sim = cosine_similarity(embs)
            within = sim[np.triu_indices(len(sim), k=1)]
            mean_sim = np.mean(within)
            cat_metrics.append((cat, mean_sim, len(embs)))
            print(f"  {cat:<20} (n={len(embs):2}): within-sim = {mean_sim:.4f}")

    print("\nCategory-level analysis (functional):")
    func_by_cat = {}
    for i, c in enumerate(func_concepts):
        cat = c["category"]
        if cat not in func_by_cat:
            func_by_cat[cat] = []
        func_by_cat[cat].append(func_embeddings[i])

    for cat, embs in sorted(func_by_cat.items()):
        embs = np.array(embs)
        if len(embs) > 1:
            sim = cosine_similarity(embs)
            within = sim[np.triu_indices(len(sim), k=1)]
            mean_sim = np.mean(within)
            print(f"  {cat:<20} (n={len(embs):2}): within-sim = {mean_sim:.4f}")

    return {
        "model": model_name,
        "phenomenal_metrics": phen_metrics,
        "functional_metrics": func_metrics,
        "phenomenal_within_sim": float(np.mean(phen_within)),
        "functional_within_sim": float(np.mean(func_within)),
        "cross_class_sim": float(np.mean(cross_sim)),
    }


def analyze_linguistic_features(phen_concepts: List[Dict], func_concepts: List[Dict]):
    """Analyze linguistic features of concepts."""
    print(f"\n{'='*60}")
    print(f"   LINGUISTIC FEATURE ANALYSIS")
    print(f"{'='*60}\n")

    # Extract features for all concepts
    phen_features = [extract_linguistic_features(c["text"]) for c in phen_concepts]
    func_features = [extract_linguistic_features(c["text"]) for c in func_concepts]

    feature_names = list(phen_features[0].keys())

    print("Feature comparison (phenomenal vs functional):")
    print("-" * 50)

    results = {}
    for feat in feature_names:
        phen_vals = [f[feat] for f in phen_features]
        func_vals = [f[feat] for f in func_features]

        phen_mean = np.mean(phen_vals)
        func_mean = np.mean(func_vals)
        diff = phen_mean - func_mean

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(phen_vals) + np.var(func_vals)) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0

        print(f"  {feat:<25} Phen={phen_mean:.4f}  Func={func_mean:.4f}  d={cohens_d:+.2f}")

        results[feat] = {
            "phenomenal_mean": phen_mean,
            "functional_mean": func_mean,
            "cohens_d": cohens_d,
        }

    return results


def load_unity_scores():
    """Load previously computed unity scores."""
    csv_path = Path("/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/consciousness_probe/h1_expanded_200_results.csv")

    if not csv_path.exists():
        print(f"Warning: Unity scores not found at {csv_path}")
        return None

    scores = {}
    with open(csv_path) as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 6:
                concept_id = parts[0].strip('"')
                unity = float(parts[5])
                scores[concept_id] = unity

    return scores


def correlate_features_with_unity(phen_concepts: List[Dict], func_concepts: List[Dict], unity_scores: Dict):
    """Correlate linguistic features with topological unity scores."""
    print(f"\n{'='*60}")
    print(f"   FEATURE-UNITY CORRELATIONS")
    print(f"{'='*60}\n")

    all_concepts = phen_concepts + func_concepts

    # Extract features and match with unity scores
    features_list = []
    unity_list = []

    for c in all_concepts:
        if c["id"] in unity_scores:
            features_list.append(extract_linguistic_features(c["text"]))
            unity_list.append(unity_scores[c["id"]])

    if not features_list:
        print("No matching unity scores found")
        return None

    print(f"Matched {len(features_list)} concepts with unity scores\n")

    feature_names = list(features_list[0].keys())
    correlations = {}

    print("Correlation with topological unity score:")
    print("-" * 50)

    for feat in feature_names:
        feat_vals = np.array([f[feat] for f in features_list])
        unity_vals = np.array(unity_list)

        # Pearson correlation
        if np.std(feat_vals) > 0 and np.std(unity_vals) > 0:
            corr = np.corrcoef(feat_vals, unity_vals)[0, 1]
        else:
            corr = 0

        correlations[feat] = corr
        print(f"  {feat:<25} r = {corr:+.3f}")

    return correlations


def main():
    print("\n" + "=" * 70)
    print("   CROSS-MODEL AND LINGUISTIC ANALYSIS")
    print("   Testing replication and feature correlations")
    print("=" * 70)

    # Load concepts
    phen_concepts, func_concepts = load_concepts()
    print(f"\nLoaded {len(phen_concepts)} phenomenal, {len(func_concepts)} functional concepts")

    results = {}

    # A: Cross-model validation
    if HAS_SBERT:
        models_to_test = [
            "all-mpnet-base-v2",  # Best general-purpose SBERT
            "all-MiniLM-L6-v2",   # Smaller, faster
        ]

        for model_name in models_to_test:
            try:
                result = analyze_with_sbert(phen_concepts, func_concepts, model_name)
                results[f"sbert_{model_name}"] = result
            except Exception as e:
                print(f"Error with {model_name}: {e}")

    # C: Linguistic feature analysis
    linguistic_results = analyze_linguistic_features(phen_concepts, func_concepts)
    results["linguistic_features"] = linguistic_results

    # Correlate with unity scores
    unity_scores = load_unity_scores()
    if unity_scores:
        correlations = correlate_features_with_unity(phen_concepts, func_concepts, unity_scores)
        results["unity_correlations"] = correlations

    # Summary
    print("\n" + "=" * 70)
    print("   SUMMARY")
    print("=" * 70 + "\n")

    if HAS_SBERT and "sbert_all-mpnet-base-v2" in results:
        r = results["sbert_all-mpnet-base-v2"]
        print("Cross-Model (Sentence-BERT all-mpnet-base-v2):")
        print(f"  Phenomenal within-class similarity: {r['phenomenal_within_sim']:.4f}")
        print(f"  Functional within-class similarity: {r['functional_within_sim']:.4f}")
        diff = r['phenomenal_within_sim'] - r['functional_within_sim']
        print(f"  Difference: {diff:+.4f}")
        if diff > 0:
            print("  → Phenomenal concepts cluster MORE tightly (consistent with BGE-M3)")
        else:
            print("  → Functional concepts cluster more tightly (OPPOSITE to BGE-M3)")

    print("\nLinguistic Features (strongest differentiators):")
    if "linguistic_features" in results:
        sorted_feats = sorted(results["linguistic_features"].items(),
                             key=lambda x: abs(x[1]["cohens_d"]), reverse=True)
        for feat, vals in sorted_feats[:3]:
            print(f"  {feat}: d={vals['cohens_d']:+.2f}")

    if "unity_correlations" in results and results["unity_correlations"]:
        print("\nUnity Correlations (strongest):")
        sorted_corrs = sorted(results["unity_correlations"].items(),
                             key=lambda x: abs(x[1]), reverse=True)
        for feat, corr in sorted_corrs[:3]:
            print(f"  {feat}: r={corr:+.3f}")

    # Save results
    output_path = Path("/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/cross_model_linguistic_analysis.json")

    # Convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
