#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Combined Signature Classifier

Combines attention entropy and topological unity signatures for phenomenal classification.
Tests whether independent mechanisms are complementary.

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers scikit-learn])" \
        --run "python3 scripts/combined_signature_classifier.py"
"""

import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

# Training data
PHENOMENAL_TRAIN = [
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

FUNCTIONAL_TRAIN = [
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

# Test data (held out)
PHENOMENAL_TEST = [
    "The qualitative feel of warmth",
    "Subjective awareness of being conscious",
    "The experience of deja vu",
    "What it feels like to be in love",
    "The phenomenal character of anxiety",
]

FUNCTIONAL_TEST = [
    "The loop iterates through the array",
    "Encryption secures the data transmission",
    "The scheduler allocates CPU time",
    "Caching improves query performance",
    "The API endpoint validates input",
]


def extract_combined_features(
    model, tokenizer, text: str, target_layer: int, device: str = "cpu"
) -> Tuple[np.ndarray, float, float]:
    """Extract both attention entropy and hidden state features."""
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    # 1. Hidden state features (for topology proxy)
    hidden = outputs.hidden_states[target_layer + 1]
    attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
    masked = hidden * attention_mask
    pooled = (masked.sum(dim=1) / attention_mask.sum(dim=1)).squeeze(0).cpu().numpy()

    # Compute unity proxy (L2 norm * inverse CV)
    norm = np.linalg.norm(pooled)
    mean_abs = np.mean(np.abs(pooled))
    std = np.std(pooled)
    cv = std / (mean_abs + 1e-10)
    unity = norm / (cv + 1e-10)  # Higher = more unified

    # 2. Attention entropy
    attn = outputs.attentions[target_layer]
    attn_np = attn.squeeze(0).mean(dim=0).cpu().numpy()  # Average over heads
    flat = attn_np.flatten() + 1e-10
    flat = flat / flat.sum()
    entropy = -np.sum(flat * np.log2(flat))

    # 3. Additional statistics
    attn_concentration = np.max(flat)  # Max attention weight

    # Reduced hidden state for additional features
    step = max(1, len(pooled) // 20)
    reduced_hidden = pooled[::step][:20]

    return reduced_hidden, entropy, unity


def build_feature_sets(
    model, tokenizer, texts: List[str], target_layer: int, device: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build different feature sets for comparison."""
    entropy_features = []
    unity_features = []
    hidden_features = []
    combined_features = []

    for text in texts:
        hidden, entropy, unity = extract_combined_features(
            model, tokenizer, text, target_layer, device
        )

        # Single features
        entropy_features.append([entropy])
        unity_features.append([unity])
        hidden_features.append(hidden)

        # Combined features
        combined = np.concatenate([[entropy, unity], hidden])
        combined_features.append(combined)

    return (
        np.array(entropy_features),
        np.array(unity_features),
        np.array(hidden_features),
        np.array(combined_features),
    )


def evaluate_classifier(X_train, y_train, X_test, y_test, name: str) -> Dict:
    """Train and evaluate a classifier."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Training accuracy
    train_acc = clf.score(X_train_scaled, y_train)

    # Test accuracy
    test_pred = clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)

    # Cross-validation on training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv)

    return {
        "name": name,
        "n_features": X_train.shape[1],
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
    }


def main():
    print("\n" + "=" * 70)
    print("   COMBINED SIGNATURE CLASSIFIER")
    print("   Testing if independent signatures are complementary")
    print("=" * 70 + "\n")

    model_name = "bert-base-uncased"
    target_layer = 10  # Layer 11 (0-indexed)

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}\n")

    # Build training features
    print("Extracting training features...")
    train_texts = PHENOMENAL_TRAIN + FUNCTIONAL_TRAIN
    train_labels = [1] * len(PHENOMENAL_TRAIN) + [0] * len(FUNCTIONAL_TRAIN)
    y_train = np.array(train_labels)

    entropy_train, unity_train, hidden_train, combined_train = build_feature_sets(
        model, tokenizer, train_texts, target_layer, device
    )

    # Build test features
    print("Extracting test features...")
    test_texts = PHENOMENAL_TEST + FUNCTIONAL_TEST
    test_labels = [1] * len(PHENOMENAL_TEST) + [0] * len(FUNCTIONAL_TEST)
    y_test = np.array(test_labels)

    entropy_test, unity_test, hidden_test, combined_test = build_feature_sets(
        model, tokenizer, test_texts, target_layer, device
    )

    # Evaluate different feature sets
    print("\nEvaluating classifiers...\n")

    results = []

    # 1. Entropy only
    result = evaluate_classifier(
        entropy_train, y_train, entropy_test, y_test, "Entropy Only"
    )
    results.append(result)
    print(
        f"  {result['name']:20s} | Train: {result['train_accuracy']:.1%} | Test: {result['test_accuracy']:.1%} | CV: {result['cv_mean']:.1%} +/- {result['cv_std']:.1%}"
    )

    # 2. Unity only
    result = evaluate_classifier(unity_train, y_train, unity_test, y_test, "Unity Only")
    results.append(result)
    print(
        f"  {result['name']:20s} | Train: {result['train_accuracy']:.1%} | Test: {result['test_accuracy']:.1%} | CV: {result['cv_mean']:.1%} +/- {result['cv_std']:.1%}"
    )

    # 3. Hidden state only
    result = evaluate_classifier(
        hidden_train, y_train, hidden_test, y_test, "Hidden State Only"
    )
    results.append(result)
    print(
        f"  {result['name']:20s} | Train: {result['train_accuracy']:.1%} | Test: {result['test_accuracy']:.1%} | CV: {result['cv_mean']:.1%} +/- {result['cv_std']:.1%}"
    )

    # 4. Entropy + Unity
    entropy_unity_train = np.hstack([entropy_train, unity_train])
    entropy_unity_test = np.hstack([entropy_test, unity_test])
    result = evaluate_classifier(
        entropy_unity_train, y_train, entropy_unity_test, y_test, "Entropy + Unity"
    )
    results.append(result)
    print(
        f"  {result['name']:20s} | Train: {result['train_accuracy']:.1%} | Test: {result['test_accuracy']:.1%} | CV: {result['cv_mean']:.1%} +/- {result['cv_std']:.1%}"
    )

    # 5. All combined
    result = evaluate_classifier(
        combined_train, y_train, combined_test, y_test, "All Combined"
    )
    results.append(result)
    print(
        f"  {result['name']:20s} | Train: {result['train_accuracy']:.1%} | Test: {result['test_accuracy']:.1%} | CV: {result['cv_mean']:.1%} +/- {result['cv_std']:.1%}"
    )

    # Analysis
    print("\n" + "=" * 70)
    print("   FEATURE COMPARISON")
    print("=" * 70 + "\n")

    print(f"{'Feature Set':<20} {'N Features':<12} {'Test Acc':<12} {'CV Mean':<12}")
    print("-" * 56)
    for r in results:
        print(
            f"{r['name']:<20} {r['n_features']:<12} {r['test_accuracy']:.1%}{'':8} {r['cv_mean']:.1%}"
        )

    # Key finding
    print("\n" + "=" * 70)
    print("   KEY FINDING")
    print("=" * 70 + "\n")

    best_single = max(
        [r for r in results if "+" not in r["name"] and "Combined" not in r["name"]],
        key=lambda x: x["test_accuracy"],
    )
    entropy_unity = [r for r in results if r["name"] == "Entropy + Unity"][0]
    all_combined = [r for r in results if r["name"] == "All Combined"][0]

    improvement_eu = entropy_unity["test_accuracy"] - best_single["test_accuracy"]
    improvement_all = all_combined["test_accuracy"] - best_single["test_accuracy"]

    print(
        f"  Best single feature: {best_single['name']} ({best_single['test_accuracy']:.1%})"
    )
    print(
        f"  Entropy + Unity: {entropy_unity['test_accuracy']:.1%} (improvement: {improvement_eu:+.1%})"
    )
    print(
        f"  All Combined: {all_combined['test_accuracy']:.1%} (improvement: {improvement_all:+.1%})"
    )

    if improvement_eu > 0.05 or improvement_all > 0.05:
        print("\n  COMBINING SIGNATURES IMPROVES CLASSIFICATION")
        print("  Independent mechanisms are COMPLEMENTARY")
    elif improvement_eu < -0.05 or improvement_all < -0.05:
        print("\n  COMBINING SIGNATURES HURTS CLASSIFICATION")
        print("  Features may be redundant or conflicting")
    else:
        print("\n  MINIMAL IMPROVEMENT FROM COMBINING")
        print("  Single features capture most of the signal")

    # Feature importance for combined model
    print("\n" + "-" * 70)
    print("   FEATURE IMPORTANCE (Combined Model)")
    print("-" * 70 + "\n")

    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(combined_train)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_combined_scaled, y_train)

    feature_names = ["entropy", "unity"] + [f"hidden_{i}" for i in range(20)]
    coefs = clf.coef_[0]

    # Sort by absolute coefficient
    sorted_idx = np.argsort(np.abs(coefs))[::-1]

    print("  Top 10 features by |coefficient|:")
    for i, idx in enumerate(sorted_idx[:10]):
        direction = "phenomenal" if coefs[idx] > 0 else "functional"
        print(
            f"    {i+1}. {feature_names[idx]:12s}: {coefs[idx]:+.4f} (predicts {direction})"
        )

    # Save results
    output = {
        "model": model_name,
        "target_layer": target_layer + 1,
        "n_train_phenomenal": len(PHENOMENAL_TRAIN),
        "n_train_functional": len(FUNCTIONAL_TRAIN),
        "n_test_phenomenal": len(PHENOMENAL_TEST),
        "n_test_functional": len(FUNCTIONAL_TEST),
        "classifier_results": results,
        "best_single": best_single["name"],
        "improvement_entropy_unity": float(improvement_eu),
        "improvement_all_combined": float(improvement_all),
    }

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/combined_signature_classifier.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
