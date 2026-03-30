#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Train Phenomenal Content Classifier

Trains a classifier to detect phenomenal content using transformer embeddings.
Uses attention entropy and layer activations as features.

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers scikit-learn])" \
        --run "python3 scripts/train_phenomenal_classifier.py"
"""

import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
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
    "Consciousness as unified awareness",
    "The feeling of being alive",
    "Introspective access to mental states",
    "The quale of sweetness",
    "Direct acquaintance with experience",
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
    "Encryption secures data transmission",
    "Variables store computed values",
    "Loop iteration processes each element",
    "Stack frames track function calls",
    "Cache improves memory access speed",
]

# Test data (held out)
PHENOMENAL_TEST = [
    "The feeling of wonder at the stars",
    "What it is like to smell a rose",
    "The subjective sense of self",
    "The experience of deja vu",
    "Emotional resonance with music",
]

FUNCTIONAL_TEST = [
    "Pointers reference memory addresses",
    "Threads execute concurrently",
    "The API returns JSON data",
    "Deadlock occurs when processes wait",
    "Regular expressions match patterns",
]


def extract_features(
    model, tokenizer, text: str, target_layer: int, device: str = "cpu"
) -> np.ndarray:
    """Extract features for classification."""
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    # Feature 1: Mean-pooled activations from target layer
    hidden = outputs.hidden_states[target_layer + 1]  # +1 for embedding layer
    attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
    masked = hidden * attention_mask
    pooled = (masked.sum(dim=1) / attention_mask.sum(dim=1)).squeeze(0).cpu().numpy()

    # Feature 2: Attention entropy at target layer
    attn = outputs.attentions[target_layer]
    attn_np = attn.squeeze(0).mean(dim=0).cpu().numpy()
    flat = attn_np.flatten() + 1e-10
    flat = flat / flat.sum()
    entropy = -np.sum(flat * np.log2(flat))

    # Feature 3: Activation statistics
    norm = np.linalg.norm(pooled)
    mean_abs = np.mean(np.abs(pooled))
    std = np.std(pooled)

    # Combine: [entropy, norm, mean, std] + top 50 PCA-like components of pooled
    stats = np.array([entropy, norm, mean_abs, std])

    # Use top components of pooled activation (dimensionality reduction)
    # Simple approach: take every nth dimension
    step = len(pooled) // 50
    reduced = pooled[::step][:50]

    features = np.concatenate([stats, reduced])
    return features


def main():
    print("\n" + "=" * 70)
    print("   TRAIN PHENOMENAL CONTENT CLASSIFIER")
    print("=" * 70 + "\n")

    model_name = "bert-base-uncased"
    target_layer = 10  # Layer 11 (0-indexed) - the phenomenal corridor

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")
    print(f"Target layer: L{target_layer + 1} (phenomenal corridor)\n")

    # Extract training features
    print("Extracting training features...")
    X_train = []
    y_train = []

    for text in PHENOMENAL_TRAIN:
        features = extract_features(model, tokenizer, text, target_layer, device)
        X_train.append(features)
        y_train.append(1)  # Phenomenal = 1

    for text in FUNCTIONAL_TRAIN:
        features = extract_features(model, tokenizer, text, target_layer, device)
        X_train.append(features)
        y_train.append(0)  # Functional = 0

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(
        f"Training samples: {len(X_train)} ({sum(y_train)} phenomenal, {len(y_train) - sum(y_train)} functional)"
    )
    print(f"Feature dimensions: {X_train.shape[1]}")

    # Extract test features
    print("\nExtracting test features...")
    X_test = []
    y_test = []

    for text in PHENOMENAL_TEST:
        features = extract_features(model, tokenizer, text, target_layer, device)
        X_test.append(features)
        y_test.append(1)

    for text in FUNCTIONAL_TEST:
        features = extract_features(model, tokenizer, text, target_layer, device)
        X_test.append(features)
        y_test.append(0)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print(f"Test samples: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classifier
    print("\n" + "-" * 70)
    print("   TRAINING LOGISTIC REGRESSION CLASSIFIER")
    print("-" * 70 + "\n")

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Cross-validation on training set
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Training accuracy
    train_acc = clf.score(X_train_scaled, y_train)
    print(f"Training accuracy: {train_acc:.3f}")

    # Test accuracy
    test_acc = clf.score(X_test_scaled, y_test)
    print(f"Test accuracy: {test_acc:.3f}")

    # Detailed test results
    y_pred = clf.predict(X_test_scaled)

    print("\n" + "-" * 70)
    print("   TEST SET PREDICTIONS")
    print("-" * 70 + "\n")

    print("Phenomenal concepts:")
    for i, text in enumerate(PHENOMENAL_TEST):
        pred = y_pred[i]
        prob = clf.predict_proba(X_test_scaled[i : i + 1])[0, 1]
        status = "CORRECT" if pred == 1 else "WRONG"
        print(f"  {text[:40]:40s} -> P={prob:.2f} [{status}]")

    print("\nFunctional concepts:")
    for i, text in enumerate(FUNCTIONAL_TEST):
        idx = len(PHENOMENAL_TEST) + i
        pred = y_pred[idx]
        prob = clf.predict_proba(X_test_scaled[idx : idx + 1])[0, 1]
        status = "CORRECT" if pred == 0 else "WRONG"
        print(f"  {text[:40]:40s} -> P={prob:.2f} [{status}]")

    # Classification report
    print("\n" + "-" * 70)
    print("   CLASSIFICATION REPORT")
    print("-" * 70 + "\n")
    print(
        classification_report(y_test, y_pred, target_names=["Functional", "Phenomenal"])
    )

    # Feature importance
    print("-" * 70)
    print("   FEATURE IMPORTANCE (Top coefficients)")
    print("-" * 70 + "\n")

    coef = clf.coef_[0]
    feature_names = ["entropy", "norm", "mean_abs", "std"] + [
        f"dim_{i}" for i in range(len(coef) - 4)
    ]

    # Top positive (phenomenal indicators)
    top_pos_idx = np.argsort(coef)[-5:][::-1]
    print("Top PHENOMENAL indicators:")
    for idx in top_pos_idx:
        print(f"  {feature_names[idx]:15s}: {coef[idx]:+.3f}")

    # Top negative (functional indicators)
    top_neg_idx = np.argsort(coef)[:5]
    print("\nTop FUNCTIONAL indicators:")
    for idx in top_neg_idx:
        print(f"  {feature_names[idx]:15s}: {coef[idx]:+.3f}")

    # Key finding
    print("\n" + "=" * 70)
    print("   KEY FINDING")
    print("=" * 70 + "\n")

    if test_acc >= 0.8:
        print(f"  HIGH ACCURACY CLASSIFIER ACHIEVED ({test_acc:.1%})")
        print("  Phenomenal content can be reliably detected from transformer features")
    elif test_acc >= 0.6:
        print(f"  MODERATE ACCURACY ({test_acc:.1%})")
        print("  Classifier shows some ability to detect phenomenal content")
    else:
        print(f"  LOW ACCURACY ({test_acc:.1%})")
        print("  Simple classifier struggles to separate classes")

    # Entropy importance
    entropy_coef = coef[0]
    if entropy_coef < -0.1:
        print(f"\n  ENTROPY COEFFICIENT: {entropy_coef:.3f}")
        print("  Lower attention entropy → More likely phenomenal")
        print("  (Confirms attention concentration finding)")

    # Save model info
    output = {
        "model": model_name,
        "target_layer": target_layer + 1,
        "n_features": int(X_train.shape[1]),
        "cv_accuracy": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "feature_importance": {
            "entropy": float(coef[0]),
            "norm": float(coef[1]),
            "mean_abs": float(coef[2]),
            "std": float(coef[3]),
        },
        "predictions": [
            {
                "text": t,
                "true": int(y),
                "pred": int(p),
                "prob": float(clf.predict_proba(X_test_scaled[i : i + 1])[0, 1]),
            }
            for i, (t, y, p) in enumerate(
                zip(PHENOMENAL_TEST + FUNCTIONAL_TEST, y_test, y_pred)
            )
        ],
    }

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/phenomenal_classifier.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
