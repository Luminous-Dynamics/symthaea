#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Real-World Phenomenal Classifier Test

Tests the phenomenal classifier on diverse real-world text:
- Philosophy of mind excerpts
- Poetry and literary descriptions
- Meditation/mindfulness text
- AI ethics discussions
- Edge cases

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers scikit-learn])" \
        --run "python3 scripts/real_world_classifier_test.py"
"""

import json
from typing import Dict, List

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

# Training data (same as before)
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

# Real-world test cases
REAL_WORLD_TESTS = {
    "Philosophy of Mind": [
        ("There is something it is like to be a bat", "phenomenal", "Nagel"),
        ("Consciousness is the hard problem", "phenomenal", "Chalmers"),
        ("Mental states are functional states", "functional", "Dennett"),
        ("Qualia are intrinsic non-representational properties", "phenomenal", "Block"),
        (
            "The brain is an information processing system",
            "functional",
            "Computational",
        ),
    ],
    "Poetry & Literature": [
        ("I felt a funeral in my brain", "phenomenal", "Dickinson"),
        ("The world is too much with us", "phenomenal", "Wordsworth"),
        ("To be or not to be that is the question", "ambiguous", "Shakespeare"),
        ("Hope is the thing with feathers", "phenomenal", "Dickinson"),
        ("The road diverged in a yellow wood", "ambiguous", "Frost"),
    ],
    "Meditation & Mindfulness": [
        ("Observe your breath without judgment", "phenomenal", "Mindfulness"),
        ("Notice the sensation in your body", "phenomenal", "Body scan"),
        ("Let thoughts arise and pass away", "phenomenal", "Meditation"),
        ("Focus your attention on the present moment", "phenomenal", "Present"),
        ("Practice loving kindness toward yourself", "phenomenal", "Metta"),
    ],
    "AI & Technology": [
        ("The neural network learns patterns from data", "functional", "ML"),
        ("GPT generates text based on probabilities", "functional", "LLM"),
        ("Does the AI have inner experience", "phenomenal", "AI consciousness"),
        ("Machine consciousness remains an open question", "phenomenal", "Philosophy"),
        ("The model optimizes a loss function", "functional", "Training"),
    ],
    "Edge Cases": [
        ("The sunset painted the sky orange", "ambiguous", "Description"),
        ("She felt deeply moved by the music", "phenomenal", "Emotion"),
        ("The sensor detected temperature changes", "functional", "Sensor"),
        ("He experienced a moment of clarity", "phenomenal", "Insight"),
        ("The system processes user requests", "functional", "System"),
    ],
    "Science": [
        ("Neurons fire action potentials", "functional", "Neuroscience"),
        ("Photons stimulate retinal cells", "functional", "Vision"),
        ("Patients report pain after surgery", "phenomenal", "Medical"),
        ("The amygdala processes emotional stimuli", "functional", "Neuro"),
        ("Subjects describe their visual experience", "phenomenal", "Psych"),
    ],
}


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

    hidden = outputs.hidden_states[target_layer + 1]
    attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
    masked = hidden * attention_mask
    pooled = (masked.sum(dim=1) / attention_mask.sum(dim=1)).squeeze(0).cpu().numpy()

    attn = outputs.attentions[target_layer]
    attn_np = attn.squeeze(0).mean(dim=0).cpu().numpy()
    flat = attn_np.flatten() + 1e-10
    flat = flat / flat.sum()
    entropy = -np.sum(flat * np.log2(flat))

    norm = np.linalg.norm(pooled)
    mean_abs = np.mean(np.abs(pooled))
    std = np.std(pooled)

    stats = np.array([entropy, norm, mean_abs, std])
    step = len(pooled) // 50
    reduced = pooled[::step][:50]

    return np.concatenate([stats, reduced])


def main():
    print("\n" + "=" * 70)
    print("   REAL-WORLD PHENOMENAL CLASSIFIER TEST")
    print("=" * 70 + "\n")

    model_name = "bert-base-uncased"
    target_layer = 10

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}\n")

    # Train classifier
    print("Training classifier on canonical examples...")
    X_train = []
    y_train = []

    for text in PHENOMENAL_TRAIN:
        X_train.append(extract_features(model, tokenizer, text, target_layer, device))
        y_train.append(1)

    for text in FUNCTIONAL_TRAIN:
        X_train.append(extract_features(model, tokenizer, text, target_layer, device))
        y_train.append(0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)
    print(f"Training accuracy: {clf.score(X_train_scaled, y_train):.1%}\n")

    # Test on real-world examples
    all_results = []
    category_accuracy = {}

    for category, examples in REAL_WORLD_TESTS.items():
        print("=" * 70)
        print(f"   {category.upper()}")
        print("=" * 70 + "\n")

        correct = 0
        total = 0

        for text, expected, source in examples:
            features = extract_features(model, tokenizer, text, target_layer, device)
            features_scaled = scaler.transform(features.reshape(1, -1))

            prob = clf.predict_proba(features_scaled)[0, 1]
            pred = "phenomenal" if prob > 0.5 else "functional"

            # For ambiguous cases, count as correct if prob is in middle range
            if expected == "ambiguous":
                is_correct = 0.3 < prob < 0.7
                status = "AMBIGUOUS" if is_correct else f"LEANING {pred}"
            else:
                is_correct = pred == expected
                status = "CORRECT" if is_correct else "WRONG"
                total += 1
                if is_correct:
                    correct += 1

            result = {
                "text": text,
                "expected": expected,
                "predicted": pred,
                "probability": float(prob),
                "correct": is_correct,
                "source": source,
                "category": category,
            }
            all_results.append(result)

            prob_bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            print(f"  [{source:12s}] P={prob:.2f} |{prob_bar}| {status}")
            print(f'    "{text[:50]}..."' if len(text) > 50 else f'    "{text}"')
            print()

        if total > 0:
            acc = correct / total
            category_accuracy[category] = acc
            print(f"  Category accuracy: {acc:.1%} ({correct}/{total})\n")

    # Overall summary
    print("\n" + "=" * 70)
    print("   OVERALL RESULTS")
    print("=" * 70 + "\n")

    non_ambig = [r for r in all_results if r["expected"] != "ambiguous"]
    correct_count = sum(1 for r in non_ambig if r["correct"])
    total_count = len(non_ambig)

    print(f"{'Category':<25} {'Accuracy':<12}")
    print("-" * 37)
    for cat, acc in category_accuracy.items():
        print(f"{cat:<25} {acc:.1%}")
    print("-" * 37)
    print(
        f"{'OVERALL':<25} {correct_count/total_count:.1%} ({correct_count}/{total_count})"
    )

    # Confidence analysis
    print("\n" + "-" * 70)
    print("   CONFIDENCE DISTRIBUTION")
    print("-" * 70 + "\n")

    probs = [r["probability"] for r in all_results]
    high_conf_phen = sum(1 for p in probs if p > 0.8)
    high_conf_func = sum(1 for p in probs if p < 0.2)
    uncertain = sum(1 for p in probs if 0.3 < p < 0.7)

    print(f"  High confidence phenomenal (P > 0.8): {high_conf_phen}")
    print(f"  High confidence functional (P < 0.2): {high_conf_func}")
    print(f"  Uncertain (0.3 < P < 0.7): {uncertain}")

    # Error analysis
    print("\n" + "-" * 70)
    print("   ERROR ANALYSIS")
    print("-" * 70 + "\n")

    errors = [r for r in non_ambig if not r["correct"]]
    if errors:
        print(f"  {len(errors)} errors:")
        for e in errors:
            print(
                f"    [{e['category']}] Expected {e['expected']}, got {e['predicted']} (P={e['probability']:.2f})"
            )
            print(f"      \"{e['text'][:60]}...\"")
    else:
        print("  No errors on non-ambiguous cases!")

    # Key finding
    print("\n" + "=" * 70)
    print("   KEY FINDING")
    print("=" * 70 + "\n")

    overall_acc = correct_count / total_count
    if overall_acc >= 0.85:
        print(f"  HIGH REAL-WORLD ACCURACY ({overall_acc:.1%})")
        print("  Classifier generalizes well to diverse text")
    elif overall_acc >= 0.70:
        print(f"  GOOD REAL-WORLD ACCURACY ({overall_acc:.1%})")
        print("  Classifier shows reasonable generalization")
    else:
        print(f"  MODERATE ACCURACY ({overall_acc:.1%})")
        print("  Classifier struggles with some real-world cases")

    # Save results
    output = {
        "model": model_name,
        "target_layer": target_layer + 1,
        "training_samples": len(PHENOMENAL_TRAIN) + len(FUNCTIONAL_TRAIN),
        "test_results": all_results,
        "category_accuracy": category_accuracy,
        "overall_accuracy": float(overall_acc),
        "total_correct": correct_count,
        "total_tested": total_count,
    }

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/real_world_classifier_test.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
