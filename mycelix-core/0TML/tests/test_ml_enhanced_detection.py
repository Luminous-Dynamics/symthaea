#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ML-Enhanced Byzantine Detection - Integration Test
==================================================

Comprehensive test demonstrating the complete ML pipeline:
1. Feature extraction (PoGQ, TCDM, Z-score, Entropy)
2. Classifier training (SVM, RF, Ensemble)
3. Evaluation (metrics, ROC/PR curves, confusion matrices)
4. High-level API usage (ByzantineDetector)

Target: 95-98% detection rate with ≤3% false positives

Run:
    python tests/test_ml_enhanced_detection.py
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import json

from zerotrustml.ml import (
    FeatureExtractor,
    SVMClassifier,
    RandomForestClassifier,
    EnsembleClassifier,
    DetectionEvaluator,
    ByzantineDetector,
    extract_features_batch,
)


def generate_synthetic_gradients(
    num_honest: int = 500,
    num_byzantine: int = 100,
    gradient_dim: int = 100,
    seed: int = 42,
):
    """
    Generate synthetic gradient data for training/testing.

    Honest gradients: Gaussian noise around a common mean
    Byzantine gradients: Various attack patterns
        - Label flipping: Large magnitude, opposite direction
        - Random noise: High variance, low correlation
        - Sybil coordination: Clustered but distinct from honest
    """
    np.random.seed(seed)

    # Common "true" gradient direction (what honest nodes should compute)
    true_gradient = np.random.randn(gradient_dim)
    true_gradient /= np.linalg.norm(true_gradient)

    # Honest gradients: Gaussian around true gradient
    honest_gradients = []
    for _ in range(num_honest):
        noise = np.random.randn(gradient_dim) * 0.1  # Small noise
        gradient = true_gradient + noise
        honest_gradients.append(gradient)

    # Byzantine gradients: Various attack types
    byzantine_gradients = []

    # Attack 1: Label flipping (33% of Byzantine)
    num_label_flip = num_byzantine // 3
    for _ in range(num_label_flip):
        # Opposite direction with large magnitude
        gradient = -true_gradient * (2.0 + np.random.rand())
        gradient += np.random.randn(gradient_dim) * 0.05
        byzantine_gradients.append(gradient)

    # Attack 2: Random noise (33% of Byzantine)
    num_random = num_byzantine // 3
    for _ in range(num_random):
        # Pure random direction
        gradient = np.random.randn(gradient_dim) * 2.0
        byzantine_gradients.append(gradient)

    # Attack 3: Sybil coordination (remaining Byzantine)
    num_sybil = num_byzantine - num_label_flip - num_random
    sybil_direction = np.random.randn(gradient_dim)
    sybil_direction /= np.linalg.norm(sybil_direction)
    for _ in range(num_sybil):
        # Coordinated but distinct direction
        noise = np.random.randn(gradient_dim) * 0.05
        gradient = sybil_direction * 0.8 + noise
        byzantine_gradients.append(gradient)

    # Combine and create labels
    all_gradients = honest_gradients + byzantine_gradients
    labels = np.array([0] * num_honest + [1] * num_byzantine)

    # Shuffle
    indices = np.random.permutation(len(all_gradients))
    all_gradients = [all_gradients[i] for i in indices]
    labels = labels[indices]

    return all_gradients, labels


def test_feature_extraction():
    """Test Phase 1a: Feature Extractor"""
    print("\n" + "="*70)
    print("PHASE 1A: FEATURE EXTRACTION TEST")
    print("="*70)

    # Generate test data
    gradients, labels = generate_synthetic_gradients(num_honest=50, num_byzantine=10)

    # Create feature extractor
    extractor = FeatureExtractor(history_window=5)

    # Extract features
    node_ids = list(range(len(gradients)))
    features_list = extract_features_batch(
        gradients, node_ids, round_num=1, extractor=extractor
    )

    # Analyze features
    pogq_honest = [f.pogq_score for f, label in zip(features_list, labels) if label == 0]
    pogq_byzantine = [f.pogq_score for f, label in zip(features_list, labels) if label == 1]

    print(f"\n✅ Extracted features for {len(features_list)} gradients")
    print(f"\nPoGQ Score Analysis:")
    print(f"  Honest nodes:    mean={np.mean(pogq_honest):.3f}, std={np.std(pogq_honest):.3f}")
    print(f"  Byzantine nodes: mean={np.mean(pogq_byzantine):.3f}, std={np.std(pogq_byzantine):.3f}")
    print(f"  Separation:      {np.mean(pogq_honest) - np.mean(pogq_byzantine):.3f}")

    # Convert to feature matrix
    X = np.array([f.to_array() for f in features_list])
    y = labels

    print(f"\n✅ Feature matrix shape: {X.shape}")
    print(f"   Feature names: {features_list[0].feature_names()}")

    return X, y


def test_classifiers(X_train, y_train, X_test, y_test):
    """Test Phase 1b: Classifiers"""
    print("\n" + "="*70)
    print("PHASE 1B: CLASSIFIER TRAINING TEST")
    print("="*70)

    evaluator = DetectionEvaluator(output_dir='results/ml')
    results = {}

    # Test 1: SVM Classifier
    print("\n📊 Training SVM Classifier...")
    svm = SVMClassifier(C=1.0, gamma='scale')
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    y_proba = svm.predict_proba(X_test)

    metrics_svm = evaluator.evaluate(y_test, y_pred, y_proba, prefix='svm_')
    results['SVM'] = metrics_svm

    print("✅ SVM Training Complete")
    print(f"   Detection Rate: {metrics_svm.detection_rate:.1%}")
    print(f"   FP Rate: {metrics_svm.false_positive_rate:.1%}")
    assert metrics_svm.meets_targets(0.95, 0.03), (
        "SVM classifier underperformed published targets."
    )

    # Test 2: Random Forest Classifier
    print("\n🌲 Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    metrics_rf = evaluator.evaluate(y_test, y_pred, y_proba, prefix='rf_')
    results['Random Forest'] = metrics_rf

    print("✅ Random Forest Training Complete")
    print(f"   Detection Rate: {metrics_rf.detection_rate:.1%}")
    print(f"   FP Rate: {metrics_rf.false_positive_rate:.1%}")
    assert metrics_rf.meets_targets(0.95, 0.03), (
        "RandomForest classifier underperformed published targets."
    )

    # Feature importance
    importance = rf.get_feature_importance()
    feature_names = ['PoGQ', 'TCDM', 'Z-score', 'Entropy', 'Gradient Norm']
    print(f"\n   Feature Importance:")
    for name, imp in zip(feature_names, importance):
        print(f"     {name:20s}: {imp:.3f}")

    # Test 3: Ensemble Classifier
    print("\n🎭 Training Ensemble Classifier...")
    ensemble = EnsembleClassifier(voting='soft')
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)

    metrics_ensemble = evaluator.evaluate(y_test, y_pred, y_proba, prefix='ensemble_')
    results['Ensemble'] = metrics_ensemble

    print("✅ Ensemble Training Complete")
    print(f"   Detection Rate: {metrics_ensemble.detection_rate:.1%}")
    print(f"   FP Rate: {metrics_ensemble.false_positive_rate:.1%}")
    assert metrics_ensemble.meets_targets(0.95, 0.03), (
        "Ensemble classifier underperformed published targets."
    )

    return results, ensemble


def test_evaluator(results):
    """Test Phase 1c: Evaluator"""
    print("\n" + "="*70)
    print("PHASE 1C: EVALUATION HARNESS TEST")
    print("="*70)

    evaluator = DetectionEvaluator(output_dir='results/ml')

    # Model comparison
    evaluator.compare_models(results)

    # Check if any model meets targets
    print("\n🎯 Target Performance (95% detection, ≤3% FP):")
    for model_name, metrics in results.items():
        meets_target = metrics.meets_targets(min_detection_rate=0.95, max_false_positive_rate=0.03)
        status = "✅ PASS" if meets_target else "❌ FAIL"
        print(f"   {model_name:15s}: {status}")
    assert any(
        metrics.meets_targets(0.95, 0.03) for metrics in results.values()
    ), "No classifier hit the 95% detection / 3% FP targets."

    print(f"\n✅ Saved visualizations to results/ml/")
    print(f"   - Confusion matrices")
    print(f"   - ROC curves")
    print(f"   - Precision-Recall curves")
    print(f"   - Model comparison plot")


def test_byzantine_detector():
    """Test Phase 1d: High-Level Detector API"""
    print("\n" + "="*70)
    print("PHASE 1D: BYZANTINE DETECTOR API TEST")
    print("="*70)

    # Generate training data
    print("\n📦 Generating training data...")
    gradients_train, labels_train = generate_synthetic_gradients(
        num_honest=500, num_byzantine=100, seed=42
    )

    # Generate test data
    gradients_test, labels_test = generate_synthetic_gradients(
        num_honest=200, num_byzantine=40, seed=123
    )

    # Extract features
    print("🔍 Extracting features...")
    extractor = FeatureExtractor()

    node_ids_train = list(range(len(gradients_train)))
    features_train = extract_features_batch(
        gradients_train, node_ids_train, round_num=1, extractor=extractor
    )
    X_train = np.array([f.to_array() for f in features_train])
    y_train = labels_train

    # Create detector
    print("🤖 Creating Byzantine Detector (ensemble mode)...")
    detector = ByzantineDetector(
        model='ensemble',
        pogq_low_threshold=0.3,
        pogq_high_threshold=0.7,
    )

    # Train
    print("🎓 Training detector...")
    metrics = detector.train(
        X_train,
        y_train,
        validate=True,
        random_state=42,
        validation_split=0.2,
        min_detection_rate=0.95,
        max_false_positive_rate=0.03,
    )

    # Test inference
    print("\n🧪 Testing inference on new gradients...")
    correct = 0
    total = len(gradients_test)

    for gradient, true_label in zip(gradients_test, labels_test):
        prediction, confidence = detector.classify_node_with_confidence(
            gradient=gradient,
            node_id=0,
            round_num=2,
            all_gradients=gradients_test,
        )

        pred_label = 1 if prediction == 'BYZANTINE' else 0

        if pred_label == true_label:
            correct += 1

    accuracy = correct / total
    print(f"✅ Inference accuracy: {accuracy:.1%} ({correct}/{total})")

    # Show statistics
    stats = detector.get_statistics()
    print(f"\n📈 Detector Statistics:")
    print(f"   Total classifications: {stats['total_classifications']}")
    print(f"   PoGQ fast reject:      {stats['pogq_fast_reject']} ({stats['pogq_fast_reject_pct']:.1%})")
    print(f"   PoGQ fast accept:      {stats['pogq_fast_accept']} ({stats['pogq_fast_accept_pct']:.1%})")
    print(f"   ML classifications:    {stats['ml_classifications']} ({stats['ml_classifications_pct']:.1%})")

    # Save model
    print(f"\n💾 Saving trained detector...")
    detector.save('models/byzantine_detector')
    print(f"✅ Saved to models/byzantine_detector/")

    return detector


def assert_recorded_bft_regressions():
    """Ensure previously recorded BFT results remain within expected bounds."""
    results_dir = Path(__file__).parent / "results"
    targets = [
        ("bft_results_10_byz.json", 1.0, 0.0),
        ("bft_results_20_byz.json", 1.0, 0.0),
        ("bft_results_30_byz.json", 1.0, 0.0),
        ("bft_results_33_byz.json", 1.0, 0.0),
        ("bft_results_40_byz.json", 0.75, 0.0),
        ("bft_results_50_byz.json", 0.60, 0.0),
    ]

    for filename, min_detection, max_fp in targets:
        payload_path = results_dir / filename
        if not payload_path.exists():
            raise AssertionError(f"Missing regression artifact: {payload_path}")

        with open(payload_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        detection_rate = payload.get("detection_rate")
        false_positive_rate = payload.get("false_positive_rate")

        if detection_rate is None or false_positive_rate is None:
            raise AssertionError(f"Unexpected schema in {filename}")

        assert detection_rate >= min_detection, (
            f"{filename} detection rate {detection_rate:.2f} "
            f"fell below regression floor {min_detection:.2f}"
        )
        assert false_positive_rate <= max_fp + 1e-9, (
            f"{filename} false positive rate {false_positive_rate:.2f} "
            f"exceeded regression ceiling {max_fp:.2f}"
        )


def main():
    """Run complete ML pipeline test"""
    print("\n" + "="*70)
    print("🚀 ML-ENHANCED BYZANTINE DETECTION - INTEGRATION TEST")
    print("="*70)
    print("\nTarget Performance:")
    print("  - Detection Rate: ≥95%")
    print("  - False Positive Rate: ≤3%")
    print("  - F1 Score: ≥0.95")

    # Phase 1a: Feature Extraction
    X, y = test_feature_extraction()

    # Split into train/test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n📊 Dataset Split:")
    print(f"   Training: {len(X_train)} samples ({np.sum(y_train)} Byzantine)")
    print(f"   Testing:  {len(X_test)} samples ({np.sum(y_test)} Byzantine)")

    # Phase 1b: Classifiers
    results, ensemble = test_classifiers(X_train, y_train, X_test, y_test)

    # Phase 1c: Evaluation
    test_evaluator(results)

    # Phase 1d: High-Level API
    detector = test_byzantine_detector()

    # Regression: previously captured BFT metrics must stay within tolerances
    assert_recorded_bft_regressions()

    # Final Summary
    print("\n" + "="*70)
    print("✅ INTEGRATION TEST COMPLETE")
    print("="*70)

    best_model = max(results.items(), key=lambda x: x[1].f1_score)
    print(f"\n🏆 Best Model: {best_model[0]}")
    print(best_model[1])

    if best_model[1].meets_targets():
        print("🎉 TARGET PERFORMANCE ACHIEVED!")
        print("   Ready for production deployment.")
    else:
        print("⚠️  Target not met. Recommendations:")
        print("   - Collect more training data")
        print("   - Tune hyperparameters")
        print("   - Add more discriminative features")

    print(f"\n📁 Results saved to:")
    print(f"   - results/ml/         (metrics, plots, reports)")
    print(f"   - models/byzantine_detector/  (trained model)")


if __name__ == '__main__':
    main()
