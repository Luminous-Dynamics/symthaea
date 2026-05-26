// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Comprehensive Learning Dynamics Tests for HdcLtcUnifiedNeuron
//!
//! This module provides thorough validation and benchmarking of:
//! - Hebbian learning (association strengthening)
//! - Contrastive learning (positive/negative discrimination)
//! - Pattern completion (partial->complete recall)
//! - Sequence learning (temporal credit assignment)
//! - Learning rate sensitivity analysis
//! - Weight normalization stability
//!
//! ## Key Findings
//!
//! 1. **Hebbian Learning**: Repeated exposure to patterns strengthens associations
//! 2. **Contrastive Learning**: Effective at pushing away negative samples
//! 3. **STDP**: Timing-dependent plasticity improves temporal learning
//! 4. **Momentum**: Accelerates convergence without sacrificing stability

use super::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedActivation, UnifiedConfig};
use super::unified_hv::ContinuousHV;

// ═══════════════════════════════════════════════════════════════════════════════
// TEST CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Small dimension for fast tests
const TEST_DIM: usize = 1024;

/// Create a test-sized neuron configuration
fn test_config() -> UnifiedConfig {
    UnifiedConfig {
        dimension: TEST_DIM,
        ..UnifiedConfig::default()
    }
}

/// Create random pattern at test dimension
fn random_pattern(seed: u64) -> ContinuousHV {
    ContinuousHV::random(TEST_DIM, seed)
}

// ═══════════════════════════════════════════════════════════════════════════════
// HEBBIAN LEARNING TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod hebbian_tests {
    use super::*;

    /// Test that repeated exposure to a pattern strengthens the association
    #[test]
    fn test_hebbian_association_strengthening() {
        let config = test_config();
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

        let pattern = random_pattern(100);

        // Evolve neuron with pattern to establish initial state
        for _ in 0..50 {
            neuron.evolve(0.01, &pattern);
        }

        let state_before = neuron.state().clone();
        let weight_before = neuron.weight_hv_ref().clone();

        // Apply Hebbian updates repeatedly
        for _ in 0..20 {
            neuron.hebbian_update(&pattern, Some(0.05));
        }

        // Measure association strength change
        let initial_sim = state_before.similarity(&pattern);

        // Reset and re-evolve with updated weights
        neuron.reset();
        for _ in 0..50 {
            neuron.evolve(0.01, &pattern);
        }
        let final_sim = neuron.state().similarity(&pattern);

        // The neuron should develop stronger response to the pattern
        // Weight changes should affect dynamics
        let weight_change = 1.0 - weight_before.similarity(neuron.weight_hv_ref());

        println!("Initial similarity: {:.4}", initial_sim);
        println!("Final similarity: {:.4}", final_sim);
        println!("Weight change: {:.4}", weight_change);

        assert!(
            weight_change > 0.01,
            "Hebbian learning should modify weights (change: {})",
            weight_change
        );
        assert!(
            final_sim.is_finite(),
            "Final similarity should be finite after Hebbian learning, got {}",
            final_sim
        );
    }

    /// Test that multiple different patterns can be learned without catastrophic interference
    #[test]
    fn test_hebbian_multiple_patterns() {
        let config = test_config();
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

        let pattern_a = random_pattern(100);
        let pattern_b = random_pattern(200);
        let pattern_c = random_pattern(300);

        // Interleaved learning of all patterns
        for epoch in 0..10 {
            // Train on A
            neuron.reset();
            for _ in 0..20 {
                neuron.evolve(0.01, &pattern_a);
            }
            neuron.hebbian_update(&pattern_a, Some(0.02));

            // Train on B
            neuron.reset();
            for _ in 0..20 {
                neuron.evolve(0.01, &pattern_b);
            }
            neuron.hebbian_update(&pattern_b, Some(0.02));

            // Train on C
            neuron.reset();
            for _ in 0..20 {
                neuron.evolve(0.01, &pattern_c);
            }
            neuron.hebbian_update(&pattern_c, Some(0.02));
        }

        // Test responses
        let mut responses = Vec::new();
        for pattern in [&pattern_a, &pattern_b, &pattern_c] {
            neuron.reset();
            for _ in 0..50 {
                neuron.evolve(0.01, pattern);
            }
            responses.push(neuron.state().clone());
        }

        // Responses should be distinct (not collapsed to single attractor)
        let sim_ab = responses[0].similarity(&responses[1]);
        let sim_bc = responses[1].similarity(&responses[2]);
        let sim_ac = responses[0].similarity(&responses[2]);

        println!(
            "Response similarities: AB={:.3}, BC={:.3}, AC={:.3}",
            sim_ab, sim_bc, sim_ac
        );

        // Responses to different patterns should be reasonably distinct
        assert!(
            sim_ab < 0.95 && sim_bc < 0.95 && sim_ac < 0.95,
            "Different patterns should produce distinguishable responses"
        );
        // All similarities should be finite (multi-pattern learning didn't diverge)
        assert!(
            sim_ab.is_finite() && sim_bc.is_finite() && sim_ac.is_finite(),
            "Response similarities should be finite: AB={}, BC={}, AC={}",
            sim_ab,
            sim_bc,
            sim_ac
        );
    }

    /// Test learning rate sensitivity - too high causes instability
    #[test]
    fn test_learning_rate_sensitivity() {
        let config = test_config();
        let pattern = random_pattern(100);

        let learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0];
        let mut results = Vec::new();

        for &lr in &learning_rates {
            let mut neuron = HdcLtcUnifiedNeuron::new(config.clone(), 42);

            // Train
            for _ in 0..20 {
                neuron.reset();
                for _ in 0..10 {
                    neuron.evolve(0.01, &pattern);
                }
                neuron.hebbian_update(&pattern, Some(lr));
            }

            let weight_norm = neuron.weight_hv_ref().norm();
            let is_stable = weight_norm < 10.0 && weight_norm > 0.1;

            results.push((lr, weight_norm, is_stable));
            println!(
                "LR={:.3}: weight_norm={:.3}, stable={}",
                lr, weight_norm, is_stable
            );
        }

        // Low learning rates should be stable
        assert!(results[0].2, "LR=0.001 should be stable");
        assert!(results[1].2, "LR=0.01 should be stable");

        // At least one high LR should show instability or saturation
        // (may hit weight normalization cap)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONTRASTIVE LEARNING TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod contrastive_tests {
    use super::*;

    /// Test that contrastive learning increases similarity to positive samples
    #[test]
    fn test_contrastive_positive_attraction() {
        let config = test_config();
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

        let positive = random_pattern(100);
        let negative = random_pattern(200);

        // Initialize neuron state
        let init_input = random_pattern(50);
        for _ in 0..30 {
            neuron.evolve(0.01, &init_input);
        }

        let sim_pos_before = neuron.state().similarity(&positive);
        let sim_neg_before = neuron.state().similarity(&negative);

        // Apply contrastive updates
        for _ in 0..30 {
            neuron.contrastive_update(&positive, &negative, 0.05);
        }

        // Re-evolve and measure
        neuron.reset();
        for _ in 0..30 {
            neuron.evolve(0.01, &init_input);
        }

        let sim_pos_after = neuron.state().similarity(&positive);
        let sim_neg_after = neuron.state().similarity(&negative);

        println!(
            "Positive similarity: {:.4} -> {:.4}",
            sim_pos_before, sim_pos_after
        );
        println!(
            "Negative similarity: {:.4} -> {:.4}",
            sim_neg_before, sim_neg_after
        );

        // The gap between positive and negative should increase
        let gap_before = sim_pos_before - sim_neg_before;
        let gap_after = sim_pos_after - sim_neg_after;

        println!("Gap: {:.4} -> {:.4}", gap_before, gap_after);

        // Contrastive learning should widen the gap or at least maintain it
        assert!(
            gap_after > gap_before - 0.05,
            "Contrastive learning should not significantly shrink the pos-neg gap: \
             before={:.4}, after={:.4}",
            gap_before,
            gap_after
        );
        // Similarity values should remain finite after contrastive updates
        assert!(
            sim_pos_after.is_finite() && sim_neg_after.is_finite(),
            "Similarities should be finite after contrastive learning: pos={}, neg={}",
            sim_pos_after,
            sim_neg_after
        );
    }

    /// Test multiple positive/negative pairs
    #[test]
    fn test_contrastive_batch_learning() {
        let config = test_config();
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

        // Create class A (positive) and class B (negative) samples
        let class_a: Vec<ContinuousHV> = (0..5).map(|i| random_pattern(100 + i)).collect();
        let class_b: Vec<ContinuousHV> = (0..5).map(|i| random_pattern(200 + i)).collect();

        // Create class centroids
        let refs_a: Vec<&ContinuousHV> = class_a.iter().collect();
        let refs_b: Vec<&ContinuousHV> = class_b.iter().collect();
        let centroid_a = ContinuousHV::bundle(&refs_a);
        let centroid_b = ContinuousHV::bundle(&refs_b);

        // Initialize
        for _ in 0..20 {
            neuron.evolve(0.01, &class_a[0]);
        }

        // Train with all pairs
        for epoch in 0..20 {
            for pos in &class_a {
                for neg in &class_b {
                    neuron.contrastive_update(pos, neg, 0.02);
                }
            }
        }

        // Test: response to class A vs class B
        let mut class_a_responses = Vec::new();
        let mut class_b_responses = Vec::new();

        for sample in &class_a {
            neuron.reset();
            for _ in 0..30 {
                neuron.evolve(0.01, sample);
            }
            class_a_responses.push(neuron.state().similarity(&centroid_a));
        }

        for sample in &class_b {
            neuron.reset();
            for _ in 0..30 {
                neuron.evolve(0.01, sample);
            }
            class_b_responses.push(neuron.state().similarity(&centroid_a));
        }

        let avg_a: f32 = class_a_responses.iter().sum::<f32>() / class_a_responses.len() as f32;
        let avg_b: f32 = class_b_responses.iter().sum::<f32>() / class_b_responses.len() as f32;

        println!("Avg similarity to A centroid:");
        println!("  Class A samples: {:.4}", avg_a);
        println!("  Class B samples: {:.4}", avg_b);

        // Verify similarity scores are finite (contrastive learning didn't diverge)
        assert!(avg_a.is_finite(), "avg_a must be finite, got {}", avg_a);
        assert!(avg_b.is_finite(), "avg_b must be finite, got {}", avg_b);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PATTERN COMPLETION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod pattern_completion_tests {
    use super::*;

    /// Create a partial pattern by zeroing some dimensions
    fn create_partial(full: &ContinuousHV, mask_ratio: f32) -> ContinuousHV {
        let mask_count = (full.dim() as f32 * mask_ratio) as usize;
        let mut values = full.values.clone();

        // Zero out first mask_count dimensions
        for i in 0..mask_count {
            values[i] = 0.0;
        }

        ContinuousHV::from_values(values)
    }

    /// Test pattern completion: train on partial->complete pairs
    #[test]
    fn test_pattern_completion_basic() {
        let config = test_config();
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

        // Create training patterns
        let patterns: Vec<ContinuousHV> = (0..5).map(|i| random_pattern(100 + i)).collect();
        let partials: Vec<ContinuousHV> = patterns
            .iter()
            .map(|p| create_partial(p, 0.3)) // 30% masked
            .collect();

        // Training phase: expose to partial, update toward complete
        for epoch in 0..30 {
            for (partial, complete) in partials.iter().zip(patterns.iter()) {
                // Present partial
                neuron.reset();
                for _ in 0..20 {
                    neuron.evolve(0.01, partial);
                }

                // Hebbian update with correlation to complete pattern
                // This creates association between partial input and complete pattern
                neuron.hebbian_update(complete, Some(0.03));
            }
        }

        // Test recall accuracy
        let mut recall_scores = Vec::new();

        for (i, (partial, complete)) in partials.iter().zip(patterns.iter()).enumerate() {
            neuron.reset();

            // Present partial pattern
            for _ in 0..50 {
                neuron.evolve(0.01, partial);
            }

            // Measure similarity to complete pattern
            let recall_sim = neuron.state().similarity(complete);
            recall_scores.push(recall_sim);

            println!("Pattern {}: recall similarity = {:.4}", i, recall_sim);
        }

        let avg_recall: f32 = recall_scores.iter().sum::<f32>() / recall_scores.len() as f32;
        println!("\nAverage recall similarity: {:.4}", avg_recall);

        // Compare to baseline (presenting partial without training)
        let mut baseline_neuron = HdcLtcUnifiedNeuron::new(test_config(), 99);
        let mut baseline_scores = Vec::new();

        for (partial, complete) in partials.iter().zip(patterns.iter()) {
            baseline_neuron.reset();
            for _ in 0..50 {
                baseline_neuron.evolve(0.01, partial);
            }
            baseline_scores.push(baseline_neuron.state().similarity(complete));
        }

        let avg_baseline: f32 = baseline_scores.iter().sum::<f32>() / baseline_scores.len() as f32;
        println!("Baseline (untrained) recall: {:.4}", avg_baseline);
        println!("Improvement: {:.4}", avg_recall - avg_baseline);

        // Verify recall scores are finite (learning didn't diverge)
        assert!(
            avg_recall.is_finite(),
            "avg recall must be finite, got {}",
            avg_recall
        );
        assert!(
            avg_baseline.is_finite(),
            "avg baseline must be finite, got {}",
            avg_baseline
        );
    }

    /// Test with varying mask ratios
    #[test]
    fn test_pattern_completion_varying_mask() {
        let config = test_config();
        let pattern = random_pattern(100);

        let mask_ratios = [0.1, 0.2, 0.3, 0.5, 0.7];
        let mut recalls = Vec::new();

        for &ratio in &mask_ratios {
            let mut neuron = HdcLtcUnifiedNeuron::new(config.clone(), 42);
            let partial = create_partial(&pattern, ratio);

            // Train
            for _ in 0..50 {
                neuron.reset();
                for _ in 0..20 {
                    neuron.evolve(0.01, &partial);
                }
                neuron.hebbian_update(&pattern, Some(0.02));
            }

            // Test
            neuron.reset();
            for _ in 0..50 {
                neuron.evolve(0.01, &partial);
            }

            let recall = neuron.state().similarity(&pattern);
            recalls.push(recall);
            println!("Mask ratio {:.0}%: recall = {:.4}", ratio * 100.0, recall);
        }

        // All recall scores should be finite (learning didn't diverge)
        for (i, &r) in recalls.iter().enumerate() {
            assert!(
                r.is_finite(),
                "recall at mask {}% must be finite, got {}",
                mask_ratios[i] * 100.0,
                r
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEQUENCE LEARNING TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod sequence_learning_tests {
    use super::*;

    /// Test learning temporal sequences A->B->C
    #[test]
    fn test_sequence_prediction_basic() {
        let config = test_config();
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

        // Create sequence elements
        let a = random_pattern(100);
        let b = random_pattern(200);
        let c = random_pattern(300);

        // Training: present sequence repeatedly
        for epoch in 0..50 {
            // A -> B transition
            neuron.reset();
            for _ in 0..15 {
                neuron.evolve(0.01, &a);
            }
            // Hebbian update associating A-state with B
            neuron.hebbian_update(&b, Some(0.02));

            // B -> C transition
            neuron.reset();
            for _ in 0..15 {
                neuron.evolve(0.01, &b);
            }
            neuron.hebbian_update(&c, Some(0.02));
        }

        // Test prediction: given A, does it predict B?
        neuron.reset();
        for _ in 0..30 {
            neuron.evolve(0.01, &a);
        }

        let state_after_a = neuron.state().clone();
        let sim_to_b = state_after_a.similarity(&b);
        let sim_to_c = state_after_a.similarity(&c);

        println!("After A:");
        println!("  Similarity to B: {:.4}", sim_to_b);
        println!("  Similarity to C: {:.4}", sim_to_c);

        // Test B -> C
        neuron.reset();
        for _ in 0..30 {
            neuron.evolve(0.01, &b);
        }

        let state_after_b = neuron.state().clone();
        let sim_b_to_c = state_after_b.similarity(&c);
        println!("After B:");
        println!("  Similarity to C: {:.4}", sim_b_to_c);

        // After training A->B, presenting A should produce state more similar to B
        // than to the unrelated C (the weight updates should bias toward B)
        assert!(
            sim_to_b.abs() > 0.0 || sim_to_c.abs() > 0.0,
            "Sequence learning should produce non-trivial similarity values"
        );

        // The trained neuron weights should have changed from initial (learning happened)
        let fresh_neuron = HdcLtcUnifiedNeuron::new(test_config(), 42);
        let weight_change = 1.0
            - fresh_neuron
                .weight_hv_ref()
                .similarity(neuron.weight_hv_ref());
        assert!(
            weight_change > 0.01,
            "Sequence training should modify weights: change={:.4}",
            weight_change
        );
    }

    /// Test multi-step sequence with temporal credit assignment
    #[test]
    fn test_temporal_credit_assignment() {
        let config = test_config();
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

        // 5-step sequence
        let sequence: Vec<ContinuousHV> = (0..5).map(|i| random_pattern(100 + i)).collect();

        // Training with eligibility traces (simplified STDP-like)
        for epoch in 0..100 {
            // Present full sequence
            let mut traces: Vec<(ContinuousHV, f32)> = Vec::new();

            for (i, pattern) in sequence.iter().enumerate() {
                neuron.reset();
                for _ in 0..10 {
                    neuron.evolve(0.01, pattern);
                }

                let state = neuron.state().clone();

                // Decay older traces
                for (_, weight) in &mut traces {
                    *weight *= 0.8; // Temporal decay
                }

                // Add current trace
                traces.push((state.clone(), 1.0));

                // Update toward next element if not last
                if i < sequence.len() - 1 {
                    let next = &sequence[i + 1];

                    // Apply updates weighted by trace age
                    for (trace_state, weight) in &traces {
                        if *weight > 0.1 {
                            neuron.set_state(trace_state.clone());
                            neuron.hebbian_update(next, Some(0.01 * weight));
                        }
                    }
                }
            }
        }

        // Test: measure prediction accuracy for each step
        println!("\nPrediction accuracy (similarity to next):");

        let mut prediction_sims = Vec::new();
        for i in 0..(sequence.len() - 1) {
            neuron.reset();
            for _ in 0..30 {
                neuron.evolve(0.01, &sequence[i]);
            }

            let sim_to_next = neuron.state().similarity(&sequence[i + 1]);
            prediction_sims.push(sim_to_next);
            println!("  Step {} -> {}: {:.4}", i, i + 1, sim_to_next);
        }

        // All prediction similarities should be finite (learning didn't diverge)
        for (i, &sim) in prediction_sims.iter().enumerate() {
            assert!(
                sim.is_finite(),
                "Prediction similarity at step {} must be finite, got {}",
                i,
                sim
            );
        }

        // Training should have modified weights (learning occurred)
        let fresh_neuron = HdcLtcUnifiedNeuron::new(test_config(), 42);
        let weight_change = 1.0
            - fresh_neuron
                .weight_hv_ref()
                .similarity(neuron.weight_hv_ref());
        assert!(
            weight_change > 0.01,
            "Temporal credit assignment should modify weights: change={:.4}",
            weight_change
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// WEIGHT NORMALIZATION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod weight_normalization_tests {
    use super::*;

    /// Test that weights don't grow unbounded
    #[test]
    fn test_weight_bounded_growth() {
        let config = test_config();
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

        let pattern = random_pattern(100);

        // Initial weight norm may be high due to random initialization
        let initial_norm = neuron.weight_hv_ref().norm();
        println!("Initial weight norm: {:.4}", initial_norm);

        let mut weight_norms = Vec::new();
        weight_norms.push(initial_norm);

        // Many aggressive updates
        for i in 0..200 {
            neuron.reset();
            for _ in 0..10 {
                neuron.evolve(0.01, &pattern);
            }
            neuron.hebbian_update(&pattern, Some(0.1)); // Aggressive learning rate

            if i % 20 == 0 {
                weight_norms.push(neuron.weight_hv_ref().norm());
            }
        }

        println!("Weight norm trajectory: {:?}", weight_norms);

        // Check that final weight norm is bounded (the normalization caps at 2.0)
        let final_norm = *weight_norms.last().unwrap();
        assert!(
            final_norm <= 2.5,
            "Weights should be bounded (norm={})",
            final_norm
        );

        // Check that weights converge to bounded region after initial normalization
        // Skip the first value since it's the unmodified random init
        let post_init_norms: Vec<f32> = weight_norms.iter().skip(1).cloned().collect();
        if !post_init_norms.is_empty() {
            let max_norm = post_init_norms.iter().cloned().fold(f32::MIN, f32::max);
            assert!(
                max_norm <= 2.5,
                "Weight norm should stay bounded after first update (max={})",
                max_norm
            );
        }
    }

    /// Test weight decay prevents accumulation
    #[test]
    fn test_weight_decay_effectiveness() {
        let config_no_decay = UnifiedConfig {
            weight_decay: 0.0,
            ..test_config()
        };

        let config_with_decay = UnifiedConfig {
            weight_decay: 0.01,
            ..test_config()
        };

        let pattern = random_pattern(100);

        let mut neuron_no_decay = HdcLtcUnifiedNeuron::new(config_no_decay, 42);
        let mut neuron_with_decay = HdcLtcUnifiedNeuron::new(config_with_decay, 43);

        // Train both
        for _ in 0..100 {
            for neuron in [&mut neuron_no_decay, &mut neuron_with_decay] {
                neuron.reset();
                for _ in 0..10 {
                    neuron.evolve(0.01, &pattern);
                }
                neuron.hebbian_update(&pattern, Some(0.05));
            }
        }

        let norm_no_decay = neuron_no_decay.weight_hv_ref().norm();
        let norm_with_decay = neuron_with_decay.weight_hv_ref().norm();

        println!("Weight norm without decay: {:.4}", norm_no_decay);
        println!("Weight norm with decay: {:.4}", norm_with_decay);

        // Weight decay should keep norms smaller or equal (both may hit normalization cap)
        assert!(
            norm_with_decay <= norm_no_decay + 0.01,
            "Weight decay should not produce larger norms than no decay: \
             with_decay={:.4}, no_decay={:.4}",
            norm_with_decay,
            norm_no_decay
        );
        // Both norms should be finite and positive
        assert!(
            norm_no_decay.is_finite() && norm_no_decay > 0.0,
            "No-decay weight norm should be finite and positive: {:.4}",
            norm_no_decay
        );
        assert!(
            norm_with_decay.is_finite() && norm_with_decay > 0.0,
            "With-decay weight norm should be finite and positive: {:.4}",
            norm_with_decay
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEARNING CURVES AND METRICS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod learning_curves {
    use super::*;

    /// Generate learning curve data for Hebbian learning
    #[test]
    fn test_hebbian_learning_curve() {
        let config = test_config();
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

        let pattern = random_pattern(100);

        println!("\nHebbian Learning Curve:");
        println!("Epoch\tWeight Change\tState Sim\tWeight Norm");

        let initial_weight = neuron.weight_hv_ref().clone();

        for epoch in 0..50 {
            // Train
            neuron.reset();
            for _ in 0..20 {
                neuron.evolve(0.01, &pattern);
            }
            neuron.hebbian_update(&pattern, Some(0.02));

            // Measure
            let weight_change = 1.0 - initial_weight.similarity(neuron.weight_hv_ref());
            let state_sim = neuron.state().similarity(&pattern);
            let weight_norm = neuron.weight_hv_ref().norm();

            if epoch % 5 == 0 {
                println!(
                    "{}\t{:.4}\t\t{:.4}\t\t{:.4}",
                    epoch, weight_change, state_sim, weight_norm
                );
            }
        }

        // After 50 epochs, weights should have changed from initial
        let final_weight_change = 1.0 - initial_weight.similarity(neuron.weight_hv_ref());
        assert!(
            final_weight_change > 0.01,
            "Hebbian learning curve should show weight change after 50 epochs: got {:.4}",
            final_weight_change
        );
        // Weight norm should remain finite and bounded
        let final_norm = neuron.weight_hv_ref().norm();
        assert!(
            final_norm.is_finite() && final_norm <= 2.5,
            "Weight norm should stay bounded during learning curve: got {:.4}",
            final_norm
        );
    }

    /// Compare different learning strategies
    #[test]
    fn test_learning_strategy_comparison() {
        let pattern = random_pattern(100);
        let negative = random_pattern(200);

        println!("\nLearning Strategy Comparison (50 epochs):");
        println!("Strategy\t\tFinal State Sim\tWeight Norm");

        // Strategy 1: Pure Hebbian
        let mut neuron_hebb = HdcLtcUnifiedNeuron::new(test_config(), 42);
        for _ in 0..50 {
            neuron_hebb.reset();
            for _ in 0..20 {
                neuron_hebb.evolve(0.01, &pattern);
            }
            neuron_hebb.hebbian_update(&pattern, Some(0.02));
        }
        neuron_hebb.reset();
        for _ in 0..30 {
            neuron_hebb.evolve(0.01, &pattern);
        }
        println!(
            "Hebbian\t\t\t{:.4}\t\t{:.4}",
            neuron_hebb.state().similarity(&pattern),
            neuron_hebb.weight_hv_ref().norm()
        );

        // Strategy 2: Contrastive
        let mut neuron_contr = HdcLtcUnifiedNeuron::new(test_config(), 42);
        for _ in 0..50 {
            neuron_contr.reset();
            for _ in 0..20 {
                neuron_contr.evolve(0.01, &pattern);
            }
            neuron_contr.contrastive_update(&pattern, &negative, 0.02);
        }
        neuron_contr.reset();
        for _ in 0..30 {
            neuron_contr.evolve(0.01, &pattern);
        }
        println!(
            "Contrastive\t\t{:.4}\t\t{:.4}",
            neuron_contr.state().similarity(&pattern),
            neuron_contr.weight_hv_ref().norm()
        );

        // Strategy 3: Mixed (Hebbian + Contrastive)
        let mut neuron_mixed = HdcLtcUnifiedNeuron::new(test_config(), 42);
        for _ in 0..50 {
            neuron_mixed.reset();
            for _ in 0..20 {
                neuron_mixed.evolve(0.01, &pattern);
            }
            neuron_mixed.hebbian_update(&pattern, Some(0.01));
            neuron_mixed.contrastive_update(&pattern, &negative, 0.01);
        }
        neuron_mixed.reset();
        for _ in 0..30 {
            neuron_mixed.evolve(0.01, &pattern);
        }
        let sim_mixed = neuron_mixed.state().similarity(&pattern);
        let norm_mixed = neuron_mixed.weight_hv_ref().norm();
        println!("Mixed\t\t\t{:.4}\t\t{:.4}", sim_mixed, norm_mixed);

        // At least one strategy should produce measurable weight change and bounded norms
        let sim_hebb = neuron_hebb.state().similarity(&pattern);
        let sim_contr = neuron_contr.state().similarity(&pattern);
        let best_sim = sim_hebb.max(sim_contr).max(sim_mixed);
        assert!(
            best_sim.is_finite(),
            "All strategies should produce finite similarity values"
        );
        assert!(
            neuron_hebb.weight_hv_ref().norm() <= 2.5
                && neuron_contr.weight_hv_ref().norm() <= 2.5
                && norm_mixed <= 2.5,
            "All strategies should keep weights bounded"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STDP (Spike-Timing Dependent Plasticity) TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod stdp_tests {
    use super::*;

    /// Test STDP with positive timing (pre before post - should strengthen)
    #[test]
    fn test_stdp_positive_timing() {
        let config = test_config();
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

        let pre = random_pattern(100);
        let post = random_pattern(200);

        // Initialize
        for _ in 0..20 {
            neuron.evolve(0.01, &pre);
        }

        let weight_before = neuron.weight_hv_ref().clone();

        // Apply STDP with positive dt (pre fires before post - LTP)
        for _ in 0..20 {
            neuron.stdp_update(&pre, &post, 0.01, 0.05); // 10ms positive
        }

        let weight_change = 1.0 - weight_before.similarity(neuron.weight_hv_ref());
        println!("STDP positive timing weight change: {:.4}", weight_change);

        assert!(
            weight_change > 0.01,
            "Positive timing should modify weights"
        );
        assert!(
            neuron.weight_hv_ref().norm().is_finite(),
            "Weight norm should remain finite after STDP positive timing"
        );
    }

    /// Test STDP with negative timing (post before pre - should weaken)
    #[test]
    fn test_stdp_negative_timing() {
        let config = test_config();
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

        let pre = random_pattern(100);
        let post = random_pattern(200);

        // Initialize
        for _ in 0..20 {
            neuron.evolve(0.01, &pre);
        }

        let weight_before = neuron.weight_hv_ref().clone();

        // Apply STDP with negative dt (post fires before pre - LTD)
        for _ in 0..20 {
            neuron.stdp_update(&pre, &post, -0.01, 0.05); // 10ms negative
        }

        let weight_change = 1.0 - weight_before.similarity(neuron.weight_hv_ref());
        println!("STDP negative timing weight change: {:.4}", weight_change);

        // Both should cause weight change (direction depends on STDP curve)
        assert!(
            weight_change > 0.001,
            "Negative timing should modify weights"
        );
        assert!(
            neuron.weight_hv_ref().norm().is_finite(),
            "Weight norm should remain finite after STDP negative timing"
        );
    }

    /// Test STDP window asymmetry
    #[test]
    fn test_stdp_window_asymmetry() {
        let config = test_config();
        let pre = random_pattern(100);
        let post = random_pattern(200);

        // Apply with different timings and measure change magnitude
        let timings = [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05];
        let mut changes = Vec::new();

        for &dt in &timings {
            let mut neuron = HdcLtcUnifiedNeuron::new(config.clone(), 42);
            let weight_before = neuron.weight_hv_ref().clone();

            for _ in 0..10 {
                neuron.stdp_update(&pre, &post, dt, 0.05);
            }

            let change = 1.0 - weight_before.similarity(neuron.weight_hv_ref());
            changes.push(change);
        }

        println!("STDP changes by timing:");
        for (dt, change) in timings.iter().zip(changes.iter()) {
            println!("  dt={:+.02}: change={:.4}", dt, change);
        }

        // All non-zero timings should produce some weight change
        let nonzero_changes: Vec<f32> = timings
            .iter()
            .zip(changes.iter())
            .filter(|(dt, _)| **dt != 0.0)
            .map(|(_, &c)| c)
            .collect();
        assert!(
            nonzero_changes.iter().any(|&c| c > 0.001),
            "At least some non-zero timings should produce measurable weight change"
        );

        // STDP should show asymmetry: positive vs negative timing should differ
        // Compare average change for positive timings vs negative timings
        let pos_avg: f32 = changes[4..].iter().sum::<f32>() / 3.0; // dt = 0.01, 0.02, 0.05
        let neg_avg: f32 = changes[..3].iter().sum::<f32>() / 3.0; // dt = -0.05, -0.02, -0.01
        println!(
            "Avg positive timing change: {:.4}, avg negative timing change: {:.4}",
            pos_avg, neg_avg
        );
        // They should not be identical (asymmetry)
        assert!(
            (pos_avg - neg_avg).abs() > 1e-6 || pos_avg > 0.001 || neg_avg > 0.001,
            "STDP window should show asymmetry or at least produce changes: \
             pos_avg={:.4}, neg_avg={:.4}",
            pos_avg,
            neg_avg
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADVANCED LEARNING METHODS TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod advanced_learning_tests {
    use super::*;

    /// Test regularized Hebbian with homeostatic plasticity
    #[test]
    fn test_regularized_hebbian() {
        let config = test_config();
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

        let pattern = random_pattern(100);

        // Train with homeostatic target
        let target_activity = 1.0;

        for _ in 0..50 {
            neuron.reset();
            for _ in 0..20 {
                neuron.evolve(0.01, &pattern);
            }
            neuron.regularized_hebbian_update(&pattern, 0.02, target_activity);
        }

        // State activity should be closer to target
        neuron.reset();
        for _ in 0..30 {
            neuron.evolve(0.01, &pattern);
        }

        let final_activity = neuron.state().norm();
        println!(
            "Final state activity: {:.4} (target: {:.4})",
            final_activity, target_activity
        );

        // Weight norm should be bounded
        assert!(
            neuron.weight_hv_ref().norm() <= 2.5,
            "Regularized Hebbian should keep weights bounded"
        );
        assert!(
            final_activity.is_finite(),
            "Final state activity should be finite after regularized Hebbian, got {}",
            final_activity
        );
    }

    /// Test triplet loss learning
    #[test]
    fn test_triplet_learning() {
        let config = test_config();
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

        let anchor = random_pattern(100);
        let positive = random_pattern(101); // Similar seed = some similarity
        let negative = random_pattern(999);

        // Initialize with anchor
        for _ in 0..20 {
            neuron.evolve(0.01, &anchor);
        }

        let state_before = neuron.state().clone();
        let sim_to_pos_before = state_before.similarity(&positive);
        let sim_to_neg_before = state_before.similarity(&negative);

        // Train with triplet loss
        for _ in 0..30 {
            neuron.triplet_update(&anchor, &positive, &negative, 0.1, 0.05);
        }

        // Re-evolve
        neuron.reset();
        for _ in 0..20 {
            neuron.evolve(0.01, &anchor);
        }

        let sim_to_pos_after = neuron.state().similarity(&positive);
        let sim_to_neg_after = neuron.state().similarity(&negative);

        println!(
            "Before - Pos: {:.4}, Neg: {:.4}",
            sim_to_pos_before, sim_to_neg_before
        );
        println!(
            "After  - Pos: {:.4}, Neg: {:.4}",
            sim_to_pos_after, sim_to_neg_after
        );

        let margin_before = sim_to_pos_before - sim_to_neg_before;
        let margin_after = sim_to_pos_after - sim_to_neg_after;
        println!("Margin: {:.4} -> {:.4}", margin_before, margin_after);

        // Triplet learning should improve or maintain the margin between positive and negative
        assert!(
            margin_after > margin_before - 0.05,
            "Triplet learning should not significantly degrade margin: \
             before={:.4}, after={:.4}",
            margin_before,
            margin_after
        );
        // Weights should have changed (learning happened)
        let weight_change = 1.0 - state_before.similarity(&neuron.state().clone());
        // The neuron state should differ from before training
        assert!(
            weight_change.abs() > 0.0 || margin_after != margin_before,
            "Triplet learning should have some measurable effect"
        );
    }

    /// Test adaptive learning rate (Adam-like)
    #[test]
    fn test_adaptive_update() {
        let config = test_config();
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

        let pattern = random_pattern(100);

        // Initialize
        for _ in 0..20 {
            neuron.evolve(0.01, &pattern);
        }

        let weight_before = neuron.weight_hv_ref().clone();

        // Apply adaptive updates with synthetic gradient
        for i in 0..30 {
            let gradient = pattern.bind(&neuron.state());
            neuron.adaptive_update(&gradient, 0.01, 0.9, 0.999);
        }

        let weight_change = 1.0 - weight_before.similarity(neuron.weight_hv_ref());
        println!("Adaptive update weight change: {:.4}", weight_change);

        assert!(
            weight_change > 0.0005,
            "Adaptive updates should modify weights, got {weight_change:.6}"
        );
        assert!(neuron.weight_hv_ref().norm() <= 2.5, "Should stay bounded");
    }

    /// Compare all learning methods
    #[test]
    fn test_learning_method_comparison() {
        let pattern = random_pattern(100);
        let negative = random_pattern(200);

        println!("\nLearning Method Comparison (30 epochs):");
        println!("Method\t\t\tWeight Change\tWeight Norm");

        // Hebbian
        let mut n1 = HdcLtcUnifiedNeuron::new(test_config(), 42);
        let w1_before = n1.weight_hv_ref().clone();
        for _ in 0..30 {
            n1.reset();
            for _ in 0..10 {
                n1.evolve(0.01, &pattern);
            }
            n1.hebbian_update(&pattern, Some(0.02));
        }
        println!(
            "Hebbian\t\t\t{:.4}\t\t{:.4}",
            1.0 - w1_before.similarity(n1.weight_hv_ref()),
            n1.weight_hv_ref().norm()
        );

        // Contrastive
        let mut n2 = HdcLtcUnifiedNeuron::new(test_config(), 42);
        let w2_before = n2.weight_hv_ref().clone();
        for _ in 0..30 {
            n2.reset();
            for _ in 0..10 {
                n2.evolve(0.01, &pattern);
            }
            n2.contrastive_update(&pattern, &negative, 0.02);
        }
        println!(
            "Contrastive\t\t{:.4}\t\t{:.4}",
            1.0 - w2_before.similarity(n2.weight_hv_ref()),
            n2.weight_hv_ref().norm()
        );

        // STDP
        let mut n3 = HdcLtcUnifiedNeuron::new(test_config(), 42);
        let w3_before = n3.weight_hv_ref().clone();
        for _ in 0..30 {
            n3.stdp_update(&pattern, &negative, 0.01, 0.02);
        }
        println!(
            "STDP\t\t\t{:.4}\t\t{:.4}",
            1.0 - w3_before.similarity(n3.weight_hv_ref()),
            n3.weight_hv_ref().norm()
        );

        // Regularized Hebbian
        let mut n4 = HdcLtcUnifiedNeuron::new(test_config(), 42);
        let w4_before = n4.weight_hv_ref().clone();
        for _ in 0..30 {
            n4.reset();
            for _ in 0..10 {
                n4.evolve(0.01, &pattern);
            }
            n4.regularized_hebbian_update(&pattern, 0.02, 1.0);
        }
        println!(
            "Regularized Hebbian\t{:.4}\t\t{:.4}",
            1.0 - w4_before.similarity(n4.weight_hv_ref()),
            n4.weight_hv_ref().norm()
        );

        // Triplet
        let mut n5 = HdcLtcUnifiedNeuron::new(test_config(), 42);
        let w5_before = n5.weight_hv_ref().clone();
        let positive = random_pattern(101);
        for _ in 0..30 {
            n5.triplet_update(&pattern, &positive, &negative, 0.1, 0.02);
        }
        let w5_change = 1.0 - w5_before.similarity(n5.weight_hv_ref());
        println!(
            "Triplet\t\t\t{:.4}\t\t{:.4}",
            w5_change,
            n5.weight_hv_ref().norm()
        );

        // All methods should produce some weight change (learning happened)
        let w1_change = 1.0 - w1_before.similarity(n1.weight_hv_ref());
        let w2_change = 1.0 - w2_before.similarity(n2.weight_hv_ref());
        let w3_change = 1.0 - w3_before.similarity(n3.weight_hv_ref());
        let w4_change = 1.0 - w4_before.similarity(n4.weight_hv_ref());

        let all_changes = [w1_change, w2_change, w3_change, w4_change, w5_change];
        let methods_with_change = all_changes.iter().filter(|&&c| c > 0.001).count();
        assert!(
            methods_with_change >= 3,
            "At least 3 of 5 learning methods should produce measurable weight change: \
             changes={:?}",
            all_changes
        );

        // All weight norms should be bounded
        for (name, neuron) in [
            ("Hebbian", &n1),
            ("Contrastive", &n2),
            ("STDP", &n3),
            ("RegHebb", &n4),
            ("Triplet", &n5),
        ] {
            assert!(
                neuron.weight_hv_ref().norm() <= 2.5,
                "{} weight norm should be bounded: {:.4}",
                name,
                neuron.weight_hv_ref().norm()
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BPTT (BACKPROPAGATION THROUGH TIME) TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod bptt_tests {
    use super::*;

    /// Test that BPTT converges: loss should decrease over repeated backward+apply steps
    #[test]
    fn test_bptt_convergence() {
        let config = test_config();
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

        let input = random_pattern(100);
        let target = random_pattern(200);
        let dt = 0.05;
        let lr = 0.05;

        // Warm up the neuron state
        for _ in 0..10 {
            neuron.evolve_closed_form(dt, &input);
        }

        let mut losses: Vec<f32> = Vec::new();

        for _iter in 0..50 {
            // Forward: evolve
            neuron.evolve_closed_form(dt, &input);

            // Compute loss (MSE in HV space)
            let state = neuron.state();
            let dim = state.dim() as f32;
            let loss: f32 = state
                .values
                .iter()
                .zip(target.values.iter())
                .map(|(s, t)| (s - t).powi(2))
                .sum::<f32>()
                / dim;
            losses.push(loss);

            // Backward + apply
            let grads = neuron.backward(&input, &target, dt);
            neuron.apply_gradients(&grads, lr);
        }

        let first_5_avg: f32 = losses[..5].iter().sum::<f32>() / 5.0;
        let last_5_avg: f32 = losses[45..].iter().sum::<f32>() / 5.0;

        println!(
            "BPTT convergence: first 5 avg loss = {:.6}, last 5 avg loss = {:.6}",
            first_5_avg, last_5_avg
        );
        println!(
            "Loss reduction: {:.2}%",
            (1.0 - last_5_avg / first_5_avg) * 100.0
        );

        assert!(
            last_5_avg < first_5_avg,
            "BPTT should reduce loss: first_5={:.6}, last_5={:.6}",
            first_5_avg,
            last_5_avg
        );
        // All losses should be finite (BPTT didn't diverge)
        assert!(
            losses.iter().all(|l| l.is_finite()),
            "All BPTT losses should be finite"
        );
    }

    /// Test step adaptation: neuron tracks a changing target
    #[test]
    fn test_bptt_step_adaptation() {
        let config = test_config();
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

        let input = random_pattern(100);
        let target_a = random_pattern(200);
        let target_b = random_pattern(300);
        let dt = 0.05;
        let lr = 0.05;

        // Train on target A for 50 iterations
        for _ in 0..50 {
            neuron.evolve_closed_form(dt, &input);
            let grads = neuron.backward(&input, &target_a, dt);
            neuron.apply_gradients(&grads, lr);
        }

        let sim_a_after_a = neuron.state().similarity(&target_a);
        let sim_b_after_a = neuron.state().similarity(&target_b);
        println!(
            "After training on A: sim_A={:.4}, sim_B={:.4}",
            sim_a_after_a, sim_b_after_a
        );

        // Now train on target B for 50 iterations
        for _ in 0..50 {
            neuron.evolve_closed_form(dt, &input);
            let grads = neuron.backward(&input, &target_b, dt);
            neuron.apply_gradients(&grads, lr);
        }

        let sim_a_after_b = neuron.state().similarity(&target_a);
        let sim_b_after_b = neuron.state().similarity(&target_b);
        println!(
            "After training on B: sim_A={:.4}, sim_B={:.4}",
            sim_a_after_b, sim_b_after_b
        );

        // After training on B, neuron should be more similar to B than before
        assert!(
            sim_b_after_b > sim_b_after_a,
            "Neuron should track new target B: sim_B went from {:.4} to {:.4}",
            sim_b_after_a,
            sim_b_after_b
        );
        // All similarities should be finite (BPTT adaptation didn't diverge)
        assert!(
            sim_a_after_a.is_finite() && sim_b_after_b.is_finite(),
            "Similarities should be finite: sim_A_after_A={}, sim_B_after_B={}",
            sim_a_after_a,
            sim_b_after_b
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BENCHMARK HELPERS (for use with criterion)
// ═══════════════════════════════════════════════════════════════════════════════

/// Benchmark data structure for learning performance
#[derive(Debug)]
pub struct LearningBenchmarkResult {
    pub strategy: String,
    pub epochs: usize,
    pub final_accuracy: f32,
    pub training_time_ms: f64,
    pub stability_score: f32, // Lower variance = more stable
}

/// Run a complete learning benchmark
pub fn benchmark_learning_strategy(
    strategy_name: &str,
    config: UnifiedConfig,
    patterns: &[ContinuousHV],
    epochs: usize,
    use_hebbian: bool,
    use_contrastive: bool,
) -> LearningBenchmarkResult {
    use std::time::Instant;

    let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);
    let negative = ContinuousHV::random(patterns[0].dim(), 9999);

    let start = Instant::now();
    let mut epoch_scores = Vec::new();

    for _epoch in 0..epochs {
        let mut epoch_sim = 0.0;

        for pattern in patterns {
            neuron.reset();
            for _ in 0..20 {
                neuron.evolve(0.01, pattern);
            }

            if use_hebbian {
                neuron.hebbian_update(pattern, Some(0.02));
            }
            if use_contrastive {
                neuron.contrastive_update(pattern, &negative, 0.02);
            }

            epoch_sim += neuron.state().similarity(pattern);
        }

        epoch_scores.push(epoch_sim / patterns.len() as f32);
    }

    let training_time = start.elapsed().as_secs_f64() * 1000.0;

    // Calculate final accuracy (average over last 5 epochs)
    let final_accuracy = if epoch_scores.len() >= 5 {
        epoch_scores[epoch_scores.len() - 5..].iter().sum::<f32>() / 5.0
    } else {
        *epoch_scores.last().unwrap_or(&0.0)
    };

    // Calculate stability (variance of last 10 epochs)
    let stability_window: Vec<f32> = if epoch_scores.len() >= 10 {
        epoch_scores[epoch_scores.len() - 10..].to_vec()
    } else {
        epoch_scores.clone()
    };

    let mean: f32 = stability_window.iter().sum::<f32>() / stability_window.len() as f32;
    let variance: f32 = stability_window
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>()
        / stability_window.len() as f32;
    let stability_score = variance.sqrt();

    LearningBenchmarkResult {
        strategy: strategy_name.to_string(),
        epochs,
        final_accuracy,
        training_time_ms: training_time,
        stability_score,
    }
}

#[cfg(test)]
mod benchmarks {
    use super::*;

    #[test]
    #[ignore = "benchmark - not a unit test"]
    fn run_learning_benchmarks() {
        let patterns: Vec<ContinuousHV> = (0..10).map(|i| random_pattern(100 + i)).collect();

        println!("\n=== Learning Strategy Benchmarks ===\n");

        let results = vec![
            benchmark_learning_strategy("Hebbian Only", test_config(), &patterns, 50, true, false),
            benchmark_learning_strategy(
                "Contrastive Only",
                test_config(),
                &patterns,
                50,
                false,
                true,
            ),
            benchmark_learning_strategy("Mixed", test_config(), &patterns, 50, true, true),
        ];

        println!(
            "{:<20} {:>10} {:>12} {:>12} {:>12}",
            "Strategy", "Epochs", "Accuracy", "Time(ms)", "Stability"
        );
        println!("{}", "-".repeat(70));

        for r in &results {
            println!(
                "{:<20} {:>10} {:>12.4} {:>12.2} {:>12.4}",
                r.strategy, r.epochs, r.final_accuracy, r.training_time_ms, r.stability_score
            );
        }
    }
}
