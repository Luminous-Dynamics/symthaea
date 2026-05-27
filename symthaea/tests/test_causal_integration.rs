// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for Causal Discovery with Symthaea HDC
//!
//! Tests causal discovery integration with:
//! - HDC hypervector analysis
//! - NixOS configuration causal analysis
//! - Consciousness hierarchy causal flows
//! - Grid search threshold tuning

use symthaea::intelligence::{
    CausalAttention, CausalConsciousness, CausalDirection, CausalDiscoveryEngine, CausalLTCBridge,
    NixOSCausalAnalyzer, RandomThresholdSearch, ThresholdTuner,
};

// ============================================================================
// HDC CAUSAL INTEGRATION TESTS
// ============================================================================

#[test]
fn test_causal_discovery_with_hdc_signals() {
    // Simulate HDC signals with causal structure
    // Signal A causes signal B through a nonlinear transformation
    let signal_a: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
    let signal_b: Vec<f64> = signal_a.iter().map(|&a| (a * 2.0).tanh() + 0.05).collect();

    let mut engine = CausalDiscoveryEngine::with_ensemble_size(42, 5);

    let (direction, confidence) = engine.predict_with_confidence(&signal_a, &signal_b);

    println!("HDC signal causal analysis:");
    println!("  Direction: {:?}", direction);
    println!("  Confidence: {:.2}%", confidence * 100.0);

    // Confidence should be reasonable
    assert!((0.0..=1.0).contains(&confidence));
}

#[test]
fn test_causal_attention_multiple_layers() {
    let mut attention = CausalAttention::new(42);

    // Simulate 5 HDC layers with causal structure
    // Layer 0 -> Layer 1 -> Layer 2 -> Layer 3 -> Layer 4
    let layer0: Vec<f64> = (0..150).map(|i| i as f64 * 0.1).collect();
    let layer1: Vec<f64> = layer0.iter().map(|&x| 0.8 * x + 0.2).collect();
    let layer2: Vec<f64> = layer1.iter().map(|&x| 0.7 * x + 0.3).collect();
    let layer3: Vec<f64> = layer2.iter().map(|&x| 0.6 * x + 0.4).collect();
    let layer4: Vec<f64> = layer3.iter().map(|&x| 0.5 * x + 0.5).collect();

    let layers = vec![layer0, layer1, layer2, layer3, layer4];
    let weights = attention.compute_attention(&layers);

    println!("Causal attention matrix (5 HDC layers):");
    for (i, row) in weights.iter().enumerate() {
        println!(
            "  Layer {} -> {:?}",
            i,
            row.iter().map(|v| format!("{:.2}", v)).collect::<Vec<_>>()
        );
    }

    // Check attention matrix is valid
    assert_eq!(weights.len(), 5);
    for row in &weights {
        assert_eq!(row.len(), 5);
        // Softmax should sum to ~1
        let sum: f64 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Attention should sum to 1, got {}",
            sum
        );
    }

    // Check that we can identify causes
    let causes_of_4 = attention.get_causes(4);
    println!("Causes of layer 4: {:?}", causes_of_4);

    let effects_of_0 = attention.get_effects(0);
    println!("Effects of layer 0: {:?}", effects_of_0);
}

#[test]
fn test_causal_ltc_bridge_hierarchy() {
    let mut bridge = CausalLTCBridge::new(7, 42);

    // Simulate 3 levels of the LTC hierarchy
    let level0: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let level1: Vec<f64> = level0.iter().map(|&x| 2.0 * x).collect();
    let level2: Vec<f64> = level1.iter().map(|&x| 0.5 * x).collect();

    // Discover cross-level causality
    let dir_0_1 = bridge.discover_cross_level(0, 1, &level0, &level1);
    let dir_1_2 = bridge.discover_cross_level(1, 2, &level1, &level2);
    let dir_0_2 = bridge.discover_cross_level(0, 2, &level0, &level2);

    println!("Cross-level causality:");
    println!("  Level 0 -> 1: {:?}", dir_0_1);
    println!("  Level 1 -> 2: {:?}", dir_1_2);
    println!("  Level 0 -> 2: {:?}", dir_0_2);

    // Get causal flow
    let flow = bridge.get_causal_flow();
    println!("Causal flow in hierarchy: {:?}", flow);
    assert!(!flow.is_empty(), "Should have detected some causal flow");

    // Compute causal Phi
    let phi = bridge.compute_causal_phi();
    println!("Causal Phi: {:.4}", phi);
    assert!(phi >= 0.0, "Causal Phi should be non-negative");
}

// ============================================================================
// NIXOS CAUSAL ANALYSIS TESTS
// ============================================================================

#[test]
fn test_nixos_causal_analyzer_full_workflow() {
    let mut analyzer = NixOSCausalAnalyzer::new(42);

    // Simulate NixOS configuration observations over time
    // Pattern: enabling nvidia driver affects xserver, which affects display manager
    for i in 0..100 {
        let t = i as f64;

        // NVIDIA enabled (simulated as stepping function)
        let nvidia_enabled = if i > 30 { 1.0 } else { 0.0 };

        // XServer follows NVIDIA with delay
        let xserver_enabled = if i > 35 { 1.0 } else { 0.0 };

        // Display manager follows XServer
        let dm_enabled = if i > 40 { 1.0 } else { 0.0 };

        // Kernel parameter affected by NVIDIA
        let kernel_param = nvidia_enabled * 0.8 + 0.1 * (t * 0.1).sin();

        analyzer.observe("hardware.nvidia.enable", nvidia_enabled);
        analyzer.observe("services.xserver.enable", xserver_enabled);
        analyzer.observe("services.displayManager.enable", dm_enabled);
        analyzer.observe("boot.kernelParams", kernel_param);
    }

    // Discover causal structure
    let edges = analyzer.discover_structure();
    println!("Discovered causal edges: {}", edges.len());
    for edge in &edges {
        println!(
            "  {} -> {} (confidence: {:.2})",
            edge.from, edge.to, edge.confidence
        );
    }

    // Analyze root causes of display manager issues
    let analysis = analyzer.analyze_root_causes("services.displayManager.enable");
    println!("\nRoot cause analysis for display manager:");
    println!("  Symptom: {}", analysis.symptom);
    for cause in &analysis.root_causes {
        println!(
            "  Root cause: {} (confidence: {:.2})",
            cause.variable, cause.confidence
        );
        println!("    Explanation: {}", cause.explanation);
    }

    // Predict side effects of changing NVIDIA
    let effects = analyzer.predict_side_effects("hardware.nvidia.enable");
    println!("\nSide effects of changing NVIDIA:");
    for effect in &effects {
        println!(
            "  {} will {} (confidence: {:.2})",
            effect.affected_variable, effect.direction, effect.confidence
        );
    }

    // Get fix recommendations
    let recommendations = analyzer.recommend_fixes("services.displayManager.enable");
    println!("\nFix recommendations:");
    for rec in &recommendations {
        println!("  - {}", rec);
    }
}

#[test]
fn test_nixos_encoder_functions() {
    // Test encoding utilities
    let bool_true = NixOSCausalAnalyzer::encode_bool(true);
    let bool_false = NixOSCausalAnalyzer::encode_bool(false);

    assert_eq!(bool_true, 1.0);
    assert_eq!(bool_false, 0.0);

    // Test option value encoding
    let val1 = NixOSCausalAnalyzer::encode_option_value("nvidia");
    let val2 = NixOSCausalAnalyzer::encode_option_value("nouveau");
    let val3 = NixOSCausalAnalyzer::encode_option_value("nvidia");

    // Same value should encode to same hash
    assert_eq!(val1, val3);
    // Different values should encode differently
    assert_ne!(val1, val2);
    // Values should be in [0, 1]
    assert!((0.0..=1.0).contains(&val1));
    assert!((0.0..=1.0).contains(&val2));
}

// ============================================================================
// THRESHOLD TUNING INTEGRATION TESTS
// ============================================================================

#[test]
fn test_threshold_tuner_with_hdc_data() {
    // Generate HDC-like causal data
    let mut training_data = Vec::new();

    // Linear causal relationships (common in HDC)
    for i in 0..20 {
        let offset = (i * 5) as f64;
        let x: Vec<f64> = (0..100).map(|j| (j as f64 + offset) * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 1.5 * xi + 0.1).collect();
        training_data.push((x, y, CausalDirection::Forward));
    }

    // Nonlinear causal relationships
    for i in 0..10 {
        let offset = (i * 3) as f64;
        let x: Vec<f64> = (0..100).map(|j| (j as f64 + offset) * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|&xi| (xi * 0.5).tanh()).collect();
        training_data.push((x, y, CausalDirection::Forward));
    }

    // Tune thresholds using a coarse grid for speed
    let mut tuner = ThresholdTuner::new(42)
        .with_noise_range(vec![0.75, 0.85, 0.95])
        .with_nonlin_range(vec![0.3, 0.5])
        .with_corr_range(vec![0.75, 0.85, 0.95])
        .with_kurtosis_range(vec![2.5, 3.5]);

    let result = tuner.grid_search(&training_data);

    println!("Threshold tuning with HDC data:");
    println!("  Best noise threshold: {:.2}", result.noise_threshold);
    println!("  Best nonlin threshold: {:.2}", result.nonlin_threshold);
    println!("  Best corr threshold: {:.2}", result.corr_threshold);
    println!(
        "  Best kurtosis threshold: {:.2}",
        result.kurtosis_threshold
    );
    println!(
        "  Accuracy: {:.1}% ({}/{})",
        result.accuracy * 100.0,
        result.n_correct,
        result.n_total
    );

    assert!(result.n_total == 30);
}

#[test]
fn test_random_search_efficiency() {
    // Generate test data
    let mut data = Vec::new();
    for i in 0..15 {
        let x: Vec<f64> = (0..80).map(|j| (j + i * 10) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + (xi * 0.02).sin()).collect();
        data.push((x, y, CausalDirection::Forward));
    }

    // Random search should be fast
    let start = std::time::Instant::now();
    let mut search = RandomThresholdSearch::new(42, 30);
    let result = search.search(&data);
    let elapsed = start.elapsed();

    println!("Random search completed in {:?}", elapsed);
    println!("  Best accuracy: {:.1}%", result.accuracy * 100.0);

    // Should complete in reasonable time
    assert!(elapsed.as_secs() < 30, "Random search should be fast");
}

// ============================================================================
// CAUSAL CONSCIOUSNESS INTEGRATION TESTS
// ============================================================================

#[test]
fn test_full_causal_consciousness_workflow() {
    let mut cc = CausalConsciousness::new(42, 7);

    // Simulate consciousness signals with causal structure
    let sensory: Vec<f64> = (0..150).map(|i| (i as f64 * 0.1).sin()).collect();
    let cognitive: Vec<f64> = sensory.iter().map(|&s| (s * 1.5).tanh()).collect();

    // Full analysis
    let result = cc.analyze(&sensory, &cognitive);

    println!("Full causal consciousness analysis:");
    println!("  Direction: {:?}", result.direction);
    println!("  Confidence: {:.2}%", result.confidence * 100.0);
    println!("  HSIC dependence: {:.4}", result.hsic_dependence);
    println!("  Is independent: {}", result.is_independent);
    println!("  P-value: {:.4}", result.p_value);
    println!(
        "  Residual independence: {:.4}",
        result.residual_independence
    );

    // Provide feedback to improve
    cc.learn(&sensory, &cognitive, CausalDirection::Forward);

    // Check stats
    let stats = cc.get_stats();
    println!("\nSystem stats after learning:");
    println!("  N predictions: {}", stats.n_predictions);
    println!("  Accuracy: {:.1}%", stats.accuracy * 100.0);
    println!(
        "  Learned noise threshold: {:.2}",
        stats.learned_noise_threshold
    );
    println!(
        "  Learned corr threshold: {:.2}",
        stats.learned_corr_threshold
    );

    // Validate confidence bounds
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    assert!(result.residual_independence >= 0.0);
}

#[test]
fn test_causal_discovery_parallel_mode() {
    // Create engine with parallel mode
    let mut engine = CausalDiscoveryEngine::with_ensemble_size(42, 10);

    // Generate test data
    let x: Vec<f64> = (0..200).map(|i| i as f64 * 0.1).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 0.1).collect();

    // Time the prediction (parallel should be fast)
    let start = std::time::Instant::now();
    let (direction, confidence) = engine.predict_with_confidence(&x, &y);
    let elapsed = start.elapsed();

    println!("Parallel prediction completed in {:?}", elapsed);
    println!("  Direction: {:?}", direction);
    println!("  Confidence: {:.2}%", confidence * 100.0);

    // Should complete quickly
    assert!(
        elapsed.as_millis() < 5000,
        "Parallel prediction should be fast"
    );
}