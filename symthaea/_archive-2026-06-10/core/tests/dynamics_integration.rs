// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Dynamics Module Integration Tests
//!
//! Integration tests for `symthaea::dynamics` covering the core CfC (Closed-form
//! Continuous-time) neural network types, hierarchical multi-scale processing,
//! crystallized concept representations, activation functions, and online learning
//! statistics.
//!
//! These types are always compiled (no feature gate). Because CfC dynamics are
//! stochastic, tests assert finiteness and dimensional correctness rather than
//! exact numeric values.

use ndarray::Array1;
use symthaea::dynamics::*;

// ============================================================================
// Helper functions
// ============================================================================

/// Build a small CfC cell config for fast tests.
fn small_cell_config() -> CfCConfig {
    CfCConfig {
        input_dim: 8,
        hidden_dim: 16,
        use_backbone: false,
        backbone_layers: 0,
        backbone_dim: 0,
        activation: ActivationType::SiLU,
        tau_range: (0.1, 10.0),
        dropout: 0.0,
        gradient_clip: 1.0,
        online_learning: None,
    }
}

/// Build a small CfC network config for fast tests.
fn small_network_config() -> CfCNetworkConfig {
    CfCNetworkConfig {
        input_dim: 8,
        hidden_dim: 16,
        num_layers: 2,
        output_dim: 4,
        cell_config: small_cell_config(),
        residual: true,
        bidirectional: false,
        enable_online_learning: false,
        online_learning_config: OnlineLearningConfig::default(),
    }
}

/// Build a small input vector of dimension 8.
fn input_8() -> Array1<f32> {
    Array1::from_vec(vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])
}

/// Assert every element of an Array1 is finite.
fn assert_all_finite(arr: &Array1<f32>, label: &str) {
    for (i, &v) in arr.iter().enumerate() {
        assert!(v.is_finite(), "{label}[{i}] = {v} is not finite");
    }
}

// ============================================================================
// 1. CfCCell construction and forward pass
// ============================================================================

#[test]
fn cfc_cell_construction_and_forward_dimensions() {
    let config = small_cell_config();
    let mut cell = CfCCell::new(config);

    // Initial state should be zero
    assert_eq!(cell.state().len(), 16);
    assert!(cell.state().iter().all(|&x| x == 0.0));

    // Forward pass with a small input
    let input = input_8();
    let output = cell.forward(&input, 0.05);

    // Output dimension must match hidden_dim
    assert_eq!(
        output.len(),
        16,
        "CfCCell forward output should have hidden_dim elements"
    );
    assert_all_finite(&output, "CfCCell forward output");
}

// ============================================================================
// 2. CfCCell forward_with_cache and backward_from_cache round-trip
// ============================================================================

#[test]
fn cfc_cell_forward_with_cache_and_backward() {
    let config = small_cell_config();
    let mut cell = CfCCell::new(config);

    let input = input_8();
    let dt = 0.1;

    // Forward with cache
    let (output, cache) = cell.forward_with_cache(&input, dt);
    assert_eq!(output.len(), 16);
    assert_all_finite(&output, "cached forward output");

    // Backward from cache should produce finite gradients
    let upstream_grad = Array1::ones(16) * 0.01;
    let grads = cell.backward_from_cache(&cache, &upstream_grad, dt);

    // Check gradient dimensions
    assert_eq!(grads.dw_in.dim(), (16, 8));
    assert_eq!(grads.dw_h.dim(), (16, 16));
    assert_eq!(grads.db_h.len(), 16);
    assert_eq!(grads.dtau.len(), 16);

    // All gradients must be finite
    assert!(
        grads.dw_in.iter().all(|v| v.is_finite()),
        "dw_in contains non-finite"
    );
    assert!(
        grads.dw_h.iter().all(|v| v.is_finite()),
        "dw_h contains non-finite"
    );
    assert_all_finite(&grads.db_h, "db_h gradient");
    assert_all_finite(&grads.dtau, "dtau gradient");
}

// ============================================================================
// 3. CfCNetwork construction and forward pass
// ============================================================================

#[test]
fn cfc_network_forward_output_dimensions_and_finiteness() {
    let config = small_network_config();
    let mut net = CfCNetwork::new(config);

    let input = input_8();
    let output = net.forward(&input, 0.1);

    assert_eq!(output.len(), 4, "Network output should match output_dim=4");
    assert_all_finite(&output, "CfCNetwork forward output");
}

// ============================================================================
// 4. CfCNetwork sequence processing
// ============================================================================

#[test]
fn cfc_network_forward_sequence() {
    let config = small_network_config();
    let mut net = CfCNetwork::new(config);

    let seq_len = 10;
    let inputs: Vec<Array1<f32>> = (0..seq_len)
        .map(|i| {
            let scale = (i as f32) * 0.1;
            Array1::from_vec(vec![scale; 8])
        })
        .collect();
    let dts = vec![0.05f32; seq_len];

    let outputs = net.forward_sequence(&inputs, &dts);

    assert_eq!(
        outputs.len(),
        seq_len,
        "Should produce one output per timestep"
    );
    for (t, out) in outputs.iter().enumerate() {
        assert_eq!(
            out.len(),
            4,
            "Output at step {t} should have output_dim=4 elements"
        );
        assert_all_finite(out, &format!("sequence output[{t}]"));
    }
}

// ============================================================================
// 5. CfCNetwork state save/restore round-trip
// ============================================================================

#[test]
fn cfc_network_state_save_restore_round_trip() {
    let config = small_network_config();
    let mut net = CfCNetwork::new(config);

    // Drive the network with a few forward passes to build up state
    for _ in 0..5 {
        net.forward(&input_8(), 0.1);
    }

    // Snapshot states
    let saved_states = net.state();
    assert_eq!(
        saved_states.len(),
        2,
        "Should have one state vector per layer"
    );

    // Run more forward passes to change state
    for _ in 0..5 {
        net.forward(&Array1::ones(8), 0.1);
    }
    let changed_states = net.state();

    // States should have changed
    let any_changed = saved_states
        .iter()
        .zip(changed_states.iter())
        .any(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-12));
    assert!(
        any_changed,
        "States should have changed after additional forward passes"
    );

    // Restore original states
    net.set_state(saved_states.clone());
    let restored_states = net.state();

    // Restored states should match saved
    for (layer_idx, (saved, restored)) in
        saved_states.iter().zip(restored_states.iter()).enumerate()
    {
        for (i, (s, r)) in saved.iter().zip(restored.iter()).enumerate() {
            assert!(
                (s - r).abs() < f32::EPSILON,
                "Layer {layer_idx} state element {i} mismatch after restore: {s} vs {r}"
            );
        }
    }
}

// ============================================================================
// 6. CfCNetwork parameter counting
// ============================================================================

#[test]
fn cfc_network_parameter_count_is_positive_and_consistent() {
    let config = small_network_config();
    let net = CfCNetwork::new(config);

    let params = net.num_parameters();
    assert!(params > 0, "Network should have a positive parameter count");

    // Manually compute expected count (backbone disabled):
    // Layer 0: input_dim(8) * hidden_dim(16) + hidden(16)*hidden(16) + bias(16) + tau(16)
    //        = 128 + 256 + 16 + 16 = 416
    // Layer 1: hidden_dim(16) * hidden_dim(16) + hidden(16)*hidden(16) + bias(16) + tau(16)
    //        = 256 + 256 + 16 + 16 = 544
    // Output: hidden_dim(16) * output_dim(4) + bias(4) = 64 + 4 = 68
    // Total = 416 + 544 + 68 = 1028
    assert_eq!(
        params, 1028,
        "Parameter count should match analytical computation"
    );
}

// ============================================================================
// 7. HierarchicalCfC multi-scale forward (4 time constants)
// ============================================================================

#[test]
fn hierarchical_cfc_multi_scale_forward() {
    let config = HierarchicalCfCConfig {
        input_dim: 8,
        output_dim: 4,
        time_constants: DEFAULT_TIME_CONSTANTS.to_vec(),
        hidden_dims: vec![16, 16, 12, 8],
        top_down_feedback: true,
        feedback_strength: 0.3,
        bottom_up_integration: true,
        integration_strength: 0.5,
        multi_scale_prediction: true,
        lr_scales: vec![0.5, 1.0, 1.0, 0.5],
    };
    let mut hcfc = HierarchicalCfC::new(config);

    assert_eq!(
        hcfc.num_layers(),
        4,
        "Should have 4 layers matching DEFAULT_TIME_CONSTANTS"
    );

    let input = input_8();
    let output = hcfc.forward_hierarchical(&input, 0.01);

    // Combined output should match output_dim
    assert_eq!(output.combined.len(), 4);
    assert_all_finite(&output.combined, "hierarchical combined output");

    // Should produce one scale output per layer
    assert_eq!(output.scale_outputs.len(), 4);
    for (i, scale_out) in output.scale_outputs.iter().enumerate() {
        assert_eq!(
            scale_out.len(),
            4,
            "Scale {i} output should match output_dim"
        );
        assert_all_finite(scale_out, &format!("scale_output[{i}]"));
    }

    // Layer states match hidden dims
    assert_eq!(output.layer_states.len(), 4);
    let expected_hidden_dims = [16, 16, 12, 8];
    for (i, state) in output.layer_states.iter().enumerate() {
        assert_eq!(
            state.len(),
            expected_hidden_dims[i],
            "Layer {i} state dim mismatch"
        );
    }

    // Effective taus returned for all layers
    assert_eq!(output.effective_taus.len(), 4);
}

// ============================================================================
// 8. HierarchicalCfC training step produces finite loss
// ============================================================================

#[test]
fn hierarchical_cfc_train_step_produces_finite_loss() {
    let config = HierarchicalCfCConfig {
        input_dim: 8,
        output_dim: 4,
        time_constants: vec![0.01, 0.1, 1.0, 10.0],
        hidden_dims: vec![16, 16, 12, 8],
        ..Default::default()
    };
    let mut hcfc = HierarchicalCfC::new(config);

    let input = input_8();
    let target = Array1::from_vec(vec![0.5, -0.3, 0.1, 0.8]);

    let loss = hcfc.train_step(&input, &target, 0.1, 0.01).unwrap();
    assert!(
        loss.is_finite(),
        "Training loss should be finite, got {loss}"
    );
    assert!(
        loss >= 0.0,
        "MSE-based loss should be non-negative, got {loss}"
    );
}

// ============================================================================
// 9. CrystalizedConcept creation, similarity, and associations
// ============================================================================

#[test]
fn crystalized_concept_creation_similarity_and_associations() {
    // Create two concepts with known embeddings
    let embedding_a = vec![1.0, 0.0, 0.0, 0.0];
    let embedding_b = vec![0.0, 1.0, 0.0, 0.0]; // Orthogonal to A
    let embedding_c = vec![1.0, 0.0, 0.0, 0.0]; // Identical to A

    let concept_a = CrystalizedConcept::new(1, "Alpha", embedding_a);
    let concept_b = CrystalizedConcept::new(2, "Beta", embedding_b);
    let concept_c = CrystalizedConcept::new(3, "AlphaClone", embedding_c);

    // Basic fields
    assert_eq!(concept_a.id, 1);
    assert_eq!(concept_a.name, "Alpha");
    assert_eq!(concept_a.dimension(), 4);
    assert_eq!(concept_a.confidence, 0.5);
    assert_eq!(concept_a.activation_count, 0);

    // Orthogonal vectors should have zero similarity
    let sim_ab = concept_a.similarity(&concept_b);
    assert!(
        sim_ab.abs() < 1e-6,
        "Orthogonal concepts should have ~0 similarity, got {sim_ab}"
    );

    // Identical vectors should have similarity 1.0
    let sim_ac = concept_a.similarity(&concept_c);
    assert!(
        (sim_ac - 1.0).abs() < 1e-6,
        "Identical embeddings should have similarity ~1.0, got {sim_ac}"
    );

    // Self-similarity should be 1.0
    let sim_aa = concept_a.similarity(&concept_a);
    assert!(
        (sim_aa - 1.0).abs() < 1e-6,
        "Self-similarity should be 1.0, got {sim_aa}"
    );

    // Associations
    let mut concept_a = concept_a;
    assert_eq!(concept_a.association_strength(2), 0.0, "No association yet");

    concept_a.add_association(2, 0.85);
    assert!(
        (concept_a.association_strength(2) - 0.85).abs() < 1e-6,
        "Association strength should be 0.85"
    );

    // Association strength is clamped to [-1, 1]
    concept_a.add_association(3, 5.0);
    assert!(
        (concept_a.association_strength(3) - 1.0).abs() < 1e-6,
        "Association should be clamped to 1.0"
    );
}

// ============================================================================
// 10. CrystalizedConcept with_details and activate
// ============================================================================

#[test]
fn crystalized_concept_with_details_and_activate() {
    let embedding = vec![0.5, 0.5, 0.5, 0.5];
    let concept = CrystalizedConcept::with_details(
        42,
        "DetailedConcept",
        "A concept created with full details",
        embedding,
        0.95,
    );

    assert_eq!(concept.id, 42);
    assert_eq!(concept.name, "DetailedConcept");
    assert_eq!(
        concept.description,
        Some("A concept created with full details".to_string())
    );
    assert!((concept.confidence - 0.95).abs() < 1e-6);

    // Activate the concept
    let mut concept = concept;
    concept.activate(1000);
    assert_eq!(concept.activation_count, 1);
    assert_eq!(concept.last_activated, 1000);

    concept.activate(2000);
    assert_eq!(concept.activation_count, 2);
    assert_eq!(concept.last_activated, 2000);
}

// ============================================================================
// 11. ActivationType::apply for each variant
// ============================================================================

#[test]
fn activation_type_apply_known_values() {
    // ReLU(-1) = 0, ReLU(1) = 1
    assert_eq!(
        ActivationType::ReLU.apply(-1.0),
        0.0,
        "ReLU(-1) should be 0"
    );
    assert_eq!(ActivationType::ReLU.apply(1.0), 1.0, "ReLU(1) should be 1");
    assert_eq!(ActivationType::ReLU.apply(0.0), 0.0, "ReLU(0) should be 0");

    // Sigmoid(0) = 0.5
    let sig_zero = ActivationType::Sigmoid.apply(0.0);
    assert!(
        (sig_zero - 0.5).abs() < 1e-6,
        "Sigmoid(0) should be 0.5, got {sig_zero}"
    );

    // Sigmoid is bounded in (0, 1)
    let sig_large = ActivationType::Sigmoid.apply(100.0);
    assert!(
        sig_large > 0.99 && sig_large <= 1.0,
        "Sigmoid(100) should be ~1.0"
    );
    let sig_neg = ActivationType::Sigmoid.apply(-100.0);
    assert!(
        (0.0..0.01).contains(&sig_neg),
        "Sigmoid(-100) should be ~0.0"
    );

    // Tanh(0) = 0
    let tanh_zero = ActivationType::Tanh.apply(0.0);
    assert!(
        tanh_zero.abs() < 1e-6,
        "Tanh(0) should be 0, got {tanh_zero}"
    );

    // Tanh is bounded in (-1, 1)
    let tanh_large = ActivationType::Tanh.apply(100.0);
    assert!((tanh_large - 1.0).abs() < 1e-4, "Tanh(100) should be ~1.0");

    // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    let silu_zero = ActivationType::SiLU.apply(0.0);
    assert!(
        silu_zero.abs() < 1e-6,
        "SiLU(0) should be 0, got {silu_zero}"
    );

    // GELU(0) = 0 (symmetric around 0)
    let gelu_zero = ActivationType::GELU.apply(0.0);
    assert!(
        gelu_zero.abs() < 1e-6,
        "GELU(0) should be ~0, got {gelu_zero}"
    );

    // All activations should produce finite values for normal inputs
    let test_values = [-5.0, -1.0, 0.0, 1.0, 5.0];
    let variants = [
        ActivationType::SiLU,
        ActivationType::GELU,
        ActivationType::ReLU,
        ActivationType::Tanh,
        ActivationType::Sigmoid,
    ];
    for variant in &variants {
        for &x in &test_values {
            let y = variant.apply(x);
            assert!(y.is_finite(), "{variant:?}.apply({x}) = {y} is not finite");
        }
    }
}

// ============================================================================
// 12. ActivationType::apply_fast agrees approximately with apply
// ============================================================================

#[test]
fn activation_type_apply_fast_approximate_agreement() {
    let test_values = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0];
    let variants = [
        ActivationType::SiLU,
        ActivationType::GELU,
        ActivationType::ReLU,
        ActivationType::Tanh,
        ActivationType::Sigmoid,
    ];

    for variant in &variants {
        for &x in &test_values {
            let exact = variant.apply(x);
            let fast = variant.apply_fast(x);
            let diff = (exact - fast).abs();

            // ReLU and Tanh should be exact; GELU uses same formula for both paths.
            // Sigmoid uses fast_sigmoid rational approximation -- diverges more at
            // larger |x| (error up to ~0.08 at |x|=3).
            // SiLU = x * sigmoid(x), so the approximation error scales with |x| --
            // at |x|=3 the difference can reach ~0.23.
            let tolerance = match variant {
                ActivationType::ReLU | ActivationType::Tanh | ActivationType::GELU => 1e-6,
                ActivationType::Sigmoid => 0.1,
                ActivationType::SiLU => 0.25,
            };
            assert!(
                diff < tolerance,
                "{variant:?}.apply_fast({x}) = {fast} differs from apply({x}) = {exact} by {diff} > {tolerance}"
            );
        }
    }
}

// ============================================================================
// 13. Online learning stats tracking
// ============================================================================

#[test]
fn online_learning_stats_tracking() {
    let online_config = OnlineLearningConfig {
        learning_rate: 0.01,
        error_threshold: 0.0, // Always adapt
        ema_alpha: 0.1,
        max_weight_delta: 0.1,
        adapt_tau: false,
        tau_lr_multiplier: 0.01,
    };

    let mut net_config = small_network_config();
    net_config.enable_online_learning = true;
    net_config.online_learning_config = online_config.clone();
    net_config.cell_config.online_learning = Some(online_config);

    let mut net = CfCNetwork::new(net_config);
    assert!(
        net.online_learning_enabled(),
        "Online learning should be enabled"
    );

    let input = input_8();
    let target = Array1::from_vec(vec![1.0, -1.0, 0.5, -0.5]);

    // Initial stats should be zero
    let stats = net.online_stats();
    assert_eq!(stats.total_adaptation_calls, 0);
    assert_eq!(stats.adaptations_applied, 0);
    assert_eq!(stats.adaptations_skipped, 0);

    // Adapt with a large error to ensure it triggers
    let adapted = net.adapt_online(10.0, &input, &target, 0.1);
    assert!(adapted, "Should have adapted with large error");

    let stats = net.online_stats();
    assert_eq!(stats.total_adaptation_calls, 1);
    assert_eq!(stats.adaptations_applied, 1);
    assert!(
        stats.cumulative_weight_change > 0.0,
        "Should have accumulated weight change"
    );
    assert!(stats.ema_error > 0.0, "EMA error should have been updated");

    // Reset stats
    net.reset_online_stats();
    let stats = net.online_stats();
    assert_eq!(stats.total_adaptation_calls, 0);
    assert_eq!(stats.adaptations_applied, 0);
    assert_eq!(stats.cumulative_weight_change, 0.0);
}

// ============================================================================
// 14. Network reset clears state
// ============================================================================

#[test]
fn cfc_network_reset_clears_state() {
    let config = small_network_config();
    let mut net = CfCNetwork::new(config);

    // Drive the network
    for _ in 0..10 {
        net.forward(&input_8(), 0.1);
    }

    // Verify state is non-zero
    let states_before = net.state();
    let has_nonzero = states_before
        .iter()
        .any(|s| s.iter().any(|&v| v.abs() > 1e-12));
    assert!(
        has_nonzero,
        "After forward passes, at least some state elements should be non-zero"
    );

    // Reset
    net.reset();

    // All states should be zero after reset
    let states_after = net.state();
    for (layer_idx, state) in states_after.iter().enumerate() {
        for (i, &v) in state.iter().enumerate() {
            assert_eq!(
                v, 0.0,
                "Layer {layer_idx} state[{i}] should be 0.0 after reset, got {v}"
            );
        }
    }
}

// ============================================================================
// 15. HierarchicalCfC num_layers and num_parameters
// ============================================================================

#[test]
fn hierarchical_cfc_num_layers_and_parameters() {
    let config = HierarchicalCfCConfig {
        input_dim: 8,
        output_dim: 4,
        time_constants: vec![0.01, 0.1, 1.0, 10.0],
        hidden_dims: vec![16, 16, 12, 8],
        ..Default::default()
    };
    let hcfc = HierarchicalCfC::new(config);

    assert_eq!(hcfc.num_layers(), 4);

    let params = hcfc.num_parameters();
    assert!(
        params > 0,
        "Hierarchical network should have positive parameter count"
    );

    // Should be substantially more than a single flat layer
    // (includes CfC layer params + inter-layer projections + output projections + combo weights)
    assert!(
        params > 500,
        "Expected substantial parameter count for 4-layer hierarchical network, got {params}"
    );
}

// ============================================================================
// 16. HierarchicalCfC reset zeroes all state
// ============================================================================

#[test]
fn hierarchical_cfc_reset_zeroes_state() {
    let config = HierarchicalCfCConfig {
        input_dim: 8,
        output_dim: 4,
        time_constants: vec![0.01, 0.1],
        hidden_dims: vec![16, 16],
        ..Default::default()
    };
    let mut hcfc = HierarchicalCfC::new(config);

    // Drive it
    for _ in 0..10 {
        hcfc.forward_hierarchical(&input_8(), 0.1);
    }

    // Reset
    hcfc.reset();

    // All layer states should be zero
    for level_states in hcfc.all_states() {
        for state in level_states {
            assert!(
                state.iter().all(|&x| x == 0.0),
                "All states should be zero after reset"
            );
        }
    }
}

// ============================================================================
// 17. CfCCell reset clears state and step counter
// ============================================================================

#[test]
fn cfc_cell_reset_clears_state() {
    let config = small_cell_config();
    let mut cell = CfCCell::new(config);

    // Run some forward passes
    for _ in 0..5 {
        cell.forward(&input_8(), 0.05);
    }

    // State should be non-zero
    assert!(
        cell.state().iter().any(|&v| v.abs() > 1e-12),
        "State should be non-zero after forward passes"
    );

    cell.reset();

    // State should be all zeros after reset
    assert!(
        cell.state().iter().all(|&v| v == 0.0),
        "State should be all zeros after reset"
    );
}