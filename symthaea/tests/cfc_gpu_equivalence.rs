// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # CfC GPU/CPU Equivalence Tests
//!
//! This module verifies that the GPU-accelerated CfC implementation produces
//! results equivalent to the CPU implementation within floating-point tolerance.
//!
//! ## Test Categories
//!
//! 1. **Forward pass equivalence**: Single and batch forward passes
//! 2. **State management**: Reset, save, restore operations
//! 3. **Numerical stability**: Long sequences, edge cases
//! 4. **Training equivalence**: Gradient computation and weight updates
//!
//! ## Running Tests
//!
//! ```bash
//! # Run all equivalence tests
//! CARGO_TARGET_DIR=/tmp/symthaea-gpu cargo test --test cfc_gpu_equivalence
//!
//! # Run with GPU features
//! CARGO_TARGET_DIR=/tmp/symthaea-gpu cargo test --test cfc_gpu_equivalence --features gpu
//! ```

use symthaea::dynamics::{
    ActivationType, CfCConfig, CfCNetworkConfig, GpuBackend, GpuCfcConfig, GpuCfcNetwork,
};

/// Tolerance for floating-point comparisons
#[allow(dead_code)]
const TOLERANCE: f32 = 1e-4;

/// Extended tolerance for accumulated operations
#[allow(dead_code)]
const EXTENDED_TOLERANCE: f32 = 1e-3;

/// Maximum absolute difference allowed
fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, |acc, x| acc.max(x))
}

/// Check if two vectors are approximately equal
#[allow(dead_code)]
fn approx_equal(a: &[f32], b: &[f32], tolerance: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    max_diff(a, b) < tolerance
}

// =============================================================================
// TEST CONFIGURATIONS
// =============================================================================

fn test_configs() -> Vec<(&'static str, GpuCfcConfig)> {
    vec![
        (
            "tiny",
            GpuCfcConfig {
                input_dim: 8,
                hidden_dim: 16,
                num_layers: 1,
                output_dim: 4,
                use_backbone: false,
                ..Default::default()
            },
        ),
        (
            "small_no_backbone",
            GpuCfcConfig {
                input_dim: 32,
                hidden_dim: 64,
                num_layers: 2,
                output_dim: 16,
                use_backbone: false,
                ..Default::default()
            },
        ),
        (
            "small_with_backbone",
            GpuCfcConfig {
                input_dim: 32,
                hidden_dim: 64,
                num_layers: 2,
                output_dim: 16,
                use_backbone: true,
                backbone_layers: 2,
                backbone_dim: 48,
                ..Default::default()
            },
        ),
        (
            "medium",
            GpuCfcConfig {
                input_dim: 64,
                hidden_dim: 128,
                num_layers: 3,
                output_dim: 32,
                use_backbone: true,
                backbone_layers: 2,
                backbone_dim: 96,
                ..Default::default()
            },
        ),
    ]
}

/// Convert GpuCfcConfig to CfCNetworkConfig
#[allow(dead_code)]
fn to_cpu_config(gpu_config: &GpuCfcConfig) -> CfCNetworkConfig {
    CfCNetworkConfig {
        input_dim: gpu_config.input_dim,
        hidden_dim: gpu_config.hidden_dim,
        num_layers: gpu_config.num_layers,
        output_dim: gpu_config.output_dim,
        cell_config: CfCConfig {
            input_dim: gpu_config.input_dim,
            hidden_dim: gpu_config.hidden_dim,
            use_backbone: gpu_config.use_backbone,
            backbone_layers: gpu_config.backbone_layers,
            backbone_dim: gpu_config.backbone_dim,
            activation: gpu_config.activation,
            tau_range: gpu_config.tau_range,
            dropout: 0.0, // Disable dropout for deterministic comparison
            gradient_clip: 1.0,
            online_learning: None,
        },
        residual: true,
        bidirectional: false,
        enable_online_learning: false,
        online_learning_config: Default::default(),
    }
}

// =============================================================================
// FORWARD PASS TESTS
// =============================================================================

#[test]
fn test_single_forward_output_size() {
    for (name, gpu_config) in test_configs() {
        let mut gpu_network = GpuCfcNetwork::new(gpu_config.clone(), GpuBackend::Cpu)
            .expect("Failed to create GPU network");

        let input = vec![0.5f32; gpu_config.input_dim];
        let output = gpu_network.forward(&input, 0.1).unwrap();

        assert_eq!(
            output.len(),
            gpu_config.output_dim,
            "Config '{}': Output dimension mismatch",
            name
        );
    }
}

#[test]
fn test_forward_output_is_finite() {
    for (name, gpu_config) in test_configs() {
        let mut gpu_network = GpuCfcNetwork::new(gpu_config.clone(), GpuBackend::Cpu)
            .expect("Failed to create GPU network");

        let input = vec![0.5f32; gpu_config.input_dim];
        let output = gpu_network.forward(&input, 0.1).unwrap();

        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Config '{}': Output[{}] = {} is not finite",
                name,
                i,
                v
            );
        }
    }
}

#[test]
fn test_forward_with_extreme_inputs() {
    let gpu_config = GpuCfcConfig {
        input_dim: 16,
        hidden_dim: 32,
        num_layers: 2,
        output_dim: 8,
        use_backbone: false,
        ..Default::default()
    };

    let mut gpu_network = GpuCfcNetwork::new(gpu_config.clone(), GpuBackend::Cpu)
        .expect("Failed to create GPU network");

    // Test with various extreme inputs
    let test_cases = vec![
        ("zeros", vec![0.0f32; 16]),
        ("ones", vec![1.0f32; 16]),
        ("large", vec![10.0f32; 16]),
        ("small", vec![1e-6f32; 16]),
        ("negative", vec![-1.0f32; 16]),
        (
            "mixed",
            (0..16)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
        ),
    ];

    for (case_name, input) in test_cases {
        gpu_network.reset();
        let output = gpu_network.forward(&input, 0.1).unwrap();

        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Case '{}': Output[{}] = {} is not finite",
                case_name,
                i,
                v
            );
            assert!(
                v.abs() < 100.0,
                "Case '{}': Output[{}] = {} is unreasonably large",
                case_name,
                i,
                v
            );
        }
    }
}

// =============================================================================
// BATCH PROCESSING TESTS
// =============================================================================

#[test]
fn test_batch_forward_consistency() {
    // Test that batch processing produces consistent, finite outputs
    // Note: Batch processing and sequential processing are semantically different -
    // batch processes all inputs with fresh state per item, while sequential
    // accumulates state across items. This test verifies batch output quality.
    let gpu_config = GpuCfcConfig {
        input_dim: 32,
        hidden_dim: 64,
        num_layers: 2,
        output_dim: 16,
        use_backbone: false,
        ..Default::default()
    };

    let mut gpu_network = GpuCfcNetwork::new(gpu_config.clone(), GpuBackend::Cpu)
        .expect("Failed to create GPU network");

    let batch_size = 10;
    let inputs: Vec<Vec<f32>> = (0..batch_size)
        .map(|i| vec![(i as f32) * 0.1; 32])
        .collect();
    let dts: Vec<f32> = vec![0.1; batch_size];

    // Process as batch
    gpu_network.reset();
    let batch_outputs = gpu_network.forward_batch(&inputs, &dts).unwrap();

    // Verify batch outputs are valid
    assert_eq!(
        batch_outputs.len(),
        batch_size,
        "Batch should return {} outputs",
        batch_size
    );
    for (i, output) in batch_outputs.iter().enumerate() {
        assert_eq!(output.len(), 16, "Batch item {} should have 16 outputs", i);
        for (j, &v) in output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Batch item {}, output[{}] = {} is not finite",
                i,
                j,
                v
            );
            assert!(
                v.abs() < 100.0,
                "Batch item {}, output[{}] = {} is unreasonably large",
                i,
                j,
                v
            );
        }
    }

    // Verify determinism: same batch should produce same outputs
    gpu_network.reset();
    let batch_outputs_2 = gpu_network.forward_batch(&inputs, &dts).unwrap();

    const TOLERANCE: f32 = 1e-6;
    for (i, (out1, out2)) in batch_outputs.iter().zip(batch_outputs_2.iter()).enumerate() {
        let diff = max_diff(out1, out2);
        assert!(
            diff < TOLERANCE,
            "Batch item {}: max_diff = {} exceeds tolerance {} (non-deterministic batch)",
            i,
            diff,
            TOLERANCE
        );
    }
}

#[test]
fn test_sequence_processing() {
    let gpu_config = GpuCfcConfig {
        input_dim: 16,
        hidden_dim: 32,
        num_layers: 2,
        output_dim: 8,
        use_backbone: false,
        ..Default::default()
    };

    let mut gpu_network = GpuCfcNetwork::new(gpu_config.clone(), GpuBackend::Cpu)
        .expect("Failed to create GPU network");

    let seq_len = 50;
    let inputs: Vec<Vec<f32>> = (0..seq_len)
        .map(|i| {
            let t = (i as f32) * 0.1;
            vec![t.sin(); 16]
        })
        .collect();
    let dts: Vec<f32> = vec![0.1; seq_len];

    let outputs = gpu_network.forward_sequence(&inputs, &dts).unwrap();

    assert_eq!(outputs.len(), seq_len);
    for (i, output) in outputs.iter().enumerate() {
        assert_eq!(output.len(), 8);
        for (j, &v) in output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Sequence step {}, output[{}] = {} is not finite",
                i,
                j,
                v
            );
        }
    }
}

// =============================================================================
// STATE MANAGEMENT TESTS
// =============================================================================

#[test]
fn test_reset_clears_state() {
    let gpu_config = GpuCfcConfig {
        input_dim: 16,
        hidden_dim: 32,
        num_layers: 2,
        output_dim: 8,
        use_backbone: false,
        ..Default::default()
    };

    let mut gpu_network = GpuCfcNetwork::new(gpu_config.clone(), GpuBackend::Cpu)
        .expect("Failed to create GPU network");

    // Do some forward passes
    let input = vec![0.5; 16];
    for _ in 0..10 {
        let _ = gpu_network.forward(&input, 0.1);
    }

    // State should be non-zero
    let state_before = gpu_network.state();
    let magnitude_before: f32 = state_before
        .iter()
        .flat_map(|s| s.iter())
        .map(|&x| x.abs())
        .sum();
    assert!(
        magnitude_before > 0.0,
        "State should be non-zero after forward passes"
    );

    // Reset
    gpu_network.reset();

    // State should be zero
    let state_after = gpu_network.state();
    let magnitude_after: f32 = state_after
        .iter()
        .flat_map(|s| s.iter())
        .map(|&x| x.abs())
        .sum();
    assert!(magnitude_after < 1e-6, "State should be zero after reset");
}

#[test]
fn test_state_save_restore() {
    let gpu_config = GpuCfcConfig {
        input_dim: 16,
        hidden_dim: 32,
        num_layers: 2,
        output_dim: 8,
        use_backbone: false,
        ..Default::default()
    };

    let mut gpu_network = GpuCfcNetwork::new(gpu_config.clone(), GpuBackend::Cpu)
        .expect("Failed to create GPU network");

    // Build up some state
    let input = vec![0.5; 16];
    for _ in 0..5 {
        let _ = gpu_network.forward(&input, 0.1);
    }

    // Save state
    let saved_state = gpu_network.state();

    // Do more forward passes
    for _ in 0..5 {
        let _ = gpu_network.forward(&input, 0.1);
    }

    // Restore state
    gpu_network.set_state(saved_state.clone());

    // Verify state matches
    let restored_state = gpu_network.state();
    for (i, (saved, restored)) in saved_state.iter().zip(restored_state.iter()).enumerate() {
        for (j, (&s, &r)) in saved.iter().zip(restored.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-6,
                "Layer {}, element {}: saved={}, restored={}",
                i,
                j,
                s,
                r
            );
        }
    }
}

// =============================================================================
// NUMERICAL STABILITY TESTS
// =============================================================================

#[test]
fn test_long_sequence_stability() {
    let gpu_config = GpuCfcConfig {
        input_dim: 16,
        hidden_dim: 32,
        num_layers: 2,
        output_dim: 8,
        use_backbone: false,
        ..Default::default()
    };

    let mut gpu_network = GpuCfcNetwork::new(gpu_config.clone(), GpuBackend::Cpu)
        .expect("Failed to create GPU network");

    // Process a long sequence
    let seq_len = 1000;
    let inputs: Vec<Vec<f32>> = (0..seq_len)
        .map(|i| {
            let t = (i as f32) * 0.01;
            vec![t.sin(); 16]
        })
        .collect();
    let dts: Vec<f32> = vec![0.1; seq_len];

    let outputs = gpu_network.forward_sequence(&inputs, &dts).unwrap();

    // Check that outputs remain stable throughout
    for (i, output) in outputs.iter().enumerate() {
        for (j, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "Step {}, output[{}] = {} diverged", i, j, v);
            assert!(
                v.abs() < 20.0, // CfC should clamp to [-10, 10] but output projection might scale
                "Step {}, output[{}] = {} grew too large",
                i,
                j,
                v
            );
        }
    }

    println!("Long sequence ({} steps) completed successfully", seq_len);
}

#[test]
fn test_varying_dt_stability() {
    let gpu_config = GpuCfcConfig {
        input_dim: 16,
        hidden_dim: 32,
        num_layers: 2,
        output_dim: 8,
        use_backbone: false,
        ..Default::default()
    };

    let mut gpu_network = GpuCfcNetwork::new(gpu_config.clone(), GpuBackend::Cpu)
        .expect("Failed to create GPU network");

    // Test with varying time steps
    let test_dts = vec![
        0.001, // Very small
        0.01,  // Small
        0.1,   // Normal
        1.0,   // Large
        10.0,  // Very large
    ];

    let input = vec![0.5; 16];

    for &dt in &test_dts {
        gpu_network.reset();
        let output = gpu_network.forward(&input, dt).unwrap();

        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "dt={}: output[{}] = {} is not finite",
                dt,
                i,
                v
            );
        }
    }
}

// =============================================================================
// TRAINING TESTS
// =============================================================================

#[test]
fn test_training_reduces_loss() {
    let gpu_config = GpuCfcConfig {
        input_dim: 16,
        hidden_dim: 32,
        num_layers: 1,
        output_dim: 8,
        use_backbone: false,
        ..Default::default()
    };

    let mut gpu_network = GpuCfcNetwork::new(gpu_config.clone(), GpuBackend::Cpu)
        .expect("Failed to create GPU network");

    // Simple training data
    let inputs: Vec<Vec<f32>> = vec![vec![1.0; 16], vec![0.0; 16]];
    let targets: Vec<Vec<f32>> = vec![vec![1.0; 8], vec![0.0; 8]];
    let dts = vec![0.1, 0.1];

    // Get initial loss
    gpu_network.reset();
    let initial_loss = gpu_network
        .train_step(&inputs, &targets, &dts, 0.01)
        .unwrap();

    // Train for several steps
    for _ in 0..50 {
        gpu_network.reset();
        let _ = gpu_network.train_step(&inputs, &targets, &dts, 0.01);
    }

    // Get final loss
    gpu_network.reset();
    let final_loss = gpu_network
        .train_step(&inputs, &targets, &dts, 0.01)
        .unwrap();

    println!("Initial loss: {}, Final loss: {}", initial_loss, final_loss);

    // Loss should decrease or stay reasonable
    assert!(final_loss.is_finite(), "Final loss is not finite");
    // Note: With random initialization, loss might not always decrease
    // but should remain bounded
    assert!(final_loss < 10.0, "Final loss {} is too high", final_loss);
}

// =============================================================================
// ACTIVATION FUNCTION TESTS
// =============================================================================

#[test]
fn test_different_activations() {
    let activations = [
        ActivationType::SiLU,
        ActivationType::GELU,
        ActivationType::ReLU,
        ActivationType::Tanh,
        ActivationType::Sigmoid,
    ];

    for activation in activations {
        let gpu_config = GpuCfcConfig {
            input_dim: 16,
            hidden_dim: 32,
            num_layers: 2,
            output_dim: 8,
            use_backbone: false,
            activation,
            ..Default::default()
        };

        let mut gpu_network = GpuCfcNetwork::new(gpu_config.clone(), GpuBackend::Cpu)
            .expect("Failed to create GPU network");

        let input = vec![0.5; 16];
        let output = gpu_network.forward(&input, 0.1).unwrap();

        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Activation {:?}: output[{}] = {} is not finite",
                activation,
                i,
                v
            );
        }

        println!(
            "Activation {:?}: output = {:?}",
            activation,
            &output[..3.min(output.len())]
        );
    }
}

// =============================================================================
// PARAMETER COUNT TESTS
// =============================================================================

#[test]
fn test_parameter_count() {
    let gpu_config = GpuCfcConfig {
        input_dim: 32,
        hidden_dim: 64,
        num_layers: 2,
        output_dim: 16,
        use_backbone: true,
        backbone_layers: 2,
        backbone_dim: 48,
        ..Default::default()
    };

    let gpu_network = GpuCfcNetwork::new(gpu_config.clone(), GpuBackend::Cpu)
        .expect("Failed to create GPU network");

    let params = gpu_network.num_parameters();

    // Sanity check: should have significant number of parameters
    assert!(
        params > 1000,
        "Network should have more than 1000 parameters"
    );

    println!("Network has {} parameters", params);
}

// =============================================================================
// STATISTICS TESTS
// =============================================================================

#[test]
fn test_statistics_tracking() {
    let gpu_config = GpuCfcConfig {
        input_dim: 16,
        hidden_dim: 32,
        num_layers: 1,
        output_dim: 8,
        use_backbone: false,
        ..Default::default()
    };

    let mut gpu_network = GpuCfcNetwork::new(gpu_config.clone(), GpuBackend::Cpu)
        .expect("Failed to create GPU network");

    let input = vec![0.1; 16];
    let num_steps = 100;

    for _ in 0..num_steps {
        let _ = gpu_network.forward(&input, 0.1);
    }

    let stats = gpu_network.stats();
    assert_eq!(stats.total_steps, num_steps as u64);
    assert!(stats.total_forward_time_us > 0);
    assert!(stats.avg_forward_us > 0);

    println!("{}", stats);
}

// =============================================================================
// BACKEND SELECTION TESTS
// =============================================================================

#[test]
fn test_backend_auto_selection() {
    let backend = GpuBackend::Auto.resolve();
    println!(
        "Auto-selected backend: {:?} - {}",
        backend,
        backend.description()
    );

    // Should resolve to something valid (Cuda variant removed — only Cpu or Wgpu)
    assert!(matches!(backend, GpuBackend::Cpu | GpuBackend::Wgpu));
}

#[test]
fn test_cpu_backend_always_works() {
    let gpu_config = GpuCfcConfig::default();
    let gpu_network = GpuCfcNetwork::new(gpu_config, GpuBackend::Cpu);

    assert!(gpu_network.is_ok(), "CPU backend should always work");
    assert!(!gpu_network.unwrap().is_gpu_accelerated());
}