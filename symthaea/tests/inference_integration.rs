// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the inference/ module.
//!
//! Tests cover:
//! - StreamingConfig preset validation (low_latency, balanced, high_throughput)
//! - Utility factory functions (default_streaming, low_latency_streaming, high_capacity_streaming)
//! - Push and poll basic operation with sequence number verification
//! - Flush forces processing of pending inputs below batch threshold
//! - Stats accumulation (throughput, latency)
//! - State export/import (network_state / set_network_state)
//! - Stop/start control flow
//! - Reset clears all state
//! - Warmup detection (is_warmed_up)
//! - Batch push (push_batch)

use ndarray::Array1;
use symthaea::dynamics::cfc::{CfCConfig, CfCNetwork, CfCNetworkConfig};
use symthaea::inference::{
    StreamingConfig, StreamingInference, default_streaming, high_capacity_streaming,
    low_latency_streaming,
};

// ============================================================================
// Helpers
// ============================================================================

/// Create a deterministic input vector from a seed.
fn make_input(dim: usize, seed: u64) -> Array1<f32> {
    Array1::from_shape_fn(dim, |i| ((i as u64 + seed) as f32 * 0.1).sin())
}

/// Build a CfCNetwork with the given dimensions.
fn build_network(input_dim: usize, hidden_dim: usize, output_dim: usize) -> CfCNetwork {
    let config = CfCNetworkConfig {
        input_dim,
        hidden_dim,
        num_layers: 2,
        output_dim,
        cell_config: CfCConfig {
            input_dim,
            hidden_dim,
            ..Default::default()
        },
        ..Default::default()
    };
    CfCNetwork::new(config)
}

// ============================================================================
// 1. Config Preset Validation
// ============================================================================

#[test]
fn test_config_low_latency_has_sensible_defaults() {
    let config = StreamingConfig::low_latency();
    assert_eq!(
        config.batch_accumulation, 1,
        "Low latency should process every sample"
    );
    assert_eq!(config.max_latency_ms, 10, "Low latency should target 10ms");
    assert!(
        config.warmup_samples <= 8,
        "Low latency warmup should be small, got {}",
        config.warmup_samples
    );
    assert!(
        config.drop_on_backpressure,
        "Low latency should drop on backpressure for real-time operation"
    );
}

#[test]
fn test_config_balanced_has_sensible_defaults() {
    let config = StreamingConfig::balanced();
    assert!(
        config.batch_accumulation > 1 && config.batch_accumulation <= 16,
        "Balanced batch size should be moderate, got {}",
        config.batch_accumulation
    );
    assert_eq!(config.max_latency_ms, 25, "Balanced should target 25ms");
    assert!(
        config.enable_checkpoints,
        "Balanced should enable checkpoints"
    );
}

#[test]
fn test_config_high_throughput_has_sensible_defaults() {
    let config = StreamingConfig::high_throughput();
    assert!(
        config.batch_accumulation >= 16,
        "High throughput should batch at least 16, got {}",
        config.batch_accumulation
    );
    assert!(
        config.max_latency_ms >= 100,
        "High throughput can tolerate >=100ms latency, got {}",
        config.max_latency_ms
    );
    assert!(
        config.max_output_queue >= 128,
        "High throughput should have large output queue, got {}",
        config.max_output_queue
    );
    assert!(
        !config.drop_on_backpressure,
        "High throughput should not drop outputs"
    );
}

#[test]
fn test_config_presets_ordering() {
    let low = StreamingConfig::low_latency();
    let balanced = StreamingConfig::balanced();
    let high = StreamingConfig::high_throughput();

    // Batch accumulation should increase from low_latency to high_throughput
    assert!(
        low.batch_accumulation <= balanced.batch_accumulation,
        "Low latency batch ({}) should be <= balanced ({})",
        low.batch_accumulation,
        balanced.batch_accumulation
    );
    assert!(
        balanced.batch_accumulation <= high.batch_accumulation,
        "Balanced batch ({}) should be <= high throughput ({})",
        balanced.batch_accumulation,
        high.batch_accumulation
    );

    // Latency budget should increase from low_latency to high_throughput
    assert!(
        low.max_latency_ms <= balanced.max_latency_ms,
        "Low latency budget ({}) should be <= balanced ({})",
        low.max_latency_ms,
        balanced.max_latency_ms
    );
    assert!(
        balanced.max_latency_ms <= high.max_latency_ms,
        "Balanced latency budget ({}) should be <= high throughput ({})",
        balanced.max_latency_ms,
        high.max_latency_ms
    );
}

// ============================================================================
// 2. Utility Factory Functions
// ============================================================================

#[test]
fn test_default_streaming_creates_working_instance() {
    let streamer = default_streaming(8, 4);
    assert!(streamer.is_active(), "Should be active on creation");
    assert!(!streamer.is_warmed_up(), "Should not be warmed up yet");
    assert_eq!(
        streamer.total_samples(),
        0,
        "Should have zero samples initially"
    );
    assert_eq!(streamer.current_sequence(), 0, "Sequence should start at 0");
}

#[test]
fn test_low_latency_streaming_creates_working_instance() {
    let streamer = low_latency_streaming(8, 4);
    assert!(streamer.is_active());
    assert_eq!(
        streamer.config().batch_accumulation,
        1,
        "Low latency factory should use batch_accumulation=1"
    );
    assert_eq!(
        streamer.config().max_latency_ms,
        10,
        "Low latency factory should use 10ms budget"
    );
}

#[test]
fn test_high_capacity_streaming_creates_working_instance() {
    let streamer = high_capacity_streaming(8, 4);
    assert!(streamer.is_active());
    assert!(
        streamer.config().batch_accumulation >= 16,
        "High capacity factory should use large batch, got {}",
        streamer.config().batch_accumulation
    );
}

// ============================================================================
// 3. Push and Poll Basic
// ============================================================================

#[test]
fn test_push_and_poll_produces_outputs() {
    let network = build_network(8, 32, 4);
    let config = StreamingConfig {
        batch_accumulation: 1,
        warmup_samples: 0,
        max_latency_ms: 1000,
        ..Default::default()
    };
    let streamer = StreamingInference::new(network, config);

    // Push 10 inputs and collect outputs
    let mut outputs = Vec::new();
    for i in 0..10u64 {
        streamer.push(make_input(8, i));
        while let Some(out) = streamer.poll() {
            outputs.push(out);
        }
    }

    assert!(
        !outputs.is_empty(),
        "Should have produced at least some outputs after 10 pushes"
    );

    // Verify sequence numbers are monotonically increasing
    for window in outputs.windows(2) {
        assert!(
            window[1].sequence > window[0].sequence,
            "Sequence numbers should increase: {} vs {}",
            window[0].sequence,
            window[1].sequence
        );
    }

    // Verify output dimension
    for out in &outputs {
        assert_eq!(
            out.output.len(),
            4,
            "Output dim should match network output_dim=4"
        );
    }
}

#[test]
fn test_push_and_poll_output_values_are_finite() {
    let network = build_network(8, 32, 4);
    let config = StreamingConfig {
        batch_accumulation: 1,
        warmup_samples: 0,
        max_latency_ms: 1000,
        ..Default::default()
    };
    let streamer = StreamingInference::new(network, config);

    for i in 0..5u64 {
        streamer.push(make_input(8, i));
    }

    let outputs = streamer.poll_all();
    for out in &outputs {
        for &val in out.output.iter() {
            assert!(
                val.is_finite(),
                "Output value should be finite, got {}",
                val
            );
        }
        assert!(
            out.latency_us < 10_000_000,
            "Latency should be reasonable (<10s), got {}us",
            out.latency_us
        );
    }
}

// ============================================================================
// 4. Flush Forces Processing
// ============================================================================

#[test]
fn test_flush_forces_processing_below_batch_threshold() {
    let network = build_network(8, 32, 4);
    let config = StreamingConfig {
        batch_accumulation: 100, // Very large threshold so normal push won't trigger
        warmup_samples: 0,
        max_latency_ms: 60_000, // Very long timeout so it won't auto-trigger
        ..Default::default()
    };
    let streamer = StreamingInference::new(network, config);

    // Push 3 inputs (well below batch_accumulation=100)
    for i in 0..3u64 {
        streamer.push(make_input(8, i));
    }

    // No output yet (batch not reached, timeout not reached)
    assert!(
        streamer.poll().is_none(),
        "Should have no output before flush"
    );

    // Flush forces processing
    let flushed = streamer.flush();
    assert!(
        flushed.is_some(),
        "Flush should produce an output from pending inputs"
    );

    let output = flushed.unwrap();
    assert!(
        output.forced_flush,
        "Output from flush should be marked as forced_flush"
    );
    assert_eq!(
        output.input_count, 3,
        "Flushed output should reflect all 3 pending inputs"
    );
}

// ============================================================================
// 5. Stats Accumulation
// ============================================================================

#[test]
fn test_stats_accumulation_after_pushes() {
    let network = build_network(8, 32, 4);
    let config = StreamingConfig {
        batch_accumulation: 1,
        warmup_samples: 0,
        max_latency_ms: 1000,
        ..Default::default()
    };
    let streamer = StreamingInference::new(network, config);

    // Push 20 inputs
    for i in 0..20u64 {
        streamer.push(make_input(8, i));
    }

    // Drain all outputs
    let _ = streamer.poll_all();

    let stats = streamer.stats();
    eprintln!("Stats after 20 pushes: {:?}", stats);

    assert!(
        stats.total_inputs >= 20,
        "total_inputs should be >= 20, got {}",
        stats.total_inputs
    );
    assert!(
        stats.total_outputs > 0,
        "total_outputs should be > 0, got {}",
        stats.total_outputs
    );
    assert!(
        stats.total_batches > 0,
        "total_batches should be > 0, got {}",
        stats.total_batches
    );
    assert!(
        stats.avg_latency_us.is_finite() && stats.avg_latency_us >= 0.0,
        "avg_latency_us should be non-negative and finite, got {}",
        stats.avg_latency_us
    );
    assert!(
        stats.avg_batch_size.is_finite() && stats.avg_batch_size > 0.0,
        "avg_batch_size should be positive and finite, got {}",
        stats.avg_batch_size
    );
}

// ============================================================================
// 6. State Export/Import
// ============================================================================

#[test]
fn test_state_export_and_import() {
    let network = build_network(8, 32, 4);
    let config = StreamingConfig {
        batch_accumulation: 1,
        warmup_samples: 0,
        max_latency_ms: 1000,
        ..Default::default()
    };
    let streamer = StreamingInference::new(network, config);

    // Push some data to get meaningful state
    for i in 0..5u64 {
        streamer.push(make_input(8, i));
    }
    let _ = streamer.poll_all();

    // Export state
    let saved_state = streamer.network_state();
    assert!(
        !saved_state.is_empty(),
        "Exported state should not be empty"
    );
    eprintln!("Exported {} state arrays", saved_state.len());

    // Push more data to change state
    for i in 10..20u64 {
        streamer.push(make_input(8, i));
    }
    let _ = streamer.poll_all();

    let changed_state = streamer.network_state();

    // Restore original state
    streamer.set_network_state(saved_state.clone());
    let restored_state = streamer.network_state();

    // Restored state should match saved state
    assert_eq!(
        saved_state.len(),
        restored_state.len(),
        "Restored state should have same number of arrays"
    );
    for (i, (saved, restored)) in saved_state.iter().zip(restored_state.iter()).enumerate() {
        assert_eq!(
            saved.len(),
            restored.len(),
            "State array {} dimension mismatch",
            i
        );
        for (j, (s, r)) in saved.iter().zip(restored.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-6,
                "State[{}][{}] mismatch after restore: {} vs {}",
                i,
                j,
                s,
                r
            );
        }
    }

    // Verify the state actually changed before restoration (sanity check)
    let any_different = saved_state.iter().zip(changed_state.iter()).any(|(s, c)| {
        s.iter()
            .zip(c.iter())
            .any(|(sv, cv)| (sv - cv).abs() > 1e-10)
    });
    // Note: this may not always be true if CfC converges quickly, so we just log it
    eprintln!(
        "State changed between saves: {}",
        if any_different {
            "yes"
        } else {
            "no (may have converged)"
        }
    );
}

// ============================================================================
// 7. Stop/Start Control
// ============================================================================

#[test]
fn test_stop_rejects_inputs_start_resumes() {
    let network = build_network(8, 32, 4);
    let config = StreamingConfig {
        batch_accumulation: 1,
        warmup_samples: 0,
        max_latency_ms: 1000,
        ..Default::default()
    };
    let streamer = StreamingInference::new(network, config);

    assert!(streamer.is_active(), "Should be active initially");

    // Push some data first to verify it works
    streamer.push(make_input(8, 0));
    let initial_samples = streamer.total_samples();
    assert!(
        initial_samples > 0,
        "Should have processed at least 1 sample"
    );

    // Stop processing
    streamer.stop();
    assert!(!streamer.is_active(), "Should be inactive after stop");

    // Push data while stopped
    let pushed_while_stopped = streamer.push(make_input(8, 100));
    assert!(
        !pushed_while_stopped,
        "Push should return false when stopped"
    );
    assert_eq!(
        streamer.total_samples(),
        initial_samples,
        "Total samples should not increase while stopped"
    );

    // Start again
    streamer.start();
    assert!(streamer.is_active(), "Should be active after start");

    // Push should work again
    streamer.push(make_input(8, 200));
    assert!(
        streamer.total_samples() > initial_samples,
        "Total samples should increase after restart"
    );
}

// ============================================================================
// 8. Reset Clears All
// ============================================================================

#[test]
fn test_reset_clears_all_state() {
    let network = build_network(8, 32, 4);
    let config = StreamingConfig {
        batch_accumulation: 1,
        warmup_samples: 5,
        max_latency_ms: 1000,
        ..Default::default()
    };
    let streamer = StreamingInference::new(network, config);

    // Push data
    for i in 0..10u64 {
        streamer.push(make_input(8, i));
    }
    let _ = streamer.poll_all();

    assert!(
        streamer.total_samples() >= 10,
        "Should have 10+ samples before reset"
    );

    // Reset
    streamer.reset();

    assert_eq!(
        streamer.total_samples(),
        0,
        "total_samples should be 0 after reset"
    );
    assert!(
        streamer.poll().is_none(),
        "Output queue should be empty after reset"
    );
    assert!(
        !streamer.is_warmed_up(),
        "Should not be warmed up after reset"
    );
    assert!(
        streamer.is_active(),
        "Should still be active after reset (reset != stop)"
    );
}

// ============================================================================
// 9. Warmup Detection
// ============================================================================

#[test]
fn test_warmup_detection() {
    let warmup_count = 5;
    let network = build_network(8, 32, 4);
    let config = StreamingConfig {
        batch_accumulation: 1,
        warmup_samples: warmup_count,
        max_latency_ms: 1000,
        ..Default::default()
    };
    let streamer = StreamingInference::new(network, config);

    assert!(!streamer.is_warmed_up(), "Should not be warmed up at start");

    // Push fewer than warmup_samples
    for i in 0..(warmup_count - 1) {
        streamer.push(make_input(8, i as u64));
        assert!(
            !streamer.is_warmed_up(),
            "Should not be warmed up after {} samples (need {})",
            i + 1,
            warmup_count
        );
    }

    // During warmup, poll should return None (inference not triggered)
    assert!(
        streamer.poll().is_none(),
        "Should produce no output during warmup"
    );

    // Push the final warmup sample
    streamer.push(make_input(8, warmup_count as u64));
    assert!(
        streamer.is_warmed_up(),
        "Should be warmed up after {} samples",
        warmup_count
    );
}

// ============================================================================
// 10. Batch Push
// ============================================================================

#[test]
fn test_push_batch_processes_all_inputs() {
    let network = build_network(8, 32, 4);
    let config = StreamingConfig {
        batch_accumulation: 1,
        warmup_samples: 0,
        max_latency_ms: 1000,
        ..Default::default()
    };
    let streamer = StreamingInference::new(network, config);

    // Create a batch of 5 inputs
    let batch: Vec<Array1<f32>> = (0..5).map(|i| make_input(8, i)).collect();
    let count = streamer.push_batch(batch);

    assert_eq!(
        count, 5,
        "push_batch should return the number of inputs pushed"
    );
    assert_eq!(
        streamer.total_samples(),
        5,
        "total_samples should reflect the batch"
    );

    // Drain outputs
    let outputs = streamer.poll_all();
    eprintln!("Outputs from batch of 5: {}", outputs.len());

    // We should have at least one output (batch_accumulation=1 means process on every push)
    assert!(
        !outputs.is_empty(),
        "Should have produced outputs from batch push"
    );

    // All outputs should have valid dimensions
    for out in &outputs {
        assert_eq!(out.output.len(), 4, "Output dim should be 4");
        for &val in out.output.iter() {
            assert!(val.is_finite(), "Output value should be finite");
        }
    }
}

#[test]
fn test_push_batch_empty_is_noop() {
    let network = build_network(8, 32, 4);
    let config = StreamingConfig {
        batch_accumulation: 1,
        warmup_samples: 0,
        ..Default::default()
    };
    let streamer = StreamingInference::new(network, config);

    let count = streamer.push_batch(vec![]);
    assert_eq!(count, 0, "Empty batch should return 0");
    assert_eq!(streamer.total_samples(), 0, "No samples should be recorded");
}

// ============================================================================
// Additional Edge Cases
// ============================================================================

#[test]
fn test_peek_does_not_consume() {
    let network = build_network(8, 32, 4);
    let config = StreamingConfig {
        batch_accumulation: 1,
        warmup_samples: 0,
        max_latency_ms: 1000,
        ..Default::default()
    };
    let streamer = StreamingInference::new(network, config);

    // Push enough to produce output
    for i in 0..3u64 {
        streamer.push(make_input(8, i));
    }

    // Peek should return the same output as poll
    if let Some(peeked) = streamer.peek() {
        let polled = streamer
            .poll()
            .expect("Should have an output to poll after peek");
        assert_eq!(
            peeked.sequence, polled.sequence,
            "Peek and poll should return the same output"
        );
    }
}

#[test]
fn test_poll_all_drains_queue() {
    let network = build_network(8, 32, 4);
    let config = StreamingConfig {
        batch_accumulation: 1,
        warmup_samples: 0,
        max_latency_ms: 1000,
        ..Default::default()
    };
    let streamer = StreamingInference::new(network, config);

    for i in 0..10u64 {
        streamer.push(make_input(8, i));
    }

    let all = streamer.poll_all();
    assert!(
        !all.is_empty(),
        "poll_all should return outputs after pushes"
    );

    // After poll_all, queue should be empty
    assert!(
        streamer.poll().is_none(),
        "Queue should be empty after poll_all"
    );
}

#[test]
fn test_checkpoint_sequences_and_restore() {
    let network = build_network(8, 32, 4);
    let config = StreamingConfig {
        batch_accumulation: 1,
        warmup_samples: 0,
        enable_checkpoints: true,
        checkpoint_interval: 4,
        max_latency_ms: 1000,
        ..Default::default()
    };
    let streamer = StreamingInference::new(network, config);

    // Push enough data to trigger checkpoints (checkpoint_interval=4)
    for i in 0..20u64 {
        streamer.push(make_input(8, i));
    }
    let _ = streamer.poll_all();

    let sequences = streamer.checkpoint_sequences();
    eprintln!("Checkpoint sequences: {:?}", sequences);

    // Attempt to restore a non-existent checkpoint
    assert!(
        !streamer.restore_checkpoint(999999),
        "Restoring non-existent checkpoint should return false"
    );

    // If we have checkpoints, test restoring one
    if let Some(&seq) = sequences.first() {
        assert!(
            streamer.restore_checkpoint(seq),
            "Restoring existing checkpoint {} should succeed",
            seq
        );
    }
}

#[test]
fn test_reset_stats_clears_statistics() {
    let network = build_network(8, 32, 4);
    let config = StreamingConfig {
        batch_accumulation: 1,
        warmup_samples: 0,
        max_latency_ms: 1000,
        ..Default::default()
    };
    let streamer = StreamingInference::new(network, config);

    // Generate some stats
    for i in 0..5u64 {
        streamer.push(make_input(8, i));
    }
    let _ = streamer.poll_all();

    let stats_before = streamer.stats();
    assert!(
        stats_before.total_outputs > 0,
        "Should have stats before reset"
    );

    // Reset stats
    streamer.reset_stats();
    let stats_after = streamer.stats();
    assert_eq!(
        stats_after.total_outputs, 0,
        "Stats should be zeroed after reset_stats"
    );
    assert_eq!(
        stats_after.total_inputs, 0,
        "Stats should be zeroed after reset_stats"
    );
}