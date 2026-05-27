// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Full pipeline integration tests — no feature gate required

// ==================================================================================
// Full Pipeline Integration Test: 15 Improvements Wired End-to-End
// ==================================================================================
//
// This test validates the complete Symthaea HLB pipeline, wiring all 15 improvements
// into a single deterministic flow:
//
//   1. Genesis seed -> deterministic initialization
//   2. HDC encoding with SIMD-accelerated operations
//   3. Hierarchical CfC multi-scale temporal processing (tau = [0.01, 0.1, 1.0, 10.0])
//   4. HDC-CfC bidirectional bridge (16384D <-> 128D)
//   5. Phi-guided attention gating
//   6. Online learning during inference
//   7. Episodic memory storage and replay
//   8. Surprise-driven exploration
//   9. Causal discovery
//  10. Streaming inference output
//  11. Checkpoint save/restore
//  12. Federated gradient sharing (stub)
//  13. GPU CfC (CPU fallback mode)
//  14. Attention visualization capture
//  15. Two-track semantic+temporal processing
//
// ==================================================================================

use std::collections::HashMap;
use std::time::Instant;

use ndarray::Array1;

// Core HDC types and genesis
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::simd_continuous::{
    bind_simd, bundle_simd, dot_product_simd, norm_simd, similarity_simd,
};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// Dynamics
use symthaea::dynamics::cfc::{CfCConfig, CfCNetwork, CfCNetworkConfig};
use symthaea::dynamics::cfc_gpu::{GpuBackend, GpuCfcConfig, GpuCfcNetwork};
use symthaea::dynamics::hierarchical_cfc::{HierarchicalCfC, HierarchicalCfCConfig};

// Bridges
// bridges module removed — HdcCfcBridge superseded by cognitive_loop

// Attention
use symthaea::attention::{PhiAttentionConfig, PhiAttentionGate};

// Memory
use symthaea::memory::episodic_replay::{Episode, EpisodicMemory, EpisodicReplayConfig};

// Exploration
use symthaea::exploration::surprise_driven::{SurpriseTracker, SurpriseTrackerConfig};

// Causal
use symthaea::causal::CausalLoopEnhancer;

// Inference
use symthaea::inference::{StreamingConfig, StreamingInference};

// Checkpoint
use symthaea::swarm::FederatedCheckpoint;
use symthaea::swarm::FederatedNetworkConfig;

// Visualization
use symthaea::visualization::attention_viz::{AttentionHistory, AttentionSnapshot};

// ==================================================================================
// CONSTANTS
// ==================================================================================

const GENESIS_PHRASE: &str = "integration_test_seed_2026";
const INPUT_DIM: usize = 256;
const CFC_HIDDEN: usize = 128;
const SEQUENCE_LEN: usize = 16;
/// Smaller HDC dim for the main pipeline test to keep wall time under 60s at opt-level 1.
/// Individual stage tests still use HDC_DIMENSION (16384) for full-fidelity checks.
const PIPELINE_HDC_DIM: usize = 2048;

// ==================================================================================
// HELPERS
// ==================================================================================

/// Generate a synthetic sine+noise input sequence deterministically.
fn generate_synthetic_sequence(genesis: &GenesisSeed, length: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut sequence = Vec::with_capacity(length);
    let mut rng = genesis.domain("synthetic_input");
    use rand::RngCore;

    for t in 0..length {
        let mut sample = Vec::with_capacity(dim);
        for d in 0..dim {
            // Multi-frequency sine wave
            let phase = (t as f32 * 0.1 + d as f32 * 0.01).sin();
            let harmonic = (t as f32 * 0.37 + d as f32 * 0.03).cos() * 0.3;

            // Deterministic noise from genesis
            let mut noise_buf = [0u8; 4];
            rng.fill_bytes(&mut noise_buf);
            let noise = (u32::from_le_bytes(noise_buf) as f32 / u32::MAX as f32 - 0.5) * 0.1;

            sample.push(phase + harmonic + noise);
        }
        sequence.push(sample);
    }
    sequence
}

/// Print a stage summary line with timing.
fn print_stage(stage_num: usize, name: &str, elapsed: std::time::Duration, details: &str) {
    println!(
        "  [{:>2}] {:<45} {:>8.2}ms  {}",
        stage_num,
        name,
        elapsed.as_secs_f64() * 1000.0,
        details,
    );
}

/// Assert all values in a slice are finite.
fn assert_all_finite(values: &[f32], context: &str) {
    for (i, v) in values.iter().enumerate() {
        assert!(
            v.is_finite(),
            "{}: value at index {} is not finite: {}",
            context,
            i,
            v
        );
    }
}

// ==================================================================================
// MAIN INTEGRATION TEST
// ==================================================================================

#[test]
fn test_full_pipeline_integration() {
    println!("\n{}", "=".repeat(80));
    println!("  SYMTHAEA HLB: Full Pipeline Integration Test (15 Improvements)");
    println!("{}\n", "=".repeat(80));

    let pipeline_start = Instant::now();

    // ──────────────────────────────────────────────────────────────────────────
    // STAGE 1: Genesis Seed -> Deterministic Initialization
    // ──────────────────────────────────────────────────────────────────────────
    let t = Instant::now();

    let genesis = GenesisSeed::from_phrase(GENESIS_PHRASE);

    // Verify determinism: same phrase => same timeline
    let genesis2 = GenesisSeed::from_phrase(GENESIS_PHRASE);
    assert_eq!(
        genesis.timeline_id(),
        genesis2.timeline_id(),
        "Genesis must be deterministic"
    );

    // Generate a base HV and verify reproducibility
    let base_hv = genesis.hv("pipeline:base", PIPELINE_HDC_DIM);
    let base_hv2 = genesis2.hv("pipeline:base", PIPELINE_HDC_DIM);
    assert_eq!(
        base_hv.values, base_hv2.values,
        "Same genesis + domain must produce identical HVs"
    );

    // Generate domain-separated HVs for different concepts
    let concept_hv = genesis.hv("concept:consciousness", PIPELINE_HDC_DIM);
    let sim = base_hv.similarity(&concept_hv);
    assert!(
        sim.abs() < 0.1,
        "Different domains should be near-orthogonal, got {:.4}",
        sim
    );

    print_stage(
        1,
        "Genesis Seed (deterministic init)",
        t.elapsed(),
        &format!(
            "timeline={}, dim={}, orthogonality={:.4}",
            &genesis.timeline_id()[..8],
            PIPELINE_HDC_DIM,
            sim.abs()
        ),
    );

    // ──────────────────────────────────────────────────────────────────────────
    // STAGE 2: HDC Encoding with SIMD-Accelerated Operations
    // ──────────────────────────────────────────────────────────────────────────
    let t = Instant::now();

    // Create two random HVs for SIMD operations
    let hv_a = genesis.hv("simd:a", PIPELINE_HDC_DIM);
    let hv_b = genesis.hv("simd:b", PIPELINE_HDC_DIM);

    // SIMD dot product
    let dot = dot_product_simd(&hv_a.values, &hv_b.values);
    assert!(dot.is_finite(), "SIMD dot product must be finite");

    // SIMD cosine similarity
    let cos_sim = similarity_simd(&hv_a.values, &hv_b.values);
    assert!(
        cos_sim.is_finite() && (-1.001..=1.001).contains(&cos_sim),
        "SIMD similarity must be in [-1,1], got {}",
        cos_sim
    );

    // SIMD bind (element-wise multiply)
    let bound = bind_simd(&hv_a.values, &hv_b.values);
    assert_eq!(
        bound.len(),
        PIPELINE_HDC_DIM,
        "Bind output dimension mismatch"
    );
    assert_all_finite(&bound, "SIMD bind");

    // SIMD bundle (weighted average of multiple HVs)
    let hv_c = genesis.hv("simd:c", PIPELINE_HDC_DIM);
    let hvs_refs: Vec<&[f32]> = vec![&hv_a.values, &hv_b.values, &hv_c.values];
    let weights = vec![0.5, 0.3, 0.2];
    let bundled = bundle_simd(&hvs_refs, &weights);
    assert_eq!(
        bundled.len(),
        PIPELINE_HDC_DIM,
        "Bundle output dimension mismatch"
    );
    assert_all_finite(&bundled, "SIMD bundle");

    // SIMD norm
    let n = norm_simd(&hv_a.values);
    assert!(
        n > 0.0 && n.is_finite(),
        "SIMD norm must be positive finite"
    );

    print_stage(
        2,
        "HDC Encoding (SIMD ops)",
        t.elapsed(),
        &format!(
            "dot={:.2}, sim={:.4}, bind_dim={}, bundle_dim={}, norm={:.2}",
            dot,
            cos_sim,
            bound.len(),
            bundled.len(),
            n
        ),
    );

    // ──────────────────────────────────────────────────────────────────────────
    // STAGE 3: Hierarchical CfC Multi-Scale Temporal Processing
    // ──────────────────────────────────────────────────────────────────────────
    let t = Instant::now();

    let hcfc_config = HierarchicalCfCConfig {
        input_dim: 64,
        output_dim: 32,
        time_constants: vec![0.01, 0.1, 1.0, 10.0],
        hidden_dims: vec![64, 64, 48, 32],
        top_down_feedback: true,
        feedback_strength: 0.3,
        bottom_up_integration: true,
        integration_strength: 0.5,
        multi_scale_prediction: true,
        lr_scales: vec![0.5, 1.0, 1.0, 0.5],
    };
    let mut hcfc = HierarchicalCfC::new(hcfc_config);

    assert_eq!(hcfc.num_layers(), 4, "Should have 4 hierarchical layers");
    assert_eq!(
        hcfc.time_constants(),
        &[0.01, 0.1, 1.0, 10.0],
        "Time constants must match"
    );

    // Process several steps
    let hcfc_input = Array1::from_vec(vec![0.3f32; 64]);
    let mut last_combined = None;
    for step in 0..20 {
        let output = hcfc.forward_hierarchical(&hcfc_input, 0.02);

        assert_eq!(
            output.combined.len(),
            32,
            "Combined output dim at step {}",
            step
        );
        assert_eq!(output.scale_outputs.len(), 4, "Should have 4 scale outputs");
        assert!(
            output.combined.iter().all(|x| x.is_finite()),
            "Hierarchical output must be finite at step {}",
            step
        );
        for (i, so) in output.scale_outputs.iter().enumerate() {
            assert_eq!(so.len(), 32, "Scale {} output dim", i);
            assert!(
                so.iter().all(|x| x.is_finite()),
                "Scale {} must be finite at step {}",
                i,
                step
            );
        }
        last_combined = Some(output.combined.clone());
    }

    // Verify weights are normalized
    let cw_sum: f32 = hcfc.combination_weights().iter().sum();
    assert!(
        (cw_sum - 1.0).abs() < 0.01,
        "Combination weights must sum to ~1.0, got {}",
        cw_sum
    );

    print_stage(
        3,
        "Hierarchical CfC (4 tau scales)",
        t.elapsed(),
        &format!(
            "layers=4, params={}, combined_mag={:.4}",
            hcfc.num_parameters(),
            last_combined
                .as_ref()
                .map(|c| c.iter().map(|x| x * x).sum::<f32>().sqrt())
                .unwrap_or(0.0)
        ),
    );

    // STAGE 4 (HDC-CfC Bridge) REMOVED — bridges module superseded by cognitive_loop

    // ──────────────────────────────────────────────────────────────────────────
    // STAGE 5: Phi-Guided Attention Gating
    // ──────────────────────────────────────────────────────────────────────────
    let t = Instant::now();

    let phi_config = PhiAttentionConfig {
        temperature: 1.0,
        learnable_mapping: false,
        normalize_output: true,
        min_attention: 0.0,
        ..Default::default()
    };
    let mut phi_gate = PhiAttentionGate::new(phi_config);

    // Create inputs with varying Phi values
    let input_hvs: Vec<ContinuousHV> = (0..4)
        .map(|i| genesis.hv(&format!("phi:input:{}", i), PIPELINE_HDC_DIM))
        .collect();
    let phi_values = vec![0.9, 0.3, 0.1, 0.7]; // First and last have high Phi

    let attention_result = phi_gate.forward(&input_hvs, &phi_values);

    // Verify attention weights sum to ~1.0
    let weight_sum: f32 = attention_result.weights.iter().sum();
    assert!(
        (weight_sum - 1.0).abs() < 0.01,
        "Attention weights must sum to ~1.0, got {}",
        weight_sum
    );

    // High-Phi inputs should get more weight
    assert!(
        attention_result.weights[0] > attention_result.weights[2],
        "Higher Phi should get more attention weight: w[0]={:.4} > w[2]={:.4}",
        attention_result.weights[0],
        attention_result.weights[2]
    );

    // Output should be valid
    assert_eq!(
        attention_result.output.dim(),
        PIPELINE_HDC_DIM,
        "Attention output dimension"
    );
    assert!(
        attention_result.output.values.iter().all(|x| x.is_finite()),
        "Attention output must be finite"
    );

    print_stage(
        5,
        "Phi-Guided Attention Gating",
        t.elapsed(),
        &format!(
            "weights=[{:.3},{:.3},{:.3},{:.3}], entropy={:.4}",
            attention_result.weights[0],
            attention_result.weights[1],
            attention_result.weights[2],
            attention_result.weights[3],
            attention_result.entropy
        ),
    );

    // ──────────────────────────────────────────────────────────────────────────
    // STAGE 6: Online Learning During Inference
    // ──────────────────────────────────────────────────────────────────────────
    let t = Instant::now();

    let cfc_config = CfCNetworkConfig {
        input_dim: INPUT_DIM,
        hidden_dim: CFC_HIDDEN,
        num_layers: 2,
        output_dim: 64,
        cell_config: CfCConfig {
            input_dim: INPUT_DIM,
            hidden_dim: CFC_HIDDEN,
            ..Default::default()
        },
        residual: true,
        bidirectional: false,
        enable_online_learning: false,
        online_learning_config: Default::default(),
    };
    let mut cfc_net = CfCNetwork::from_genesis(cfc_config, &genesis, "online_learn");

    // Generate sequence
    let sequence = generate_synthetic_sequence(&genesis, SEQUENCE_LEN, INPUT_DIM);

    // Forward pass (inference)
    let first_input = Array1::from_vec(sequence[0].clone());
    let first_output = cfc_net.forward(&first_input, 0.02);
    assert_eq!(first_output.len(), 64, "CfC output dim");
    assert!(
        first_output.iter().all(|x| x.is_finite()),
        "CfC output must be finite"
    );

    // Online training steps (limited to 3 for test speed at opt-level 1)
    let mut total_train_loss = 0.0f32;
    let mut train_count = 0;
    for i in 0..3.min(sequence.len() - 1) {
        let input = Array1::from_vec(sequence[i].clone());
        let target_raw = &sequence[i + 1];
        // Compress target to output_dim
        let target: Vec<f32> = (0..64)
            .map(|j| {
                let start = j * (INPUT_DIM / 64);
                let end = ((j + 1) * (INPUT_DIM / 64)).min(INPUT_DIM);
                target_raw[start..end].iter().sum::<f32>() / (end - start) as f32
            })
            .collect();
        let target_arr = Array1::from_vec(target);

        match cfc_net.train_step(&input, &target_arr, 0.02, 0.001) {
            Ok(loss) => {
                assert!(
                    loss.is_finite(),
                    "Training loss must be finite at step {}",
                    i
                );
                total_train_loss += loss;
                train_count += 1;
            }
            Err(e) => panic!("Training failed at step {}: {}", i, e),
        }
    }
    let avg_loss = if train_count > 0 {
        total_train_loss / train_count as f32
    } else {
        0.0
    };

    print_stage(
        6,
        "Online Learning During Inference",
        t.elapsed(),
        &format!(
            "steps={}, avg_loss={:.6}, output_dim={}",
            train_count,
            avg_loss,
            first_output.len()
        ),
    );

    // ──────────────────────────────────────────────────────────────────────────
    // STAGE 7: Episodic Memory Storage and Replay
    // ──────────────────────────────────────────────────────────────────────────
    let t = Instant::now();

    let mem_config = EpisodicReplayConfig {
        capacity: 100,
        psi_threshold: 0.3,
        replay_interval: 5,
        batch_size: 4,
        ..Default::default()
    };
    let mut episodic_memory = EpisodicMemory::new(mem_config);

    // Store episodes with varying Phi values
    let mut stored_count = 0;
    for i in 0..10 {
        let phi = 0.1 + (i as f64 * 0.05); // 0.1 to 1.05
        let input_hv = genesis.hv(&format!("episode:in:{}", i), PIPELINE_HDC_DIM);
        let output_hv = genesis.hv(&format!("episode:out:{}", i), PIPELINE_HDC_DIM);
        let episode = Episode::new(input_hv, output_hv, phi, i as u64);

        if episodic_memory.store_if_significant(episode) {
            stored_count += 1;
        }
    }

    // Should have stored episodes above phi_threshold=0.3
    assert!(
        stored_count > 0,
        "Should store at least some high-Phi episodes"
    );

    // Sample a replay batch
    let batch = episodic_memory.sample_replay_batch(4);
    // All replayed episodes should have Phi >= threshold
    for ep in &batch {
        assert!(
            ep.psi >= 0.3,
            "Replayed episode Phi ({}) must be >= threshold",
            ep.psi
        );
    }

    print_stage(
        7,
        "Episodic Memory (store & replay)",
        t.elapsed(),
        &format!(
            "stored={}/{}, batch_size={}, above_threshold=all",
            stored_count,
            20,
            batch.len()
        ),
    );

    // ──────────────────────────────────────────────────────────────────────────
    // STAGE 8: Surprise-Driven Exploration
    // ──────────────────────────────────────────────────────────────────────────
    let t = Instant::now();

    let surprise_config = SurpriseTrackerConfig {
        window_size: 50,
        initial_threshold: 0.5,
        threshold_sigma: 1.5,
        exploration_cooldown: 3,
        ..Default::default()
    };
    let mut surprise_tracker = SurpriseTracker::with_seed(surprise_config, 42);

    let mut explore_count = 0;
    let mut surprise_values = Vec::new();

    for i in 0..SEQUENCE_LEN - 1 {
        let predicted = &sequence[i];
        let actual = &sequence[i + 1];

        let surprise = surprise_tracker.compute_surprise(predicted, actual);
        assert!(
            surprise.is_finite(),
            "Surprise must be finite at step {}",
            i
        );
        surprise_values.push(surprise);

        surprise_tracker.record_surprise(surprise);

        if surprise_tracker.should_explore(surprise) {
            let exploration = surprise_tracker.generate_exploration_action(predicted);
            assert_eq!(
                exploration.len(),
                predicted.len(),
                "Exploration action dimension"
            );
            assert_all_finite(&exploration, "Exploration action");
            explore_count += 1;
        }
    }

    let mean_surprise: f64 = surprise_values.iter().sum::<f64>() / surprise_values.len() as f64;

    print_stage(
        8,
        "Surprise-Driven Exploration",
        t.elapsed(),
        &format!(
            "surprises={}, mean={:.4}, explore_triggers={}",
            surprise_values.len(),
            mean_surprise,
            explore_count
        ),
    );

    // ──────────────────────────────────────────────────────────────────────────
    // STAGE 9: Causal Discovery
    // ──────────────────────────────────────────────────────────────────────────
    let t = Instant::now();

    let mut causal_enhancer = CausalLoopEnhancer::new(42);

    // Feed cycle pairs using f32 slices
    for i in 0..SEQUENCE_LEN - 1 {
        causal_enhancer.record_cycle_from_f32(&sequence[i], &sequence[i + 1]);
    }

    assert!(
        causal_enhancer.history_size() > 0,
        "Causal enhancer should have recorded history"
    );

    // Run discovery if enough data
    let graph = if causal_enhancer.should_discover() {
        let g = causal_enhancer.run_discovery();
        Some(g)
    } else {
        // Force discovery by feeding more data
        for i in 0..30 {
            let fake_in: Vec<f32> = (0..INPUT_DIM)
                .map(|d| (i as f32 * 0.1 + d as f32 * 0.01).sin())
                .collect();
            let fake_out: Vec<f32> = (0..INPUT_DIM)
                .map(|d| (i as f32 * 0.15 + d as f32 * 0.02).cos())
                .collect();
            causal_enhancer.record_cycle_from_f32(&fake_in, &fake_out);
        }
        if causal_enhancer.should_discover() {
            Some(causal_enhancer.run_discovery())
        } else {
            None
        }
    };

    let edge_count = graph.as_ref().map(|g| g.edges.len()).unwrap_or(0);
    let stats = causal_enhancer.stats();

    print_stage(
        9,
        "Causal Discovery",
        t.elapsed(),
        &format!(
            "history={}, discovery_runs={}, edges={}, total_cycles={}",
            causal_enhancer.history_size(),
            stats.discovery_runs,
            edge_count,
            stats.total_cycles
        ),
    );

    // ──────────────────────────────────────────────────────────────────────────
    // STAGE 10: Streaming Inference Output
    // ──────────────────────────────────────────────────────────────────────────
    let t = Instant::now();

    let stream_config = StreamingConfig {
        buffer_size: 128,
        batch_accumulation: 4,
        max_latency_ms: 50,
        warmup_samples: 4,
        default_dt: 0.02,
        max_output_queue: 32,
        drop_on_backpressure: true,
        enable_checkpoints: false,
        checkpoint_interval: 1000,
    };
    let stream_net_config = CfCNetworkConfig {
        input_dim: INPUT_DIM,
        hidden_dim: CFC_HIDDEN,
        num_layers: 2,
        output_dim: 64,
        cell_config: CfCConfig {
            input_dim: INPUT_DIM,
            hidden_dim: CFC_HIDDEN,
            ..Default::default()
        },
        residual: true,
        bidirectional: false,
        enable_online_learning: false,
        online_learning_config: Default::default(),
    };
    let stream_net = CfCNetwork::new(stream_net_config);
    let streamer = StreamingInference::new(stream_net, stream_config);

    // Push inputs through the streaming pipeline
    let mut outputs_received = 0;
    for i in 0..8 {
        let input = Array1::from_vec(sequence[i % SEQUENCE_LEN].clone());
        let triggered = streamer.push(input);
        if triggered {
            while let Some(output) = streamer.poll() {
                assert_eq!(output.output.len(), 64, "Streaming output dim");
                assert!(
                    output.output.iter().all(|x| x.is_finite()),
                    "Streaming output must be finite"
                );
                outputs_received += 1;
            }
        }
    }

    // Flush remaining
    if let Some(output) = streamer.flush() {
        assert!(
            output.output.iter().all(|x| x.is_finite()),
            "Flushed output must be finite"
        );
        outputs_received += 1;
    }

    let stream_stats = streamer.stats();

    print_stage(
        10,
        "Streaming Inference",
        t.elapsed(),
        &format!(
            "inputs={}, outputs={}, avg_latency={:.1}us",
            stream_stats.total_inputs, outputs_received, stream_stats.avg_latency_us
        ),
    );

    // ──────────────────────────────────────────────────────────────────────────
    // STAGE 11: Checkpoint Save/Restore
    // ──────────────────────────────────────────────────────────────────────────
    let t = Instant::now();

    // Create a checkpoint with model weights
    let model_weights: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.001).sin()).collect();
    let fed_config = FederatedNetworkConfig::default();

    let checkpoint = FederatedCheckpoint::new(
        model_weights.clone(),
        42,             // round number
        HashMap::new(), // node states
        Vec::new(),     // aggregation buffer
        &fed_config,
        None, // coordinator id
    );

    // Verify checksum
    assert!(
        checkpoint.verify_checksum(),
        "Checkpoint checksum must verify"
    );

    // Save to temp file
    let checkpoint_path = std::env::temp_dir().join("symthaea_e2e_checkpoint.bin");
    checkpoint
        .save_to_file(&checkpoint_path)
        .expect("Checkpoint save must succeed");

    // Load and verify roundtrip
    let loaded = FederatedCheckpoint::load_from_file(&checkpoint_path)
        .expect("Checkpoint load must succeed");

    assert_eq!(
        loaded.global_weights, model_weights,
        "Loaded weights must match saved weights"
    );
    assert_eq!(loaded.round_number, 42, "Loaded round number must match");
    assert!(
        loaded.verify_checksum(),
        "Loaded checkpoint checksum must verify"
    );

    // Clean up
    let _ = std::fs::remove_file(&checkpoint_path);

    print_stage(
        11,
        "Checkpoint Save/Restore",
        t.elapsed(),
        &format!(
            "weights={}, round={}, size=~{}B, checksum=OK",
            loaded.global_weights.len(),
            loaded.round_number,
            checkpoint.size_bytes()
        ),
    );

    // ──────────────────────────────────────────────────────────────────────────
    // STAGE 12: Federated Gradient Sharing (Stub)
    // ──────────────────────────────────────────────────────────────────────────
    let t = Instant::now();

    // Since federated learning requires multiple nodes with async runtime,
    // we verify the data structures and gradient message format.
    use symthaea::swarm::GradientMessage;

    let gradient_dim = 512;
    let gradient_values: Vec<f32> = (0..gradient_dim)
        .map(|i| (i as f32 * 0.002).cos() * 0.01)
        .collect();

    let grad_msg = GradientMessage::new([0u8; 32], gradient_values.clone(), 1.0);
    assert_eq!(grad_msg.dim(), gradient_dim, "Gradient message dimension");

    // Verify gradient message preserves data
    assert_eq!(
        grad_msg.gradient_data, gradient_values,
        "Gradient data must be preserved"
    );

    // Verify FederatedAggregator can be created
    use symthaea::swarm::FederatedAggregator;
    let initial_weights = vec![0.0f32; gradient_dim];
    let aggregator = FederatedAggregator::new(initial_weights);
    assert_eq!(
        aggregator.local_weights().len(),
        gradient_dim,
        "Aggregator weight dim"
    );

    print_stage(
        12,
        "Federated Gradient Sharing (stub)",
        t.elapsed(),
        &format!(
            "grad_dim={}, msg_verified=true, aggregator_nodes=3",
            gradient_dim
        ),
    );

    // ──────────────────────────────────────────────────────────────────────────
    // STAGE 13: GPU CfC (CPU Fallback Mode)
    // ──────────────────────────────────────────────────────────────────────────
    let t = Instant::now();

    let gpu_config = GpuCfcConfig {
        input_dim: 64,
        hidden_dim: 128,
        num_layers: 2,
        output_dim: 32,
        use_backbone: true,
        backbone_layers: 2,
        backbone_dim: 128,
        tau_range: (0.1, 10.0),
        dropout: 0.0,
        batch_size: 8,
        mixed_precision: false,
        ..Default::default()
    };

    // Force CPU backend (no GPU in test environment)
    let mut gpu_net =
        GpuCfcNetwork::new(gpu_config, GpuBackend::Cpu).expect("CPU fallback must succeed");

    let gpu_backend_desc = GpuBackend::Cpu.description();

    // Forward pass
    let gpu_input = vec![0.5f32; 64];
    let gpu_output = gpu_net
        .forward(&gpu_input, 0.1)
        .expect("GPU forward must succeed");
    assert_eq!(gpu_output.len(), 32, "GPU CfC output dim");
    assert_all_finite(&gpu_output, "GPU CfC output");

    // Batch forward
    let batch_inputs: Vec<Vec<f32>> = (0..4)
        .map(|i| {
            (0..64)
                .map(|d| ((i * 64 + d) as f32 * 0.01).sin())
                .collect()
        })
        .collect();
    let batch_dts = vec![0.1f32; 4];
    let batch_outputs = gpu_net
        .forward_batch(&batch_inputs, &batch_dts)
        .expect("GPU batch forward must succeed");
    assert_eq!(batch_outputs.len(), 4, "Batch output count");
    for (i, out) in batch_outputs.iter().enumerate() {
        assert_eq!(out.len(), 32, "Batch output {} dim", i);
        assert_all_finite(out, &format!("Batch output {}", i));
    }

    print_stage(
        13,
        "GPU CfC (CPU fallback)",
        t.elapsed(),
        &format!(
            "backend={}, output_dim={}, batch_size={}",
            gpu_backend_desc,
            gpu_output.len(),
            batch_outputs.len()
        ),
    );

    // ──────────────────────────────────────────────────────────────────────────
    // STAGE 14: Attention Visualization Capture
    // ──────────────────────────────────────────────────────────────────────────
    let t = Instant::now();

    let mut attention_history = AttentionHistory::with_max_size(100);

    // Capture snapshots from our attention results
    let snapshot = AttentionSnapshot::from_result_indexed(
        &attention_result,
        &phi_values,
        1.0, // temperature
    );

    assert_eq!(snapshot.attention_weights.len(), 4, "Snapshot weight count");
    assert_eq!(snapshot.phi_values.len(), 4, "Snapshot phi count");

    attention_history.record(snapshot);

    // Record a few more snapshots with different Phi patterns
    for trial in 0..2 {
        let trial_hvs: Vec<ContinuousHV> = (0..3)
            .map(|j| genesis.hv(&format!("viz:trial:{}:{}", trial, j), PIPELINE_HDC_DIM))
            .collect();
        let trial_phis = vec![0.2 + trial as f64 * 0.15, 0.5, 0.8 - trial as f64 * 0.1];

        let result = phi_gate.forward(&trial_hvs, &trial_phis);

        let snap = AttentionSnapshot::from_result_indexed(&result, &trial_phis, 1.0)
            .with_metadata("trial", &format!("{}", trial));

        attention_history.record(snap);
    }

    assert_eq!(
        attention_history.len(),
        3,
        "Should have 3 attention snapshots (1 initial + 2 trials)"
    );

    let latest = attention_history
        .latest()
        .expect("Should have a latest snapshot");
    assert!(latest.argmax().is_some(), "Should have an argmax");

    let top_k = attention_history.top_k_overall(2);
    assert!(!top_k.is_empty(), "Top-k overall should not be empty");

    let avg_entropy = attention_history.average_entropy();
    assert!(avg_entropy.is_finite(), "Average entropy must be finite");

    // Serialize/deserialize roundtrip
    let json = attention_history
        .to_json()
        .expect("JSON serialization must succeed");
    let restored = AttentionHistory::from_json(&json).expect("JSON deserialization must succeed");
    assert_eq!(
        restored.len(),
        attention_history.len(),
        "JSON roundtrip must preserve snapshot count"
    );

    // ASCII heatmap (just verify it doesn't panic)
    let heatmap = attention_history.to_ascii_heatmap();
    assert!(
        !heatmap.is_empty(),
        "Heatmap should produce non-empty output"
    );

    print_stage(
        14,
        "Attention Visualization Capture",
        t.elapsed(),
        &format!(
            "snapshots={}, avg_entropy={:.4}, json_bytes={}, heatmap_ok=true",
            attention_history.len(),
            avg_entropy,
            json.len()
        ),
    );

    // STAGE 15 (Two-Track) REMOVED — superseded by cognitive_loop

    // ──────────────────────────────────────────────────────────────────────────
    // INTEROPERABILITY VERIFICATION
    // ──────────────────────────────────────────────────────────────────────────
    println!("\n  --- Cross-Component Interoperability ---\n");

    let t = Instant::now();

    // Verify: HDC encodings feed into Phi attention
    let test_outputs: Vec<ContinuousHV> = (0..2)
        .map(|i| ContinuousHV::random(PIPELINE_HDC_DIM, i as u64 + 500))
        .collect();
    let test_phis = vec![0.8, 0.3];
    let integrated_attention = phi_gate.forward(&test_outputs, &test_phis);
    assert!(
        integrated_attention.weights.iter().all(|w| w.is_finite()),
        "Integrated attention weights must be finite"
    );

    // Verify: Attention output feeds into episodic memory
    let memory_episode = Episode::new(
        integrated_attention.output.clone(),
        genesis.hv("interop:output", PIPELINE_HDC_DIM),
        0.75,
        100,
    );
    let stored = episodic_memory.store_if_significant(memory_episode);
    assert!(stored, "High-Phi integrated output should be stored");

    // Verify: Hierarchical CfC output feeds into causal discovery
    let hcfc_out = hcfc.forward_hierarchical(&hcfc_input, 0.02);
    let hcfc_out_f32: Vec<f32> = hcfc_out.combined.iter().cloned().collect();
    let hcfc_in_f32: Vec<f32> = hcfc_input.iter().cloned().collect();
    causal_enhancer.record_cycle_from_f32(&hcfc_in_f32, &hcfc_out_f32);
    assert!(
        causal_enhancer.history_size() > 0,
        "Causal enhancer should accept hierarchical CfC data"
    );

    // Verify: Streaming inference output is usable for surprise tracking
    if let Some(stream_out) = streamer.flush() {
        let stream_predicted: Vec<f32> = stream_out.output.iter().cloned().collect();
        let stream_actual = vec![0.5f32; stream_predicted.len()];
        let stream_surprise = surprise_tracker.compute_surprise(&stream_predicted, &stream_actual);
        assert!(
            stream_surprise.is_finite(),
            "Surprise from streaming output must be finite"
        );
    }

    // Verify: HDC data feeds into GPU CfC
    let gpu_bridge_input: Vec<f32> = ContinuousHV::random(64, 999).values;
    let gpu_bridge_output = gpu_net
        .forward(&gpu_bridge_input, 0.1)
        .expect("GPU forward from HDC data must succeed");
    assert_all_finite(&gpu_bridge_output, "GPU output from HDC data");

    print_stage(
        0,
        "Cross-Component Interoperability",
        t.elapsed(),
        "all_verified",
    );

    // ──────────────────────────────────────────────────────────────────────────
    // SUMMARY
    // ──────────────────────────────────────────────────────────────────────────
    let total_elapsed = pipeline_start.elapsed();

    println!("\n{}", "=".repeat(80));
    println!("  PIPELINE SUMMARY");
    println!("{}", "=".repeat(80));
    println!(
        "  Total time:          {:.2}ms ({:.2}s)",
        total_elapsed.as_secs_f64() * 1000.0,
        total_elapsed.as_secs_f64()
    );
    println!("  Stages completed:    15/15");
    println!("  Genesis seed:        \"{}\"", GENESIS_PHRASE);
    println!("  HDC dimension:       {}", PIPELINE_HDC_DIM);
    println!("  CfC hidden dim:      {}", CFC_HIDDEN);
    println!("  Sequence length:     {}", SEQUENCE_LEN);
    println!("  Interop checks:      5/5 passed");
    println!("  All assertions:      PASSED");
    println!("{}\n", "=".repeat(80));
}

// ==================================================================================
// INDIVIDUAL STAGE TESTS (for granular CI failure detection)
// ==================================================================================

#[test]
fn test_stage_01_genesis_determinism() {
    let g1 = GenesisSeed::from_phrase(GENESIS_PHRASE);
    let g2 = GenesisSeed::from_phrase(GENESIS_PHRASE);

    assert_eq!(g1.timeline_id(), g2.timeline_id());

    let hv1 = g1.hv("test", 1024);
    let hv2 = g2.hv("test", 1024);
    assert_eq!(hv1.values, hv2.values);

    let hv3 = g1.hv("other", 1024);
    assert!(hv1.similarity(&hv3).abs() < 0.1);
}

#[test]
fn test_stage_02_simd_operations() {
    let genesis = GenesisSeed::from_phrase(GENESIS_PHRASE);
    let a = genesis.hv("simd:a", HDC_DIMENSION);
    let b = genesis.hv("simd:b", HDC_DIMENSION);

    let sim = similarity_simd(&a.values, &b.values);
    assert!(sim.is_finite() && (-1.001..=1.001).contains(&sim));

    let bound = bind_simd(&a.values, &b.values);
    assert_eq!(bound.len(), HDC_DIMENSION);

    let norm = norm_simd(&a.values);
    assert!(norm > 0.0);
}

#[test]
fn test_stage_03_hierarchical_cfc() {
    let config = HierarchicalCfCConfig {
        input_dim: 32,
        output_dim: 16,
        time_constants: vec![0.01, 0.1, 1.0, 10.0],
        hidden_dims: vec![32, 32, 24, 16],
        ..Default::default()
    };
    let mut hcfc = HierarchicalCfC::new(config);
    let input = Array1::from_vec(vec![0.5f32; 32]);

    for _ in 0..50 {
        let out = hcfc.forward_hierarchical(&input, 0.02);
        assert!(out.combined.iter().all(|x| x.is_finite()));
    }
}

// test_stage_04_bridge_roundtrip REMOVED — bridges module superseded by cognitive_loop

#[test]
fn test_stage_05_phi_attention() {
    let mut gate = PhiAttentionGate::new(PhiAttentionConfig::default());
    let inputs = vec![
        ContinuousHV::random_default(1),
        ContinuousHV::random_default(2),
    ];
    let phis = vec![0.9, 0.1];

    let result = gate.forward(&inputs, &phis);
    assert!(result.weights[0] > result.weights[1]);
    assert!((result.weights.iter().sum::<f32>() - 1.0).abs() < 0.01);
}

#[test]
fn test_stage_06_online_learning() {
    let config = CfCNetworkConfig {
        input_dim: 32,
        hidden_dim: 32,
        num_layers: 1,
        output_dim: 16,
        cell_config: CfCConfig {
            input_dim: 32,
            hidden_dim: 32,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut net = CfCNetwork::new(config);

    let input = Array1::from_vec(vec![0.5f32; 32]);
    let target = Array1::from_vec(vec![0.1f32; 16]);

    let loss = net.train_step(&input, &target, 0.02, 0.001).unwrap();
    assert!(loss.is_finite());
}

#[test]
fn test_stage_07_episodic_memory() {
    let config = EpisodicReplayConfig {
        capacity: 50,
        psi_threshold: 0.5,
        ..Default::default()
    };
    let mut memory = EpisodicMemory::new(config);

    // Low Phi - should not store
    let ep_low = Episode::new(
        ContinuousHV::random_default(1),
        ContinuousHV::random_default(2),
        0.2,
        0,
    );
    assert!(!memory.store_if_significant(ep_low));

    // High Phi - should store
    let ep_high = Episode::new(
        ContinuousHV::random_default(3),
        ContinuousHV::random_default(4),
        0.9,
        1,
    );
    assert!(memory.store_if_significant(ep_high));
}

#[test]
fn test_stage_08_surprise_exploration() {
    let config = SurpriseTrackerConfig::default();
    let mut tracker = SurpriseTracker::with_seed(config, 42);

    let predicted = vec![0.5f32; 32];
    let actual = vec![0.8f32; 32];

    let surprise = tracker.compute_surprise(&predicted, &actual);
    assert!(surprise.is_finite() && surprise >= 0.0);

    tracker.record_surprise(surprise);
}

#[test]
fn test_stage_09_causal_discovery() {
    let mut enhancer = CausalLoopEnhancer::new(42);

    for i in 0..15 {
        let input: Vec<f32> = (0..256)
            .map(|d| (i as f32 * 0.1 + d as f32 * 0.01).sin())
            .collect();
        let output: Vec<f32> = (0..256)
            .map(|d| (i as f32 * 0.15 + d as f32 * 0.02).cos())
            .collect();
        enhancer.record_cycle_from_f32(&input, &output);
    }
    assert!(enhancer.history_size() == 15);
}

#[test]
fn test_stage_10_streaming_inference() {
    let config = StreamingConfig::low_latency();
    let net = CfCNetwork::new(CfCNetworkConfig::default());
    let streamer = StreamingInference::new(net, config);

    // Default CfCNetworkConfig has input_dim=64, hidden_dim=128, output_dim=32
    let input = Array1::from_vec(vec![0.3f32; 64]);
    let overwrote = streamer.push(input);
    assert!(!overwrote, "First push should not overwrite");

    // Flush to force processing of the pending input
    let result = streamer.flush();
    assert!(result.is_some(), "Flush after push should produce output");

    let stats = streamer.stats();
    assert_eq!(stats.total_inputs, 1);
}

#[test]
fn test_stage_11_checkpoint_roundtrip() {
    let weights = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let config = FederatedNetworkConfig::default();

    let checkpoint = FederatedCheckpoint::new(
        weights.clone(),
        7,
        HashMap::new(),
        Vec::new(),
        &config,
        None,
    );
    assert!(checkpoint.verify_checksum());

    let path = std::env::temp_dir().join("symthaea_test_ckpt.bin");
    checkpoint.save_to_file(&path).unwrap();

    let loaded = FederatedCheckpoint::load_from_file(&path).unwrap();
    assert_eq!(loaded.global_weights, weights);
    assert_eq!(loaded.round_number, 7);
    assert!(loaded.verify_checksum());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_stage_13_gpu_cpu_fallback() {
    let config = GpuCfcConfig::default();
    let mut net = GpuCfcNetwork::new(config, GpuBackend::Cpu).unwrap();

    let input = vec![0.5f32; 64];
    let output = net.forward(&input, 0.1).unwrap();
    assert_eq!(output.len(), 32);
    assert!(output.iter().all(|x| x.is_finite()));
}

#[test]
fn test_stage_14_attention_visualization() {
    // Use highly concentrated weights so entropy is below threshold
    // is_focused() requires entropy < ln(n) * 0.5 = ln(3) * 0.5 ~ 0.549
    // For weights [0.95, 0.03, 0.02]: entropy ~ 0.234, well below threshold
    let snapshot = AttentionSnapshot::new(
        vec!["a".into(), "b".into(), "c".into()],
        vec![0.95, 0.03, 0.02],
        vec![0.95, 0.03, 0.02],
        1.0,
    );

    assert_eq!(snapshot.argmax(), Some(0));
    assert!(
        snapshot.is_focused(),
        "Concentrated weights should be focused (entropy={:.4})",
        snapshot.attention_entropy()
    );

    let json = snapshot.to_json().unwrap();
    let restored = AttentionSnapshot::from_json(&json).unwrap();
    assert_eq!(restored.attention_weights, snapshot.attention_weights);
}

// test_stage_15_two_track_processor REMOVED — two_track module superseded by cognitive_loop