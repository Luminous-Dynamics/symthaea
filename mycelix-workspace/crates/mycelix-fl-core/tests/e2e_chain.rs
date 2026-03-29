// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-End Integration Test: Symthaea → Core → SDK → Holochain
//!
//! Proves the complete FL chain works by simulating each conversion step:
//!
//! 1. Symthaea `GradientMessage` (f32, [u8;32] ID) → `ReputationGradient`
//! 2. → core `GradientUpdate` (f32, String ID) → `UnifiedPipeline`
//! 3. → `AggregatedGradient` (f32)
//! 4. → SDK f64 wrapper conversion
//! 5. → Holochain bridge `ZomeTrainingRound` / `ZomeByzantineRecord`
//!
//! Run: `cargo test --test e2e_chain`

use std::collections::HashMap;

use mycelix_fl_core::consciousness_plugin::ConsciousnessAwareByzantinePlugin;
use mycelix_fl_core::meta_learning::MetaLearningByzantinePlugin;
use mycelix_fl_core::pipeline::{ExternalWeightMap, ParticipantWeightAdjustment};
use mycelix_fl_core::plugins::{ByzantinePlugin, PipelinePlugins};
use mycelix_fl_core::{
    convert::{aggregated_to_f64, gradients_to_f32, gradients_to_f64, update_to_f32},
    AggregatedGradient, AggregationMethod, GradientUpdate, PipelineConfig, UnifiedPipeline,
};

/// Simulates Symthaea's `GradientMessage` (the f32 + [u8;32] ID format).
/// In real Symthaea: `swarm::federated_cfc::GradientMessage`.
struct SimulatedGradientMessage {
    sender_id: [u8; 32],
    model_version: u64,
    gradients: Vec<f32>,
    batch_size: u32,
    loss: f32,
}

/// Simulates Symthaea's `ReputationGradient` (wraps GradientUpdate + reputation).
/// In real Symthaea: `hybrid_bft::ReputationGradient`.
struct SimulatedReputationGradient {
    update: GradientUpdate,
    reputation: f32,
}

/// Convert [u8;32] sender ID to hex string (what Symthaea does for core compat).
fn sender_id_to_string(id: &[u8; 32]) -> String {
    id.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Stage 1: Symthaea GradientMessage → ReputationGradient → core GradientUpdate
fn stage1_symthaea_to_core(
    messages: &[SimulatedGradientMessage],
    node_reputations: &HashMap<[u8; 32], f32>,
) -> (Vec<GradientUpdate>, HashMap<String, f32>) {
    let mut updates = Vec::new();
    let mut reps = HashMap::new();

    for msg in messages {
        let pid = sender_id_to_string(&msg.sender_id);
        let rep = node_reputations.get(&msg.sender_id).copied().unwrap_or(0.5);

        let update = GradientUpdate::new(
            pid.clone(),
            msg.model_version,
            msg.gradients.clone(),
            msg.batch_size,
            msg.loss,
        );
        updates.push(update);
        reps.insert(pid, rep);
    }

    (updates, reps)
}

/// Stage 2: Run through core UnifiedPipeline
fn stage2_pipeline(
    updates: &[GradientUpdate],
    reps: &HashMap<String, f32>,
    config: PipelineConfig,
) -> mycelix_fl_core::pipeline::PipelineResult {
    let mut pipeline = UnifiedPipeline::new(config);
    pipeline
        .aggregate(updates, reps)
        .expect("Pipeline should succeed")
}

/// Stage 3: SDK f64 conversion (simulates UnifiedPipelineF64)
fn stage3_sdk_f64_conversion(result: &AggregatedGradient) -> Vec<f64> {
    aggregated_to_f64(result)
}

/// Stage 4: SDK → creates inputs from f64 API (simulates SDK client)
fn stage4_sdk_creates_updates(
    gradients_f64: &[Vec<f64>],
    participant_ids: &[String],
    model_version: u64,
) -> Vec<GradientUpdate> {
    gradients_f64
        .iter()
        .zip(participant_ids.iter())
        .map(|(g, pid)| update_to_f32(pid.clone(), model_version, g, 100, 0.5, None, 0))
        .collect()
}

// === Tests ===

#[test]
fn test_e2e_stage1_symthaea_conversion() {
    let mut id = [0u8; 32];
    id[0] = 0xAB;
    id[1] = 0xCD;

    let msg = SimulatedGradientMessage {
        sender_id: id,
        model_version: 1,
        gradients: vec![0.1, 0.2, 0.3],
        batch_size: 100,
        loss: 0.5,
    };

    let mut node_reps = HashMap::new();
    node_reps.insert(id, 0.9);

    let (updates, reps) = stage1_symthaea_to_core(&[msg], &node_reps);

    assert_eq!(updates.len(), 1);
    assert_eq!(updates[0].gradients, vec![0.1, 0.2, 0.3]);
    let pid = &updates[0].participant_id;
    assert!(
        pid.starts_with("abcd"),
        "Hex ID should start with abcd, got {}",
        pid
    );
    assert_eq!(*reps.get(pid).unwrap(), 0.9);
}

#[test]
fn test_e2e_full_chain_honest() {
    // Stage 1: Create Symthaea-style messages
    let n_nodes = 5;
    let dim = 20;
    let mut messages = Vec::new();
    let mut node_reps = HashMap::new();

    for i in 0..n_nodes {
        let mut id = [0u8; 32];
        id[0] = i as u8;
        let gradients: Vec<f32> = (0..dim)
            .map(|d| 0.1 + d as f32 * 0.01 + i as f32 * 0.001)
            .collect();
        messages.push(SimulatedGradientMessage {
            sender_id: id,
            model_version: 1,
            gradients,
            batch_size: 100 + i as u32 * 10,
            loss: 0.5 - i as f32 * 0.01,
        });
        node_reps.insert(id, 0.85 + i as f32 * 0.01);
    }

    // Stage 1: Symthaea → Core types
    let (updates, reps) = stage1_symthaea_to_core(&messages, &node_reps);
    assert_eq!(updates.len(), n_nodes);
    assert_eq!(reps.len(), n_nodes);

    // Stage 2: Core pipeline
    let config = PipelineConfig::default();
    let result = stage2_pipeline(&updates, &reps, config);
    assert_eq!(result.aggregated.gradients.len(), dim);
    assert!(result.aggregated.participant_count > 0);

    // Verify aggregated values are near the honest mean
    for val in &result.aggregated.gradients {
        assert!(val.is_finite());
        assert!((*val - 0.1).abs() < 1.0, "Unexpected value: {}", val);
    }

    // Stage 3: SDK f64 conversion
    let f64_result = stage3_sdk_f64_conversion(&result.aggregated);
    assert_eq!(f64_result.len(), dim);
    for (f32_val, f64_val) in result.aggregated.gradients.iter().zip(f64_result.iter()) {
        let error = (*f32_val as f64 - f64_val).abs();
        assert!(
            error < 1e-6,
            "f32→f64 precision loss too high: {} vs {}",
            f32_val,
            f64_val
        );
    }
}

#[test]
fn test_e2e_full_chain_with_byzantine() {
    let dim = 20;
    let mut messages = Vec::new();
    let mut node_reps = HashMap::new();

    // 7 honest nodes
    for i in 0..7 {
        let mut id = [0u8; 32];
        id[0] = i as u8;
        let gradients: Vec<f32> = (0..dim).map(|d| 0.5 + d as f32 * 0.001).collect();
        messages.push(SimulatedGradientMessage {
            sender_id: id,
            model_version: 1,
            gradients,
            batch_size: 100,
            loss: 0.5,
        });
        node_reps.insert(id, 0.9);
    }

    // 3 Byzantine nodes (30%)
    for i in 0..3 {
        let mut id = [0u8; 32];
        id[0] = (100 + i) as u8;
        let gradients: Vec<f32> = (0..dim)
            .map(|_| if i % 2 == 0 { 100.0 } else { -100.0 })
            .collect();
        messages.push(SimulatedGradientMessage {
            sender_id: id,
            model_version: 1,
            gradients,
            batch_size: 100,
            loss: 0.5,
        });
        node_reps.insert(id, 0.15); // Low reputation → will be gated
    }

    let (updates, reps) = stage1_symthaea_to_core(&messages, &node_reps);
    let result = stage2_pipeline(&updates, &reps, PipelineConfig::default());

    // Byzantine should be gated out (rep 0.15 < min_rep 0.3)
    for val in &result.aggregated.gradients {
        assert!(
            (*val - 0.5).abs() < 1.0,
            "Byzantine influence leaked: {}",
            val
        );
    }
}

#[test]
fn test_e2e_sdk_f64_roundtrip() {
    // Simulate SDK creating updates in f64, converting to f32 for core, getting f64 result
    let participant_ids: Vec<String> = (0..5).map(|i| format!("node_{}", i)).collect();
    let gradients_f64: Vec<Vec<f64>> = (0..5)
        .map(|i| {
            (0..10)
                .map(|d| 0.1 + d as f64 * 0.01 + i as f64 * 0.001)
                .collect()
        })
        .collect();

    // SDK converts to f32 for core
    let updates = stage4_sdk_creates_updates(&gradients_f64, &participant_ids, 1);
    assert_eq!(updates.len(), 5);
    for u in &updates {
        assert_eq!(u.gradients.len(), 10);
    }

    // Run through pipeline
    let mut reps = HashMap::new();
    for pid in &participant_ids {
        reps.insert(pid.clone(), 0.85);
    }
    let config = PipelineConfig {
        multi_signal_detection: false, // Disable for small input
        ..PipelineConfig::performance()
    };
    let result = stage2_pipeline(&updates, &reps, config);

    // Convert result back to f64
    let result_f64 = stage3_sdk_f64_conversion(&result.aggregated);

    // Verify precision: f64→f32→aggregate→f32→f64 should have < 1e-5 relative error per dim
    // compared to pure f64 aggregation
    let expected: Vec<f64> = {
        let mut avg = vec![0.0f64; 10];
        for g in &gradients_f64 {
            for (i, &v) in g.iter().enumerate() {
                avg[i] += v / 5.0;
            }
        }
        avg
    };

    for (i, (&exp, &actual)) in expected.iter().zip(result_f64.iter()).enumerate() {
        let rel_err = if exp.abs() > 1e-10 {
            (exp - actual).abs() / exp.abs()
        } else {
            (exp - actual).abs()
        };
        assert!(
            rel_err < 1e-4,
            "Roundtrip dim {}: expected {:.8e}, got {:.8e} (rel_err={:.2e})",
            i,
            exp,
            actual,
            rel_err
        );
    }
}

#[test]
fn test_e2e_with_meta_learning_plugin() {
    let dim = 15;
    let n_rounds = 5;

    let mut meta_plugin = MetaLearningByzantinePlugin::new();
    let config = PipelineConfig::adaptive();
    let mut pipeline = UnifiedPipeline::new(config);

    for round in 0..n_rounds {
        let mut updates = Vec::new();
        let mut reps = HashMap::new();

        // 6 honest
        for i in 0..6 {
            updates.push(GradientUpdate::new(
                format!("h{}", i),
                round as u64 + 1,
                vec![0.5; dim],
                100,
                0.5,
            ));
            reps.insert(format!("h{}", i), 0.9);
        }

        // 2 Byzantine
        for i in 0..2 {
            updates.push(GradientUpdate::new(
                format!("byz{}", i),
                round as u64 + 1,
                vec![if i == 0 { 100.0 } else { -100.0 }; dim],
                100,
                0.5,
            ));
            reps.insert(format!("byz{}", i), 0.15);
        }

        let mut plugins = PipelinePlugins {
            compression: None,
            byzantine: vec![&mut meta_plugin],
            verification: None,
        };

        let result = pipeline
            .aggregate_with_plugins(&updates, &reps, &mut plugins)
            .expect("Pipeline should succeed");

        // Result should be near honest mean
        for val in &result.result.aggregated.gradients {
            assert!(
                (*val - 0.5).abs() < 1.0,
                "Round {}: value {} too far from 0.5",
                round,
                val
            );
        }
    }

    // After 5 rounds, meta plugin should have tracked participants
    let byz_profile = meta_plugin.get_participant_profile("byz0");
    assert!(
        byz_profile.is_some(),
        "Should have tracked Byzantine participant"
    );

    let honest_profile = meta_plugin.get_participant_profile("h0");
    assert!(
        honest_profile.is_some(),
        "Should have tracked honest participant"
    );
}

#[test]
fn test_e2e_with_consciousness_plugin() {
    let dim = 15;
    let mut consciousness_plugin = ConsciousnessAwareByzantinePlugin::new();

    // Set consciousness scores: honest nodes have high score, Byzantine have very low
    let mut consciousness_scores = HashMap::new();
    for i in 0..8 {
        consciousness_scores.insert(format!("h{}", i), 0.7);
    }
    for i in 0..2 {
        consciousness_scores.insert(format!("byz{}", i), 0.05); // Below veto threshold
    }
    consciousness_plugin.set_consciousness_scores(consciousness_scores);

    let config = PipelineConfig::adaptive();
    let mut pipeline = UnifiedPipeline::new(config);

    let mut updates = Vec::new();
    let mut reps = HashMap::new();

    // 8 honest
    for i in 0..8 {
        updates.push(GradientUpdate::new(
            format!("h{}", i),
            1,
            vec![0.5; dim],
            100,
            0.5,
        ));
        reps.insert(format!("h{}", i), 0.9);
    }

    // 2 Byzantine (extreme gradients)
    for i in 0..2 {
        updates.push(GradientUpdate::new(
            format!("byz{}", i),
            1,
            vec![if i == 0 { 100.0 } else { -100.0 }; dim],
            100,
            0.5,
        ));
        reps.insert(format!("byz{}", i), 0.5); // Medium reputation — consciousness gating must catch them
    }

    let mut plugins = PipelinePlugins {
        compression: None,
        byzantine: vec![&mut consciousness_plugin],
        verification: None,
    };

    let result = pipeline
        .aggregate_with_plugins(&updates, &reps, &mut plugins)
        .expect("Pipeline should succeed with consciousness plugin");

    // Byzantine nodes should be vetoed (phi=0.05 < veto_threshold=0.1)
    for val in &result.result.aggregated.gradients {
        assert!(
            (*val - 0.5).abs() < 1.0,
            "Byzantine influence should be suppressed: {}",
            val
        );
    }
}

#[test]
fn test_e2e_meta_learning_plus_consciousness() {
    let dim = 15;
    let mut meta_plugin = MetaLearningByzantinePlugin::new();
    let mut consciousness_plugin = ConsciousnessAwareByzantinePlugin::new();

    // Set phi scores
    let mut consciousness_scores = HashMap::new();
    for i in 0..6 {
        consciousness_scores.insert(format!("h{}", i), 0.8);
    }
    for i in 0..2 {
        consciousness_scores.insert(format!("byz{}", i), 0.15); // Low but above veto
    }
    consciousness_plugin.set_consciousness_scores(consciousness_scores);

    let config = PipelineConfig::adaptive();
    let mut pipeline = UnifiedPipeline::new(config);

    let mut updates = Vec::new();
    let mut reps = HashMap::new();

    // 6 honest
    for i in 0..6 {
        updates.push(GradientUpdate::new(
            format!("h{}", i),
            1,
            vec![0.5; dim],
            100,
            0.5,
        ));
        reps.insert(format!("h{}", i), 0.9);
    }

    // 2 Byzantine
    for i in 0..2 {
        updates.push(GradientUpdate::new(
            format!("byz{}", i),
            1,
            vec![if i == 0 { 50.0 } else { -50.0 }; dim],
            100,
            0.5,
        ));
        reps.insert(format!("byz{}", i), 0.5);
    }

    // Compose both plugins
    let mut plugins = PipelinePlugins {
        compression: None,
        byzantine: vec![&mut meta_plugin, &mut consciousness_plugin],
        verification: None,
    };

    let result = pipeline
        .aggregate_with_plugins(&updates, &reps, &mut plugins)
        .expect("Pipeline should succeed with composed plugins");

    // With both plugins active, Byzantine influence should be suppressed
    for val in &result.result.aggregated.gradients {
        assert!(
            (*val - 0.5).abs() < 5.0,
            "Composed plugins should suppress Byzantine influence: {}",
            val
        );
    }
}

#[cfg(feature = "holochain")]
mod holochain_tests {
    use super::*;
    use mycelix_fl_core::holochain_bridge::*;

    #[test]
    fn test_e2e_holochain_bridge_conversion() {
        // Run a pipeline round
        let mut updates = Vec::new();
        let mut reps = HashMap::new();
        for i in 0..5 {
            updates.push(GradientUpdate::new(
                format!("node_{}", i),
                1,
                vec![0.5; 10],
                100,
                0.5,
            ));
            reps.insert(format!("node_{}", i), 0.9);
        }

        let config = PipelineConfig::default();
        let result = stage2_pipeline(&updates, &reps, config);

        // Convert to Holochain zome types
        let training_round = ZomeTrainingRound::from_pipeline_result(&result, 1, &updates);
        assert_eq!(training_round.round_number, 1);
        assert_eq!(training_round.model_version, 1);
        assert_eq!(
            training_round.participant_count as usize,
            result.aggregated.participant_count
        );

        // Convert individual updates to gradient entries
        for update in &updates {
            let zome_gradient = ZomeModelGradient::from_update(update);
            assert_eq!(zome_gradient.participant_id, update.participant_id);
            assert_eq!(zome_gradient.model_version, 1);
            assert!(!zome_gradient.gradient_hash.is_empty());
        }

        // Round-trip through JSON
        let json = serde_json::to_string(&training_round).unwrap();
        let parsed: ZomeTrainingRound = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.round_number, training_round.round_number);
        assert_eq!(parsed.participant_count, training_round.participant_count);
    }
}

#[test]
fn test_e2e_deterministic() {
    // Same inputs should produce identical outputs across runs
    let make_updates = || -> (Vec<GradientUpdate>, HashMap<String, f32>) {
        let mut updates = Vec::new();
        let mut reps = HashMap::new();
        for i in 0..5 {
            updates.push(GradientUpdate::new(
                format!("n{}", i),
                1,
                vec![0.1 * (i as f32 + 1.0); 10],
                100,
                0.5,
            ));
            reps.insert(format!("n{}", i), 0.85);
        }
        (updates, reps)
    };

    let (u1, r1) = make_updates();
    let (u2, r2) = make_updates();

    let config = PipelineConfig {
        dp_config: None, // DP is stochastic
        multi_signal_detection: false,
        ..PipelineConfig::performance()
    };

    let result1 = stage2_pipeline(&u1, &r1, config.clone());
    let result2 = stage2_pipeline(&u2, &r2, config);

    assert_eq!(
        result1.aggregated.gradients, result2.aggregated.gradients,
        "Same inputs should produce identical outputs"
    );
    assert_eq!(
        result1.aggregated.participant_count,
        result2.aggregated.participant_count
    );
}
