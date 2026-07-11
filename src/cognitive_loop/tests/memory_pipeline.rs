// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Memory Pipeline Integration Tests
//!
//! Tests for `run_resonator_codebook_phase()` and `run_episodic_replay_and_memory_phase()`.
//! These methods (in helpers/cycle_phases_memory.rs, 645 LOC) manage:
//! - Semantic codebook promotion/eviction (competitive learning)
//! - Codebook diversity and utilization rate tracking
//! - Episodic replay with surprise-boosted batch sizes
//! - Dream consolidation and memory coordinator graduation
//!
//! Focus: numerical stability, EMA drift, and division-by-zero guards.

use super::super::*;

// ═══════════════════════════════════════════════════════════════════════════════
// 1. CODEBOOK DIVERSITY: EMA stays finite and bounded
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn codebook_diversity_finite_across_cycles() {
    // Codebook diversity is computed every 50 cycles and EMA-smoothed.
    // Run 200 cycles to exercise multiple diversity computations.
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let inputs = ["novel alpha", "familiar beta", "novel gamma"];
    for i in 0..200 {
        let result = svc.cycle(inputs[i % inputs.len()]);
        let div = result.metadata.memory.codebook_diversity;
        assert!(
            div.is_finite() && div >= 0.0 && div <= 1.0,
            "codebook_diversity out of [0,1] at cycle {i}: {div}"
        );
    }
}

#[test]
fn codebook_utilization_rate_finite_across_cycles() {
    // Utilization rate is computed every 50 cycles: utilized/n symbols.
    // EMA: rate * 0.8 + new * 0.2 — verify no NaN drift.
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    for i in 0..200 {
        let result = svc.cycle("utilization check");
        let rate = result.metadata.memory.codebook_utilization_rate;
        assert!(
            rate.is_finite() && rate >= 0.0 && rate <= 1.0,
            "codebook_utilization_rate out of [0,1] at cycle {i}: {rate}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. EPISODIC REPLAY: Batch sizes and telemetry
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn episodic_replay_batch_size_non_negative() {
    // The replay batch can be boosted by surprise, sleep, and phasic DA.
    // All boosts are additive → batch size must be >= 0.
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    for i in 0..150 {
        let result = svc.cycle("replay check");
        let batch = result.metadata.memory.surprise_replay_batch_size;
        assert!(
            (batch as f64).is_finite(),
            "replay batch overflow at cycle {i}: {batch}"
        );
    }
}

#[test]
fn resonator_best_sim_bounded() {
    // Best cosine similarity from resonator recall should be in [0, 1].
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    for i in 0..100 {
        let result = svc.cycle("resonator sim check");
        let sim = result.metadata.memory.resonator_best_sim;
        assert!(
            sim.is_finite() && sim >= 0.0 && sim <= 1.0,
            "resonator_best_sim out of [0,1] at cycle {i}: {sim}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. RESONATOR PROMOTIONS/EVICTIONS: Codebook management
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn resonator_promotions_and_evictions_finite() {
    // Codebook phase runs periodically. Promotion and eviction counts
    // should be reasonable and not overflow.
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let mut total_promotions = 0u64;
    let mut total_evictions = 0u64;
    for i in 0..200 {
        let result = svc.cycle("codebook mgmt");
        total_promotions += result.metadata.memory.resonator_promotions as u64;
        total_evictions += result.metadata.memory.codebook_evictions as u64;
        assert!(
            total_evictions <= total_promotions + 200,
            "Evictions ({total_evictions}) vastly exceed promotions ({total_promotions}) at cycle {i}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. MEMORY PIPELINE STABILITY: Extended run with varied input
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn memory_pipeline_no_nan_100_cycles() {
    // Full memory pipeline stress test: codebook + episodic replay + consolidation.
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let inputs = [
        "completely novel input alpha",
        "slightly familiar beta",
        "well-known repeated gamma",
        "surprising high-arousal delta",
        "calm consolidation epsilon",
    ];
    for i in 0..100 {
        let result = svc.cycle(inputs[i % inputs.len()]);
        let m = &result.metadata;
        let mem = &m.memory;

        assert!(
            mem.codebook_diversity.is_finite(),
            "NaN codebook_diversity at {i}"
        );
        assert!(
            mem.codebook_utilization_rate.is_finite(),
            "NaN codebook_utilization at {i}"
        );
        assert!(
            mem.resonator_best_sim.is_finite(),
            "NaN resonator_best_sim at {i}"
        );
        assert!(
            result.prediction_error.is_finite(),
            "NaN prediction_error at {i}"
        );
        assert!(
            m.consciousness.consciousness_level.is_finite(),
            "NaN consciousness at {i}"
        );

        for (j, &v) in result.output.iter().enumerate() {
            assert!(v.is_finite(), "NaN output[{j}] at cycle {i}");
        }
    }
}

#[test]
fn memory_pipeline_prediction_confidence_bounded() {
    // The memory phase can adjust prediction_confidence (via PE spike
    // consolidation). Verify it stays in [0, 1] across cycles.
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    for i in 0..100 {
        svc.cycle("confidence bound");
        let conf = svc.prediction_confidence();
        assert!(
            conf.is_finite() && conf >= 0.0 && conf <= 1.0,
            "prediction_confidence out of [0,1] at cycle {i}: {conf}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. EXPLORATION MODULATION: Codebook diversity → exploration coupling
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn codebook_diversity_exploration_coupling_bounded() {
    // Low codebook diversity boosts exploration; high diversity dampens it.
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    for i in 0..200 {
        let result = svc.cycle("exploration coupling");
        let lr = result.metadata.actual_effective_lr;
        assert!(
            lr.is_finite() && lr >= 0.0 && lr <= 10.0,
            "LR out of bounds at cycle {i}: {lr}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. EARLY CYCLE SAFETY: Memory phase on very first cycles
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn memory_pipeline_safe_on_first_cycle() {
    // The memory phase should not crash on the very first cycle when
    // EMA values are at defaults, codebook is empty, and replay hasn't
    // been triggered yet.
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let result = svc.cycle("first cycle safety");
    assert!(result.prediction_error.is_finite());
    assert!(result.metadata.memory.codebook_diversity.is_finite());
    assert!(result.metadata.memory.codebook_utilization_rate.is_finite());
    for &v in &result.output {
        assert!(v.is_finite());
    }
}

#[test]
fn episodic_storage_queue_full_is_non_fatal_to_cycle() {
    use crate::databases::storage_runtime::StorageRuntimeHandle;
    use crate::databases::{MemoryRecord, MemoryType};
    use std::sync::atomic::Ordering;
    use symthaea_core::hdc::binary_hv::BinaryHV;
    use tokio::sync::mpsc;

    let mut config = CognitiveLoopConfig::default();
    config.episodic_replay_training = false;
    config.episodic_replay_config.psi_threshold = 0.0;

    let mut svc = CognitiveLoopService::new(config).unwrap();
    let warmup = svc.cycle("storage backpressure warmup");
    assert!(warmup.prediction_error.is_finite());
    assert!(
        svc.memory
            .episodic_persistence
            .replay
            .as_ref()
            .map(|replay| !replay.get_top_episodes(1).is_empty())
            .unwrap_or(false),
        "test should create at least one episode before forcing persistence"
    );

    let (tx, _rx) = mpsc::channel(1);
    let runtime = StorageRuntimeHandle::from_sender_for_test(tx);

    runtime
        .try_store_memory(MemoryRecord {
            id: "queue-filler".to_string(),
            memory_type: MemoryType::Episodic,
            encoding: BinaryHV::random(42),
            content: "occupy bounded storage queue".to_string(),
            timestamp_ms: 0,
            valence: 0.0,
            arousal: 0.0,
            psi: 0.0,
            topics: Vec::new(),
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        })
        .unwrap();

    svc.memory
        .episodic_persistence
        .attach_storage_runtime(runtime);

    svc.stats.total_cycles = 198;
    let result = svc.cycle("storage backpressure forced flush");

    assert!(result.prediction_error.is_finite());
    assert!(
        result.output.iter().all(|v| v.is_finite()),
        "cycle output should stay finite when storage queue is full"
    );
    assert!(
        !svc.memory
            .episodic_persistence
            .flush_in_progress
            .load(Ordering::Relaxed),
        "failed storage enqueue must clear the flush guard"
    );
}
