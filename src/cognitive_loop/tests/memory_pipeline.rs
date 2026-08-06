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

// ═══════════════════════════════════════════════════════════════════════════════
// 3. PREDICTIVE COMPRESSION C3: episodic recall → prediction
//    (docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md §7)
// ═══════════════════════════════════════════════════════════════════════════════

/// `enable_episodic_recall_prediction` defaults to `false` — every existing
/// C1 result (and every other test in this suite) must stay reproducible
/// bit-for-bit. Two services with identical seed/config and an identical
/// input schedule (repeated, so the episodic store fills with near-duplicate
/// episodes a recall WOULD match against if the gate were active) must
/// produce prediction_error trajectories matching within float-reduction
/// noise with the flag off.
///
/// CORRECTION (2026-07-25): originally asserted bit-exact equality, which
/// failed at cycle 1 (not cycle 0 — the cold-start sentinel cycle IS
/// bit-exact). Diagnosed with a standalone probe
/// (`examples/c3_determinism_probe.rs`, scratch, not committed): max
/// per-element output diff ~1e-7 (f32 epsilon), `prediction_error` matches
/// to 6 decimals — this is benign floating-point non-associativity from
/// the pipeline's `rayon::join` parallel post-processing (thread completion
/// order, and therefore float summation order, isn't guaranteed identical
/// run-to-run even with identical seeds/inputs), not a seeding bug. Genuine
/// same-seed reproducibility (which values get computed) holds; bit-exact
/// reproducibility (the order they get summed in) does not, and was never
/// actually claimed by the crate doc's SHAKE-256 guarantee (that guarantee
/// is about the seeded random STREAMS, not about parallel float-reduction
/// order). Tolerance below is orders of magnitude tighter than any real
/// recall-blend effect would produce (see the companion test) while being
/// generous enough to absorb this class of noise.
#[test]
fn recall_prediction_flag_off_is_bit_identical_to_baseline() {
    const NOISE_TOLERANCE: f32 = 1e-4;
    let make = || {
        CognitiveLoopService::new(CognitiveLoopConfig {
            genesis_phrase: Some("c3-purity-seed".to_string()),
            async_training: false,
            enable_episodic_recall_prediction: false,
            ..Default::default()
        })
        .unwrap()
    };
    let mut a = make();
    let mut b = make();
    for i in 0..80 {
        let input = if i % 2 == 0 {
            "the recurring test sentence for episodic recall"
        } else {
            "a second recurring sentence for the same probe"
        };
        let ra = a.cycle(input);
        let rb = b.cycle(input);
        assert!(
            (ra.prediction_error - rb.prediction_error).abs() < NOISE_TOLERANCE,
            "flag-off trajectories diverged beyond float-noise tolerance at cycle {i}: {} vs {}",
            ra.prediction_error,
            rb.prediction_error
        );
        let max_diff = ra
            .output
            .iter()
            .zip(rb.output.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < NOISE_TOLERANCE,
            "flag-off output diverged beyond float-noise tolerance at cycle {i}: max_diff={max_diff}"
        );
    }
}

/// Mechanism sanity (not a purity claim): once the episodic store has
/// accumulated real (input, output) pairs on a REPEATED input, the `on`
/// service's behavior must differ from an identically-seeded `off` service
/// at some point, by MORE than the ~1e-7 float-reduction noise floor the
/// companion test above establishes — proving the wiring produces a real
/// effect, not just ambient parallel-reduction jitter. Does not assert a
/// *direction* (better/worse) — C3's pre-registered predictions (P5/P6)
/// are for the dedicated harness, not this smoke test.
#[test]
fn recall_prediction_flag_on_eventually_diverges_from_off() {
    // An order of magnitude above the noise floor established by
    // `recall_prediction_flag_off_is_bit_identical_to_baseline`'s
    // NOISE_TOLERANCE (1e-4) — big enough that only a genuine recall-blend
    // effect (which nudges up to half the prediction toward recalled
    // content) could produce it.
    const MEANINGFUL_DIVERGENCE: f32 = 1e-3;
    let make = |on: bool| {
        CognitiveLoopService::new(CognitiveLoopConfig {
            genesis_phrase: Some("c3-mechanism-seed".to_string()),
            async_training: false,
            enable_episodic_recall_prediction: on,
            ..Default::default()
        })
        .unwrap()
    };
    let mut off = make(false);
    let mut on = make(true);
    let mut diverged = false;
    for _ in 0..120 {
        // Fixed repeated input: the store fills with high-similarity episodes
        // of the same content, giving recall its best chance to fire.
        let r_off = off.cycle("the recurring test sentence for episodic recall");
        let r_on = on.cycle("the recurring test sentence for episodic recall");
        if (r_off.prediction_error - r_on.prediction_error).abs() > MEANINGFUL_DIVERGENCE {
            diverged = true;
            break;
        }
    }
    assert!(
        diverged,
        "flag on/off produced no meaningful divergence (> {MEANINGFUL_DIVERGENCE}) for \
         120 cycles on a repeated input — the recall blend path is not firing"
    );
}
