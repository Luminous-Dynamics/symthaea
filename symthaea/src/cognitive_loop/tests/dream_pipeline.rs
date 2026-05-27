// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dream Pipeline Tests
//!
//! Tests for `run_dream_phase()` (helpers/cycle_phases_dream.rs, 225 LOC).
//! This phase manages:
//! - Surprise event recording (Phi-weighted, narrative-boosted)
//! - Dream simulation during Cruise urgency
//! - Wisdom application → exploration/confidence modulation
//! - Dream feedback bridge for context-aware priors
//!
//! Focus: numerical stability, telemetry bounds, and first-cycle safety.

use super::super::*;

// ═══════════════════════════════════════════════════════════════════════════════
// 1. BASIC SAFETY: Dream phase should not panic on any cycle
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn dream_phase_safe_on_first_cycle() {
    // Dream engine may not exist on first cycle — run_dream_phase must handle None gracefully.
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let result = svc.cycle("dream first cycle");
    // Dream fields should be finite defaults
    assert!(result.prediction_error.is_finite());
    assert!(
        result
            .metadata
            .consciousness
            .consciousness_level
            .is_finite()
    );
}

#[test]
fn dream_phase_no_nan_100_cycles() {
    // Extended run to exercise dream recording, simulation, and wisdom accumulation.
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let inputs = [
        "novel surprise alpha",
        "calm steady beta",
        "surprising gamma burst",
        "familiar delta pattern",
        "emotional epsilon peak",
    ];
    for i in 0..100 {
        let result = svc.cycle(inputs[i % inputs.len()]);
        assert!(
            result.prediction_error.is_finite(),
            "NaN prediction_error at cycle {i}"
        );
        assert!(
            result
                .metadata
                .consciousness
                .consciousness_level
                .is_finite(),
            "NaN consciousness at cycle {i}"
        );
        for (j, &v) in result.output.iter().enumerate() {
            assert!(v.is_finite(), "NaN output[{j}] at cycle {i}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. EXPLORATION AND CONFIDENCE: Dream wisdom modulation stays bounded
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn dream_exploration_bounded_across_cycles() {
    // Dream wisdom can boost exploration via adjust_exploration("dream_wisdom", ...).
    // Verify the effective LR (which includes exploration) stays bounded.
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    for i in 0..150 {
        let result = svc.cycle("dream exploration");
        let lr = result.metadata.actual_effective_lr;
        assert!(
            lr.is_finite() && lr >= 0.0 && lr <= 10.0,
            "LR out of bounds at cycle {i}: {lr}"
        );
    }
}

#[test]
fn dream_confidence_bounded_across_cycles() {
    // Dream Phi insights can boost confidence via adjust_confidence("dream_phi_insight", ...).
    // Verify prediction_confidence stays in [0, 1].
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    for i in 0..150 {
        svc.cycle("dream confidence");
        let conf = svc.prediction_confidence();
        assert!(
            conf.is_finite() && conf >= 0.0 && conf <= 1.0,
            "prediction_confidence out of [0,1] at cycle {i}: {conf}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. LEARNING SIGNAL: Dream Phi improvement modulates FEP learning
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn dream_learning_signal_finite() {
    // Dream insights multiply learning_signal by 1.0 + (phi_improvement * 0.2).min(0.15).
    // Verify the FEP telemetry stays finite across cycles.
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    for i in 0..100 {
        let result = svc.cycle("dream learning signal");
        let fep_surprise = result.metadata.fep.fep_surprise;
        assert!(
            fep_surprise.is_finite(),
            "FEP surprise non-finite at cycle {i}: {fep_surprise}"
        );
        let td_err = result.metadata.fep.fep_td_error;
        assert!(
            td_err.is_finite(),
            "FEP TD error non-finite at cycle {i}: {td_err}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. URGENCY INTERACTION: Dream runs more during Cruise
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn dream_runs_without_panic_across_urgency_transitions() {
    // Rapidly changing inputs create urgency transitions (Normal→Cruise→Critical).
    // Dream phase adapts its interval. Verify no panics across transitions.
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    for i in 0..200 {
        // Alternate between high-error and low-error inputs to create urgency swings
        let input = if i % 10 < 3 {
            "CRISIS ALERT EMERGENCY" // high arousal → higher urgency
        } else {
            "steady calm peaceful" // low arousal → Cruise
        };
        let result = svc.cycle(input);
        assert!(
            result.prediction_error.is_finite(),
            "Non-finite PE during urgency transition at cycle {i}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. CONSOLIDATION PRESSURE: Memory load + consolidation pressure interaction
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn dream_interval_dynamic_with_consolidation_pressure() {
    // The dream interval adapts based on consolidation_pressure + memory_load.
    // Run enough cycles for consolidation to build up and verify stability.
    let mut svc = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    for i in 0..300 {
        let result = svc.cycle("consolidation pressure test");
        // The key assertion: even with pressure-driven interval changes,
        // the system produces finite outputs.
        assert!(
            result.metadata.actual_effective_lr.is_finite(),
            "Non-finite LR under consolidation pressure at cycle {i}"
        );
    }
}
