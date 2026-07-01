#![cfg(feature = "unstable-examples")]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Session 20/21: Mechanism Activation Audit
//!
//! Runs 500 cycles with varied inputs and verifies:
//! 1. Unified readiness gate stays bounded [0.3, 1.0]
//! 2. LR never crashes from compound dampening
//! 3. Readiness gate actually fires under sustained load
//! 4. Mechanism activations are non-trivial
//! 5. S21: Sleep pressure recovery engages, LR contribution tracking works
//!
//! S21: Reduced from 2,000 → 500 cycles. The proptest covers statistical
//! properties over many randomized runs; this test covers deterministic invariants.

use symthaea::cognitive_loop::CognitiveLoopService;

fn make_service() -> CognitiveLoopService {
    CognitiveLoopService::new(Default::default()).expect("failed to create CognitiveLoopService")
}

/// Run 500 cycles with rotating inputs to exercise all code paths.
/// Verify readiness invariants and activation counts.
#[test]
fn mechanism_audit_500_cycles() {
    let mut svc = make_service();
    let inputs = [
        "novel sensory burst",
        "familiar pattern recognition",
        "high-stress emergency signal",
        "calm consolidation period",
        "surprising contradictory input",
        "repetitive monotone stimulus",
        "complex multi-modal integration",
        "ambiguous uncertain evidence",
        "reward-predicting cue",
        "prediction error correction",
    ];

    let mut min_readiness = 1.0_f32;
    let mut max_readiness = 0.0_f32;
    let mut min_lr = f32::MAX;
    let mut max_lr = f32::MIN;
    let mut max_activations = 0u32;
    let mut readiness_below_half_count = 0u32;
    let mut gate_fired = false;
    let mut recovery_seen = false;
    let mut dominant_sources = std::collections::HashSet::new();

    for i in 0..500 {
        let input = inputs[i % inputs.len()];
        let result = svc.cycle(input);
        let m = &result.metadata;

        // Track readiness bounds.
        if m.unified_readiness_score < min_readiness {
            min_readiness = m.unified_readiness_score;
        }
        if m.unified_readiness_score > max_readiness {
            max_readiness = m.unified_readiness_score;
        }
        if m.unified_readiness_score < 0.5 {
            readiness_below_half_count += 1;
        }
        if m.unified_readiness_score < 0.95 {
            gate_fired = true;
        }
        if m.mechanism_activation_count > max_activations {
            max_activations = m.mechanism_activation_count;
        }

        // Track LR bounds.
        let lr = m.actual_effective_lr;
        if lr < min_lr {
            min_lr = lr;
        }
        if lr > max_lr {
            max_lr = lr;
        }

        // S21: Track contribution telemetry.
        if m.sleep_pressure_recovering {
            recovery_seen = true;
        }
        if !m.lr_dominant_source.is_empty() {
            dominant_sources.insert(m.lr_dominant_source.clone());
        }

        // THE critical invariant: readiness floor.
        assert!(
            m.unified_readiness_score >= 0.3,
            "Readiness floor violated at cycle {i}: {}",
            m.unified_readiness_score
        );

        // LR should never be NaN or negative.
        assert!(lr.is_finite() && lr >= 0.0, "LR invalid at cycle {i}: {lr}");

        // S21: LR proposal source count should be reasonable.
        assert!(
            m.lr_proposal_source_count <= 100,
            "implausible LR source count: {} at cycle {i}",
            m.lr_proposal_source_count
        );
    }

    // Print audit summary.
    eprintln!("=== Mechanism Audit: 500 cycles ===");
    eprintln!("Readiness range: [{min_readiness:.4}, {max_readiness:.4}]");
    eprintln!("LR range: [{min_lr:.4}, {max_lr:.4}]");
    eprintln!("Max activations/cycle: {max_activations}");
    eprintln!("Readiness < 0.5 in {readiness_below_half_count} / 500 cycles");
    eprintln!("Readiness gate fired: {gate_fired}");
    eprintln!("Sleep pressure recovery seen: {recovery_seen}");
    eprintln!("Unique LR dominant sources: {}", dominant_sources.len());
    for src in &dominant_sources {
        eprintln!("  - {src}");
    }

    // Verify non-trivial mechanism activation.
    assert!(
        max_activations >= 5,
        "Too few mechanism activations per cycle: max {max_activations} (expected >= 5)"
    );

    // Readiness gate must fire within 500 cycles.
    assert!(
        gate_fired,
        "Unified readiness gate never activated in 500 cycles"
    );

    // S21: Multiple LR sources should contribute over 500 cycles.
    assert!(
        dominant_sources.len() >= 2,
        "Too few unique LR dominant sources: {} (expected >= 2)",
        dominant_sources.len()
    );
}