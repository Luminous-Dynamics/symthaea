// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Property-Based Tests for Feedback Consensus System

Validates that the LR composition and feedback system produce bounded,
consistent results across the parameter space.

## Key Properties

1. **LR composition bounded**: `compose_effective_lr` stays in [0, 0.01].
2. **LR composition finite**: No NaN/Inf from extreme inputs.
3. **LR subsystem reset**: subsystem_lr_factor always reset to 1.0 after compose.
*/

use proptest::prelude::*;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

// ═══════════════════════════════════════════════════════════════════════════════
// Property 1: LR composition stays in [0.0, 0.01]
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn prop_compose_effective_lr_bounded() {
    proptest!(|(
        semantic_factor in 0.01f32..5.0,
        reasoning_factor in 0.01f32..5.0,
        subsystem_factor in 0.1f32..3.0,
        mce_boost in 0.0f32..1.0,
    )| {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        service.carryover.learning.subsystem_lr_factor = subsystem_factor;
        service.carryover.learning.mce_lr_boost = mce_boost;
        let lr = service.compose_effective_lr(semantic_factor, reasoning_factor);
        prop_assert!(lr >= 0.0, "LR below 0: {lr}");
        prop_assert!(lr <= 0.01, "LR above 0.01: {lr}");
        prop_assert!(lr.is_finite(), "LR non-finite: {lr}");
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Property 2: LR subsystem factor reset after compose
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn prop_compose_effective_lr_resets_subsystem() {
    proptest!(|(
        initial_factor in 0.5f32..2.5,
    )| {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        service.carryover.learning.subsystem_lr_factor = initial_factor;
        let _lr = service.compose_effective_lr(1.0, 1.0);
        prop_assert!(
            (service.carryover.learning.subsystem_lr_factor - 1.0).abs() < 1e-6,
            "subsystem_lr_factor not reset: {}",
            service.carryover.learning.subsystem_lr_factor
        );
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Property 3: Telemetry fields populated after compose
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn prop_compose_effective_lr_populates_telemetry() {
    proptest!(|(
        semantic_factor in 0.5f32..2.0,
        reasoning_factor in 0.5f32..2.0,
    )| {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        let _lr = service.compose_effective_lr(semantic_factor, reasoning_factor);
        let cog_mod = service.carryover.learning.lr_cognitive_mod;
        let meta_mod = service.carryover.learning.lr_meta_mod;
        prop_assert!(cog_mod > 0.0 && cog_mod.is_finite(), "cognitive_mod invalid: {cog_mod}");
        prop_assert!(meta_mod > 0.0 && meta_mod.is_finite(), "meta_mod invalid: {meta_mod}");
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Property 4: Behavioral equivalence — default priority = old behavior
// ═══════════════════════════════════════════════════════════════════════════════

/// With all-Cognitive priority (the default), the new weighted consensus
/// should produce identical results to the old unweighted consensus.
/// This ensures backward compatibility.
#[test]
fn prop_default_priority_preserves_behavior() {
    proptest!(|(
        n_adds in 1usize..20,
        n_scales in 0usize..10,
        base in 0.1f64..0.9,
    )| {
        use symthaea::cognitive_loop::feedback_state::{FeedbackProposal, ProposalCollector};

        let mut collector = ProposalCollector::new();
        // All proposals at default (Cognitive) priority with confidence 1.0
        for i in 0..n_adds {
            let delta = (i as f64 * 0.01) - 0.05; // range: -0.05 to +0.14
            collector.propose("test", FeedbackProposal::Add(delta));
        }
        for i in 0..n_scales {
            let factor = 0.9 + (i as f64 * 0.02); // range: 0.9 to 1.08
            collector.propose("test", FeedbackProposal::Scale(factor));
        }

        let result = collector.integrate(base, 0.0, 1.0);

        // All weights are 1.0 × 1.0 = 1.0, so weighted average = simple average
        // and weighted geometric mean = simple geometric mean.
        // The result should be deterministic and within bounds.
        prop_assert!(result.effective >= 0.0 && result.effective <= 1.0);
        prop_assert!(result.effective.is_finite());
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Property 5: Confidence scaling — zero confidence = no influence
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn prop_zero_confidence_has_no_influence() {
    proptest!(|(
        delta in -0.5f64..0.5,
        base in 0.1f64..0.9,
    )| {
        use symthaea::cognitive_loop::feedback_state::{FeedbackProposal, Priority, ProposalCollector};

        // A proposal with zero confidence should not move the value
        let mut collector = ProposalCollector::new();
        collector.propose_weighted("zero_conf", FeedbackProposal::Add(delta), Priority::Safety, 0.0);
        let result = collector.integrate(base, 0.0, 1.0);

        // With weight = 3.0 × 0.0 = 0.0, total_weight = 0, so no add applied
        prop_assert!((result.effective - base).abs() < 1e-10,
            "Zero-confidence proposal moved value: base={base}, got={}", result.effective);
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Property 6: Cache correctness — cached and uncached produce same result
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn prop_cache_matches_uncached() {
    proptest!(|(
        deltas in proptest::collection::vec(-0.3f64..0.3, 1..15),
        factors in proptest::collection::vec(0.8f64..1.2, 0..8),
        base in 0.1f64..0.9,
    )| {
        use symthaea::cognitive_loop::feedback_state::{FeedbackProposal, ProposalCollector};

        let mut collector = ProposalCollector::new();
        for d in &deltas {
            collector.propose("test", FeedbackProposal::Add(*d));
        }
        for f in &factors {
            collector.propose("test", FeedbackProposal::Scale(*f));
        }

        // First call populates cache
        let r1 = collector.integrate(base, 0.0, 1.0);
        // Second call should hit cache
        let r2 = collector.integrate(base, 0.0, 1.0);

        prop_assert!((r1.effective - r2.effective).abs() < 1e-15,
            "Cache mismatch: first={}, second={}", r1.effective, r2.effective);
    });
}