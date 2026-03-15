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
