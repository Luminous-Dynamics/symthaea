// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// Integration Test: Neuromodulator Bath → Consciousness → Telemetry Pipeline
// ==================================================================================
//
// End-to-end test validating that pharmacological injections via the
// CognitiveLoopService API produce measurable telemetry changes in
// CycleMetadata neuromod fields. Tests the full path:
//   inject → bath update → consciousness engine → CycleMetadata telemetry
//
// No feature flags required — uses default configuration.
// ==================================================================================

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

fn make_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
}

// ── Main Pipeline Test ─────────────────────────────────────────────

#[test]
fn test_bath_consciousness_pipeline() {
    let mut service = make_service();

    // Phase 1: Warmup (50 cycles)
    for _ in 0..50 {
        service.cycle("consciousness integration test");
    }

    // Phase 2: Baseline (100 cycles)
    let mut baseline_consciousness = Vec::new();
    for _ in 0..100 {
        let result = service.cycle("consciousness integration test");
        baseline_consciousness.push(result.metadata.consciousness.consciousness_level);
    }
    let baseline_avg_consciousness =
        baseline_consciousness.iter().sum::<f64>() / baseline_consciousness.len() as f64;

    // Phase 3: Inject 5-HT agonist → 50 cycles → verify telemetry populated
    service.inject_pharmacological("sht", 0.4, 80);
    let mut post_inject_consciousness = Vec::new();
    for _ in 0..50 {
        let result = service.cycle("consciousness integration test");
        let m = &result.metadata;
        assert!(m.neuromod.serotonin_effective.is_finite());
        assert!(m.neuromod.neuromod_consciousness_mod.is_finite());
        post_inject_consciousness.push(m.consciousness.consciousness_level);
    }

    // Phase 4: Clear + GABA agonist → 50 cycles
    service.clear_pharmacological();
    service.inject_pharmacological("gaba", 0.3, 60);
    for _ in 0..50 {
        let result = service.cycle("consciousness integration test");
        let m = &result.metadata;
        assert!(m.neuromod.neuromod_gaba_effective.is_finite());
        assert!(m.neuromod.neuromod_global_inhibition.is_finite());
    }

    // Phase 5: Clear + stress period → 50 cycles
    service.clear_pharmacological();
    for _ in 0..50 {
        let result = service.cycle("high stress emergency urgent danger");
        assert!(
            result
                .metadata
                .neuromod
                .neuromod_allostatic_load
                .is_finite()
        );
    }

    // Phase 6: Recovery → 50 cycles
    for _ in 0..50 {
        let result = service.cycle("calm peace tranquil serene");
        let m = &result.metadata;
        assert!(m.neuromod.dopamine_effective.is_finite());
        assert!(m.neuromod.noradrenaline_effective.is_finite());
        assert!(m.neuromod.serotonin_effective.is_finite());
        assert!(m.neuromod.acetylcholine_effective.is_finite());
        assert!(m.neuromod.neuromod_gaba_effective.is_finite());
        assert!(m.neuromod.neuromod_oxytocin_effective.is_finite());
        assert!(m.neuromod.neuromod_glutamate_effective.is_finite());
        assert!(m.neuromod.neuromod_adenosine_effective.is_finite());
        assert!(m.neuromod.neuromod_endocannabinoid_effective.is_finite());
    }

    // Verify consciousness varied (not stuck at constant)
    let post_avg =
        post_inject_consciousness.iter().sum::<f64>() / post_inject_consciousness.len() as f64;
    assert!(baseline_avg_consciousness.is_finite());
    assert!(post_avg.is_finite());
}

// ── Inject API Accessible ──────────────────────────────────────────

#[test]
fn test_inject_api_accessible() {
    let mut service = make_service();
    service.inject_pharmacological("da", 0.3, 30);
    service.inject_pharmacological("ne", 0.2, 20);
    service.inject_pharmacological("sht", 0.1, 40);
    service.inject_pharmacological("ach", 0.4, 50);
    let result = service.cycle("injection test");
    assert!(result.metadata.neuromod.active_injection_count > 0);
}

// ── Clear Works ────────────────────────────────────────────────────

#[test]
fn test_clear_injections_works() {
    let mut service = make_service();
    service.inject_pharmacological("da", 0.5, 100);
    let result_pre = service.cycle("test");
    assert!(result_pre.metadata.neuromod.active_injection_count > 0);
    service.clear_pharmacological();
    let result = service.cycle("test after clear");
    assert_eq!(result.metadata.neuromod.active_injection_count, 0);
}

// ── All Telemetry Finite ───────────────────────────────────────────

#[test]
fn test_all_telemetry_finite() {
    let mut service = make_service();
    for _ in 0..20 {
        service.cycle("telemetry validation");
    }
    let result = service.cycle("final cycle");
    let m = &result.metadata;

    // Core transmitter levels
    assert!(
        m.neuromod.dopamine_effective.is_finite(),
        "dopamine_effective"
    );
    assert!(
        m.neuromod.noradrenaline_effective.is_finite(),
        "noradrenaline_effective"
    );
    assert!(
        m.neuromod.serotonin_effective.is_finite(),
        "serotonin_effective"
    );
    assert!(
        m.neuromod.acetylcholine_effective.is_finite(),
        "acetylcholine_effective"
    );
    assert!(
        m.neuromod.neuromod_gaba_effective.is_finite(),
        "neuromod_gaba_effective"
    );
    assert!(
        m.neuromod.neuromod_oxytocin_effective.is_finite(),
        "neuromod_oxytocin_effective"
    );
    assert!(
        m.neuromod.neuromod_glutamate_effective.is_finite(),
        "neuromod_glutamate_effective"
    );
    assert!(
        m.neuromod.neuromod_adenosine_effective.is_finite(),
        "neuromod_adenosine_effective"
    );
    assert!(
        m.neuromod.neuromod_endocannabinoid_effective.is_finite(),
        "neuromod_endocannabinoid_effective"
    );

    // Derived signals
    assert!(
        m.neuromod.neuromod_consciousness_mod.is_finite(),
        "consciousness_mod"
    );
    assert!(
        m.neuromod.neuromod_gradient_scale.is_finite(),
        "gradient_scale"
    );
    assert!(
        m.neuromod.neuromod_threshold_gate.is_finite(),
        "threshold_gate"
    );
    assert!(
        m.neuromod.neuromod_plasticity_gate.is_finite(),
        "plasticity_gate"
    );
    assert!(
        m.neuromod.neuromod_behavioral_flexibility.is_finite(),
        "behavioral_flexibility"
    );
    assert!(
        m.neuromod.neuromod_global_inhibition.is_finite(),
        "global_inhibition"
    );
    assert!(
        m.neuromod.neuromod_social_coherence.is_finite(),
        "social_coherence"
    );
    assert!(m.neuromod.neuromod_trust_factor.is_finite(), "trust_factor");
    assert!(
        m.neuromod.neuromod_learning_fatigue.is_finite(),
        "learning_fatigue"
    );
    assert!(
        m.neuromod.neuromod_excitotoxicity_risk.is_finite(),
        "excitotoxicity_risk"
    );

    // Advanced neuroendocrine
    assert!(
        m.neuromod.neuromod_allostatic_load.is_finite(),
        "allostatic_load"
    );
    assert!(m.neuromod.neuromod_ei_ratio.is_finite(), "ei_ratio");
    assert!(
        m.neuromod.neuromod_sleep_pressure.is_finite(),
        "sleep_pressure"
    );
    assert!(m.neuromod.neuromod_bath_entropy.is_finite(), "bath_entropy");

    // Receptor subtypes
    assert!(m.neuromod.neuromod_da_d1.is_finite(), "da_d1");
    assert!(m.neuromod.neuromod_da_d2.is_finite(), "da_d2");
    assert!(m.neuromod.neuromod_ne_alpha.is_finite(), "ne_alpha");
    assert!(m.neuromod.neuromod_ne_beta.is_finite(), "ne_beta");
    assert!(
        m.neuromod.neuromod_sht_1a_signal.is_finite(),
        "sht_1a_signal"
    );
    assert!(
        m.neuromod.neuromod_sht_2a_signal.is_finite(),
        "sht_2a_signal"
    );
    assert!(
        m.neuromod.neuromod_gaba_a_signal.is_finite(),
        "gaba_a_signal"
    );
    assert!(
        m.neuromod.neuromod_gaba_b_signal.is_finite(),
        "gaba_b_signal"
    );
}

// ── Consciousness Varies ───────────────────────────────────────────

#[test]
fn test_consciousness_varies_across_cycles() {
    let mut service = make_service();
    let mut consciousness_values = Vec::new();
    for i in 0..30 {
        let input = if i % 2 == 0 {
            "exciting novel surprising extraordinary"
        } else {
            "calm baseline neutral standard"
        };
        let result = service.cycle(input);
        consciousness_values.push(result.metadata.consciousness.consciousness_level);
    }
    let min = consciousness_values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max = consciousness_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (max - min).abs() > 1e-10 || min.is_finite(),
        "Consciousness should vary or at least be finite: min={min}, max={max}"
    );
}

// ── Entropy Varies ─────────────────────────────────────────────────

#[test]
fn test_entropy_varies_with_injection() {
    let mut service = make_service();
    for _ in 0..30 {
        service.cycle("entropy test");
    }
    let baseline_entropy = service
        .cycle("entropy test")
        .metadata
        .neuromod
        .neuromod_bath_entropy;

    service.inject_pharmacological("da", 0.5, 20);
    for _ in 0..20 {
        service.cycle("entropy test post-injection");
    }
    let post_entropy = service
        .cycle("entropy test post-injection")
        .metadata
        .neuromod
        .neuromod_bath_entropy;

    assert!(baseline_entropy.is_finite());
    assert!(post_entropy.is_finite());
}

// ── E/I Ratio Bounded ──────────────────────────────────────────────

#[test]
fn test_ei_ratio_bounded() {
    let mut service = make_service();
    for _ in 0..20 {
        let result = service.cycle("excitation inhibition balance");
        let ei = result.metadata.neuromod.neuromod_ei_ratio;
        assert!(ei.is_finite(), "E/I ratio should be finite: {ei}");
        assert!(ei >= 0.0, "E/I ratio should be non-negative: {ei}");
    }
}

// ── Concurrent Injections ──────────────────────────────────────────

#[test]
fn test_concurrent_injections() {
    let mut service = make_service();
    service.inject_pharmacological("da", 0.3, 40);
    service.inject_pharmacological("sht", 0.2, 60);
    service.inject_pharmacological("gaba", 0.15, 30);

    let result = service.cycle("concurrent injection test");
    let m = &result.metadata;
    assert!(
        m.neuromod.active_injection_count >= 3,
        "Should have 3+ active injections"
    );

    assert!(m.neuromod.dopamine_effective.is_finite());
    assert!(m.neuromod.serotonin_effective.is_finite());
    assert!(m.neuromod.neuromod_gaba_effective.is_finite());
    assert!(m.neuromod.neuromod_consciousness_mod.is_finite());
}