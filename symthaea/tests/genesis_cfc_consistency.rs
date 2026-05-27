#![cfg(feature = "genesis")]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-crate CfC consistency validation for genesis pipeline predictors.
//!
//! All 5 genesis crates use `HdcLtcUnifiedNeuron::evolve_closed_form()` for O(1)
//! temporal jumps. This test suite validates three invariants:
//!
//! 1. **Transitivity**: Evolving t=0->100 in one step equals t=0->50 then t=50->100
//!    (the closed-form solution is path-independent, up to floating point tolerance).
//! 2. **Finiteness**: All predictors produce finite, non-NaN outputs for large dt.
//! 3. **Non-degeneracy**: Different inputs produce distinguishable outputs.

use std::time::Instant;

use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// ─── Genesis crate imports ──────────────────────────────────────────────────

use symthaea_cell_foundry::MultiScalePredictor;
use symthaea_ectogenesis::TemporalPlanner;
use symthaea_genomics::DegradationModel;
use symthaea_nurture::{CriticalPeriodDomain, CriticalPeriodModel, DevelopmentalAge};
use symthaea_population::PopulationTrajectoryPredictor;

// =============================================================================
// Helpers
// =============================================================================

/// Build a fresh HdcLtcUnifiedNeuron with a given tau and seed, matching the
/// pattern used across genesis crates.
fn make_neuron(tau_base: f32, seed: u64) -> HdcLtcUnifiedNeuron {
    let config = UnifiedConfig {
        tau_base,
        backbone_tau: 0.1,
        dimension: HDC_DIMENSION,
        ..UnifiedConfig::default()
    };
    HdcLtcUnifiedNeuron::new(config, seed)
}

/// Assert all values in a ContinuousHV are finite (not NaN or Inf).
fn assert_finite(hv: &ContinuousHV, label: &str) {
    for (i, v) in hv.values.iter().enumerate() {
        assert!(
            v.is_finite(),
            "{label}: non-finite value at dimension {i}: {v}"
        );
    }
}

// =============================================================================
// TEST 1: Raw CfC transitivity (neuron-level)
// =============================================================================

/// The closed-form CfC solution satisfies:
///   evolve(dt_total, input) == evolve(dt_a, input) then evolve(dt_b, input)
/// when dt_a + dt_b == dt_total, because:
///   x(t+dt) = x_inf + (x(t) - x_inf) * exp(-dt/tau)
/// is composable: exp(-dt_a/tau) * exp(-dt_b/tau) == exp(-(dt_a+dt_b)/tau).
///
/// Note: The gating and backbone_tau introduce mild state-dependent nonlinearity,
/// so we use a relaxed tolerance rather than exact equality.
#[test]
fn test_raw_cfc_transitivity() {
    let input = ContinuousHV::random(HDC_DIMENSION, 0xCFC0_CAFE);

    // Single jump: t=0 -> t=100
    let mut neuron_single = make_neuron(10.0, 42);
    neuron_single.evolve_closed_form(100.0, &input);
    let state_single = neuron_single.state().clone();

    // Two-step jump: t=0 -> t=50 -> t=100
    let mut neuron_two = make_neuron(10.0, 42);
    neuron_two.evolve_closed_form(50.0, &input);
    neuron_two.evolve_closed_form(50.0, &input);
    let state_two = neuron_two.state().clone();

    let sim = state_single.similarity(&state_two);
    assert!(
        sim > 0.95,
        "CfC transitivity: single-step and two-step should be highly similar, got sim={sim}"
    );
}

// =============================================================================
// TEST 2: Genomics DegradationModel transitivity
// =============================================================================

#[test]
fn test_genomics_degradation_transitivity() {
    // Single jump: 0 -> 10,000 years at 15C
    let mut model_single = DegradationModel::new(15.0);
    let profile_single = model_single.predict_damage_profile(10_000.0);

    // Two-step: 0 -> 5,000 years, then reset-and-predict-5000-from-intermediate
    // Note: DegradationModel mutates internal neuron state, so the second call
    // to predict_damage_profile starts from the state after the first call.
    let mut model_two = DegradationModel::new(15.0);
    let _intermediate = model_two.predict_damage_profile(5_000.0);
    let profile_two = model_two.predict_damage_profile(5_000.0);

    // Both should be finite
    assert_finite(&profile_single, "genomics single-step");
    assert_finite(&profile_two, "genomics two-step");

    // The accumulated state after 5k+5k should be similar to 10k
    // (not exact due to neuron state mutation between calls)
    let sim = profile_single.similarity(&profile_two);
    assert!(
        sim > 0.8,
        "Genomics degradation transitivity: sim={sim} (expected >0.8)"
    );
}

// =============================================================================
// TEST 3: Cell Foundry MultiScalePredictor transitivity via observe()
// =============================================================================

/// Cell Foundry's observe() mutates internal neuron state, so we can test
/// CfC transitivity by comparing one observe(dt) vs two observe(dt/2) calls
/// with the same input held constant.
#[test]
fn test_cell_foundry_transitivity() {
    let input = ContinuousHV::random(HDC_DIMENSION, 0xCE11_F0AD);

    // Single observation: dt=200s
    let mut predictor_single = MultiScalePredictor::new();
    predictor_single.observe(&input, 200.0);
    let state_single = predictor_single.state().clone();

    // Two-step observation: dt=100s + dt=100s with same input
    let mut predictor_two = MultiScalePredictor::new();
    predictor_two.observe(&input, 100.0);
    predictor_two.observe(&input, 100.0);
    let state_two = predictor_two.state().clone();

    assert_finite(&state_single, "cell-foundry single-step");
    assert_finite(&state_two, "cell-foundry two-step");

    // CfC closed-form with constant input is composable:
    // exp(-dt/tau) == exp(-dt/2/tau) * exp(-dt/2/tau)
    // Gating introduces mild nonlinearity, so we use relaxed tolerance.
    let sim = state_single.similarity(&state_two);
    assert!(
        sim > 0.95,
        "Cell foundry transitivity: sim={sim} (expected >0.95)"
    );
}

// =============================================================================
// TEST 4: Ectogenesis TemporalPlanner deterministic reproducibility
// =============================================================================

/// TemporalPlanner's predict_at_week() calls set_state() then evolve_closed_form(),
/// making it a pure function of (state, current_week, target_week). The CfC
/// closed-form solution is deterministic: same inputs must yield same output.
#[test]
fn test_ectogenesis_deterministic_reproducibility() {
    let state = ContinuousHV::random(HDC_DIMENSION, 0xFE1A_5EED);

    // Same prediction twice should be identical
    let mut planner_a = TemporalPlanner::new();
    let pred_a = planner_a.predict_at_week(&state, 10, 30);

    let mut planner_b = TemporalPlanner::new();
    let pred_b = planner_b.predict_at_week(&state, 10, 30);

    assert_finite(&pred_a, "ectogenesis prediction A");
    assert_finite(&pred_b, "ectogenesis prediction B");

    let sim = pred_a.similarity(&pred_b);
    assert!(
        (sim - 1.0).abs() < 1e-4,
        "Ectogenesis determinism: same inputs should yield identical outputs, sim={sim}"
    );

    // Different initial states should produce different predictions
    // (different target weeks converge to x_inf for large dt/tau ratios,
    // so we test non-degeneracy via input variation instead)
    let state_alt = ContinuousHV::random(HDC_DIMENSION, 0xDEAD_BEEF);
    let mut planner_c = TemporalPlanner::new();
    let pred_alt = planner_c.predict_at_week(&state_alt, 10, 30);

    let sim_diff = pred_a.similarity(&pred_alt);
    assert!(
        sim_diff < 0.999,
        "Ectogenesis: different initial states should diverge, sim={sim_diff}"
    );
}

// =============================================================================
// TEST 5: Population trajectory predictor deterministic reproducibility
// =============================================================================

/// PopulationTrajectoryPredictor's predict_at_generation() calls set_state()
/// then evolve_closed_form(), so it is a deterministic function of (state, r#gen).
/// Same inputs must yield identical outputs (CfC closed-form is deterministic).
#[test]
fn test_population_deterministic_reproducibility() {
    let initial = ContinuousHV::random(HDC_DIMENSION, 0xA0A0_FACE);

    // Same prediction twice should be identical
    let mut pred_a = PopulationTrajectoryPredictor::with_generation_time(1.0);
    let state_a = pred_a.predict_at_generation(&initial, 100);

    let mut pred_b = PopulationTrajectoryPredictor::with_generation_time(1.0);
    let state_b = pred_b.predict_at_generation(&initial, 100);

    assert_finite(&state_a, "population prediction A");
    assert_finite(&state_b, "population prediction B");

    let sim = state_a.similarity(&state_b);
    assert!(
        (sim - 1.0).abs() < 1e-4,
        "Population determinism: same inputs should yield identical outputs, sim={sim}"
    );

    // Different generation counts should produce different states
    let mut pred_c = PopulationTrajectoryPredictor::with_generation_time(1.0);
    let state_short = pred_c.predict_at_generation(&initial, 10);
    let mut pred_d = PopulationTrajectoryPredictor::with_generation_time(1.0);
    let state_long = pred_d.predict_at_generation(&initial, 1000);

    let sim_diff = state_short.similarity(&state_long);
    assert!(
        sim_diff < 0.999,
        "Population: different generation counts should diverge, sim={sim_diff}"
    );
}

// =============================================================================
// TEST 6: Nurture CriticalPeriodModel CfC neuron state finiteness
// =============================================================================

#[test]
fn test_nurture_cfc_finiteness_across_domains() {
    for domain in [
        CriticalPeriodDomain::Attachment,
        CriticalPeriodDomain::Language,
        CriticalPeriodDomain::Vision,
    ] {
        let mut model = CriticalPeriodModel::new();

        // Evolve through a range of ages (0 to 120 months)
        for month in [0.0, 6.0, 12.0, 24.0, 48.0, 72.0, 120.0] {
            let age = DevelopmentalAge::new(month);
            let plasticity = model.plasticity_at(domain, &age);
            assert!(
                plasticity.is_finite(),
                "{domain:?} plasticity at {month}mo should be finite, got {plasticity}"
            );
            assert!(
                plasticity >= 0.0 && plasticity <= 1.0,
                "{domain:?} plasticity at {month}mo should be in [0,1], got {plasticity}"
            );
        }

        // After all those evolutions, the neuron state should still be finite
        let state = model.neuron_state();
        assert_finite(state, &format!("nurture {domain:?} final state"));
        assert!(
            state.norm() > 0.0,
            "Nurture {domain:?} neuron state should be non-zero after evolution"
        );
    }
}

// =============================================================================
// TEST 7: All predictors produce finite outputs for extreme dt values
// =============================================================================

#[test]
fn test_all_predictors_finite_for_large_dt() {
    // Genomics: 1 million years
    let mut genomics = DegradationModel::new(15.0);
    let profile = genomics.predict_damage_profile(1_000_000.0);
    assert_finite(&profile, "genomics 1M years");
    assert!(profile.norm() > 0.0, "genomics 1M years should be non-zero");

    // Cell foundry: 9 months (max biological horizon)
    let cell = MultiScalePredictor::new();
    let input = ContinuousHV::random(HDC_DIMENSION, 42);
    let pred = cell.predict_at_horizon(&input, 23_328_000.0);
    assert_finite(&pred, "cell-foundry 9 months");

    // Ectogenesis: full term (week 40)
    let mut ecto = TemporalPlanner::new();
    let fetal_state = ContinuousHV::random(HDC_DIMENSION, 0xBABE);
    let term = ecto.predict_at_week(&fetal_state, 1, 40);
    assert_finite(&term, "ectogenesis week 1->40");

    // Population: 10,000 generations
    let mut pop = PopulationTrajectoryPredictor::with_generation_time(1.0);
    let pop_state = ContinuousHV::random(HDC_DIMENSION, 0xD0A0);
    let evolved = pop.predict_at_generation(&pop_state, 10_000);
    assert_finite(&evolved, "population 10k generations");

    // Nurture: extreme age
    let mut nurture = CriticalPeriodModel::new();
    let extreme_age = DevelopmentalAge::new(600.0); // 50 years in months
    let p = nurture.plasticity_at(CriticalPeriodDomain::Language, &extreme_age);
    assert!(
        p.is_finite(),
        "nurture plasticity at 50 years should be finite"
    );
    assert_finite(nurture.neuron_state(), "nurture neuron at 50 years");
}

// =============================================================================
// TEST 8: Non-degeneracy — different inputs yield different outputs
// =============================================================================

#[test]
fn test_genomics_non_degenerate() {
    // Use a long time span (100,000 years) and wide temperature gap so the
    // Arrhenius scaling factor creates a measurable difference in the CfC
    // evolution. At short timescales, the exponential decay barely diverges.
    let mut model_cold = DegradationModel::new(-20.0);
    let mut model_hot = DegradationModel::new(50.0);

    let profile_cold = model_cold.predict_damage_profile(100_000.0);
    let profile_hot = model_hot.predict_damage_profile(100_000.0);

    // Verify both are finite
    assert_finite(&profile_cold, "genomics cold profile");
    assert_finite(&profile_hot, "genomics hot profile");

    // At least verify the states are not identical (both evolved from zero
    // with different effective dt due to Arrhenius scaling)
    let sim = profile_cold.similarity(&profile_hot);
    assert!(
        sim < 1.0 - 1e-6,
        "Genomics: different temperatures should produce different damage profiles, sim={sim}"
    );
}

#[test]
fn test_cell_foundry_non_degenerate() {
    let predictor = MultiScalePredictor::new();
    let input_a = ContinuousHV::random(HDC_DIMENSION, 100);
    let input_b = ContinuousHV::random(HDC_DIMENSION, 200);

    let pred_a = predictor.predict_at_horizon(&input_a, 3_600.0);
    let pred_b = predictor.predict_at_horizon(&input_b, 3_600.0);

    let sim = pred_a.similarity(&pred_b);
    assert!(
        sim < 0.99,
        "Cell foundry: different cell states should predict differently, sim={sim}"
    );
}

#[test]
fn test_ectogenesis_non_degenerate() {
    let state_a = ContinuousHV::random(HDC_DIMENSION, 0xAAAA);
    let state_b = ContinuousHV::random(HDC_DIMENSION, 0xBBBB);

    let mut planner_a = TemporalPlanner::new();
    let mut planner_b = TemporalPlanner::new();

    let pred_a = planner_a.predict_at_week(&state_a, 10, 30);
    let pred_b = planner_b.predict_at_week(&state_b, 10, 30);

    let sim = pred_a.similarity(&pred_b);
    assert!(
        sim < 0.99,
        "Ectogenesis: different fetal states should predict differently, sim={sim}"
    );
}

#[test]
fn test_population_non_degenerate() {
    let state_a = ContinuousHV::random(HDC_DIMENSION, 0x1111);
    let state_b = ContinuousHV::random(HDC_DIMENSION, 0x2222);

    let mut pred_a = PopulationTrajectoryPredictor::with_generation_time(1.0);
    let mut pred_b = PopulationTrajectoryPredictor::with_generation_time(1.0);

    let evolved_a = pred_a.predict_at_generation(&state_a, 50);
    let evolved_b = pred_b.predict_at_generation(&state_b, 50);

    let sim = evolved_a.similarity(&evolved_b);
    assert!(
        sim < 0.99,
        "Population: different initial states should evolve differently, sim={sim}"
    );
}

// =============================================================================
// TEST 12: O(1) wall-clock time property across crates
// =============================================================================

/// Verify that small dt and large dt take approximately the same wall-clock
/// time, confirming the O(1) closed-form property holds at the API level.
#[test]
fn test_o1_wall_clock_across_crates() {
    let iterations = 200;
    let input = ContinuousHV::random(HDC_DIMENSION, 0xBEEF);

    // Cell Foundry: dt=1s vs dt=23_328_000s (9 months)
    let predictor = MultiScalePredictor::new();

    let start_small = Instant::now();
    for _ in 0..iterations {
        let _ = predictor.predict_at_horizon(&input, 1.0);
    }
    let time_small = start_small.elapsed();

    let start_large = Instant::now();
    for _ in 0..iterations {
        let _ = predictor.predict_at_horizon(&input, 23_328_000.0);
    }
    let time_large = start_large.elapsed();

    let ratio = time_large.as_nanos() as f64 / time_small.as_nanos().max(1) as f64;
    assert!(
        ratio < 5.0 && ratio > 0.2,
        "O(1) property: small dt took {time_small:?}, large dt took {time_large:?}, ratio={ratio}"
    );
}