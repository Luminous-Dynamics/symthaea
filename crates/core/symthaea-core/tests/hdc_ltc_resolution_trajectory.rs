//! Research-only deterministic trajectory fixtures for adaptive HDC-LTC resolution.
//!
//! This suite deliberately starts with the public neuron API rather than a
//! production resolution-transition API. It establishes controls and a
//! trajectory-level oracle before any adaptive conversion is allowed to affect
//! production state.

use symthaea_core::hdc::{ContinuousHV, HdcLtcUnifiedNeuron, UnifiedActivation, UnifiedConfig};

const DIMS: [usize; 7] = [1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536];
const SEED: u64 = 0x4844_432d_4c54_4301;
const INPUT_SEED: u64 = 0x5452414a_2d494e50;
const DT_SCHEDULE: [f32; 8] = [0.01, 0.017, 0.031, 0.007, 0.023, 0.041, 0.013, 0.029];

fn config(dim: usize) -> UnifiedConfig {
    UnifiedConfig {
        dimension: dim,
        activation: UnifiedActivation::Tanh,
        // Keep the fixture deterministic and sensitive to state/input geometry.
        tau_base: 0.1,
        backbone_tau: 0.5,
        gating_steepness: 1.0,
        interp_bias: 0.0,
        fourier_frequencies: Vec::new(),
        fourier_amplitude: 0.1,
        learning_rate: 0.01,
        momentum: 0.9,
        weight_decay: 0.0001,
    }
}

fn input(dim: usize, step: usize) -> ContinuousHV {
    ContinuousHV::random(dim, INPUT_SEED.wrapping_add(step as u64 * 7919))
}

fn neuron(dim: usize) -> HdcLtcUnifiedNeuron {
    HdcLtcUnifiedNeuron::new(config(dim), SEED.wrapping_add(dim as u64))
}

fn normalized_l2_error(a: &ContinuousHV, b: &ContinuousHV) -> f32 {
    assert_eq!(a.dim(), b.dim());
    let denom = a.norm().max(1e-12);
    a.subtract(b).norm() / denom
}

fn trajectory(dim: usize, steps: usize, irregular: bool) -> Vec<ContinuousHV> {
    let mut n = neuron(dim);
    let mut out = Vec::with_capacity(steps + 1);
    out.push(n.state().clone());

    for step in 0..steps {
        let dt = if irregular {
            DT_SCHEDULE[step % DT_SCHEDULE.len()]
        } else {
            0.02
        };
        let u = input(dim, step);
        n.evolve_closed_form(dt, &u);
        out.push(n.state().clone());
    }

    out
}

#[test]
fn identity_control_is_bitwise_deterministic_for_fixed_schedule() {
    let a = trajectory(4_096, 16, true);
    let b = trajectory(4_096, 16, true);

    assert_eq!(a.len(), b.len());
    for (lhs, rhs) in a.iter().zip(b.iter()) {
        assert_eq!(lhs.dim(), rhs.dim());
        assert_eq!(lhs.values, rhs.values);
    }
}

#[test]
fn same_dimension_identity_has_zero_representation_error() {
    let mut a = neuron(4_096);
    let mut b = neuron(4_096);

    for step in 0..16 {
        let u = input(4_096, step);
        let dt = DT_SCHEDULE[step % DT_SCHEDULE.len()];
        a.evolve_closed_form(dt, &u);
        b.evolve_closed_form(dt, &u);

        assert_eq!(a.state().values, b.state().values);
        assert_eq!(a.effective_tau(&u).to_bits(), b.effective_tau(&u).to_bits());
    }
}

#[test]
fn irregular_dt_changes_the_fixture_trajectory_but_remains_deterministic() {
    let fixed = trajectory(4_096, 32, false);
    let irregular_a = trajectory(4_096, 32, true);
    let irregular_b = trajectory(4_096, 32, true);

    for (a, b) in irregular_a.iter().zip(irregular_b.iter()) {
        assert_eq!(a.values, b.values);
    }

    let terminal_difference = normalized_l2_error(
        fixed.last().expect("fixed trajectory"),
        irregular_a.last().expect("irregular trajectory"),
    );
    assert!(terminal_difference.is_finite());
    assert!(terminal_difference > 1e-7);
}

#[test]
fn legacy_dilate_round_trip_is_measured_at_multiple_scales() {
    for &(hi, lo) in &[(16_384, 4_096), (32_768, 8_192), (65_536, 4_096)] {
        let source = ContinuousHV::random(hi, SEED.wrapping_add(hi as u64));
        let low = source.dilate(lo);
        let round_trip = low.dilate(hi);

        assert_eq!(low.dim(), lo);
        assert_eq!(round_trip.dim(), hi);
        let error = normalized_l2_error(&source, &round_trip);

        // This is a characterization, not a semantic-preservation gate.
        assert!(error.is_finite(), "non-finite round-trip error at {hi}->{lo}->{hi}");
    }
}

#[test]
fn legacy_dilate_can_be_characterized_against_liquid_trajectory() {
    for &(hi, lo) in &[(16_384, 4_096), (16_384, 8_192), (32_768, 8_192)] {
        let high = trajectory(hi, 8);
        let low_start = high[0].dilate(lo);

        let mut low = neuron(lo);
        low.set_state(low_start);

        let mut converted_high = Vec::with_capacity(9);
        converted_high.push(high[0].clone());

        for step in 0..8 {
            let dt = DT_SCHEDULE[step];
            let high_input = input(hi, step);
            let low_input = high_input.dilate(lo);

            // Reference trajectory evolves at high resolution.
            // Candidate trajectory evolves after conversion.
            // Compare only after projecting the high-resolution state.
            let mut reference = neuron(hi);
            reference.set_state(converted_high.last().expect("state").clone());
            reference.evolve_closed_form(dt, &high_input);
            let reference_state = reference.state().clone();
            converted_high.push(reference_state.clone());

            low.evolve_closed_form(dt, &low_input);
            let projected_reference = reference_state.dilate(lo);

            let error = normalized_l2_error(low.state(), &projected_reference);
            let reference_tau = reference.effective_tau(&high_input);
            let low_tau = low.effective_tau(&low_input);
            let tau_error = (reference_tau - low_tau).abs();

            assert!(error.is_finite());
            assert!(tau_error.is_finite());
        }
    }
}

#[test]
fn round_trip_hysteresis_is_explicitly_nonzero_or_zero_but_never_unknown() {
    for &(hi, lo) in &[(16_384, 4_096), (65_536, 1_024)] {
        let source = ContinuousHV::random(hi, SEED.wrapping_add(hi as u64));
        let round_trip = source.dilate(lo).dilate(hi);
        let error = normalized_l2_error(&source, &round_trip);

        assert!(error.is_finite());
        // Do not impose a quality threshold here. The purpose is to make
        // irreversible loss observable before a policy threshold is chosen.
    }
}

#[test]
fn resolution_ladder_has_expected_storage_order() {
    for pair in DIMS.windows(2) {
        assert!(pair[0] < pair[1]);
    }
}
