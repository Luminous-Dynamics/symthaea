//! Independent numerical-oracle tests for piecewise-linear thermal lethality.
//!
//! These tests verify the numerical integral under the crate's stated
//! piecewise-linear temperature assumption. They do not validate the biological
//! adequacy of any pathogen D/z parameters or certify a real cooking process.

use symthaea_culinary::thermal::ThermalTrajectory;

const D_REF: f64 = 0.396;
const T_REF: f64 = 60.0;
const Z: f64 = 5.56;

fn assert_relative_close(actual: f64, expected: f64, tolerance: f64, context: &str) {
    assert!(actual.is_finite(), "{context}: actual must be finite, got {actual}");
    assert!(expected.is_finite(), "{context}: oracle must be finite, got {expected}");
    let relative_error = (actual - expected).abs() / expected.abs().max(f64::MIN_POSITIVE);
    assert!(
        relative_error <= tolerance,
        "relative error {relative_error:e} exceeds {tolerance:e}: {context}; actual={actual:e}; expected={expected:e}"
    );
}

/// Independent closed-form oracle for one linear segment, expressed as the
/// endpoint-rate difference rather than the production exp_m1(mean-rate) form.
fn analytic_segment_log_reduction(
    duration_min: f64,
    start_c: f64,
    end_c: f64,
    d_ref: f64,
    t_ref: f64,
    z: f64,
) -> f64 {
    let start_rate = 10.0_f64.powf((start_c - t_ref) / z) / d_ref;
    if start_c == end_c {
        return duration_min * start_rate;
    }
    let end_rate = 10.0_f64.powf((end_c - t_ref) / z) / d_ref;
    duration_min * (end_rate - start_rate)
        / (std::f64::consts::LN_10 / z * (end_c - start_c))
}

fn model(trajectory: &ThermalTrajectory) -> f64 {
    trajectory.log_reduction(D_REF, T_REF, Z)
}

#[test]
fn constant_temperature_holds_match_the_analytical_rate() {
    for &(temperature_c, duration_min) in &[
        (40.0, 0.25),
        (55.0, 3.0),
        (60.0, 3.96),
        (72.0, 8.0),
        (100.0, 2.0),
        (180.0, 0.5),
    ] {
        let actual = model(&ThermalTrajectory::hold(temperature_c, duration_min));
        let expected = duration_min * 10.0_f64.powf((temperature_c - T_REF) / Z) / D_REF;
        assert_relative_close(
            actual,
            expected,
            2e-14,
            &format!("hold {temperature_c} C for {duration_min} min"),
        );
    }
}

#[test]
fn heating_and_cooling_ramps_match_independent_closed_form_oracle() {
    for &(start_c, end_c, duration_min) in &[
        (40.0, 80.0, 10.0),
        (40.0, 100.0, 10.0),
        (40.0, 200.0, 10.0),
        (80.0, 200.0, 0.5),
        (200.0, 40.0, 10.0),
        (250.0, 80.0, 3.0),
    ] {
        let trajectory =
            ThermalTrajectory::new(vec![(0.0, start_c), (duration_min, end_c)]);
        let actual = model(&trajectory);
        let expected = analytic_segment_log_reduction(
            duration_min,
            start_c,
            end_c,
            D_REF,
            T_REF,
            Z,
        );
        assert_relative_close(
            actual,
            expected,
            2e-12,
            &format!("{start_c} C -> {end_c} C over {duration_min} min"),
        );
    }
}

#[test]
fn near_constant_temperature_segment_uses_a_stable_small_delta_limit() {
    let duration_min = 7.0;
    let trajectory = ThermalTrajectory::new(vec![(0.0, 60.0), (duration_min, 60.0 + 1e-10)]);
    let actual = model(&trajectory);
    // The mean temperature differs from 60 C by only 5e-11 C; this limit
    // avoids an endpoint-rate difference oracle dominated by cancellation.
    let constant_limit = duration_min / D_REF;
    assert_relative_close(
        actual,
        constant_limit,
        5e-11,
        "near-constant temperature stability",
    );
}

#[test]
fn inserting_a_collinear_sample_does_not_change_integrated_lethality() {
    let unsegmented = ThermalTrajectory::new(vec![(0.0, 30.0), (10.0, 200.0)]);
    let segmented = ThermalTrajectory::new(vec![
        (0.0, 30.0),
        (2.5, 72.5),
        (5.0, 115.0),
        (7.5, 157.5),
        (10.0, 200.0),
    ]);

    assert_relative_close(
        model(&segmented),
        model(&unsegmented),
        2e-14,
        "collinear resampling invariance",
    );
}

#[test]
fn duration_scaling_scales_integrated_lethality_linearly() {
    let original = ThermalTrajectory::new(vec![(0.0, 40.0), (3.0, 80.0), (8.0, 120.0)]);
    let scaled = ThermalTrajectory::new(vec![(0.0, 40.0), (9.0, 80.0), (24.0, 120.0)]);

    assert_relative_close(
        model(&scaled),
        3.0 * model(&original),
        2e-14,
        "threefold duration scaling",
    );
}

#[test]
fn shifting_every_temperature_by_one_z_value_scales_lethality_tenfold() {
    let baseline = ThermalTrajectory::new(vec![(0.0, 40.0), (3.0, 80.0), (8.0, 120.0)]);
    let warmer = ThermalTrajectory::new(vec![(0.0, 40.0 + Z), (3.0, 80.0 + Z), (8.0, 120.0 + Z)]);

    assert_relative_close(
        model(&warmer),
        10.0 * model(&baseline),
        2e-13,
        "temperature shift by z",
    );
}

#[test]
fn subdividing_a_ramp_into_many_collinear_segments_remains_stable() {
    let single = ThermalTrajectory::new(vec![(0.0, 30.0), (10.0, 250.0)]);
    let mut many_points = Vec::with_capacity(1001);
    for i in 0..=1000 {
        let fraction = i as f64 / 1000.0;
        many_points.push((10.0 * fraction, 30.0 + (250.0 - 30.0) * fraction));
    }
    let subdivided = ThermalTrajectory::new(many_points);

    assert_relative_close(
        model(&subdivided),
        model(&single),
        5e-13,
        "1000-way collinear subdivision",
    );
}

#[test]
fn warmer_complete_trajectory_has_greater_lethality_for_the_same_d_z_model() {
    let cooler = ThermalTrajectory::new(vec![(0.0, 45.0), (2.0, 60.0), (8.0, 70.0)]);
    let warmer = ThermalTrajectory::new(vec![(0.0, 50.0), (2.0, 65.0), (8.0, 75.0)]);

    assert!(
        model(&warmer) > model(&cooler),
        "warmer trajectory must not deliver less modelled lethality"
    );
}
