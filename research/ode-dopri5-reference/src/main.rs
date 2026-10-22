//! Matched analytical work–precision comparison against ode_solvers 0.6.2.
//!
//! The reference method is Dormand–Prince 5(4) (Dopri5), not a claim of universal
//! solver superiority. Both solvers are evaluated on one frozen 101-point output grid
//! and compared against the exact oscillator, including pointwise solver disagreement.
//! It intentionally excludes wall-clock timing from the deterministic receipt.

use std::env;
use std::error::Error;

use ode_solvers::{Dopri5, OutputType, System, Vector2};
use ode_verify_001_harness::dynamical_system::{HarmonicOscillator, Rk45Integrator, Rk45Tolerances};

const T_END: f64 = 10.0;
const INITIAL_DT: f64 = 0.2;
const DENSE_OUTPUT_DT: f64 = 0.1;
const COMMON_OUTPUT_INTERVALS: usize = 100;
const SCALES: [f64; 3] = [1.0e-6, 1.0, 1.0e6];
const RELATIVE_TOLERANCES: [f64; 3] = [1.0e-3, 1.0e-5, 1.0e-7];
const GLOBAL_ERROR_MULTIPLIER: f64 = 10.0;

/// x' = v, v' = -x — identical to Symthaea's simple harmonic oscillator.
struct UnitHarmonicOscillator;

impl System<f64, Vector2<f64>> for UnitHarmonicOscillator {
    fn system(&self, _time: f64, state: &Vector2<f64>, derivative: &mut Vector2<f64>) {
        derivative[0] = state[1];
        derivative[1] = -state[0];
    }
}

fn scaled_state_error(state: [f64; 2], scale: f64, time: f64) -> f64 {
    let exact = [scale * time.cos(), -scale * time.sin()];
    ((state[0] - exact[0]).powi(2) + (state[1] - exact[1]).powi(2)).sqrt() / scale
}

fn max_sampled_scaled_trajectory_error(
    samples: impl Iterator<Item = (f64, [f64; 2])>,
    scale: f64,
) -> f64 {
    samples
        .map(|(time, state)| scaled_state_error(state, scale, time))
        .fold(0.0_f64, f64::max)
}

fn max_relative_energy_drift(states: impl Iterator<Item = [f64; 2]>, scale: f64) -> f64 {
    let initial_energy = 0.5 * scale * scale;
    states
        .map(|state| {
            let energy = 0.5 * (state[0] * state[0] + state[1] * state[1]);
            (energy - initial_energy).abs() / initial_energy
        })
        .fold(0.0_f64, f64::max)
}

/// Resample an adaptive Cash–Karp trajectory at the same times as Dopri5's
/// dense-output grid. The quintic Hermite interpolant uses the harmonic oscillator's
/// known first and second derivatives. This is benchmark post-processing, not production
/// dense output, and its arithmetic is not counted as integration RHS work.
fn quintic_hermite_sample_grid(
    times: &[f64],
    states: &[Vec<f64>],
) -> Option<Vec<(f64, [f64; 2])>> {
    if times.len() != states.len()
        || times.len() < 2
        || times[0].abs() > 32.0 * f64::EPSILON
        || (times[times.len() - 1] - T_END).abs()
            > 32.0 * f64::EPSILON * T_END.abs().max(1.0)
        || states
            .iter()
            .any(|state| state.len() != 2 || state.iter().any(|value| !value.is_finite()))
        || times.iter().any(|time| !time.is_finite())
        || times.windows(2).any(|window| window[1] <= window[0])
    {
        return None;
    }

    let mut samples = Vec::with_capacity(COMMON_OUTPUT_INTERVALS + 1);
    for grid_index in 0..=COMMON_OUTPUT_INTERVALS {
        let target = T_END * grid_index as f64 / COMMON_OUTPUT_INTERVALS as f64;
        if grid_index == 0 {
            samples.push((target, [states[0][0], states[0][1]]));
            continue;
        }
        if grid_index == COMMON_OUTPUT_INTERVALS {
            let last = states.last()?;
            samples.push((target, [last[0], last[1]]));
            continue;
        }

        let interval = (0..times.len() - 1)
            .find(|index| times[*index] <= target && target <= times[*index + 1])?;
        let t0 = times[interval];
        let t1 = times[interval + 1];
        let h = t1 - t0;
        if !h.is_finite() || h <= 0.0 {
            return None;
        }
        let s = (target - t0) / h;
        if !s.is_finite() || !(0.0..=1.0).contains(&s) {
            return None;
        }

        let y0 = &states[interval];
        let y1 = &states[interval + 1];
        let s2 = s * s;
        let s3 = s2 * s;
        let s4 = s3 * s;
        let s5 = s4 * s;
        let h00 = 1.0 - 10.0 * s3 + 15.0 * s4 - 6.0 * s5;
        let h10 = s - 6.0 * s3 + 8.0 * s4 - 3.0 * s5;
        let h20 = 0.5 * (s2 - 3.0 * s3 + 3.0 * s4 - s5);
        let h01 = 10.0 * s3 - 15.0 * s4 + 6.0 * s5;
        let h11 = -4.0 * s3 + 7.0 * s4 - 3.0 * s5;
        let h21 = 0.5 * (s3 - 2.0 * s4 + s5);

        let x0 = y0[0];
        let v0 = y0[1];
        let x1 = y1[0];
        let v1 = y1[1];
        let h_squared = h * h;
        let x = h00 * x0
            + h10 * h * v0
            + h20 * h_squared * (-x0)
            + h01 * x1
            + h11 * h * v1
            + h21 * h_squared * (-x1);
        let v = h00 * v0
            + h10 * h * (-x0)
            + h20 * h_squared * (-v0)
            + h01 * v1
            + h11 * h * (-x1)
            + h21 * h_squared * (-v1);
        if !x.is_finite() || !v.is_finite() {
            return None;
        }
        samples.push((target, [x, v]));
    }
    Some(samples)
}

fn max_common_grid_disagreement(
    custom: &[(f64, [f64; 2])],
    reference: &[(f64, [f64; 2])],
    scale: f64,
) -> Option<f64> {
    if custom.len() != COMMON_OUTPUT_INTERVALS + 1 || reference.len() != custom.len() {
        return None;
    }
    let time_tolerance = 32.0 * f64::EPSILON * T_END.abs().max(1.0);
    let mut maximum = 0.0_f64;
    for ((custom_time, custom_state), (reference_time, reference_state)) in
        custom.iter().zip(reference.iter())
    {
        if !custom_time.is_finite()
            || !reference_time.is_finite()
            || (custom_time - reference_time).abs() > time_tolerance
        {
            return None;
        }
        let error = ((custom_state[0] - reference_state[0]).powi(2)
            + (custom_state[1] - reference_state[1]).powi(2))
        .sqrt()
            / scale;
        if !error.is_finite() {
            return None;
        }
        maximum = maximum.max(error);
    }
    Some(maximum)
}

fn format_optional_f64(value: Option<f64>) -> String {
    value
        .filter(|number| number.is_finite())
        .map(|number| format!("{number:.17e}"))
        .unwrap_or_else(|| "null".to_owned())
}

fn main() -> Result<(), Box<dyn Error>> {
    let source_sha = env::var("ODE_BENCH_SOURCE_SHA").unwrap_or_else(|_| "unknown".to_string());
    let checkout_sha = env::var("ODE_BENCH_CHECKOUT_SHA").unwrap_or_else(|_| "unknown".to_string());
    let lock_sha = env::var("ODE_BENCH_LOCK_SHA").unwrap_or_else(|_| "unknown".to_string());

    let custom_integrator = Rk45Integrator::new(1.0e-8);
    let custom_system = HarmonicOscillator::simple(1.0);
    let mut observations = Vec::new();
    let mut all_passed = true;

    for scale in SCALES {
        for rtol in RELATIVE_TOLERANCES {
            let atol = scale * rtol * 1.0e-3;
            let endpoint_error_limit = GLOBAL_ERROR_MULTIPLIER * rtol;
            let energy_drift_limit = GLOBAL_ERROR_MULTIPLIER * rtol;
            let tolerances = Rk45Tolerances::new(rtol, vec![atol; 2]);

            let custom = custom_integrator.try_simulate_with_tolerances(
                &custom_system,
                &[scale, 0.0],
                0.0,
                T_END,
                INITIAL_DT,
                &tolerances,
            );
            let mut custom_samples: Option<Vec<(f64, [f64; 2])>> = None;

            let custom_fields = match custom {
                Ok(report) => match quintic_hermite_sample_grid(
                    &report.simulation.times,
                    &report.simulation.states,
                ) {
                    Some(samples) => {
                        let endpoint = samples.last().map(|sample| sample.1).unwrap_or([f64::NAN; 2]);
                        let endpoint_error = scaled_state_error(endpoint, scale, T_END);
                        let trajectory_error = max_sampled_scaled_trajectory_error(
                            samples.iter().copied(),
                            scale,
                        );
                        let energy_drift = max_relative_energy_drift(
                            samples.iter().map(|sample| sample.1),
                            scale,
                        );
                        let common_grid_complete = samples.len() == COMMON_OUTPUT_INTERVALS + 1;
                        let finite = samples.iter().all(|(_, state)| {
                            state.iter().all(|value| value.is_finite())
                        }) && report.simulation.states.iter().flatten().all(|value| value.is_finite());
                        let pass = common_grid_complete
                            && finite
                            && endpoint_error <= endpoint_error_limit
                            && trajectory_error <= endpoint_error_limit
                            && energy_drift <= energy_drift_limit
                            && report.diagnostics.max_error_norm <= 1.0;
                        all_passed &= pass;
                        custom_samples = Some(samples.clone());
                        format!(
                            "\"custom_endpoint_error_scaled\":{:.17e},\"custom_max_sampled_scaled_trajectory_error\":{:.17e},\"custom_energy_drift\":{:.17e},\"custom_accepted_steps\":{},\"custom_rejected_steps\":{},\"custom_rhs_evaluations\":{},\"custom_max_error_norm\":{:.17e},\"custom_output_samples\":{},\"custom_common_grid_complete\":{},\"custom_integration_steps\":{},\"custom_integration_samples\":{},\"custom_finite\":{},\"custom_pass\":{}",
                            endpoint_error,
                            trajectory_error,
                            energy_drift,
                            report.diagnostics.accepted_steps,
                            report.diagnostics.rejected_steps,
                            report.diagnostics.derivative_evaluations,
                            report.diagnostics.max_error_norm,
                            samples.len(),
                            common_grid_complete,
                            report.diagnostics.accepted_steps,
                            report.simulation.states.len(),
                            finite,
                            pass
                        )
                    }
                    None => {
                        all_passed = false;
                        format!(
                            "\"custom_common_grid_complete\":false,\"custom_pass\":false,\"custom_error\":\"trajectory_resampling_failed\""
                        )
                    }
                },
                Err(error) => {
                    all_passed = false;
                    format!(
                        "\"custom_common_grid_complete\":false,\"custom_pass\":false,\"custom_error\":\"{:?}\"",
                        error
                    )
                }
            };

            let y0 = Vector2::new(scale, 0.0);
            let mut dopri5 = Dopri5::new(
                UnitHarmonicOscillator,
                0.0,
                T_END,
                DENSE_OUTPUT_DT,
                y0,
                rtol,
                atol,
            );
            // Keep the reference method's adaptive accepted steps. Dense-output interval
            // is irrelevant in Sparse mode; both methods are reconstructed with the
            // same harness-only quintic-Hermite interpolation below.
            dopri5.set_output(OutputType::Sparse);
            let reference = dopri5.integrate();

            let reference_fields = match reference {
                Ok(stats) => {
                    let raw_reference_states = dopri5.y_out();
                    let raw_reference_times = dopri5.x_out();
                    let endpoint_time = raw_reference_times.last().copied().unwrap_or(f64::NAN);
                    let terminal_time_tolerance = 32.0 * f64::EPSILON * T_END.abs().max(1.0);
                    let integration_completed = endpoint_time.is_finite()
                        && (endpoint_time - T_END).abs() <= terminal_time_tolerance;
                    let raw_reference_states_vec: Vec<Vec<f64>> = raw_reference_states
                        .iter()
                        .map(|state| vec![state[0], state[1]])
                        .collect();
                    let interpolated_reference = quintic_hermite_sample_grid(
                        raw_reference_times,
                        &raw_reference_states_vec,
                    );
                    let mut common_grid_matches = false;
                    let mut max_common_grid_disagreement = None;
                    let mut endpoint_error = f64::MAX;
                    let mut trajectory_error = f64::MAX;
                    let mut energy_drift = f64::MAX;
                    let mut finite = false;
                    let mut common_grid_samples = 0usize;
                    let mut pass = false;

                    if let Some(reference_samples) = interpolated_reference {
                        common_grid_samples = reference_samples.len();
                        if let Some(endpoint) = reference_samples.last().map(|sample| sample.1) {
                            endpoint_error = scaled_state_error(endpoint, scale, T_END);
                        }
                        trajectory_error = max_sampled_scaled_trajectory_error(
                            reference_samples.iter().copied(),
                            scale,
                        );
                        energy_drift = max_relative_energy_drift(
                            reference_samples.iter().map(|sample| sample.1),
                            scale,
                        );
                        max_common_grid_disagreement = custom_samples.as_deref().and_then(|custom| {
                            max_common_grid_disagreement(custom, &reference_samples, scale)
                        });
                        common_grid_matches = max_common_grid_disagreement.is_some()
                            && common_grid_samples == COMMON_OUTPUT_INTERVALS + 1;
                        finite = endpoint_error.is_finite()
                            && trajectory_error.is_finite()
                            && energy_drift.is_finite()
                            && raw_reference_states
                                .iter()
                                .all(|state| state.iter().all(|value| value.is_finite()));
                        pass = integration_completed
                            && finite
                            && common_grid_matches
                            && endpoint_error <= endpoint_error_limit
                            && trajectory_error <= endpoint_error_limit
                            && max_common_grid_disagreement
                                .is_some_and(|error| error <= 2.0 * endpoint_error_limit)
                            && energy_drift <= energy_drift_limit;
                    }
                    all_passed &= pass;
                    format!(
                        "\"dopri5_terminal_time\":{:.17e},\"dopri5_terminal_time_error\":{:.17e},\"dopri5_terminal_time_tolerance\":{:.17e},\"dopri5_endpoint_error_scaled\":{:.17e},\"dopri5_max_sampled_scaled_trajectory_error\":{:.17e},\"dopri5_sampled_energy_drift\":{:.17e},\"dopri5_accepted_steps\":{},\"dopri5_rejected_steps\":{},\"dopri5_rhs_evaluations\":{},\"dopri5_integration_samples\":{},\"dopri5_output_samples\":{},\"common_grid_matches\":{},\"max_common_grid_disagreement_scaled\":{},\"dopri5_complete\":{},\"dopri5_finite\":{},\"dopri5_pass\":{}",
                        endpoint_time,
                        (endpoint_time - T_END).abs(),
                        terminal_time_tolerance,
                        endpoint_error,
                        trajectory_error,
                        energy_drift,
                        stats.accepted_steps,
                        stats.rejected_steps,
                        stats.num_eval,
                        raw_reference_states.len(),
                        common_grid_samples,
                        common_grid_matches,
                        format_optional_f64(max_common_grid_disagreement),
                        integration_completed,
                        finite,
                        pass
                    )
                }
                Err(error) => {
                    all_passed = false;
                    format!(
                        "\"dopri5_integration_samples\":0,\"dopri5_output_samples\":0,\"common_grid_matches\":false,\"max_common_grid_disagreement_scaled\":null,\"dopri5_pass\":false,\"dopri5_error\":\"{:?}\"",
                        error
                    )
                }
            };

            observations.push(format!(
                "{{\"scale\":{:.17e},\"rtol\":{:.17e},\"atol\":{:.17e},\"global_endpoint_error_limit\":{:.17e},\"global_energy_drift_limit\":{:.17e},{},{}}}",
                scale, rtol, atol, endpoint_error_limit, energy_drift_limit, custom_fields, reference_fields
            ));
        }
    }

    println!(
        "{{\"schema_version\":\"0.1.0\",\"benchmark_id\":\"ODE-WP-DOPRI5-COMPARISON-001\",\"source_sha\":\"{}\",\"checkout_sha\":\"{}\",\"lock_sha\":\"{}\",\"custom_solver\":\"Symthaea Cash-Karp RK45, scale-aware weighted RMS\",\"reference_solver\":\"ode_solvers 0.6.2 Dopri5 (Dormand-Prince 5(4))\",\"reference_equation\":\"x'=v; v'=-x\",\"common_output_grid\":{{\"start\":0.0,\"end\":10.0,\"intervals\":100,\"step\":0.1,\"samples\":101}},\"t_end\":{:.17e},\"observations\":[",
        source_sha, checkout_sha, lock_sha, T_END
    );
    for (index, observation) in observations.iter().enumerate() {
        if index > 0 {
            println!(",");
        }
        print!("{observation}");
    }
    println!("]}}");

    if !all_passed {
        return Err("one or more matched analytical solver-comparison cases failed".into());
    }
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quintic_hermite_resampling_emits_the_frozen_grid() {
        let mut times = Vec::new();
        let mut states = Vec::new();
        for index in 0..=40 {
            let time = T_END * index as f64 / 40.0;
            times.push(time);
            states.push(vec![time.cos(), -time.sin()]);
        }

        let samples = quintic_hermite_sample_grid(&times, &states).unwrap();
        assert_eq!(samples.len(), COMMON_OUTPUT_INTERVALS + 1);
        assert_eq!(samples.first().unwrap().0, 0.0);
        assert_eq!(samples.last().unwrap().0, T_END);
        for (index, (time, _)) in samples.iter().enumerate() {
            let expected = T_END * index as f64 / COMMON_OUTPUT_INTERVALS as f64;
            assert!(
                (time - expected).abs() <= 4.0 * f64::EPSILON * T_END.max(1.0),
                "grid index {index}: got {time}, expected {expected}"
            );
        }

        // Quintic Hermite is a sixth-order local interpolation approximation on this smooth fixture.
        let maximum_error = max_sampled_scaled_trajectory_error(samples.iter().copied(), 1.0);
        assert!(maximum_error < 1.0e-7, "max interpolation error={maximum_error}");
    }

    #[test]
    fn common_grid_disagreement_rejects_mismatched_times_and_shapes() {
        let samples: Vec<(f64, [f64; 2])> = (0..=COMMON_OUTPUT_INTERVALS)
            .map(|index| {
                let time = T_END * index as f64 / COMMON_OUTPUT_INTERVALS as f64;
                (time, [time.cos(), -time.sin()])
            })
            .collect();
        assert_eq!(max_common_grid_disagreement(&samples, &samples, 1.0), Some(0.0));

        let mut wrong_time = samples.clone();
        wrong_time[20].0 += 1.0e-6;
        assert!(max_common_grid_disagreement(&samples, &wrong_time, 1.0).is_none());

        let truncated = &samples[..samples.len() - 1];
        assert!(max_common_grid_disagreement(&samples, truncated, 1.0).is_none());
    }

    #[test]
    fn quintic_hermite_resampling_rejects_malformed_adaptive_trajectories() {
        assert!(quintic_hermite_sample_grid(&[], &[]).is_none());
        assert!(quintic_hermite_sample_grid(&[0.0, 10.0], &[vec![0.0], vec![0.0]]).is_none());
        assert!(quintic_hermite_sample_grid(
            &[0.0, 10.0],
            &[vec![0.0, 0.0], vec![f64::NAN, 0.0]]
        )
        .is_none());
        assert!(quintic_hermite_sample_grid(
            &[0.0, 9.0],
            &[vec![0.0, 0.0], vec![0.0, 0.0]]
        )
        .is_none());
    }
}
