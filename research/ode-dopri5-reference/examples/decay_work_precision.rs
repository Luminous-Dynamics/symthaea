//! Multi-regime analytical work–precision comparison for exponential decay.
//!
//! The decay family checks transient and strongly damped behavior in addition
//! to the smooth oscillatory fixture. It compares Symthaea's scale-aware
//! Cash–Karp method against ode_solvers 0.6.2 Dopri5 under the same declared
//! SI/unitless scalar state, initial scale, final time, rtol, and atol.
//! The receipt excludes wall-clock time, preserving deterministic output.

use std::env;
use std::error::Error;

use ode_solvers::{Dopri5, System, Vector2};
use ode_verify_001_harness::dynamical_system::{
    DynamicalSystem, Rk45Integrator, Rk45Tolerances,
};

const T_END: f64 = 1.0;
const INITIAL_DT: f64 = 0.05;
const DENSE_OUTPUT_DT: f64 = 0.01;
const RATES: [f64; 3] = [1.0, 10.0, 50.0];
const SCALES: [f64; 3] = [1.0e-6, 1.0, 1.0e6];
const RELATIVE_TOLERANCES: [f64; 3] = [1.0e-3, 1.0e-5, 1.0e-7];
const GLOBAL_ERROR_MULTIPLIER: f64 = 10.0;

/// y' = -rate*y; z' = 0. The second state is a zero-valued sentinel so both
/// solvers use the same fixed two-component shape as the oscillator comparison.
struct ExponentialDecay {
    rate: f64,
}

impl DynamicalSystem for ExponentialDecay {
    fn state_dim(&self) -> usize {
        2
    }

    fn derivatives(&self, state: &[f64], _t: f64) -> Vec<f64> {
        vec![-self.rate * state[0], 0.0]
    }

    fn name(&self) -> &str {
        "ExponentialDecay"
    }
}

struct ExponentialDecayReference {
    rate: f64,
}

impl System<f64, Vector2<f64>> for ExponentialDecayReference {
    fn system(&self, _time: f64, state: &Vector2<f64>, derivative: &mut Vector2<f64>) {
        derivative[0] = -self.rate * state[0];
        derivative[1] = 0.0;
    }
}

#[derive(Debug)]
struct TrajectoryMetrics {
    max_scaled_absolute_error: f64,
    final_scaled_absolute_error: f64,
    max_monotonicity_increase_scaled: f64,
    monotonicity_violations: usize,
    nonnegative_within_tolerance: bool,
    finite: bool,
}

fn evaluate_samples(
    samples: &[(f64, [f64; 2])],
    rate: f64,
    scale: f64,
    atol: f64,
) -> TrajectoryMetrics {
    let mut max_scaled_absolute_error = 0.0_f64;
    let mut max_monotonicity_increase_scaled = 0.0_f64;
    let mut monotonicity_violations = 0usize;
    let mut nonnegative_within_tolerance = true;
    let mut finite = !samples.is_empty();
    let slack = 10.0 * atol;
    let mut previous: Option<f64> = None;

    for (time, state) in samples {
        let exact_y = scale * (-rate * *time).exp();
        let scaled_error = (state[0] - exact_y).abs() / scale;
        let sentinel_error = state[1].abs() / scale;
        if !time.is_finite()
            || !state[0].is_finite()
            || !state[1].is_finite()
            || !scaled_error.is_finite()
            || !sentinel_error.is_finite()
        {
            finite = false;
        }
        max_scaled_absolute_error =
            max_scaled_absolute_error.max(scaled_error).max(sentinel_error);
        if state[0] < -slack {
            nonnegative_within_tolerance = false;
        }
        if let Some(prev) = previous {
            let increase = state[0] - prev;
            if increase > slack {
                monotonicity_violations += 1;
            }
            max_monotonicity_increase_scaled =
                max_monotonicity_increase_scaled.max(increase.max(0.0) / scale);
        }
        previous = Some(state[0]);
    }

    let final_scaled_absolute_error = samples
        .last()
        .map(|(time, state)| {
            (state[0] - scale * (-rate * *time).exp()).abs() / scale
        })
        .unwrap_or(f64::INFINITY);

    TrajectoryMetrics {
        max_scaled_absolute_error,
        final_scaled_absolute_error,
        max_monotonicity_increase_scaled,
        monotonicity_violations,
        nonnegative_within_tolerance,
        finite,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let source_sha = env::var("ODE_BENCH_SOURCE_SHA").unwrap_or_else(|_| "unknown".to_string());
    let checkout_sha =
        env::var("ODE_BENCH_CHECKOUT_SHA").unwrap_or_else(|_| "unknown".to_string());
    let lock_sha = env::var("ODE_BENCH_LOCK_SHA").unwrap_or_else(|_| "unknown".to_string());

    let integrator = Rk45Integrator::new(1.0e-8);
    let mut observations = Vec::new();
    let mut all_passed = true;

    for rate in RATES {
        for scale in SCALES {
            for rtol in RELATIVE_TOLERANCES {
                let atol = scale * rtol * 1.0e-3;
                let global_error_limit = GLOBAL_ERROR_MULTIPLIER * rtol;
                let tolerances = Rk45Tolerances::new(rtol, vec![atol; 2]);
                let custom_system = ExponentialDecay { rate };

                let custom_fields = match integrator.try_simulate_with_tolerances(
                    &custom_system,
                    &[scale, 0.0],
                    0.0,
                    T_END,
                    INITIAL_DT,
                    &tolerances,
                ) {
                    Ok(report) => {
                        let samples: Vec<(f64, [f64; 2])> = report
                            .simulation
                            .times
                            .iter()
                            .zip(report.simulation.states.iter())
                            .map(|(time, state)| (*time, [state[0], state[1]]))
                            .collect();
                        let metrics = evaluate_samples(&samples, rate, scale, atol);
                        let completed = report.simulation.times.last().is_some_and(|time| {
                            (*time - T_END).abs() <= f64::EPSILON * T_END.max(1.0)
                        });
                        let error_within_limit =
                            metrics.max_scaled_absolute_error <= global_error_limit;
                        let pass = completed
                            && metrics.finite
                            && metrics.nonnegative_within_tolerance
                            && metrics.monotonicity_violations == 0
                            && error_within_limit
                            && report.diagnostics.max_error_norm <= 1.0;
                        all_passed &= pass;
                        format!(
                            "\"custom_max_scaled_trajectory_error\":{:.17e},\"custom_final_scaled_error\":{:.17e},\"custom_max_monotonicity_increase_scaled\":{:.17e},\"custom_monotonicity_violations\":{},\"custom_nonnegative_within_tolerance\":{},\"custom_accepted_steps\":{},\"custom_rejected_steps\":{},\"custom_rhs_evaluations\":{},\"custom_max_error_norm\":{:.17e},\"custom_complete\":{},\"custom_finite\":{},\"custom_error_within_limit\":{},\"custom_pass\":{}",
                            metrics.max_scaled_absolute_error,
                            metrics.final_scaled_absolute_error,
                            metrics.max_monotonicity_increase_scaled,
                            metrics.monotonicity_violations,
                            metrics.nonnegative_within_tolerance,
                            report.diagnostics.accepted_steps,
                            report.diagnostics.rejected_steps,
                            report.diagnostics.derivative_evaluations,
                            report.diagnostics.max_error_norm,
                            completed,
                            metrics.finite,
                            error_within_limit,
                            pass
                        )
                    }
                    Err(error) => {
                        all_passed = false;
                        format!("\"custom_pass\":false,\"custom_error\":\"{:?}\"", error)
                    }
                };

                let mut dopri5 = Dopri5::new(
                    ExponentialDecayReference { rate },
                    0.0,
                    T_END,
                    DENSE_OUTPUT_DT,
                    Vector2::new(scale, 0.0),
                    rtol,
                    atol,
                );
                let reference_fields = match dopri5.integrate() {
                    Ok(stats) => {
                        let times = dopri5.x_out();
                        let states = dopri5.y_out();
                        let samples: Vec<(f64, [f64; 2])> = times
                            .iter()
                            .zip(states.iter())
                            .map(|(time, state)| (*time, [state[0], state[1]]))
                            .collect();
                        let metrics = evaluate_samples(&samples, rate, scale, atol);
                        let completed = times.last().is_some_and(|time| {
                            (*time - T_END).abs() <= f64::EPSILON * T_END.max(1.0)
                        });
                        let error_within_limit =
                            metrics.max_scaled_absolute_error <= global_error_limit;
                        let pass = completed
                            && metrics.finite
                            && metrics.nonnegative_within_tolerance
                            && metrics.monotonicity_violations == 0
                            && error_within_limit;
                        all_passed &= pass;
                        format!(
                            "\"dopri5_max_scaled_trajectory_error\":{:.17e},\"dopri5_final_scaled_error\":{:.17e},\"dopri5_max_monotonicity_increase_scaled\":{:.17e},\"dopri5_monotonicity_violations\":{},\"dopri5_nonnegative_within_tolerance\":{},\"dopri5_accepted_steps\":{},\"dopri5_rejected_steps\":{},\"dopri5_rhs_evaluations\":{},\"dopri5_output_samples\":{},\"dopri5_complete\":{},\"dopri5_finite\":{},\"dopri5_error_within_limit\":{},\"dopri5_pass\":{}",
                            metrics.max_scaled_absolute_error,
                            metrics.final_scaled_absolute_error,
                            metrics.max_monotonicity_increase_scaled,
                            metrics.monotonicity_violations,
                            metrics.nonnegative_within_tolerance,
                            stats.accepted_steps,
                            stats.rejected_steps,
                            stats.num_eval,
                            samples.len(),
                            completed,
                            metrics.finite,
                            error_within_limit,
                            pass
                        )
                    }
                    Err(error) => {
                        all_passed = false;
                        format!("\"dopri5_pass\":false,\"dopri5_error\":\"{:?}\"", error)
                    }
                };

                observations.push(format!(
                    "{{\"rate\":{:.17e},\"scale\":{:.17e},\"rtol\":{:.17e},\"atol\":{:.17e},\"global_scaled_trajectory_error_limit\":{:.17e},{},{}}}",
                    rate, scale, rtol, atol, global_error_limit, custom_fields, reference_fields
                ));
            }
        }
    }

    println!(
        "{{\"schema_version\":\"0.1.0\",\"benchmark_id\":\"ODE-WP-DECAY-DOPRI5-001\",\"source_sha\":\"{}\",\"checkout_sha\":\"{}\",\"lock_sha\":\"{}\",\"custom_solver\":\"Symthaea Cash-Karp RK45, scale-aware weighted RMS\",\"reference_solver\":\"ode_solvers 0.6.2 Dopri5 (Dormand-Prince 5(4))\",\"reference_equation\":\"y'=-rate*y; z'=0\",\"t_end\":{:.17e},\"dense_output_dt\":{:.17e},\"observations\":[",
        source_sha, checkout_sha, lock_sha, T_END, DENSE_OUTPUT_DT
    );
    for (index, observation) in observations.iter().enumerate() {
        if index > 0 {
            println!(",");
        }
        print!("{observation}");
    }
    println!("]}}");

    if !all_passed {
        return Err("one or more matched exponential-decay comparison cases failed".into());
    }
    Ok(())
}
