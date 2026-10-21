//! Matched analytical work–precision comparison against ode_solvers 0.6.2.
//!
//! The reference method is Dormand–Prince 5(4) (Dopri5), not a claim of universal
//! solver superiority. This benchmark compares endpoint error and evaluation work
//! for the same scaled harmonic oscillator at the same declared rtol / atol.
//! It intentionally excludes wall-clock timing from the deterministic receipt.

use std::env;
use std::error::Error;

use ode_solvers::{Dopri5, System, Vector2};
use ode_verify_001_harness::dynamical_system::{
    HarmonicOscillator, Rk45Integrator, Rk45Tolerances,
};

const T_END: f64 = 10.0;
const INITIAL_DT: f64 = 0.2;
const DENSE_OUTPUT_DT: f64 = 0.1;
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

fn scaled_endpoint_error(state: &[f64], scale: f64, t_end: f64) -> f64 {
    let exact = [scale * t_end.cos(), -scale * t_end.sin()];
    ((state[0] - exact[0]).powi(2) + (state[1] - exact[1]).powi(2)).sqrt() / scale
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

            let custom_fields = match custom {
                Ok(report) => {
                    let endpoint = report.simulation.final_state();
                    let endpoint_error = scaled_endpoint_error(endpoint, scale, T_END);
                    let energy_drift = max_relative_energy_drift(
                        report
                            .simulation
                            .states
                            .iter()
                            .map(|state| [state[0], state[1]]),
                        scale,
                    );
                    let completed =
                        report.simulation.times.last().is_some_and(|time| {
                            (*time - T_END).abs() <= f64::EPSILON * T_END.max(1.0)
                        });
                    let finite = endpoint_error.is_finite()
                        && energy_drift.is_finite()
                        && report
                            .simulation
                            .states
                            .iter()
                            .flatten()
                            .all(|value| value.is_finite());
                    let pass = completed
                        && finite
                        && endpoint_error <= endpoint_error_limit
                        && energy_drift <= energy_drift_limit
                        && report.diagnostics.max_error_norm <= 1.0;
                    all_passed &= pass;
                    format!(
                        "\"custom_endpoint_error_scaled\":{:.17e},\"custom_energy_drift\":{:.17e},\"custom_accepted_steps\":{},\"custom_rejected_steps\":{},\"custom_rhs_evaluations\":{},\"custom_max_error_norm\":{:.17e},\"custom_complete\":{},\"custom_finite\":{},\"custom_pass\":{}",
                        endpoint_error,
                        energy_drift,
                        report.diagnostics.accepted_steps,
                        report.diagnostics.rejected_steps,
                        report.diagnostics.derivative_evaluations,
                        report.diagnostics.max_error_norm,
                        completed,
                        finite,
                        pass
                    )
                }
                Err(error) => {
                    all_passed = false;
                    format!("\"custom_pass\":false,\"custom_error\":\"{:?}\"", error)
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
            let reference = dopri5.integrate();

            let reference_fields = match reference {
                Ok(stats) => {
                    let reference_states = dopri5.y_out();
                    let reference_times = dopri5.x_out();
                    let Some(endpoint) = reference_states.last() else {
                        all_passed = false;
                        observations.push(format!(
                            "{{\"scale\":{:.17e},\"rtol\":{:.17e},\"atol\":{:.17e},\"global_endpoint_error_limit\":{:.17e},\"global_energy_drift_limit\":{:.17e},{},\"dopri5_pass\":false,\"dopri5_error\":\"no_output_states\"}}",
                            scale,
                            rtol,
                            atol,
                            endpoint_error_limit,
                            energy_drift_limit,
                            custom_fields
                        ));
                        continue;
                    };
                    let endpoint_time = reference_times.last().copied().unwrap_or(f64::NAN);
                    let endpoint_error =
                        scaled_endpoint_error(&[endpoint[0], endpoint[1]], scale, T_END);
                    let energy_drift = max_relative_energy_drift(
                        reference_states.iter().map(|state| [state[0], state[1]]),
                        scale,
                    );
                    let completed = endpoint_time.is_finite()
                        && (endpoint_time - T_END).abs() <= f64::EPSILON * T_END.max(1.0);
                    let finite = endpoint_error.is_finite()
                        && energy_drift.is_finite()
                        && reference_states
                            .iter()
                            .all(|state| state.iter().all(|value| value.is_finite()));
                    let pass = completed
                        && finite
                        && endpoint_error <= endpoint_error_limit
                        && energy_drift <= energy_drift_limit;
                    all_passed &= pass;
                    format!(
                        "\"dopri5_endpoint_error_scaled\":{:.17e},\"dopri5_sampled_energy_drift\":{:.17e},\"dopri5_accepted_steps\":{},\"dopri5_rejected_steps\":{},\"dopri5_rhs_evaluations\":{},\"dopri5_output_samples\":{},\"dopri5_complete\":{},\"dopri5_finite\":{},\"dopri5_pass\":{}",
                        endpoint_error,
                        energy_drift,
                        stats.accepted_steps,
                        stats.rejected_steps,
                        stats.num_eval,
                        reference_states.len(),
                        completed,
                        finite,
                        pass
                    )
                }
                Err(error) => {
                    all_passed = false;
                    format!("\"dopri5_pass\":false,\"dopri5_error\":\"{:?}\"", error)
                }
            };

            observations.push(format!(
                "{{\"scale\":{:.17e},\"rtol\":{:.17e},\"atol\":{:.17e},\"global_endpoint_error_limit\":{:.17e},\"global_energy_drift_limit\":{:.17e},{},{}}}",
                scale, rtol, atol, endpoint_error_limit, energy_drift_limit, custom_fields, reference_fields
            ));
        }
    }

    println!(
        "{{\"schema_version\":\"0.1.0\",\"benchmark_id\":\"ODE-WP-DOPRI5-COMPARISON-001\",\"source_sha\":\"{}\",\"checkout_sha\":\"{}\",\"lock_sha\":\"{}\",\"custom_solver\":\"Symthaea Cash-Karp RK45, scale-aware weighted RMS\",\"reference_solver\":\"ode_solvers 0.6.2 Dopri5 (Dormand-Prince 5(4))\",\"reference_equation\":\"x'=v; v'=-x\",\"t_end\":{:.17e},\"observations\":[",
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
