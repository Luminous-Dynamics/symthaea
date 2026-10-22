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
    DynamicalSystem, HarmonicOscillator, Rk45Controller, Rk45Integrator, Rk45Tolerances,
    SimulationError,
};

const T_END: f64 = 10.0;
const INITIAL_DT: f64 = 0.2;
const DENSE_OUTPUT_DT: f64 = 0.1;
const COMMON_OUTPUT_INTERVALS: usize = 100;
const MAX_CUSTOM_ACCEPTED_STEPS: usize = 1_000_000;
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

struct CustomTrajectory {
    samples: Vec<(f64, [f64; 2])>,
    accepted_steps: usize,
    rejected_steps: usize,
    rhs_evaluations: usize,
    max_accepted_error_norm: f64,
    completed: bool,
    finite: bool,
}

fn integrate_custom_on_common_grid(
    integrator: &Rk45Integrator,
    system: &dyn DynamicalSystem,
    scale: f64,
    tolerances: &Rk45Tolerances,
) -> Result<CustomTrajectory, SimulationError> {
    let mut controller = Rk45Controller::new(INITIAL_DT)?;
    let mut state = [scale, 0.0];
    let mut time = 0.0;
    let mut accepted_steps = 0usize;
    let mut rejected_steps = 0usize;
    let mut rhs_evaluations = 0usize;
    let mut max_accepted_error_norm = 0.0_f64;
    let mut finite = true;
    let mut samples = Vec::with_capacity(COMMON_OUTPUT_INTERVALS + 1);
    samples.push((time, state));

    for grid_index in 1..=COMMON_OUTPUT_INTERVALS {
        let target = T_END * grid_index as f64 / COMMON_OUTPUT_INTERVALS as f64;
        while time < target {
            if accepted_steps >= MAX_CUSTOM_ACCEPTED_STEPS {
                return Err(SimulationError::StepBudgetExceeded {
                    maximum_steps: MAX_CUSTOM_ACCEPTED_STEPS,
                });
            }

            let remaining = target - time;
            let step = controller.try_step(
                integrator,
                system,
                &state,
                time,
                remaining,
                tolerances,
            )?;
            if !step.accepted_dt.is_finite()
                || step.accepted_dt <= 0.0
                || step.accepted_dt > remaining
                || step.state.len() != 2
                || step.state.iter().any(|value| !value.is_finite())
            {
                return Err(SimulationError::InvalidIntegratorStepSize {
                    step: accepted_steps + 1,
                });
            }

            let previous_time = time;
            let next_time = if step.accepted_dt >= remaining {
                target
            } else {
                time + step.accepted_dt
            };
            if !next_time.is_finite() || next_time <= previous_time {
                return Err(SimulationError::TimeDidNotAdvance {
                    step: accepted_steps + 1,
                });
            }

            accepted_steps += 1;
            rejected_steps = rejected_steps.saturating_add(step.rejected_steps);
            rhs_evaluations = rhs_evaluations.saturating_add(step.derivative_evaluations);
            max_accepted_error_norm = max_accepted_error_norm.max(step.error_norm);
            state = [step.state[0], step.state[1]];
            finite &= state.iter().all(|value| value.is_finite());
            time = next_time;
        }
        time = target;
        samples.push((target, state));
    }

    Ok(CustomTrajectory {
        samples,
        accepted_steps,
        rejected_steps,
        rhs_evaluations,
        max_accepted_error_norm,
        completed: time == T_END,
        finite,
    })
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

            let custom = integrate_custom_on_common_grid(
                &custom_integrator,
                &custom_system,
                scale,
                &tolerances,
            );
            let mut custom_samples: Option<Vec<(f64, [f64; 2])>> = None;

            let custom_fields = match custom {
                Ok(report) => {
                    custom_samples = Some(report.samples.clone());
                    let endpoint = report.samples.last().map(|sample| sample.1).unwrap_or([f64::NAN; 2]);
                    let endpoint_error = scaled_state_error(endpoint, scale, T_END);
                    let trajectory_error = max_sampled_scaled_trajectory_error(
                        report.samples.iter().copied(),
                        scale,
                    );
                    let energy_drift = max_relative_energy_drift(
                        report.samples.iter().map(|sample| sample.1),
                        scale,
                    );
                    let common_grid_complete = report.samples.len() == COMMON_OUTPUT_INTERVALS + 1;
                    let pass = report.completed
                        && common_grid_complete
                        && report.finite
                        && endpoint_error <= endpoint_error_limit
                        && trajectory_error <= endpoint_error_limit
                        && energy_drift <= energy_drift_limit
                        && report.max_accepted_error_norm <= 1.0;
                    all_passed &= pass;
                    format!(
                        "\"custom_endpoint_error_scaled\":{:.17e},\"custom_max_sampled_scaled_trajectory_error\":{:.17e},\"custom_energy_drift\":{:.17e},\"custom_accepted_steps\":{},\"custom_rejected_steps\":{},\"custom_rhs_evaluations\":{},\"custom_max_error_norm\":{:.17e},\"custom_output_samples\":{},\"custom_common_grid_complete\":{},\"custom_complete\":{},\"custom_finite\":{},\"custom_pass\":{}",
                        endpoint_error,
                        trajectory_error,
                        energy_drift,
                        report.accepted_steps,
                        report.rejected_steps,
                        report.rhs_evaluations,
                        report.max_accepted_error_norm,
                        report.samples.len(),
                        common_grid_complete,
                        report.completed,
                        report.finite,
                        pass
                    )
                }
                Err(error) => {
                    all_passed = false;
                    format!("\"custom_common_grid_complete\":false,\"custom_pass\":false,\"custom_error\":\"{:?}\"", error)
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
                        scaled_state_error([endpoint[0], endpoint[1]], scale, T_END);
                    let reference_samples: Vec<(f64, [f64; 2])> = reference_times
                        .iter()
                        .copied()
                        .zip(reference_states.iter().map(|state| [state[0], state[1]]))
                        .collect();
                    let trajectory_error = max_sampled_scaled_trajectory_error(
                        reference_samples.iter().copied(),
                        scale,
                    );
                    let common_grid_disagreement = custom_samples.as_deref().and_then(|custom| {
                        max_common_grid_disagreement(custom, &reference_samples, scale)
                    });
                    let common_grid_matches = common_grid_disagreement.is_some();
                    let energy_drift = max_relative_energy_drift(
                        reference_samples.iter().map(|sample| sample.1),
                        scale,
                    );
                    // `ode_solvers` output times are accumulated floating-point samples;
                    // terminal-time validation needs a roundoff envelope, not a single ULP.
                    let terminal_time_tolerance = 32.0 * f64::EPSILON * T_END.abs().max(1.0);
                    let completed = endpoint_time.is_finite()
                        && (endpoint_time - T_END).abs() <= terminal_time_tolerance;
                    let finite = endpoint_error.is_finite()
                        && energy_drift.is_finite()
                        && reference_states
                            .iter()
                            .all(|state| state.iter().all(|value| value.is_finite()));
                    let pass = completed
                        && finite
                        && common_grid_matches
                        && endpoint_error <= endpoint_error_limit
                        && trajectory_error <= endpoint_error_limit
                        && common_grid_disagreement
                            .is_some_and(|error| error <= 2.0 * endpoint_error_limit)
                        && energy_drift <= energy_drift_limit;
                    all_passed &= pass;
                    format!(
                        "\"dopri5_terminal_time\":{:.17e},\"dopri5_terminal_time_error\":{:.17e},\"dopri5_terminal_time_tolerance\":{:.17e},\"dopri5_endpoint_error_scaled\":{:.17e},\"dopri5_max_sampled_scaled_trajectory_error\":{:.17e},\"dopri5_sampled_energy_drift\":{:.17e},\"dopri5_accepted_steps\":{},\"dopri5_rejected_steps\":{},\"dopri5_rhs_evaluations\":{},\"dopri5_output_samples\":{},\"common_grid_matches\":{},\"max_common_grid_disagreement_scaled\":{},\"dopri5_complete\":{},\"dopri5_finite\":{},\"dopri5_pass\":{}",
                        endpoint_time,
                        (endpoint_time - T_END).abs(),
                        terminal_time_tolerance,
                        endpoint_error,
                        trajectory_error,
                        energy_drift,
                        stats.accepted_steps,
                        stats.rejected_steps,
                        stats.num_eval,
                        reference_states.len(),
                        common_grid_matches,
                        format_optional_f64(common_grid_disagreement),
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
