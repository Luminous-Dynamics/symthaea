//! Deterministic work–precision ledger for the scale-aware RK45 path.
//!
//! This is an analytical oscillator benchmark, not an external-SOTA comparison.
//! Wall-clock timing is deliberately excluded from the deterministic receipt;
//! first compare derivative evaluations against endpoint error, then time repeated
//! runs under a separately frozen machine/toolchain profile.

use std::env;
use std::error::Error;

use ode_verify_001_harness::dynamical_system::{
    HarmonicOscillator, Rk45Integrator, Rk45Tolerances,
};

const T_END: f64 = 10.0;
const INITIAL_DT: f64 = 0.2;
const SCALES: [f64; 3] = [1.0e-6, 1.0, 1.0e6];
const RELATIVE_TOLERANCES: [f64; 3] = [1.0e-3, 1.0e-5, 1.0e-7];
const GLOBAL_ERROR_MULTIPLIER: f64 = 10.0;

fn main() -> Result<(), Box<dyn Error>> {
    let source_sha = env::var("ODE_BENCH_SOURCE_SHA").unwrap_or_else(|_| "unknown".to_string());
    let checkout_sha =
        env::var("ODE_BENCH_CHECKOUT_SHA").unwrap_or_else(|_| "unknown".to_string());
    let oscillator = HarmonicOscillator::simple(1.0);
    let integrator = Rk45Integrator::new(1.0e-8);
    let mut observations = Vec::new();
    let mut all_passed = true;

    for scale in SCALES {
        for relative_tolerance in RELATIVE_TOLERANCES {
            // Scale the absolute floor with the state units so this experiment
            // tests relative-error semantics, not arbitrary unit selection.
            let absolute_tolerance = scale * relative_tolerance * 1.0e-3;
            let tolerances = Rk45Tolerances::new(relative_tolerance, vec![absolute_tolerance; 2]);

            match integrator.try_simulate_with_tolerances(
                &oscillator,
                &[scale, 0.0],
                0.0,
                T_END,
                INITIAL_DT,
                &tolerances,
            ) {
                Ok(result) => {
                    let exact = [scale * T_END.cos(), -scale * T_END.sin()];
                    let endpoint_error = result
                        .simulation
                        .final_state()
                        .iter()
                        .zip(exact.iter())
                        .map(|(actual, reference)| (actual - reference).powi(2))
                        .sum::<f64>()
                        .sqrt()
                        / scale;

                    let initial_energy = 0.5 * scale * scale;
                    let max_relative_energy_drift = result
                        .simulation
                        .states
                        .iter()
                        .map(|state| {
                            let energy = 0.5 * (state[0] * state[0] + state[1] * state[1]);
                            (energy - initial_energy).abs() / initial_energy
                        })
                        .fold(0.0f64, f64::max);

                    let endpoint_error_limit = GLOBAL_ERROR_MULTIPLIER * relative_tolerance;
                    let energy_drift_limit = GLOBAL_ERROR_MULTIPLIER * relative_tolerance;
                    let completed = result.simulation.times.last().is_some_and(|time| {
                        (*time - T_END).abs() <= f64::EPSILON * T_END.abs().max(1.0)
                    });
                    let finite = endpoint_error.is_finite()
                        && max_relative_energy_drift.is_finite()
                        && result
                            .simulation
                            .states
                            .iter()
                            .flatten()
                            .all(|value| value.is_finite());
                    let endpoint_error_within_limit = endpoint_error <= endpoint_error_limit;
                    let energy_drift_within_limit = max_relative_energy_drift <= energy_drift_limit;
                    let accepted = completed
                        && finite
                        && result.diagnostics.max_error_norm <= 1.0
                        && endpoint_error_within_limit
                        && energy_drift_within_limit
                        && result.diagnostics.accepted_steps == result.simulation.steps;
                    all_passed &= accepted;

                    observations.push(format!(
                        "{{\"scale\":{:.17e},\"relative_tolerance\":{:.17e},\"absolute_tolerance\":{:.17e},\"endpoint_error_l2_scaled\":{:.17e},\"endpoint_error_limit\":{:.17e},\"max_relative_energy_drift\":{:.17e},\"energy_drift_limit\":{:.17e},\"accepted_steps\":{},\"rejected_steps\":{},\"derivative_evaluations\":{},\"max_accepted_error_norm\":{:.17e},\"completed\":{},\"finite\":{},\"endpoint_error_within_limit\":{},\"energy_drift_within_limit\":{},\"numerical_contract_pass\":{}}}",
                        scale,
                        relative_tolerance,
                        absolute_tolerance,
                        endpoint_error,
                        endpoint_error_limit,
                        max_relative_energy_drift,
                        energy_drift_limit,
                        result.diagnostics.accepted_steps,
                        result.diagnostics.rejected_steps,
                        result.diagnostics.derivative_evaluations,
                        result.diagnostics.max_error_norm,
                        completed,
                        finite,
                        endpoint_error_within_limit,
                        energy_drift_within_limit,
                        accepted
                    ));
                }
                Err(error) => {
                    all_passed = false;
                    observations.push(format!(
                        "{{\"scale\":{:.17e},\"relative_tolerance\":{:.17e},\"absolute_tolerance\":{:.17e},\"endpoint_error_limit\":{:.17e},\"energy_drift_limit\":{:.17e},\"numerical_contract_pass\":false,\"error\":\"{:?}\"}}",
                        scale,
                        relative_tolerance,
                        absolute_tolerance,
                        GLOBAL_ERROR_MULTIPLIER * relative_tolerance,
                        GLOBAL_ERROR_MULTIPLIER * relative_tolerance,
                        error
                    ));
                }
            }
        }
    }

    println!(
        "{{\"schema_version\":\"0.1.0\",\"benchmark_id\":\"ODE-WP-ANALYTIC-OSCILLATOR-001\",\"source_sha\":\"{}\",\"checkout_sha\":\"{}\",\"reference\":\"analytic_harmonic_oscillator\",\"t_end\":{:.17e},\"observations\":[",
        source_sha, checkout_sha, T_END
    );
    for (index, observation) in observations.iter().enumerate() {
        if index > 0 {
            println!(",");
        }
        print!("{observation}");
    }
    println!("]}}");

    if !all_passed {
        return Err("one or more analytical work-precision cases failed".into());
    }
    Ok(())
}
