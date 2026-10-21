//! Matched-work / precision report for fixed-step harmonic-oscillator integration.
//!
//! Run in release mode for interpretable timing:
//! cargo run --locked --release --manifest-path research/ode-verify-001/Cargo.toml --example oscillator_work_precision
//!
//! Timings include the checked driver and full trajectory recording. They are
//! not integrator-kernel-only numbers and must not be generalized to other
//! equation families or hardware.

use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ode_verify_001_harness::dynamical_system::{
    try_simulate, DynamicalSystem, EulerIntegrator, HarmonicOscillator, Integrator, Rk4Integrator,
    VerletIntegrator,
};

const PERIODS: usize = 4;
const STEPS_PER_PERIOD: [usize; 5] = [16, 32, 64, 128, 256];
const TIMING_REPEATS: usize = 5;
const OMEGA: f64 = 1.0;
const INITIAL_STATE: [f64; 2] = [1.0, 0.0];

struct CountedOscillator {
    inner: HarmonicOscillator,
    derivative_evaluations: AtomicUsize,
}

impl CountedOscillator {
    fn new() -> Self {
        Self {
            inner: HarmonicOscillator::simple(OMEGA),
            derivative_evaluations: AtomicUsize::new(0),
        }
    }
}

impl DynamicalSystem for CountedOscillator {
    fn state_dim(&self) -> usize {
        self.inner.state_dim()
    }

    fn derivatives(&self, state: &[f64], t: f64) -> Vec<f64> {
        self.derivative_evaluations.fetch_add(1, Ordering::Relaxed);
        self.inner.derivatives(state, t)
    }

    fn name(&self) -> &str {
        "counted harmonic oscillator"
    }
}

#[derive(Clone, Debug)]
struct Metrics {
    integrator: &'static str,
    steps_per_period: usize,
    step_size: f64,
    steps: usize,
    derivative_evaluations: usize,
    final_state_error_linf: f64,
    max_trajectory_error_linf: f64,
    max_relative_energy_drift: f64,
    median_runtime_ns: u128,
}

fn median(values: &mut [u128]) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn state_error(time: f64, state: &[f64]) -> f64 {
    let exact_x = (OMEGA * time).cos();
    let exact_v = -OMEGA * (OMEGA * time).sin();
    (state[0] - exact_x).abs().max((state[1] - exact_v).abs())
}

fn evaluate_case(
    integrator: &dyn Integrator,
    steps_per_period: usize,
    timing_repeats: usize,
) -> Result<Metrics, String> {
    if steps_per_period == 0 || timing_repeats == 0 {
        return Err("steps_per_period and timing_repeats must be positive".to_string());
    }

    let period = std::f64::consts::TAU / OMEGA;
    let step_size = period / steps_per_period as f64;
    let requested_steps = PERIODS
        .checked_mul(steps_per_period)
        .ok_or_else(|| "requested step count overflowed".to_string())?;
    let end_time = step_size * requested_steps as f64;
    let mut elapsed_ns = Vec::with_capacity(timing_repeats);
    let mut final_run = None;
    let mut expected_steps_observed = None;
    let mut expected_evaluations_observed = None;

    for _ in 0..timing_repeats {
        let system = CountedOscillator::new();
        let start = Instant::now();
        let result = try_simulate(
            &system,
            integrator,
            &INITIAL_STATE,
            0.0,
            end_time,
            step_size,
        )
        .map_err(|error| format!("{} rejected benchmark request: {error:?}", integrator.name()))?;
        let elapsed = start.elapsed().as_nanos();
        black_box(&result);

        if result.times.len() != result.states.len() || result.states.is_empty() {
            return Err(format!(
                "{} returned inconsistent time/state samples",
                integrator.name()
            ));
        }
        if (result.times[0]).abs() > f64::EPSILON
            || (result.times[result.times.len() - 1] - end_time).abs() > 1e-10
        {
            return Err(format!(
                "{} failed the declared benchmark horizon",
                integrator.name()
            ));
        }
        if result
            .states
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(format!(
                "{} emitted a non-finite trajectory",
                integrator.name()
            ));
        }

        let derivative_evaluations = system.derivative_evaluations.load(Ordering::Relaxed);
        if expected_steps_observed
            .is_some_and(|steps| steps != result.steps)
            || expected_evaluations_observed
                .is_some_and(|count| count != derivative_evaluations)
        {
            return Err(format!(
                "{} did not repeat with stable step/evaluation counts",
                integrator.name()
            ));
        }
        expected_steps_observed = Some(result.steps);
        expected_evaluations_observed = Some(derivative_evaluations);
        elapsed_ns.push(elapsed);
        final_run = Some((result, derivative_evaluations));
    }

    let (result, derivative_evaluations) =
        final_run.ok_or_else(|| "benchmark repeated zero times".to_string())?;
    let oscillator = HarmonicOscillator::simple(OMEGA);
    let initial_energy = oscillator.energy(&INITIAL_STATE).abs().max(f64::MIN_POSITIVE);
    let mut max_trajectory_error_linf = 0.0_f64;
    let mut max_relative_energy_drift = 0.0_f64;

    for (time, state) in result.times.iter().zip(result.states.iter()) {
        max_trajectory_error_linf =
            max_trajectory_error_linf.max(state_error(*time, state));
        let relative_drift = (oscillator.energy(state) - initial_energy).abs() / initial_energy;
        max_relative_energy_drift = max_relative_energy_drift.max(relative_drift);
    }

    let final_time = *result
        .times
        .last()
        .ok_or_else(|| "trajectory contained no final time".to_string())?;
    let final_state = result
        .final_state();
    let final_state_error_linf = state_error(final_time, final_state);

    Ok(Metrics {
        integrator: integrator.name(),
        steps_per_period,
        step_size,
        steps: result.steps,
        derivative_evaluations,
        final_state_error_linf,
        max_trajectory_error_linf,
        max_relative_energy_drift,
        median_runtime_ns: median(&mut elapsed_ns),
    })
}

fn print_header() {
    println!(
        "integrator,steps_per_period,dt,periods,steps,derivative_evaluations,\
final_state_error_linf,max_trajectory_error_linf,max_relative_energy_drift,median_runtime_ns"
    );
}

fn print_metrics(metrics: &Metrics) {
    println!(
        "{},{},{:.17},{},{},{},{:.12e},{:.12e},{:.12e},{}",
        metrics.integrator,
        metrics.steps_per_period,
        metrics.step_size,
        PERIODS,
        metrics.steps,
        metrics.derivative_evaluations,
        metrics.final_state_error_linf,
        metrics.max_trajectory_error_linf,
        metrics.max_relative_energy_drift,
        metrics.median_runtime_ns
    );
}

fn main() {
    let integrators: [(&dyn Integrator); 3] =
        [&EulerIntegrator, &Rk4Integrator, &VerletIntegrator];

    print_header();
    let mut failures = 0usize;
    for integrator in integrators {
        for steps_per_period in STEPS_PER_PERIOD {
            match evaluate_case(integrator, steps_per_period, TIMING_REPEATS) {
                Ok(metrics) => print_metrics(&metrics),
                Err(error) => {
                    eprintln!("{error}");
                    failures += 1;
                }
            }
        }
    }

    if failures > 0 {
        eprintln!("{failures} work-precision case(s) failed");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refinement_reduces_rk4_trajectory_error() {
        let coarse = evaluate_case(&Rk4Integrator, 64, 1).expect("coarse RK4 case");
        let fine = evaluate_case(&Rk4Integrator, 128, 1).expect("fine RK4 case");
        assert!(
            fine.max_trajectory_error_linf < coarse.max_trajectory_error_linf,
            "RK4 error did not improve under refinement: coarse={:.3e}, fine={:.3e}",
            coarse.max_trajectory_error_linf,
            fine.max_trajectory_error_linf
        );
        let ratio = coarse.max_trajectory_error_linf / fine.max_trajectory_error_linf;
        assert!(
            (10.0..=24.0).contains(&ratio),
            "RK4 refinement ratio {ratio:.3} is outside the preregistered diagnostic band 10..=24"
        );
    }

    #[test]
    fn refinement_reduces_verlet_trajectory_error() {
        let coarse = evaluate_case(&VerletIntegrator, 32, 1).expect("coarse Verlet case");
        let fine = evaluate_case(&VerletIntegrator, 64, 1).expect("fine Verlet case");
        assert!(
            fine.max_trajectory_error_linf < coarse.max_trajectory_error_linf,
            "Verlet error did not improve under refinement: coarse={:.3e}, fine={:.3e}",
            coarse.max_trajectory_error_linf,
            fine.max_trajectory_error_linf
        );
    }

    #[test]
    fn all_methods_emit_finite_errors_and_nonzero_work_counts() {
        for integrator in [&EulerIntegrator as &dyn Integrator, &Rk4Integrator, &VerletIntegrator] {
            let metrics = evaluate_case(integrator, 64, 1).expect("finite work-precision case");
            assert!(metrics.steps > 0);
            assert!(metrics.derivative_evaluations > 0);
            assert!(metrics.median_runtime_ns > 0);
            assert!(metrics.final_state_error_linf.is_finite());
            assert!(metrics.max_trajectory_error_linf.is_finite());
            assert!(metrics.max_relative_energy_drift.is_finite());
        }
    }
}
