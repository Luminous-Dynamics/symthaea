//! Bounded explicit Cash–Karp stress probe for the Robertson stiff kinetics system.
//!
//! This is intentionally diagnostic-only. It records work and structural defects,
//! but does not use an approved reference trajectory and cannot qualify solver accuracy.

use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};

use ode_verify_001_harness::dynamical_system::{
    DynamicalSystem, Rk45Controller, Rk45Integrator, Rk45Tolerances, SimulationError,
};

const OUTPUT_TIMES: [f64; 17] = [
    0.0, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0, 10.0, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6,
    1.0e7, 1.0e8, 1.0e9, 1.0e10, 1.0e11,
];
const INITIAL_STATE: [f64; 3] = [1.0, 0.0, 0.0];
const RELATIVE_TOLERANCE: f64 = 1.0e-8;
const ABSOLUTE_TOLERANCES: [f64; 3] = [1.0e-12, 1.0e-16, 1.0e-12];
const INITIAL_DT: f64 = 1.0e-8;
const MAX_ATTEMPTS: usize = 100_000;
const TOLERANCE_MULTIPLIER: f64 = 10.0;

struct Robertson {
    derivative_evaluations: AtomicUsize,
}

impl DynamicalSystem for Robertson {
    fn state_dim(&self) -> usize {
        3
    }

    fn derivatives(&self, state: &[f64], _time: f64) -> Vec<f64> {
        self.derivative_evaluations.fetch_add(1, Ordering::Relaxed);
        let y1 = state[0];
        let y2 = state[1];
        let y3 = state[2];
        vec![
            -0.04 * y1 + 1.0e4 * y2 * y3,
            0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2,
            3.0e7 * y2 * y2,
        ]
    }

    fn name(&self) -> &str {
        "RobertsonStiffKinetics"
    }
}

fn json_number(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.17e}")
    } else {
        "null".to_owned()
    }
}

fn json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    for ch in value.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => escaped.push(' '),
            ch => escaped.push(ch),
        }
    }
    escaped
}

fn state_json(state: &[f64; 3]) -> String {
    format!(
        "[{},{},{}]",
        json_number(state[0]),
        json_number(state[1]),
        json_number(state[2])
    )
}

fn sample_json(samples: &[(f64, [f64; 3])]) -> String {
    samples
        .iter()
        .map(|(time, state)| {
            format!(
                "{{\"time\":{},\"state\":{}}}",
                json_number(*time),
                state_json(state)
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn vector_json(values: &[f64; 3]) -> String {
    state_json(values)
}

fn update_structural_metrics(
    state: &[f64; 3],
    min_state: &mut [f64; 3],
    max_negative_excursion: &mut [f64; 3],
    max_invariant_defect: &mut f64,
    all_finite: &mut bool,
) {
    for index in 0..3 {
        if !state[index].is_finite() {
            *all_finite = false;
        } else {
            min_state[index] = min_state[index].min(state[index]);
            max_negative_excursion[index] =
                max_negative_excursion[index].max((-state[index]).max(0.0));
        }
    }
    let invariant = state[0] + state[1] + state[2];
    if invariant.is_finite() {
        *max_invariant_defect = max_invariant_defect.max((invariant - 1.0).abs());
    } else {
        *all_finite = false;
    }
}

fn main() {
    let source_sha = env::var("ODE_BENCH_SOURCE_SHA").unwrap_or_else(|_| "unknown".to_owned());
    let checkout_sha =
        env::var("ODE_BENCH_CHECKOUT_SHA").unwrap_or_else(|_| "unknown".to_owned());

    let integrator = Rk45Integrator {
        tol: RELATIVE_TOLERANCE,
        dt_min: 1.0e-24,
        dt_max: 1.0e11,
    };
    let tolerances = Rk45Tolerances::new(
        RELATIVE_TOLERANCE,
        ABSOLUTE_TOLERANCES.to_vec(),
    );
    let system = Robertson {
        derivative_evaluations: AtomicUsize::new(0),
    };
    let mut controller = match Rk45Controller::new(INITIAL_DT) {
        Ok(controller) => controller,
        Err(error) => {
            print_failure_receipt(
                &source_sha,
                &checkout_sha,
                "invalid_controller",
                &format!("{error:?}"),
            );
            return;
        }
    };

    let mut time = OUTPUT_TIMES[0];
    let mut state = INITIAL_STATE;
    let mut target_index = 1usize;
    let mut samples = vec![(time, state)];
    let mut accepted_steps = 0usize;
    let mut rejected_steps = 0usize;
    let mut attempted_steps = 0usize;
    let mut derivative_evaluations = 0usize;
    let mut min_state = INITIAL_STATE;
    let mut max_negative_excursion = [0.0_f64; 3];
    let mut max_invariant_defect = 0.0_f64;
    let mut all_finite = true;
    let mut status = "attempt_budget_exceeded";
    let mut status_detail = String::new();
    let mut failed_step_work_complete = true;

    update_structural_metrics(
        &state,
        &mut min_state,
        &mut max_negative_excursion,
        &mut max_invariant_defect,
        &mut all_finite,
    );

    while target_index < OUTPUT_TIMES.len() && attempted_steps < MAX_ATTEMPTS {
        let target = OUTPUT_TIMES[target_index];
        let remaining = target - time;
        if !remaining.is_finite() || remaining <= 0.0 {
            status = "invalid_target_interval";
            status_detail = format!("time={time:.17e}; target={target:.17e}");
            failed_step_work_complete = false;
            break;
        }

        let rhs_before = system.derivative_evaluations.load(Ordering::Relaxed);
        let step_result = controller.try_step(
            &integrator,
            &system,
            &state,
            time,
            remaining,
            &tolerances,
        );
        let rhs_after = system.derivative_evaluations.load(Ordering::Relaxed);
        let rhs_delta = rhs_after.saturating_sub(rhs_before);
        derivative_evaluations = rhs_after;

        match step_result {
            Ok(step) => {
                attempted_steps = attempted_steps.saturating_add(step.attempted_steps);
                rejected_steps = rejected_steps.saturating_add(step.rejected_steps);
                if rhs_delta != step.derivative_evaluations {
                    status = "rhs_counter_mismatch";
                    status_detail = format!(
                        "independent RHS callback count {rhs_delta} differs from step report {}",
                        step.derivative_evaluations
                    );
                    failed_step_work_complete = false;
                    break;
                }
                accepted_steps = accepted_steps.saturating_add(1);

                if !step.accepted_dt.is_finite()
                    || step.accepted_dt <= 0.0
                    || step.state.len() != 3
                {
                    status = "invalid_step_report";
                    status_detail =
                        "accepted step report had an invalid dt or state dimension".to_owned();
                    failed_step_work_complete = false;
                    break;
                }

                let previous_time = time;
                time += step.accepted_dt;
                if !time.is_finite() || time <= previous_time {
                    status = "time_did_not_advance";
                    status_detail = format!(
                        "previous_time={previous_time:.17e}; accepted_dt={:.17e}",
                        step.accepted_dt
                    );
                    failed_step_work_complete = false;
                    break;
                }

                state = [step.state[0], step.state[1], step.state[2]];
                update_structural_metrics(
                    &state,
                    &mut min_state,
                    &mut max_negative_excursion,
                    &mut max_invariant_defect,
                    &mut all_finite,
                );

                let target_slack = 16.0 * f64::EPSILON * target.abs().max(1.0);
                if (time - target).abs() <= target_slack {
                    time = target;
                    samples.push((time, state));
                    target_index += 1;
                } else if time > target + target_slack {
                    status = "sample_target_overshoot";
                    status_detail = format!("time={time:.17e}; target={target:.17e}");
                    failed_step_work_complete = false;
                    break;
                }

                if !all_finite {
                    status = "non_finite_state";
                    status_detail = "non-finite accepted state or invariant".to_owned();
                    break;
                }
            }
            Err(error) => {
                status = match &error {
                    SimulationError::AdaptiveToleranceNotMet => "tolerance_not_met",
                    SimulationError::TimeDidNotAdvance { .. } => "time_did_not_advance",
                    _ => "solver_error",
                };
                status_detail = format!("{error:?}");
                // The current API returns no partial step report on errors, so
                // work attempted within this failed step cannot be counted here.
                failed_step_work_complete = false;
                break;
            }
        }
    }

    if target_index == OUTPUT_TIMES.len() {
        status = "completed";
        status_detail.clear();
    } else if attempted_steps >= MAX_ATTEMPTS && status == "attempt_budget_exceeded" {
        status_detail = "fixed attempt budget reached before the final sample time".to_owned();
    }

    let nonnegative_within_tolerance = (0..3).all(|i| {
        min_state[i] >= -TOLERANCE_MULTIPLIER * ABSOLUTE_TOLERANCES[i]
    });
    let reached_final_time = target_index == OUTPUT_TIMES.len()
        && (time - OUTPUT_TIMES[OUTPUT_TIMES.len() - 1]).abs()
            <= 16.0 * f64::EPSILON * time.abs().max(1.0);
    let state_is_finite = state.iter().all(|value| value.is_finite());
    let qualification_status = "UNQUALIFIED_DIAGNOSTIC_ONLY";
    let final_state_json = state_json(&state);
    let detail_json = json_string(&status_detail);
    let sample_data = sample_json(&samples);

    println!(
        concat!(
            "{{\"schema_version\":\"0.1.0\",",
            "\"benchmark_id\":\"ODE-ROBERTSON-EXPLICIT-PROBE-001\",",
            "\"source_sha\":\"{}\",\"checkout_sha\":\"{}\",",
            "\"qualification_status\":\"{}\",\"solver\":\"SymthaeaCashKarpRK45\",",
            "\"status\":\"{}\",\"status_detail\":\"{}\",",
            "\"t_start\":{},\"t_end_requested\":{},\"last_time\":{},",
            "\"output_target_count\":{},\"output_sample_count\":{},",
            "\"attempt_budget\":{},\"attempted_steps_reported\":{},",
            "\"accepted_steps_reported\":{},\"rejected_steps_reported\":{},",
            "\"derivative_evaluations_reported\":{},",
            "\"failed_step_work_complete\":{},",
            "\"reached_final_time\":{},\"state_is_finite\":{},",
            "\"nonnegative_within_tolerance_at_accepted_endpoints\":{},",
            "\"max_invariant_defect_at_accepted_endpoints\":{},",
            "\"min_state_at_accepted_endpoints\":{},",
            "\"max_negative_excursion_at_accepted_endpoints\":{},",
            "\"final_state\":{},\"output_samples\":[{}]}}"
        ),
        json_string(&source_sha),
        json_string(&checkout_sha),
        qualification_status,
        status,
        detail_json,
        json_number(OUTPUT_TIMES[0]),
        json_number(OUTPUT_TIMES[OUTPUT_TIMES.len() - 1]),
        json_number(time),
        OUTPUT_TIMES.len(),
        samples.len(),
        MAX_ATTEMPTS,
        attempted_steps,
        accepted_steps,
        rejected_steps,
        derivative_evaluations,
        failed_step_work_complete,
        reached_final_time,
        state_is_finite,
        nonnegative_within_tolerance,
        json_number(max_invariant_defect),
        vector_json(&min_state),
        vector_json(&max_negative_excursion),
        final_state_json,
        sample_data
    );
}

fn print_failure_receipt(source_sha: &str, checkout_sha: &str, status: &str, detail: &str) {
    println!(
        "{{\"schema_version\":\"0.1.0\",\"benchmark_id\":\"ODE-ROBERTSON-EXPLICIT-PROBE-001\",\"source_sha\":\"{}\",\"checkout_sha\":\"{}\",\"qualification_status\":\"UNQUALIFIED_DIAGNOSTIC_ONLY\",\"solver\":\"SymthaeaCashKarpRK45\",\"status\":\"{}\",\"status_detail\":\"{}\",\"output_target_count\":{},\"output_sample_count\":1,\"attempt_budget\":{},\"attempted_steps_reported\":0,\"accepted_steps_reported\":0,\"rejected_steps_reported\":0,\"derivative_evaluations_reported\":0,\"failed_step_work_complete\":false,\"reached_final_time\":false,\"state_is_finite\":true,\"nonnegative_within_tolerance_at_accepted_endpoints\":true,\"max_invariant_defect_at_accepted_endpoints\":0.0,\"min_state_at_accepted_endpoints\":[1.0,0.0,0.0],\"max_negative_excursion_at_accepted_endpoints\":[0.0,0.0,0.0],\"final_state\":[1.0,0.0,0.0],\"output_samples\":[{{\"time\":0.0,\"state\":[1.0,0.0,0.0]}}]}}",
        json_string(source_sha),
        json_string(checkout_sha),
        status,
        json_string(detail),
        OUTPUT_TIMES.len(),
        MAX_ATTEMPTS
    );
}
