//! Deterministic robust-gain selector for IG-COORD-SELECT-001.
//!
//! This consumes calibration estimates; it does not infer them or run agents.
//! Evaluator-only oracle outcomes are deliberately absent from SelectorInput.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::env;
use std::fs;

#[derive(Debug, Deserialize)]
struct FixtureSet {
    schema_version: u32,
    protocol_id: String,
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Clone, Deserialize)]
struct Fixture {
    id: String,
    input: SelectorInput,
    expected: Expected,
}

#[derive(Debug, Clone, Deserialize)]
struct SelectorInput {
    task_id: String,
    task_decomposability: String,
    sequential_coupling: String,
    latent_information_asymmetry: String,
    distribution_shift: bool,
    single_agent: Estimate,
    candidate_architectures: Vec<Estimate>,
    policy: Policy,
}

#[derive(Debug, Clone, Deserialize)]
struct Estimate {
    architecture_id: String,
    quality_mean_milli: u16,
    quality_lower_milli: u16,
    quality_upper_milli: u16,
    token_cost: u64,
    latency_ms: u64,
    risk_milli: u16,
    calibration_blocks: u32,
    out_of_distribution: bool,
    available: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct Policy {
    min_calibration_blocks: u32,
    max_quality_interval_width_milli: u16,
    min_robust_quality_gain_milli: u16,
    min_robust_net_utility_gain_milli: i32,
    token_penalty_milli_per_1000: u32,
    latency_penalty_milli_per_second: u32,
    risk_penalty_milli_per_1000: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct Decision {
    task_id: String,
    decision: String,
    selected_architecture_id: Option<String>,
    reason: String,
    robust_net_gain_milli: Option<i32>,
}

#[derive(Debug, Clone, Deserialize)]
struct Expected {
    decision: String,
    selected_architecture_id: Option<String>,
    reason: String,
    robust_net_gain_milli: Option<i32>,
}

fn estimate_is_valid(estimate: &Estimate) -> bool {
    estimate.quality_lower_milli <= estimate.quality_mean_milli
        && estimate.quality_mean_milli <= estimate.quality_upper_milli
        && estimate.quality_upper_milli <= 1_000
        && estimate.risk_milli <= 1_000
        && !estimate.architecture_id.trim().is_empty()
}

fn validate_input(input: &SelectorInput) -> Result<(), String> {
    if input.task_id.trim().is_empty() {
        return Err("task_id must not be empty".to_string());
    }
    if !matches!(input.task_decomposability.as_str(), "low" | "medium" | "high")
        || !matches!(input.sequential_coupling.as_str(), "low" | "medium" | "high")
        || !matches!(
            input.latent_information_asymmetry.as_str(),
            "low" | "medium" | "high"
        )
    {
        return Err(format!("task {} has an invalid descriptor", input.task_id));
    }
    if !estimate_is_valid(&input.single_agent) {
        return Err(format!("task {} has an invalid single-agent estimate", input.task_id));
    }

    let mut ids = HashSet::new();
    for candidate in &input.candidate_architectures {
        if !estimate_is_valid(candidate) {
            return Err(format!(
                "task {} has an invalid estimate for {}",
                input.task_id, candidate.architecture_id
            ));
        }
        if candidate.architecture_id == input.single_agent.architecture_id {
            return Err("candidate architecture id collides with single-agent id".to_string());
        }
        if !ids.insert(candidate.architecture_id.clone()) {
            return Err(format!("duplicate candidate architecture id: {}", candidate.architecture_id));
        }
    }

    if input.policy.min_calibration_blocks == 0 {
        return Err("min_calibration_blocks must be positive".to_string());
    }
    Ok(())
}

fn penalty(estimate: &Estimate, policy: &Policy) -> i64 {
    let token_penalty = (estimate.token_cost as u128
        * policy.token_penalty_milli_per_1000 as u128
        / 1_000) as i64;
    let latency_penalty = (estimate.latency_ms as u128
        * policy.latency_penalty_milli_per_second as u128
        / 1_000) as i64;
    let risk_penalty = (estimate.risk_milli as u128
        * policy.risk_penalty_milli_per_1000 as u128
        / 1_000) as i64;
    token_penalty + latency_penalty + risk_penalty
}

fn utility_lower(estimate: &Estimate, policy: &Policy) -> i64 {
    estimate.quality_lower_milli as i64 - penalty(estimate, policy)
}

fn utility_upper(estimate: &Estimate, policy: &Policy) -> i64 {
    estimate.quality_upper_milli as i64 - penalty(estimate, policy)
}

fn is_ready(estimate: &Estimate, policy: &Policy) -> bool {
    !estimate.out_of_distribution
        && estimate.calibration_blocks >= policy.min_calibration_blocks
        && estimate.quality_upper_milli - estimate.quality_lower_milli
            <= policy.max_quality_interval_width_milli
}

fn decision(
    input: &SelectorInput,
    kind: &str,
    architecture: Option<&str>,
    reason: &str,
    robust_net_gain_milli: Option<i32>,
) -> Decision {
    Decision {
        task_id: input.task_id.clone(),
        decision: kind.to_string(),
        selected_architecture_id: architecture.map(str::to_string),
        reason: reason.to_string(),
        robust_net_gain_milli,
    }
}

/// Selects a team only when its conservative quality and net-utility gains
/// both exceed the declared policy thresholds. The answer key and the
/// evaluator-only oracle are never inputs to this function.
fn select_architecture(input: &SelectorInput) -> Result<Decision, String> {
    validate_input(input)?;
    let policy = &input.policy;
    let baseline = &input.single_agent;

    if input.distribution_shift {
        return Ok(decision(
            input,
            "CALIBRATE",
            None,
            "task_distribution_shift",
            None,
        ));
    }
    if !is_ready(baseline, policy) {
        return Ok(decision(
            input,
            "CALIBRATE",
            None,
            "single_baseline_not_calibrated",
            None,
        ));
    }

    let baseline_lower_utility = utility_lower(baseline, policy);
    let baseline_upper_utility = utility_upper(baseline, policy);
    let mut promising_uncertain_architecture = false;
    let mut qualifying: Vec<(String, i32)> = Vec::new();

    for candidate in &input.candidate_architectures {
        if !candidate.available {
            continue;
        }

        if !is_ready(candidate, policy) {
            let plausible_quality_gain =
                candidate.quality_upper_milli as i64 - baseline.quality_lower_milli as i64;
            let plausible_net_gain =
                utility_upper(candidate, policy) - baseline_lower_utility;
            if plausible_quality_gain >= policy.min_robust_quality_gain_milli as i64
                && plausible_net_gain >= policy.min_robust_net_utility_gain_milli as i64
            {
                promising_uncertain_architecture = true;
            }
            continue;
        }

        let robust_quality_gain =
            candidate.quality_lower_milli as i64 - baseline.quality_upper_milli as i64;
        let robust_net_gain =
            utility_lower(candidate, policy) - baseline_upper_utility;

        if robust_quality_gain >= policy.min_robust_quality_gain_milli as i64
            && robust_net_gain >= policy.min_robust_net_utility_gain_milli as i64
        {
            qualifying.push((
                candidate.architecture_id.clone(),
                robust_net_gain.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            ));
        }
    }

    if promising_uncertain_architecture {
        return Ok(decision(
            input,
            "CALIBRATE",
            None,
            "promising_architecture_needs_calibration",
            None,
        ));
    }

    qualifying.sort_by(|left, right| {
        match right.1.cmp(&left.1) {
            Ordering::Equal => left.0.cmp(&right.0),
            ordering => ordering,
        }
    });

    if let Some((architecture_id, robust_net_gain)) = qualifying.first() {
        return Ok(decision(
            input,
            "SELECT",
            Some(architecture_id),
            "robust_positive_net_gain",
            Some(*robust_net_gain),
        ));
    }

    Ok(decision(
        input,
        "SINGLE",
        Some("single"),
        "no_robust_positive_net_gain",
        None,
    ))
}

fn expected_matches(actual: &Decision, expected: &Expected) -> bool {
    actual.decision == expected.decision
        && actual.selected_architecture_id == expected.selected_architecture_id
        && actual.reason == expected.reason
        && actual.robust_net_gain_milli == expected.robust_net_gain_milli
}

fn run(path: &str) -> Result<(), String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("cannot read selector fixtures {path}: {error}"))?;
    let fixture_set: FixtureSet = serde_json::from_str(&raw)
        .map_err(|error| format!("invalid selector fixture JSON: {error}"))?;

    if fixture_set.schema_version != 1
        || fixture_set.protocol_id != "IG-COORD-SELECT-001"
    {
        return Err("unsupported coordination selector fixture schema".to_string());
    }
    if fixture_set.fixtures.is_empty() {
        return Err("selector fixture set must not be empty".to_string());
    }

    let mut ids = HashSet::new();
    let mut selected = 0usize;
    let mut single = 0usize;
    let mut calibrate = 0usize;
    let mut mismatches = 0usize;

    for fixture in &fixture_set.fixtures {
        if !ids.insert(fixture.id.clone()) {
            return Err(format!("duplicate fixture id: {}", fixture.id));
        }

        let actual = select_architecture(&fixture.input)?;
        match actual.decision.as_str() {
            "SELECT" => selected += 1,
            "SINGLE" => single += 1,
            "CALIBRATE" => calibrate += 1,
            other => return Err(format!("unexpected selector disposition: {other}")),
        }

        if !expected_matches(&actual, &fixture.expected) {
            mismatches += 1;
            eprintln!(
                "EXPECTATION_MISMATCH {} expected={:?} actual={:?}",
                fixture.id, fixture.expected, actual
            );
        }
        println!(
            "{}",
            serde_json::to_string(&actual)
                .map_err(|error| format!("cannot serialize selector result: {error}"))?
        );
    }

    println!(
        "SUMMARY fixtures={} selected={} single={} calibrate={} expectation_mismatches={}",
        fixture_set.fixtures.len(),
        selected,
        single,
        calibrate,
        mismatches
    );

    if mismatches > 0 {
        return Err(format!("{mismatches} selector expectation(s) did not match"));
    }
    Ok(())
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: ig-coord-select-qualifier <COORD_SELECT_FIXTURES_V0_1.json>");
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("expected exactly one fixture path");
        std::process::exit(2);
    }

    if let Err(error) = run(&path) {
        eprintln!("coordination selector qualification failed: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURES: &str = include_str!("../COORD_SELECT_FIXTURES_V0_1.json");

    fn fixtures() -> FixtureSet {
        serde_json::from_str(FIXTURES).expect("coordination selector fixtures parse")
    }

    #[test]
    fn all_frozen_selector_expectations_match() {
        let set = fixtures();
        assert_eq!(set.schema_version, 1);
        assert_eq!(set.protocol_id, "IG-COORD-SELECT-001");
        for fixture in &set.fixtures {
            let actual = select_architecture(&fixture.input).expect("fixture input is valid");
            assert!(
                expected_matches(&actual, &fixture.expected),
                "fixture {} mismatch: actual={actual:?}, expected={:?}",
                fixture.id,
                fixture.expected
            );
        }
    }

    #[test]
    fn capability_saturation_selects_single_agent() {
        let set = fixtures();
        let fixture = set.fixtures.iter()
            .find(|fixture| fixture.id == "CS-002-capability-saturation-single")
            .expect("saturation fixture");
        let actual = select_architecture(&fixture.input).expect("valid input");
        assert_eq!(actual.decision, "SINGLE");
        assert_eq!(actual.selected_architecture_id.as_deref(), Some("single"));
    }

    #[test]
    fn decomposable_task_can_justify_parallel_architecture() {
        let set = fixtures();
        let fixture = set.fixtures.iter()
            .find(|fixture| fixture.id == "CS-003-decomposable-parallel")
            .expect("parallel fixture");
        let actual = select_architecture(&fixture.input).expect("valid input");
        assert_eq!(actual.decision, "SELECT");
        assert_eq!(actual.selected_architecture_id.as_deref(), Some("parallel_independent"));
    }

    #[test]
    fn sequentially_coupled_task_does_not_get_team_by_default() {
        let set = fixtures();
        let fixture = set.fixtures.iter()
            .find(|fixture| fixture.id == "CS-004-sequential-coupling-single")
            .expect("sequential fixture");
        let actual = select_architecture(&fixture.input).expect("valid input");
        assert_eq!(actual.decision, "SINGLE");
    }

    #[test]
    fn uncertain_promising_team_returns_calibrate() {
        let set = fixtures();
        let fixture = set.fixtures.iter()
            .find(|fixture| fixture.id == "CS-005-uncertain-promising-team")
            .expect("uncertain fixture");
        let actual = select_architecture(&fixture.input).expect("valid input");
        assert_eq!(actual.decision, "CALIBRATE");
    }

    #[test]
    fn out_of_distribution_task_never_forces_a_team() {
        let set = fixtures();
        let fixture = set.fixtures.iter()
            .find(|fixture| fixture.id == "CS-006-out-of-distribution-task")
            .expect("shift fixture");
        let actual = select_architecture(&fixture.input).expect("valid input");
        assert_eq!(actual.decision, "CALIBRATE");
    }

    #[test]
    fn resource_cost_can_erase_predicted_quality_gain() {
        let set = fixtures();
        let fixture = set.fixtures.iter()
            .find(|fixture| fixture.id == "CS-007-team-gain-eaten-by-cost")
            .expect("cost fixture");
        let actual = select_architecture(&fixture.input).expect("valid input");
        assert_eq!(actual.decision, "SINGLE");
    }

    #[test]
    fn deterministic_tie_break_uses_architecture_id() {
        let set = fixtures();
        let fixture = set.fixtures.iter()
            .find(|fixture| fixture.id == "CS-012-deterministic-tie-break")
            .expect("tie fixture");
        let actual = select_architecture(&fixture.input).expect("valid input");
        assert_eq!(actual.selected_architecture_id.as_deref(), Some("alpha_team"));
    }

    #[test]
    fn evaluator_only_fields_are_not_part_of_selector_input() {
        // Fixture has evaluator_only oracle labels, but SelectorInput does not
        // deserialize or expose them to select_architecture.
        let set = fixtures();
        let fixture = set.fixtures.iter()
            .find(|fixture| fixture.id == "CS-001-targeted-hidden-info")
            .expect("hidden-info fixture");
        let actual = select_architecture(&fixture.input).expect("valid input");
        assert_eq!(actual.selected_architecture_id.as_deref(), Some("targeted_discovery"));
    }

    #[test]
    fn duplicate_candidate_ids_are_rejected() {
        let set = fixtures();
        let mut input = set.fixtures.iter()
            .find(|fixture| fixture.id == "CS-003-decomposable-parallel")
            .expect("parallel fixture")
            .input.clone();
        input.candidate_architectures.push(input.candidate_architectures[0].clone());
        assert!(validate_input(&input).is_err());
    }
}
