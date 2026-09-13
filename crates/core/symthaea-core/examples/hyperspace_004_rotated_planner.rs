// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! HYPERSPACE-004 preregistered blind rotated-basis planner discovery study.
//!
//! The target seeded RRT sees only an external four-coordinate problem, rotated
//! oracle, conservative sampling envelope, seed and config. Canonical semantic
//! diagnostics are computed only after the planner returns.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use serde_json::{Value, json};
use symthaea_core::continuous_reachability::{
    ContinuousPathReplay, ContinuousReachabilityError, ContinuousValidityOracle,
    EuclideanPlanningProblem, OracleVerdict, validate_euclidean_path,
};
use symthaea_core::hyperspace_benchmark::{
    FiniteWShellOracle, HyperspaceDimension, canonical_shell_problem,
    certify_radial_separation,
};
use symthaea_core::sampling_reachability::{
    BoundedEuclideanSamplingOracle, RrtConfig, SamplingReachabilityResult, seeded_rrt,
};

#[path = "hyperspace_support/orthogonal.rs"]
mod orthogonal;
use orthogonal::{
    MATERIAL_CLEARANCE, ROUNDTRIP_TOLERANCE, RotatedSamplingOracle, TransformSpec,
    frozen_transforms, matrix_bits, path_roundtrip_error,
};

const SCHEMA: &str = "symthaea.hyperspace-004.rotated-planner.v1";
const PREREG_ISSUE: u64 = 2435;
const QUALIFIED_H003R_HEAD: &str = "8d588e50ce68b16b6d888a9e3919fbf9c261569c";
const QUALIFIED_H003R_ARTIFACT_DIGEST: &str =
    "sha256:77dc89358ebdcf28438b89834cb611cb27aa56e3060da9280bb809300e9d6faa";
const HISTORICAL_H003_SUBJECT: &str = "962c3b72350251713c54b7b6328960a2a6f35595";
const W_SHELL: f64 = 0.25;
const SEEDS: [u64; 8] = [2, 3, 5, 7, 11, 13, 17, 19];
const SUCCESS_GATE_PER_TRANSFORM: usize = 6;
const MAX_SUCCESS_SPREAD: usize = 2;

fn main() -> Result<(), Box<dyn Error>> {
    let (subject, output) = parse_args()?;
    let study = match run_study() {
        Ok(value) => value,
        Err(error) => json!({
            "pass": false,
            "status": "ERROR",
            "error": error.to_string(),
        }),
    };
    let pass = study.get("pass").and_then(Value::as_bool) == Some(true);
    let report = json!({
        "schema": SCHEMA,
        "source_subject": subject,
        "preregistration_issue": PREREG_ISSUE,
        "qualified_hyperspace_003r_head": QUALIFIED_H003R_HEAD,
        "qualified_hyperspace_003r_artifact_digest": QUALIFIED_H003R_ARTIFACT_DIGEST,
        "historical_hyperspace_003_subject": HISTORICAL_H003_SUBJECT,
        "historical_hyperspace_003_verdict": "FAIL",
        "target_receives_canonical_witness": false,
        "target_receives_privileged_w_axis": false,
        "claim_scope": "blind deterministic seeded-RRT discovery across six frozen tilted representations of the synthetic finite-w shell only",
        "verdict": if pass { "PASS" } else { "FAIL" },
        "study": study,
    });

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "HYPERSPACE-004 rotated-planner verdict: {}",
        report["verdict"].as_str().unwrap_or("FAIL")
    );
    println!("receipt: {}", output.display());
    Ok(())
}

fn parse_args() -> Result<(String, PathBuf), Box<dyn Error>> {
    let mut subject = None;
    let mut output = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--subject" => subject = args.next(),
            "--output" => output = args.next().map(PathBuf::from),
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    let subject = subject.ok_or("--subject is required")?;
    if subject.len() != 40 || !subject.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("--subject must be an exact 40-hex commit SHA".into());
    }
    Ok((subject.to_ascii_lowercase(), output.ok_or("--output is required")?))
}

fn run_study() -> Result<Value, Box<dyn Error>> {
    let feasible_canonical = FiniteWShellOracle::new(
        HyperspaceDimension::R4,
        1.0,
        2.0,
        W_SHELL,
        4.0,
        2.0,
    )?;
    let feasible_canonical_problem = canonical_shell_problem(&feasible_canonical, 3.0)?;

    let blocked_canonical = FiniteWShellOracle::new(
        HyperspaceDimension::R4,
        1.0,
        2.0,
        W_SHELL,
        4.0,
        0.20,
    )?;
    let blocked_canonical_problem = canonical_shell_problem(&blocked_canonical, 3.0)?;
    let blocked_certificate =
        certify_radial_separation(&blocked_canonical_problem, &blocked_canonical)?;

    let projected_r3 = FiniteWShellOracle::new(
        HyperspaceDimension::R3,
        1.0,
        2.0,
        0.0,
        4.0,
        0.0,
    )?;

    let transforms = frozen_transforms();
    if transforms.len() != 6 {
        return Err(format!("expected six frozen transforms, got {}", transforms.len()).into());
    }

    let mut transform_reports = Vec::with_capacity(6);
    let mut success_counts = Vec::with_capacity(6);
    let mut all_transforms_pass = true;

    for spec in transforms {
        let feasible_oracle = RotatedSamplingOracle::new(feasible_canonical.clone(), spec.matrix)?;
        let transform_identity = hex32(feasible_oracle.transform_identity());
        if transform_identity != spec.expected_feasible_transform_identity {
            return Err(format!(
                "{} transform identity drift: expected {}, got {}",
                spec.name, spec.expected_feasible_transform_identity, transform_identity
            )
            .into());
        }
        validate_envelope(&feasible_oracle)?;
        let feasible_problem = rotated_problem(&feasible_canonical_problem, &feasible_oracle)?;

        let blocked_oracle = RotatedSamplingOracle::new(blocked_canonical.clone(), spec.matrix)?;
        validate_envelope(&blocked_oracle)?;
        let blocked_problem = rotated_problem(&blocked_canonical_problem, &blocked_oracle)?;

        let mut feasible_runs = Vec::with_capacity(SEEDS.len());
        let mut blocked_runs = Vec::with_capacity(SEEDS.len());
        let mut successes = 0_usize;
        let mut quality_failures = 0_usize;
        let mut blocked_false_feasible = 0_usize;
        let mut execution_errors = 0_usize;

        for seed in SEEDS {
            match run_escape_seed(
                seed,
                &feasible_problem,
                &feasible_oracle,
                &projected_r3,
            ) {
                Ok((record, qualified)) => {
                    if qualified {
                        successes += 1;
                    } else if record.get("planner_kind").and_then(Value::as_str)
                        == Some("Feasible")
                    {
                        quality_failures += 1;
                    }
                    feasible_runs.push(record);
                }
                Err(error) => {
                    execution_errors += 1;
                    feasible_runs.push(json!({
                        "seed": seed,
                        "status": "ERROR",
                        "error": error.to_string(),
                    }));
                }
            }

            match run_blocked_seed(seed, &blocked_problem, &blocked_oracle) {
                Ok((record, false_feasible)) => {
                    if false_feasible {
                        blocked_false_feasible += 1;
                    }
                    blocked_runs.push(record);
                }
                Err(error) => {
                    execution_errors += 1;
                    blocked_runs.push(json!({
                        "seed": seed,
                        "status": "ERROR",
                        "error": error.to_string(),
                    }));
                }
            }
        }

        let transform_pass = successes >= SUCCESS_GATE_PER_TRANSFORM
            && quality_failures == 0
            && blocked_false_feasible == 0
            && execution_errors == 0;
        all_transforms_pass &= transform_pass;
        success_counts.push(successes);

        transform_reports.push(json!({
            "name": spec.name,
            "expected_feasible_transform_identity": spec.expected_feasible_transform_identity,
            "feasible_transform_identity": transform_identity,
            "blocked_transform_identity": hex32(blocked_oracle.transform_identity()),
            "matrix_bits": matrix_bits(feasible_oracle.matrix()),
            "feasible_oracle_identity": hex32(feasible_oracle.profile().identity()),
            "blocked_oracle_identity": hex32(blocked_oracle.profile().identity()),
            "feasible_problem_identity": hex32(feasible_problem.identity()),
            "blocked_problem_identity": hex32(blocked_problem.identity()),
            "feasible_sampling_min_bits": bits(feasible_oracle.sampling_min()),
            "feasible_sampling_max_bits": bits(feasible_oracle.sampling_max()),
            "blocked_sampling_min_bits": bits(blocked_oracle.sampling_min()),
            "blocked_sampling_max_bits": bits(blocked_oracle.sampling_max()),
            "feasible_envelope_volume_ratio": feasible_oracle.envelope_volume_ratio(),
            "blocked_envelope_volume_ratio": blocked_oracle.envelope_volume_ratio(),
            "successes": successes,
            "success_gate": SUCCESS_GATE_PER_TRANSFORM,
            "quality_failures": quality_failures,
            "blocked_false_feasible": blocked_false_feasible,
            "execution_errors": execution_errors,
            "pass": transform_pass,
            "escape_runs": feasible_runs,
            "blocked_runs": blocked_runs,
        }));
    }

    let min_successes = *success_counts.iter().min().ok_or("missing success counts")?;
    let max_successes = *success_counts.iter().max().ok_or("missing success counts")?;
    let success_spread = max_successes - min_successes;
    let spread_gate = success_spread <= MAX_SUCCESS_SPREAD;
    let pass = all_transforms_pass && spread_gate;

    Ok(json!({
        "pass": pass,
        "status": if pass { "PASS" } else { "FAIL" },
        "seeds": SEEDS,
        "transform_count": transform_reports.len(),
        "required_transform_count": 6,
        "preregistered_gates": {
            "successes_per_transform_at_least": SUCCESS_GATE_PER_TRANSFORM,
            "blocked_false_feasible_per_transform": 0,
            "quality_failures_per_transform": 0,
            "execution_errors_per_transform": 0,
            "max_success_count_spread": MAX_SUCCESS_SPREAD,
            "material_hidden_coordinate_threshold": W_SHELL + MATERIAL_CLEARANCE,
            "roundtrip_tolerance": ROUNDTRIP_TOLERANCE,
        },
        "escape_rrt_profile": {
            "max_iterations": 5000,
            "step_size": 0.75,
            "goal_bias": 0.10,
            "goal_connection_radius": 5.0,
        },
        "blocked_rrt_profile": {
            "max_iterations": 2000,
            "step_size": 0.4,
            "goal_bias": 0.05,
            "goal_connection_radius": 0.6,
        },
        "canonical_feasible_oracle_identity": hex32(feasible_canonical.profile().identity()),
        "canonical_feasible_problem_identity": hex32(feasible_canonical_problem.identity()),
        "canonical_blocked_oracle_identity": hex32(blocked_canonical.profile().identity()),
        "canonical_blocked_problem_identity": hex32(blocked_canonical_problem.identity()),
        "canonical_blocked_certificate_identity": hex32(blocked_certificate.identity()),
        "projected_r3_oracle_identity": hex32(projected_r3.profile().identity()),
        "success_counts": success_counts,
        "success_count_spread": success_spread,
        "spread_gate": spread_gate,
        "transforms": transform_reports,
    }))
}

fn rotated_problem(
    canonical_problem: &EuclideanPlanningProblem,
    oracle: &RotatedSamplingOracle,
) -> Result<EuclideanPlanningProblem, ContinuousReachabilityError> {
    EuclideanPlanningProblem::new(
        4,
        oracle.to_external(canonical_problem.start())?,
        oracle.to_external(canonical_problem.goal())?,
        oracle,
    )
}

fn run_escape_seed(
    seed: u64,
    problem: &EuclideanPlanningProblem,
    oracle: &RotatedSamplingOracle,
    projected_r3: &FiniteWShellOracle,
) -> Result<(Value, bool), Box<dyn Error>> {
    let config = RrtConfig::new(seed, 5_000, 0.75, 0.10, 5.0)?;
    let config_identity = config.identity();
    match seeded_rrt(problem, oracle, config)? {
        SamplingReachabilityResult::Feasible {
            path,
            validation,
            solver,
        } => {
            let replay = match validate_euclidean_path(problem, oracle, &path)? {
                ContinuousPathReplay::Valid(receipt) => receipt,
                other => {
                    return Err(format!(
                        "seed {seed} feasible candidate contradicted external replay: {other:?}"
                    )
                    .into());
                }
            };
            let replay_identity_match = replay.path_identity() == validation.path_identity();
            if !replay_identity_match {
                return Err(format!("seed {seed} external replay identity mismatch").into());
            }

            let canonical_path: Vec<Vec<f64>> = path
                .iter()
                .map(|state| oracle.to_canonical(state))
                .collect::<Result<_, _>>()?;
            let max_abs_canonical_w = canonical_path
                .iter()
                .map(|state| state[3].abs())
                .fold(0.0_f64, f64::max);
            let material_hidden_coordinate =
                max_abs_canonical_w > W_SHELL + MATERIAL_CLEARANCE;
            let roundtrip_error = path_roundtrip_error(oracle, &path)?;
            let projected_collision = projection_has_collision(projected_r3, &canonical_path)?;
            let qualified = replay_identity_match
                && material_hidden_coordinate
                && roundtrip_error <= ROUNDTRIP_TOLERANCE
                && projected_collision;

            Ok((
                json!({
                    "seed": seed,
                    "status": if qualified { "PASS" } else { "FAIL" },
                    "planner_kind": "Feasible",
                    "config_identity": hex32(config_identity),
                    "solver_identity": hex32(solver.solver_identity()),
                    "iterations": solver.iterations(),
                    "accepted_nodes": solver.accepted_nodes(),
                    "state_queries": solver.state_queries(),
                    "segment_queries": solver.segment_queries(),
                    "path_identity": hex32(replay.path_identity()),
                    "total_cost": replay.total_cost(),
                    "replay_identity_match": replay_identity_match,
                    "max_abs_canonical_w": max_abs_canonical_w,
                    "material_hidden_coordinate": material_hidden_coordinate,
                    "material_threshold": W_SHELL + MATERIAL_CLEARANCE,
                    "external_roundtrip_error": roundtrip_error,
                    "roundtrip_limit": ROUNDTRIP_TOLERANCE,
                    "projected_collision": projected_collision,
                }),
                qualified,
            ))
        }
        SamplingReachabilityResult::Unknown {
            reason,
            detail,
            best_partial_path,
            solver,
        } => Ok((
            json!({
                "seed": seed,
                "status": "UNKNOWN",
                "planner_kind": "Unknown",
                "reason": format!("{reason:?}"),
                "detail": detail,
                "config_identity": hex32(config_identity),
                "solver_identity": hex32(solver.solver_identity()),
                "iterations": solver.iterations(),
                "accepted_nodes": solver.accepted_nodes(),
                "state_queries": solver.state_queries(),
                "segment_queries": solver.segment_queries(),
                "best_partial_waypoints": best_partial_path.len(),
            }),
            false,
        )),
    }
}

fn run_blocked_seed(
    seed: u64,
    problem: &EuclideanPlanningProblem,
    oracle: &RotatedSamplingOracle,
) -> Result<(Value, bool), Box<dyn Error>> {
    let config = RrtConfig::new(seed, 2_000, 0.4, 0.05, 0.6)?;
    let config_identity = config.identity();
    match seeded_rrt(problem, oracle, config)? {
        SamplingReachabilityResult::Feasible {
            path,
            validation,
            solver,
        } => {
            let replay = match validate_euclidean_path(problem, oracle, &path)? {
                ContinuousPathReplay::Valid(receipt) => receipt,
                other => {
                    return Err(format!(
                        "blocked seed {seed} solver/replay contradiction: {other:?}"
                    )
                    .into());
                }
            };
            Ok((
                json!({
                    "seed": seed,
                    "status": "FALSE_FEASIBLE",
                    "planner_kind": "Feasible",
                    "config_identity": hex32(config_identity),
                    "solver_identity": hex32(solver.solver_identity()),
                    "iterations": solver.iterations(),
                    "accepted_nodes": solver.accepted_nodes(),
                    "state_queries": solver.state_queries(),
                    "segment_queries": solver.segment_queries(),
                    "path_identity": hex32(replay.path_identity()),
                    "solver_validation_identity": hex32(validation.path_identity()),
                }),
                true,
            ))
        }
        SamplingReachabilityResult::Unknown {
            reason,
            detail,
            best_partial_path,
            solver,
        } => Ok((
            json!({
                "seed": seed,
                "status": "UNKNOWN",
                "planner_kind": "Unknown",
                "reason": format!("{reason:?}"),
                "detail": detail,
                "config_identity": hex32(config_identity),
                "solver_identity": hex32(solver.solver_identity()),
                "iterations": solver.iterations(),
                "accepted_nodes": solver.accepted_nodes(),
                "state_queries": solver.state_queries(),
                "segment_queries": solver.segment_queries(),
                "best_partial_waypoints": best_partial_path.len(),
            }),
            false,
        )),
    }
}

fn projection_has_collision(
    projected_r3: &FiniteWShellOracle,
    canonical_path: &[Vec<f64>],
) -> Result<bool, ContinuousReachabilityError> {
    if canonical_path.len() < 2 {
        return Err(ContinuousReachabilityError::OracleEvaluation {
            reason: "projection diagnostic requires at least two waypoints".to_string(),
        });
    }
    for segment in canonical_path.windows(2) {
        let from = &segment[0][0..3];
        let to = &segment[1][0..3];
        match projected_r3.segment_verdict(from, to)? {
            OracleVerdict::Invalid { .. } => return Ok(true),
            OracleVerdict::Valid => {}
            OracleVerdict::Unknown { reason } => {
                return Err(ContinuousReachabilityError::OracleEvaluation {
                    reason: format!("R3 projection diagnostic returned Unknown: {reason}"),
                });
            }
        }
    }
    Ok(false)
}

fn validate_envelope(oracle: &RotatedSamplingOracle) -> Result<(), Box<dyn Error>> {
    if oracle.sampling_min().len() != 4 || oracle.sampling_max().len() != 4 {
        return Err("rotated sampling envelope must contain four coordinates".into());
    }
    for (&min, &max) in oracle.sampling_min().iter().zip(oracle.sampling_max()) {
        if !min.is_finite() || !max.is_finite() || min >= max {
            return Err("rotated sampling envelope must be finite with min < max".into());
        }
    }
    Ok(())
}

fn bits(values: &[f64]) -> Vec<String> {
    values
        .iter()
        .map(|value| format!("{:016x}", value.to_bits()))
        .collect()
}

fn hex32(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}
