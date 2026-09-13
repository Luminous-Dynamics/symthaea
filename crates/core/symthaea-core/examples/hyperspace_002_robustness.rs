// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! HYPERSPACE-002 preregistered coordinate-permutation and seed robustness study.
//!
//! This is a follow-up to HYPERSPACE-001. It must never overwrite or rescue the
//! primary HYPERSPACE-001 verdict. The study asks whether the same exact synthetic
//! R4 navigation problem remains solvable when the extra coordinate is represented
//! at different vector indices, while analytically blocked controls remain free of
//! false feasible planner results.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use serde_json::{Value, json};
use symthaea_core::continuous_reachability::{
    ContinuousPathReplay, ContinuousReachabilityError, ContinuousValidityOracle,
    ContinuousValidityOracleProfile, EuclideanPlanningProblem, OracleVerdict,
    validate_euclidean_path,
};
use symthaea_core::hyperspace_benchmark::{
    FiniteWShellOracle, HyperspaceDimension, canonical_shell_problem,
    certify_radial_separation, evaluator_reference_escape_path, qualify_projection_trap,
};
use symthaea_core::reachability::UnknownReason;
use symthaea_core::sampling_reachability::{
    BoundedEuclideanSamplingOracle, RrtConfig, SamplingReachabilityResult, seeded_rrt,
};
use symthaea_core::state_space::{EuclideanSpace, StateSpace};

const SCHEMA: &str = "symthaea.hyperspace-002.robustness.v1";
const PREREG_ISSUE: u64 = 2317;
const PRIMARY_HYPERSPACE_001_HEAD: &str = "52c411aa7c89482e3aac66e0c15114f560afa835";
const SEEDS: [u64; 8] = [2, 3, 5, 7, 11, 13, 17, 19];
const W_SHELL: f64 = 0.25;
const SUCCESS_GATE_PER_AXIS: usize = 6;
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
        "primary_hyperspace_001_head": PRIMARY_HYPERSPACE_001_HEAD,
        "primary_verdict_is_independent": true,
        "combined_claim_requires_primary_pass": true,
        "claim_scope": "synthetic Euclidean coordinate-representation robustness only; not evidence for physically accessible extra dimensions",
        "verdict": if pass { "PASS" } else { "FAIL" },
        "study": study,
    });

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!("HYPERSPACE-002 robustness verdict: {}", report["verdict"].as_str().unwrap_or("FAIL"));
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

#[derive(Clone, Debug)]
struct PermutedShellOracle {
    canonical: FiniteWShellOracle,
    canonical_from_external: [usize; 4],
    external_from_canonical: [usize; 4],
    sampling_min: Vec<f64>,
    sampling_max: Vec<f64>,
    profile: ContinuousValidityOracleProfile,
}

impl PermutedShellOracle {
    fn new(canonical: FiniteWShellOracle, external_w_axis: usize) -> Result<Self, Box<dyn Error>> {
        if canonical.benchmark_dimension() != HyperspaceDimension::R4 {
            return Err("permutation wrapper requires canonical R4 oracle".into());
        }
        if external_w_axis >= 4 {
            return Err(format!("external w axis must be in 0..4, got {external_w_axis}").into());
        }
        let spatial_external: Vec<usize> = (0..4).filter(|axis| *axis != external_w_axis).collect();
        let canonical_from_external = [
            spatial_external[0],
            spatial_external[1],
            spatial_external[2],
            external_w_axis,
        ];
        let mut external_from_canonical = [0_usize; 4];
        for (canonical_axis, external_axis) in canonical_from_external.iter().copied().enumerate() {
            external_from_canonical[external_axis] = canonical_axis;
        }
        let mut sampling_min = vec![0.0; 4];
        let mut sampling_max = vec![0.0; 4];
        for external_axis in 0..4 {
            let canonical_axis = external_from_canonical[external_axis];
            sampling_min[external_axis] = canonical.sampling_min()[canonical_axis];
            sampling_max[external_axis] = canonical.sampling_max()[canonical_axis];
        }
        let state_space = EuclideanSpace::new(4);
        let mut parameters = Vec::new();
        parameters.extend_from_slice(&canonical.profile().identity());
        parameters.extend(canonical_from_external.iter().map(|axis| *axis as u8));
        let profile = ContinuousValidityOracleProfile::new(
            state_space.profile().identity(),
            "hyperspace-coordinate-permutation-v1",
            parameters,
        )?;
        Ok(Self {
            canonical,
            canonical_from_external,
            external_from_canonical,
            sampling_min,
            sampling_max,
            profile,
        })
    }

    fn to_canonical(&self, external: &[f64]) -> Result<Vec<f64>, ContinuousReachabilityError> {
        if external.len() != 4 || external.iter().any(|value| !value.is_finite()) {
            return Err(ContinuousReachabilityError::OracleEvaluation {
                reason: "permuted shell state must contain exactly four finite coordinates".to_string(),
            });
        }
        let mut canonical = vec![0.0; 4];
        for canonical_axis in 0..4 {
            canonical[canonical_axis] = external[self.canonical_from_external[canonical_axis]];
        }
        Ok(canonical)
    }

    fn to_external(&self, canonical: &[f64]) -> Result<Vec<f64>, ContinuousReachabilityError> {
        if canonical.len() != 4 || canonical.iter().any(|value| !value.is_finite()) {
            return Err(ContinuousReachabilityError::OracleEvaluation {
                reason: "canonical shell state must contain exactly four finite coordinates".to_string(),
            });
        }
        let mut external = vec![0.0; 4];
        for external_axis in 0..4 {
            external[external_axis] = canonical[self.external_from_canonical[external_axis]];
        }
        Ok(external)
    }

    fn permutation(&self) -> [usize; 4] {
        self.canonical_from_external
    }
}

impl ContinuousValidityOracle for PermutedShellOracle {
    fn profile(&self) -> &ContinuousValidityOracleProfile {
        &self.profile
    }
    fn state_verdict(&self, state: &[f64]) -> Result<OracleVerdict, ContinuousReachabilityError> {
        self.canonical.state_verdict(&self.to_canonical(state)?)
    }
    fn segment_verdict(&self, from: &[f64], to: &[f64]) -> Result<OracleVerdict, ContinuousReachabilityError> {
        self.canonical.segment_verdict(&self.to_canonical(from)?, &self.to_canonical(to)?)
    }
}

impl BoundedEuclideanSamplingOracle for PermutedShellOracle {
    fn sampling_min(&self) -> &[f64] { &self.sampling_min }
    fn sampling_max(&self) -> &[f64] { &self.sampling_max }
}

fn run_study() -> Result<Value, Box<dyn Error>> {
    let feasible_canonical = FiniteWShellOracle::new(HyperspaceDimension::R4, 1.0, 2.0, W_SHELL, 4.0, 2.0)?;
    let feasible_canonical_problem = canonical_shell_problem(&feasible_canonical, 3.0)?;
    let canonical_witness = evaluator_reference_escape_path(&feasible_canonical_problem, &feasible_canonical, 0.25)?;
    let blocked_canonical = FiniteWShellOracle::new(HyperspaceDimension::R4, 1.0, 2.0, W_SHELL, 4.0, 0.20)?;
    let blocked_canonical_problem = canonical_shell_problem(&blocked_canonical, 3.0)?;
    let blocked_certificate = certify_radial_separation(&blocked_canonical_problem, &blocked_canonical)?;

    let mut axis_reports = Vec::with_capacity(4);
    let mut success_counts = Vec::with_capacity(4);
    let mut all_axes_pass = true;

    for external_w_axis in 0..4 {
        let feasible_oracle = PermutedShellOracle::new(feasible_canonical.clone(), external_w_axis)?;
        let feasible_problem = permuted_problem(&feasible_canonical_problem, &feasible_oracle)?;
        let transformed_witness: Vec<Vec<f64>> = canonical_witness.iter().map(|state| feasible_oracle.to_external(state)).collect::<Result<_, _>>()?;
        let witness_validation = match validate_euclidean_path(&feasible_problem, &feasible_oracle, &transformed_witness)? {
            ContinuousPathReplay::Valid(validation) => validation,
            other => return Err(format!("axis {external_w_axis} transformed evaluator witness did not replay valid: {other:?}").into()),
        };
        let blocked_oracle = PermutedShellOracle::new(blocked_canonical.clone(), external_w_axis)?;
        let blocked_problem = permuted_problem(&blocked_canonical_problem, &blocked_oracle)?;
        let mut feasible_runs = Vec::with_capacity(SEEDS.len());
        let mut blocked_runs = Vec::with_capacity(SEEDS.len());
        let mut successes = 0_usize;
        let mut quality_failures = 0_usize;
        let mut blocked_false_feasible = 0_usize;
        let mut execution_errors = 0_usize;

        for seed in SEEDS {
            match run_escape_seed(seed, external_w_axis, &feasible_problem, &feasible_oracle, &feasible_canonical_problem, &feasible_canonical) {
                Ok((record, qualified_success)) => {
                    if qualified_success { successes += 1; }
                    else if record.get("planner_kind").and_then(Value::as_str) == Some("Feasible") { quality_failures += 1; }
                    feasible_runs.push(record);
                }
                Err(error) => {
                    execution_errors += 1;
                    feasible_runs.push(json!({"seed": seed, "status": "ERROR", "error": error.to_string()}));
                }
            }
            match run_blocked_seed(seed, &blocked_problem, &blocked_oracle) {
                Ok((record, false_feasible)) => {
                    if false_feasible { blocked_false_feasible += 1; }
                    blocked_runs.push(record);
                }
                Err(error) => {
                    execution_errors += 1;
                    blocked_runs.push(json!({"seed": seed, "status": "ERROR", "error": error.to_string()}));
                }
            }
        }

        let axis_pass = successes >= SUCCESS_GATE_PER_AXIS && quality_failures == 0 && blocked_false_feasible == 0 && execution_errors == 0;
        all_axes_pass &= axis_pass;
        success_counts.push(successes);
        axis_reports.push(json!({
            "external_w_axis": external_w_axis,
            "canonical_from_external": feasible_oracle.permutation(),
            "permutation_identity": hex32(feasible_oracle.profile().identity()),
            "feasible_problem_identity": hex32(feasible_problem.identity()),
            "blocked_problem_identity": hex32(blocked_problem.identity()),
            "transformed_witness_path_identity": hex32(witness_validation.path_identity()),
            "blocked_certificate_identity": hex32(blocked_certificate.identity()),
            "successes": successes,
            "success_gate": SUCCESS_GATE_PER_AXIS,
            "quality_failures": quality_failures,
            "blocked_false_feasible": blocked_false_feasible,
            "execution_errors": execution_errors,
            "pass": axis_pass,
            "escape_runs": feasible_runs,
            "blocked_runs": blocked_runs,
        }));
    }

    let min_successes = *success_counts.iter().min().ok_or("missing success counts")?;
    let max_successes = *success_counts.iter().max().ok_or("missing success counts")?;
    let success_spread = max_successes - min_successes;
    let representation_invariance = success_spread <= MAX_SUCCESS_SPREAD;
    let pass = all_axes_pass && representation_invariance;
    Ok(json!({
        "pass": pass,
        "status": if pass { "PASS" } else { "FAIL" },
        "seeds": SEEDS,
        "preregistered_gates": {
            "successes_per_axis_at_least": SUCCESS_GATE_PER_AXIS,
            "blocked_false_feasible_per_axis": 0,
            "quality_failures_per_axis": 0,
            "max_success_count_spread": MAX_SUCCESS_SPREAD,
        },
        "escape_rrt_profile": {"max_iterations": 5000, "step_size": 0.75, "goal_bias": 0.10, "goal_connection_radius": 5.0},
        "blocked_rrt_profile": {"max_iterations": 2000, "step_size": 0.4, "goal_bias": 0.05, "goal_connection_radius": 0.6},
        "canonical_feasible_problem_identity": hex32(feasible_canonical_problem.identity()),
        "canonical_feasible_oracle_identity": hex32(feasible_canonical.profile().identity()),
        "canonical_blocked_problem_identity": hex32(blocked_canonical_problem.identity()),
        "canonical_blocked_oracle_identity": hex32(blocked_canonical.profile().identity()),
        "canonical_blocked_certificate_identity": hex32(blocked_certificate.identity()),
        "success_counts": success_counts,
        "success_count_spread": success_spread,
        "representation_invariance_gate": representation_invariance,
        "axes": axis_reports,
    }))
}

fn permuted_problem(canonical_problem: &EuclideanPlanningProblem, oracle: &PermutedShellOracle) -> Result<EuclideanPlanningProblem, ContinuousReachabilityError> {
    EuclideanPlanningProblem::new(4, oracle.to_external(canonical_problem.start())?, oracle.to_external(canonical_problem.goal())?, oracle)
}

fn run_escape_seed(seed: u64, external_w_axis: usize, problem: &EuclideanPlanningProblem, oracle: &PermutedShellOracle, canonical_problem: &EuclideanPlanningProblem, canonical_oracle: &FiniteWShellOracle) -> Result<(Value, bool), Box<dyn Error>> {
    let config = RrtConfig::new(seed, 5_000, 0.75, 0.10, 5.0)?;
    let config_identity = config.identity();
    match seeded_rrt(problem, oracle, config)? {
        SamplingReachabilityResult::Feasible { path, validation, solver } => {
            let replay = match validate_euclidean_path(problem, oracle, &path)? {
                ContinuousPathReplay::Valid(replay) => replay,
                other => return Err(format!("seed {seed} feasible candidate contradicted independent replay: {other:?}").into()),
            };
            if replay.path_identity() != validation.path_identity() { return Err(format!("seed {seed} replay identity mismatch").into()); }
            let max_abs_w = path.iter().map(|state| state[external_w_axis].abs()).fold(0.0_f64, f64::max);
            let material_w = max_abs_w > W_SHELL;
            let canonical_path: Vec<Vec<f64>> = path.iter().map(|state| oracle.to_canonical(state)).collect::<Result<_, _>>()?;
            let projection = qualify_projection_trap(canonical_problem, canonical_oracle, &canonical_path);
            let projection_ok = projection.is_ok();
            let projection_identity = projection.ok().map(|receipt| hex32(receipt.identity()));
            let qualified = material_w && projection_ok;
            Ok((json!({
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
                "max_abs_w": max_abs_w,
                "material_extra_coordinate": material_w,
                "projection_trap_established": projection_ok,
                "projection_trap_identity": projection_identity,
            }), qualified))
        }
        SamplingReachabilityResult::Unknown { reason, detail, best_partial_path, solver } => Ok((json!({
            "seed": seed,
            "status": "UNKNOWN",
            "planner_kind": "Unknown",
            "reason": unknown_reason_name(&reason),
            "detail": detail,
            "config_identity": hex32(config_identity),
            "solver_identity": hex32(solver.solver_identity()),
            "iterations": solver.iterations(),
            "accepted_nodes": solver.accepted_nodes(),
            "state_queries": solver.state_queries(),
            "segment_queries": solver.segment_queries(),
            "best_partial_waypoints": best_partial_path.len(),
        }), false)),
    }
}

fn run_blocked_seed(seed: u64, problem: &EuclideanPlanningProblem, oracle: &PermutedShellOracle) -> Result<(Value, bool), Box<dyn Error>> {
    let config = RrtConfig::new(seed, 2_000, 0.4, 0.05, 0.6)?;
    let config_identity = config.identity();
    match seeded_rrt(problem, oracle, config)? {
        SamplingReachabilityResult::Feasible { path, validation, solver } => Ok((json!({
            "seed": seed,
            "status": "FALSE_FEASIBLE",
            "planner_kind": "Feasible",
            "config_identity": hex32(config_identity),
            "solver_identity": hex32(solver.solver_identity()),
            "iterations": solver.iterations(),
            "accepted_nodes": solver.accepted_nodes(),
            "path_identity": hex32(validation.path_identity()),
            "waypoints": path.len(),
        }), true)),
        SamplingReachabilityResult::Unknown { reason, detail, solver, .. } => Ok((json!({
            "seed": seed,
            "status": "EXPECTED_UNKNOWN",
            "planner_kind": "Unknown",
            "reason": unknown_reason_name(&reason),
            "detail": detail,
            "config_identity": hex32(config_identity),
            "solver_identity": hex32(solver.solver_identity()),
            "iterations": solver.iterations(),
            "accepted_nodes": solver.accepted_nodes(),
            "state_queries": solver.state_queries(),
            "segment_queries": solver.segment_queries(),
        }), false)),
    }
}

fn unknown_reason_name(reason: &UnknownReason) -> &'static str {
    match reason {
        UnknownReason::BudgetExceeded => "BudgetExceeded",
        UnknownReason::NoPathFound => "NoPathFound",
        UnknownReason::ValidityOracleIncomplete => "ValidityOracleIncomplete",
        UnknownReason::NumericalFailure => "NumericalFailure",
        UnknownReason::ModelOutOfDomain => "ModelOutOfDomain",
        UnknownReason::ConstraintProjectionFailure => "ConstraintProjectionFailure",
        UnknownReason::UnsupportedSpace => "UnsupportedSpace",
    }
}

fn hex32(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}
