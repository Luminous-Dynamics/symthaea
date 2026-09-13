// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! HYPERSPACE-002Q exact coordinate-permutation equivariance control.
//!
//! This deterministic experiment isolates representation equivalence from the
//! stochastic planner robustness study in HYPERSPACE-002. It enumerates all 24
//! permutations of canonical `(x, y, z, w)` and requires exact oracle/validator
//! agreement on a frozen analytic corpus. It makes no physical extra-dimension
//! claim and cannot rescue a HYPERSPACE-002 planner failure.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use blake3::Hasher;
use serde_json::{Value, json};
use symthaea_core::continuous_reachability::{
    ContinuousPathReplay, ContinuousPathValidationReceipt, ContinuousReachabilityError,
    ContinuousValidityOracle, EuclideanPlanningProblem, validate_euclidean_path,
};
use symthaea_core::hyperspace_benchmark::{
    FiniteWShellOracle, HyperspaceDimension, canonical_shell_problem,
    evaluator_reference_escape_path,
};
use symthaea_core::sampling_reachability::BoundedEuclideanSamplingOracle;

#[path = "hyperspace_support/permutation.rs"]
mod permutation;
use permutation::PermutationOracle;

const SCHEMA: &str = "symthaea.hyperspace-002q.equivariance.v1";
const PREREG_ISSUE: u64 = 2346;
const W_SHELL: f64 = 0.25;
const BINARY_OFFSET: f64 = 0.000_976_562_5; // 2^-10

fn main() -> Result<(), Box<dyn Error>> {
    let (subject, hyperspace_002_head, output) = parse_args()?;
    let control = match run_control() {
        Ok(value) => value,
        Err(error) => json!({
            "pass": false,
            "status": "ERROR",
            "error": error.to_string(),
        }),
    };
    let pass = control.get("pass").and_then(Value::as_bool) == Some(true);
    let report = json!({
        "schema": SCHEMA,
        "source_subject": subject,
        "preregistration_issue": PREREG_ISSUE,
        "hyperspace_002_head": hyperspace_002_head,
        "permutation_adapter": "examples/hyperspace_support/permutation.rs",
        "claim_scope": "synthetic Euclidean oracle/validator coordinate-permutation equivariance only; not planner robustness or physical extra-dimension evidence",
        "can_rescue_hyperspace_002_failure": false,
        "verdict": if pass { "PASS" } else { "FAIL" },
        "control": control,
    });

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "HYPERSPACE-002Q equivariance verdict: {}",
        report["verdict"].as_str().unwrap_or("FAIL")
    );
    println!("receipt: {}", output.display());
    Ok(())
}

fn parse_args() -> Result<(String, String, PathBuf), Box<dyn Error>> {
    let mut subject = None;
    let mut h002 = None;
    let mut output = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--subject" => subject = args.next(),
            "--hyperspace-002-head" => h002 = args.next(),
            "--output" => output = args.next().map(PathBuf::from),
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    let subject = validate_sha(subject.ok_or("--subject is required")?, "--subject")?;
    let h002 = validate_sha(
        h002.ok_or("--hyperspace-002-head is required")?,
        "--hyperspace-002-head",
    )?;
    Ok((subject, h002, output.ok_or("--output is required")?))
}

fn validate_sha(value: String, flag: &str) -> Result<String, Box<dyn Error>> {
    if value.len() != 40 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{flag} must be an exact 40-hex commit SHA").into());
    }
    Ok(value.to_ascii_lowercase())
}

#[derive(Clone, Debug)]
struct StateCase {
    name: &'static str,
    state: Vec<f64>,
}

#[derive(Clone, Debug)]
struct SegmentCase {
    name: &'static str,
    from: Vec<f64>,
    to: Vec<f64>,
}

#[derive(Clone, Debug)]
struct PathCase {
    name: &'static str,
    path: Vec<Vec<f64>>,
}

fn run_control() -> Result<Value, Box<dyn Error>> {
    let canonical = FiniteWShellOracle::new(
        HyperspaceDimension::R4,
        1.0,
        2.0,
        W_SHELL,
        4.0,
        2.0,
    )?;
    let canonical_problem = canonical_shell_problem(&canonical, 3.0)?;
    let reference_path =
        evaluator_reference_escape_path(&canonical_problem, &canonical, 0.25)?;
    let canonical_reference = require_valid_replay(validate_euclidean_path(
        &canonical_problem,
        &canonical,
        &reference_path,
    )?)?;

    let states = frozen_states(&canonical_problem);
    let segments = frozen_segments(&canonical_problem);
    let paths = frozen_paths(&canonical_problem, &reference_path);
    let corpus_identity = corpus_identity(&canonical, &states, &segments, &paths);
    let permutations = permutations4();
    if permutations.len() != 24 {
        return Err(format!("expected 24 permutations, got {}", permutations.len()).into());
    }

    let mut permutation_reports = Vec::with_capacity(24);
    let mut total_mismatches = 0_usize;

    for permutation in permutations {
        let oracle = PermutationOracle::new(canonical.clone(), permutation)?;
        let problem = EuclideanPlanningProblem::new(
            4,
            oracle.to_external(canonical_problem.start())?,
            oracle.to_external(canonical_problem.goal())?,
            &oracle,
        )?;
        let mut mismatches = Vec::<String>::new();

        check_bounds_roundtrip(&canonical, &oracle, &mut mismatches)?;
        check_problem_roundtrip(&canonical_problem, &problem, &oracle, &mut mismatches)?;

        for case in &states {
            let external = oracle.to_external(&case.state)?;
            let roundtrip = oracle.to_canonical(&external)?;
            if !same_bits(&roundtrip, &case.state) {
                mismatches.push(format!("state {} roundtrip mismatch", case.name));
            }
            let canonical_verdict = canonical.state_verdict(&case.state)?;
            let external_verdict = oracle.state_verdict(&external)?;
            if canonical_verdict != external_verdict {
                mismatches.push(format!(
                    "state {} verdict mismatch: canonical={canonical_verdict:?} external={external_verdict:?}",
                    case.name
                ));
            }
        }

        for case in &segments {
            let external_from = oracle.to_external(&case.from)?;
            let external_to = oracle.to_external(&case.to)?;
            let canonical_verdict = canonical.segment_verdict(&case.from, &case.to)?;
            let external_verdict = oracle.segment_verdict(&external_from, &external_to)?;
            if canonical_verdict != external_verdict {
                mismatches.push(format!(
                    "segment {} verdict mismatch: canonical={canonical_verdict:?} external={external_verdict:?}",
                    case.name
                ));
            }
            if !same_bits(&oracle.to_canonical(&external_from)?, &case.from)
                || !same_bits(&oracle.to_canonical(&external_to)?, &case.to)
            {
                mismatches.push(format!("segment {} endpoint roundtrip mismatch", case.name));
            }
        }

        let mut path_reports = Vec::with_capacity(paths.len());
        for case in &paths {
            let external_path = transform_path_to_external(&oracle, &case.path)?;
            let canonical_replay =
                validate_euclidean_path(&canonical_problem, &canonical, &case.path)?;
            let external_replay = validate_euclidean_path(&problem, &oracle, &external_path)?;
            let canonical_signature = replay_signature(&canonical_replay);
            let external_signature = replay_signature(&external_replay);
            let signature_equal = canonical_signature == external_signature;
            if !signature_equal {
                mismatches.push(format!(
                    "path {} replay class/signature mismatch: canonical={} external={}",
                    case.name, canonical_signature, external_signature
                ));
            }

            let inverse_path = transform_path_to_canonical(&oracle, &external_path)?;
            if !same_path_bits(&inverse_path, &case.path) {
                mismatches.push(format!("path {} inverse-transform mismatch", case.name));
            }

            let mut cost_bits_equal = None;
            let mut canonical_path_identity = None;
            let mut external_path_identity = None;
            let mut inverse_replay_identity_equal = None;
            if let (
                ContinuousPathReplay::Valid(canonical_valid),
                ContinuousPathReplay::Valid(external_valid),
            ) = (&canonical_replay, &external_replay)
            {
                let equal =
                    canonical_valid.total_cost().to_bits() == external_valid.total_cost().to_bits();
                cost_bits_equal = Some(equal);
                if !equal {
                    mismatches.push(format!(
                        "path {} total-cost bits mismatch: canonical={:016x} external={:016x}",
                        case.name,
                        canonical_valid.total_cost().to_bits(),
                        external_valid.total_cost().to_bits()
                    ));
                }
                canonical_path_identity = Some(hex32(canonical_valid.path_identity()));
                external_path_identity = Some(hex32(external_valid.path_identity()));

                let inverse_replay = require_valid_replay(validate_euclidean_path(
                    &canonical_problem,
                    &canonical,
                    &inverse_path,
                )?)?;
                let identity_equal =
                    inverse_replay.path_identity() == canonical_valid.path_identity();
                inverse_replay_identity_equal = Some(identity_equal);
                if !identity_equal {
                    mismatches.push(format!(
                        "path {} inverse canonical replay identity mismatch",
                        case.name
                    ));
                }
            }

            path_reports.push(json!({
                "name": case.name,
                "canonical_replay": canonical_signature,
                "external_replay": external_signature,
                "signature_equal": signature_equal,
                "cost_bits_equal": cost_bits_equal,
                "canonical_path_identity": canonical_path_identity,
                "external_path_identity": external_path_identity,
                "inverse_replay_identity_equal": inverse_replay_identity_equal,
            }));
        }

        let transformed_reference = transform_path_to_external(&oracle, &reference_path)?;
        let transformed_reference_replay = require_valid_replay(validate_euclidean_path(
            &problem,
            &oracle,
            &transformed_reference,
        )?)?;
        if transformed_reference_replay.total_cost().to_bits()
            != canonical_reference.total_cost().to_bits()
        {
            mismatches.push("reference path exact total-cost mismatch".to_string());
        }

        let mismatch_count = mismatches.len();
        let permutation_pass = mismatch_count == 0;
        total_mismatches += mismatch_count;
        permutation_reports.push(json!({
            "canonical_to_external": permutation,
            "permutation_identity": hex32(oracle.profile().identity()),
            "problem_identity": hex32(problem.identity()),
            "state_cases": states.len(),
            "segment_cases": segments.len(),
            "path_cases": paths.len(),
            "mismatch_count": mismatch_count,
            "mismatches": mismatches,
            "reference_external_path_identity": hex32(transformed_reference_replay.path_identity()),
            "reference_total_cost_bits": format!("{:016x}", transformed_reference_replay.total_cost().to_bits()),
            "path_results": path_reports,
            "pass": permutation_pass,
        }));
    }

    let pass = total_mismatches == 0 && permutation_reports.len() == 24;
    Ok(json!({
        "pass": pass,
        "status": if pass { "PASS" } else { "FAIL" },
        "permutation_count": permutation_reports.len(),
        "required_permutation_count": 24,
        "total_mismatches": total_mismatches,
        "zero_mismatch_gate": total_mismatches == 0,
        "corpus_identity": hex32(corpus_identity),
        "canonical_oracle_identity": hex32(canonical.profile().identity()),
        "canonical_problem_identity": hex32(canonical_problem.identity()),
        "canonical_reference_path_identity": hex32(canonical_reference.path_identity()),
        "canonical_reference_total_cost_bits": format!("{:016x}", canonical_reference.total_cost().to_bits()),
        "corpus_counts": {
            "states": states.len(),
            "segments": segments.len(),
            "paths": paths.len(),
        },
        "permutations": permutation_reports,
    }))
}

fn check_bounds_roundtrip(
    canonical: &FiniteWShellOracle,
    oracle: &PermutationOracle,
    mismatches: &mut Vec<String>,
) -> Result<(), ContinuousReachabilityError> {
    let canonical_min = oracle.to_canonical(oracle.sampling_min())?;
    let canonical_max = oracle.to_canonical(oracle.sampling_max())?;
    if !same_bits(&canonical_min, canonical.sampling_min()) {
        mismatches.push("sampling_min inverse mapping mismatch".to_string());
    }
    if !same_bits(&canonical_max, canonical.sampling_max()) {
        mismatches.push("sampling_max inverse mapping mismatch".to_string());
    }
    Ok(())
}

fn check_problem_roundtrip(
    canonical_problem: &EuclideanPlanningProblem,
    problem: &EuclideanPlanningProblem,
    oracle: &PermutationOracle,
    mismatches: &mut Vec<String>,
) -> Result<(), ContinuousReachabilityError> {
    if !same_bits(
        &oracle.to_canonical(problem.start())?,
        canonical_problem.start(),
    ) {
        mismatches.push("problem start inverse mapping mismatch".to_string());
    }
    if !same_bits(
        &oracle.to_canonical(problem.goal())?,
        canonical_problem.goal(),
    ) {
        mismatches.push("problem goal inverse mapping mismatch".to_string());
    }
    Ok(())
}

fn frozen_states(problem: &EuclideanPlanningProblem) -> Vec<StateCase> {
    vec![
        StateCase { name: "start", state: problem.start().to_vec() },
        StateCase { name: "goal", state: problem.goal().to_vec() },
        StateCase { name: "inside_inner", state: vec![0.5, 0.0, 0.0, 0.0] },
        StateCase { name: "just_inside_inner_boundary", state: vec![1.0 - BINARY_OFFSET, 0.0, 0.0, 0.0] },
        StateCase { name: "inner_boundary", state: vec![1.0, 0.0, 0.0, 0.0] },
        StateCase { name: "just_inside_shell", state: vec![1.0 + BINARY_OFFSET, 0.0, 0.0, 0.0] },
        StateCase { name: "shell_mid", state: vec![1.5, 0.0, 0.0, 0.0] },
        StateCase { name: "just_before_outer_boundary", state: vec![2.0 - BINARY_OFFSET, 0.0, 0.0, 0.0] },
        StateCase { name: "outer_boundary", state: vec![2.0, 0.0, 0.0, 0.0] },
        StateCase { name: "just_outside_outer_boundary", state: vec![2.0 + BINARY_OFFSET, 0.0, 0.0, 0.0] },
        StateCase { name: "w_zero_shell", state: vec![1.5, 0.0, 0.0, 0.0] },
        StateCase { name: "w_positive_boundary", state: vec![1.5, 0.0, 0.0, W_SHELL] },
        StateCase { name: "w_negative_boundary", state: vec![1.5, 0.0, 0.0, -W_SHELL] },
        StateCase { name: "w_positive_just_beyond", state: vec![1.5, 0.0, 0.0, W_SHELL + BINARY_OFFSET] },
        StateCase { name: "w_negative_just_beyond", state: vec![1.5, 0.0, 0.0, -(W_SHELL + BINARY_OFFSET)] },
        StateCase { name: "domain_boundary", state: vec![4.0, 0.0, 0.0, 0.0] },
        StateCase { name: "outside_domain", state: vec![4.0 + BINARY_OFFSET, 0.0, 0.0, 0.0] },
    ]
}

fn frozen_segments(problem: &EuclideanPlanningProblem) -> Vec<SegmentCase> {
    vec![
        SegmentCase { name: "direct_blocked", from: problem.start().to_vec(), to: problem.goal().to_vec() },
        SegmentCase { name: "inner_boundary_tangent", from: vec![-1.0, 1.0, 0.0, 0.0], to: vec![1.0, 1.0, 0.0, 0.0] },
        SegmentCase { name: "outer_boundary_tangent", from: vec![-1.0, 2.0, 0.0, 0.0], to: vec![1.0, 2.0, 0.0, 0.0] },
        SegmentCase { name: "w_boundary_crossing", from: vec![0.0, 0.0, 0.0, W_SHELL], to: vec![3.0, 0.0, 0.0, W_SHELL] },
        SegmentCase { name: "above_shell_crossing", from: vec![0.0, 0.0, 0.0, 0.5], to: vec![3.0, 0.0, 0.0, 0.5] },
        SegmentCase { name: "below_shell_crossing", from: vec![0.0, 0.0, 0.0, -0.5], to: vec![3.0, 0.0, 0.0, -0.5] },
        SegmentCase { name: "inner_region_valid", from: vec![0.0, 0.0, 0.0, 0.0], to: vec![0.5, 0.0, 0.0, 0.0] },
        SegmentCase { name: "outside_domain_endpoint", from: vec![3.0, 0.0, 0.0, 0.0], to: vec![4.0 + BINARY_OFFSET, 0.0, 0.0, 0.0] },
    ]
}

fn frozen_paths(problem: &EuclideanPlanningProblem, reference_path: &[Vec<f64>]) -> Vec<PathCase> {
    vec![
        PathCase { name: "reference_escape", path: reference_path.to_vec() },
        PathCase { name: "direct_blocked", path: vec![problem.start().to_vec(), problem.goal().to_vec()] },
        PathCase { name: "invalid_middle_state", path: vec![problem.start().to_vec(), vec![1.5, 0.0, 0.0, 0.0], problem.goal().to_vec()] },
        PathCase { name: "w_boundary_path", path: vec![problem.start().to_vec(), vec![0.0, 0.0, 0.0, W_SHELL], vec![3.0, 0.0, 0.0, W_SHELL], problem.goal().to_vec()] },
    ]
}

fn transform_path_to_external(oracle: &PermutationOracle, path: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, ContinuousReachabilityError> {
    path.iter().map(|state| oracle.to_external(state)).collect()
}

fn transform_path_to_canonical(oracle: &PermutationOracle, path: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, ContinuousReachabilityError> {
    path.iter().map(|state| oracle.to_canonical(state)).collect()
}

fn require_valid_replay(replay: ContinuousPathReplay) -> Result<ContinuousPathValidationReceipt, Box<dyn Error>> {
    match replay {
        ContinuousPathReplay::Valid(receipt) => Ok(receipt),
        other => Err(format!("expected valid replay, got {other:?}").into()),
    }
}

fn replay_signature(replay: &ContinuousPathReplay) -> String {
    match replay {
        ContinuousPathReplay::Valid(receipt) => format!(
            "Valid(states={},segments={},cost={:016x})",
            receipt.state_queries(), receipt.segment_queries(), receipt.total_cost().to_bits()
        ),
        ContinuousPathReplay::Invalid { reason, state_index, segment_index } => format!(
            "Invalid(reason={reason:?},state={state_index:?},segment={segment_index:?})"
        ),
        ContinuousPathReplay::Unknown { reason, state_index, segment_index } => format!(
            "Unknown(reason={reason:?},state={state_index:?},segment={segment_index:?})"
        ),
    }
}

fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(left, right)| left.to_bits() == right.to_bits())
}

fn same_path_bits(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(left, right)| same_bits(left, right))
}

fn permutations4() -> Vec<[usize; 4]> {
    let mut out = Vec::with_capacity(24);
    for a in 0..4 {
        for b in 0..4 {
            if b == a { continue; }
            for c in 0..4 {
                if c == a || c == b { continue; }
                for d in 0..4 {
                    if d == a || d == b || d == c { continue; }
                    out.push([a, b, c, d]);
                }
            }
        }
    }
    out
}

fn corpus_identity(canonical: &FiniteWShellOracle, states: &[StateCase], segments: &[SegmentCase], paths: &[PathCase]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-hyperspace-002q-corpus-v1\0");
    hasher.update(&canonical.profile().identity());
    hash_states(&mut hasher, states);
    hash_segments(&mut hasher, segments);
    hash_paths(&mut hasher, paths);
    *hasher.finalize().as_bytes()
}

fn hash_states(hasher: &mut Hasher, states: &[StateCase]) {
    hasher.update(&(states.len() as u64).to_le_bytes());
    for case in states {
        hash_bytes(hasher, case.name.as_bytes());
        hash_vector(hasher, &case.state);
    }
}

fn hash_segments(hasher: &mut Hasher, segments: &[SegmentCase]) {
    hasher.update(&(segments.len() as u64).to_le_bytes());
    for case in segments {
        hash_bytes(hasher, case.name.as_bytes());
        hash_vector(hasher, &case.from);
        hash_vector(hasher, &case.to);
    }
}

fn hash_paths(hasher: &mut Hasher, paths: &[PathCase]) {
    hasher.update(&(paths.len() as u64).to_le_bytes());
    for case in paths {
        hash_bytes(hasher, case.name.as_bytes());
        hasher.update(&(case.path.len() as u64).to_le_bytes());
        for state in &case.path {
            hash_vector(hasher, state);
        }
    }
}

fn hash_vector(hasher: &mut Hasher, values: &[f64]) {
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(&value.to_bits().to_le_bytes());
    }
}

fn hash_bytes(hasher: &mut Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn hex32(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}
