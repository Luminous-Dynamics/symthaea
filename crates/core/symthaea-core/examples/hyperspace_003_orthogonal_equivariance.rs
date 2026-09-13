// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! HYPERSPACE-003 preregistered orthogonal basis-mixing evaluator control.
//!
//! This deterministic experiment advances beyond coordinate relabeling. It applies
//! six frozen orthogonal basis changes that genuinely mix canonical `w` with xyz,
//! while keeping the canonical finite-w shell oracle as the sole validity authority.
//! The result is a synthetic numerical/evaluator theorem only.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use blake3::Hasher;
use serde_json::{Value, json};
use symthaea_core::continuous_reachability::{
    ContinuousPathReplay, ContinuousReachabilityError, ContinuousValidityOracle,
    ContinuousValidityOracleProfile, EuclideanPlanningProblem, OracleVerdict,
    validate_euclidean_path,
};
use symthaea_core::hyperspace_benchmark::{
    FiniteWShellOracle, HyperspaceDimension, canonical_shell_problem,
    evaluator_reference_escape_path, qualify_projection_trap,
};
use symthaea_core::state_space::{EuclideanSpace, StateSpace};

const SCHEMA: &str = "symthaea.hyperspace-003.orthogonal-equivariance.v1";
const PREREG_ISSUE: u64 = 2417;
const W_SHELL: f64 = 0.25;
const C: f64 = 0.6; // 3/5, identity-bound by f64 bits.
const S: f64 = 0.8; // 4/5, identity-bound by f64 bits.
const ORTHOGONALITY_TOLERANCE: f64 = 3.552_713_678_800_501e-15; // 2^-48
const ROUNDTRIP_TOLERANCE: f64 = 9.094_947_017_729_282e-13; // 2^-40
const LENGTH_TOLERANCE: f64 = 9.094_947_017_729_282e-13; // 2^-40
const CLEARANCE_OFFSET: f64 = 9.536_743_164_062_5e-7; // 2^-20

type Mat4 = [[f64; 4]; 4];

fn main() -> Result<(), Box<dyn Error>> {
    let (subject, h002q_head, output) = parse_args()?;
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
        "hyperspace_002q_head": h002q_head,
        "claim_scope": "synthetic Euclidean evaluator invariance under six preregistered orthogonal basis changes; not arbitrary O(4), planner robustness, curved-manifold competence, or physical extra-dimension evidence",
        "verdict": if pass { "PASS" } else { "FAIL" },
        "control": control,
    });

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "HYPERSPACE-003 orthogonal-equivariance verdict: {}",
        report["verdict"].as_str().unwrap_or("FAIL")
    );
    println!("receipt: {}", output.display());
    Ok(())
}

fn parse_args() -> Result<(String, String, PathBuf), Box<dyn Error>> {
    let mut subject = None;
    let mut h002q = None;
    let mut output = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--subject" => subject = args.next(),
            "--hyperspace-002q-head" => h002q = args.next(),
            "--output" => output = args.next().map(PathBuf::from),
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok((
        validate_sha(subject.ok_or("--subject is required")?, "--subject")?,
        validate_sha(
            h002q.ok_or("--hyperspace-002q-head is required")?,
            "--hyperspace-002q-head",
        )?,
        output.ok_or("--output is required")?,
    ))
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

#[derive(Clone, Debug)]
struct TransformSpec {
    name: &'static str,
    matrix: Mat4,
}

#[derive(Clone, Debug)]
struct OrthogonalMixingOracle {
    canonical: FiniteWShellOracle,
    matrix: Mat4,
    inverse: Mat4,
    profile: ContinuousValidityOracleProfile,
}

impl OrthogonalMixingOracle {
    fn new(canonical: FiniteWShellOracle, matrix: Mat4) -> Result<Self, Box<dyn Error>> {
        for row in matrix {
            if row.iter().any(|value| !value.is_finite()) {
                return Err("orthogonal transform matrix must be finite".into());
            }
        }
        let inverse = transpose(matrix);
        let state_space = EuclideanSpace::new(4);
        let mut parameters = Vec::new();
        parameters.extend_from_slice(&canonical.profile().identity());
        parameters.extend_from_slice(&numerical_policy_identity());
        for row in matrix {
            for value in row {
                parameters.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        let profile = ContinuousValidityOracleProfile::new(
            state_space.profile().identity(),
            "hyperspace-orthogonal-mixing-v1",
            parameters,
        )?;
        Ok(Self {
            canonical,
            matrix,
            inverse,
            profile,
        })
    }

    fn to_external(&self, canonical: &[f64]) -> Result<Vec<f64>, ContinuousReachabilityError> {
        require_four_finite(canonical, "canonical")?;
        mat_vec(&self.matrix, canonical)
    }

    fn to_canonical(&self, external: &[f64]) -> Result<Vec<f64>, ContinuousReachabilityError> {
        require_four_finite(external, "external")?;
        mat_vec(&self.inverse, external)
    }
}

impl ContinuousValidityOracle for OrthogonalMixingOracle {
    fn profile(&self) -> &ContinuousValidityOracleProfile {
        &self.profile
    }

    fn state_verdict(&self, state: &[f64]) -> Result<OracleVerdict, ContinuousReachabilityError> {
        self.canonical.state_verdict(&self.to_canonical(state)?)
    }

    fn segment_verdict(
        &self,
        from: &[f64],
        to: &[f64],
    ) -> Result<OracleVerdict, ContinuousReachabilityError> {
        self.canonical
            .segment_verdict(&self.to_canonical(from)?, &self.to_canonical(to)?)
    }
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

    let states = frozen_states(&canonical_problem);
    let segments = frozen_segments(&canonical_problem);
    let paths = frozen_paths(&canonical_problem, &reference_path);
    let corners = domain_corners();
    let policy_identity = numerical_policy_identity();
    let corpus_identity = corpus_identity(
        &canonical,
        &policy_identity,
        &states,
        &segments,
        &paths,
        &corners,
    );
    let transforms = frozen_transforms();
    if transforms.len() != 6 {
        return Err(format!("expected 6 transforms, got {}", transforms.len()).into());
    }

    let mut reports = Vec::with_capacity(transforms.len());
    let mut total_mismatches = 0_usize;
    let mut global_max_orthogonality_error = 0.0_f64;
    let mut global_max_roundtrip_error = 0.0_f64;
    let mut global_max_length_error = 0.0_f64;

    for transform in transforms {
        let oracle = OrthogonalMixingOracle::new(canonical.clone(), transform.matrix)?;
        let problem = EuclideanPlanningProblem::new(
            4,
            oracle.to_external(canonical_problem.start())?,
            oracle.to_external(canonical_problem.goal())?,
            &oracle,
        )?;
        let mut mismatches = Vec::<String>::new();
        let orthogonality_error = orthogonality_error(&transform.matrix);
        global_max_orthogonality_error =
            global_max_orthogonality_error.max(orthogonality_error);
        if orthogonality_error > ORTHOGONALITY_TOLERANCE {
            mismatches.push(format!(
                "orthogonality error {orthogonality_error:e} exceeds {ORTHOGONALITY_TOLERANCE:e}"
            ));
        }
        if !materially_mixes_w(&transform.matrix) {
            mismatches.push("transform does not materially mix canonical w with xyz".to_string());
        }

        let start_error = roundtrip_error(&oracle, canonical_problem.start())?;
        let goal_error = roundtrip_error(&oracle, canonical_problem.goal())?;
        let mut transform_max_roundtrip = start_error.max(goal_error);
        if start_error > ROUNDTRIP_TOLERANCE {
            mismatches.push(format!("start roundtrip error {start_error:e}"));
        }
        if goal_error > ROUNDTRIP_TOLERANCE {
            mismatches.push(format!("goal roundtrip error {goal_error:e}"));
        }

        for (index, corner) in corners.iter().enumerate() {
            let error = roundtrip_error(&oracle, corner)?;
            transform_max_roundtrip = transform_max_roundtrip.max(error);
            if error > ROUNDTRIP_TOLERANCE {
                mismatches.push(format!("domain corner {index} roundtrip error {error:e}"));
            }
        }

        let mut state_reports = Vec::with_capacity(states.len());
        for case in &states {
            let external = oracle.to_external(&case.state)?;
            let inverse = oracle.to_canonical(&external)?;
            let error = max_abs_diff(&inverse, &case.state)?;
            transform_max_roundtrip = transform_max_roundtrip.max(error);
            let canonical_verdict = canonical.state_verdict(&case.state)?;
            let external_verdict = oracle.state_verdict(&external)?;
            let verdict_equal = canonical_verdict == external_verdict;
            if error > ROUNDTRIP_TOLERANCE {
                mismatches.push(format!("state {} roundtrip error {error:e}", case.name));
            }
            if !verdict_equal {
                mismatches.push(format!(
                    "state {} verdict mismatch: canonical={canonical_verdict:?} external={external_verdict:?}",
                    case.name
                ));
            }
            state_reports.push(json!({
                "name": case.name,
                "roundtrip_error": error,
                "canonical_verdict": verdict_name(&canonical_verdict),
                "external_verdict": verdict_name(&external_verdict),
                "verdict_equal": verdict_equal,
            }));
        }

        let mut segment_reports = Vec::with_capacity(segments.len());
        for case in &segments {
            let external_from = oracle.to_external(&case.from)?;
            let external_to = oracle.to_external(&case.to)?;
            let from_inverse = oracle.to_canonical(&external_from)?;
            let to_inverse = oracle.to_canonical(&external_to)?;
            let endpoint_error = max_abs_diff(&from_inverse, &case.from)?
                .max(max_abs_diff(&to_inverse, &case.to)?);
            transform_max_roundtrip = transform_max_roundtrip.max(endpoint_error);
            let canonical_verdict = canonical.segment_verdict(&case.from, &case.to)?;
            let external_verdict = oracle.segment_verdict(&external_from, &external_to)?;
            let verdict_equal = canonical_verdict == external_verdict;
            if endpoint_error > ROUNDTRIP_TOLERANCE {
                mismatches.push(format!(
                    "segment {} endpoint roundtrip error {endpoint_error:e}",
                    case.name
                ));
            }
            if !verdict_equal {
                mismatches.push(format!(
                    "segment {} verdict mismatch: canonical={canonical_verdict:?} external={external_verdict:?}",
                    case.name
                ));
            }
            segment_reports.push(json!({
                "name": case.name,
                "endpoint_roundtrip_error": endpoint_error,
                "canonical_verdict": verdict_name(&canonical_verdict),
                "external_verdict": verdict_name(&external_verdict),
                "verdict_equal": verdict_equal,
            }));
        }

        let mut path_reports = Vec::with_capacity(paths.len());
        let mut transform_max_length_error = 0.0_f64;
        for case in &paths {
            let external_path = transform_path_to_external(&oracle, &case.path)?;
            let inverse_path = transform_path_to_canonical(&oracle, &external_path)?;
            let waypoint_error = path_max_abs_diff(&inverse_path, &case.path)?;
            transform_max_roundtrip = transform_max_roundtrip.max(waypoint_error);
            if waypoint_error > ROUNDTRIP_TOLERANCE {
                mismatches.push(format!("path {} waypoint roundtrip error {waypoint_error:e}", case.name));
            }

            let canonical_replay =
                validate_euclidean_path(&canonical_problem, &canonical, &case.path)?;
            let external_replay = validate_euclidean_path(&problem, &oracle, &external_path)?;
            let canonical_class = replay_class(&canonical_replay);
            let external_class = replay_class(&external_replay);
            let class_equal = canonical_class == external_class;
            if !class_equal {
                mismatches.push(format!(
                    "path {} replay class mismatch: canonical={canonical_class} external={external_class}",
                    case.name
                ));
            }

            let mut length_error = None;
            let mut length_limit = None;
            if let (
                ContinuousPathReplay::Valid(canonical_valid),
                ContinuousPathReplay::Valid(external_valid),
            ) = (&canonical_replay, &external_replay)
            {
                let error = (canonical_valid.total_cost() - external_valid.total_cost()).abs();
                let limit = LENGTH_TOLERANCE * canonical_valid.total_cost().abs().max(1.0);
                transform_max_length_error = transform_max_length_error.max(error);
                if error > limit {
                    mismatches.push(format!(
                        "path {} length error {error:e} exceeds {limit:e}",
                        case.name
                    ));
                }
                length_error = Some(error);
                length_limit = Some(limit);
            }

            path_reports.push(json!({
                "name": case.name,
                "canonical_replay": canonical_class,
                "external_replay": external_class,
                "class_equal": class_equal,
                "waypoint_roundtrip_error": waypoint_error,
                "length_error": length_error,
                "length_limit": length_limit,
            }));
        }

        let transformed_reference = transform_path_to_external(&oracle, &reference_path)?;
        let transformed_reference_replay =
            validate_euclidean_path(&problem, &oracle, &transformed_reference)?;
        if replay_class(&transformed_reference_replay) != "Valid" {
            mismatches.push("transformed evaluator reference path is not valid".to_string());
        }
        let inverse_reference = transform_path_to_canonical(&oracle, &transformed_reference)?;
        let reference_roundtrip_error = path_max_abs_diff(&inverse_reference, &reference_path)?;
        transform_max_roundtrip = transform_max_roundtrip.max(reference_roundtrip_error);
        if reference_roundtrip_error > ROUNDTRIP_TOLERANCE {
            mismatches.push(format!(
                "reference path roundtrip error {reference_roundtrip_error:e}"
            ));
        }
        let h6 = qualify_projection_trap(&canonical_problem, &canonical, &inverse_reference);
        let h6_established = h6.is_ok();
        if !h6_established {
            mismatches.push("inverse-transformed evaluator reference did not re-establish H6".to_string());
        }
        let h6_identity = h6.ok().map(|receipt| hex32(receipt.identity()));

        global_max_roundtrip_error = global_max_roundtrip_error.max(transform_max_roundtrip);
        global_max_length_error = global_max_length_error.max(transform_max_length_error);
        let mismatch_count = mismatches.len();
        total_mismatches += mismatch_count;
        reports.push(json!({
            "name": transform.name,
            "transform_identity": hex32(oracle.profile().identity()),
            "matrix_bits": matrix_bits(&transform.matrix),
            "orthogonality_error": orthogonality_error,
            "materially_mixes_w": materially_mixes_w(&transform.matrix),
            "max_roundtrip_error": transform_max_roundtrip,
            "max_length_error": transform_max_length_error,
            "reference_h6_established": h6_established,
            "reference_h6_identity": h6_identity,
            "state_results": state_reports,
            "segment_results": segment_reports,
            "path_results": path_reports,
            "mismatch_count": mismatch_count,
            "mismatches": mismatches,
            "pass": mismatch_count == 0,
        }));
    }

    let pass = total_mismatches == 0 && reports.len() == 6;
    Ok(json!({
        "pass": pass,
        "status": if pass { "PASS" } else { "FAIL" },
        "transform_count": reports.len(),
        "required_transform_count": 6,
        "total_mismatches": total_mismatches,
        "zero_mismatch_gate": total_mismatches == 0,
        "canonical_oracle_identity": hex32(canonical.profile().identity()),
        "canonical_problem_identity": hex32(canonical_problem.identity()),
        "numerical_policy_identity": hex32(policy_identity),
        "corpus_identity": hex32(corpus_identity),
        "corpus_counts": {
            "states": states.len(),
            "segments": segments.len(),
            "paths": paths.len(),
            "domain_corners": corners.len(),
        },
        "numerical_policy": {
            "orthogonality_tolerance": ORTHOGONALITY_TOLERANCE,
            "roundtrip_tolerance": ROUNDTRIP_TOLERANCE,
            "length_tolerance_factor": LENGTH_TOLERANCE,
            "clearance_offset": CLEARANCE_OFFSET,
        },
        "observed_maxima": {
            "orthogonality_error": global_max_orthogonality_error,
            "roundtrip_error": global_max_roundtrip_error,
            "length_error": global_max_length_error,
        },
        "transforms": reports,
    }))
}

fn frozen_transforms() -> Vec<TransformSpec> {
    let rxw = givens(0, 3);
    let ryw = givens(1, 3);
    let rzw = givens(2, 3);
    vec![
        TransformSpec { name: "R_xw", matrix: rxw },
        TransformSpec { name: "R_yw", matrix: ryw },
        TransformSpec { name: "R_zw", matrix: rzw },
        TransformSpec { name: "R_yw_R_xw", matrix: mat_mul(&ryw, &rxw) },
        TransformSpec { name: "R_zw_R_xw", matrix: mat_mul(&rzw, &rxw) },
        TransformSpec {
            name: "R_zw_R_yw_R_xw",
            matrix: mat_mul(&rzw, &mat_mul(&ryw, &rxw)),
        },
    ]
}

fn frozen_states(problem: &EuclideanPlanningProblem) -> Vec<StateCase> {
    vec![
        StateCase { name: "start", state: problem.start().to_vec() },
        StateCase { name: "goal", state: problem.goal().to_vec() },
        StateCase { name: "inner", state: vec![0.5, 0.0, 0.0, 0.0] },
        StateCase { name: "outer", state: vec![2.5, 0.0, 0.0, 0.0] },
        StateCase { name: "obstacle_mid", state: vec![1.5, 0.0, 0.0, 0.0] },
        StateCase { name: "above_shell", state: vec![1.5, 0.0, 0.0, 0.5] },
        StateCase { name: "below_shell", state: vec![1.5, 0.0, 0.0, -0.5] },
        StateCase { name: "inner_minus", state: vec![1.0 - CLEARANCE_OFFSET, 0.0, 0.0, 0.0] },
        StateCase { name: "inner_plus", state: vec![1.0 + CLEARANCE_OFFSET, 0.0, 0.0, 0.0] },
        StateCase { name: "outer_minus", state: vec![2.0 - CLEARANCE_OFFSET, 0.0, 0.0, 0.0] },
        StateCase { name: "outer_plus", state: vec![2.0 + CLEARANCE_OFFSET, 0.0, 0.0, 0.0] },
        StateCase { name: "w_pos_inside", state: vec![1.5, 0.0, 0.0, W_SHELL - CLEARANCE_OFFSET] },
        StateCase { name: "w_pos_outside", state: vec![1.5, 0.0, 0.0, W_SHELL + CLEARANCE_OFFSET] },
        StateCase { name: "w_neg_inside", state: vec![1.5, 0.0, 0.0, -W_SHELL + CLEARANCE_OFFSET] },
        StateCase { name: "w_neg_outside", state: vec![1.5, 0.0, 0.0, -W_SHELL - CLEARANCE_OFFSET] },
        StateCase { name: "domain_inside", state: vec![4.0 - CLEARANCE_OFFSET, 0.0, 0.0, 0.0] },
        StateCase { name: "domain_outside", state: vec![4.0 + CLEARANCE_OFFSET, 0.0, 0.0, 0.0] },
        StateCase { name: "mixed_free", state: vec![1.2, 0.4, 0.3, 0.5] },
    ]
}

fn frozen_segments(problem: &EuclideanPlanningProblem) -> Vec<SegmentCase> {
    vec![
        SegmentCase { name: "direct_blocked", from: problem.start().to_vec(), to: problem.goal().to_vec() },
        SegmentCase { name: "above_crossing", from: vec![0.0, 0.0, 0.0, 0.5], to: vec![3.0, 0.0, 0.0, 0.5] },
        SegmentCase { name: "below_crossing", from: vec![0.0, 0.0, 0.0, -0.5], to: vec![3.0, 0.0, 0.0, -0.5] },
        SegmentCase { name: "obstacle_interior", from: vec![1.25, 0.0, 0.0, 0.0], to: vec![1.75, 0.0, 0.0, 0.0] },
        SegmentCase { name: "outer_clearance", from: vec![-3.0, 2.0 + CLEARANCE_OFFSET, 0.0, 0.0], to: vec![3.0, 2.0 + CLEARANCE_OFFSET, 0.0, 0.0] },
        SegmentCase { name: "inner_region", from: vec![-0.5, 0.0, 0.0, 0.0], to: vec![0.5, 0.0, 0.0, 0.0] },
        SegmentCase { name: "w_just_outside", from: vec![0.0, 0.0, 0.0, W_SHELL + CLEARANCE_OFFSET], to: vec![3.0, 0.0, 0.0, W_SHELL + CLEARANCE_OFFSET] },
        SegmentCase { name: "w_just_inside", from: vec![0.0, 0.0, 0.0, W_SHELL - CLEARANCE_OFFSET], to: vec![3.0, 0.0, 0.0, W_SHELL - CLEARANCE_OFFSET] },
    ]
}

fn frozen_paths(
    problem: &EuclideanPlanningProblem,
    reference_path: &[Vec<f64>],
) -> Vec<PathCase> {
    vec![
        PathCase { name: "reference_escape", path: reference_path.to_vec() },
        PathCase { name: "direct_blocked", path: vec![problem.start().to_vec(), problem.goal().to_vec()] },
        PathCase { name: "invalid_middle", path: vec![problem.start().to_vec(), vec![1.5, 0.0, 0.0, 0.0], problem.goal().to_vec()] },
        PathCase { name: "below_escape", path: vec![problem.start().to_vec(), vec![0.0, 0.0, 0.0, -0.5], vec![3.0, 0.0, 0.0, -0.5], problem.goal().to_vec()] },
    ]
}

fn domain_corners() -> Vec<Vec<f64>> {
    let mut corners = Vec::with_capacity(16);
    for mask in 0_u8..16 {
        corners.push(vec![
            if mask & 1 == 0 { -4.0 } else { 4.0 },
            if mask & 2 == 0 { -4.0 } else { 4.0 },
            if mask & 4 == 0 { -4.0 } else { 4.0 },
            if mask & 8 == 0 { -2.0 } else { 2.0 },
        ]);
    }
    corners
}

fn givens(a: usize, b: usize) -> Mat4 {
    let mut matrix = identity();
    matrix[a][a] = C;
    matrix[a][b] = -S;
    matrix[b][a] = S;
    matrix[b][b] = C;
    matrix
}

fn identity() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn transpose(matrix: Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for (row, values) in out.iter_mut().enumerate() {
        for (column, value) in values.iter_mut().enumerate() {
            *value = matrix[column][row];
        }
    }
    out
}

fn mat_mul(left: &Mat4, right: &Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            out[row][column] = (0..4).map(|k| left[row][k] * right[k][column]).sum();
        }
    }
    out
}

fn mat_vec(matrix: &Mat4, vector: &[f64]) -> Result<Vec<f64>, ContinuousReachabilityError> {
    require_four_finite(vector, "transform input")?;
    let mut out = vec![0.0; 4];
    for row in 0..4 {
        out[row] = (0..4).map(|column| matrix[row][column] * vector[column]).sum();
        if !out[row].is_finite() {
            return Err(ContinuousReachabilityError::OracleEvaluation {
                reason: "orthogonal transform produced non-finite coordinate".to_string(),
            });
        }
        out[row] += 0.0;
    }
    Ok(out)
}

fn orthogonality_error(matrix: &Mat4) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..4 {
        for j in 0..4 {
            let dot: f64 = (0..4).map(|k| matrix[k][i] * matrix[k][j]).sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((dot - expected).abs());
        }
    }
    worst
}

fn materially_mixes_w(matrix: &Mat4) -> bool {
    (0..4).any(|external| {
        matrix[external][3] != 0.0 && (0..3).any(|spatial| matrix[external][spatial] != 0.0)
    })
}

fn roundtrip_error(
    oracle: &OrthogonalMixingOracle,
    canonical: &[f64],
) -> Result<f64, ContinuousReachabilityError> {
    let external = oracle.to_external(canonical)?;
    let inverse = oracle.to_canonical(&external)?;
    max_abs_diff(&inverse, canonical)
}

fn max_abs_diff(left: &[f64], right: &[f64]) -> Result<f64, ContinuousReachabilityError> {
    if left.len() != right.len() {
        return Err(ContinuousReachabilityError::OracleEvaluation {
            reason: "numeric comparison dimension mismatch".to_string(),
        });
    }
    let mut worst = 0.0_f64;
    for (&a, &b) in left.iter().zip(right) {
        let error = (a - b).abs();
        if !error.is_finite() {
            return Err(ContinuousReachabilityError::OracleEvaluation {
                reason: "numeric comparison became non-finite".to_string(),
            });
        }
        worst = worst.max(error);
    }
    Ok(worst)
}

fn path_max_abs_diff(
    left: &[Vec<f64>],
    right: &[Vec<f64>],
) -> Result<f64, ContinuousReachabilityError> {
    if left.len() != right.len() {
        return Err(ContinuousReachabilityError::OracleEvaluation {
            reason: "path comparison waypoint-count mismatch".to_string(),
        });
    }
    let mut worst = 0.0_f64;
    for (a, b) in left.iter().zip(right) {
        worst = worst.max(max_abs_diff(a, b)?);
    }
    Ok(worst)
}

fn transform_path_to_external(
    oracle: &OrthogonalMixingOracle,
    path: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, ContinuousReachabilityError> {
    path.iter().map(|state| oracle.to_external(state)).collect()
}

fn transform_path_to_canonical(
    oracle: &OrthogonalMixingOracle,
    path: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, ContinuousReachabilityError> {
    path.iter().map(|state| oracle.to_canonical(state)).collect()
}

fn replay_class(replay: &ContinuousPathReplay) -> &'static str {
    match replay {
        ContinuousPathReplay::Valid(_) => "Valid",
        ContinuousPathReplay::Invalid { .. } => "Invalid",
        ContinuousPathReplay::Unknown { .. } => "Unknown",
    }
}

fn verdict_name(verdict: &OracleVerdict) -> &'static str {
    match verdict {
        OracleVerdict::Valid => "Valid",
        OracleVerdict::Invalid { .. } => "Invalid",
        OracleVerdict::Unknown { .. } => "Unknown",
    }
}

fn require_four_finite(state: &[f64], role: &str) -> Result<(), ContinuousReachabilityError> {
    if state.len() != 4 || state.iter().any(|value| !value.is_finite()) {
        return Err(ContinuousReachabilityError::OracleEvaluation {
            reason: format!("{role} must contain exactly four finite coordinates"),
        });
    }
    Ok(())
}

fn numerical_policy_identity() -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-hyperspace-003-numerical-policy-v1\0");
    for value in [
        ORTHOGONALITY_TOLERANCE,
        ROUNDTRIP_TOLERANCE,
        LENGTH_TOLERANCE,
        CLEARANCE_OFFSET,
        C,
        S,
    ] {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn corpus_identity(
    canonical: &FiniteWShellOracle,
    policy_identity: &[u8; 32],
    states: &[StateCase],
    segments: &[SegmentCase],
    paths: &[PathCase],
    corners: &[Vec<f64>],
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-hyperspace-003-corpus-v1\0");
    hasher.update(&canonical.profile().identity());
    hasher.update(policy_identity);
    hasher.update(&(states.len() as u64).to_le_bytes());
    for case in states {
        hash_bytes(&mut hasher, case.name.as_bytes());
        hash_vector(&mut hasher, &case.state);
    }
    hasher.update(&(segments.len() as u64).to_le_bytes());
    for case in segments {
        hash_bytes(&mut hasher, case.name.as_bytes());
        hash_vector(&mut hasher, &case.from);
        hash_vector(&mut hasher, &case.to);
    }
    hasher.update(&(paths.len() as u64).to_le_bytes());
    for case in paths {
        hash_bytes(&mut hasher, case.name.as_bytes());
        hasher.update(&(case.path.len() as u64).to_le_bytes());
        for state in &case.path {
            hash_vector(&mut hasher, state);
        }
    }
    hasher.update(&(corners.len() as u64).to_le_bytes());
    for corner in corners {
        hash_vector(&mut hasher, corner);
    }
    *hasher.finalize().as_bytes()
}

fn hash_bytes(hasher: &mut Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn hash_vector(hasher: &mut Hasher, vector: &[f64]) {
    hasher.update(&(vector.len() as u64).to_le_bytes());
    for value in vector {
        hasher.update(&value.to_bits().to_le_bytes());
    }
}

fn matrix_bits(matrix: &Mat4) -> Vec<Vec<String>> {
    matrix
        .iter()
        .map(|row| row.iter().map(|value| format!("{:016x}", value.to_bits())).collect())
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
