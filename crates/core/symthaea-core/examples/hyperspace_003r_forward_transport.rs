// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! HYPERSPACE-003R exact forward representation-transport repair theorem.
//!
//! HYPERSPACE-003 remains an executed FAIL. This experiment does not approximate
//! canonical identity through an inverse rotation. Instead it starts from the exact
//! canonical H6-qualified witness, transports it forward exactly once into each
//! frozen rotated representation, and independently validates that exact external
//! path against an external problem built from the same forward-transformed endpoints.

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

const SCHEMA: &str = "symthaea.hyperspace-003r.forward-transport.v1";
const PREREG_ISSUE: u64 = 2427;
const QUALIFIED_H002Q_HEAD: &str = "9b7bf7c22b673aabdd735f8c15f4620a3e7fb2f0";
const FAILED_H003_SUBJECT: &str = "962c3b72350251713c54b7b6328960a2a6f35595";
const FAILED_H003_ARTIFACT_DIGEST: &str =
    "sha256:03673207c9c0a12691b0708c4f92fbb7096fc79eaeee9b981767a776680a6851";
const W_SHELL: f64 = 0.25;
const C: f64 = 0.6; // Exact f64 used by H003 for 3/5.
const S: f64 = 0.8; // Exact f64 used by H003 for 4/5.
const ORTHOGONALITY_TOLERANCE: f64 = 3.552_713_678_800_501e-15; // 2^-48
const ROUNDTRIP_TOLERANCE: f64 = 9.094_947_017_729_282e-13; // 2^-40
const LENGTH_TOLERANCE: f64 = 9.094_947_017_729_282e-13; // 2^-40
const CLEARANCE_OFFSET: f64 = 9.536_743_164_062_5e-7; // 2^-20, H003 policy binding.

type Mat4 = [[f64; 4]; 4];

fn main() -> Result<(), Box<dyn Error>> {
    let (subject, output) = parse_args()?;
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
        "qualified_hyperspace_002q_head": QUALIFIED_H002Q_HEAD,
        "failed_hyperspace_003_subject": FAILED_H003_SUBJECT,
        "failed_hyperspace_003_artifact_digest": FAILED_H003_ARTIFACT_DIGEST,
        "historical_hyperspace_003_verdict": "FAIL",
        "can_rewrite_hyperspace_003": false,
        "claim_scope": "exact forward transport of the canonical H6-qualified witness into six frozen orthogonal representations; inverse reconstruction is diagnostic only",
        "verdict": if pass { "PASS" } else { "FAIL" },
        "control": control,
    });

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "HYPERSPACE-003R forward-transport verdict: {}",
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

#[derive(Clone, Debug)]
struct TransformSpec {
    name: &'static str,
    matrix: Mat4,
}

#[derive(Clone, Debug)]
struct RotatedOracle {
    canonical: FiniteWShellOracle,
    matrix: Mat4,
    inverse: Mat4,
    profile: ContinuousValidityOracleProfile,
}

impl RotatedOracle {
    fn new(canonical: FiniteWShellOracle, matrix: Mat4) -> Result<Self, Box<dyn Error>> {
        if matrix.iter().flatten().any(|value| !value.is_finite()) {
            return Err("rotation matrix must be finite".into());
        }
        let inverse = transpose(matrix);
        let state_space = EuclideanSpace::new(4);
        let mut parameters = Vec::new();
        parameters.extend_from_slice(&canonical.profile().identity());
        parameters.extend_from_slice(&h003_numerical_policy_identity());
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
        mat_vec(&self.matrix, canonical)
    }

    fn to_canonical(&self, external: &[f64]) -> Result<Vec<f64>, ContinuousReachabilityError> {
        mat_vec(&self.inverse, external)
    }
}

impl ContinuousValidityOracle for RotatedOracle {
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
    let canonical_path =
        evaluator_reference_escape_path(&canonical_problem, &canonical, 0.25)?;
    let canonical_validation = require_valid(validate_euclidean_path(
        &canonical_problem,
        &canonical,
        &canonical_path,
    )?)?;
    let canonical_h6 = qualify_projection_trap(&canonical_problem, &canonical, &canonical_path)?;
    let canonical_path_identity = canonical_validation.path_identity();
    let canonical_h6_identity = canonical_h6.identity();
    let policy_identity = h003_numerical_policy_identity();

    let transforms = frozen_transforms();
    if transforms.len() != 6 {
        return Err(format!("expected six frozen transforms, got {}", transforms.len()).into());
    }

    let mut reports = Vec::with_capacity(6);
    let mut failures = 0_usize;
    let mut negative_controls_rejected = 0_usize;
    let mut observed_max_roundtrip = 0.0_f64;
    let mut observed_max_length_error = 0.0_f64;

    for transform in transforms {
        let oracle = RotatedOracle::new(canonical.clone(), transform.matrix)?;
        let external_start = oracle.to_external(canonical_problem.start())?;
        let external_goal = oracle.to_external(canonical_problem.goal())?;
        let external_problem = EuclideanPlanningProblem::new(
            4,
            external_start.clone(),
            external_goal.clone(),
            &oracle,
        )?;
        let external_path = transform_path_forward(&oracle, &canonical_path)?;
        let fresh_forward = transform_path_forward(&oracle, &canonical_path)?;
        let exact_forward_equal = same_path_bits(&external_path, &fresh_forward);
        let exact_endpoints = same_bits(&external_path[0], external_problem.start())
            && same_bits(
                external_path.last().expect("reference path has an end"),
                external_problem.goal(),
            );

        let external_validation = require_valid(validate_euclidean_path(
            &external_problem,
            &oracle,
            &external_path,
        )?)?;

        let inverse_path = transform_path_inverse(&oracle, &external_path)?;
        let inverse_roundtrip_error = path_max_abs_diff(&inverse_path, &canonical_path)?;
        observed_max_roundtrip = observed_max_roundtrip.max(inverse_roundtrip_error);

        let length_error =
            (external_validation.total_cost() - canonical_validation.total_cost()).abs();
        let length_limit = LENGTH_TOLERANCE * canonical_validation.total_cost().abs().max(1.0);
        observed_max_length_error = observed_max_length_error.max(length_error);

        let transport_identity = hash_transport_receipt(
            &canonical,
            &canonical_problem,
            canonical_path_identity,
            canonical_h6_identity,
            &external_problem,
            external_validation.path_identity(),
            oracle.profile().identity(),
            &external_path,
            policy_identity,
        );

        let mut tampered = external_path.clone();
        tampered[1][0] = next_up(tampered[1][0])?;
        let negative_rejected = !verify_exact_forward_transport(
            &oracle,
            &canonical_path,
            &tampered,
        )?;
        if negative_rejected {
            negative_controls_rejected += 1;
        }
        let tampered_replay = validate_euclidean_path(&external_problem, &oracle, &tampered)?;

        let transform_pass = exact_forward_equal
            && exact_endpoints
            && inverse_roundtrip_error <= ROUNDTRIP_TOLERANCE
            && length_error <= length_limit
            && negative_rejected;
        if !transform_pass {
            failures += 1;
        }

        reports.push(json!({
            "name": transform.name,
            "transform_identity": hex32(oracle.profile().identity()),
            "matrix_bits": matrix_bits(&transform.matrix),
            "external_problem_identity": hex32(external_problem.identity()),
            "external_path_identity": hex32(external_validation.path_identity()),
            "transport_identity": hex32(transport_identity),
            "exact_forward_equal": exact_forward_equal,
            "exact_external_endpoints": exact_endpoints,
            "inverse_roundtrip_error": inverse_roundtrip_error,
            "roundtrip_limit": ROUNDTRIP_TOLERANCE,
            "length_error": length_error,
            "length_limit": length_limit,
            "negative_control_rejected": negative_rejected,
            "tampered_replay_class": replay_class(&tampered_replay),
            "pass": transform_pass,
        }));
    }

    let pass = failures == 0 && negative_controls_rejected == 6 && reports.len() == 6;
    Ok(json!({
        "pass": pass,
        "status": if pass { "PASS" } else { "FAIL" },
        "transform_count": reports.len(),
        "required_transform_count": 6,
        "failed_transform_count": failures,
        "negative_controls_rejected": negative_controls_rejected,
        "required_negative_controls_rejected": 6,
        "canonical_oracle_identity": hex32(canonical.profile().identity()),
        "canonical_problem_identity": hex32(canonical_problem.identity()),
        "canonical_reference_path_identity": hex32(canonical_path_identity),
        "canonical_h6_receipt_identity": hex32(canonical_h6_identity),
        "h003_numerical_policy_identity": hex32(policy_identity),
        "failed_h003_subject": FAILED_H003_SUBJECT,
        "failed_h003_artifact_digest": FAILED_H003_ARTIFACT_DIGEST,
        "historical_h003_verdict": "FAIL",
        "inverse_is_authority_for_canonical_identity": false,
        "observed_max_inverse_roundtrip_error": observed_max_roundtrip,
        "observed_max_length_error": observed_max_length_error,
        "transforms": reports,
    }))
}

fn verify_exact_forward_transport(
    oracle: &RotatedOracle,
    canonical_path: &[Vec<f64>],
    candidate_external: &[Vec<f64>],
) -> Result<bool, ContinuousReachabilityError> {
    let expected = transform_path_forward(oracle, canonical_path)?;
    Ok(same_path_bits(&expected, candidate_external))
}

fn hash_transport_receipt(
    canonical: &FiniteWShellOracle,
    canonical_problem: &EuclideanPlanningProblem,
    canonical_path_identity: [u8; 32],
    canonical_h6_identity: [u8; 32],
    external_problem: &EuclideanPlanningProblem,
    external_path_identity: [u8; 32],
    transform_identity: [u8; 32],
    external_path: &[Vec<f64>],
    policy_identity: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-hyperspace-003r-forward-transport-receipt-v1\0");
    hasher.update(&canonical.profile().identity());
    hasher.update(&canonical_problem.identity());
    hasher.update(&canonical_path_identity);
    hasher.update(&canonical_h6_identity);
    hasher.update(&external_problem.identity());
    hasher.update(&external_path_identity);
    hasher.update(&transform_identity);
    hasher.update(&policy_identity);
    hasher.update(QUALIFIED_H002Q_HEAD.as_bytes());
    hasher.update(FAILED_H003_SUBJECT.as_bytes());
    hasher.update(FAILED_H003_ARTIFACT_DIGEST.as_bytes());
    hash_path_bits(&mut hasher, external_path);
    *hasher.finalize().as_bytes()
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
    for row in 0..4 {
        for column in 0..4 {
            out[row][column] = matrix[column][row];
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
    if vector.len() != 4 || vector.iter().any(|value| !value.is_finite()) {
        return Err(ContinuousReachabilityError::OracleEvaluation {
            reason: "orthogonal transport requires exactly four finite coordinates".to_string(),
        });
    }
    let mut out = vec![0.0; 4];
    for row in 0..4 {
        out[row] = (0..4).map(|column| matrix[row][column] * vector[column]).sum();
        if !out[row].is_finite() {
            return Err(ContinuousReachabilityError::OracleEvaluation {
                reason: "orthogonal transport produced non-finite coordinate".to_string(),
            });
        }
        out[row] += 0.0;
    }
    Ok(out)
}

fn transform_path_forward(
    oracle: &RotatedOracle,
    path: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, ContinuousReachabilityError> {
    path.iter().map(|state| oracle.to_external(state)).collect()
}

fn transform_path_inverse(
    oracle: &RotatedOracle,
    path: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, ContinuousReachabilityError> {
    path.iter().map(|state| oracle.to_canonical(state)).collect()
}

fn require_valid(
    replay: ContinuousPathReplay,
) -> Result<symthaea_core::continuous_reachability::ContinuousPathValidationReceipt, Box<dyn Error>> {
    match replay {
        ContinuousPathReplay::Valid(receipt) => Ok(receipt),
        other => Err(format!("expected independently valid path, got {other:?}").into()),
    }
}

fn replay_class(replay: &ContinuousPathReplay) -> &'static str {
    match replay {
        ContinuousPathReplay::Valid(_) => "Valid",
        ContinuousPathReplay::Invalid { .. } => "Invalid",
        ContinuousPathReplay::Unknown { .. } => "Unknown",
    }
}

fn path_max_abs_diff(
    left: &[Vec<f64>],
    right: &[Vec<f64>],
) -> Result<f64, ContinuousReachabilityError> {
    if left.len() != right.len() {
        return Err(ContinuousReachabilityError::OracleEvaluation {
            reason: "path diagnostic waypoint-count mismatch".to_string(),
        });
    }
    let mut worst = 0.0_f64;
    for (a, b) in left.iter().zip(right) {
        if a.len() != b.len() {
            return Err(ContinuousReachabilityError::OracleEvaluation {
                reason: "path diagnostic dimension mismatch".to_string(),
            });
        }
        for (&x, &y) in a.iter().zip(b) {
            let error = (x - y).abs();
            if !error.is_finite() {
                return Err(ContinuousReachabilityError::OracleEvaluation {
                    reason: "path diagnostic became non-finite".to_string(),
                });
            }
            worst = worst.max(error);
        }
    }
    Ok(worst)
}

fn same_path_bits(left: &[Vec<f64>], right: &[Vec<f64>]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(a, b)| same_bits(a, b))
}

fn same_bits(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

fn next_up(value: f64) -> Result<f64, Box<dyn Error>> {
    if !value.is_finite() {
        return Err("cannot mutate a non-finite value".into());
    }
    if value == 0.0 {
        return Ok(f64::from_bits(1));
    }
    let bits = value.to_bits();
    Ok(if value > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    })
}

fn h003_numerical_policy_identity() -> [u8; 32] {
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

fn hash_path_bits(hasher: &mut Hasher, path: &[Vec<f64>]) {
    hasher.update(&(path.len() as u64).to_le_bytes());
    for state in path {
        hasher.update(&(state.len() as u64).to_le_bytes());
        for value in state {
            hasher.update(&value.to_bits().to_le_bytes());
        }
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
