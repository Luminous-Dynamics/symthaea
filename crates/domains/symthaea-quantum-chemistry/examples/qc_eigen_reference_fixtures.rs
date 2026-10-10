// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Emit a small, deterministic checked-eigensolver corpus for independent NumPy comparison.
//!
//! Run from a clean checkout:
//! `cargo run -p symthaea-quantum-chemistry --example qc_eigen_reference_fixtures > /tmp/qc-eigen-native.json`

use serde::Serialize;
use std::error::Error;
use std::process::Command;
use symthaea_quantum_chemistry::scf::generalized_eigen::{
    CheckedSymmetricEigenResult, EigenStoppingReason, symmetric_eigen_checked,
};

const REPORT_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u16,
    producer: &'static str,
    producer_version: &'static str,
    source_revision: String,
    source_tree_sha: String,
    worktree_clean: bool,
    cases: Vec<MatrixCaseRecord>,
}

#[derive(Debug, Serialize)]
struct MatrixCaseRecord {
    case_id: &'static str,
    dimension: usize,
    matrix_row_major: Vec<f64>,
    native_result: NativeEigenResult,
}

#[derive(Debug, Serialize)]
struct NativeEigenResult {
    status: &'static str,
    eigenvalues: Option<Vec<f64>>,
    eigenvectors_row_major: Option<Vec<f64>>,
    converged: bool,
    iterations: Option<usize>,
    stopping_reason: Option<&'static str>,
    max_off_diagonal: Option<f64>,
    max_relative_eigenpair_residual: Option<f64>,
    max_orthogonality_residual: Option<f64>,
    failure_reason: Option<String>,
}

fn git_output(args: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new("git").args(args).output()?;
    if !output.status.success() {
        return Err(format!(
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn source_provenance() -> Result<(String, String), Box<dyn Error>> {
    let status = git_output(&["status", "--porcelain", "--untracked-files=all"])?;
    if !status.is_empty() {
        return Err("refusing to emit eigen fixtures from a dirty Git worktree".into());
    }
    let revision = git_output(&["rev-parse", "--verify", "HEAD^{commit}"])?;
    let tree = git_output(&["rev-parse", "HEAD^{tree}"])?;
    for (kind, sha) in [("commit", &revision), ("tree", &tree)] {
        if sha.len() != 40 || !sha.chars().all(|ch| ch.is_ascii_hexdigit()) {
            return Err(format!("Git returned an invalid {kind} SHA").into());
        }
    }
    Ok((revision, tree))
}

fn record_case(
    case_id: &'static str,
    dimension: usize,
    matrix_row_major: Vec<f64>,
) -> MatrixCaseRecord {
    let max_iterations = 100_usize.saturating_mul(dimension).saturating_mul(dimension);
    let result = symmetric_eigen_checked(&matrix_row_major, dimension, max_iterations);
    MatrixCaseRecord {
        case_id,
        dimension,
        matrix_row_major,
        native_result: match result {
            Ok(result) => native_result(result),
            Err(error) => NativeEigenResult {
                status: "invalid_input",
                eigenvalues: None,
                eigenvectors_row_major: None,
                converged: false,
                iterations: None,
                stopping_reason: None,
                max_off_diagonal: None,
                max_relative_eigenpair_residual: None,
                max_orthogonality_residual: None,
                failure_reason: Some(error.to_string()),
            },
        },
    }
}

fn native_result(result: CheckedSymmetricEigenResult) -> NativeEigenResult {
    let status = if result.converged {
        "converged"
    } else {
        "non_converged"
    };
    let stopping_reason = match result.stopping_reason {
        EigenStoppingReason::Converged => {
            "converged"
        }
        EigenStoppingReason::IterationLimit => {
            "iteration_limit"
        }
        EigenStoppingReason::ResidualContractFailed => {
            "residual_contract_failed"
        }
    };
    NativeEigenResult {
        status,
        eigenvalues: Some(result.eigenvalues),
        eigenvectors_row_major: Some(result.eigenvectors),
        converged: result.converged,
        iterations: Some(result.iterations),
        stopping_reason: Some(stopping_reason),
        max_off_diagonal: Some(result.max_off_diagonal),
        max_relative_eigenpair_residual: Some(result.max_relative_eigenpair_residual),
        max_orthogonality_residual: Some(result.max_orthogonality_residual),
        failure_reason: None,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let (source_revision, source_tree_sha) = source_provenance()?;

    // Small, frozen symmetric matrices span scalar, diagonal, degenerate,
    // dense, negative off-diagonal, near-dependent, and very small-scale cases.
    let cases = vec![
        record_case("scalar_3_25", 1, vec![3.25]),
        record_case("diagonal_3_1", 2, vec![3.0, 0.0, 0.0, 1.0]),
        record_case(
            "degenerate_identity_scaled",
            3,
            vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0],
        ),
        record_case("dense_symmetric_3x3", 3, vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 5.0]),
        record_case("negative_off_diagonal", 2, vec![1.0, -0.75, -0.75, 1.0]),
        record_case("near_linear_dependence", 2, vec![1.0, 0.9999999, 0.9999999, 1.0]),
        record_case("small_scale_symmetric", 2, vec![1e-16, 0.5e-16, 0.5e-16, 1e-16]),
        record_case("subnormal_scale_symmetric", 2, vec![1e-310, 0.5e-310, 0.5e-310, 1e-310]),
    ];

    let (ending_revision, ending_tree_sha) = source_provenance()?;
    if ending_revision != source_revision || ending_tree_sha != source_tree_sha {
        return Err("Git HEAD/tree changed while eigen fixtures were running".into());
    }

    let report = Report {
        schema_version: REPORT_SCHEMA_VERSION,
        producer: "symthaea-quantum-chemistry",
        producer_version: env!("CARGO_PKG_VERSION"),
        source_revision,
        source_tree_sha,
        worktree_clean: true,
        cases,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
