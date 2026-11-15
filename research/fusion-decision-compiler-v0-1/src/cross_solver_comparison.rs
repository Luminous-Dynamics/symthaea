//! Matched-case, covariance-aware solver output comparison for FECOMP.
//!
//! This module composes source-pinned solver outputs, observable-scoped
//! dependency manifests and externally computed joint covariance receipts.
//! It does not implement solver physics, units, normalization or covariance
//! arithmetic, and it does not establish physical truth.

use super::evidence_dependence::{
    DependencyValidationError, ObservableDependenceAssessment, RunDependencyManifest,
    compare_observable_dependencies,
};

pub const CROSS_SOLVER_COMPARISON_VERSION: &str = "fusion-cross-solver-comparison/0.1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrozenComparisonCase {
    pub case_id: String,
    pub candidate_id: String,
    pub candidate_revision: String,
    pub geometry_sha256: String,
    pub service_boundary_ref: String,
    pub observable_id: String,
    pub canonical_quantity_ref: String,
    pub unit_ref: String,
    pub frame_ref: String,
    pub sign_convention_ref: String,
    pub time_basis_ref: String,
    pub normalization_contract_ref: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplicabilityState {
    InDeclaredDomain,
    Unknown,
    OutOfDomain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceState {
    Converged,
    NonConverged,
    NotReported,
    NotExecuted,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolverObservableOutput {
    pub run_id: String,
    pub solver_id: String,
    pub solver_revision: String,
    pub case_id: String,
    pub candidate_id: String,
    pub candidate_revision: String,
    pub geometry_sha256: String,
    pub service_boundary_ref: String,
    pub observable_id: String,
    pub canonical_quantity_ref: String,
    pub unit_ref: String,
    pub frame_ref: String,
    pub sign_convention_ref: String,
    pub time_basis_ref: String,
    pub normalization_contract_ref: String,
    /// Numeric value after the exact declared normalization contract.
    pub normalized_value: f64,
    /// Identity of the solver-output payload; this module does not hash bytes.
    pub output_sha256: String,
    /// Reference to the source-pinned solver run receipt.
    pub run_receipt_ref: String,
    /// Reference to the observable-specific output evidence.
    pub output_evidence_ref: String,
    pub applicability: ApplicabilityState,
    pub convergence: ConvergenceState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossCovarianceState {
    Known,
    Unknown,
    NotAssessed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossCovarianceEvidence {
    pub state: CrossCovarianceState,
    pub receipt_ref: Option<String>,
    /// Ordered pair identity: first run then second run.
    pub first_run_id: Option<String>,
    pub second_run_id: Option<String>,
    pub observable_id: Option<String>,
    pub first_output_sha256: Option<String>,
    pub second_output_sha256: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossCovarianceDisposition {
    BoundToExactOutputPair,
    Unknown,
    NotAssessed,
    StaleOrMismatched,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossSolverComparisonDisposition {
    NotComparable,
    ComparableDescriptiveOnly,
    ComparableWithJointCovarianceReceipt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonDiagnostic {
    DifferentCase,
    DifferentCandidate,
    DifferentCandidateRevision,
    DifferentGeometry,
    DifferentServiceBoundary,
    DifferentObservable,
    DifferentCanonicalQuantity,
    DifferentUnit,
    DifferentFrame,
    DifferentSignConvention,
    DifferentTimeBasis,
    DifferentNormalizationContract,
    FirstRunApplicabilityUnknown,
    FirstRunOutOfDomain,
    SecondRunApplicabilityUnknown,
    SecondRunOutOfDomain,
    FirstRunNotConverged,
    FirstRunConvergenceNotReported,
    FirstRunNotExecuted,
    SecondRunNotConverged,
    SecondRunConvergenceNotReported,
    SecondRunNotExecuted,
    NonFiniteDifference,
    CrossCovarianceReceiptMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComparisonValidationError {
    EmptyField(&'static str),
    NonFiniteNumber(&'static str),
    InvalidSha256(&'static str),
    DuplicateRunIdentity(String),
    DependencyManifestRunMismatch {
        output_run_id: String,
        manifest_run_id: String,
    },
    InvalidDependencyManifests(Vec<DependencyValidationError>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CrossSolverComparison {
    pub contract_version: &'static str,
    pub case_id: String,
    pub observable_id: String,
    pub first_run_id: String,
    pub second_run_id: String,
    pub first_solver_id: String,
    pub second_solver_id: String,
    pub first_output_sha256: String,
    pub second_output_sha256: String,
    pub disposition: CrossSolverComparisonDisposition,
    /// Defined as second normalized output minus first normalized output.
    pub second_minus_first: Option<f64>,
    pub absolute_difference: Option<f64>,
    pub cross_covariance: CrossCovarianceDisposition,
    pub covariance_receipt_ref: Option<String>,
    pub dependence: ObservableDependenceAssessment,
    pub diagnostics: Vec<ComparisonDiagnostic>,
    /// This module never promotes a software comparison to physical validation.
    pub claim_ceiling: &'static str,
}

/// Compare two outputs against one frozen, canonical comparison case.
///
/// A raw difference is reported only after exact case, observable, quantity,
/// unit, frame, sign, time and normalization identity match. If those match
/// but cross-covariance is not bound to these exact two output artifacts, the
/// disposition remains descriptive-only. Even a bound covariance receipt only
/// enables downstream uncertainty review; it does not assert significance.
pub fn compare_solver_outputs(
    case: &FrozenComparisonCase,
    first: &SolverObservableOutput,
    second: &SolverObservableOutput,
    first_dependencies: &RunDependencyManifest,
    second_dependencies: &RunDependencyManifest,
    covariance: &CrossCovarianceEvidence,
) -> Result<CrossSolverComparison, Vec<ComparisonValidationError>> {
    let mut errors = validate_case(case);
    errors.extend(validate_output(first, "first"));
    errors.extend(validate_output(second, "second"));

    if first.run_id == second.run_id {
        errors.push(ComparisonValidationError::DuplicateRunIdentity(
            first.run_id.clone(),
        ));
    }
    if first_dependencies.run_id != first.run_id {
        errors.push(ComparisonValidationError::DependencyManifestRunMismatch {
            output_run_id: first.run_id.clone(),
            manifest_run_id: first_dependencies.run_id.clone(),
        });
    }
    if second_dependencies.run_id != second.run_id {
        errors.push(ComparisonValidationError::DependencyManifestRunMismatch {
            output_run_id: second.run_id.clone(),
            manifest_run_id: second_dependencies.run_id.clone(),
        });
    }

    let dependence = compare_observable_dependencies(
        first_dependencies,
        second_dependencies,
        std::slice::from_ref(&case.observable_id),
    )
    .map_err(|dependency_errors| {
        vec![ComparisonValidationError::InvalidDependencyManifests(
            dependency_errors,
        )]
    })?
    .into_iter()
    .next()
    .expect("one requested observable produces one dependence assessment");

    if !errors.is_empty() {
        return Err(errors);
    }

    let mut diagnostics = Vec::new();
    compare_case_identity(case, first, &mut diagnostics);
    compare_case_identity(case, second, &mut diagnostics);
    compare_pair_identity(first, second, &mut diagnostics);
    evaluate_run_status(first, true, &mut diagnostics);
    evaluate_run_status(second, false, &mut diagnostics);

    let pair_is_comparable = diagnostics.is_empty();
    let difference = if pair_is_comparable {
        let value = second.normalized_value - first.normalized_value;
        if value.is_finite() {
            Some(value)
        } else {
            diagnostics.push(ComparisonDiagnostic::NonFiniteDifference);
            None
        }
    } else {
        None
    };

    let covariance_disposition = evaluate_covariance_binding(case, first, second, covariance);
    if covariance_disposition == CrossCovarianceDisposition::StaleOrMismatched {
        diagnostics.push(ComparisonDiagnostic::CrossCovarianceReceiptMismatch);
    }

    let disposition = if difference.is_none() {
        CrossSolverComparisonDisposition::NotComparable
    } else if covariance_disposition == CrossCovarianceDisposition::BoundToExactOutputPair {
        CrossSolverComparisonDisposition::ComparableWithJointCovarianceReceipt
    } else {
        CrossSolverComparisonDisposition::ComparableDescriptiveOnly
    };

    let covariance_receipt_ref =
        if covariance_disposition == CrossCovarianceDisposition::BoundToExactOutputPair {
            covariance.receipt_ref.clone()
        } else {
            None
        };

    Ok(CrossSolverComparison {
        contract_version: CROSS_SOLVER_COMPARISON_VERSION,
        case_id: case.case_id.clone(),
        observable_id: case.observable_id.clone(),
        first_run_id: first.run_id.clone(),
        second_run_id: second.run_id.clone(),
        first_solver_id: first.solver_id.clone(),
        second_solver_id: second.solver_id.clone(),
        first_output_sha256: first.output_sha256.clone(),
        second_output_sha256: second.output_sha256.clone(),
        disposition,
        second_minus_first: difference,
        absolute_difference: difference.map(f64::abs),
        cross_covariance: covariance_disposition,
        covariance_receipt_ref,
        dependence,
        diagnostics,
        claim_ceiling: "matched normalized solver-output comparison only",
    })
}

fn validate_case(case: &FrozenComparisonCase) -> Vec<ComparisonValidationError> {
    let mut errors = Vec::new();
    for (value, field) in [
        (case.case_id.as_str(), "case.case_id"),
        (case.candidate_id.as_str(), "case.candidate_id"),
        (case.candidate_revision.as_str(), "case.candidate_revision"),
        (
            case.service_boundary_ref.as_str(),
            "case.service_boundary_ref",
        ),
        (case.observable_id.as_str(), "case.observable_id"),
        (
            case.canonical_quantity_ref.as_str(),
            "case.canonical_quantity_ref",
        ),
        (case.unit_ref.as_str(), "case.unit_ref"),
        (case.frame_ref.as_str(), "case.frame_ref"),
        (
            case.sign_convention_ref.as_str(),
            "case.sign_convention_ref",
        ),
        (case.time_basis_ref.as_str(), "case.time_basis_ref"),
        (
            case.normalization_contract_ref.as_str(),
            "case.normalization_contract_ref",
        ),
    ] {
        if value.trim().is_empty() {
            errors.push(ComparisonValidationError::EmptyField(field));
        }
    }
    if !is_sha256(&case.geometry_sha256) {
        errors.push(ComparisonValidationError::InvalidSha256(
            "case.geometry_sha256",
        ));
    }
    errors
}

fn validate_output(
    output: &SolverObservableOutput,
    prefix: &'static str,
) -> Vec<ComparisonValidationError> {
    let mut errors = Vec::new();
    for (value, field) in [
        (output.run_id.as_str(), "run_id"),
        (output.solver_id.as_str(), "solver_id"),
        (output.solver_revision.as_str(), "solver_revision"),
        (output.case_id.as_str(), "case_id"),
        (output.candidate_id.as_str(), "candidate_id"),
        (output.candidate_revision.as_str(), "candidate_revision"),
        (output.service_boundary_ref.as_str(), "service_boundary_ref"),
        (output.observable_id.as_str(), "observable_id"),
        (
            output.canonical_quantity_ref.as_str(),
            "canonical_quantity_ref",
        ),
        (output.unit_ref.as_str(), "unit_ref"),
        (output.frame_ref.as_str(), "frame_ref"),
        (output.sign_convention_ref.as_str(), "sign_convention_ref"),
        (output.time_basis_ref.as_str(), "time_basis_ref"),
        (
            output.normalization_contract_ref.as_str(),
            "normalization_contract_ref",
        ),
        (output.run_receipt_ref.as_str(), "run_receipt_ref"),
        (output.output_evidence_ref.as_str(), "output_evidence_ref"),
    ] {
        if value.trim().is_empty() {
            errors.push(ComparisonValidationError::EmptyField(match prefix {
                "first" => match field {
                    "run_id" => "first.run_id",
                    "solver_id" => "first.solver_id",
                    "solver_revision" => "first.solver_revision",
                    "case_id" => "first.case_id",
                    "candidate_id" => "first.candidate_id",
                    "candidate_revision" => "first.candidate_revision",
                    "service_boundary_ref" => "first.service_boundary_ref",
                    "observable_id" => "first.observable_id",
                    "canonical_quantity_ref" => "first.canonical_quantity_ref",
                    "unit_ref" => "first.unit_ref",
                    "frame_ref" => "first.frame_ref",
                    "sign_convention_ref" => "first.sign_convention_ref",
                    "time_basis_ref" => "first.time_basis_ref",
                    "normalization_contract_ref" => "first.normalization_contract_ref",
                    "run_receipt_ref" => "first.run_receipt_ref",
                    _ => "first.output_evidence_ref",
                },
                _ => match field {
                    "run_id" => "second.run_id",
                    "solver_id" => "second.solver_id",
                    "solver_revision" => "second.solver_revision",
                    "case_id" => "second.case_id",
                    "candidate_id" => "second.candidate_id",
                    "candidate_revision" => "second.candidate_revision",
                    "service_boundary_ref" => "second.service_boundary_ref",
                    "observable_id" => "second.observable_id",
                    "canonical_quantity_ref" => "second.canonical_quantity_ref",
                    "unit_ref" => "second.unit_ref",
                    "frame_ref" => "second.frame_ref",
                    "sign_convention_ref" => "second.sign_convention_ref",
                    "time_basis_ref" => "second.time_basis_ref",
                    "normalization_contract_ref" => "second.normalization_contract_ref",
                    "run_receipt_ref" => "second.run_receipt_ref",
                    _ => "second.output_evidence_ref",
                },
            }));
        }
    }
    if !is_sha256(&output.geometry_sha256) {
        errors.push(ComparisonValidationError::InvalidSha256(match prefix {
            "first" => "first.geometry_sha256",
            _ => "second.geometry_sha256",
        }));
    }
    if !is_sha256(&output.output_sha256) {
        errors.push(ComparisonValidationError::InvalidSha256(match prefix {
            "first" => "first.output_sha256",
            _ => "second.output_sha256",
        }));
    }
    if !output.normalized_value.is_finite() {
        errors.push(ComparisonValidationError::NonFiniteNumber(match prefix {
            "first" => "first.normalized_value",
            _ => "second.normalized_value",
        }));
    }
    errors
}

fn compare_case_identity(
    case: &FrozenComparisonCase,
    output: &SolverObservableOutput,
    diagnostics: &mut Vec<ComparisonDiagnostic>,
) {
    if output.case_id != case.case_id {
        push_once(diagnostics, ComparisonDiagnostic::DifferentCase);
    }
    if output.candidate_id != case.candidate_id {
        push_once(diagnostics, ComparisonDiagnostic::DifferentCandidate);
    }
    if output.candidate_revision != case.candidate_revision {
        push_once(
            diagnostics,
            ComparisonDiagnostic::DifferentCandidateRevision,
        );
    }
    if output.geometry_sha256 != case.geometry_sha256 {
        push_once(diagnostics, ComparisonDiagnostic::DifferentGeometry);
    }
    if output.service_boundary_ref != case.service_boundary_ref {
        push_once(diagnostics, ComparisonDiagnostic::DifferentServiceBoundary);
    }
    if output.observable_id != case.observable_id {
        push_once(diagnostics, ComparisonDiagnostic::DifferentObservable);
    }
    if output.canonical_quantity_ref != case.canonical_quantity_ref {
        push_once(
            diagnostics,
            ComparisonDiagnostic::DifferentCanonicalQuantity,
        );
    }
    if output.unit_ref != case.unit_ref {
        push_once(diagnostics, ComparisonDiagnostic::DifferentUnit);
    }
    if output.frame_ref != case.frame_ref {
        push_once(diagnostics, ComparisonDiagnostic::DifferentFrame);
    }
    if output.sign_convention_ref != case.sign_convention_ref {
        push_once(diagnostics, ComparisonDiagnostic::DifferentSignConvention);
    }
    if output.time_basis_ref != case.time_basis_ref {
        push_once(diagnostics, ComparisonDiagnostic::DifferentTimeBasis);
    }
    if output.normalization_contract_ref != case.normalization_contract_ref {
        push_once(
            diagnostics,
            ComparisonDiagnostic::DifferentNormalizationContract,
        );
    }
}

fn compare_pair_identity(
    first: &SolverObservableOutput,
    second: &SolverObservableOutput,
    diagnostics: &mut Vec<ComparisonDiagnostic>,
) {
    if first.case_id != second.case_id {
        push_once(diagnostics, ComparisonDiagnostic::DifferentCase);
    }
    if first.candidate_id != second.candidate_id {
        push_once(diagnostics, ComparisonDiagnostic::DifferentCandidate);
    }
    if first.candidate_revision != second.candidate_revision {
        push_once(
            diagnostics,
            ComparisonDiagnostic::DifferentCandidateRevision,
        );
    }
    if first.geometry_sha256 != second.geometry_sha256 {
        push_once(diagnostics, ComparisonDiagnostic::DifferentGeometry);
    }
    if first.service_boundary_ref != second.service_boundary_ref {
        push_once(diagnostics, ComparisonDiagnostic::DifferentServiceBoundary);
    }
    if first.observable_id != second.observable_id {
        push_once(diagnostics, ComparisonDiagnostic::DifferentObservable);
    }
    if first.canonical_quantity_ref != second.canonical_quantity_ref {
        push_once(
            diagnostics,
            ComparisonDiagnostic::DifferentCanonicalQuantity,
        );
    }
    if first.unit_ref != second.unit_ref {
        push_once(diagnostics, ComparisonDiagnostic::DifferentUnit);
    }
    if first.frame_ref != second.frame_ref {
        push_once(diagnostics, ComparisonDiagnostic::DifferentFrame);
    }
    if first.sign_convention_ref != second.sign_convention_ref {
        push_once(diagnostics, ComparisonDiagnostic::DifferentSignConvention);
    }
    if first.time_basis_ref != second.time_basis_ref {
        push_once(diagnostics, ComparisonDiagnostic::DifferentTimeBasis);
    }
    if first.normalization_contract_ref != second.normalization_contract_ref {
        push_once(
            diagnostics,
            ComparisonDiagnostic::DifferentNormalizationContract,
        );
    }
}

fn evaluate_run_status(
    output: &SolverObservableOutput,
    first: bool,
    diagnostics: &mut Vec<ComparisonDiagnostic>,
) {
    match output.applicability {
        ApplicabilityState::InDeclaredDomain => {}
        ApplicabilityState::Unknown => push_once(
            diagnostics,
            if first {
                ComparisonDiagnostic::FirstRunApplicabilityUnknown
            } else {
                ComparisonDiagnostic::SecondRunApplicabilityUnknown
            },
        ),
        ApplicabilityState::OutOfDomain => push_once(
            diagnostics,
            if first {
                ComparisonDiagnostic::FirstRunOutOfDomain
            } else {
                ComparisonDiagnostic::SecondRunOutOfDomain
            },
        ),
    }

    match output.convergence {
        ConvergenceState::Converged => {}
        ConvergenceState::NonConverged => push_once(
            diagnostics,
            if first {
                ComparisonDiagnostic::FirstRunNotConverged
            } else {
                ComparisonDiagnostic::SecondRunNotConverged
            },
        ),
        ConvergenceState::NotReported => push_once(
            diagnostics,
            if first {
                ComparisonDiagnostic::FirstRunConvergenceNotReported
            } else {
                ComparisonDiagnostic::SecondRunConvergenceNotReported
            },
        ),
        ConvergenceState::NotExecuted => push_once(
            diagnostics,
            if first {
                ComparisonDiagnostic::FirstRunNotExecuted
            } else {
                ComparisonDiagnostic::SecondRunNotExecuted
            },
        ),
    }
}

fn evaluate_covariance_binding(
    case: &FrozenComparisonCase,
    first: &SolverObservableOutput,
    second: &SolverObservableOutput,
    evidence: &CrossCovarianceEvidence,
) -> CrossCovarianceDisposition {
    match evidence.state {
        CrossCovarianceState::Unknown => CrossCovarianceDisposition::Unknown,
        CrossCovarianceState::NotAssessed => CrossCovarianceDisposition::NotAssessed,
        CrossCovarianceState::Known => {
            let matches = evidence
                .receipt_ref
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
                && evidence.first_run_id.as_deref() == Some(first.run_id.as_str())
                && evidence.second_run_id.as_deref() == Some(second.run_id.as_str())
                && evidence.observable_id.as_deref() == Some(case.observable_id.as_str())
                && evidence.first_output_sha256.as_deref() == Some(first.output_sha256.as_str())
                && evidence.second_output_sha256.as_deref() == Some(second.output_sha256.as_str());
            if matches {
                CrossCovarianceDisposition::BoundToExactOutputPair
            } else {
                CrossCovarianceDisposition::StaleOrMismatched
            }
        }
    }
}

fn push_once(diagnostics: &mut Vec<ComparisonDiagnostic>, diagnostic: ComparisonDiagnostic) {
    if !diagnostics.contains(&diagnostic) {
        diagnostics.push(diagnostic);
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_dependence::{DependencyIdentity, DependencyKind, ObservableDependency};

    fn digest(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn case() -> FrozenComparisonCase {
        FrozenComparisonCase {
            case_id: "iter-reference-case-v1".to_string(),
            candidate_id: "candidate-iter-reference".to_string(),
            candidate_revision: "design-r1".to_string(),
            geometry_sha256: digest('a'),
            service_boundary_ref: "net-electric-export-boundary-v1".to_string(),
            observable_id: "net-electric-export-mw".to_string(),
            canonical_quantity_ref: "quantity-power-net-electric".to_string(),
            unit_ref: "unit-megawatt-electric".to_string(),
            frame_ref: "frame-grid-interface".to_string(),
            sign_convention_ref: "grid-export-positive-v1".to_string(),
            time_basis_ref: "steady-state-window-300s-v1".to_string(),
            normalization_contract_ref: "plant-output-normalization-v1".to_string(),
        }
    }

    fn output(run_id: &str, value: f64, output_digest: char) -> SolverObservableOutput {
        let frozen = case();
        SolverObservableOutput {
            run_id: run_id.to_string(),
            solver_id: if run_id == "run-process" {
                "PROCESS".to_string()
            } else {
                "D0FUS".to_string()
            },
            solver_revision: "pinned-commit".to_string(),
            case_id: frozen.case_id,
            candidate_id: frozen.candidate_id,
            candidate_revision: frozen.candidate_revision,
            geometry_sha256: frozen.geometry_sha256,
            service_boundary_ref: frozen.service_boundary_ref,
            observable_id: frozen.observable_id,
            canonical_quantity_ref: frozen.canonical_quantity_ref,
            unit_ref: frozen.unit_ref,
            frame_ref: frozen.frame_ref,
            sign_convention_ref: frozen.sign_convention_ref,
            time_basis_ref: frozen.time_basis_ref,
            normalization_contract_ref: frozen.normalization_contract_ref,
            normalized_value: value,
            output_sha256: digest(output_digest),
            run_receipt_ref: format!("run-receipt-{run_id}"),
            output_evidence_ref: format!("output-evidence-{run_id}"),
            applicability: ApplicabilityState::InDeclaredDomain,
            convergence: ConvergenceState::Converged,
        }
    }

    fn dependencies(run_id: &str, source: &str, complete: bool) -> RunDependencyManifest {
        RunDependencyManifest {
            run_id: run_id.to_string(),
            output_observables: vec!["net-electric-export-mw".to_string()],
            complete_observables: if complete {
                vec!["net-electric-export-mw".to_string()]
            } else {
                Vec::new()
            },
            dependencies: vec![ObservableDependency {
                observable_id: "net-electric-export-mw".to_string(),
                dependency: DependencyIdentity {
                    node_id: source.to_string(),
                    revision: "source-r1".to_string(),
                    kind: DependencyKind::InputDataset,
                    content_sha256: None,
                    source_evidence_ref: Some(format!("source-receipt-{source}")),
                },
            }],
        }
    }

    fn unknown_covariance() -> CrossCovarianceEvidence {
        CrossCovarianceEvidence {
            state: CrossCovarianceState::Unknown,
            receipt_ref: None,
            first_run_id: None,
            second_run_id: None,
            observable_id: None,
            first_output_sha256: None,
            second_output_sha256: None,
        }
    }

    fn known_covariance(
        first: &SolverObservableOutput,
        second: &SolverObservableOutput,
    ) -> CrossCovarianceEvidence {
        CrossCovarianceEvidence {
            state: CrossCovarianceState::Known,
            receipt_ref: Some("symtropy-joint-covariance-receipt-v1".to_string()),
            first_run_id: Some(first.run_id.clone()),
            second_run_id: Some(second.run_id.clone()),
            observable_id: Some(first.observable_id.clone()),
            first_output_sha256: Some(first.output_sha256.clone()),
            second_output_sha256: Some(second.output_sha256.clone()),
        }
    }

    fn compare(
        first: &SolverObservableOutput,
        second: &SolverObservableOutput,
        covariance: &CrossCovarianceEvidence,
        first_manifest: &RunDependencyManifest,
        second_manifest: &RunDependencyManifest,
    ) -> CrossSolverComparison {
        compare_solver_outputs(
            &case(),
            first,
            second,
            first_manifest,
            second_manifest,
            covariance,
        )
        .unwrap()
    }

    #[test]
    fn matched_outputs_without_cross_covariance_are_descriptive_only() {
        let first = output("run-process", 1200.0, 'b');
        let second = output("run-d0fus", 1185.0, 'c');
        let result = compare(
            &first,
            &second,
            &unknown_covariance(),
            &dependencies("run-process", "dataset-a", true),
            &dependencies("run-d0fus", "dataset-b", true),
        );
        assert_eq!(
            result.disposition,
            CrossSolverComparisonDisposition::ComparableDescriptiveOnly
        );
        assert_eq!(result.second_minus_first, Some(-15.0));
        assert_eq!(result.absolute_difference, Some(15.0));
        assert_eq!(result.cross_covariance, CrossCovarianceDisposition::Unknown);
        assert_eq!(
            result.dependence.state,
            super::super::evidence_dependence::ObservableDependenceState::NoSharedDependencyInCompleteDeclarations
        );
    }

    #[test]
    fn exact_joint_covariance_receipt_only_enables_downstream_review() {
        let first = output("run-process", 1200.0, 'b');
        let second = output("run-d0fus", 1185.0, 'c');
        let result = compare(
            &first,
            &second,
            &known_covariance(&first, &second),
            &dependencies("run-process", "dataset-a", true),
            &dependencies("run-d0fus", "dataset-b", true),
        );
        assert_eq!(
            result.disposition,
            CrossSolverComparisonDisposition::ComparableWithJointCovarianceReceipt
        );
        assert_eq!(
            result.cross_covariance,
            CrossCovarianceDisposition::BoundToExactOutputPair
        );
        assert_eq!(
            result.covariance_receipt_ref.as_deref(),
            Some("symtropy-joint-covariance-receipt-v1")
        );
    }

    #[test]
    fn shared_calibration_is_exposed_even_when_outputs_are_comparable() {
        let first = output("run-process", 1200.0, 'b');
        let second = output("run-d0fus", 1185.0, 'c');
        let result = compare(
            &first,
            &second,
            &unknown_covariance(),
            &dependencies("run-process", "shared-calibration", true),
            &dependencies("run-d0fus", "shared-calibration", true),
        );
        assert_eq!(
            result.dependence.state,
            super::super::evidence_dependence::ObservableDependenceState::SharedDependencyFound
        );
        assert_eq!(
            result.disposition,
            CrossSolverComparisonDisposition::ComparableDescriptiveOnly
        );
    }

    #[test]
    fn missing_dependency_inventory_is_not_independence() {
        let first = output("run-process", 1200.0, 'b');
        let second = output("run-d0fus", 1185.0, 'c');
        let result = compare(
            &first,
            &second,
            &unknown_covariance(),
            &dependencies("run-process", "dataset-a", false),
            &dependencies("run-d0fus", "dataset-b", true),
        );
        assert_eq!(
            result.dependence.state,
            super::super::evidence_dependence::ObservableDependenceState::UnknownDependencyInventory
        );
    }

    #[test]
    fn different_geometry_is_not_comparable() {
        let first = output("run-process", 1200.0, 'b');
        let mut second = output("run-d0fus", 1185.0, 'c');
        second.geometry_sha256 = digest('d');
        let result = compare(
            &first,
            &second,
            &unknown_covariance(),
            &dependencies("run-process", "dataset-a", true),
            &dependencies("run-d0fus", "dataset-b", true),
        );
        assert_eq!(
            result.disposition,
            CrossSolverComparisonDisposition::NotComparable
        );
        assert!(
            result
                .diagnostics
                .contains(&ComparisonDiagnostic::DifferentGeometry)
        );
        assert_eq!(result.second_minus_first, None);
    }

    #[test]
    fn service_boundary_mismatch_is_not_comparable() {
        let first = output("run-process", 1200.0, 'b');
        let mut second = output("run-d0fus", 1185.0, 'c');
        second.service_boundary_ref = "plasma-gain-boundary-v1".to_string();
        let result = compare(
            &first,
            &second,
            &unknown_covariance(),
            &dependencies("run-process", "dataset-a", true),
            &dependencies("run-d0fus", "dataset-b", true),
        );
        assert!(
            result
                .diagnostics
                .contains(&ComparisonDiagnostic::DifferentServiceBoundary)
        );
        assert_eq!(
            result.disposition,
            CrossSolverComparisonDisposition::NotComparable
        );
    }

    #[test]
    fn quantity_unit_frame_sign_time_and_normalization_mismatches_refuse_comparison() {
        let first = output("run-process", 1200.0, 'b');
        let mut second = output("run-d0fus", 1185.0, 'c');
        second.unit_ref = "unit-watt".to_string();
        second.frame_ref = "frame-machine-axis".to_string();
        second.sign_convention_ref = "grid-import-positive-v1".to_string();
        second.time_basis_ref = "steady-state-window-60s-v1".to_string();
        second.normalization_contract_ref = "normalization-v2".to_string();
        second.canonical_quantity_ref = "quantity-thermal-power".to_string();
        let result = compare(
            &first,
            &second,
            &unknown_covariance(),
            &dependencies("run-process", "dataset-a", true),
            &dependencies("run-d0fus", "dataset-b", true),
        );
        assert_eq!(
            result.disposition,
            CrossSolverComparisonDisposition::NotComparable
        );
        for diagnostic in [
            ComparisonDiagnostic::DifferentUnit,
            ComparisonDiagnostic::DifferentFrame,
            ComparisonDiagnostic::DifferentSignConvention,
            ComparisonDiagnostic::DifferentTimeBasis,
            ComparisonDiagnostic::DifferentNormalizationContract,
            ComparisonDiagnostic::DifferentCanonicalQuantity,
        ] {
            assert!(result.diagnostics.contains(&diagnostic));
        }
    }

    #[test]
    fn candidate_revision_mismatch_is_not_comparable() {
        let first = output("run-process", 1200.0, 'b');
        let mut second = output("run-d0fus", 1185.0, 'c');
        second.candidate_revision = "design-r2".to_string();
        let result = compare(
            &first,
            &second,
            &unknown_covariance(),
            &dependencies("run-process", "dataset-a", true),
            &dependencies("run-d0fus", "dataset-b", true),
        );
        assert!(
            result
                .diagnostics
                .contains(&ComparisonDiagnostic::DifferentCandidateRevision)
        );
    }

    #[test]
    fn out_of_domain_and_nonconverged_runs_are_not_successful_comparisons() {
        let mut first = output("run-process", 1200.0, 'b');
        let second = output("run-d0fus", 1185.0, 'c');
        first.applicability = ApplicabilityState::OutOfDomain;
        first.convergence = ConvergenceState::NonConverged;
        let result = compare(
            &first,
            &second,
            &unknown_covariance(),
            &dependencies("run-process", "dataset-a", true),
            &dependencies("run-d0fus", "dataset-b", true),
        );
        assert_eq!(
            result.disposition,
            CrossSolverComparisonDisposition::NotComparable
        );
        assert!(
            result
                .diagnostics
                .contains(&ComparisonDiagnostic::FirstRunOutOfDomain)
        );
        assert!(
            result
                .diagnostics
                .contains(&ComparisonDiagnostic::FirstRunNotConverged)
        );
    }

    #[test]
    fn stale_covariance_receipt_is_not_attached_to_new_outputs() {
        let first = output("run-process", 1200.0, 'b');
        let second = output("run-d0fus", 1185.0, 'c');
        let mut covariance = known_covariance(&first, &second);
        covariance.second_output_sha256 = Some(digest('e'));
        let result = compare(
            &first,
            &second,
            &covariance,
            &dependencies("run-process", "dataset-a", true),
            &dependencies("run-d0fus", "dataset-b", true),
        );
        assert_eq!(
            result.disposition,
            CrossSolverComparisonDisposition::ComparableDescriptiveOnly
        );
        assert_eq!(
            result.cross_covariance,
            CrossCovarianceDisposition::StaleOrMismatched
        );
        assert!(
            result
                .diagnostics
                .contains(&ComparisonDiagnostic::CrossCovarianceReceiptMismatch)
        );
        assert_eq!(result.covariance_receipt_ref, None);
    }

    #[test]
    fn swapped_covariance_run_order_is_stale() {
        let first = output("run-process", 1200.0, 'b');
        let second = output("run-d0fus", 1185.0, 'c');
        let mut covariance = known_covariance(&first, &second);
        covariance.first_run_id = Some(second.run_id.clone());
        covariance.second_run_id = Some(first.run_id.clone());
        let result = compare(
            &first,
            &second,
            &covariance,
            &dependencies("run-process", "dataset-a", true),
            &dependencies("run-d0fus", "dataset-b", true),
        );
        assert_eq!(
            result.cross_covariance,
            CrossCovarianceDisposition::StaleOrMismatched
        );
    }

    #[test]
    fn non_finite_value_and_malformed_artifact_digests_are_refused() {
        let mut first = output("run-process", f64::NAN, 'x');
        first.output_sha256 = "bad".to_string();
        let second = output("run-d0fus", 1185.0, 'c');
        let errors = compare_solver_outputs(
            &case(),
            &first,
            &second,
            &dependencies("run-process", "dataset-a", true),
            &dependencies("run-d0fus", "dataset-b", true),
            &unknown_covariance(),
        )
        .unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            ComparisonValidationError::InvalidSha256("first.output_sha256")
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            ComparisonValidationError::NonFiniteNumber("first.normalized_value")
        )));
    }

    #[test]
    fn dependency_manifest_must_match_exact_run_id() {
        let first = output("run-process", 1200.0, 'b');
        let second = output("run-d0fus", 1185.0, 'c');
        let errors = compare_solver_outputs(
            &case(),
            &first,
            &second,
            &dependencies("old-run", "dataset-a", true),
            &dependencies("run-d0fus", "dataset-b", true),
            &unknown_covariance(),
        )
        .unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            ComparisonValidationError::DependencyManifestRunMismatch { .. }
        )));
    }

    #[test]
    fn covariance_unknown_and_not_assessed_are_distinct() {
        let first = output("run-process", 1200.0, 'b');
        let second = output("run-d0fus", 1185.0, 'c');
        let result = compare(
            &first,
            &second,
            &CrossCovarianceEvidence {
                state: CrossCovarianceState::NotAssessed,
                receipt_ref: None,
                first_run_id: None,
                second_run_id: None,
                observable_id: None,
                first_output_sha256: None,
                second_output_sha256: None,
            },
            &dependencies("run-process", "dataset-a", true),
            &dependencies("run-d0fus", "dataset-b", true),
        );
        assert_eq!(
            result.cross_covariance,
            CrossCovarianceDisposition::NotAssessed
        );
        assert_eq!(
            result.disposition,
            CrossSolverComparisonDisposition::ComparableDescriptiveOnly
        );
    }

    #[test]
    fn subtraction_overflow_withholds_the_numeric_difference() {
        let first = output("run-process", -f64::MAX, 'b');
        let second = output("run-d0fus", f64::MAX, 'c');
        let result = compare(
            &first,
            &second,
            &unknown_covariance(),
            &dependencies("run-process", "dataset-a", true),
            &dependencies("run-d0fus", "dataset-b", true),
        );
        assert_eq!(
            result.disposition,
            CrossSolverComparisonDisposition::NotComparable
        );
        assert!(
            result
                .diagnostics
                .contains(&ComparisonDiagnostic::NonFiniteDifference)
        );
        assert_eq!(result.second_minus_first, None);
    }
}
