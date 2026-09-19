// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Authority gate for cross-solver federation.
//!
//! The raw federation engine compares exact solver receipts and pairwise
//! results. This module adds the missing authority boundary: every solver
//! receipt must first be bound to a non-forgeable `SatisfiedPhysicalModel`
//! capability. External callers therefore cannot federate solvers against a
//! model whose physical contract was incomplete, violated, invalid, or merely
//! claimed satisfied by supplying a digest string.

use serde::Serialize;
use symthaea_science_research::{ResearchId, Sha256Digest, SharedRootKind};

use crate::{
    SatisfiedPhysicalModel,
    solver_federation::{
        FederationCoverage, FederationFinding, FrozenSolverFederationSpec, PairLineage,
        SolverAgreementState, SolverFederationReport, SolverMethodFamily, SolverPairComparison,
        SolverReceipt, evaluate_solver_federation as evaluate_solver_federation_raw,
    },
};

const VALIDATED_FEDERATION_REPORT_DOMAIN: &str =
    "symthaea.validated-solver-federation-report.identity.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverReceiptBindingIssue {
    SubjectMismatch {
        expected: Sha256Digest,
        found: Sha256Digest,
    },
    ContractMismatch {
        expected: Sha256Digest,
        found: Sha256Digest,
    },
}

/// Solver receipt bound to the exact satisfied model-validation capability.
///
/// Private validation fields prevent callers from relabeling a receipt as
/// belonging to another model artifact or another successful validation run.
/// Serialization is allowed for retained evidence; deserialization is
/// deliberately absent so wire data cannot mint this capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ValidatedSolverReceipt {
    receipt: SolverReceipt,
    model_artifact_sha256: Sha256Digest,
    model_validation_report_sha256: Sha256Digest,
}

impl ValidatedSolverReceipt {
    pub fn receipt(&self) -> &SolverReceipt {
        &self.receipt
    }

    pub fn solver_id(&self) -> &ResearchId {
        &self.receipt.solver_id
    }

    pub fn model_artifact_sha256(&self) -> &Sha256Digest {
        &self.model_artifact_sha256
    }

    pub fn model_validation_report_sha256(&self) -> &Sha256Digest {
        &self.model_validation_report_sha256
    }
}

pub fn bind_solver_receipt(
    model: &SatisfiedPhysicalModel,
    receipt: SolverReceipt,
) -> Result<ValidatedSolverReceipt, SolverReceiptBindingIssue> {
    if &receipt.subject_sha256 != model.subject_sha256() {
        return Err(SolverReceiptBindingIssue::SubjectMismatch {
            expected: model.subject_sha256().clone(),
            found: receipt.subject_sha256,
        });
    }
    if &receipt.model_contract_sha256 != model.contract_sha256() {
        return Err(SolverReceiptBindingIssue::ContractMismatch {
            expected: model.contract_sha256().clone(),
            found: receipt.model_contract_sha256,
        });
    }

    Ok(ValidatedSolverReceipt {
        receipt,
        model_artifact_sha256: model.model_artifact_sha256().clone(),
        model_validation_report_sha256: model.report_sha256().clone(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SolverFederationGateIssue {
    SpecSubjectMismatch,
    SpecContractMismatch,
    ReceiptModelArtifactMismatch { solver_id: ResearchId },
    ReceiptValidationReportMismatch { solver_id: ResearchId },
}

/// Federation report explicitly bound to the exact model artifact and exact
/// satisfied validation report used to admit its solver receipts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ValidatedSolverFederationReport {
    report: SolverFederationReport,
    model_artifact_sha256: Sha256Digest,
    model_validation_report_sha256: Sha256Digest,
    report_sha256: Sha256Digest,
}

impl ValidatedSolverFederationReport {
    pub fn report(&self) -> &SolverFederationReport {
        &self.report
    }

    pub fn model_artifact_sha256(&self) -> &Sha256Digest {
        &self.model_artifact_sha256
    }

    pub fn model_validation_report_sha256(&self) -> &Sha256Digest {
        &self.model_validation_report_sha256
    }

    pub fn report_sha256(&self) -> &Sha256Digest {
        &self.report_sha256
    }
}

/// Evaluate solver federation only after all receipts have been bound to the
/// exact satisfied physical-model capability.
///
/// Gate failures are returned separately from numerical agreement/disagreement:
/// an identity/authority substitution is not another solver outcome.
pub fn evaluate_validated_solver_federation(
    model: &SatisfiedPhysicalModel,
    frozen: &FrozenSolverFederationSpec,
    receipts: &[ValidatedSolverReceipt],
    comparisons: &[SolverPairComparison],
) -> Result<ValidatedSolverFederationReport, Vec<SolverFederationGateIssue>> {
    let spec = frozen.spec();
    let mut issues = Vec::new();

    if &spec.subject_sha256 != model.subject_sha256() {
        issues.push(SolverFederationGateIssue::SpecSubjectMismatch);
    }
    if &spec.model_contract_sha256 != model.contract_sha256() {
        issues.push(SolverFederationGateIssue::SpecContractMismatch);
    }

    for receipt in receipts {
        if receipt.model_artifact_sha256() != model.model_artifact_sha256() {
            issues.push(SolverFederationGateIssue::ReceiptModelArtifactMismatch {
                solver_id: receipt.solver_id().clone(),
            });
        }
        if receipt.model_validation_report_sha256() != model.report_sha256() {
            issues.push(SolverFederationGateIssue::ReceiptValidationReportMismatch {
                solver_id: receipt.solver_id().clone(),
            });
        }
    }

    if !issues.is_empty() {
        issues.sort();
        issues.dedup();
        return Err(issues);
    }

    let raw_receipts: Vec<_> = receipts.iter().map(|item| item.receipt.clone()).collect();
    let report = evaluate_solver_federation_raw(frozen, &raw_receipts, comparisons);
    let report_sha256 = validated_report_digest(
        &report,
        model.model_artifact_sha256(),
        model.report_sha256(),
    );

    Ok(ValidatedSolverFederationReport {
        report,
        model_artifact_sha256: model.model_artifact_sha256().clone(),
        model_validation_report_sha256: model.report_sha256().clone(),
        report_sha256,
    })
}

fn validated_report_digest(
    report: &SolverFederationReport,
    model_artifact_sha256: &Sha256Digest,
    model_validation_report_sha256: &Sha256Digest,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(VALIDATED_FEDERATION_REPORT_DOMAIN);
    digest.text(report.federation_sha256.as_str());
    digest.text(report.subject_sha256.as_str());
    digest.text(report.model_contract_sha256.as_str());
    digest.text(model_artifact_sha256.as_str());
    digest.text(model_validation_report_sha256.as_str());

    for solver_id in &report.solver_ids {
        digest.text("solver");
        digest.text(solver_id.as_str());
    }
    for family in &report.distinct_method_families {
        digest.text("method-family");
        digest.text(method_family_tag(*family));
    }
    digest.text(coverage_tag(report.coverage));
    digest.text(agreement_tag(report.agreement));

    let mut lineage = report.lineage.clone();
    lineage.sort_by(|left, right| {
        (&left.left_solver_id, &left.right_solver_id)
            .cmp(&(&right.left_solver_id, &right.right_solver_id))
    });
    for pair in &lineage {
        digest_pair_lineage(&mut digest, pair);
    }

    let mut findings = report.findings.clone();
    findings.sort();
    for finding in &findings {
        digest_finding(&mut digest, finding);
    }
    digest.text(if report.independence_established {
        "independence-established"
    } else {
        "independence-not-established"
    });
    digest.finish()
}

fn digest_pair_lineage(digest: &mut FramedDigest, pair: &PairLineage) {
    digest.text("pair-lineage");
    digest.text(pair.left_solver_id.as_str());
    digest.text(pair.right_solver_id.as_str());
    digest.text(bool_tag(pair.same_method_family));
    digest.text(bool_tag(pair.shared_implementation));
    digest.text(bool_tag(pair.shared_environment));
    digest.text(bool_tag(pair.shared_input));
    for root in &pair.shared_declared_roots {
        digest.text("shared-root");
        digest.text(shared_root_kind_tag(root.kind));
        digest.text(root.digest.as_str());
    }
}

fn digest_finding(digest: &mut FramedDigest, finding: &FederationFinding) {
    match finding {
        FederationFinding::DuplicateSolver { solver_id } => {
            digest.text("duplicate-solver");
            digest.text(solver_id.as_str());
        }
        FederationFinding::SolverSubjectMismatch { solver_id } => {
            digest.text("solver-subject-mismatch");
            digest.text(solver_id.as_str());
        }
        FederationFinding::SolverContractMismatch { solver_id } => {
            digest.text("solver-contract-mismatch");
            digest.text(solver_id.as_str());
        }
        FederationFinding::DuplicatePair { left, right } => {
            digest_pair_ids(digest, "duplicate-pair", left, right);
        }
        FederationFinding::SelfComparison { solver_id } => {
            digest.text("self-comparison");
            digest.text(solver_id.as_str());
        }
        FederationFinding::UnknownSolver { solver_id } => {
            digest.text("unknown-solver");
            digest.text(solver_id.as_str());
        }
        FederationFinding::ResultSubstitution { solver_id } => {
            digest.text("result-substitution");
            digest.text(solver_id.as_str());
        }
        FederationFinding::MetricSubstitution { left, right } => {
            digest_pair_ids(digest, "metric-substitution", left, right);
        }
        FederationFinding::AgreementPolicySubstitution { left, right } => {
            digest_pair_ids(digest, "agreement-policy-substitution", left, right);
        }
        FederationFinding::ComparisonArtifactMissing { left, right } => {
            digest_pair_ids(digest, "comparison-artifact-missing", left, right);
        }
        FederationFinding::ComparisonJustificationMissing { left, right } => {
            digest_pair_ids(digest, "comparison-justification-missing", left, right);
        }
        FederationFinding::MissingComparison { left, right } => {
            digest_pair_ids(digest, "missing-comparison", left, right);
        }
        FederationFinding::InsufficientSolvers { observed, required } => {
            digest.text("insufficient-solvers");
            digest.text(&observed.to_string());
            digest.text(&required.to_string());
        }
        FederationFinding::InsufficientMethodDiversity { observed, required } => {
            digest.text("insufficient-method-diversity");
            digest.text(&observed.to_string());
            digest.text(&required.to_string());
        }
        FederationFinding::SharedResultArtifact { left, right } => {
            digest_pair_ids(digest, "shared-result-artifact", left, right);
        }
    }
}

fn digest_pair_ids(digest: &mut FramedDigest, tag: &str, left: &ResearchId, right: &ResearchId) {
    digest.text(tag);
    digest.text(left.as_str());
    digest.text(right.as_str());
}

const fn method_family_tag(family: SolverMethodFamily) -> &'static str {
    match family {
        SolverMethodFamily::Analytic => "analytic",
        SolverMethodFamily::Symbolic => "symbolic",
        SolverMethodFamily::OdeIntegrator => "ode-integrator",
        SolverMethodFamily::FiniteDifference => "finite-difference",
        SolverMethodFamily::FiniteVolume => "finite-volume",
        SolverMethodFamily::FiniteElement => "finite-element",
        SolverMethodFamily::Spectral => "spectral",
        SolverMethodFamily::Lattice => "lattice",
        SolverMethodFamily::MonteCarlo => "monte-carlo",
        SolverMethodFamily::ReducedOrder => "reduced-order",
        SolverMethodFamily::PhysicsInformedNeuralNetwork => "pinn",
        SolverMethodFamily::NeuralOperator => "neural-operator",
        SolverMethodFamily::HdcDiscoveredModel => "hdc-discovered-model",
        SolverMethodFamily::Other => "other",
    }
}

const fn shared_root_kind_tag(kind: SharedRootKind) -> &'static str {
    match kind {
        SharedRootKind::RawData => "raw-data",
        SharedRootKind::Calibration => "calibration",
        SharedRootKind::Preprocessing => "preprocessing",
        SharedRootKind::Implementation => "implementation",
        SharedRootKind::SolverLibrary => "solver-library",
        SharedRootKind::ModelCheckpoint => "model-checkpoint",
        SharedRootKind::Organization => "organization",
        SharedRootKind::Instrument => "instrument",
        SharedRootKind::Other => "other",
    }
}

const fn coverage_tag(coverage: FederationCoverage) -> &'static str {
    match coverage {
        FederationCoverage::Complete => "complete",
        FederationCoverage::Incomplete => "incomplete",
        FederationCoverage::Invalid => "invalid",
    }
}

const fn agreement_tag(agreement: SolverAgreementState) -> &'static str {
    match agreement {
        SolverAgreementState::AgreementObserved => "agreement-observed",
        SolverAgreementState::DisagreementObserved => "disagreement-observed",
        SolverAgreementState::Incomplete => "incomplete",
        SolverAgreementState::Invalid => "invalid",
    }
}

const fn bool_tag(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

struct FramedDigest {
    bytes: Vec<u8>,
}

impl FramedDigest {
    fn new(domain: &str) -> Self {
        let mut digest = Self { bytes: Vec::new() };
        digest.text(domain);
        digest
    }

    fn text(&mut self, value: &str) {
        self.bytes
            .extend_from_slice(&(value.len() as u64).to_be_bytes());
        self.bytes.extend_from_slice(value.as_bytes());
    }

    fn finish(self) -> Sha256Digest {
        Sha256Digest::of_bytes(&self.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    use crate::{
        DimensionalSignature, FrozenPhysicalModelContract, ObligationEvaluation, ObligationKind,
        ObligationOutcome, PHYSICAL_MODEL_CONTRACT_SCHEMA, PairwiseAgreement,
        PhysicalModelContract, PhysicalModelEvaluationInput, PhysicalObligationSpec,
        SOLVER_FEDERATION_SCHEMA, SolverFederationSpec, SolverMethodFamily,
        SolverPairComparison, UnitMap, evaluate_satisfied_physical_model,
    };
    use symthaea_core::hdc::conjecture_engine::{BinOp, Expr};

    fn id(value: &str) -> ResearchId {
        ResearchId::parse(value).unwrap()
    }

    fn sha(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }

    fn contract() -> FrozenPhysicalModelContract {
        PhysicalModelContract {
            schema_version: PHYSICAL_MODEL_CONTRACT_SCHEMA.into(),
            contract_id: id("PHYS-CONTRACT-1"),
            subject_sha256: sha("subject"),
            model_artifact_sha256: sha("model"),
            dimensional_expression_sha256: sha("expression"),
            unit_context_sha256: sha("units"),
            expected_dimensions: DimensionalSignature::ENERGY,
            applicability_regime_sha256: sha("regime"),
            numerical_convergence_policy_sha256: None,
            obligations: vec![PhysicalObligationSpec {
                obligation_id: id("ENERGY-CONSERVATION"),
                kind: ObligationKind::ConservationEnergy,
                predicate_sha256: sha("predicate"),
                evaluator_sha256: sha("evaluator"),
                required: true,
                allow_not_applicable: false,
            }],
            supersedes_contract_sha256: None,
        }
        .freeze()
        .unwrap()
    }

    fn satisfied_model() -> SatisfiedPhysicalModel {
        let frozen = contract();
        let expression = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("m".into())),
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("v".into())),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let units: UnitMap = [
            ("m".to_owned(), DimensionalSignature::MASS),
            ("v".to_owned(), DimensionalSignature::VELOCITY),
        ]
        .into_iter()
        .collect();
        let obligation = &frozen.contract().obligations[0];
        let evaluations = [ObligationEvaluation {
            obligation_id: obligation.obligation_id.clone(),
            predicate_sha256: obligation.predicate_sha256.clone(),
            evaluator_sha256: obligation.evaluator_sha256.clone(),
            outcome: ObligationOutcome::Pass,
            result_artifact_sha256: Some(sha("physical-check")),
            justification_sha256: None,
        }];
        evaluate_satisfied_physical_model(
            &frozen,
            PhysicalModelEvaluationInput {
                expression: &expression,
                units: &units,
                observed_model_artifact_sha256: &sha("model"),
                observed_expression_sha256: &sha("expression"),
                observed_unit_context_sha256: &sha("units"),
                evaluations: &evaluations,
            },
        )
        .unwrap()
    }

    fn spec(model: &SatisfiedPhysicalModel) -> FrozenSolverFederationSpec {
        SolverFederationSpec {
            schema_version: SOLVER_FEDERATION_SCHEMA.into(),
            federation_id: id("FED-1"),
            subject_sha256: model.subject_sha256().clone(),
            model_contract_sha256: model.contract_sha256().clone(),
            comparison_metric_sha256: sha("metric"),
            agreement_policy_sha256: sha("policy"),
            minimum_solvers: 2,
            minimum_distinct_method_families: 2,
        }
        .freeze()
        .unwrap()
    }

    fn raw_receipt(
        model: &SatisfiedPhysicalModel,
        name: &str,
        family: SolverMethodFamily,
    ) -> SolverReceipt {
        SolverReceipt {
            solver_id: id(name),
            subject_sha256: model.subject_sha256().clone(),
            model_contract_sha256: model.contract_sha256().clone(),
            method_family: family,
            implementation_sha256: sha(&format!("implementation-{name}")),
            environment_sha256: sha(&format!("environment-{name}")),
            input_sha256: sha("shared-input"),
            result_sha256: sha(&format!("result-{name}")),
            diagnostics_sha256: sha(&format!("diagnostics-{name}")),
            dependency_roots: BTreeSet::new(),
        }
    }

    fn comparison(left: &SolverReceipt, right: &SolverReceipt) -> SolverPairComparison {
        SolverPairComparison {
            left_solver_id: left.solver_id.clone(),
            right_solver_id: right.solver_id.clone(),
            left_result_sha256: left.result_sha256.clone(),
            right_result_sha256: right.result_sha256.clone(),
            comparison_metric_sha256: sha("metric"),
            agreement_policy_sha256: sha("policy"),
            outcome: PairwiseAgreement::Agree,
            comparison_artifact_sha256: Some(sha("comparison")),
            justification_sha256: None,
        }
    }

    #[test]
    fn receipt_binding_rejects_subject_substitution() {
        let model = satisfied_model();
        let mut receipt = raw_receipt(&model, "A", SolverMethodFamily::FiniteElement);
        receipt.subject_sha256 = sha("other-subject");
        assert!(matches!(
            bind_solver_receipt(&model, receipt),
            Err(SolverReceiptBindingIssue::SubjectMismatch { .. })
        ));
    }

    #[test]
    fn validated_federation_binds_exact_model_validation() {
        let model = satisfied_model();
        let a_raw = raw_receipt(&model, "A", SolverMethodFamily::FiniteElement);
        let b_raw = raw_receipt(&model, "B", SolverMethodFamily::Spectral);
        let pair = comparison(&a_raw, &b_raw);
        let a = bind_solver_receipt(&model, a_raw).unwrap();
        let b = bind_solver_receipt(&model, b_raw).unwrap();
        let report = evaluate_validated_solver_federation(
            &model,
            &spec(&model),
            &[a, b],
            &[pair],
        )
        .unwrap();
        assert_eq!(report.model_artifact_sha256(), model.model_artifact_sha256());
        assert_eq!(
            report.model_validation_report_sha256(),
            model.report_sha256()
        );
        assert_eq!(
            report.report_sha256(),
            &validated_report_digest(
                report.report(),
                report.model_artifact_sha256(),
                report.model_validation_report_sha256(),
            )
        );
        assert!(!report.report().independence_established);
    }

    #[test]
    fn federation_spec_cannot_switch_contract_after_validation() {
        let model = satisfied_model();
        let wrong = SolverFederationSpec {
            schema_version: SOLVER_FEDERATION_SCHEMA.into(),
            federation_id: id("FED-1"),
            subject_sha256: model.subject_sha256().clone(),
            model_contract_sha256: sha("different-contract"),
            comparison_metric_sha256: sha("metric"),
            agreement_policy_sha256: sha("policy"),
            minimum_solvers: 2,
            minimum_distinct_method_families: 2,
        }
        .freeze()
        .unwrap();
        assert!(matches!(
            evaluate_validated_solver_federation(&model, &wrong, &[], &[]),
            Err(issues) if issues.contains(&SolverFederationGateIssue::SpecContractMismatch)
        ));
    }
}
