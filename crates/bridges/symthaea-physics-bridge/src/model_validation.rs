// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Non-forgeable satisfied-model capability for authority-bearing solver work.
//!
//! A frozen physical contract is only a specification. Solver federation must
//! additionally bind the exact validation result that demonstrated the exact
//! model instance satisfied that contract. This module provides that boundary:
//! callers cannot construct `SatisfiedPhysicalModel` directly or deserialize it
//! from claimed wire data; it is minted only by rerunning `evaluate_physical_model`
//! and observing a satisfied closure with passing dimensional assessment.

use symthaea_core::hdc::conjecture_engine::Expr;
use symthaea_science_research::Sha256Digest;

use crate::{
    ContractClosure, ContractFinding, DimensionAssessment, FrozenPhysicalModelContract,
    ObligationEvaluation, PhysicalModelReport, UnitMap, evaluate_physical_model,
};

pub const PHYSICAL_MODEL_REPORT_SCHEMA: &str = "symthaea.physical-model-report.v1";
const REPORT_DIGEST_DOMAIN: &str = "symthaea.physical-model-report.identity.v1";

/// Exact inputs required to rerun model validation.
pub struct PhysicalModelEvaluationInput<'a> {
    pub expression: &'a Expr,
    pub units: &'a UnitMap,
    pub observed_model_artifact_sha256: &'a Sha256Digest,
    pub observed_expression_sha256: &'a Sha256Digest,
    pub observed_unit_context_sha256: &'a Sha256Digest,
    pub evaluations: &'a [ObligationEvaluation],
}

/// Capability proving that the exact model instance satisfied the exact frozen
/// physical contract under the exact retained evaluation inputs.
///
/// Fields are private and the type is intentionally not deserializable. A SHA
/// string naming a purported report is not enough to construct this capability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SatisfiedPhysicalModel {
    report: PhysicalModelReport,
    report_sha256: Sha256Digest,
}

impl SatisfiedPhysicalModel {
    pub fn report(&self) -> &PhysicalModelReport {
        &self.report
    }

    pub fn report_sha256(&self) -> &Sha256Digest {
        &self.report_sha256
    }

    pub fn subject_sha256(&self) -> &Sha256Digest {
        &self.report.subject_sha256
    }

    pub fn contract_sha256(&self) -> &Sha256Digest {
        &self.report.contract_sha256
    }

    pub fn model_artifact_sha256(&self) -> &Sha256Digest {
        &self.report.model_artifact_sha256
    }
}

/// Rerun physical-model validation and mint a capability only for a satisfied
/// result. The full non-satisfied report is returned on failure so diagnostics
/// are not lost or compressed into a boolean.
pub fn evaluate_satisfied_physical_model(
    frozen: &FrozenPhysicalModelContract,
    input: PhysicalModelEvaluationInput<'_>,
) -> Result<SatisfiedPhysicalModel, PhysicalModelReport> {
    let report = evaluate_physical_model(
        frozen,
        input.expression,
        input.units,
        input.observed_model_artifact_sha256,
        input.observed_expression_sha256,
        input.observed_unit_context_sha256,
        input.evaluations,
    );

    let dimension_passes = matches!(
        &report.dimension_assessment,
        DimensionAssessment::Pass { .. }
    );
    if report.closure != ContractClosure::Satisfied || !dimension_passes {
        return Err(report);
    }

    let report_sha256 = physical_model_report_digest(&report);
    Ok(SatisfiedPhysicalModel {
        report,
        report_sha256,
    })
}

/// Deterministic semantic identity for a physical-model validation report.
///
/// This intentionally hashes typed report semantics rather than Debug or JSON
/// formatting. Findings are sorted before hashing so equivalent reports retain
/// one identity even if a future caller constructs the vector in another order.
pub fn physical_model_report_digest(report: &PhysicalModelReport) -> Sha256Digest {
    let mut digest = FramedDigest::new(REPORT_DIGEST_DOMAIN);
    digest.text(PHYSICAL_MODEL_REPORT_SCHEMA);
    digest.text(report.contract_sha256.as_str());
    digest.text(report.subject_sha256.as_str());
    digest.text(report.model_artifact_sha256.as_str());
    digest_dimension_assessment(&mut digest, &report.dimension_assessment);
    digest.text(closure_tag(report.closure));

    for obligation_id in &report.evaluated_obligation_ids {
        digest.text("evaluated-obligation");
        digest.text(obligation_id.as_str());
    }

    let mut findings = report.findings.clone();
    findings.sort();
    for finding in &findings {
        let (tag, obligation_id) = finding_parts(finding);
        digest.text("finding");
        digest.text(tag);
        digest.text(obligation_id.as_str());
    }
    digest.finish()
}

fn digest_dimension_assessment(digest: &mut FramedDigest, assessment: &DimensionAssessment) {
    match assessment {
        DimensionAssessment::Pass { observed } => {
            digest.text("dimension-pass");
            digest_dimensions(digest, *observed);
        }
        DimensionAssessment::Rejected {
            unknown_variables,
            inconsistent,
        } => {
            digest.text("dimension-rejected");
            digest.text(if *inconsistent { "inconsistent" } else { "consistent" });
            for variable in unknown_variables {
                digest.text("unknown-variable");
                digest.text(variable);
            }
        }
        DimensionAssessment::Mismatch { expected, observed } => {
            digest.text("dimension-mismatch");
            digest_dimensions(digest, *expected);
            digest_dimensions(digest, *observed);
        }
        DimensionAssessment::IdentityMismatch => digest.text("dimension-identity-mismatch"),
    }
}

fn digest_dimensions(digest: &mut FramedDigest, dimensions: crate::DimensionalSignature) {
    for exponent in dimensions.as_array() {
        digest.text(&exponent.to_string());
    }
}

const fn closure_tag(closure: ContractClosure) -> &'static str {
    match closure {
        ContractClosure::Satisfied => "satisfied",
        ContractClosure::Incomplete => "incomplete",
        ContractClosure::Violated => "violated",
        ContractClosure::Invalid => "invalid",
    }
}

fn finding_parts(finding: &ContractFinding) -> (&'static str, &symthaea_science_research::ResearchId) {
    match finding {
        ContractFinding::DuplicateEvaluation { obligation_id } => ("duplicate-evaluation", obligation_id),
        ContractFinding::UnknownObligation { obligation_id } => ("unknown-obligation", obligation_id),
        ContractFinding::PredicateSubstitution { obligation_id } => ("predicate-substitution", obligation_id),
        ContractFinding::EvaluatorSubstitution { obligation_id } => ("evaluator-substitution", obligation_id),
        ContractFinding::ExecutedOutcomeMissingArtifact { obligation_id } => ("executed-outcome-missing-artifact", obligation_id),
        ContractFinding::NonExecutedOutcomeMissingJustification { obligation_id } => ("nonexecuted-outcome-missing-justification", obligation_id),
        ContractFinding::MissingRequiredEvaluation { obligation_id } => ("missing-required-evaluation", obligation_id),
        ContractFinding::RequiredUnknown { obligation_id } => ("required-unknown", obligation_id),
        ContractFinding::RequiredNotApplicable { obligation_id } => ("required-not-applicable", obligation_id),
        ContractFinding::ForbiddenNotApplicable { obligation_id } => ("forbidden-not-applicable", obligation_id),
        ContractFinding::RequiredInvalid { obligation_id } => ("required-invalid", obligation_id),
        ContractFinding::RequiredFailure { obligation_id } => ("required-failure", obligation_id),
        ContractFinding::AdvisoryFailure { obligation_id } => ("advisory-failure", obligation_id),
        ContractFinding::AdvisoryUnknown { obligation_id } => ("advisory-unknown", obligation_id),
        ContractFinding::AdvisoryInvalid { obligation_id } => ("advisory-invalid", obligation_id),
    }
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
    use crate::{
        DimensionalSignature, ObligationKind, ObligationOutcome, PhysicalModelContract,
        PhysicalObligationSpec, PHYSICAL_MODEL_CONTRACT_SCHEMA,
    };
    use symthaea_core::hdc::conjecture_engine::{BinOp, Expr};
    use symthaea_science_research::ResearchId;

    fn id(value: &str) -> ResearchId {
        ResearchId::parse(value).unwrap()
    }

    fn sha(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }

    fn energy_expr() -> Expr {
        Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("m".into())),
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("v".into())),
                Box::new(Expr::Const(2.0)),
            )),
        )
    }

    fn units() -> UnitMap {
        [
            ("m".to_owned(), DimensionalSignature::MASS),
            ("v".to_owned(), DimensionalSignature::VELOCITY),
        ]
        .into_iter()
        .collect()
    }

    fn frozen_contract() -> FrozenPhysicalModelContract {
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

    fn pass_eval(frozen: &FrozenPhysicalModelContract) -> ObligationEvaluation {
        let spec = &frozen.contract().obligations[0];
        ObligationEvaluation {
            obligation_id: spec.obligation_id.clone(),
            predicate_sha256: spec.predicate_sha256.clone(),
            evaluator_sha256: spec.evaluator_sha256.clone(),
            outcome: ObligationOutcome::Pass,
            result_artifact_sha256: Some(sha("result")),
            justification_sha256: None,
        }
    }

    #[test]
    fn satisfied_report_mints_capability() {
        let frozen = frozen_contract();
        let expression = energy_expr();
        let units = units();
        let evaluations = [pass_eval(&frozen)];
        let satisfied = evaluate_satisfied_physical_model(
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
        .unwrap();
        assert_eq!(satisfied.report().closure, ContractClosure::Satisfied);
        assert_eq!(
            satisfied.report_sha256(),
            &physical_model_report_digest(satisfied.report())
        );
    }

    #[test]
    fn incomplete_report_cannot_mint_capability() {
        let frozen = frozen_contract();
        let expression = energy_expr();
        let units = UnitMap::new();
        let evaluations = [pass_eval(&frozen)];
        let report = evaluate_satisfied_physical_model(
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
        .unwrap_err();
        assert_eq!(report.closure, ContractClosure::Incomplete);
    }

    #[test]
    fn violated_report_cannot_mint_capability() {
        let frozen = frozen_contract();
        let expression = energy_expr();
        let units = units();
        let mut failed = pass_eval(&frozen);
        failed.outcome = ObligationOutcome::Fail;
        let evaluations = [failed];
        let report = evaluate_satisfied_physical_model(
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
        .unwrap_err();
        assert_eq!(report.closure, ContractClosure::Violated);
    }

    #[test]
    fn report_digest_changes_when_semantics_change() {
        let frozen = frozen_contract();
        let expression = energy_expr();
        let units = units();
        let evaluations = [pass_eval(&frozen)];
        let satisfied = evaluate_satisfied_physical_model(
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
        .unwrap();
        let original = physical_model_report_digest(satisfied.report());
        let mut changed = satisfied.report().clone();
        changed.closure = ContractClosure::Incomplete;
        assert_ne!(original, physical_model_report_digest(&changed));
    }
}
