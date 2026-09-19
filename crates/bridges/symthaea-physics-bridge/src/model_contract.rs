// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bearing physical-model contracts.
//!
//! Discovery may tolerate incomplete physical metadata. Authority-bearing
//! validation may not. A contract binds an exact scientific subject and model
//! artifact to dimensional expectations, an applicability regime, and typed
//! physical obligations. Unknown dimensions remain unknown, failed laws remain
//! failures, and omitted required checks remain incomplete.
//!
//! The shared layer intentionally does not pretend that conservation,
//! thermodynamics, boundary conditions, stability, and numerical convergence
//! share one universal evaluator. Each obligation instead binds the exact
//! predicate and evaluator identity used by its domain-specific implementation.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::conjecture_engine::Expr;
use symthaea_science_research::{ResearchId, Sha256Digest};

use crate::{DimensionalSignature, StrictInferenceResult, UnitMap, infer_dimensions_strict};

pub const PHYSICAL_MODEL_CONTRACT_SCHEMA: &str = "symthaea.physical-model-contract.v1";
const CONTRACT_DIGEST_DOMAIN: &str = "symthaea.physical-model-contract.identity.v1";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum ObligationKind {
    ConservationEnergy,
    ConservationMomentum,
    ConservationAngularMomentum,
    ConservationCharge,
    ConservationMass,
    ConservationParticleNumber,
    ProbabilityNormalization,
    Symmetry,
    Positivity,
    Causality,
    InitialCondition,
    BoundaryCondition,
    LimitingBehavior,
    ThermodynamicConsistency,
    Monotonicity,
    NumericalConvergence,
    ApplicabilityRegime,
    Stability,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalObligationSpec {
    pub obligation_id: ResearchId,
    pub kind: ObligationKind,
    pub predicate_sha256: Sha256Digest,
    pub evaluator_sha256: Sha256Digest,
    pub required: bool,
    pub allow_not_applicable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalModelContract {
    pub schema_version: String,
    pub contract_id: ResearchId,
    pub subject_sha256: Sha256Digest,
    pub model_artifact_sha256: Sha256Digest,
    pub dimensional_expression_sha256: Sha256Digest,
    pub unit_context_sha256: Sha256Digest,
    pub expected_dimensions: DimensionalSignature,
    pub applicability_regime_sha256: Sha256Digest,
    /// Required when a required `NumericalConvergence` obligation exists.
    /// This is separate from the evaluator implementation so acceptance
    /// tolerances cannot be silently changed with solver code.
    pub numerical_convergence_policy_sha256: Option<Sha256Digest>,
    pub obligations: Vec<PhysicalObligationSpec>,
    /// Optional lineage only. Supersession never transfers authority.
    pub supersedes_contract_sha256: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhysicalModelContractIssue {
    WrongSchemaVersion { found: String },
    MissingObligations,
    MissingRequiredObligation,
    DuplicateObligation { obligation_id: ResearchId },
    NumericalConvergencePolicyMissing,
}

impl PhysicalModelContract {
    pub fn validate(&self) -> Vec<PhysicalModelContractIssue> {
        let mut issues = Vec::new();
        if self.schema_version != PHYSICAL_MODEL_CONTRACT_SCHEMA {
            issues.push(PhysicalModelContractIssue::WrongSchemaVersion {
                found: self.schema_version.clone(),
            });
        }
        if self.obligations.is_empty() {
            issues.push(PhysicalModelContractIssue::MissingObligations);
        }
        if !self.obligations.iter().any(|item| item.required) {
            issues.push(PhysicalModelContractIssue::MissingRequiredObligation);
        }

        let mut ids = BTreeSet::new();
        for obligation in &self.obligations {
            if !ids.insert(obligation.obligation_id.clone()) {
                issues.push(PhysicalModelContractIssue::DuplicateObligation {
                    obligation_id: obligation.obligation_id.clone(),
                });
            }
        }

        let requires_convergence = self.obligations.iter().any(|item| {
            item.required && item.kind == ObligationKind::NumericalConvergence
        });
        if requires_convergence && self.numerical_convergence_policy_sha256.is_none() {
            issues.push(PhysicalModelContractIssue::NumericalConvergencePolicyMissing);
        }
        issues
    }

    pub fn freeze(self) -> Result<FrozenPhysicalModelContract, Vec<PhysicalModelContractIssue>> {
        let issues = self.validate();
        if !issues.is_empty() {
            return Err(issues);
        }
        let contract_sha256 = contract_digest(&self);
        Ok(FrozenPhysicalModelContract {
            contract: self,
            contract_sha256,
        })
    }
}

/// Imports must deserialize `PhysicalModelContract`, validate, and freeze again.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrozenPhysicalModelContract {
    contract: PhysicalModelContract,
    contract_sha256: Sha256Digest,
}

impl FrozenPhysicalModelContract {
    pub fn contract(&self) -> &PhysicalModelContract {
        &self.contract
    }

    pub fn contract_sha256(&self) -> &Sha256Digest {
        &self.contract_sha256
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum ObligationOutcome {
    Pass,
    Fail,
    Unknown,
    NotApplicable,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObligationEvaluation {
    pub obligation_id: ResearchId,
    pub predicate_sha256: Sha256Digest,
    pub evaluator_sha256: Sha256Digest,
    pub outcome: ObligationOutcome,
    /// Required for executed Pass/Fail outcomes.
    pub result_artifact_sha256: Option<Sha256Digest>,
    /// Required for Unknown/NotApplicable/Invalid outcomes.
    pub justification_sha256: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum DimensionAssessment {
    Pass {
        observed: DimensionalSignature,
    },
    /// A rejected strict inference retains both dimensions of failure. A tree
    /// can simultaneously contain unknown annotations and an inconsistent
    /// fully-annotated branch; neither fact is discarded.
    Rejected {
        unknown_variables: BTreeSet<String>,
        inconsistent: bool,
    },
    Mismatch {
        expected: DimensionalSignature,
        observed: DimensionalSignature,
    },
    IdentityMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum ContractFinding {
    DuplicateEvaluation { obligation_id: ResearchId },
    UnknownObligation { obligation_id: ResearchId },
    PredicateSubstitution { obligation_id: ResearchId },
    EvaluatorSubstitution { obligation_id: ResearchId },
    ExecutedOutcomeMissingArtifact { obligation_id: ResearchId },
    NonExecutedOutcomeMissingJustification { obligation_id: ResearchId },
    MissingRequiredEvaluation { obligation_id: ResearchId },
    RequiredUnknown { obligation_id: ResearchId },
    RequiredNotApplicable { obligation_id: ResearchId },
    ForbiddenNotApplicable { obligation_id: ResearchId },
    RequiredInvalid { obligation_id: ResearchId },
    RequiredFailure { obligation_id: ResearchId },
    AdvisoryFailure { obligation_id: ResearchId },
    AdvisoryUnknown { obligation_id: ResearchId },
    AdvisoryInvalid { obligation_id: ResearchId },
}

/// Variant order is the conservative escalation order.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum ContractClosure {
    Satisfied,
    Incomplete,
    Violated,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PhysicalModelReport {
    pub contract_sha256: Sha256Digest,
    pub subject_sha256: Sha256Digest,
    pub model_artifact_sha256: Sha256Digest,
    pub dimension_assessment: DimensionAssessment,
    pub closure: ContractClosure,
    pub evaluated_obligation_ids: BTreeSet<ResearchId>,
    pub findings: Vec<ContractFinding>,
}

/// Evaluate an exact model instance against a frozen physical contract.
///
/// The observed SHA-256 values come from the provenance layer. Matching these
/// identities is mandatory even when the supplied expression happens to pass
/// dimensional inference.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_physical_model(
    frozen: &FrozenPhysicalModelContract,
    expression: &Expr,
    units: &UnitMap,
    observed_model_artifact_sha256: &Sha256Digest,
    observed_expression_sha256: &Sha256Digest,
    observed_unit_context_sha256: &Sha256Digest,
    evaluations: &[ObligationEvaluation],
) -> PhysicalModelReport {
    let contract = frozen.contract();
    let identity_matches = observed_model_artifact_sha256 == &contract.model_artifact_sha256
        && observed_expression_sha256 == &contract.dimensional_expression_sha256
        && observed_unit_context_sha256 == &contract.unit_context_sha256;

    let dimension_assessment = if identity_matches {
        assess_dimensions(expression, units, contract.expected_dimensions)
    } else {
        DimensionAssessment::IdentityMismatch
    };

    let mut closure = match &dimension_assessment {
        DimensionAssessment::Pass { .. } => ContractClosure::Satisfied,
        DimensionAssessment::Rejected {
            unknown_variables,
            inconsistent,
        } => {
            if *inconsistent {
                ContractClosure::Violated
            } else if !unknown_variables.is_empty() {
                ContractClosure::Incomplete
            } else {
                ContractClosure::Invalid
            }
        }
        DimensionAssessment::Mismatch { .. } => ContractClosure::Violated,
        DimensionAssessment::IdentityMismatch => ContractClosure::Invalid,
    };

    let specs: BTreeMap<_, _> = contract
        .obligations
        .iter()
        .map(|spec| (spec.obligation_id.clone(), spec))
        .collect();
    let mut seen = BTreeSet::new();
    let mut valid_evaluated = BTreeSet::new();
    let mut findings = Vec::new();

    for evaluation in evaluations {
        if !seen.insert(evaluation.obligation_id.clone()) {
            findings.push(ContractFinding::DuplicateEvaluation {
                obligation_id: evaluation.obligation_id.clone(),
            });
            closure = closure.max(ContractClosure::Invalid);
            continue;
        }

        let Some(spec) = specs.get(&evaluation.obligation_id) else {
            findings.push(ContractFinding::UnknownObligation {
                obligation_id: evaluation.obligation_id.clone(),
            });
            closure = closure.max(ContractClosure::Invalid);
            continue;
        };

        if evaluation.predicate_sha256 != spec.predicate_sha256 {
            findings.push(ContractFinding::PredicateSubstitution {
                obligation_id: evaluation.obligation_id.clone(),
            });
            closure = closure.max(ContractClosure::Invalid);
            continue;
        }
        if evaluation.evaluator_sha256 != spec.evaluator_sha256 {
            findings.push(ContractFinding::EvaluatorSubstitution {
                obligation_id: evaluation.obligation_id.clone(),
            });
            closure = closure.max(ContractClosure::Invalid);
            continue;
        }

        let executed = matches!(
            evaluation.outcome,
            ObligationOutcome::Pass | ObligationOutcome::Fail
        );
        if executed && evaluation.result_artifact_sha256.is_none() {
            findings.push(ContractFinding::ExecutedOutcomeMissingArtifact {
                obligation_id: evaluation.obligation_id.clone(),
            });
            closure = closure.max(ContractClosure::Invalid);
            continue;
        }
        if !executed && evaluation.justification_sha256.is_none() {
            findings.push(ContractFinding::NonExecutedOutcomeMissingJustification {
                obligation_id: evaluation.obligation_id.clone(),
            });
            closure = closure.max(ContractClosure::Invalid);
            continue;
        }
        if evaluation.outcome == ObligationOutcome::NotApplicable && !spec.allow_not_applicable {
            findings.push(ContractFinding::ForbiddenNotApplicable {
                obligation_id: evaluation.obligation_id.clone(),
            });
            closure = closure.max(ContractClosure::Invalid);
            continue;
        }

        valid_evaluated.insert(evaluation.obligation_id.clone());
        match (spec.required, evaluation.outcome) {
            (true, ObligationOutcome::Pass) => {}
            (true, ObligationOutcome::Fail) => {
                findings.push(ContractFinding::RequiredFailure {
                    obligation_id: evaluation.obligation_id.clone(),
                });
                closure = closure.max(ContractClosure::Violated);
            }
            (true, ObligationOutcome::Unknown) => {
                findings.push(ContractFinding::RequiredUnknown {
                    obligation_id: evaluation.obligation_id.clone(),
                });
                closure = closure.max(ContractClosure::Incomplete);
            }
            (true, ObligationOutcome::NotApplicable) => {
                findings.push(ContractFinding::RequiredNotApplicable {
                    obligation_id: evaluation.obligation_id.clone(),
                });
            }
            (true, ObligationOutcome::Invalid) => {
                findings.push(ContractFinding::RequiredInvalid {
                    obligation_id: evaluation.obligation_id.clone(),
                });
                closure = closure.max(ContractClosure::Invalid);
            }
            (false, ObligationOutcome::Pass | ObligationOutcome::NotApplicable) => {}
            (false, ObligationOutcome::Fail) => {
                findings.push(ContractFinding::AdvisoryFailure {
                    obligation_id: evaluation.obligation_id.clone(),
                });
            }
            (false, ObligationOutcome::Unknown) => {
                findings.push(ContractFinding::AdvisoryUnknown {
                    obligation_id: evaluation.obligation_id.clone(),
                });
            }
            (false, ObligationOutcome::Invalid) => {
                findings.push(ContractFinding::AdvisoryInvalid {
                    obligation_id: evaluation.obligation_id.clone(),
                });
            }
        }
    }

    for spec in contract.obligations.iter().filter(|spec| spec.required) {
        if !valid_evaluated.contains(&spec.obligation_id) {
            findings.push(ContractFinding::MissingRequiredEvaluation {
                obligation_id: spec.obligation_id.clone(),
            });
            closure = closure.max(ContractClosure::Incomplete);
        }
    }
    findings.sort();

    PhysicalModelReport {
        contract_sha256: frozen.contract_sha256().clone(),
        subject_sha256: contract.subject_sha256.clone(),
        model_artifact_sha256: contract.model_artifact_sha256.clone(),
        dimension_assessment,
        closure,
        evaluated_obligation_ids: valid_evaluated,
        findings,
    }
}

fn assess_dimensions(
    expression: &Expr,
    units: &UnitMap,
    expected: DimensionalSignature,
) -> DimensionAssessment {
    match infer_dimensions_strict(expression, units) {
        StrictInferenceResult::Inferred(observed) if observed == expected => {
            DimensionAssessment::Pass { observed }
        }
        StrictInferenceResult::Inferred(observed) => DimensionAssessment::Mismatch {
            expected,
            observed,
        },
        StrictInferenceResult::Rejected(failure) => DimensionAssessment::Rejected {
            unknown_variables: failure.unknown_variables().clone(),
            inconsistent: failure.is_inconsistent(),
        },
    }
}

fn contract_digest(contract: &PhysicalModelContract) -> Sha256Digest {
    let mut digest = FramedDigest::new(CONTRACT_DIGEST_DOMAIN);
    digest.text(PHYSICAL_MODEL_CONTRACT_SCHEMA);
    digest.text(contract.contract_id.as_str());
    digest.text(contract.subject_sha256.as_str());
    digest.text(contract.model_artifact_sha256.as_str());
    digest.text(contract.dimensional_expression_sha256.as_str());
    digest.text(contract.unit_context_sha256.as_str());
    for exponent in contract.expected_dimensions.as_array() {
        digest.text(&exponent.to_string());
    }
    digest.text(contract.applicability_regime_sha256.as_str());
    digest.optional_sha(contract.numerical_convergence_policy_sha256.as_ref());

    let mut obligations = contract.obligations.clone();
    obligations.sort_by(|left, right| left.obligation_id.cmp(&right.obligation_id));
    for obligation in obligations {
        digest.text("obligation");
        digest.text(obligation.obligation_id.as_str());
        digest.text(obligation_kind_tag(obligation.kind));
        digest.text(obligation.predicate_sha256.as_str());
        digest.text(obligation.evaluator_sha256.as_str());
        digest.text(if obligation.required { "required" } else { "advisory" });
        digest.text(if obligation.allow_not_applicable {
            "na-allowed"
        } else {
            "na-forbidden"
        });
    }
    digest.optional_sha(contract.supersedes_contract_sha256.as_ref());
    digest.finish()
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

    fn optional_sha(&mut self, value: Option<&Sha256Digest>) {
        match value {
            Some(value) => {
                self.text("some");
                self.text(value.as_str());
            }
            None => self.text("none"),
        }
    }

    fn finish(self) -> Sha256Digest {
        Sha256Digest::of_bytes(&self.bytes)
    }
}

const fn obligation_kind_tag(kind: ObligationKind) -> &'static str {
    match kind {
        ObligationKind::ConservationEnergy => "conservation-energy",
        ObligationKind::ConservationMomentum => "conservation-momentum",
        ObligationKind::ConservationAngularMomentum => "conservation-angular-momentum",
        ObligationKind::ConservationCharge => "conservation-charge",
        ObligationKind::ConservationMass => "conservation-mass",
        ObligationKind::ConservationParticleNumber => "conservation-particle-number",
        ObligationKind::ProbabilityNormalization => "probability-normalization",
        ObligationKind::Symmetry => "symmetry",
        ObligationKind::Positivity => "positivity",
        ObligationKind::Causality => "causality",
        ObligationKind::InitialCondition => "initial-condition",
        ObligationKind::BoundaryCondition => "boundary-condition",
        ObligationKind::LimitingBehavior => "limiting-behavior",
        ObligationKind::ThermodynamicConsistency => "thermodynamic-consistency",
        ObligationKind::Monotonicity => "monotonicity",
        ObligationKind::NumericalConvergence => "numerical-convergence",
        ObligationKind::ApplicabilityRegime => "applicability-regime",
        ObligationKind::Stability => "stability",
        ObligationKind::Other => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::conjecture_engine::{BinOp, Expr};

    fn id(value: &str) -> ResearchId {
        ResearchId::parse(value).unwrap()
    }

    fn sha(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }

    fn units(pairs: &[(&str, DimensionalSignature)]) -> UnitMap {
        pairs
            .iter()
            .map(|(name, dimension)| ((*name).to_owned(), *dimension))
            .collect()
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

    fn obligation(value: &str, kind: ObligationKind, required: bool) -> PhysicalObligationSpec {
        PhysicalObligationSpec {
            obligation_id: id(value),
            kind,
            predicate_sha256: sha(&format!("predicate-{value}")),
            evaluator_sha256: sha(&format!("evaluator-{value}")),
            required,
            allow_not_applicable: false,
        }
    }

    fn contract() -> PhysicalModelContract {
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
            obligations: vec![obligation(
                "ENERGY-CONSERVATION",
                ObligationKind::ConservationEnergy,
                true,
            )],
            supersedes_contract_sha256: None,
        }
    }

    fn pass_eval(spec: &PhysicalObligationSpec) -> ObligationEvaluation {
        ObligationEvaluation {
            obligation_id: spec.obligation_id.clone(),
            predicate_sha256: spec.predicate_sha256.clone(),
            evaluator_sha256: spec.evaluator_sha256.clone(),
            outcome: ObligationOutcome::Pass,
            result_artifact_sha256: Some(sha("result")),
            justification_sha256: None,
        }
    }

    fn energy_units() -> UnitMap {
        units(&[
            ("m", DimensionalSignature::MASS),
            ("v", DimensionalSignature::VELOCITY),
        ])
    }

    fn evaluate(
        frozen: &FrozenPhysicalModelContract,
        units: &UnitMap,
        evaluations: &[ObligationEvaluation],
    ) -> PhysicalModelReport {
        evaluate_physical_model(
            frozen,
            &energy_expr(),
            units,
            &sha("model"),
            &sha("expression"),
            &sha("units"),
            evaluations,
        )
    }

    #[test]
    fn duplicate_obligation_ids_fail_freeze() {
        let mut draft = contract();
        draft.obligations.push(draft.obligations[0].clone());
        assert!(draft.freeze().is_err());
    }

    #[test]
    fn required_convergence_needs_explicit_policy() {
        let mut draft = contract();
        draft.obligations.push(obligation(
            "CONVERGENCE",
            ObligationKind::NumericalConvergence,
            true,
        ));
        assert!(draft.validate().iter().any(|issue| matches!(
            issue,
            PhysicalModelContractIssue::NumericalConvergencePolicyMissing
        )));
        draft.numerical_convergence_policy_sha256 = Some(sha("convergence-policy"));
        assert!(draft.freeze().is_ok());
    }

    #[test]
    fn fully_bound_model_satisfies_contract() {
        let frozen = contract().freeze().unwrap();
        let report = evaluate(
            &frozen,
            &energy_units(),
            &[pass_eval(&frozen.contract().obligations[0])],
        );
        assert_eq!(report.closure, ContractClosure::Satisfied);
        assert!(report.findings.is_empty());
    }

    #[test]
    fn unknown_dimension_annotation_is_incomplete_not_dimensionless() {
        let frozen = contract().freeze().unwrap();
        let report = evaluate(
            &frozen,
            &UnitMap::new(),
            &[pass_eval(&frozen.contract().obligations[0])],
        );
        assert_eq!(report.closure, ContractClosure::Incomplete);
        assert!(matches!(
            report.dimension_assessment,
            DimensionAssessment::Rejected {
                inconsistent: false,
                ..
            }
        ));
    }

    #[test]
    fn dimension_mismatch_violates_contract() {
        let frozen = contract().freeze().unwrap();
        let wrong = units(&[
            ("m", DimensionalSignature::MASS),
            ("v", DimensionalSignature::LENGTH),
        ]);
        let report = evaluate(
            &frozen,
            &wrong,
            &[pass_eval(&frozen.contract().obligations[0])],
        );
        assert_eq!(report.closure, ContractClosure::Violated);
    }

    #[test]
    fn missing_required_obligation_is_incomplete() {
        let frozen = contract().freeze().unwrap();
        let report = evaluate(&frozen, &energy_units(), &[]);
        assert_eq!(report.closure, ContractClosure::Incomplete);
        assert!(report.findings.iter().any(|finding| matches!(
            finding,
            ContractFinding::MissingRequiredEvaluation { .. }
        )));
    }

    #[test]
    fn predicate_substitution_is_invalid() {
        let frozen = contract().freeze().unwrap();
        let mut evaluation = pass_eval(&frozen.contract().obligations[0]);
        evaluation.predicate_sha256 = sha("substituted");
        assert_eq!(
            evaluate(&frozen, &energy_units(), &[evaluation]).closure,
            ContractClosure::Invalid
        );
    }

    #[test]
    fn required_failure_violates_contract() {
        let frozen = contract().freeze().unwrap();
        let mut evaluation = pass_eval(&frozen.contract().obligations[0]);
        evaluation.outcome = ObligationOutcome::Fail;
        assert_eq!(
            evaluate(&frozen, &energy_units(), &[evaluation]).closure,
            ContractClosure::Violated
        );
    }

    #[test]
    fn observed_model_identity_substitution_is_invalid() {
        let frozen = contract().freeze().unwrap();
        let report = evaluate_physical_model(
            &frozen,
            &energy_expr(),
            &energy_units(),
            &sha("different-model"),
            &sha("expression"),
            &sha("units"),
            &[pass_eval(&frozen.contract().obligations[0])],
        );
        assert_eq!(report.closure, ContractClosure::Invalid);
        assert_eq!(
            report.dimension_assessment,
            DimensionAssessment::IdentityMismatch
        );
    }

    #[test]
    fn contract_identity_is_obligation_order_independent() {
        let mut a = contract();
        a.obligations
            .push(obligation("POSITIVITY", ObligationKind::Positivity, false));
        let mut b = a.clone();
        b.obligations.reverse();
        assert_eq!(
            a.freeze().unwrap().contract_sha256(),
            b.freeze().unwrap().contract_sha256()
        );
    }

    #[test]
    fn forbidden_not_applicable_is_invalid() {
        let frozen = contract().freeze().unwrap();
        let spec = &frozen.contract().obligations[0];
        let evaluation = ObligationEvaluation {
            obligation_id: spec.obligation_id.clone(),
            predicate_sha256: spec.predicate_sha256.clone(),
            evaluator_sha256: spec.evaluator_sha256.clone(),
            outcome: ObligationOutcome::NotApplicable,
            result_artifact_sha256: None,
            justification_sha256: Some(sha("why-na")),
        };
        assert_eq!(
            evaluate(&frozen, &energy_units(), &[evaluation]).closure,
            ContractClosure::Invalid
        );
    }
}
