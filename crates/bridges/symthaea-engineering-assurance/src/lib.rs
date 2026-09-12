// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Plan-bound Engineering Trust Kernel assurance composition.
//!
//! ```text
//! semantic evidence plan
//! != plan-bound admitted evidence
//! != plan-bound discharge receipt
//! != current plan-bound discharge fact
//! != complete requirement-verification contract
//! != current requirement satisfaction
//! != qualified design / certification / manufacturing / actuation authority
//! ```
//!
//! V2 preserves ETK V1 admission/currentness as a lower defense-in-depth
//! theorem, but never hashes the float-sensitive V1 capability identity into a
//! V2 authority identity. Every V2 engineering float uses Canonical Binary64,
//! and every authority capability retains enough immutable witness material to
//! reconstruct its own identity preimage for audit.

#![deny(unsafe_code)]

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use symthaea_engineering_evidence_plan::{
    AcceptedRequirementRevisionIdV1, AcceptedRequirementRevisionV1, CurrentnessAssertionIdV1,
    CurrentnessAssertionV1, MetricOperatorV1, ObligationRevisionIdV1, Sha256DigestV1,
    SimulationEvidencePlanV1, SimulationEvidencePolicyRevisionV1, SimulationRequestRevisionV1,
    SubjectRevisionIdV1, SubjectRevisionV1, TwinRevisionIdV1, TwinRevisionV1,
    ValidityDomainRevisionIdV1, ValidityDomainRevisionV1, WarningPolicyV1,
};
use symthaea_engineering_requirement_binding::{
    RequirementObligationBindingIdV1, RequirementObligationBindingV1,
};
use symthaea_engineering_trust::{
    AdmissionDenialReasonV1, AdmittedSimulationEvidenceV1, DischargeContextV1,
    EvidenceCurrentnessV1, MetricAcceptancePolicyV1, ObligationDischargeReceiptV1,
    SimulationAdmissionPolicyV1, SimulationCandidateBindingV1, ThresholdOperatorV1,
    admit_simulation_evidence_v1, is_obligation_discharged_v1,
    issue_obligation_discharge_receipt_v1,
};
use symthaea_formal_safety::ProofObligation;
use symthaea_sim_bridge::{
    EngineeringDomain, ExecutionMode, SimulationRequest, SimulationResult, SolverKind,
    UncertaintyEstimate,
};
use thiserror::Error;

const REQUEST_SCHEMA_V2: &str = "symthaea.etk-simulation-request.v2";
const REQUEST_DOMAIN_V2: &[u8] = b"symthaea.etk-simulation-request.v2\0";
const POLICY_SCHEMA_V2: &str = "symthaea.etk-simulation-evidence-policy.v2";
const POLICY_DOMAIN_V2: &[u8] = b"symthaea.etk-simulation-evidence-policy.v2\0";
const PLAN_SCHEMA_V2: &str = "symthaea.etk-simulation-evidence-plan.v2";
const PLAN_DOMAIN_V2: &[u8] = b"symthaea.etk-simulation-evidence-plan.v2\0";
const ADMITTED_SCHEMA_V2: &str = "symthaea.etk-plan-bound-admitted-simulation-evidence.v2";
const ADMITTED_DOMAIN_V2: &[u8] = b"symthaea.etk-plan-bound-admitted-simulation-evidence.v2\0";
const RECEIPT_SCHEMA_V2: &str = "symthaea.etk-plan-bound-obligation-discharge-receipt.v2";
const RECEIPT_DOMAIN_V2: &[u8] = b"symthaea.etk-plan-bound-obligation-discharge-receipt.v2\0";
const FACT_SCHEMA_V2: &str = "symthaea.etk-current-plan-bound-obligation-discharge-fact.v2";
const FACT_DOMAIN_V2: &[u8] = b"symthaea.etk-current-plan-bound-obligation-discharge-fact.v2\0";
const CONTRACT_SCHEMA_V2: &str = "symthaea.etk-requirement-verification-contract.v2";
const CONTRACT_DOMAIN_V2: &[u8] = b"symthaea.etk-requirement-verification-contract.v2\0";
const SATISFACTION_SCHEMA_V2: &str = "symthaea.etk-requirement-satisfaction-receipt.v2";
const SATISFACTION_DOMAIN_V2: &[u8] = b"symthaea.etk-requirement-satisfaction-receipt.v2\0";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AssuranceErrorV2 {
    #[error("{0} cannot be empty or have leading/trailing whitespace")]
    InvalidText(&'static str),
    #[error("{0} is not a canonical SHA-256 identity")]
    InvalidDigest(&'static str),
    #[error("non-finite binary64 in {0}")]
    NonFinite(&'static str),
    #[error("invalid uncertainty in {0}")]
    InvalidUncertainty(&'static str),
    #[error("duplicate simulation parameter {0}")]
    DuplicateParameter(String),
    #[error("duplicate requested metric {0}")]
    DuplicateRequestedMetric(String),
    #[error("duplicate result metric {0}")]
    DuplicateResultMetric(String),
    #[error("duplicate solver warning {0}")]
    DuplicateWarning(String),
    #[error("V1 semantic evidence-plan validation failed: {0}")]
    SemanticPlanValidation(String),
    #[error("the supplied proof obligation does not match the semantic plan")]
    PlanObligationMismatch,
    #[error("solver result request/input binding does not match the semantic plan")]
    PlanResultBindingMismatch,
    #[error("lower ETK admission denied: {0:?}")]
    LowerAdmissionDenied(Vec<AdmissionDenialReasonV1>),
    #[error("solver warning denied by the semantic evidence plan")]
    WarningDenied,
    #[error("solver warning requires review under the semantic evidence plan")]
    WarningReviewRequired,
    #[error("solver warning is not in the exact plan allowlist: {0}")]
    WarningNotAllowed(String),
    #[error("lower discharge receipt issuance failed: {0}")]
    LowerReceiptIssuance(String),
    #[error("plan-bound admitted evidence targets a different semantic plan")]
    PlanBoundAdmissionMismatch,
    #[error("plan-bound discharge receipt targets a historical/different semantic plan")]
    HistoricalPlan,
    #[error("lower ETK receipt no longer applies to this exact semantic plan")]
    LowerReceiptNotCurrent,
    #[error("requirement/obligation relationship targets a different accepted requirement")]
    RelationshipRequirementMismatch,
    #[error("relationship obligation and evidence-plan obligation differ")]
    RelationshipPlanObligationMismatch,
    #[error("an AllOf verification contract requires at least one verification member")]
    EmptyAllOf,
    #[error("the same exact evidence plan is counted more than once in one AllOf contract")]
    DuplicateEvidencePlan,
    #[error("current twin does not belong to the supplied current subject")]
    CurrentTwinSubjectMismatch,
}

macro_rules! authority_id {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(Sha256DigestV1);

        impl $name {
            fn from_digest(digest: Sha256DigestV1) -> Self {
                Self(digest)
            }

            pub fn as_str(&self) -> &str {
                self.0.as_str()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

macro_rules! premise_id {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(Sha256DigestV1);

        impl $name {
            pub fn parse(value: impl Into<String>) -> Result<Self, AssuranceErrorV2> {
                Sha256DigestV1::parse(value.into())
                    .map(Self)
                    .map_err(|_| AssuranceErrorV2::InvalidDigest(stringify!($name)))
            }

            pub fn as_str(&self) -> &str {
                self.0.as_str()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

authority_id!(SimulationRequestRevisionIdV2);
authority_id!(EvidencePolicyRevisionIdV2);
authority_id!(EvidencePlanIdV2);
authority_id!(PlanBoundAdmittedEvidenceIdV2);
authority_id!(PlanBoundDischargeReceiptIdV2);
authority_id!(CurrentPlanBoundDischargeFactIdV2);
authority_id!(RequirementVerificationContractIdV2);
authority_id!(RequirementSatisfactionReceiptIdV2);
premise_id!(RequirementDecompositionPolicyRevisionDigestV2);
premise_id!(RequirementDecompositionAcceptanceRecordDigestV2);
premise_id!(RequirementCurrentnessAssertionIdV2);

/// Cross-language identity encoding for one finite IEEE-754 binary64 value.
/// Signed zero is intentionally normalized because it is not an ETK V2
/// engineering semantic distinction.
pub fn canonical_binary64_v2(value: f64) -> Result<String, AssuranceErrorV2> {
    if !value.is_finite() {
        return Err(AssuranceErrorV2::NonFinite("canonical binary64"));
    }
    let normalized = if value == 0.0 { 0.0 } else { value };
    Ok(format!("f64:{:016x}", normalized.to_bits()))
}

/// Exact semantic plan under which one solver result may later be admitted.
#[must_use = "an evidence plan is binding structure, not admitted evidence"]
#[derive(Debug, Clone, PartialEq)]
pub struct PlanBoundSimulationEvidencePlanV2 {
    plan_id: EvidencePlanIdV2,
    subject_revision_id: SubjectRevisionIdV1,
    twin_revision_id: TwinRevisionIdV1,
    requirement_revision_id: AcceptedRequirementRevisionIdV1,
    obligation_id: String,
    obligation_revision_id: ObligationRevisionIdV1,
    logical_request_id: String,
    request_revision_id: SimulationRequestRevisionIdV2,
    evidence_policy_revision_id: EvidencePolicyRevisionIdV2,
    validity_domain_revision_id: ValidityDomainRevisionIdV1,
    currentness_assertion_id: CurrentnessAssertionIdV1,
    expected_rendered_input_digest: Sha256DigestV1,
    metric_name: String,
    metric_unit: String,
    metric_operator: MetricOperatorV1,
    metric_threshold: f64,
    max_epistemic: f64,
    max_aleatoric: f64,
    warning_policy: WarningPolicyV1,
}

impl PlanBoundSimulationEvidencePlanV2 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &SubjectRevisionV1,
        twin: &TwinRevisionV1,
        requirement: &AcceptedRequirementRevisionV1,
        obligation: &ProofObligation,
        raw_request: &SimulationRequest,
        evidence_policy: &SimulationEvidencePolicyRevisionV1,
        validity_domain: &ValidityDomainRevisionV1,
        currentness: &CurrentnessAssertionV1,
        expected_rendered_input_digest: Sha256DigestV1,
    ) -> Result<Self, AssuranceErrorV2> {
        // V1 remains an independent semantic validator, but its float-sensitive
        // identities are deliberately excluded from the V2 authority preimage.
        let request_v1 = SimulationRequestRevisionV1::from_request(raw_request)
            .map_err(|error| AssuranceErrorV2::SemanticPlanValidation(error.to_string()))?;
        SimulationEvidencePlanV1::new(
            subject,
            twin,
            requirement,
            obligation,
            &request_v1,
            evidence_policy,
            validity_domain,
            currentness,
            expected_rendered_input_digest.clone(),
        )
        .map_err(|error| AssuranceErrorV2::SemanticPlanValidation(error.to_string()))?;

        let request_revision_id = SimulationRequestRevisionIdV2::from_digest(domain_hash(
            REQUEST_DOMAIN_V2,
            &request_value_v2(raw_request)?,
        ));
        let evidence_policy_revision_id = EvidencePolicyRevisionIdV2::from_digest(domain_hash(
            POLICY_DOMAIN_V2,
            &policy_value_v2(evidence_policy)?,
        ));
        let obligation_revision_id = ObligationRevisionIdV1::for_obligation(obligation)
            .map_err(|_| AssuranceErrorV2::InvalidDigest("obligation revision"))?;
        let obligation_id = obligation.id.to_string();
        let preimage = json!({
            "currentness_assertion_id": currentness.assertion_id().as_str(),
            "evidence_policy_revision_id": evidence_policy_revision_id.as_str(),
            "expected_rendered_input_digest": expected_rendered_input_digest.as_str(),
            "obligation_id": obligation_id,
            "obligation_revision_id": obligation_revision_id.as_str(),
            "request_id": raw_request.id,
            "request_revision_id": request_revision_id.as_str(),
            "requirement_revision_id": requirement.revision_id().as_str(),
            "schema": PLAN_SCHEMA_V2,
            "subject_revision_id": subject.revision_id().as_str(),
            "twin_revision_id": twin.revision_id().as_str(),
            "validity_domain_revision_id": validity_domain.revision_id().as_str(),
        });
        let plan_id = EvidencePlanIdV2::from_digest(domain_hash(PLAN_DOMAIN_V2, &preimage));
        let metric = evidence_policy.required_metric();

        Ok(Self {
            plan_id,
            subject_revision_id: subject.revision_id().clone(),
            twin_revision_id: twin.revision_id().clone(),
            requirement_revision_id: requirement.revision_id().clone(),
            obligation_id,
            obligation_revision_id,
            logical_request_id: raw_request.id.clone(),
            request_revision_id,
            evidence_policy_revision_id,
            validity_domain_revision_id: validity_domain.revision_id().clone(),
            currentness_assertion_id: currentness.assertion_id().clone(),
            expected_rendered_input_digest,
            metric_name: metric.name().to_string(),
            metric_unit: metric.unit().to_string(),
            metric_operator: metric.operator(),
            metric_threshold: metric.threshold(),
            max_epistemic: metric.max_epistemic(),
            max_aleatoric: metric.max_aleatoric(),
            warning_policy: evidence_policy.warning_policy().clone(),
        })
    }

    pub fn plan_id(&self) -> &EvidencePlanIdV2 {
        &self.plan_id
    }
    pub fn subject_revision_id(&self) -> &SubjectRevisionIdV1 {
        &self.subject_revision_id
    }
    pub fn twin_revision_id(&self) -> &TwinRevisionIdV1 {
        &self.twin_revision_id
    }
    pub fn requirement_revision_id(&self) -> &AcceptedRequirementRevisionIdV1 {
        &self.requirement_revision_id
    }
    pub fn obligation_id(&self) -> &str {
        &self.obligation_id
    }
    pub fn obligation_revision_id(&self) -> &ObligationRevisionIdV1 {
        &self.obligation_revision_id
    }
    pub fn logical_request_id(&self) -> &str {
        &self.logical_request_id
    }
    pub fn request_revision_id(&self) -> &SimulationRequestRevisionIdV2 {
        &self.request_revision_id
    }
    pub fn evidence_policy_revision_id(&self) -> &EvidencePolicyRevisionIdV2 {
        &self.evidence_policy_revision_id
    }
    pub fn validity_domain_revision_id(&self) -> &ValidityDomainRevisionIdV1 {
        &self.validity_domain_revision_id
    }
    pub fn currentness_assertion_id(&self) -> &CurrentnessAssertionIdV1 {
        &self.currentness_assertion_id
    }
    pub fn expected_rendered_input_digest(&self) -> &Sha256DigestV1 {
        &self.expected_rendered_input_digest
    }

    pub fn audit_record_v2(&self) -> Value {
        json!({
            "authority": "binding-plan-only",
            "currentness_assertion_id": self.currentness_assertion_id.as_str(),
            "evidence_plan_id": self.plan_id.as_str(),
            "evidence_policy_revision_id": self.evidence_policy_revision_id.as_str(),
            "expected_rendered_input_digest": self.expected_rendered_input_digest.as_str(),
            "obligation_id": self.obligation_id,
            "obligation_revision_id": self.obligation_revision_id.as_str(),
            "request_id": self.logical_request_id,
            "request_revision_id": self.request_revision_id.as_str(),
            "requirement_revision_id": self.requirement_revision_id.as_str(),
            "subject_revision_id": self.subject_revision_id.as_str(),
            "twin_revision_id": self.twin_revision_id.as_str(),
            "validity_domain_revision_id": self.validity_domain_revision_id.as_str(),
        })
    }
}

/// Authority-bearing proof that one exact solver result was admitted under one
/// exact semantic evidence plan. The normalized V2 preimage is retained solely
/// for deterministic audit/recomputation; there is no data -> capability parser.
#[must_use = "plan-bound admitted evidence is not a discharge receipt"]
#[derive(Debug, Clone, PartialEq)]
pub struct PlanBoundAdmittedSimulationEvidenceV2 {
    admitted_evidence_id: PlanBoundAdmittedEvidenceIdV2,
    plan_id: EvidencePlanIdV2,
    output_digest: Sha256DigestV1,
    normalized_identity: Value,
    lower_admitted: AdmittedSimulationEvidenceV1,
}

impl PlanBoundAdmittedSimulationEvidenceV2 {
    pub fn admitted_evidence_id(&self) -> &PlanBoundAdmittedEvidenceIdV2 {
        &self.admitted_evidence_id
    }
    pub fn plan_id(&self) -> &EvidencePlanIdV2 {
        &self.plan_id
    }
    pub fn output_digest(&self) -> &Sha256DigestV1 {
        &self.output_digest
    }

    pub fn audit_record_v2(&self) -> Value {
        json!({
            "authority": "plan-bound-admitted-evidence-only",
            "admitted_evidence_id": self.admitted_evidence_id.as_str(),
            "identity_preimage": self.normalized_identity.clone(),
            "plan_id": self.plan_id.as_str(),
            "solver_output_digest": self.output_digest.as_str(),
        })
    }
}

pub fn admit_plan_bound_simulation_v2(
    plan: &PlanBoundSimulationEvidencePlanV2,
    obligation: &ProofObligation,
    result: &SimulationResult,
    candidate_artifact_id: impl Into<String>,
    source_lineage_id: impl Into<String>,
) -> Result<PlanBoundAdmittedSimulationEvidenceV2, AssuranceErrorV2> {
    let obligation_revision = ObligationRevisionIdV1::for_obligation(obligation)
        .map_err(|_| AssuranceErrorV2::InvalidDigest("obligation revision"))?;
    if obligation.id.to_string() != plan.obligation_id
        || obligation_revision != plan.obligation_revision_id
    {
        return Err(AssuranceErrorV2::PlanObligationMismatch);
    }

    if result.request_id != plan.logical_request_id {
        return Err(AssuranceErrorV2::PlanResultBindingMismatch);
    }
    let input_digest = required_sha256_option(&result.evidence.input_digest, "solver input digest")?;
    if input_digest != plan.expected_rendered_input_digest {
        return Err(AssuranceErrorV2::PlanResultBindingMismatch);
    }
    let output_digest = required_sha256_option(&result.evidence.output_digest, "solver output digest")?;
    let warnings = canonicalize_warnings(&result.warnings)?;
    enforce_warning_policy(&plan.warning_policy, &warnings)?;

    let candidate_artifact_id = canonical_text(candidate_artifact_id.into(), "candidate artifact id")?;
    let source_lineage_id = canonical_text(source_lineage_id.into(), "source lineage id")?;
    let lower_metric = MetricAcceptancePolicyV1::new(
        plan.metric_name.clone(),
        plan.metric_unit.clone(),
        lower_operator(plan.metric_operator),
        plan.metric_threshold,
        plan.max_epistemic,
        plan.max_aleatoric,
    )
    .map_err(|error| AssuranceErrorV2::SemanticPlanValidation(error.to_string()))?;
    let lower_policy = SimulationAdmissionPolicyV1::for_obligation(
        obligation,
        plan.subject_revision_id.as_str(),
        plan.twin_revision_id.as_str(),
        plan.requirement_revision_id.as_str(),
        plan.evidence_policy_revision_id.as_str(),
        plan.logical_request_id.clone(),
        plan.validity_domain_revision_id.as_str(),
        lower_metric,
        plan.expected_rendered_input_digest.as_str(),
    )
    .map_err(|error| AssuranceErrorV2::SemanticPlanValidation(error.to_string()))?;
    let lower_binding = SimulationCandidateBindingV1::for_policy(
        &lower_policy,
        candidate_artifact_id.clone(),
        EvidenceCurrentnessV1::Current,
        plan.currentness_assertion_id.as_str(),
        source_lineage_id.clone(),
    )
    .map_err(|error| AssuranceErrorV2::SemanticPlanValidation(error.to_string()))?;
    let lower_admitted = admit_simulation_evidence_v1(&lower_policy, &lower_binding, result)
        .map_err(|denial| AssuranceErrorV2::LowerAdmissionDenied(denial.reasons().to_vec()))?;

    let normalized_identity = normalized_plan_bound_admission_v2(
        plan,
        result,
        &candidate_artifact_id,
        &source_lineage_id,
        &input_digest,
        &output_digest,
        &warnings,
    )?;
    let admitted_evidence_id = PlanBoundAdmittedEvidenceIdV2::from_digest(domain_hash(
        ADMITTED_DOMAIN_V2,
        &normalized_identity,
    ));

    Ok(PlanBoundAdmittedSimulationEvidenceV2 {
        admitted_evidence_id,
        plan_id: plan.plan_id.clone(),
        output_digest,
        normalized_identity,
        lower_admitted,
    })
}

/// Immutable historical receipt for one exact plan-bound admission.
#[must_use = "a discharge receipt is historical evidence, not present requirement satisfaction"]
#[derive(Debug, Clone, PartialEq)]
pub struct PlanBoundObligationDischargeReceiptV2 {
    receipt_id: PlanBoundDischargeReceiptIdV2,
    plan_id: EvidencePlanIdV2,
    obligation_id: String,
    obligation_revision_id: ObligationRevisionIdV1,
    plan_bound_admitted_evidence_id: PlanBoundAdmittedEvidenceIdV2,
    lower_receipt: ObligationDischargeReceiptV1,
}

impl PlanBoundObligationDischargeReceiptV2 {
    pub fn receipt_id(&self) -> &PlanBoundDischargeReceiptIdV2 {
        &self.receipt_id
    }
    pub fn plan_id(&self) -> &EvidencePlanIdV2 {
        &self.plan_id
    }
    pub fn obligation_revision_id(&self) -> &ObligationRevisionIdV1 {
        &self.obligation_revision_id
    }
    pub fn admitted_evidence_id(&self) -> &PlanBoundAdmittedEvidenceIdV2 {
        &self.plan_bound_admitted_evidence_id
    }

    pub fn audit_record_v2(&self) -> Value {
        json!({
            "authority": "historical-plan-bound-discharge-only",
            "obligation_id": self.obligation_id,
            "obligation_revision_id": self.obligation_revision_id.as_str(),
            "plan_bound_admitted_evidence_id": self.plan_bound_admitted_evidence_id.as_str(),
            "plan_bound_discharge_receipt_id": self.receipt_id.as_str(),
            "plan_id": self.plan_id.as_str(),
        })
    }
}

pub fn issue_plan_bound_discharge_receipt_v2(
    plan: &PlanBoundSimulationEvidencePlanV2,
    obligation: &ProofObligation,
    admitted: &PlanBoundAdmittedSimulationEvidenceV2,
) -> Result<PlanBoundObligationDischargeReceiptV2, AssuranceErrorV2> {
    let obligation_revision_id = ObligationRevisionIdV1::for_obligation(obligation)
        .map_err(|_| AssuranceErrorV2::InvalidDigest("obligation revision"))?;
    if admitted.plan_id != plan.plan_id
        || obligation.id.to_string() != plan.obligation_id
        || obligation_revision_id != plan.obligation_revision_id
    {
        return Err(AssuranceErrorV2::PlanBoundAdmissionMismatch);
    }

    let lower_receipt = issue_obligation_discharge_receipt_v1(obligation, &admitted.lower_admitted)
        .map_err(|error| AssuranceErrorV2::LowerReceiptIssuance(error.to_string()))?;
    let preimage = json!({
        "obligation_id": plan.obligation_id,
        "obligation_revision_id": plan.obligation_revision_id.as_str(),
        "plan_bound_admitted_evidence_id": admitted.admitted_evidence_id.as_str(),
        "plan_id": plan.plan_id.as_str(),
        "schema": RECEIPT_SCHEMA_V2,
    });
    let receipt_id = PlanBoundDischargeReceiptIdV2::from_digest(domain_hash(
        RECEIPT_DOMAIN_V2,
        &preimage,
    ));

    Ok(PlanBoundObligationDischargeReceiptV2 {
        receipt_id,
        plan_id: plan.plan_id.clone(),
        obligation_id: plan.obligation_id.clone(),
        obligation_revision_id,
        plan_bound_admitted_evidence_id: admitted.admitted_evidence_id.clone(),
        lower_receipt,
    })
}

/// Present-tense fact for one exact semantic plan, retaining its historical
/// receipt witness and all identity components required to audit the preimage.
#[must_use = "a current plan-bound discharge fact is not requirement satisfaction"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentPlanBoundDischargeFactV2 {
    fact_id: CurrentPlanBoundDischargeFactIdV2,
    plan_id: EvidencePlanIdV2,
    obligation_id: String,
    obligation_revision_id: ObligationRevisionIdV1,
    subject_revision_id: SubjectRevisionIdV1,
    twin_revision_id: TwinRevisionIdV1,
    requirement_revision_id: AcceptedRequirementRevisionIdV1,
    request_revision_id: SimulationRequestRevisionIdV2,
    evidence_policy_revision_id: EvidencePolicyRevisionIdV2,
    validity_domain_revision_id: ValidityDomainRevisionIdV1,
    currentness_assertion_id: CurrentnessAssertionIdV1,
    witness_receipt_id: PlanBoundDischargeReceiptIdV2,
}

impl CurrentPlanBoundDischargeFactV2 {
    pub fn fact_id(&self) -> &CurrentPlanBoundDischargeFactIdV2 {
        &self.fact_id
    }
    pub fn plan_id(&self) -> &EvidencePlanIdV2 {
        &self.plan_id
    }
    pub fn obligation_revision_id(&self) -> &ObligationRevisionIdV1 {
        &self.obligation_revision_id
    }
    pub fn subject_revision_id(&self) -> &SubjectRevisionIdV1 {
        &self.subject_revision_id
    }
    pub fn twin_revision_id(&self) -> &TwinRevisionIdV1 {
        &self.twin_revision_id
    }
    pub fn requirement_revision_id(&self) -> &AcceptedRequirementRevisionIdV1 {
        &self.requirement_revision_id
    }
    pub fn witness_receipt_id(&self) -> &PlanBoundDischargeReceiptIdV2 {
        &self.witness_receipt_id
    }

    pub fn audit_record_v2(&self) -> Value {
        json!({
            "authority": "current-plan-bound-discharge-only",
            "current_discharge_fact_id": self.fact_id.as_str(),
            "currentness_assertion_id": self.currentness_assertion_id.as_str(),
            "evidence_plan_id": self.plan_id.as_str(),
            "evidence_policy_revision_id": self.evidence_policy_revision_id.as_str(),
            "obligation_id": self.obligation_id,
            "obligation_revision_id": self.obligation_revision_id.as_str(),
            "plan_bound_discharge_receipt_id": self.witness_receipt_id.as_str(),
            "request_revision_id": self.request_revision_id.as_str(),
            "requirement_revision_id": self.requirement_revision_id.as_str(),
            "subject_revision_id": self.subject_revision_id.as_str(),
            "twin_revision_id": self.twin_revision_id.as_str(),
            "validity_domain_revision_id": self.validity_domain_revision_id.as_str(),
        })
    }
}

pub fn derive_current_plan_bound_discharge_fact_v2(
    current_plan: &PlanBoundSimulationEvidencePlanV2,
    obligation: &ProofObligation,
    receipt: &PlanBoundObligationDischargeReceiptV2,
) -> Result<CurrentPlanBoundDischargeFactV2, AssuranceErrorV2> {
    let obligation_revision_id = ObligationRevisionIdV1::for_obligation(obligation)
        .map_err(|_| AssuranceErrorV2::InvalidDigest("obligation revision"))?;
    if receipt.plan_id != current_plan.plan_id
        || receipt.obligation_id != current_plan.obligation_id
        || receipt.obligation_revision_id != current_plan.obligation_revision_id
        || obligation_revision_id != current_plan.obligation_revision_id
    {
        return Err(AssuranceErrorV2::HistoricalPlan);
    }

    let context = DischargeContextV1::new(
        current_plan.subject_revision_id.as_str(),
        current_plan.twin_revision_id.as_str(),
        current_plan.requirement_revision_id.as_str(),
        current_plan.validity_domain_revision_id.as_str(),
        current_plan.currentness_assertion_id.as_str(),
    )
    .map_err(|error| AssuranceErrorV2::SemanticPlanValidation(error.to_string()))?;
    if !is_obligation_discharged_v1(
        obligation,
        &context,
        std::slice::from_ref(&receipt.lower_receipt),
    ) {
        return Err(AssuranceErrorV2::LowerReceiptNotCurrent);
    }

    let preimage = json!({
        "currentness_assertion_id": current_plan.currentness_assertion_id.as_str(),
        "evidence_plan_id": current_plan.plan_id.as_str(),
        "evidence_policy_revision_id": current_plan.evidence_policy_revision_id.as_str(),
        "obligation_id": current_plan.obligation_id,
        "obligation_revision_id": current_plan.obligation_revision_id.as_str(),
        "plan_bound_discharge_receipt_id": receipt.receipt_id.as_str(),
        "request_revision_id": current_plan.request_revision_id.as_str(),
        "requirement_revision_id": current_plan.requirement_revision_id.as_str(),
        "schema": FACT_SCHEMA_V2,
        "subject_revision_id": current_plan.subject_revision_id.as_str(),
        "twin_revision_id": current_plan.twin_revision_id.as_str(),
        "validity_domain_revision_id": current_plan.validity_domain_revision_id.as_str(),
    });
    let fact_id = CurrentPlanBoundDischargeFactIdV2::from_digest(domain_hash(
        FACT_DOMAIN_V2,
        &preimage,
    ));

    Ok(CurrentPlanBoundDischargeFactV2 {
        fact_id,
        plan_id: current_plan.plan_id.clone(),
        obligation_id: current_plan.obligation_id.clone(),
        obligation_revision_id,
        subject_revision_id: current_plan.subject_revision_id.clone(),
        twin_revision_id: current_plan.twin_revision_id.clone(),
        requirement_revision_id: current_plan.requirement_revision_id.clone(),
        request_revision_id: current_plan.request_revision_id.clone(),
        evidence_policy_revision_id: current_plan.evidence_policy_revision_id.clone(),
        validity_domain_revision_id: current_plan.validity_domain_revision_id.clone(),
        currentness_assertion_id: current_plan.currentness_assertion_id.clone(),
        witness_receipt_id: receipt.receipt_id.clone(),
    })
}

/// One required `(traceability relationship, exact evidence plan)` member.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequirementVerificationMemberV2 {
    relationship_id: RequirementObligationBindingIdV1,
    requirement_revision_id: AcceptedRequirementRevisionIdV1,
    obligation_revision_id: ObligationRevisionIdV1,
    evidence_plan_id: EvidencePlanIdV2,
}

impl RequirementVerificationMemberV2 {
    pub fn new(
        binding: &RequirementObligationBindingV1,
        plan: &PlanBoundSimulationEvidencePlanV2,
    ) -> Result<Self, AssuranceErrorV2> {
        if binding.requirement_revision_id() != &plan.requirement_revision_id {
            return Err(AssuranceErrorV2::RelationshipRequirementMismatch);
        }
        if binding.obligation_revision_id() != &plan.obligation_revision_id {
            return Err(AssuranceErrorV2::RelationshipPlanObligationMismatch);
        }
        Ok(Self {
            relationship_id: binding.binding_id().clone(),
            requirement_revision_id: binding.requirement_revision_id().clone(),
            obligation_revision_id: binding.obligation_revision_id().clone(),
            evidence_plan_id: plan.plan_id.clone(),
        })
    }

    pub fn evidence_plan_id(&self) -> &EvidencePlanIdV2 {
        &self.evidence_plan_id
    }
    pub fn obligation_revision_id(&self) -> &ObligationRevisionIdV1 {
        &self.obligation_revision_id
    }

    fn as_value(&self) -> Value {
        json!({
            "evidence_plan_id": self.evidence_plan_id.as_str(),
            "obligation_revision_id": self.obligation_revision_id.as_str(),
            "relationship_id": self.relationship_id.as_str(),
            "requirement_revision_id": self.requirement_revision_id.as_str(),
        })
    }
}

/// Conservative V2 completeness contract. `AllOf` is the only composition
/// mode; the same relationship may appear under several distinct plans.
#[must_use = "a verification contract is completeness structure, not satisfaction"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequirementVerificationContractV2 {
    contract_id: RequirementVerificationContractIdV2,
    requirement_revision_id: AcceptedRequirementRevisionIdV1,
    members: Vec<RequirementVerificationMemberV2>,
    decomposition_policy_revision_id: RequirementDecompositionPolicyRevisionDigestV2,
    decomposition_acceptance_record_digest: RequirementDecompositionAcceptanceRecordDigestV2,
}

impl RequirementVerificationContractV2 {
    pub fn all_of(
        requirement: &AcceptedRequirementRevisionV1,
        mut members: Vec<RequirementVerificationMemberV2>,
        decomposition_policy_revision_id: RequirementDecompositionPolicyRevisionDigestV2,
        decomposition_acceptance_record_digest: RequirementDecompositionAcceptanceRecordDigestV2,
    ) -> Result<Self, AssuranceErrorV2> {
        if members.is_empty() {
            return Err(AssuranceErrorV2::EmptyAllOf);
        }

        let mut plans = BTreeSet::new();
        for member in &members {
            if member.requirement_revision_id != *requirement.revision_id() {
                return Err(AssuranceErrorV2::RelationshipRequirementMismatch);
            }
            if !plans.insert(member.evidence_plan_id.as_str().to_string()) {
                return Err(AssuranceErrorV2::DuplicateEvidencePlan);
            }
        }
        members.sort_by(|left, right| {
            left.evidence_plan_id
                .as_str()
                .cmp(right.evidence_plan_id.as_str())
                .then_with(|| left.relationship_id.as_str().cmp(right.relationship_id.as_str()))
                .then_with(|| {
                    left.obligation_revision_id
                        .as_str()
                        .cmp(right.obligation_revision_id.as_str())
                })
        });
        let member_values = members
            .iter()
            .map(RequirementVerificationMemberV2::as_value)
            .collect::<Vec<_>>();
        let preimage = json!({
            "composition": "AllOf",
            "decomposition_acceptance_record_digest": decomposition_acceptance_record_digest.as_str(),
            "decomposition_policy_revision_id": decomposition_policy_revision_id.as_str(),
            "members": member_values,
            "requirement_revision_id": requirement.revision_id().as_str(),
            "schema": CONTRACT_SCHEMA_V2,
        });
        let contract_id = RequirementVerificationContractIdV2::from_digest(domain_hash(
            CONTRACT_DOMAIN_V2,
            &preimage,
        ));

        Ok(Self {
            contract_id,
            requirement_revision_id: requirement.revision_id().clone(),
            members,
            decomposition_policy_revision_id,
            decomposition_acceptance_record_digest,
        })
    }

    pub fn contract_id(&self) -> &RequirementVerificationContractIdV2 {
        &self.contract_id
    }
    pub fn requirement_revision_id(&self) -> &AcceptedRequirementRevisionIdV1 {
        &self.requirement_revision_id
    }
    pub fn members(&self) -> &[RequirementVerificationMemberV2] {
        &self.members
    }

    pub fn audit_record_v2(&self) -> Value {
        json!({
            "authority": "verification-completeness-contract-only",
            "composition": "AllOf",
            "decomposition_acceptance_record_digest": self.decomposition_acceptance_record_digest.as_str(),
            "decomposition_policy_revision_id": self.decomposition_policy_revision_id.as_str(),
            "members": self.members.iter().map(RequirementVerificationMemberV2::as_value).collect::<Vec<_>>(),
            "requirement_revision_id": self.requirement_revision_id.as_str(),
            "verification_contract_id": self.contract_id.as_str(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequirementUnsatisfiedV2 {
    missing_evidence_plan_ids: Vec<EvidencePlanIdV2>,
}

impl RequirementUnsatisfiedV2 {
    pub fn missing_evidence_plan_ids(&self) -> &[EvidencePlanIdV2] {
        &self.missing_evidence_plan_ids
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UsedCurrentFactV2 {
    current_discharge_fact_id: CurrentPlanBoundDischargeFactIdV2,
    evidence_plan_id: EvidencePlanIdV2,
    obligation_revision_id: ObligationRevisionIdV1,
}

impl UsedCurrentFactV2 {
    fn as_value(&self) -> Value {
        json!({
            "current_discharge_fact_id": self.current_discharge_fact_id.as_str(),
            "evidence_plan_id": self.evidence_plan_id.as_str(),
            "obligation_revision_id": self.obligation_revision_id.as_str(),
        })
    }
}

/// Present requirement-satisfaction capability. The exact higher currentness
/// assertion and exact fact witnesses are retained for audit/recomputation.
#[must_use = "requirement satisfaction is not design qualification"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequirementSatisfactionReceiptV2 {
    receipt_id: RequirementSatisfactionReceiptIdV2,
    verification_contract_id: RequirementVerificationContractIdV2,
    requirement_revision_id: AcceptedRequirementRevisionIdV1,
    current_subject_revision_id: SubjectRevisionIdV1,
    current_twin_revision_id: TwinRevisionIdV1,
    currentness_assertion_id: RequirementCurrentnessAssertionIdV2,
    used_facts: Vec<UsedCurrentFactV2>,
}

impl RequirementSatisfactionReceiptV2 {
    pub fn receipt_id(&self) -> &RequirementSatisfactionReceiptIdV2 {
        &self.receipt_id
    }
    pub fn verification_contract_id(&self) -> &RequirementVerificationContractIdV2 {
        &self.verification_contract_id
    }

    pub fn audit_record_v2(&self) -> Value {
        json!({
            "authority": "current-requirement-satisfaction-only",
            "current_discharge_facts": self.used_facts.iter().map(UsedCurrentFactV2::as_value).collect::<Vec<_>>(),
            "current_subject_revision_id": self.current_subject_revision_id.as_str(),
            "current_twin_revision_id": self.current_twin_revision_id.as_str(),
            "currentness_assertion_id": self.currentness_assertion_id.as_str(),
            "requirement_revision_id": self.requirement_revision_id.as_str(),
            "requirement_satisfaction_receipt_id": self.receipt_id.as_str(),
            "verification_contract_id": self.verification_contract_id.as_str(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequirementSatisfactionDecisionV2 {
    HistoricalVerificationContract,
    RequirementUnsatisfied(RequirementUnsatisfiedV2),
    CurrentRequirementSatisfied(RequirementSatisfactionReceiptV2),
}

pub fn evaluate_requirement_satisfaction_v2(
    contract: &RequirementVerificationContractV2,
    current_requirement: &AcceptedRequirementRevisionV1,
    current_subject: &SubjectRevisionV1,
    current_twin: &TwinRevisionV1,
    facts: &[CurrentPlanBoundDischargeFactV2],
    currentness_assertion_id: RequirementCurrentnessAssertionIdV2,
) -> Result<RequirementSatisfactionDecisionV2, AssuranceErrorV2> {
    if current_twin.subject_revision_id() != current_subject.revision_id() {
        return Err(AssuranceErrorV2::CurrentTwinSubjectMismatch);
    }
    if current_requirement.revision_id() != contract.requirement_revision_id() {
        return Ok(RequirementSatisfactionDecisionV2::HistoricalVerificationContract);
    }

    let mut applicable: BTreeMap<String, &CurrentPlanBoundDischargeFactV2> = BTreeMap::new();
    for fact in facts {
        if fact.subject_revision_id != *current_subject.revision_id()
            || fact.twin_revision_id != *current_twin.revision_id()
            || fact.requirement_revision_id != *current_requirement.revision_id()
        {
            continue;
        }
        let key = fact.plan_id.as_str().to_string();
        match applicable.get(&key) {
            Some(previous) if previous.fact_id.as_str() <= fact.fact_id.as_str() => {}
            _ => {
                applicable.insert(key, fact);
            }
        }
    }

    let mut missing = Vec::new();
    let mut used_facts = Vec::with_capacity(contract.members.len());
    for member in &contract.members {
        if let Some(fact) = applicable.get(member.evidence_plan_id.as_str()) {
            used_facts.push(UsedCurrentFactV2 {
                current_discharge_fact_id: fact.fact_id.clone(),
                evidence_plan_id: member.evidence_plan_id.clone(),
                obligation_revision_id: member.obligation_revision_id.clone(),
            });
        } else {
            missing.push(member.evidence_plan_id.clone());
        }
    }
    if !missing.is_empty() {
        missing.sort_by(|left, right| left.as_str().cmp(right.as_str()));
        return Ok(RequirementSatisfactionDecisionV2::RequirementUnsatisfied(
            RequirementUnsatisfiedV2 {
                missing_evidence_plan_ids: missing,
            },
        ));
    }

    used_facts.sort_by(|left, right| {
        left.evidence_plan_id
            .as_str()
            .cmp(right.evidence_plan_id.as_str())
            .then_with(|| {
                left.current_discharge_fact_id
                    .as_str()
                    .cmp(right.current_discharge_fact_id.as_str())
            })
    });
    let used_values = used_facts
        .iter()
        .map(UsedCurrentFactV2::as_value)
        .collect::<Vec<_>>();
    let preimage = json!({
        "current_discharge_facts": used_values,
        "current_subject_revision_id": current_subject.revision_id().as_str(),
        "current_twin_revision_id": current_twin.revision_id().as_str(),
        "currentness_assertion_id": currentness_assertion_id.as_str(),
        "requirement_revision_id": current_requirement.revision_id().as_str(),
        "schema": SATISFACTION_SCHEMA_V2,
        "verification_contract_id": contract.contract_id.as_str(),
    });
    let receipt_id = RequirementSatisfactionReceiptIdV2::from_digest(domain_hash(
        SATISFACTION_DOMAIN_V2,
        &preimage,
    ));

    Ok(RequirementSatisfactionDecisionV2::CurrentRequirementSatisfied(
        RequirementSatisfactionReceiptV2 {
            receipt_id,
            verification_contract_id: contract.contract_id.clone(),
            requirement_revision_id: current_requirement.revision_id().clone(),
            current_subject_revision_id: current_subject.revision_id().clone(),
            current_twin_revision_id: current_twin.revision_id().clone(),
            currentness_assertion_id,
            used_facts,
        },
    ))
}

fn normalized_plan_bound_admission_v2(
    plan: &PlanBoundSimulationEvidencePlanV2,
    result: &SimulationResult,
    candidate_artifact_id: &str,
    source_lineage_id: &str,
    input_digest: &Sha256DigestV1,
    output_digest: &Sha256DigestV1,
    warnings: &[String],
) -> Result<Value, AssuranceErrorV2> {
    let mut seen = BTreeSet::new();
    let mut metrics = Vec::with_capacity(result.metrics.len());
    for metric in &result.metrics {
        let name = canonical_text(metric.name.clone(), "result metric name")?;
        if !seen.insert(name.clone()) {
            return Err(AssuranceErrorV2::DuplicateResultMetric(name));
        }
        metrics.push(json!({
            "name": name,
            "uncertainty": optional_uncertainty_value_v2(metric.uncertainty)?,
            "unit": canonical_text(metric.unit.clone(), "result metric unit")?,
            "value": canonical_binary64_field(metric.value, "result metric value")?,
        }));
    }
    metrics.sort_by(|left, right| {
        left["name"]
            .as_str()
            .cmp(&right["name"].as_str())
            .then_with(|| left["unit"].as_str().cmp(&right["unit"].as_str()))
    });

    let evidence = &result.evidence;
    let execution = json!({
        "backend": required_option_text(&evidence.backend, "solver backend")?,
        "input_digest": input_digest.as_str(),
        "mode": execution_mode_name(evidence.mode),
        "output_digest": output_digest.as_str(),
        "parser_version": required_option_text(&evidence.parser_version, "parser version")?,
        "solver_version": required_option_text(&evidence.solver_version, "solver version")?,
    });

    Ok(json!({
        "candidate_artifact_id": candidate_artifact_id,
        "confidence": canonical_binary64_field(result.confidence, "confidence")?,
        "converged": result.converged,
        "execution": execution,
        "metrics": metrics,
        "plan_id": plan.plan_id.as_str(),
        "run_uncertainty": uncertainty_value_v2(result.uncertainty)?,
        "schema": ADMITTED_SCHEMA_V2,
        "source_lineage_id": source_lineage_id,
        "warnings": warnings,
    }))
}

fn request_value_v2(request: &SimulationRequest) -> Result<Value, AssuranceErrorV2> {
    let mut parameters = request.parameters.iter().collect::<Vec<_>>();
    parameters.sort_by(|left, right| left.name.cmp(&right.name));
    let mut seen = BTreeSet::new();
    let mut values = Vec::with_capacity(parameters.len());
    for parameter in parameters {
        let name = canonical_text(parameter.name.clone(), "simulation parameter name")?;
        if !seen.insert(name.clone()) {
            return Err(AssuranceErrorV2::DuplicateParameter(name));
        }
        values.push(json!({
            "name": name,
            "provenance": canonical_text(parameter.provenance.clone(), "simulation parameter provenance")?,
            "uncertainty": optional_uncertainty_value_v2(parameter.uncertainty)?,
            "unit": canonical_text(parameter.unit.clone(), "simulation parameter unit")?,
            "value": canonical_binary64_field(parameter.value, "simulation parameter value")?,
        }));
    }

    let mut metrics = request.requested_metrics.clone();
    for metric in &metrics {
        canonical_text(metric.clone(), "requested metric")?;
    }
    metrics.sort();
    for pair in metrics.windows(2) {
        if pair[0] == pair[1] {
            return Err(AssuranceErrorV2::DuplicateRequestedMetric(pair[0].clone()));
        }
    }

    Ok(json!({
        "domain": engineering_domain_name(request.domain),
        "logical_request_id": canonical_text(request.id.clone(), "simulation request id")?,
        "objective": canonical_text(request.objective.clone(), "simulation objective")?,
        "parameters": values,
        "requested_metrics": metrics,
        "schema": REQUEST_SCHEMA_V2,
        "solver": solver_kind_name(request.solver),
    }))
}

fn policy_value_v2(policy: &SimulationEvidencePolicyRevisionV1) -> Result<Value, AssuranceErrorV2> {
    let metric = policy.required_metric();
    Ok(json!({
        "execution_mode": "external_solver",
        "policy_label": canonical_text(policy.policy_label().to_string(), "evidence policy label")?,
        "required_metric": {
            "max_aleatoric": canonical_binary64_field(metric.max_aleatoric(), "maximum aleatoric uncertainty")?,
            "max_epistemic": canonical_binary64_field(metric.max_epistemic(), "maximum epistemic uncertainty")?,
            "name": metric.name(),
            "operator": metric_operator_name(metric.operator()),
            "threshold": canonical_binary64_field(metric.threshold(), "required metric threshold")?,
            "unit": metric.unit(),
        },
        "schema": POLICY_SCHEMA_V2,
        "warning_policy": warning_policy_value(policy.warning_policy()),
    }))
}

fn uncertainty_value_v2(value: UncertaintyEstimate) -> Result<Value, AssuranceErrorV2> {
    if !value.epistemic.is_finite()
        || !value.aleatoric.is_finite()
        || !(0.0..=1.0).contains(&value.epistemic)
        || !(0.0..=1.0).contains(&value.aleatoric)
    {
        return Err(AssuranceErrorV2::InvalidUncertainty("uncertainty"));
    }
    let interval = match value.interval {
        Some(interval) => {
            if !interval.lower.is_finite()
                || !interval.upper.is_finite()
                || interval.lower > interval.upper
            {
                return Err(AssuranceErrorV2::InvalidUncertainty("interval"));
            }
            json!({
                "lower": canonical_binary64_field(interval.lower, "interval lower")?,
                "upper": canonical_binary64_field(interval.upper, "interval upper")?,
            })
        }
        None => Value::Null,
    };
    Ok(json!({
        "aleatoric": canonical_binary64_field(value.aleatoric, "aleatoric uncertainty")?,
        "epistemic": canonical_binary64_field(value.epistemic, "epistemic uncertainty")?,
        "interval": interval,
    }))
}

fn optional_uncertainty_value_v2(
    value: Option<UncertaintyEstimate>,
) -> Result<Value, AssuranceErrorV2> {
    match value {
        Some(value) => uncertainty_value_v2(value),
        None => Ok(Value::Null),
    }
}

fn canonicalize_warnings(warnings: &[String]) -> Result<Vec<String>, AssuranceErrorV2> {
    let mut normalized = warnings
        .iter()
        .map(|warning| canonical_text(warning.clone(), "solver warning"))
        .collect::<Result<Vec<_>, _>>()?;
    normalized.sort();
    for pair in normalized.windows(2) {
        if pair[0] == pair[1] {
            return Err(AssuranceErrorV2::DuplicateWarning(pair[0].clone()));
        }
    }
    Ok(normalized)
}

fn enforce_warning_policy(
    policy: &WarningPolicyV1,
    warnings: &[String],
) -> Result<(), AssuranceErrorV2> {
    if warnings.is_empty() {
        return Ok(());
    }
    match policy {
        WarningPolicyV1::DenyAny => Err(AssuranceErrorV2::WarningDenied),
        WarningPolicyV1::ReviewRequired => Err(AssuranceErrorV2::WarningReviewRequired),
        WarningPolicyV1::AllowExact(allowed) => {
            for warning in warnings {
                if !allowed.iter().any(|candidate| candidate == warning) {
                    return Err(AssuranceErrorV2::WarningNotAllowed(warning.clone()));
                }
            }
            Ok(())
        }
    }
}

fn warning_policy_value(policy: &WarningPolicyV1) -> Value {
    match policy {
        WarningPolicyV1::DenyAny => json!({"mode": "deny_any"}),
        WarningPolicyV1::ReviewRequired => json!({"mode": "review_required"}),
        WarningPolicyV1::AllowExact(messages) => {
            json!({"allowed_exact_messages": messages, "mode": "allow_exact"})
        }
    }
}

fn canonical_binary64_field(
    value: f64,
    field: &'static str,
) -> Result<String, AssuranceErrorV2> {
    canonical_binary64_v2(value).map_err(|_| AssuranceErrorV2::NonFinite(field))
}

fn canonical_text(
    value: impl Into<String>,
    field: &'static str,
) -> Result<String, AssuranceErrorV2> {
    let value = value.into();
    if value.is_empty() || value.trim() != value {
        Err(AssuranceErrorV2::InvalidText(field))
    } else {
        Ok(value)
    }
}

fn required_option_text(
    value: &Option<String>,
    field: &'static str,
) -> Result<String, AssuranceErrorV2> {
    match value {
        Some(value) => canonical_text(value.clone(), field),
        None => Err(AssuranceErrorV2::InvalidText(field)),
    }
}

fn required_sha256_option(
    value: &Option<String>,
    field: &'static str,
) -> Result<Sha256DigestV1, AssuranceErrorV2> {
    let value = value
        .as_ref()
        .ok_or(AssuranceErrorV2::InvalidDigest(field))?;
    Sha256DigestV1::parse(value.clone()).map_err(|_| AssuranceErrorV2::InvalidDigest(field))
}

fn lower_operator(operator: MetricOperatorV1) -> ThresholdOperatorV1 {
    match operator {
        MetricOperatorV1::Lt => ThresholdOperatorV1::Lt,
        MetricOperatorV1::Le => ThresholdOperatorV1::Le,
        MetricOperatorV1::Gt => ThresholdOperatorV1::Gt,
        MetricOperatorV1::Ge => ThresholdOperatorV1::Ge,
    }
}

fn metric_operator_name(operator: MetricOperatorV1) -> &'static str {
    match operator {
        MetricOperatorV1::Lt => "<",
        MetricOperatorV1::Le => "<=",
        MetricOperatorV1::Gt => ">",
        MetricOperatorV1::Ge => ">=",
    }
}

fn execution_mode_name(mode: ExecutionMode) -> &'static str {
    match mode {
        ExecutionMode::Unknown => "unknown",
        ExecutionMode::DryRun => "dry_run",
        ExecutionMode::ExternalSolver => "external_solver",
    }
}

fn engineering_domain_name(domain: EngineeringDomain) -> &'static str {
    match domain {
        EngineeringDomain::Civil => "Civil",
        EngineeringDomain::Mechanical => "Mechanical",
        EngineeringDomain::Electrical => "Electrical",
        EngineeringDomain::Aerospace => "Aerospace",
        EngineeringDomain::ChemicalProcess => "ChemicalProcess",
        EngineeringDomain::Robotics => "Robotics",
        EngineeringDomain::Nuclear => "Nuclear",
        EngineeringDomain::Materials => "Materials",
        EngineeringDomain::Environmental => "Environmental",
        EngineeringDomain::Systems => "Systems",
    }
}

fn solver_kind_name(solver: SolverKind) -> &'static str {
    match solver {
        SolverKind::FiniteElement => "FiniteElement",
        SolverKind::ComputationalFluidDynamics => "ComputationalFluidDynamics",
        SolverKind::MultibodyDynamics => "MultibodyDynamics",
        SolverKind::Circuit => "Circuit",
        SolverKind::Process => "Process",
        SolverKind::CadGeometry => "CadGeometry",
        SolverKind::MultiPhysics => "MultiPhysics",
        SolverKind::Custom => "Custom",
    }
}

fn domain_hash(domain: &[u8], value: &Value) -> Sha256DigestV1 {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(canonical_json(value).as_bytes());
    Sha256DigestV1::parse(format!("sha256:{}", hex::encode(hasher.finalize())))
        .expect("SHA-256 output is canonical lowercase hex")
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) =>
            serde_json::to_string(value).expect("serializing an in-memory JSON string cannot fail"),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Object(map) => {
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            let body = keys
                .into_iter()
                .map(|key| {
                    let encoded = serde_json::to_string(key)
                        .expect("serializing an in-memory JSON key cannot fail");
                    format!("{encoded}:{}", canonical_json(&map[key]))
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
    }
}
