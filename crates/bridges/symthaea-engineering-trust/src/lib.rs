// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Narrow Engineering Trust Kernel boundary for engineering evidence.
//!
//! ```text
//! solver result
//! != admitted engineering evidence
//! != discharged obligation
//! != qualified design
//! != manufacturing or actuation authority
//! ```
//!
//! The authority-bearing output types in this crate keep their fields private
//! and intentionally do not implement Serde deserialization. They may expose
//! explicit audit records because an audit record is data, not a capability.

#![deny(unsafe_code)]

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use symthaea_formal_safety::{EvidenceKind, ProofObligation};
use symthaea_sim_bridge::{
    ExecutionMode, SimulationMetric, SimulationResult, UncertaintyEstimate,
};
use thiserror::Error;

const ADMISSION_SCHEMA_V1: &str = "symthaea.etk-simulation-evidence-admission.v1";
const ADMITTED_EVIDENCE_DOMAIN_V1: &[u8] = b"symthaea.etk-admitted-simulation-evidence.v1\0";
const OBLIGATION_SNAPSHOT_DOMAIN_V1: &[u8] = b"symthaea.etk-proof-obligation-snapshot.v1\0";
const DISCHARGE_RECEIPT_DOMAIN_V1: &[u8] = b"symthaea.etk-obligation-discharge-receipt.v1\0";

/// Scalar threshold operators supported by ETK simulation admission v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThresholdOperatorV1 {
    Lt,
    Le,
    Gt,
    Ge,
}

impl ThresholdOperatorV1 {
    fn as_str(self) -> &'static str {
        match self {
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
        }
    }

    fn evaluate(self, value: f64, threshold: f64) -> bool {
        match self {
            Self::Lt => value < threshold,
            Self::Le => value <= threshold,
            Self::Gt => value > threshold,
            Self::Ge => value >= threshold,
        }
    }
}

/// Errors while constructing non-authoritative ETK contract inputs.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ContractErrorV1 {
    #[error("{0} cannot be empty")]
    EmptyField(&'static str),
    #[error("{0} must be finite")]
    NonFinite(&'static str),
    #[error("{0} must be within [0, 1]")]
    UnitInterval(&'static str),
    #[error("proof obligation does not expect Simulation evidence")]
    NotSimulationObligation,
}

/// One exact metric predicate plus an epistemic/aleatoric uncertainty budget.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricAcceptancePolicyV1 {
    name: String,
    unit: String,
    operator: ThresholdOperatorV1,
    threshold: f64,
    max_epistemic: f64,
    max_aleatoric: f64,
}

impl MetricAcceptancePolicyV1 {
    pub fn new(
        name: impl Into<String>,
        unit: impl Into<String>,
        operator: ThresholdOperatorV1,
        threshold: f64,
        max_epistemic: f64,
        max_aleatoric: f64,
    ) -> Result<Self, ContractErrorV1> {
        let name = checked_string(name.into(), "metric name")?;
        let unit = checked_string(unit.into(), "metric unit")?;
        if !threshold.is_finite() {
            return Err(ContractErrorV1::NonFinite("metric threshold"));
        }
        checked_unit_interval(max_epistemic, "maximum epistemic uncertainty")?;
        checked_unit_interval(max_aleatoric, "maximum aleatoric uncertainty")?;
        Ok(Self {
            name,
            unit,
            operator,
            threshold,
            max_epistemic,
            max_aleatoric,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn unit(&self) -> &str {
        &self.unit
    }

    pub fn operator(&self) -> ThresholdOperatorV1 {
        self.operator
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    pub fn max_epistemic(&self) -> f64 {
        self.max_epistemic
    }

    pub fn max_aleatoric(&self) -> f64 {
        self.max_aleatoric
    }
}

/// Exact obligation/context against which one simulation artifact is admitted.
///
/// The obligation revision is content-addressed from the obligation definition,
/// not maintained by a caller-controlled counter.
#[derive(Debug, Clone, PartialEq)]
pub struct SimulationAdmissionPolicyV1 {
    obligation_id: String,
    obligation_revision: String,
    subject_id: String,
    twin_revision: String,
    requirement_revision: String,
    evidence_policy_id: String,
    request_id: String,
    validity_domain_id: String,
    required_metric: MetricAcceptancePolicyV1,
    expected_input_digest: String,
}

impl SimulationAdmissionPolicyV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn for_obligation(
        obligation: &ProofObligation,
        subject_id: impl Into<String>,
        twin_revision: impl Into<String>,
        requirement_revision: impl Into<String>,
        evidence_policy_id: impl Into<String>,
        request_id: impl Into<String>,
        validity_domain_id: impl Into<String>,
        required_metric: MetricAcceptancePolicyV1,
        expected_input_digest: impl Into<String>,
    ) -> Result<Self, ContractErrorV1> {
        if obligation.expected_evidence != EvidenceKind::Simulation {
            return Err(ContractErrorV1::NotSimulationObligation);
        }
        Ok(Self {
            obligation_id: obligation.id.to_string(),
            obligation_revision: proof_obligation_snapshot_id_v1(obligation),
            subject_id: checked_string(subject_id.into(), "subject id")?,
            twin_revision: checked_string(twin_revision.into(), "twin revision")?,
            requirement_revision: checked_string(
                requirement_revision.into(),
                "requirement revision",
            )?,
            evidence_policy_id: checked_string(evidence_policy_id.into(), "evidence policy id")?,
            request_id: checked_string(request_id.into(), "simulation request id")?,
            validity_domain_id: checked_string(validity_domain_id.into(), "validity domain id")?,
            required_metric,
            expected_input_digest: checked_string(
                expected_input_digest.into(),
                "expected input digest",
            )?,
        })
    }

    pub fn obligation_id(&self) -> &str {
        &self.obligation_id
    }

    pub fn obligation_revision(&self) -> &str {
        &self.obligation_revision
    }

    pub fn subject_id(&self) -> &str {
        &self.subject_id
    }

    pub fn twin_revision(&self) -> &str {
        &self.twin_revision
    }

    pub fn requirement_revision(&self) -> &str {
        &self.requirement_revision
    }

    pub fn evidence_policy_id(&self) -> &str {
        &self.evidence_policy_id
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn validity_domain_id(&self) -> &str {
        &self.validity_domain_id
    }

    pub fn required_metric(&self) -> &MetricAcceptancePolicyV1 {
        &self.required_metric
    }

    pub fn expected_input_digest(&self) -> &str {
        &self.expected_input_digest
    }
}

/// Candidate applicability at admission time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvidenceCurrentnessV1 {
    Current,
    HistoricallyValid,
}

impl EvidenceCurrentnessV1 {
    fn as_str(self) -> &'static str {
        match self {
            Self::Current => "Current",
            Self::HistoricallyValid => "HistoricallyValid",
        }
    }
}

/// Non-authoritative metadata binding a solver result to an engineering context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulationCandidateBindingV1 {
    candidate_artifact_id: String,
    binds_obligation_id: String,
    obligation_revision: String,
    subject_id: String,
    twin_revision: String,
    requirement_revision: String,
    validity_domain_id: String,
    currentness: EvidenceCurrentnessV1,
    currentness_proof_id: String,
    source_lineage_id: String,
}

impl SimulationCandidateBindingV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        candidate_artifact_id: impl Into<String>,
        binds_obligation_id: impl Into<String>,
        obligation_revision: impl Into<String>,
        subject_id: impl Into<String>,
        twin_revision: impl Into<String>,
        requirement_revision: impl Into<String>,
        validity_domain_id: impl Into<String>,
        currentness: EvidenceCurrentnessV1,
        currentness_proof_id: impl Into<String>,
        source_lineage_id: impl Into<String>,
    ) -> Result<Self, ContractErrorV1> {
        Ok(Self {
            candidate_artifact_id: checked_string(
                candidate_artifact_id.into(),
                "candidate artifact id",
            )?,
            binds_obligation_id: checked_string(
                binds_obligation_id.into(),
                "bound obligation id",
            )?,
            obligation_revision: checked_string(
                obligation_revision.into(),
                "obligation revision",
            )?,
            subject_id: checked_string(subject_id.into(), "subject id")?,
            twin_revision: checked_string(twin_revision.into(), "twin revision")?,
            requirement_revision: checked_string(
                requirement_revision.into(),
                "requirement revision",
            )?,
            validity_domain_id: checked_string(validity_domain_id.into(), "validity domain id")?,
            currentness,
            currentness_proof_id: checked_string(
                currentness_proof_id.into(),
                "currentness proof id",
            )?,
            source_lineage_id: checked_string(source_lineage_id.into(), "source lineage id")?,
        })
    }

    pub fn for_policy(
        policy: &SimulationAdmissionPolicyV1,
        candidate_artifact_id: impl Into<String>,
        currentness: EvidenceCurrentnessV1,
        currentness_proof_id: impl Into<String>,
        source_lineage_id: impl Into<String>,
    ) -> Result<Self, ContractErrorV1> {
        Self::new(
            candidate_artifact_id,
            policy.obligation_id.clone(),
            policy.obligation_revision.clone(),
            policy.subject_id.clone(),
            policy.twin_revision.clone(),
            policy.requirement_revision.clone(),
            policy.validity_domain_id.clone(),
            currentness,
            currentness_proof_id,
            source_lineage_id,
        )
    }
}

/// Stable semantic denial reasons for production admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AdmissionDenialReasonV1 {
    ObligationBindingMismatch,
    ObligationRevisionMismatch,
    SubjectMismatch,
    TwinRevisionMismatch,
    RequirementRevisionMismatch,
    RequestIdMismatch,
    ValidityDomainMismatch,
    CandidateNotCurrent,
    ExecutionModeNotExternalSolver,
    SimulationNotConverged,
    ConfidenceInvalid,
    RunUncertaintyInvalid,
    MetricsMissing,
    MetricInvalid,
    RequiredMetricMissing,
    MetricUnitMismatch,
    MetricUncertaintyInvalid,
    MetricValueOutsideUncertaintyInterval,
    UncertaintyBudgetExceeded,
    AcceptancePredicateFailed,
    ProvenanceIncomplete,
    InputDigestMismatch,
    SimulationBridgeRejected,
}

/// Fail-closed result of the admission theorem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulationAdmissionDenialV1 {
    reasons: Vec<AdmissionDenialReasonV1>,
}

impl SimulationAdmissionDenialV1 {
    pub fn reasons(&self) -> &[AdmissionDenialReasonV1] {
        &self.reasons
    }
}

/// Authority-bearing proof that one exact simulation artifact passed ETK v1
/// admission for one exact obligation/context.
///
/// This is still not an obligation-discharge receipt.
#[must_use = "admitted evidence is not equivalent to an obligation discharge receipt"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedSimulationEvidenceV1 {
    admitted_evidence_id: String,
    candidate_artifact_id: String,
    obligation_id: String,
    obligation_revision: String,
    subject_id: String,
    twin_revision: String,
    requirement_revision: String,
    evidence_policy_id: String,
    request_id: String,
    validity_domain_id: String,
    currentness_proof_id: String,
    source_lineage_id: String,
}

impl AdmittedSimulationEvidenceV1 {
    pub fn admitted_evidence_id(&self) -> &str {
        &self.admitted_evidence_id
    }

    pub fn candidate_artifact_id(&self) -> &str {
        &self.candidate_artifact_id
    }

    pub fn obligation_id(&self) -> &str {
        &self.obligation_id
    }

    pub fn obligation_revision(&self) -> &str {
        &self.obligation_revision
    }

    pub fn subject_id(&self) -> &str {
        &self.subject_id
    }

    pub fn twin_revision(&self) -> &str {
        &self.twin_revision
    }

    pub fn requirement_revision(&self) -> &str {
        &self.requirement_revision
    }

    pub fn evidence_policy_id(&self) -> &str {
        &self.evidence_policy_id
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn validity_domain_id(&self) -> &str {
        &self.validity_domain_id
    }

    pub fn currentness_proof_id(&self) -> &str {
        &self.currentness_proof_id
    }

    pub fn source_lineage_id(&self) -> &str {
        &self.source_lineage_id
    }

    /// Explicit non-authoritative audit representation.
    pub fn audit_record_v1(&self) -> Value {
        json!({
            "admitted_evidence_id": self.admitted_evidence_id.as_str(),
            "candidate_artifact_id": self.candidate_artifact_id.as_str(),
            "currentness_proof_id": self.currentness_proof_id.as_str(),
            "evidence_policy_id": self.evidence_policy_id.as_str(),
            "obligation_id": self.obligation_id.as_str(),
            "obligation_revision": self.obligation_revision.as_str(),
            "request_id": self.request_id.as_str(),
            "requirement_revision": self.requirement_revision.as_str(),
            "source_lineage_id": self.source_lineage_id.as_str(),
            "subject_id": self.subject_id.as_str(),
            "twin_revision": self.twin_revision.as_str(),
            "validity_domain_id": self.validity_domain_id.as_str(),
        })
    }
}

/// Admit one normalized simulation result for one exact ETK v1 policy.
///
/// This is an independent production implementation and never invokes the
/// Python reference oracle.
pub fn admit_simulation_evidence_v1(
    policy: &SimulationAdmissionPolicyV1,
    binding: &SimulationCandidateBindingV1,
    result: &SimulationResult,
) -> Result<AdmittedSimulationEvidenceV1, SimulationAdmissionDenialV1> {
    let mut reasons = BTreeSet::new();

    if binding.binds_obligation_id != policy.obligation_id {
        reasons.insert(AdmissionDenialReasonV1::ObligationBindingMismatch);
    }
    if binding.obligation_revision != policy.obligation_revision {
        reasons.insert(AdmissionDenialReasonV1::ObligationRevisionMismatch);
    }
    if binding.subject_id != policy.subject_id {
        reasons.insert(AdmissionDenialReasonV1::SubjectMismatch);
    }
    if binding.twin_revision != policy.twin_revision {
        reasons.insert(AdmissionDenialReasonV1::TwinRevisionMismatch);
    }
    if binding.requirement_revision != policy.requirement_revision {
        reasons.insert(AdmissionDenialReasonV1::RequirementRevisionMismatch);
    }
    if result.request_id != policy.request_id {
        reasons.insert(AdmissionDenialReasonV1::RequestIdMismatch);
    }
    if binding.validity_domain_id != policy.validity_domain_id {
        reasons.insert(AdmissionDenialReasonV1::ValidityDomainMismatch);
    }
    if binding.currentness != EvidenceCurrentnessV1::Current {
        reasons.insert(AdmissionDenialReasonV1::CandidateNotCurrent);
    }
    if result.evidence.mode != ExecutionMode::ExternalSolver {
        reasons.insert(AdmissionDenialReasonV1::ExecutionModeNotExternalSolver);
    }
    if !result.converged {
        reasons.insert(AdmissionDenialReasonV1::SimulationNotConverged);
    }
    if !result.confidence.is_finite() || !(0.0..=1.0).contains(&result.confidence) {
        reasons.insert(AdmissionDenialReasonV1::ConfidenceInvalid);
    }
    if !uncertainty_is_valid(result.uncertainty) {
        reasons.insert(AdmissionDenialReasonV1::RunUncertaintyInvalid);
    }
    if result.metrics.is_empty() {
        reasons.insert(AdmissionDenialReasonV1::MetricsMissing);
    }

    for metric in &result.metrics {
        if metric.name.trim().is_empty() || metric.unit.trim().is_empty() || !metric.value.is_finite()
        {
            reasons.insert(AdmissionDenialReasonV1::MetricInvalid);
        }
        if let Some(uncertainty) = metric.uncertainty
            && !uncertainty_is_valid(uncertainty)
        {
            reasons.insert(AdmissionDenialReasonV1::MetricUncertaintyInvalid);
        }
    }

    let matching_metrics: Vec<&SimulationMetric> = result
        .metrics
        .iter()
        .filter(|metric| metric.name == policy.required_metric.name)
        .collect();
    if matching_metrics.len() != 1 {
        reasons.insert(AdmissionDenialReasonV1::RequiredMetricMissing);
    } else {
        let metric = matching_metrics[0];
        if metric.unit != policy.required_metric.unit {
            reasons.insert(AdmissionDenialReasonV1::MetricUnitMismatch);
        }
        let effective_uncertainty = metric.uncertainty.unwrap_or(result.uncertainty);
        if uncertainty_is_valid(effective_uncertainty) && metric.value.is_finite() {
            if let Some(interval) = effective_uncertainty.interval
                && !(interval.lower <= metric.value && metric.value <= interval.upper)
            {
                reasons.insert(AdmissionDenialReasonV1::MetricValueOutsideUncertaintyInterval);
            }
            if effective_uncertainty.epistemic > policy.required_metric.max_epistemic
                || effective_uncertainty.aleatoric > policy.required_metric.max_aleatoric
            {
                reasons.insert(AdmissionDenialReasonV1::UncertaintyBudgetExceeded);
            }
            let conservative_value = match effective_uncertainty.interval {
                Some(interval) => match policy.required_metric.operator {
                    ThresholdOperatorV1::Lt | ThresholdOperatorV1::Le => interval.upper,
                    ThresholdOperatorV1::Gt | ThresholdOperatorV1::Ge => interval.lower,
                },
                None => metric.value,
            };
            if metric.unit == policy.required_metric.unit
                && !policy
                    .required_metric
                    .operator
                    .evaluate(conservative_value, policy.required_metric.threshold)
            {
                reasons.insert(AdmissionDenialReasonV1::AcceptancePredicateFailed);
            }
        }
    }

    let evidence = &result.evidence;
    if !option_nonempty(&evidence.backend)
        || !option_nonempty(&evidence.solver_version)
        || !option_nonempty(&evidence.input_digest)
        || !option_nonempty(&evidence.output_digest)
        || !option_nonempty(&evidence.parser_version)
    {
        reasons.insert(AdmissionDenialReasonV1::ProvenanceIncomplete);
    }
    if evidence.input_digest.as_deref() != Some(policy.expected_input_digest.as_str()) {
        reasons.insert(AdmissionDenialReasonV1::InputDigestMismatch);
    }

    // Defense in depth: if the shared simulation bridge becomes stricter,
    // ETK fails closed instead of admitting a bridge-rejected result.
    if reasons.is_empty() && !result.is_engineering_evidence() {
        reasons.insert(AdmissionDenialReasonV1::SimulationBridgeRejected);
    }

    if !reasons.is_empty() {
        return Err(SimulationAdmissionDenialV1 {
            reasons: reasons.into_iter().collect(),
        });
    }

    let identity = normalized_admission_identity_v1(policy, binding, result);
    Ok(AdmittedSimulationEvidenceV1 {
        admitted_evidence_id: domain_hash(ADMITTED_EVIDENCE_DOMAIN_V1, &identity),
        candidate_artifact_id: binding.candidate_artifact_id.clone(),
        obligation_id: policy.obligation_id.clone(),
        obligation_revision: policy.obligation_revision.clone(),
        subject_id: policy.subject_id.clone(),
        twin_revision: policy.twin_revision.clone(),
        requirement_revision: policy.requirement_revision.clone(),
        evidence_policy_id: policy.evidence_policy_id.clone(),
        request_id: policy.request_id.clone(),
        validity_domain_id: policy.validity_domain_id.clone(),
        currentness_proof_id: binding.currentness_proof_id.clone(),
        source_lineage_id: binding.source_lineage_id.clone(),
    })
}

/// Present-tense applicability context for evaluating discharge receipts.
///
/// Supplying the currentness-proof ID on each query is intentional: a receipt
/// that was current when issued must not remain automatically applicable after
/// its currentness attestation changes while all design IDs remain unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DischargeContextV1 {
    subject_id: String,
    twin_revision: String,
    requirement_revision: String,
    validity_domain_id: String,
    currentness_proof_id: String,
}

impl DischargeContextV1 {
    pub fn new(
        subject_id: impl Into<String>,
        twin_revision: impl Into<String>,
        requirement_revision: impl Into<String>,
        validity_domain_id: impl Into<String>,
        currentness_proof_id: impl Into<String>,
    ) -> Result<Self, ContractErrorV1> {
        Ok(Self {
            subject_id: checked_string(subject_id.into(), "subject id")?,
            twin_revision: checked_string(twin_revision.into(), "twin revision")?,
            requirement_revision: checked_string(
                requirement_revision.into(),
                "requirement revision",
            )?,
            validity_domain_id: checked_string(validity_domain_id.into(), "validity domain id")?,
            currentness_proof_id: checked_string(
                currentness_proof_id.into(),
                "currentness proof id",
            )?,
        })
    }

    pub fn subject_id(&self) -> &str {
        &self.subject_id
    }

    pub fn twin_revision(&self) -> &str {
        &self.twin_revision
    }

    pub fn requirement_revision(&self) -> &str {
        &self.requirement_revision
    }

    pub fn validity_domain_id(&self) -> &str {
        &self.validity_domain_id
    }

    pub fn currentness_proof_id(&self) -> &str {
        &self.currentness_proof_id
    }
}

/// Failure to derive a receipt from already-admitted evidence.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DischargeReceiptErrorV1 {
    #[error("proof obligation does not expect Simulation evidence")]
    NotSimulationObligation,
    #[error("admitted evidence targets a different proof obligation")]
    ObligationIdMismatch,
    #[error("admitted evidence targets a stale/different obligation revision")]
    ObligationRevisionMismatch,
}

/// Immutable receipt proving that one admitted-evidence token was applied to
/// one exact proof-obligation snapshot and engineering context.
#[must_use = "a discharge receipt is distinct from evidence admission and downstream authority"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObligationDischargeReceiptV1 {
    receipt_id: String,
    obligation_id: String,
    obligation_revision: String,
    admitted_evidence_id: String,
    candidate_artifact_id: String,
    subject_id: String,
    twin_revision: String,
    requirement_revision: String,
    evidence_policy_id: String,
    validity_domain_id: String,
    currentness_proof_id: String,
}

impl ObligationDischargeReceiptV1 {
    pub fn receipt_id(&self) -> &str {
        &self.receipt_id
    }

    pub fn obligation_id(&self) -> &str {
        &self.obligation_id
    }

    pub fn obligation_revision(&self) -> &str {
        &self.obligation_revision
    }

    pub fn admitted_evidence_id(&self) -> &str {
        &self.admitted_evidence_id
    }

    pub fn candidate_artifact_id(&self) -> &str {
        &self.candidate_artifact_id
    }

    pub fn subject_id(&self) -> &str {
        &self.subject_id
    }

    pub fn twin_revision(&self) -> &str {
        &self.twin_revision
    }

    pub fn requirement_revision(&self) -> &str {
        &self.requirement_revision
    }

    pub fn evidence_policy_id(&self) -> &str {
        &self.evidence_policy_id
    }

    pub fn validity_domain_id(&self) -> &str {
        &self.validity_domain_id
    }

    pub fn currentness_proof_id(&self) -> &str {
        &self.currentness_proof_id
    }

    /// Explicit non-authoritative audit representation.
    pub fn audit_record_v1(&self) -> Value {
        json!({
            "admitted_evidence_id": self.admitted_evidence_id.as_str(),
            "candidate_artifact_id": self.candidate_artifact_id.as_str(),
            "currentness_proof_id": self.currentness_proof_id.as_str(),
            "evidence_policy_id": self.evidence_policy_id.as_str(),
            "obligation_id": self.obligation_id.as_str(),
            "obligation_revision": self.obligation_revision.as_str(),
            "receipt_id": self.receipt_id.as_str(),
            "requirement_revision": self.requirement_revision.as_str(),
            "subject_id": self.subject_id.as_str(),
            "twin_revision": self.twin_revision.as_str(),
            "validity_domain_id": self.validity_domain_id.as_str(),
        })
    }

    fn applies_to(&self, obligation: &ProofObligation, context: &DischargeContextV1) -> bool {
        obligation.expected_evidence == EvidenceKind::Simulation
            && self.obligation_id == obligation.id.to_string()
            && self.obligation_revision == proof_obligation_snapshot_id_v1(obligation)
            && self.subject_id == context.subject_id
            && self.twin_revision == context.twin_revision
            && self.requirement_revision == context.requirement_revision
            && self.validity_domain_id == context.validity_domain_id
            && self.currentness_proof_id == context.currentness_proof_id
    }
}

/// Cross the distinct `admitted evidence -> discharge receipt` boundary.
pub fn issue_obligation_discharge_receipt_v1(
    obligation: &ProofObligation,
    admitted: &AdmittedSimulationEvidenceV1,
) -> Result<ObligationDischargeReceiptV1, DischargeReceiptErrorV1> {
    if obligation.expected_evidence != EvidenceKind::Simulation {
        return Err(DischargeReceiptErrorV1::NotSimulationObligation);
    }
    let obligation_id = obligation.id.to_string();
    if admitted.obligation_id != obligation_id {
        return Err(DischargeReceiptErrorV1::ObligationIdMismatch);
    }
    let obligation_revision = proof_obligation_snapshot_id_v1(obligation);
    if admitted.obligation_revision != obligation_revision {
        return Err(DischargeReceiptErrorV1::ObligationRevisionMismatch);
    }

    let receipt_preimage = json!({
        "admitted_evidence_id": admitted.admitted_evidence_id.as_str(),
        "candidate_artifact_id": admitted.candidate_artifact_id.as_str(),
        "currentness_proof_id": admitted.currentness_proof_id.as_str(),
        "evidence_policy_id": admitted.evidence_policy_id.as_str(),
        "obligation_id": obligation_id.as_str(),
        "obligation_revision": obligation_revision.as_str(),
        "requirement_revision": admitted.requirement_revision.as_str(),
        "subject_id": admitted.subject_id.as_str(),
        "twin_revision": admitted.twin_revision.as_str(),
        "validity_domain_id": admitted.validity_domain_id.as_str(),
    });

    Ok(ObligationDischargeReceiptV1 {
        receipt_id: domain_hash(DISCHARGE_RECEIPT_DOMAIN_V1, &receipt_preimage),
        obligation_id,
        obligation_revision,
        admitted_evidence_id: admitted.admitted_evidence_id.clone(),
        candidate_artifact_id: admitted.candidate_artifact_id.clone(),
        subject_id: admitted.subject_id.clone(),
        twin_revision: admitted.twin_revision.clone(),
        requirement_revision: admitted.requirement_revision.clone(),
        evidence_policy_id: admitted.evidence_policy_id.clone(),
        validity_domain_id: admitted.validity_domain_id.clone(),
        currentness_proof_id: admitted.currentness_proof_id.clone(),
    })
}

/// Derive present-tense discharge from immutable receipts.
///
/// Historical receipts remain auditable but stop applying automatically when
/// the obligation definition, subject, twin revision, accepted requirement,
/// validity domain, or currentness proof changes.
pub fn is_obligation_discharged_v1(
    obligation: &ProofObligation,
    context: &DischargeContextV1,
    receipts: &[ObligationDischargeReceiptV1],
) -> bool {
    receipts
        .iter()
        .any(|receipt| receipt.applies_to(obligation, context))
}

/// Content-addressed revision identity for the proposition being justified.
///
/// Lifecycle status and attached evidence refs are excluded: they are history
/// and consequences, not the semantic obligation definition itself.
pub fn proof_obligation_snapshot_id_v1(obligation: &ProofObligation) -> String {
    let preimage = json!({
        "claim": obligation.claim.as_str(),
        "expected_evidence_kind": evidence_kind_name(obligation.expected_evidence),
        "obligation_id": obligation.id.to_string(),
        "schema": "symthaea.etk-proof-obligation-snapshot.v1",
    });
    domain_hash(OBLIGATION_SNAPSHOT_DOMAIN_V1, &preimage)
}

fn checked_string(value: String, field: &'static str) -> Result<String, ContractErrorV1> {
    if value.trim().is_empty() {
        Err(ContractErrorV1::EmptyField(field))
    } else {
        Ok(value)
    }
}

fn checked_unit_interval(value: f64, field: &'static str) -> Result<(), ContractErrorV1> {
    if !value.is_finite() {
        return Err(ContractErrorV1::NonFinite(field));
    }
    if !(0.0..=1.0).contains(&value) {
        return Err(ContractErrorV1::UnitInterval(field));
    }
    Ok(())
}

fn option_nonempty(value: &Option<String>) -> bool {
    value.as_deref().is_some_and(|text| !text.trim().is_empty())
}

fn uncertainty_is_valid(uncertainty: UncertaintyEstimate) -> bool {
    if !uncertainty.epistemic.is_finite()
        || !(0.0..=1.0).contains(&uncertainty.epistemic)
        || !uncertainty.aleatoric.is_finite()
        || !(0.0..=1.0).contains(&uncertainty.aleatoric)
    {
        return false;
    }
    match uncertainty.interval {
        Some(interval) => {
            interval.lower.is_finite()
                && interval.upper.is_finite()
                && interval.lower <= interval.upper
        }
        None => true,
    }
}

fn evidence_kind_name(kind: EvidenceKind) -> &'static str {
    match kind {
        EvidenceKind::FormalProof => "FormalProof",
        EvidenceKind::Simulation => "Simulation",
        EvidenceKind::Test => "Test",
        EvidenceKind::Telemetry => "Telemetry",
        EvidenceKind::Standard => "Standard",
    }
}

fn execution_mode_name(mode: ExecutionMode) -> &'static str {
    match mode {
        ExecutionMode::Unknown => "unknown",
        ExecutionMode::DryRun => "dry_run",
        ExecutionMode::ExternalSolver => "external_solver",
    }
}

fn uncertainty_value(uncertainty: UncertaintyEstimate) -> Value {
    let interval = match uncertainty.interval {
        Some(interval) => json!({
            "lower": interval.lower,
            "upper": interval.upper,
        }),
        None => Value::Null,
    };
    json!({
        "aleatoric": uncertainty.aleatoric,
        "epistemic": uncertainty.epistemic,
        "interval": interval,
    })
}

fn normalized_admission_identity_v1(
    policy: &SimulationAdmissionPolicyV1,
    binding: &SimulationCandidateBindingV1,
    result: &SimulationResult,
) -> Value {
    let metrics: Vec<Value> = result
        .metrics
        .iter()
        .map(|metric| {
            json!({
                "name": metric.name.as_str(),
                "uncertainty": metric
                    .uncertainty
                    .map(uncertainty_value)
                    .unwrap_or(Value::Null),
                "unit": metric.unit.as_str(),
                "value": metric.value,
            })
        })
        .collect();

    json!({
        "candidate": {
            "binds_obligation_id": binding.binds_obligation_id.as_str(),
            "candidate_artifact_id": binding.candidate_artifact_id.as_str(),
            "confidence": result.confidence,
            "converged": result.converged,
            "currentness": binding.currentness.as_str(),
            "currentness_proof_id": binding.currentness_proof_id.as_str(),
            "evidence_kind": "Simulation",
            "execution": {
                "backend": result.evidence.backend.as_deref().unwrap_or(""),
                "input_digest": result.evidence.input_digest.as_deref().unwrap_or(""),
                "mode": execution_mode_name(result.evidence.mode),
                "output_digest": result.evidence.output_digest.as_deref().unwrap_or(""),
                "parser_version": result.evidence.parser_version.as_deref().unwrap_or(""),
                "solver_version": result.evidence.solver_version.as_deref().unwrap_or(""),
            },
            "metrics": metrics,
            "obligation_revision": binding.obligation_revision.as_str(),
            "request_id": result.request_id.as_str(),
            "requirement_revision": binding.requirement_revision.as_str(),
            "run_uncertainty": uncertainty_value(result.uncertainty),
            "source_lineage_id": binding.source_lineage_id.as_str(),
            "subject_id": binding.subject_id.as_str(),
            "twin_revision": binding.twin_revision.as_str(),
            "validity_domain_id": binding.validity_domain_id.as_str(),
        },
        "expected": {
            "input_digest": policy.expected_input_digest.as_str(),
        },
        "obligation": {
            "evidence_policy_id": policy.evidence_policy_id.as_str(),
            "expected_evidence_kind": "Simulation",
            "obligation_id": policy.obligation_id.as_str(),
            "obligation_revision": policy.obligation_revision.as_str(),
            "request_id": policy.request_id.as_str(),
            "required_metric": {
                "max_aleatoric": policy.required_metric.max_aleatoric,
                "max_epistemic": policy.required_metric.max_epistemic,
                "name": policy.required_metric.name.as_str(),
                "operator": policy.required_metric.operator.as_str(),
                "threshold": policy.required_metric.threshold,
                "unit": policy.required_metric.unit.as_str(),
            },
            "requirement_revision": policy.requirement_revision.as_str(),
            "subject_id": policy.subject_id.as_str(),
            "twin_revision": policy.twin_revision.as_str(),
            "validity_domain_id": policy.validity_domain_id.as_str(),
        },
        "schema": ADMISSION_SCHEMA_V1,
    })
}

fn domain_hash(domain: &[u8], value: &Value) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(canonical_json(value).as_bytes());
    format!("sha256:{}", hex::encode(hasher.finalize()))
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => serde_json::to_string(value)
            .expect("serializing an in-memory JSON string cannot fail"),
        Value::Array(values) => {
            let body = values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{body}]")
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            let body = keys
                .into_iter()
                .map(|key| {
                    let encoded_key = serde_json::to_string(key)
                        .expect("serializing an in-memory JSON key cannot fail");
                    format!("{encoded_key}:{}", canonical_json(&map[key]))
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_formal_safety::ObligationStatus;
    use symthaea_sim_bridge::{Interval, SimulationEvidence};

    fn metric_policy() -> MetricAcceptancePolicyV1 {
        MetricAcceptancePolicyV1::new(
            "max_stress_mpa",
            "MPa",
            ThresholdOperatorV1::Le,
            250.0,
            0.2,
            0.1,
        )
        .unwrap()
    }

    fn external_result() -> SimulationResult {
        SimulationResult {
            request_id: "sim-static-G17-LC9".into(),
            converged: true,
            confidence: 0.94,
            uncertainty: UncertaintyEstimate {
                epistemic: 0.12,
                aleatoric: 0.05,
                interval: None,
            },
            metrics: vec![
                SimulationMetric {
                    name: "max_stress_mpa".into(),
                    value: 181.2,
                    unit: "MPa".into(),
                    uncertainty: Some(UncertaintyEstimate {
                        epistemic: 0.08,
                        aleatoric: 0.04,
                        interval: Some(Interval {
                            lower: 175.0,
                            upper: 190.0,
                        }),
                    }),
                },
                SimulationMetric {
                    name: "max_displacement_mm".into(),
                    value: 0.82,
                    unit: "mm".into(),
                    uncertainty: Some(UncertaintyEstimate {
                        epistemic: 0.1,
                        aleatoric: 0.05,
                        interval: None,
                    }),
                },
            ],
            warnings: Vec::new(),
            evidence: SimulationEvidence {
                mode: ExecutionMode::ExternalSolver,
                backend: Some("calculix".into()),
                solver_version: Some("2.22".into()),
                input_digest: Some("sha256:input-G17-LC9".into()),
                output_digest: Some("sha256:output-run-0007".into()),
                parser_version: Some("symthaea-calculix-parser-v1".into()),
            },
        }
    }

    fn oracle_fixture_policy() -> SimulationAdmissionPolicyV1 {
        SimulationAdmissionPolicyV1 {
            obligation_id: "O-structural-stress-42".into(),
            obligation_revision: "O-structural-stress-42:r3".into(),
            subject_id: "bracket-alpha".into(),
            twin_revision: "design:G17".into(),
            requirement_revision: "REQ-STRESS:r5".into(),
            evidence_policy_id: "ETK-SIM-ADMISSION-V1".into(),
            request_id: "sim-static-G17-LC9".into(),
            validity_domain_id: "VD-static-G17-LC9".into(),
            required_metric: metric_policy(),
            expected_input_digest: "sha256:input-G17-LC9".into(),
        }
    }

    fn oracle_fixture_binding() -> SimulationCandidateBindingV1 {
        SimulationCandidateBindingV1::new(
            "solver-output:run-0007",
            "O-structural-stress-42",
            "O-structural-stress-42:r3",
            "bracket-alpha",
            "design:G17",
            "REQ-STRESS:r5",
            "VD-static-G17-LC9",
            EvidenceCurrentnessV1::Current,
            "currentness:design-G17:fixture-v1",
            "calculix:2.22:mesh-M14:material-M4",
        )
        .unwrap()
    }

    fn live_context(currentness_proof_id: &str) -> DischargeContextV1 {
        DischargeContextV1::new(
            "bracket-alpha",
            "design:G17",
            "REQ-STRESS:r5",
            "VD-static-G17-LC9",
            currentness_proof_id,
        )
        .unwrap()
    }

    #[test]
    fn production_hash_matches_independent_oracle_vector() {
        let admitted = admit_simulation_evidence_v1(
            &oracle_fixture_policy(),
            &oracle_fixture_binding(),
            &external_result(),
        )
        .unwrap();
        assert_eq!(
            admitted.admitted_evidence_id(),
            "sha256:7066c8509f0563484acc3a2d16d5b9b689606a2250389da56ad66560dbc83ff8"
        );
    }

    #[test]
    fn admission_alone_does_not_discharge_obligation() {
        let obligation = ProofObligation::new(
            "stress remains below allowable under service load",
            EvidenceKind::Simulation,
        );
        let policy = SimulationAdmissionPolicyV1::for_obligation(
            &obligation,
            "bracket-alpha",
            "design:G17",
            "REQ-STRESS:r5",
            "ETK-SIM-ADMISSION-V1",
            "sim-static-G17-LC9",
            "VD-static-G17-LC9",
            metric_policy(),
            "sha256:input-G17-LC9",
        )
        .unwrap();
        let binding = SimulationCandidateBindingV1::for_policy(
            &policy,
            "solver-output:run-0007",
            EvidenceCurrentnessV1::Current,
            "currentness:design-G17:test",
            "calculix:2.22:mesh-M14:material-M4",
        )
        .unwrap();
        let admitted = admit_simulation_evidence_v1(&policy, &binding, &external_result()).unwrap();

        assert_eq!(obligation.status, ObligationStatus::Open);
        assert!(!is_obligation_discharged_v1(
            &obligation,
            &live_context("currentness:design-G17:test"),
            &[],
        ));

        let receipt = issue_obligation_discharge_receipt_v1(&obligation, &admitted).unwrap();
        assert_eq!(obligation.status, ObligationStatus::Open);
        assert!(is_obligation_discharged_v1(
            &obligation,
            &live_context("currentness:design-G17:test"),
            &[receipt],
        ));
    }

    #[test]
    fn obligation_mutation_invalidates_old_receipt_without_manual_version_bump() {
        let mut obligation = ProofObligation::new("stress <= allowable", EvidenceKind::Simulation);
        let policy = SimulationAdmissionPolicyV1::for_obligation(
            &obligation,
            "bracket-alpha",
            "design:G17",
            "REQ-STRESS:r5",
            "ETK-SIM-ADMISSION-V1",
            "sim-static-G17-LC9",
            "VD-static-G17-LC9",
            metric_policy(),
            "sha256:input-G17-LC9",
        )
        .unwrap();
        let binding = SimulationCandidateBindingV1::for_policy(
            &policy,
            "solver-output:run-0007",
            EvidenceCurrentnessV1::Current,
            "currentness:design-G17:test",
            "calculix:2.22:mesh-M14:material-M4",
        )
        .unwrap();
        let admitted = admit_simulation_evidence_v1(&policy, &binding, &external_result()).unwrap();
        let receipt = issue_obligation_discharge_receipt_v1(&obligation, &admitted).unwrap();

        obligation.claim = "stress <= revised allowable with fatigue margin".into();
        assert!(!is_obligation_discharged_v1(
            &obligation,
            &live_context("currentness:design-G17:test"),
            &[receipt],
        ));
        assert_eq!(
            issue_obligation_discharge_receipt_v1(&obligation, &admitted),
            Err(DischargeReceiptErrorV1::ObligationRevisionMismatch)
        );
    }

    #[test]
    fn currentness_refresh_makes_old_receipt_inapplicable() {
        let obligation = ProofObligation::new("stress <= allowable", EvidenceKind::Simulation);
        let policy = SimulationAdmissionPolicyV1::for_obligation(
            &obligation,
            "bracket-alpha",
            "design:G17",
            "REQ-STRESS:r5",
            "ETK-SIM-ADMISSION-V1",
            "sim-static-G17-LC9",
            "VD-static-G17-LC9",
            metric_policy(),
            "sha256:input-G17-LC9",
        )
        .unwrap();
        let binding = SimulationCandidateBindingV1::for_policy(
            &policy,
            "solver-output:run-0007",
            EvidenceCurrentnessV1::Current,
            "currentness:design-G17:attestation-1",
            "calculix:2.22:mesh-M14:material-M4",
        )
        .unwrap();
        let admitted = admit_simulation_evidence_v1(&policy, &binding, &external_result()).unwrap();
        let receipt = issue_obligation_discharge_receipt_v1(&obligation, &admitted).unwrap();

        assert!(is_obligation_discharged_v1(
            &obligation,
            &live_context("currentness:design-G17:attestation-1"),
            std::slice::from_ref(&receipt),
        ));
        assert!(!is_obligation_discharged_v1(
            &obligation,
            &live_context("currentness:design-G17:attestation-2"),
            &[receipt],
        ));
    }

    #[test]
    fn dry_run_cannot_be_admitted() {
        let mut result = external_result();
        result.evidence.mode = ExecutionMode::DryRun;
        let denial = admit_simulation_evidence_v1(
            &oracle_fixture_policy(),
            &oracle_fixture_binding(),
            &result,
        )
        .unwrap_err();
        assert!(
            denial
                .reasons()
                .contains(&AdmissionDenialReasonV1::ExecutionModeNotExternalSolver)
        );
    }

    #[test]
    fn passing_mean_with_failing_interval_is_denied() {
        let mut result = external_result();
        result.metrics[0].value = 249.0;
        result.metrics[0].uncertainty = Some(UncertaintyEstimate {
            epistemic: 0.08,
            aleatoric: 0.04,
            interval: Some(Interval {
                lower: 240.0,
                upper: 260.0,
            }),
        });
        let denial = admit_simulation_evidence_v1(
            &oracle_fixture_policy(),
            &oracle_fixture_binding(),
            &result,
        )
        .unwrap_err();
        assert!(
            denial
                .reasons()
                .contains(&AdmissionDenialReasonV1::AcceptancePredicateFailed)
        );
    }

    #[test]
    fn excessive_metric_uncertainty_is_denied() {
        let mut result = external_result();
        result.metrics[0].uncertainty = Some(UncertaintyEstimate {
            epistemic: 0.25,
            aleatoric: 0.04,
            interval: Some(Interval {
                lower: 175.0,
                upper: 190.0,
            }),
        });
        let denial = admit_simulation_evidence_v1(
            &oracle_fixture_policy(),
            &oracle_fixture_binding(),
            &result,
        )
        .unwrap_err();
        assert!(
            denial
                .reasons()
                .contains(&AdmissionDenialReasonV1::UncertaintyBudgetExceeded)
        );
    }

    #[test]
    fn malformed_secondary_metric_fails_closed() {
        let mut result = external_result();
        result.metrics[1].value = f64::NAN;
        let denial = admit_simulation_evidence_v1(
            &oracle_fixture_policy(),
            &oracle_fixture_binding(),
            &result,
        )
        .unwrap_err();
        assert!(
            denial
                .reasons()
                .contains(&AdmissionDenialReasonV1::MetricInvalid)
        );
    }

    #[test]
    fn audit_records_do_not_mutate_or_discharge() {
        let admitted = admit_simulation_evidence_v1(
            &oracle_fixture_policy(),
            &oracle_fixture_binding(),
            &external_result(),
        )
        .unwrap();
        let record = admitted.audit_record_v1();
        assert_eq!(
            record["admitted_evidence_id"],
            Value::String(admitted.admitted_evidence_id().to_string())
        );
    }
}
