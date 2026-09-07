// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Semantic-canonical public confirmatory-simulation contract.
//!
//! This facade strengthens the PA-16 v3 contract without widening physical
//! authority. `AllCriteria` claim ordering is canonicalized, exact duplicate
//! criteria are rejected, and IEEE-754 signed zero is normalized because the
//! comparison semantics treat `-0.0` and `+0.0` identically.
//!
//! The earlier v3 facade remains crate-private for regression coverage. External
//! callers therefore cannot mint qualification from its order-sensitive claim
//! transcript.

use crate::confirmatory_contract as v3;
use crate::deliberation::SelectedCandidate;
use crate::outcome_claim::{
    ClaimAggregation, ConfirmatoryClaimOutcome, ConfirmatorySimulationEvidence, MetricCriterion,
    MetricPredicate, PreparedConfirmatorySimulation, SatisfiedSimulationClaim,
    SimulationOutcomeClaim,
};
use crate::safety_preregistration::{
    PreparedSafetyConfirmatorySimulation, SafetyPreregisteredConfirmatoryEvidence,
};
use crate::strict_context::StrictSimulationRegistry;
use symthaea_formal_safety::{EvidenceKind, SafetyCase};
use symthaea_sim_bridge::SimulationRequest;
use thiserror::Error;

const SEMANTIC_CLAIM_TRANSCRIPT_DOMAIN: &[u8] =
    b"symthaea.physical-agency.outcome-claim.semantic.v1";

/// Canonical semantic identity of one preregistered outcome claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalClaimTranscript {
    bytes: Vec<u8>,
}

impl CanonicalClaimTranscript {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Strict simulation qualification bound to the semantic-canonical claim.
///
/// This remains simulation evidence only. It grants no HAL access, actuator
/// capability, execution permit, or physical authority.
#[derive(Debug, Clone, PartialEq)]
pub struct ClaimBoundConfirmatorySimulationQualification {
    inner: v3::ClaimBoundConfirmatorySimulationQualification,
    claim_transcript: CanonicalClaimTranscript,
}

impl ClaimBoundConfirmatorySimulationQualification {
    pub fn selected(&self) -> &SelectedCandidate {
        self.inner.selected()
    }

    pub fn assessment(&self) -> &crate::portfolio::CandidateAssessment {
        self.inner.assessment()
    }

    pub fn world_snapshot(&self) -> &crate::deliberation::WorldSnapshotRef {
        self.inner.world_snapshot()
    }

    pub fn satisfied_claim(&self) -> &SatisfiedSimulationClaim {
        self.inner.satisfied_claim()
    }

    pub fn safety_case_id(&self) -> &str {
        self.inner.safety_case_id()
    }

    pub fn backend(&self) -> &str {
        self.inner.backend()
    }

    pub fn solver_version(&self) -> &str {
        self.inner.solver_version()
    }

    pub fn parser_version(&self) -> &str {
        self.inner.parser_version()
    }

    pub fn input_digest(&self) -> &str {
        self.inner.input_digest()
    }

    pub fn output_digest(&self) -> &str {
        self.inner.output_digest()
    }

    pub fn request_transcript(&self) -> &crate::strict_context::CanonicalRequestTranscript {
        self.inner.request_transcript()
    }

    pub fn contexts(&self) -> &[crate::strict_context::SimulationContextRef] {
        self.inner.contexts()
    }

    pub fn claim_transcript(&self) -> &CanonicalClaimTranscript {
        &self.claim_transcript
    }
}

pub fn prepare_confirmatory_simulation(
    selected: &SelectedCandidate,
    request: SimulationRequest,
    claim: SimulationOutcomeClaim,
) -> Result<PreparedConfirmatorySimulation, ConfirmatoryContractError> {
    let _ = canonical_claim_transcript(&claim)?;
    Ok(v3::prepare_confirmatory_simulation(selected, request, claim)?)
}

pub fn preregister_confirmatory_safety(
    prepared: PreparedConfirmatorySimulation,
    safety_case: &SafetyCase,
) -> Result<PreparedSafetyConfirmatorySimulation, ConfirmatoryContractError> {
    Ok(v3::preregister_confirmatory_safety(prepared, safety_case)?)
}

pub fn run_preregistered_safety_confirmatory_simulation(
    registry: &StrictSimulationRegistry,
    prepared: &PreparedSafetyConfirmatorySimulation,
) -> Result<SafetyPreregisteredConfirmatoryEvidence, ConfirmatoryContractError> {
    Ok(v3::run_preregistered_safety_confirmatory_simulation(
        registry, prepared,
    )?)
}

pub fn evaluate_confirmatory_claim(
    evidence: &ConfirmatorySimulationEvidence,
) -> Result<ConfirmatoryClaimOutcome, ConfirmatoryContractError> {
    Ok(v3::evaluate_confirmatory_claim(evidence)?)
}

/// Semantic-canonical v4 evidence reference.
///
/// PA-15's v2 exact-run lineage is nested directly with the semantic-canonical
/// full claim transcript. The older order-sensitive v3 label is deliberately
/// excluded from v4 identity.
pub fn required_confirmatory_safety_evidence_ref(
    evidence: &SafetyPreregisteredConfirmatoryEvidence,
    satisfied: &SatisfiedSimulationClaim,
) -> Result<String, ConfirmatoryContractError> {
    let v2 = crate::safety_preregistration::required_preregistered_safety_evidence_ref(
        evidence, satisfied,
    )?;
    let claim = canonical_claim_transcript(satisfied.claim())?;

    let mut reference = String::from("physical-agency-confirmatory:v4");
    push_field(&mut reference, &v2);
    push_field(&mut reference, &bytes_to_hex(claim.as_bytes()));
    Ok(reference)
}

/// Qualify only when the completed preregistered SafetyCase cites v4.
///
/// The crate-private v3 facade is satisfied only on an ephemeral clone so the
/// weaker v3 label never becomes externally sufficient evidence.
pub fn qualify_confirmatory_simulation(
    evidence: &SafetyPreregisteredConfirmatoryEvidence,
    satisfied: &SatisfiedSimulationClaim,
    completed_safety_case: &SafetyCase,
) -> Result<ClaimBoundConfirmatorySimulationQualification, ConfirmatoryContractError> {
    let required_v4 = required_confirmatory_safety_evidence_ref(evidence, satisfied)?;
    let Some(index) = completed_safety_case
        .obligations
        .iter()
        .position(|obligation| {
            obligation.expected_evidence == EvidenceKind::Simulation
                && obligation
                    .evidence_refs
                    .iter()
                    .any(|reference| reference == &required_v4)
        })
    else {
        return Err(ConfirmatoryContractError::MissingSemanticClaimBoundSafetyEvidence);
    };

    let required_v3 = v3::required_confirmatory_safety_evidence_ref(evidence, satisfied)?;
    let mut v3_case = completed_safety_case.clone();
    if !v3_case.obligations[index]
        .evidence_refs
        .iter()
        .any(|reference| reference == &required_v3)
    {
        v3_case.obligations[index].evidence_refs.push(required_v3);
    }

    let inner = v3::qualify_confirmatory_simulation(evidence, satisfied, &v3_case)?;
    let claim_transcript = canonical_claim_transcript(satisfied.claim())?;
    Ok(ClaimBoundConfirmatorySimulationQualification {
        inner,
        claim_transcript,
    })
}

pub fn canonical_claim_transcript(
    claim: &SimulationOutcomeClaim,
) -> Result<CanonicalClaimTranscript, ConfirmatoryContractError> {
    // Reuse the lower contract's validation so unsupported schemas and invalid
    // finite/order constraints fail before canonical encoding.
    let _ = v3::canonical_claim_transcript(claim)?;

    let mut criteria = claim
        .criteria
        .iter()
        .map(canonical_criterion_bytes)
        .collect::<Vec<_>>();
    criteria.sort();
    if criteria.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(ConfirmatoryContractError::DuplicateClaimCriterion);
    }

    let mut bytes = Vec::new();
    push_bytes(&mut bytes, SEMANTIC_CLAIM_TRANSCRIPT_DOMAIN);
    bytes.extend_from_slice(&claim.schema_version.to_le_bytes());
    push_bytes(&mut bytes, claim.claim_id.as_bytes());
    push_bytes(&mut bytes, claim.transition_id.as_bytes());
    push_bytes(&mut bytes, claim.proposal_id.as_bytes());
    bytes.push(match claim.aggregation {
        ClaimAggregation::AllCriteria => 0,
    });
    bytes.extend_from_slice(&(criteria.len() as u64).to_le_bytes());
    for criterion in criteria {
        push_bytes(&mut bytes, &criterion);
    }

    Ok(CanonicalClaimTranscript { bytes })
}

fn canonical_criterion_bytes(criterion: &MetricCriterion) -> Vec<u8> {
    let mut bytes = Vec::new();
    push_bytes(&mut bytes, criterion.metric_name.as_bytes());
    push_bytes(&mut bytes, criterion.unit.as_bytes());
    match &criterion.predicate {
        MetricPredicate::AtLeast(value) => {
            bytes.push(0);
            push_f64(&mut bytes, *value);
        }
        MetricPredicate::AtMost(value) => {
            bytes.push(1);
            push_f64(&mut bytes, *value);
        }
        MetricPredicate::InsideClosedInterval { lower, upper } => {
            bytes.push(2);
            push_f64(&mut bytes, *lower);
            push_f64(&mut bytes, *upper);
        }
        MetricPredicate::OutsideOpenInterval { lower, upper } => {
            bytes.push(3);
            push_f64(&mut bytes, *lower);
            push_f64(&mut bytes, *upper);
        }
    }
    bytes.push(match criterion.uncertainty_policy {
        crate::outcome_claim::MetricUncertaintyPolicy::RequireInterval => 0,
        crate::outcome_claim::MetricUncertaintyPolicy::AllowPointEstimate => 1,
    });
    bytes
}

fn push_field(reference: &mut String, field: &str) {
    reference.push('|');
    reference.push_str(&field.len().to_string());
    reference.push(':');
    reference.push_str(field);
}

fn push_bytes(output: &mut Vec<u8>, bytes: &[u8]) {
    output.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    output.extend_from_slice(bytes);
}

fn push_f64(output: &mut Vec<u8>, value: f64) {
    let normalized = if value == 0.0 { 0.0 } else { value };
    output.extend_from_slice(&normalized.to_bits().to_le_bytes());
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

#[derive(Debug, Error)]
pub enum ConfirmatoryContractError {
    #[error("lower confirmatory contract failed: {0}")]
    Lower(#[from] v3::ConfirmatoryContractError),
    #[error("claim contains an exact duplicate criterion")]
    DuplicateClaimCriterion,
    #[error("completed safety case does not cite semantic-canonical v4 simulation lineage")]
    MissingSemanticClaimBoundSafetyEvidence,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MetricUncertaintyPolicy, SimulationOutcomeClaim};

    fn criterion(metric: &str, predicate: MetricPredicate) -> MetricCriterion {
        MetricCriterion {
            metric_name: metric.into(),
            unit: "1".into(),
            predicate,
            uncertainty_policy: MetricUncertaintyPolicy::RequireInterval,
        }
    }

    fn claim(criteria: Vec<MetricCriterion>) -> SimulationOutcomeClaim {
        SimulationOutcomeClaim::all_criteria("claim", "transition", "proposal", criteria)
    }

    #[test]
    fn all_criteria_order_does_not_change_canonical_identity() {
        let a = claim(vec![
            criterion("quality", MetricPredicate::AtLeast(0.8)),
            criterion("risk", MetricPredicate::AtMost(0.2)),
        ]);
        let b = claim(vec![
            criterion("risk", MetricPredicate::AtMost(0.2)),
            criterion("quality", MetricPredicate::AtLeast(0.8)),
        ]);
        assert_eq!(
            canonical_claim_transcript(&a).unwrap(),
            canonical_claim_transcript(&b).unwrap()
        );
    }

    #[test]
    fn signed_zero_has_one_semantic_identity() {
        let negative = claim(vec![criterion("offset", MetricPredicate::AtLeast(-0.0))]);
        let positive = claim(vec![criterion("offset", MetricPredicate::AtLeast(0.0))]);
        assert_eq!(
            canonical_claim_transcript(&negative).unwrap(),
            canonical_claim_transcript(&positive).unwrap()
        );
    }

    #[test]
    fn exact_duplicate_criteria_are_rejected() {
        let duplicate = criterion("quality", MetricPredicate::AtLeast(0.8));
        assert!(matches!(
            canonical_claim_transcript(&claim(vec![duplicate.clone(), duplicate])),
            Err(ConfirmatoryContractError::DuplicateClaimCriterion)
        ));
    }

    #[test]
    fn materially_different_thresholds_remain_distinct() {
        let a = claim(vec![criterion("quality", MetricPredicate::AtLeast(0.8))]);
        let b = claim(vec![criterion("quality", MetricPredicate::AtLeast(0.9))]);
        assert_ne!(
            canonical_claim_transcript(&a).unwrap(),
            canonical_claim_transcript(&b).unwrap()
        );
    }
}
