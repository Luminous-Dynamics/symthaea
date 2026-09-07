// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Normalized-canonical public confirmatory-simulation contract.
//!
//! This facade strengthens the PA-16 v3 contract without widening physical
//! authority. `AllCriteria` claim ordering is canonicalized, exact duplicate
//! criteria are rejected, IEEE-754 signed zero is normalized because the
//! comparison semantics treat `-0.0` and `+0.0` identically, and mathematically
//! unsatisfiable scalar conjunctions fail before solver execution.
//!
//! "Normalized-canonical" is intentionally narrower than a universal semantic
//! normal form: metric units remain typed only as strings and logically
//! redundant-but-distinct criteria are still represented explicitly.
//!
//! The strict v4 profile permits exactly one `EvidenceKind::Simulation` safety
//! obligation: it is structurally designated as the exact confirmatory-outcome
//! obligation and must cite the exact v4 lineage. Additional simulation-derived
//! safety claims require a future typed per-obligation/multi-run evidence
//! contract rather than free-form obligation text plus opaque refs.
//!
//! The earlier v3 facade remains crate-private for regression coverage. External
//! callers therefore cannot mint qualification from its order-sensitive claim
//! transcript.

use crate::confirmatory_contract as v3;
use crate::deliberation::SelectedCandidate;
use crate::outcome_claim::{
    ClaimAggregation, ConfirmatoryClaimOutcome, ConfirmatorySimulationEvidence, MetricCriterion,
    MetricPredicate, OutcomeClaimError, PreparedConfirmatorySimulation, SatisfiedSimulationClaim,
    SimulationOutcomeClaim,
};
use crate::safety_preregistration::{
    PreparedSafetyConfirmatorySimulation, SafetyPreregisteredConfirmatoryEvidence,
    SafetyPreregistrationError,
};
use crate::strict_context::StrictSimulationRegistry;
use std::collections::BTreeMap;
use symthaea_formal_safety::{EvidenceKind, SafetyCase};
use symthaea_sim_bridge::{SimulationMetric, SimulationRequest};
use thiserror::Error;

const NORMALIZED_CLAIM_TRANSCRIPT_DOMAIN: &[u8] =
    b"symthaea.physical-agency.outcome-claim.normalized.v1";

/// Canonical normalized identity of one preregistered outcome claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalClaimTranscript {
    bytes: Vec<u8>,
}

impl CanonicalClaimTranscript {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Strict simulation qualification bound to the normalized-canonical claim.
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
    let simulation_obligation_count = safety_case
        .obligations
        .iter()
        .filter(|obligation| obligation.expected_evidence == EvidenceKind::Simulation)
        .count();

    match simulation_obligation_count {
        1 => {}
        0 => {
            return Err(ConfirmatoryContractError::Safety(
                SafetyPreregistrationError::MissingSimulationObligation,
            ));
        }
        count => {
            return Err(
                ConfirmatoryContractError::MultipleSimulationObligationsRequireTypedBindings(
                    count,
                ),
            );
        }
    }

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
    validate_claimed_metric_interval_consistency(evidence)?;
    Ok(v3::evaluate_confirmatory_claim(evidence)?)
}

fn validate_claimed_metric_interval_consistency(
    evidence: &ConfirmatorySimulationEvidence,
) -> Result<(), ConfirmatoryContractError> {
    let result = evidence.selection_evidence().validated().result();

    for criterion in &evidence.claim().criteria {
        let mut matching = result.metrics.iter().filter(|metric| {
            metric.name == criterion.metric_name && metric.unit == criterion.unit
        });
        let Some(metric) = matching.next() else {
            continue;
        };
        if matching.next().is_some() {
            continue;
        }
        validate_metric_interval_consistency(metric)?;
    }

    Ok(())
}

fn validate_metric_interval_consistency(
    metric: &SimulationMetric,
) -> Result<(), ConfirmatoryContractError> {
    let Some(interval) = metric.uncertainty.and_then(|uncertainty| uncertainty.interval) else {
        return Ok(());
    };

    if !interval.contains(metric.value) {
        return Err(ConfirmatoryContractError::MetricEstimateOutsideInterval {
            metric_name: metric.name.clone(),
            unit: metric.unit.clone(),
            value: metric.value,
            lower: interval.lower,
            upper: interval.upper,
        });
    }

    Ok(())
}

/// Normalized-canonical v4 evidence reference.
///
/// PA-15's v2 exact-run lineage is nested directly with the normalized complete
/// claim transcript. The older order-sensitive v3 label is deliberately
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

/// Qualify only when the preregistered Simulation obligation cites v4.
///
/// The crate-private v3 facade is satisfied only on an ephemeral clone so the
/// weaker v3 label never becomes externally sufficient evidence. Exactly one
/// Simulation obligation is permitted by the public preregistration path.
pub fn qualify_confirmatory_simulation(
    evidence: &SafetyPreregisteredConfirmatoryEvidence,
    satisfied: &SatisfiedSimulationClaim,
    completed_safety_case: &SafetyCase,
) -> Result<ClaimBoundConfirmatorySimulationQualification, ConfirmatoryContractError> {
    let required_v4 = required_confirmatory_safety_evidence_ref(evidence, satisfied)?;
    let mut first_simulation_index = None;

    for frozen in evidence
        .safety_plan()
        .obligations()
        .iter()
        .filter(|obligation| obligation.expected_evidence() == EvidenceKind::Simulation)
    {
        let Some((index, actual)) = completed_safety_case
            .obligations
            .iter()
            .enumerate()
            .find(|(_, obligation)| obligation.id.to_string() == frozen.id())
        else {
            return Err(
                ConfirmatoryContractError::SimulationObligationMissingNormalizedEvidence(
                    frozen.id().to_string(),
                ),
            );
        };

        if !actual
            .evidence_refs
            .iter()
            .any(|reference| reference == &required_v4)
        {
            return Err(
                ConfirmatoryContractError::SimulationObligationMissingNormalizedEvidence(
                    frozen.id().to_string(),
                ),
            );
        }

        if first_simulation_index.is_none() {
            first_simulation_index = Some(index);
        }
    }

    let Some(index) = first_simulation_index else {
        return Err(ConfirmatoryContractError::MissingNormalizedClaimBoundSafetyEvidence);
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
    // finite/order constraints fail before normalized encoding.
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

    validate_all_criteria_satisfiable(claim)?;

    let mut bytes = Vec::new();
    push_bytes(&mut bytes, NORMALIZED_CLAIM_TRANSCRIPT_DOMAIN);
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

#[derive(Debug, Clone, Copy, PartialEq)]
struct ClosedRange {
    lower: Option<f64>,
    upper: Option<f64>,
}

impl ClosedRange {
    const ALL_REAL: Self = Self {
        lower: None,
        upper: None,
    };
}

fn validate_all_criteria_satisfiable(
    claim: &SimulationOutcomeClaim,
) -> Result<(), ConfirmatoryContractError> {
    let mut groups: BTreeMap<(String, String), Vec<&MetricPredicate>> = BTreeMap::new();
    for criterion in &claim.criteria {
        groups
            .entry((criterion.metric_name.clone(), criterion.unit.clone()))
            .or_default()
            .push(&criterion.predicate);
    }

    for ((metric_name, unit), predicates) in groups {
        let mut feasible = vec![ClosedRange::ALL_REAL];
        for predicate in predicates {
            feasible = intersect_range_sets(&feasible, &predicate_ranges(predicate));
            if feasible.is_empty() {
                return Err(ConfirmatoryContractError::UnsatisfiableClaim {
                    metric_name,
                    unit,
                });
            }
        }
    }

    Ok(())
}

fn predicate_ranges(predicate: &MetricPredicate) -> Vec<ClosedRange> {
    match predicate {
        MetricPredicate::AtLeast(value) => vec![ClosedRange {
            lower: Some(*value),
            upper: None,
        }],
        MetricPredicate::AtMost(value) => vec![ClosedRange {
            lower: None,
            upper: Some(*value),
        }],
        MetricPredicate::InsideClosedInterval { lower, upper } => vec![ClosedRange {
            lower: Some(*lower),
            upper: Some(*upper),
        }],
        MetricPredicate::OutsideOpenInterval { lower, upper } => vec![
            ClosedRange {
                lower: None,
                upper: Some(*lower),
            },
            ClosedRange {
                lower: Some(*upper),
                upper: None,
            },
        ],
    }
}

fn intersect_range_sets(left: &[ClosedRange], right: &[ClosedRange]) -> Vec<ClosedRange> {
    let mut intersections = Vec::new();
    for left_range in left {
        for right_range in right {
            if let Some(intersection) = intersect_ranges(*left_range, *right_range) {
                intersections.push(intersection);
            }
        }
    }
    normalize_ranges(intersections)
}

fn intersect_ranges(left: ClosedRange, right: ClosedRange) -> Option<ClosedRange> {
    let lower = max_lower(left.lower, right.lower);
    let upper = min_upper(left.upper, right.upper);
    if matches!((lower, upper), (Some(lower), Some(upper)) if lower > upper) {
        None
    } else {
        Some(ClosedRange { lower, upper })
    }
}

fn normalize_ranges(mut ranges: Vec<ClosedRange>) -> Vec<ClosedRange> {
    ranges.sort_by(|left, right| compare_lower(left.lower, right.lower));
    let mut normalized: Vec<ClosedRange> = Vec::with_capacity(ranges.len());

    for range in ranges {
        if let Some(last) = normalized.last_mut() {
            if ranges_overlap_or_touch(*last, range) {
                last.upper = max_upper(last.upper, range.upper);
                continue;
            }
        }
        normalized.push(range);
    }

    normalized
}

fn compare_lower(left: Option<f64>, right: Option<f64>) -> std::cmp::Ordering {
    match (left, right) {
        (None, None) => std::cmp::Ordering::Equal,
        (None, Some(_)) => std::cmp::Ordering::Less,
        (Some(_), None) => std::cmp::Ordering::Greater,
        (Some(left), Some(right)) => left.total_cmp(&right),
    }
}

fn ranges_overlap_or_touch(left: ClosedRange, right: ClosedRange) -> bool {
    match (left.upper, right.lower) {
        (None, _) | (_, None) => true,
        (Some(upper), Some(lower)) => lower <= upper,
    }
}

fn max_lower(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (None, value) | (value, None) => value,
        (Some(left), Some(right)) => Some(left.max(right)),
    }
}

fn min_upper(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (None, value) | (value, None) => value,
        (Some(left), Some(right)) => Some(left.min(right)),
    }
}

fn max_upper(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (None, _) | (_, None) => None,
        (Some(left), Some(right)) => Some(left.max(right)),
    }
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
    #[error("outcome-claim contract failed: {0}")]
    Outcome(OutcomeClaimError),
    #[error("safety preregistration/qualification failed: {0}")]
    Safety(SafetyPreregistrationError),
    #[error("claim metric {0:?} was not explicitly requested by the simulation request")]
    ClaimMetricNotRequested(String),
    #[error("claim metric {0:?} appears more than once in simulation requested_metrics")]
    ClaimMetricRequestedMultipleTimes(String),
    #[error("claim contains an exact duplicate criterion")]
    DuplicateClaimCriterion,
    #[error("all_criteria claim is unsatisfiable for metric {metric_name:?} unit {unit:?}")]
    UnsatisfiableClaim { metric_name: String, unit: String },
    #[error("strict v4 preregistration permits exactly one Simulation obligation; found {0}")]
    MultipleSimulationObligationsRequireTypedBindings(usize),
    #[error("claimed metric {metric_name:?} unit {unit:?} reports estimate {value} outside uncertainty interval [{lower}, {upper}]")]
    MetricEstimateOutsideInterval {
        metric_name: String,
        unit: String,
        value: f64,
        lower: f64,
        upper: f64,
    },
    #[error("preregistered Simulation obligation {0:?} does not cite normalized-canonical v4 lineage")]
    SimulationObligationMissingNormalizedEvidence(String),
    #[error("completed safety case does not cite normalized-canonical v4 simulation lineage")]
    MissingNormalizedClaimBoundSafetyEvidence,
    #[error("private v3 structural qualification unexpectedly lacked its exact simulation lineage")]
    InternalV3StructuralEvidenceMissing,
}

impl From<v3::ConfirmatoryContractError> for ConfirmatoryContractError {
    fn from(error: v3::ConfirmatoryContractError) -> Self {
        match error {
            v3::ConfirmatoryContractError::Outcome(error) => Self::Outcome(error),
            v3::ConfirmatoryContractError::Safety(error) => Self::Safety(error),
            v3::ConfirmatoryContractError::ClaimMetricNotRequested(metric) => {
                Self::ClaimMetricNotRequested(metric)
            }
            v3::ConfirmatoryContractError::ClaimMetricRequestedMultipleTimes(metric) => {
                Self::ClaimMetricRequestedMultipleTimes(metric)
            }
            v3::ConfirmatoryContractError::MissingClaimBoundSafetyEvidence => {
                Self::InternalV3StructuralEvidenceMissing
            }
        }
    }
}

impl From<SafetyPreregistrationError> for ConfirmatoryContractError {
    fn from(error: SafetyPreregistrationError) -> Self {
        Self::Safety(error)
    }
}

impl From<OutcomeClaimError> for ConfirmatoryContractError {
    fn from(error: OutcomeClaimError) -> Self {
        Self::Outcome(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MetricUncertaintyPolicy, SimulationOutcomeClaim};
    use symthaea_sim_bridge::{Interval, UncertaintyEstimate};

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
    fn signed_zero_has_one_normalized_identity() {
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
    fn contradictory_lower_and_upper_bounds_are_rejected() {
        let impossible = claim(vec![
            criterion("quality", MetricPredicate::AtLeast(0.9)),
            criterion("quality", MetricPredicate::AtMost(0.8)),
        ]);
        assert!(matches!(
            canonical_claim_transcript(&impossible),
            Err(ConfirmatoryContractError::UnsatisfiableClaim { metric_name, unit })
                if metric_name == "quality" && unit == "1"
        ));
    }

    #[test]
    fn inside_and_outside_conflict_is_rejected() {
        let impossible = claim(vec![
            criterion(
                "quality",
                MetricPredicate::InsideClosedInterval {
                    lower: 0.3,
                    upper: 0.7,
                },
            ),
            criterion(
                "quality",
                MetricPredicate::OutsideOpenInterval {
                    lower: 0.2,
                    upper: 0.8,
                },
            ),
        ]);
        assert!(matches!(
            canonical_claim_transcript(&impossible),
            Err(ConfirmatoryContractError::UnsatisfiableClaim { .. })
        ));
    }

    #[test]
    fn outside_open_interval_preserves_boundary_solutions() {
        let feasible = claim(vec![
            criterion(
                "quality",
                MetricPredicate::InsideClosedInterval {
                    lower: 0.2,
                    upper: 0.8,
                },
            ),
            criterion(
                "quality",
                MetricPredicate::OutsideOpenInterval {
                    lower: 0.2,
                    upper: 0.8,
                },
            ),
        ]);
        assert!(canonical_claim_transcript(&feasible).is_ok());
    }

    #[test]
    fn claimed_metric_estimate_must_lie_inside_its_interval() {
        let metric = SimulationMetric {
            name: "quality".into(),
            value: 0.1,
            unit: "1".into(),
            uncertainty: Some(
                UncertaintyEstimate::new(0.1, 0.1).with_interval(Interval::new(0.9, 1.0)),
            ),
        };

        assert!(matches!(
            validate_metric_interval_consistency(&metric),
            Err(ConfirmatoryContractError::MetricEstimateOutsideInterval { .. })
        ));
    }

    #[test]
    fn claimed_metric_estimate_inside_interval_is_consistent() {
        let metric = SimulationMetric {
            name: "quality".into(),
            value: 0.95,
            unit: "1".into(),
            uncertainty: Some(
                UncertaintyEstimate::new(0.1, 0.1).with_interval(Interval::new(0.9, 1.0)),
            ),
        };

        assert!(validate_metric_interval_consistency(&metric).is_ok());
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

    #[test]
    fn safety_errors_are_not_hidden_behind_private_v3_surface() {
        let public = ConfirmatoryContractError::from(v3::ConfirmatoryContractError::Safety(
            SafetyPreregistrationError::MissingIndependentObligation,
        ));
        assert!(matches!(
            public,
            ConfirmatoryContractError::Safety(
                SafetyPreregistrationError::MissingIndependentObligation
            )
        ));
    }
}
