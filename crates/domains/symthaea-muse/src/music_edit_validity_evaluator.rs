// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic, fail-closed evaluation of musical edit obligations against
//! source-bound typed observations. This module reduces evidence records; it
//! does not authenticate measurement producers or claim perceptual validity.

use crate::music_edit_validity_envelope::{
    music_edit_validity_envelope_digest_v1, validate_music_edit_validity_envelope_v1,
    MusicEditValidityConstraintV1, MusicEditValidityEnvelopeV1,
    MusicEditValidityEvidenceV1, MusicEditValidityPropertyV1,
    MusicEditValidityRelationV1, MusicEditValidityScopeV1,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const MUSIC_EDIT_VALIDITY_EVALUATOR_SCHEMA_V1: &str =
    "mel-music-edit-validity-evaluation-v1";
pub const MUSIC_EDIT_VALIDITY_EVALUATOR_VERSION_V1: &str =
    "mel-validity-evaluator-v1";
const EVALUATOR_DOMAIN: &[u8] = b"luminous.mel.music-edit-validity-evaluator.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditObservationSourceFamilyV1 {
    Symbolic,
    Acoustic,
    HumanAdjudication,
    RendererBound,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditObservationSourceV1 {
    pub family: MusicEditObservationSourceFamilyV1,
    pub evidence_sha256: String,
    pub method_id: String,
    pub method_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MusicEditObservationPayloadV1 {
    Text { value: String },
    Scalar { value: f64, unit: String, uncertainty: f64 },
    Presence { present: bool },
    TextComparison { before: String, after: String },
    NumericComparison { before: f64, after: f64, unit: String, uncertainty: f64 },
    NumericSeries { values: Vec<f64>, unit: String, uncertainty: f64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditValidityObservationV1 {
    pub constraint_id: String,
    pub property: MusicEditValidityPropertyV1,
    pub scope: MusicEditValidityScopeV1,
    pub envelope_sha256: String,
    pub candidate_output_sha256: String,
    pub source_lineage_sha256: String,
    pub sources: Vec<MusicEditObservationSourceV1>,
    pub payload: MusicEditObservationPayloadV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditValidityConstraintSectionV1 {
    RequestedChange,
    ProtectedInvariant,
    PermittedVariation,
    ForbiddenChange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditValidityObservationStatusV1 {
    Satisfied,
    Violated,
    Indeterminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditValidityObservationReasonV1 {
    RelationSatisfied,
    RelationContradicted,
    MissingObservation,
    InsufficientEvidenceFamilies,
    InvalidEvidenceDigest,
    EmptyEvidenceMethod,
    NonCanonicalEvidenceSources,
    WrongPayloadKind,
    UnitMismatch,
    InvalidUncertainty,
    NonFiniteMeasurement,
    EmptyOrShortSeries,
    NonMonotonicSeries,
    UncertainMonotonicity,
    MalformedExpectedValue,
    UncertainEquality,
    BoundaryUncertainty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditValidityDecisionV1 {
    Acceptable,
    Rejected,
    Abstain,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditConstraintEvaluationV1 {
    pub constraint_id: String,
    pub section: MusicEditValidityConstraintSectionV1,
    pub status: MusicEditValidityObservationStatusV1,
    pub reason: MusicEditValidityObservationReasonV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MusicEditValidityEvaluationIssueV1 {
    InvalidEnvelope(Vec<String>),
    InvalidCandidateDigest,
    SourceLineageMismatch,
    InvalidObservationEnvelopeDigest { constraint_id: String },
    InvalidObservationCandidateDigest { constraint_id: String },
    InvalidObservationLineage { constraint_id: String },
    InvalidObservationDigest { constraint_id: String },
    DuplicateObservation { constraint_id: String },
    UnexpectedObservation { constraint_id: String },
    NonCanonicalObservationOrder,
    SerializationError,
    ReceiptSchemaMismatch,
    ReceiptEnvelopeMismatch,
    ReceiptCandidateMismatch,
    ReceiptLineageMismatch,
    ReceiptObservationBundleMismatch,
    ReceiptOutcomeMismatch,
    ReceiptDigestMismatch,
}

impl std::fmt::Display for MusicEditValidityEvaluationIssueV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for MusicEditValidityEvaluationIssueV1 {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicEditValidityEvaluationReceiptV1 {
    pub schema_version: String,
    pub evaluator_version: String,
    pub envelope_sha256: String,
    pub candidate_output_sha256: String,
    pub source_lineage_sha256: String,
    pub observation_bundle_sha256: String,
    pub outcomes: Vec<MusicEditConstraintEvaluationV1>,
    pub decision: MusicEditValidityDecisionV1,
    pub receipt_sha256: String,
}

#[derive(Serialize)]
struct EvaluationReceiptCommitmentV1<'a> {
    schema_version: &'a str,
    evaluator_version: &'a str,
    envelope_sha256: &'a str,
    candidate_output_sha256: &'a str,
    source_lineage_sha256: &'a str,
    observation_bundle_sha256: &'a str,
    outcomes: &'a [MusicEditConstraintEvaluationV1],
    decision: MusicEditValidityDecisionV1,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(EVALUATOR_DOMAIN);
    hasher.update(b"\0");
    hasher.update(domain);
    hasher.update(b"\0");
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn json_digest<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<String, MusicEditValidityEvaluationIssueV1> {
    let bytes = serde_json::to_vec(value)
        .map_err(|_| MusicEditValidityEvaluationIssueV1::SerializationError)?;
    Ok(domain_hash(domain, &bytes))
}

fn all_constraints(
    envelope: &MusicEditValidityEnvelopeV1,
) -> Vec<(&MusicEditValidityConstraintV1, MusicEditValidityConstraintSectionV1)> {
    let mut result = Vec::new();
    result.extend(envelope.requested_change.iter().map(|item| {
        (item, MusicEditValidityConstraintSectionV1::RequestedChange)
    }));
    result.extend(envelope.protected_invariants.iter().map(|item| {
        (item, MusicEditValidityConstraintSectionV1::ProtectedInvariant)
    }));
    result.extend(envelope.permitted_variations.iter().map(|item| {
        (item, MusicEditValidityConstraintSectionV1::PermittedVariation)
    }));
    result.extend(envelope.forbidden_changes.iter().map(|item| {
        (item, MusicEditValidityConstraintSectionV1::ForbiddenChange)
    }));
    result.sort_by(|left, right| left.0.constraint_id.cmp(&right.0.constraint_id));
    result
}

fn expected_source_family(
    evidence: MusicEditValidityEvidenceV1,
) -> Option<MusicEditObservationSourceFamilyV1> {
    match evidence {
        MusicEditValidityEvidenceV1::Symbolic => {
            Some(MusicEditObservationSourceFamilyV1::Symbolic)
        }
        MusicEditValidityEvidenceV1::Acoustic => {
            Some(MusicEditObservationSourceFamilyV1::Acoustic)
        }
        MusicEditValidityEvidenceV1::HumanAdjudication => {
            Some(MusicEditObservationSourceFamilyV1::HumanAdjudication)
        }
        MusicEditValidityEvidenceV1::RendererBound => {
            Some(MusicEditObservationSourceFamilyV1::RendererBound)
        }
        MusicEditValidityEvidenceV1::Combined => None,
    }
}

fn validate_observation_sources(
    constraint: &MusicEditValidityConstraintV1,
    sources: &[MusicEditObservationSourceV1],
) -> Option<MusicEditValidityObservationReasonV1> {
    use MusicEditValidityObservationReasonV1 as Reason;

    if sources.is_empty() {
        return Some(Reason::InsufficientEvidenceFamilies);
    }
    if !sources.windows(2).all(|pair| pair[0].family < pair[1].family) {
        return Some(Reason::NonCanonicalEvidenceSources);
    }
    for source in sources {
        if !is_sha256(&source.evidence_sha256) || !is_sha256(&source.method_sha256) {
            return Some(Reason::InvalidEvidenceDigest);
        }
        if source.method_id.trim().is_empty() {
            return Some(Reason::EmptyEvidenceMethod);
        }
    }
    match expected_source_family(constraint.evidence) {
        Some(expected) if sources.iter().any(|source| source.family == expected) => None,
        Some(_) => Some(Reason::InsufficientEvidenceFamilies),
        None if sources.len() >= 2 => None,
        None => Some(Reason::InsufficientEvidenceFamilies),
    }
}

fn relation_result(
    constraint: &MusicEditValidityConstraintV1,
    payload: &MusicEditObservationPayloadV1,
) -> (MusicEditValidityObservationStatusV1, MusicEditValidityObservationReasonV1) {
    use MusicEditObservationPayloadV1 as Payload;
    use MusicEditValidityObservationReasonV1 as Reason;
    use MusicEditValidityObservationStatusV1 as Status;
    use MusicEditValidityRelationV1 as Relation;

    let satisfied = |condition: bool| {
        if condition {
            (Status::Satisfied, Reason::RelationSatisfied)
        } else {
            (Status::Violated, Reason::RelationContradicted)
        }
    };
    let indeterminate = |reason| (Status::Indeterminate, reason);

    match constraint.relation {
        Relation::EqualTo | Relation::DifferentFrom => {
            let Some(expected) = constraint.expected_value.as_deref() else {
                return indeterminate(Reason::MalformedExpectedValue);
            };
            let equal = match payload {
                Payload::Text { value } => value == expected,
                Payload::Scalar {
                    value,
                    unit,
                    uncertainty,
                } => {
                    if !uncertainty.is_finite() || *uncertainty < 0.0 {
                        return indeterminate(Reason::InvalidUncertainty);
                    }
                    if constraint.unit.as_deref().is_some_and(|wanted| wanted != unit) {
                        return indeterminate(Reason::UnitMismatch);
                    }
                    let Ok(expected_number) = expected.parse::<f64>() else {
                        return indeterminate(Reason::MalformedExpectedValue);
                    };
                    if !value.is_finite() || !expected_number.is_finite() {
                        return indeterminate(Reason::NonFiniteMeasurement);
                    }
                    if *uncertainty > 0.0 {
                        if constraint.relation == Relation::EqualTo {
                            if (*value - expected_number).abs() + *uncertainty == 0.0 {
                                true
                            } else if (*value - expected_number).abs() > *uncertainty {
                                false
                            } else {
                                return indeterminate(Reason::UncertainEquality);
                            }
                        } else if (*value - expected_number).abs() > *uncertainty {
                            false
                        } else {
                            return indeterminate(Reason::UncertainEquality);
                        }
                    } else {
                        value == &expected_number
                    }
                }
                _ => return indeterminate(Reason::WrongPayloadKind),
            };
            if constraint.relation == Relation::EqualTo {
                satisfied(equal)
            } else {
                satisfied(!equal)
            }
        }
        Relation::WithinRange => {
            let Payload::Scalar {
                value,
                unit,
                uncertainty,
            } = payload
            else {
                return indeterminate(Reason::WrongPayloadKind);
            };
            if !value.is_finite() {
                return indeterminate(Reason::NonFiniteMeasurement);
            }
            if !uncertainty.is_finite() || *uncertainty < 0.0 {
                return indeterminate(Reason::InvalidUncertainty);
            }
            if constraint.unit.as_deref() != Some(unit.as_str()) {
                return indeterminate(Reason::UnitMismatch);
            }
            let (Some(minimum), Some(maximum)) = (constraint.minimum, constraint.maximum) else {
                return indeterminate(Reason::MalformedExpectedValue);
            };
            if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
                return indeterminate(Reason::MalformedExpectedValue);
            }
            let lower = *value - *uncertainty;
            let upper = *value + *uncertainty;
            if lower >= minimum && upper <= maximum {
                (Status::Satisfied, Reason::RelationSatisfied)
            } else if upper < minimum || lower > maximum {
                (Status::Violated, Reason::RelationContradicted)
            } else {
                indeterminate(Reason::BoundaryUncertainty)
            }
        }
        Relation::MemberOfSet | Relation::NotMemberOfSet => {
            let Payload::Text { value } = payload else {
                return indeterminate(Reason::WrongPayloadKind);
            };
            let member = constraint.allowed_values.binary_search(value).is_ok();
            if constraint.relation == Relation::MemberOfSet {
                satisfied(member)
            } else {
                satisfied(!member)
            }
        }
        Relation::Present | Relation::Absent => {
            let Payload::Presence { present } = payload else {
                return indeterminate(Reason::WrongPayloadKind);
            };
            if constraint.relation == Relation::Present {
                satisfied(*present)
            } else {
                satisfied(!present)
            }
        }
        Relation::NoChange => match payload {
            Payload::TextComparison { before, after } => satisfied(before == after),
            Payload::NumericComparison {
                before,
                after,
                unit,
                uncertainty,
            } => {
                if !before.is_finite() || !after.is_finite() {
                    return indeterminate(Reason::NonFiniteMeasurement);
                }
                if !uncertainty.is_finite() || *uncertainty < 0.0 {
                    return indeterminate(Reason::InvalidUncertainty);
                }
                if constraint.unit.as_deref().is_some_and(|wanted| wanted != unit) {
                    return indeterminate(Reason::UnitMismatch);
                }
                let difference = (before - after).abs();
                if difference == 0.0 && *uncertainty == 0.0 {
                    (Status::Satisfied, Reason::RelationSatisfied)
                } else if difference > *uncertainty {
                    (Status::Violated, Reason::RelationContradicted)
                } else {
                    indeterminate(Reason::UncertainEquality)
                }
            }
            _ => indeterminate(Reason::WrongPayloadKind),
        },
        Relation::MonotonicIncrease | Relation::MonotonicDecrease => {
            let Payload::NumericSeries {
                values,
                unit,
                uncertainty,
            } = payload
            else {
                return indeterminate(Reason::WrongPayloadKind);
            };
            if values.len() < 2 {
                return indeterminate(Reason::EmptyOrShortSeries);
            }
            if values.iter().any(|value| !value.is_finite()) {
                return indeterminate(Reason::NonFiniteMeasurement);
            }
            if !uncertainty.is_finite() || *uncertainty < 0.0 {
                return indeterminate(Reason::InvalidUncertainty);
            }
            if constraint.unit.as_deref().is_some_and(|wanted| wanted != unit) {
                return indeterminate(Reason::UnitMismatch);
            }
            let mut ambiguous = false;
            let mut changed = false;
            for pair in values.windows(2) {
                if constraint.relation == Relation::MonotonicIncrease {
                    if pair[1] + *uncertainty < pair[0] - *uncertainty {
                        return (Status::Violated, Reason::NonMonotonicSeries);
                    }
                    if pair[1] - *uncertainty < pair[0] + *uncertainty {
                        ambiguous = true;
                    }
                    if pair[1] - *uncertainty > pair[0] + *uncertainty {
                        changed = true;
                    }
                } else {
                    if pair[1] - uncertainty > pair[0] + uncertainty {
                        return (Status::Violated, Reason::NonMonotonicSeries);
                    }
                    if pair[1] + *uncertainty > pair[0] - *uncertainty {
                        ambiguous = true;
                    }
                    if pair[1] + *uncertainty < pair[0] - *uncertainty {
                        changed = true;
                    }
                }
            }
            if ambiguous {
                indeterminate(Reason::UncertainMonotonicity)
            } else {
                satisfied(changed)
            }
        }
    }
}

fn evaluate_one(
    constraint: &MusicEditValidityConstraintV1,
    section: MusicEditValidityConstraintSectionV1,
    observation: Option<&MusicEditValidityObservationV1>,
) -> MusicEditConstraintEvaluationV1 {
    use MusicEditValidityObservationReasonV1 as Reason;
    use MusicEditValidityObservationStatusV1 as Status;

    let Some(observation) = observation else {
        return MusicEditConstraintEvaluationV1 {
            constraint_id: constraint.constraint_id.clone(),
            section,
            status: Status::Indeterminate,
            reason: Reason::MissingObservation,
        };
    };
    if let Some(reason) = validate_observation_sources(constraint, &observation.sources) {
        return MusicEditConstraintEvaluationV1 {
            constraint_id: constraint.constraint_id.clone(),
            section,
            status: Status::Indeterminate,
            reason,
        };
    }
    let (status, reason) = relation_result(constraint, &observation.payload);
    MusicEditConstraintEvaluationV1 {
        constraint_id: constraint.constraint_id.clone(),
        section,
        status,
        reason,
    }
}

fn decision_from_outcomes(
    outcomes: &[MusicEditConstraintEvaluationV1],
) -> MusicEditValidityDecisionV1 {
    use MusicEditValidityDecisionV1 as Decision;
    use MusicEditValidityObservationStatusV1 as Status;
    if outcomes.iter().any(|outcome| outcome.status == Status::Violated) {
        Decision::Rejected
    } else if outcomes.iter().any(|outcome| outcome.status == Status::Indeterminate) {
        Decision::Abstain
    } else {
        Decision::Acceptable
    }
}

fn receipt_digest(
    receipt: &MusicEditValidityEvaluationReceiptV1,
) -> Result<String, MusicEditValidityEvaluationIssueV1> {
    let commitment = EvaluationReceiptCommitmentV1 {
        schema_version: &receipt.schema_version,
        evaluator_version: &receipt.evaluator_version,
        envelope_sha256: &receipt.envelope_sha256,
        candidate_output_sha256: &receipt.candidate_output_sha256,
        source_lineage_sha256: &receipt.source_lineage_sha256,
        observation_bundle_sha256: &receipt.observation_bundle_sha256,
        outcomes: &receipt.outcomes,
        decision: receipt.decision,
    };
    json_digest(b"evaluation-receipt", &commitment)
}

/// Evaluate each declared constraint exactly once. Missing observations become
/// Indeterminate. Duplicate/extra observations and digest substitution are
/// structural errors. Numeric uncertainty is conservative: its interval must
/// fit wholly inside an allowed range; overlap with a boundary abstains.
pub fn evaluate_music_edit_validity_v1(
    envelope: &MusicEditValidityEnvelopeV1,
    candidate_output_sha256: &str,
    source_lineage_sha256: &str,
    observations: &[MusicEditValidityObservationV1],
) -> Result<MusicEditValidityEvaluationReceiptV1, Vec<MusicEditValidityEvaluationIssueV1>> {
    let envelope_issues = validate_music_edit_validity_envelope_v1(envelope);
    if !envelope_issues.is_empty() {
        return Err(vec![MusicEditValidityEvaluationIssueV1::InvalidEnvelope(
            envelope_issues.iter().map(|issue| format!("{issue:?}")).collect(),
        )]);
    }
    if !is_sha256(candidate_output_sha256) {
        return Err(vec![MusicEditValidityEvaluationIssueV1::InvalidCandidateDigest]);
    }
    if !is_sha256(source_lineage_sha256) || source_lineage_sha256 != envelope.source_lineage_sha256 {
        return Err(vec![MusicEditValidityEvaluationIssueV1::SourceLineageMismatch]);
    }
    let envelope_sha256 = music_edit_validity_envelope_digest_v1(envelope).map_err(|issue| {
        vec![MusicEditValidityEvaluationIssueV1::InvalidEnvelope(vec![format!("{issue:?}")])]
    })?;

    let constraints = all_constraints(envelope);
    let expected_ids: BTreeSet<&str> = constraints
        .iter()
        .map(|(constraint, _)| constraint.constraint_id.as_str())
        .collect();
    let mut observation_map =
        BTreeMap::<&str, &MusicEditValidityObservationV1>::new();
    let mut issues = Vec::new();
    if !observations.windows(2).all(|pair| {
        pair[0].constraint_id.as_str() < pair[1].constraint_id.as_str()
    }) {
        issues.push(MusicEditValidityEvaluationIssueV1::NonCanonicalObservationOrder);
    }
    for observation in observations {
        if !expected_ids.contains(observation.constraint_id.as_str()) {
            issues.push(MusicEditValidityEvaluationIssueV1::UnexpectedObservation {
                constraint_id: observation.constraint_id.clone(),
            });
        }
        if observation_map
            .insert(observation.constraint_id.as_str(), observation)
            .is_some()
        {
            issues.push(MusicEditValidityEvaluationIssueV1::DuplicateObservation {
                constraint_id: observation.constraint_id.clone(),
            });
        }
        if observation.envelope_sha256 != envelope_sha256 {
            issues.push(MusicEditValidityEvaluationIssueV1::InvalidObservationEnvelopeDigest {
                constraint_id: observation.constraint_id.clone(),
            });
        }
        if observation.candidate_output_sha256 != candidate_output_sha256 {
            issues.push(MusicEditValidityEvaluationIssueV1::InvalidObservationCandidateDigest {
                constraint_id: observation.constraint_id.clone(),
            });
        }
        if observation.source_lineage_sha256 != source_lineage_sha256 {
            issues.push(MusicEditValidityEvaluationIssueV1::InvalidObservationLineage {
                constraint_id: observation.constraint_id.clone(),
            });
        }
        if let Some((constraint, _)) = constraints
            .iter()
            .find(|(constraint, _)| constraint.constraint_id == observation.constraint_id)
        {
            if observation.property != constraint.property || observation.scope != constraint.scope {
                issues.push(MusicEditValidityEvaluationIssueV1::InvalidObservationDigest {
                    constraint_id: observation.constraint_id.clone(),
                });
            }
        }
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let mut outcomes = Vec::with_capacity(constraints.len());
    for (constraint, section) in constraints {
        outcomes.push(evaluate_one(
            constraint,
            section,
            observation_map.get(constraint.constraint_id.as_str()).copied(),
        ));
    }
    let decision = decision_from_outcomes(&outcomes);
    let observation_bundle_sha256 =
        json_digest(b"observation-bundle", observations).map_err(|issue| vec![issue])?;
    let mut receipt = MusicEditValidityEvaluationReceiptV1 {
        schema_version: MUSIC_EDIT_VALIDITY_EVALUATOR_SCHEMA_V1.to_owned(),
        evaluator_version: MUSIC_EDIT_VALIDITY_EVALUATOR_VERSION_V1.to_owned(),
        envelope_sha256,
        candidate_output_sha256: candidate_output_sha256.to_owned(),
        source_lineage_sha256: source_lineage_sha256.to_owned(),
        observation_bundle_sha256,
        outcomes,
        decision,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = receipt_digest(&receipt).map_err(|issue| vec![issue])?;
    Ok(receipt)
}

/// Re-evaluate the supplied evidence and compare every receipt field. This
/// verifies the deterministic reducer result; it does not rerun audio/human
/// measurement adapters.
pub fn verify_music_edit_validity_evaluation_receipt_v1(
    receipt: &MusicEditValidityEvaluationReceiptV1,
    envelope: &MusicEditValidityEnvelopeV1,
    candidate_output_sha256: &str,
    source_lineage_sha256: &str,
    observations: &[MusicEditValidityObservationV1],
) -> Vec<MusicEditValidityEvaluationIssueV1> {
    let mut issues = Vec::new();
    if receipt.schema_version != MUSIC_EDIT_VALIDITY_EVALUATOR_SCHEMA_V1 {
        issues.push(MusicEditValidityEvaluationIssueV1::ReceiptSchemaMismatch);
    }
    let expected = match evaluate_music_edit_validity_v1(
        envelope,
        candidate_output_sha256,
        source_lineage_sha256,
        observations,
    ) {
        Ok(expected) => expected,
        Err(evaluation_issues) => return evaluation_issues,
    };
    if receipt.envelope_sha256 != expected.envelope_sha256 {
        issues.push(MusicEditValidityEvaluationIssueV1::ReceiptEnvelopeMismatch);
    }
    if receipt.candidate_output_sha256 != expected.candidate_output_sha256 {
        issues.push(MusicEditValidityEvaluationIssueV1::ReceiptCandidateMismatch);
    }
    if receipt.source_lineage_sha256 != expected.source_lineage_sha256 {
        issues.push(MusicEditValidityEvaluationIssueV1::ReceiptLineageMismatch);
    }
    if receipt.observation_bundle_sha256 != expected.observation_bundle_sha256 {
        issues.push(MusicEditValidityEvaluationIssueV1::ReceiptObservationBundleMismatch);
    }
    if receipt.outcomes != expected.outcomes || receipt.decision != expected.decision {
        issues.push(MusicEditValidityEvaluationIssueV1::ReceiptOutcomeMismatch);
    }
    match receipt_digest(receipt) {
        Ok(actual) if actual == receipt.receipt_sha256 => {}
        _ => issues.push(MusicEditValidityEvaluationIssueV1::ReceiptDigestMismatch),
    }
    issues
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::music_edit_benchmark_adapter::{
        MusicEditOperationV1, MusicEditSplitV1, MusicEditTargetScopeV1,
    };
    use crate::music_edit_validity_envelope::{
        MusicEditReferencePolicyV1, MusicEditValidityConstraintV1,
    };

    const CANDIDATE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const LINEAGE: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn digest(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn constraint(
        id: &str,
        property: MusicEditValidityPropertyV1,
        scope: MusicEditValidityScopeV1,
        relation: MusicEditValidityRelationV1,
        expected_value: Option<&str>,
        allowed_values: Vec<String>,
    ) -> MusicEditValidityConstraintV1 {
        MusicEditValidityConstraintV1 {
            constraint_id: id.to_owned(),
            property,
            scope,
            relation,
            expected_value: expected_value.map(str::to_owned),
            allowed_values,
            minimum: None,
            maximum: None,
            unit: None,
            evidence: MusicEditValidityEvidenceV1::Symbolic,
            rationale: format!("synthetic evaluator test: {id}"),
        }
    }

    fn envelope() -> MusicEditValidityEnvelopeV1 {
        MusicEditValidityEnvelopeV1 {
            schema_version: crate::music_edit_validity_envelope::MUSIC_EDIT_VALIDITY_ENVELOPE_SCHEMA_V1.to_owned(),
            envelope_id: "synthetic.evaluator.instrument-replacement".to_owned(),
            envelope_version: "v1".to_owned(),
            dataset_id: "synthetic-evaluator-dataset".to_owned(),
            dataset_version: "fixture-v1".to_owned(),
            benchmark_track_id: "development".to_owned(),
            sample_id: "sample-001".to_owned(),
            source_lineage_sha256: LINEAGE.to_owned(),
            split: MusicEditSplitV1::Development,
            operation: MusicEditOperationV1::ReplaceInstrument,
            target_scope: MusicEditTargetScopeV1::Instrument,
            target_selector_sha256: digest(0x10),
            requested_change: vec![
                constraint(
                    "change.from-original",
                    MusicEditValidityPropertyV1::InstrumentIdentity,
                    MusicEditValidityScopeV1::TargetPart,
                    MusicEditValidityRelationV1::DifferentFrom,
                    Some("clarinet"),
                    Vec::new(),
                ),
                constraint(
                    "change.instrument",
                    MusicEditValidityPropertyV1::InstrumentIdentity,
                    MusicEditValidityScopeV1::TargetPart,
                    MusicEditValidityRelationV1::MemberOfSet,
                    None,
                    vec!["flute".to_owned(), "oboe".to_owned()],
                ),
            ],
            protected_invariants: vec![constraint(
                "protect.part-lineage",
                MusicEditValidityPropertyV1::EventLineage,
                MusicEditValidityScopeV1::TargetPart,
                MusicEditValidityRelationV1::NoChange,
                None,
                Vec::new(),
            )],
            permitted_variations: Vec::new(),
            forbidden_changes: Vec::new(),
            references: Vec::new(),
            reference_policy: MusicEditReferencePolicyV1::ConstraintOnly,
        }
    }

    fn source(
        family: MusicEditObservationSourceFamilyV1,
        byte: u8,
    ) -> MusicEditObservationSourceV1 {
        MusicEditObservationSourceV1 {
            family,
            evidence_sha256: digest(byte),
            method_id: format!("synthetic.{family:?}.method"),
            method_sha256: digest(byte.saturating_add(1)),
        }
    }

    fn observations(
        envelope: &MusicEditValidityEnvelopeV1,
        candidate: &str,
    ) -> Vec<MusicEditValidityObservationV1> {
        let envelope_sha = music_edit_validity_envelope_digest_v1(envelope).unwrap();
        let mut rows = vec![
            MusicEditValidityObservationV1 {
                constraint_id: "change.from-original".to_owned(),
                property: MusicEditValidityPropertyV1::InstrumentIdentity,
                scope: MusicEditValidityScopeV1::TargetPart,
                envelope_sha256: envelope_sha.clone(),
                candidate_output_sha256: candidate.to_owned(),
                source_lineage_sha256: LINEAGE.to_owned(),
                sources: vec![source(MusicEditObservationSourceFamilyV1::Symbolic, 0x20)],
                payload: MusicEditObservationPayloadV1::Text {
                    value: "flute".to_owned(),
                },
            },
            MusicEditValidityObservationV1 {
                constraint_id: "change.instrument".to_owned(),
                property: MusicEditValidityPropertyV1::InstrumentIdentity,
                scope: MusicEditValidityScopeV1::TargetPart,
                envelope_sha256: envelope_sha.clone(),
                candidate_output_sha256: candidate.to_owned(),
                source_lineage_sha256: LINEAGE.to_owned(),
                sources: vec![source(MusicEditObservationSourceFamilyV1::Symbolic, 0x22)],
                payload: MusicEditObservationPayloadV1::Text {
                    value: "flute".to_owned(),
                },
            },
            MusicEditValidityObservationV1 {
                constraint_id: "protect.part-lineage".to_owned(),
                property: MusicEditValidityPropertyV1::EventLineage,
                scope: MusicEditValidityScopeV1::TargetPart,
                envelope_sha256: envelope_sha,
                candidate_output_sha256: candidate.to_owned(),
                source_lineage_sha256: LINEAGE.to_owned(),
                sources: vec![source(MusicEditObservationSourceFamilyV1::Symbolic, 0x24)],
                payload: MusicEditObservationPayloadV1::TextComparison {
                    before: "source-event-lineage".to_owned(),
                    after: "source-event-lineage".to_owned(),
                },
            },
        ];
        rows.sort_by(|left, right| left.constraint_id.cmp(&right.constraint_id));
        rows
    }

    fn evaluate(
        envelope: &MusicEditValidityEnvelopeV1,
        observations: &[MusicEditValidityObservationV1],
    ) -> MusicEditValidityEvaluationReceiptV1 {
        evaluate_music_edit_validity_v1(envelope, CANDIDATE, LINEAGE, observations).unwrap()
    }

    #[test]
    fn all_required_obligations_satisfied_and_receipt_replays() {
        let e = envelope();
        let o = observations(&e, CANDIDATE);
        let receipt = evaluate(&e, &o);
        assert_eq!(receipt.decision, MusicEditValidityDecisionV1::Acceptable);
        assert!(receipt.outcomes.iter().all(|outcome| {
            outcome.status == MusicEditValidityObservationStatusV1::Satisfied
        }));
        assert!(verify_music_edit_validity_evaluation_receipt_v1(
            &receipt, &e, CANDIDATE, LINEAGE, &o
        ).is_empty());
    }

    #[test]
    fn no_op_fails_requested_change_even_when_lineage_is_preserved() {
        let e = envelope();
        let mut o = observations(&e, CANDIDATE);
        o[0].payload = MusicEditObservationPayloadV1::Text {
            value: "clarinet".to_owned(),
        };
        let receipt = evaluate(&e, &o);
        assert_eq!(receipt.decision, MusicEditValidityDecisionV1::Rejected);
        assert!(receipt.outcomes.iter().any(|outcome| {
            outcome.constraint_id == "change.from-original"
                && outcome.status == MusicEditValidityObservationStatusV1::Violated
        }));
    }

    #[test]
    fn missing_observation_abstains_not_accepts() {
        let e = envelope();
        let mut o = observations(&e, CANDIDATE);
        o.pop();
        let receipt = evaluate(&e, &o);
        assert_eq!(receipt.decision, MusicEditValidityDecisionV1::Abstain);
        assert!(receipt.outcomes.iter().any(|outcome| {
            outcome.status == MusicEditValidityObservationStatusV1::Indeterminate
                && outcome.reason == MusicEditValidityObservationReasonV1::MissingObservation
        }));
    }

    #[test]
    fn changed_protected_event_lineage_rejects() {
        let e = envelope();
        let mut o = observations(&e, CANDIDATE);
        let protected = o.iter_mut()
            .find(|item| item.constraint_id == "protect.part-lineage")
            .unwrap();
        protected.payload = MusicEditObservationPayloadV1::TextComparison {
            before: "source-event-lineage".to_owned(),
            after: "new-event-lineage".to_owned(),
        };
        let receipt = evaluate(&e, &o);
        assert_eq!(receipt.decision, MusicEditValidityDecisionV1::Rejected);
    }

    #[test]
    fn candidate_or_lineage_substitution_is_structural_error() {
        let e = envelope();
        let mut o = observations(&e, CANDIDATE);
        o[0].candidate_output_sha256 = digest(0x77);
        assert!(evaluate_music_edit_validity_v1(&e, CANDIDATE, LINEAGE, &o).is_err());
        let o = observations(&e, CANDIDATE);
        assert!(evaluate_music_edit_validity_v1(&e, CANDIDATE, &digest(0x78), &o).is_err());
    }

    #[test]
    fn duplicate_or_extra_observations_reject() {
        let e = envelope();
        let mut o = observations(&e, CANDIDATE);
        o.push(o[0].clone());
        o.sort_by(|left, right| left.constraint_id.cmp(&right.constraint_id));
        assert!(evaluate_music_edit_validity_v1(&e, CANDIDATE, LINEAGE, &o).is_err());

        let mut o = observations(&e, CANDIDATE);
        o.push(MusicEditValidityObservationV1 {
            constraint_id: "not-in-envelope".to_owned(),
            property: MusicEditValidityPropertyV1::Harmony,
            scope: MusicEditValidityScopeV1::Composition,
            envelope_sha256: music_edit_validity_envelope_digest_v1(&e).unwrap(),
            candidate_output_sha256: CANDIDATE.to_owned(),
            source_lineage_sha256: LINEAGE.to_owned(),
            sources: vec![source(MusicEditObservationSourceFamilyV1::Symbolic, 0x50)],
            payload: MusicEditObservationPayloadV1::Text { value: "x".to_owned() },
        });
        o.sort_by(|left, right| left.constraint_id.cmp(&right.constraint_id));
        assert!(evaluate_music_edit_validity_v1(&e, CANDIDATE, LINEAGE, &o).is_err());
    }

    #[test]
    fn numeric_range_uncertainty_straddling_boundary_abstains() {
        let mut e = envelope();
        let mut range = constraint(
            "variation.dynamic-range",
            MusicEditValidityPropertyV1::DynamicContour,
            MusicEditValidityScopeV1::TargetPart,
            MusicEditValidityRelationV1::WithinRange,
            None,
            Vec::new(),
        );
        range.minimum = Some(-1.0);
        range.maximum = Some(1.0);
        range.unit = Some("dB".to_owned());
        range.evidence = MusicEditValidityEvidenceV1::Acoustic;
        e.permitted_variations.push(range);
        let mut o = observations(&e, CANDIDATE);
        o.push(MusicEditValidityObservationV1 {
            constraint_id: "variation.dynamic-range".to_owned(),
            property: MusicEditValidityPropertyV1::DynamicContour,
            scope: MusicEditValidityScopeV1::TargetPart,
            envelope_sha256: music_edit_validity_envelope_digest_v1(&e).unwrap(),
            candidate_output_sha256: CANDIDATE.to_owned(),
            source_lineage_sha256: LINEAGE.to_owned(),
            sources: vec![source(MusicEditObservationSourceFamilyV1::Acoustic, 0x60)],
            payload: MusicEditObservationPayloadV1::Scalar {
                value: 0.95,
                unit: "dB".to_owned(),
                uncertainty: 0.10,
            },
        });
        o.sort_by(|left, right| left.constraint_id.cmp(&right.constraint_id));
        let receipt = evaluate(&e, &o);
        assert_eq!(receipt.decision, MusicEditValidityDecisionV1::Abstain);
        assert!(receipt.outcomes.iter().any(|outcome| {
            outcome.constraint_id == "variation.dynamic-range"
                && outcome.reason == MusicEditValidityObservationReasonV1::BoundaryUncertainty
        }));
    }

    #[test]
    fn out_of_range_measurement_rejects_even_with_uncertainty() {
        let mut e = envelope();
        let mut range = constraint(
            "variation.dynamic-range",
            MusicEditValidityPropertyV1::DynamicContour,
            MusicEditValidityScopeV1::TargetPart,
            MusicEditValidityRelationV1::WithinRange,
            None,
            Vec::new(),
        );
        range.minimum = Some(-1.0);
        range.maximum = Some(1.0);
        range.unit = Some("dB".to_owned());
        range.evidence = MusicEditValidityEvidenceV1::Acoustic;
        e.permitted_variations.push(range);
        let mut o = observations(&e, CANDIDATE);
        o.push(MusicEditValidityObservationV1 {
            constraint_id: "variation.dynamic-range".to_owned(),
            property: MusicEditValidityPropertyV1::DynamicContour,
            scope: MusicEditValidityScopeV1::TargetPart,
            envelope_sha256: music_edit_validity_envelope_digest_v1(&e).unwrap(),
            candidate_output_sha256: CANDIDATE.to_owned(),
            source_lineage_sha256: LINEAGE.to_owned(),
            sources: vec![source(MusicEditObservationSourceFamilyV1::Acoustic, 0x61)],
            payload: MusicEditObservationPayloadV1::Scalar {
                value: 1.5,
                unit: "dB".to_owned(),
                uncertainty: 0.1,
            },
        });
        o.sort_by(|left, right| left.constraint_id.cmp(&right.constraint_id));
        assert_eq!(evaluate(&e, &o).decision, MusicEditValidityDecisionV1::Rejected);
    }

    #[test]
    fn forbidden_change_is_a_hard_rejection() {
        let mut e = envelope();
        let mut forbidden = constraint(
            "forbid.global-tempo",
            MusicEditValidityPropertyV1::Tempo,
            MusicEditValidityScopeV1::WholeMixture,
            MusicEditValidityRelationV1::NoChange,
            None,
            Vec::new(),
        );
        e.forbidden_changes.push(forbidden);
        let mut o = observations(&e, CANDIDATE);
        o.push(MusicEditValidityObservationV1 {
            constraint_id: "forbid.global-tempo".to_owned(),
            property: MusicEditValidityPropertyV1::Tempo,
            scope: MusicEditValidityScopeV1::WholeMixture,
            envelope_sha256: music_edit_validity_envelope_digest_v1(&e).unwrap(),
            candidate_output_sha256: CANDIDATE.to_owned(),
            source_lineage_sha256: LINEAGE.to_owned(),
            sources: vec![source(MusicEditObservationSourceFamilyV1::Symbolic, 0x62)],
            payload: MusicEditObservationPayloadV1::TextComparison {
                before: "120".to_owned(),
                after: "124".to_owned(),
            },
        });
        o.sort_by(|left, right| left.constraint_id.cmp(&right.constraint_id));
        assert_eq!(evaluate(&e, &o).decision, MusicEditValidityDecisionV1::Rejected);
    }

    #[test]
    fn monotonic_series_detects_reversal_as_violation_and_flatline_as_violation() {
        let mut e = envelope();
        let mut trend = constraint(
            "change.dynamic-contour",
            MusicEditValidityPropertyV1::DynamicContour,
            MusicEditValidityScopeV1::TargetPart,
            MusicEditValidityRelationV1::MonotonicIncrease,
            None,
            Vec::new(),
        );
        trend.evidence = MusicEditValidityEvidenceV1::Acoustic;
        e.requested_change.push(trend);
        e.requested_change.sort_by(|left, right| left.constraint_id.cmp(&right.constraint_id));
        let mut o = observations(&e, CANDIDATE);
        o.push(MusicEditValidityObservationV1 {
            constraint_id: "change.dynamic-contour".to_owned(),
            property: MusicEditValidityPropertyV1::DynamicContour,
            scope: MusicEditValidityScopeV1::TargetPart,
            envelope_sha256: music_edit_validity_envelope_digest_v1(&e).unwrap(),
            candidate_output_sha256: CANDIDATE.to_owned(),
            source_lineage_sha256: LINEAGE.to_owned(),
            sources: vec![source(MusicEditObservationSourceFamilyV1::Acoustic, 0x63)],
            payload: MusicEditObservationPayloadV1::NumericSeries {
                values: vec![0.0, 1.0, 0.5, 3.0],
                unit: "dB".to_owned(),
                uncertainty: 0.0,
            },
        });
        o.sort_by(|left, right| left.constraint_id.cmp(&right.constraint_id));
        let receipt = evaluate(&e, &o);
        assert_eq!(receipt.decision, MusicEditValidityDecisionV1::Rejected);
        assert!(receipt.outcomes.iter().any(|outcome| {
            outcome.reason == MusicEditValidityObservationReasonV1::NonMonotonicSeries
        }));

        let mut o = observations(&e, CANDIDATE);
        o.push(MusicEditValidityObservationV1 {
            constraint_id: "change.dynamic-contour".to_owned(),
            property: MusicEditValidityPropertyV1::DynamicContour,
            scope: MusicEditValidityScopeV1::TargetPart,
            envelope_sha256: music_edit_validity_envelope_digest_v1(&e).unwrap(),
            candidate_output_sha256: CANDIDATE.to_owned(),
            source_lineage_sha256: LINEAGE.to_owned(),
            sources: vec![source(MusicEditObservationSourceFamilyV1::Acoustic, 0x64)],
            payload: MusicEditObservationPayloadV1::NumericSeries {
                values: vec![0.0, 0.0, 0.0],
                unit: "dB".to_owned(),
                uncertainty: 0.0,
            },
        });
        o.sort_by(|left, right| left.constraint_id.cmp(&right.constraint_id));
        assert_eq!(evaluate(&e, &o).decision, MusicEditValidityDecisionV1::Rejected);
    }

    #[test]
    fn combined_evidence_requires_two_distinct_source_families() {
        let mut e = envelope();
        e.requested_change[0].evidence = MusicEditValidityEvidenceV1::Combined;
        let mut o = observations(&e, CANDIDATE);
        assert_eq!(evaluate(&e, &o).decision, MusicEditValidityDecisionV1::Abstain);
        o[0].sources.push(source(MusicEditObservationSourceFamilyV1::Acoustic, 0x70));
        o[0].sources.sort_by_key(|item| item.family);
        assert_eq!(evaluate(&e, &o).decision, MusicEditValidityDecisionV1::Acceptable);
    }

    #[test]
    fn observation_bundle_change_changes_receipt_commitment() {
        let e = envelope();
        let mut o = observations(&e, CANDIDATE);
        let before = evaluate(&e, &o);
        o[0].payload = MusicEditObservationPayloadV1::Text {
            value: "oboe".to_owned(),
        };
        let after = evaluate(&e, &o);
        assert_ne!(before.receipt_sha256, after.receipt_sha256);
    }
}
