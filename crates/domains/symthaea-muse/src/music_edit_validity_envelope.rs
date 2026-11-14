// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Contract model for musical edit validity envelopes and multiple references.
//! A paired target is one realization, not necessarily the only valid outcome.

use crate::music_edit_benchmark_adapter::{
    MusicEditOperationV1, MusicEditSplitV1, MusicEditTargetScopeV1,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const MUSIC_EDIT_VALIDITY_ENVELOPE_SCHEMA_V1: &str =
    "mel-music-edit-validity-envelope-v1";
const ENVELOPE_DOMAIN: &[u8] = b"luminous.mel.music-edit-validity-envelope.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditValidityPropertyV1 {
    InstrumentIdentity, DrumKitIdentity, EventPresence, EventLineage, MelodyPitch,
    Harmony, OnsetTicks, DurationTicks, Rhythm, Meter, Form, Tempo, VelocityContour,
    DynamicContour, TuningCents, TimbreFamily, StereoImage, GlobalLoudness,
    AudioArtifact, PhraseCoherence, Articulation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditValidityScopeV1 {
    TargetPart, EditedSupport, UnaffectedParts, WholeMixture, Composition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditValidityRelationV1 {
    EqualTo, DifferentFrom, WithinRange, MemberOfSet, NotMemberOfSet, Present,
    Absent, NoChange, MonotonicIncrease, MonotonicDecrease,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditValidityEvidenceV1 {
    Symbolic, Acoustic, HumanAdjudication, RendererBound, Combined,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditValidityConstraintV1 {
    pub constraint_id: String,
    pub property: MusicEditValidityPropertyV1,
    pub scope: MusicEditValidityScopeV1,
    pub relation: MusicEditValidityRelationV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_value: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_values: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    pub evidence: MusicEditValidityEvidenceV1,
    pub rationale: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditReferenceRoleV1 { ValidAlternative, Demonstration, NegativeControl }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditReferenceAdequacyV1 { Validated, Ambiguous, Inaudible, Invalid, Unreviewed }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditReferenceReviewV1 {
    /// Commitment to a review record; this is not an authenticated reviewer identity.
    pub review_evidence_sha256: String,
    pub reviewer_set_sha256: String,
    pub reviewer_count: u16,
    /// Attestation only; not cryptographic proof of independence.
    pub independent_reviewers_attested: bool,
    pub adjudication_method_id: String,
    pub reviewed_at_utc: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditValidityReferenceV1 {
    pub reference_id: String,
    pub role: MusicEditReferenceRoleV1,
    pub adequacy: MusicEditReferenceAdequacyV1,
    pub source_lineage_sha256: String,
    pub audio_sha256: String,
    pub renderer_profile_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symbolic_target_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub review: Option<MusicEditReferenceReviewV1>,
    pub notes: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditReferencePolicyV1 {
    ConstraintOnly,
    RequireReviewedAlternative,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditValidityEnvelopeV1 {
    pub schema_version: String,
    pub envelope_id: String,
    pub envelope_version: String,
    pub dataset_id: String,
    pub dataset_version: String,
    pub benchmark_track_id: String,
    pub sample_id: String,
    pub source_lineage_sha256: String,
    pub split: MusicEditSplitV1,
    pub operation: MusicEditOperationV1,
    pub target_scope: MusicEditTargetScopeV1,
    pub target_selector_sha256: String,
    pub requested_change: Vec<MusicEditValidityConstraintV1>,
    pub protected_invariants: Vec<MusicEditValidityConstraintV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub permitted_variations: Vec<MusicEditValidityConstraintV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub forbidden_changes: Vec<MusicEditValidityConstraintV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub references: Vec<MusicEditValidityReferenceV1>,
    pub reference_policy: MusicEditReferencePolicyV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstraintSection { Requested, Protected, Permitted, Forbidden }

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicEditValidityEnvelopeIssueV1 {
    WrongSchema, EmptyField(&'static str), MutableVersion, InvalidDigest(&'static str),
    EmptyRequestedChange, EmptyProtectedInvariants, DuplicateConstraintId(String),
    NonCanonicalConstraintOrder(&'static str), InvalidConstraintField(String),
    InvalidConstraintForSection { id: String, section: &'static str },
    DuplicateReferenceId(String), NonCanonicalReferenceOrder,
    ReferenceLineageMismatch(String), ValidAlternativeNotValidated(String),
    ValidatedReferenceMissingReview(String), InvalidReviewRecord(String),
    NegativeControlNotInvalid(String), RequiredAlternativeMissing,
    ConflictingConstraintSections {
        property: MusicEditValidityPropertyV1,
        scope: MusicEditValidityScopeV1,
    },
    OperationScopeMismatch(MusicEditOperationV1),
    MissingOperationRequirement { operation: MusicEditOperationV1, property: MusicEditValidityPropertyV1 },
    InvalidEnvelope, SerializationError,
}

impl std::fmt::Display for MusicEditValidityEnvelopeIssueV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for MusicEditValidityEnvelopeIssueV1 {}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require(issues: &mut Vec<MusicEditValidityEnvelopeIssueV1>, field: &'static str, value: &str) {
    if value.trim().is_empty() {
        issues.push(MusicEditValidityEnvelopeIssueV1::EmptyField(field));
    }
}

fn check_digest(
    issues: &mut Vec<MusicEditValidityEnvelopeIssueV1>,
    field: &'static str,
    value: &str,
) {
    if !is_sha256(value) {
        issues.push(MusicEditValidityEnvelopeIssueV1::InvalidDigest(field));
    }
}

fn validate_constraint(
    item: &MusicEditValidityConstraintV1,
    section: ConstraintSection,
) -> Vec<MusicEditValidityEnvelopeIssueV1> {
    use MusicEditValidityRelationV1 as R;
    let mut issues = Vec::new();
    let id = item.constraint_id.clone();
    let fields_valid = !id.trim().is_empty()
        && !item.rationale.trim().is_empty()
        && !item.unit.as_ref().is_some_and(|unit| unit.trim().is_empty());
    if !fields_valid {
        issues.push(MusicEditValidityEnvelopeIssueV1::InvalidConstraintField(id.clone()));
    }

    let payload_valid = match item.relation {
        R::EqualTo | R::DifferentFrom => {
            item.expected_value.as_ref().is_some_and(|value| !value.trim().is_empty())
                && item.allowed_values.is_empty()
                && item.minimum.is_none()
                && item.maximum.is_none()
                && item.unit.is_none()
        }
        R::WithinRange => {
            item.expected_value.is_none()
                && item.allowed_values.is_empty()
                && item.minimum.is_some_and(|value| value.is_finite())
                && item.maximum.is_some_and(|value| value.is_finite())
                && item.minimum <= item.maximum
                && item.unit.as_ref().is_some_and(|unit| !unit.trim().is_empty())
        }
        R::MemberOfSet | R::NotMemberOfSet => {
            item.expected_value.is_none()
                && !item.allowed_values.is_empty()
                && item.minimum.is_none()
                && item.maximum.is_none()
                && item.unit.is_none()
                && item.allowed_values.iter().all(|value| !value.trim().is_empty())
                && item.allowed_values.windows(2)
                    .all(|pair| pair[0].as_str() < pair[1].as_str())
        }
        R::Present | R::Absent | R::NoChange | R::MonotonicIncrease | R::MonotonicDecrease => {
            item.expected_value.is_none()
                && item.allowed_values.is_empty()
                && item.minimum.is_none()
                && item.maximum.is_none()
                && item.unit.is_none()
        }
    };
    if !payload_valid {
        issues.push(MusicEditValidityEnvelopeIssueV1::InvalidConstraintField(id.clone()));
    }

    let section_valid = match section {
        ConstraintSection::Requested => item.relation != R::NoChange,
        ConstraintSection::Protected => matches!(
            item.relation,
            R::EqualTo | R::WithinRange | R::MemberOfSet | R::NotMemberOfSet
                | R::Present | R::Absent | R::NoChange
        ),
        ConstraintSection::Permitted => {
            matches!(item.relation, R::EqualTo | R::WithinRange | R::MemberOfSet)
        }
        ConstraintSection::Forbidden => matches!(item.relation, R::NoChange | R::NotMemberOfSet),
    };
    if !section_valid {
        let label = match section {
            ConstraintSection::Requested => "requested_change",
            ConstraintSection::Protected => "protected_invariants",
            ConstraintSection::Permitted => "permitted_variations",
            ConstraintSection::Forbidden => "forbidden_changes",
        };
        issues.push(MusicEditValidityEnvelopeIssueV1::InvalidConstraintForSection {
            id,
            section: label,
        });
    }
    issues
}

fn validate_section(
    values: &[MusicEditValidityConstraintV1],
    section: ConstraintSection,
    label: &'static str,
    seen: &mut BTreeSet<String>,
) -> Vec<MusicEditValidityEnvelopeIssueV1> {
    let mut issues = Vec::new();
    if !values.windows(2).all(|pair| {
        pair[0].constraint_id.as_str() < pair[1].constraint_id.as_str()
    }) {
        issues.push(MusicEditValidityEnvelopeIssueV1::NonCanonicalConstraintOrder(label));
    }
    for value in values {
        if !seen.insert(value.constraint_id.clone()) {
            issues.push(MusicEditValidityEnvelopeIssueV1::DuplicateConstraintId(
                value.constraint_id.clone(),
            ));
        }
        issues.extend(validate_constraint(value, section));
    }
    issues
}


fn has_requested(
    envelope: &MusicEditValidityEnvelopeV1,
    property: MusicEditValidityPropertyV1,
    scope: MusicEditValidityScopeV1,
    relation: MusicEditValidityRelationV1,
) -> bool {
    envelope.requested_change.iter().any(|constraint| {
        constraint.property == property
            && constraint.scope == scope
            && constraint.relation == relation
    })
}

fn has_protected(
    envelope: &MusicEditValidityEnvelopeV1,
    property: MusicEditValidityPropertyV1,
    scope: MusicEditValidityScopeV1,
    relation: MusicEditValidityRelationV1,
) -> bool {
    envelope.protected_invariants.iter().any(|constraint| {
        constraint.property == property
            && constraint.scope == scope
            && constraint.relation == relation
    })
}

fn operation_scope_valid(envelope: &MusicEditValidityEnvelopeV1) -> bool {
    use MusicEditOperationV1 as O;
    use MusicEditTargetScopeV1 as T;
    match envelope.operation {
        O::RemoveInstrument | O::ReplaceInstrument | O::ExtractInstrument | O::DoubleInstrument
        | O::Staccato | O::Legato => envelope.target_scope == T::Instrument,
        O::RemoveDrums | O::ReplaceDrums | O::ExtractDrums => envelope.target_scope == T::DrumKit,
        O::Crescendo | O::Decrescendo | O::TuningCorrection440 => {
            matches!(envelope.target_scope, T::Instrument | T::Mixture)
        }
    }
}

fn validate_cross_section_conflicts(
    envelope: &MusicEditValidityEnvelopeV1,
) -> Vec<MusicEditValidityEnvelopeIssueV1> {
    use MusicEditValidityRelationV1 as R;

    let mut issues = Vec::new();
    let overlaps = |left: &MusicEditValidityConstraintV1,
                    right: &MusicEditValidityConstraintV1| {
        left.property == right.property && left.scope == right.scope
    };
    let mut add_conflict = |property, scope| {
        let issue = MusicEditValidityEnvelopeIssueV1::ConflictingConstraintSections {
            property,
            scope,
        };
        if !issues.contains(&issue) {
            issues.push(issue);
        }
    };

    for requested in &envelope.requested_change {
        let protected_conflict = envelope.protected_invariants.iter().any(|protected| {
            overlaps(requested, protected)
                && protected.relation == R::NoChange
                && requested.relation != R::NoChange
        });
        let forbidden_conflict = envelope.forbidden_changes.iter().any(|forbidden| {
            overlaps(requested, forbidden)
                && forbidden.relation == R::NoChange
                && requested.relation != R::NoChange
        });
        if protected_conflict || forbidden_conflict {
            add_conflict(requested.property, requested.scope);
        }
    }

    for permitted in &envelope.permitted_variations {
        let protected_conflict = envelope.protected_invariants.iter().any(|protected| {
            overlaps(permitted, protected) && protected.relation == R::NoChange
        });
        let forbidden_conflict = envelope.forbidden_changes.iter().any(|forbidden| {
            overlaps(permitted, forbidden) && forbidden.relation == R::NoChange
        });
        if protected_conflict || forbidden_conflict {
            add_conflict(permitted.property, permitted.scope);
        }
    }
    issues
}

/// Enforces a minimal semantic contract per declared operation. These checks
/// validate the envelope only; actual symbolic/audio observations need separate
/// operation-specific verifiers and renderer-bound evidence.
fn validate_operation_contract(
    envelope: &MusicEditValidityEnvelopeV1,
) -> Vec<MusicEditValidityEnvelopeIssueV1> {
    use MusicEditOperationV1 as O;
    use MusicEditValidityEvidenceV1 as E;
    use MusicEditValidityPropertyV1 as P;
    use MusicEditValidityRelationV1 as R;
    use MusicEditValidityScopeV1 as S;
    use MusicEditTargetScopeV1 as T;

    let mut issues = Vec::new();
    if !operation_scope_valid(envelope) {
        issues.push(MusicEditValidityEnvelopeIssueV1::OperationScopeMismatch(
            envelope.operation,
        ));
        return issues;
    }

    let missing = |property| MusicEditValidityEnvelopeIssueV1::MissingOperationRequirement {
        operation: envelope.operation,
        property,
    };

    match envelope.operation {
        O::RemoveInstrument => {
            let symbolic_absence = envelope.requested_change.iter().any(|constraint| {
                constraint.property == P::InstrumentIdentity
                    && constraint.scope == S::TargetPart
                    && constraint.relation == R::Absent
                    && matches!(constraint.evidence, E::Symbolic | E::Combined)
            });
            let acoustic_absence = envelope.requested_change.iter().any(|constraint| {
                constraint.property == P::EventPresence
                    && constraint.scope == S::TargetPart
                    && constraint.relation == R::Absent
                    && matches!(constraint.evidence, E::Acoustic | E::Combined)
            });
            if !symbolic_absence {
                issues.push(missing(P::InstrumentIdentity));
            }
            if !acoustic_absence {
                issues.push(missing(P::EventPresence));
            }
            if !has_protected(envelope, P::EventLineage, S::UnaffectedParts, R::NoChange) {
                issues.push(missing(P::EventLineage));
            }
        }
        O::ReplaceInstrument => {
            let allowed_identity = has_requested(envelope, P::InstrumentIdentity, S::TargetPart, R::MemberOfSet);
            let changes_source_identity = envelope.requested_change.iter().any(|constraint| {
                constraint.property == P::InstrumentIdentity
                    && constraint.scope == S::TargetPart
                    && constraint.relation == R::DifferentFrom
                    && constraint.expected_value.as_ref().is_some_and(|value| !value.trim().is_empty())
            });
            if !allowed_identity || !changes_source_identity {
                issues.push(missing(P::InstrumentIdentity));
            }
            for property in [P::EventLineage, P::MelodyPitch, P::OnsetTicks, P::DurationTicks] {
                if !has_protected(envelope, property, S::TargetPart, R::NoChange) {
                    issues.push(missing(property));
                }
            }
        }
        O::ExtractInstrument => {
            let identity_bound = has_requested(envelope, P::InstrumentIdentity, S::TargetPart, R::EqualTo)
                || has_requested(envelope, P::InstrumentIdentity, S::TargetPart, R::MemberOfSet);
            let target_present = envelope.requested_change.iter().any(|constraint| {
                constraint.property == P::EventPresence
                    && constraint.scope == S::TargetPart
                    && constraint.relation == R::Present
                    && matches!(constraint.evidence, E::Acoustic | E::Combined)
            });
            let non_target_absent = envelope.requested_change.iter().any(|constraint| {
                constraint.property == P::EventPresence
                    && constraint.scope == S::UnaffectedParts
                    && constraint.relation == R::Absent
                    && matches!(constraint.evidence, E::Acoustic | E::Combined)
            });
            if !identity_bound {
                issues.push(missing(P::InstrumentIdentity));
            }
            if !target_present || !non_target_absent {
                issues.push(missing(P::EventPresence));
            }
        }
        O::DoubleInstrument => {
            if !has_requested(envelope, P::InstrumentIdentity, S::TargetPart, R::MemberOfSet)
                || !has_requested(envelope, P::EventLineage, S::EditedSupport, R::Present)
            {
                issues.push(missing(P::InstrumentIdentity));
            }
            for property in [P::EventLineage, P::MelodyPitch, P::OnsetTicks, P::DurationTicks] {
                if !has_protected(envelope, property, S::TargetPart, R::NoChange) {
                    issues.push(missing(property));
                }
            }
        }
        O::RemoveDrums => {
            let symbolic_absence = envelope.requested_change.iter().any(|constraint| {
                constraint.property == P::DrumKitIdentity
                    && constraint.scope == S::TargetPart
                    && constraint.relation == R::Absent
                    && matches!(constraint.evidence, E::Symbolic | E::Combined)
            });
            let acoustic_absence = envelope.requested_change.iter().any(|constraint| {
                constraint.property == P::EventPresence
                    && constraint.scope == S::TargetPart
                    && constraint.relation == R::Absent
                    && matches!(constraint.evidence, E::Acoustic | E::Combined)
            });
            if !symbolic_absence {
                issues.push(missing(P::DrumKitIdentity));
            }
            if !acoustic_absence {
                issues.push(missing(P::EventPresence));
            }
            if !has_protected(envelope, P::Rhythm, S::UnaffectedParts, R::NoChange) {
                issues.push(missing(P::Rhythm));
            }
        }
        O::ReplaceDrums => {
            let allowed_kit = has_requested(envelope, P::DrumKitIdentity, S::TargetPart, R::MemberOfSet);
            let changes_source_kit = envelope.requested_change.iter().any(|constraint| {
                constraint.property == P::DrumKitIdentity
                    && constraint.scope == S::TargetPart
                    && constraint.relation == R::DifferentFrom
                    && constraint.expected_value.as_ref().is_some_and(|value| !value.trim().is_empty())
            });
            if !allowed_kit || !changes_source_kit {
                issues.push(missing(P::DrumKitIdentity));
            }
            for property in [P::EventLineage, P::Rhythm, P::OnsetTicks, P::DurationTicks] {
                if !has_protected(envelope, property, S::TargetPart, R::NoChange) {
                    issues.push(missing(property));
                }
            }
        }
        O::ExtractDrums => {
            let identity_bound = has_requested(envelope, P::DrumKitIdentity, S::TargetPart, R::EqualTo)
                || has_requested(envelope, P::DrumKitIdentity, S::TargetPart, R::MemberOfSet);
            let target_present = envelope.requested_change.iter().any(|constraint| {
                constraint.property == P::EventPresence
                    && constraint.scope == S::TargetPart
                    && constraint.relation == R::Present
                    && matches!(constraint.evidence, E::Acoustic | E::Combined)
            });
            let non_drum_absent = envelope.requested_change.iter().any(|constraint| {
                constraint.property == P::EventPresence
                    && constraint.scope == S::UnaffectedParts
                    && constraint.relation == R::Absent
                    && matches!(constraint.evidence, E::Acoustic | E::Combined)
            });
            if !identity_bound {
                issues.push(missing(P::DrumKitIdentity));
            }
            if !target_present || !non_drum_absent {
                issues.push(missing(P::EventPresence));
            }
        }
        O::Staccato | O::Legato => {
            let articulation = if envelope.operation == O::Staccato { "staccato" } else { "legato" };
            let has_articulation = envelope.requested_change.iter().any(|constraint| {
                constraint.property == P::Articulation
                    && constraint.scope == S::TargetPart
                    && constraint.relation == R::EqualTo
                    && constraint.expected_value.as_deref() == Some(articulation)
                    && matches!(constraint.evidence, E::Acoustic | E::Combined)
            });
            let has_duration_delta = envelope.requested_change.iter().any(|constraint| {
                constraint.property == P::DurationTicks
                    && constraint.scope == S::TargetPart
                    && constraint.relation == R::DifferentFrom
                    && constraint.expected_value.as_deref() == Some("source_event_duration")
                    && matches!(constraint.evidence, E::Symbolic | E::Combined)
            });
            let has_duration_range = envelope.requested_change.iter().any(|constraint| {
                constraint.property == P::DurationTicks
                    && constraint.scope == S::TargetPart
                    && constraint.relation == R::WithinRange
                    && constraint.unit.as_deref() == Some("ticks")
                    && constraint.minimum.is_some_and(|minimum| minimum >= 1.0 && minimum.fract() == 0.0)
                    && constraint.maximum.is_some_and(|maximum| maximum >= 1.0 && maximum.fract() == 0.0)
                    && matches!(constraint.evidence, E::Symbolic | E::Combined)
            });
            if !has_articulation {
                issues.push(missing(P::Articulation));
            }
            if !has_duration_delta || !has_duration_range {
                issues.push(missing(P::DurationTicks));
            }
            for property in [P::MelodyPitch, P::OnsetTicks] {
                if !has_protected(envelope, property, S::TargetPart, R::NoChange) {
                    issues.push(missing(property));
                }
            }
        }
        O::Crescendo | O::Decrescendo => {
            let scope = if envelope.target_scope == T::Instrument { S::TargetPart } else { S::WholeMixture };
            let increasing = envelope.operation == O::Crescendo;
            let relation = if increasing { R::MonotonicIncrease } else { R::MonotonicDecrease };
            let has_contour = envelope.requested_change.iter().any(|constraint| {
                constraint.property == P::DynamicContour
                    && constraint.scope == scope
                    && constraint.relation == relation
                    && matches!(constraint.evidence, E::Acoustic | E::Combined)
            });
            let has_audible_delta = envelope.requested_change.iter().any(|constraint| {
                constraint.property == P::DynamicContour
                    && constraint.scope == scope
                    && constraint.relation == R::WithinRange
                    && constraint.unit.as_deref() == Some("dB_delta")
                    && matches!(constraint.evidence, E::Acoustic | E::Combined)
                    && if increasing {
                        constraint.minimum.is_some_and(|minimum| minimum >= 3.0)
                            && constraint.maximum.is_some_and(|maximum| maximum >= 3.0)
                    } else {
                        constraint.minimum.is_some_and(|minimum| minimum <= -3.0)
                            && constraint.maximum.is_some_and(|maximum| maximum <= -3.0)
                    }
            });
            if !has_contour || !has_audible_delta {
                issues.push(missing(P::DynamicContour));
            }
        }
        O::TuningCorrection440 => {
            let scope = if envelope.target_scope == T::Instrument { S::TargetPart } else { S::WholeMixture };
            let has_cent_tolerance = envelope.requested_change.iter().any(|constraint| {
                constraint.property == P::TuningCents
                    && constraint.scope == scope
                    && constraint.relation == R::WithinRange
                    && constraint.unit.as_deref() == Some("cents")
                    && constraint.minimum.is_some_and(|minimum| (-5.0..=0.0).contains(&minimum))
                    && constraint.maximum.is_some_and(|maximum| (0.0..=5.0).contains(&maximum))
                    && matches!(constraint.evidence, E::Acoustic | E::Combined)
            });
            let anchors_a440 = envelope.requested_change.iter().any(|constraint| {
                constraint.property == P::TuningCents
                    && constraint.scope == scope
                    && constraint.relation == R::EqualTo
                    && constraint.expected_value.as_deref() == Some("A4=440Hz")
                    && matches!(constraint.evidence, E::Acoustic | E::Combined)
            });
            if !has_cent_tolerance || !anchors_a440 {
                issues.push(missing(P::TuningCents));
            }
            if !has_protected(envelope, P::MelodyPitch, scope, R::NoChange) {
                issues.push(missing(P::MelodyPitch));
            }
        }
    }
    issues
}

/// Checks the contract structure; it does not score audio or infer listener preference.
pub fn validate_music_edit_validity_envelope_v1(
    envelope: &MusicEditValidityEnvelopeV1,
) -> Vec<MusicEditValidityEnvelopeIssueV1> {
    use MusicEditReferenceAdequacyV1 as A;
    use MusicEditReferenceRoleV1 as R;

    let mut issues = Vec::new();
    if envelope.schema_version != MUSIC_EDIT_VALIDITY_ENVELOPE_SCHEMA_V1 {
        issues.push(MusicEditValidityEnvelopeIssueV1::WrongSchema);
    }
    for (field, value) in [
        ("envelope_id", envelope.envelope_id.as_str()),
        ("envelope_version", envelope.envelope_version.as_str()),
        ("dataset_id", envelope.dataset_id.as_str()),
        ("dataset_version", envelope.dataset_version.as_str()),
        ("benchmark_track_id", envelope.benchmark_track_id.as_str()),
        ("sample_id", envelope.sample_id.as_str()),
    ] {
        require(&mut issues, field, value);
    }
    if ["latest", "current", "stable"].contains(&envelope.envelope_version.to_ascii_lowercase().as_str())
        || ["latest", "current", "stable"].contains(&envelope.dataset_version.to_ascii_lowercase().as_str())
    {
        issues.push(MusicEditValidityEnvelopeIssueV1::MutableVersion);
    }
    check_digest(&mut issues, "source_lineage_sha256", &envelope.source_lineage_sha256);
    check_digest(&mut issues, "target_selector_sha256", &envelope.target_selector_sha256);
    if envelope.requested_change.is_empty() {
        issues.push(MusicEditValidityEnvelopeIssueV1::EmptyRequestedChange);
    }
    if envelope.protected_invariants.is_empty() {
        issues.push(MusicEditValidityEnvelopeIssueV1::EmptyProtectedInvariants);
    }

    let mut constraint_ids = BTreeSet::new();
    issues.extend(validate_section(
        &envelope.requested_change, ConstraintSection::Requested, "requested_change", &mut constraint_ids,
    ));
    issues.extend(validate_section(
        &envelope.protected_invariants, ConstraintSection::Protected, "protected_invariants", &mut constraint_ids,
    ));
    issues.extend(validate_section(
        &envelope.permitted_variations, ConstraintSection::Permitted, "permitted_variations", &mut constraint_ids,
    ));
    issues.extend(validate_section(
        &envelope.forbidden_changes, ConstraintSection::Forbidden, "forbidden_changes", &mut constraint_ids,
    ));

    if !envelope.references.windows(2).all(|pair| {
        pair[0].reference_id.as_str() < pair[1].reference_id.as_str()
    }) {
        issues.push(MusicEditValidityEnvelopeIssueV1::NonCanonicalReferenceOrder);
    }
    let mut reference_ids = BTreeSet::new();
    let mut valid_alternatives = 0_usize;
    for reference in &envelope.references {
        require(&mut issues, "reference_id", &reference.reference_id);
        require(&mut issues, "reference_notes", &reference.notes);
        if !reference_ids.insert(reference.reference_id.clone()) {
            issues.push(MusicEditValidityEnvelopeIssueV1::DuplicateReferenceId(
                reference.reference_id.clone(),
            ));
        }
        if reference.source_lineage_sha256 != envelope.source_lineage_sha256 {
            issues.push(MusicEditValidityEnvelopeIssueV1::ReferenceLineageMismatch(
                reference.reference_id.clone(),
            ));
        }
        check_digest(&mut issues, "reference_source_lineage_sha256", &reference.source_lineage_sha256);
        check_digest(&mut issues, "reference_audio_sha256", &reference.audio_sha256);
        check_digest(&mut issues, "reference_renderer_profile_sha256", &reference.renderer_profile_sha256);
        if let Some(digest) = reference.symbolic_target_sha256.as_deref() {
            check_digest(&mut issues, "reference_symbolic_target_sha256", digest);
        }

        match reference.role {
            R::ValidAlternative => {
                if reference.adequacy != A::Validated {
                    issues.push(MusicEditValidityEnvelopeIssueV1::ValidAlternativeNotValidated(
                        reference.reference_id.clone(),
                    ));
                } else {
                    valid_alternatives += 1;
                }
            }
            R::NegativeControl if reference.adequacy != A::Invalid => {
                issues.push(MusicEditValidityEnvelopeIssueV1::NegativeControlNotInvalid(
                    reference.reference_id.clone(),
                ));
            }
            R::Demonstration | R::NegativeControl => {}
        }

        if reference.adequacy == A::Validated {
            match reference.review.as_ref() {
                None => issues.push(MusicEditValidityEnvelopeIssueV1::ValidatedReferenceMissingReview(
                    reference.reference_id.clone(),
                )),
                Some(review) => {
                    check_digest(&mut issues, "review_evidence_sha256", &review.review_evidence_sha256);
                    check_digest(&mut issues, "reviewer_set_sha256", &review.reviewer_set_sha256);
                    if review.reviewer_count < 2
                        || !review.independent_reviewers_attested
                        || review.adjudication_method_id.trim().is_empty()
                        || review.reviewed_at_utc.trim().is_empty()
                    {
                        issues.push(MusicEditValidityEnvelopeIssueV1::InvalidReviewRecord(
                            reference.reference_id.clone(),
                        ));
                    }
                }
            }
        }
    }

    if envelope.reference_policy == MusicEditReferencePolicyV1::RequireReviewedAlternative
        && valid_alternatives == 0
    {
        issues.push(MusicEditValidityEnvelopeIssueV1::RequiredAlternativeMissing);
    }
    issues.extend(validate_cross_section_conflicts(envelope));
    issues.extend(validate_operation_contract(envelope));
    issues
}

/// A content commitment to a valid envelope. Review commitments are not reviewer authentication.
pub fn music_edit_validity_envelope_digest_v1(
    envelope: &MusicEditValidityEnvelopeV1,
) -> Result<String, MusicEditValidityEnvelopeIssueV1> {
    if !validate_music_edit_validity_envelope_v1(envelope).is_empty() {
        return Err(MusicEditValidityEnvelopeIssueV1::InvalidEnvelope);
    }
    let bytes = serde_json::to_vec(envelope)
        .map_err(|_| MusicEditValidityEnvelopeIssueV1::SerializationError)?;
    let mut hasher = Sha256::new();
    hasher.update(ENVELOPE_DOMAIN);
    hasher.update(b"\0");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> String { format!("{byte:02x}").repeat(32) }

    fn constraint(
        id: &str, property: MusicEditValidityPropertyV1,
        scope: MusicEditValidityScopeV1, relation: MusicEditValidityRelationV1,
    ) -> MusicEditValidityConstraintV1 {
        MusicEditValidityConstraintV1 {
            constraint_id: id.to_owned(),
            property, scope, relation,
            expected_value: None, allowed_values: Vec::new(),
            minimum: None, maximum: None, unit: None,
            evidence: MusicEditValidityEvidenceV1::Symbolic,
            rationale: format!("synthetic test constraint: {id}"),
        }
    }

    fn valid_review() -> MusicEditReferenceReviewV1 {
        MusicEditReferenceReviewV1 {
            review_evidence_sha256: digest(0x11),
            reviewer_set_sha256: digest(0x12),
            reviewer_count: 2,
            independent_reviewers_attested: true,
            adjudication_method_id: "blind-panel-v1".to_owned(),
            reviewed_at_utc: "2026-11-14T00:00:00Z".to_owned(),
        }
    }

    fn reference(id: &str, role: MusicEditReferenceRoleV1, adequacy: MusicEditReferenceAdequacyV1, byte: u8)
        -> MusicEditValidityReferenceV1 {
        MusicEditValidityReferenceV1 {
            reference_id: id.to_owned(), role, adequacy,
            source_lineage_sha256: digest(1),
            audio_sha256: digest(byte),
            renderer_profile_sha256: digest(2),
            symbolic_target_sha256: Some(digest(3)),
            review: (adequacy == MusicEditReferenceAdequacyV1::Validated).then(valid_review),
            notes: format!("synthetic reference {id}"),
        }
    }

    fn valid_envelope() -> MusicEditValidityEnvelopeV1 {
        let mut target = constraint(
            "change.instrument", MusicEditValidityPropertyV1::InstrumentIdentity,
            MusicEditValidityScopeV1::TargetPart, MusicEditValidityRelationV1::MemberOfSet,
        );
        target.allowed_values = vec!["flute".to_owned(), "oboe".to_owned()];
        let source_identity = MusicEditValidityConstraintV1 {
            constraint_id: "change.from-original".to_owned(),
            property: MusicEditValidityPropertyV1::InstrumentIdentity,
            scope: MusicEditValidityScopeV1::TargetPart,
            relation: MusicEditValidityRelationV1::DifferentFrom,
            expected_value: Some("clarinet".to_owned()),
            allowed_values: Vec::new(),
            minimum: None,
            maximum: None,
            unit: None,
            evidence: MusicEditValidityEvidenceV1::Symbolic,
            rationale: "the new instrument differs from the source instrument".to_owned(),
        };
        let variation = MusicEditValidityConstraintV1 {
            constraint_id: "variation.dynamic-range".to_owned(),
            property: MusicEditValidityPropertyV1::DynamicContour,
            scope: MusicEditValidityScopeV1::TargetPart,
            relation: MusicEditValidityRelationV1::WithinRange,
            expected_value: None, allowed_values: Vec::new(),
            minimum: Some(-3.0), maximum: Some(3.0), unit: Some("dB".to_owned()),
            evidence: MusicEditValidityEvidenceV1::Acoustic,
            rationale: "limited expressive variation is permitted".to_owned(),
        };
        let mut references = vec![
            reference("alt-a", MusicEditReferenceRoleV1::ValidAlternative, MusicEditReferenceAdequacyV1::Validated, 0x21),
            reference("alt-b", MusicEditReferenceRoleV1::ValidAlternative, MusicEditReferenceAdequacyV1::Validated, 0x22),
            reference("wrong-instrument", MusicEditReferenceRoleV1::NegativeControl, MusicEditReferenceAdequacyV1::Invalid, 0x23),
        ];
        references.sort_by(|a, b| a.reference_id.cmp(&b.reference_id));
        MusicEditValidityEnvelopeV1 {
            schema_version: MUSIC_EDIT_VALIDITY_ENVELOPE_SCHEMA_V1.to_owned(),
            envelope_id: "synthetic.instrument-replacement".to_owned(),
            envelope_version: "v1".to_owned(),
            dataset_id: "synthetic-validity-dataset".to_owned(),
            dataset_version: "synthetic-2026-11-14".to_owned(),
            benchmark_track_id: "public-development".to_owned(),
            sample_id: "sample-001".to_owned(),
            source_lineage_sha256: digest(1),
            split: MusicEditSplitV1::Development,
            operation: MusicEditOperationV1::ReplaceInstrument,
            target_scope: MusicEditTargetScopeV1::Instrument,
            target_selector_sha256: digest(4),
            requested_change: vec![source_identity, target],
            protected_invariants: vec![
                constraint("protect.harmony", MusicEditValidityPropertyV1::Harmony, MusicEditValidityScopeV1::Composition, MusicEditValidityRelationV1::NoChange),
                constraint("protect.melody", MusicEditValidityPropertyV1::MelodyPitch, MusicEditValidityScopeV1::UnaffectedParts, MusicEditValidityRelationV1::NoChange),
                constraint("protect.part-lineage", MusicEditValidityPropertyV1::EventLineage, MusicEditValidityScopeV1::TargetPart, MusicEditValidityRelationV1::NoChange),
                constraint("protect.part-pitch", MusicEditValidityPropertyV1::MelodyPitch, MusicEditValidityScopeV1::TargetPart, MusicEditValidityRelationV1::NoChange),
                constraint("protect.part-onset", MusicEditValidityPropertyV1::OnsetTicks, MusicEditValidityScopeV1::TargetPart, MusicEditValidityRelationV1::NoChange),
                constraint("protect.part-duration", MusicEditValidityPropertyV1::DurationTicks, MusicEditValidityScopeV1::TargetPart, MusicEditValidityRelationV1::NoChange),
            ],
            permitted_variations: vec![variation],
            forbidden_changes: vec![
                constraint("forbid.global-tempo", MusicEditValidityPropertyV1::Tempo, MusicEditValidityScopeV1::WholeMixture, MusicEditValidityRelationV1::NoChange),
            ],
            references,
            reference_policy: MusicEditReferencePolicyV1::RequireReviewedAlternative,
        }
    }

    #[test]
    fn multi_reference_envelope_and_digest_pass() {
        let envelope = valid_envelope();
        assert!(validate_music_edit_validity_envelope_v1(&envelope).is_empty());
        assert!(music_edit_validity_envelope_digest_v1(&envelope).is_ok());
    }


    #[test]
    fn operation_contract_requires_target_identity_and_event_lineage() {
        let mut envelope = valid_envelope();
        assert!(validate_music_edit_validity_envelope_v1(&envelope).is_empty());

        envelope.protected_invariants.retain(|constraint| {
            constraint.property != MusicEditValidityPropertyV1::EventLineage
        });
        assert!(validate_music_edit_validity_envelope_v1(&envelope).contains(
            &MusicEditValidityEnvelopeIssueV1::MissingOperationRequirement {
                operation: MusicEditOperationV1::ReplaceInstrument,
                property: MusicEditValidityPropertyV1::EventLineage,
            }
        ));

        envelope = valid_envelope();
        envelope.requested_change.retain(|constraint| {
            constraint.relation != MusicEditValidityRelationV1::DifferentFrom
        });
        assert!(validate_music_edit_validity_envelope_v1(&envelope).contains(
            &MusicEditValidityEnvelopeIssueV1::MissingOperationRequirement {
                operation: MusicEditOperationV1::ReplaceInstrument,
                property: MusicEditValidityPropertyV1::InstrumentIdentity,
            }
        ));
    }

    #[test]
    fn staccato_requires_symbolic_duration_and_acoustic_articulation_obligations() {
        let mut envelope = valid_envelope();
        envelope.operation = MusicEditOperationV1::Staccato;
        envelope.requested_change.clear();
        envelope.requested_change.push(MusicEditValidityConstraintV1 {
            constraint_id: "change.articulation".to_owned(),
            property: MusicEditValidityPropertyV1::Articulation,
            scope: MusicEditValidityScopeV1::TargetPart,
            relation: MusicEditValidityRelationV1::EqualTo,
            expected_value: Some("staccato".to_owned()),
            allowed_values: Vec::new(),
            minimum: None,
            maximum: None,
            unit: None,
            evidence: MusicEditValidityEvidenceV1::Acoustic,
            rationale: "the rendered articulation should be audibly staccato".to_owned(),
        });
        envelope.requested_change.push(MusicEditValidityConstraintV1 {
            constraint_id: "change.duration-delta".to_owned(),
            property: MusicEditValidityPropertyV1::DurationTicks,
            scope: MusicEditValidityScopeV1::TargetPart,
            relation: MusicEditValidityRelationV1::DifferentFrom,
            expected_value: Some("source_event_duration".to_owned()),
            allowed_values: Vec::new(),
            minimum: None,
            maximum: None,
            unit: None,
            evidence: MusicEditValidityEvidenceV1::Symbolic,
            rationale: "duration changes from its source-event value".to_owned(),
        });
        envelope.requested_change.push(MusicEditValidityConstraintV1 {
            constraint_id: "change.duration".to_owned(),
            property: MusicEditValidityPropertyV1::DurationTicks,
            scope: MusicEditValidityScopeV1::TargetPart,
            relation: MusicEditValidityRelationV1::WithinRange,
            expected_value: None,
            allowed_values: Vec::new(),
            minimum: Some(1.0),
            maximum: Some(120.0),
            unit: Some("ticks".to_owned()),
            evidence: MusicEditValidityEvidenceV1::Symbolic,
            rationale: "note durations are constrained in exact score ticks".to_owned(),
        });
        envelope.requested_change.sort_by(|a, b| a.constraint_id.cmp(&b.constraint_id));
        envelope.references.clear();
        envelope.reference_policy = MusicEditReferencePolicyV1::ConstraintOnly;
        assert!(validate_music_edit_validity_envelope_v1(&envelope).is_empty());
        envelope.requested_change[0].evidence = MusicEditValidityEvidenceV1::Symbolic;
        assert!(validate_music_edit_validity_envelope_v1(&envelope).contains(
            &MusicEditValidityEnvelopeIssueV1::MissingOperationRequirement {
                operation: MusicEditOperationV1::Staccato,
                property: MusicEditValidityPropertyV1::Articulation,
            }
        ));
    }

    #[test]
    fn crescendo_scope_must_be_explicit_and_acoustically_evidenced() {
        let mut envelope = valid_envelope();
        envelope.operation = MusicEditOperationV1::Crescendo;
        envelope.requested_change.clear();
        envelope.requested_change.push(MusicEditValidityConstraintV1 {
            constraint_id: "change.dynamic-contour".to_owned(),
            property: MusicEditValidityPropertyV1::DynamicContour,
            scope: MusicEditValidityScopeV1::TargetPart,
            relation: MusicEditValidityRelationV1::MonotonicIncrease,
            expected_value: None,
            allowed_values: Vec::new(),
            minimum: None,
            maximum: None,
            unit: None,
            evidence: MusicEditValidityEvidenceV1::Acoustic,
            rationale: "the selected part must have an increasing rendered dynamic contour".to_owned(),
        });
        envelope.requested_change.push(MusicEditValidityConstraintV1 {
            constraint_id: "change.dynamic-range".to_owned(),
            property: MusicEditValidityPropertyV1::DynamicContour,
            scope: MusicEditValidityScopeV1::TargetPart,
            relation: MusicEditValidityRelationV1::WithinRange,
            expected_value: None,
            allowed_values: Vec::new(),
            minimum: Some(6.0),
            maximum: Some(12.0),
            unit: Some("dB_delta".to_owned()),
            evidence: MusicEditValidityEvidenceV1::Acoustic,
            rationale: "the increase must reach a minimum audible dynamic change".to_owned(),
        });
        envelope.requested_change.sort_by(|a, b| a.constraint_id.cmp(&b.constraint_id));
        envelope.references.clear();
        envelope.reference_policy = MusicEditReferencePolicyV1::ConstraintOnly;
        assert!(validate_music_edit_validity_envelope_v1(&envelope).is_empty());
        envelope.target_scope = MusicEditTargetScopeV1::Mixture;
        assert!(validate_music_edit_validity_envelope_v1(&envelope).contains(
            &MusicEditValidityEnvelopeIssueV1::MissingOperationRequirement {
                operation: MusicEditOperationV1::Crescendo,
                property: MusicEditValidityPropertyV1::DynamicContour,
            }
        ));
    }

    #[test]
    fn constraint_only_evaluation_can_omit_reference_audio() {
        let mut envelope = valid_envelope();
        envelope.references.clear();
        envelope.reference_policy = MusicEditReferencePolicyV1::ConstraintOnly;
        assert!(validate_music_edit_validity_envelope_v1(&envelope).is_empty());
    }

    #[test]
    fn required_reference_policy_rejects_without_valid_alternative() {
        let mut envelope = valid_envelope();
        envelope.references.retain(|r| r.role != MusicEditReferenceRoleV1::ValidAlternative);
        assert!(validate_music_edit_validity_envelope_v1(&envelope)
            .contains(&MusicEditValidityEnvelopeIssueV1::RequiredAlternativeMissing));
    }

    #[test]
    fn unreviewed_alternative_rejects() {
        let mut envelope = valid_envelope();
        envelope.references[0].adequacy = MusicEditReferenceAdequacyV1::Unreviewed;
        envelope.references[0].review = None;
        assert!(validate_music_edit_validity_envelope_v1(&envelope)
            .contains(&MusicEditValidityEnvelopeIssueV1::ValidAlternativeNotValidated("alt-a".to_owned())));
    }

    #[test]
    fn lineage_mismatch_rejects() {
        let mut envelope = valid_envelope();
        envelope.references[0].source_lineage_sha256 = digest(9);
        assert!(validate_music_edit_validity_envelope_v1(&envelope)
            .contains(&MusicEditValidityEnvelopeIssueV1::ReferenceLineageMismatch("alt-a".to_owned())));
    }

    #[test]
    fn invalid_negative_control_cannot_be_promoted_to_valid() {
        let mut envelope = valid_envelope();
        let control = envelope.references.iter_mut()
            .find(|r| r.role == MusicEditReferenceRoleV1::NegativeControl).unwrap();
        control.adequacy = MusicEditReferenceAdequacyV1::Validated;
        control.review = Some(valid_review());
        assert!(validate_music_edit_validity_envelope_v1(&envelope)
            .contains(&MusicEditValidityEnvelopeIssueV1::NegativeControlNotInvalid("wrong-instrument".to_owned())));
    }

    #[test]
    fn duplicate_constraint_id_across_sections_rejects() {
        let mut envelope = valid_envelope();
        envelope.forbidden_changes[0].constraint_id = "protect.harmony".to_owned();
        assert!(validate_music_edit_validity_envelope_v1(&envelope)
            .contains(&MusicEditValidityEnvelopeIssueV1::DuplicateConstraintId("protect.harmony".to_owned())));
    }

    #[test]
    fn invalid_numeric_range_rejects() {
        let mut envelope = valid_envelope();
        envelope.permitted_variations[0].minimum = Some(5.0);
        envelope.permitted_variations[0].maximum = Some(-5.0);
        assert!(validate_music_edit_validity_envelope_v1(&envelope)
            .contains(&MusicEditValidityEnvelopeIssueV1::InvalidConstraintField("variation.dynamic-range".to_owned())));
    }

    #[test]
    fn digest_changes_when_alternative_bytes_change() {
        let mut envelope = valid_envelope();
        let before = music_edit_validity_envelope_digest_v1(&envelope).unwrap();
        envelope.references[0].audio_sha256 = digest(0x31);
        let after = music_edit_validity_envelope_digest_v1(&envelope).unwrap();
        assert_ne!(before, after);
    }

    #[test]
    fn mutable_versions_reject() {
        let mut envelope = valid_envelope();
        envelope.dataset_version = "latest".to_owned();
        assert!(validate_music_edit_validity_envelope_v1(&envelope)
            .contains(&MusicEditValidityEnvelopeIssueV1::MutableVersion));
    }


    #[test]
    fn drum_replacement_requires_kit_identity_and_preserved_event_lineage() {
        let mut envelope = valid_envelope();
        envelope.operation = MusicEditOperationV1::ReplaceDrums;
        envelope.target_scope = MusicEditTargetScopeV1::DrumKit;
        envelope.requested_change[0].property = MusicEditValidityPropertyV1::DrumKitIdentity;
        envelope.requested_change[0].allowed_values = vec!["kit-a".to_owned(), "kit-b".to_owned()];
        envelope.requested_change.push(MusicEditValidityConstraintV1 {
            constraint_id: "change.from-kit".to_owned(),
            property: MusicEditValidityPropertyV1::DrumKitIdentity,
            scope: MusicEditValidityScopeV1::TargetPart,
            relation: MusicEditValidityRelationV1::DifferentFrom,
            expected_value: Some("kit-original".to_owned()),
            allowed_values: Vec::new(),
            minimum: None,
            maximum: None,
            unit: None,
            evidence: MusicEditValidityEvidenceV1::Symbolic,
            rationale: "the replacement kit differs from the original kit".to_owned(),
        });
        envelope.requested_change.sort_by(|a, b| a.constraint_id.cmp(&b.constraint_id));
        envelope.protected_invariants.push(constraint(
            "protect.drum-events",
            MusicEditValidityPropertyV1::EventLineage,
            MusicEditValidityScopeV1::TargetPart,
            MusicEditValidityRelationV1::NoChange,
        ));
        envelope.protected_invariants.sort_by(|a, b| a.constraint_id.cmp(&b.constraint_id));
        assert!(validate_music_edit_validity_envelope_v1(&envelope).is_empty());
    }


    #[test]
    fn articulation_duration_bounds_must_use_positive_integer_ticks() {
        let mut envelope = valid_envelope();
        envelope.operation = MusicEditOperationV1::Staccato;
        envelope.requested_change.clear();
        envelope.requested_change.extend([
            MusicEditValidityConstraintV1 {
                constraint_id: "change.articulation".to_owned(),
                property: MusicEditValidityPropertyV1::Articulation,
                scope: MusicEditValidityScopeV1::TargetPart,
                relation: MusicEditValidityRelationV1::EqualTo,
                expected_value: Some("staccato".to_owned()),
                allowed_values: Vec::new(), minimum: None, maximum: None, unit: None,
                evidence: MusicEditValidityEvidenceV1::Acoustic,
                rationale: "audible staccato articulation".to_owned(),
            },
            MusicEditValidityConstraintV1 {
                constraint_id: "change.duration".to_owned(),
                property: MusicEditValidityPropertyV1::DurationTicks,
                scope: MusicEditValidityScopeV1::TargetPart,
                relation: MusicEditValidityRelationV1::WithinRange,
                expected_value: None, allowed_values: Vec::new(),
                minimum: Some(1.5), maximum: Some(120.0), unit: Some("ticks".to_owned()),
                evidence: MusicEditValidityEvidenceV1::Symbolic,
                rationale: "synthetic deliberately fractional tick range".to_owned(),
            },
        ]);
        envelope.requested_change.sort_by(|a, b| a.constraint_id.cmp(&b.constraint_id));
        envelope.references.clear();
        envelope.reference_policy = MusicEditReferencePolicyV1::ConstraintOnly;
        assert!(validate_music_edit_validity_envelope_v1(&envelope).contains(
            &MusicEditValidityEnvelopeIssueV1::MissingOperationRequirement {
                operation: MusicEditOperationV1::Staccato,
                property: MusicEditValidityPropertyV1::DurationTicks,
            }
        ));
    }

    #[test]
    fn tuning_correction_requires_a440_anchor_cent_tolerance_and_pitch_protection() {
        let mut envelope = valid_envelope();
        envelope.operation = MusicEditOperationV1::TuningCorrection440;
        envelope.requested_change.clear();
        envelope.requested_change.extend([
            MusicEditValidityConstraintV1 {
                constraint_id: "change.a440-anchor".to_owned(),
                property: MusicEditValidityPropertyV1::TuningCents,
                scope: MusicEditValidityScopeV1::TargetPart,
                relation: MusicEditValidityRelationV1::EqualTo,
                expected_value: Some("A4=440Hz".to_owned()),
                allowed_values: Vec::new(),
                minimum: None,
                maximum: None,
                unit: None,
                evidence: MusicEditValidityEvidenceV1::Acoustic,
                rationale: "the tuning reference is concert A at 440 Hz".to_owned(),
            },
            MusicEditValidityConstraintV1 {
                constraint_id: "change.cent-tolerance".to_owned(),
                property: MusicEditValidityPropertyV1::TuningCents,
                scope: MusicEditValidityScopeV1::TargetPart,
                relation: MusicEditValidityRelationV1::WithinRange,
                expected_value: None,
                allowed_values: Vec::new(),
                minimum: Some(-2.0),
                maximum: Some(2.0),
                unit: Some("cents".to_owned()),
                evidence: MusicEditValidityEvidenceV1::Acoustic,
                rationale: "the corrected result must fall within two cents of the reference".to_owned(),
            },
        ]);
        envelope.protected_invariants.push(constraint(
            "protect.equal-tempered-note",
            MusicEditValidityPropertyV1::MelodyPitch,
            MusicEditValidityScopeV1::TargetPart,
            MusicEditValidityRelationV1::NoChange,
        ));
        envelope.protected_invariants.sort_by(|a, b| a.constraint_id.cmp(&b.constraint_id));
        envelope.references.clear();
        envelope.reference_policy = MusicEditReferencePolicyV1::ConstraintOnly;
        assert!(validate_music_edit_validity_envelope_v1(&envelope).is_empty());

        envelope.requested_change[0].expected_value = Some("MIDI-69".to_owned());
        assert!(validate_music_edit_validity_envelope_v1(&envelope).contains(
            &MusicEditValidityEnvelopeIssueV1::MissingOperationRequirement {
                operation: MusicEditOperationV1::TuningCorrection440,
                property: MusicEditValidityPropertyV1::TuningCents,
            }
        ));
    }


    #[test]
    fn instrument_removal_requires_symbolic_part_absence_acoustic_absence_and_other_part_protection() {
        let mut envelope = valid_envelope();
        envelope.operation = MusicEditOperationV1::RemoveInstrument;
        envelope.requested_change = vec![
            MusicEditValidityConstraintV1 {
                constraint_id: "remove.part.identity".to_owned(),
                property: MusicEditValidityPropertyV1::InstrumentIdentity,
                scope: MusicEditValidityScopeV1::TargetPart,
                relation: MusicEditValidityRelationV1::Absent,
                expected_value: None, allowed_values: Vec::new(),
                minimum: None, maximum: None, unit: None,
                evidence: MusicEditValidityEvidenceV1::Symbolic,
                rationale: "the complete selected part is absent in the symbolic target".to_owned(),
            },
            MusicEditValidityConstraintV1 {
                constraint_id: "remove.part.acoustic".to_owned(),
                property: MusicEditValidityPropertyV1::EventPresence,
                scope: MusicEditValidityScopeV1::TargetPart,
                relation: MusicEditValidityRelationV1::Absent,
                expected_value: None, allowed_values: Vec::new(),
                minimum: None, maximum: None, unit: None,
                evidence: MusicEditValidityEvidenceV1::Acoustic,
                rationale: "no audible target-part residual remains in the rendered target".to_owned(),
            },
        ];
        envelope.requested_change.sort_by(|a, b| a.constraint_id.cmp(&b.constraint_id));
        envelope.protected_invariants = vec![constraint(
            "protect.other-lineage",
            MusicEditValidityPropertyV1::EventLineage,
            MusicEditValidityScopeV1::UnaffectedParts,
            MusicEditValidityRelationV1::NoChange,
        )];
        envelope.references.clear();
        envelope.reference_policy = MusicEditReferencePolicyV1::ConstraintOnly;
        assert!(validate_music_edit_validity_envelope_v1(&envelope).is_empty());

        envelope.requested_change[1].evidence = MusicEditValidityEvidenceV1::Symbolic;
        assert!(validate_music_edit_validity_envelope_v1(&envelope).contains(
            &MusicEditValidityEnvelopeIssueV1::MissingOperationRequirement {
                operation: MusicEditOperationV1::RemoveInstrument,
                property: MusicEditValidityPropertyV1::EventPresence,
            }
        ));
    }

    #[test]
    fn extract_instrument_requires_acoustic_target_presence_and_non_target_leakage_check() {
        let mut envelope = valid_envelope();
        envelope.operation = MusicEditOperationV1::ExtractInstrument;
        envelope.requested_change = vec![
            constraint("extract.instrument", MusicEditValidityPropertyV1::InstrumentIdentity,
                MusicEditValidityScopeV1::TargetPart, MusicEditValidityRelationV1::EqualTo),
            constraint("extract.target-present", MusicEditValidityPropertyV1::EventPresence,
                MusicEditValidityScopeV1::TargetPart, MusicEditValidityRelationV1::Present),
            constraint("extract.other-absent", MusicEditValidityPropertyV1::EventPresence,
                MusicEditValidityScopeV1::UnaffectedParts, MusicEditValidityRelationV1::Absent),
        ];
        envelope.requested_change[0].expected_value = Some("flute".to_owned());
        envelope.requested_change[1].evidence = MusicEditValidityEvidenceV1::Acoustic;
        envelope.requested_change[2].evidence = MusicEditValidityEvidenceV1::Acoustic;
        envelope.requested_change.sort_by(|a, b| a.constraint_id.cmp(&b.constraint_id));
        envelope.references.clear();
        envelope.reference_policy = MusicEditReferencePolicyV1::ConstraintOnly;
        assert!(validate_music_edit_validity_envelope_v1(&envelope).is_empty());
        envelope.requested_change[1].evidence = MusicEditValidityEvidenceV1::Symbolic;
        assert!(validate_music_edit_validity_envelope_v1(&envelope).contains(
            &MusicEditValidityEnvelopeIssueV1::MissingOperationRequirement {
                operation: MusicEditOperationV1::ExtractInstrument,
                property: MusicEditValidityPropertyV1::EventPresence,
            }
        ));
    }

    #[test]
    fn tuning_tolerance_wider_than_five_cents_rejects() {
        let mut envelope = valid_envelope();
        envelope.operation = MusicEditOperationV1::TuningCorrection440;
        envelope.requested_change = vec![
            MusicEditValidityConstraintV1 {
                constraint_id: "tuning.a440-anchor".to_owned(),
                property: MusicEditValidityPropertyV1::TuningCents,
                scope: MusicEditValidityScopeV1::TargetPart,
                relation: MusicEditValidityRelationV1::EqualTo,
                expected_value: Some("A4=440Hz".to_owned()),
                allowed_values: Vec::new(), minimum: None, maximum: None, unit: None,
                evidence: MusicEditValidityEvidenceV1::Acoustic,
                rationale: "the reference tuning anchor is concert A".to_owned(),
            },
            MusicEditValidityConstraintV1 {
                constraint_id: "tuning.cent-tolerance".to_owned(),
                property: MusicEditValidityPropertyV1::TuningCents,
                scope: MusicEditValidityScopeV1::TargetPart,
                relation: MusicEditValidityRelationV1::WithinRange,
                expected_value: None, allowed_values: Vec::new(),
                minimum: Some(-10.0), maximum: Some(5.0), unit: Some("cents".to_owned()),
                evidence: MusicEditValidityEvidenceV1::Acoustic,
                rationale: "the final tuning tolerance is bounded in cents".to_owned(),
            },
        ];
        envelope.protected_invariants = vec![constraint(
            "tuning.pitches-retained", MusicEditValidityPropertyV1::MelodyPitch,
            MusicEditValidityScopeV1::TargetPart, MusicEditValidityRelationV1::NoChange,
        )];
        envelope.references.clear();
        envelope.reference_policy = MusicEditReferencePolicyV1::ConstraintOnly;
        assert!(validate_music_edit_validity_envelope_v1(&envelope).contains(
            &MusicEditValidityEnvelopeIssueV1::MissingOperationRequirement {
                operation: MusicEditOperationV1::TuningCorrection440,
                property: MusicEditValidityPropertyV1::TuningCents,
            }
        ));
    }

    #[test]
    fn permitted_variation_cannot_overlap_a_protected_no_change_invariant() {
        let mut envelope = valid_envelope();
        envelope.permitted_variations.push(MusicEditValidityConstraintV1 {
            constraint_id: "variation.harmony".to_owned(),
            property: MusicEditValidityPropertyV1::Harmony,
            scope: MusicEditValidityScopeV1::Composition,
            relation: MusicEditValidityRelationV1::WithinRange,
            expected_value: None,
            allowed_values: Vec::new(),
            minimum: Some(0.0),
            maximum: Some(1.0),
            unit: Some("normalized".to_owned()),
            evidence: MusicEditValidityEvidenceV1::Symbolic,
            rationale: "deliberately conflicting synthetic variation".to_owned(),
        });
        envelope.permitted_variations.sort_by(|a, b| a.constraint_id.cmp(&b.constraint_id));
        assert!(validate_music_edit_validity_envelope_v1(&envelope).contains(
            &MusicEditValidityEnvelopeIssueV1::ConflictingConstraintSections {
                property: MusicEditValidityPropertyV1::Harmony,
                scope: MusicEditValidityScopeV1::Composition,
            }
        ));
    }

    #[test]
    fn noncanonical_reference_order_rejects() {
        let mut envelope = valid_envelope();
        envelope.references.reverse();
        assert!(validate_music_edit_validity_envelope_v1(&envelope)
            .contains(&MusicEditValidityEnvelopeIssueV1::NonCanonicalReferenceOrder));
    }

    #[test]
    fn requested_change_cannot_be_no_change() {
        let mut envelope = valid_envelope();
        envelope.requested_change[0].relation = MusicEditValidityRelationV1::NoChange;
        envelope.requested_change[0].allowed_values.clear();
        assert!(validate_music_edit_validity_envelope_v1(&envelope).iter().any(|issue| {
            matches!(issue, MusicEditValidityEnvelopeIssueV1::InvalidConstraintForSection { section: "requested_change", .. })
        }));
    }
}
