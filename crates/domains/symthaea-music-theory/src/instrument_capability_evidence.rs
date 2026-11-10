// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bound instrument capability claims.
//!
//! This module is deliberately separate from the instrument/asset inventory
//! and from orchestration feasibility. It qualifies claims about one exact
//! instrument instance/profile and their evidence. A generic family label,
//! GM program, filename, decoded WAV, or caller-supplied range is not by
//! itself a qualified performer/renderer capability.
//!
//! The validator checks provenance shape, scope, claim/value consistency,
//! promotion rules and derived-claim lineage. It does not authenticate
//! external sources, measure an instrument, prove license ownership, or
//! certify musical idiom. Those require independent evidence collection.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::orchestration::{
    InstrumentCapabilityV1, InstrumentKindV1, MAX_INSTRUMENT_POLYPHONY,
};

pub const INSTRUMENT_CAPABILITY_EVIDENCE_SCHEMA_V1: &str =
    "instrument-capability-evidence-v1";
pub const MAX_CAPABILITY_CLAIMS_V1: usize = 512;
const MAX_LINEAGE_DEPTH_V1: usize = 64;

/// Where a capability value came from. Evidence classes are not interchangeable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityEvidenceClassV1 {
    Measured,
    SourceCitedModelled,
    Derived,
    Estimated,
    Unverified,
}

/// Whether the claim may be consumed by a downstream hard feasibility gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityDispositionV1 {
    /// Eligible for hard checks only within the claim's exact scope.
    Qualified,
    /// Usable only when a downstream policy explicitly permits the condition.
    Conditional,
    /// Retained for research/display; never a hard feasibility pass.
    ResearchOnly,
    /// Evidence is insufficient or contradicted.
    Unqualified,
    /// The system must abstain from using this capability.
    Abstain,
}

/// Claim domain. Each kind has exactly one compatible value representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityKindV1 {
    SoundingRange,
    Tessitura,
    MaximumPolyphony,
    WrittenToSoundingTransposition,
    ArticulationSupport,
    RendererPatchMapping,
}

/// Scoped performance/realization conditions.
///
/// Missing dimensions remain unknown; no default performer level, technique,
/// renderer, patch or articulation is inferred from an instrument name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CapabilityScopeV1 {
    pub performer_level: Option<String>,
    pub technique_id: Option<String>,
    pub renderer_id: Option<String>,
    pub patch_id: Option<String>,
    pub articulation_id: Option<String>,
    pub velocity_min: Option<u8>,
    pub velocity_max: Option<u8>,
}

/// Typed values avoid interpreting a generic JSON number differently by caller.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "value_type", rename_all = "snake_case")]
pub enum CapabilityValueV1 {
    SoundingRange {
        lowest_midi: u8,
        highest_midi: u8,
    },
    Tessitura {
        lowest_midi: u8,
        highest_midi: u8,
    },
    MaximumPolyphony {
        simultaneous_notes: u8,
    },
    WrittenToSoundingTransposition {
        semitones: i8,
    },
    ArticulationSupport {
        articulation_id: String,
        supported: bool,
    },
    RendererPatchMapping {
        renderer_id: String,
        patch_id: String,
        status: RendererMappingStatusV1,
        asset_sha256: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RendererMappingStatusV1 {
    Exact,
    ExplicitFallback,
    Unavailable,
}

/// License/consent state for assets or human-contributed evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceUseStatusV1 {
    Authorized,
    AttributionRequired,
    Restricted,
    Unknown,
    NotApplicable,
}

/// Source citation or measurement artifact identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySourceV1 {
    /// URL, publication ID, repository+path, instrument log ID, or other stable locator.
    pub locator: String,
    /// Source revision, publication version, capture session or measurement run.
    pub revision: String,
    /// Optional content digest of the cited source/artifact.
    pub sha256: Option<String>,
}

/// One capability claim with exact scope and evidence lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstrumentCapabilityClaimV1 {
    pub id: String,
    pub kind: CapabilityKindV1,
    pub value: CapabilityValueV1,
    pub scope: CapabilityScopeV1,
    pub evidence_class: CapabilityEvidenceClassV1,
    pub disposition: CapabilityDispositionV1,
    pub source: Option<CapabilitySourceV1>,
    /// Required for measured claims; binds the captured measurement artifact.
    pub artifact_sha256: Option<String>,
    /// Required for measured and derived claims; names procedure or transform.
    pub method_or_transform: Option<String>,
    /// Required for derived claims; references source claims in this profile.
    pub parent_claim_ids: Vec<String>,
    pub use_status: EvidenceUseStatusV1,
    /// Stable receipt/reference for an attribution obligation, when required.
    pub attribution_record_id: Option<String>,
}

/// Evidence profile is bound to the same instance ID and kind as the
/// orchestration capability record; it does not silently replace that record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstrumentCapabilityEvidenceProfileV1 {
    pub schema: String,
    pub instrument_id: String,
    pub instrument_kind: InstrumentKindV1,
    pub profile_revision: String,
    pub claims: Vec<InstrumentCapabilityClaimV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityEvidenceIssueCodeV1 {
    InvalidSchema,
    EmptyInstrumentId,
    InstrumentIdMismatch,
    InstrumentKindMismatch,
    EmptyProfileRevision,
    ResourceLimitExceeded,
    EmptyClaimId,
    DuplicateClaimId,
    InvalidScope,
    InsufficientPhysicalCapabilityScope,
    InvalidVelocityScope,
    KindValueMismatch,
    InvalidRange,
    InvalidPolyphony,
    InvalidTransposition,
    EmptyArticulationId,
    RendererScopeMismatch,
    InvalidAssetDigest,
    InvalidSource,
    InvalidArtifactDigest,
    MissingMeasurementMethod,
    MissingDerivationMethod,
    MissingParentClaim,
    ParentClaimCycle,
    ExcessiveLineageDepth,
    PromotionWithoutEvidence,
    EstimatedOrUnverifiedQualified,
    RestrictedEvidenceQualified,
    MissingAttribution,
    FallbackMappingCannotBeHardQualified,
    UnavailableRendererMappingQualified,
    ContradictoryQualifiedClaims,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityEvidenceIssueV1 {
    pub code: CapabilityEvidenceIssueCodeV1,
    pub claim_id: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstrumentCapabilityEvidenceReportV1 {
    pub valid: bool,
    pub instrument_id: String,
    pub claim_count: usize,
    pub qualified_claim_count: usize,
    pub conditional_claim_count: usize,
    pub research_only_claim_count: usize,
    pub issues: Vec<CapabilityEvidenceIssueV1>,
}

impl InstrumentCapabilityEvidenceReportV1 {
    fn push(
        &mut self,
        code: CapabilityEvidenceIssueCodeV1,
        claim_id: Option<&str>,
        message: impl Into<String>,
    ) {
        self.issues.push(CapabilityEvidenceIssueV1 {
            code,
            claim_id: claim_id.map(str::to_owned),
            message: message.into(),
        });
        self.valid = false;
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn non_empty(value: &Option<String>) -> bool {
    value.as_ref().is_some_and(|text| !text.trim().is_empty())
}

fn value_matches_kind(kind: CapabilityKindV1, value: &CapabilityValueV1) -> bool {
    matches!(
        (kind, value),
        (CapabilityKindV1::SoundingRange, CapabilityValueV1::SoundingRange { .. })
            | (CapabilityKindV1::Tessitura, CapabilityValueV1::Tessitura { .. })
            | (
                CapabilityKindV1::MaximumPolyphony,
                CapabilityValueV1::MaximumPolyphony { .. }
            )
            | (
                CapabilityKindV1::WrittenToSoundingTransposition,
                CapabilityValueV1::WrittenToSoundingTransposition { .. }
            )
            | (
                CapabilityKindV1::ArticulationSupport,
                CapabilityValueV1::ArticulationSupport { .. }
            )
            | (
                CapabilityKindV1::RendererPatchMapping,
                CapabilityValueV1::RendererPatchMapping { .. }
            )
    )
}

fn validate_value(
    report: &mut InstrumentCapabilityEvidenceReportV1,
    claim: &InstrumentCapabilityClaimV1,
) {
    if !value_matches_kind(claim.kind, &claim.value) {
        report.push(
            CapabilityEvidenceIssueCodeV1::KindValueMismatch,
            Some(&claim.id),
            "claim kind and typed value variant do not match",
        );
        return;
    }
    match &claim.value {
        CapabilityValueV1::SoundingRange {
            lowest_midi,
            highest_midi,
        }
        | CapabilityValueV1::Tessitura {
            lowest_midi,
            highest_midi,
        } => {
            if lowest_midi > highest_midi || *highest_midi > 127 {
                report.push(
                    CapabilityEvidenceIssueCodeV1::InvalidRange,
                    Some(&claim.id),
                    "inclusive MIDI range requires 0 <= lowest_midi <= highest_midi <= 127",
                );
            }
        }
        CapabilityValueV1::MaximumPolyphony { simultaneous_notes } => {
            if *simultaneous_notes == 0 || *simultaneous_notes > MAX_INSTRUMENT_POLYPHONY {
                report.push(
                    CapabilityEvidenceIssueCodeV1::InvalidPolyphony,
                    Some(&claim.id),
                    "simultaneous-note capacity must be within the declared global limit",
                );
            }
        }
        CapabilityValueV1::WrittenToSoundingTransposition { semitones } => {
            if !(-48..=48).contains(semitones) {
                report.push(
                    CapabilityEvidenceIssueCodeV1::InvalidTransposition,
                    Some(&claim.id),
                    "written-to-sounding transposition must be within the v1 safety bound",
                );
            }
        }
        CapabilityValueV1::ArticulationSupport {
            articulation_id, ..
        } => {
            if articulation_id.trim().is_empty()
                || claim.scope.articulation_id.as_deref() != Some(articulation_id.as_str())
            {
                report.push(
                    CapabilityEvidenceIssueCodeV1::EmptyArticulationId,
                    Some(&claim.id),
                    "articulation support requires a non-empty articulation ID",
                );
            }
        }
        CapabilityValueV1::RendererPatchMapping {
            renderer_id,
            patch_id,
            status,
            asset_sha256,
        } => {
            if *status == RendererMappingStatusV1::Unavailable
                && claim.disposition == CapabilityDispositionV1::Qualified
            {
                report.push(
                    CapabilityEvidenceIssueCodeV1::UnavailableRendererMappingQualified,
                    Some(&claim.id),
                    "an unavailable renderer/patch mapping cannot be hard-qualified",
                );
            }
            if *status == RendererMappingStatusV1::ExplicitFallback
                && claim.disposition == CapabilityDispositionV1::Qualified
            {
                report.push(
                    CapabilityEvidenceIssueCodeV1::FallbackMappingCannotBeHardQualified,
                    Some(&claim.id),
                    "an explicit fallback is a conditional realization, not an exact instrument/patch qualification",
                );
            }
            if renderer_id.trim().is_empty()
                || patch_id.trim().is_empty()
                || claim.scope.renderer_id.as_deref() != Some(renderer_id.as_str())
                || claim.scope.patch_id.as_deref() != Some(patch_id.as_str())
            {
                report.push(
                    CapabilityEvidenceIssueCodeV1::RendererScopeMismatch,
                    Some(&claim.id),
                    "renderer/patch value must match non-empty renderer_id and patch_id in its exact scope",
                );
            }
            if asset_sha256
                .as_ref()
                .is_some_and(|digest| !is_sha256(digest))
            {
                report.push(
                    CapabilityEvidenceIssueCodeV1::InvalidAssetDigest,
                    Some(&claim.id),
                    "asset_sha256 must be 64 lowercase hexadecimal characters",
                );
            }
        }
    }
}

fn validate_source_and_promotion(
    report: &mut InstrumentCapabilityEvidenceReportV1,
    claim: &InstrumentCapabilityClaimV1,
) {
    if let Some(source) = &claim.source {
        if source.locator.trim().is_empty()
            || source.revision.trim().is_empty()
            || source.sha256.as_ref().is_some_and(|digest| !is_sha256(digest))
        {
            report.push(
                CapabilityEvidenceIssueCodeV1::InvalidSource,
                Some(&claim.id),
                "source locator/revision must be non-empty and any source digest must be valid",
            );
        }
    }

    if claim
        .artifact_sha256
        .as_ref()
        .is_some_and(|digest| !is_sha256(digest))
    {
        report.push(
            CapabilityEvidenceIssueCodeV1::InvalidArtifactDigest,
            Some(&claim.id),
            "artifact_sha256 must be 64 lowercase hexadecimal characters",
        );
    }

    match claim.evidence_class {
        CapabilityEvidenceClassV1::Measured => {
            if claim.artifact_sha256.is_none() || !non_empty(&claim.method_or_transform) {
                report.push(
                    CapabilityEvidenceIssueCodeV1::MissingMeasurementMethod,
                    Some(&claim.id),
                    "measured claims require an artifact digest and a non-empty measurement method",
                );
            }
        }
        CapabilityEvidenceClassV1::SourceCitedModelled => {
            if claim.source.is_none() {
                report.push(
                    CapabilityEvidenceIssueCodeV1::InvalidSource,
                    Some(&claim.id),
                    "source-cited/modelled claims require a source locator and revision",
                );
            }
        }
        CapabilityEvidenceClassV1::Derived => {
            if claim.parent_claim_ids.is_empty() || !non_empty(&claim.method_or_transform) {
                report.push(
                    CapabilityEvidenceIssueCodeV1::MissingDerivationMethod,
                    Some(&claim.id),
                    "derived claims require parent claim IDs and a versioned transformation/method",
                );
            }
        }
        CapabilityEvidenceClassV1::Estimated | CapabilityEvidenceClassV1::Unverified => {}
    }

    if claim.disposition == CapabilityDispositionV1::Qualified {
        if matches!(
            claim.evidence_class,
            CapabilityEvidenceClassV1::Estimated | CapabilityEvidenceClassV1::Unverified
        ) {
            report.push(
                CapabilityEvidenceIssueCodeV1::EstimatedOrUnverifiedQualified,
                Some(&claim.id),
                "estimated or unverified evidence cannot be promoted to a hard-qualified capability",
            );
        }
        if matches!(
            claim.evidence_class,
            CapabilityEvidenceClassV1::Measured
                | CapabilityEvidenceClassV1::SourceCitedModelled
                | CapabilityEvidenceClassV1::Derived
        ) {
            let has_required = match claim.evidence_class {
                CapabilityEvidenceClassV1::Measured => {
                    claim.artifact_sha256.is_some() && non_empty(&claim.method_or_transform)
                }
                CapabilityEvidenceClassV1::SourceCitedModelled => claim.source.is_some(),
                CapabilityEvidenceClassV1::Derived => {
                    !claim.parent_claim_ids.is_empty() && non_empty(&claim.method_or_transform)
                }
                CapabilityEvidenceClassV1::Estimated | CapabilityEvidenceClassV1::Unverified => false,
            };
            if !has_required {
                report.push(
                    CapabilityEvidenceIssueCodeV1::PromotionWithoutEvidence,
                    Some(&claim.id),
                    "qualified disposition requires the complete evidence fields for its evidence class",
                );
            }
        }
        if claim.use_status == EvidenceUseStatusV1::Restricted
            || claim.use_status == EvidenceUseStatusV1::Unknown
        {
            report.push(
                CapabilityEvidenceIssueCodeV1::RestrictedEvidenceQualified,
                Some(&claim.id),
                "restricted or unknown-use evidence cannot satisfy a qualified hard capability",
            );
        }
        if claim.use_status == EvidenceUseStatusV1::AttributionRequired
            && !non_empty(&claim.attribution_record_id)
        {
            report.push(
                CapabilityEvidenceIssueCodeV1::MissingAttribution,
                Some(&claim.id),
                "attribution-required evidence needs a stable attribution record ID before qualification",
            );
        }
    }
}

fn visit_lineage(
    id: &str,
    claims: &BTreeMap<&str, &InstrumentCapabilityClaimV1>,
    active: &mut BTreeSet<String>,
    done: &mut BTreeSet<String>,
    depth: usize,
) -> Result<(), CapabilityEvidenceIssueCodeV1> {
    if depth > MAX_LINEAGE_DEPTH_V1 {
        return Err(CapabilityEvidenceIssueCodeV1::ExcessiveLineageDepth);
    }
    if done.contains(id) {
        return Ok(());
    }
    if !active.insert(id.to_owned()) {
        return Err(CapabilityEvidenceIssueCodeV1::ParentClaimCycle);
    }
    if let Some(claim) = claims.get(id) {
        for parent in &claim.parent_claim_ids {
            if !claims.contains_key(parent.as_str()) {
                return Err(CapabilityEvidenceIssueCodeV1::MissingParentClaim);
            }
            visit_lineage(parent, claims, active, done, depth + 1)?;
        }
    }
    active.remove(id);
    done.insert(id.to_owned());
    Ok(())
}

/// Validate a source-bound capability profile against the exact instance
/// record that a downstream orchestration plan refers to.
///
/// This function validates evidence structure and promotion policy only. A
/// downstream consumer must still request the exact claim kind and scope it
/// needs; a qualified renderer claim does not imply a qualified playable range.
pub fn validate_instrument_capability_evidence(
    instrument: &InstrumentCapabilityV1,
    profile: &InstrumentCapabilityEvidenceProfileV1,
) -> InstrumentCapabilityEvidenceReportV1 {
    let mut report = InstrumentCapabilityEvidenceReportV1 {
        valid: true,
        instrument_id: instrument.id.clone(),
        claim_count: profile.claims.len(),
        qualified_claim_count: 0,
        conditional_claim_count: 0,
        research_only_claim_count: 0,
        issues: Vec::new(),
    };

    if profile.schema != INSTRUMENT_CAPABILITY_EVIDENCE_SCHEMA_V1 {
        report.push(
            CapabilityEvidenceIssueCodeV1::InvalidSchema,
            None,
            "unsupported instrument capability evidence schema",
        );
    }
    if instrument.id.trim().is_empty() || profile.instrument_id.trim().is_empty() {
        report.push(
            CapabilityEvidenceIssueCodeV1::EmptyInstrumentId,
            None,
            "instrument instance IDs must be non-empty",
        );
    }
    if instrument.id != profile.instrument_id {
        report.push(
            CapabilityEvidenceIssueCodeV1::InstrumentIdMismatch,
            None,
            "profile instrument_id must match the concrete orchestration instance ID",
        );
    }
    if instrument.kind != profile.instrument_kind {
        report.push(
            CapabilityEvidenceIssueCodeV1::InstrumentKindMismatch,
            None,
            "profile instrument_kind must match the concrete orchestration instance kind",
        );
    }
    if profile.profile_revision.trim().is_empty() {
        report.push(
            CapabilityEvidenceIssueCodeV1::EmptyProfileRevision,
            None,
            "profile_revision must be non-empty",
        );
    }
    if profile.claims.len() > MAX_CAPABILITY_CLAIMS_V1 {
        report.push(
            CapabilityEvidenceIssueCodeV1::ResourceLimitExceeded,
            None,
            "capability claim count exceeds the v1 resource limit",
        );
        return report;
    }

    let mut claims = BTreeMap::<&str, &InstrumentCapabilityClaimV1>::new();
    for claim in &profile.claims {
        if claim.id.trim().is_empty() {
            report.push(
                CapabilityEvidenceIssueCodeV1::EmptyClaimId,
                Some(&claim.id),
                "claim ID must be non-empty",
            );
        } else if claims.insert(claim.id.as_str(), claim).is_some() {
            report.push(
                CapabilityEvidenceIssueCodeV1::DuplicateClaimId,
                Some(&claim.id),
                "claim IDs must be unique within the profile",
            );
        }

        if claim.scope.performer_level.as_ref().is_some_and(|value| value.trim().is_empty())
            || claim.scope.technique_id.as_ref().is_some_and(|value| value.trim().is_empty())
            || claim.scope.renderer_id.as_ref().is_some_and(|value| value.trim().is_empty())
            || claim.scope.patch_id.as_ref().is_some_and(|value| value.trim().is_empty())
            || claim.scope.articulation_id.as_ref().is_some_and(|value| value.trim().is_empty())
        {
            report.push(
                CapabilityEvidenceIssueCodeV1::InvalidScope,
                Some(&claim.id),
                "optional scope fields must be absent or non-empty",
            );
        }
        if matches!(
            claim.kind,
            CapabilityKindV1::SoundingRange
                | CapabilityKindV1::Tessitura
                | CapabilityKindV1::MaximumPolyphony
                | CapabilityKindV1::ArticulationSupport
        ) && claim.scope.performer_level.is_none()
            && claim.scope.technique_id.is_none()
        {
            report.push(
                CapabilityEvidenceIssueCodeV1::InsufficientPhysicalCapabilityScope,
                Some(&claim.id),
                "physical capability claims require at least a performer-level or technique scope",
            );
        }
        if let (Some(min), Some(max)) = (claim.scope.velocity_min, claim.scope.velocity_max) {
            if min > max || max > 127 {
                report.push(
                    CapabilityEvidenceIssueCodeV1::InvalidVelocityScope,
                    Some(&claim.id),
                    "MIDI velocity scope requires 0 <= velocity_min <= velocity_max <= 127",
                );
            }
        }
        validate_value(&mut report, claim);
        validate_source_and_promotion(&mut report, claim);

        match claim.disposition {
            CapabilityDispositionV1::Qualified => report.qualified_claim_count += 1,
            CapabilityDispositionV1::Conditional => report.conditional_claim_count += 1,
            CapabilityDispositionV1::ResearchOnly => report.research_only_claim_count += 1,
            CapabilityDispositionV1::Unqualified | CapabilityDispositionV1::Abstain => {}
        }
    }

    for claim in &profile.claims {
        if claim.evidence_class == CapabilityEvidenceClassV1::Derived {
            for parent in &claim.parent_claim_ids {
                if !claims.contains_key(parent.as_str()) {
                    report.push(
                        CapabilityEvidenceIssueCodeV1::MissingParentClaim,
                        Some(&claim.id),
                        format!("derived claim references missing parent claim {parent}"),
                    );
                }
            }
        }
    }

    // Conflicting values for the same capability and exact scope remain
    // visible. Never average them or select one by insertion order.
    for (index, left) in profile.claims.iter().enumerate() {
        if left.disposition != CapabilityDispositionV1::Qualified {
            continue;
        }
        for right in profile.claims.iter().skip(index + 1) {
            if right.disposition == CapabilityDispositionV1::Qualified
                && left.kind == right.kind
                && left.scope == right.scope
                && left.value != right.value
            {
                report.push(
                    CapabilityEvidenceIssueCodeV1::ContradictoryQualifiedClaims,
                    Some(&right.id),
                    format!("claim conflicts with qualified claim {} at the same capability scope", left.id),
                );
            }
        }
    }

    let mut done = BTreeSet::new();
    for claim in &profile.claims {
        if claim.evidence_class != CapabilityEvidenceClassV1::Derived {
            continue;
        }
        let mut active = BTreeSet::new();
        if let Err(code) = visit_lineage(
            &claim.id,
            &claims,
            &mut active,
            &mut done,
            0,
        ) {
            report.push(
                code,
                Some(&claim.id),
                "derived capability lineage is missing, cyclic, or exceeds the v1 depth limit",
            );
        }
        if claim.disposition == CapabilityDispositionV1::Qualified
            && claim.parent_claim_ids.iter().any(|parent| {
                claims
                    .get(parent.as_str())
                    .is_some_and(|source| source.disposition != CapabilityDispositionV1::Qualified)
            })
        {
            report.push(
                CapabilityEvidenceIssueCodeV1::PromotionWithoutEvidence,
                Some(&claim.id),
                "qualified derived claim requires all direct parent claims to be qualified",
            );
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestration::{SoundingRangeV1};

    fn instrument() -> InstrumentCapabilityV1 {
        InstrumentCapabilityV1 {
            id: "cello-solo-01".into(),
            kind: InstrumentKindV1::Cello,
            sounding_range: SoundingRangeV1 {
                lowest_midi: 36,
                highest_midi: 88,
            },
            max_polyphony: 4,
        }
    }

    fn measured_range_claim() -> InstrumentCapabilityClaimV1 {
        InstrumentCapabilityClaimV1 {
            id: "cello-range-measured-2026-01".into(),
            kind: CapabilityKindV1::SoundingRange,
            value: CapabilityValueV1::SoundingRange {
                lowest_midi: 36,
                highest_midi: 88,
            },
            scope: CapabilityScopeV1 {
                performer_level: Some("professional".into()),
                technique_id: Some("ordinary_bowed".into()),
                ..Default::default()
            },
            evidence_class: CapabilityEvidenceClassV1::Measured,
            disposition: CapabilityDispositionV1::Qualified,
            source: None,
            artifact_sha256: Some("a".repeat(64)),
            method_or_transform: Some("chromatic playable-note protocol v1".into()),
            parent_claim_ids: Vec::new(),
            use_status: EvidenceUseStatusV1::NotApplicable,
            attribution_record_id: None,
        }
    }

    fn profile(claims: Vec<InstrumentCapabilityClaimV1>) -> InstrumentCapabilityEvidenceProfileV1 {
        InstrumentCapabilityEvidenceProfileV1 {
            schema: INSTRUMENT_CAPABILITY_EVIDENCE_SCHEMA_V1.into(),
            instrument_id: "cello-solo-01".into(),
            instrument_kind: InstrumentKindV1::Cello,
            profile_revision: "2026-11-10.1".into(),
            claims,
        }
    }

    #[test]
    fn accepts_a_scoped_measured_range_claim() {
        let report = validate_instrument_capability_evidence(
            &instrument(),
            &profile(vec![measured_range_claim()]),
        );
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.qualified_claim_count, 1);
    }

    #[test]
    fn rejects_pitch_values_outside_the_midi_seven_bit_domain() {
        let mut claim = measured_range_claim();
        claim.value = CapabilityValueV1::SoundingRange {
            lowest_midi: 36,
            highest_midi: 200,
        };
        let report = validate_instrument_capability_evidence(
            &instrument(),
            &profile(vec![claim]),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::InvalidRange
        }));
    }

    #[test]
    fn rejects_unscoped_physical_capability_claims() {
        let mut claim = measured_range_claim();
        claim.scope.performer_level = None;
        claim.scope.technique_id = None;
        let report = validate_instrument_capability_evidence(
            &instrument(),
            &profile(vec![claim]),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::InsufficientPhysicalCapabilityScope
        }));
    }

    #[test]
    fn rejects_estimated_claim_promoted_to_qualified() {
        let mut claim = measured_range_claim();
        claim.evidence_class = CapabilityEvidenceClassV1::Estimated;
        claim.artifact_sha256 = None;
        claim.method_or_transform = None;
        let report = validate_instrument_capability_evidence(&instrument(), &profile(vec![claim]));
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::EstimatedOrUnverifiedQualified
        }));
    }

    #[test]
    fn rejects_profile_instrument_id_or_kind_mismatch() {
        let mut wrong_id = profile(vec![measured_range_claim()]);
        wrong_id.instrument_id = "another-cello".into();
        let report = validate_instrument_capability_evidence(&instrument(), &wrong_id);
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::InstrumentIdMismatch
        }));

        let mut wrong_kind = profile(vec![measured_range_claim()]);
        wrong_kind.instrument_kind = InstrumentKindV1::Violin;
        let report = validate_instrument_capability_evidence(&instrument(), &wrong_kind);
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::InstrumentKindMismatch
        }));
    }

    #[test]
    fn rejects_renderer_patch_claim_with_mismatched_scope() {
        let mut claim = measured_range_claim();
        claim.id = "patch".into();
        claim.kind = CapabilityKindV1::RendererPatchMapping;
        claim.value = CapabilityValueV1::RendererPatchMapping {
            renderer_id: "fluidsynth-2".into(),
            patch_id: "soundfont:0:42".into(),
            status: RendererMappingStatusV1::Exact,
            asset_sha256: Some("b".repeat(64)),
        };
        let report = validate_instrument_capability_evidence(&instrument(), &profile(vec![claim]));
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::RendererScopeMismatch
        }));
    }

    #[test]
    fn rejects_unknown_or_cyclic_derived_lineage() {
        let mut parent = measured_range_claim();
        parent.disposition = CapabilityDispositionV1::Conditional;
        let mut derived = measured_range_claim();
        derived.id = "derived".into();
        derived.evidence_class = CapabilityEvidenceClassV1::Derived;
        derived.artifact_sha256 = None;
        derived.method_or_transform = Some("range-transform-v1".into());
        derived.parent_claim_ids = vec!["missing".into()];
        let report = validate_instrument_capability_evidence(
            &instrument(),
            &profile(vec![parent.clone(), derived]),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::MissingParentClaim
        }));

        parent.id = "parent".into();
        parent.evidence_class = CapabilityEvidenceClassV1::Derived;
        parent.artifact_sha256 = None;
        parent.method_or_transform = Some("identity-transform-v1".into());
        parent.parent_claim_ids = vec!["derived".into()];
        let mut child = measured_range_claim();
        child.id = "derived".into();
        child.evidence_class = CapabilityEvidenceClassV1::Derived;
        child.artifact_sha256 = None;
        child.method_or_transform = Some("range-transform-v1".into());
        child.parent_claim_ids = vec!["parent".into()];
        let report = validate_instrument_capability_evidence(
            &instrument(),
            &profile(vec![parent, child]),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::ParentClaimCycle
        }));
    }

    #[test]
    fn rejects_restricted_evidence_and_missing_attribution() {
        let mut restricted = measured_range_claim();
        restricted.use_status = EvidenceUseStatusV1::Restricted;
        let mut no_attribution = measured_range_claim();
        no_attribution.id = "attributed".into();
        no_attribution.use_status = EvidenceUseStatusV1::AttributionRequired;
        no_attribution.attribution_attached = false;
        let report = validate_instrument_capability_evidence(
            &instrument(),
            &profile(vec![restricted, no_attribution]),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::RestrictedEvidenceQualified
        }));
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::MissingAttribution
        }));
    }

    #[test]
    fn rejects_duplicate_ids_and_kind_value_mismatch() {
        let first = measured_range_claim();
        let mut second = measured_range_claim();
        second.kind = CapabilityKindV1::MaximumPolyphony;
        let report = validate_instrument_capability_evidence(
            &instrument(),
            &profile(vec![first, second]),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::DuplicateClaimId
        }));
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::KindValueMismatch
        }));
    }

    #[test]
    fn refuses_to_average_conflicting_qualified_values() {
        let first = measured_range_claim();
        let mut second = measured_range_claim();
        second.id = "range-conflict".into();
        second.value = CapabilityValueV1::SoundingRange {
            lowest_midi: 38,
            highest_midi: 86,
        };
        let report = validate_instrument_capability_evidence(
            &instrument(),
            &profile(vec![first, second]),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::ContradictoryQualifiedClaims
        }));
    }

    #[test]
    fn explicit_renderer_fallback_cannot_be_promoted_to_exact_qualification() {
        let mut claim = measured_range_claim();
        claim.id = "renderer-fallback".into();
        claim.kind = CapabilityKindV1::RendererPatchMapping;
        claim.value = CapabilityValueV1::RendererPatchMapping {
            renderer_id: "fluidsynth-2".into(),
            patch_id: "soundfont:0:42".into(),
            status: RendererMappingStatusV1::ExplicitFallback,
            asset_sha256: Some("c".repeat(64)),
        };
        claim.scope.renderer_id = Some("fluidsynth-2".into());
        claim.scope.patch_id = Some("soundfont:0:42".into());
        let report = validate_instrument_capability_evidence(
            &instrument(),
            &profile(vec![claim]),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::FallbackMappingCannotBeHardQualified
        }));
    }

    #[test]
    fn rejects_invalid_asset_digest_and_velocity_scope() {
        let mut claim = measured_range_claim();
        claim.id = "bad-scope".into();
        claim.scope.velocity_min = Some(100);
        claim.scope.velocity_max = Some(10);
        claim.kind = CapabilityKindV1::RendererPatchMapping;
        claim.value = CapabilityValueV1::RendererPatchMapping {
            renderer_id: "native-renderer".into(),
            patch_id: "cello".into(),
            status: RendererMappingStatusV1::Exact,
            asset_sha256: Some("not-a-digest".into()),
        };
        claim.scope.renderer_id = Some("native-renderer".into());
        claim.scope.patch_id = Some("cello".into());
        let report = validate_instrument_capability_evidence(&instrument(), &profile(vec![claim]));
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::InvalidAssetDigest
        }));
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::InvalidVelocityScope
        }));
    }
}
