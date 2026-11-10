// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bound instrument capability claims for score-conditioned orchestration.
//!
//! A capability is scoped to a concrete instrument model, performer/technique,
//! dynamic, renderer, patch and asset revision. A family label, GM program,
//! declared sample path or non-silent render is not sufficient evidence.
//!
//! This module does not discover assets or measure instruments. It validates
//! provenance supplied by the source inventory / qualification process and
//! materializes only explicitly qualified, scope-matched hard claims into the
//! existing renderer-independent InstrumentCapabilityV1 contract.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

use symthaea_music_theory::orchestration::{
    InstrumentCapabilityV1, InstrumentKindV1, SoundingRangeV1,
};

pub const INSTRUMENT_CAPABILITY_EVIDENCE_SCHEMA_V1: &str =
    "muse-instrument-capability-evidence-v1";
pub const INSTRUMENT_CAPABILITY_PROFILE_DIGEST_SCHEMA_V1: &str =
    "muse-instrument-capability-profile-sha256-v1";
pub const MAX_CAPABILITY_CLAIMS_V1: usize = 128;
pub const MAX_PARENT_EVIDENCE_IDS_V1: usize = 32;
pub const MAX_POLYPHONY_V1: u8 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceClassV1 {
    Measured,
    SourceCitedModelled,
    Derived,
    Estimated,
    Unverified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationDispositionV1 {
    Qualified,
    Conditional,
    ResearchOnly,
    Unqualified,
    Abstain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReusePermissionV1 {
    Permitted,
    Restricted,
    Unknown,
    NotApplicable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityScopeV1 {
    /// Exact model or instrument specification; never inferred from family.
    pub instrument_model: Option<String>,
    /// Explicit performer population/level where the claim depends on a player.
    pub performer_level: Option<String>,
    /// Technique or playing mode to which the claim applies.
    pub technique: Option<String>,
    /// Dynamic/velocity region where the claim applies.
    pub dynamic: Option<String>,
    /// Exact renderer/build and patch scope for renderability claims.
    pub renderer_id: Option<String>,
    pub patch_id: Option<String>,
    /// Exact sample/asset set when the claim depends on sample-backed rendering.
    pub asset_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceProvenanceV1 {
    pub evidence_id: String,
    pub evidence_class: EvidenceClassV1,
    pub source_uri: Option<String>,
    pub source_revision: Option<String>,
    /// SHA-256 of the captured evidence artifact, not a path/size manifest.
    pub content_sha256: Option<String>,
    /// Exact asset bytes used by this claim, distinct from the evidence-record digest.
    pub bound_asset_sha256: Option<String>,
    pub method: Option<String>,
    pub environment: Option<String>,
    pub parent_evidence_ids: Vec<String>,
    pub transformation_id: Option<String>,
    pub license: ReusePermissionV1,
    pub consent: ReusePermissionV1,
    /// Standard uncertainty for measured/estimated numeric claims, when known.
    pub uncertainty: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "claim", rename_all = "snake_case")]
pub enum CapabilityValueV1 {
    SoundingRange {
        lowest_midi: u8,
        highest_midi: u8,
    },
    Tessitura {
        lowest_midi: u8,
        highest_midi: u8,
    },
    MaxPolyphony {
        notes: u8,
    },
    WrittenToSoundingTransposition {
        semitones: i16,
    },
    ArticulationSupport {
        articulation: String,
        supported: bool,
    },
    ControllerSupport {
        controller: String,
        supported: bool,
    },
    RendererPatchAvailability {
        available: bool,
    },
    SampleCoverage {
        lowest_midi: u8,
        highest_midi: u8,
        velocity_layers: Vec<u8>,
        articulations: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityClaimV1 {
    pub claim_id: String,
    pub value: CapabilityValueV1,
    pub scope: CapabilityScopeV1,
    pub disposition: QualificationDispositionV1,
    /// Record identifying the independent qualification/review decision.
    /// Merely setting disposition=qualified is not sufficient.
    pub qualification_record_id: Option<String>,
    pub qualification_record_sha256: Option<String>,
    pub qualification_authority_id: Option<String>,
    pub qualification_record_uri: Option<String>,
    pub evidence: EvidenceProvenanceV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstrumentCapabilityEvidenceProfileV1 {
    pub schema: String,
    /// Concrete performer/instrument instance used by an orchestration plan.
    pub instrument_instance_id: String,
    pub instrument_kind: InstrumentKindV1,
    pub claims: Vec<CapabilityClaimV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityEvidenceIssueCodeV1 {
    InvalidSchema,
    EmptyInstrumentInstanceId,
    TooManyClaims,
    EmptyClaimId,
    DuplicateClaimId,
    EmptyEvidenceId,
    DuplicateEvidenceId,
    InvalidScope,
    InvalidCapabilityValue,
    MalformedContentDigest,
    MalformedQualificationRecordDigest,
    MissingQualificationAuthority,
    AssetDigestMismatch,
    QualifiedAssetWithoutLicense,
    EmptyClaims,
    InvalidUncertainty,
    MissingMeasuredEvidence,
    MissingSourceCitation,
    MissingDerivedLineage,
    UnknownDerivedParent,
    QualifiedClaimWithoutReviewRecord,
    UnqualifiedEvidencePromoted,
    QualifiedClaimHasRestrictedReuse,
    DuplicateQualifiedCapability,
    MissingQualifiedSoundingRange,
    MissingQualifiedPolyphony,
    MissingQualifiedRendererPatch,
    RendererPatchUnavailable,
    QualificationNotTrusted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityEvidenceIssueV1 {
    pub code: CapabilityEvidenceIssueCodeV1,
    pub claim_id: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityEvidenceReportV1 {
    pub valid: bool,
    pub claim_count: usize,
    pub qualified_claim_count: usize,
    pub issues: Vec<CapabilityEvidenceIssueV1>,
}

impl CapabilityEvidenceReportV1 {
    fn push(
        &mut self,
        code: CapabilityEvidenceIssueCodeV1,
        claim_id: Option<&str>,
        detail: impl Into<String>,
    ) {
        self.issues.push(CapabilityEvidenceIssueV1 {
            code,
            claim_id: claim_id.map(str::to_owned),
            detail: detail.into(),
        });
    }

    fn finish(mut self) -> Self {
        self.valid = self.issues.is_empty();
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityMaterializationErrorV1 {
    InvalidProfile,
    ProfileSerialization,
    ScopeMismatch,
    MissingQualifiedSoundingRange,
    MissingQualifiedPolyphony,
    DuplicateQualifiedSoundingRange,
    DuplicateQualifiedPolyphony,
    MissingQualifiedRendererPatch,
    RendererPatchUnavailable,
}

/// Evidence receipt kept beside the existing orchestration capability.
/// Downstream planners should retain this value instead of copying only the
/// range/polyphony numbers and losing their qualification lineage.
/// Approved qualification attestations supplied by a separately authenticated
/// trust registry. The registry loader must verify authority/signature or other
/// governance proof before constructing this set; this module checks only
/// exact authority + record ID + content-digest membership.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct QualificationTrustPolicyV1 {
    pub approved_records: BTreeSet<(String, String, String)>,
}

/// Opaque, materializer-issued capability receipt. Fields are crate-visible
/// for the planner, but external callers cannot construct or deserialize a
/// claimed-qualified value. External consumers use the read-only accessors.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QualifiedInstrumentCapabilityV1 {
    pub(crate) capability: InstrumentCapabilityV1,
    pub(crate) profile_sha256: String,
    pub(crate) profile_digest_schema: String,
    pub(crate) performance_scope: CapabilityScopeV1,
    pub(crate) renderer_scope: Option<CapabilityScopeV1>,
    pub(crate) claim_ids: Vec<String>,
    pub(crate) evidence_ids: Vec<String>,
    pub(crate) qualification_record_ids: Vec<String>,
    pub(crate) qualification_record_sha256s: Vec<String>,
}

impl QualifiedInstrumentCapabilityV1 {
    pub fn capability(&self) -> &InstrumentCapabilityV1 { &self.capability }
    pub fn profile_sha256(&self) -> &str { &self.profile_sha256 }
    pub fn profile_digest_schema(&self) -> &str { &self.profile_digest_schema }
    pub fn performance_scope(&self) -> &CapabilityScopeV1 { &self.performance_scope }
    pub fn renderer_scope(&self) -> Option<&CapabilityScopeV1> { self.renderer_scope.as_ref() }
    pub fn claim_ids(&self) -> &[String] { &self.claim_ids }
    pub fn evidence_ids(&self) -> &[String] { &self.evidence_ids }
    pub fn qualification_record_ids(&self) -> &[String] { &self.qualification_record_ids }
    pub fn qualification_record_sha256s(&self) -> &[String] { &self.qualification_record_sha256s }
}

pub fn instrument_capability_profile_sha256_v1(
    profile: &InstrumentCapabilityEvidenceProfileV1,
) -> Result<String, serde_json::Error> {
    let serialized = serde_json::to_vec(profile)?;
    let mut hasher = Sha256::new();
    hasher.update(INSTRUMENT_CAPABILITY_PROFILE_DIGEST_SCHEMA_V1.as_bytes());
    hasher.update([0]);
    hasher.update(serialized);
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn valid_sha256_hex_v1(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &Option<String>) -> bool {
    value.as_ref().is_some_and(|text| !text.trim().is_empty())
}

fn validate_scope(scope: &CapabilityScopeV1) -> bool {
    let optional_text_is_valid = [
        &scope.instrument_model,
        &scope.performer_level,
        &scope.technique,
        &scope.dynamic,
        &scope.renderer_id,
        &scope.patch_id,
    ]
    .into_iter()
    .all(|value| value.as_ref().is_none_or(|text| !text.trim().is_empty()));
    let patch_requires_renderer = scope.patch_id.is_none() || scope.renderer_id.is_some();
    let asset_requires_renderer_patch = scope.asset_sha256.is_none()
        || (scope.renderer_id.is_some() && scope.patch_id.is_some());
    let asset_digest_is_valid = scope
        .asset_sha256
        .as_ref()
        .is_none_or(|digest| valid_sha256_hex_v1(digest));
    optional_text_is_valid
        && patch_requires_renderer
        && asset_requires_renderer_patch
        && asset_digest_is_valid
}

fn validate_value(value: &CapabilityValueV1) -> bool {
    match value {
        CapabilityValueV1::SoundingRange {
            lowest_midi,
            highest_midi,
        }
        | CapabilityValueV1::Tessitura {
            lowest_midi,
            highest_midi,
        }
        | CapabilityValueV1::SampleCoverage {
            lowest_midi,
            highest_midi,
            ..
        } => lowest_midi <= highest_midi && *highest_midi <= 127,
        CapabilityValueV1::MaxPolyphony { notes } => {
            *notes > 0 && *notes <= MAX_POLYPHONY_V1
        }
        CapabilityValueV1::ArticulationSupport { articulation, .. } => {
            !articulation.trim().is_empty()
        }
        CapabilityValueV1::ControllerSupport { controller, .. } => {
            !controller.trim().is_empty()
        }
        CapabilityValueV1::RendererPatchAvailability { .. } => true,
        CapabilityValueV1::WrittenToSoundingTransposition { semitones } => {
            (-48..=48).contains(semitones)
        }
    }
}

fn validate_evidence(
    claim: &CapabilityClaimV1,
    report: &mut CapabilityEvidenceReportV1,
) {
    let evidence = &claim.evidence;
    if evidence.evidence_id.trim().is_empty() {
        report.push(
            CapabilityEvidenceIssueCodeV1::EmptyEvidenceId,
            Some(&claim.claim_id),
            "evidence_id must be non-empty",
        );
    }
    if evidence
        .content_sha256
        .as_ref()
        .is_some_and(|digest| !valid_sha256_hex_v1(digest))
        || evidence
            .bound_asset_sha256
            .as_ref()
            .is_some_and(|digest| !valid_sha256_hex_v1(digest))
    {
        report.push(
            CapabilityEvidenceIssueCodeV1::MalformedContentDigest,
            Some(&claim.claim_id),
            "evidence and bound-asset digests must be 64 lowercase hexadecimal characters",
        );
    }
    if let Some(asset_sha256) = &claim.scope.asset_sha256 {
        if evidence.bound_asset_sha256.as_ref() != Some(asset_sha256) {
            report.push(
                CapabilityEvidenceIssueCodeV1::AssetDigestMismatch,
                Some(&claim.claim_id),
                "asset-scoped claims must bind the exact asset SHA-256 in their evidence record",
            );
        }
        if claim.disposition == QualificationDispositionV1::Qualified
            && evidence.license != ReusePermissionV1::Permitted
        {
            report.push(
                CapabilityEvidenceIssueCodeV1::QualifiedAssetWithoutLicense,
                Some(&claim.claim_id),
                "qualified asset-scoped claims require explicit permitted license status",
            );
        }
    }
    if evidence
        .uncertainty
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        report.push(
            CapabilityEvidenceIssueCodeV1::InvalidUncertainty,
            Some(&claim.claim_id),
            "uncertainty must be finite and non-negative",
        );
    }

    match evidence.evidence_class {
        EvidenceClassV1::Measured => {
            if !nonempty(&evidence.content_sha256) || !nonempty(&evidence.method) || !nonempty(&evidence.environment) {
                report.push(
                    CapabilityEvidenceIssueCodeV1::MissingMeasuredEvidence,
                    Some(&claim.claim_id),
                    "measured evidence requires a content digest, measurement method and environment",
                );
            }
        }
        EvidenceClassV1::SourceCitedModelled => {
            if !nonempty(&evidence.source_uri) || !nonempty(&evidence.source_revision) {
                report.push(
                    CapabilityEvidenceIssueCodeV1::MissingSourceCitation,
                    Some(&claim.claim_id),
                    "source-cited/modelled evidence requires a source URI and pinned source revision",
                );
            }
        }
        EvidenceClassV1::Derived => {
            if evidence.parent_evidence_ids.is_empty()
                || evidence.parent_evidence_ids.len() > MAX_PARENT_EVIDENCE_IDS_V1
                || !nonempty(&evidence.transformation_id)
            {
                report.push(
                    CapabilityEvidenceIssueCodeV1::MissingDerivedLineage,
                    Some(&claim.claim_id),
                    "derived evidence requires bounded parent evidence IDs and a transformation ID/version",
                );
            }
        }
        EvidenceClassV1::Estimated | EvidenceClassV1::Unverified => {}
    }

    if claim.disposition == QualificationDispositionV1::Qualified {
        if !nonempty(&claim.qualification_record_id) {
            report.push(
                CapabilityEvidenceIssueCodeV1::QualifiedClaimWithoutReviewRecord,
                Some(&claim.claim_id),
                "qualified claims require an independent qualification_record_id",
            );
        }
        if !claim
            .qualification_record_sha256
            .as_ref()
            .is_some_and(|digest| valid_sha256_hex_v1(digest))
        {
            report.push(
                CapabilityEvidenceIssueCodeV1::MalformedQualificationRecordDigest,
                Some(&claim.claim_id),
                "qualified claims require a 64-character lowercase SHA-256 of the qualification record",
            );
        }
        if !nonempty(&claim.qualification_authority_id)
            || !nonempty(&claim.qualification_record_uri)
        {
            report.push(
                CapabilityEvidenceIssueCodeV1::MissingQualificationAuthority,
                Some(&claim.claim_id),
                "qualified claims require an authority identifier and qualification record URI",
            );
        }
        if matches!(
            evidence.evidence_class,
            EvidenceClassV1::Estimated | EvidenceClassV1::Unverified
        ) {
            report.push(
                CapabilityEvidenceIssueCodeV1::UnqualifiedEvidencePromoted,
                Some(&claim.claim_id),
                "estimated/unverified evidence cannot satisfy a qualified hard capability claim",
            );
        }
        if matches!(
            evidence.license,
            ReusePermissionV1::Restricted | ReusePermissionV1::Unknown
        ) || matches!(
            evidence.consent,
            ReusePermissionV1::Restricted | ReusePermissionV1::Unknown
        ) {
            report.push(
                CapabilityEvidenceIssueCodeV1::QualifiedClaimHasRestrictedReuse,
                Some(&claim.claim_id),
                "restricted or unknown license/consent cannot qualify a planner-consumed capability",
            );
        }
    }
}

pub fn validate_instrument_capability_evidence_v1(
    profile: &InstrumentCapabilityEvidenceProfileV1,
) -> CapabilityEvidenceReportV1 {
    let mut report = CapabilityEvidenceReportV1 {
        valid: false,
        claim_count: profile.claims.len(),
        qualified_claim_count: profile
            .claims
            .iter()
            .filter(|claim| claim.disposition == QualificationDispositionV1::Qualified)
            .count(),
        issues: Vec::new(),
    };

    if profile.schema != INSTRUMENT_CAPABILITY_EVIDENCE_SCHEMA_V1 {
        report.push(
            CapabilityEvidenceIssueCodeV1::InvalidSchema,
            None,
            "instrument capability evidence schema identifier is not supported",
        );
    }
    if profile.instrument_instance_id.trim().is_empty() {
        report.push(
            CapabilityEvidenceIssueCodeV1::EmptyInstrumentInstanceId,
            None,
            "instrument_instance_id must be non-empty",
        );
    }
    if profile.claims.len() > MAX_CAPABILITY_CLAIMS_V1 {
        report.push(
            CapabilityEvidenceIssueCodeV1::TooManyClaims,
            None,
            "profile exceeds the bounded claim count",
        );
    }

    if profile.claims.is_empty() {
        report.push(
            CapabilityEvidenceIssueCodeV1::EmptyClaims,
            None,
            "an evidence profile must contain at least one capability claim",
        );
    }
    let mut claim_ids = BTreeSet::new();
    // One immutable evidence artifact may support several claims. Repeated
    // evidence IDs are allowed only when their provenance records are identical.
    let mut evidence_records = BTreeMap::<String, EvidenceProvenanceV1>::new();
    for claim in &profile.claims {
        if claim.claim_id.trim().is_empty() {
            report.push(
                CapabilityEvidenceIssueCodeV1::EmptyClaimId,
                None,
                "claim_id must be non-empty",
            );
        } else if !claim_ids.insert(claim.claim_id.as_str()) {
            report.push(
                CapabilityEvidenceIssueCodeV1::DuplicateClaimId,
                Some(&claim.claim_id),
                "claim_id must be unique within the profile",
            );
        }
        if !claim.evidence.evidence_id.trim().is_empty() {
            if let Some(existing) = evidence_records.get(&claim.evidence.evidence_id) {
                if existing != &claim.evidence {
                    report.push(
                        CapabilityEvidenceIssueCodeV1::DuplicateEvidenceId,
                        Some(&claim.claim_id),
                        "a reused evidence_id must have identical provenance in every claim",
                    );
                }
            } else {
                evidence_records.insert(
                    claim.evidence.evidence_id.clone(),
                    claim.evidence.clone(),
                );
            }
        }
        if !validate_scope(&claim.scope) {
            report.push(
                CapabilityEvidenceIssueCodeV1::InvalidScope,
                Some(&claim.claim_id),
                "claim scope contains an empty field, a patch without renderer, or malformed asset digest",
            );
        }
        if !validate_value(&claim.value) {
            report.push(
                CapabilityEvidenceIssueCodeV1::InvalidCapabilityValue,
                Some(&claim.claim_id),
                "capability value is outside its schema bounds",
            );
        }
        validate_evidence(claim, &mut report);
    }

    // Derived claims must resolve every parent against the exact evidence
    // records present in this profile. Merely listing arbitrary parent IDs is
    // not a provenance chain.
    for claim in &profile.claims {
        if claim.evidence.evidence_class == EvidenceClassV1::Derived {
            for parent in &claim.evidence.parent_evidence_ids {
                if !evidence_records.contains_key(parent) {
                    report.push(
                        CapabilityEvidenceIssueCodeV1::UnknownDerivedParent,
                        Some(&claim.claim_id),
                        format!("derived evidence references unknown parent evidence ID {parent}"),
                    );
                }
            }
        }
    }

    report.finish()
}

fn qualified_values<'a>(
    profile: &'a InstrumentCapabilityEvidenceProfileV1,
    scope: &CapabilityScopeV1,
) -> impl Iterator<Item = &'a CapabilityClaimV1> {
    profile.claims.iter().filter(move |claim| {
        claim.disposition == QualificationDispositionV1::Qualified && &claim.scope == scope
    })
}

/// Materialize the existing orchestration capability only from exact-scope,
/// explicitly qualified range/polyphony claims. Renderer-scoped profiles also
/// require a positive renderer/patch availability claim.
pub fn materialize_qualified_instrument_capability_v1(
    profile: &InstrumentCapabilityEvidenceProfileV1,
    required_performance_scope: &CapabilityScopeV1,
    required_renderer_scope: Option<&CapabilityScopeV1>,
    trust_policy: &QualificationTrustPolicyV1,
) -> Result<QualifiedInstrumentCapabilityV1, CapabilityMaterializationErrorV1> {
    let validation = validate_instrument_capability_evidence_v1(profile);
    if !validation.valid {
        return Err(CapabilityMaterializationErrorV1::InvalidProfile);
    }

    if !validate_scope(required_performance_scope)
        || required_renderer_scope.is_some_and(|scope| {
            !validate_scope(scope) || scope.renderer_id.is_none() || scope.patch_id.is_none()
        })
    {
        return Err(CapabilityMaterializationErrorV1::ScopeMismatch);
    }
    let performance_claims: Vec<&CapabilityClaimV1> =
        qualified_values(profile, required_performance_scope).collect();
    let ranges: Vec<&CapabilityClaimV1> = performance_claims
        .iter()
        .copied()
        .filter(|claim| matches!(&claim.value, CapabilityValueV1::SoundingRange { .. }))
        .collect();
    let polyphony: Vec<&CapabilityClaimV1> = performance_claims
        .iter()
        .copied()
        .filter(|claim| matches!(&claim.value, CapabilityValueV1::MaxPolyphony { .. }))
        .collect();

    if ranges.is_empty() {
        let exists_at_another_scope = profile.claims.iter().any(|claim| {
            claim.disposition == QualificationDispositionV1::Qualified
                && matches!(&claim.value, CapabilityValueV1::SoundingRange { .. })
        });
        return Err(if exists_at_another_scope {
            CapabilityMaterializationErrorV1::ScopeMismatch
        } else {
            CapabilityMaterializationErrorV1::MissingQualifiedSoundingRange
        });
    }
    if ranges.len() > 1 {
        return Err(CapabilityMaterializationErrorV1::DuplicateQualifiedSoundingRange);
    }
    if polyphony.is_empty() {
        let exists_at_another_scope = profile.claims.iter().any(|claim| {
            claim.disposition == QualificationDispositionV1::Qualified
                && matches!(&claim.value, CapabilityValueV1::MaxPolyphony { .. })
        });
        return Err(if exists_at_another_scope {
            CapabilityMaterializationErrorV1::ScopeMismatch
        } else {
            CapabilityMaterializationErrorV1::MissingQualifiedPolyphony
        });
    }
    if polyphony.len() > 1 {
        return Err(CapabilityMaterializationErrorV1::DuplicateQualifiedPolyphony);
    }

    let renderer_claims: Vec<&CapabilityClaimV1> = if let Some(renderer_scope) =
        required_renderer_scope
    {
        qualified_values(profile, renderer_scope)
            .filter(|claim| {
                matches!(
                    &claim.value,
                    CapabilityValueV1::RendererPatchAvailability { .. }
                )
            })
            .collect()
    } else {
        Vec::new()
    };
    if required_renderer_scope.is_some() && renderer_claims.is_empty() {
        let exists_at_another_scope = profile.claims.iter().any(|claim| {
            claim.disposition == QualificationDispositionV1::Qualified
                && matches!(
                    &claim.value,
                    CapabilityValueV1::RendererPatchAvailability { .. }
                )
        });
        return Err(if exists_at_another_scope {
            CapabilityMaterializationErrorV1::ScopeMismatch
        } else {
            CapabilityMaterializationErrorV1::MissingQualifiedRendererPatch
        });
    }
    if renderer_claims.len() > 1 {
        return Err(CapabilityMaterializationErrorV1::DuplicateQualifiedCapability);
    }
    if renderer_claims.first().is_some_and(|claim| {
        !matches!(
            &claim.value,
            CapabilityValueV1::RendererPatchAvailability { available: true }
        )
    }) {
        return Err(CapabilityMaterializationErrorV1::RendererPatchUnavailable);
    }

    let sounding_range = match &ranges[0].value {
        CapabilityValueV1::SoundingRange {
            lowest_midi,
            highest_midi,
        } => SoundingRangeV1 {
            lowest_midi: *lowest_midi,
            highest_midi: *highest_midi,
        },
        _ => unreachable!("range candidates were filtered by variant"),
    };
    let max_polyphony = match &polyphony[0].value {
        CapabilityValueV1::MaxPolyphony { notes } => *notes,
        _ => unreachable!("polyphony candidates were filtered by variant"),
    };

    let mut required_claims = vec![ranges[0], polyphony[0]];
    if let Some(renderer_claim) = renderer_claims.first() {
        required_claims.push(*renderer_claim);
    }
    required_claims.sort_by(|a, b| a.claim_id.cmp(&b.claim_id));

    // Do not trust the profile's own "qualified" string. The calling boundary
    // must supply a trust policy loaded from an authenticated registry.
    for claim in &required_claims {
        let trusted_key = (
            claim.qualification_authority_id.clone().unwrap_or_default(),
            claim.qualification_record_id.clone().unwrap_or_default(),
            claim.qualification_record_sha256.clone().unwrap_or_default(),
        );
        if !trust_policy.approved_records.contains(&trusted_key) {
            return Err(CapabilityMaterializationErrorV1::QualificationNotTrusted);
        }
    }

    let profile_sha256 = instrument_capability_profile_sha256_v1(profile)
        .map_err(|_| CapabilityMaterializationErrorV1::ProfileSerialization)?;
    let claim_ids = required_claims
        .iter()
        .map(|claim| claim.claim_id.clone())
        .collect();
    let evidence_ids = required_claims
        .iter()
        .map(|claim| claim.evidence.evidence_id.clone())
        .collect();
    let qualification_record_ids = required_claims
        .iter()
        .filter_map(|claim| claim.qualification_record_id.clone())
        .collect();
    let qualification_record_sha256s = required_claims
        .iter()
        .filter_map(|claim| claim.qualification_record_sha256.clone())
        .collect();

    Ok(QualifiedInstrumentCapabilityV1 {
        capability: InstrumentCapabilityV1 {
            id: profile.instrument_instance_id.clone(),
            kind: profile.instrument_kind,
            sounding_range,
            max_polyphony,
        },
        profile_sha256,
        profile_digest_schema: INSTRUMENT_CAPABILITY_PROFILE_DIGEST_SCHEMA_V1.into(),
        performance_scope: required_performance_scope.clone(),
        renderer_scope: required_renderer_scope.cloned(),
        claim_ids,
        evidence_ids,
        qualification_record_ids,
        qualification_record_sha256s,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(ch: char) -> String {
        std::iter::repeat(ch).take(64).collect()
    }

    fn performance_scope() -> CapabilityScopeV1 {
        CapabilityScopeV1 {
            instrument_model: Some("cello-model-a".into()),
            performer_level: Some("professional".into()),
            technique: Some("arco".into()),
            dynamic: Some("mf".into()),
            renderer_id: None,
            patch_id: None,
            asset_sha256: None,
        }
    }

    fn renderer_scope() -> CapabilityScopeV1 {
        CapabilityScopeV1 {
            instrument_model: Some("cello-model-a".into()),
            performer_level: None,
            technique: None,
            dynamic: None,
            renderer_id: Some("fluidsynth-2.4".into()),
            patch_id: Some("orchestral-cello-sustain".into()),
            asset_sha256: Some(sha('a')),
        }
    }

    fn measured(evidence_id: &str) -> EvidenceProvenanceV1 {
        EvidenceProvenanceV1 {
            evidence_id: evidence_id.into(),
            evidence_class: EvidenceClassV1::Measured,
            source_uri: Some("https://example.invalid/measurement-record".into()),
            source_revision: Some("fixture-v1".into()),
            content_sha256: Some(sha('b')),
            bound_asset_sha256: None,
            method: Some("calibrated note-range protocol".into()),
            environment: Some("fixture performer, room and recording chain".into()),
            parent_evidence_ids: Vec::new(),
            transformation_id: None,
            license: ReusePermissionV1::NotApplicable,
            consent: ReusePermissionV1::Granted,
            uncertainty: Some(0.0),
        }
    }

    fn claim(
        claim_id: &str,
        value: CapabilityValueV1,
        scope: CapabilityScopeV1,
        evidence_id: &str,
    ) -> CapabilityClaimV1 {
        let mut evidence = measured(evidence_id);
        if let Some(asset_sha256) = &scope.asset_sha256 {
            evidence.bound_asset_sha256 = Some(asset_sha256.clone());
            evidence.license = ReusePermissionV1::Permitted;
        }
        CapabilityClaimV1 {
            claim_id: claim_id.into(),
            value,
            scope,
            disposition: QualificationDispositionV1::Qualified,
            qualification_record_id: Some(format!("review-{claim_id}")),
            qualification_record_sha256: Some(sha('d')),
            qualification_authority_id: Some("music-performer-review-board".into()),
            qualification_record_uri: Some(format!("https://example.invalid/reviews/{claim_id}")),
            evidence,
        }
    }

    fn trust_policy(profile: &InstrumentCapabilityEvidenceProfileV1) -> QualificationTrustPolicyV1 {
        QualificationTrustPolicyV1 {
            approved_records: profile.claims.iter().filter_map(|claim| {
                Some((
                    claim.qualification_authority_id.clone()?,
                    claim.qualification_record_id.clone()?,
                    claim.qualification_record_sha256.clone()?,
                ))
            }).collect(),
        }
    }

    fn valid_profile(
        include_renderer: bool,
    ) -> (
        InstrumentCapabilityEvidenceProfileV1,
        CapabilityScopeV1,
        Option<CapabilityScopeV1>,
    ) {
        let required_performance_scope = performance_scope();
        let required_renderer_scope = include_renderer.then(renderer_scope);
        let mut claims = vec![
            claim(
                "range",
                CapabilityValueV1::SoundingRange {
                    lowest_midi: 36,
                    highest_midi: 84,
                },
                required_performance_scope.clone(),
                "range-measurement",
            ),
            claim(
                "polyphony",
                CapabilityValueV1::MaxPolyphony { notes: 1 },
                required_performance_scope.clone(),
                "polyphony-measurement",
            ),
        ];
        if let Some(renderer_scope) = &required_renderer_scope {
            claims.push(claim(
                "renderer-patch",
                CapabilityValueV1::RendererPatchAvailability { available: true },
                renderer_scope.clone(),
                "renderer-probe",
            ));
        }
        (
            InstrumentCapabilityEvidenceProfileV1 {
                schema: INSTRUMENT_CAPABILITY_EVIDENCE_SCHEMA_V1.into(),
                instrument_instance_id: "solo-cello-01".into(),
                instrument_kind: InstrumentKindV1::Cello,
                claims,
            },
            required_performance_scope,
            required_renderer_scope,
        )
    }

    #[test]
    fn exact_scope_measured_claims_materialize_existing_capability_contract() {
        let (profile, required_performance_scope, required_renderer_scope) = valid_profile(true);
        let renderer_scope = required_renderer_scope.as_ref().unwrap();
        let policy = trust_policy(&profile);
        let capability = materialize_qualified_instrument_capability_v1(
            &profile,
            &required_performance_scope,
            Some(renderer_scope),
            &policy,
        )
        .unwrap();
        assert_eq!(capability.capability.id, "solo-cello-01");
        assert_eq!(capability.capability.kind, InstrumentKindV1::Cello);
        assert_eq!(capability.capability.sounding_range.lowest_midi, 36);
        assert_eq!(capability.capability.sounding_range.highest_midi, 84);
        assert_eq!(capability.capability.max_polyphony, 1);
        assert!(valid_sha256_hex_v1(&capability.profile_sha256));
        assert_eq!(capability.claim_ids.len(), 3);
        assert_eq!(capability.evidence_ids.len(), 3);
        assert_eq!(capability.qualification_record_ids.len(), 3);
        assert_eq!(capability.qualification_record_sha256s.len(), 3);
        assert!(capability
            .qualification_record_sha256s
            .iter()
            .all(|digest| valid_sha256_hex_v1(digest)));
        assert_eq!(capability.performance_scope, required_performance_scope);
        assert_eq!(capability.renderer_scope, Some(renderer_scope.clone()));
    }

    #[test]
    fn estimated_evidence_cannot_be_promoted_to_qualified_hard_capability() {
        let (mut profile, _, _) = valid_profile(false);
        profile.claims[0].evidence.evidence_class = EvidenceClassV1::Estimated;
        let report = validate_instrument_capability_evidence_v1(&profile);
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::UnqualifiedEvidencePromoted
        }));
        assert!(!report.valid);
    }

    #[test]
    fn qualified_claim_requires_permitted_reuse_and_a_review_record() {
        let (mut profile, _, _) = valid_profile(false);
        profile.claims[0].evidence.license = ReusePermissionV1::Unknown;
        profile.claims[0].qualification_record_id = None;
        let report = validate_instrument_capability_evidence_v1(&profile);
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::QualifiedClaimHasRestrictedReuse
        }));
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::QualifiedClaimWithoutReviewRecord
        }));
    }

    #[test]
    fn renderer_scoped_capability_abstains_on_scope_mismatch() {
        let (profile, performance_scope, _) = valid_profile(true);
        let mut wrong_scope = performance_scope;
        wrong_scope.instrument_model = Some("cello-model-b".into());
        assert_eq!(
            materialize_qualified_instrument_capability_v1(
                &profile,
                &wrong_scope,
                None,
                &trust_policy(&profile),
            ),
            Err(CapabilityMaterializationErrorV1::ScopeMismatch)
        );
    }

    #[test]
    fn asset_scoped_claim_rejects_a_different_asset_digest() {
        let (mut profile, _, _) = valid_profile(true);
        profile.claims[2].evidence.bound_asset_sha256 = Some(sha('c'));
        let report = validate_instrument_capability_evidence_v1(&profile);
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::AssetDigestMismatch
        }));
    }

    #[test]
    fn asset_scoped_qualified_claim_requires_explicit_license_permission() {
        let (mut profile, _, _) = valid_profile(true);
        profile.claims[2].evidence.license = ReusePermissionV1::Unknown;
        let report = validate_instrument_capability_evidence_v1(&profile);
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::QualifiedAssetWithoutLicense
        }));
    }

    #[test]
    fn renderer_scoped_capability_requires_a_positive_exact_patch_probe() {
        let (mut profile, performance_scope, renderer_scope) = valid_profile(true);
        profile.claims.pop();
        let renderer_scope = renderer_scope.as_ref().unwrap();
        assert_eq!(
            materialize_qualified_instrument_capability_v1(
                &profile,
                &performance_scope,
                Some(renderer_scope),
                &trust_policy(&profile),
            ),
            Err(CapabilityMaterializationErrorV1::MissingQualifiedRendererPatch)
        );
    }

    #[test]
    fn malformed_evidence_digest_is_rejected() {
        let (mut profile, _, _) = valid_profile(false);
        profile.claims[0].evidence.content_sha256 = Some("not-a-digest".into());
        let report = validate_instrument_capability_evidence_v1(&profile);
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::MalformedContentDigest
        }));
    }

    #[test]
    fn qualified_claim_requires_a_valid_qualification_record_digest() {
        let (mut profile, _, _) = valid_profile(false);
        profile.claims[0].qualification_record_sha256 = Some("not-a-digest".into());
        let report = validate_instrument_capability_evidence_v1(&profile);
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::MalformedQualificationRecordDigest
        }));
    }

    #[test]
    fn qualified_claims_need_an_authenticated_trust_policy_match() {
        let (profile, performance_scope, _) = valid_profile(false);
        assert_eq!(
            materialize_qualified_instrument_capability_v1(
                &profile,
                &performance_scope,
                None,
                &QualificationTrustPolicyV1::default(),
            ),
            Err(CapabilityMaterializationErrorV1::QualificationNotTrusted)
        );
    }

    #[test]
    fn renderer_scope_must_name_both_renderer_and_patch() {
        let (profile, performance_scope, _) = valid_profile(false);
        assert_eq!(
            materialize_qualified_instrument_capability_v1(
                &profile,
                &performance_scope,
                Some(&performance_scope),
                &trust_policy(&profile),
            ),
            Err(CapabilityMaterializationErrorV1::ScopeMismatch)
        );
    }

    #[test]
    fn derived_claim_rejects_unknown_parent_evidence_ids() {
        let (mut profile, _, _) = valid_profile(false);
        profile.claims[0].evidence.evidence_class = EvidenceClassV1::Derived;
        profile.claims[0].evidence.parent_evidence_ids = vec!["not-in-profile".into()];
        profile.claims[0].evidence.transformation_id = Some("range-transform-v1".into());
        let report = validate_instrument_capability_evidence_v1(&profile);
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::UnknownDerivedParent
        }));
    }

    #[test]
    fn asset_scope_requires_exact_renderer_and_patch() {
        let mut invalid = performance_scope();
        invalid.asset_sha256 = Some(sha('a'));
        assert!(!validate_scope(&invalid));
    }

    #[test]
    fn derived_claim_requires_parent_evidence_and_transform_identity() {
        let (mut profile, _, _) = valid_profile(false);
        profile.claims[0].evidence.evidence_class = EvidenceClassV1::Derived;
        profile.claims[0].evidence.parent_evidence_ids.clear();
        profile.claims[0].evidence.transformation_id = None;
        let report = validate_instrument_capability_evidence_v1(&profile);
        assert!(report.issues.iter().any(|issue| {
            issue.code == CapabilityEvidenceIssueCodeV1::MissingDerivedLineage
        }));
    }
}
