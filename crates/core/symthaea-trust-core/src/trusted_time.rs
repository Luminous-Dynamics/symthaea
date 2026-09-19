// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Authenticated trusted-time capability.
//!
//! TRUST-008A established only structural interval consistency. This layer turns
//! that into authority only when every counted time source is itself authorized
//! under the expected root role and signs an exact, non-circular time-source
//! statement. Distinct source IDs alone are not treated as organizational
//! independence; this layer establishes authenticated source provenance and a
//! bounded consensus interval, not global time truth.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::{
    AuthorizedTrustRoleAttestation, FramedDigest, FrozenTimeEvidence, Sha256Digest,
    TimeAssessment, TimeAssessmentClosure, TimeAssessmentPolicy, TimeEvidenceDraft,
    TimeEvidenceKind, TimeEvidenceIssue, TrustRole, MAX_TIME_SOURCE_ID_BYTES, assess_time,
};

const TIME_SOURCE_STATEMENT_DOMAIN: &str = "symthaea.time-source-statement.identity.v1";
const TRUSTED_TIME_AUTHORITY_DOMAIN: &str = "symthaea.trusted-time-authority.identity.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimeSourceStatementIssue {
    InvalidSourceId,
    SourceIdTooLong,
    InvalidInterval,
}

/// Authority-independent time statement. This object exists specifically to
/// break the otherwise circular dependency between evidence identity and the
/// authority that signs that evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TimeSourceStatement {
    source_id: String,
    kind: TimeEvidenceKind,
    earliest_unix_s: u64,
    latest_unix_s: u64,
    source_artifact_sha256: Sha256Digest,
    statement_sha256: Sha256Digest,
}

impl TimeSourceStatement {
    pub fn new(
        source_id: impl Into<String>,
        kind: TimeEvidenceKind,
        earliest_unix_s: u64,
        latest_unix_s: u64,
        source_artifact_sha256: Sha256Digest,
    ) -> Result<Self, Vec<TimeSourceStatementIssue>> {
        let source_id = source_id.into();
        let mut issues = Vec::new();
        if !canonical_identifier(&source_id) {
            issues.push(TimeSourceStatementIssue::InvalidSourceId);
        }
        if source_id.len() > MAX_TIME_SOURCE_ID_BYTES {
            issues.push(TimeSourceStatementIssue::SourceIdTooLong);
        }
        if earliest_unix_s > latest_unix_s {
            issues.push(TimeSourceStatementIssue::InvalidInterval);
        }
        if !issues.is_empty() {
            return Err(issues);
        }
        let statement_sha256 = time_source_statement_digest_fields(
            &source_id,
            kind,
            earliest_unix_s,
            latest_unix_s,
            &source_artifact_sha256,
        );
        Ok(Self {
            source_id,
            kind,
            earliest_unix_s,
            latest_unix_s,
            source_artifact_sha256,
            statement_sha256,
        })
    }

    pub fn source_id(&self) -> &str { &self.source_id }
    pub fn kind(&self) -> TimeEvidenceKind { self.kind }
    pub fn earliest_unix_s(&self) -> u64 { self.earliest_unix_s }
    pub fn latest_unix_s(&self) -> u64 { self.latest_unix_s }
    pub fn source_artifact_sha256(&self) -> &Sha256Digest { &self.source_artifact_sha256 }
    pub fn statement_sha256(&self) -> &Sha256Digest { &self.statement_sha256 }

    pub fn bind_authority(
        &self,
        source_authority_sha256: Sha256Digest,
    ) -> Result<FrozenTimeEvidence, Vec<TimeEvidenceIssue>> {
        TimeEvidenceDraft {
            source_id: self.source_id.clone(),
            kind: self.kind,
            earliest_unix_s: self.earliest_unix_s,
            latest_unix_s: self.latest_unix_s,
            source_artifact_sha256: self.source_artifact_sha256.clone(),
            source_authority_sha256: Some(source_authority_sha256),
        }
        .freeze()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustedTimeIssue {
    StructuralAssessmentNotConsistent,
    StructuralAssessmentIdentityMismatch,
    MinimumKindTooWeak,
    DuplicateSourceArtifact { source_id: String },
    DuplicateSourceAuthority { source_id: String },
    MissingSourceAuthority { source_id: String },
    UnknownSourceAuthority { source_id: String },
    WrongAuthorityRole {
        source_id: String,
        expected: TrustRole,
        actual: TrustRole,
    },
    SourceArtifactMismatch { source_id: String },
    SourceStatementMismatch { source_id: String },
    RootAuthorityMismatch { source_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrustedTimeBinding {
    source_id: String,
    kind: TimeEvidenceKind,
    evidence_sha256: Sha256Digest,
    source_artifact_sha256: Sha256Digest,
    source_statement_sha256: Sha256Digest,
    source_authority_sha256: Sha256Digest,
    role: TrustRole,
    root_authority_sha256: Sha256Digest,
    trust_snapshot_authority_sha256: Sha256Digest,
}

impl TrustedTimeBinding {
    pub fn source_id(&self) -> &str { &self.source_id }
    pub fn kind(&self) -> TimeEvidenceKind { self.kind }
    pub fn evidence_sha256(&self) -> &Sha256Digest { &self.evidence_sha256 }
    pub fn source_artifact_sha256(&self) -> &Sha256Digest { &self.source_artifact_sha256 }
    pub fn source_statement_sha256(&self) -> &Sha256Digest { &self.source_statement_sha256 }
    pub fn source_authority_sha256(&self) -> &Sha256Digest { &self.source_authority_sha256 }
    pub fn role(&self) -> TrustRole { self.role }
    pub fn root_authority_sha256(&self) -> &Sha256Digest { &self.root_authority_sha256 }
    pub fn trust_snapshot_authority_sha256(&self) -> &Sha256Digest {
        &self.trust_snapshot_authority_sha256
    }
}

/// Non-deserializable capability establishing an authenticated bounded time
/// interval under one exact root authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrustedTime {
    structural_assessment_sha256: Sha256Digest,
    root_authority_sha256: Sha256Digest,
    consensus_earliest_unix_s: u64,
    consensus_latest_unix_s: u64,
    bindings: Vec<TrustedTimeBinding>,
    authority_sha256: Sha256Digest,
}

impl TrustedTime {
    pub fn structural_assessment_sha256(&self) -> &Sha256Digest {
        &self.structural_assessment_sha256
    }
    pub fn root_authority_sha256(&self) -> &Sha256Digest { &self.root_authority_sha256 }
    pub fn consensus_interval(&self) -> (u64, u64) {
        (self.consensus_earliest_unix_s, self.consensus_latest_unix_s)
    }
    pub fn bindings(&self) -> &[TrustedTimeBinding] { &self.bindings }
    pub fn authority_sha256(&self) -> &Sha256Digest { &self.authority_sha256 }
    pub const fn trusted_time_established(&self) -> bool { true }
    /// Current wall-clock truth still depends on root/snapshot currentness and
    /// source freshness at the point of use.
    pub const fn current_time_established(&self) -> bool { false }
}

pub fn establish_trusted_time(
    structural_policy: TimeAssessmentPolicy,
    structural_assessment: &TimeAssessment,
    evidence: &[FrozenTimeEvidence],
    authorities: &[AuthorizedTrustRoleAttestation],
) -> Result<TrustedTime, Vec<TrustedTimeIssue>> {
    let mut issues = Vec::new();
    if structural_assessment.closure() != TimeAssessmentClosure::Consistent {
        issues.push(TrustedTimeIssue::StructuralAssessmentNotConsistent);
    }
    let recomputed = assess_time(structural_policy.clone(), evidence);
    if recomputed.assessment_sha256() != structural_assessment.assessment_sha256() {
        issues.push(TrustedTimeIssue::StructuralAssessmentIdentityMismatch);
    }
    if structural_policy.minimum_kind.strength() < TimeEvidenceKind::TransparencyIntegrated.strength() {
        issues.push(TrustedTimeIssue::MinimumKindTooWeak);
    }

    let authority_by_digest: BTreeMap<_, _> = authorities
        .iter()
        .map(|authority| (authority.authority_sha256().clone(), authority))
        .collect();
    let mut source_artifacts = BTreeSet::new();
    let mut source_authorities = BTreeSet::new();
    let mut root_authority = None;
    let mut bindings = Vec::new();

    for item in evidence {
        let Some(source_authority_sha256) = item.source_authority_sha256() else {
            issues.push(TrustedTimeIssue::MissingSourceAuthority {
                source_id: item.source_id().to_string(),
            });
            continue;
        };
        if !source_artifacts.insert(item.source_artifact_sha256().clone()) {
            issues.push(TrustedTimeIssue::DuplicateSourceArtifact {
                source_id: item.source_id().to_string(),
            });
        }
        if !source_authorities.insert(source_authority_sha256.clone()) {
            issues.push(TrustedTimeIssue::DuplicateSourceAuthority {
                source_id: item.source_id().to_string(),
            });
        }
        let Some(authority) = authority_by_digest.get(source_authority_sha256).copied() else {
            issues.push(TrustedTimeIssue::UnknownSourceAuthority {
                source_id: item.source_id().to_string(),
            });
            continue;
        };
        let expected_role = time_role(item.kind());
        if authority.role() != expected_role {
            issues.push(TrustedTimeIssue::WrongAuthorityRole {
                source_id: item.source_id().to_string(),
                expected: expected_role,
                actual: authority.role(),
            });
        }
        if authority.subject_sha256() != item.source_artifact_sha256() {
            issues.push(TrustedTimeIssue::SourceArtifactMismatch {
                source_id: item.source_id().to_string(),
            });
        }
        let statement_sha256 = time_source_statement_digest(item);
        if authority.payload_sha256() != &statement_sha256 {
            issues.push(TrustedTimeIssue::SourceStatementMismatch {
                source_id: item.source_id().to_string(),
            });
        }
        match &root_authority {
            None => root_authority = Some(authority.root_authority_sha256().clone()),
            Some(expected) if expected == authority.root_authority_sha256() => {}
            Some(_) => issues.push(TrustedTimeIssue::RootAuthorityMismatch {
                source_id: item.source_id().to_string(),
            }),
        }
        bindings.push(TrustedTimeBinding {
            source_id: item.source_id().to_string(),
            kind: item.kind(),
            evidence_sha256: item.evidence_sha256().clone(),
            source_artifact_sha256: item.source_artifact_sha256().clone(),
            source_statement_sha256: statement_sha256,
            source_authority_sha256: source_authority_sha256.clone(),
            role: authority.role(),
            root_authority_sha256: authority.root_authority_sha256().clone(),
            trust_snapshot_authority_sha256: authority.trust_snapshot_authority_sha256().clone(),
        });
    }

    if !issues.is_empty() {
        return Err(issues);
    }
    let Some((earliest, latest)) = structural_assessment.consensus_interval() else {
        return Err(vec![TrustedTimeIssue::StructuralAssessmentNotConsistent]);
    };
    let Some(root_authority_sha256) = root_authority else {
        return Err(vec![TrustedTimeIssue::StructuralAssessmentNotConsistent]);
    };
    bindings.sort_by(|left, right| left.source_id.cmp(&right.source_id));
    let authority_sha256 = trusted_time_digest(
        structural_assessment.assessment_sha256(),
        &root_authority_sha256,
        earliest,
        latest,
        &bindings,
    );
    Ok(TrustedTime {
        structural_assessment_sha256: structural_assessment.assessment_sha256().clone(),
        root_authority_sha256,
        consensus_earliest_unix_s: earliest,
        consensus_latest_unix_s: latest,
        bindings,
        authority_sha256,
    })
}

/// Reconstruct the exact authority-independent statement digest from frozen
/// evidence. This must equal the payload signed by the source authority.
pub fn time_source_statement_digest(item: &FrozenTimeEvidence) -> Sha256Digest {
    time_source_statement_digest_fields(
        item.source_id(),
        item.kind(),
        item.earliest_unix_s(),
        item.latest_unix_s(),
        item.source_artifact_sha256(),
    )
}

fn time_source_statement_digest_fields(
    source_id: &str,
    kind: TimeEvidenceKind,
    earliest_unix_s: u64,
    latest_unix_s: u64,
    source_artifact_sha256: &Sha256Digest,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(TIME_SOURCE_STATEMENT_DOMAIN);
    digest.text(source_id);
    digest.text(time_kind_tag(kind));
    digest.text(&earliest_unix_s.to_string());
    digest.text(&latest_unix_s.to_string());
    digest.text(source_artifact_sha256.as_str());
    digest.digest()
}

fn trusted_time_digest(
    structural_assessment_sha256: &Sha256Digest,
    root_authority_sha256: &Sha256Digest,
    earliest: u64,
    latest: u64,
    bindings: &[TrustedTimeBinding],
) -> Sha256Digest {
    let mut digest = FramedDigest::new(TRUSTED_TIME_AUTHORITY_DOMAIN);
    digest.text(structural_assessment_sha256.as_str());
    digest.text(root_authority_sha256.as_str());
    digest.text(&earliest.to_string());
    digest.text(&latest.to_string());
    for binding in bindings {
        digest.text("binding");
        digest.text(&binding.source_id);
        digest.text(time_kind_tag(binding.kind));
        digest.text(binding.evidence_sha256.as_str());
        digest.text(binding.source_artifact_sha256.as_str());
        digest.text(binding.source_statement_sha256.as_str());
        digest.text(binding.source_authority_sha256.as_str());
        digest.text(role_tag(binding.role));
        digest.text(binding.root_authority_sha256.as_str());
        digest.text(binding.trust_snapshot_authority_sha256.as_str());
    }
    digest.digest()
}

const fn time_role(kind: TimeEvidenceKind) -> TrustRole {
    match kind {
        TimeEvidenceKind::TransparencyIntegrated => TrustRole::TransparencyLog,
        TimeEvidenceKind::WitnessedCheckpoint => TrustRole::TransparencyWitness,
        TimeEvidenceKind::ExternalTimestampAuthority => TrustRole::Freshness,
        TimeEvidenceKind::LocalClockDeclared | TimeEvidenceKind::MonotonicSystemClock => {
            TrustRole::Freshness
        }
    }
}

const fn time_kind_tag(kind: TimeEvidenceKind) -> &'static str {
    match kind {
        TimeEvidenceKind::LocalClockDeclared => "local-clock-declared",
        TimeEvidenceKind::MonotonicSystemClock => "monotonic-system-clock",
        TimeEvidenceKind::TransparencyIntegrated => "transparency-integrated",
        TimeEvidenceKind::WitnessedCheckpoint => "witnessed-checkpoint",
        TimeEvidenceKind::ExternalTimestampAuthority => "external-timestamp-authority",
    }
}

const fn role_tag(role: TrustRole) -> &'static str {
    match role {
        TrustRole::Root => "root",
        TrustRole::Freshness => "freshness",
        TrustRole::KeyLifecycle => "key-lifecycle",
        TrustRole::QualificationProfile => "qualification-profile",
        TrustRole::QualificationDecision => "qualification-decision",
        TrustRole::QualificationLifecycle => "qualification-lifecycle",
        TrustRole::TransparencyLog => "transparency-log",
        TrustRole::TransparencyWitness => "transparency-witness",
        TrustRole::EmergencyRecovery => "emergency-recovery",
    }
}

fn canonical_identifier(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'/' | b'-')
        })
}
