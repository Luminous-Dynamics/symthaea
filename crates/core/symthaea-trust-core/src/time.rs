// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded, content-addressed time evidence.
//!
//! Authority-sensitive code should not silently equate a local wall-clock read
//! with externally witnessed time. This module preserves time as an interval plus
//! provenance class and exact source artifact. TRUST-008A remains deliberately
//! non-authorizing: source-authority digests are commitments, not proof that the
//! named source was itself valid. A later gate may consume authenticated source
//! capabilities and mint trusted-time authority.

use std::collections::BTreeSet;

use serde::Serialize;

use crate::{FramedDigest, Sha256Digest};

pub const TIME_EVIDENCE_SCHEMA: &str = "symthaea.time-evidence.v1";
pub const TIME_ASSESSMENT_SCHEMA: &str = "symthaea.time-assessment.v1";
const TIME_EVIDENCE_DOMAIN: &str = "symthaea.time-evidence.identity.v1";
const TIME_ASSESSMENT_DOMAIN: &str = "symthaea.time-assessment.identity.v1";
pub const MAX_TIME_SOURCE_ID_BYTES: usize = 256;
pub const MAX_TIME_EVIDENCE_ITEMS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum TimeEvidenceKind {
    LocalClockDeclared,
    MonotonicSystemClock,
    TransparencyIntegrated,
    WitnessedCheckpoint,
    ExternalTimestampAuthority,
}

impl TimeEvidenceKind {
    pub const fn strength(self) -> u8 {
        match self {
            Self::LocalClockDeclared => 0,
            Self::MonotonicSystemClock => 1,
            Self::TransparencyIntegrated => 2,
            Self::WitnessedCheckpoint => 3,
            Self::ExternalTimestampAuthority => 4,
        }
    }

    pub const fn requires_source_authority(self) -> bool {
        matches!(
            self,
            Self::TransparencyIntegrated
                | Self::WitnessedCheckpoint
                | Self::ExternalTimestampAuthority
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeEvidenceDraft {
    pub source_id: String,
    pub kind: TimeEvidenceKind,
    pub earliest_unix_s: u64,
    pub latest_unix_s: u64,
    pub source_artifact_sha256: Sha256Digest,
    pub source_authority_sha256: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimeEvidenceIssue {
    InvalidSourceId,
    SourceIdTooLong,
    InvalidInterval,
    MissingRequiredSourceAuthority,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrozenTimeEvidence {
    source_id: String,
    kind: TimeEvidenceKind,
    earliest_unix_s: u64,
    latest_unix_s: u64,
    source_artifact_sha256: Sha256Digest,
    source_authority_sha256: Option<Sha256Digest>,
    evidence_sha256: Sha256Digest,
}

impl TimeEvidenceDraft {
    pub fn freeze(self) -> Result<FrozenTimeEvidence, Vec<TimeEvidenceIssue>> {
        let mut issues = Vec::new();
        if !canonical_identifier(&self.source_id) {
            issues.push(TimeEvidenceIssue::InvalidSourceId);
        }
        if self.source_id.len() > MAX_TIME_SOURCE_ID_BYTES {
            issues.push(TimeEvidenceIssue::SourceIdTooLong);
        }
        if self.earliest_unix_s > self.latest_unix_s {
            issues.push(TimeEvidenceIssue::InvalidInterval);
        }
        if self.kind.requires_source_authority() && self.source_authority_sha256.is_none() {
            issues.push(TimeEvidenceIssue::MissingRequiredSourceAuthority);
        }
        if !issues.is_empty() {
            return Err(issues);
        }
        let evidence_sha256 = evidence_digest(&self);
        Ok(FrozenTimeEvidence {
            source_id: self.source_id,
            kind: self.kind,
            earliest_unix_s: self.earliest_unix_s,
            latest_unix_s: self.latest_unix_s,
            source_artifact_sha256: self.source_artifact_sha256,
            source_authority_sha256: self.source_authority_sha256,
            evidence_sha256,
        })
    }
}

impl FrozenTimeEvidence {
    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    pub fn kind(&self) -> TimeEvidenceKind {
        self.kind
    }

    pub fn earliest_unix_s(&self) -> u64 {
        self.earliest_unix_s
    }

    pub fn latest_unix_s(&self) -> u64 {
        self.latest_unix_s
    }

    pub fn source_artifact_sha256(&self) -> &Sha256Digest {
        &self.source_artifact_sha256
    }

    pub fn source_authority_sha256(&self) -> Option<&Sha256Digest> {
        self.source_authority_sha256.as_ref()
    }

    pub fn evidence_sha256(&self) -> &Sha256Digest {
        &self.evidence_sha256
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TimeAssessmentPolicy {
    pub minimum_distinct_sources: usize,
    pub minimum_kind: TimeEvidenceKind,
    pub maximum_consensus_width_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimeAssessmentPolicyIssue {
    ZeroSourceThreshold,
    TooManyRequiredSources,
}

impl TimeAssessmentPolicy {
    pub fn validate(&self) -> Result<(), Vec<TimeAssessmentPolicyIssue>> {
        let mut issues = Vec::new();
        if self.minimum_distinct_sources == 0 {
            issues.push(TimeAssessmentPolicyIssue::ZeroSourceThreshold);
        }
        if self.minimum_distinct_sources > MAX_TIME_EVIDENCE_ITEMS {
            issues.push(TimeAssessmentPolicyIssue::TooManyRequiredSources);
        }
        if issues.is_empty() {
            Ok(())
        } else {
            Err(issues)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum TimeAssessmentClosure {
    Consistent,
    Insufficient,
    Inconsistent,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum TimeAssessmentFinding {
    InvalidPolicy,
    EmptyEvidence,
    TooManyEvidenceItems,
    DuplicateSource { source_id: String },
    EvidenceBelowMinimumKind { source_id: String },
    InsufficientDistinctSources { actual: usize, required: usize },
    NoIntervalIntersection,
    ConsensusIntervalTooWide { actual_s: u64, maximum_s: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TimeAssessment {
    policy: TimeAssessmentPolicy,
    evidence_sha256s: Vec<Sha256Digest>,
    consensus_earliest_unix_s: Option<u64>,
    consensus_latest_unix_s: Option<u64>,
    findings: Vec<TimeAssessmentFinding>,
    closure: TimeAssessmentClosure,
    assessment_sha256: Sha256Digest,
}

impl TimeAssessment {
    pub fn closure(&self) -> TimeAssessmentClosure {
        self.closure
    }

    pub fn consensus_interval(&self) -> Option<(u64, u64)> {
        Some((
            self.consensus_earliest_unix_s?,
            self.consensus_latest_unix_s?,
        ))
    }

    pub fn findings(&self) -> &[TimeAssessmentFinding] {
        &self.findings
    }

    pub fn assessment_sha256(&self) -> &Sha256Digest {
        &self.assessment_sha256
    }

    /// TRUST-008A only establishes deterministic structural consistency. Exact
    /// source-authority digests are not yet authenticated here.
    pub const fn trusted_time_established(&self) -> bool {
        false
    }
}

pub fn assess_time(
    policy: TimeAssessmentPolicy,
    evidence: &[FrozenTimeEvidence],
) -> TimeAssessment {
    let mut findings = Vec::new();
    let mut invalid = false;
    let mut insufficient = false;
    let mut inconsistent = false;

    if policy.validate().is_err() {
        findings.push(TimeAssessmentFinding::InvalidPolicy);
        invalid = true;
    }
    if evidence.is_empty() {
        findings.push(TimeAssessmentFinding::EmptyEvidence);
        insufficient = true;
    }
    if evidence.len() > MAX_TIME_EVIDENCE_ITEMS {
        findings.push(TimeAssessmentFinding::TooManyEvidenceItems);
        invalid = true;
    }

    let mut sources = BTreeSet::new();
    let mut accepted = Vec::new();
    for item in evidence {
        if !sources.insert(item.source_id.clone()) {
            findings.push(TimeAssessmentFinding::DuplicateSource {
                source_id: item.source_id.clone(),
            });
            invalid = true;
            continue;
        }
        if item.kind.strength() < policy.minimum_kind.strength() {
            findings.push(TimeAssessmentFinding::EvidenceBelowMinimumKind {
                source_id: item.source_id.clone(),
            });
            continue;
        }
        accepted.push(item);
    }

    if accepted.len() < policy.minimum_distinct_sources {
        findings.push(TimeAssessmentFinding::InsufficientDistinctSources {
            actual: accepted.len(),
            required: policy.minimum_distinct_sources,
        });
        insufficient = true;
    }

    let consensus_earliest_unix_s = accepted.iter().map(|item| item.earliest_unix_s).max();
    let consensus_latest_unix_s = accepted.iter().map(|item| item.latest_unix_s).min();
    if let (Some(earliest), Some(latest)) =
        (consensus_earliest_unix_s, consensus_latest_unix_s)
    {
        if earliest > latest {
            findings.push(TimeAssessmentFinding::NoIntervalIntersection);
            inconsistent = true;
        } else {
            let width = latest - earliest;
            if width > policy.maximum_consensus_width_s {
                findings.push(TimeAssessmentFinding::ConsensusIntervalTooWide {
                    actual_s: width,
                    maximum_s: policy.maximum_consensus_width_s,
                });
                inconsistent = true;
            }
        }
    }

    findings.sort_by_key(|finding| format!("{finding:?}"));
    let closure = if invalid {
        TimeAssessmentClosure::Invalid
    } else if inconsistent {
        TimeAssessmentClosure::Inconsistent
    } else if insufficient {
        TimeAssessmentClosure::Insufficient
    } else {
        TimeAssessmentClosure::Consistent
    };

    let mut evidence_sha256s: Vec<_> = evidence
        .iter()
        .map(|item| item.evidence_sha256.clone())
        .collect();
    evidence_sha256s.sort();
    let assessment_sha256 = assessment_digest(
        &policy,
        &evidence_sha256s,
        consensus_earliest_unix_s,
        consensus_latest_unix_s,
        closure,
    );

    TimeAssessment {
        policy,
        evidence_sha256s,
        consensus_earliest_unix_s,
        consensus_latest_unix_s,
        findings,
        closure,
        assessment_sha256,
    }
}

fn evidence_digest(draft: &TimeEvidenceDraft) -> Sha256Digest {
    let mut digest = FramedDigest::new(TIME_EVIDENCE_DOMAIN);
    digest.text(TIME_EVIDENCE_SCHEMA);
    digest.text(&draft.source_id);
    digest.text(time_kind_tag(draft.kind));
    digest.text(&draft.earliest_unix_s.to_string());
    digest.text(&draft.latest_unix_s.to_string());
    digest.text(draft.source_artifact_sha256.as_str());
    digest.optional_sha(draft.source_authority_sha256.as_ref());
    digest.digest()
}

fn assessment_digest(
    policy: &TimeAssessmentPolicy,
    evidence_sha256s: &[Sha256Digest],
    consensus_earliest_unix_s: Option<u64>,
    consensus_latest_unix_s: Option<u64>,
    closure: TimeAssessmentClosure,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(TIME_ASSESSMENT_DOMAIN);
    digest.text(TIME_ASSESSMENT_SCHEMA);
    digest.text(&policy.minimum_distinct_sources.to_string());
    digest.text(time_kind_tag(policy.minimum_kind));
    digest.text(&policy.maximum_consensus_width_s.to_string());
    for evidence_sha256 in evidence_sha256s {
        digest.text("evidence");
        digest.text(evidence_sha256.as_str());
    }
    match consensus_earliest_unix_s {
        Some(value) => {
            digest.text("consensus-earliest");
            digest.text(&value.to_string());
        }
        None => digest.text("no-consensus-earliest"),
    }
    match consensus_latest_unix_s {
        Some(value) => {
            digest.text("consensus-latest");
            digest.text(&value.to_string());
        }
        None => digest.text("no-consensus-latest"),
    }
    digest.text(match closure {
        TimeAssessmentClosure::Consistent => "consistent",
        TimeAssessmentClosure::Insufficient => "insufficient",
        TimeAssessmentClosure::Inconsistent => "inconsistent",
        TimeAssessmentClosure::Invalid => "invalid",
    });
    digest.text("trusted-time-not-established");
    digest.digest()
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

fn canonical_identifier(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'/' | b'-')
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(
        source: &str,
        kind: TimeEvidenceKind,
        earliest: u64,
        latest: u64,
    ) -> FrozenTimeEvidence {
        TimeEvidenceDraft {
            source_id: source.into(),
            kind,
            earliest_unix_s: earliest,
            latest_unix_s: latest,
            source_artifact_sha256: Sha256Digest::of_bytes(source.as_bytes()),
            source_authority_sha256: kind
                .requires_source_authority()
                .then(|| Sha256Digest::of_bytes(format!("authority:{source}").as_bytes())),
        }
        .freeze()
        .unwrap()
    }

    #[test]
    fn overlapping_intervals_produce_bounded_consensus() {
        let policy = TimeAssessmentPolicy {
            minimum_distinct_sources: 2,
            minimum_kind: TimeEvidenceKind::TransparencyIntegrated,
            maximum_consensus_width_s: 10,
        };
        let report = assess_time(
            policy,
            &[
                evidence("log-a", TimeEvidenceKind::TransparencyIntegrated, 100, 110),
                evidence("witness-b", TimeEvidenceKind::WitnessedCheckpoint, 105, 115),
            ],
        );
        assert_eq!(report.closure(), TimeAssessmentClosure::Consistent);
        assert_eq!(report.consensus_interval(), Some((105, 110)));
        assert!(!report.trusted_time_established());
    }

    #[test]
    fn disjoint_intervals_fail_closed() {
        let policy = TimeAssessmentPolicy {
            minimum_distinct_sources: 2,
            minimum_kind: TimeEvidenceKind::TransparencyIntegrated,
            maximum_consensus_width_s: 10,
        };
        let report = assess_time(
            policy,
            &[
                evidence("log-a", TimeEvidenceKind::TransparencyIntegrated, 100, 101),
                evidence("witness-b", TimeEvidenceKind::WitnessedCheckpoint, 110, 111),
            ],
        );
        assert_eq!(report.closure(), TimeAssessmentClosure::Inconsistent);
    }

    #[test]
    fn duplicate_source_cannot_fake_quorum() {
        let policy = TimeAssessmentPolicy {
            minimum_distinct_sources: 2,
            minimum_kind: TimeEvidenceKind::TransparencyIntegrated,
            maximum_consensus_width_s: 10,
        };
        let report = assess_time(
            policy,
            &[
                evidence("same", TimeEvidenceKind::TransparencyIntegrated, 100, 105),
                evidence("same", TimeEvidenceKind::WitnessedCheckpoint, 100, 105),
            ],
        );
        assert_eq!(report.closure(), TimeAssessmentClosure::Invalid);
    }
}
