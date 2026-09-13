// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bitemporal lifecycle and current-reliance semantics for authenticated
//! verifier-profile provenance.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea_evidence_independence_provenance::{
    AuthenticatedProfileProvenance, ProfileProvenancePolicy, VersionedVerifierProfileRecord,
};

pub const PROFILE_LIFECYCLE_LEDGER_SCHEMA_V1: &str =
    "symthaea.assurance.verifier-profile-lifecycle-ledger.v1";
pub const MAX_LIFECYCLE_EVENTS: usize = 4_096;
pub const MAX_EVIDENCE_REFS: usize = 128;
pub const MAX_TEXT_BYTES: usize = 512;

const LEDGER_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.verifier-profile-lifecycle-ledger.digest.v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfileLifecycleEventKind {
    /// Normal replacement. This changes which profile should be used for events
    /// at or after `successor_effective_from_ms`, but does not rewrite earlier
    /// valid history.
    Superseded {
        successor_record_digest: String,
        successor_effective_from_ms: u64,
        reason_ref: String,
    },
    /// Corrective information says this record was not reliable from an earlier
    /// effective time. This can invalidate current reliance on historical use.
    CorrectivelyRevoked {
        invalid_from_ms: u64,
        reason_ref: String,
    },
    /// A substantive contradiction places the record in quarantine from the
    /// supplied effective time. Resolution records disposition but does not
    /// reactivate this record; a replacement is required for renewed reliance.
    Contradicted {
        contradiction_id: String,
        effective_from_ms: u64,
        contradiction_ref: String,
    },
    ContradictionResolved {
        contradiction_id: String,
        resolution_ref: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileLifecycleEvent {
    pub event_id: String,
    pub recorded_at_ms: u64,
    pub kind: ProfileLifecycleEventKind,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileLifecycleLedger {
    pub schema_version: String,
    pub ledger_id: String,
    pub profile_id: String,
    pub record_digest: String,
    pub events: Vec<ProfileLifecycleEvent>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfileLifecycleDisposition {
    Invalid,
    Blocked,
    Usable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfileLifecycleIssue {
    InvalidRecord,
    RecordDigestMismatch,
    ProfileIdMismatch,
    InvalidLedger,
    AssessmentPredatesVerification,
    FutureRecordedEvent(String),
    RecordNotApplicableAtVerification,
    ProvenanceNotProspectivelyEstablished,
    SupersededAtVerification {
        successor_record_digest: String,
        successor_effective_from_ms: u64,
    },
    CorrectivelyRevokedAtVerification {
        invalid_from_ms: u64,
    },
    ContradictedAtVerification {
        contradiction_id: String,
        effective_from_ms: u64,
    },
    DuplicateContradiction(String),
    DuplicateContradictionResolution(String),
    ResolutionWithoutContradiction(String),
    ResolutionNotAfterContradiction(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileLifecycleReport {
    pub disposition: ProfileLifecycleDisposition,
    pub profile_id: String,
    pub record_digest: Option<String>,
    pub provenance_attestation_digest: String,
    pub graph_digest: String,
    pub relation_completeness_digest: String,
    pub ledger_id: String,
    pub ledger_digest: Option<String>,
    pub verification_at_ms: u64,
    pub assessed_at_ms: u64,
    pub issues: Vec<ProfileLifecycleIssue>,
}

impl ProfileLifecycleReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveVerifierProfile {
    record_digest: String,
    graph_digest: String,
    relation_completeness_digest: String,
    provenance_attestation_digest: String,
    lifecycle_digest: String,
    verification_at_ms: u64,
    assessed_at_ms: u64,
}

impl ActiveVerifierProfile {
    pub fn record_digest(&self) -> &str {
        &self.record_digest
    }

    pub fn graph_digest(&self) -> &str {
        &self.graph_digest
    }

    pub fn relation_completeness_digest(&self) -> &str {
        &self.relation_completeness_digest
    }

    pub fn provenance_attestation_digest(&self) -> &str {
        &self.provenance_attestation_digest
    }

    pub fn lifecycle_digest(&self) -> &str {
        &self.lifecycle_digest
    }

    pub const fn verification_at_ms(&self) -> u64 {
        self.verification_at_ms
    }

    pub const fn assessed_at_ms(&self) -> u64 {
        self.assessed_at_ms
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileLifecycleAssessment {
    pub report: ProfileLifecycleReport,
    active: Option<ActiveVerifierProfile>,
}

impl ProfileLifecycleAssessment {
    pub fn active(&self) -> Option<&ActiveVerifierProfile> {
        self.active.as_ref()
    }

    pub fn into_active(self) -> Option<ActiveVerifierProfile> {
        self.active
    }
}

impl ProfileLifecycleLedger {
    pub fn validate(&self) -> bool {
        if self.schema_version != PROFILE_LIFECYCLE_LEDGER_SCHEMA_V1
            || !canonical_text(&self.ledger_id)
            || !canonical_text(&self.profile_id)
            || !digest_text(&self.record_digest)
            || self.events.len() > MAX_LIFECYCLE_EVENTS
            || !valid_refs(&self.evidence_refs)
        {
            return false;
        }

        let mut ids = BTreeSet::new();
        self.events.iter().all(|event| {
            canonical_text(&event.event_id)
                && ids.insert(event.event_id.as_str())
                && valid_event_kind(&event.kind)
                && valid_refs(&event.evidence_refs)
        })
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(LEDGER_DIGEST_DOMAIN);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.ledger_id);
        push_field(&mut hasher, &self.profile_id);
        push_field(&mut hasher, &self.record_digest);

        let mut events = self.events.iter().collect::<Vec<_>>();
        events.sort_by(|left, right| {
            (left.recorded_at_ms, left.event_id.as_str())
                .cmp(&(right.recorded_at_ms, right.event_id.as_str()))
        });
        for event in events {
            push_field(&mut hasher, &event.event_id);
            push_u64(&mut hasher, event.recorded_at_ms);
            push_event_kind(&mut hasher, &event.kind);
            push_sorted_refs(&mut hasher, &event.evidence_refs);
        }
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

pub fn assess_profile_lifecycle(
    record: &VersionedVerifierProfileRecord,
    provenance: &AuthenticatedProfileProvenance,
    ledger: &ProfileLifecycleLedger,
    verification_at_ms: u64,
    assessed_at_ms: u64,
) -> ProfileLifecycleAssessment {
    let record_digest = record.canonical_digest();
    let ledger_digest = ledger.canonical_digest();
    let mut issues = Vec::new();

    if !record.validate() {
        issues.push(ProfileLifecycleIssue::InvalidRecord);
    }
    if record_digest.as_deref() != Some(provenance.record_digest())
        || record_digest.as_deref() != Some(ledger.record_digest.as_str())
    {
        issues.push(ProfileLifecycleIssue::RecordDigestMismatch);
    }
    if record.profile_id != ledger.profile_id {
        issues.push(ProfileLifecycleIssue::ProfileIdMismatch);
    }
    if !ledger.validate() {
        issues.push(ProfileLifecycleIssue::InvalidLedger);
    }
    if assessed_at_ms < verification_at_ms {
        issues.push(ProfileLifecycleIssue::AssessmentPredatesVerification);
    }

    if issues.iter().any(is_structural_issue) {
        return assessment(
            record,
            provenance,
            ledger,
            record_digest,
            ledger_digest,
            verification_at_ms,
            assessed_at_ms,
            ProfileLifecycleDisposition::Invalid,
            issues,
            None,
        );
    }

    for event in &ledger.events {
        if event.recorded_at_ms > assessed_at_ms {
            issues.push(ProfileLifecycleIssue::FutureRecordedEvent(
                event.event_id.clone(),
            ));
        }
    }

    validate_contradiction_sequences(ledger, &mut issues);
    if issues.iter().any(is_structural_issue) {
        return assessment(
            record,
            provenance,
            ledger,
            record_digest,
            ledger_digest,
            verification_at_ms,
            assessed_at_ms,
            ProfileLifecycleDisposition::Invalid,
            issues,
            None,
        );
    }

    if !provenance.record_applies_at(verification_at_ms) {
        issues.push(ProfileLifecycleIssue::RecordNotApplicableAtVerification);
    } else if !provenance.prospectively_established_at(verification_at_ms) {
        issues.push(ProfileLifecycleIssue::ProvenanceNotProspectivelyEstablished);
    }

    let mut events = ledger.events.iter().collect::<Vec<_>>();
    events.sort_by(|left, right| {
        (left.recorded_at_ms, left.event_id.as_str())
            .cmp(&(right.recorded_at_ms, right.event_id.as_str()))
    });
    for event in events {
        if event.recorded_at_ms > assessed_at_ms {
            continue;
        }
        match &event.kind {
            ProfileLifecycleEventKind::Superseded {
                successor_record_digest,
                successor_effective_from_ms,
                ..
            } => {
                if *successor_effective_from_ms <= verification_at_ms {
                    issues.push(ProfileLifecycleIssue::SupersededAtVerification {
                        successor_record_digest: successor_record_digest.clone(),
                        successor_effective_from_ms: *successor_effective_from_ms,
                    });
                }
            }
            ProfileLifecycleEventKind::CorrectivelyRevoked {
                invalid_from_ms, ..
            } => {
                if *invalid_from_ms <= verification_at_ms {
                    issues.push(ProfileLifecycleIssue::CorrectivelyRevokedAtVerification {
                        invalid_from_ms: *invalid_from_ms,
                    });
                }
            }
            ProfileLifecycleEventKind::Contradicted {
                contradiction_id,
                effective_from_ms,
                ..
            } => {
                if *effective_from_ms <= verification_at_ms {
                    issues.push(ProfileLifecycleIssue::ContradictedAtVerification {
                        contradiction_id: contradiction_id.clone(),
                        effective_from_ms: *effective_from_ms,
                    });
                }
            }
            ProfileLifecycleEventKind::ContradictionResolved { .. } => {
                // Deliberately non-reactivating. Resolution is auditable but a
                // contradicted record remains quarantined; replacement is needed.
            }
        }
    }

    let blocked = issues.iter().any(is_blocking_issue);
    let disposition = if blocked {
        ProfileLifecycleDisposition::Blocked
    } else {
        ProfileLifecycleDisposition::Usable
    };
    let active = if disposition == ProfileLifecycleDisposition::Usable {
        Some(ActiveVerifierProfile {
            record_digest: record_digest.clone().expect("validated record"),
            graph_digest: provenance.graph_digest().to_string(),
            relation_completeness_digest: provenance
                .relation_completeness_digest()
                .to_string(),
            provenance_attestation_digest: provenance.attestation_digest().to_string(),
            lifecycle_digest: ledger_digest.clone().expect("validated ledger"),
            verification_at_ms,
            assessed_at_ms,
        })
    } else {
        None
    };

    assessment(
        record,
        provenance,
        ledger,
        record_digest,
        ledger_digest,
        verification_at_ms,
        assessed_at_ms,
        disposition,
        issues,
        active,
    )
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProvenancePolicyTracker {
    latest_sequence: Option<u64>,
    latest_issued_at_ms: Option<u64>,
    latest_digest: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProvenancePolicyTrackingError {
    InvalidPolicy,
    SequenceRollback { latest: u64, proposed: u64 },
    SequenceCollision { sequence: u64 },
    IssuedAtRegressed { latest: u64, proposed: u64 },
}

impl ProvenancePolicyTracker {
    pub fn accept(
        &mut self,
        policy: &ProfileProvenancePolicy,
    ) -> Result<String, ProvenancePolicyTrackingError> {
        if !policy.validate() {
            return Err(ProvenancePolicyTrackingError::InvalidPolicy);
        }
        let digest = policy
            .canonical_digest()
            .ok_or(ProvenancePolicyTrackingError::InvalidPolicy)?;
        if let Some(latest) = self.latest_sequence {
            if policy.sequence < latest {
                return Err(ProvenancePolicyTrackingError::SequenceRollback {
                    latest,
                    proposed: policy.sequence,
                });
            }
            if policy.sequence == latest {
                if self.latest_digest.as_deref() == Some(digest.as_str()) {
                    return Ok(digest);
                }
                return Err(ProvenancePolicyTrackingError::SequenceCollision {
                    sequence: policy.sequence,
                });
            }
        }
        if let Some(latest) = self.latest_issued_at_ms {
            if policy.issued_at_ms < latest {
                return Err(ProvenancePolicyTrackingError::IssuedAtRegressed {
                    latest,
                    proposed: policy.issued_at_ms,
                });
            }
        }
        self.latest_sequence = Some(policy.sequence);
        self.latest_issued_at_ms = Some(policy.issued_at_ms);
        self.latest_digest = Some(digest.clone());
        Ok(digest)
    }

    pub const fn latest_sequence(&self) -> Option<u64> {
        self.latest_sequence
    }

    pub fn latest_digest(&self) -> Option<&str> {
        self.latest_digest.as_deref()
    }
}

#[allow(clippy::too_many_arguments)]
fn assessment(
    record: &VersionedVerifierProfileRecord,
    provenance: &AuthenticatedProfileProvenance,
    ledger: &ProfileLifecycleLedger,
    record_digest: Option<String>,
    ledger_digest: Option<String>,
    verification_at_ms: u64,
    assessed_at_ms: u64,
    disposition: ProfileLifecycleDisposition,
    issues: Vec<ProfileLifecycleIssue>,
    active: Option<ActiveVerifierProfile>,
) -> ProfileLifecycleAssessment {
    ProfileLifecycleAssessment {
        report: ProfileLifecycleReport {
            disposition,
            profile_id: record.profile_id.clone(),
            record_digest,
            provenance_attestation_digest: provenance.attestation_digest().to_string(),
            graph_digest: provenance.graph_digest().to_string(),
            relation_completeness_digest: provenance
                .relation_completeness_digest()
                .to_string(),
            ledger_id: ledger.ledger_id.clone(),
            ledger_digest,
            verification_at_ms,
            assessed_at_ms,
            issues,
        },
        active,
    }
}

fn validate_contradiction_sequences(
    ledger: &ProfileLifecycleLedger,
    issues: &mut Vec<ProfileLifecycleIssue>,
) {
    let mut contradictions = BTreeMap::<&str, u64>::new();
    let mut resolutions = BTreeSet::<&str>::new();
    for event in &ledger.events {
        match &event.kind {
            ProfileLifecycleEventKind::Contradicted {
                contradiction_id, ..
            } => {
                if contradictions
                    .insert(contradiction_id.as_str(), event.recorded_at_ms)
                    .is_some()
                {
                    issues.push(ProfileLifecycleIssue::DuplicateContradiction(
                        contradiction_id.clone(),
                    ));
                }
            }
            ProfileLifecycleEventKind::ContradictionResolved {
                contradiction_id, ..
            } => {
                if !resolutions.insert(contradiction_id.as_str()) {
                    issues.push(ProfileLifecycleIssue::DuplicateContradictionResolution(
                        contradiction_id.clone(),
                    ));
                }
            }
            _ => {}
        }
    }
    for event in &ledger.events {
        let ProfileLifecycleEventKind::ContradictionResolved {
            contradiction_id, ..
        } = &event.kind
        else {
            continue;
        };
        let Some(contradiction_time) = contradictions.get(contradiction_id.as_str()) else {
            issues.push(ProfileLifecycleIssue::ResolutionWithoutContradiction(
                contradiction_id.clone(),
            ));
            continue;
        };
        if event.recorded_at_ms <= *contradiction_time {
            issues.push(ProfileLifecycleIssue::ResolutionNotAfterContradiction(
                contradiction_id.clone(),
            ));
        }
    }
}

fn valid_event_kind(kind: &ProfileLifecycleEventKind) -> bool {
    match kind {
        ProfileLifecycleEventKind::Superseded {
            successor_record_digest,
            reason_ref,
            ..
        } => digest_text(successor_record_digest) && canonical_text(reason_ref),
        ProfileLifecycleEventKind::CorrectivelyRevoked { reason_ref, .. } => {
            canonical_text(reason_ref)
        }
        ProfileLifecycleEventKind::Contradicted {
            contradiction_id,
            contradiction_ref,
            ..
        } => canonical_text(contradiction_id) && canonical_text(contradiction_ref),
        ProfileLifecycleEventKind::ContradictionResolved {
            contradiction_id,
            resolution_ref,
        } => canonical_text(contradiction_id) && canonical_text(resolution_ref),
    }
}

fn is_structural_issue(issue: &ProfileLifecycleIssue) -> bool {
    matches!(
        issue,
        ProfileLifecycleIssue::InvalidRecord
            | ProfileLifecycleIssue::RecordDigestMismatch
            | ProfileLifecycleIssue::ProfileIdMismatch
            | ProfileLifecycleIssue::InvalidLedger
            | ProfileLifecycleIssue::AssessmentPredatesVerification
            | ProfileLifecycleIssue::FutureRecordedEvent(_)
            | ProfileLifecycleIssue::DuplicateContradiction(_)
            | ProfileLifecycleIssue::DuplicateContradictionResolution(_)
            | ProfileLifecycleIssue::ResolutionWithoutContradiction(_)
            | ProfileLifecycleIssue::ResolutionNotAfterContradiction(_)
    )
}

fn is_blocking_issue(issue: &ProfileLifecycleIssue) -> bool {
    matches!(
        issue,
        ProfileLifecycleIssue::RecordNotApplicableAtVerification
            | ProfileLifecycleIssue::ProvenanceNotProspectivelyEstablished
            | ProfileLifecycleIssue::SupersededAtVerification { .. }
            | ProfileLifecycleIssue::CorrectivelyRevokedAtVerification { .. }
            | ProfileLifecycleIssue::ContradictedAtVerification { .. }
    )
}

fn push_event_kind(hasher: &mut blake3::Hasher, kind: &ProfileLifecycleEventKind) {
    match kind {
        ProfileLifecycleEventKind::Superseded {
            successor_record_digest,
            successor_effective_from_ms,
            reason_ref,
        } => {
            push_field(hasher, "superseded");
            push_field(hasher, successor_record_digest);
            push_u64(hasher, *successor_effective_from_ms);
            push_field(hasher, reason_ref);
        }
        ProfileLifecycleEventKind::CorrectivelyRevoked {
            invalid_from_ms,
            reason_ref,
        } => {
            push_field(hasher, "correctively-revoked");
            push_u64(hasher, *invalid_from_ms);
            push_field(hasher, reason_ref);
        }
        ProfileLifecycleEventKind::Contradicted {
            contradiction_id,
            effective_from_ms,
            contradiction_ref,
        } => {
            push_field(hasher, "contradicted");
            push_field(hasher, contradiction_id);
            push_u64(hasher, *effective_from_ms);
            push_field(hasher, contradiction_ref);
        }
        ProfileLifecycleEventKind::ContradictionResolved {
            contradiction_id,
            resolution_ref,
        } => {
            push_field(hasher, "contradiction-resolved");
            push_field(hasher, contradiction_id);
            push_field(hasher, resolution_ref);
        }
    }
}

fn canonical_text(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty() && trimmed == value && value.len() <= MAX_TEXT_BYTES
}

fn digest_text(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("blake3:") else {
        return false;
    };
    hex.len() == 64
        && hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_refs(refs: &[String]) -> bool {
    !refs.is_empty()
        && refs.len() <= MAX_EVIDENCE_REFS
        && refs.iter().all(|value| canonical_text(value))
        && refs.iter().collect::<BTreeSet<_>>().len() == refs.len()
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn push_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_be_bytes());
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.iter().collect::<Vec<_>>();
    refs.sort();
    for value in refs {
        push_field(hasher, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use symthaea_evidence_independence_provenance::{
        ProfileAttestationEnvelope, ProfileClaimScope, ProfileIssuerKeyRecord,
        PROFILE_ATTESTATION_SCHEMA_V1, PROFILE_PROVENANCE_POLICY_SCHEMA_V1,
        PROFILE_RECORD_SCHEMA_V1, verify_profile_provenance,
    };
    use symthaea_evidence_verifier_diversity::VerifierFaultDomainProfile;

    fn digest(byte: char) -> String {
        format!("blake3:{}", byte.to_string().repeat(64))
    }

    fn fixture() -> (
        VersionedVerifierProfileRecord,
        ProfileProvenancePolicy,
        AuthenticatedProfileProvenance,
    ) {
        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let record = VersionedVerifierProfileRecord {
            schema_version: PROFILE_RECORD_SCHEMA_V1.into(),
            profile_id: "profile:a".into(),
            revision: 1,
            profile: VerifierFaultDomainProfile {
                verifier_ref: "verifier:a".into(),
                organization_domain: "org:a".into(),
                review_process_domain: "process:a".into(),
                toolchain_domain: "tool:a".into(),
                evidence_source_domain: "source:a".into(),
                evidence_refs: vec!["profile:evidence".into()],
            },
            effective_from_ms: 1_000,
            effective_until_ms: Some(10_000),
            graph_digest: digest('a'),
            relation_completeness_digest: digest('b'),
            evidence_refs: vec!["record:evidence".into()],
        };
        let scopes = vec![
            ProfileClaimScope::ProfileIdentity,
            ProfileClaimScope::FaultDomainAssignment,
            ProfileClaimScope::GraphBinding,
            ProfileClaimScope::RelationCompletenessBinding,
        ];
        let policy = ProfileProvenancePolicy {
            schema_version: PROFILE_PROVENANCE_POLICY_SCHEMA_V1.into(),
            policy_id: "profile-trust:v1".into(),
            sequence: 3,
            issued_at_ms: 500,
            expires_at_ms: 20_000,
            required_scopes: scopes.clone(),
            trusted_keys: vec![ProfileIssuerKeyRecord {
                key_id: "issuer:a".into(),
                public_key_ed25519_hex: hex::encode(signing_key.verifying_key().as_bytes()),
                valid_from_ms: 400,
                valid_until_ms: Some(15_000),
                revoked_at_ms: None,
                allowed_scopes: scopes.clone(),
                evidence_refs: vec!["issuer:evidence".into()],
            }],
            evidence_refs: vec!["policy:evidence".into()],
        };
        let mut attestation = ProfileAttestationEnvelope {
            schema_version: PROFILE_ATTESTATION_SCHEMA_V1.into(),
            record_digest: record.canonical_digest().unwrap(),
            policy_digest: policy.canonical_digest().unwrap(),
            issuer_key_id: "issuer:a".into(),
            issuer_public_key_ed25519_hex: hex::encode(signing_key.verifying_key().as_bytes()),
            issued_at_ms: 900,
            expires_at_ms: Some(12_000),
            scopes,
            nonce_blake3_hex: "c".repeat(64),
            signature_ed25519_hex: "0".repeat(128),
        };
        let message = attestation.canonical_unsigned_bytes().unwrap();
        attestation.signature_ed25519_hex = hex::encode(signing_key.sign(&message).to_bytes());
        let authenticated = verify_profile_provenance(&record, &policy, &attestation)
            .into_authenticated()
            .unwrap();
        (record, policy, authenticated)
    }

    fn ledger(
        record: &VersionedVerifierProfileRecord,
        events: Vec<ProfileLifecycleEvent>,
    ) -> ProfileLifecycleLedger {
        ProfileLifecycleLedger {
            schema_version: PROFILE_LIFECYCLE_LEDGER_SCHEMA_V1.into(),
            ledger_id: "lifecycle:v1".into(),
            profile_id: record.profile_id.clone(),
            record_digest: record.canonical_digest().unwrap(),
            events,
            evidence_refs: vec!["lifecycle:evidence".into()],
        }
    }

    fn event(id: &str, recorded: u64, kind: ProfileLifecycleEventKind) -> ProfileLifecycleEvent {
        ProfileLifecycleEvent {
            event_id: id.into(),
            recorded_at_ms: recorded,
            kind,
            evidence_refs: vec![format!("event:{id}")],
        }
    }

    #[test]
    fn ordinary_later_supersession_preserves_historical_reliance() {
        let (record, _, provenance) = fixture();
        let ledger = ledger(
            &record,
            vec![event(
                "supersede",
                5_000,
                ProfileLifecycleEventKind::Superseded {
                    successor_record_digest: digest('d'),
                    successor_effective_from_ms: 4_000,
                    reason_ref: "reason:normal-rotation".into(),
                },
            )],
        );
        let assessment = assess_profile_lifecycle(&record, &provenance, &ledger, 2_000, 6_000);
        assert_eq!(assessment.report.disposition, ProfileLifecycleDisposition::Usable);
        assert!(assessment.active().is_some());
    }

    #[test]
    fn supersession_effective_before_verification_blocks_old_record() {
        let (record, _, provenance) = fixture();
        let ledger = ledger(
            &record,
            vec![event(
                "supersede",
                2_000,
                ProfileLifecycleEventKind::Superseded {
                    successor_record_digest: digest('d'),
                    successor_effective_from_ms: 1_500,
                    reason_ref: "reason:normal-rotation".into(),
                },
            )],
        );
        let assessment = assess_profile_lifecycle(&record, &provenance, &ledger, 2_000, 3_000);
        assert_eq!(assessment.report.disposition, ProfileLifecycleDisposition::Blocked);
    }

    #[test]
    fn later_corrective_revocation_can_poison_historical_reliance() {
        let (record, _, provenance) = fixture();
        let ledger = ledger(
            &record,
            vec![event(
                "correction",
                5_000,
                ProfileLifecycleEventKind::CorrectivelyRevoked {
                    invalid_from_ms: 1_000,
                    reason_ref: "reason:misattributed-toolchain".into(),
                },
            )],
        );
        let assessment = assess_profile_lifecycle(&record, &provenance, &ledger, 2_000, 6_000);
        assert_eq!(assessment.report.disposition, ProfileLifecycleDisposition::Blocked);
        assert!(assessment.report.issues.iter().any(|issue| matches!(
            issue,
            ProfileLifecycleIssue::CorrectivelyRevokedAtVerification { .. }
        )));
    }

    #[test]
    fn correction_effective_after_verification_preserves_earlier_history() {
        let (record, _, provenance) = fixture();
        let ledger = ledger(
            &record,
            vec![event(
                "correction",
                5_000,
                ProfileLifecycleEventKind::CorrectivelyRevoked {
                    invalid_from_ms: 3_000,
                    reason_ref: "reason:later-boundary".into(),
                },
            )],
        );
        let assessment = assess_profile_lifecycle(&record, &provenance, &ledger, 2_000, 6_000);
        assert_eq!(assessment.report.disposition, ProfileLifecycleDisposition::Usable);
    }

    #[test]
    fn contradiction_stays_blocking_after_resolution() {
        let (record, _, provenance) = fixture();
        let ledger = ledger(
            &record,
            vec![
                event(
                    "contradiction",
                    4_000,
                    ProfileLifecycleEventKind::Contradicted {
                        contradiction_id: "contradiction:x".into(),
                        effective_from_ms: 1_500,
                        contradiction_ref: "evidence:contradiction".into(),
                    },
                ),
                event(
                    "resolution",
                    5_000,
                    ProfileLifecycleEventKind::ContradictionResolved {
                        contradiction_id: "contradiction:x".into(),
                        resolution_ref: "review:resolution".into(),
                    },
                ),
            ],
        );
        let assessment = assess_profile_lifecycle(&record, &provenance, &ledger, 2_000, 6_000);
        assert_eq!(assessment.report.disposition, ProfileLifecycleDisposition::Blocked);
    }

    #[test]
    fn orphan_resolution_is_structurally_invalid() {
        let (record, _, provenance) = fixture();
        let ledger = ledger(
            &record,
            vec![event(
                "resolution",
                5_000,
                ProfileLifecycleEventKind::ContradictionResolved {
                    contradiction_id: "contradiction:missing".into(),
                    resolution_ref: "review:resolution".into(),
                },
            )],
        );
        let assessment = assess_profile_lifecycle(&record, &provenance, &ledger, 2_000, 6_000);
        assert_eq!(assessment.report.disposition, ProfileLifecycleDisposition::Invalid);
    }

    #[test]
    fn policy_tracker_rejects_rollback_and_collision() {
        let (_, policy, _) = fixture();
        let mut tracker = ProvenancePolicyTracker::default();
        tracker.accept(&policy).unwrap();

        let mut rollback = policy.clone();
        rollback.sequence = 2;
        rollback.issued_at_ms = 600;
        assert!(matches!(
            tracker.accept(&rollback),
            Err(ProvenancePolicyTrackingError::SequenceRollback { .. })
        ));

        let mut collision = policy.clone();
        collision.evidence_refs = vec!["policy:different".into()];
        assert!(matches!(
            tracker.accept(&collision),
            Err(ProvenancePolicyTrackingError::SequenceCollision { .. })
        ));
    }

    #[test]
    fn policy_tracker_rejects_issued_at_regression() {
        let (_, mut policy, _) = fixture();
        let mut tracker = ProvenancePolicyTracker::default();
        tracker.accept(&policy).unwrap();
        policy.sequence += 1;
        policy.issued_at_ms = 400;
        assert!(matches!(
            tracker.accept(&policy),
            Err(ProvenancePolicyTrackingError::IssuedAtRegressed { .. })
        ));
    }

    #[test]
    fn lifecycle_digest_is_event_order_independent() {
        let (record, _, _) = fixture();
        let a = event(
            "a",
            4_000,
            ProfileLifecycleEventKind::CorrectivelyRevoked {
                invalid_from_ms: 3_000,
                reason_ref: "reason:a".into(),
            },
        );
        let b = event(
            "b",
            5_000,
            ProfileLifecycleEventKind::Superseded {
                successor_record_digest: digest('d'),
                successor_effective_from_ms: 4_500,
                reason_ref: "reason:b".into(),
            },
        );
        let left = ledger(&record, vec![a.clone(), b.clone()]);
        let right = ledger(&record, vec![b, a]);
        assert_eq!(left.canonical_digest(), right.canonical_digest());
    }
}
