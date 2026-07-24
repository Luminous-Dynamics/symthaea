// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Incremental export and verification of tamper-evident audit evidence.

use crate::audit::{AuditAppendError, AuditEvent, AuditJournal, compute_audit_event_hash};
use crate::crypto_digest::{Sha256, Sha256Digest};
use serde::{Deserialize, Serialize};

pub const AUDIT_SEGMENT_SCHEMA: &str = "symthaea.fabrication.audit-segment.v1";
pub const MAX_AUDIT_SEGMENT_EVENTS: usize = 8192;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditSegment {
    pub schema_version: String,
    pub start_sequence: u64,
    pub previous_head: Option<Sha256Digest>,
    pub events: Vec<AuditEvent>,
    pub segment_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditSegmentError {
    JournalNotIntact,
    StartSequenceZero,
    StartBeyondJournal {
        start_sequence: u64,
        event_count: usize,
    },
    EmptySegment,
    TooManyEvents {
        actual: usize,
        maximum: usize,
    },
    UnsupportedSchema,
    StartSequenceMismatch,
    SequenceMismatch {
        index: usize,
        actual: u64,
        expected: u64,
    },
    TimestampRegressed {
        index: usize,
    },
    PreviousHashMismatch {
        index: usize,
    },
    RecordHashMismatch {
        index: usize,
    },
    SegmentDigestMismatch,
    Encoding(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditSegmentVerificationReport {
    pub violations: Vec<AuditSegmentError>,
    pub verified_head: Option<Sha256Digest>,
}

impl AuditSegmentVerificationReport {
    pub fn intact(&self) -> bool {
        self.violations.is_empty()
    }
}

pub fn export_audit_segment(
    journal: &AuditJournal,
    start_sequence: u64,
    maximum_events: usize,
) -> Result<AuditSegment, AuditSegmentError> {
    if !journal.verify().intact() {
        return Err(AuditSegmentError::JournalNotIntact);
    }
    if start_sequence == 0 {
        return Err(AuditSegmentError::StartSequenceZero);
    }
    if maximum_events == 0 {
        return Err(AuditSegmentError::EmptySegment);
    }
    let start_index =
        usize::try_from(start_sequence - 1).map_err(|_| AuditSegmentError::StartBeyondJournal {
            start_sequence,
            event_count: journal.events.len(),
        })?;
    if start_index >= journal.events.len() {
        return Err(AuditSegmentError::StartBeyondJournal {
            start_sequence,
            event_count: journal.events.len(),
        });
    }
    let count = maximum_events.min(MAX_AUDIT_SEGMENT_EVENTS);
    let end = start_index.saturating_add(count).min(journal.events.len());
    let events = journal.events[start_index..end].to_vec();
    let previous_head = start_index
        .checked_sub(1)
        .map(|index| journal.events[index].record_hash);
    let segment_digest = digest_segment_fields(start_sequence, previous_head, &events)?;
    Ok(AuditSegment {
        schema_version: AUDIT_SEGMENT_SCHEMA.into(),
        start_sequence,
        previous_head,
        events,
        segment_digest,
    })
}

pub fn verify_audit_segment(
    segment: &AuditSegment,
    expected_previous_head: Option<Sha256Digest>,
) -> AuditSegmentVerificationReport {
    let mut violations = Vec::new();
    if segment.schema_version != AUDIT_SEGMENT_SCHEMA {
        violations.push(AuditSegmentError::UnsupportedSchema);
    }
    if segment.start_sequence == 0 {
        violations.push(AuditSegmentError::StartSequenceZero);
    }
    if segment.events.is_empty() {
        violations.push(AuditSegmentError::EmptySegment);
    }
    if segment.events.len() > MAX_AUDIT_SEGMENT_EVENTS {
        violations.push(AuditSegmentError::TooManyEvents {
            actual: segment.events.len(),
            maximum: MAX_AUDIT_SEGMENT_EVENTS,
        });
    }
    if segment.previous_head != expected_previous_head {
        violations.push(AuditSegmentError::PreviousHashMismatch { index: 0 });
    }
    if segment
        .events
        .first()
        .is_some_and(|event| event.sequence != segment.start_sequence)
    {
        violations.push(AuditSegmentError::StartSequenceMismatch);
    }

    let mut previous_hash = segment.previous_head;
    let mut previous_timestamp = None;
    let mut verified_head = None;
    for (index, event) in segment.events.iter().enumerate() {
        let expected_sequence = segment.start_sequence.saturating_add(index as u64);
        if event.sequence != expected_sequence {
            violations.push(AuditSegmentError::SequenceMismatch {
                index,
                actual: event.sequence,
                expected: expected_sequence,
            });
        }
        if previous_timestamp.is_some_and(|timestamp| event.timestamp_unix_s < timestamp) {
            violations.push(AuditSegmentError::TimestampRegressed { index });
        }
        if event.previous_hash != previous_hash {
            violations.push(AuditSegmentError::PreviousHashMismatch { index });
        }
        match compute_audit_event_hash(event) {
            Ok(expected) if expected == event.record_hash => verified_head = Some(expected),
            _ => violations.push(AuditSegmentError::RecordHashMismatch { index }),
        }
        previous_hash = Some(event.record_hash);
        previous_timestamp = Some(event.timestamp_unix_s);
    }
    match digest_segment_fields(
        segment.start_sequence,
        segment.previous_head,
        &segment.events,
    ) {
        Ok(expected) if expected == segment.segment_digest => {}
        Ok(_) => violations.push(AuditSegmentError::SegmentDigestMismatch),
        Err(error) => violations.push(error),
    }
    AuditSegmentVerificationReport {
        violations,
        verified_head,
    }
}

fn digest_segment_fields(
    start_sequence: u64,
    previous_head: Option<Sha256Digest>,
    events: &[AuditEvent],
) -> Result<Sha256Digest, AuditSegmentError> {
    #[derive(Serialize)]
    struct Body<'a> {
        start_sequence: u64,
        previous_head: Option<Sha256Digest>,
        events: &'a [AuditEvent],
    }
    let bytes = serde_json::to_vec(&Body {
        start_sequence,
        previous_head,
        events,
    })
    .map_err(|error| AuditSegmentError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.audit-segment-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

impl From<AuditAppendError> for AuditSegmentError {
    fn from(error: AuditAppendError) -> Self {
        Self::Encoding(format!("{error:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::{AuditAction, AuditJournal};
    use crate::crypto_digest::sha256;

    fn journal() -> AuditJournal {
        let mut journal = AuditJournal::default();
        for sequence in 0..5 {
            journal
                .append(
                    100 + sequence,
                    "operator",
                    AuditAction::Other(format!("step-{sequence}")),
                    sha256(&[sequence as u8]),
                    None,
                )
                .unwrap();
        }
        journal
    }

    #[test]
    fn segment_verifies_from_known_predecessor() {
        let journal = journal();
        let segment = export_audit_segment(&journal, 3, 2).unwrap();
        let report = verify_audit_segment(&segment, Some(journal.events[1].record_hash));
        assert!(report.intact());
        assert_eq!(report.verified_head, Some(journal.events[3].record_hash));
    }

    #[test]
    fn wrong_predecessor_and_event_tamper_are_detected() {
        let journal = journal();
        let mut segment = export_audit_segment(&journal, 2, 3).unwrap();
        assert!(!verify_audit_segment(&segment, None).intact());
        segment.events[1].actor = "intruder".into();
        let report = verify_audit_segment(&segment, Some(journal.events[0].record_hash));
        assert!(report.violations.iter().any(|violation| matches!(
            violation,
            AuditSegmentError::RecordHashMismatch { index: 1 }
        )));
    }
}

pub const AUDIT_ANCHOR_SCHEMA: &str = "symthaea.fabrication.audit-anchor.v1";
pub const MAX_AUDIT_ANCHOR_ID_BYTES: usize = 256;
pub const MAX_AUDIT_ANCHOR_SIGNATURE_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditAnchor {
    pub schema_version: String,
    pub anchor_id: String,
    pub anchored_at_unix_s: u64,
    pub event_count: u64,
    pub journal_digest: Sha256Digest,
    pub journal_head: Sha256Digest,
    pub trust_snapshot_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedAuditAnchor {
    pub anchor: AuditAnchor,
    pub algorithm: crate::attestation::SignatureAlgorithm,
    pub key_id: String,
    pub signature: Vec<u8>,
}

pub trait AuditAnchorSigner {
    fn algorithm(&self) -> crate::attestation::SignatureAlgorithm;
    fn key_id(&self) -> &str;
    fn sign_audit_anchor(&self, message: &[u8]) -> Result<Vec<u8>, String>;
}

pub trait AuditAnchorVerifier {
    fn verify_audit_anchor(
        &self,
        algorithm: &crate::attestation::SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditAnchorError {
    EmptyJournal,
    InvalidAnchorId,
    AnchorIdTooLong,
    InvalidAlgorithm,
    InvalidKeyId,
    InvalidTrustSnapshot,
    TrustSnapshotStale,
    AnchorFromFuture,
    AnchorOutsideSnapshot,
    SignerIneligible(crate::trust::KeyEligibility),
    JournalNotIntact,
    JournalDigestMismatch,
    JournalHeadMismatch,
    EventCountMismatch,
    TrustSnapshotMismatch,
    UnsupportedSchema,
    EmptySignature,
    SignatureTooLarge,
    Signing(String),
    VerificationProvider(String),
    InvalidSignature,
    Encoding(String),
}

#[derive(Debug, Clone)]
pub struct VerifiedAuditAnchor {
    anchor: AuditAnchor,
    algorithm: crate::attestation::SignatureAlgorithm,
    key_id: String,
}

impl VerifiedAuditAnchor {
    pub fn anchor(&self) -> &AuditAnchor {
        &self.anchor
    }
    pub fn algorithm(&self) -> &crate::attestation::SignatureAlgorithm {
        &self.algorithm
    }
    pub fn key_id(&self) -> &str {
        &self.key_id
    }
}

pub fn sign_audit_anchor(
    journal: &AuditJournal,
    snapshot: &crate::trust::TrustSnapshot,
    anchored_at_unix_s: u64,
    anchor_id: impl Into<String>,
    signer: &dyn AuditAnchorSigner,
) -> Result<SignedAuditAnchor, AuditAnchorError> {
    let anchor_id = anchor_id.into();
    validate_anchor_id(&anchor_id)?;
    if journal.events.is_empty() {
        return Err(AuditAnchorError::EmptyJournal);
    }
    if !journal.verify().intact() {
        return Err(AuditAnchorError::JournalNotIntact);
    }
    snapshot
        .validate()
        .map_err(|_| AuditAnchorError::InvalidTrustSnapshot)?;
    if !snapshot.is_fresh_at(anchored_at_unix_s) {
        return Err(AuditAnchorError::TrustSnapshotStale);
    }
    let eligibility = snapshot.key_eligibility(
        &signer.algorithm(),
        signer.key_id(),
        crate::trust::KeyUsage::AuditAnchor,
        anchored_at_unix_s,
    );
    if eligibility != crate::trust::KeyEligibility::Eligible {
        return Err(AuditAnchorError::SignerIneligible(eligibility));
    }
    if !signer.algorithm().is_canonical() {
        return Err(AuditAnchorError::InvalidAlgorithm);
    }
    if !canonical_identity(signer.key_id()) {
        return Err(AuditAnchorError::InvalidKeyId);
    }
    let journal_digest = crate::audit::digest_audit_journal(journal)
        .map_err(|_| AuditAnchorError::JournalNotIntact)?;
    let trust_snapshot_digest = crate::trust::digest_trust_snapshot(snapshot)
        .map_err(|_| AuditAnchorError::InvalidTrustSnapshot)?;
    let anchor = AuditAnchor {
        schema_version: AUDIT_ANCHOR_SCHEMA.into(),
        anchor_id,
        anchored_at_unix_s,
        event_count: journal.events.len() as u64,
        journal_digest,
        journal_head: journal.head().ok_or(AuditAnchorError::EmptyJournal)?,
        trust_snapshot_digest,
    };
    let message = audit_anchor_message(&anchor)?;
    let signature = signer
        .sign_audit_anchor(&message)
        .map_err(AuditAnchorError::Signing)?;
    if signature.is_empty() {
        return Err(AuditAnchorError::EmptySignature);
    }
    if signature.len() > MAX_AUDIT_ANCHOR_SIGNATURE_BYTES {
        return Err(AuditAnchorError::SignatureTooLarge);
    }
    Ok(SignedAuditAnchor {
        anchor,
        algorithm: signer.algorithm(),
        key_id: signer.key_id().to_string(),
        signature,
    })
}

pub fn verify_signed_audit_anchor(
    signed: &SignedAuditAnchor,
    journal: &AuditJournal,
    snapshot: &crate::trust::TrustSnapshot,
    evaluation_time_unix_s: u64,
    verifier: &dyn AuditAnchorVerifier,
) -> Result<VerifiedAuditAnchor, Vec<AuditAnchorError>> {
    let mut errors = Vec::new();
    if signed.anchor.schema_version != AUDIT_ANCHOR_SCHEMA {
        errors.push(AuditAnchorError::UnsupportedSchema);
    }
    if let Err(error) = validate_anchor_id(&signed.anchor.anchor_id) {
        errors.push(error);
    }
    if !signed.algorithm.is_canonical() {
        errors.push(AuditAnchorError::InvalidAlgorithm);
    }
    if !canonical_identity(&signed.key_id) {
        errors.push(AuditAnchorError::InvalidKeyId);
    }
    if signed.signature.is_empty() {
        errors.push(AuditAnchorError::EmptySignature);
    } else if signed.signature.len() > MAX_AUDIT_ANCHOR_SIGNATURE_BYTES {
        errors.push(AuditAnchorError::SignatureTooLarge);
    }
    if signed.anchor.anchored_at_unix_s > evaluation_time_unix_s {
        errors.push(AuditAnchorError::AnchorFromFuture);
    }
    if !snapshot.is_fresh_at(signed.anchor.anchored_at_unix_s) {
        errors.push(AuditAnchorError::AnchorOutsideSnapshot);
    }
    if !journal.verify().intact() {
        errors.push(AuditAnchorError::JournalNotIntact);
    }
    if journal.events.len() as u64 != signed.anchor.event_count {
        errors.push(AuditAnchorError::EventCountMismatch);
    }
    if journal.head() != Some(signed.anchor.journal_head) {
        errors.push(AuditAnchorError::JournalHeadMismatch);
    }
    match crate::audit::digest_audit_journal(journal) {
        Ok(digest) if digest == signed.anchor.journal_digest => {}
        _ => errors.push(AuditAnchorError::JournalDigestMismatch),
    }
    if snapshot.validate().is_err() {
        errors.push(AuditAnchorError::InvalidTrustSnapshot);
    } else {
        if !snapshot.is_fresh_at(evaluation_time_unix_s) {
            errors.push(AuditAnchorError::TrustSnapshotStale);
        }
        match crate::trust::digest_trust_snapshot(snapshot) {
            Ok(digest) if digest == signed.anchor.trust_snapshot_digest => {}
            _ => errors.push(AuditAnchorError::TrustSnapshotMismatch),
        }
        for eligibility_time in [signed.anchor.anchored_at_unix_s, evaluation_time_unix_s] {
            let eligibility = snapshot.key_eligibility(
                &signed.algorithm,
                &signed.key_id,
                crate::trust::KeyUsage::AuditAnchor,
                eligibility_time,
            );
            if eligibility != crate::trust::KeyEligibility::Eligible {
                errors.push(AuditAnchorError::SignerIneligible(eligibility));
            }
        }
    }
    match audit_anchor_message(&signed.anchor) {
        Ok(message) => match verifier.verify_audit_anchor(
            &signed.algorithm,
            &signed.key_id,
            &message,
            &signed.signature,
        ) {
            Ok(true) => {}
            Ok(false) => errors.push(AuditAnchorError::InvalidSignature),
            Err(reason) => errors.push(AuditAnchorError::VerificationProvider(reason)),
        },
        Err(error) => errors.push(error),
    }
    if errors.is_empty() {
        Ok(VerifiedAuditAnchor {
            anchor: signed.anchor.clone(),
            algorithm: signed.algorithm.clone(),
            key_id: signed.key_id.clone(),
        })
    } else {
        Err(errors)
    }
}

pub fn digest_audit_anchor(anchor: &AuditAnchor) -> Result<Sha256Digest, AuditAnchorError> {
    if anchor.schema_version != AUDIT_ANCHOR_SCHEMA {
        return Err(AuditAnchorError::UnsupportedSchema);
    }
    validate_anchor_id(&anchor.anchor_id)?;
    let bytes = serde_json::to_vec(anchor)
        .map_err(|error| AuditAnchorError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.audit-anchor-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

fn audit_anchor_message(anchor: &AuditAnchor) -> Result<Vec<u8>, AuditAnchorError> {
    let bytes = serde_json::to_vec(anchor)
        .map_err(|error| AuditAnchorError::Encoding(error.to_string()))?;
    let mut message = b"symthaea.fabrication.audit-anchor-signature.v1\0".to_vec();
    message.extend_from_slice(&bytes);
    Ok(message)
}

fn validate_anchor_id(value: &str) -> Result<(), AuditAnchorError> {
    if value.trim().is_empty() || value != value.trim() {
        return Err(AuditAnchorError::InvalidAnchorId);
    }
    if value.len() > MAX_AUDIT_ANCHOR_ID_BYTES {
        return Err(AuditAnchorError::AnchorIdTooLong);
    }
    Ok(())
}

fn canonical_identity(value: &str) -> bool {
    !value.trim().is_empty() && value == value.trim() && value.len() <= 256
}

#[cfg(test)]
mod anchor_tests {
    use super::*;
    use crate::attestation::SignatureAlgorithm;
    use crate::audit::{AuditAction, AuditJournal};
    use crate::crypto_digest::sha256;
    use crate::trust::{KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot};
    use std::collections::BTreeSet;

    struct Provider;
    impl AuditAnchorSigner for Provider {
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::Ed25519
        }
        fn key_id(&self) -> &str {
            "audit-root"
        }
        fn sign_audit_anchor(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            Ok(sha256(message).0.to_vec())
        }
    }
    impl AuditAnchorVerifier for Provider {
        fn verify_audit_anchor(
            &self,
            algorithm: &SignatureAlgorithm,
            key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            Ok(algorithm == &SignatureAlgorithm::Ed25519
                && key_id == "audit-root"
                && signature == sha256(message).0.as_slice())
        }
    }

    fn evidence() -> (AuditJournal, TrustSnapshot) {
        let mut journal = AuditJournal::default();
        journal
            .append(
                100,
                "operator",
                AuditAction::JobSubmitted,
                sha256(b"job"),
                None,
            )
            .unwrap();
        let snapshot = TrustSnapshot::new(
            1,
            50,
            200,
            vec![KeyTrustRecord {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "audit-root".into(),
                not_before_unix_s: 1,
                not_after_unix_s: None,
                status: KeyLifecycleStatus::Active,
                usages: BTreeSet::from([KeyUsage::AuditAnchor]),
            }],
        )
        .unwrap();
        (journal, snapshot)
    }

    #[test]
    fn signed_anchor_binds_exact_journal_head_and_snapshot() {
        let (journal, snapshot) = evidence();
        let signed =
            sign_audit_anchor(&journal, &snapshot, 100, "daily-anchor-1", &Provider).unwrap();
        let verified =
            verify_signed_audit_anchor(&signed, &journal, &snapshot, 100, &Provider).unwrap();
        assert_eq!(verified.anchor().journal_head, journal.head().unwrap());
    }

    #[test]
    fn appended_or_tampered_journal_does_not_match_old_anchor() {
        let (mut journal, snapshot) = evidence();
        let signed =
            sign_audit_anchor(&journal, &snapshot, 100, "daily-anchor-1", &Provider).unwrap();
        journal
            .append(
                101,
                "operator",
                AuditAction::Other("later".into()),
                sha256(b"later"),
                None,
            )
            .unwrap();
        assert!(verify_signed_audit_anchor(&signed, &journal, &snapshot, 101, &Provider).is_err());
    }
}
