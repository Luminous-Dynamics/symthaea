// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Decision-scoped evidence qualification gate for MACE retrieval candidates.
//!
//! Retrieval-policy memory may propose search procedures, but it cannot issue
//! this receipt or authorize evidence. A separately configured verifier must
//! validate the external verifier signature and exact request binding.
//!
//! This is a boundary contract, not a production signature algorithm, issuer
//! registry, rights engine or source-authentication service. Production must
//! inject a verifier backed by separately governed trust roots and re-check
//! live authorization at the consequential use.

use sha2::{Digest, Sha256};

pub const MAX_QUALIFICATION_ENTRIES_V1: usize = 10_000;
pub const MAX_QUALIFICATION_TTL_SECONDS_V1: u64 = 15 * 60;
pub const MAX_SIGNATURE_BYTES_V1: usize = 4_096;
pub const MAX_IDENTIFIER_BYTES_V1: usize = 256;

/// Admissibility under one exact policy, purpose and decision need. This is
/// NOT a claim that the underlying statement is objectively true.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum EvidenceDispositionV1 {
    Admissible,
    Rejected,
    Indeterminate,
}

impl EvidenceDispositionV1 {
    fn tag(self) -> u8 {
        match self {
            Self::Admissible => 1,
            Self::Rejected => 2,
            Self::Indeterminate => 3,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceQualificationEntryV1 {
    pub evidence_id: String,
    pub disposition: EvidenceDispositionV1,
    /// Digest of the independently inspected basis bundle, not a source signature.
    pub basis_receipt_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceQualificationReceiptV1 {
    pub run_id: String,
    pub query_invocation_id: String,
    pub domain_id: String,
    pub environment_revision_sha256: String,
    pub tool_profile_sha256: String,
    pub query_sha256: String,
    pub candidate_set_sha256: String,
    pub decision_need_sha256: String,
    pub purpose_sha256: String,
    pub qualification_policy_sha256: String,
    pub lifecycle_snapshot_sha256: String,
    pub verifier_profile_sha256: String,
    pub key_id: String,
    pub issued_at_unix: u64,
    pub expires_at_unix: u64,
    /// Must be sorted by evidence_id and cover every candidate exactly once.
    pub entries: Vec<EvidenceQualificationEntryV1>,
    /// Opaque signature bytes validated by the configured external verifier.
    pub signature: Vec<u8>,
}

/// Expected request context supplied by the consumer, never copied from the
/// receipt. Candidate IDs must be in canonical ascending order.
pub struct EvidenceQualificationBindingV1<'a> {
    pub run_id: &'a str,
    pub query_invocation_id: &'a str,
    pub domain_id: &'a str,
    pub environment_revision_sha256: &'a str,
    pub tool_profile_sha256: &'a str,
    pub query_sha256: &'a str,
    pub candidate_set_sha256: &'a str,
    pub candidate_evidence_ids: &'a [String],
    pub decision_need_sha256: &'a str,
    pub purpose_sha256: &'a str,
    pub qualification_policy_sha256: &'a str,
    pub lifecycle_snapshot_sha256: &'a str,
    pub verifier_profile_sha256: &'a str,
    pub now_unix: u64,
}

/// Cryptographic verification and trust-root policy live outside this crate.
/// Implementations must reject unknown/revoked keys and verify the signature
/// over the supplied domain-separated payload digest.
pub trait EvidenceQualificationSignatureVerifierV1 {
    fn verify_signature(
        &self,
        verifier_profile_sha256: &str,
        key_id: &str,
        payload_sha256: &str,
        signature: &[u8],
    ) -> Result<(), EvidenceQualificationRejectV1>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceQualificationRejectV1 {
    InvalidIdentifier,
    InvalidDigest,
    EmptyCandidateSet,
    TooManyCandidates,
    CandidateIdsNotCanonical,
    DuplicateCandidateId,
    BindingMismatch,
    ReceiptEntryCountMismatch,
    ReceiptEntriesNotCanonical,
    DuplicateReceiptEntry,
    InvalidReceiptTtl,
    ReceiptNotYetValid,
    ReceiptExpired,
    ReceiptTooOld,
    EmptySignature,
    SignatureTooLarge,
    SignatureRejected,
}

/// Capability token created only by verify_and_bind. Callers cannot construct
/// it directly from unverified receipt fields.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedEvidenceQualificationV1 {
    receipt_sha256: String,
    admissible_evidence_ids: Vec<String>,
    rejected_evidence_ids: Vec<String>,
    indeterminate_evidence_ids: Vec<String>,
    expires_at_unix: u64,
}

impl VerifiedEvidenceQualificationV1 {
    pub fn verify_and_bind(
        receipt: &EvidenceQualificationReceiptV1,
        binding: &EvidenceQualificationBindingV1<'_>,
        verifier: &impl EvidenceQualificationSignatureVerifierV1,
    ) -> Result<Self, EvidenceQualificationRejectV1> {
        validate_binding(binding)?;
        validate_receipt_shape(receipt)?;

        for (actual, expected) in [
            (receipt.run_id.as_str(), binding.run_id),
            (receipt.query_invocation_id.as_str(), binding.query_invocation_id),
            (receipt.domain_id.as_str(), binding.domain_id),
            (receipt.environment_revision_sha256.as_str(), binding.environment_revision_sha256),
            (receipt.tool_profile_sha256.as_str(), binding.tool_profile_sha256),
            (receipt.query_sha256.as_str(), binding.query_sha256),
            (receipt.candidate_set_sha256.as_str(), binding.candidate_set_sha256),
            (receipt.decision_need_sha256.as_str(), binding.decision_need_sha256),
            (receipt.purpose_sha256.as_str(), binding.purpose_sha256),
            (receipt.qualification_policy_sha256.as_str(), binding.qualification_policy_sha256),
            (receipt.lifecycle_snapshot_sha256.as_str(), binding.lifecycle_snapshot_sha256),
            (receipt.verifier_profile_sha256.as_str(), binding.verifier_profile_sha256),
        ] {
            if actual != expected {
                return Err(EvidenceQualificationRejectV1::BindingMismatch);
            }
        }

        if receipt.entries.len() != binding.candidate_evidence_ids.len() {
            return Err(EvidenceQualificationRejectV1::ReceiptEntryCountMismatch);
        }
        for (entry, candidate_id) in receipt.entries.iter().zip(binding.candidate_evidence_ids) {
            if entry.evidence_id.as_str() != candidate_id.as_str() {
                return Err(EvidenceQualificationRejectV1::BindingMismatch);
            }
        }

        if receipt.issued_at_unix > binding.now_unix {
            return Err(EvidenceQualificationRejectV1::ReceiptNotYetValid);
        }
        if binding.now_unix >= receipt.expires_at_unix {
            return Err(EvidenceQualificationRejectV1::ReceiptExpired);
        }
        if receipt.expires_at_unix <= receipt.issued_at_unix
            || receipt.expires_at_unix - receipt.issued_at_unix
                > MAX_QUALIFICATION_TTL_SECONDS_V1
        {
            return Err(EvidenceQualificationRejectV1::InvalidReceiptTtl);
        }
        if binding.now_unix - receipt.issued_at_unix > MAX_QUALIFICATION_TTL_SECONDS_V1 {
            return Err(EvidenceQualificationRejectV1::ReceiptTooOld);
        }

        let payload_sha256 = receipt.signing_payload_sha256();
        verifier
            .verify_signature(
                &receipt.verifier_profile_sha256,
                &receipt.key_id,
                &payload_sha256,
                &receipt.signature,
            )
            .map_err(|_| EvidenceQualificationRejectV1::SignatureRejected)?;

        let mut admissible = Vec::new();
        let mut rejected = Vec::new();
        let mut indeterminate = Vec::new();
        for entry in &receipt.entries {
            match entry.disposition {
                EvidenceDispositionV1::Admissible => admissible.push(entry.evidence_id.clone()),
                EvidenceDispositionV1::Rejected => rejected.push(entry.evidence_id.clone()),
                EvidenceDispositionV1::Indeterminate => indeterminate.push(entry.evidence_id.clone()),
            }
        }

        Ok(Self {
            receipt_sha256: receipt.receipt_sha256(),
            admissible_evidence_ids: admissible,
            rejected_evidence_ids: rejected,
            indeterminate_evidence_ids: indeterminate,
            expires_at_unix: receipt.expires_at_unix,
        })
    }

    /// Only IDs that passed the independent, decision-scoped admissibility
    /// check. The executor must still re-check live authority and lifecycle.
    pub fn admissible_evidence_ids(&self) -> &[String] { &self.admissible_evidence_ids }
    pub fn rejected_evidence_ids(&self) -> &[String] { &self.rejected_evidence_ids }
    pub fn indeterminate_evidence_ids(&self) -> &[String] { &self.indeterminate_evidence_ids }
    pub fn receipt_sha256(&self) -> &str { &self.receipt_sha256 }
    pub fn expires_at_unix(&self) -> u64 { self.expires_at_unix }
    pub fn is_empty(&self) -> bool { self.admissible_evidence_ids.is_empty() }
}

impl EvidenceQualificationReceiptV1 {
    /// Domain-separated payload digest. The signature itself is excluded;
    /// key identity and every qualification binding are included.
    pub fn signing_payload_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"luminous:mace:evidence-qualification:v1\0");
        for value in [
            self.run_id.as_str(),
            self.query_invocation_id.as_str(),
            self.domain_id.as_str(),
            self.environment_revision_sha256.as_str(),
            self.tool_profile_sha256.as_str(),
            self.query_sha256.as_str(),
            self.candidate_set_sha256.as_str(),
            self.decision_need_sha256.as_str(),
            self.purpose_sha256.as_str(),
            self.qualification_policy_sha256.as_str(),
            self.lifecycle_snapshot_sha256.as_str(),
            self.verifier_profile_sha256.as_str(),
            self.key_id.as_str(),
        ] {
            hash_field(&mut hasher, value);
        }
        hasher.update(self.issued_at_unix.to_be_bytes());
        hasher.update(self.expires_at_unix.to_be_bytes());
        hasher.update((self.entries.len() as u64).to_be_bytes());
        for entry in &self.entries {
            hash_field(&mut hasher, &entry.evidence_id);
            hasher.update([entry.disposition.tag()]);
            hash_field(&mut hasher, &entry.basis_receipt_sha256);
        }
        hex_digest(&hasher.finalize())
    }

    /// Full commitment includes the signature bytes as well as its payload.
    pub fn receipt_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"luminous:mace:evidence-qualification-receipt:v1\0");
        hash_field(&mut hasher, &self.signing_payload_sha256());
        hasher.update((self.signature.len() as u64).to_be_bytes());
        hasher.update(&self.signature);
        hex_digest(&hasher.finalize())
    }
}

fn validate_binding(binding: &EvidenceQualificationBindingV1<'_>)
    -> Result<(), EvidenceQualificationRejectV1>
{
    for value in [binding.run_id, binding.query_invocation_id, binding.domain_id] {
        if !valid_atom(value) || value.len() > MAX_IDENTIFIER_BYTES_V1 {
            return Err(EvidenceQualificationRejectV1::InvalidIdentifier);
        }
    }
    for digest in [
        binding.environment_revision_sha256, binding.tool_profile_sha256,
        binding.query_sha256, binding.candidate_set_sha256, binding.decision_need_sha256,
        binding.purpose_sha256, binding.qualification_policy_sha256,
        binding.lifecycle_snapshot_sha256, binding.verifier_profile_sha256,
    ] {
        if !valid_sha256(digest) { return Err(EvidenceQualificationRejectV1::InvalidDigest); }
    }
    if binding.candidate_evidence_ids.is_empty() {
        return Err(EvidenceQualificationRejectV1::EmptyCandidateSet);
    }
    if binding.candidate_evidence_ids.len() > MAX_QUALIFICATION_ENTRIES_V1 {
        return Err(EvidenceQualificationRejectV1::TooManyCandidates);
    }
    for (index, id) in binding.candidate_evidence_ids.iter().enumerate() {
        if !valid_atom(id) || id.len() > MAX_IDENTIFIER_BYTES_V1 {
            return Err(EvidenceQualificationRejectV1::InvalidIdentifier);
        }
        if index > 0 {
            match binding.candidate_evidence_ids[index - 1].cmp(id) {
                std::cmp::Ordering::Equal =>
                    return Err(EvidenceQualificationRejectV1::DuplicateCandidateId),
                std::cmp::Ordering::Greater =>
                    return Err(EvidenceQualificationRejectV1::CandidateIdsNotCanonical),
                std::cmp::Ordering::Less => {}
            }
        }
    }
    Ok(())
}

fn validate_receipt_shape(receipt: &EvidenceQualificationReceiptV1)
    -> Result<(), EvidenceQualificationRejectV1>
{
    for value in [receipt.run_id.as_str(), receipt.query_invocation_id.as_str(),
        receipt.domain_id.as_str(), receipt.key_id.as_str()] {
        if !valid_atom(value) || value.len() > MAX_IDENTIFIER_BYTES_V1 {
            return Err(EvidenceQualificationRejectV1::InvalidIdentifier);
        }
    }
    for digest in [
        receipt.environment_revision_sha256.as_str(), receipt.tool_profile_sha256.as_str(),
        receipt.query_sha256.as_str(), receipt.candidate_set_sha256.as_str(),
        receipt.decision_need_sha256.as_str(), receipt.purpose_sha256.as_str(),
        receipt.qualification_policy_sha256.as_str(), receipt.lifecycle_snapshot_sha256.as_str(),
        receipt.verifier_profile_sha256.as_str(),
    ] {
        if !valid_sha256(digest) { return Err(EvidenceQualificationRejectV1::InvalidDigest); }
    }
    if receipt.entries.is_empty() { return Err(EvidenceQualificationRejectV1::EmptyCandidateSet); }
    if receipt.entries.len() > MAX_QUALIFICATION_ENTRIES_V1 {
        return Err(EvidenceQualificationRejectV1::TooManyCandidates);
    }
    if receipt.signature.is_empty() { return Err(EvidenceQualificationRejectV1::EmptySignature); }
    if receipt.signature.len() > MAX_SIGNATURE_BYTES_V1 {
        return Err(EvidenceQualificationRejectV1::SignatureTooLarge);
    }
    for (index, entry) in receipt.entries.iter().enumerate() {
        if !valid_atom(&entry.evidence_id) || entry.evidence_id.len() > MAX_IDENTIFIER_BYTES_V1 {
            return Err(EvidenceQualificationRejectV1::InvalidIdentifier);
        }
        if !valid_sha256(&entry.basis_receipt_sha256) {
            return Err(EvidenceQualificationRejectV1::InvalidDigest);
        }
        if index > 0 {
            match receipt.entries[index - 1].evidence_id.cmp(&entry.evidence_id) {
                std::cmp::Ordering::Equal =>
                    return Err(EvidenceQualificationRejectV1::DuplicateReceiptEntry),
                std::cmp::Ordering::Greater =>
                    return Err(EvidenceQualificationRejectV1::ReceiptEntriesNotCanonical),
                std::cmp::Ordering::Less => {}
            }
        }
    }
    Ok(())
}

fn valid_atom(value: &str) -> bool {
    !value.trim().is_empty() && value == value.trim() && !value.chars().any(char::is_control)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestSignatureVerifier;

    impl EvidenceQualificationSignatureVerifierV1 for TestSignatureVerifier {
        fn verify_signature(
            &self, profile: &str, key_id: &str, payload: &str, signature: &[u8],
        ) -> Result<(), EvidenceQualificationRejectV1> {
            if valid_sha256(profile) && key_id == "test-key" && signature == payload.as_bytes() {
                Ok(())
            } else {
                Err(EvidenceQualificationRejectV1::SignatureRejected)
            }
        }
    }

    fn digest(byte: char) -> String { std::iter::repeat_n(byte, 64).collect() }

    fn entry(id: &str, disposition: EvidenceDispositionV1) -> EvidenceQualificationEntryV1 {
        EvidenceQualificationEntryV1 {
            evidence_id: id.to_owned(),
            disposition,
            basis_receipt_sha256: digest('a'),
        }
    }

    fn receipt() -> EvidenceQualificationReceiptV1 {
        EvidenceQualificationReceiptV1 {
            run_id: "run-1".to_owned(),
            query_invocation_id: "invocation-1".to_owned(),
            domain_id: "web".to_owned(),
            environment_revision_sha256: digest('b'),
            tool_profile_sha256: digest('c'),
            query_sha256: digest('d'),
            candidate_set_sha256: digest('e'),
            decision_need_sha256: digest('f'),
            purpose_sha256: digest('1'),
            qualification_policy_sha256: digest('2'),
            lifecycle_snapshot_sha256: digest('3'),
            verifier_profile_sha256: digest('4'),
            key_id: "test-key".to_owned(),
            issued_at_unix: 100,
            expires_at_unix: 200,
            entries: vec![
                entry("evidence-a", EvidenceDispositionV1::Admissible),
                entry("evidence-b", EvidenceDispositionV1::Rejected),
                entry("evidence-c", EvidenceDispositionV1::Indeterminate),
            ],
            signature: Vec::new(),
        }
    }

    fn signed_receipt() -> EvidenceQualificationReceiptV1 {
        let mut value = receipt();
        value.signature = value.signing_payload_sha256().into_bytes();
        value
    }

    fn candidate_ids() -> Vec<String> {
        vec!["evidence-a".to_owned(), "evidence-b".to_owned(), "evidence-c".to_owned()]
    }

    fn binding<'a>(
        receipt: &'a EvidenceQualificationReceiptV1, ids: &'a [String], now: u64,
    ) -> EvidenceQualificationBindingV1<'a> {
        EvidenceQualificationBindingV1 {
            run_id: &receipt.run_id,
            query_invocation_id: &receipt.query_invocation_id,
            domain_id: &receipt.domain_id,
            environment_revision_sha256: &receipt.environment_revision_sha256,
            tool_profile_sha256: &receipt.tool_profile_sha256,
            query_sha256: &receipt.query_sha256,
            candidate_set_sha256: &receipt.candidate_set_sha256,
            candidate_evidence_ids: ids,
            decision_need_sha256: &receipt.decision_need_sha256,
            purpose_sha256: &receipt.purpose_sha256,
            qualification_policy_sha256: &receipt.qualification_policy_sha256,
            lifecycle_snapshot_sha256: &receipt.lifecycle_snapshot_sha256,
            verifier_profile_sha256: &receipt.verifier_profile_sha256,
            now_unix: now,
        }
    }

    #[test]
    fn only_independently_admissible_ids_cross_the_gate() {
        let receipt = signed_receipt();
        let ids = candidate_ids();
        let binding = binding(&receipt, &ids, 150);
        let verified = VerifiedEvidenceQualificationV1::verify_and_bind(
            &receipt, &binding, &TestSignatureVerifier,
        ).expect("valid scoped receipt");
        assert_eq!(verified.admissible_evidence_ids(), &["evidence-a".to_owned()]);
        assert_eq!(verified.rejected_evidence_ids(), &["evidence-b".to_owned()]);
        assert_eq!(verified.indeterminate_evidence_ids(), &["evidence-c".to_owned()]);
        assert_eq!(verified.expires_at_unix(), 200);
    }

    #[test]
    fn decision_need_and_purpose_are_part_of_the_binding() {
        let receipt = signed_receipt();
        let ids = candidate_ids();
        let changed_need = digest('5');
        let mut binding = binding(&receipt, &ids, 150);
        binding.decision_need_sha256 = &changed_need;
        assert_eq!(
            VerifiedEvidenceQualificationV1::verify_and_bind(&receipt, &binding, &TestSignatureVerifier),
            Err(EvidenceQualificationRejectV1::BindingMismatch)
        );
    }

    #[test]
    fn expired_receipt_is_rejected_at_consumption_time() {
        let receipt = signed_receipt();
        let ids = candidate_ids();
        let binding = binding(&receipt, &ids, 200);
        assert_eq!(
            VerifiedEvidenceQualificationV1::verify_and_bind(&receipt, &binding, &TestSignatureVerifier),
            Err(EvidenceQualificationRejectV1::ReceiptExpired)
        );
    }

    #[test]
    fn missing_candidate_disposition_fails_closed() {
        let receipt = signed_receipt();
        let ids = vec!["evidence-a".to_owned(), "evidence-b".to_owned()];
        let binding = binding(&receipt, &ids, 150);
        assert_eq!(
            VerifiedEvidenceQualificationV1::verify_and_bind(&receipt, &binding, &TestSignatureVerifier),
            Err(EvidenceQualificationRejectV1::ReceiptEntryCountMismatch)
        );
    }

    #[test]
    fn duplicate_candidate_ids_are_rejected() {
        let receipt = signed_receipt();
        let ids = vec!["evidence-a".to_owned(), "evidence-a".to_owned(), "evidence-c".to_owned()];
        let binding = binding(&receipt, &ids, 150);
        assert_eq!(
            VerifiedEvidenceQualificationV1::verify_and_bind(&receipt, &binding, &TestSignatureVerifier),
            Err(EvidenceQualificationRejectV1::DuplicateCandidateId)
        );
    }

    #[test]
    fn altered_signature_fails_closed() {
        let mut receipt = signed_receipt();
        receipt.signature[0] ^= 1;
        let ids = candidate_ids();
        let binding = binding(&receipt, &ids, 150);
        assert_eq!(
            VerifiedEvidenceQualificationV1::verify_and_bind(&receipt, &binding, &TestSignatureVerifier),
            Err(EvidenceQualificationRejectV1::SignatureRejected)
        );
    }

    #[test]
    fn environment_revision_cannot_be_rebound() {
        let receipt = signed_receipt();
        let ids = candidate_ids();
        let changed_environment = digest('5');
        let mut binding = binding(&receipt, &ids, 150);
        binding.environment_revision_sha256 = &changed_environment;
        assert_eq!(
            VerifiedEvidenceQualificationV1::verify_and_bind(&receipt, &binding, &TestSignatureVerifier),
            Err(EvidenceQualificationRejectV1::BindingMismatch)
        );
    }

    #[test]
    fn zero_admissible_candidates_is_an_abstention_signal_not_permission() {
        let mut receipt = receipt();
        for item in &mut receipt.entries { item.disposition = EvidenceDispositionV1::Rejected; }
        receipt.signature = receipt.signing_payload_sha256().into_bytes();
        let ids = candidate_ids();
        let binding = binding(&receipt, &ids, 150);
        let verified = VerifiedEvidenceQualificationV1::verify_and_bind(
            &receipt, &binding, &TestSignatureVerifier,
        ).expect("valid receipt can qualify zero candidates");
        assert!(verified.is_empty());
        assert!(verified.admissible_evidence_ids().is_empty());
    }
}
