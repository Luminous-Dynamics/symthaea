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

use serde_json::{json, Value};
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
    /// Digest of the effective tenant/consumer authorization scope.
    pub subject_scope_sha256: String,
    pub environment_revision_sha256: String,
    pub tool_profile_sha256: String,
    pub query_sha256: String,
    pub candidate_set_sha256: String,
    /// Independently recomputed digest of the exact sorted evidence-ID set.
    pub candidate_evidence_ids_sha256: String,
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
    pub subject_scope_sha256: &'a str,
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
        issued_at_unix: u64,
        now_unix: u64,
    ) -> Result<(), EvidenceQualificationRejectV1>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceQualificationRejectV1 {
    InvalidJson,
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
    TrustProfileInvalid,
    TrustProfileDigestMismatch,
    UnknownKey,
    RevokedKey,
    KeyNotYetValid,
    KeyExpired,
    InvalidPublicKey,
    InvalidSignatureEncoding,
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
            (receipt.subject_scope_sha256.as_str(), binding.subject_scope_sha256),
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
        if receipt.candidate_evidence_ids_sha256 != candidate_ids_sha256(binding.candidate_evidence_ids) {
            return Err(EvidenceQualificationRejectV1::BindingMismatch);
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
                receipt.issued_at_unix,
                binding.now_unix,
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
    pub fn admissible_evidence_ids_at(
        &self,
        now_unix: u64,
    ) -> Result<&[String], EvidenceQualificationRejectV1> {
        if now_unix >= self.expires_at_unix {
            return Err(EvidenceQualificationRejectV1::ReceiptExpired);
        }
        Ok(&self.admissible_evidence_ids)
    }
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
            self.subject_scope_sha256.as_str(),
            self.environment_revision_sha256.as_str(),
            self.tool_profile_sha256.as_str(),
            self.query_sha256.as_str(),
            self.candidate_set_sha256.as_str(),
            self.candidate_evidence_ids_sha256.as_str(),
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


    /// Strictly decode the versioned JSON wire representation. This validates
    /// structure and digests but does not establish signer trust; callers must
    /// still invoke verify_and_bind with an external signature verifier.
    pub fn from_json_value(value: &Value) -> Result<Self, EvidenceQualificationRejectV1> {
        let object = value.as_object().ok_or(EvidenceQualificationRejectV1::InvalidJson)?;
        if object.len() != 20
            || object.get("schema_version").and_then(Value::as_str)
                != Some("mace_evidence_qualification_receipt_v1")
        {
            return Err(EvidenceQualificationRejectV1::InvalidJson);
        }

        let required_string = |field: &str| -> Result<String, EvidenceQualificationRejectV1> {
            object.get(field).and_then(Value::as_str).map(str::to_owned)
                .ok_or(EvidenceQualificationRejectV1::InvalidJson)
        };
        let required_u64 = |field: &str| -> Result<u64, EvidenceQualificationRejectV1> {
            object.get(field).and_then(Value::as_u64)
                .ok_or(EvidenceQualificationRejectV1::InvalidJson)
        };

        let entry_values = object.get("entries").and_then(Value::as_array)
            .ok_or(EvidenceQualificationRejectV1::InvalidJson)?;
        let mut entries = Vec::with_capacity(entry_values.len());
        for value in entry_values {
            let item = value.as_object().ok_or(EvidenceQualificationRejectV1::InvalidJson)?;
            if item.len() != 3 {
                return Err(EvidenceQualificationRejectV1::InvalidJson);
            }
            let evidence_id = item.get("evidence_id").and_then(Value::as_str)
                .ok_or(EvidenceQualificationRejectV1::InvalidJson)?.to_owned();
            let basis_receipt_sha256 = item.get("basis_receipt_sha256").and_then(Value::as_str)
                .ok_or(EvidenceQualificationRejectV1::InvalidJson)?.to_owned();
            let disposition = match item.get("disposition").and_then(Value::as_str) {
                Some("admissible") => EvidenceDispositionV1::Admissible,
                Some("rejected") => EvidenceDispositionV1::Rejected,
                Some("indeterminate") => EvidenceDispositionV1::Indeterminate,
                _ => return Err(EvidenceQualificationRejectV1::InvalidJson),
            };
            entries.push(EvidenceQualificationEntryV1 {
                evidence_id,
                disposition,
                basis_receipt_sha256,
            });
        }

        let signature_values = object.get("signature").and_then(Value::as_array)
            .ok_or(EvidenceQualificationRejectV1::InvalidJson)?;
        if signature_values.len() > MAX_SIGNATURE_BYTES_V1 {
            return Err(EvidenceQualificationRejectV1::SignatureTooLarge);
        }
        let mut signature = Vec::with_capacity(signature_values.len());
        for value in signature_values {
            let byte = value.as_u64().ok_or(EvidenceQualificationRejectV1::InvalidJson)?;
            signature.push(u8::try_from(byte).map_err(|_| EvidenceQualificationRejectV1::InvalidJson)?);
        }

        let receipt = Self {
            run_id: required_string("run_id")?,
            query_invocation_id: required_string("query_invocation_id")?,
            domain_id: required_string("domain_id")?,
            subject_scope_sha256: required_string("subject_scope_sha256")?,
            environment_revision_sha256: required_string("environment_revision_sha256")?,
            tool_profile_sha256: required_string("tool_profile_sha256")?,
            query_sha256: required_string("query_sha256")?,
            candidate_set_sha256: required_string("candidate_set_sha256")?,
            candidate_evidence_ids_sha256: required_string("candidate_evidence_ids_sha256")?,
            decision_need_sha256: required_string("decision_need_sha256")?,
            purpose_sha256: required_string("purpose_sha256")?,
            qualification_policy_sha256: required_string("qualification_policy_sha256")?,
            lifecycle_snapshot_sha256: required_string("lifecycle_snapshot_sha256")?,
            verifier_profile_sha256: required_string("verifier_profile_sha256")?,
            key_id: required_string("key_id")?,
            issued_at_unix: required_u64("issued_at_unix")?,
            expires_at_unix: required_u64("expires_at_unix")?,
            entries,
            signature,
        };
        validate_receipt_shape(&receipt)?;
        Ok(receipt)
    }

    /// Deterministic typed projection for the versioned JSON wire schema.
    /// Signature bytes are represented as an array of unsigned octets.
    pub fn to_json_value(&self) -> Value {
        json!({
            "schema_version": "mace_evidence_qualification_receipt_v1",
            "run_id": self.run_id,
            "query_invocation_id": self.query_invocation_id,
            "domain_id": self.domain_id,
            "subject_scope_sha256": self.subject_scope_sha256,
            "environment_revision_sha256": self.environment_revision_sha256,
            "tool_profile_sha256": self.tool_profile_sha256,
            "query_sha256": self.query_sha256,
            "candidate_set_sha256": self.candidate_set_sha256,
            "candidate_evidence_ids_sha256": self.candidate_evidence_ids_sha256,
            "decision_need_sha256": self.decision_need_sha256,
            "purpose_sha256": self.purpose_sha256,
            "qualification_policy_sha256": self.qualification_policy_sha256,
            "lifecycle_snapshot_sha256": self.lifecycle_snapshot_sha256,
            "verifier_profile_sha256": self.verifier_profile_sha256,
            "key_id": self.key_id,
            "issued_at_unix": self.issued_at_unix,
            "expires_at_unix": self.expires_at_unix,
            "entries": self.entries.iter().map(|entry| {
                let disposition = match entry.disposition {
                    EvidenceDispositionV1::Admissible => "admissible",
                    EvidenceDispositionV1::Rejected => "rejected",
                    EvidenceDispositionV1::Indeterminate => "indeterminate",
                };
                json!({
                    "evidence_id": entry.evidence_id,
                    "disposition": disposition,
                    "basis_receipt_sha256": entry.basis_receipt_sha256,
                })
            }).collect::<Vec<_>>(),
            "signature": self.signature,
        })
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
        binding.subject_scope_sha256, binding.environment_revision_sha256, binding.tool_profile_sha256,
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
        receipt.subject_scope_sha256.as_str(), receipt.environment_revision_sha256.as_str(), receipt.tool_profile_sha256.as_str(),
        receipt.query_sha256.as_str(), receipt.candidate_set_sha256.as_str(),
        receipt.candidate_evidence_ids_sha256.as_str(), receipt.decision_need_sha256.as_str(), receipt.purpose_sha256.as_str(),
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

fn candidate_ids_sha256(ids: &[String]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:candidate-evidence-ids:v1\0");
    hasher.update((ids.len() as u64).to_be_bytes());
    for id in ids { hash_field(&mut hasher, id); }
    hex_digest(&hasher.finalize())
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
    use crate::retrieval_policy_memory::{
        RetrievalOperationOutcomeV1,
        RetrievalOperationV1,
        RetrievalPolicyMemoryV1,
        RetrievalStopReasonV1,
        RetrievalTraceReceiptV1,
        RetrievalTraceRefV1,
    };

    struct TestSignatureVerifier;

    impl EvidenceQualificationSignatureVerifierV1 for TestSignatureVerifier {
        fn verify_signature(
            &self, profile: &str, key_id: &str, payload: &str, signature: &[u8],
            _issued_at_unix: u64, _now_unix: u64,
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
            subject_scope_sha256: digest('7'),
            environment_revision_sha256: digest('b'),
            tool_profile_sha256: digest('c'),
            query_sha256: digest('d'),
            candidate_set_sha256: digest('e'),
            candidate_evidence_ids_sha256: candidate_ids_sha256(&vec!["evidence-a".to_owned(), "evidence-b".to_owned(), "evidence-c".to_owned()]),
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

    fn successful_candidate_trace(
        invocation: &str,
        query_digest_byte: char,
        created_at_unix: u64,
    ) -> RetrievalTraceReceiptV1 {
        RetrievalTraceReceiptV1 {
            run_id: "run-1".to_owned(),
            query_invocation_id: invocation.to_owned(),
            domain_id: "web".to_owned(),
            environment_revision_sha256: digest('b'),
            tool_profile_sha256: digest('c'),
            query_sha256: digest(query_digest_byte),
            candidate_set_sha256: Some(digest('e')),
            operations: vec![RetrievalOperationV1 {
                tool_id: "trajectory.bm25.search".to_owned(),
                tool_revision_sha256: digest('c'),
                action_template_sha256: digest('d'),
                observed_evidence_ids: vec!["poisoned-candidate".to_owned()],
                outcome: RetrievalOperationOutcomeV1::EvidenceObserved,
                elapsed_ns: 7,
                inspected_bytes: 0,
                inspected_bytes_measured: false,
            }],
            stop_reason: RetrievalStopReasonV1::EvidenceSufficient,
            tool_call_budget: 1,
            tool_calls_used: 1,
            elapsed_ns: 7,
            inspected_bytes: 0,
            inspected_bytes_complete: false,
            created_at_unix,
        }
    }

    fn binding<'a>(
        receipt: &'a EvidenceQualificationReceiptV1, ids: &'a [String], now: u64,
    ) -> EvidenceQualificationBindingV1<'a> {
        EvidenceQualificationBindingV1 {
            run_id: &receipt.run_id,
            query_invocation_id: &receipt.query_invocation_id,
            domain_id: &receipt.domain_id,
            subject_scope_sha256: &receipt.subject_scope_sha256,
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
        assert_eq!(verified.admissible_evidence_ids_at(150).expect("still valid"), &["evidence-a".to_owned()]);
        assert_eq!(verified.rejected_evidence_ids(), &["evidence-b".to_owned()]);
        assert_eq!(verified.indeterminate_evidence_ids(), &["evidence-c".to_owned()]);
        assert_eq!(verified.expires_at_unix(), 200);
    }

    #[test]
    fn decision_need_and_purpose_are_part_of_the_binding() {
        let receipt = signed_receipt();
        let ids = candidate_ids();
        let changed_need = digest('6');
        let mut binding = binding(&receipt, &ids, 150);
        binding.decision_need_sha256 = &changed_need;
        assert_eq!(
            VerifiedEvidenceQualificationV1::verify_and_bind(&receipt, &binding, &TestSignatureVerifier),
            Err(EvidenceQualificationRejectV1::BindingMismatch)
        );
    }

    #[test]
    fn purpose_is_part_of_the_binding() {
        let receipt = signed_receipt();
        let ids = candidate_ids();
        let changed_purpose = digest('5');
        let mut binding = binding(&receipt, &ids, 150);
        binding.purpose_sha256 = &changed_purpose;
        assert_eq!(
            VerifiedEvidenceQualificationV1::verify_and_bind(&receipt, &binding, &TestSignatureVerifier),
            Err(EvidenceQualificationRejectV1::BindingMismatch)
        );
    }

    #[test]
    fn candidate_id_set_commitment_must_match_exact_sorted_ids() {
        let mut receipt = signed_receipt();
        receipt.candidate_evidence_ids_sha256 = digest('5');
        receipt.signature = receipt.signing_payload_sha256().into_bytes();
        let ids = candidate_ids();
        let binding = binding(&receipt, &ids, 150);
        assert_eq!(
            VerifiedEvidenceQualificationV1::verify_and_bind(&receipt, &binding, &TestSignatureVerifier),
            Err(EvidenceQualificationRejectV1::BindingMismatch)
        );
    }

    #[test]
    fn subject_scope_cannot_be_rebound_across_tenants_or_consumers() {
        let receipt = signed_receipt();
        let ids = candidate_ids();
        let changed_scope = digest('8');
        let mut binding = binding(&receipt, &ids, 150);
        binding.subject_scope_sha256 = &changed_scope;
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
    fn receipt_json_round_trips_and_rejects_unknown_fields() {
        let receipt = signed_receipt();
        let value = receipt.to_json_value();
        let decoded = EvidenceQualificationReceiptV1::from_json_value(&value)
            .expect("strict JSON round trip");
        assert_eq!(decoded, receipt);
        assert_eq!(decoded.receipt_sha256(), receipt.receipt_sha256());

        let mut extended = value;
        extended.as_object_mut().expect("object").insert(
            "unrecognized".to_owned(),
            Value::Bool(true),
        );
        assert_eq!(
            EvidenceQualificationReceiptV1::from_json_value(&extended),
            Err(EvidenceQualificationRejectV1::InvalidJson)
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
    fn verified_token_cannot_be_used_after_its_expiry() {
        let receipt = signed_receipt();
        let ids = candidate_ids();
        let binding = binding(&receipt, &ids, 150);
        let verified = VerifiedEvidenceQualificationV1::verify_and_bind(
            &receipt, &binding, &TestSignatureVerifier,
        ).expect("valid at issuance");
        assert_eq!(
            verified.admissible_evidence_ids_at(200),
            Err(EvidenceQualificationRejectV1::ReceiptExpired)
        );
    }

    #[test]
    fn promoted_search_strategy_never_upgrades_rejected_candidate_evidence() {
        let mut policy_memory = RetrievalPolicyMemoryV1::default();
        let traces = [
            successful_candidate_trace("invocation-1", '8', 100),
            successful_candidate_trace("invocation-2", '9', 101),
            successful_candidate_trace("invocation-3", 'a', 102),
        ];
        let mut references = Vec::new();
        for trace in traces {
            policy_memory.record_trace(trace.clone()).expect("record candidate-producing trace");
            references.push(RetrievalTraceRefV1 {
                run_id: trace.run_id,
                query_invocation_id: trace.query_invocation_id,
            });
        }

        let note = policy_memory.promote_strategy(
            &references,
            &digest('5'),
            120,
            180,
        ).expect("three distinct invocations may promote an advisory strategy");
        assert_eq!(
            policy_memory.applicable_notes("web", &digest('b'), &digest('c'), 130).len(),
            1
        );

        let mut qualification = receipt();
        qualification.query_invocation_id = "invocation-1".to_owned();
        qualification.query_sha256 = digest('8');
        qualification.candidate_set_sha256 = digest('e');
        qualification.entries = vec![entry("poisoned-candidate", EvidenceDispositionV1::Rejected)];
        qualification.candidate_evidence_ids_sha256 =
            candidate_ids_sha256(&vec!["poisoned-candidate".to_owned()]);
        qualification.signature = qualification.signing_payload_sha256().into_bytes();

        let candidate_ids = vec!["poisoned-candidate".to_owned()];
        let binding = EvidenceQualificationBindingV1 {
            run_id: &qualification.run_id,
            query_invocation_id: &qualification.query_invocation_id,
            domain_id: &qualification.domain_id,
            subject_scope_sha256: &qualification.subject_scope_sha256,
            environment_revision_sha256: &qualification.environment_revision_sha256,
            tool_profile_sha256: &qualification.tool_profile_sha256,
            query_sha256: &qualification.query_sha256,
            candidate_set_sha256: &qualification.candidate_set_sha256,
            candidate_evidence_ids: &candidate_ids,
            decision_need_sha256: &qualification.decision_need_sha256,
            purpose_sha256: &qualification.purpose_sha256,
            qualification_policy_sha256: &qualification.qualification_policy_sha256,
            lifecycle_snapshot_sha256: &qualification.lifecycle_snapshot_sha256,
            verifier_profile_sha256: &qualification.verifier_profile_sha256,
            now_unix: 150,
        };
        let verified = VerifiedEvidenceQualificationV1::verify_and_bind(
            &qualification,
            &binding,
            &TestSignatureVerifier,
        ).expect("external qualification result is validly bound");
        assert!(verified.is_empty());
        assert!(verified.admissible_evidence_ids_at(150).expect("fresh token").is_empty());
        assert!(policy_memory.applicable_notes("web", &digest('b'), &digest('c'), 130)
            .iter().any(|candidate| candidate.note_sha256 == note.note_sha256));
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
        assert!(verified.admissible_evidence_ids_at(150).expect("still valid").is_empty());
    }
}
