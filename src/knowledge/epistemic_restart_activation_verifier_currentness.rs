// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! EKM-082 — independent current-head attestation for the EKM-081 activation
//! verifier trust state.
//!
//! EKM-081 proves local policy acceptance and monotonic continuity relative to a
//! caller-held verifier checkpoint. That still does not prove that the candidate
//! trust snapshot is the current/latest head. EKM-082 asks that question through
//! a separate provider contract whose evidence semantics explicitly mean current
//! monotonic head state.
//!
//! A generic detached signature is deliberately not an accepted evidence kind.
//! Currentness is time-bounded audit evidence only; no epoch issuance, mutation,
//! activation, or trusted-state mutation authority is granted.

use super::{
    ActivationVerifierTrustReviewDigestV1, ActivationVerifierTrustReviewReceiptV1,
    CanonicalVerifierTrustProfileDigestV1,
};
use std::error::Error;
use std::fmt;

pub const MAX_ACTIVATION_VERIFIER_CURRENTNESS_AUTHORITY_ID_BYTES: usize = 256;
pub const MAX_ACTIVATION_VERIFIER_CURRENTNESS_PROOF_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationVerifierCurrentnessVersion {
    V1,
}

/// Evidence classes whose provider contract explicitly means current monotonic
/// head state. A generic signature is intentionally absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationVerifierCurrentnessEvidenceKindV1 {
    MonotonicProtectedState,
    HardwareMonotonicCounter,
    TransparencyLogHead,
    CurrentHeadWitnessQuorum,
}

impl ActivationVerifierCurrentnessEvidenceKindV1 {
    fn tag(self) -> u8 {
        match self {
            Self::MonotonicProtectedState => 1,
            Self::HardwareMonotonicCounter => 2,
            Self::TransparencyLogHead => 3,
            Self::CurrentHeadWitnessQuorum => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActivationVerifierCurrentnessStatementDigestV1([u8; 32]);

impl ActivationVerifierCurrentnessStatementDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        hex32(self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VerifiedActivationVerifierCurrentnessDigestV1([u8; 32]);

impl VerifiedActivationVerifierCurrentnessDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        hex32(self.0)
    }
}

/// Exact current-head question presented to the deployment trust provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationVerifierCurrentnessStatementV1 {
    version: ActivationVerifierCurrentnessVersion,
    trust_review_digest: ActivationVerifierTrustReviewDigestV1,
    activation_evidence_receipt_digest: [u8; 32],
    issuance_record_digest: [u8; 32],
    verifier_profile_digest: CanonicalVerifierTrustProfileDigestV1,
    trust_policy_digest: [u8; 32],
    reference_checkpoint_digest: [u8; 32],
    verifier_id: String,
    implementation_id: String,
    implementation_version: String,
    trust_snapshot_digest: [u8; 32],
    trust_snapshot_sequence: u64,
    configuration_digest: [u8; 32],
    profile_valid_until_cycle: u64,
    trust_reviewed_at_cycle: u64,
    attested_at_cycle: u64,
    expires_at_cycle: u64,
    authority_id: String,
    evidence_kind: ActivationVerifierCurrentnessEvidenceKindV1,
}

impl ActivationVerifierCurrentnessStatementV1 {
    pub fn new(
        review: &ActivationVerifierTrustReviewReceiptV1,
        attested_at_cycle: u64,
        expires_at_cycle: u64,
        authority_id: impl Into<String>,
        evidence_kind: ActivationVerifierCurrentnessEvidenceKindV1,
    ) -> Result<Self, ActivationVerifierCurrentnessError> {
        if !review.local_policy_accepted()
            || !review.continuity_relative_to_caller_checkpoint_proven()
            || !review.profile_fresh_at_review_cycle()
            || review.provider_trust_independently_established()
            || review.global_current_head_independently_proven()
            || review.epoch_issuance_chain_verified()
            || review.epoch_issuance_authorized()
            || review.mutation_authority()
            || review.activation_authorized()
        {
            return Err(ActivationVerifierCurrentnessError::UnexpectedReviewAuthority);
        }
        if attested_at_cycle < review.reviewed_at_cycle() {
            return Err(ActivationVerifierCurrentnessError::AttestationPredatesTrustReview {
                attested_at_cycle,
                trust_reviewed_at_cycle: review.reviewed_at_cycle(),
            });
        }
        let profile = review.candidate_profile();
        if attested_at_cycle >= profile.valid_until_cycle() {
            return Err(ActivationVerifierCurrentnessError::ProfileExpiredBeforeAttestation {
                attested_at_cycle,
                profile_valid_until_cycle: profile.valid_until_cycle(),
            });
        }
        if expires_at_cycle <= attested_at_cycle || expires_at_cycle > profile.valid_until_cycle() {
            return Err(ActivationVerifierCurrentnessError::InvalidValidityWindow {
                attested_at_cycle,
                expires_at_cycle,
                profile_valid_until_cycle: profile.valid_until_cycle(),
            });
        }

        let statement = Self {
            version: ActivationVerifierCurrentnessVersion::V1,
            trust_review_digest: review.review_digest(),
            activation_evidence_receipt_digest: review
                .activation_evidence_receipt_digest()
                .as_bytes(),
            issuance_record_digest: review.issuance_record_digest().as_bytes(),
            verifier_profile_digest: review.candidate_profile_digest(),
            trust_policy_digest: review.trust_policy_digest().as_bytes(),
            reference_checkpoint_digest: review.reference_checkpoint_digest().as_bytes(),
            verifier_id: profile.verifier_id().to_string(),
            implementation_id: profile.implementation_id().to_string(),
            implementation_version: profile.implementation_version().to_string(),
            trust_snapshot_digest: profile.trust_snapshot_digest(),
            trust_snapshot_sequence: profile.trust_snapshot_sequence(),
            configuration_digest: profile.configuration_digest(),
            profile_valid_until_cycle: profile.valid_until_cycle(),
            trust_reviewed_at_cycle: review.reviewed_at_cycle(),
            attested_at_cycle,
            expires_at_cycle,
            authority_id: authority_id.into(),
            evidence_kind,
        };
        statement.validate_shape()?;
        Ok(statement)
    }

    pub fn version(&self) -> ActivationVerifierCurrentnessVersion { self.version }
    pub fn trust_review_digest(&self) -> ActivationVerifierTrustReviewDigestV1 { self.trust_review_digest }
    pub fn activation_evidence_receipt_digest(&self) -> [u8; 32] { self.activation_evidence_receipt_digest }
    pub fn issuance_record_digest(&self) -> [u8; 32] { self.issuance_record_digest }
    pub fn verifier_profile_digest(&self) -> CanonicalVerifierTrustProfileDigestV1 { self.verifier_profile_digest }
    pub fn trust_policy_digest(&self) -> [u8; 32] { self.trust_policy_digest }
    pub fn reference_checkpoint_digest(&self) -> [u8; 32] { self.reference_checkpoint_digest }
    pub fn verifier_id(&self) -> &str { &self.verifier_id }
    pub fn implementation_id(&self) -> &str { &self.implementation_id }
    pub fn implementation_version(&self) -> &str { &self.implementation_version }
    pub fn trust_snapshot_digest(&self) -> [u8; 32] { self.trust_snapshot_digest }
    pub fn trust_snapshot_sequence(&self) -> u64 { self.trust_snapshot_sequence }
    pub fn configuration_digest(&self) -> [u8; 32] { self.configuration_digest }
    pub fn profile_valid_until_cycle(&self) -> u64 { self.profile_valid_until_cycle }
    pub fn trust_reviewed_at_cycle(&self) -> u64 { self.trust_reviewed_at_cycle }
    pub fn attested_at_cycle(&self) -> u64 { self.attested_at_cycle }
    pub fn expires_at_cycle(&self) -> u64 { self.expires_at_cycle }
    pub fn authority_id(&self) -> &str { &self.authority_id }
    pub fn evidence_kind(&self) -> ActivationVerifierCurrentnessEvidenceKindV1 { self.evidence_kind }

    fn validate_shape(&self) -> Result<(), ActivationVerifierCurrentnessError> {
        if self.trust_snapshot_sequence == 0 {
            return Err(ActivationVerifierCurrentnessError::InvalidTrustSnapshotSequence);
        }
        for (label, digest) in [
            ("activation-evidence-receipt", self.activation_evidence_receipt_digest),
            ("issuance-record", self.issuance_record_digest),
            ("trust-policy", self.trust_policy_digest),
            ("reference-checkpoint", self.reference_checkpoint_digest),
            ("trust-snapshot", self.trust_snapshot_digest),
            ("configuration", self.configuration_digest),
        ] {
            if digest == [0; 32] {
                return Err(ActivationVerifierCurrentnessError::ZeroDigest(label));
            }
        }
        validate_identifier(&self.verifier_id)?;
        validate_identifier(&self.implementation_id)?;
        validate_identifier(&self.implementation_version)?;
        if self.authority_id.trim().is_empty()
            || self.authority_id != self.authority_id.trim()
            || self.authority_id.len() > MAX_ACTIVATION_VERIFIER_CURRENTNESS_AUTHORITY_ID_BYTES
        {
            return Err(ActivationVerifierCurrentnessError::InvalidAuthorityId);
        }
        if self.attested_at_cycle < self.trust_reviewed_at_cycle {
            return Err(ActivationVerifierCurrentnessError::AttestationPredatesTrustReview {
                attested_at_cycle: self.attested_at_cycle,
                trust_reviewed_at_cycle: self.trust_reviewed_at_cycle,
            });
        }
        if self.attested_at_cycle >= self.profile_valid_until_cycle {
            return Err(ActivationVerifierCurrentnessError::ProfileExpiredBeforeAttestation {
                attested_at_cycle: self.attested_at_cycle,
                profile_valid_until_cycle: self.profile_valid_until_cycle,
            });
        }
        if self.expires_at_cycle <= self.attested_at_cycle
            || self.expires_at_cycle > self.profile_valid_until_cycle
        {
            return Err(ActivationVerifierCurrentnessError::InvalidValidityWindow {
                attested_at_cycle: self.attested_at_cycle,
                expires_at_cycle: self.expires_at_cycle,
                profile_valid_until_cycle: self.profile_valid_until_cycle,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationVerifierCurrentnessEvidenceV1 {
    statement: ActivationVerifierCurrentnessStatementV1,
    proof: Vec<u8>,
}

impl ActivationVerifierCurrentnessEvidenceV1 {
    pub fn new(
        statement: ActivationVerifierCurrentnessStatementV1,
        proof: Vec<u8>,
    ) -> Result<Self, ActivationVerifierCurrentnessError> {
        statement.validate_shape()?;
        if proof.is_empty() {
            return Err(ActivationVerifierCurrentnessError::EmptyProof);
        }
        if proof.len() > MAX_ACTIVATION_VERIFIER_CURRENTNESS_PROOF_BYTES {
            return Err(ActivationVerifierCurrentnessError::ProofTooLarge {
                actual: proof.len(),
                maximum: MAX_ACTIVATION_VERIFIER_CURRENTNESS_PROOF_BYTES,
            });
        }
        Ok(Self { statement, proof })
    }

    pub fn statement(&self) -> &ActivationVerifierCurrentnessStatementV1 { &self.statement }
    pub fn proof(&self) -> &[u8] { &self.proof }
}

/// Returning true means the provider attests that the exact EKM-081 candidate
/// verifier trust state named by the statement is the current monotonic/head state
/// under the selected evidence kind at the statement's attestation cycle.
pub trait ActivationVerifierCurrentnessVerifierV1 {
    fn verify_current_activation_verifier_state(
        &self,
        evidence_kind: ActivationVerifierCurrentnessEvidenceKindV1,
        authority_id: &str,
        statement_digest: [u8; 32],
        proof: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedActivationVerifierCurrentnessV1 {
    statement: ActivationVerifierCurrentnessStatementV1,
    statement_digest: ActivationVerifierCurrentnessStatementDigestV1,
    proof_digest: [u8; 32],
    verified_at_cycle: u64,
    global_current_head_independently_proven: bool,
    verifier_correctness_independently_proven: bool,
    trusted_state_mutated: bool,
    epoch_issuance_chain_verified: bool,
    epoch_issuance_authorized: bool,
    mutation_authority: bool,
    activation_authorized: bool,
    receipt_digest: VerifiedActivationVerifierCurrentnessDigestV1,
}

impl VerifiedActivationVerifierCurrentnessV1 {
    pub fn statement(&self) -> &ActivationVerifierCurrentnessStatementV1 { &self.statement }
    pub fn statement_digest(&self) -> ActivationVerifierCurrentnessStatementDigestV1 { self.statement_digest }
    pub fn proof_digest(&self) -> [u8; 32] { self.proof_digest }
    pub fn verified_at_cycle(&self) -> u64 { self.verified_at_cycle }
    pub fn global_current_head_independently_proven(&self) -> bool { self.global_current_head_independently_proven }
    pub fn verifier_correctness_independently_proven(&self) -> bool { self.verifier_correctness_independently_proven }
    pub fn trusted_state_mutated(&self) -> bool { self.trusted_state_mutated }
    pub fn epoch_issuance_chain_verified(&self) -> bool { self.epoch_issuance_chain_verified }
    pub fn epoch_issuance_authorized(&self) -> bool { self.epoch_issuance_authorized }
    pub fn mutation_authority(&self) -> bool { self.mutation_authority }
    pub fn activation_authorized(&self) -> bool { self.activation_authorized }
    pub fn receipt_digest(&self) -> VerifiedActivationVerifierCurrentnessDigestV1 { self.receipt_digest }

    pub fn verify_internal(&self) -> Result<(), ActivationVerifierCurrentnessError> {
        self.statement.validate_shape()?;
        if digest_statement(&self.statement)? != self.statement_digest {
            return Err(ActivationVerifierCurrentnessError::StatementDigestMismatch);
        }
        if !self.global_current_head_independently_proven
            || self.verifier_correctness_independently_proven
            || self.trusted_state_mutated
            || self.epoch_issuance_chain_verified
            || self.epoch_issuance_authorized
            || self.mutation_authority
            || self.activation_authorized
        {
            return Err(ActivationVerifierCurrentnessError::UnexpectedAuthority);
        }
        if digest_verified_receipt(self)? != self.receipt_digest {
            return Err(ActivationVerifierCurrentnessError::ReceiptDigestMismatch);
        }
        Ok(())
    }
}

pub fn verify_activation_verifier_currentness(
    evidence: &ActivationVerifierCurrentnessEvidenceV1,
    review: &ActivationVerifierTrustReviewReceiptV1,
    observed_at_cycle: u64,
    verifier: &dyn ActivationVerifierCurrentnessVerifierV1,
) -> Result<VerifiedActivationVerifierCurrentnessV1, ActivationVerifierCurrentnessError> {
    evidence.statement.validate_shape()?;
    validate_statement_binding(&evidence.statement, review)?;
    if observed_at_cycle < evidence.statement.attested_at_cycle {
        return Err(ActivationVerifierCurrentnessError::ObservationPredatesAttestation {
            observed_at_cycle,
            attested_at_cycle: evidence.statement.attested_at_cycle,
        });
    }
    if observed_at_cycle >= evidence.statement.expires_at_cycle {
        return Err(ActivationVerifierCurrentnessError::CurrentnessExpired {
            observed_at_cycle,
            expires_at_cycle: evidence.statement.expires_at_cycle,
        });
    }

    let statement_digest = digest_statement(&evidence.statement)?;
    let accepted = verifier
        .verify_current_activation_verifier_state(
            evidence.statement.evidence_kind,
            &evidence.statement.authority_id,
            statement_digest.as_bytes(),
            &evidence.proof,
        )
        .map_err(ActivationVerifierCurrentnessError::VerificationProvider)?;
    if !accepted {
        return Err(ActivationVerifierCurrentnessError::ProofRejected);
    }

    let mut proof_hasher = blake3::Hasher::new();
    proof_hasher.update(b"symthaea-ekm-activation-verifier-currentness-proof-v1");
    proof_hasher.update(&evidence.proof);

    let mut result = VerifiedActivationVerifierCurrentnessV1 {
        statement: evidence.statement.clone(),
        statement_digest,
        proof_digest: *proof_hasher.finalize().as_bytes(),
        verified_at_cycle: observed_at_cycle,
        global_current_head_independently_proven: true,
        verifier_correctness_independently_proven: false,
        trusted_state_mutated: false,
        epoch_issuance_chain_verified: false,
        epoch_issuance_authorized: false,
        mutation_authority: false,
        activation_authorized: false,
        receipt_digest: VerifiedActivationVerifierCurrentnessDigestV1([0; 32]),
    };
    result.receipt_digest = digest_verified_receipt(&result)?;
    Ok(result)
}

fn validate_statement_binding(
    statement: &ActivationVerifierCurrentnessStatementV1,
    review: &ActivationVerifierTrustReviewReceiptV1,
) -> Result<(), ActivationVerifierCurrentnessError> {
    let profile = review.candidate_profile();
    if statement.trust_review_digest != review.review_digest()
        || statement.activation_evidence_receipt_digest
            != review.activation_evidence_receipt_digest().as_bytes()
        || statement.issuance_record_digest != review.issuance_record_digest().as_bytes()
        || statement.verifier_profile_digest != review.candidate_profile_digest()
        || statement.trust_policy_digest != review.trust_policy_digest().as_bytes()
        || statement.reference_checkpoint_digest != review.reference_checkpoint_digest().as_bytes()
        || statement.verifier_id != profile.verifier_id()
        || statement.implementation_id != profile.implementation_id()
        || statement.implementation_version != profile.implementation_version()
        || statement.trust_snapshot_digest != profile.trust_snapshot_digest()
        || statement.trust_snapshot_sequence != profile.trust_snapshot_sequence()
        || statement.configuration_digest != profile.configuration_digest()
        || statement.profile_valid_until_cycle != profile.valid_until_cycle()
        || statement.trust_reviewed_at_cycle != review.reviewed_at_cycle()
    {
        return Err(ActivationVerifierCurrentnessError::ReviewBindingMismatch);
    }
    if !review.local_policy_accepted()
        || !review.continuity_relative_to_caller_checkpoint_proven()
        || !review.profile_fresh_at_review_cycle()
        || review.provider_trust_independently_established()
        || review.global_current_head_independently_proven()
        || review.epoch_issuance_chain_verified()
        || review.epoch_issuance_authorized()
        || review.mutation_authority()
        || review.activation_authorized()
    {
        return Err(ActivationVerifierCurrentnessError::UnexpectedReviewAuthority);
    }
    Ok(())
}

fn digest_statement(
    statement: &ActivationVerifierCurrentnessStatementV1,
) -> Result<ActivationVerifierCurrentnessStatementDigestV1, ActivationVerifierCurrentnessError> {
    statement.validate_shape()?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-activation-verifier-currentness-statement-v1");
    hasher.update(&[1]);
    hasher.update(&statement.trust_review_digest.as_bytes());
    hasher.update(&statement.activation_evidence_receipt_digest);
    hasher.update(&statement.issuance_record_digest);
    hasher.update(&statement.verifier_profile_digest.as_bytes());
    hasher.update(&statement.trust_policy_digest);
    hasher.update(&statement.reference_checkpoint_digest);
    hash_string(&mut hasher, &statement.verifier_id)?;
    hash_string(&mut hasher, &statement.implementation_id)?;
    hash_string(&mut hasher, &statement.implementation_version)?;
    hasher.update(&statement.trust_snapshot_digest);
    hasher.update(&statement.trust_snapshot_sequence.to_le_bytes());
    hasher.update(&statement.configuration_digest);
    hasher.update(&statement.profile_valid_until_cycle.to_le_bytes());
    hasher.update(&statement.trust_reviewed_at_cycle.to_le_bytes());
    hasher.update(&statement.attested_at_cycle.to_le_bytes());
    hasher.update(&statement.expires_at_cycle.to_le_bytes());
    hash_string(&mut hasher, &statement.authority_id)?;
    hasher.update(&[statement.evidence_kind.tag()]);
    Ok(ActivationVerifierCurrentnessStatementDigestV1(*hasher.finalize().as_bytes()))
}

fn digest_verified_receipt(
    receipt: &VerifiedActivationVerifierCurrentnessV1,
) -> Result<VerifiedActivationVerifierCurrentnessDigestV1, ActivationVerifierCurrentnessError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-verified-activation-verifier-currentness-v1");
    hasher.update(&[1]);
    hasher.update(&receipt.statement_digest.as_bytes());
    hasher.update(&receipt.proof_digest);
    hasher.update(&receipt.verified_at_cycle.to_le_bytes());
    for value in [
        receipt.global_current_head_independently_proven,
        receipt.verifier_correctness_independently_proven,
        receipt.trusted_state_mutated,
        receipt.epoch_issuance_chain_verified,
        receipt.epoch_issuance_authorized,
        receipt.mutation_authority,
        receipt.activation_authorized,
    ] {
        hasher.update(&[u8::from(value)]);
    }
    Ok(VerifiedActivationVerifierCurrentnessDigestV1(*hasher.finalize().as_bytes()))
}

fn validate_identifier(value: &str) -> Result<(), ActivationVerifierCurrentnessError> {
    if value.trim().is_empty() || value != value.trim() || value.len() > 16 * 1024 {
        return Err(ActivationVerifierCurrentnessError::InvalidIdentifier);
    }
    Ok(())
}

fn hash_string(
    hasher: &mut blake3::Hasher,
    value: &str,
) -> Result<(), ActivationVerifierCurrentnessError> {
    let len = u64::try_from(value.len()).map_err(|_| ActivationVerifierCurrentnessError::LengthOverflow)?;
    hasher.update(&len.to_le_bytes());
    hasher.update(value.as_bytes());
    Ok(())
}

fn hex32(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationVerifierCurrentnessError {
    UnexpectedReviewAuthority,
    InvalidTrustSnapshotSequence,
    ZeroDigest(&'static str),
    InvalidIdentifier,
    InvalidAuthorityId,
    AttestationPredatesTrustReview { attested_at_cycle: u64, trust_reviewed_at_cycle: u64 },
    ProfileExpiredBeforeAttestation { attested_at_cycle: u64, profile_valid_until_cycle: u64 },
    InvalidValidityWindow { attested_at_cycle: u64, expires_at_cycle: u64, profile_valid_until_cycle: u64 },
    EmptyProof,
    ProofTooLarge { actual: usize, maximum: usize },
    ReviewBindingMismatch,
    ObservationPredatesAttestation { observed_at_cycle: u64, attested_at_cycle: u64 },
    CurrentnessExpired { observed_at_cycle: u64, expires_at_cycle: u64 },
    VerificationProvider(String),
    ProofRejected,
    StatementDigestMismatch,
    ReceiptDigestMismatch,
    UnexpectedAuthority,
    LengthOverflow,
}

impl fmt::Display for ActivationVerifierCurrentnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "activation verifier currentness invalid: {self:?}")
    }
}
impl Error for ActivationVerifierCurrentnessError {}

#[cfg(test)]
mod tests {
    use super::*;

    struct AcceptOnly(&'static [u8]);
    impl ActivationVerifierCurrentnessVerifierV1 for AcceptOnly {
        fn verify_current_activation_verifier_state(
            &self,
            _evidence_kind: ActivationVerifierCurrentnessEvidenceKindV1,
            _authority_id: &str,
            _statement_digest: [u8; 32],
            proof: &[u8],
        ) -> Result<bool, String> {
            Ok(proof == self.0)
        }
    }

    #[test]
    fn evidence_kinds_exclude_generic_signature_semantics() {
        let kinds = [
            ActivationVerifierCurrentnessEvidenceKindV1::MonotonicProtectedState,
            ActivationVerifierCurrentnessEvidenceKindV1::HardwareMonotonicCounter,
            ActivationVerifierCurrentnessEvidenceKindV1::TransparencyLogHead,
            ActivationVerifierCurrentnessEvidenceKindV1::CurrentHeadWitnessQuorum,
        ];
        assert_eq!(kinds.len(), 4);
        assert_eq!(kinds[0].tag(), 1);
        assert_eq!(kinds[3].tag(), 4);
    }

    #[test]
    fn provider_interface_can_reject_nonmatching_proof() {
        let verifier = AcceptOnly(b"current");
        assert!(!verifier
            .verify_current_activation_verifier_state(
                ActivationVerifierCurrentnessEvidenceKindV1::TransparencyLogHead,
                "authority",
                [1; 32],
                b"stale",
            )
            .unwrap());
    }
}
