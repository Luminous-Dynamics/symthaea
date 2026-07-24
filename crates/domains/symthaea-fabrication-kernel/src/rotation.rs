// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Policy-controlled, lifecycle-governed trust snapshot rotation.

use crate::attestation::SignatureAlgorithm;
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::trust::{
    KeyEligibility, KeyLifecycleStatus, KeyUsage, TrustSnapshot, TrustSnapshotError,
    digest_trust_snapshot,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const TRUST_ROTATION_SCHEMA: &str = "symthaea.fabrication.trust-rotation.v1";
pub const TRUST_ROTATION_POLICY_SCHEMA: &str = "symthaea.fabrication.trust-rotation-policy.v1";
pub const MAX_ROTATION_SIGNATURES: usize = 64;
pub const MAX_ROTATION_SIGNATURE_BYTES: usize = 64 * 1024;
pub const MAX_ROTATION_KEY_ID_BYTES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyRotationPolicy {
    pub schema_version: String,
    pub minimum_valid_signatures: usize,
    pub maximum_signatures: usize,
    pub require_algorithm_diversity: bool,
    pub maximum_activation_delay_s: u64,
    pub minimum_overlap_s: u64,
    pub allow_emergency_revocation: bool,
    pub minimum_active_keys_per_usage: BTreeMap<KeyUsage, u16>,
}

impl Default for KeyRotationPolicy {
    fn default() -> Self {
        Self {
            schema_version: TRUST_ROTATION_POLICY_SCHEMA.into(),
            minimum_valid_signatures: 2,
            maximum_signatures: 16,
            require_algorithm_diversity: true,
            maximum_activation_delay_s: 24 * 60 * 60,
            minimum_overlap_s: 60 * 60,
            allow_emergency_revocation: true,
            minimum_active_keys_per_usage: BTreeMap::from([
                (KeyUsage::FabricationManifest, 1),
                (KeyUsage::MachineSession, 1),
                (KeyUsage::MachineTelemetry, 1),
                (KeyUsage::OperatorCommand, 1),
                (KeyUsage::GatewayConsensus, 1),
                (KeyUsage::IncidentEvidence, 1),
                (KeyUsage::ReleaseCertification, 2),
                (KeyUsage::TrustRotation, 2),
                (KeyUsage::RecoveryAuthorization, 1),
                (KeyUsage::AuditAnchor, 1),
                (KeyUsage::ThresholdCeremony, 2),
                (KeyUsage::GatewayMembership, 2),
                (KeyUsage::TransparencyLog, 1),
                (KeyUsage::ReleasePromotion, 2),
            ]),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustRotationProposal {
    pub schema_version: String,
    pub current_snapshot_digest: Sha256Digest,
    pub proposed_snapshot: TrustSnapshot,
    pub activates_at_unix_s: u64,
    pub emergency: bool,
    pub reason_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustRotationSignature {
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedTrustRotationProposal {
    pub proposal: TrustRotationProposal,
    pub signatures: Vec<TrustRotationSignature>,
}

pub trait TrustRotationSigner {
    fn algorithm(&self) -> SignatureAlgorithm;
    fn key_id(&self) -> &str;
    fn sign_rotation(&self, message: &[u8]) -> Result<Vec<u8>, String>;
}

pub trait TrustRotationVerifier {
    fn verify_rotation(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustRotationViolation {
    UnsupportedPolicySchema,
    InvalidSignatureBounds,
    InvalidActivationDelay,
    EmptyCoverageRequirement(KeyUsage),
    UnsupportedProposalSchema,
    CurrentSnapshot(TrustSnapshotError),
    ProposedSnapshot(TrustSnapshotError),
    CurrentSnapshotStale,
    CurrentDigestMismatch,
    SequenceNotMonotonic {
        current: u64,
        proposed: u64,
    },
    ActivationDoesNotMatchIssueTime,
    ActivationInPast,
    ActivationTooFarInFuture,
    EmergencyRotationNotAllowed,
    TooManySignatures {
        actual: usize,
        maximum: usize,
    },
    DuplicateSigner {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
    InvalidAlgorithm,
    InvalidKeyId,
    SignatureTooLarge {
        actual: usize,
        maximum: usize,
    },
    SignatureVerification(String),
    SignatureInvalid {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
    SignerIneligible {
        algorithm: SignatureAlgorithm,
        key_id: String,
        eligibility: KeyEligibility,
    },
    InsufficientSignatures {
        actual: usize,
        required: usize,
    },
    MissingAlgorithmDiversity,
    KeyRemovedWithoutLifecycleRecord {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
    UsageCoverageMissing {
        usage: KeyUsage,
        actual: usize,
        required: usize,
    },
    OverlapCoverageMissing {
        usage: KeyUsage,
        actual: usize,
        required: usize,
    },
    Encoding(String),
}

#[derive(Debug, Clone)]
pub struct AuthorizedTrustRotation {
    proposal: TrustRotationProposal,
    proposal_digest: Sha256Digest,
    policy_digest: Sha256Digest,
    proposed_snapshot_digest: Sha256Digest,
    valid_signers: Vec<(SignatureAlgorithm, String)>,
    authorized_at_unix_s: u64,
}

impl AuthorizedTrustRotation {
    pub fn current_snapshot_digest(&self) -> Sha256Digest {
        self.proposal.current_snapshot_digest
    }

    pub fn activates_at_unix_s(&self) -> u64 {
        self.proposal.activates_at_unix_s
    }

    pub fn proposal(&self) -> &TrustRotationProposal {
        &self.proposal
    }

    pub fn proposal_digest(&self) -> Sha256Digest {
        self.proposal_digest
    }

    pub fn policy_digest(&self) -> Sha256Digest {
        self.policy_digest
    }

    pub fn proposed_snapshot_digest(&self) -> Sha256Digest {
        self.proposed_snapshot_digest
    }

    pub fn valid_signers(&self) -> &[(SignatureAlgorithm, String)] {
        &self.valid_signers
    }

    pub fn authorized_at_unix_s(&self) -> u64 {
        self.authorized_at_unix_s
    }

    pub fn into_proposed_snapshot(self) -> TrustSnapshot {
        self.proposal.proposed_snapshot
    }
}

impl KeyRotationPolicy {
    pub fn validate(&self) -> Result<(), Vec<TrustRotationViolation>> {
        let mut violations = Vec::new();
        if self.schema_version != TRUST_ROTATION_POLICY_SCHEMA {
            violations.push(TrustRotationViolation::UnsupportedPolicySchema);
        }
        if self.minimum_valid_signatures == 0
            || self.maximum_signatures == 0
            || self.minimum_valid_signatures > self.maximum_signatures
            || self.maximum_signatures > MAX_ROTATION_SIGNATURES
        {
            violations.push(TrustRotationViolation::InvalidSignatureBounds);
        }
        if self.maximum_activation_delay_s == 0 {
            violations.push(TrustRotationViolation::InvalidActivationDelay);
        }
        for (usage, minimum) in &self.minimum_active_keys_per_usage {
            if *minimum == 0 {
                violations.push(TrustRotationViolation::EmptyCoverageRequirement(*usage));
            }
        }
        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }
}

pub fn canonical_rotation_policy_bytes(
    policy: &KeyRotationPolicy,
) -> Result<Vec<u8>, Vec<TrustRotationViolation>> {
    policy.validate()?;
    serde_json::to_vec(policy)
        .map_err(|error| vec![TrustRotationViolation::Encoding(error.to_string())])
}

pub fn digest_rotation_policy(
    policy: &KeyRotationPolicy,
) -> Result<Sha256Digest, Vec<TrustRotationViolation>> {
    let bytes = canonical_rotation_policy_bytes(policy)?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.trust-rotation-policy-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn canonical_rotation_proposal_bytes(
    proposal: &TrustRotationProposal,
) -> Result<Vec<u8>, TrustRotationViolation> {
    if proposal.schema_version != TRUST_ROTATION_SCHEMA {
        return Err(TrustRotationViolation::UnsupportedProposalSchema);
    }
    proposal
        .proposed_snapshot
        .validate()
        .map_err(TrustRotationViolation::ProposedSnapshot)?;
    serde_json::to_vec(proposal)
        .map_err(|error| TrustRotationViolation::Encoding(error.to_string()))
}

pub fn digest_rotation_proposal(
    proposal: &TrustRotationProposal,
) -> Result<Sha256Digest, TrustRotationViolation> {
    let bytes = canonical_rotation_proposal_bytes(proposal)?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.trust-rotation-proposal-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn sign_trust_rotation_proposal(
    proposal: TrustRotationProposal,
    signers: &[&dyn TrustRotationSigner],
) -> Result<SignedTrustRotationProposal, Vec<TrustRotationViolation>> {
    let bytes = canonical_rotation_proposal_bytes(&proposal).map_err(|error| vec![error])?;
    let mut identities = BTreeSet::new();
    let mut signatures = Vec::with_capacity(signers.len());
    let mut violations = Vec::new();
    if signers.len() > MAX_ROTATION_SIGNATURES {
        violations.push(TrustRotationViolation::TooManySignatures {
            actual: signers.len(),
            maximum: MAX_ROTATION_SIGNATURES,
        });
    }
    for signer in signers.iter().take(MAX_ROTATION_SIGNATURES) {
        let algorithm = signer.algorithm();
        let key_id = signer.key_id().to_string();
        if !algorithm.is_canonical() {
            violations.push(TrustRotationViolation::InvalidAlgorithm);
            continue;
        }
        if !canonical_key_id(&key_id) {
            violations.push(TrustRotationViolation::InvalidKeyId);
            continue;
        }
        if !identities.insert((algorithm.clone(), key_id.clone())) {
            violations.push(TrustRotationViolation::DuplicateSigner { algorithm, key_id });
            continue;
        }
        match signer.sign_rotation(&bytes) {
            Ok(signature)
                if !signature.is_empty() && signature.len() <= MAX_ROTATION_SIGNATURE_BYTES =>
            {
                signatures.push(TrustRotationSignature {
                    algorithm,
                    key_id,
                    signature,
                });
            }
            Ok(signature) => violations.push(TrustRotationViolation::SignatureTooLarge {
                actual: signature.len(),
                maximum: MAX_ROTATION_SIGNATURE_BYTES,
            }),
            Err(error) => violations.push(TrustRotationViolation::SignatureVerification(error)),
        }
    }
    if violations.is_empty() {
        Ok(SignedTrustRotationProposal {
            proposal,
            signatures,
        })
    } else {
        Err(violations)
    }
}

pub fn authorize_trust_rotation(
    signed: SignedTrustRotationProposal,
    current_snapshot: &TrustSnapshot,
    policy: &KeyRotationPolicy,
    evaluation_time_unix_s: u64,
    verifier: &dyn TrustRotationVerifier,
) -> Result<AuthorizedTrustRotation, Vec<TrustRotationViolation>> {
    let mut violations = Vec::new();
    if let Err(mut policy_violations) = policy.validate() {
        violations.append(&mut policy_violations);
    }
    if let Err(error) = current_snapshot.validate() {
        violations.push(TrustRotationViolation::CurrentSnapshot(error));
    }
    if let Err(error) = signed.proposal.proposed_snapshot.validate() {
        violations.push(TrustRotationViolation::ProposedSnapshot(error));
    }
    if signed.proposal.schema_version != TRUST_ROTATION_SCHEMA {
        violations.push(TrustRotationViolation::UnsupportedProposalSchema);
    }
    if !current_snapshot.is_fresh_at(evaluation_time_unix_s) {
        violations.push(TrustRotationViolation::CurrentSnapshotStale);
    }
    let current_digest = digest_trust_snapshot(current_snapshot)
        .map_err(|error| vec![TrustRotationViolation::CurrentSnapshot(error)])?;
    if signed.proposal.current_snapshot_digest != current_digest {
        violations.push(TrustRotationViolation::CurrentDigestMismatch);
    }
    if signed.proposal.proposed_snapshot.sequence != current_snapshot.sequence.saturating_add(1) {
        violations.push(TrustRotationViolation::SequenceNotMonotonic {
            current: current_snapshot.sequence,
            proposed: signed.proposal.proposed_snapshot.sequence,
        });
    }
    if signed.proposal.activates_at_unix_s != signed.proposal.proposed_snapshot.issued_at_unix_s {
        violations.push(TrustRotationViolation::ActivationDoesNotMatchIssueTime);
    }
    if signed.proposal.activates_at_unix_s < evaluation_time_unix_s {
        violations.push(TrustRotationViolation::ActivationInPast);
    }
    if signed.proposal.activates_at_unix_s
        > evaluation_time_unix_s.saturating_add(policy.maximum_activation_delay_s)
    {
        violations.push(TrustRotationViolation::ActivationTooFarInFuture);
    }
    if signed.proposal.emergency && !policy.allow_emergency_revocation {
        violations.push(TrustRotationViolation::EmergencyRotationNotAllowed);
    }
    if signed.signatures.len() > policy.maximum_signatures {
        violations.push(TrustRotationViolation::TooManySignatures {
            actual: signed.signatures.len(),
            maximum: policy.maximum_signatures,
        });
    }

    evaluate_snapshot_transition(current_snapshot, &signed.proposal, policy, &mut violations);

    let bytes = canonical_rotation_proposal_bytes(&signed.proposal).map_err(|error| vec![error])?;
    let mut signer_identities = BTreeSet::new();
    let mut valid_signers = Vec::new();
    for signature in &signed.signatures {
        if !signature.algorithm.is_canonical() {
            violations.push(TrustRotationViolation::InvalidAlgorithm);
            continue;
        }
        if !canonical_key_id(&signature.key_id) {
            violations.push(TrustRotationViolation::InvalidKeyId);
            continue;
        }
        if signature.signature.is_empty()
            || signature.signature.len() > MAX_ROTATION_SIGNATURE_BYTES
        {
            violations.push(TrustRotationViolation::SignatureTooLarge {
                actual: signature.signature.len(),
                maximum: MAX_ROTATION_SIGNATURE_BYTES,
            });
            continue;
        }
        let identity = (signature.algorithm.clone(), signature.key_id.clone());
        if !signer_identities.insert(identity.clone()) {
            violations.push(TrustRotationViolation::DuplicateSigner {
                algorithm: identity.0,
                key_id: identity.1,
            });
            continue;
        }
        let eligibility = current_snapshot.key_eligibility(
            &signature.algorithm,
            &signature.key_id,
            KeyUsage::TrustRotation,
            evaluation_time_unix_s,
        );
        if eligibility != KeyEligibility::Eligible {
            violations.push(TrustRotationViolation::SignerIneligible {
                algorithm: signature.algorithm.clone(),
                key_id: signature.key_id.clone(),
                eligibility,
            });
            continue;
        }
        match verifier.verify_rotation(
            &signature.algorithm,
            &signature.key_id,
            &bytes,
            &signature.signature,
        ) {
            Ok(true) => valid_signers.push(identity),
            Ok(false) => violations.push(TrustRotationViolation::SignatureInvalid {
                algorithm: signature.algorithm.clone(),
                key_id: signature.key_id.clone(),
            }),
            Err(error) => violations.push(TrustRotationViolation::SignatureVerification(error)),
        }
    }
    valid_signers.sort();
    if valid_signers.len() < policy.minimum_valid_signatures {
        violations.push(TrustRotationViolation::InsufficientSignatures {
            actual: valid_signers.len(),
            required: policy.minimum_valid_signatures,
        });
    }
    if policy.require_algorithm_diversity
        && valid_signers
            .iter()
            .map(|(algorithm, _)| algorithm)
            .collect::<BTreeSet<_>>()
            .len()
            < 2
    {
        violations.push(TrustRotationViolation::MissingAlgorithmDiversity);
    }

    if !violations.is_empty() {
        return Err(violations);
    }
    let proposal_digest =
        digest_rotation_proposal(&signed.proposal).map_err(|error| vec![error])?;
    let policy_digest = digest_rotation_policy(policy)?;
    let proposed_snapshot_digest = digest_trust_snapshot(&signed.proposal.proposed_snapshot)
        .map_err(|error| vec![TrustRotationViolation::ProposedSnapshot(error)])?;
    Ok(AuthorizedTrustRotation {
        proposal: signed.proposal,
        proposal_digest,
        policy_digest,
        proposed_snapshot_digest,
        valid_signers,
        authorized_at_unix_s: evaluation_time_unix_s,
    })
}

fn evaluate_snapshot_transition(
    current: &TrustSnapshot,
    proposal: &TrustRotationProposal,
    policy: &KeyRotationPolicy,
    violations: &mut Vec<TrustRotationViolation>,
) {
    let proposed = &proposal.proposed_snapshot;
    let proposed_identities: BTreeSet<_> = proposed
        .keys
        .iter()
        .map(|key| (key.algorithm.clone(), key.key_id.clone()))
        .collect();
    for key in &current.keys {
        if !proposed_identities.contains(&(key.algorithm.clone(), key.key_id.clone())) {
            violations.push(TrustRotationViolation::KeyRemovedWithoutLifecycleRecord {
                algorithm: key.algorithm.clone(),
                key_id: key.key_id.clone(),
            });
        }
    }

    let activation = proposal.activates_at_unix_s;
    let overlap_end = activation.saturating_add(policy.minimum_overlap_s);
    for (usage, required) in &policy.minimum_active_keys_per_usage {
        let required = *required as usize;
        let active_at_activation = proposed
            .keys
            .iter()
            .filter(|key| eligible_record_at(key, *usage, activation))
            .count();
        if active_at_activation < required {
            violations.push(TrustRotationViolation::UsageCoverageMissing {
                usage: *usage,
                actual: active_at_activation,
                required,
            });
        }
        if !proposal.emergency && policy.minimum_overlap_s > 0 {
            let current_identities: BTreeSet<_> = current
                .keys
                .iter()
                .filter(|key| eligible_record_at(key, *usage, activation))
                .map(|key| (key.algorithm.clone(), key.key_id.clone()))
                .collect();
            let overlap = proposed
                .keys
                .iter()
                .filter(|key| {
                    current_identities.contains(&(key.algorithm.clone(), key.key_id.clone()))
                        && eligible_record_at(key, *usage, overlap_end)
                })
                .count();
            if overlap < required {
                violations.push(TrustRotationViolation::OverlapCoverageMissing {
                    usage: *usage,
                    actual: overlap,
                    required,
                });
            }
        }
    }
}

fn eligible_record_at(key: &crate::trust::KeyTrustRecord, usage: KeyUsage, unix_s: u64) -> bool {
    key.status == KeyLifecycleStatus::Active
        && unix_s >= key.not_before_unix_s
        && key
            .not_after_unix_s
            .is_none_or(|not_after| unix_s < not_after)
        && key.usages.contains(&usage)
}

fn canonical_key_id(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_ROTATION_KEY_ID_BYTES
        && !value.chars().any(char::is_control)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;
    use crate::trust::KeyTrustRecord;

    struct Provider {
        algorithm: SignatureAlgorithm,
        key_id: String,
    }

    impl TrustRotationSigner for Provider {
        fn algorithm(&self) -> SignatureAlgorithm {
            self.algorithm.clone()
        }
        fn key_id(&self) -> &str {
            &self.key_id
        }
        fn sign_rotation(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            let mut bytes = self.key_id.as_bytes().to_vec();
            bytes.extend_from_slice(message);
            Ok(sha256(&bytes).0.to_vec())
        }
    }

    struct Verifier;

    impl TrustRotationVerifier for Verifier {
        fn verify_rotation(
            &self,
            _algorithm: &SignatureAlgorithm,
            key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            let mut bytes = key_id.as_bytes().to_vec();
            bytes.extend_from_slice(message);
            Ok(signature == sha256(&bytes).0.as_slice())
        }
    }

    fn usages() -> BTreeSet<KeyUsage> {
        BTreeSet::from([
            KeyUsage::FabricationManifest,
            KeyUsage::MachineSession,
            KeyUsage::MachineTelemetry,
            KeyUsage::OperatorCommand,
            KeyUsage::GatewayConsensus,
            KeyUsage::IncidentEvidence,
            KeyUsage::ReleaseCertification,
            KeyUsage::TrustRotation,
            KeyUsage::RecoveryAuthorization,
            KeyUsage::AuditAnchor,
            KeyUsage::ThresholdCeremony,
            KeyUsage::GatewayMembership,
            KeyUsage::TransparencyLog,
            KeyUsage::ReleasePromotion,
        ])
    }

    fn key(algorithm: SignatureAlgorithm, key_id: &str) -> KeyTrustRecord {
        KeyTrustRecord {
            algorithm,
            key_id: key_id.into(),
            not_before_unix_s: 100,
            not_after_unix_s: Some(10_000),
            status: KeyLifecycleStatus::Active,
            usages: usages(),
        }
    }

    fn current() -> TrustSnapshot {
        TrustSnapshot::new(
            1,
            100,
            5_000,
            vec![
                key(SignatureAlgorithm::Ed25519, "old-ed"),
                key(SignatureAlgorithm::MlDsa65, "old-pq"),
            ],
        )
        .unwrap()
    }

    fn proposal(current: &TrustSnapshot) -> TrustRotationProposal {
        let mut keys = current.keys.clone();
        keys.push(KeyTrustRecord {
            algorithm: SignatureAlgorithm::MlDsa87,
            key_id: "new-pq".into(),
            not_before_unix_s: 600,
            not_after_unix_s: Some(12_000),
            status: KeyLifecycleStatus::Active,
            usages: usages(),
        });
        TrustRotationProposal {
            schema_version: TRUST_ROTATION_SCHEMA.into(),
            current_snapshot_digest: digest_trust_snapshot(current).unwrap(),
            proposed_snapshot: TrustSnapshot::new(2, 600, 6_000, keys).unwrap(),
            activates_at_unix_s: 600,
            emergency: false,
            reason_digest: sha256(b"scheduled rotation"),
        }
    }

    #[test]
    fn scheduled_rotation_requires_current_diverse_quorum_and_overlap() {
        let current = current();
        let proposal = proposal(&current);
        let ed = Provider {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "old-ed".into(),
        };
        let pq = Provider {
            algorithm: SignatureAlgorithm::MlDsa65,
            key_id: "old-pq".into(),
        };
        let signed = sign_trust_rotation_proposal(proposal, &[&ed, &pq]).unwrap();
        let authorized = authorize_trust_rotation(
            signed,
            &current,
            &KeyRotationPolicy::default(),
            500,
            &Verifier,
        )
        .unwrap();
        assert_eq!(authorized.valid_signers().len(), 2);
        assert_eq!(authorized.proposal().proposed_snapshot.sequence, 2);
    }

    #[test]
    fn silent_key_removal_fails_closed() {
        let current = current();
        let mut proposal = proposal(&current);
        proposal
            .proposed_snapshot
            .keys
            .retain(|key| key.key_id != "old-ed");
        let ed = Provider {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "old-ed".into(),
        };
        let pq = Provider {
            algorithm: SignatureAlgorithm::MlDsa65,
            key_id: "old-pq".into(),
        };
        let signed = sign_trust_rotation_proposal(proposal, &[&ed, &pq]).unwrap();
        let violations = authorize_trust_rotation(
            signed,
            &current,
            &KeyRotationPolicy::default(),
            500,
            &Verifier,
        )
        .unwrap_err();
        assert!(violations.iter().any(|violation| matches!(
            violation,
            TrustRotationViolation::KeyRemovedWithoutLifecycleRecord { key_id, .. }
                if key_id == "old-ed"
        )));
    }

    #[test]
    fn emergency_rotation_must_be_explicitly_allowed() {
        let current = current();
        let mut proposal = proposal(&current);
        proposal.emergency = true;
        let ed = Provider {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "old-ed".into(),
        };
        let pq = Provider {
            algorithm: SignatureAlgorithm::MlDsa65,
            key_id: "old-pq".into(),
        };
        let signed = sign_trust_rotation_proposal(proposal, &[&ed, &pq]).unwrap();
        let mut policy = KeyRotationPolicy::default();
        policy.allow_emergency_revocation = false;
        let violations =
            authorize_trust_rotation(signed, &current, &policy, 500, &Verifier).unwrap_err();
        assert!(violations.contains(&TrustRotationViolation::EmergencyRotationNotAllowed));
    }
}
