// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact raw-signature evidence binding for witnessed policy-head authority.
//!
//! Upstream opaque checkpoint/witness capabilities prove that a signer verified one exact logical
//! checkpoint or witness statement. They do not expose the exact signature bytes that their
//! verifier consumed. This layer closes that last identity gap by requiring independent verifier
//! providers to verify the exact raw signed evidence bytes already committed by the observed head.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::transparency_checkpoint::SignedTransparencyCheckpoint;
use symthaea_fabrication_kernel::transparency_witness::SignedTransparencyWitness;
use symthaea_fabrication_policy_head_observation::{
    QuorumObservedPolicyHeadIdV1, QuorumObservedPolicyHeadV1,
};
use symthaea_fabrication_witness_authority::{
    RegistryBoundPolicyHeadIdV1, RegistryBoundPolicyHeadV1,
};

pub const EXACT_EVIDENCE_BOUND_POLICY_HEAD_SCHEMA: &str =
    "symthaea.fabrication.exact-evidence-bound-policy-head.v1";
pub const MAX_EXACT_EVIDENCE_VERIFIERS: usize = 16;

const SIGNED_CHECKPOINT_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.policy-lineage-head-checkpoint-evidence.v1\0";
const SIGNED_WITNESS_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.policy-lineage-head-witness-evidence.v1\0";
const WITNESS_SET_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.policy-lineage-head-witness-set.v1\0";
const VERIFIER_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.policy-head-exact-evidence-verifier-set.v1\0";
const EXACT_EVIDENCE_HEAD_DOMAIN: &[u8] =
    b"symthaea.fabrication.exact-evidence-bound-policy-head.v1\0";
const CHECKPOINT_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea.fabrication.transparency-checkpoint-signature.v1\0";
const WITNESS_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea.fabrication.transparency-witness-signature.v1\0";

/// Runtime cryptographic verification provider for exact signed evidence.
///
/// `provider_id` and `verification_policy_digest` are committed into the resulting capability so
/// the evidence theorem records which verifier implementations/policies actually checked the raw
/// bytes. This trait does not itself claim those provider identities are externally governed.
pub trait ExactPolicyHeadEvidenceVerifierV1 {
    fn provider_id(&self) -> &str;
    fn verification_policy_digest(&self) -> Sha256Digest;

    fn verify_checkpoint_signature(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;

    fn verify_witness_signature(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactEvidenceVerificationPolicyV1 {
    pub minimum_distinct_providers: usize,
    pub maximum_providers: usize,
}

impl Default for ExactEvidenceVerificationPolicyV1 {
    fn default() -> Self {
        Self {
            minimum_distinct_providers: 2,
            maximum_providers: 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExactEvidenceBoundPolicyHeadIdV1(Sha256Digest);

impl ExactEvidenceBoundPolicyHeadIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque proof that the exact raw checkpoint signature and every exact raw witness signature
/// committed upstream were independently reverified by the committed provider set.
#[derive(Debug, Clone)]
#[must_use]
pub struct ExactEvidenceBoundPolicyHeadV1 {
    id: ExactEvidenceBoundPolicyHeadIdV1,
    observed_head_id: QuorumObservedPolicyHeadIdV1,
    registry_bound_head_id: RegistryBoundPolicyHeadIdV1,
    checkpoint_evidence_digest: Sha256Digest,
    witness_set_evidence_digest: Sha256Digest,
    verifier_set_digest: Sha256Digest,
    verifier_count: usize,
    witness_count: usize,
}

impl ExactEvidenceBoundPolicyHeadV1 {
    pub fn id(&self) -> ExactEvidenceBoundPolicyHeadIdV1 {
        self.id
    }
    pub fn observed_head_id(&self) -> QuorumObservedPolicyHeadIdV1 {
        self.observed_head_id
    }
    pub fn registry_bound_head_id(&self) -> RegistryBoundPolicyHeadIdV1 {
        self.registry_bound_head_id
    }
    pub fn checkpoint_evidence_digest(&self) -> Sha256Digest {
        self.checkpoint_evidence_digest
    }
    pub fn witness_set_evidence_digest(&self) -> Sha256Digest {
        self.witness_set_evidence_digest
    }
    pub fn verifier_set_digest(&self) -> Sha256Digest {
        self.verifier_set_digest
    }
    pub fn verifier_count(&self) -> usize {
        self.verifier_count
    }
    pub fn witness_count(&self) -> usize {
        self.witness_count
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExactEvidenceBindingError {
    InvalidPolicy,
    InsufficientProviders { actual: usize, required: usize },
    TooManyProviders { actual: usize, maximum: usize },
    InvalidProviderId(String),
    ZeroProviderPolicyDigest(String),
    DuplicateProvider(String),
    ObservedHeadMismatch,
    CheckpointEvidenceMismatch,
    WitnessSetEvidenceMismatch,
    CheckpointDigestMismatch,
    InvalidCheckpointSigner,
    InvalidWitnessSigner(String),
    WitnessCheckpointMismatch(String),
    CheckpointSignatureRejected(String),
    WitnessSignatureRejected { provider: String, key_id: String },
    VerificationProviderError { provider: String, reason: String },
    Encoding(String),
}

pub fn bind_exact_policy_head_signature_evidence_v1(
    registry_bound: &RegistryBoundPolicyHeadV1,
    observed: &QuorumObservedPolicyHeadV1,
    signed_checkpoint: &SignedTransparencyCheckpoint,
    signed_witnesses: &[SignedTransparencyWitness],
    policy: &ExactEvidenceVerificationPolicyV1,
    providers: &[&dyn ExactPolicyHeadEvidenceVerifierV1],
) -> Result<ExactEvidenceBoundPolicyHeadV1, Vec<ExactEvidenceBindingError>> {
    let mut violations = Vec::new();

    if policy.minimum_distinct_providers == 0
        || policy.maximum_providers == 0
        || policy.minimum_distinct_providers > policy.maximum_providers
        || policy.maximum_providers > MAX_EXACT_EVIDENCE_VERIFIERS
    {
        violations.push(ExactEvidenceBindingError::InvalidPolicy);
    }
    if providers.len() < policy.minimum_distinct_providers {
        violations.push(ExactEvidenceBindingError::InsufficientProviders {
            actual: providers.len(),
            required: policy.minimum_distinct_providers,
        });
    }
    if providers.len() > policy.maximum_providers {
        violations.push(ExactEvidenceBindingError::TooManyProviders {
            actual: providers.len(),
            maximum: policy.maximum_providers,
        });
    }
    if registry_bound.observed_head_id() != observed.id() {
        violations.push(ExactEvidenceBindingError::ObservedHeadMismatch);
    }

    let checkpoint_evidence_digest = match hash_serializable(
        SIGNED_CHECKPOINT_EVIDENCE_DOMAIN,
        signed_checkpoint,
    ) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };
    if checkpoint_evidence_digest != observed.signed_checkpoint_evidence_digest() {
        violations.push(ExactEvidenceBindingError::CheckpointEvidenceMismatch);
    }

    let witness_set_evidence_digest = match digest_exact_witness_set(signed_witnesses) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };
    if witness_set_evidence_digest != observed.witness_set_evidence_digest()
        || witness_set_evidence_digest != registry_bound.witness_set_evidence_digest()
    {
        violations.push(ExactEvidenceBindingError::WitnessSetEvidenceMismatch);
    }

    if signed_checkpoint.checkpoint_digest != observed.checkpoint_digest() {
        violations.push(ExactEvidenceBindingError::CheckpointDigestMismatch);
    }
    if !signed_checkpoint.signature.algorithm.is_canonical()
        || invalid_identifier(&signed_checkpoint.signature.key_id)
        || signed_checkpoint.signature.signature.is_empty()
    {
        violations.push(ExactEvidenceBindingError::InvalidCheckpointSigner);
    }

    let mut seen_witnesses = BTreeSet::new();
    for witness in signed_witnesses {
        let key_id = witness.signature.key_id.clone();
        if !witness.signature.algorithm.is_canonical()
            || invalid_identifier(&key_id)
            || witness.signature.signature.is_empty()
        {
            violations.push(ExactEvidenceBindingError::InvalidWitnessSigner(key_id));
            continue;
        }
        if witness.statement.checkpoint_digest != observed.checkpoint_digest() {
            violations.push(ExactEvidenceBindingError::WitnessCheckpointMismatch(key_id));
        }
        if !seen_witnesses.insert((
            witness.signature.algorithm.clone(),
            witness.signature.key_id.clone(),
            witness.statement_digest,
        )) {
            violations.push(ExactEvidenceBindingError::InvalidWitnessSigner(
                witness.signature.key_id.clone(),
            ));
        }
    }

    let mut provider_commitments = Vec::new();
    let mut seen_providers = BTreeSet::new();
    for provider in providers {
        let provider_id = provider.provider_id().to_string();
        if invalid_identifier(&provider_id) {
            violations.push(ExactEvidenceBindingError::InvalidProviderId(provider_id));
            continue;
        }
        let policy_digest = provider.verification_policy_digest();
        if policy_digest == Sha256Digest([0; 32]) {
            violations.push(ExactEvidenceBindingError::ZeroProviderPolicyDigest(
                provider_id,
            ));
            continue;
        }
        if !seen_providers.insert(provider_id.clone()) {
            violations.push(ExactEvidenceBindingError::DuplicateProvider(provider_id));
            continue;
        }
        provider_commitments.push(VerifierCommitment {
            provider_id: provider_id.clone(),
            verification_policy_digest: policy_digest.to_hex(),
        });

        let checkpoint_message = checkpoint_signature_message(signed_checkpoint.checkpoint_digest);
        match provider.verify_checkpoint_signature(
            &signed_checkpoint.signature.algorithm,
            &signed_checkpoint.signature.key_id,
            &checkpoint_message,
            &signed_checkpoint.signature.signature,
        ) {
            Ok(true) => {}
            Ok(false) => violations.push(ExactEvidenceBindingError::CheckpointSignatureRejected(
                provider_id.clone(),
            )),
            Err(reason) => violations.push(ExactEvidenceBindingError::VerificationProviderError {
                provider: provider_id.clone(),
                reason,
            }),
        }

        for witness in signed_witnesses {
            let message = witness_signature_message(witness.statement_digest);
            match provider.verify_witness_signature(
                &witness.signature.algorithm,
                &witness.signature.key_id,
                &message,
                &witness.signature.signature,
            ) {
                Ok(true) => {}
                Ok(false) => violations.push(ExactEvidenceBindingError::WitnessSignatureRejected {
                    provider: provider_id.clone(),
                    key_id: witness.signature.key_id.clone(),
                }),
                Err(reason) => {
                    violations.push(ExactEvidenceBindingError::VerificationProviderError {
                        provider: provider_id.clone(),
                        reason: format!("{}: {reason}", witness.signature.key_id),
                    })
                }
            }
        }
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    provider_commitments.sort_by(|left, right| left.provider_id.cmp(&right.provider_id));
    let verifier_set_digest = hash_serializable(VERIFIER_SET_DOMAIN, &provider_commitments)
        .map_err(|error| vec![error])?;
    let id = ExactEvidenceBoundPolicyHeadIdV1(
        digest_exact_evidence_head(
            observed.id(),
            registry_bound.id(),
            checkpoint_evidence_digest,
            witness_set_evidence_digest,
            verifier_set_digest,
            provider_commitments.len(),
            signed_witnesses.len(),
        )
        .map_err(|error| vec![error])?,
    );

    Ok(ExactEvidenceBoundPolicyHeadV1 {
        id,
        observed_head_id: observed.id(),
        registry_bound_head_id: registry_bound.id(),
        checkpoint_evidence_digest,
        witness_set_evidence_digest,
        verifier_set_digest,
        verifier_count: provider_commitments.len(),
        witness_count: signed_witnesses.len(),
    })
}

fn checkpoint_signature_message(digest: Sha256Digest) -> Vec<u8> {
    let mut message = CHECKPOINT_SIGNATURE_DOMAIN.to_vec();
    message.extend_from_slice(&digest.0);
    message
}

fn witness_signature_message(digest: Sha256Digest) -> Vec<u8> {
    let mut message = WITNESS_SIGNATURE_DOMAIN.to_vec();
    message.extend_from_slice(&digest.0);
    message
}

fn digest_exact_witness_set(
    signed_witnesses: &[SignedTransparencyWitness],
) -> Result<Sha256Digest, ExactEvidenceBindingError> {
    let mut evidence_digests = signed_witnesses
        .iter()
        .map(|witness| hash_serializable(SIGNED_WITNESS_EVIDENCE_DOMAIN, witness))
        .collect::<Result<Vec<_>, _>>()?;
    evidence_digests.sort();
    let mut hasher = Sha256::new();
    hasher.update(WITNESS_SET_EVIDENCE_DOMAIN);
    hasher.update(&(evidence_digests.len() as u64).to_le_bytes());
    for digest in evidence_digests {
        hasher.update(&digest.0);
    }
    Ok(hasher.finalize())
}

fn invalid_identifier(value: &str) -> bool {
    value.trim().is_empty()
        || value != value.trim()
        || value.len() > 256
        || value.chars().any(char::is_control)
}

#[derive(Debug, Clone, Serialize)]
struct VerifierCommitment {
    provider_id: String,
    verification_policy_digest: String,
}

#[derive(Serialize)]
struct ExactEvidenceHeadCommitment {
    schema: &'static str,
    observed_head_id: String,
    registry_bound_head_id: String,
    checkpoint_evidence_digest: String,
    witness_set_evidence_digest: String,
    verifier_set_digest: String,
    verifier_count: usize,
    witness_count: usize,
}

#[allow(clippy::too_many_arguments)]
fn digest_exact_evidence_head(
    observed_head_id: QuorumObservedPolicyHeadIdV1,
    registry_bound_head_id: RegistryBoundPolicyHeadIdV1,
    checkpoint_evidence_digest: Sha256Digest,
    witness_set_evidence_digest: Sha256Digest,
    verifier_set_digest: Sha256Digest,
    verifier_count: usize,
    witness_count: usize,
) -> Result<Sha256Digest, ExactEvidenceBindingError> {
    hash_serializable(
        EXACT_EVIDENCE_HEAD_DOMAIN,
        &ExactEvidenceHeadCommitment {
            schema: EXACT_EVIDENCE_BOUND_POLICY_HEAD_SCHEMA,
            observed_head_id: observed_head_id.to_hex(),
            registry_bound_head_id: registry_bound_head_id.to_hex(),
            checkpoint_evidence_digest: checkpoint_evidence_digest.to_hex(),
            witness_set_evidence_digest: witness_set_evidence_digest.to_hex(),
            verifier_set_digest: verifier_set_digest.to_hex(),
            verifier_count,
            witness_count,
        },
    )
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, ExactEvidenceBindingError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| ExactEvidenceBindingError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
