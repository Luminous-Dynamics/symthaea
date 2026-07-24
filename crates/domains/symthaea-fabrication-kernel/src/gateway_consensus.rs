// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-gateway quorum over one exact durable gateway-state generation.

use crate::attestation::{DetachedSignature, SignatureAlgorithm};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::gateway_state::FabricationGatewayState;
use crate::trust::{KeyEligibility, KeyUsage, TrustSnapshot, digest_trust_snapshot};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const GATEWAY_ENDORSEMENT_SCHEMA: &str = "symthaea.fabrication.gateway-endorsement.v1";
pub const SIGNED_GATEWAY_ENDORSEMENT_SCHEMA: &str =
    "symthaea.fabrication.signed-gateway-endorsement.v1";
pub const MAX_GATEWAY_ENDORSEMENTS: usize = 64;
pub const MAX_GATEWAY_ENDORSEMENT_SIGNATURE_BYTES: usize = 64 * 1024;
pub const MAX_GATEWAY_ID_BYTES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayEndorsement {
    pub schema_version: String,
    pub gateway_id: String,
    pub gateway_state_digest: Sha256Digest,
    pub gateway_generation: u64,
    pub previous_gateway_state_digest: Option<Sha256Digest>,
    pub state_committed_at_unix_ms: u64,
    pub issued_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedGatewayEndorsement {
    pub schema_version: String,
    pub endorsement: GatewayEndorsement,
    pub endorsement_digest: Sha256Digest,
    pub signature: DetachedSignature,
}

pub trait GatewayEndorsementSigner {
    fn algorithm(&self) -> SignatureAlgorithm;
    fn key_id(&self) -> &str;
    fn sign_gateway_endorsement(&self, message: &[u8]) -> Result<Vec<u8>, String>;
}

pub trait GatewayEndorsementVerifier {
    fn verify_gateway_endorsement(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayConsensusError {
    InvalidState(String),
    UnsupportedSchema,
    InvalidWindow,
    EmptyGatewayId,
    NonCanonicalGatewayId,
    GatewayIdTooLong { actual: usize, maximum: usize },
    InvalidAlgorithm,
    EmptyKeyId,
    NonCanonicalKeyId,
    EmptySignature,
    SignatureTooLarge { actual: usize, maximum: usize },
    Signing(String),
    Encoding(String),
}

impl GatewayEndorsement {
    pub fn validate(&self) -> Result<(), GatewayConsensusError> {
        if self.schema_version != GATEWAY_ENDORSEMENT_SCHEMA {
            return Err(GatewayConsensusError::UnsupportedSchema);
        }
        validate_gateway_id(&self.gateway_id)?;
        if self.gateway_generation == 0 {
            return Err(GatewayConsensusError::InvalidState(
                "gateway generation is zero".into(),
            ));
        }
        if self.issued_at_unix_ms >= self.expires_at_unix_ms {
            return Err(GatewayConsensusError::InvalidWindow);
        }
        Ok(())
    }
}

pub fn canonical_gateway_endorsement_bytes(
    endorsement: &GatewayEndorsement,
) -> Result<Vec<u8>, GatewayConsensusError> {
    endorsement.validate()?;
    serde_json::to_vec(endorsement)
        .map_err(|error| GatewayConsensusError::Encoding(error.to_string()))
}

pub fn digest_gateway_endorsement(
    endorsement: &GatewayEndorsement,
) -> Result<Sha256Digest, GatewayConsensusError> {
    let bytes = canonical_gateway_endorsement_bytes(endorsement)?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.gateway-endorsement-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn endorse_gateway_state(
    state: &FabricationGatewayState,
    gateway_id: impl Into<String>,
    issued_at_unix_ms: u64,
    expires_at_unix_ms: u64,
    signer: &dyn GatewayEndorsementSigner,
) -> Result<SignedGatewayEndorsement, GatewayConsensusError> {
    state
        .validate()
        .map_err(|error| GatewayConsensusError::InvalidState(format!("{error:?}")))?;
    let gateway_id = gateway_id.into();
    validate_gateway_id(&gateway_id)?;
    let endorsement = GatewayEndorsement {
        schema_version: GATEWAY_ENDORSEMENT_SCHEMA.into(),
        gateway_id,
        gateway_state_digest: state
            .digest()
            .map_err(|error| GatewayConsensusError::InvalidState(format!("{error:?}")))?,
        gateway_generation: state.generation,
        previous_gateway_state_digest: state.previous_state_digest,
        state_committed_at_unix_ms: state.committed_at_unix_ms,
        issued_at_unix_ms,
        expires_at_unix_ms,
    };
    endorsement.validate()?;
    let algorithm = signer.algorithm();
    if !algorithm.is_canonical() {
        return Err(GatewayConsensusError::InvalidAlgorithm);
    }
    let key_id = signer.key_id();
    if key_id.trim().is_empty() {
        return Err(GatewayConsensusError::EmptyKeyId);
    }
    if key_id != key_id.trim() {
        return Err(GatewayConsensusError::NonCanonicalKeyId);
    }
    let endorsement_digest = digest_gateway_endorsement(&endorsement)?;
    let message = endorsement_signature_message(endorsement_digest);
    let signature = signer
        .sign_gateway_endorsement(&message)
        .map_err(GatewayConsensusError::Signing)?;
    if signature.is_empty() {
        return Err(GatewayConsensusError::EmptySignature);
    }
    if signature.len() > MAX_GATEWAY_ENDORSEMENT_SIGNATURE_BYTES {
        return Err(GatewayConsensusError::SignatureTooLarge {
            actual: signature.len(),
            maximum: MAX_GATEWAY_ENDORSEMENT_SIGNATURE_BYTES,
        });
    }
    Ok(SignedGatewayEndorsement {
        schema_version: SIGNED_GATEWAY_ENDORSEMENT_SCHEMA.into(),
        endorsement,
        endorsement_digest,
        signature: DetachedSignature {
            algorithm,
            key_id: key_id.to_string(),
            signature,
        },
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayConsensusPolicy {
    pub minimum_distinct_gateways: usize,
    pub maximum_endorsements: usize,
    pub require_algorithm_diversity: bool,
    pub required_gateway_ids: BTreeSet<String>,
    pub allowed_gateway_ids: Option<BTreeSet<String>>,
}

impl Default for GatewayConsensusPolicy {
    fn default() -> Self {
        Self {
            minimum_distinct_gateways: 1,
            maximum_endorsements: MAX_GATEWAY_ENDORSEMENTS,
            require_algorithm_diversity: false,
            required_gateway_ids: BTreeSet::new(),
            allowed_gateway_ids: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayConsensusViolation {
    InvalidPolicy,
    TooManyEndorsements {
        actual: usize,
        maximum: usize,
    },
    UnsupportedSchema,
    InvalidEndorsement(GatewayConsensusError),
    EndorsementDigestMismatch(String),
    StateDigestMismatch(String),
    StateGenerationMismatch(String),
    PreviousStateMismatch(String),
    StateCommitTimeMismatch(String),
    NotYetValid(String),
    Expired(String),
    GatewayNotAllowed(String),
    DuplicateGateway(String),
    DuplicateSigner {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
    SignatureTooLarge {
        gateway_id: String,
        actual: usize,
        maximum: usize,
    },
    SignerUnknown(String),
    SignerNotYetValid(String),
    SignerExpired(String),
    SignerRetired(String),
    SignerRevoked(String),
    SignerUsageNotAllowed(String),
    InvalidSignature(String),
    VerificationProviderError {
        gateway_id: String,
        reason: String,
    },
    InsufficientDistinctGateways {
        actual: usize,
        required: usize,
    },
    MissingRequiredGateway(String),
    MissingAlgorithmDiversity,
    TrustSnapshotInvalid(String),
    TrustSnapshotStale,
}

#[derive(Debug, Clone)]
pub struct VerifiedGatewayConsensus {
    state_digest: Sha256Digest,
    generation: u64,
    consensus_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    gateways: Vec<String>,
    signers: Vec<(SignatureAlgorithm, String)>,
}

impl VerifiedGatewayConsensus {
    pub fn state_digest(&self) -> Sha256Digest {
        self.state_digest
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn consensus_digest(&self) -> Sha256Digest {
        self.consensus_digest
    }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }
    pub fn gateways(&self) -> &[String] {
        &self.gateways
    }
    pub fn signers(&self) -> &[(SignatureAlgorithm, String)] {
        &self.signers
    }
}

pub fn verify_gateway_consensus(
    state: &FabricationGatewayState,
    endorsements: &[SignedGatewayEndorsement],
    policy: &GatewayConsensusPolicy,
    trust_snapshot: &TrustSnapshot,
    now_unix_ms: u64,
    verifier: &dyn GatewayEndorsementVerifier,
) -> Result<VerifiedGatewayConsensus, Vec<GatewayConsensusViolation>> {
    let mut violations = Vec::new();
    if policy.minimum_distinct_gateways == 0
        || policy.maximum_endorsements == 0
        || policy.minimum_distinct_gateways > policy.maximum_endorsements
        || policy
            .required_gateway_ids
            .iter()
            .any(|id| validate_gateway_id(id).is_err())
        || policy
            .allowed_gateway_ids
            .as_ref()
            .is_some_and(|ids| ids.iter().any(|id| validate_gateway_id(id).is_err()))
    {
        violations.push(GatewayConsensusViolation::InvalidPolicy);
    }
    if endorsements.len() > policy.maximum_endorsements {
        violations.push(GatewayConsensusViolation::TooManyEndorsements {
            actual: endorsements.len(),
            maximum: policy.maximum_endorsements,
        });
    }
    let state_digest = match state.digest() {
        Ok(digest) => digest,
        Err(error) => {
            violations.push(GatewayConsensusViolation::InvalidEndorsement(
                GatewayConsensusError::InvalidState(format!("{error:?}")),
            ));
            Sha256Digest([0; 32])
        }
    };
    if trust_snapshot.validate().is_err() {
        violations.push(GatewayConsensusViolation::TrustSnapshotInvalid(
            "snapshot validation failed".into(),
        ));
    }
    let now_unix_s = now_unix_ms / 1_000;
    if !trust_snapshot.is_fresh_at(now_unix_s) {
        violations.push(GatewayConsensusViolation::TrustSnapshotStale);
    }

    let mut seen_gateways = BTreeSet::new();
    let mut valid_gateways = BTreeSet::new();
    let mut signer_ids = BTreeSet::new();
    let mut algorithms = BTreeSet::new();
    let mut valid_signers = Vec::new();
    let mut valid_digests = Vec::new();
    for signed in endorsements {
        let gateway_id = signed.endorsement.gateway_id.clone();
        if signed.schema_version != SIGNED_GATEWAY_ENDORSEMENT_SCHEMA {
            violations.push(GatewayConsensusViolation::UnsupportedSchema);
            continue;
        }
        if let Err(error) = signed.endorsement.validate() {
            violations.push(GatewayConsensusViolation::InvalidEndorsement(error));
            continue;
        }
        match digest_gateway_endorsement(&signed.endorsement) {
            Ok(digest) if digest != signed.endorsement_digest => {
                violations.push(GatewayConsensusViolation::EndorsementDigestMismatch(
                    gateway_id.clone(),
                ));
                continue;
            }
            Err(error) => {
                violations.push(GatewayConsensusViolation::InvalidEndorsement(error));
                continue;
            }
            Ok(_) => {}
        }
        if signed.endorsement.gateway_state_digest != state_digest {
            violations.push(GatewayConsensusViolation::StateDigestMismatch(
                gateway_id.clone(),
            ));
        }
        if signed.endorsement.gateway_generation != state.generation {
            violations.push(GatewayConsensusViolation::StateGenerationMismatch(
                gateway_id.clone(),
            ));
        }
        if signed.endorsement.previous_gateway_state_digest != state.previous_state_digest {
            violations.push(GatewayConsensusViolation::PreviousStateMismatch(
                gateway_id.clone(),
            ));
        }
        if signed.endorsement.state_committed_at_unix_ms != state.committed_at_unix_ms {
            violations.push(GatewayConsensusViolation::StateCommitTimeMismatch(
                gateway_id.clone(),
            ));
        }
        if now_unix_ms < signed.endorsement.issued_at_unix_ms {
            violations.push(GatewayConsensusViolation::NotYetValid(gateway_id.clone()));
        }
        if now_unix_ms >= signed.endorsement.expires_at_unix_ms {
            violations.push(GatewayConsensusViolation::Expired(gateway_id.clone()));
        }
        if policy
            .allowed_gateway_ids
            .as_ref()
            .is_some_and(|allowed| !allowed.contains(&gateway_id))
        {
            violations.push(GatewayConsensusViolation::GatewayNotAllowed(
                gateway_id.clone(),
            ));
        }
        if !seen_gateways.insert(gateway_id.clone()) {
            violations.push(GatewayConsensusViolation::DuplicateGateway(
                gateway_id.clone(),
            ));
            continue;
        }
        if signed.signature.signature.len() > MAX_GATEWAY_ENDORSEMENT_SIGNATURE_BYTES {
            violations.push(GatewayConsensusViolation::SignatureTooLarge {
                gateway_id: gateway_id.clone(),
                actual: signed.signature.signature.len(),
                maximum: MAX_GATEWAY_ENDORSEMENT_SIGNATURE_BYTES,
            });
            continue;
        }
        let signer_id = (
            signed.signature.algorithm.clone(),
            signed.signature.key_id.clone(),
        );
        if !signer_ids.insert(signer_id.clone()) {
            violations.push(GatewayConsensusViolation::DuplicateSigner {
                algorithm: signer_id.0,
                key_id: signer_id.1,
            });
            continue;
        }
        match trust_snapshot.key_eligibility(
            &signed.signature.algorithm,
            &signed.signature.key_id,
            KeyUsage::GatewayConsensus,
            now_unix_s,
        ) {
            KeyEligibility::Eligible => {}
            KeyEligibility::Unknown => {
                violations.push(GatewayConsensusViolation::SignerUnknown(
                    signed.signature.key_id.clone(),
                ));
                continue;
            }
            KeyEligibility::NotYetValid => {
                violations.push(GatewayConsensusViolation::SignerNotYetValid(
                    signed.signature.key_id.clone(),
                ));
                continue;
            }
            KeyEligibility::Expired => {
                violations.push(GatewayConsensusViolation::SignerExpired(
                    signed.signature.key_id.clone(),
                ));
                continue;
            }
            KeyEligibility::Retired => {
                violations.push(GatewayConsensusViolation::SignerRetired(
                    signed.signature.key_id.clone(),
                ));
                continue;
            }
            KeyEligibility::Revoked => {
                violations.push(GatewayConsensusViolation::SignerRevoked(
                    signed.signature.key_id.clone(),
                ));
                continue;
            }
            KeyEligibility::UsageNotAllowed => {
                violations.push(GatewayConsensusViolation::SignerUsageNotAllowed(
                    signed.signature.key_id.clone(),
                ));
                continue;
            }
        }
        let message = endorsement_signature_message(signed.endorsement_digest);
        match verifier.verify_gateway_endorsement(
            &signed.signature.algorithm,
            &signed.signature.key_id,
            &message,
            &signed.signature.signature,
        ) {
            Ok(true) => {
                algorithms.insert(signed.signature.algorithm.clone());
                valid_gateways.insert(gateway_id);
                valid_signers.push(signer_id);
                valid_digests.push(signed.endorsement_digest);
            }
            Ok(false) => violations.push(GatewayConsensusViolation::InvalidSignature(gateway_id)),
            Err(reason) => violations
                .push(GatewayConsensusViolation::VerificationProviderError { gateway_id, reason }),
        }
    }
    if valid_gateways.len() < policy.minimum_distinct_gateways {
        violations.push(GatewayConsensusViolation::InsufficientDistinctGateways {
            actual: valid_gateways.len(),
            required: policy.minimum_distinct_gateways,
        });
    }
    for required in &policy.required_gateway_ids {
        if !valid_gateways.contains(required) {
            violations.push(GatewayConsensusViolation::MissingRequiredGateway(
                required.clone(),
            ));
        }
    }
    if policy.require_algorithm_diversity && algorithms.len() < 2 {
        violations.push(GatewayConsensusViolation::MissingAlgorithmDiversity);
    }
    let trust_snapshot_digest = digest_trust_snapshot(trust_snapshot).map_err(|error| {
        vec![GatewayConsensusViolation::TrustSnapshotInvalid(format!(
            "{error:?}"
        ))]
    })?;
    if !violations.is_empty() {
        return Err(violations);
    }
    valid_digests.sort();
    let consensus_digest = digest_consensus_set(state_digest, &valid_digests);
    Ok(VerifiedGatewayConsensus {
        state_digest,
        generation: state.generation,
        consensus_digest,
        trust_snapshot_digest,
        gateways: valid_gateways.into_iter().collect(),
        signers: valid_signers,
    })
}

fn endorsement_signature_message(digest: Sha256Digest) -> Vec<u8> {
    let mut message = b"symthaea.fabrication.gateway-endorsement-signature.v1\0".to_vec();
    message.extend_from_slice(&digest.0);
    message
}

fn digest_consensus_set(
    state_digest: Sha256Digest,
    endorsement_digests: &[Sha256Digest],
) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.gateway-consensus-digest.v1\0");
    hasher.update(&state_digest.0);
    hasher.update(&(endorsement_digests.len() as u64).to_le_bytes());
    for digest in endorsement_digests {
        hasher.update(&digest.0);
    }
    hasher.finalize()
}

fn validate_gateway_id(value: &str) -> Result<(), GatewayConsensusError> {
    if value.trim().is_empty() {
        return Err(GatewayConsensusError::EmptyGatewayId);
    }
    if value != value.trim() {
        return Err(GatewayConsensusError::NonCanonicalGatewayId);
    }
    if value.len() > MAX_GATEWAY_ID_BYTES {
        return Err(GatewayConsensusError::GatewayIdTooLong {
            actual: value.len(),
            maximum: MAX_GATEWAY_ID_BYTES,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::AuditJournal;
    use crate::crypto_digest::sha256;
    use crate::gateway_consensus_tracker::GatewayConsensusTracker;
    use crate::gateway_state::FabricationGatewayState;
    use crate::incident_ledger::IncidentLedger;
    use crate::operator_command_tracker::OperatorCommandTracker;
    use crate::session::MachineSessionTracker;
    use crate::submission_ledger::SubmissionLedger;
    use crate::telemetry_tracker::MachineTelemetryTracker;
    use crate::trust::{KeyLifecycleStatus, KeyTrustRecord};

    struct Provider {
        algorithm: SignatureAlgorithm,
        key_id: &'static str,
    }
    impl GatewayEndorsementSigner for Provider {
        fn algorithm(&self) -> SignatureAlgorithm {
            self.algorithm.clone()
        }
        fn key_id(&self) -> &str {
            self.key_id
        }
        fn sign_gateway_endorsement(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            Ok(sha256(message).0.to_vec())
        }
    }
    impl GatewayEndorsementVerifier for Provider {
        fn verify_gateway_endorsement(
            &self,
            _algorithm: &SignatureAlgorithm,
            _key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            Ok(signature == sha256(message).0.as_slice())
        }
    }

    fn state() -> FabricationGatewayState {
        let trust = TrustSnapshot::new(
            1,
            100,
            1_000,
            vec![KeyTrustRecord {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "gateway-a-key".into(),
                not_before_unix_s: 100,
                not_after_unix_s: None,
                status: KeyLifecycleStatus::Active,
                usages: BTreeSet::from([KeyUsage::GatewayConsensus]),
            }],
        )
        .unwrap();
        FabricationGatewayState::genesis(
            500_000,
            trust,
            AuditJournal::default(),
            MachineSessionTracker::default(),
            MachineTelemetryTracker::default(),
            SubmissionLedger::default(),
            OperatorCommandTracker::default(),
            GatewayConsensusTracker::default(),
            IncidentLedger::default(),
        )
        .unwrap()
    }

    #[test]
    fn exact_state_endorsement_reaches_quorum() {
        let state = state();
        let provider = Provider {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "gateway-a-key",
        };
        let endorsement =
            endorse_gateway_state(&state, "gateway-a", 500_000, 510_000, &provider).unwrap();
        let verified = verify_gateway_consensus(
            &state,
            &[endorsement],
            &GatewayConsensusPolicy::default(),
            &state.trust_snapshot,
            501_000,
            &provider,
        )
        .unwrap();
        assert_eq!(verified.generation(), 1);
    }

    #[test]
    fn divergent_state_digest_is_rejected() {
        let state = state();
        let provider = Provider {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "gateway-a-key",
        };
        let mut endorsement =
            endorse_gateway_state(&state, "gateway-a", 500_000, 510_000, &provider).unwrap();
        endorsement.endorsement.gateway_state_digest = sha256(b"fork");
        endorsement.endorsement_digest =
            digest_gateway_endorsement(&endorsement.endorsement).unwrap();
        let message = endorsement_signature_message(endorsement.endorsement_digest);
        endorsement.signature.signature = sha256(&message).0.to_vec();
        let violations = verify_gateway_consensus(
            &state,
            &[endorsement],
            &GatewayConsensusPolicy::default(),
            &state.trust_snapshot,
            501_000,
            &provider,
        )
        .unwrap_err();
        assert!(violations.iter().any(|violation| matches!(
            violation,
            GatewayConsensusViolation::StateDigestMismatch(_)
        )));
    }
}
