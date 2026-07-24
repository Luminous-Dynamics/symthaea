// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Immutable, signed incident evidence bundles.

use crate::attestation::{DetachedSignature, SignatureAlgorithm};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::gateway_state::{GatewayStateEnvelope, GatewayStateError};
use crate::trust::{KeyEligibility, KeyUsage, TrustSnapshot, digest_trust_snapshot};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const INCIDENT_BUNDLE_SCHEMA: &str = "symthaea.fabrication.incident-bundle.v1";
pub const SIGNED_INCIDENT_BUNDLE_SCHEMA: &str = "symthaea.fabrication.signed-incident-bundle.v1";
pub const MAX_INCIDENT_ID_BYTES: usize = 256;
pub const MAX_INCIDENT_SUMMARY_BYTES: usize = 4096;
pub const MAX_INCIDENT_SIGNATURES: usize = 16;
pub const MAX_INCIDENT_SIGNATURE_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IncidentKind {
    OperatorPause,
    OperatorCancel,
    EmergencyStop,
    RuntimeContainment,
    SubmissionUncertain,
    SubmissionReconciliationFailure,
    GatewayDivergence,
    RecoveryFailure,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IncidentBundle {
    pub schema_version: String,
    pub incident_id: String,
    pub occurred_at_unix_ms: u64,
    pub kind: IncidentKind,
    pub summary: String,
    pub manifest_digest: Option<Sha256Digest>,
    pub machine_id: Option<String>,
    pub session_digest: Option<Sha256Digest>,
    pub printer_job_id: Option<String>,
    pub trigger_digest: Sha256Digest,
    pub gateway_state: GatewayStateEnvelope,
    pub gateway_state_digest: Sha256Digest,
    pub audit_journal_digest: Sha256Digest,
    pub submission_ledger_digest: Sha256Digest,
    pub telemetry_tracker_digest: Sha256Digest,
    pub operator_command_tracker_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedIncidentBundle {
    pub schema_version: String,
    pub bundle: IncidentBundle,
    pub bundle_digest: Sha256Digest,
    pub signatures: Vec<DetachedSignature>,
}

pub trait IncidentBundleSigner {
    fn algorithm(&self) -> SignatureAlgorithm;
    fn key_id(&self) -> &str;
    fn sign_incident_bundle(&self, message: &[u8]) -> Result<Vec<u8>, String>;
}

pub trait IncidentBundleVerifier {
    fn verify_incident_bundle(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IncidentBundleError {
    UnsupportedSchema,
    UnsupportedSignedSchema,
    EmptyIncidentId,
    NonCanonicalIncidentId,
    IncidentIdTooLong,
    InvalidSummary,
    SummaryTooLong,
    InvalidOptionalIdentifier(&'static str),
    IncidentBeforeGatewayCommit,
    GatewayState(GatewayStateError),
    GatewayDigestMismatch,
    AuditDigestMismatch,
    SubmissionDigestMismatch,
    TelemetryDigestMismatch,
    OperatorCommandDigestMismatch,
    InvalidAlgorithm,
    EmptyKeyId,
    NonCanonicalKeyId,
    EmptySignature,
    SignatureTooLarge {
        actual: usize,
        maximum: usize,
    },
    TooManySignatures {
        actual: usize,
        maximum: usize,
    },
    DuplicateSigner {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
    Signing {
        key_id: String,
        reason: String,
    },
    Encoding(String),
}

impl IncidentBundle {
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        incident_id: impl Into<String>,
        occurred_at_unix_ms: u64,
        kind: IncidentKind,
        summary: impl Into<String>,
        manifest_digest: Option<Sha256Digest>,
        machine_id: Option<String>,
        session_digest: Option<Sha256Digest>,
        printer_job_id: Option<String>,
        trigger_digest: Sha256Digest,
        gateway_state: GatewayStateEnvelope,
    ) -> Result<Self, IncidentBundleError> {
        let evidence = gateway_state
            .state
            .evidence_digests()
            .map_err(IncidentBundleError::GatewayState)?;
        let bundle = Self {
            schema_version: INCIDENT_BUNDLE_SCHEMA.into(),
            incident_id: incident_id.into(),
            occurred_at_unix_ms,
            kind,
            summary: summary.into(),
            manifest_digest,
            machine_id,
            session_digest,
            printer_job_id,
            trigger_digest,
            gateway_state_digest: gateway_state.state_digest,
            audit_journal_digest: evidence.audit_journal,
            submission_ledger_digest: evidence.submission_ledger,
            telemetry_tracker_digest: evidence.telemetry_tracker,
            operator_command_tracker_digest: evidence.operator_command_tracker,
            gateway_state,
        };
        bundle.validate()?;
        Ok(bundle)
    }

    pub fn validate(&self) -> Result<(), IncidentBundleError> {
        if self.schema_version != INCIDENT_BUNDLE_SCHEMA {
            return Err(IncidentBundleError::UnsupportedSchema);
        }
        validate_id(&self.incident_id).map_err(|error| match error {
            IdError::Empty => IncidentBundleError::EmptyIncidentId,
            IdError::NonCanonical => IncidentBundleError::NonCanonicalIncidentId,
            IdError::TooLong => IncidentBundleError::IncidentIdTooLong,
        })?;
        if self.summary.trim().is_empty() || self.summary != self.summary.trim() {
            return Err(IncidentBundleError::InvalidSummary);
        }
        if self.summary.len() > MAX_INCIDENT_SUMMARY_BYTES {
            return Err(IncidentBundleError::SummaryTooLong);
        }
        for (name, value) in [
            ("machine_id", self.machine_id.as_deref()),
            ("printer_job_id", self.printer_job_id.as_deref()),
        ] {
            if value.is_some_and(|value| validate_id(value).is_err()) {
                return Err(IncidentBundleError::InvalidOptionalIdentifier(name));
            }
        }
        let state = self
            .gateway_state
            .clone()
            .open()
            .map_err(IncidentBundleError::GatewayState)?;
        if self.occurred_at_unix_ms < state.committed_at_unix_ms {
            return Err(IncidentBundleError::IncidentBeforeGatewayCommit);
        }
        if self.gateway_state_digest != self.gateway_state.state_digest {
            return Err(IncidentBundleError::GatewayDigestMismatch);
        }
        let evidence = state
            .evidence_digests()
            .map_err(IncidentBundleError::GatewayState)?;
        if self.audit_journal_digest != evidence.audit_journal {
            return Err(IncidentBundleError::AuditDigestMismatch);
        }
        if self.submission_ledger_digest != evidence.submission_ledger {
            return Err(IncidentBundleError::SubmissionDigestMismatch);
        }
        if self.telemetry_tracker_digest != evidence.telemetry_tracker {
            return Err(IncidentBundleError::TelemetryDigestMismatch);
        }
        if self.operator_command_tracker_digest != evidence.operator_command_tracker {
            return Err(IncidentBundleError::OperatorCommandDigestMismatch);
        }
        Ok(())
    }
}

pub fn digest_incident_bundle(
    bundle: &IncidentBundle,
) -> Result<Sha256Digest, IncidentBundleError> {
    bundle.validate()?;
    let bytes = serde_json::to_vec(bundle)
        .map_err(|error| IncidentBundleError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.incident-bundle-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn sign_incident_bundle(
    bundle: IncidentBundle,
    signers: &[&dyn IncidentBundleSigner],
) -> Result<SignedIncidentBundle, IncidentBundleError> {
    bundle.validate()?;
    if signers.len() > MAX_INCIDENT_SIGNATURES {
        return Err(IncidentBundleError::TooManySignatures {
            actual: signers.len(),
            maximum: MAX_INCIDENT_SIGNATURES,
        });
    }
    let bundle_digest = digest_incident_bundle(&bundle)?;
    let message = incident_signature_message(bundle_digest);
    let mut identities = BTreeSet::new();
    let mut signatures = Vec::with_capacity(signers.len());
    for signer in signers {
        let algorithm = signer.algorithm();
        if !algorithm.is_canonical() {
            return Err(IncidentBundleError::InvalidAlgorithm);
        }
        let key_id = signer.key_id();
        if key_id.trim().is_empty() {
            return Err(IncidentBundleError::EmptyKeyId);
        }
        if key_id != key_id.trim() {
            return Err(IncidentBundleError::NonCanonicalKeyId);
        }
        if !identities.insert((algorithm.clone(), key_id.to_string())) {
            return Err(IncidentBundleError::DuplicateSigner {
                algorithm,
                key_id: key_id.to_string(),
            });
        }
        let signature = signer.sign_incident_bundle(&message).map_err(|reason| {
            IncidentBundleError::Signing {
                key_id: key_id.to_string(),
                reason,
            }
        })?;
        if signature.is_empty() {
            return Err(IncidentBundleError::EmptySignature);
        }
        if signature.len() > MAX_INCIDENT_SIGNATURE_BYTES {
            return Err(IncidentBundleError::SignatureTooLarge {
                actual: signature.len(),
                maximum: MAX_INCIDENT_SIGNATURE_BYTES,
            });
        }
        signatures.push(DetachedSignature {
            algorithm,
            key_id: key_id.to_string(),
            signature,
        });
    }
    Ok(SignedIncidentBundle {
        schema_version: SIGNED_INCIDENT_BUNDLE_SCHEMA.into(),
        bundle,
        bundle_digest,
        signatures,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IncidentVerificationViolation {
    InvalidBundle(IncidentBundleError),
    UnsupportedSchema,
    DigestMismatch,
    TooManySignatures,
    DuplicateSigner(String),
    SignatureTooLarge {
        key_id: String,
        actual: usize,
        maximum: usize,
    },
    TrustSnapshotInvalid,
    TrustSnapshotStale,
    SignerIneligible(String),
    InvalidSignature(String),
    VerificationProviderError {
        key_id: String,
        reason: String,
    },
    InsufficientSignatures {
        actual: usize,
        required: usize,
    },
}

#[derive(Debug, Clone)]
pub struct VerifiedIncidentBundle {
    signed: SignedIncidentBundle,
    valid_signers: Vec<(SignatureAlgorithm, String)>,
    trust_snapshot_digest: Sha256Digest,
}

impl VerifiedIncidentBundle {
    pub fn bundle(&self) -> &IncidentBundle {
        &self.signed.bundle
    }
    pub fn bundle_digest(&self) -> Sha256Digest {
        self.signed.bundle_digest
    }
    pub fn valid_signers(&self) -> &[(SignatureAlgorithm, String)] {
        &self.valid_signers
    }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }
}

pub fn verify_incident_bundle(
    signed: SignedIncidentBundle,
    trust_snapshot: &TrustSnapshot,
    evaluation_time_unix_s: u64,
    minimum_valid_signatures: usize,
    verifier: &dyn IncidentBundleVerifier,
) -> Result<VerifiedIncidentBundle, Vec<IncidentVerificationViolation>> {
    let mut violations = Vec::new();
    if signed.schema_version != SIGNED_INCIDENT_BUNDLE_SCHEMA {
        violations.push(IncidentVerificationViolation::UnsupportedSchema);
    }
    if let Err(error) = signed.bundle.validate() {
        violations.push(IncidentVerificationViolation::InvalidBundle(error));
    }
    match digest_incident_bundle(&signed.bundle) {
        Ok(digest) if digest != signed.bundle_digest => {
            violations.push(IncidentVerificationViolation::DigestMismatch)
        }
        Err(error) => violations.push(IncidentVerificationViolation::InvalidBundle(error)),
        Ok(_) => {}
    }
    if signed.signatures.len() > MAX_INCIDENT_SIGNATURES {
        violations.push(IncidentVerificationViolation::TooManySignatures);
    }
    if trust_snapshot.validate().is_err() {
        violations.push(IncidentVerificationViolation::TrustSnapshotInvalid);
    }
    if !trust_snapshot.is_fresh_at(evaluation_time_unix_s) {
        violations.push(IncidentVerificationViolation::TrustSnapshotStale);
    }
    let message = incident_signature_message(signed.bundle_digest);
    let mut seen = BTreeSet::new();
    let mut valid_signers = Vec::new();
    for signature in &signed.signatures {
        if signature.signature.len() > MAX_INCIDENT_SIGNATURE_BYTES {
            violations.push(IncidentVerificationViolation::SignatureTooLarge {
                key_id: signature.key_id.clone(),
                actual: signature.signature.len(),
                maximum: MAX_INCIDENT_SIGNATURE_BYTES,
            });
            continue;
        }
        let identity = (signature.algorithm.clone(), signature.key_id.clone());
        if !seen.insert(identity.clone()) {
            violations.push(IncidentVerificationViolation::DuplicateSigner(
                signature.key_id.clone(),
            ));
            continue;
        }
        if trust_snapshot.key_eligibility(
            &signature.algorithm,
            &signature.key_id,
            KeyUsage::IncidentEvidence,
            evaluation_time_unix_s,
        ) != KeyEligibility::Eligible
        {
            violations.push(IncidentVerificationViolation::SignerIneligible(
                signature.key_id.clone(),
            ));
            continue;
        }
        match verifier.verify_incident_bundle(
            &signature.algorithm,
            &signature.key_id,
            &message,
            &signature.signature,
        ) {
            Ok(true) => valid_signers.push(identity),
            Ok(false) => violations.push(IncidentVerificationViolation::InvalidSignature(
                signature.key_id.clone(),
            )),
            Err(reason) => {
                violations.push(IncidentVerificationViolation::VerificationProviderError {
                    key_id: signature.key_id.clone(),
                    reason,
                })
            }
        }
    }
    if valid_signers.len() < minimum_valid_signatures {
        violations.push(IncidentVerificationViolation::InsufficientSignatures {
            actual: valid_signers.len(),
            required: minimum_valid_signatures,
        });
    }
    let trust_snapshot_digest = digest_trust_snapshot(trust_snapshot)
        .map_err(|_| vec![IncidentVerificationViolation::TrustSnapshotInvalid])?;
    if !violations.is_empty() {
        return Err(violations);
    }
    Ok(VerifiedIncidentBundle {
        signed,
        valid_signers,
        trust_snapshot_digest,
    })
}

fn incident_signature_message(digest: Sha256Digest) -> Vec<u8> {
    let mut message = b"symthaea.fabrication.incident-bundle-signature.v1\0".to_vec();
    message.extend_from_slice(&digest.0);
    message
}

enum IdError {
    Empty,
    NonCanonical,
    TooLong,
}
fn validate_id(value: &str) -> Result<(), IdError> {
    if value.trim().is_empty() {
        return Err(IdError::Empty);
    }
    if value != value.trim() {
        return Err(IdError::NonCanonical);
    }
    if value.len() > MAX_INCIDENT_ID_BYTES {
        return Err(IdError::TooLong);
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

    struct Provider;
    impl IncidentBundleSigner for Provider {
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::Ed25519
        }
        fn key_id(&self) -> &str {
            "incident-key"
        }
        fn sign_incident_bundle(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            Ok(sha256(message).0.to_vec())
        }
    }
    impl IncidentBundleVerifier for Provider {
        fn verify_incident_bundle(
            &self,
            _algorithm: &SignatureAlgorithm,
            _key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            Ok(signature == sha256(message).0.as_slice())
        }
    }

    fn state() -> GatewayStateEnvelope {
        GatewayStateEnvelope::seal(
            FabricationGatewayState::genesis(
                500_000,
                TrustSnapshot::new(
                    1,
                    100,
                    1_000,
                    vec![KeyTrustRecord {
                        algorithm: SignatureAlgorithm::Ed25519,
                        key_id: "incident-key".into(),
                        not_before_unix_s: 100,
                        not_after_unix_s: None,
                        status: KeyLifecycleStatus::Active,
                        usages: BTreeSet::from([KeyUsage::IncidentEvidence]),
                    }],
                )
                .unwrap(),
                AuditJournal::default(),
                MachineSessionTracker::default(),
                MachineTelemetryTracker::default(),
                SubmissionLedger::default(),
                OperatorCommandTracker::default(),
                GatewayConsensusTracker::default(),
                IncidentLedger::default(),
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn signed_incident_is_bound_to_gateway_evidence() {
        let envelope = state();
        let bundle = IncidentBundle::capture(
            "incident-1",
            501_000,
            IncidentKind::EmergencyStop,
            "thermal containment triggered",
            Some(sha256(b"manifest")),
            Some("machine".into()),
            Some(sha256(b"session")),
            Some("job".into()),
            sha256(b"trigger"),
            envelope.clone(),
        )
        .unwrap();
        let signed = sign_incident_bundle(bundle, &[&Provider]).unwrap();
        let verified =
            verify_incident_bundle(signed, &envelope.state.trust_snapshot, 501, 1, &Provider)
                .unwrap();
        assert_eq!(verified.bundle().incident_id, "incident-1");
    }
}
