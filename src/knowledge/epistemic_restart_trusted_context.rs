// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Joint trusted context for read-only epistemic restart admission.
//!
//! EKM-051 accepts a trusted anchor tracker and trusted verifier checkpoint as two
//! independent inputs. EKM-052 binds those inputs into one immutable trust epoch so
//! a caller cannot accidentally pair checkpoints from different deployment states.
//!
//! This module remains read-only: it does not persist, advance, hydrate, or activate
//! any trusted or epistemic state.

use super::epistemic_restart_anchor::{
    RestartAnchorDigestV1, RestartAnchorEvidenceV1, RestartAnchorTrackerV1,
};
use super::epistemic_restart_anchor_policy::RestartAnchorTrustPolicyV1;
use super::epistemic_restart_quarantine_admission::{
    AdmittedEpistemicRestartQuarantineV2, EpistemicRestartQuarantineAdmissionDigest,
    EpistemicRestartQuarantineAdmissionError,
};
use super::epistemic_restart_validation_receipt::EpistemicRestartValidationReceiptDigest;
use super::epistemic_restart_verifier_continuity::TrustedRestartVerifierStateV1;
use super::epistemic_restart_verifier_provenance::{
    ProfiledRestartAnchorEvidenceVerifierV1, RestartVerifierProfileDigestV1,
    RestartVerifierProvenanceDigestV1,
};
use super::epistemic_restart_wire_v2::EpistemicRestartWireSnapshotV2;
use std::error::Error;
use std::fmt;

pub const MAX_RESTART_DEPLOYMENT_ID_BYTES: usize = 256;
pub const MAX_RESTART_TRUST_DOMAIN_ID_BYTES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointTrustedRestartContextVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JointTrustedRestartContextDigestV1([u8; 32]);

impl JointTrustedRestartContextDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            use std::fmt::Write as _;
            write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
        }
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContextualizedRestartQuarantineDigestV1([u8; 32]);

impl ContextualizedRestartQuarantineDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Immutable pairing of the trusted restart-anchor state and trusted verifier state.
#[derive(Debug, Clone)]
pub struct JointTrustedRestartContextV1 {
    version: JointTrustedRestartContextVersion,
    deployment_id: String,
    trust_domain_id: String,
    committed_at_cycle: u64,
    anchor_tracker: RestartAnchorTrackerV1,
    verifier_state: TrustedRestartVerifierStateV1,
    anchor_sequence: u64,
    anchor_digest: RestartAnchorDigestV1,
    anchor_receipt_digest: EpistemicRestartValidationReceiptDigest,
    anchor_capture_cycle: u64,
    verifier_trust_snapshot_sequence: u64,
    verifier_profile_digest: RestartVerifierProfileDigestV1,
    verifier_provenance_digest: RestartVerifierProvenanceDigestV1,
    context_digest: JointTrustedRestartContextDigestV1,
}

impl JointTrustedRestartContextV1 {
    pub fn capture(
        deployment_id: impl Into<String>,
        trust_domain_id: impl Into<String>,
        committed_at_cycle: u64,
        anchor_tracker: &RestartAnchorTrackerV1,
        verifier_state: &TrustedRestartVerifierStateV1,
    ) -> Result<Self, JointTrustedRestartContextError> {
        let deployment_id = deployment_id.into();
        let trust_domain_id = trust_domain_id.into();
        validate_identifier(
            &deployment_id,
            MAX_RESTART_DEPLOYMENT_ID_BYTES,
            JointTrustedRestartContextField::DeploymentId,
        )?;
        validate_identifier(
            &trust_domain_id,
            MAX_RESTART_TRUST_DOMAIN_ID_BYTES,
            JointTrustedRestartContextField::TrustDomainId,
        )?;
        let anchor_sequence = anchor_tracker
            .latest_sequence()
            .ok_or(JointTrustedRestartContextError::UninitializedAnchorTracker)?;
        let anchor_digest = anchor_tracker
            .latest_anchor_digest()
            .ok_or(JointTrustedRestartContextError::UninitializedAnchorTracker)?;
        let anchor_receipt_digest = anchor_tracker
            .latest_receipt_digest()
            .ok_or(JointTrustedRestartContextError::UninitializedAnchorTracker)?;
        let anchor_capture_cycle = anchor_tracker
            .latest_captured_at_cycle()
            .ok_or(JointTrustedRestartContextError::UninitializedAnchorTracker)?;
        if committed_at_cycle < anchor_capture_cycle {
            return Err(JointTrustedRestartContextError::CommitPredatesAnchor {
                committed_at_cycle,
                anchor_capture_cycle,
            });
        }
        let verifier_trust_snapshot_sequence = verifier_state.trust_snapshot_sequence();
        if verifier_trust_snapshot_sequence == 0 {
            return Err(JointTrustedRestartContextError::InvalidVerifierSequence);
        }

        let mut context = Self {
            version: JointTrustedRestartContextVersion::V1,
            deployment_id,
            trust_domain_id,
            committed_at_cycle,
            anchor_tracker: anchor_tracker.clone(),
            verifier_state: verifier_state.clone(),
            anchor_sequence,
            anchor_digest,
            anchor_receipt_digest,
            anchor_capture_cycle,
            verifier_trust_snapshot_sequence,
            verifier_profile_digest: verifier_state.profile_digest(),
            verifier_provenance_digest: verifier_state.provenance_digest(),
            context_digest: JointTrustedRestartContextDigestV1([0; 32]),
        };
        context.context_digest = digest_context(&context)?;
        context.verify_integrity()?;
        Ok(context)
    }

    pub fn version(&self) -> JointTrustedRestartContextVersion {
        self.version
    }

    pub fn deployment_id(&self) -> &str {
        &self.deployment_id
    }

    pub fn trust_domain_id(&self) -> &str {
        &self.trust_domain_id
    }

    pub fn committed_at_cycle(&self) -> u64 {
        self.committed_at_cycle
    }

    pub fn anchor_sequence(&self) -> u64 {
        self.anchor_sequence
    }

    pub fn anchor_capture_cycle(&self) -> u64 {
        self.anchor_capture_cycle
    }

    pub fn verifier_trust_snapshot_sequence(&self) -> u64 {
        self.verifier_trust_snapshot_sequence
    }

    pub fn context_digest(&self) -> JointTrustedRestartContextDigestV1 {
        self.context_digest
    }

    pub fn verify_integrity(&self) -> Result<(), JointTrustedRestartContextError> {
        if self.anchor_tracker.latest_sequence() != Some(self.anchor_sequence)
            || self.anchor_tracker.latest_anchor_digest() != Some(self.anchor_digest)
            || self.anchor_tracker.latest_receipt_digest() != Some(self.anchor_receipt_digest)
            || self.anchor_tracker.latest_captured_at_cycle() != Some(self.anchor_capture_cycle)
        {
            return Err(JointTrustedRestartContextError::AnchorSnapshotMismatch);
        }
        if self.verifier_state.trust_snapshot_sequence() != self.verifier_trust_snapshot_sequence
            || self.verifier_state.profile_digest() != self.verifier_profile_digest
            || self.verifier_state.provenance_digest() != self.verifier_provenance_digest
        {
            return Err(JointTrustedRestartContextError::VerifierSnapshotMismatch);
        }
        if self.committed_at_cycle < self.anchor_capture_cycle {
            return Err(JointTrustedRestartContextError::CommitPredatesAnchor {
                committed_at_cycle: self.committed_at_cycle,
                anchor_capture_cycle: self.anchor_capture_cycle,
            });
        }
        if digest_context(self)? != self.context_digest {
            return Err(JointTrustedRestartContextError::ContextDigestMismatch);
        }
        Ok(())
    }

    fn anchor_tracker(&self) -> &RestartAnchorTrackerV1 {
        &self.anchor_tracker
    }

    fn verifier_state(&self) -> &TrustedRestartVerifierStateV1 {
        &self.verifier_state
    }
}

#[derive(Debug, Clone)]
pub struct ContextualizedEpistemicRestartQuarantineV2 {
    quarantine: AdmittedEpistemicRestartQuarantineV2,
    trusted_context_digest: JointTrustedRestartContextDigestV1,
    reviewed_at_cycle: u64,
    trusted_context_mutated: bool,
    writable_hydration_authorized: bool,
    activation_authorized: bool,
    contextualized_digest: ContextualizedRestartQuarantineDigestV1,
}

impl ContextualizedEpistemicRestartQuarantineV2 {
    #[allow(clippy::too_many_arguments)]
    pub fn validate_and_construct(
        snapshot: &EpistemicRestartWireSnapshotV2,
        policy: &RestartAnchorTrustPolicyV1,
        candidate_anchor_evidence: &RestartAnchorEvidenceV1,
        verifier: &dyn ProfiledRestartAnchorEvidenceVerifierV1,
        trusted_context: &JointTrustedRestartContextV1,
        observed_at_cycle: u64,
    ) -> Result<Self, JointTrustedRestartContextError> {
        trusted_context.verify_integrity()?;
        if observed_at_cycle < trusted_context.committed_at_cycle {
            return Err(JointTrustedRestartContextError::ObservationPredatesContext {
                observed_at_cycle,
                committed_at_cycle: trusted_context.committed_at_cycle,
            });
        }
        let quarantine = AdmittedEpistemicRestartQuarantineV2::validate_and_construct(
            snapshot,
            policy,
            candidate_anchor_evidence,
            verifier,
            trusted_context.anchor_tracker(),
            trusted_context.verifier_state(),
            observed_at_cycle,
        )
        .map_err(JointTrustedRestartContextError::QuarantineAdmission)?;

        if quarantine.trusted_anchor_sequence_before_review() != trusted_context.anchor_sequence
            || quarantine.trusted_verifier_sequence_before_review()
                != trusted_context.verifier_trust_snapshot_sequence
        {
            return Err(JointTrustedRestartContextError::QuarantineContextMismatch);
        }

        let mut out = Self {
            quarantine,
            trusted_context_digest: trusted_context.context_digest,
            reviewed_at_cycle: observed_at_cycle,
            trusted_context_mutated: false,
            writable_hydration_authorized: false,
            activation_authorized: false,
            contextualized_digest: ContextualizedRestartQuarantineDigestV1([0; 32]),
        };
        out.contextualized_digest = digest_contextualized_quarantine(&out);
        out.verify_read_only()?;
        Ok(out)
    }

    pub fn quarantine(&self) -> &AdmittedEpistemicRestartQuarantineV2 {
        &self.quarantine
    }

    pub fn trusted_context_digest(&self) -> JointTrustedRestartContextDigestV1 {
        self.trusted_context_digest
    }

    pub fn reviewed_at_cycle(&self) -> u64 {
        self.reviewed_at_cycle
    }

    pub fn trusted_context_mutated(&self) -> bool {
        self.trusted_context_mutated
    }

    pub fn writable_hydration_authorized(&self) -> bool {
        self.writable_hydration_authorized
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }

    pub fn contextualized_digest(&self) -> ContextualizedRestartQuarantineDigestV1 {
        self.contextualized_digest
    }

    pub fn verify_read_only(&self) -> Result<(), JointTrustedRestartContextError> {
        self.quarantine
            .verify_read_only()
            .map_err(JointTrustedRestartContextError::QuarantineAdmission)?;
        if self.trusted_context_mutated
            || self.writable_hydration_authorized
            || self.activation_authorized
        {
            return Err(JointTrustedRestartContextError::UnexpectedAuthority);
        }
        if digest_contextualized_quarantine(self) != self.contextualized_digest {
            return Err(JointTrustedRestartContextError::ContextualizedDigestMismatch);
        }
        Ok(())
    }
}

fn digest_context(
    context: &JointTrustedRestartContextV1,
) -> Result<JointTrustedRestartContextDigestV1, JointTrustedRestartContextError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-joint-trusted-restart-context-v1");
    hasher.update(&[1]);
    hash_bytes(&mut hasher, context.deployment_id.as_bytes())?;
    hash_bytes(&mut hasher, context.trust_domain_id.as_bytes())?;
    hasher.update(&context.committed_at_cycle.to_le_bytes());
    hasher.update(&context.anchor_sequence.to_le_bytes());
    hasher.update(&context.anchor_digest.as_bytes());
    hasher.update(&context.anchor_receipt_digest.as_bytes());
    hasher.update(&context.anchor_capture_cycle.to_le_bytes());
    hasher.update(&context.verifier_trust_snapshot_sequence.to_le_bytes());
    hasher.update(&context.verifier_profile_digest.as_bytes());
    hasher.update(&context.verifier_provenance_digest.as_bytes());
    Ok(JointTrustedRestartContextDigestV1(*hasher.finalize().as_bytes()))
}

fn digest_contextualized_quarantine(
    quarantine: &ContextualizedEpistemicRestartQuarantineV2,
) -> ContextualizedRestartQuarantineDigestV1 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-contextualized-restart-quarantine-v1");
    hasher.update(&quarantine.quarantine.admission_digest().as_bytes());
    hasher.update(&quarantine.trusted_context_digest.as_bytes());
    hasher.update(&quarantine.reviewed_at_cycle.to_le_bytes());
    hasher.update(&[u8::from(quarantine.trusted_context_mutated)]);
    hasher.update(&[u8::from(quarantine.writable_hydration_authorized)]);
    hasher.update(&[u8::from(quarantine.activation_authorized)]);
    ContextualizedRestartQuarantineDigestV1(*hasher.finalize().as_bytes())
}

fn validate_identifier(
    value: &str,
    maximum: usize,
    field: JointTrustedRestartContextField,
) -> Result<(), JointTrustedRestartContextError> {
    if value.trim().is_empty() || value != value.trim() || value.len() > maximum {
        return Err(JointTrustedRestartContextError::InvalidIdentifier {
            field,
            actual_length: value.len(),
            maximum,
        });
    }
    Ok(())
}

fn hash_bytes(
    hasher: &mut blake3::Hasher,
    bytes: &[u8],
) -> Result<(), JointTrustedRestartContextError> {
    let length = u64::try_from(bytes.len()).map_err(|_| JointTrustedRestartContextError::LengthOverflow)?;
    hasher.update(&length.to_le_bytes());
    hasher.update(bytes);
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointTrustedRestartContextField {
    DeploymentId,
    TrustDomainId,
}

#[derive(Debug, Clone, PartialEq)]
pub enum JointTrustedRestartContextError {
    InvalidIdentifier {
        field: JointTrustedRestartContextField,
        actual_length: usize,
        maximum: usize,
    },
    UninitializedAnchorTracker,
    InvalidVerifierSequence,
    CommitPredatesAnchor {
        committed_at_cycle: u64,
        anchor_capture_cycle: u64,
    },
    ObservationPredatesContext {
        observed_at_cycle: u64,
        committed_at_cycle: u64,
    },
    AnchorSnapshotMismatch,
    VerifierSnapshotMismatch,
    ContextDigestMismatch,
    QuarantineContextMismatch,
    QuarantineAdmission(EpistemicRestartQuarantineAdmissionError),
    UnexpectedAuthority,
    ContextualizedDigestMismatch,
    LengthOverflow,
}

impl fmt::Display for JointTrustedRestartContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "joint trusted restart context rejected: {self:?}")
    }
}

impl Error for JointTrustedRestartContextError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_and_contextualized_quarantine_types_expose_no_default_authority() {
        fn context_type(_: Option<&JointTrustedRestartContextV1>) {}
        fn quarantine_type(_: Option<&ContextualizedEpistemicRestartQuarantineV2>) {}
        context_type(None);
        quarantine_type(None);
    }
}
