// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sealed public facade for read-only epistemic restart quarantine inspection.
//!
//! EKM-051 and EKM-052 intentionally build rich internal quarantine objects. Those
//! objects retain a cloned ledger and full decoded snapshot so they can re-verify
//! themselves, but exposing those implementation types publicly also exposes a raw
//! cloneable ledger/snapshot surface. EKM-053 closes that capability boundary.
//!
//! Public callers receive only opaque trust/quarantine handles, record-level
//! immutable inspection, stable digests, counts, and verification. There is no raw
//! ledger, raw wire snapshot, inner quarantine, hydration, or activation escape hatch.

use super::claim_evidence::{
    ClaimId, EvidenceId, EvidenceRecord, KnowledgeClaim, ProvenanceId, ProvenanceRecord,
};
use super::epistemic_restart_anchor::{RestartAnchorEvidenceV1, RestartAnchorTrackerV1};
use super::epistemic_restart_anchor_policy::RestartAnchorTrustPolicyV1;
use super::epistemic_restart_trusted_context::{
    ContextualizedEpistemicRestartQuarantineV2, JointTrustedRestartContextV1,
};
use super::epistemic_restart_verifier_continuity::TrustedRestartVerifierStateV1;
use super::epistemic_restart_verifier_provenance::ProfiledRestartAnchorEvidenceVerifierV1;
use super::epistemic_restart_wire_v2::EpistemicRestartWireSnapshotV2;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RestartTrustContextDigestV1([u8; 32]);

impl RestartTrustContextDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        hex32(self.0)
    }

    /// Internal-only constructor for canonical decoders and qualification tests.
    /// Public callers cannot mint a trust-context handle from digest bytes alone.
    pub(crate) fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReadOnlyRestartQuarantineDigestV1([u8; 32]);

impl ReadOnlyRestartQuarantineDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        hex32(self.0)
    }
}

/// Opaque public handle to one jointly bound trusted restart epoch.
///
/// The inner anchor/verifier checkpoints are deliberately not exposed.
#[derive(Debug, Clone)]
pub struct RestartTrustContextHandleV1 {
    inner: JointTrustedRestartContextV1,
    digest: RestartTrustContextDigestV1,
}

impl RestartTrustContextHandleV1 {
    pub fn capture(
        deployment_id: impl Into<String>,
        trust_domain_id: impl Into<String>,
        committed_at_cycle: u64,
        anchor_tracker: &RestartAnchorTrackerV1,
        verifier_state: &TrustedRestartVerifierStateV1,
    ) -> Result<Self, RestartQuarantineFacadeError> {
        let inner = JointTrustedRestartContextV1::capture(
            deployment_id,
            trust_domain_id,
            committed_at_cycle,
            anchor_tracker,
            verifier_state,
        )
        .map_err(|error| RestartQuarantineFacadeError::TrustContextRejected {
            detail: format!("{error:?}"),
        })?;
        let digest = RestartTrustContextDigestV1(inner.context_digest().as_bytes());
        Ok(Self { inner, digest })
    }

    pub fn deployment_id(&self) -> &str {
        self.inner.deployment_id()
    }

    pub fn trust_domain_id(&self) -> &str {
        self.inner.trust_domain_id()
    }

    pub fn committed_at_cycle(&self) -> u64 {
        self.inner.committed_at_cycle()
    }

    pub fn anchor_sequence(&self) -> u64 {
        self.inner.anchor_sequence()
    }

    pub fn anchor_capture_cycle(&self) -> u64 {
        self.inner.anchor_capture_cycle()
    }

    pub fn verifier_trust_snapshot_sequence(&self) -> u64 {
        self.inner.verifier_trust_snapshot_sequence()
    }

    pub fn digest(&self) -> RestartTrustContextDigestV1 {
        self.digest
    }

    pub fn verify(&self) -> Result<(), RestartQuarantineFacadeError> {
        self.inner
            .verify_integrity()
            .map_err(|error| RestartQuarantineFacadeError::TrustContextRejected {
                detail: format!("{error:?}"),
            })?;
        if self.digest.0 != self.inner.context_digest().as_bytes() {
            return Err(RestartQuarantineFacadeError::TrustContextDigestMismatch);
        }
        Ok(())
    }
}

/// Public read-only inspection handle for an admitted restart-v2 quarantine.
///
/// This type intentionally does not implement `Clone` and never exposes its inner
/// contextualized quarantine, reconstructed ledger, or decoded wire snapshot.
#[derive(Debug)]
pub struct ReadOnlyEpistemicRestartQuarantineV2 {
    inner: ContextualizedEpistemicRestartQuarantineV2,
    trust_context_digest: RestartTrustContextDigestV1,
    quarantine_digest: ReadOnlyRestartQuarantineDigestV1,
}

impl ReadOnlyEpistemicRestartQuarantineV2 {
    #[allow(clippy::too_many_arguments)]
    pub fn validate_and_admit(
        snapshot: &EpistemicRestartWireSnapshotV2,
        policy: &RestartAnchorTrustPolicyV1,
        candidate_anchor_evidence: &RestartAnchorEvidenceV1,
        verifier: &dyn ProfiledRestartAnchorEvidenceVerifierV1,
        trusted_context: &RestartTrustContextHandleV1,
        observed_at_cycle: u64,
    ) -> Result<Self, RestartQuarantineFacadeError> {
        trusted_context.verify()?;
        let inner = ContextualizedEpistemicRestartQuarantineV2::validate_and_construct(
            snapshot,
            policy,
            candidate_anchor_evidence,
            verifier,
            &trusted_context.inner,
            observed_at_cycle,
        )
        .map_err(|error| RestartQuarantineFacadeError::AdmissionRejected {
            detail: format!("{error:?}"),
        })?;

        let trust_context_digest = trusted_context.digest;
        let quarantine_digest = ReadOnlyRestartQuarantineDigestV1(
            inner.contextualized_digest().as_bytes(),
        );
        let out = Self {
            inner,
            trust_context_digest,
            quarantine_digest,
        };
        out.verify()?;
        Ok(out)
    }

    pub fn captured_at_cycle(&self) -> u64 {
        self.inner.quarantine().captured_at_cycle()
    }

    pub fn claim_count(&self) -> usize {
        self.inner.quarantine().ledger().claim_count()
    }

    pub fn evidence_count(&self) -> usize {
        self.inner.quarantine().ledger().evidence_count()
    }

    pub fn provenance_count(&self) -> usize {
        self.inner.quarantine().ledger().provenance_count()
    }

    pub fn claim(&self, id: ClaimId) -> Option<&KnowledgeClaim> {
        self.inner.quarantine().ledger().claim(id)
    }

    pub fn evidence(&self, id: EvidenceId) -> Option<&EvidenceRecord> {
        self.inner.quarantine().ledger().evidence(id)
    }

    pub fn provenance(&self, id: ProvenanceId) -> Option<&ProvenanceRecord> {
        self.inner.quarantine().ledger().provenance(id)
    }

    pub fn source_outer_checksum(&self) -> [u8; 32] {
        self.inner.quarantine().source_snapshot().outer_checksum
    }

    pub fn source_v2_digest(&self) -> [u8; 32] {
        self.inner.quarantine().source_snapshot().claimed_v2_digest
    }

    pub fn trust_context_digest(&self) -> RestartTrustContextDigestV1 {
        self.trust_context_digest
    }

    pub fn quarantine_digest(&self) -> ReadOnlyRestartQuarantineDigestV1 {
        self.quarantine_digest
    }

    pub fn writable_hydration_authorized(&self) -> bool {
        false
    }

    pub fn activation_authorized(&self) -> bool {
        false
    }

    pub fn verify(&self) -> Result<(), RestartQuarantineFacadeError> {
        self.inner
            .verify_read_only()
            .map_err(|error| RestartQuarantineFacadeError::VerificationRejected {
                detail: format!("{error:?}"),
            })?;
        if self.quarantine_digest.0 != self.inner.contextualized_digest().as_bytes() {
            return Err(RestartQuarantineFacadeError::QuarantineDigestMismatch);
        }
        if self.trust_context_digest.0 != self.inner.trusted_context_digest().as_bytes() {
            return Err(RestartQuarantineFacadeError::TrustContextDigestMismatch);
        }
        if self.inner.trusted_context_mutated()
            || self.inner.writable_hydration_authorized()
            || self.inner.activation_authorized()
        {
            return Err(RestartQuarantineFacadeError::UnexpectedAuthority);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestartQuarantineFacadeError {
    TrustContextRejected { detail: String },
    AdmissionRejected { detail: String },
    VerificationRejected { detail: String },
    TrustContextDigestMismatch,
    QuarantineDigestMismatch,
    UnexpectedAuthority,
}

impl fmt::Display for RestartQuarantineFacadeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "restart quarantine facade rejected: {self:?}")
    }
}

impl Error for RestartQuarantineFacadeError {}

fn hex32(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_quarantine_handle_has_no_clone_or_inner_escape_contract() {
        fn context_type(_: Option<&RestartTrustContextHandleV1>) {}
        fn quarantine_type(_: Option<&ReadOnlyEpistemicRestartQuarantineV2>) {}
        context_type(None);
        quarantine_type(None);
    }
}
