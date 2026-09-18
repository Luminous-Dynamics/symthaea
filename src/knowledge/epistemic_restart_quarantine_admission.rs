// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Read-only quarantine admission for externally supplied epistemic restart state.
//!
//! EKM-051 is the first restart tranche allowed to construct a verified in-memory
//! quarantine object from an externally decoded EKM-042 snapshot. It deliberately
//! does not hydrate writable belief state, advance trusted checkpoints, or activate
//! the candidate.
//!
//! Admission is transactional in the read-only sense: EKM-049 verifier provenance
//! and EKM-050 verifier continuity are re-derived internally from the exact candidate
//! rather than accepted as caller-supplied PASS objects.

use super::claim_evidence::EpistemicLedger;
use super::epistemic_restart_admission_review::{
    EpistemicRestartAdmissionReviewDigest, EpistemicRestartAdmissionReviewReceiptV1,
};
use super::epistemic_restart_anchor::{
    RestartAnchorEvidenceV1, RestartAnchorEvidenceVerifierV1, RestartAnchorTrackerV1,
};
use super::epistemic_restart_anchor_policy::RestartAnchorTrustPolicyV1;
use super::epistemic_restart_verifier_continuity::{
    RestartVerifierContinuityDispositionV1, RestartVerifierContinuityGateV1,
    TrustedRestartVerifierStateV1,
};
use super::epistemic_restart_verifier_provenance::{
    ProfiledRestartAnchorEvidenceVerifierV1, RestartVerifierProvenanceDigestV1,
    RestartVerifierProvenanceError, RestartVerifierProvenanceReceiptV1,
};
use super::epistemic_restart_wire_v2::EpistemicRestartWireSnapshotV2;
use super::epistemic_restart_wire_v2_validation::{
    EpistemicRestartWireV2ValidationError, EpistemicRestartWireV2Validator,
};
use super::epistemic_restart_wire_validation::EpistemicRestartWireValidator;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicRestartQuarantineAdmissionVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpistemicRestartQuarantineAdmissionDigest([u8; 32]);

impl EpistemicRestartQuarantineAdmissionDigest {
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

/// Read-only quarantine reconstructed from one exact externally supplied snapshot.
///
/// All fields are private. The object exposes only immutable views and carries no
/// writable support store, revision-history writer, activation handle, or trusted
/// checkpoint mutation primitive.
#[derive(Debug, Clone)]
pub struct AdmittedEpistemicRestartQuarantineV2 {
    version: EpistemicRestartQuarantineAdmissionVersion,
    captured_at_cycle: u64,
    ledger: EpistemicLedger,
    snapshot: EpistemicRestartWireSnapshotV2,
    admission_review_digest: EpistemicRestartAdmissionReviewDigest,
    verifier_provenance_digest: RestartVerifierProvenanceDigestV1,
    verifier_continuity_disposition: RestartVerifierContinuityDispositionV1,
    trusted_anchor_sequence_before_review: u64,
    trusted_verifier_sequence_before_review: u64,
    anchor_tracker_mutated: bool,
    trusted_verifier_state_mutated: bool,
    writable_hydration_authorized: bool,
    activation_authorized: bool,
    admission_digest: EpistemicRestartQuarantineAdmissionDigest,
}

impl AdmittedEpistemicRestartQuarantineV2 {
    #[allow(clippy::too_many_arguments)]
    pub fn validate_and_construct(
        snapshot: &EpistemicRestartWireSnapshotV2,
        policy: &RestartAnchorTrustPolicyV1,
        candidate_anchor_evidence: &RestartAnchorEvidenceV1,
        verifier: &dyn ProfiledRestartAnchorEvidenceVerifierV1,
        trusted_anchor_tracker: &RestartAnchorTrackerV1,
        trusted_verifier_state: &TrustedRestartVerifierStateV1,
        observed_at_cycle: u64,
    ) -> Result<Self, EpistemicRestartQuarantineAdmissionError> {
        // Re-run complete EKM-049. This internally re-runs EKM-044 through EKM-048
        // against the exact candidate snapshot and does not advance the real tracker.
        let (admission_review, verifier_provenance) =
            RestartVerifierProvenanceReceiptV1::validate_and_review(
                snapshot,
                policy,
                candidate_anchor_evidence,
                verifier,
                trusted_anchor_tracker,
                observed_at_cycle,
            )
            .map_err(EpistemicRestartQuarantineAdmissionError::VerifierProvenance)?;

        // Re-run EKM-050 from the exact EKM-049 receipt. No caller-supplied PASS
        // object can be substituted here.
        let verifier_continuity = RestartVerifierContinuityGateV1::evaluate(
            trusted_verifier_state,
            &verifier_provenance,
        );
        if !verifier_continuity.further_review_eligible()
            || matches!(
                verifier_continuity.disposition(),
                RestartVerifierContinuityDispositionV1::Rejected
            )
        {
            return Err(EpistemicRestartQuarantineAdmissionError::VerifierContinuityRejected);
        }
        if verifier_continuity.trusted_state_mutated()
            || verifier_continuity.quarantine_construction_authorized()
            || verifier_continuity.activation_authorized()
        {
            return Err(EpistemicRestartQuarantineAdmissionError::UnexpectedVerifierAuthority);
        }
        if admission_review.anchor_tracker_mutated()
            || admission_review.quarantine_construction_authorized()
            || admission_review.activation_authorized()
        {
            return Err(EpistemicRestartQuarantineAdmissionError::UnexpectedAnchorAuthority);
        }
        if verifier_provenance.quarantine_construction_authorized()
            || verifier_provenance.activation_authorized()
        {
            return Err(EpistemicRestartQuarantineAdmissionError::UnexpectedProvenanceAuthority);
        }

        // Independently re-run the complete V2 semantic validator immediately
        // before reconstruction so the quarantine never relies only on an older
        // review receipt.
        let validation = EpistemicRestartWireV2Validator::validate(snapshot)
            .map_err(EpistemicRestartQuarantineAdmissionError::SemanticValidation)?;
        if validation.captured_at_cycle != admission_review.candidate_capture_cycle() {
            return Err(EpistemicRestartQuarantineAdmissionError::CaptureCycleMismatch);
        }

        let ledger = reconstruct_read_only_ledger(snapshot)?;
        verify_reconstructed_ledger(snapshot, &ledger)?;

        let trusted_anchor_sequence_before_review = trusted_anchor_tracker
            .latest_sequence()
            .ok_or(EpistemicRestartQuarantineAdmissionError::UninitializedTrustedAnchor)?;
        if trusted_anchor_sequence_before_review != admission_review.prior_anchor_sequence() {
            return Err(EpistemicRestartQuarantineAdmissionError::TrustedAnchorChangedDuringReview);
        }
        let trusted_verifier_sequence_before_review =
            trusted_verifier_state.trust_snapshot_sequence();

        let mut quarantine = Self {
            version: EpistemicRestartQuarantineAdmissionVersion::V1,
            captured_at_cycle: validation.captured_at_cycle,
            ledger,
            snapshot: snapshot.clone(),
            admission_review_digest: admission_review.review_digest(),
            verifier_provenance_digest: verifier_provenance.provenance_digest(),
            verifier_continuity_disposition: verifier_continuity.disposition(),
            trusted_anchor_sequence_before_review,
            trusted_verifier_sequence_before_review,
            anchor_tracker_mutated: false,
            trusted_verifier_state_mutated: false,
            writable_hydration_authorized: false,
            activation_authorized: false,
            admission_digest: EpistemicRestartQuarantineAdmissionDigest([0; 32]),
        };
        quarantine.admission_digest = digest_quarantine_admission(&quarantine);
        quarantine.verify_read_only()?;
        Ok(quarantine)
    }

    pub fn version(&self) -> EpistemicRestartQuarantineAdmissionVersion {
        self.version
    }

    pub fn captured_at_cycle(&self) -> u64 {
        self.captured_at_cycle
    }

    pub fn ledger(&self) -> &EpistemicLedger {
        &self.ledger
    }

    pub fn source_snapshot(&self) -> &EpistemicRestartWireSnapshotV2 {
        &self.snapshot
    }

    pub fn admission_review_digest(&self) -> EpistemicRestartAdmissionReviewDigest {
        self.admission_review_digest
    }

    pub fn verifier_provenance_digest(&self) -> RestartVerifierProvenanceDigestV1 {
        self.verifier_provenance_digest
    }

    pub fn verifier_continuity_disposition(&self) -> RestartVerifierContinuityDispositionV1 {
        self.verifier_continuity_disposition
    }

    pub fn trusted_anchor_sequence_before_review(&self) -> u64 {
        self.trusted_anchor_sequence_before_review
    }

    pub fn trusted_verifier_sequence_before_review(&self) -> u64 {
        self.trusted_verifier_sequence_before_review
    }

    pub fn anchor_tracker_mutated(&self) -> bool {
        self.anchor_tracker_mutated
    }

    pub fn trusted_verifier_state_mutated(&self) -> bool {
        self.trusted_verifier_state_mutated
    }

    pub fn writable_hydration_authorized(&self) -> bool {
        self.writable_hydration_authorized
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }

    pub fn admission_digest(&self) -> EpistemicRestartQuarantineAdmissionDigest {
        self.admission_digest
    }

    /// Re-run all checks available from the immutable quarantine representation.
    pub fn verify_read_only(&self) -> Result<(), EpistemicRestartQuarantineAdmissionError> {
        let validation = EpistemicRestartWireV2Validator::validate(&self.snapshot)
            .map_err(EpistemicRestartQuarantineAdmissionError::SemanticValidation)?;
        if validation.captured_at_cycle != self.captured_at_cycle {
            return Err(EpistemicRestartQuarantineAdmissionError::CaptureCycleMismatch);
        }
        verify_reconstructed_ledger(&self.snapshot, &self.ledger)?;
        if self.anchor_tracker_mutated
            || self.trusted_verifier_state_mutated
            || self.writable_hydration_authorized
            || self.activation_authorized
        {
            return Err(EpistemicRestartQuarantineAdmissionError::UnexpectedQuarantineAuthority);
        }
        let digest = digest_quarantine_admission(self);
        if digest != self.admission_digest {
            return Err(EpistemicRestartQuarantineAdmissionError::AdmissionDigestMismatch);
        }
        Ok(())
    }
}

fn reconstruct_read_only_ledger(
    snapshot: &EpistemicRestartWireSnapshotV2,
) -> Result<EpistemicLedger, EpistemicRestartQuarantineAdmissionError> {
    // EKM-035 validates the base snapshot's complete graph/state-machine first.
    EpistemicRestartWireValidator::validate(&snapshot.base)
        .map_err(EpistemicRestartQuarantineAdmissionError::BaseValidation)?;

    let mut ledger = EpistemicLedger::new();
    for record in &snapshot.base.provenance {
        let id = ledger
            .add_provenance(
                record.source_label.clone(),
                record.source_uri.clone(),
                record.content_hash.clone(),
                record.recorded_at_cycle,
                record.parent_ids.clone(),
            )
            .map_err(|_| EpistemicRestartQuarantineAdmissionError::LedgerRebuildRejected)?;
        if id != record.id {
            return Err(EpistemicRestartQuarantineAdmissionError::LedgerRebuildIdMismatch);
        }
    }
    for record in &snapshot.base.claims {
        let id = ledger.add_claim(
            record.statement.clone(),
            record.kind,
            record.domain.clone(),
            record.scope.clone(),
            record.created_at_cycle,
        );
        if id != record.id {
            return Err(EpistemicRestartQuarantineAdmissionError::LedgerRebuildIdMismatch);
        }
    }
    for record in &snapshot.base.evidence {
        let id = ledger
            .add_evidence(
                record.claim_id,
                record.kind,
                record.polarity,
                record.provenance_id,
                record.observed_at_cycle,
                record.context.clone(),
                record.method.clone(),
            )
            .map_err(|_| EpistemicRestartQuarantineAdmissionError::LedgerRebuildRejected)?;
        if id != record.id {
            return Err(EpistemicRestartQuarantineAdmissionError::LedgerRebuildIdMismatch);
        }
    }
    Ok(ledger)
}

fn verify_reconstructed_ledger(
    snapshot: &EpistemicRestartWireSnapshotV2,
    ledger: &EpistemicLedger,
) -> Result<(), EpistemicRestartQuarantineAdmissionError> {
    if ledger.provenance_count() != snapshot.base.provenance.len()
        || ledger.claim_count() != snapshot.base.claims.len()
        || ledger.evidence_count() != snapshot.base.evidence.len()
    {
        return Err(EpistemicRestartQuarantineAdmissionError::LedgerCountMismatch);
    }
    for record in &snapshot.base.provenance {
        let rebuilt = ledger
            .provenance(record.id)
            .ok_or(EpistemicRestartQuarantineAdmissionError::LedgerSemanticMismatch)?;
        if rebuilt.source_label != record.source_label
            || rebuilt.source_uri != record.source_uri
            || rebuilt.content_hash != record.content_hash
            || rebuilt.recorded_at_cycle != record.recorded_at_cycle
            || rebuilt.parent_ids != record.parent_ids
        {
            return Err(EpistemicRestartQuarantineAdmissionError::LedgerSemanticMismatch);
        }
    }
    for record in &snapshot.base.claims {
        let rebuilt = ledger
            .claim(record.id)
            .ok_or(EpistemicRestartQuarantineAdmissionError::LedgerSemanticMismatch)?;
        if rebuilt.statement != record.statement
            || rebuilt.kind != record.kind
            || rebuilt.domain != record.domain
            || rebuilt.scope != record.scope
            || rebuilt.created_at_cycle != record.created_at_cycle
            || rebuilt.evidence_ids != record.evidence_ids
        {
            return Err(EpistemicRestartQuarantineAdmissionError::LedgerSemanticMismatch);
        }
    }
    for record in &snapshot.base.evidence {
        let rebuilt = ledger
            .evidence(record.id)
            .ok_or(EpistemicRestartQuarantineAdmissionError::LedgerSemanticMismatch)?;
        if rebuilt.claim_id != record.claim_id
            || rebuilt.kind != record.kind
            || rebuilt.polarity != record.polarity
            || rebuilt.provenance_id != record.provenance_id
            || rebuilt.observed_at_cycle != record.observed_at_cycle
            || rebuilt.context != record.context
            || rebuilt.method != record.method
        {
            return Err(EpistemicRestartQuarantineAdmissionError::LedgerSemanticMismatch);
        }
    }
    Ok(())
}

fn digest_quarantine_admission(
    quarantine: &AdmittedEpistemicRestartQuarantineV2,
) -> EpistemicRestartQuarantineAdmissionDigest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-read-only-quarantine-admission-v1");
    hasher.update(&[1]);
    hasher.update(&quarantine.captured_at_cycle.to_le_bytes());
    hasher.update(&quarantine.snapshot.outer_checksum);
    hasher.update(&quarantine.snapshot.claimed_v2_digest);
    hasher.update(&quarantine.admission_review_digest.as_bytes());
    hasher.update(&quarantine.verifier_provenance_digest.as_bytes());
    hasher.update(&[verifier_continuity_tag(
        quarantine.verifier_continuity_disposition,
    )]);
    hasher.update(&quarantine.trusted_anchor_sequence_before_review.to_le_bytes());
    hasher.update(&quarantine.trusted_verifier_sequence_before_review.to_le_bytes());
    hasher.update(&[u8::from(quarantine.anchor_tracker_mutated)]);
    hasher.update(&[u8::from(quarantine.trusted_verifier_state_mutated)]);
    hasher.update(&[u8::from(quarantine.writable_hydration_authorized)]);
    hasher.update(&[u8::from(quarantine.activation_authorized)]);
    EpistemicRestartQuarantineAdmissionDigest(*hasher.finalize().as_bytes())
}

fn verifier_continuity_tag(disposition: RestartVerifierContinuityDispositionV1) -> u8 {
    match disposition {
        RestartVerifierContinuityDispositionV1::StableTrustReuse => 1,
        RestartVerifierContinuityDispositionV1::TrustSnapshotAdvance => 2,
        RestartVerifierContinuityDispositionV1::Rejected => 3,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EpistemicRestartQuarantineAdmissionError {
    VerifierProvenance(RestartVerifierProvenanceError),
    VerifierContinuityRejected,
    UnexpectedVerifierAuthority,
    UnexpectedAnchorAuthority,
    UnexpectedProvenanceAuthority,
    SemanticValidation(EpistemicRestartWireV2ValidationError),
    BaseValidation(super::epistemic_restart_wire_validation::EpistemicRestartWireValidationError),
    CaptureCycleMismatch,
    UninitializedTrustedAnchor,
    TrustedAnchorChangedDuringReview,
    LedgerRebuildRejected,
    LedgerRebuildIdMismatch,
    LedgerCountMismatch,
    LedgerSemanticMismatch,
    UnexpectedQuarantineAuthority,
    AdmissionDigestMismatch,
}

impl fmt::Display for EpistemicRestartQuarantineAdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epistemic restart quarantine admission rejected: {self:?}")
    }
}

impl Error for EpistemicRestartQuarantineAdmissionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarantine_type_exposes_no_authority_by_default_contract() {
        // Compile-level guard: the public type exists only through validate_and_construct;
        // there is no Default, public field construction, into_live, hydrate, or activate API.
        fn assert_type(_: Option<&AdmittedEpistemicRestartQuarantineV2>) {}
        assert_type(None);
    }
}
