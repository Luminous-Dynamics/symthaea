// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Corrected public surface for durable welfare-authority replay recovery.
//!
//! The implementation module retains the persistence/recovery machinery, but its diagnostic
//! snapshot reconstruction helper is intentionally not exposed: persisted snapshots are owned by
//! the persistence boundary and are the source of truth. Runtime code may observe only generation
//! and the exact latest committed digest.

use symthaea_core::intervention_interlock::InterventionRequest;
use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use symthaea_fabrication_kernel::trust::TrustSnapshot;
use symthaea_psych_bench::moral_patient::{MoralPatientEvidenceProfile, PrecautionPolicy};
use symthaea_welfare_authority::{
    SignedWelfareInterventionAuthority, VerifiedWelfareInterventionAuthority,
    WelfareAuthorityPolicyManifest, WelfareAuthoritySignatureVerifier, WelfareAuthorityTracker,
};
use symthaea_welfare_consent::{SubjectConsentLedger, SubjectIdentityRegistry};

use crate::evidence_context::EvidenceBoundInterventionPermit;

#[path = "replay_recovery.rs"]
mod implementation;

pub use implementation::{
    DURABLE_REPLAY_SCHEMA, DurableAuthorizationError, DurableAuthorityReplaySnapshot,
    DurableReplayError, DurableReplayReservation, DurableReplayReservationError,
    ReplayFencePersistence, ReplayScopeCursor, ReplayScopeRecord, digest_replay_snapshot,
};

/// Public durable replay fence. Deliberately omits any API that tries to reconstruct the
/// externally persisted snapshot from partial live state.
#[derive(Debug, Clone)]
pub struct DurableAuthorityReplayFence {
    inner: implementation::DurableAuthorityReplayFence,
}

impl DurableAuthorityReplayFence {
    pub fn new() -> Result<Self, DurableReplayError> {
        implementation::DurableAuthorityReplayFence::new().map(|inner| Self { inner })
    }

    pub fn recover_anchored(
        snapshot: &DurableAuthorityReplaySnapshot,
        expected_latest_digest: Sha256Digest,
    ) -> Result<Self, DurableReplayError> {
        implementation::DurableAuthorityReplayFence::recover_anchored(
            snapshot,
            expected_latest_digest,
        )
        .map(|inner| Self { inner })
    }

    pub fn generation(&self) -> u64 {
        self.inner.generation()
    }

    pub fn current_digest(&self) -> Sha256Digest {
        self.inner.current_digest()
    }

    pub fn reserve_verified<P: ReplayFencePersistence>(
        &mut self,
        authority: &VerifiedWelfareInterventionAuthority,
        persistence: &mut P,
    ) -> Result<DurableReplayReservation, DurableReplayReservationError<P::Error>> {
        self.inner.reserve_verified(authority, persistence)
    }
}

/// Verify evidence-bound authority, durably reserve its replay identity, then run the normal
/// hardened authorization path. The reservation is persisted before permit minting continues.
#[allow(clippy::too_many_arguments)]
pub fn authorize_evidence_bound_intervention_durable<P: ReplayFencePersistence>(
    base_nonce: &str,
    profile: &MoralPatientEvidenceProfile,
    precaution_policy: &PrecautionPolicy,
    subject_id: &str,
    consent_ledger: &SubjectConsentLedger,
    subject_registry: &SubjectIdentityRegistry,
    signed_authority: &SignedWelfareInterventionAuthority,
    authority_manifest: &WelfareAuthorityPolicyManifest,
    authority_trust_snapshot: &TrustSnapshot,
    authority_verifier: &dyn WelfareAuthoritySignatureVerifier,
    volatile_authority_tracker: &mut WelfareAuthorityTracker,
    durable_replay_fence: &mut DurableAuthorityReplayFence,
    replay_persistence: &mut P,
    request: &InterventionRequest,
    unix_s: u64,
) -> Result<
    (EvidenceBoundInterventionPermit, DurableReplayReservation),
    DurableAuthorizationError<P::Error>,
> {
    implementation::authorize_evidence_bound_intervention_durable(
        base_nonce,
        profile,
        precaution_policy,
        subject_id,
        consent_ledger,
        subject_registry,
        signed_authority,
        authority_manifest,
        authority_trust_snapshot,
        authority_verifier,
        volatile_authority_tracker,
        &mut durable_replay_fence.inner,
        replay_persistence,
        request,
        unix_s,
    )
}
