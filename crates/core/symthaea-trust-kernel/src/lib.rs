// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared lifecycle-governed trust and quorum-clock substrate.
//!
//! This crate is introduced as a parallel compatibility implementation. It has
//! no live authority consumers yet. Existing `symthaea.fabrication.*` schema
//! and digest domain strings are preserved intentionally so extraction does not
//! silently become a protocol migration.
//!
//! ```text
//! valid signature
//! != trusted signer
//! != purpose-authorized active key
//! != fresh trust snapshot
//! != quorum-derived clock window
//! != continuity across clock epochs
//! ```

#![deny(unsafe_code)]

mod accepted_clock_basis;
mod clock;
mod clock_bootstrap_authority;
mod clock_evaluation_permit;
mod clock_governance_time;
mod clock_operational;
mod clock_witness;
mod continuity;
mod digest;
mod signature;
mod trust;

pub use accepted_clock_basis::{
    ACCEPTED_CLOCK_BASIS_SCHEMA, AcceptedClockBasisError, AcceptedClockBasisIdV5,
    AcceptedClockBasisV5, accept_bootstrap_clock_basis_v5,
};
pub use clock::{
    CLOCK_OBSERVATION_SCHEMA, ClockEpochTracker, ClockObservation, ClockObservationVerifier,
    ClockQuorumPolicy, ClockTrackingError, ClockViolation, VerifiedClockWindow,
    canonical_clock_observation_bytes, digest_clock_epoch_tracker, digest_clock_observation,
    verify_clock_quorum,
};
pub use clock_bootstrap_authority::{
    CLOCK_BOOTSTRAP_AUTHORITY_EVIDENCE_SCHEMA, CLOCK_BOOTSTRAP_CLAIM_SCHEMA,
    VERIFIED_CLOCK_BOOTSTRAP_AUTHORITY_SCHEMA, ClockBootstrapAuthorityError,
    ClockBootstrapAuthorityEvidenceV2, ClockBootstrapAuthorityVerifier, ClockBootstrapClaimV2,
    VerifiedClockBootstrapAuthorityIdV2, VerifiedClockBootstrapAuthorityV2,
    verify_clock_bootstrap_authority,
};
pub use clock_evaluation_permit::{
    CLOCK_BOOTSTRAP_ANCHOR_SCHEMA, CLOCK_CONTINUITY_POLICY_SCHEMA,
    CLOCK_EVALUATION_PERMIT_SCHEMA, CLOCK_EVALUATION_POLICY_SCHEMA,
    CLOCK_QUORUM_POLICY_SCHEMA, ClockAuthorityKeyV1, ClockBootstrapAnchorIdV2,
    ClockContinuityPolicyRevisionIdV1, ClockContinuityPolicyRevisionV1,
    ClockEvaluationPermitError, ClockEvaluationPermitIdV4, ClockEvaluationPermitV4,
    ClockEvaluationPolicyIdV4, ClockEvaluationPolicyV4, ClockQuorumPolicyRevisionIdV1,
    ClockQuorumPolicyRevisionV1, derive_bootstrap_clock_evaluation_permit_v4,
};
pub use clock_governance_time::{
    CLOCK_GOVERNANCE_EVALUATION_ENVELOPE_SCHEMA, ClockGovernanceEvaluationEnvelopeIdV1,
    ClockGovernanceEvaluationEnvelopeV1, ClockGovernanceTimeError,
    derive_clock_governance_evaluation_envelope_v1,
};
pub use clock_operational::{
    OPERATIONAL_CLOCK_BASIS_SCHEMA, OPERATIONAL_CLOCK_SUCCESSOR_PERMIT_SCHEMA,
    ClockSuccessorEvaluationPermitIdV2, ClockSuccessorEvaluationPermitV2,
    OperationalClockAuthorityKeyV1, OperationalClockBasisIdV1, OperationalClockBasisKindV1,
    OperationalClockBasisV1, OperationalClockError, accept_operational_clock_successor_v1,
    bind_bootstrap_operational_clock_basis_v1, derive_operational_clock_successor_permit_v2,
};
pub use clock_witness::{
    AcceptedClockObservationV1, CLOCK_WINDOW_EVALUATION_WITNESS_SCHEMA, ClockSignerV1,
    ClockWindowEvaluationWitnessV1, ClockWindowWitnessError,
    digest_clock_window_evaluation_witness, verify_clock_quorum_with_witness,
    verify_clock_window_evaluation_witness,
};
pub use continuity::{
    CLOCK_CONTINUITY_SCHEMA, ClockContinuityError, ClockContinuityPolicy,
    VerifiedClockContinuity, digest_clock_continuity, verify_clock_continuity,
};
pub use digest::{DigestParseError, Sha256Digest, sha256};
pub use signature::{DetachedSignature, MAX_SIGNATURE_ALGORITHM_NAME_BYTES, SignatureAlgorithm};
pub use trust::{
    KeyEligibility, KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot,
    TrustSnapshotError, TrustSnapshotTracker, TrustSnapshotTrackingError,
    canonical_trust_snapshot_bytes, digest_trust_snapshot,
};
