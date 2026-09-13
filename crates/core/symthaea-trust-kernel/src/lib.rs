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

mod clock;
mod continuity;
mod digest;
mod signature;
mod trust;

pub use clock::{
    CLOCK_OBSERVATION_SCHEMA, ClockEpochTracker, ClockObservation, ClockObservationVerifier,
    ClockQuorumPolicy, ClockTrackingError, ClockViolation, VerifiedClockWindow,
    canonical_clock_observation_bytes, digest_clock_epoch_tracker, digest_clock_observation,
    verify_clock_quorum,
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
