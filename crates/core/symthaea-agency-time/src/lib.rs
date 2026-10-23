// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Project-neutral temporal ABI for bounded consequential agency.
//!
//! This crate freezes the first type boundary from AGENCY-TIME-001 without
//! selecting a production clock provider. Its central rule is **No Time
//! Laundering**:
//!
//! ```text
//! civil timestamp
//! != monotonic elapsed duration
//! != causal ordering
//! != runtime incarnation
//! != lease freshness
//! != finality
//! ```
//!
//! This source tranche deliberately cannot mint either a live monotonic
//! deadline or a positive freshness/currentness witness. Production
//! construction and verification remain blocked on the verifier-owned
//! clock-provider theorem in AGENCY-TIME-001A.

#![deny(unsafe_code)]

/// Schema version for the first Agency temporal ABI.
pub const AGENCY_TIME_ABI_SCHEMA_VERSION: u16 = 1;

/// Closed semantic roles that must not be interchanged merely because their
/// representations are numeric or time-like.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum TemporalRole {
    /// Externally meaningful civil/wall-clock time evidence.
    ExternalCivilTime = 0,
    /// Runtime-local elapsed duration measured by a monotonic provider.
    MonotonicElapsed = 1,
    /// Ordering inside an identified causal lineage.
    CausalOrdering = 2,
    /// Identity of one live runtime/executor incarnation.
    RuntimeIncarnation = 3,
    /// Freshness/currentness of one bounded live lease.
    LeaseFreshness = 4,
    /// Explicit domain proof that a state/effect is final under that domain.
    Finality = 5,
}

/// Opaque identity of one live runtime incarnation.
///
/// The bytes are intentionally not caller-constructible through the public API.
/// Possessing or comparing this identifier does not grant execution authority.
///
/// ```compile_fail
/// use symthaea_agency_time::RuntimeIncarnationId;
/// let _forged = RuntimeIncarnationId([7; 32]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimeIncarnationId([u8; 32]);

impl RuntimeIncarnationId {
    /// Read-only stable identity bytes for evidence correlation.
    ///
    /// These bytes are an identifier, not a constructor input or live-currentness
    /// proof.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Opaque identity of the runtime-time semantics/provider profile.
///
/// A profile identity names semantics; it does not prove the provider is live,
/// current, or correctly configured.
///
/// ```compile_fail
/// use symthaea_agency_time::RuntimeTimeProfileId;
/// let _forged = RuntimeTimeProfileId([3; 16]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimeTimeProfileId([u8; 16]);

impl RuntimeTimeProfileId {
    /// Read-only stable profile identity bytes for same-profile checks.
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

/// Live monotonic deadline bound to one exact runtime incarnation and one exact
/// time-provider/profile identity.
///
/// There is intentionally no public constructor. The private deadline coordinate
/// is not exposed through accessors or formatting because callers must not replay
/// a raw provider tick as freshness proof. A later verifier-owned provider inside
/// this crate may construct and evaluate this type after AGENCY-TIME-001A is
/// implemented.
///
/// ```compile_fail
/// use symthaea_agency_time::MonotonicLeaseDeadline;
/// let _forged = MonotonicLeaseDeadline {};
/// ```
pub struct MonotonicLeaseDeadline {
    runtime_incarnation: RuntimeIncarnationId,
    profile_id: RuntimeTimeProfileId,
    /// Sealed until the trusted provider successor (#5082) consumes it. Keeping
    /// this private and unreadable is part of the ABI theorem, not accidental
    /// dead code in this source-only predecessor.
    #[allow(dead_code)]
    deadline_coordinate: u64,
}

impl MonotonicLeaseDeadline {
    /// Runtime incarnation this live deadline belongs to.
    pub const fn runtime_incarnation(&self) -> RuntimeIncarnationId {
        self.runtime_incarnation
    }

    /// Runtime-time profile under which this deadline was created.
    pub const fn profile_id(&self) -> RuntimeTimeProfileId {
        self.profile_id
    }
}

/// Negative result vocabulary for verifier-owned live deadline evaluation.
///
/// This enum intentionally has **no positive `Current` variant**. A public enum
/// variant is caller-constructible and therefore cannot serve as proof of live
/// freshness. A future positive currentness witness must be opaque and
/// verifier-owned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeadlineRejection {
    /// The verifier-owned monotonic deadline has elapsed.
    Expired,
    /// The deadline belongs to another runtime incarnation.
    RuntimeIncarnationMismatch,
    /// The verifier/provider profile differs from the one bound into the deadline.
    ProfileMismatch,
    /// The verifier cannot currently establish deadline currentness.
    Indeterminate,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn incarnation(byte: u8) -> RuntimeIncarnationId {
        RuntimeIncarnationId([byte; 32])
    }

    fn profile(byte: u8) -> RuntimeTimeProfileId {
        RuntimeTimeProfileId([byte; 16])
    }

    fn deadline(
        runtime_incarnation: RuntimeIncarnationId,
        profile_id: RuntimeTimeProfileId,
        deadline_coordinate: u64,
    ) -> MonotonicLeaseDeadline {
        MonotonicLeaseDeadline {
            runtime_incarnation,
            profile_id,
            deadline_coordinate,
        }
    }

    #[test]
    fn temporal_roles_are_distinct() {
        let roles = [
            TemporalRole::ExternalCivilTime,
            TemporalRole::MonotonicElapsed,
            TemporalRole::CausalOrdering,
            TemporalRole::RuntimeIncarnation,
            TemporalRole::LeaseFreshness,
            TemporalRole::Finality,
        ];
        let unique = roles.into_iter().collect::<BTreeSet<_>>();
        assert_eq!(unique.len(), roles.len());
    }

    #[test]
    fn runtime_incarnation_identity_is_exact() {
        assert_ne!(incarnation(1), incarnation(2));
        assert_eq!(incarnation(3).as_bytes(), &[3; 32]);
    }

    #[test]
    fn runtime_time_profile_identity_is_exact() {
        assert_ne!(profile(1), profile(2));
        assert_eq!(profile(4).as_bytes(), &[4; 16]);
    }

    #[test]
    fn live_deadline_binds_exact_runtime_and_profile() {
        let runtime = incarnation(9);
        let time_profile = profile(8);
        let live = deadline(runtime, time_profile, 42);

        assert_eq!(live.runtime_incarnation(), runtime);
        assert_eq!(live.profile_id(), time_profile);
        assert_eq!(live.deadline_coordinate, 42);
    }

    #[test]
    fn rejection_states_do_not_collapse() {
        let states = [
            DeadlineRejection::Expired,
            DeadlineRejection::RuntimeIncarnationMismatch,
            DeadlineRejection::ProfileMismatch,
            DeadlineRejection::Indeterminate,
        ];
        let unique = states.into_iter().collect::<std::collections::HashSet<_>>();
        assert_eq!(unique.len(), states.len());
    }
}
