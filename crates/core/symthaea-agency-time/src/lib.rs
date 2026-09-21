// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Project-neutral temporal ABI for bounded consequential agency.
//!
//! Time-like values are deliberately separated by semantic role. The Linux
//! implementation further splits deadline **issuance** from deadline
//! **verification** so a high-rate firewall can check currentness without
//! gaining a deadline-renewal capability.

#![deny(unsafe_code)]

pub const AGENCY_TIME_ABI_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum TemporalRole {
    ExternalCivilTime = 0,
    MonotonicElapsed = 1,
    CausalOrdering = 2,
    RuntimeIncarnation = 3,
    LeaseFreshness = 4,
    Finality = 5,
}

/// Opaque identifier for one live runtime incarnation. Identity is not authority.
///
/// ```compile_fail
/// use symthaea_agency_time::RuntimeIncarnationId;
/// let _forged = RuntimeIncarnationId([7; 32]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimeIncarnationId([u8; 32]);

impl RuntimeIncarnationId {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Opaque identity of one runtime-time semantic profile.
///
/// ```compile_fail
/// use symthaea_agency_time::RuntimeTimeProfileId;
/// let _forged = RuntimeTimeProfileId([3; 16]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimeTimeProfileId([u8; 16]);

impl RuntimeTimeProfileId {
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

/// Live monotonic deadline bound to one exact runtime incarnation and profile.
///
/// There is intentionally no public constructor, raw-coordinate accessor,
/// `Debug`, cloning, or serialization surface.
///
/// ```compile_fail
/// use symthaea_agency_time::MonotonicLeaseDeadline;
/// let _forged = MonotonicLeaseDeadline {};
/// ```
pub struct MonotonicLeaseDeadline {
    runtime_incarnation: RuntimeIncarnationId,
    profile_id: RuntimeTimeProfileId,
    deadline_coordinate: u64,
}

impl MonotonicLeaseDeadline {
    pub const fn runtime_incarnation(&self) -> RuntimeIncarnationId {
        self.runtime_incarnation
    }

    pub const fn profile_id(&self) -> RuntimeTimeProfileId {
        self.profile_id
    }
}

/// Negative result vocabulary for verifier-owned live deadline evaluation.
/// There is deliberately no caller-constructible positive `Current` value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeadlineRejection {
    Expired,
    RuntimeIncarnationMismatch,
    ProfileMismatch,
    Indeterminate,
}

#[cfg(target_os = "linux")]
mod linux_boottime;
#[cfg(target_os = "linux")]
pub use linux_boottime::{LinuxBoottimeIssuer, LinuxBoottimeVerifier, RuntimeTimeError};

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
        assert_eq!(roles.into_iter().collect::<BTreeSet<_>>().len(), roles.len());
    }

    #[test]
    fn runtime_and_profile_identities_are_exact() {
        assert_ne!(incarnation(1), incarnation(2));
        assert_ne!(profile(1), profile(2));
        assert_eq!(incarnation(3).as_bytes(), &[3; 32]);
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
        assert_eq!(
            states.into_iter().collect::<std::collections::HashSet<_>>().len(),
            states.len()
        );
    }
}
