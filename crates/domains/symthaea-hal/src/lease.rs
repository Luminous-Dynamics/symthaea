// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Expiring actuation-lease contract for host-side command admission.
//!
//! A lease binds one command evidence identity to a monotonic sequence and a
//! bounded validity interval. This first tranche is deliberately host-process
//! enforced: the guard must be called by a live process and therefore is not an
//! independent deadman. MCU/drive/watchdog enforcement requires a later tranche
//! and separate qualification.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeaseEnforcementClass {
    /// The current host process evaluates lease validity. Process death itself
    /// cannot be detected by this class without an independent lower layer.
    HostProcessOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActuationLeaseV1 {
    pub sequence: u64,
    pub issued_at_monotonic_us: u64,
    pub valid_until_monotonic_us: u64,
    pub controller_id: String,
    pub safety_profile_id: String,
    pub fallback_profile_id: String,
    /// Content/evidence identity supplied by the command producer. The lease
    /// admission API requires an exact match with the command being admitted.
    pub command_evidence_id: String,
}

impl ActuationLeaseV1 {
    pub fn duration_us(&self) -> Option<u64> {
        self.valid_until_monotonic_us
            .checked_sub(self.issued_at_monotonic_us)
    }

    pub fn validate_structure(&self) -> Result<(), LeaseRejection> {
        if self.sequence == 0 {
            return Err(LeaseRejection::ZeroSequence);
        }
        if self.duration_us().is_none() {
            return Err(LeaseRejection::InvalidInterval);
        }
        if self.controller_id.trim().is_empty() {
            return Err(LeaseRejection::MissingControllerIdentity);
        }
        if self.safety_profile_id.trim().is_empty() {
            return Err(LeaseRejection::MissingSafetyProfileIdentity);
        }
        if self.fallback_profile_id.trim().is_empty() {
            return Err(LeaseRejection::MissingFallbackProfileIdentity);
        }
        if self.command_evidence_id.trim().is_empty() {
            return Err(LeaseRejection::MissingCommandIdentity);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeaseRejection {
    ZeroSequence,
    InvalidInterval,
    MissingControllerIdentity,
    MissingSafetyProfileIdentity,
    MissingFallbackProfileIdentity,
    MissingCommandIdentity,
    IssuedInFuture {
        issued_at_monotonic_us: u64,
        observed_at_monotonic_us: u64,
    },
    Expired {
        valid_until_monotonic_us: u64,
        observed_at_monotonic_us: u64,
    },
    DurationExceedsPolicy {
        duration_us: u64,
        maximum_duration_us: u64,
    },
    ReplayOrReorder {
        sequence: u64,
        last_admitted_sequence: u64,
    },
    CommandIdentityMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeaseAdmission {
    pub sequence: u64,
    pub observed_at_monotonic_us: u64,
    pub remaining_us: u64,
    pub controller_id: String,
    pub safety_profile_id: String,
    pub fallback_profile_id: String,
    pub command_evidence_id: String,
    pub enforcement_class: LeaseEnforcementClass,
    /// Explicitly false for this tranche. A true independent-deadman claim must
    /// come from a lower-level executor/watchdog and separate evidence.
    pub independent_process_death_enforcement: bool,
}

#[derive(Debug, Clone)]
pub struct CommandLeaseGuard {
    maximum_duration_us: u64,
    last_admitted_sequence: Option<u64>,
}

impl CommandLeaseGuard {
    pub fn new(maximum_duration_us: u64) -> Self {
        Self {
            maximum_duration_us,
            last_admitted_sequence: None,
        }
    }

    pub fn maximum_duration_us(&self) -> u64 {
        self.maximum_duration_us
    }

    pub fn last_admitted_sequence(&self) -> Option<u64> {
        self.last_admitted_sequence
    }

    pub fn reset_sequence(&mut self) {
        self.last_admitted_sequence = None;
    }

    /// Admit one lease against the command identity being considered for
    /// actuation. All timestamps are in the same caller-defined monotonic clock
    /// domain; cross-clock conversion is intentionally outside this contract.
    pub fn admit(
        &mut self,
        lease: &ActuationLeaseV1,
        observed_at_monotonic_us: u64,
        expected_command_evidence_id: &str,
    ) -> Result<LeaseAdmission, LeaseRejection> {
        lease.validate_structure()?;

        if lease.issued_at_monotonic_us > observed_at_monotonic_us {
            return Err(LeaseRejection::IssuedInFuture {
                issued_at_monotonic_us: lease.issued_at_monotonic_us,
                observed_at_monotonic_us,
            });
        }
        if observed_at_monotonic_us >= lease.valid_until_monotonic_us {
            return Err(LeaseRejection::Expired {
                valid_until_monotonic_us: lease.valid_until_monotonic_us,
                observed_at_monotonic_us,
            });
        }

        let duration_us = lease.duration_us().ok_or(LeaseRejection::InvalidInterval)?;
        if duration_us > self.maximum_duration_us {
            return Err(LeaseRejection::DurationExceedsPolicy {
                duration_us,
                maximum_duration_us: self.maximum_duration_us,
            });
        }

        if let Some(last) = self.last_admitted_sequence
            && lease.sequence <= last
        {
            return Err(LeaseRejection::ReplayOrReorder {
                sequence: lease.sequence,
                last_admitted_sequence: last,
            });
        }

        if lease.command_evidence_id != expected_command_evidence_id {
            return Err(LeaseRejection::CommandIdentityMismatch);
        }

        let remaining_us = lease.valid_until_monotonic_us - observed_at_monotonic_us;
        self.last_admitted_sequence = Some(lease.sequence);
        Ok(LeaseAdmission {
            sequence: lease.sequence,
            observed_at_monotonic_us,
            remaining_us,
            controller_id: lease.controller_id.clone(),
            safety_profile_id: lease.safety_profile_id.clone(),
            fallback_profile_id: lease.fallback_profile_id.clone(),
            command_evidence_id: lease.command_evidence_id.clone(),
            enforcement_class: LeaseEnforcementClass::HostProcessOnly,
            independent_process_death_enforcement: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lease(sequence: u64) -> ActuationLeaseV1 {
        ActuationLeaseV1 {
            sequence,
            issued_at_monotonic_us: 1_000,
            valid_until_monotonic_us: 1_100,
            controller_id: "controller-A".to_string(),
            safety_profile_id: "safety-v1".to_string(),
            fallback_profile_id: "hold-v1".to_string(),
            command_evidence_id: format!("cmd-{sequence}"),
        }
    }

    #[test]
    fn valid_short_lease_is_admitted_as_host_only() {
        let mut guard = CommandLeaseGuard::new(500);
        let lease = lease(1);
        let admission = guard.admit(&lease, 1_050, "cmd-1").unwrap();
        assert_eq!(admission.remaining_us, 50);
        assert_eq!(
            admission.enforcement_class,
            LeaseEnforcementClass::HostProcessOnly
        );
        assert!(!admission.independent_process_death_enforcement);
    }

    #[test]
    fn expired_lease_is_rejected() {
        let mut guard = CommandLeaseGuard::new(500);
        let lease = lease(1);
        assert!(matches!(
            guard.admit(&lease, 1_100, "cmd-1"),
            Err(LeaseRejection::Expired { .. })
        ));
    }

    #[test]
    fn future_issued_lease_is_rejected() {
        let mut guard = CommandLeaseGuard::new(500);
        let mut lease = lease(1);
        lease.issued_at_monotonic_us = 1_060;
        assert!(matches!(
            guard.admit(&lease, 1_050, "cmd-1"),
            Err(LeaseRejection::IssuedInFuture { .. })
        ));
    }

    #[test]
    fn replayed_or_reordered_sequence_is_rejected() {
        let mut guard = CommandLeaseGuard::new(500);
        let first = lease(2);
        guard.admit(&first, 1_050, "cmd-2").unwrap();
        let replay = lease(2);
        assert!(matches!(
            guard.admit(&replay, 1_051, "cmd-2"),
            Err(LeaseRejection::ReplayOrReorder { .. })
        ));
        let older = lease(1);
        assert!(matches!(
            guard.admit(&older, 1_051, "cmd-1"),
            Err(LeaseRejection::ReplayOrReorder { .. })
        ));
    }

    #[test]
    fn lease_must_bind_the_exact_command_identity() {
        let mut guard = CommandLeaseGuard::new(500);
        let lease = lease(1);
        assert_eq!(
            guard.admit(&lease, 1_050, "different-command"),
            Err(LeaseRejection::CommandIdentityMismatch)
        );
    }

    #[test]
    fn overlong_lease_is_rejected() {
        let mut guard = CommandLeaseGuard::new(50);
        let lease = lease(1);
        assert!(matches!(
            guard.admit(&lease, 1_020, "cmd-1"),
            Err(LeaseRejection::DurationExceedsPolicy { .. })
        ));
    }

    #[test]
    fn missing_fallback_identity_is_rejected() {
        let mut guard = CommandLeaseGuard::new(500);
        let mut lease = lease(1);
        lease.fallback_profile_id.clear();
        assert_eq!(
            guard.admit(&lease, 1_050, "cmd-1"),
            Err(LeaseRejection::MissingFallbackProfileIdentity)
        );
    }
}
