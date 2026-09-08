// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Uncertainty-aware trusted-time binding for safety-profile authorization commits.
//!
//! An authorization window is only meaningful if the commit boundary knows how
//! trustworthy the local clock is. This module wraps the exact root/head commit
//! preconditions with a stable clock-source identity, monotone clock epoch, and an
//! uncertainty interval that must fit wholly inside the authorization window.

use crate::admission::SafetyProfileAuthorizationHead;
use crate::commit_preconditions::{
    ProfileAuthorityRootSnapshot, SafetyProfileAuthorizationCommitError,
    SafetyProfileAuthorizationCommitPreconditions, SafetyProfileAuthorizationCommitState,
};
use thiserror::Error;

/// One trusted local time observation expressed as a closed uncertainty interval.
///
/// The real wall-clock time is asserted to lie somewhere in
/// `[earliest_unix_ms, latest_unix_ms]`. `source_id` identifies the clock authority
/// or discipline source and `epoch` changes whenever that source is reset,
/// reprovisioned, or loses continuity in a way that invalidates in-flight checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustedAuthorizationClockObservation {
    source_id: String,
    epoch: u64,
    earliest_unix_ms: i64,
    latest_unix_ms: i64,
}

impl TrustedAuthorizationClockObservation {
    pub fn new(
        source_id: impl Into<String>,
        epoch: u64,
        earliest_unix_ms: i64,
        latest_unix_ms: i64,
    ) -> Result<Self, TrustedAuthorizationTimeError> {
        let source_id = source_id.into();
        if source_id.trim().is_empty() {
            return Err(TrustedAuthorizationTimeError::EmptyClockSourceId);
        }
        if epoch == 0 {
            return Err(TrustedAuthorizationTimeError::ZeroClockEpoch);
        }
        if earliest_unix_ms > latest_unix_ms {
            return Err(TrustedAuthorizationTimeError::InvalidClockInterval {
                earliest_unix_ms,
                latest_unix_ms,
            });
        }
        Ok(Self {
            source_id,
            epoch,
            earliest_unix_ms,
            latest_unix_ms,
        })
    }

    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn earliest_unix_ms(&self) -> i64 {
        self.earliest_unix_ms
    }

    pub fn latest_unix_ms(&self) -> i64 {
        self.latest_unix_ms
    }
}

/// Production-oriented wrapper around exact commit preconditions that also binds
/// the clock authority observed when the commit attempt was prepared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeBoundSafetyProfileAuthorizationCommitPreconditions {
    inner: SafetyProfileAuthorizationCommitPreconditions,
    expected_clock_source_id: String,
    expected_clock_epoch: u64,
}

impl TimeBoundSafetyProfileAuthorizationCommitPreconditions {
    /// Bind exact commit preconditions to one trusted clock lineage.
    ///
    /// The entire initial uncertainty interval must already fit inside the signed
    /// authorization validity window. Merely having the midpoint or nominal time
    /// inside the window is insufficient.
    pub fn new(
        inner: SafetyProfileAuthorizationCommitPreconditions,
        checked_at: &TrustedAuthorizationClockObservation,
    ) -> Result<Self, TrustedAuthorizationTimeError> {
        validate_entire_interval(&inner, checked_at)?;
        Ok(Self {
            inner,
            expected_clock_source_id: checked_at.source_id().to_owned(),
            expected_clock_epoch: checked_at.epoch(),
        })
    }

    pub fn inner(&self) -> &SafetyProfileAuthorizationCommitPreconditions {
        &self.inner
    }

    pub fn expected_clock_source_id(&self) -> &str {
        &self.expected_clock_source_id
    }

    pub fn expected_clock_epoch(&self) -> u64 {
        self.expected_clock_epoch
    }

    /// Recheck root, head, trusted-clock lineage, and the entire current time
    /// uncertainty interval immediately before persistence.
    ///
    /// If the exact candidate head is already present, idempotent recovery wins
    /// before clock/root freshness checks, matching the lower-level commit contract:
    /// this only acknowledges a historical write and does not assert that the
    /// authorization is currently executable.
    pub fn recheck_commit_observation(
        &self,
        current_clock: &TrustedAuthorizationClockObservation,
        current_root: &ProfileAuthorityRootSnapshot,
        current_head: &SafetyProfileAuthorizationHead,
    ) -> Result<SafetyProfileAuthorizationCommitState, TrustedAuthorizationTimeError> {
        if current_head == self.inner.candidate_head() {
            return Ok(SafetyProfileAuthorizationCommitState::AlreadyCommitted);
        }

        if current_clock.source_id() != self.expected_clock_source_id
            || current_clock.epoch() != self.expected_clock_epoch
        {
            return Err(TrustedAuthorizationTimeError::ClockLineageChanged {
                expected_source_id: self.expected_clock_source_id.clone(),
                expected_epoch: self.expected_clock_epoch,
                observed_source_id: current_clock.source_id().to_owned(),
                observed_epoch: current_clock.epoch(),
            });
        }

        validate_entire_interval(&self.inner, current_clock)?;

        // The complete uncertainty interval is already proven admissible, so using
        // the conservative latest bound for the lower exact-time primitive cannot
        // make an invalid interval appear valid.
        Ok(self.inner.recheck_commit_observation(
            current_clock.latest_unix_ms(),
            current_root,
            current_head,
        )?)
    }
}

fn validate_entire_interval(
    preconditions: &SafetyProfileAuthorizationCommitPreconditions,
    clock: &TrustedAuthorizationClockObservation,
) -> Result<(), TrustedAuthorizationTimeError> {
    let valid_from = preconditions.valid_from_unix_ms();
    let valid_until = preconditions.valid_until_unix_ms();

    if clock.latest_unix_ms() < valid_from {
        return Err(TrustedAuthorizationTimeError::DefinitelyNotYetValid {
            latest_unix_ms: clock.latest_unix_ms(),
            valid_from_unix_ms: valid_from,
        });
    }
    if clock.earliest_unix_ms() >= valid_until {
        return Err(TrustedAuthorizationTimeError::DefinitelyExpired {
            earliest_unix_ms: clock.earliest_unix_ms(),
            valid_until_unix_ms: valid_until,
        });
    }
    if clock.earliest_unix_ms() < valid_from || clock.latest_unix_ms() >= valid_until {
        return Err(TrustedAuthorizationTimeError::UncertaintyCrossesValidityBoundary {
            earliest_unix_ms: clock.earliest_unix_ms(),
            latest_unix_ms: clock.latest_unix_ms(),
            valid_from_unix_ms: valid_from,
            valid_until_unix_ms: valid_until,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TrustedAuthorizationTimeError {
    #[error(transparent)]
    Commit(#[from] SafetyProfileAuthorizationCommitError),
    #[error("trusted authorization clock source id must not be empty")]
    EmptyClockSourceId,
    #[error("trusted authorization clock epoch must be greater than zero")]
    ZeroClockEpoch,
    #[error("invalid trusted clock interval: earliest {earliest_unix_ms} > latest {latest_unix_ms}")]
    InvalidClockInterval {
        earliest_unix_ms: i64,
        latest_unix_ms: i64,
    },
    #[error("trusted clock lineage changed: expected {expected_source_id}@{expected_epoch}, observed {observed_source_id}@{observed_epoch}")]
    ClockLineageChanged {
        expected_source_id: String,
        expected_epoch: u64,
        observed_source_id: String,
        observed_epoch: u64,
    },
    #[error("authorization is definitely not yet valid: latest possible time {latest_unix_ms} < valid-from {valid_from_unix_ms}")]
    DefinitelyNotYetValid {
        latest_unix_ms: i64,
        valid_from_unix_ms: i64,
    },
    #[error("authorization is definitely expired: earliest possible time {earliest_unix_ms} >= valid-until {valid_until_unix_ms}")]
    DefinitelyExpired {
        earliest_unix_ms: i64,
        valid_until_unix_ms: i64,
    },
    #[error("trusted clock uncertainty [{earliest_unix_ms}, {latest_unix_ms}] crosses authorization validity [{valid_from_unix_ms}, {valid_until_unix_ms})")]
    UncertaintyCrossesValidityBoundary {
        earliest_unix_ms: i64,
        latest_unix_ms: i64,
        valid_from_unix_ms: i64,
        valid_until_unix_ms: i64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::{
        PolicyCheckedSafetyProfileAuthorization, SafetyProfileAuthorizationAdmissionPolicy,
        SafetyProfileAuthorizationHead,
    };
    use crate::authorization::{ProfileAuthorityRootDigest, SafetyProfileAuthorizationSubject};
    use crate::commit_preconditions::ProfileAuthorityRootSnapshot;
    use crate::transition::SafetyProfileAuthorizationTransition;
    use crate::{
        ComponentRequirement, SafetyConfigurationProfile, SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1,
    };

    fn root(byte: u8) -> ProfileAuthorityRootDigest {
        ProfileAuthorityRootDigest::Blake3_256([byte; 32])
    }

    fn profile() -> SafetyConfigurationProfile {
        SafetyConfigurationProfile {
            schema_version: SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1.to_owned(),
            profile_id: "compute-commons-autonomous-node-v1".to_owned(),
            hardware_inventory: ComponentRequirement::Required,
            firmware: ComponentRequirement::Required,
            software_closure: ComponentRequirement::Required,
            electrical_topology: ComponentRequirement::Required,
            thermal_topology: ComponentRequirement::Required,
            protection_settings: ComponentRequirement::Required,
            sensor_map: ComponentRequirement::Required,
            actuator_map: ComponentRequirement::Required,
            calibration: ComponentRequirement::Required,
            network_topology: ComponentRequirement::Required,
        }
    }

    fn checked_bootstrap() -> (
        PolicyCheckedSafetyProfileAuthorization,
        ProfileAuthorityRootSnapshot,
    ) {
        let profile = profile();
        let subject = SafetyProfileAuthorizationSubject::new(
            "auth-1",
            "facility-profile-root-v1",
            root(0x33),
            "compute-campus",
            1,
            1_000,
            2_000,
            &profile,
        )
        .unwrap();
        let transition = SafetyProfileAuthorizationTransition::bootstrap(subject).unwrap();
        let policy = SafetyProfileAuthorizationAdmissionPolicy::new(
            "compute-campus",
            root(0x33),
            SafetyProfileAuthorizationHead::Uninitialized,
        )
        .unwrap();
        let checked = policy.check(1_500, &transition, &profile).unwrap();
        let root_snapshot =
            ProfileAuthorityRootSnapshot::new("facility-profile-root-v1", root(0x33), 1).unwrap();
        (checked, root_snapshot)
    }

    fn clock(
        source: &str,
        epoch: u64,
        earliest: i64,
        latest: i64,
    ) -> TrustedAuthorizationClockObservation {
        TrustedAuthorizationClockObservation::new(source, epoch, earliest, latest).unwrap()
    }

    fn time_bound() -> (
        PolicyCheckedSafetyProfileAuthorization,
        ProfileAuthorityRootSnapshot,
        TimeBoundSafetyProfileAuthorizationCommitPreconditions,
    ) {
        let (checked, root_snapshot) = checked_bootstrap();
        let base = SafetyProfileAuthorizationCommitPreconditions::from_policy_checked(
            &checked,
            root_snapshot.clone(),
        )
        .unwrap();
        let bounded = TimeBoundSafetyProfileAuthorizationCommitPreconditions::new(
            base,
            &clock("secure-rtc-v1", 7, 1_400, 1_600),
        )
        .unwrap();
        (checked, root_snapshot, bounded)
    }

    #[test]
    fn entire_uncertainty_interval_inside_window_is_ready() {
        let (checked, root_snapshot, bounded) = time_bound();
        assert_eq!(
            bounded
                .recheck_commit_observation(
                    &clock("secure-rtc-v1", 7, 1_450, 1_650),
                    &root_snapshot,
                    checked.expected_predecessor_head(),
                )
                .unwrap(),
            SafetyProfileAuthorizationCommitState::ReadyToCommit
        );
    }

    #[test]
    fn uncertainty_crossing_valid_from_fails_closed() {
        let (checked, root_snapshot) = checked_bootstrap();
        let base = SafetyProfileAuthorizationCommitPreconditions::from_policy_checked(
            &checked,
            root_snapshot,
        )
        .unwrap();
        assert!(matches!(
            TimeBoundSafetyProfileAuthorizationCommitPreconditions::new(
                base,
                &clock("secure-rtc-v1", 7, 900, 1_100),
            ),
            Err(TrustedAuthorizationTimeError::UncertaintyCrossesValidityBoundary { .. })
        ));
    }

    #[test]
    fn uncertainty_crossing_expiry_fails_closed() {
        let (checked, root_snapshot, bounded) = time_bound();
        assert!(matches!(
            bounded.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_900, 2_050),
                &root_snapshot,
                checked.expected_predecessor_head(),
            ),
            Err(TrustedAuthorizationTimeError::UncertaintyCrossesValidityBoundary { .. })
        ));
    }

    #[test]
    fn clock_epoch_change_invalidates_in_flight_commit() {
        let (checked, root_snapshot, bounded) = time_bound();
        assert!(matches!(
            bounded.recheck_commit_observation(
                &clock("secure-rtc-v1", 8, 1_450, 1_650),
                &root_snapshot,
                checked.expected_predecessor_head(),
            ),
            Err(TrustedAuthorizationTimeError::ClockLineageChanged { .. })
        ));
    }

    #[test]
    fn clock_source_change_invalidates_in_flight_commit() {
        let (checked, root_snapshot, bounded) = time_bound();
        assert!(matches!(
            bounded.recheck_commit_observation(
                &clock("network-time-v2", 7, 1_450, 1_650),
                &root_snapshot,
                checked.expected_predecessor_head(),
            ),
            Err(TrustedAuthorizationTimeError::ClockLineageChanged { .. })
        ));
    }

    #[test]
    fn already_committed_remains_idempotent_history_after_clock_change() {
        let (checked, root_snapshot, bounded) = time_bound();
        let changed_clock = clock("different-clock", 99, 9_000, 9_100);
        let changed_root = ProfileAuthorityRootSnapshot::new("new-root", root(0x44), 9).unwrap();

        assert_eq!(
            bounded
                .recheck_commit_observation(
                    &changed_clock,
                    &changed_root,
                    checked.candidate_head(),
                )
                .unwrap(),
            SafetyProfileAuthorizationCommitState::AlreadyCommitted
        );
        let _ = root_snapshot;
    }

    #[test]
    fn clock_observation_rejects_invalid_shape() {
        assert_eq!(
            TrustedAuthorizationClockObservation::new("", 1, 1_000, 1_000),
            Err(TrustedAuthorizationTimeError::EmptyClockSourceId)
        );
        assert_eq!(
            TrustedAuthorizationClockObservation::new("rtc", 0, 1_000, 1_000),
            Err(TrustedAuthorizationTimeError::ZeroClockEpoch)
        );
        assert_eq!(
            TrustedAuthorizationClockObservation::new("rtc", 1, 2_000, 1_000),
            Err(TrustedAuthorizationTimeError::InvalidClockInterval {
                earliest_unix_ms: 2_000,
                latest_unix_ms: 1_000,
            })
        );
    }
}
