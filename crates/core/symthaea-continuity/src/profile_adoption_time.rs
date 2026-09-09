// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Clock-lineage and uncertainty binding for verifier-profile adoption commits.
//!
//! Root and persisted-head currentness are not sufficient to make an authorization
//! validity window meaningful. A fresh commit must also know which local clock
//! lineage produced its time observation and how uncertain that observation is.
//!
//! This module does not claim that a caller-constructed clock observation is proof
//! of freshness. Production must obtain observations from the designated local
//! clock/trust boundary. The type theorem here is narrower: once a clock lineage is
//! selected, fresh commit may proceed only while the same source/epoch remains in
//! use and the entire uncertainty interval is inside the adoption validity window.

use thiserror::Error;

use crate::profile_adoption_admission::VerifierProfileAdoptionHeadV1;
use crate::profile_adoption_commit::{
    VerifierProfileAdoptionCommitError, VerifierProfileAdoptionCommitPreconditionsV1,
    VerifierProfileAdoptionCommitStateV1,
};
use crate::profile_adoption_root::VerifierProfileAdoptionAuthorityRootSnapshotV1;

const CLOCK_OBSERVATION_DOMAIN: &[u8] =
    b"symthaea.continuity.verifier-profile-adoption.clock-observation.v1\0";

/// Content identity of one exact local clock observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VerifierProfileAdoptionClockObservationId([u8; 32]);

impl VerifierProfileAdoptionClockObservationId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// One local clock observation expressed as a closed uncertainty interval.
///
/// The actual wall-clock time is asserted by the local clock boundary to lie in
/// `[earliest_unix_ms, latest_unix_ms]`. `source_epoch` must change whenever the
/// source is reset, reprovisioned, or loses continuity in a way that should
/// invalidate in-flight authorization checks.
///
/// This type is deliberately non-Serde. Transport bytes cannot manufacture a
/// clock observation that downstream code mistakes for locally obtained state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierProfileAdoptionClockObservationV1 {
    source_id: String,
    source_epoch: u64,
    earliest_unix_ms: u64,
    latest_unix_ms: u64,
    observation_id: VerifierProfileAdoptionClockObservationId,
}

impl VerifierProfileAdoptionClockObservationV1 {
    pub fn new(
        source_id: impl Into<String>,
        source_epoch: u64,
        earliest_unix_ms: u64,
        latest_unix_ms: u64,
    ) -> Result<Self, VerifierProfileAdoptionTimeError> {
        let source_id = checked_text("clock_source_id", source_id.into())?;
        if source_epoch == 0 {
            return Err(VerifierProfileAdoptionTimeError::ZeroClockEpoch);
        }
        if earliest_unix_ms > latest_unix_ms {
            return Err(VerifierProfileAdoptionTimeError::InvalidClockInterval {
                earliest_unix_ms,
                latest_unix_ms,
            });
        }
        let observation_id = VerifierProfileAdoptionClockObservationId(hash_observation(
            &source_id,
            source_epoch,
            earliest_unix_ms,
            latest_unix_ms,
        ));
        Ok(Self {
            source_id,
            source_epoch,
            earliest_unix_ms,
            latest_unix_ms,
            observation_id,
        })
    }

    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    pub fn source_epoch(&self) -> u64 {
        self.source_epoch
    }

    pub fn earliest_unix_ms(&self) -> u64 {
        self.earliest_unix_ms
    }

    pub fn latest_unix_ms(&self) -> u64 {
        self.latest_unix_ms
    }

    pub fn uncertainty_ms(&self) -> u64 {
        self.latest_unix_ms - self.earliest_unix_ms
    }

    pub fn id(&self) -> VerifierProfileAdoptionClockObservationId {
        self.observation_id
    }

    pub fn validate(&self) -> Result<(), VerifierProfileAdoptionTimeError> {
        checked_text("clock_source_id", self.source_id.clone())?;
        if self.source_epoch == 0 {
            return Err(VerifierProfileAdoptionTimeError::ZeroClockEpoch);
        }
        if self.earliest_unix_ms > self.latest_unix_ms {
            return Err(VerifierProfileAdoptionTimeError::InvalidClockInterval {
                earliest_unix_ms: self.earliest_unix_ms,
                latest_unix_ms: self.latest_unix_ms,
            });
        }
        let expected = VerifierProfileAdoptionClockObservationId(hash_observation(
            &self.source_id,
            self.source_epoch,
            self.earliest_unix_ms,
            self.latest_unix_ms,
        ));
        if expected != self.observation_id {
            return Err(VerifierProfileAdoptionTimeError::ClockObservationIdentityMismatch);
        }
        Ok(())
    }
}

/// Commit preconditions strengthened with one exact local clock lineage and a
/// maximum accepted uncertainty width.
///
/// The initial clock observation is retained by identity for audit/provenance, but
/// fresh commit does not require byte-for-byte equality with that observation. It
/// requires a new/current observation from the same source+epoch whose uncertainty
/// remains within policy and whose entire interval remains within validity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeBoundVerifierProfileAdoptionCommitPreconditionsV1 {
    inner: VerifierProfileAdoptionCommitPreconditionsV1,
    expected_clock_source_id: String,
    expected_clock_epoch: u64,
    checked_clock_observation_id: VerifierProfileAdoptionClockObservationId,
    max_uncertainty_ms: u64,
}

impl TimeBoundVerifierProfileAdoptionCommitPreconditionsV1 {
    pub fn new(
        inner: VerifierProfileAdoptionCommitPreconditionsV1,
        checked_at: &VerifierProfileAdoptionClockObservationV1,
        max_uncertainty_ms: u64,
    ) -> Result<Self, VerifierProfileAdoptionTimeError> {
        checked_at.validate()?;
        validate_entire_interval(&inner, checked_at, max_uncertainty_ms)?;
        Ok(Self {
            inner,
            expected_clock_source_id: checked_at.source_id().to_owned(),
            expected_clock_epoch: checked_at.source_epoch(),
            checked_clock_observation_id: checked_at.id(),
            max_uncertainty_ms,
        })
    }

    pub fn inner(&self) -> &VerifierProfileAdoptionCommitPreconditionsV1 {
        &self.inner
    }

    pub fn expected_clock_source_id(&self) -> &str {
        &self.expected_clock_source_id
    }

    pub fn expected_clock_epoch(&self) -> u64 {
        self.expected_clock_epoch
    }

    pub fn checked_clock_observation_id(&self) -> VerifierProfileAdoptionClockObservationId {
        self.checked_clock_observation_id
    }

    pub fn max_uncertainty_ms(&self) -> u64 {
        self.max_uncertainty_ms
    }

    /// Recheck clock lineage/uncertainty together with the lower root/head commit
    /// theorem immediately before persistence.
    ///
    /// Exact candidate-head equality remains historical idempotent recovery and is
    /// intentionally recognized before present-time/root freshness checks. It says
    /// only that this exact candidate appears already committed, not that the
    /// verifier remains authorized now.
    pub fn recheck_commit_observation(
        &self,
        current_clock: &VerifierProfileAdoptionClockObservationV1,
        current_root: &VerifierProfileAdoptionAuthorityRootSnapshotV1,
        current_head: &VerifierProfileAdoptionHeadV1,
    ) -> Result<VerifierProfileAdoptionCommitStateV1, VerifierProfileAdoptionTimeError> {
        if current_head == self.inner.candidate_head() {
            return Ok(VerifierProfileAdoptionCommitStateV1::AlreadyCommitted);
        }

        current_clock.validate()?;
        if current_clock.source_id() != self.expected_clock_source_id
            || current_clock.source_epoch() != self.expected_clock_epoch
        {
            return Err(VerifierProfileAdoptionTimeError::ClockLineageChanged {
                expected_source_id: self.expected_clock_source_id.clone(),
                expected_epoch: self.expected_clock_epoch,
                observed_source_id: current_clock.source_id().to_owned(),
                observed_epoch: current_clock.source_epoch(),
            });
        }

        validate_entire_interval(&self.inner, current_clock, self.max_uncertainty_ms)?;

        // The entire interval has already been proven to lie inside validity. Using
        // the conservative latest bound when delegating to the exact-time lower
        // primitive cannot turn an invalid interval into a valid one.
        Ok(self.inner.recheck_commit_observation(
            current_clock.latest_unix_ms(),
            current_root,
            current_head,
        )?)
    }
}

fn validate_entire_interval(
    preconditions: &VerifierProfileAdoptionCommitPreconditionsV1,
    clock: &VerifierProfileAdoptionClockObservationV1,
    max_uncertainty_ms: u64,
) -> Result<(), VerifierProfileAdoptionTimeError> {
    let uncertainty_ms = clock.uncertainty_ms();
    if uncertainty_ms > max_uncertainty_ms {
        return Err(VerifierProfileAdoptionTimeError::ClockUncertaintyExceedsPolicy {
            observed_ms: uncertainty_ms,
            maximum_ms: max_uncertainty_ms,
        });
    }

    let valid_from = preconditions.valid_from_unix_ms();
    let valid_until = preconditions.valid_until_unix_ms();

    if clock.latest_unix_ms() < valid_from {
        return Err(VerifierProfileAdoptionTimeError::DefinitelyNotYetValid {
            latest_unix_ms: clock.latest_unix_ms(),
            valid_from_unix_ms: valid_from,
        });
    }
    if clock.earliest_unix_ms() >= valid_until {
        return Err(VerifierProfileAdoptionTimeError::DefinitelyExpired {
            earliest_unix_ms: clock.earliest_unix_ms(),
            valid_until_unix_ms: valid_until,
        });
    }
    if clock.earliest_unix_ms() < valid_from || clock.latest_unix_ms() >= valid_until {
        return Err(
            VerifierProfileAdoptionTimeError::UncertaintyCrossesValidityBoundary {
                earliest_unix_ms: clock.earliest_unix_ms(),
                latest_unix_ms: clock.latest_unix_ms(),
                valid_from_unix_ms: valid_from,
                valid_until_unix_ms: valid_until,
            },
        );
    }
    Ok(())
}

fn checked_text(
    field: &'static str,
    value: String,
) -> Result<String, VerifierProfileAdoptionTimeError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(VerifierProfileAdoptionTimeError::BlankText { field });
    }
    if trimmed.len() > 1024 {
        return Err(VerifierProfileAdoptionTimeError::TextTooLong { field });
    }
    if trimmed.chars().any(char::is_control) {
        return Err(VerifierProfileAdoptionTimeError::ControlCharacters { field });
    }
    Ok(trimmed.to_owned())
}

fn hash_observation(
    source_id: &str,
    source_epoch: u64,
    earliest_unix_ms: u64,
    latest_unix_ms: u64,
) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(source_id.len() + 32);
    bytes.extend_from_slice(&(source_id.len() as u64).to_le_bytes());
    bytes.extend_from_slice(source_id.as_bytes());
    bytes.extend_from_slice(&source_epoch.to_le_bytes());
    bytes.extend_from_slice(&earliest_unix_ms.to_le_bytes());
    bytes.extend_from_slice(&latest_unix_ms.to_le_bytes());
    let mut hasher = blake3::Hasher::new();
    hasher.update(CLOCK_OBSERVATION_DOMAIN);
    hasher.update(&bytes);
    *hasher.finalize().as_bytes()
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierProfileAdoptionTimeError {
    #[error(transparent)]
    Commit(#[from] VerifierProfileAdoptionCommitError),
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds 1024 bytes")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("verifier-adoption clock epoch must be greater than zero")]
    ZeroClockEpoch,
    #[error("invalid verifier-adoption clock interval: earliest {earliest_unix_ms} > latest {latest_unix_ms}")]
    InvalidClockInterval {
        earliest_unix_ms: u64,
        latest_unix_ms: u64,
    },
    #[error("stored verifier-adoption clock observation identity is not canonical")]
    ClockObservationIdentityMismatch,
    #[error("verifier-adoption clock lineage changed: expected {expected_source_id}@{expected_epoch}, observed {observed_source_id}@{observed_epoch}")]
    ClockLineageChanged {
        expected_source_id: String,
        expected_epoch: u64,
        observed_source_id: String,
        observed_epoch: u64,
    },
    #[error("verifier-adoption clock uncertainty {observed_ms}ms exceeds policy maximum {maximum_ms}ms")]
    ClockUncertaintyExceedsPolicy {
        observed_ms: u64,
        maximum_ms: u64,
    },
    #[error("verifier adoption is definitely not yet valid: latest possible time {latest_unix_ms} < valid-from {valid_from_unix_ms}")]
    DefinitelyNotYetValid {
        latest_unix_ms: u64,
        valid_from_unix_ms: u64,
    },
    #[error("verifier adoption is definitely expired: earliest possible time {earliest_unix_ms} >= valid-until {valid_until_unix_ms}")]
    DefinitelyExpired {
        earliest_unix_ms: u64,
        valid_until_unix_ms: u64,
    },
    #[error("verifier-adoption clock uncertainty [{earliest_unix_ms}, {latest_unix_ms}] crosses validity [{valid_from_unix_ms}, {valid_until_unix_ms})")]
    UncertaintyCrossesValidityBoundary {
        earliest_unix_ms: u64,
        latest_unix_ms: u64,
        valid_from_unix_ms: u64,
        valid_until_unix_ms: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EvidenceClass, RootBoundPolicyCheckedVerifierProfileAdoptionV1,
        VerifierAdoptionScopeV1, VerifierProfileAdoptionAdmissionPolicyV1,
        VerifierProfileAdoptionAuthorityRootSnapshotV1, VerifierProfileAdoptionHeadV1,
        VerifierProfileAdoptionSubjectV1, VerifierProfileAdoptionTransitionV1, VerifierProfileV1,
    };

    fn profile() -> VerifierProfileV1 {
        VerifierProfileV1::new(
            "hardware-verifier-v1",
            [0x22; 32],
            7,
            EvidenceClass::HardwareVerified,
        )
        .unwrap()
    }

    fn root(epoch: u64) -> VerifierProfileAdoptionAuthorityRootSnapshotV1 {
        VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            epoch,
        )
        .unwrap()
    }

    fn preconditions() -> VerifierProfileAdoptionCommitPreconditionsV1 {
        let profile = profile();
        let subject = VerifierProfileAdoptionSubjectV1::new(
            "adopt-1",
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            &profile,
            1,
            1_000,
            2_000,
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
        .unwrap();
        let transition = VerifierProfileAdoptionTransitionV1::bootstrap(subject).unwrap();
        let checked = VerifierProfileAdoptionAdmissionPolicyV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            "hardware-verifier-v1",
            VerifierProfileAdoptionHeadV1::Uninitialized,
        )
        .unwrap()
        .check(1_500, &transition, &profile, None)
        .unwrap();
        let bound = RootBoundPolicyCheckedVerifierProfileAdoptionV1::bind(checked, root(9)).unwrap();
        VerifierProfileAdoptionCommitPreconditionsV1::from_root_bound(bound).unwrap()
    }

    fn clock(
        source: &str,
        epoch: u64,
        earliest: u64,
        latest: u64,
    ) -> VerifierProfileAdoptionClockObservationV1 {
        VerifierProfileAdoptionClockObservationV1::new(source, epoch, earliest, latest).unwrap()
    }

    fn time_bound() -> TimeBoundVerifierProfileAdoptionCommitPreconditionsV1 {
        TimeBoundVerifierProfileAdoptionCommitPreconditionsV1::new(
            preconditions(),
            &clock("secure-rtc-v1", 7, 1_400, 1_600),
            250,
        )
        .unwrap()
    }

    #[test]
    fn entire_uncertainty_interval_inside_window_is_ready() {
        let bounded = time_bound();
        assert_eq!(
            bounded
                .recheck_commit_observation(
                    &clock("secure-rtc-v1", 7, 1_450, 1_650),
                    &root(9),
                    bounded.inner().expected_predecessor_head(),
                )
                .unwrap(),
            VerifierProfileAdoptionCommitStateV1::ReadyToCommit
        );
    }

    #[test]
    fn clock_epoch_change_blocks_fresh_commit() {
        let bounded = time_bound();
        assert!(matches!(
            bounded.recheck_commit_observation(
                &clock("secure-rtc-v1", 8, 1_450, 1_650),
                &root(9),
                bounded.inner().expected_predecessor_head(),
            ),
            Err(VerifierProfileAdoptionTimeError::ClockLineageChanged { .. })
        ));
    }

    #[test]
    fn clock_source_change_blocks_fresh_commit() {
        let bounded = time_bound();
        assert!(matches!(
            bounded.recheck_commit_observation(
                &clock("ntp-fallback", 7, 1_450, 1_650),
                &root(9),
                bounded.inner().expected_predecessor_head(),
            ),
            Err(VerifierProfileAdoptionTimeError::ClockLineageChanged { .. })
        ));
    }

    #[test]
    fn initial_uncertainty_crossing_valid_from_fails_closed() {
        assert!(matches!(
            TimeBoundVerifierProfileAdoptionCommitPreconditionsV1::new(
                preconditions(),
                &clock("secure-rtc-v1", 7, 900, 1_100),
                250,
            ),
            Err(VerifierProfileAdoptionTimeError::UncertaintyCrossesValidityBoundary { .. })
        ));
    }

    #[test]
    fn current_uncertainty_crossing_expiry_fails_closed() {
        let bounded = time_bound();
        assert!(matches!(
            bounded.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_900, 2_050),
                &root(9),
                bounded.inner().expected_predecessor_head(),
            ),
            Err(VerifierProfileAdoptionTimeError::UncertaintyCrossesValidityBoundary { .. })
        ));
    }

    #[test]
    fn uncertainty_quality_policy_is_frozen_and_rechecked() {
        assert!(matches!(
            TimeBoundVerifierProfileAdoptionCommitPreconditionsV1::new(
                preconditions(),
                &clock("secure-rtc-v1", 7, 1_300, 1_700),
                250,
            ),
            Err(VerifierProfileAdoptionTimeError::ClockUncertaintyExceedsPolicy {
                observed_ms: 400,
                maximum_ms: 250,
            })
        ));

        let bounded = time_bound();
        assert!(matches!(
            bounded.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_300, 1_700),
                &root(9),
                bounded.inner().expected_predecessor_head(),
            ),
            Err(VerifierProfileAdoptionTimeError::ClockUncertaintyExceedsPolicy {
                observed_ms: 400,
                maximum_ms: 250,
            })
        ));
    }

    #[test]
    fn same_clock_lineage_does_not_hide_root_or_head_currentness() {
        let bounded = time_bound();
        assert!(matches!(
            bounded.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_450, 1_650),
                &root(10),
                bounded.inner().expected_predecessor_head(),
            ),
            Err(VerifierProfileAdoptionTimeError::Commit(
                VerifierProfileAdoptionCommitError::RootBinding(_)
            ))
        ));
    }

    #[test]
    fn exact_candidate_is_historical_idempotent_even_after_clock_root_and_time_change() {
        let bounded = time_bound();
        let replacement_root = VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "replacement-root",
            [0x77; 32],
            42,
        )
        .unwrap();
        let unrelated_clock = clock("unrelated-clock", 99, 9_000, 10_000);

        assert_eq!(
            bounded
                .recheck_commit_observation(
                    &unrelated_clock,
                    &replacement_root,
                    bounded.inner().candidate_head(),
                )
                .unwrap(),
            VerifierProfileAdoptionCommitStateV1::AlreadyCommitted
        );
    }

    #[test]
    fn clock_observation_identity_binds_source_epoch_and_interval() {
        let a = clock("secure-rtc-v1", 7, 1_400, 1_600);
        let same = clock("secure-rtc-v1", 7, 1_400, 1_600);
        let new_epoch = clock("secure-rtc-v1", 8, 1_400, 1_600);
        let new_interval = clock("secure-rtc-v1", 7, 1_401, 1_600);

        assert_eq!(a.id(), same.id());
        assert_ne!(a.id(), new_epoch.id());
        assert_ne!(a.id(), new_interval.id());
        a.validate().unwrap();
    }

    #[test]
    fn invalid_clock_structure_fails_closed() {
        assert!(matches!(
            VerifierProfileAdoptionClockObservationV1::new(" ", 7, 1_000, 1_100),
            Err(VerifierProfileAdoptionTimeError::BlankText { .. })
        ));
        assert_eq!(
            VerifierProfileAdoptionClockObservationV1::new("secure-rtc-v1", 0, 1_000, 1_100),
            Err(VerifierProfileAdoptionTimeError::ZeroClockEpoch)
        );
        assert_eq!(
            VerifierProfileAdoptionClockObservationV1::new("secure-rtc-v1", 7, 1_101, 1_100),
            Err(VerifierProfileAdoptionTimeError::InvalidClockInterval {
                earliest_unix_ms: 1_101,
                latest_unix_ms: 1_100,
            })
        );
    }

    #[test]
    fn time_bound_capsule_retains_exact_lower_authority_identity() {
        let bounded = time_bound();
        assert_eq!(bounded.expected_clock_source_id(), "secure-rtc-v1");
        assert_eq!(bounded.expected_clock_epoch(), 7);
        assert_eq!(bounded.max_uncertainty_ms(), 250);
        assert_eq!(
            bounded.inner().canonical_transition_bytes(),
            bounded.inner().transition().canonical_signing_bytes().unwrap()
        );
        assert_eq!(
            bounded.inner().expected_root_snapshot().provisioning_epoch(),
            9
        );
    }
}
