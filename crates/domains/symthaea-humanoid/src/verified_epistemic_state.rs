// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verifier-owned current epistemic state for humanoid execution.
//!
//! `StateEstimatorReport` and `HumanoidStateUncertaintyEnvelope` are serializable
//! evidence. They must not become positive physical authority merely because a
//! caller can construct an `accepted = true` report. This module keeps the real
//! fused estimator as the owner of the positive transition and adds a short-lived
//! process-local permit bound to one estimator session, validation epoch, and
//! exact accepted measurement sequence.

use std::num::NonZeroU64;

use crate::execution::HumanoidAuthorityEnvelope;
use crate::morphology::HumanoidMorphology;
use crate::state_estimation::{
    FusedHumanoidStateEstimator, ProprioceptiveMeasurement, StateEstimatorConfig,
    StateEstimatorError,
};
use crate::state_uncertainty::HumanoidStateUncertaintyEnvelope;
use crate::types::HumanoidState;

/// Opaque proof that the owner estimator accepted one measurement at one local
/// validation epoch. It must be rechecked against the same owner before use.
#[derive(Debug)]
pub struct HumanoidEpistemicStatePermit {
    estimator_session_id: String,
    validation_epoch: NonZeroU64,
    sequence: u64,
    sampled_at_s: f64,
    received_at_s: f64,
    uncertainty: HumanoidStateUncertaintyEnvelope,
}

impl HumanoidEpistemicStatePermit {
    pub fn estimator_session_id(&self) -> &str {
        &self.estimator_session_id
    }

    pub const fn validation_epoch(&self) -> NonZeroU64 {
        self.validation_epoch
    }

    pub const fn sequence(&self) -> u64 {
        self.sequence
    }
}

/// Process-local current epistemic witness accepted against the current owner
/// state and a caller-supplied local time observation.
///
/// Intentionally not `Clone`, `Copy`, `Serialize`, or `Deserialize`. A later
/// physical execution admission should consume this value and re-check its
/// `valid_until_s` against the same trusted local time domain.
#[derive(Debug)]
pub struct VerifiedCurrentHumanoidEpistemicState {
    estimator_session_id: String,
    validation_epoch: NonZeroU64,
    sequence: u64,
    valid_until_s: f64,
    uncertainty: HumanoidStateUncertaintyEnvelope,
}

impl VerifiedCurrentHumanoidEpistemicState {
    pub fn estimator_session_id(&self) -> &str {
        &self.estimator_session_id
    }

    pub const fn validation_epoch(&self) -> NonZeroU64 {
        self.validation_epoch
    }

    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub const fn valid_until_s(&self) -> f64 {
        self.valid_until_s
    }

    pub const fn uncertainty(&self) -> HumanoidStateUncertaintyEnvelope {
        self.uncertainty
    }

    pub fn epistemic_authority(&self) -> f32 {
        self.uncertainty.epistemic_authority()
    }

    pub fn is_current_at(&self, now_s: f64) -> bool {
        now_s.is_finite() && now_s <= self.valid_until_s
    }

    /// Epistemic state may only tighten an already established authority value.
    pub fn restrict_authority(
        &self,
        authority: HumanoidAuthorityEnvelope,
    ) -> HumanoidAuthorityEnvelope {
        self.uncertainty.restrict_authority(authority)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifiedEpistemicStateError {
    InvalidEstimatorSession,
    InvalidEstimatorConfig,
    EpochExhausted,
    Estimator(StateEstimatorError),
    NoCurrentEstimate,
    SessionMismatch,
    StaleValidationEpoch,
    SequenceMismatch,
    NonFiniteCurrentTime,
    CurrentTimeBeforeReceipt,
    StaleAtUse,
}

impl std::fmt::Display for VerifiedEpistemicStateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidEstimatorSession => "estimator session identity is empty",
            Self::InvalidEstimatorConfig => "estimator configuration is invalid for authority use",
            Self::EpochExhausted => "estimator validation epoch exhausted",
            Self::Estimator(_) => "state estimator rejected the measurement",
            Self::NoCurrentEstimate => "no current accepted estimator sequence exists",
            Self::SessionMismatch => "epistemic permit belongs to another estimator session",
            Self::StaleValidationEpoch => "epistemic permit belongs to an older validation epoch",
            Self::SequenceMismatch => "epistemic permit is not bound to the current accepted sequence",
            Self::NonFiniteCurrentTime => "current epistemic time observation is non-finite",
            Self::CurrentTimeBeforeReceipt => "current epistemic time precedes measurement receipt",
            Self::StaleAtUse => "accepted state estimate is stale at use time",
        };
        f.write_str(message)
    }
}

impl std::error::Error for VerifiedEpistemicStateError {}

/// Stateful owner of one fused-estimator session and its live validation epoch.
///
/// Successful estimator updates advance the epoch. Rejected updates are
/// transactional and leave the prior epoch current until its natural freshness
/// deadline. Reset also advances the epoch and invalidates outstanding permits.
#[derive(Debug)]
pub struct HumanoidVerifiedStateEstimator {
    estimator: FusedHumanoidStateEstimator,
    config: StateEstimatorConfig,
    estimator_session_id: String,
    validation_epoch: u64,
    epoch_exhausted: bool,
    current_sequence: Option<u64>,
}

impl HumanoidVerifiedStateEstimator {
    pub fn new(
        morphology: HumanoidMorphology,
        estimator_session_id: impl Into<String>,
        config: StateEstimatorConfig,
    ) -> Result<Self, VerifiedEpistemicStateError> {
        let estimator_session_id = estimator_session_id.into();
        if estimator_session_id.trim().is_empty() {
            return Err(VerifiedEpistemicStateError::InvalidEstimatorSession);
        }
        if !config_valid_for_authority(&config) {
            return Err(VerifiedEpistemicStateError::InvalidEstimatorConfig);
        }
        Ok(Self {
            estimator: FusedHumanoidStateEstimator::with_config(morphology, config.clone()),
            config,
            estimator_session_id,
            validation_epoch: 0,
            epoch_exhausted: false,
            current_sequence: None,
        })
    }

    pub fn estimator_session_id(&self) -> &str {
        &self.estimator_session_id
    }

    pub const fn validation_epoch(&self) -> u64 {
        self.validation_epoch
    }

    pub const fn current_sequence(&self) -> Option<u64> {
        self.current_sequence
    }

    pub fn estimate(&self) -> &HumanoidState {
        self.estimator.estimate()
    }

    /// Run the real fused estimator and mint a process-local permit only after
    /// that owner accepts the measurement. Public report data alone cannot call
    /// this transition.
    pub fn update(
        &mut self,
        measurement: &ProprioceptiveMeasurement,
    ) -> Result<HumanoidEpistemicStatePermit, VerifiedEpistemicStateError> {
        let (_, report) = self
            .estimator
            .update(measurement)
            .map_err(VerifiedEpistemicStateError::Estimator)?;
        let uncertainty = HumanoidStateUncertaintyEnvelope::from_estimator_report(
            report,
            &self.config,
        );
        let epoch = self.advance_epoch()?;
        self.current_sequence = Some(report.sequence);
        Ok(HumanoidEpistemicStatePermit {
            estimator_session_id: self.estimator_session_id.clone(),
            validation_epoch: epoch,
            sequence: report.sequence,
            sampled_at_s: measurement.sampled_at_s,
            received_at_s: measurement.received_at_s,
            uncertainty,
        })
    }

    /// Recheck a one-update permit against the same current estimator owner and
    /// current local time before allowing it to become an execution input.
    pub fn accept_current(
        &self,
        permit: HumanoidEpistemicStatePermit,
        now_s: f64,
    ) -> Result<VerifiedCurrentHumanoidEpistemicState, VerifiedEpistemicStateError> {
        if self.epoch_exhausted {
            return Err(VerifiedEpistemicStateError::EpochExhausted);
        }
        let Some(current_sequence) = self.current_sequence else {
            return Err(VerifiedEpistemicStateError::NoCurrentEstimate);
        };
        if permit.estimator_session_id != self.estimator_session_id {
            return Err(VerifiedEpistemicStateError::SessionMismatch);
        }
        if permit.validation_epoch.get() != self.validation_epoch {
            return Err(VerifiedEpistemicStateError::StaleValidationEpoch);
        }
        if permit.sequence != current_sequence || permit.uncertainty.sequence != current_sequence {
            return Err(VerifiedEpistemicStateError::SequenceMismatch);
        }
        if !now_s.is_finite() {
            return Err(VerifiedEpistemicStateError::NonFiniteCurrentTime);
        }
        if now_s < permit.received_at_s {
            return Err(VerifiedEpistemicStateError::CurrentTimeBeforeReceipt);
        }
        let valid_until_s = permit.sampled_at_s + self.config.maximum_measurement_age_s;
        if !valid_until_s.is_finite() || now_s > valid_until_s {
            return Err(VerifiedEpistemicStateError::StaleAtUse);
        }

        Ok(VerifiedCurrentHumanoidEpistemicState {
            estimator_session_id: permit.estimator_session_id,
            validation_epoch: permit.validation_epoch,
            sequence: permit.sequence,
            valid_until_s,
            uncertainty: permit.uncertainty,
        })
    }

    /// Resetting the fused estimator invalidates every permit from the previous
    /// estimator state. It never resumes old epistemic authority automatically.
    pub fn reset(&mut self, state: &HumanoidState) -> Result<(), VerifiedEpistemicStateError> {
        self.estimator
            .reset(state)
            .map_err(VerifiedEpistemicStateError::Estimator)?;
        let _ = self.advance_epoch()?;
        self.current_sequence = None;
        Ok(())
    }

    fn advance_epoch(&mut self) -> Result<NonZeroU64, VerifiedEpistemicStateError> {
        if self.epoch_exhausted {
            return Err(VerifiedEpistemicStateError::EpochExhausted);
        }
        let Some(next) = self.validation_epoch.checked_add(1) else {
            self.epoch_exhausted = true;
            return Err(VerifiedEpistemicStateError::EpochExhausted);
        };
        self.validation_epoch = next;
        NonZeroU64::new(next).ok_or(VerifiedEpistemicStateError::EpochExhausted)
    }
}

fn config_valid_for_authority(config: &StateEstimatorConfig) -> bool {
    let unit = [
        config.orientation_correction,
        config.linear_velocity_correction,
        config.angular_velocity_correction,
        config.joint_position_correction,
        config.joint_velocity_correction,
        config.double_support_velocity_damping,
    ];
    let positive = [
        config.maximum_measurement_age_s,
        config.maximum_dt_s,
        config.maximum_orientation_innovation_rad,
        config.maximum_linear_velocity_innovation_mps,
        config.maximum_joint_position_innovation_rad,
    ];
    unit.into_iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
        && positive
            .into_iter()
            .all(|value| value.is_finite() && value > 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contact::ContactFrame;

    fn estimator() -> HumanoidVerifiedStateEstimator {
        HumanoidVerifiedStateEstimator::new(
            HumanoidMorphology::Dmc21,
            "estimator-session-17",
            StateEstimatorConfig::default(),
        )
        .unwrap()
    }

    fn measurement(sequence: u64, timestamp: f64) -> ProprioceptiveMeasurement {
        let mut state = HumanoidState::standing();
        state.timestamp = timestamp;
        let contact = ContactFrame::estimated_from_state(&state, 0.05);
        ProprioceptiveMeasurement::from_simulator(
            HumanoidMorphology::Dmc21,
            sequence,
            state,
            contact,
        )
    }

    #[test]
    fn accepted_estimator_update_mints_current_process_local_witness() {
        let mut estimator = estimator();
        let permit = estimator.update(&measurement(1, 1.0)).unwrap();
        assert_eq!(permit.validation_epoch().get(), 1);
        let verified = estimator.accept_current(permit, 1.01).unwrap();
        assert_eq!(verified.sequence(), 1);
        assert_eq!(verified.estimator_session_id(), "estimator-session-17");
        assert!(verified.is_current_at(1.01));
        assert!(verified.epistemic_authority() > 0.0);
    }

    #[test]
    fn later_accepted_update_invalidates_previous_permit() {
        let mut estimator = estimator();
        let old = estimator.update(&measurement(1, 1.0)).unwrap();
        estimator.update(&measurement(2, 1.01)).unwrap();
        assert_eq!(
            estimator.accept_current(old, 1.02).unwrap_err(),
            VerifiedEpistemicStateError::StaleValidationEpoch
        );
    }

    #[test]
    fn stale_estimate_is_rejected_at_use_even_if_it_was_fresh_when_received() {
        let mut estimator = estimator();
        let permit = estimator.update(&measurement(1, 1.0)).unwrap();
        let now = 1.0 + StateEstimatorConfig::default().maximum_measurement_age_s + 0.001;
        assert_eq!(
            estimator.accept_current(permit, now).unwrap_err(),
            VerifiedEpistemicStateError::StaleAtUse
        );
    }

    #[test]
    fn rejected_update_does_not_invalidate_prior_current_epoch() {
        let mut estimator = estimator();
        let prior = estimator.update(&measurement(1, 1.0)).unwrap();
        let mut corrupted = measurement(2, 1.01);
        corrupted.state.root_quaternion = [0.0, 1.0, 0.0, 0.0];
        assert!(matches!(
            estimator.update(&corrupted),
            Err(VerifiedEpistemicStateError::Estimator(
                StateEstimatorError::InnovationRejected
            ))
        ));
        let verified = estimator.accept_current(prior, 1.02).unwrap();
        assert_eq!(verified.sequence(), 1);
    }

    #[test]
    fn reset_invalidates_outstanding_permit_and_requires_new_measurement() {
        let mut estimator = estimator();
        let prior = estimator.update(&measurement(1, 1.0)).unwrap();
        estimator.reset(&HumanoidState::standing()).unwrap();
        assert_eq!(
            estimator.accept_current(prior, 1.01).unwrap_err(),
            VerifiedEpistemicStateError::NoCurrentEstimate
        );
    }

    #[test]
    fn verified_epistemic_state_only_restricts_existing_authority() {
        let mut estimator = estimator();
        let permit = estimator.update(&measurement(1, 1.0)).unwrap();
        let verified = estimator.accept_current(permit, 1.01).unwrap();
        let authority = verified.restrict_authority(HumanoidAuthorityEnvelope {
            epistemic: 0.2,
            ..HumanoidAuthorityEnvelope::fully_admitted()
        });
        assert!(authority.epistemic <= 0.2);
    }
}
