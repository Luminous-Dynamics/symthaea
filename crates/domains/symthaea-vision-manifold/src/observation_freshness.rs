// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Observation freshness and visibility lifecycle for persistent visual beliefs.
//!
//! The central invariant is that prediction may explain an unseen interval but can never refresh
//! the timestamp/frame of the last real observation. Freshness therefore ages against sensor time
//! even while prediction confidence remains high.

use serde::Serialize;
use std::{cmp::Ordering, fmt};

use crate::entity_identity::VisualEntityHypothesisRef;
use crate::epistemic::{VisualEvidence, VisualObservationRef, VisualOrigin};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityVisibilityState {
    /// Supported by a current observation and not materially occluded.
    Visible,
    /// Supported by a current observation but only partially visible.
    PartiallyOccluded,
    /// No sufficiently recent observation; no prediction is currently asserted.
    Unobserved,
    /// No sufficiently recent observation, but a prediction supports continued presence.
    OccludedPredicted,
    /// Observation age exceeded the configured retention horizon.
    Lost,
    /// Explicitly removed from active belief tracking. This state is terminal in v1.
    Retired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ObservationFreshnessPolicy {
    unobserved_after_us: u64,
    lost_after_us: u64,
}

impl ObservationFreshnessPolicy {
    pub fn new(unobserved_after_us: u64, lost_after_us: u64) -> Result<Self, FreshnessError> {
        if lost_after_us == 0 || unobserved_after_us >= lost_after_us {
            return Err(FreshnessError::InvalidPolicy);
        }
        Ok(Self {
            unobserved_after_us,
            lost_after_us,
        })
    }

    pub const fn unobserved_after_us(self) -> u64 {
        self.unobserved_after_us
    }

    pub const fn lost_after_us(self) -> u64 {
        self.lost_after_us
    }
}

/// Observation-backed freshness state for one belief-layer entity hypothesis.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EntityObservationFreshness {
    subject: VisualEntityHypothesisRef,
    last_observed: VisualObservationRef,
    last_perception_evidence: VisualEvidence,
    state: EntityVisibilityState,
    latest_prediction: Option<VisualEvidence>,
}

impl EntityObservationFreshness {
    /// Initialize from a current perception inference that cites the exact observation.
    pub fn new_observed(
        subject: VisualEntityHypothesisRef,
        observation: VisualObservationRef,
        perception_evidence: VisualEvidence,
        partially_occluded: bool,
    ) -> Result<Self, FreshnessError> {
        validate_current_perception(&perception_evidence, observation)?;
        Ok(Self {
            subject,
            last_observed: observation,
            last_perception_evidence: perception_evidence,
            state: if partially_occluded {
                EntityVisibilityState::PartiallyOccluded
            } else {
                EntityVisibilityState::Visible
            },
            latest_prediction: None,
        })
    }

    pub const fn subject(&self) -> VisualEntityHypothesisRef {
        self.subject
    }

    pub const fn last_observed(&self) -> VisualObservationRef {
        self.last_observed
    }

    pub fn last_perception_evidence(&self) -> &VisualEvidence {
        &self.last_perception_evidence
    }

    pub const fn state(&self) -> EntityVisibilityState {
        self.state
    }

    pub fn latest_prediction(&self) -> Option<&VisualEvidence> {
        self.latest_prediction.as_ref()
    }

    /// Refresh the last-real-observation marker from a newer/current perception inference.
    ///
    /// Remembered or generated evidence cannot call this transition successfully.
    pub fn refresh_observation(
        &mut self,
        observation: VisualObservationRef,
        perception_evidence: VisualEvidence,
        partially_occluded: bool,
    ) -> Result<(), FreshnessError> {
        if self.state == EntityVisibilityState::Retired {
            return Err(FreshnessError::RetiredStateIsTerminal);
        }
        validate_current_perception(&perception_evidence, observation)?;
        validate_forward_observation(self.last_observed, observation)?;

        self.last_observed = observation;
        self.last_perception_evidence = perception_evidence;
        self.latest_prediction = None;
        self.state = if partially_occluded {
            EntityVisibilityState::PartiallyOccluded
        } else {
            EntityVisibilityState::Visible
        };
        Ok(())
    }

    /// Record a prediction of continued presence without changing `last_observed`.
    ///
    /// The prediction must descend from the last real observation so an unrelated generated state
    /// cannot attach itself to this entity's occlusion interval.
    pub fn apply_occlusion_prediction(
        &mut self,
        prediction: VisualEvidence,
    ) -> Result<(), FreshnessError> {
        if self.state == EntityVisibilityState::Retired {
            return Err(FreshnessError::RetiredStateIsTerminal);
        }
        if prediction.origin() != VisualOrigin::Predicted {
            return Err(FreshnessError::OcclusionSupportMustBePredicted);
        }
        if !prediction.parent_observations().contains(&self.last_observed) {
            return Err(FreshnessError::PredictionMissingLastObservationLineage);
        }
        self.latest_prediction = Some(prediction);
        if matches!(self.state, EntityVisibilityState::Unobserved | EntityVisibilityState::Lost) {
            // A prediction may describe a lost/unseen entity, but does not itself resurrect a lost
            // belief into observed continuity. `Lost` therefore remains `Lost`.
            if self.state == EntityVisibilityState::Unobserved {
                self.state = EntityVisibilityState::OccludedPredicted;
            }
        }
        Ok(())
    }

    /// Explicitly note absence of current visual support without inventing an occlusion cause.
    pub fn mark_unobserved(&mut self) -> Result<(), FreshnessError> {
        if self.state == EntityVisibilityState::Retired {
            return Err(FreshnessError::RetiredStateIsTerminal);
        }
        if self.state != EntityVisibilityState::Lost {
            self.state = if self.latest_prediction.is_some() {
                EntityVisibilityState::OccludedPredicted
            } else {
                EntityVisibilityState::Unobserved
            };
        }
        Ok(())
    }

    /// Re-evaluate freshness at another concrete visual observation.
    ///
    /// The reference observation is used only as an evidence-bearing clock instant. It does not
    /// imply that the subject entity was present in that frame.
    pub fn evaluate_at(
        &mut self,
        reference_observation: VisualObservationRef,
        policy: ObservationFreshnessPolicy,
    ) -> Result<u64, FreshnessError> {
        if self.state == EntityVisibilityState::Retired {
            return Err(FreshnessError::RetiredStateIsTerminal);
        }
        let age_us = observation_age_us(self.last_observed, reference_observation)?;

        if age_us >= policy.lost_after_us {
            self.state = EntityVisibilityState::Lost;
        } else if age_us >= policy.unobserved_after_us {
            self.state = if self.latest_prediction.is_some() {
                EntityVisibilityState::OccludedPredicted
            } else {
                EntityVisibilityState::Unobserved
            };
        }
        Ok(age_us)
    }

    pub fn retire(&mut self) {
        self.state = EntityVisibilityState::Retired;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreshnessError {
    InvalidPolicy,
    PerceptionMustBeInferred,
    PerceptionMissingObservationLineage,
    OcclusionSupportMustBePredicted,
    PredictionMissingLastObservationLineage,
    IncomparableObservationClocks,
    RetrogradeObservation,
    RetiredStateIsTerminal,
}

impl fmt::Display for FreshnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidPolicy => {
                "freshness policy requires 0 <= unobserved_after_us < lost_after_us and nonzero lost horizon"
            }
            Self::PerceptionMustBeInferred => {
                "refreshing last_observed requires current inferred perception evidence"
            }
            Self::PerceptionMissingObservationLineage => {
                "perception evidence must cite the exact observation used to refresh last_observed"
            }
            Self::OcclusionSupportMustBePredicted => {
                "occlusion persistence support must use VisualOrigin::Predicted"
            }
            Self::PredictionMissingLastObservationLineage => {
                "occlusion prediction must descend from the entity's last real observation"
            }
            Self::IncomparableObservationClocks => {
                "freshness cannot be computed across observations with incomparable clock semantics"
            }
            Self::RetrogradeObservation => {
                "freshness/reference observation cannot precede the last real observation"
            }
            Self::RetiredStateIsTerminal => "retired visual entity freshness state is terminal",
        };
        f.write_str(message)
    }
}

impl std::error::Error for FreshnessError {}

fn validate_current_perception(
    evidence: &VisualEvidence,
    observation: VisualObservationRef,
) -> Result<(), FreshnessError> {
    if evidence.origin() != VisualOrigin::Inferred {
        return Err(FreshnessError::PerceptionMustBeInferred);
    }
    if !evidence.parent_observations().contains(&observation) {
        return Err(FreshnessError::PerceptionMissingObservationLineage);
    }
    Ok(())
}

fn validate_forward_observation(
    previous: VisualObservationRef,
    next: VisualObservationRef,
) -> Result<(), FreshnessError> {
    if previous.stream() == next.stream() {
        if matches!(previous.frame_ordering(&next), Some(Ordering::Greater)) {
            return Err(FreshnessError::RetrogradeObservation);
        }
        if let Some(Ordering::Greater) = previous.timestamp_ordering(&next) {
            return Err(FreshnessError::RetrogradeObservation);
        }
        return Ok(());
    }

    match previous.timestamp_ordering(&next) {
        Some(Ordering::Greater) => Err(FreshnessError::RetrogradeObservation),
        Some(_) => Ok(()),
        None => Err(FreshnessError::IncomparableObservationClocks),
    }
}

fn observation_age_us(
    last_observed: VisualObservationRef,
    reference: VisualObservationRef,
) -> Result<u64, FreshnessError> {
    match last_observed.timestamp_ordering(&reference) {
        Some(Ordering::Greater) => Err(FreshnessError::RetrogradeObservation),
        Some(_) => reference
            .captured_at_us()
            .checked_sub(last_observed.captured_at_us())
            .ok_or(FreshnessError::RetrogradeObservation),
        None => Err(FreshnessError::IncomparableObservationClocks),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epistemic::{VisualCaptureClock, VisualStreamRef};

    fn observation(frame: u64, time_us: u64) -> VisualObservationRef {
        VisualObservationRef::new(
            VisualStreamRef::new(5, 7).unwrap(),
            frame,
            time_us,
            VisualCaptureClock::StreamMonotonic,
        )
    }

    fn inferred(obs: VisualObservationRef) -> VisualEvidence {
        VisualEvidence::inferred(vec![obs], 0.8).unwrap()
    }

    fn subject() -> VisualEntityHypothesisRef {
        VisualEntityHypothesisRef::new(11, 1).unwrap()
    }

    #[test]
    fn prediction_never_refreshes_last_observed() {
        let first = observation(1, 1_000);
        let mut freshness =
            EntityObservationFreshness::new_observed(subject(), first, inferred(first), false)
                .unwrap();
        let prediction = VisualEvidence::predicted(vec![first], 0.99).unwrap();
        freshness.apply_occlusion_prediction(prediction).unwrap();
        assert_eq!(freshness.last_observed(), first);
    }

    #[test]
    fn confident_prediction_cannot_prevent_lost_state() {
        let first = observation(1, 1_000);
        let mut freshness =
            EntityObservationFreshness::new_observed(subject(), first, inferred(first), false)
                .unwrap();
        freshness
            .apply_occlusion_prediction(VisualEvidence::predicted(vec![first], 1.0).unwrap())
            .unwrap();
        let policy = ObservationFreshnessPolicy::new(100, 500).unwrap();
        let age = freshness.evaluate_at(observation(2, 1_600), policy).unwrap();
        assert_eq!(age, 600);
        assert_eq!(freshness.state(), EntityVisibilityState::Lost);
        assert_eq!(freshness.last_observed(), first);
    }

    #[test]
    fn absence_without_prediction_is_unobserved_not_predicted_occlusion() {
        let first = observation(1, 1_000);
        let mut freshness =
            EntityObservationFreshness::new_observed(subject(), first, inferred(first), false)
                .unwrap();
        let policy = ObservationFreshnessPolicy::new(100, 500).unwrap();
        freshness.evaluate_at(observation(2, 1_200), policy).unwrap();
        assert_eq!(freshness.state(), EntityVisibilityState::Unobserved);
    }

    #[test]
    fn remembered_evidence_cannot_refresh_last_observed() {
        let first = observation(1, 1_000);
        let second = observation(2, 1_100);
        let mut freshness =
            EntityObservationFreshness::new_observed(subject(), first, inferred(first), false)
                .unwrap();
        let remembered = VisualEvidence::remembered(vec![second], 0.8).unwrap();
        assert_eq!(
            freshness.refresh_observation(second, remembered, false),
            Err(FreshnessError::PerceptionMustBeInferred)
        );
        assert_eq!(freshness.last_observed(), first);
    }

    #[test]
    fn newer_real_observation_refreshes_and_clears_prediction() {
        let first = observation(1, 1_000);
        let second = observation(2, 1_200);
        let mut freshness =
            EntityObservationFreshness::new_observed(subject(), first, inferred(first), false)
                .unwrap();
        freshness
            .apply_occlusion_prediction(VisualEvidence::predicted(vec![first], 0.9).unwrap())
            .unwrap();
        freshness.mark_unobserved().unwrap();
        assert_eq!(freshness.state(), EntityVisibilityState::OccludedPredicted);

        freshness
            .refresh_observation(second, inferred(second), true)
            .unwrap();
        assert_eq!(freshness.last_observed(), second);
        assert_eq!(freshness.state(), EntityVisibilityState::PartiallyOccluded);
        assert!(freshness.latest_prediction().is_none());
    }

    #[test]
    fn incomparable_cross_stream_clocks_fail_closed() {
        let first = observation(1, 1_000);
        let other = VisualObservationRef::new(
            VisualStreamRef::new(99, 1).unwrap(),
            1,
            2_000,
            VisualCaptureClock::DeviceLocal,
        );
        let mut freshness =
            EntityObservationFreshness::new_observed(subject(), first, inferred(first), false)
                .unwrap();
        assert_eq!(
            freshness.evaluate_at(
                other,
                ObservationFreshnessPolicy::new(100, 500).unwrap()
            ),
            Err(FreshnessError::IncomparableObservationClocks)
        );
    }

    #[test]
    fn retrograde_refresh_is_rejected() {
        let first = observation(2, 2_000);
        let older = observation(1, 1_000);
        let mut freshness =
            EntityObservationFreshness::new_observed(subject(), first, inferred(first), false)
                .unwrap();
        assert_eq!(
            freshness.refresh_observation(older, inferred(older), false),
            Err(FreshnessError::RetrogradeObservation)
        );
    }
}
