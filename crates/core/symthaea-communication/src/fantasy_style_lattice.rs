// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Non-authoritative multidimensional style proposals for adult fantasy dialogue.
//!
//! Style may shape language only inside an already-active adult fantasy session
//! and an allowed topic. It cannot activate a session, unblock a topic, authorize
//! a likeness, grant physical contact, or create motor authority.

use crate::adult_fantasy_session::{AdultFantasySessionV1, AdultFantasyStopReasonV1};
use crate::fantasy_preferences::{
    FantasyPreferenceModelV1, FantasyPreferenceSourceV1, FantasyStyleDimensionV1,
};
use crate::fantasy_world_state::FantasyWorldStateV1;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const FANTASY_STYLE_LATTICE_SCHEMA_V1: &str =
    "symthaea.communication.fantasy-style-lattice.v1";

pub const ALL_FANTASY_STYLE_DIMENSIONS_V1: [FantasyStyleDimensionV1; 10] = [
    FantasyStyleDimensionV1::Romance,
    FantasyStyleDimensionV1::Playfulness,
    FantasyStyleDimensionV1::Directness,
    FantasyStyleDimensionV1::VerbalIntensity,
    FantasyStyleDimensionV1::Tenderness,
    FantasyStyleDimensionV1::Initiative,
    FantasyStyleDimensionV1::Suspense,
    FantasyStyleDimensionV1::Humor,
    FantasyStyleDimensionV1::NarrativeDensity,
    FantasyStyleDimensionV1::CallbackDensity,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FantasyStyleAxisSourceV1 {
    ExplicitTurnRequest,
    ExplicitStoredPreference,
    RepeatedExplicitFeedback,
    BehavioralInference,
    PopulationPrior,
    Unknown,
}

impl From<FantasyPreferenceSourceV1> for FantasyStyleAxisSourceV1 {
    fn from(value: FantasyPreferenceSourceV1) -> Self {
        match value {
            FantasyPreferenceSourceV1::ExplicitUserPreference => Self::ExplicitStoredPreference,
            FantasyPreferenceSourceV1::RepeatedExplicitFeedback => Self::RepeatedExplicitFeedback,
            FantasyPreferenceSourceV1::BehavioralInference => Self::BehavioralInference,
            FantasyPreferenceSourceV1::PopulationPrior => Self::PopulationPrior,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FantasyStyleAxisProposalV1 {
    pub dimension: FantasyStyleDimensionV1,
    /// `None` is a first-class unknown state. No downstream component may
    /// silently replace it with a demographic/default intensity claim.
    pub value: Option<f32>,
    pub confidence: f32,
    pub source: FantasyStyleAxisSourceV1,
    pub evidence_id: Option<String>,
}

impl FantasyStyleAxisProposalV1 {
    fn unknown(dimension: FantasyStyleDimensionV1) -> Self {
        Self {
            dimension,
            value: None,
            confidence: 0.0,
            source: FantasyStyleAxisSourceV1::Unknown,
            evidence_id: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FantasyPacingSignalV1 {
    Continue,
    Hold,
    SlowDown,
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FantasyPacingTrajectoryV1 {
    Maintain,
    Hold,
    EaseDown,
    ClarifyBeforeIncreasing,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FantasyStyleTurnOverrideV1 {
    pub dimension: FantasyStyleDimensionV1,
    pub value: f32,
}

impl FantasyStyleTurnOverrideV1 {
    pub fn new(
        dimension: FantasyStyleDimensionV1,
        value: f32,
    ) -> Result<Self, FantasyStyleLatticeErrorV1> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(FantasyStyleLatticeErrorV1::InvalidAxisValue);
        }
        Ok(Self { dimension, value })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FantasyTurnStyleRequestV1 {
    pub topic_id: String,
    pub overrides: Vec<FantasyStyleTurnOverrideV1>,
    /// Descriptive-vs-conversational balance for this turn. `0` is maximally
    /// conversational; `1` is maximally descriptive. This is creative style,
    /// not permission or topic authority.
    pub descriptive_balance: Option<f32>,
    pub pacing_signal: FantasyPacingSignalV1,
}

impl FantasyTurnStyleRequestV1 {
    pub fn new(
        topic_id: impl Into<String>,
        overrides: Vec<FantasyStyleTurnOverrideV1>,
        descriptive_balance: Option<f32>,
        pacing_signal: FantasyPacingSignalV1,
    ) -> Result<Self, FantasyStyleLatticeErrorV1> {
        let topic_id = canonical_topic_id(topic_id.into())?;
        if let Some(value) = descriptive_balance {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(FantasyStyleLatticeErrorV1::InvalidDescriptiveBalance);
            }
        }
        let mut seen = BTreeSet::new();
        for override_value in &overrides {
            if !override_value.value.is_finite() || !(0.0..=1.0).contains(&override_value.value) {
                return Err(FantasyStyleLatticeErrorV1::InvalidAxisValue);
            }
            if !seen.insert(override_value.dimension) {
                return Err(FantasyStyleLatticeErrorV1::DuplicateTurnOverride);
            }
        }
        Ok(Self {
            topic_id,
            overrides,
            descriptive_balance,
            pacing_signal,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FantasyStyleProposalV1 {
    pub proposal_epoch: u64,
    pub session_id: String,
    pub session_epoch: u64,
    pub world_id: String,
    pub scene_id: String,
    pub scene_epoch: u64,
    pub boundary_epoch: u64,
    pub topic_id: String,
    pub axes: BTreeMap<FantasyStyleDimensionV1, FantasyStyleAxisProposalV1>,
    pub descriptive_balance: Option<f32>,
    pub pacing: FantasyPacingTrajectoryV1,
    /// `Hold` and `SlowDown` prohibit a proposal from increasing the
    /// intensity-bearing axes beyond the previous proposal in the same session.
    pub must_not_increase_intensity: bool,
    /// `SlowDown` additionally requires downstream realization to decrease
    /// intensity rather than merely stay below the ceiling. The planner does not
    /// invent a universal numeric decay rate.
    pub must_decrease_intensity: bool,
    pub previous_verbal_intensity_ceiling: Option<f32>,
    pub previous_initiative_ceiling: Option<f32>,
    pub previous_suspense_ceiling: Option<f32>,
    pub uncertain_dimensions: BTreeSet<FantasyStyleDimensionV1>,
}

impl FantasyStyleProposalV1 {
    pub fn axis(&self, dimension: FantasyStyleDimensionV1) -> &FantasyStyleAxisProposalV1 {
        self.axes
            .get(&dimension)
            .expect("all style dimensions are materialized")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum FantasyStylePlanOutcomeV1 {
    Proposal(FantasyStyleProposalV1),
    SessionStopped,
}

#[derive(Debug, Default)]
pub struct FantasyStylePlannerV1 {
    proposal_epoch: u64,
    last_proposal: Option<FantasyStyleProposalV1>,
}

impl FantasyStylePlannerV1 {
    pub fn last_proposal(&self) -> Option<&FantasyStyleProposalV1> {
        self.last_proposal.as_ref()
    }

    pub fn plan_next(
        &mut self,
        session: &mut AdultFantasySessionV1,
        preferences: &FantasyPreferenceModelV1,
        world: &FantasyWorldStateV1,
        request: &FantasyTurnStyleRequestV1,
    ) -> Result<FantasyStylePlanOutcomeV1, FantasyStyleLatticeErrorV1> {
        if !session.is_active() {
            return Err(FantasyStyleLatticeErrorV1::FantasySessionNotActive);
        }

        if request.pacing_signal == FantasyPacingSignalV1::Stop {
            session
                .stop(AdultFantasyStopReasonV1::ExplicitExit)
                .map_err(|_| FantasyStyleLatticeErrorV1::SessionStopFailed)?;
            self.last_proposal = None;
            return Ok(FantasyStylePlanOutcomeV1::SessionStopped);
        }

        let activation = session
            .active_activation()
            .ok_or(FantasyStyleLatticeErrorV1::FantasySessionNotActive)?;
        if activation.session_id() != world.session_id() {
            return Err(FantasyStyleLatticeErrorV1::SessionWorldMismatch);
        }
        if !preferences.may_apply_preference_to_topic(&request.topic_id) {
            return Err(FantasyStyleLatticeErrorV1::TopicBlockedOrInvalid);
        }

        let overrides: BTreeMap<FantasyStyleDimensionV1, f32> = request
            .overrides
            .iter()
            .map(|entry| (entry.dimension, entry.value))
            .collect();

        let mut axes = BTreeMap::new();
        let mut uncertain_dimensions = BTreeSet::new();
        for dimension in ALL_FANTASY_STYLE_DIMENSIONS_V1 {
            let axis = if let Some(value) = overrides.get(&dimension).copied() {
                FantasyStyleAxisProposalV1 {
                    dimension,
                    value: Some(value),
                    confidence: 1.0,
                    source: FantasyStyleAxisSourceV1::ExplicitTurnRequest,
                    evidence_id: None,
                }
            } else if let Some(estimate) = preferences.effective_estimate(dimension) {
                FantasyStyleAxisProposalV1 {
                    dimension,
                    value: Some(estimate.value),
                    confidence: estimate.confidence,
                    source: estimate.source.into(),
                    evidence_id: Some(estimate.evidence_id),
                }
            } else {
                uncertain_dimensions.insert(dimension);
                FantasyStyleAxisProposalV1::unknown(dimension)
            };
            axes.insert(dimension, axis);
        }

        let sensitive_unknown = [
            FantasyStyleDimensionV1::Directness,
            FantasyStyleDimensionV1::VerbalIntensity,
            FantasyStyleDimensionV1::Initiative,
        ]
        .into_iter()
        .any(|dimension| uncertain_dimensions.contains(&dimension));

        let must_not_increase_intensity = matches!(
            request.pacing_signal,
            FantasyPacingSignalV1::Hold | FantasyPacingSignalV1::SlowDown
        );
        let must_decrease_intensity =
            request.pacing_signal == FantasyPacingSignalV1::SlowDown;
        let pacing = match request.pacing_signal {
            FantasyPacingSignalV1::Continue if sensitive_unknown => {
                FantasyPacingTrajectoryV1::ClarifyBeforeIncreasing
            }
            FantasyPacingSignalV1::Continue => FantasyPacingTrajectoryV1::Maintain,
            FantasyPacingSignalV1::Hold => FantasyPacingTrajectoryV1::Hold,
            FantasyPacingSignalV1::SlowDown => FantasyPacingTrajectoryV1::EaseDown,
            FantasyPacingSignalV1::Stop => unreachable!("handled before proposal construction"),
        };

        let previous_verbal_intensity_ceiling = self.previous_axis_value(
            activation.session_id(),
            activation.session_epoch(),
            FantasyStyleDimensionV1::VerbalIntensity,
            must_not_increase_intensity,
        );
        let previous_initiative_ceiling = self.previous_axis_value(
            activation.session_id(),
            activation.session_epoch(),
            FantasyStyleDimensionV1::Initiative,
            must_not_increase_intensity,
        );
        let previous_suspense_ceiling = self.previous_axis_value(
            activation.session_id(),
            activation.session_epoch(),
            FantasyStyleDimensionV1::Suspense,
            must_not_increase_intensity,
        );

        cap_axis_to_previous(&mut axes, FantasyStyleDimensionV1::VerbalIntensity, previous_verbal_intensity_ceiling);
        cap_axis_to_previous(&mut axes, FantasyStyleDimensionV1::Initiative, previous_initiative_ceiling);
        cap_axis_to_previous(&mut axes, FantasyStyleDimensionV1::Suspense, previous_suspense_ceiling);

        let proposal_epoch = self
            .proposal_epoch
            .checked_add(1)
            .ok_or(FantasyStyleLatticeErrorV1::ProposalEpochExhausted)?;
        self.proposal_epoch = proposal_epoch;
        let scene = world.scene();
        let proposal = FantasyStyleProposalV1 {
            proposal_epoch,
            session_id: activation.session_id().to_owned(),
            session_epoch: activation.session_epoch(),
            world_id: world.world_id().to_owned(),
            scene_id: scene.scene_id().to_owned(),
            scene_epoch: scene.scene_epoch(),
            boundary_epoch: preferences.boundaries().boundary_epoch(),
            topic_id: request.topic_id.clone(),
            axes,
            descriptive_balance: request.descriptive_balance,
            pacing,
            must_not_increase_intensity,
            must_decrease_intensity,
            previous_verbal_intensity_ceiling,
            previous_initiative_ceiling,
            previous_suspense_ceiling,
            uncertain_dimensions,
        };
        self.last_proposal = Some(proposal.clone());
        Ok(FantasyStylePlanOutcomeV1::Proposal(proposal))
    }

    fn previous_axis_value(
        &self,
        session_id: &str,
        session_epoch: u64,
        dimension: FantasyStyleDimensionV1,
        enabled: bool,
    ) -> Option<f32> {
        if !enabled {
            return None;
        }
        self.last_proposal
            .as_ref()
            .filter(|proposal| {
                proposal.session_id == session_id && proposal.session_epoch == session_epoch
            })
            .and_then(|proposal| proposal.axis(dimension).value)
    }
}

fn cap_axis_to_previous(
    axes: &mut BTreeMap<FantasyStyleDimensionV1, FantasyStyleAxisProposalV1>,
    dimension: FantasyStyleDimensionV1,
    ceiling: Option<f32>,
) {
    let Some(ceiling) = ceiling else {
        return;
    };
    if let Some(axis) = axes.get_mut(&dimension) {
        if let Some(value) = axis.value {
            axis.value = Some(value.min(ceiling));
        }
    }
}

fn canonical_topic_id(value: String) -> Result<String, FantasyStyleLatticeErrorV1> {
    let value = value.trim().to_ascii_lowercase();
    if value.is_empty() || value.len() > 128 {
        return Err(FantasyStyleLatticeErrorV1::InvalidTopicId);
    }
    if !value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        return Err(FantasyStyleLatticeErrorV1::InvalidTopicId);
    }
    Ok(value)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FantasyStyleLatticeErrorV1 {
    InvalidTopicId,
    InvalidAxisValue,
    InvalidDescriptiveBalance,
    DuplicateTurnOverride,
    FantasySessionNotActive,
    SessionWorldMismatch,
    TopicBlockedOrInvalid,
    SessionStopFailed,
    ProposalEpochExhausted,
}
