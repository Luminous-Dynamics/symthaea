// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Non-authoritative multimodal orchestration for an already-admitted adult fantasy turn.

use crate::fantasy_preferences::FantasyStyleDimensionV1;
use crate::fantasy_style_lattice::{
    FantasyPacingTrajectoryV1, FantasyStyleProposalV1,
};
use serde::{Deserialize, Serialize};

pub const INTIMACY_DIRECTOR_PLAN_SCHEMA_V1: &str =
    "symthaea.communication.intimacy-director-plan.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntimacyCueDirectionV1 {
    Deescalating,
    Neutral,
    Escalating,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntimacyCueRequestV1 {
    pub cue_ref: String,
    pub direction: IntimacyCueDirectionV1,
}

impl IntimacyCueRequestV1 {
    pub fn new(
        cue_ref: impl Into<String>,
        direction: IntimacyCueDirectionV1,
    ) -> Result<Self, IntimacyDirectorErrorV1> {
        Ok(Self {
            cue_ref: canonical_ref(cue_ref.into())?,
            direction,
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntimacyDirectorCueBundleV1 {
    pub music: Option<IntimacyCueRequestV1>,
    pub lighting: Option<IntimacyCueRequestV1>,
    pub expression: Option<IntimacyCueRequestV1>,
    pub proximity: Option<IntimacyCueRequestV1>,
    pub touch: Option<IntimacyCueRequestV1>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntimacyDirectorCapabilitiesV1 {
    pub music: bool,
    pub lighting: bool,
    pub expression: bool,
    pub proximity_proposal: bool,
    pub touch_proposal: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndependentPhysicalEligibilityV1 {
    /// Opaque reference to a separately governed proximity/motion eligibility decision.
    pub proximity_eligibility_ref: Option<String>,
    /// Opaque reference to a separately governed contact eligibility decision.
    pub touch_eligibility_ref: Option<String>,
}

impl IndependentPhysicalEligibilityV1 {
    pub fn new(
        proximity_eligibility_ref: Option<String>,
        touch_eligibility_ref: Option<String>,
    ) -> Result<Self, IntimacyDirectorErrorV1> {
        Ok(Self {
            proximity_eligibility_ref: proximity_eligibility_ref
                .map(canonical_ref)
                .transpose()?,
            touch_eligibility_ref: touch_eligibility_ref.map(canonical_ref).transpose()?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntimacyDirectorCurrentStateV1 {
    pub session_id: String,
    pub session_epoch: u64,
    pub world_id: String,
    pub scene_id: String,
    pub scene_epoch: u64,
    pub boundary_epoch: u64,
}

impl IntimacyDirectorCurrentStateV1 {
    pub fn new(
        session_id: impl Into<String>,
        session_epoch: u64,
        world_id: impl Into<String>,
        scene_id: impl Into<String>,
        scene_epoch: u64,
        boundary_epoch: u64,
    ) -> Result<Self, IntimacyDirectorErrorV1> {
        if session_epoch == 0 || scene_epoch == 0 || boundary_epoch == 0 {
            return Err(IntimacyDirectorErrorV1::InvalidEpoch);
        }
        Ok(Self {
            session_id: canonical_ref(session_id.into())?,
            session_epoch,
            world_id: canonical_ref(world_id.into())?,
            scene_id: canonical_ref(scene_id.into())?,
            scene_epoch,
            boundary_epoch,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IntimacyProsodyIntentV1 {
    pub tenderness: Option<f32>,
    pub playfulness: Option<f32>,
    pub directness: Option<f32>,
    pub verbal_intensity_ceiling: Option<f32>,
    pub pacing: FantasyPacingTrajectoryV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IntimacyModalityV1 {
    Music,
    Lighting,
    Expression,
    Proximity,
    Touch,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntimacyModalityProposalV1 {
    pub modality: IntimacyModalityV1,
    pub cue_ref: String,
    pub direction: IntimacyCueDirectionV1,
    /// Present only for physical modalities. This is evidence of an independent
    /// eligibility decision, not authority created by the director.
    pub independent_eligibility_ref: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntimacyProposalSuppressionReasonV1 {
    CapabilityUnavailable,
    IndependentEligibilityMissing,
    PacingConstraint,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntimacySuppressedProposalV1 {
    pub modality: IntimacyModalityV1,
    pub cue_ref: String,
    pub reason: IntimacyProposalSuppressionReasonV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IntimacyDirectorPlanV1 {
    pub schema: String,
    pub style_proposal_epoch: u64,
    pub session_id: String,
    pub session_epoch: u64,
    pub world_id: String,
    pub scene_id: String,
    pub scene_epoch: u64,
    pub boundary_epoch: u64,
    pub topic_id: String,
    pub prosody: IntimacyProsodyIntentV1,
    pub proposals: Vec<IntimacyModalityProposalV1>,
    pub suppressed: Vec<IntimacySuppressedProposalV1>,
}

pub fn compose_intimacy_director_plan_v1(
    style: &FantasyStyleProposalV1,
    current: &IntimacyDirectorCurrentStateV1,
    capabilities: IntimacyDirectorCapabilitiesV1,
    physical_eligibility: &IndependentPhysicalEligibilityV1,
    cues: &IntimacyDirectorCueBundleV1,
) -> Result<IntimacyDirectorPlanV1, IntimacyDirectorErrorV1> {
    validate_style_freshness(style, current)?;

    let prosody = IntimacyProsodyIntentV1 {
        tenderness: style.axis(FantasyStyleDimensionV1::Tenderness).value,
        playfulness: style.axis(FantasyStyleDimensionV1::Playfulness).value,
        directness: style.axis(FantasyStyleDimensionV1::Directness).value,
        verbal_intensity_ceiling: style
            .axis(FantasyStyleDimensionV1::VerbalIntensity)
            .value,
        pacing: style.pacing,
    };

    let mut proposals = Vec::new();
    let mut suppressed = Vec::new();

    admit_nonphysical(
        IntimacyModalityV1::Music,
        cues.music.as_ref(),
        capabilities.music,
        style,
        &mut proposals,
        &mut suppressed,
    );
    admit_nonphysical(
        IntimacyModalityV1::Lighting,
        cues.lighting.as_ref(),
        capabilities.lighting,
        style,
        &mut proposals,
        &mut suppressed,
    );
    admit_nonphysical(
        IntimacyModalityV1::Expression,
        cues.expression.as_ref(),
        capabilities.expression,
        style,
        &mut proposals,
        &mut suppressed,
    );
    admit_physical(
        IntimacyModalityV1::Proximity,
        cues.proximity.as_ref(),
        capabilities.proximity_proposal,
        physical_eligibility.proximity_eligibility_ref.as_deref(),
        style,
        &mut proposals,
        &mut suppressed,
    );
    admit_physical(
        IntimacyModalityV1::Touch,
        cues.touch.as_ref(),
        capabilities.touch_proposal,
        physical_eligibility.touch_eligibility_ref.as_deref(),
        style,
        &mut proposals,
        &mut suppressed,
    );

    proposals.sort_by_key(|entry| entry.modality);
    suppressed.sort_by_key(|entry| entry.modality);

    Ok(IntimacyDirectorPlanV1 {
        schema: INTIMACY_DIRECTOR_PLAN_SCHEMA_V1.into(),
        style_proposal_epoch: style.proposal_epoch,
        session_id: style.session_id.clone(),
        session_epoch: style.session_epoch,
        world_id: style.world_id.clone(),
        scene_id: style.scene_id.clone(),
        scene_epoch: style.scene_epoch,
        boundary_epoch: style.boundary_epoch,
        topic_id: style.topic_id.clone(),
        prosody,
        proposals,
        suppressed,
    })
}

fn validate_style_freshness(
    style: &FantasyStyleProposalV1,
    current: &IntimacyDirectorCurrentStateV1,
) -> Result<(), IntimacyDirectorErrorV1> {
    if style.session_id != current.session_id || style.session_epoch != current.session_epoch {
        return Err(IntimacyDirectorErrorV1::StaleSessionBinding);
    }
    if style.world_id != current.world_id
        || style.scene_id != current.scene_id
        || style.scene_epoch != current.scene_epoch
    {
        return Err(IntimacyDirectorErrorV1::StaleSceneBinding);
    }
    if style.boundary_epoch != current.boundary_epoch {
        return Err(IntimacyDirectorErrorV1::StaleBoundaryBinding);
    }
    Ok(())
}

fn admit_nonphysical(
    modality: IntimacyModalityV1,
    request: Option<&IntimacyCueRequestV1>,
    capability: bool,
    style: &FantasyStyleProposalV1,
    proposals: &mut Vec<IntimacyModalityProposalV1>,
    suppressed: &mut Vec<IntimacySuppressedProposalV1>,
) {
    let Some(request) = request else {
        return;
    };
    if !capability {
        suppress(modality, request, IntimacyProposalSuppressionReasonV1::CapabilityUnavailable, suppressed);
        return;
    }
    if !direction_allowed(style, request.direction) {
        suppress(modality, request, IntimacyProposalSuppressionReasonV1::PacingConstraint, suppressed);
        return;
    }
    proposals.push(IntimacyModalityProposalV1 {
        modality,
        cue_ref: request.cue_ref.clone(),
        direction: request.direction,
        independent_eligibility_ref: None,
    });
}

fn admit_physical(
    modality: IntimacyModalityV1,
    request: Option<&IntimacyCueRequestV1>,
    capability: bool,
    independent_eligibility_ref: Option<&str>,
    style: &FantasyStyleProposalV1,
    proposals: &mut Vec<IntimacyModalityProposalV1>,
    suppressed: &mut Vec<IntimacySuppressedProposalV1>,
) {
    let Some(request) = request else {
        return;
    };
    if !capability {
        suppress(modality, request, IntimacyProposalSuppressionReasonV1::CapabilityUnavailable, suppressed);
        return;
    }
    if !direction_allowed(style, request.direction) {
        suppress(modality, request, IntimacyProposalSuppressionReasonV1::PacingConstraint, suppressed);
        return;
    }
    let Some(reference) = independent_eligibility_ref else {
        suppress(
            modality,
            request,
            IntimacyProposalSuppressionReasonV1::IndependentEligibilityMissing,
            suppressed,
        );
        return;
    };
    proposals.push(IntimacyModalityProposalV1 {
        modality,
        cue_ref: request.cue_ref.clone(),
        direction: request.direction,
        independent_eligibility_ref: Some(reference.to_owned()),
    });
}

fn direction_allowed(style: &FantasyStyleProposalV1, direction: IntimacyCueDirectionV1) -> bool {
    if style.must_decrease_intensity {
        return direction == IntimacyCueDirectionV1::Deescalating;
    }
    if style.must_not_increase_intensity
        || style.pacing == FantasyPacingTrajectoryV1::ClarifyBeforeIncreasing
    {
        return direction != IntimacyCueDirectionV1::Escalating;
    }
    true
}

fn suppress(
    modality: IntimacyModalityV1,
    request: &IntimacyCueRequestV1,
    reason: IntimacyProposalSuppressionReasonV1,
    suppressed: &mut Vec<IntimacySuppressedProposalV1>,
) {
    suppressed.push(IntimacySuppressedProposalV1 {
        modality,
        cue_ref: request.cue_ref.clone(),
        reason,
    });
}

fn canonical_ref(value: String) -> Result<String, IntimacyDirectorErrorV1> {
    let value = value.trim().to_owned();
    if value.is_empty() || value.len() > 512 {
        return Err(IntimacyDirectorErrorV1::InvalidReference);
    }
    Ok(value)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntimacyDirectorErrorV1 {
    InvalidReference,
    InvalidEpoch,
    StaleSessionBinding,
    StaleSceneBinding,
    StaleBoundaryBinding,
}
