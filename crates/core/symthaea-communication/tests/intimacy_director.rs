// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/adult_fantasy_session.rs"]
mod adult_fantasy_session;
#[path = "../src/fantasy_preferences.rs"]
mod fantasy_preferences;
#[path = "../src/fantasy_style_lattice.rs"]
mod fantasy_style_lattice;
#[path = "../src/fantasy_world_state.rs"]
mod fantasy_world_state;
#[path = "../src/intimacy_director.rs"]
mod intimacy_director;

use fantasy_preferences::FantasyStyleDimensionV1;
use fantasy_style_lattice::{
    FantasyPacingTrajectoryV1, FantasyStyleAxisProposalV1, FantasyStyleAxisSourceV1,
    FantasyStyleProposalV1,
};
use intimacy_director::*;
use std::collections::{BTreeMap, BTreeSet};

fn style(
    pacing: FantasyPacingTrajectoryV1,
    must_not_increase: bool,
    must_decrease: bool,
) -> FantasyStyleProposalV1 {
    let mut axes = BTreeMap::new();
    for (dimension, value) in [
        (FantasyStyleDimensionV1::Romance, 0.8),
        (FantasyStyleDimensionV1::Playfulness, 0.6),
        (FantasyStyleDimensionV1::Directness, 0.4),
        (FantasyStyleDimensionV1::VerbalIntensity, 0.5),
        (FantasyStyleDimensionV1::Tenderness, 0.9),
        (FantasyStyleDimensionV1::Initiative, 0.4),
        (FantasyStyleDimensionV1::Suspense, 0.3),
        (FantasyStyleDimensionV1::Humor, 0.2),
        (FantasyStyleDimensionV1::NarrativeDensity, 0.7),
        (FantasyStyleDimensionV1::CallbackDensity, 0.6),
    ] {
        axes.insert(
            dimension,
            FantasyStyleAxisProposalV1 {
                dimension,
                value: Some(value),
                confidence: 1.0,
                source: FantasyStyleAxisSourceV1::ExplicitTurnRequest,
                evidence_id: None,
            },
        );
    }
    FantasyStyleProposalV1 {
        proposal_epoch: 9,
        session_id: "session-a".into(),
        session_epoch: 4,
        world_id: "world-a".into(),
        scene_id: "scene-a".into(),
        scene_epoch: 3,
        boundary_epoch: 7,
        topic_id: "topic.alpha".into(),
        axes,
        descriptive_balance: Some(0.5),
        pacing,
        must_not_increase_intensity: must_not_increase,
        must_decrease_intensity: must_decrease,
        previous_verbal_intensity_ceiling: None,
        previous_initiative_ceiling: None,
        previous_suspense_ceiling: None,
        uncertain_dimensions: BTreeSet::new(),
    }
}

fn current() -> IntimacyDirectorCurrentStateV1 {
    IntimacyDirectorCurrentStateV1::new("session-a", 4, "world-a", "scene-a", 3, 7).unwrap()
}

fn cue(reference: &str, direction: IntimacyCueDirectionV1) -> IntimacyCueRequestV1 {
    IntimacyCueRequestV1::new(reference, direction).unwrap()
}

fn all_capabilities() -> IntimacyDirectorCapabilitiesV1 {
    IntimacyDirectorCapabilitiesV1 {
        music: true,
        lighting: true,
        expression: true,
        proximity_proposal: true,
        touch_proposal: true,
    }
}

#[test]
fn stale_style_binding_is_rejected_before_any_modality_plan() {
    let style = style(FantasyPacingTrajectoryV1::Maintain, false, false);
    let stale = IntimacyDirectorCurrentStateV1::new(
        "session-a",
        5,
        "world-a",
        "scene-a",
        3,
        7,
    )
    .unwrap();
    assert_eq!(
        compose_intimacy_director_plan_v1(
            &style,
            &stale,
            all_capabilities(),
            &IndependentPhysicalEligibilityV1::default(),
            &IntimacyDirectorCueBundleV1::default(),
        ),
        Err(IntimacyDirectorErrorV1::StaleSessionBinding)
    );
}

#[test]
fn physical_proposals_require_independent_eligibility_but_do_not_block_dialogue_adjacent_cues() {
    let style = style(FantasyPacingTrajectoryV1::Maintain, false, false);
    let cues = IntimacyDirectorCueBundleV1 {
        music: Some(cue("cue:music", IntimacyCueDirectionV1::Neutral)),
        touch: Some(cue("proposal:touch", IntimacyCueDirectionV1::Neutral)),
        ..Default::default()
    };
    let plan = compose_intimacy_director_plan_v1(
        &style,
        &current(),
        all_capabilities(),
        &IndependentPhysicalEligibilityV1::default(),
        &cues,
    )
    .unwrap();

    assert!(plan.proposals.iter().any(|p| p.modality == IntimacyModalityV1::Music));
    assert!(!plan.proposals.iter().any(|p| p.modality == IntimacyModalityV1::Touch));
    assert!(plan.suppressed.iter().any(|p| {
        p.modality == IntimacyModalityV1::Touch
            && p.reason == IntimacyProposalSuppressionReasonV1::IndependentEligibilityMissing
    }));
}

#[test]
fn touch_eligibility_is_carried_as_read_only_provenance_not_created_by_director() {
    let style = style(FantasyPacingTrajectoryV1::Maintain, false, false);
    let cues = IntimacyDirectorCueBundleV1 {
        touch: Some(cue("proposal:touch", IntimacyCueDirectionV1::Neutral)),
        ..Default::default()
    };
    let eligibility = IndependentPhysicalEligibilityV1::new(
        None,
        Some("authority:contact-eligibility-17".into()),
    )
    .unwrap();
    let plan = compose_intimacy_director_plan_v1(
        &style,
        &current(),
        all_capabilities(),
        &eligibility,
        &cues,
    )
    .unwrap();

    let touch = plan
        .proposals
        .iter()
        .find(|p| p.modality == IntimacyModalityV1::Touch)
        .unwrap();
    assert_eq!(
        touch.independent_eligibility_ref.as_deref(),
        Some("authority:contact-eligibility-17")
    );
}

#[test]
fn hold_suppresses_escalating_cues_across_modalities() {
    let style = style(FantasyPacingTrajectoryV1::Hold, true, false);
    let cues = IntimacyDirectorCueBundleV1 {
        music: Some(cue("cue:music-up", IntimacyCueDirectionV1::Escalating)),
        lighting: Some(cue("cue:lighting-neutral", IntimacyCueDirectionV1::Neutral)),
        proximity: Some(cue(
            "proposal:proximity-up",
            IntimacyCueDirectionV1::Escalating,
        )),
        ..Default::default()
    };
    let eligibility = IndependentPhysicalEligibilityV1::new(
        Some("authority:proximity-17".into()),
        None,
    )
    .unwrap();
    let plan = compose_intimacy_director_plan_v1(
        &style,
        &current(),
        all_capabilities(),
        &eligibility,
        &cues,
    )
    .unwrap();

    assert!(plan.proposals.iter().any(|p| p.modality == IntimacyModalityV1::Lighting));
    assert!(plan.suppressed.iter().any(|p| {
        p.modality == IntimacyModalityV1::Music
            && p.reason == IntimacyProposalSuppressionReasonV1::PacingConstraint
    }));
    assert!(plan.suppressed.iter().any(|p| {
        p.modality == IntimacyModalityV1::Proximity
            && p.reason == IntimacyProposalSuppressionReasonV1::PacingConstraint
    }));
}

#[test]
fn slowdown_allows_only_deescalating_optional_cues() {
    let style = style(FantasyPacingTrajectoryV1::EaseDown, true, true);
    let cues = IntimacyDirectorCueBundleV1 {
        music: Some(cue("cue:music-soften", IntimacyCueDirectionV1::Deescalating)),
        lighting: Some(cue("cue:lighting-neutral", IntimacyCueDirectionV1::Neutral)),
        expression: Some(cue("cue:expression-up", IntimacyCueDirectionV1::Escalating)),
        ..Default::default()
    };
    let plan = compose_intimacy_director_plan_v1(
        &style,
        &current(),
        all_capabilities(),
        &IndependentPhysicalEligibilityV1::default(),
        &cues,
    )
    .unwrap();

    assert_eq!(plan.proposals.len(), 1);
    assert_eq!(plan.proposals[0].modality, IntimacyModalityV1::Music);
    assert_eq!(plan.suppressed.len(), 2);
    assert!(plan
        .suppressed
        .iter()
        .all(|entry| entry.reason == IntimacyProposalSuppressionReasonV1::PacingConstraint));
}

#[test]
fn uncertainty_never_allows_escalating_cross_modal_cue() {
    let style = style(
        FantasyPacingTrajectoryV1::ClarifyBeforeIncreasing,
        false,
        false,
    );
    let cues = IntimacyDirectorCueBundleV1 {
        expression: Some(cue("cue:expression-up", IntimacyCueDirectionV1::Escalating)),
        music: Some(cue("cue:music-neutral", IntimacyCueDirectionV1::Neutral)),
        ..Default::default()
    };
    let plan = compose_intimacy_director_plan_v1(
        &style,
        &current(),
        all_capabilities(),
        &IndependentPhysicalEligibilityV1::default(),
        &cues,
    )
    .unwrap();

    assert!(plan.proposals.iter().any(|p| p.modality == IntimacyModalityV1::Music));
    assert!(plan.suppressed.iter().any(|p| p.modality == IntimacyModalityV1::Expression));
}

#[test]
fn capability_absence_is_local_to_that_modality() {
    let style = style(FantasyPacingTrajectoryV1::Maintain, false, false);
    let cues = IntimacyDirectorCueBundleV1 {
        music: Some(cue("cue:music", IntimacyCueDirectionV1::Neutral)),
        lighting: Some(cue("cue:lighting", IntimacyCueDirectionV1::Neutral)),
        ..Default::default()
    };
    let mut capabilities = all_capabilities();
    capabilities.music = false;
    let plan = compose_intimacy_director_plan_v1(
        &style,
        &current(),
        capabilities,
        &IndependentPhysicalEligibilityV1::default(),
        &cues,
    )
    .unwrap();

    assert!(plan.proposals.iter().any(|p| p.modality == IntimacyModalityV1::Lighting));
    assert!(plan.suppressed.iter().any(|p| {
        p.modality == IntimacyModalityV1::Music
            && p.reason == IntimacyProposalSuppressionReasonV1::CapabilityUnavailable
    }));
}

#[test]
fn prosody_is_a_semantic_projection_of_style_not_a_new_authority_source() {
    let style = style(FantasyPacingTrajectoryV1::Maintain, false, false);
    let plan = compose_intimacy_director_plan_v1(
        &style,
        &current(),
        all_capabilities(),
        &IndependentPhysicalEligibilityV1::default(),
        &IntimacyDirectorCueBundleV1::default(),
    )
    .unwrap();

    assert_eq!(plan.prosody.tenderness, Some(0.9));
    assert_eq!(plan.prosody.playfulness, Some(0.6));
    assert_eq!(plan.prosody.directness, Some(0.4));
    assert_eq!(plan.prosody.verbal_intensity_ceiling, Some(0.5));
    assert!(plan.proposals.is_empty());
}
