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

use adult_fantasy_session::*;
use fantasy_preferences::*;
use fantasy_style_lattice::*;
use fantasy_world_state::*;

struct AllowVerifier;
impl AdultEligibilityVerifier for AllowVerifier {
    fn verify_adult_eligibility(
        &self,
        _evidence: &AdultEligibilityEvidenceHandleV1,
        _participant_id: &str,
        _now_ns: u64,
    ) -> bool {
        true
    }
}

fn active_session(epoch: u64) -> AdultFantasySessionV1 {
    let mut session = AdultFantasySessionV1::new("participant-a", "session-a").unwrap();
    let activation = AdultFantasyActivationV1::new(
        "participant-a",
        "session-a",
        epoch,
        FantasyRealityFrameV1::ExplicitRoleplay,
        FantasyRetentionPolicyV1::Ephemeral,
        FantasyIdentityPolicyV1::OriginalOrFictionalOnly,
    )
    .unwrap();
    let evidence =
        AdultEligibilityEvidenceHandleV1::new("adult-evidence", "participant-a", 1, 10_000)
            .unwrap();
    session
        .activate(activation, evidence, &AllowVerifier, 100)
        .unwrap();
    session
}

fn world(session_id: &str) -> FantasyWorldStateV1 {
    FantasyWorldStateV1::new(
        "world-a",
        session_id,
        FantasySceneStateV1::new("scene-a", 1, Some("tone:warm".into())).unwrap(),
    )
    .unwrap()
}

fn preference(
    id: &str,
    dimension: FantasyStyleDimensionV1,
    value: f32,
    source: FantasyPreferenceSourceV1,
) -> FantasyPreferenceEvidenceV1 {
    FantasyPreferenceEvidenceV1::new(
        id,
        dimension,
        value,
        0.9,
        source,
        100,
        format!("source:{id}"),
        FantasyPreferenceRetentionV1::EphemeralSession,
    )
    .unwrap()
}

fn request(
    overrides: Vec<FantasyStyleTurnOverrideV1>,
    pacing: FantasyPacingSignalV1,
) -> FantasyTurnStyleRequestV1 {
    FantasyTurnStyleRequestV1::new("topic.alpha", overrides, Some(0.4), pacing).unwrap()
}

fn proposal(outcome: FantasyStylePlanOutcomeV1) -> FantasyStyleProposalV1 {
    match outcome {
        FantasyStylePlanOutcomeV1::Proposal(proposal) => proposal,
        FantasyStylePlanOutcomeV1::SessionStopped => panic!("expected proposal"),
    }
}

#[test]
fn explicit_turn_override_beats_stored_preference_without_mutating_it() {
    let mut session = active_session(1);
    let world = world("session-a");
    let mut preferences = FantasyPreferenceModelV1::new(1).unwrap();
    preferences
        .record(preference(
            "stored-playful",
            FantasyStyleDimensionV1::Playfulness,
            0.25,
            FantasyPreferenceSourceV1::ExplicitUserPreference,
        ))
        .unwrap();

    let mut planner = FantasyStylePlannerV1::default();
    let planned = proposal(
        planner
            .plan_next(
                &mut session,
                &preferences,
                &world,
                &request(
                    vec![FantasyStyleTurnOverrideV1::new(
                        FantasyStyleDimensionV1::Playfulness,
                        0.85,
                    )
                    .unwrap()],
                    FantasyPacingSignalV1::Continue,
                ),
            )
            .unwrap(),
    );

    let axis = planned.axis(FantasyStyleDimensionV1::Playfulness);
    assert_eq!(axis.value, Some(0.85));
    assert_eq!(axis.source, FantasyStyleAxisSourceV1::ExplicitTurnRequest);
    assert_eq!(
        preferences
            .effective_estimate(FantasyStyleDimensionV1::Playfulness)
            .unwrap()
            .value,
        0.25
    );
}

#[test]
fn unknown_sensitive_axes_require_clarification_before_increasing() {
    let mut session = active_session(1);
    let world = world("session-a");
    let preferences = FantasyPreferenceModelV1::new(1).unwrap();
    let mut planner = FantasyStylePlannerV1::default();

    let planned = proposal(
        planner
            .plan_next(
                &mut session,
                &preferences,
                &world,
                &request(Vec::new(), FantasyPacingSignalV1::Continue),
            )
            .unwrap(),
    );

    assert_eq!(
        planned.pacing,
        FantasyPacingTrajectoryV1::ClarifyBeforeIncreasing
    );
    assert!(
        planned
            .uncertain_dimensions
            .contains(&FantasyStyleDimensionV1::Directness)
    );
    assert_eq!(
        planned.axis(FantasyStyleDimensionV1::VerbalIntensity).value,
        None
    );
    assert_eq!(
        planned.axis(FantasyStyleDimensionV1::Initiative).source,
        FantasyStyleAxisSourceV1::Unknown
    );
}

#[test]
fn hard_topic_boundary_dominates_explicit_positive_style_preference() {
    let mut session = active_session(1);
    let world = world("session-a");
    let mut preferences = FantasyPreferenceModelV1::new(1).unwrap();
    preferences
        .record(preference(
            "explicit-intensity",
            FantasyStyleDimensionV1::VerbalIntensity,
            1.0,
            FantasyPreferenceSourceV1::ExplicitUserPreference,
        ))
        .unwrap();
    let mut boundary = FantasyTopicBoundaryV1::new(2).unwrap();
    boundary.block_topic("topic.alpha").unwrap();
    preferences.replace_boundaries(boundary).unwrap();

    let mut planner = FantasyStylePlannerV1::default();
    assert_eq!(
        planner.plan_next(
            &mut session,
            &preferences,
            &world,
            &request(Vec::new(), FantasyPacingSignalV1::Continue),
        ),
        Err(FantasyStyleLatticeErrorV1::TopicBlockedOrInvalid)
    );
}

#[test]
fn style_plan_requires_world_bound_to_active_session() {
    let mut session = active_session(1);
    let other_world = world("different-session");
    let preferences = FantasyPreferenceModelV1::new(1).unwrap();
    let mut planner = FantasyStylePlannerV1::default();

    assert_eq!(
        planner.plan_next(
            &mut session,
            &preferences,
            &other_world,
            &request(Vec::new(), FantasyPacingSignalV1::Continue),
        ),
        Err(FantasyStyleLatticeErrorV1::SessionWorldMismatch)
    );
}

#[test]
fn hold_caps_intensity_bearing_axes_to_previous_same_session_proposal() {
    let mut session = active_session(1);
    let world = world("session-a");
    let preferences = FantasyPreferenceModelV1::new(1).unwrap();
    let mut planner = FantasyStylePlannerV1::default();

    let first = proposal(
        planner
            .plan_next(
                &mut session,
                &preferences,
                &world,
                &request(
                    vec![
                        FantasyStyleTurnOverrideV1::new(
                            FantasyStyleDimensionV1::VerbalIntensity,
                            0.45,
                        )
                        .unwrap(),
                        FantasyStyleTurnOverrideV1::new(
                            FantasyStyleDimensionV1::Initiative,
                            0.35,
                        )
                        .unwrap(),
                        FantasyStyleTurnOverrideV1::new(
                            FantasyStyleDimensionV1::Suspense,
                            0.55,
                        )
                        .unwrap(),
                    ],
                    FantasyPacingSignalV1::Continue,
                ),
            )
            .unwrap(),
    );
    assert_eq!(
        first.axis(FantasyStyleDimensionV1::VerbalIntensity).value,
        Some(0.45)
    );

    let held = proposal(
        planner
            .plan_next(
                &mut session,
                &preferences,
                &world,
                &request(
                    vec![
                        FantasyStyleTurnOverrideV1::new(
                            FantasyStyleDimensionV1::VerbalIntensity,
                            0.95,
                        )
                        .unwrap(),
                        FantasyStyleTurnOverrideV1::new(
                            FantasyStyleDimensionV1::Initiative,
                            0.95,
                        )
                        .unwrap(),
                        FantasyStyleTurnOverrideV1::new(
                            FantasyStyleDimensionV1::Suspense,
                            0.95,
                        )
                        .unwrap(),
                    ],
                    FantasyPacingSignalV1::Hold,
                ),
            )
            .unwrap(),
    );

    assert_eq!(held.pacing, FantasyPacingTrajectoryV1::Hold);
    assert!(held.must_not_increase_intensity);
    assert!(!held.must_decrease_intensity);
    assert_eq!(
        held.axis(FantasyStyleDimensionV1::VerbalIntensity).value,
        Some(0.45)
    );
    assert_eq!(
        held.axis(FantasyStyleDimensionV1::Initiative).value,
        Some(0.35)
    );
    assert_eq!(
        held.axis(FantasyStyleDimensionV1::Suspense).value,
        Some(0.55)
    );
}

#[test]
fn slowdown_requires_decrease_and_cannot_raise_prior_ceiling() {
    let mut session = active_session(1);
    let world = world("session-a");
    let preferences = FantasyPreferenceModelV1::new(1).unwrap();
    let mut planner = FantasyStylePlannerV1::default();

    planner
        .plan_next(
            &mut session,
            &preferences,
            &world,
            &request(
                vec![FantasyStyleTurnOverrideV1::new(
                    FantasyStyleDimensionV1::VerbalIntensity,
                    0.6,
                )
                .unwrap()],
                FantasyPacingSignalV1::Continue,
            ),
        )
        .unwrap();

    let slowed = proposal(
        planner
            .plan_next(
                &mut session,
                &preferences,
                &world,
                &request(
                    vec![FantasyStyleTurnOverrideV1::new(
                        FantasyStyleDimensionV1::VerbalIntensity,
                        0.9,
                    )
                    .unwrap()],
                    FantasyPacingSignalV1::SlowDown,
                ),
            )
            .unwrap(),
    );

    assert_eq!(slowed.pacing, FantasyPacingTrajectoryV1::EaseDown);
    assert!(slowed.must_not_increase_intensity);
    assert!(slowed.must_decrease_intensity);
    assert_eq!(slowed.previous_verbal_intensity_ceiling, Some(0.6));
    assert_eq!(
        slowed.axis(FantasyStyleDimensionV1::VerbalIntensity).value,
        Some(0.6)
    );
}

#[test]
fn stop_latches_session_and_emits_no_style_proposal() {
    let mut session = active_session(1);
    let world = world("session-a");
    let preferences = FantasyPreferenceModelV1::new(1).unwrap();
    let mut planner = FantasyStylePlannerV1::default();

    planner
        .plan_next(
            &mut session,
            &preferences,
            &world,
            &request(Vec::new(), FantasyPacingSignalV1::Continue),
        )
        .unwrap();

    let outcome = planner
        .plan_next(
            &mut session,
            &preferences,
            &world,
            &request(Vec::new(), FantasyPacingSignalV1::Stop),
        )
        .unwrap();

    assert_eq!(outcome, FantasyStylePlanOutcomeV1::SessionStopped);
    assert!(!session.is_active());
    assert!(planner.last_proposal().is_none());
    assert!(matches!(
        session.state(),
        AdultFantasySessionStateV1::Stopped {
            reason: AdultFantasyStopReasonV1::ExplicitExit,
            ..
        }
    ));
}

#[test]
fn malformed_or_duplicate_turn_overrides_fail_closed() {
    assert_eq!(
        FantasyTurnStyleRequestV1::new(
            "not a valid topic",
            Vec::new(),
            None,
            FantasyPacingSignalV1::Continue,
        ),
        Err(FantasyStyleLatticeErrorV1::InvalidTopicId)
    );
    let duplicate = vec![
        FantasyStyleTurnOverrideV1::new(FantasyStyleDimensionV1::Romance, 0.2).unwrap(),
        FantasyStyleTurnOverrideV1::new(FantasyStyleDimensionV1::Romance, 0.7).unwrap(),
    ];
    assert_eq!(
        FantasyTurnStyleRequestV1::new(
            "topic.alpha",
            duplicate,
            None,
            FantasyPacingSignalV1::Continue,
        ),
        Err(FantasyStyleLatticeErrorV1::DuplicateTurnOverride)
    );
}

#[test]
fn proposal_binds_scene_boundary_and_session_epochs() {
    let mut session = active_session(7);
    let world = FantasyWorldStateV1::new(
        "world-seven",
        "session-a",
        FantasySceneStateV1::new("scene-seven", 3, None).unwrap(),
    )
    .unwrap();
    let preferences = FantasyPreferenceModelV1::new(11).unwrap();
    let mut planner = FantasyStylePlannerV1::default();

    let planned = proposal(
        planner
            .plan_next(
                &mut session,
                &preferences,
                &world,
                &request(Vec::new(), FantasyPacingSignalV1::Continue),
            )
            .unwrap(),
    );

    assert_eq!(planned.session_id, "session-a");
    assert_eq!(planned.session_epoch, 7);
    assert_eq!(planned.world_id, "world-seven");
    assert_eq!(planned.scene_id, "scene-seven");
    assert_eq!(planned.scene_epoch, 3);
    assert_eq!(planned.boundary_epoch, 11);
}
