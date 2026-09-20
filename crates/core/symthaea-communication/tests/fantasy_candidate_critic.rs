// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/adult_fantasy_session.rs"]
mod adult_fantasy_session;
#[path = "../src/fantasy_candidate_critic.rs"]
mod fantasy_candidate_critic;
#[path = "../src/fantasy_preferences.rs"]
mod fantasy_preferences;
#[path = "../src/fantasy_style_lattice.rs"]
mod fantasy_style_lattice;
#[path = "../src/fantasy_world_state.rs"]
mod fantasy_world_state;

use fantasy_candidate_critic::*;
use fantasy_preferences::FantasyStyleDimensionV1;
use fantasy_style_lattice::{
    FantasyPacingTrajectoryV1, FantasyStyleAxisProposalV1, FantasyStyleAxisSourceV1,
    FantasyStyleProposalV1,
};
use std::collections::{BTreeMap, BTreeSet};

fn style() -> FantasyStyleProposalV1 {
    let mut axes = BTreeMap::new();
    for dimension in [
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
    ] {
        axes.insert(
            dimension,
            FantasyStyleAxisProposalV1 {
                dimension,
                value: Some(0.5),
                confidence: 1.0,
                source: FantasyStyleAxisSourceV1::ExplicitTurnRequest,
                evidence_id: None,
            },
        );
    }
    FantasyStyleProposalV1 {
        proposal_epoch: 3,
        session_id: "session-a".into(),
        session_epoch: 2,
        world_id: "world-a".into(),
        scene_id: "scene-a".into(),
        scene_epoch: 1,
        boundary_epoch: 4,
        topic_id: "topic.alpha".into(),
        axes,
        descriptive_balance: Some(0.5),
        pacing: FantasyPacingTrajectoryV1::Maintain,
        must_not_increase_intensity: false,
        must_decrease_intensity: false,
        previous_verbal_intensity_ceiling: None,
        previous_initiative_ceiling: None,
        previous_suspense_ceiling: None,
        uncertain_dimensions: BTreeSet::new(),
    }
}

fn candidate(id: &str) -> FantasyDialogueCandidateV1 {
    FantasyDialogueCandidateV1::new(
        id,
        "session-a",
        2,
        3,
        "topic.alpha",
        format!("generator:{id}"),
        format!("private-content:{id}"),
        format!("blake3:{id}"),
    )
    .unwrap()
}

fn scores(default: f32) -> BTreeMap<FantasyCandidateSoftMetricV1, f32> {
    ALL_FANTASY_CANDIDATE_SOFT_METRICS_V1
        .into_iter()
        .map(|metric| (metric, default))
        .collect()
}

fn report(
    report_id: &str,
    critic_id: &str,
    candidate_id: &str,
    style_fit: f32,
    novelty: f32,
    hard_failures: BTreeSet<FantasyCandidateHardFailureV1>,
    uncertainty: bool,
) -> FantasyCandidateCriticReportV1 {
    let mut values = scores(0.7);
    values.insert(FantasyCandidateSoftMetricV1::StyleFit, style_fit);
    values.insert(FantasyCandidateSoftMetricV1::Novelty, novelty);
    FantasyCandidateCriticReportV1::new(
        report_id,
        critic_id,
        candidate_id,
        hard_failures,
        values,
        uncertainty,
    )
    .unwrap()
}

fn two_reports(candidate_id: &str, style_fit: f32, novelty: f32) -> Vec<FantasyCandidateCriticReportV1> {
    vec![
        report(
            &format!("r1-{candidate_id}"),
            "critic-a",
            candidate_id,
            style_fit,
            novelty,
            BTreeSet::new(),
            false,
        ),
        report(
            &format!("r2-{candidate_id}"),
            "critic-b",
            candidate_id,
            style_fit,
            novelty,
            BTreeSet::new(),
            false,
        ),
    ]
}

#[test]
fn any_hard_failure_disqualifies_even_perfect_soft_scores() {
    let style = style();
    let candidates = vec![candidate("unsafe"), candidate("safe")];
    let mut reports = Vec::new();
    let mut hard = BTreeSet::new();
    hard.insert(FantasyCandidateHardFailureV1::HardBoundary);
    reports.push(report(
        "r-unsafe-a",
        "critic-a",
        "unsafe",
        1.0,
        1.0,
        hard,
        false,
    ));
    reports.push(report(
        "r-unsafe-b",
        "critic-b",
        "unsafe",
        1.0,
        1.0,
        BTreeSet::new(),
        false,
    ));
    reports.extend(two_reports("safe", 0.6, 0.2));

    let receipt = select_fantasy_dialogue_candidate_v1(
        &style,
        &candidates,
        &reports,
        &FantasyCandidateSelectionPolicyV1::default(),
    )
    .unwrap();

    assert_eq!(receipt.selected_candidate_id.as_deref(), Some("safe"));
    let unsafe_assessment = receipt
        .assessments
        .iter()
        .find(|assessment| assessment.candidate_id == "unsafe")
        .unwrap();
    assert_eq!(
        unsafe_assessment.admissibility,
        FantasyCandidateAdmissibilityV1::Disqualified
    );
    assert!(
        unsafe_assessment
            .hard_failures
            .contains(&FantasyCandidateHardFailureV1::HardBoundary)
    );
}

#[test]
fn material_critic_disagreement_is_retained_as_uncertainty() {
    let style = style();
    let candidates = vec![candidate("a")];
    let reports = vec![
        report(
            "r1",
            "critic-a",
            "a",
            0.95,
            0.5,
            BTreeSet::new(),
            false,
        ),
        report(
            "r2",
            "critic-b",
            "a",
            0.20,
            0.5,
            BTreeSet::new(),
            false,
        ),
    ];

    let receipt = select_fantasy_dialogue_candidate_v1(
        &style,
        &candidates,
        &reports,
        &FantasyCandidateSelectionPolicyV1::default(),
    )
    .unwrap();

    assert!(receipt.selected_candidate_id.is_none());
    assert!(receipt.revision_required);
    assert_eq!(
        receipt.assessments[0].admissibility,
        FantasyCandidateAdmissibilityV1::Uncertain
    );
    assert!(
        receipt.assessments[0]
            .disagreement_metrics
            .contains(&FantasyCandidateSoftMetricV1::StyleFit)
    );
}

#[test]
fn insufficient_independent_critics_cannot_auto_select() {
    let style = style();
    let candidates = vec![candidate("a")];
    let reports = vec![report(
        "r1",
        "critic-a",
        "a",
        0.9,
        0.9,
        BTreeSet::new(),
        false,
    )];

    let receipt = select_fantasy_dialogue_candidate_v1(
        &style,
        &candidates,
        &reports,
        &FantasyCandidateSelectionPolicyV1::default(),
    )
    .unwrap();
    assert!(receipt.selected_candidate_id.is_none());
    assert_eq!(
        receipt.assessments[0].admissibility,
        FantasyCandidateAdmissibilityV1::Uncertain
    );
}

#[test]
fn scope_mismatch_is_hard_disqualification_not_soft_penalty() {
    let style = style();
    let mut wrong = candidate("wrong");
    wrong.style_proposal_epoch = 99;
    let reports = two_reports("wrong", 1.0, 1.0);
    let receipt = select_fantasy_dialogue_candidate_v1(
        &style,
        &[wrong],
        &reports,
        &FantasyCandidateSelectionPolicyV1::default(),
    )
    .unwrap();

    assert!(receipt.selected_candidate_id.is_none());
    assert!(
        receipt.assessments[0]
            .hard_failures
            .contains(&FantasyCandidateHardFailureV1::ScopeBinding)
    );
}

#[test]
fn default_soft_selection_is_lexicographic_not_novelty_dominated() {
    let style = style();
    let candidates = vec![candidate("better-style"), candidate("more-novel")];
    let mut reports = two_reports("better-style", 0.90, 0.10);
    reports.extend(two_reports("more-novel", 0.80, 1.00));

    let receipt = select_fantasy_dialogue_candidate_v1(
        &style,
        &candidates,
        &reports,
        &FantasyCandidateSelectionPolicyV1::default(),
    )
    .unwrap();

    assert_eq!(
        receipt.selected_candidate_id.as_deref(),
        Some("better-style")
    );
}

#[test]
fn duplicate_critic_for_same_candidate_fails_closed() {
    let style = style();
    let candidates = vec![candidate("a")];
    let reports = vec![
        report(
            "r1",
            "critic-a",
            "a",
            0.8,
            0.8,
            BTreeSet::new(),
            false,
        ),
        report(
            "r2",
            "critic-a",
            "a",
            0.8,
            0.8,
            BTreeSet::new(),
            false,
        ),
    ];

    assert_eq!(
        select_fantasy_dialogue_candidate_v1(
            &style,
            &candidates,
            &reports,
            &FantasyCandidateSelectionPolicyV1::default(),
        ),
        Err(FantasyCandidateCriticErrorV1::DuplicateCriticForCandidate)
    );
}

#[test]
fn critic_material_uncertainty_blocks_automatic_selection_without_fabricated_failure() {
    let style = style();
    let candidates = vec![candidate("a")];
    let reports = vec![
        report(
            "r1",
            "critic-a",
            "a",
            0.9,
            0.9,
            BTreeSet::new(),
            true,
        ),
        report(
            "r2",
            "critic-b",
            "a",
            0.9,
            0.9,
            BTreeSet::new(),
            false,
        ),
    ];

    let receipt = select_fantasy_dialogue_candidate_v1(
        &style,
        &candidates,
        &reports,
        &FantasyCandidateSelectionPolicyV1::default(),
    )
    .unwrap();
    assert!(receipt.selected_candidate_id.is_none());
    assert!(receipt.assessments[0].hard_failures.is_empty());
    assert!(receipt.assessments[0].material_uncertainty);
}

#[test]
fn incomplete_soft_scores_are_rejected_at_report_construction() {
    let mut incomplete = scores(0.7);
    incomplete.remove(&FantasyCandidateSoftMetricV1::Novelty);
    assert_eq!(
        FantasyCandidateCriticReportV1::new(
            "r1",
            "critic-a",
            "a",
            BTreeSet::new(),
            incomplete,
            false,
        ),
        Err(FantasyCandidateCriticErrorV1::IncompleteSoftScores)
    );
}
