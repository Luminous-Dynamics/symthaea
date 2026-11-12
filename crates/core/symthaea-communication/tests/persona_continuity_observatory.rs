// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/persona_continuity_observatory.rs"]
mod persona_continuity_observatory;

use persona_continuity_observatory::{
    PersonaBandV1, PersonaEvidenceStatusV1, PersonaFacetCriticalityV1,
    PersonaFacetPolicyV1, PersonaObservationV1, PersonaProfileV1,
    evaluate_persona_continuity,
};

fn band(min: f32, max: f32) -> PersonaBandV1 {
    PersonaBandV1::new(min, max).unwrap()
}

fn profile() -> PersonaProfileV1 {
    PersonaProfileV1::new(
        "symthaea-persona",
        "v1",
        0.20,
        vec![
            PersonaFacetPolicyV1::new(
                "warmth",
                band(0.45, 0.90),
                PersonaFacetCriticalityV1::Critical,
                true,
                false,
                vec![
                    ("technical:rust".into(), band(0.45, 0.70)),
                    ("playful".into(), band(0.60, 0.90)),
                ],
            )
            .unwrap(),
            PersonaFacetPolicyV1::new(
                "playfulness",
                band(0.10, 0.90),
                PersonaFacetCriticalityV1::NonCritical,
                true,
                false,
                vec![
                    ("technical:rust".into(), band(0.10, 0.40)),
                    ("playful".into(), band(0.55, 0.90)),
                ],
            )
            .unwrap(),
            PersonaFacetPolicyV1::new(
                "precision",
                band(0.65, 1.0),
                PersonaFacetCriticalityV1::Critical,
                true,
                false,
                vec![],
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

fn obs(
    id: &str,
    episode: &str,
    turn: u64,
    context: &str,
    facet: &str,
    value: f32,
    evaluator: &str,
) -> PersonaObservationV1 {
    PersonaObservationV1::new(
        id,
        episode,
        turn,
        context,
        facet,
        value,
        evaluator,
        format!("evidence:{id}"),
    )
    .unwrap()
}

#[test]
fn stable_core_can_vary_across_contexts_without_becoming_one_fixed_tone() {
    let profile = profile();
    let observations = vec![
        obs("w-tech", "episode-1", 1, "technical:rust", "warmth", 0.55, "critic-a"),
        obs("p-tech", "episode-1", 1, "technical:rust", "playfulness", 0.25, "critic-a"),
        obs("prec", "episode-1", 1, "technical:rust", "precision", 0.90, "critic-a"),
        obs("w-play", "episode-1", 2, "playful", "warmth", 0.80, "critic-a"),
        obs("p-play", "episode-1", 2, "playful", "playfulness", 0.75, "critic-a"),
    ];

    let receipt = evaluate_persona_continuity(&profile, &observations).unwrap();
    assert_eq!(receipt.evidence_status, PersonaEvidenceStatusV1::Established);
    assert_eq!(receipt.critical_violation_count, 0);
    assert_eq!(receipt.noncritical_violation_count, 0);
    assert_eq!(receipt.observed_context_count, 2);
    assert_eq!(receipt.observed_facet_count, 3);
}

#[test]
fn context_specific_drift_is_reported_without_calling_it_global_identity_failure() {
    let profile = profile();
    let observations = vec![
        obs("w", "episode-1", 1, "technical:rust", "warmth", 0.55, "critic-a"),
        obs("p", "episode-1", 1, "technical:rust", "playfulness", 0.70, "critic-a"),
        obs("prec", "episode-1", 1, "technical:rust", "precision", 0.90, "critic-a"),
    ];

    let receipt = evaluate_persona_continuity(&profile, &observations).unwrap();
    assert_eq!(receipt.context_band_violation_count, 1);
    assert_eq!(receipt.global_band_violation_count, 0);
    assert_eq!(receipt.noncritical_violation_count, 1);
    assert_eq!(receipt.critical_violation_count, 0);
}

#[test]
fn global_core_violation_is_reported_separately() {
    let profile = profile();
    let observations = vec![
        obs("w", "episode-1", 1, "ordinary", "warmth", 0.60, "critic-a"),
        obs("p", "episode-1", 1, "ordinary", "playfulness", 0.40, "critic-a"),
        obs("prec", "episode-1", 1, "ordinary", "precision", 0.30, "critic-a"),
    ];

    let receipt = evaluate_persona_continuity(&profile, &observations).unwrap();
    assert_eq!(receipt.global_band_violation_count, 1);
    assert_eq!(receipt.critical_violation_count, 1);
}

#[test]
fn missing_required_facets_yields_partial_not_success() {
    let profile = profile();
    let observations = vec![obs(
        "w",
        "episode-1",
        1,
        "ordinary",
        "warmth",
        0.60,
        "critic-a",
    )];

    let receipt = evaluate_persona_continuity(&profile, &observations).unwrap();
    assert_eq!(receipt.evidence_status, PersonaEvidenceStatusV1::Partial);
    assert_eq!(
        receipt.unobserved_required_facets,
        vec!["playfulness".to_string(), "precision".to_string()]
    );
}

#[test]
fn no_observations_is_not_established() {
    let profile = profile();
    let receipt = evaluate_persona_continuity(&profile, &[]).unwrap();
    assert_eq!(receipt.evidence_status, PersonaEvidenceStatusV1::NotEstablished);
    assert_eq!(receipt.total_observation_count, 0);
}

#[test]
fn evaluator_disagreement_is_uncertainty_not_a_fabricated_persona_violation() {
    let profile = profile();
    let observations = vec![
        obs("w-a", "episode-1", 1, "ordinary", "warmth", 0.50, "critic-a"),
        obs("w-b", "episode-1", 1, "ordinary", "warmth", 0.80, "critic-b"),
        obs("p", "episode-1", 1, "ordinary", "playfulness", 0.40, "critic-a"),
        obs("prec", "episode-1", 1, "ordinary", "precision", 0.90, "critic-a"),
    ];

    let receipt = evaluate_persona_continuity(&profile, &observations).unwrap();
    assert_eq!(receipt.evidence_status, PersonaEvidenceStatusV1::Uncertain);
    assert_eq!(receipt.material_disagreement_group_count, 1);
    assert_eq!(receipt.critical_violation_count, 0);
}

#[test]
fn duplicate_evaluator_for_same_exact_slot_fails_closed() {
    let profile = profile();
    let observations = vec![
        obs("a", "episode-1", 1, "ordinary", "warmth", 0.55, "critic-a"),
        obs("b", "episode-1", 1, "ordinary", "warmth", 0.60, "critic-a"),
    ];
    assert!(evaluate_persona_continuity(&profile, &observations).is_err());
}

#[test]
fn context_band_cannot_silently_broaden_global_identity_range() {
    let invalid = PersonaFacetPolicyV1::new(
        "warmth",
        band(0.45, 0.80),
        PersonaFacetCriticalityV1::Critical,
        true,
        false,
        vec![("playful".into(), band(0.40, 0.90))],
    );
    assert!(invalid.is_err());
}

#[test]
fn explicit_context_broadening_is_possible_when_profile_declares_it() {
    let facet = PersonaFacetPolicyV1::new(
        "expressiveness",
        band(0.30, 0.70),
        PersonaFacetCriticalityV1::NonCritical,
        true,
        true,
        vec![("performance".into(), band(0.10, 0.95))],
    )
    .unwrap();
    let profile = PersonaProfileV1::new("persona", "v1", 0.20, vec![facet]).unwrap();
    let observations = vec![obs(
        "obs",
        "episode-1",
        1,
        "performance",
        "expressiveness",
        0.90,
        "critic-a",
    )];
    let receipt = evaluate_persona_continuity(&profile, &observations).unwrap();
    assert_eq!(receipt.noncritical_violation_count, 0);
    assert_eq!(receipt.context_band_violation_count, 0);
}

#[test]
fn unknown_facet_fails_closed() {
    let profile = profile();
    let observation = obs(
        "obs",
        "episode-1",
        1,
        "ordinary",
        "invented-facet",
        0.5,
        "critic-a",
    );
    assert!(evaluate_persona_continuity(&profile, &[observation]).is_err());
}

#[test]
fn observation_order_does_not_change_commitments() {
    let profile = profile();
    let a = obs("a", "episode-1", 1, "ordinary", "warmth", 0.6, "critic-a");
    let b = obs("b", "episode-1", 1, "ordinary", "playfulness", 0.4, "critic-a");
    let c = obs("c", "episode-1", 1, "ordinary", "precision", 0.9, "critic-a");

    let first = evaluate_persona_continuity(&profile, &[a.clone(), b.clone(), c.clone()]).unwrap();
    let second = evaluate_persona_continuity(&profile, &[c, a, b]).unwrap();
    assert_eq!(first.observation_set_commitment, second.observation_set_commitment);
    assert_eq!(first.receipt_commitment, second.receipt_commitment);
}

#[test]
fn tampered_profile_commitment_fails_closed() {
    let mut profile = profile();
    profile.profile_version = "v2".into();
    assert!(evaluate_persona_continuity(&profile, &[]).is_err());
}
