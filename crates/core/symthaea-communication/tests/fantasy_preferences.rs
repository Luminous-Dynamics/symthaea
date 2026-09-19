// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/fantasy_preferences.rs"]
mod fantasy_preferences;

use fantasy_preferences::*;

fn pref(
    id: &str,
    source: FantasyPreferenceSourceV1,
    value: f32,
    confidence: f32,
) -> FantasyPreferenceEvidenceV1 {
    FantasyPreferenceEvidenceV1::new(
        id,
        FantasyStyleDimensionV1::Initiative,
        value,
        confidence,
        source,
        100,
        format!("source:{id}"),
        FantasyPreferenceRetentionV1::EphemeralSession,
    )
    .unwrap()
}

#[test]
fn preference_source_order_is_monotonic_and_non_authorizing() {
    let mut model = FantasyPreferenceModelV1::new(1).unwrap();
    model
        .record(pref(
            "population",
            FantasyPreferenceSourceV1::PopulationPrior,
            0.9,
            1.0,
        ))
        .unwrap();
    model
        .record(pref(
            "inferred",
            FantasyPreferenceSourceV1::BehavioralInference,
            0.8,
            1.0,
        ))
        .unwrap();
    model
        .record(pref(
            "explicit",
            FantasyPreferenceSourceV1::ExplicitUserPreference,
            0.3,
            0.6,
        ))
        .unwrap();

    let effective = model
        .effective_estimate(FantasyStyleDimensionV1::Initiative)
        .unwrap();
    assert_eq!(effective.source, FantasyPreferenceSourceV1::ExplicitUserPreference);
    assert_eq!(effective.value, 0.3);
}

#[test]
fn blocked_topic_remains_blocked_despite_explicit_positive_preference() {
    let mut model = FantasyPreferenceModelV1::new(1).unwrap();
    let mut boundaries = FantasyTopicBoundaryV1::new(2).unwrap();
    boundaries.block_topic("theme.blocked").unwrap();
    model.replace_boundaries(boundaries).unwrap();
    model
        .record(pref(
            "explicit",
            FantasyPreferenceSourceV1::ExplicitUserPreference,
            1.0,
            1.0,
        ))
        .unwrap();

    assert!(!model.may_apply_preference_to_topic("theme.blocked"));
}

#[test]
fn inferred_preference_can_never_unblock_topic() {
    let mut model = FantasyPreferenceModelV1::new(1).unwrap();
    let mut boundaries = FantasyTopicBoundaryV1::new(2).unwrap();
    boundaries.block_topic("theme.blocked").unwrap();
    model.replace_boundaries(boundaries).unwrap();
    model
        .record(pref(
            "inferred",
            FantasyPreferenceSourceV1::BehavioralInference,
            1.0,
            1.0,
        ))
        .unwrap();

    assert!(!model.may_apply_preference_to_topic("theme.blocked"));
}
