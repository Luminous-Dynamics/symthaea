// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/intimacy_psychology.rs"]
mod intimacy_psychology;

use intimacy_psychology::*;

#[allow(clippy::too_many_arguments)]
fn evidence_at(
    id: &str,
    source: IntimacyPsychologySourceV1,
    temporal: IntimacyTemporalScopeV1,
    reality: IntimacyRealityScopeV1,
    context: &str,
    value: f32,
    confidence: f32,
    observed_at_ns: u64,
    valid_until_ns: Option<u64>,
) -> IntimacyPsychologyEvidenceV1 {
    IntimacyPsychologyEvidenceV1::new(
        id,
        IntimacyPsychologyDimensionV1::CommunicationDirectness,
        value,
        confidence,
        source,
        temporal,
        reality,
        context,
        observed_at_ns,
        valid_until_ns,
        format!("source:{id}"),
        IntimacyPsychologyRetentionV1::EphemeralSession,
    )
    .unwrap()
}

fn evidence(
    id: &str,
    source: IntimacyPsychologySourceV1,
    temporal: IntimacyTemporalScopeV1,
    reality: IntimacyRealityScopeV1,
    context: &str,
    value: f32,
    confidence: f32,
    valid_until_ns: Option<u64>,
) -> IntimacyPsychologyEvidenceV1 {
    evidence_at(
        id,
        source,
        temporal,
        reality,
        context,
        value,
        confidence,
        100,
        valid_until_ns,
    )
}

#[test]
fn no_evidence_remains_unknown() {
    let model = IntimacyPsychologyModelV1::default();
    assert!(model
        .estimate_at(
            IntimacyPsychologyDimensionV1::CommunicationDirectness,
            IntimacyTemporalScopeV1::TraitLike,
            IntimacyRealityScopeV1::RealWorld,
            "relationship:alpha",
            100,
        )
        .is_none());
}

#[test]
fn current_explicit_statement_beats_population_prior() {
    let mut model = IntimacyPsychologyModelV1::default();
    model
        .record(evidence(
            "population",
            IntimacyPsychologySourceV1::PopulationPrior,
            IntimacyTemporalScopeV1::CurrentState,
            IntimacyRealityScopeV1::RealWorld,
            "session:alpha",
            0.9,
            1.0,
            Some(200),
        ))
        .unwrap();
    model
        .record(evidence(
            "explicit",
            IntimacyPsychologySourceV1::ExplicitUserStatement,
            IntimacyTemporalScopeV1::CurrentState,
            IntimacyRealityScopeV1::RealWorld,
            "session:alpha",
            0.2,
            0.6,
            Some(200),
        ))
        .unwrap();

    let estimate = model
        .estimate_at(
            IntimacyPsychologyDimensionV1::CommunicationDirectness,
            IntimacyTemporalScopeV1::CurrentState,
            IntimacyRealityScopeV1::RealWorld,
            "session:alpha",
            150,
        )
        .unwrap();
    assert_eq!(estimate.evidence_id, "explicit");
    assert_eq!(estimate.value, 0.2);
}

#[test]
fn newer_explicit_correction_beats_older_higher_confidence_statement() {
    let mut model = IntimacyPsychologyModelV1::default();
    model
        .record(evidence_at(
            "old-explicit",
            IntimacyPsychologySourceV1::ExplicitUserStatement,
            IntimacyTemporalScopeV1::CurrentState,
            IntimacyRealityScopeV1::RealWorld,
            "session:alpha",
            0.9,
            1.0,
            100,
            Some(300),
        ))
        .unwrap();
    model
        .record(evidence_at(
            "new-correction",
            IntimacyPsychologySourceV1::ExplicitUserStatement,
            IntimacyTemporalScopeV1::CurrentState,
            IntimacyRealityScopeV1::RealWorld,
            "session:alpha",
            0.1,
            0.6,
            200,
            Some(300),
        ))
        .unwrap();

    let estimate = model
        .estimate_at(
            IntimacyPsychologyDimensionV1::CommunicationDirectness,
            IntimacyTemporalScopeV1::CurrentState,
            IntimacyRealityScopeV1::RealWorld,
            "session:alpha",
            250,
        )
        .unwrap();
    assert_eq!(estimate.evidence_id, "new-correction");
    assert_eq!(estimate.value, 0.1);
}

#[test]
fn reality_and_context_are_jointly_scoped() {
    let mut model = IntimacyPsychologyModelV1::default();
    model
        .record(evidence(
            "fantasy-only",
            IntimacyPsychologySourceV1::ExplicitUserStatement,
            IntimacyTemporalScopeV1::TraitLike,
            IntimacyRealityScopeV1::FantasyOnly,
            "story:alpha",
            1.0,
            1.0,
            None,
        ))
        .unwrap();

    assert!(model
        .estimate_at(
            IntimacyPsychologyDimensionV1::CommunicationDirectness,
            IntimacyTemporalScopeV1::TraitLike,
            IntimacyRealityScopeV1::RealWorld,
            "relationship:alpha",
            100,
        )
        .is_none());
}

#[test]
fn physiological_signal_cannot_be_promoted_to_trait() {
    let result = IntimacyPsychologyEvidenceV1::new(
        "physiology",
        IntimacyPsychologyDimensionV1::SexualExcitationPropensity,
        0.8,
        0.8,
        IntimacyPsychologySourceV1::PhysiologicalInference,
        IntimacyTemporalScopeV1::TraitLike,
        IntimacyRealityScopeV1::RealWorld,
        "session:alpha",
        100,
        Some(120),
        "sensor:opaque-ref",
        IntimacyPsychologyRetentionV1::EphemeralSession,
    );
    assert_eq!(
        result,
        Err(IntimacyPsychologyErrorV1::PhysiologyCannotEstablishTrait)
    );
}

#[test]
fn stale_current_state_is_not_reused() {
    let mut model = IntimacyPsychologyModelV1::default();
    model
        .record(evidence(
            "current",
            IntimacyPsychologySourceV1::ExplicitUserStatement,
            IntimacyTemporalScopeV1::CurrentState,
            IntimacyRealityScopeV1::RealWorld,
            "session:alpha",
            0.7,
            1.0,
            Some(150),
        ))
        .unwrap();

    assert!(model
        .estimate_at(
            IntimacyPsychologyDimensionV1::CommunicationDirectness,
            IntimacyTemporalScopeV1::CurrentState,
            IntimacyRealityScopeV1::RealWorld,
            "session:alpha",
            151,
        )
        .is_none());
}