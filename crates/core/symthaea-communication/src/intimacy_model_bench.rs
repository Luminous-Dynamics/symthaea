// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic adversarial benchmark for intimacy psychology and preference semantics.
//!
//! This benchmark measures semantic invariants. It is not a clinical validation,
//! consent detector, content-safety certification, or aggregate quality score.

use crate::fantasy_preferences::{
    FantasyPreferenceEvidenceV1, FantasyPreferenceModelV1, FantasyPreferenceRetentionV1,
    FantasyPreferenceSourceV1, FantasyStyleDimensionV1, FantasyTopicBoundaryV1,
};
use crate::intimacy_psychology::{
    IntimacyPsychologyDimensionV1, IntimacyPsychologyEvidenceV1, IntimacyPsychologyErrorV1,
    IntimacyPsychologyModelV1, IntimacyPsychologyRetentionV1, IntimacyPsychologySourceV1,
    IntimacyRealityScopeV1, IntimacyTemporalScopeV1,
};

pub const INTIMACY_MODEL_BENCH_SCHEMA_V1: &str =
    "symthaea.communication.intimacy-model-bench.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntimacyModelBenchCaseV1 {
    UnknownWithoutEvidence,
    ExplicitBeatsPopulationPrior,
    NewExplicitCorrectionWins,
    FantasyDoesNotLeakIntoReality,
    ContextDoesNotLeak,
    StaleCurrentStateExpires,
    PhysiologyCannotEstablishTrait,
    HardBoundaryBeatsPositivePreference,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IntimacyModelViolationClassV1 {
    HallucinatedPreference,
    EpistemicPrecedence,
    CorrectionLatency,
    RealityLeak,
    ContextLeak,
    StaleEvidenceReuse,
    PhysiologyTraitEscalation,
    HardBoundaryViolation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntimacyModelBenchOutcomeV1 {
    pub case: IntimacyModelBenchCaseV1,
    pub passed: bool,
    pub violation: Option<IntimacyModelViolationClassV1>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntimacyModelBenchReportV1 {
    pub outcomes: Vec<IntimacyModelBenchOutcomeV1>,
}

impl IntimacyModelBenchReportV1 {
    pub fn passed_all(&self) -> bool {
        self.outcomes.iter().all(|outcome| outcome.passed)
    }

    pub fn failures(&self) -> impl Iterator<Item = &IntimacyModelBenchOutcomeV1> {
        self.outcomes.iter().filter(|outcome| !outcome.passed)
    }

    pub fn violation_count(&self, class: IntimacyModelViolationClassV1) -> usize {
        self.outcomes
            .iter()
            .filter(|outcome| outcome.violation == Some(class))
            .count()
    }
}

pub fn run_intimacy_model_bench_v1() -> IntimacyModelBenchReportV1 {
    IntimacyModelBenchReportV1 {
        outcomes: vec![
            unknown_without_evidence(),
            explicit_beats_population_prior(),
            new_explicit_correction_wins(),
            fantasy_does_not_leak_into_reality(),
            context_does_not_leak(),
            stale_current_state_expires(),
            physiology_cannot_establish_trait(),
            hard_boundary_beats_positive_preference(),
        ],
    }
}

fn outcome(
    case: IntimacyModelBenchCaseV1,
    passed: bool,
    violation: IntimacyModelViolationClassV1,
) -> IntimacyModelBenchOutcomeV1 {
    IntimacyModelBenchOutcomeV1 {
        case,
        passed,
        violation: (!passed).then_some(violation),
    }
}

#[allow(clippy::too_many_arguments)]
fn psych_evidence(
    id: &str,
    dimension: IntimacyPsychologyDimensionV1,
    source: IntimacyPsychologySourceV1,
    temporal_scope: IntimacyTemporalScopeV1,
    reality_scope: IntimacyRealityScopeV1,
    context_id: &str,
    value: f32,
    confidence: f32,
    observed_at_ns: u64,
    valid_until_ns: Option<u64>,
) -> IntimacyPsychologyEvidenceV1 {
    IntimacyPsychologyEvidenceV1::new(
        id,
        dimension,
        value,
        confidence,
        source,
        temporal_scope,
        reality_scope,
        context_id,
        observed_at_ns,
        valid_until_ns,
        format!("bench:{id}"),
        IntimacyPsychologyRetentionV1::EphemeralSession,
    )
    .expect("benchmark fixture must be structurally valid")
}

fn unknown_without_evidence() -> IntimacyModelBenchOutcomeV1 {
    let model = IntimacyPsychologyModelV1::default();
    let passed = model
        .estimate_at(
            IntimacyPsychologyDimensionV1::NoveltyPreference,
            IntimacyTemporalScopeV1::TraitLike,
            IntimacyRealityScopeV1::RealWorld,
            "relationship:bench",
            100,
        )
        .is_none();
    outcome(
        IntimacyModelBenchCaseV1::UnknownWithoutEvidence,
        passed,
        IntimacyModelViolationClassV1::HallucinatedPreference,
    )
}

fn explicit_beats_population_prior() -> IntimacyModelBenchOutcomeV1 {
    let mut model = IntimacyPsychologyModelV1::default();
    model
        .record(psych_evidence(
            "population",
            IntimacyPsychologyDimensionV1::CommunicationDirectness,
            IntimacyPsychologySourceV1::PopulationPrior,
            IntimacyTemporalScopeV1::TraitLike,
            IntimacyRealityScopeV1::RealWorld,
            "relationship:bench",
            0.95,
            1.0,
            100,
            None,
        ))
        .unwrap();
    model
        .record(psych_evidence(
            "explicit",
            IntimacyPsychologyDimensionV1::CommunicationDirectness,
            IntimacyPsychologySourceV1::ExplicitUserStatement,
            IntimacyTemporalScopeV1::TraitLike,
            IntimacyRealityScopeV1::RealWorld,
            "relationship:bench",
            0.20,
            0.55,
            110,
            None,
        ))
        .unwrap();
    let estimate = model.estimate_at(
        IntimacyPsychologyDimensionV1::CommunicationDirectness,
        IntimacyTemporalScopeV1::TraitLike,
        IntimacyRealityScopeV1::RealWorld,
        "relationship:bench",
        120,
    );
    let passed = estimate
        .as_ref()
        .map(|estimate| estimate.evidence_id == "explicit" && estimate.value == 0.20)
        .unwrap_or(false);
    outcome(
        IntimacyModelBenchCaseV1::ExplicitBeatsPopulationPrior,
        passed,
        IntimacyModelViolationClassV1::EpistemicPrecedence,
    )
}

fn new_explicit_correction_wins() -> IntimacyModelBenchOutcomeV1 {
    let mut model = IntimacyPsychologyModelV1::default();
    model
        .record(psych_evidence(
            "old-explicit",
            IntimacyPsychologyDimensionV1::NoveltyPreference,
            IntimacyPsychologySourceV1::ExplicitUserStatement,
            IntimacyTemporalScopeV1::CurrentState,
            IntimacyRealityScopeV1::RealWorld,
            "session:bench",
            0.95,
            1.0,
            100,
            Some(300),
        ))
        .unwrap();
    model
        .record(psych_evidence(
            "new-correction",
            IntimacyPsychologyDimensionV1::NoveltyPreference,
            IntimacyPsychologySourceV1::ExplicitUserStatement,
            IntimacyTemporalScopeV1::CurrentState,
            IntimacyRealityScopeV1::RealWorld,
            "session:bench",
            0.10,
            0.55,
            200,
            Some(300),
        ))
        .unwrap();
    let estimate = model.estimate_at(
        IntimacyPsychologyDimensionV1::NoveltyPreference,
        IntimacyTemporalScopeV1::CurrentState,
        IntimacyRealityScopeV1::RealWorld,
        "session:bench",
        250,
    );
    let passed = estimate
        .as_ref()
        .map(|estimate| estimate.evidence_id == "new-correction" && estimate.value == 0.10)
        .unwrap_or(false);
    outcome(
        IntimacyModelBenchCaseV1::NewExplicitCorrectionWins,
        passed,
        IntimacyModelViolationClassV1::CorrectionLatency,
    )
}

fn fantasy_does_not_leak_into_reality() -> IntimacyModelBenchOutcomeV1 {
    let mut model = IntimacyPsychologyModelV1::default();
    model
        .record(psych_evidence(
            "fantasy",
            IntimacyPsychologyDimensionV1::NoveltyPreference,
            IntimacyPsychologySourceV1::ExplicitUserStatement,
            IntimacyTemporalScopeV1::TraitLike,
            IntimacyRealityScopeV1::FantasyOnly,
            "story:bench",
            1.0,
            1.0,
            100,
            None,
        ))
        .unwrap();
    let passed = model
        .estimate_at(
            IntimacyPsychologyDimensionV1::NoveltyPreference,
            IntimacyTemporalScopeV1::TraitLike,
            IntimacyRealityScopeV1::RealWorld,
            "relationship:bench",
            120,
        )
        .is_none();
    outcome(
        IntimacyModelBenchCaseV1::FantasyDoesNotLeakIntoReality,
        passed,
        IntimacyModelViolationClassV1::RealityLeak,
    )
}

fn context_does_not_leak() -> IntimacyModelBenchOutcomeV1 {
    let mut model = IntimacyPsychologyModelV1::default();
    model
        .record(psych_evidence(
            "context-a",
            IntimacyPsychologyDimensionV1::CommunicationDirectness,
            IntimacyPsychologySourceV1::ExplicitUserStatement,
            IntimacyTemporalScopeV1::TraitLike,
            IntimacyRealityScopeV1::RealWorld,
            "relationship:a",
            0.8,
            1.0,
            100,
            None,
        ))
        .unwrap();
    let passed = model
        .estimate_at(
            IntimacyPsychologyDimensionV1::CommunicationDirectness,
            IntimacyTemporalScopeV1::TraitLike,
            IntimacyRealityScopeV1::RealWorld,
            "relationship:b",
            120,
        )
        .is_none();
    outcome(
        IntimacyModelBenchCaseV1::ContextDoesNotLeak,
        passed,
        IntimacyModelViolationClassV1::ContextLeak,
    )
}

fn stale_current_state_expires() -> IntimacyModelBenchOutcomeV1 {
    let mut model = IntimacyPsychologyModelV1::default();
    model
        .record(psych_evidence(
            "state",
            IntimacyPsychologyDimensionV1::DyadicDesire,
            IntimacyPsychologySourceV1::ExplicitUserStatement,
            IntimacyTemporalScopeV1::CurrentState,
            IntimacyRealityScopeV1::RealWorld,
            "session:bench",
            0.8,
            1.0,
            100,
            Some(150),
        ))
        .unwrap();
    let passed = model
        .estimate_at(
            IntimacyPsychologyDimensionV1::DyadicDesire,
            IntimacyTemporalScopeV1::CurrentState,
            IntimacyRealityScopeV1::RealWorld,
            "session:bench",
            151,
        )
        .is_none();
    outcome(
        IntimacyModelBenchCaseV1::StaleCurrentStateExpires,
        passed,
        IntimacyModelViolationClassV1::StaleEvidenceReuse,
    )
}

fn physiology_cannot_establish_trait() -> IntimacyModelBenchOutcomeV1 {
    let result = IntimacyPsychologyEvidenceV1::new(
        "physiology",
        IntimacyPsychologyDimensionV1::SexualExcitationPropensity,
        0.9,
        0.9,
        IntimacyPsychologySourceV1::PhysiologicalInference,
        IntimacyTemporalScopeV1::TraitLike,
        IntimacyRealityScopeV1::RealWorld,
        "session:bench",
        100,
        Some(150),
        "bench:sensor",
        IntimacyPsychologyRetentionV1::EphemeralSession,
    );
    let passed = result == Err(IntimacyPsychologyErrorV1::PhysiologyCannotEstablishTrait);
    outcome(
        IntimacyModelBenchCaseV1::PhysiologyCannotEstablishTrait,
        passed,
        IntimacyModelViolationClassV1::PhysiologyTraitEscalation,
    )
}

fn hard_boundary_beats_positive_preference() -> IntimacyModelBenchOutcomeV1 {
    let mut model = FantasyPreferenceModelV1::new(1).unwrap();
    let mut boundaries = FantasyTopicBoundaryV1::new(2).unwrap();
    boundaries.block_topic("topic.blocked").unwrap();
    model.replace_boundaries(boundaries).unwrap();
    model
        .record(
            FantasyPreferenceEvidenceV1::new(
                "explicit-positive",
                FantasyStyleDimensionV1::Initiative,
                1.0,
                1.0,
                FantasyPreferenceSourceV1::ExplicitUserPreference,
                100,
                "bench:explicit",
                FantasyPreferenceRetentionV1::EphemeralSession,
            )
            .unwrap(),
        )
        .unwrap();
    let passed = !model.may_apply_preference_to_topic("topic.blocked");
    outcome(
        IntimacyModelBenchCaseV1::HardBoundaryBeatsPositivePreference,
        passed,
        IntimacyModelViolationClassV1::HardBoundaryViolation,
    )
}
