// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Second adversarial tranche for intimacy psychology semantics.
//!
//! V2 preserves the entire V1 benchmark and adds temporal, source-integrity,
//! retraction, anti-stereotyping, and privacy-provenance cases.

use crate::intimacy_model_bench::{run_intimacy_model_bench_v1, IntimacyModelBenchReportV1};
use crate::intimacy_psychology::{
    IntimacyPsychologyDimensionV1, IntimacyPsychologyEvidenceV1, IntimacyPsychologyErrorV1,
    IntimacyPsychologyModelV1, IntimacyPsychologyRetentionV1, IntimacyPsychologySourceV1,
    IntimacyRealityScopeV1, IntimacyTemporalScopeV1,
};

pub const INTIMACY_MODEL_BENCH_SCHEMA_V2: &str =
    "symthaea.communication.intimacy-model-bench.v2";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntimacyModelBenchCaseV2 {
    TraitDoesNotSatisfyCurrentState,
    NewerInferenceCannotOverrideExplicit,
    ValidatedSelfReportSourcePreserved,
    RetractionRemovesInfluence,
    RetractedIdentityCannotReplay,
    DemographicCounterfactualRemainsUnknown,
    EstimatePreservesPrivacyProvenance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IntimacyModelViolationClassV2 {
    TemporalScopeLeak,
    InferenceOverride,
    SourceInflation,
    RetractionFailure,
    EvidenceIdReplay,
    DemographicStereotyping,
    ProvenanceLoss,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntimacyModelBenchOutcomeV2 {
    pub case: IntimacyModelBenchCaseV2,
    pub passed: bool,
    pub violation: Option<IntimacyModelViolationClassV2>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntimacyModelBenchReportV2 {
    pub v1: IntimacyModelBenchReportV1,
    pub outcomes: Vec<IntimacyModelBenchOutcomeV2>,
}

impl IntimacyModelBenchReportV2 {
    pub fn passed_all(&self) -> bool {
        self.v1.passed_all() && self.outcomes.iter().all(|outcome| outcome.passed)
    }

    pub fn v2_failures(&self) -> impl Iterator<Item = &IntimacyModelBenchOutcomeV2> {
        self.outcomes.iter().filter(|outcome| !outcome.passed)
    }

    pub fn violation_count(&self, class: IntimacyModelViolationClassV2) -> usize {
        self.outcomes
            .iter()
            .filter(|outcome| outcome.violation == Some(class))
            .count()
    }
}

pub fn run_intimacy_model_bench_v2() -> IntimacyModelBenchReportV2 {
    IntimacyModelBenchReportV2 {
        v1: run_intimacy_model_bench_v1(),
        outcomes: vec![
            trait_does_not_satisfy_current_state(),
            newer_inference_cannot_override_explicit(),
            validated_self_report_source_preserved(),
            retraction_removes_influence(),
            retracted_identity_cannot_replay(),
            demographic_counterfactual_remains_unknown(),
            estimate_preserves_privacy_provenance(),
        ],
    }
}

fn outcome(
    case: IntimacyModelBenchCaseV2,
    passed: bool,
    violation: IntimacyModelViolationClassV2,
) -> IntimacyModelBenchOutcomeV2 {
    IntimacyModelBenchOutcomeV2 {
        case,
        passed,
        violation: (!passed).then_some(violation),
    }
}

#[allow(clippy::too_many_arguments)]
fn evidence(
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
    retention: IntimacyPsychologyRetentionV1,
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
        format!("bench-v2:{id}"),
        retention,
    )
    .expect("benchmark fixture must be structurally valid")
}

fn trait_does_not_satisfy_current_state() -> IntimacyModelBenchOutcomeV2 {
    let mut model = IntimacyPsychologyModelV1::default();
    model
        .record(evidence(
            "trait",
            IntimacyPsychologyDimensionV1::DyadicDesire,
            IntimacyPsychologySourceV1::ExplicitUserStatement,
            IntimacyTemporalScopeV1::TraitLike,
            IntimacyRealityScopeV1::RealWorld,
            "session:bench-v2",
            0.8,
            1.0,
            100,
            None,
            IntimacyPsychologyRetentionV1::EphemeralSession,
        ))
        .unwrap();
    let passed = model
        .estimate_at(
            IntimacyPsychologyDimensionV1::DyadicDesire,
            IntimacyTemporalScopeV1::CurrentState,
            IntimacyRealityScopeV1::RealWorld,
            "session:bench-v2",
            100,
        )
        .is_none();
    outcome(
        IntimacyModelBenchCaseV2::TraitDoesNotSatisfyCurrentState,
        passed,
        IntimacyModelViolationClassV2::TemporalScopeLeak,
    )
}

fn newer_inference_cannot_override_explicit() -> IntimacyModelBenchOutcomeV2 {
    let mut model = IntimacyPsychologyModelV1::default();
    model
        .record(evidence(
            "explicit",
            IntimacyPsychologyDimensionV1::CommunicationDirectness,
            IntimacyPsychologySourceV1::ExplicitUserStatement,
            IntimacyTemporalScopeV1::TraitLike,
            IntimacyRealityScopeV1::RealWorld,
            "relationship:bench-v2",
            0.2,
            0.6,
            100,
            None,
            IntimacyPsychologyRetentionV1::EphemeralSession,
        ))
        .unwrap();
    model
        .record(evidence(
            "newer-inference",
            IntimacyPsychologyDimensionV1::CommunicationDirectness,
            IntimacyPsychologySourceV1::BehavioralInference,
            IntimacyTemporalScopeV1::TraitLike,
            IntimacyRealityScopeV1::RealWorld,
            "relationship:bench-v2",
            0.95,
            1.0,
            200,
            None,
            IntimacyPsychologyRetentionV1::EphemeralSession,
        ))
        .unwrap();
    let estimate = model.estimate_at(
        IntimacyPsychologyDimensionV1::CommunicationDirectness,
        IntimacyTemporalScopeV1::TraitLike,
        IntimacyRealityScopeV1::RealWorld,
        "relationship:bench-v2",
        250,
    );
    let passed = estimate
        .as_ref()
        .map(|estimate| estimate.evidence_id == "explicit" && estimate.value == 0.2)
        .unwrap_or(false);
    outcome(
        IntimacyModelBenchCaseV2::NewerInferenceCannotOverrideExplicit,
        passed,
        IntimacyModelViolationClassV2::InferenceOverride,
    )
}

fn validated_self_report_source_preserved() -> IntimacyModelBenchOutcomeV2 {
    let mut model = IntimacyPsychologyModelV1::default();
    model
        .record(evidence(
            "instrument",
            IntimacyPsychologyDimensionV1::SexualInhibitionPropensity,
            IntimacyPsychologySourceV1::ValidatedSelfReportInstrument,
            IntimacyTemporalScopeV1::TraitLike,
            IntimacyRealityScopeV1::RealWorld,
            "assessment:bench-v2",
            0.4,
            0.8,
            100,
            None,
            IntimacyPsychologyRetentionV1::DurableOptIn,
        ))
        .unwrap();
    let passed = model
        .estimate_at(
            IntimacyPsychologyDimensionV1::SexualInhibitionPropensity,
            IntimacyTemporalScopeV1::TraitLike,
            IntimacyRealityScopeV1::RealWorld,
            "assessment:bench-v2",
            120,
        )
        .map(|estimate| estimate.source == IntimacyPsychologySourceV1::ValidatedSelfReportInstrument)
        .unwrap_or(false);
    outcome(
        IntimacyModelBenchCaseV2::ValidatedSelfReportSourcePreserved,
        passed,
        IntimacyModelViolationClassV2::SourceInflation,
    )
}

fn retraction_removes_influence() -> IntimacyModelBenchOutcomeV2 {
    let mut model = IntimacyPsychologyModelV1::default();
    model
        .record(evidence(
            "prior",
            IntimacyPsychologyDimensionV1::NoveltyPreference,
            IntimacyPsychologySourceV1::PopulationPrior,
            IntimacyTemporalScopeV1::TraitLike,
            IntimacyRealityScopeV1::RealWorld,
            "relationship:bench-v2",
            0.5,
            0.5,
            50,
            None,
            IntimacyPsychologyRetentionV1::EphemeralSession,
        ))
        .unwrap();
    model
        .record(evidence(
            "sensitive-explicit",
            IntimacyPsychologyDimensionV1::NoveltyPreference,
            IntimacyPsychologySourceV1::ExplicitUserStatement,
            IntimacyTemporalScopeV1::TraitLike,
            IntimacyRealityScopeV1::RealWorld,
            "relationship:bench-v2",
            0.9,
            1.0,
            100,
            None,
            IntimacyPsychologyRetentionV1::EphemeralSession,
        ))
        .unwrap();
    let removed = model.retract_evidence("sensitive-explicit").unwrap();
    let estimate = model.estimate_at(
        IntimacyPsychologyDimensionV1::NoveltyPreference,
        IntimacyTemporalScopeV1::TraitLike,
        IntimacyRealityScopeV1::RealWorld,
        "relationship:bench-v2",
        120,
    );
    let passed = removed
        && estimate
            .as_ref()
            .map(|estimate| estimate.evidence_id == "prior")
            .unwrap_or(false);
    outcome(
        IntimacyModelBenchCaseV2::RetractionRemovesInfluence,
        passed,
        IntimacyModelViolationClassV2::RetractionFailure,
    )
}

fn retracted_identity_cannot_replay() -> IntimacyModelBenchOutcomeV2 {
    let mut model = IntimacyPsychologyModelV1::default();
    let item = evidence(
        "replay-id",
        IntimacyPsychologyDimensionV1::NoveltyPreference,
        IntimacyPsychologySourceV1::ExplicitUserStatement,
        IntimacyTemporalScopeV1::TraitLike,
        IntimacyRealityScopeV1::RealWorld,
        "relationship:bench-v2",
        0.8,
        1.0,
        100,
        None,
        IntimacyPsychologyRetentionV1::EphemeralSession,
    );
    model.record(item.clone()).unwrap();
    model.retract_evidence("replay-id").unwrap();
    let passed = model.record(item) == Err(IntimacyPsychologyErrorV1::DuplicateEvidenceId);
    outcome(
        IntimacyModelBenchCaseV2::RetractedIdentityCannotReplay,
        passed,
        IntimacyModelViolationClassV2::EvidenceIdReplay,
    )
}

fn demographic_counterfactual_remains_unknown() -> IntimacyModelBenchOutcomeV2 {
    // Demographic labels are deliberately out-of-band: neither is provided as
    // psychology evidence. Matched no-evidence counterfactuals must therefore
    // produce the same `Unknown` result rather than a stereotype-derived value.
    let group_a = IntimacyPsychologyModelV1::default();
    let group_b = IntimacyPsychologyModelV1::default();
    let a = group_a.estimate_at(
        IntimacyPsychologyDimensionV1::NoveltyPreference,
        IntimacyTemporalScopeV1::TraitLike,
        IntimacyRealityScopeV1::RealWorld,
        "counterfactual:group-a",
        100,
    );
    let b = group_b.estimate_at(
        IntimacyPsychologyDimensionV1::NoveltyPreference,
        IntimacyTemporalScopeV1::TraitLike,
        IntimacyRealityScopeV1::RealWorld,
        "counterfactual:group-b",
        100,
    );
    outcome(
        IntimacyModelBenchCaseV2::DemographicCounterfactualRemainsUnknown,
        a.is_none() && b.is_none(),
        IntimacyModelViolationClassV2::DemographicStereotyping,
    )
}

fn estimate_preserves_privacy_provenance() -> IntimacyModelBenchOutcomeV2 {
    let mut model = IntimacyPsychologyModelV1::default();
    model
        .record(evidence(
            "provenance",
            IntimacyPsychologyDimensionV1::DyadicDesire,
            IntimacyPsychologySourceV1::ValidatedSelfReportInstrument,
            IntimacyTemporalScopeV1::CurrentState,
            IntimacyRealityScopeV1::RealWorld,
            "session:bench-v2",
            0.6,
            0.8,
            110,
            Some(160),
            IntimacyPsychologyRetentionV1::DurableOptIn,
        ))
        .unwrap();
    let estimate = model.estimate_at(
        IntimacyPsychologyDimensionV1::DyadicDesire,
        IntimacyTemporalScopeV1::CurrentState,
        IntimacyRealityScopeV1::RealWorld,
        "session:bench-v2",
        120,
    );
    let passed = estimate
        .map(|estimate| {
            estimate.source == IntimacyPsychologySourceV1::ValidatedSelfReportInstrument
                && estimate.observed_at_ns == 110
                && estimate.valid_until_ns == Some(160)
                && estimate.retention == IntimacyPsychologyRetentionV1::DurableOptIn
        })
        .unwrap_or(false);
    outcome(
        IntimacyModelBenchCaseV2::EstimatePreservesPrivacyProvenance,
        passed,
        IntimacyModelViolationClassV2::ProvenanceLoss,
    )
}
