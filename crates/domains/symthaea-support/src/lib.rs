// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Support — IT support intelligence sub-crate
//!
//! Provides triage, diagnostics, knowledge management, privacy scrubbing,
//! action engine, predictive engine, and evidence-bound IT system intelligence
//! for universal IT support.

pub mod actions;
pub mod change_timeline;
pub mod diagnostic_beliefs;
pub mod diagnostics;
pub mod federation;
pub mod knowledge;
#[cfg(feature = "logparse-adapter")]
pub mod logparse_adapter;
pub mod predictive;
pub mod privacy;
pub mod scrubber;
pub mod system_state;
pub mod technology;
pub mod telemetry;
pub mod triage;
pub mod types;

pub use change_timeline::{
    temporal_relation, ChangeClockV1, ChangeId, ChangeKindV1, ChangeTimelineError,
    ChangeTimelineV1, FailureWindowV1, SystemChangeV1, TemporalRelationV1,
};
pub use diagnostic_beliefs::{
    rank_tests_by_information_gain, CausalHypothesisV1, DiagnosticBeliefError,
    DiagnosticOutcomeId, DiagnosticTestId, DiagnosticTestModelV1, ExpectedInformationGainV1,
    HypothesisDistributionV1, HypothesisId, HypothesisStatusV1,
};
#[cfg(feature = "logparse-adapter")]
pub use logparse_adapter::{
    LogObservationAdapterConfigV1, LogObservationAdapterError, LogObservationAdapterV1,
    LogObservationPolicyV1,
};
pub use system_state::{
    CurrentnessStatusV1, EntityId, EntityKindV1, ObservationClockV1, ObservationId,
    ObservationProvenanceV1, ObservationSourceKindV1, RelationId, RelationKindV1, StateValueV1,
    SystemEntityV1, SystemObservationV1, SystemRelationV1, SystemStateGraphError,
    SystemStateGraphV1,
};
pub use technology::{
    ApplicabilityAssessmentV1, ApplicabilityScopeV1, ApplicabilityStatusV1, StringSelectorV1,
    TechnologyIdentityError, TechnologyIdentityV1,
};
