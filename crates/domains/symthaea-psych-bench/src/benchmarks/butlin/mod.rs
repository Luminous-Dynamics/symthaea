// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Butlin et al. (2023) consciousness indicators (arXiv:2308.08708, Table 1).
//!
//! Tests architectural properties against the paper's actual 14 indicators:
//! RPT (Recurrent Processing), GWT (Global Workspace), HOT (Higher-Order),
//! PP (Predictive Processing, one indicator), AST (Attention Schema), AE
//! (Agency and Embodiment). The paper explicitly excludes IIT.
//!
//! When `symthaea-backend` is enabled, the ablation module provides
//! mechanistic ablation tests that prove each indicator is load-bearing, and
//! `ButlinIndicatorSuite::run()` merges that evidence into the report via
//! `report::annotate_with_ablation_results`. See
//! `BUTLIN_EVIDENCE_TIER_DESIGN.md` (crate root) for the evidence-tier model.

#[cfg(feature = "symthaea-backend")]
pub mod ablation;
#[cfg(feature = "symthaea-backend")]
pub mod ae2_empirical_runner;
#[cfg(feature = "symthaea-backend")]
pub mod gwt1_causal_envelope;
#[cfg(feature = "symthaea-backend")]
pub mod gwt1_causal_resolution;
pub mod gwt1_evidence_envelope;
#[cfg(feature = "symthaea-backend")]
pub mod gwt1_end_to_end;
pub mod gwt1_qualification;
pub mod indicators;
pub mod qualification_design;
pub mod qualification_runtime;
pub mod report;

#[cfg(feature = "symthaea-backend")]
pub use gwt1_causal_envelope::{
    GWT1_CAUSAL_EVIDENCE_ENVELOPE_SCHEMA_V1, Gwt1CausalArtifactDescriptorV1,
    Gwt1CausalArtifactFailureV1, Gwt1CausalEnvelopedEvidenceV1,
    Gwt1CausalEvidenceEnvelopeResolutionV1, Gwt1CausalEvidenceEnvelopeV1,
    Gwt1CausalIdentityFailureV1, build_gwt1_causal_envelope_v1,
    describe_gwt1_causal_artifact_v1, resolve_gwt1_causal_envelope_v1,
    run_gwt1_causal_enveloped_evidence_v1,
};
#[cfg(feature = "symthaea-backend")]
pub use gwt1_causal_resolution::{
    GWT1_CAUSAL_QUALIFICATION_SCHEMA_V1, Gwt1CausalEvidenceV1,
    Gwt1CausalQualificationFailureV1, Gwt1CausalQualificationOutcomeV1,
    Gwt1CausalQualificationResolutionV1, resolve_gwt1_causal_v1,
    run_gwt1_causal_evidence_v1,
};
pub use gwt1_evidence_envelope::{
    GWT1_EVIDENCE_ENVELOPE_SCHEMA_V1, GWT1_RAW_OBSERVATION_MEDIA_TYPE_V1,
    GWT1_RAW_OBSERVATION_SCHEMA_V1, Gwt1ArtifactIntegrityFailureV1,
    Gwt1EvidenceEnvelopeResolutionV1, Gwt1EvidenceEnvelopeV1, Gwt1RawObservationArtifactV1,
    describe_raw_observations_v1, raw_observation_blake3, resolve_gwt1_evidence_envelope_v1,
};
#[cfg(feature = "symthaea-backend")]
pub use gwt1_end_to_end::{
    Gwt1EndToEndErrorV1, Gwt1EndToEndEvidenceV1, Gwt1ExecutionIdentityV1,
    build_gwt1_evidence_v1, run_gwt1_end_to_end_v1,
};
pub use gwt1_qualification::{
    GWT1_MIN_TRAJECTORY_STEPS_V1, GWT1_PERTURBATIONS_V1, GWT1_QUALIFICATION_SCHEMA_V1,
    GWT1_REQUIRED_WORKERS_V1, GWT1_SPECIALISTS_V1, Gwt1PerturbationObservationV1,
    Gwt1QualificationFailureV1, Gwt1QualificationOutcomeV1, Gwt1QualificationResolutionV1,
    Gwt1SpecialistIdentityV1, Gwt1SpecialistQualificationReceiptV1, resolve_gwt1_v1,
};
pub use indicators::ButlinIndicatorSuite;
pub use qualification_design::{
    Comparison, ControlPurpose, ControlReadiness, DesignViolation, EffectDirection,
    EvidenceDependency, ExpectedEffect, MatchedDimension, PositiveControlId, PositiveControlPlan,
    ProbeValidity, QualificationDesign, ShamControlPlan, SharedGroup, planned_designs,
    shared_groups, validate_designs,
};
pub use qualification_runtime::{
    QualificationFailure, QualificationRunError, RuntimeQualification,
    check_identity_against_registry, resolve_outcome,
};
pub use report::{
    AblationResult, ButlinEvidenceBundle, ButlinIndicatorReport, EffectEstimate,
    EvidenceAnnotation, EvidenceMergeError, EvidenceOutcome, IndicatorEvidence, ProbeQuality,
    RuntimeConsciousnessData, SupportTier, annotate_with_ablation_results,
};
