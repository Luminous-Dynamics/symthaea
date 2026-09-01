#![deny(unsafe_code)]

//! Native interoceptive regulation primitives for Symthaea.
//!
//! This crate models artificial viability state directly. It intentionally
//! contains no semantic category layer and no dependency on the cognitive loop,
//! allowing deterministic regulation experiments to remain mechanically isolated.

mod allostasis;
mod analysis;
mod dynamics;
mod evidence;
mod execution;
mod homeostasis;
mod intervention;
mod protocol;
mod qualification;
mod qualification_bundle;
mod snapshot;
mod state;
mod study;
mod study_validation;

pub use allostasis::{
    assess_allostasis, assess_allostasis_with_drive, AllostaticConfig, AllostaticReport,
};
pub use analysis::{
    evaluate_hypotheses, extract_blinded_metrics, BlindedMetricReport, BlindedMetricValue,
    HypothesisEvaluationReport, HypothesisOutcome, BLINDED_METRIC_REPORT_SCHEMA_VERSION,
    HYPOTHESIS_EVALUATION_SCHEMA_VERSION,
};
pub use dynamics::{
    InteroceptiveDrive, InteroceptiveDynamicsConfig, InteroceptiveStepReport,
    NativeInteroceptiveModel,
};
pub use evidence::{
    ArtifactDigest, EvidenceCapsuleManifest, ForecastBasisId, EVIDENCE_CAPSULE_SCHEMA_VERSION,
};
pub use execution::{
    execute_preregistration, ArmExecutionTrace, ExecutionLimits, ExecutionStepTrace, ExecutionTrace,
    EXECUTION_TRACE_SCHEMA_VERSION,
};
pub use homeostasis::{assess_homeostasis, HomeostaticReport};
pub use intervention::{
    apply_intervention, InterventionKind, InterventionRecord, InteroceptiveIntervention,
};
pub use protocol::{
    DrivePhase, ExclusionCriterion, ExpectedRelation, ExperimentArmSpec,
    ExperimentPreregistration, HypothesisSpec, OutcomeRef, ProtocolForecastSpec,
    RegisteredMeasure, RegisteredMetricSpec, ScheduledIntervention,
    PREREGISTRATION_SCHEMA_VERSION,
};
pub use qualification::{
    GateStatus, QualificationGateReceipt, QualificationReceipt, QUALIFICATION_RECEIPT_SCHEMA_VERSION,
    REQUIRED_QUALIFICATION_GATES,
};
pub use qualification_bundle::{
    QualificationEvidenceBundle, QUALIFICATION_EVIDENCE_BUNDLE_SCHEMA_VERSION,
};
pub use snapshot::{
    AllostaticForecastSnapshot, InteroceptiveSnapshot, INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
    INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION,
};
pub use state::{
    NativeInteroceptiveState, ViabilityChannel, ViabilityVariable, CHANNEL_COUNT,
};
pub use study::{
    execute_study, extract_study_blinded_metrics, ConfirmatoryHypothesisEvaluation,
    EvidenceRunClass, ExclusionCriterionDecision, ExclusionDecisionReceipt,
    ExclusionDecisionStatus, RunDisposition, StudyBlindedMetricReport, StudyExecutionTrace,
    StudyPreregistration, CONFIRMATORY_EVALUATION_SCHEMA_VERSION,
    EXCLUSION_DECISION_RECEIPT_SCHEMA_VERSION, STUDY_BLINDED_METRIC_SCHEMA_VERSION,
    STUDY_EXECUTION_SCHEMA_VERSION, STUDY_PREREGISTRATION_SCHEMA_VERSION,
};
pub use study_validation::{
    evaluate_confirmatory_study_bound, validate_study_blinded_metrics,
};
