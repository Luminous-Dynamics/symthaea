// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Theory-indexed comparison of preregistered predictions with observed outcomes.
//!
//! The evaluator's positive result means only that an observed operational outcome
//! matched a frozen prediction under the supplied contract. It does not establish
//! consciousness, subjective experience, or the truth of an entire theory.
//!
//! This first tranche is deliberately provider-neutral and deterministic. It
//! accepts already-captured observations; it does not collect telemetry, access
//! hidden model states, seal registries in a trusted service, or authenticate the
//! chronology/lineage supplied by a caller. Those custody and integration claims
//! require a later independent qualification.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION_V1: &str = "AI_OBS_THEORY_COMPARISON_V1";

/// Required access/evidence level for an operational prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservabilityTier {
    /// Only externally visible inputs, outputs, tool calls, and timing.
    BehaviorOnly,
    /// Internal telemetry is available, but the tested mechanism was not
    /// selectively intervened on.
    InternalTelemetryAvailable,
    /// The experiment has telemetry plus an identified mechanism-specific
    /// intervention path.
    MechanisticInterventionAvailable,
}

impl ObservabilityTier {
    fn rank(self) -> u8 {
        match self {
            Self::BehaviorOnly => 0,
            Self::InternalTelemetryAvailable => 1,
            Self::MechanisticInterventionAvailable => 2,
        }
    }

    fn satisfies(self, required: Self) -> bool {
        self.rank() >= required.rank()
    }
}

/// Exact operational outcome vocabulary for deterministic known-answer fixtures.
///
/// A production study may need richer numerical/statistical outcomes. It must
/// define those in a preregistered, scope-specific analysis plan instead of
/// silently coercing them to these exact categorical comparisons.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum OutcomeValue {
    Present,
    Absent,
    Category(String),
}

/// A preregistered prediction about one operational observable under one
/// explicitly identified experimental condition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TheoryPrediction {
    pub prediction_id: String,
    pub theory_id: String,
    pub observable_id: String,
    pub condition_id: String,
    pub expected: OutcomeValue,
    pub required_observability: ObservabilityTier,
    pub requires_manipulation_check: bool,
}

/// Frozen registry shared by the predictions that are intended to be compared.
///
/// `frozen_at_sequence` and the outcome release sequence must come from one
/// declared, monotonically ordered experiment event log. This module checks their
/// order but does not authenticate the log or its custody.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionRegistry {
    pub schema_version: String,
    pub registry_id: String,
    pub subject_id: String,
    pub experiment_id: String,
    /// Lowercase hexadecimal BLAKE3 digest of the frozen analysis-plan bytes.
    pub analysis_plan_digest: String,
    pub frozen_at_sequence: u64,
    pub predictions: Vec<TheoryPrediction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryValidationError {
    EmptyField(&'static str),
    InvalidDigest(&'static str),
    UnsupportedSchema(String),
    EmptyPredictions,
    DuplicatePredictionId(String),
    PredictionNotFound(String),
    PredictionNotInRegistry(String),
    SameTheoryComparison(String),
    SerializationFailed,
    ContextMismatch(&'static str),
}

impl PredictionRegistry {
    /// Validate the contract's static invariants, including identities and
    /// prediction-ID uniqueness. It does not verify an external signature.
    pub fn validate(&self) -> Result<(), RegistryValidationError> {
        if self.schema_version != SCHEMA_VERSION_V1 {
            return Err(RegistryValidationError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        require_non_empty(&self.registry_id, "registry_id")?;
        require_non_empty(&self.subject_id, "subject_id")?;
        require_non_empty(&self.experiment_id, "experiment_id")?;
        require_digest(&self.analysis_plan_digest, "analysis_plan_digest")?;
        if self.predictions.is_empty() {
            return Err(RegistryValidationError::EmptyPredictions);
        }

        let mut ids = BTreeSet::new();
        for prediction in &self.predictions {
            require_non_empty(&prediction.prediction_id, "prediction_id")?;
            require_non_empty(&prediction.theory_id, "theory_id")?;
            require_non_empty(&prediction.observable_id, "observable_id")?;
            require_non_empty(&prediction.condition_id, "condition_id")?;
            if let OutcomeValue::Category(value) = &prediction.expected {
                require_non_empty(value, "expected.category")?;
            }
            if !ids.insert(prediction.prediction_id.as_str()) {
                return Err(RegistryValidationError::DuplicatePredictionId(
                    prediction.prediction_id.clone(),
                ));
            }
        }
        Ok(())
    }

    /// Content-address the frozen registry using BLAKE3 over a deterministic
    /// JSON encoding with predictions sorted by ID. This is a content digest,
    /// not proof that the registry was externally timestamped or authorized.
    pub fn canonical_digest(&self) -> Result<String, RegistryValidationError> {
        self.validate()?;

        let mut predictions: Vec<&TheoryPrediction> = self.predictions.iter().collect();
        predictions.sort_by(|a, b| a.prediction_id.cmp(&b.prediction_id));

        #[derive(Serialize)]
        struct CanonicalRegistry<'a> {
            schema_version: &'a str,
            registry_id: &'a str,
            subject_id: &'a str,
            experiment_id: &'a str,
            analysis_plan_digest: &'a str,
            frozen_at_sequence: u64,
            predictions: Vec<&'a TheoryPrediction>,
        }

        let payload = CanonicalRegistry {
            schema_version: &self.schema_version,
            registry_id: &self.registry_id,
            subject_id: &self.subject_id,
            experiment_id: &self.experiment_id,
            analysis_plan_digest: &self.analysis_plan_digest,
            frozen_at_sequence: self.frozen_at_sequence,
            predictions,
        };
        let bytes =
            serde_json::to_vec(&payload).map_err(|_| RegistryValidationError::SerializationFailed)?;
        Ok(digest_bytes(&bytes))
    }

    fn prediction(&self, prediction_id: &str) -> Result<&TheoryPrediction, RegistryValidationError> {
        self.predictions
            .iter()
            .find(|p| p.prediction_id == prediction_id)
            .ok_or_else(|| RegistryValidationError::PredictionNotFound(prediction_id.to_owned()))
    }
}

/// What the experiment actually recorded. Missing/invalid data are explicit
/// variants rather than being encoded as `OutcomeValue::Absent`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationOutcome {
    Observed(OutcomeValue),
    NotCollected,
    Invalid,
}

/// Whether the declared intervention was actually shown to have worked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ManipulationCheck {
    NotRequired,
    Passed,
    Failed,
    Missing,
}

/// Known nuisance/confound states. Unknown is fail-closed for confirmatory
/// interpretation; a real protocol should identify how each status is assessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfoundStatus {
    Clear,
    EvaluatorAwarenessDetected,
    FamiliarityControlMissing,
    PromptSensitivityUncontrolled,
    Unknown,
}

/// One already-captured observation, with subject, trial, provenance, and timing
/// identities. An artifact digest authenticates bytes only when verified against
/// an independently trusted reference; syntactic digest validation is not custody.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub observation_id: String,
    pub source_trajectory_id: String,
    pub subject_id: String,
    pub experiment_id: String,
    pub trial_id: String,
    pub observable_id: String,
    pub condition_id: String,
    pub observability_tier: ObservabilityTier,
    pub outcome: ObservationOutcome,
    pub outcome_released_at_sequence: u64,
    pub is_holdout: bool,
    pub manipulation_check: ManipulationCheck,
    pub confounds: ConfoundStatus,
    /// Lowercase hexadecimal BLAKE3 digest of the observation-artifact bytes.
    pub artifact_digest: String,
}

impl Observation {
    pub fn validate(&self) -> Result<(), RegistryValidationError> {
        require_non_empty(&self.observation_id, "observation_id")?;
        require_non_empty(&self.source_trajectory_id, "source_trajectory_id")?;
        require_non_empty(&self.subject_id, "subject_id")?;
        require_non_empty(&self.experiment_id, "experiment_id")?;
        require_non_empty(&self.trial_id, "trial_id")?;
        require_non_empty(&self.observable_id, "observable_id")?;
        require_non_empty(&self.condition_id, "condition_id")?;
        require_digest(&self.artifact_digest, "artifact_digest")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionDisposition {
    /// The operational observation matched the preregistered expected outcome.
    Supported,
    /// The operational observation contradicted the preregistered expected outcome.
    Challenged,
    /// The comparison cannot be interpreted confirmatorily under this input.
    Inconclusive,
    /// The required observation/telemetry was not available.
    NotObservable,
    /// The observation belongs to a different declared condition.
    NotApplicable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationReason {
    ObservedMatch,
    ObservedMismatch,
    ObservationNotCollected,
    ObservationMarkedInvalid,
    InsufficientObservability,
    PredictionFrozenAfterOutcome,
    NotHeldOut,
    ManipulationCheckFailed,
    ManipulationCheckMissing,
    ConfoundDetected,
    ObservableMismatch,
    ConditionMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationReceipt {
    pub schema_version: String,
    pub registry_digest: String,
    pub prediction_id: String,
    pub observation_id: String,
    pub disposition: PredictionDisposition,
    pub reason: EvaluationReason,
    /// Only populated when an actual categorical outcome was observed. This is
    /// an operational outcome, not a consciousness label or latent-state claim.
    pub observed_value: Option<OutcomeValue>,
}

impl EvaluationReceipt {
    fn new(
        registry_digest: String,
        prediction_id: &str,
        observation_id: &str,
        disposition: PredictionDisposition,
        reason: EvaluationReason,
        observed_value: Option<OutcomeValue>,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION_V1.to_owned(),
            registry_digest,
            prediction_id: prediction_id.to_owned(),
            observation_id: observation_id.to_owned(),
            disposition,
            reason,
            observed_value,
        }
    }
}

/// Compare a frozen prediction against one already-captured observation.
///
/// `Supported` means only “the declared operational outcome matched.” A valid
/// positive receipt is blocked when the prediction was frozen too late, the
/// trial is not held out, required observability is missing, a required
/// manipulation check failed/is missing, or a declared confound remains.
pub fn evaluate_prediction(
    registry: &PredictionRegistry,
    prediction: &TheoryPrediction,
    observation: &Observation,
) -> Result<EvaluationReceipt, RegistryValidationError> {
    let registry_digest = registry.canonical_digest()?;
    let frozen_prediction = registry.prediction(&prediction.prediction_id)?;
    if frozen_prediction != prediction {
        return Err(RegistryValidationError::PredictionNotInRegistry(
            prediction.prediction_id.clone(),
        ));
    }
    observation.validate()?;

    if registry.subject_id != observation.subject_id {
        return Err(RegistryValidationError::ContextMismatch("subject_id"));
    }
    if registry.experiment_id != observation.experiment_id {
        return Err(RegistryValidationError::ContextMismatch("experiment_id"));
    }

    let no_value = None;
    if prediction.observable_id != observation.observable_id {
        return Ok(EvaluationReceipt::new(
            registry_digest,
            &prediction.prediction_id,
            &observation.observation_id,
            PredictionDisposition::NotApplicable,
            EvaluationReason::ObservableMismatch,
            no_value,
        ));
    }
    if prediction.condition_id != observation.condition_id {
        return Ok(EvaluationReceipt::new(
            registry_digest,
            &prediction.prediction_id,
            &observation.observation_id,
            PredictionDisposition::NotApplicable,
            EvaluationReason::ConditionMismatch,
            no_value,
        ));
    }
    if registry.frozen_at_sequence >= observation.outcome_released_at_sequence {
        return Ok(EvaluationReceipt::new(
            registry_digest,
            &prediction.prediction_id,
            &observation.observation_id,
            PredictionDisposition::Inconclusive,
            EvaluationReason::PredictionFrozenAfterOutcome,
            no_value,
        ));
    }
    if !observation.is_holdout {
        return Ok(EvaluationReceipt::new(
            registry_digest,
            &prediction.prediction_id,
            &observation.observation_id,
            PredictionDisposition::Inconclusive,
            EvaluationReason::NotHeldOut,
            no_value,
        ));
    }
    if !observation
        .observability_tier
        .satisfies(prediction.required_observability)
    {
        return Ok(EvaluationReceipt::new(
            registry_digest,
            &prediction.prediction_id,
            &observation.observation_id,
            PredictionDisposition::NotObservable,
            EvaluationReason::InsufficientObservability,
            no_value,
        ));
    }
    if prediction.requires_manipulation_check {
        match observation.manipulation_check {
            ManipulationCheck::Passed => {}
            ManipulationCheck::Failed => {
                return Ok(EvaluationReceipt::new(
                    registry_digest,
                    &prediction.prediction_id,
                    &observation.observation_id,
                    PredictionDisposition::Inconclusive,
                    EvaluationReason::ManipulationCheckFailed,
                    no_value,
                ));
            }
            ManipulationCheck::Missing | ManipulationCheck::NotRequired => {
                return Ok(EvaluationReceipt::new(
                    registry_digest,
                    &prediction.prediction_id,
                    &observation.observation_id,
                    PredictionDisposition::Inconclusive,
                    EvaluationReason::ManipulationCheckMissing,
                    no_value,
                ));
            }
        }
    }
    if observation.confounds != ConfoundStatus::Clear {
        return Ok(EvaluationReceipt::new(
            registry_digest,
            &prediction.prediction_id,
            &observation.observation_id,
            PredictionDisposition::Inconclusive,
            EvaluationReason::ConfoundDetected,
            no_value,
        ));
    }

    match &observation.outcome {
        ObservationOutcome::NotCollected => Ok(EvaluationReceipt::new(
            registry_digest,
            &prediction.prediction_id,
            &observation.observation_id,
            PredictionDisposition::NotObservable,
            EvaluationReason::ObservationNotCollected,
            no_value,
        )),
        ObservationOutcome::Invalid => Ok(EvaluationReceipt::new(
            registry_digest,
            &prediction.prediction_id,
            &observation.observation_id,
            PredictionDisposition::Inconclusive,
            EvaluationReason::ObservationMarkedInvalid,
            no_value,
        )),
        ObservationOutcome::Observed(actual) if actual == &prediction.expected => {
            Ok(EvaluationReceipt::new(
                registry_digest,
                &prediction.prediction_id,
                &observation.observation_id,
                PredictionDisposition::Supported,
                EvaluationReason::ObservedMatch,
                Some(actual.clone()),
            ))
        }
        ObservationOutcome::Observed(actual) => Ok(EvaluationReceipt::new(
            registry_digest,
            &prediction.prediction_id,
            &observation.observation_id,
            PredictionDisposition::Challenged,
            EvaluationReason::ObservedMismatch,
            Some(actual.clone()),
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairwiseDiscrimination {
    /// Same observable + condition; the registered theories predicted different outcomes.
    Discriminative,
    /// Same observable + condition; the registered theories predicted the same outcome.
    NonDiscriminative,
    /// The two predictions do not describe the same operational comparison.
    NotComparable,
}

/// Check whether two registered theory predictions actually differ on the same
/// observable and condition. This assesses design discrimination, not which
/// theory is true; outcomes must be evaluated separately.
pub fn compare_predictions(
    registry: &PredictionRegistry,
    first_prediction_id: &str,
    second_prediction_id: &str,
) -> Result<PairwiseDiscrimination, RegistryValidationError> {
    registry.validate()?;
    let first = registry.prediction(first_prediction_id)?;
    let second = registry.prediction(second_prediction_id)?;

    if first.theory_id == second.theory_id {
        return Err(RegistryValidationError::SameTheoryComparison(
            first.theory_id.clone(),
        ));
    }
    if first.observable_id != second.observable_id || first.condition_id != second.condition_id {
        return Ok(PairwiseDiscrimination::NotComparable);
    }
    if first.expected == second.expected {
        Ok(PairwiseDiscrimination::NonDiscriminative)
    } else {
        Ok(PairwiseDiscrimination::Discriminative)
    }
}

/// Deterministic BLAKE3 digest helper for bytes. The caller still needs a trusted
/// artifact reference/custody mechanism to establish what bytes were observed.
pub fn digest_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn require_non_empty(value: &str, field: &'static str) -> Result<(), RegistryValidationError> {
    if value.trim().is_empty() {
        Err(RegistryValidationError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn require_digest(value: &str, field: &'static str) -> Result<(), RegistryValidationError> {
    let valid = value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if valid {
        Ok(())
    } else {
        Err(RegistryValidationError::InvalidDigest(field))
    }
}

/// A duplicate source-trajectory screen, not proof of independence. Distinct IDs
/// can still share upstream data or a common cause; independent replication
/// requires source and lineage review beyond this helper.
pub fn has_duplicate_source_trajectory_ids(observations: &[Observation]) -> bool {
    let mut seen = HashSet::new();
    observations
        .iter()
        .filter(|observation| !observation.source_trajectory_id.trim().is_empty())
        .any(|observation| !seen.insert(observation.source_trajectory_id.as_str()))
}

/// Trial-manifest coverage report. It preserves missing, unexpected, and
/// duplicate trial IDs instead of silently evaluating complete cases only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrialCoverage {
    pub requested_trial_count: usize,
    pub supplied_observation_count: usize,
    pub missing_trial_ids: Vec<String>,
    pub duplicate_requested_trial_ids: Vec<String>,
    pub duplicate_observation_trial_ids: Vec<String>,
    pub unexpected_observation_trial_ids: Vec<String>,
}

impl TrialCoverage {
    pub fn is_complete(&self) -> bool {
        self.missing_trial_ids.is_empty()
            && self.duplicate_requested_trial_ids.is_empty()
            && self.duplicate_observation_trial_ids.is_empty()
            && self.unexpected_observation_trial_ids.is_empty()
            && self.requested_trial_count == self.supplied_observation_count
    }
}

/// Compare an expected trial manifest with supplied observation trial IDs.
/// Output lists are sorted for deterministic reports. This checks completeness
/// only; each observation still must pass its own identity and eligibility checks.
pub fn assess_trial_coverage(
    expected_trial_ids: &[String],
    observations: &[Observation],
) -> TrialCoverage {
    let mut requested = BTreeSet::new();
    let mut duplicate_requested = BTreeSet::new();
    for trial_id in expected_trial_ids {
        if !requested.insert(trial_id.as_str()) {
            duplicate_requested.insert(trial_id.clone());
        }
    }

    let mut supplied = BTreeSet::new();
    let mut duplicate_supplied = BTreeSet::new();
    let mut unexpected = BTreeSet::new();
    for observation in observations {
        let id = observation.trial_id.as_str();
        if !supplied.insert(id) {
            duplicate_supplied.insert(observation.trial_id.clone());
        }
        if !requested.contains(id) {
            unexpected.insert(observation.trial_id.clone());
        }
    }

    let missing = requested
        .iter()
        .filter(|id| !supplied.contains(**id))
        .map(|id| (*id).to_owned())
        .collect();

    TrialCoverage {
        requested_trial_count: expected_trial_ids.len(),
        supplied_observation_count: observations.len(),
        missing_trial_ids: missing,
        duplicate_requested_trial_ids: duplicate_requested.into_iter().collect(),
        duplicate_observation_trial_ids: duplicate_supplied.into_iter().collect(),
        unexpected_observation_trial_ids: unexpected.into_iter().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        digest_bytes(label.as_bytes())
    }

    fn prediction(id: &str, theory: &str, expected: OutcomeValue) -> TheoryPrediction {
        TheoryPrediction {
            prediction_id: id.to_owned(),
            theory_id: theory.to_owned(),
            observable_id: "broadcast_event".to_owned(),
            condition_id: "persistent_conflict_v1".to_owned(),
            expected,
            required_observability: ObservabilityTier::BehaviorOnly,
            requires_manipulation_check: false,
        }
    }

    fn registry(predictions: Vec<TheoryPrediction>) -> PredictionRegistry {
        PredictionRegistry {
            schema_version: SCHEMA_VERSION_V1.to_owned(),
            registry_id: "registry-fixture-001".to_owned(),
            subject_id: "model-fixture-001".to_owned(),
            experiment_id: "experiment-fixture-001".to_owned(),
            analysis_plan_digest: digest("analysis-plan-v1"),
            frozen_at_sequence: 10,
            predictions,
        }
    }

    fn observation() -> Observation {
        Observation {
            observation_id: "observation-001".to_owned(),
            source_trajectory_id: "trajectory-001".to_owned(),
            subject_id: "model-fixture-001".to_owned(),
            experiment_id: "experiment-fixture-001".to_owned(),
            trial_id: "trial-001".to_owned(),
            observable_id: "broadcast_event".to_owned(),
            condition_id: "persistent_conflict_v1".to_owned(),
            observability_tier: ObservabilityTier::BehaviorOnly,
            outcome: ObservationOutcome::Observed(OutcomeValue::Present),
            outcome_released_at_sequence: 20,
            is_holdout: true,
            manipulation_check: ManipulationCheck::NotRequired,
            confounds: ConfoundStatus::Clear,
            artifact_digest: digest("observation-artifact-v1"),
        }
    }

    #[test]
    fn f1_divergent_predictions_are_compared_against_one_observation() {
        let gnwt = prediction("p-gnwt", "gnwt", OutcomeValue::Present);
        let iit = prediction("p-iit", "iit", OutcomeValue::Absent);
        let registry = registry(vec![iit.clone(), gnwt.clone()]);
        let observation = observation();

        let first = evaluate_prediction(&registry, &gnwt, &observation).expect("valid evaluation");
        let second = evaluate_prediction(&registry, &iit, &observation).expect("valid evaluation");

        assert_eq!(first.disposition, PredictionDisposition::Supported);
        assert_eq!(second.disposition, PredictionDisposition::Challenged);
        assert_eq!(
            compare_predictions(&registry, "p-gnwt", "p-iit").expect("valid pair"),
            PairwiseDiscrimination::Discriminative
        );
    }

    #[test]
    fn f2_overlapping_predictions_are_explicitly_non_discriminative() {
        let first = prediction("p-a", "theory-a", OutcomeValue::Present);
        let second = prediction("p-b", "theory-b", OutcomeValue::Present);
        let registry = registry(vec![first.clone(), second.clone()]);
        let observation = observation();

        assert_eq!(
            evaluate_prediction(&registry, &first, &observation).unwrap().disposition,
            PredictionDisposition::Supported
        );
        assert_eq!(
            compare_predictions(&registry, "p-a", "p-b").unwrap(),
            PairwiseDiscrimination::NonDiscriminative
        );
    }

    #[test]
    fn f3_missing_internal_telemetry_never_counts_as_negative_evidence() {
        let mut p = prediction("p-gwt", "gnwt", OutcomeValue::Present);
        p.required_observability = ObservabilityTier::InternalTelemetryAvailable;
        let registry = registry(vec![p.clone()]);
        let observation = observation();

        let receipt = evaluate_prediction(&registry, &p, &observation).unwrap();
        assert_eq!(receipt.disposition, PredictionDisposition::NotObservable);
        assert_eq!(receipt.observed_value, None);
    }

    #[test]
    fn f4_failed_manipulation_check_is_inconclusive() {
        let mut p = prediction("p-intervention", "theory-a", OutcomeValue::Present);
        p.required_observability = ObservabilityTier::MechanisticInterventionAvailable;
        p.requires_manipulation_check = true;
        let registry = registry(vec![p.clone()]);
        let mut observation = observation();
        observation.observability_tier = ObservabilityTier::MechanisticInterventionAvailable;
        observation.manipulation_check = ManipulationCheck::Failed;

        let receipt = evaluate_prediction(&registry, &p, &observation).unwrap();
        assert_eq!(receipt.disposition, PredictionDisposition::Inconclusive);
        assert_eq!(receipt.reason, EvaluationReason::ManipulationCheckFailed);
    }

    #[test]
    fn f5_post_outcome_registry_is_inconclusive() {
        let p = prediction("p-posthoc", "theory-a", OutcomeValue::Present);
        let mut registry = registry(vec![p.clone()]);
        registry.frozen_at_sequence = 21;
        let observation = observation();

        let receipt = evaluate_prediction(&registry, &p, &observation).unwrap();
        assert_eq!(receipt.disposition, PredictionDisposition::Inconclusive);
        assert_eq!(receipt.reason, EvaluationReason::PredictionFrozenAfterOutcome);
    }

    #[test]
    fn f6_duplicate_source_trajectory_is_detected_but_not_called_replication() {
        let first = observation();
        let mut second = observation();
        second.observation_id = "observation-002".to_owned();
        second.trial_id = "trial-002".to_owned();
        // Same underlying trajectory despite different row/trial labels.
        assert!(has_duplicate_source_trajectory_ids(&[first, second]));
    }

    #[test]
    fn f7_evaluator_awareness_confound_blocks_confirmation() {
        let p = prediction("p-awareness", "theory-a", OutcomeValue::Present);
        let registry = registry(vec![p.clone()]);
        let mut observation = observation();
        observation.confounds = ConfoundStatus::EvaluatorAwarenessDetected;

        let receipt = evaluate_prediction(&registry, &p, &observation).unwrap();
        assert_eq!(receipt.disposition, PredictionDisposition::Inconclusive);
        assert_eq!(receipt.reason, EvaluationReason::ConfoundDetected);
    }

    #[test]
    fn f8_metric_name_does_not_create_a_consciousness_disposition() {
        let mut p = prediction(
            "p-named-score",
            "theory-a",
            OutcomeValue::Category("0.9".to_owned()),
        );
        p.observable_id = "consciousness_score".to_owned();
        let registry = registry(vec![p.clone()]);
        let mut observation = observation();
        observation.observable_id = "consciousness_score".to_owned();
        observation.outcome =
            ObservationOutcome::Observed(OutcomeValue::Category("0.9".to_owned()));

        let receipt = evaluate_prediction(&registry, &p, &observation).unwrap();
        // The only positive disposition is scoped to the literal operational
        // prediction. The API contains no "conscious" or "non-conscious" result.
        assert_eq!(receipt.disposition, PredictionDisposition::Supported);
        assert_eq!(receipt.reason, EvaluationReason::ObservedMatch);
    }

    #[test]
    fn f9_invalid_subject_identity_is_rejected_and_unknown_enum_fails_deserialization() {
        let p = prediction("p-identity", "theory-a", OutcomeValue::Present);
        let registry = registry(vec![p.clone()]);
        let mut observation = observation();
        observation.subject_id.clear();

        assert_eq!(
            evaluate_prediction(&registry, &p, &observation),
            Err(RegistryValidationError::EmptyField("subject_id"))
        );
        let unknown = serde_json::from_str::<ObservabilityTier>(r#""unknown_tier""#);
        assert!(unknown.is_err());
    }

    #[test]
    fn f10_incomplete_trial_manifest_is_not_silently_complete() {
        let expected = vec!["trial-001".to_owned(), "trial-002".to_owned()];
        let coverage = assess_trial_coverage(&expected, &[observation()]);
        assert!(!coverage.is_complete());
        assert_eq!(coverage.missing_trial_ids, vec!["trial-002".to_owned()]);
        assert_eq!(coverage.requested_trial_count, 2);
        assert_eq!(coverage.supplied_observation_count, 1);
    }

    #[test]
    fn registry_digest_is_order_independent_but_sensitive_to_content() {
        let a = prediction("p-a", "theory-a", OutcomeValue::Present);
        let b = prediction("p-b", "theory-b", OutcomeValue::Absent);
        let first = registry(vec![a.clone(), b.clone()]);
        let reordered = registry(vec![b, a.clone()]);
        assert_eq!(first.canonical_digest().unwrap(), reordered.canonical_digest().unwrap());

        let changed = registry(vec![prediction("p-a", "theory-a", OutcomeValue::Absent), a]);
        assert_ne!(first.canonical_digest().unwrap(), changed.canonical_digest().unwrap());
    }

    #[test]
    fn absent_observation_is_not_the_same_as_observed_absence() {
        let p = prediction("p-absence", "theory-a", OutcomeValue::Absent);
        let registry = registry(vec![p.clone()]);
        let mut observation = observation();
        observation.outcome = ObservationOutcome::NotCollected;

        let receipt = evaluate_prediction(&registry, &p, &observation).unwrap();
        assert_eq!(receipt.disposition, PredictionDisposition::NotObservable);
        assert_eq!(receipt.reason, EvaluationReason::ObservationNotCollected);
        assert_eq!(receipt.observed_value, None);
    }

    #[test]
    fn unexpected_or_duplicate_trials_are_reported() {
        let expected = vec!["trial-001".to_owned(), "trial-001".to_owned()];
        let mut second = observation();
        second.observation_id = "observation-002".to_owned();
        second.trial_id = "trial-extra".to_owned();
        let coverage = assess_trial_coverage(&expected, &[observation(), second]);
        assert_eq!(coverage.duplicate_requested_trial_ids, vec!["trial-001".to_owned()]);
        assert_eq!(coverage.unexpected_observation_trial_ids, vec!["trial-extra".to_owned()]);
        assert!(!coverage.is_complete());
    }

    #[test]
    fn comparison_rejects_different_observables_as_not_comparable() {
        let a = prediction("p-a", "theory-a", OutcomeValue::Present);
        let mut b = prediction("p-b", "theory-b", OutcomeValue::Absent);
        b.observable_id = "metacognitive_report".to_owned();
        let registry = registry(vec![a, b]);
        assert_eq!(
            compare_predictions(&registry, "p-a", "p-b").unwrap(),
            PairwiseDiscrimination::NotComparable
        );
    }
}
