// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! R4.3 context-bound, trainable closed-form continuous-time predictor.
//!
//! R4.3 intentionally starts with one narrow learned claim: predict privileged-
//! truth body-root X linear velocity on the R4.2 nominal simple-backend context
//! profile. Every other semantic output explicitly abstains. This avoids turning
//! a first successful system-identification experiment into an unearned claim of
//! full humanoid world-model coverage.
//!
//! The learned scalar follows a CfC-style closed-form update:
//!
//! `y(t+dt) = drive + (y(t) - drive) * exp(-dt / tau)`
//!
//! with `drive = tanh(w · features)`. Features are versioned and deterministic:
//! exact R3/R4 HDC state block means + backend-applied physical-action summaries
//! + exact R4.2 external force. `tau` and `w` are learned by deterministic,
//! canonically ordered gradient descent.
//!
//! Training cases are content-addressed as
//! `transition_digest + contextual_input_digest`. Structural validation alone
//! cannot identify a target/context swap when the R4.1 base input is identical;
//! the case commitment detects mutation after capture. Authenticity of the
//! original capture remains an evidence-lineage/signature concern, not something
//! a local hash can infer retroactively.
//!
//! Nothing here grants control, safety, capability, or motor authority.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;
use std::fmt::Write as _;

use symthaea_core::hdc::sensorimotor_contingencies::{
    SensorimotorAddressV1, SensorimotorComponentV1, SensorimotorObservationV1,
    SensorimotorQuantityV1, SensorimotorSubjectV1,
};

use crate::continuous_predictor::{
    HumanoidContinuousPredictorDescriptorV1, HumanoidContinuousPredictorFamilyV1,
    HumanoidPredictionUnavailableReasonV1, HumanoidPredictedPhysicalValueV1,
    HumanoidPredictedValueV1, HumanoidPredictorActionSemanticsV1,
    HumanoidPredictorMemorySemanticsV1, HumanoidPredictorProvenanceV1,
    HumanoidPredictorTimeSemanticsV1,
};
use crate::morphology::HumanoidMorphology;
use crate::predictor_context::{
    capture_contextual_predictor_case_v1, ContextInstrumentedSimpleHumanoidSimulator,
    HumanoidContextualPredictorInputV1, HumanoidPredictorContextErrorV1,
    HumanoidPredictorContextEvidenceV1, HumanoidPredictorContextSourceV1,
};
use crate::transition_evidence::{
    HumanoidTransitionEvidenceErrorV1, HumanoidTransitionEvidenceV1,
};
use crate::types::{ActuationMode, HumanoidCommand};

const ARCHITECTURE_ID_V1: &str =
    "symthaea.humanoid.root-velocity-x.diagonal-cfc-hdc-context.v1";
const FEATURE_EXTRACTOR_ID_V1: &str =
    "symthaea.humanoid.contextual-cfc-features.block-mean.v1";
const TRAINER_ID_V1: &str =
    "symthaea.humanoid.root-velocity-x.diagonal-cfc-trainer.v1";

const TRAINING_CASE_DOMAIN_V1: &[u8] =
    b"symthaea.humanoid.contextual-cfc-training-case.v1\0";
const TRAINING_LINEAGE_DOMAIN_V1: &[u8] =
    b"symthaea.humanoid.contextual-cfc-training-lineage.v1\0";
const CHECKPOINT_DOMAIN_V1: &[u8] =
    b"symthaea.humanoid.contextual-cfc-checkpoint.v1\0";
const DESCRIPTOR_DOMAIN_V1: &[u8] =
    b"symthaea.humanoid.contextual-cfc-descriptor.v1\0";
const PREDICTION_DOMAIN_V1: &[u8] =
    b"symthaea.humanoid.contextual-cfc-prediction.v1\0";

const HDC_FEATURES: usize = 8;
const ACTION_FEATURES: usize = 4;
const FORCE_FEATURES: usize = 3;
const FEATURE_DIM: usize = 1 + HDC_FEATURES + ACTION_FEATURES + FORCE_FEATURES;

const MIN_TAU_SECONDS: f64 = 0.002;
const MAX_TAU_SECONDS: f64 = 2.0;

/// Deterministic optimizer and feature scaling configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HumanoidContextualCfcTrainingConfigV1 {
    pub epochs: u32,
    pub learning_rate: f64,
    pub initial_tau_seconds: f64,
    pub force_feature_scale_n: f64,
    pub weight_clip: f64,
}

impl Default for HumanoidContextualCfcTrainingConfigV1 {
    fn default() -> Self {
        Self {
            epochs: 320,
            learning_rate: 0.08,
            initial_tau_seconds: 0.10,
            force_feature_scale_n: 100.0,
            weight_clip: 8.0,
        }
    }
}

impl HumanoidContextualCfcTrainingConfigV1 {
    pub fn validate(&self) -> Result<(), HumanoidContextualCfcErrorV1> {
        if self.epochs == 0
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || self.learning_rate > 1.0
            || !self.initial_tau_seconds.is_finite()
            || !(MIN_TAU_SECONDS..=MAX_TAU_SECONDS).contains(&self.initial_tau_seconds)
            || !self.force_feature_scale_n.is_finite()
            || self.force_feature_scale_n <= 0.0
            || !self.weight_clip.is_finite()
            || self.weight_clip <= 0.0
        {
            return Err(HumanoidContextualCfcErrorV1::InvalidTrainingConfig);
        }
        Ok(())
    }
}

/// One target-bearing training datum sealed at capture time.
///
/// The target is never part of inference input. `case_digest_hex` records the
/// selected target/context pairing so later dataset mutation is detectable.
#[derive(Debug, Clone)]
pub struct HumanoidContextualCfcTrainingCaseV1 {
    pub transition: HumanoidTransitionEvidenceV1,
    pub input: HumanoidContextualPredictorInputV1,
    pub case_digest_hex: String,
}

impl HumanoidContextualCfcTrainingCaseV1 {
    fn seal(
        transition: HumanoidTransitionEvidenceV1,
        input: HumanoidContextualPredictorInputV1,
    ) -> Result<Self, HumanoidContextualCfcErrorV1> {
        validate_case_structure(&transition, &input)?;
        let case_digest_hex = compute_case_digest(&transition, &input);
        let case = Self {
            transition,
            input,
            case_digest_hex,
        };
        case.validate()?;
        Ok(case)
    }

    /// Load already-bound evidence. This validates the supplied commitment
    /// instead of silently creating a new commitment for a potentially swapped
    /// target/context pair.
    pub fn from_bound_evidence(
        transition: HumanoidTransitionEvidenceV1,
        input: HumanoidContextualPredictorInputV1,
        case_digest_hex: impl Into<String>,
    ) -> Result<Self, HumanoidContextualCfcErrorV1> {
        let case = Self {
            transition,
            input,
            case_digest_hex: case_digest_hex.into(),
        };
        case.validate()?;
        Ok(case)
    }

    pub fn validate(&self) -> Result<(), HumanoidContextualCfcErrorV1> {
        validate_case_structure(&self.transition, &self.input)?;
        if !valid_hex_digest(&self.case_digest_hex) {
            return Err(HumanoidContextualCfcErrorV1::InvalidTrainingCaseDigest);
        }
        if self.case_digest_hex != compute_case_digest(&self.transition, &self.input) {
            return Err(HumanoidContextualCfcErrorV1::TrainingCaseDigestMismatch);
        }
        Ok(())
    }
}

/// Capture and immediately seal one R4.3 training case.
///
/// Caller-controlled values are validated by R4.2 before backend mutation.
pub fn capture_contextual_cfc_training_case_v1(
    simulator: &mut ContextInstrumentedSimpleHumanoidSimulator,
    requested_command: &HumanoidCommand,
    requested_dt_seconds: f64,
    external_force_world_n: [f64; 3],
    clock_domain_id: impl Into<String>,
    episode_id: impl Into<String>,
) -> Result<HumanoidContextualCfcTrainingCaseV1, HumanoidContextualCfcErrorV1> {
    let (transition, input) = capture_contextual_predictor_case_v1(
        simulator,
        requested_command,
        requested_dt_seconds,
        external_force_world_n,
        clock_domain_id,
        episode_id,
    )
    .map_err(HumanoidContextualCfcErrorV1::Context)?;
    HumanoidContextualCfcTrainingCaseV1::seal(transition, input)
}

fn validate_case_structure(
    transition: &HumanoidTransitionEvidenceV1,
    input: &HumanoidContextualPredictorInputV1,
) -> Result<(), HumanoidContextualCfcErrorV1> {
    transition
        .validate()
        .map_err(HumanoidContextualCfcErrorV1::Transition)?;
    input
        .validate()
        .map_err(HumanoidContextualCfcErrorV1::Context)?;
    let expected = crate::continuous_predictor::HumanoidPredictorInputV1::from_transition(
        transition,
    )
    .map_err(HumanoidContextualCfcErrorV1::Predictor)?;
    if expected.input_digest_hex != input.base_input.input_digest_hex {
        return Err(HumanoidContextualCfcErrorV1::TrainingPairMismatch);
    }
    Ok(())
}

fn compute_case_digest(
    transition: &HumanoidTransitionEvidenceV1,
    input: &HumanoidContextualPredictorInputV1,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(TRAINING_CASE_DOMAIN_V1);
    feed_str(&mut hasher, &transition.transition_digest_hex);
    feed_str(&mut hasher, &input.contextual_input_digest_hex);
    digest_hex(hasher.finalize().as_bytes())
}

/// Context-aware learned predictor identity.
///
/// It deliberately does not reuse the ordinary R4.1 predicted-state identity:
/// this model consumes privileged R4.2 context and must remain visibly bound to
/// that fact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanoidContextualCfcDescriptorV1 {
    pub schema_id: String,
    pub predictor_id: String,
    pub family: HumanoidContinuousPredictorFamilyV1,
    pub provenance: HumanoidPredictorProvenanceV1,
    pub time_semantics: HumanoidPredictorTimeSemanticsV1,
    pub action_semantics: HumanoidPredictorActionSemanticsV1,
    pub memory_semantics: HumanoidPredictorMemorySemanticsV1,
    pub physical_action_mode: ActuationMode,
    pub context_schema_id: String,
    pub context_source: HumanoidPredictorContextSourceV1,
    pub feature_extractor_id: String,
    pub state_hdc_profile_id: String,
    pub output_representation_id: String,
    pub output_role_digest_hex: String,
}

impl HumanoidContextualCfcDescriptorV1 {
    pub const SCHEMA_ID: &'static str =
        "symthaea.humanoid.contextual-cfc-descriptor.v1";
    pub const OUTPUT_REPRESENTATION_ID: &'static str =
        "symthaea.humanoid.contextual-cfc-prediction.v1";

    pub fn validate(&self) -> Result<(), HumanoidContextualCfcErrorV1> {
        if self.schema_id != Self::SCHEMA_ID
            || self.predictor_id.trim().is_empty()
            || self.family != HumanoidContinuousPredictorFamilyV1::ClosedFormContinuousTime
            || self.time_semantics
                != HumanoidPredictorTimeSemanticsV1::ExplicitBackendAppliedDeltaSeconds
            || self.action_semantics
                != HumanoidPredictorActionSemanticsV1::BackendAppliedPhysicalScalars
            || self.memory_semantics != HumanoidPredictorMemorySemanticsV1::PureOneStep
            || self.physical_action_mode != ActuationMode::TorqueNewtonMetres
            || self.context_schema_id != HumanoidPredictorContextEvidenceV1::SCHEMA_ID
            || self.context_source != HumanoidPredictorContextSourceV1::PrivilegedSimulatorTruth
            || self.feature_extractor_id != FEATURE_EXTRACTOR_ID_V1
            || self.state_hdc_profile_id
                != HumanoidContinuousPredictorDescriptorV1::STATE_HDC_PROFILE_ID
            || self.output_representation_id != Self::OUTPUT_REPRESENTATION_ID
            || !valid_hex_digest(&self.output_role_digest_hex)
        {
            return Err(HumanoidContextualCfcErrorV1::InvalidDescriptor);
        }
        match &self.provenance {
            HumanoidPredictorProvenanceV1::LearnedCheckpoint {
                architecture_id,
                checkpoint_digest_hex,
                training_lineage_digest_hex,
            } if architecture_id == ARCHITECTURE_ID_V1
                && valid_hex_digest(checkpoint_digest_hex)
                && valid_hex_digest(training_lineage_digest_hex) => {}
            _ => return Err(HumanoidContextualCfcErrorV1::InvalidDescriptor),
        }
        Ok(())
    }

    pub fn descriptor_digest_hex(&self) -> Result<String, HumanoidContextualCfcErrorV1> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(DESCRIPTOR_DOMAIN_V1);
        feed_str(&mut hasher, &self.schema_id);
        feed_str(&mut hasher, &self.predictor_id);
        feed_str(&mut hasher, "closed_form_continuous_time");
        if let HumanoidPredictorProvenanceV1::LearnedCheckpoint {
            architecture_id,
            checkpoint_digest_hex,
            training_lineage_digest_hex,
        } = &self.provenance
        {
            feed_str(&mut hasher, architecture_id);
            feed_str(&mut hasher, checkpoint_digest_hex);
            feed_str(&mut hasher, training_lineage_digest_hex);
        }
        feed_str(&mut hasher, "explicit_backend_applied_delta_seconds");
        feed_str(&mut hasher, "backend_applied_physical_scalars");
        feed_str(&mut hasher, "pure_one_step");
        feed_str(&mut hasher, "torque_newton_metres");
        feed_str(&mut hasher, &self.context_schema_id);
        feed_str(&mut hasher, "privileged_simulator_truth");
        feed_str(&mut hasher, &self.feature_extractor_id);
        feed_str(&mut hasher, &self.state_hdc_profile_id);
        feed_str(&mut hasher, &self.output_representation_id);
        feed_str(&mut hasher, &self.output_role_digest_hex);
        Ok(digest_hex(hasher.finalize().as_bytes()))
    }
}

#[derive(Debug, Clone)]
struct RootVelocityXCfcParametersV1 {
    address: SensorimotorAddressV1,
    contract_digest: [u8; 32],
    weights: Vec<f64>,
    log_tau: f64,
}

impl RootVelocityXCfcParametersV1 {
    fn tau_seconds(&self) -> f64 {
        self.log_tau
            .exp()
            .clamp(MIN_TAU_SECONDS, MAX_TAU_SECONDS)
    }

    fn predict_normalized(&self, current: f64, features: &[f64], dt: f64) -> f64 {
        let drive_pre: f64 = self
            .weights
            .iter()
            .zip(features.iter())
            .map(|(weight, feature)| weight * feature)
            .sum();
        let drive = drive_pre.tanh();
        let decay = (-dt / self.tau_seconds()).exp();
        drive + (current - drive) * decay
    }
}

/// First learned context-aware continuous-time humanoid model.
///
/// V1 intentionally learns only body-root X linear velocity. Other roles are
/// emitted as explicit abstentions.
#[derive(Debug, Clone)]
pub struct HumanoidRootVelocityXCfcPredictorV1 {
    descriptor: HumanoidContextualCfcDescriptorV1,
    config: HumanoidContextualCfcTrainingConfigV1,
    parameters: RootVelocityXCfcParametersV1,
}

impl HumanoidRootVelocityXCfcPredictorV1 {
    pub fn train(
        cases: &[HumanoidContextualCfcTrainingCaseV1],
        config: HumanoidContextualCfcTrainingConfigV1,
    ) -> Result<Self, HumanoidContextualCfcErrorV1> {
        config.validate()?;
        if cases.is_empty() {
            return Err(HumanoidContextualCfcErrorV1::EmptyTrainingSet);
        }

        let mut ordered: Vec<&HumanoidContextualCfcTrainingCaseV1> = cases.iter().collect();
        for case in &ordered {
            case.validate()?;
            if case.input.base_input.backend_applied_action.physical_mode
                != ActuationMode::TorqueNewtonMetres
            {
                return Err(HumanoidContextualCfcErrorV1::UnsupportedActionMode);
            }
        }
        ordered.sort_by(|left, right| left.case_digest_hex.cmp(&right.case_digest_hex));

        let output_address =
            root_velocity_x_address(&ordered[0].input.base_input.pre_state.policy_observations)?
                .clone();
        let output_digest = output_address
            .semantic_digest()
            .map_err(HumanoidContextualCfcErrorV1::Sensorimotor)?;

        for case in &ordered {
            let pre = root_velocity_x_address(
                &case.input.base_input.pre_state.policy_observations,
            )?;
            let post = root_velocity_x_address(&case.transition.post_state.policy_observations)?;
            if pre
                .semantic_digest()
                .map_err(HumanoidContextualCfcErrorV1::Sensorimotor)?
                != output_digest
                || post
                    .semantic_digest()
                    .map_err(HumanoidContextualCfcErrorV1::Sensorimotor)?
                    != output_digest
            {
                return Err(HumanoidContextualCfcErrorV1::OutputRoleMismatch);
            }
        }

        let mut rows = Vec::with_capacity(ordered.len());
        for case in &ordered {
            rows.push(PreparedRowV1 {
                features: contextual_features_v1(&case.input, &config)?,
                current: normalize_physical(
                    root_velocity_x_value(
                        &case.input.base_input.pre_state.policy_observations,
                    )?,
                    &output_address,
                )?,
                target: normalize_physical(
                    root_velocity_x_value(&case.transition.post_state.policy_observations)?,
                    &output_address,
                )?,
                dt: case.input.base_input.applied_dt_seconds,
            });
        }

        let mut parameters = RootVelocityXCfcParametersV1 {
            address: output_address,
            contract_digest: output_digest,
            weights: vec![0.0; FEATURE_DIM],
            log_tau: config.initial_tau_seconds.ln(),
        };

        let min_log_tau = MIN_TAU_SECONDS.ln();
        let max_log_tau = MAX_TAU_SECONDS.ln();
        for _ in 0..config.epochs {
            for row in &rows {
                let tau = parameters.tau_seconds();
                let drive_pre: f64 = parameters
                    .weights
                    .iter()
                    .zip(row.features.iter())
                    .map(|(weight, feature)| weight * feature)
                    .sum();
                let drive = drive_pre.tanh();
                let decay = (-row.dt / tau).exp();
                let prediction = drive + (row.current - drive) * decay;
                let error = prediction - row.target;
                let drive_gradient = (1.0 - decay) * (1.0 - drive * drive);

                for (weight, feature) in parameters.weights.iter_mut().zip(row.features.iter()) {
                    *weight -= config.learning_rate * error * drive_gradient * feature;
                    *weight = weight.clamp(-config.weight_clip, config.weight_clip);
                }

                let tau_gradient =
                    (row.current - drive) * decay * (row.dt / tau);
                parameters.log_tau -= config.learning_rate * error * tau_gradient;
                parameters.log_tau = parameters.log_tau.clamp(min_log_tau, max_log_tau);
            }
        }

        if parameters
            .weights
            .iter()
            .any(|weight| !weight.is_finite())
            || !parameters.log_tau.is_finite()
        {
            return Err(HumanoidContextualCfcErrorV1::NonFiniteCheckpoint);
        }

        let training_lineage_digest_hex = training_lineage_digest(&ordered, &config)?;
        let checkpoint_digest_hex =
            checkpoint_digest(&parameters, &config, &training_lineage_digest_hex)?;
        let predictor_id = format!("{ARCHITECTURE_ID_V1}:{}", &checkpoint_digest_hex[..16]);
        let descriptor = HumanoidContextualCfcDescriptorV1 {
            schema_id: HumanoidContextualCfcDescriptorV1::SCHEMA_ID.to_string(),
            predictor_id,
            family: HumanoidContinuousPredictorFamilyV1::ClosedFormContinuousTime,
            provenance: HumanoidPredictorProvenanceV1::LearnedCheckpoint {
                architecture_id: ARCHITECTURE_ID_V1.to_string(),
                checkpoint_digest_hex,
                training_lineage_digest_hex,
            },
            time_semantics:
                HumanoidPredictorTimeSemanticsV1::ExplicitBackendAppliedDeltaSeconds,
            action_semantics:
                HumanoidPredictorActionSemanticsV1::BackendAppliedPhysicalScalars,
            memory_semantics: HumanoidPredictorMemorySemanticsV1::PureOneStep,
            physical_action_mode: ActuationMode::TorqueNewtonMetres,
            context_schema_id: HumanoidPredictorContextEvidenceV1::SCHEMA_ID.to_string(),
            context_source: HumanoidPredictorContextSourceV1::PrivilegedSimulatorTruth,
            feature_extractor_id: FEATURE_EXTRACTOR_ID_V1.to_string(),
            state_hdc_profile_id:
                HumanoidContinuousPredictorDescriptorV1::STATE_HDC_PROFILE_ID.to_string(),
            output_representation_id:
                HumanoidContextualCfcDescriptorV1::OUTPUT_REPRESENTATION_ID.to_string(),
            output_role_digest_hex: digest_hex(&output_digest),
        };
        descriptor.validate()?;

        Ok(Self {
            descriptor,
            config,
            parameters,
        })
    }

    pub fn descriptor(&self) -> &HumanoidContextualCfcDescriptorV1 {
        &self.descriptor
    }

    pub fn config(&self) -> HumanoidContextualCfcTrainingConfigV1 {
        self.config
    }

    pub fn tau_seconds(&self) -> f64 {
        self.parameters.tau_seconds()
    }

    pub fn predict(
        &self,
        input: &HumanoidContextualPredictorInputV1,
    ) -> Result<HumanoidContextualCfcPredictionV1, HumanoidContextualCfcErrorV1> {
        input
            .validate()
            .map_err(HumanoidContextualCfcErrorV1::Context)?;
        self.descriptor.validate()?;
        if input.base_input.backend_applied_action.physical_mode
            != self.descriptor.physical_action_mode
        {
            return Err(HumanoidContextualCfcErrorV1::UnsupportedActionMode);
        }

        let features = contextual_features_v1(input, &self.config)?;
        let dt = input.base_input.applied_dt_seconds;
        let mut predictions =
            Vec::with_capacity(input.base_input.pre_state.policy_observations.len());

        for observation in &input.base_input.pre_state.policy_observations {
            let address = observation.address().clone();
            let digest = address
                .semantic_digest()
                .map_err(HumanoidContextualCfcErrorV1::Sensorimotor)?;
            let prediction = match observation {
                SensorimotorObservationV1::Missing { .. } => {
                    HumanoidPredictedValueV1::Unavailable(
                        HumanoidPredictionUnavailableReasonV1::InputMissing,
                    )
                }
                SensorimotorObservationV1::Measured(measurement)
                    if digest == self.parameters.contract_digest =>
                {
                    let current = normalize_physical(measurement.value, &address)?;
                    let predicted = self
                        .parameters
                        .predict_normalized(current, &features, dt);
                    HumanoidPredictedValueV1::Predicted(denormalize_physical(
                        predicted,
                        &address,
                    )?)
                }
                SensorimotorObservationV1::Measured(_) => {
                    HumanoidPredictedValueV1::Unavailable(
                        HumanoidPredictionUnavailableReasonV1::UnsupportedRole,
                    )
                }
            };
            predictions.push(HumanoidPredictedPhysicalValueV1 {
                address,
                prediction,
            });
        }

        HumanoidContextualCfcPredictionV1::new(
            &self.descriptor,
            input,
            predictions,
        )
    }
}

/// Context-bound, role-scoped prediction.
///
/// It still carries the complete semantic contract set: unsupported roles are
/// explicit abstentions rather than silently omitted fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidContextualCfcPredictionV1 {
    pub schema_id: String,
    pub predictor_id: String,
    pub predictor_descriptor_digest_hex: String,
    pub contextual_input_digest_hex: String,
    pub morphology: HumanoidMorphology,
    pub target_timestamp_seconds: f64,
    pub policy_predictions: Vec<HumanoidPredictedPhysicalValueV1>,
    pub prediction_digest_hex: String,
}

impl HumanoidContextualCfcPredictionV1 {
    pub const SCHEMA_ID: &'static str =
        "symthaea.humanoid.contextual-cfc-prediction.v1";

    pub fn new(
        descriptor: &HumanoidContextualCfcDescriptorV1,
        input: &HumanoidContextualPredictorInputV1,
        policy_predictions: Vec<HumanoidPredictedPhysicalValueV1>,
    ) -> Result<Self, HumanoidContextualCfcErrorV1> {
        descriptor.validate()?;
        input
            .validate()
            .map_err(HumanoidContextualCfcErrorV1::Context)?;

        let mut keyed = Vec::with_capacity(policy_predictions.len());
        for prediction in policy_predictions {
            validate_prediction_value(&prediction)?;
            let digest = prediction
                .address
                .semantic_digest()
                .map_err(HumanoidContextualCfcErrorV1::Sensorimotor)?;
            keyed.push((digest, prediction));
        }
        keyed.sort_by_key(|(digest, _)| *digest);

        let mut result = Self {
            schema_id: Self::SCHEMA_ID.to_string(),
            predictor_id: descriptor.predictor_id.clone(),
            predictor_descriptor_digest_hex: descriptor.descriptor_digest_hex()?,
            contextual_input_digest_hex: input.contextual_input_digest_hex.clone(),
            morphology: input.base_input.morphology,
            target_timestamp_seconds: input.base_input.target_timestamp_seconds,
            policy_predictions: keyed
                .into_iter()
                .map(|(_, prediction)| prediction)
                .collect(),
            prediction_digest_hex: String::new(),
        };
        result.validate_without_digest(descriptor, input)?;
        result.prediction_digest_hex = result.compute_digest_hex()?;
        result.validate_against(descriptor, input)?;
        Ok(result)
    }

    pub fn validate_against(
        &self,
        descriptor: &HumanoidContextualCfcDescriptorV1,
        input: &HumanoidContextualPredictorInputV1,
    ) -> Result<(), HumanoidContextualCfcErrorV1> {
        self.validate_without_digest(descriptor, input)?;
        if self.prediction_digest_hex != self.compute_digest_hex()? {
            return Err(HumanoidContextualCfcErrorV1::PredictionDigestMismatch);
        }
        Ok(())
    }

    fn validate_without_digest(
        &self,
        descriptor: &HumanoidContextualCfcDescriptorV1,
        input: &HumanoidContextualPredictorInputV1,
    ) -> Result<(), HumanoidContextualCfcErrorV1> {
        descriptor.validate()?;
        input
            .validate()
            .map_err(HumanoidContextualCfcErrorV1::Context)?;
        if self.schema_id != Self::SCHEMA_ID
            || self.predictor_id != descriptor.predictor_id
            || self.predictor_descriptor_digest_hex != descriptor.descriptor_digest_hex()?
            || self.contextual_input_digest_hex != input.contextual_input_digest_hex
            || self.morphology != input.base_input.morphology
            || self.target_timestamp_seconds.to_bits()
                != input.base_input.target_timestamp_seconds.to_bits()
        {
            return Err(HumanoidContextualCfcErrorV1::InvalidPrediction);
        }

        let expected = observation_contract_set(
            &input.base_input.pre_state.policy_observations,
        )?;
        let output_digest = parse_hex_digest(&descriptor.output_role_digest_hex)?;
        let mut actual = BTreeSet::new();
        let mut previous: Option<[u8; 32]> = None;
        for prediction in &self.policy_predictions {
            validate_prediction_value(prediction)?;
            let digest = prediction
                .address
                .semantic_digest()
                .map_err(HumanoidContextualCfcErrorV1::Sensorimotor)?;
            if previous.is_some_and(|prior| prior >= digest) || !actual.insert(digest) {
                return Err(HumanoidContextualCfcErrorV1::ContractSetMismatch);
            }
            previous = Some(digest);

            let source = input
                .base_input
                .pre_state
                .policy_observations
                .iter()
                .find(|observation| {
                    observation
                        .address()
                        .semantic_digest()
                        .is_ok_and(|candidate| candidate == digest)
                })
                .ok_or(HumanoidContextualCfcErrorV1::ContractSetMismatch)?;

            match (source, prediction.prediction) {
                (SensorimotorObservationV1::Missing { .. },
                    HumanoidPredictedValueV1::Unavailable(
                        HumanoidPredictionUnavailableReasonV1::InputMissing,
                    )) => {}
                (SensorimotorObservationV1::Measured(_),
                    HumanoidPredictedValueV1::Predicted(_))
                    if digest == output_digest => {}
                (SensorimotorObservationV1::Measured(_),
                    HumanoidPredictedValueV1::Unavailable(
                        HumanoidPredictionUnavailableReasonV1::UnsupportedRole,
                    ))
                    if digest != output_digest => {}
                _ => return Err(HumanoidContextualCfcErrorV1::InvalidPrediction),
            }
        }
        if actual != expected {
            return Err(HumanoidContextualCfcErrorV1::ContractSetMismatch);
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> Result<String, HumanoidContextualCfcErrorV1> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(PREDICTION_DOMAIN_V1);
        feed_str(&mut hasher, &self.schema_id);
        feed_str(&mut hasher, &self.predictor_id);
        feed_str(&mut hasher, &self.predictor_descriptor_digest_hex);
        feed_str(&mut hasher, &self.contextual_input_digest_hex);
        feed_str(&mut hasher, self.morphology.schema_id());
        hasher.update(&self.target_timestamp_seconds.to_bits().to_le_bytes());
        hasher.update(&(self.policy_predictions.len() as u64).to_le_bytes());
        for prediction in &self.policy_predictions {
            let digest = prediction
                .address
                .semantic_digest()
                .map_err(HumanoidContextualCfcErrorV1::Sensorimotor)?;
            hasher.update(&digest);
            match prediction.prediction {
                HumanoidPredictedValueV1::Predicted(value) => {
                    hasher.update(&[1]);
                    hasher.update(&value.to_bits().to_le_bytes());
                }
                HumanoidPredictedValueV1::Unavailable(reason) => {
                    hasher.update(&[0]);
                    feed_str(&mut hasher, unavailable_reason_token(reason));
                }
            }
        }
        Ok(digest_hex(hasher.finalize().as_bytes()))
    }
}

struct PreparedRowV1 {
    features: Vec<f64>,
    current: f64,
    target: f64,
    dt: f64,
}

fn contextual_features_v1(
    input: &HumanoidContextualPredictorInputV1,
    config: &HumanoidContextualCfcTrainingConfigV1,
) -> Result<Vec<f64>, HumanoidContextualCfcErrorV1> {
    input
        .validate()
        .map_err(HumanoidContextualCfcErrorV1::Context)?;
    config.validate()?;

    let hdc = input
        .base_input
        .encode_pre_policy_hdc()
        .map_err(HumanoidContextualCfcErrorV1::Predictor)?
        .ok_or(HumanoidContextualCfcErrorV1::MissingStateHdc)?;
    if hdc.values.is_empty() {
        return Err(HumanoidContextualCfcErrorV1::MissingStateHdc);
    }

    let mut features = Vec::with_capacity(FEATURE_DIM);
    features.push(1.0);

    // V1 is intentionally explicit about depending on HDC coordinate order.
    // The feature extractor ID commits to this exact block-mean reduction.
    let chunk_size = (hdc.values.len() + HDC_FEATURES - 1) / HDC_FEATURES;
    for bucket in 0..HDC_FEATURES {
        let start = bucket * chunk_size;
        if start >= hdc.values.len() {
            features.push(0.0);
            continue;
        }
        let end = (start + chunk_size).min(hdc.values.len());
        let sum: f64 = hdc.values[start..end]
            .iter()
            .map(|value| f64::from(*value))
            .sum();
        features.push(sum / (end - start) as f64);
    }

    let action = &input.base_input.backend_applied_action.values;
    let mut action_sum = [0.0f64; ACTION_FEATURES];
    let mut action_count = [0usize; ACTION_FEATURES];
    for (index, value) in action.iter().enumerate() {
        let max_abs = value
            .address
            .value_contract
            .min
            .abs()
            .max(value.address.value_contract.max.abs())
            .max(1.0e-12);
        let normalized = (value.value / max_abs).clamp(-4.0, 4.0);
        let bucket =
            (index * ACTION_FEATURES / action.len().max(1)).min(ACTION_FEATURES - 1);
        action_sum[bucket] += normalized;
        action_count[bucket] += 1;
    }
    for bucket in 0..ACTION_FEATURES {
        features.push(if action_count[bucket] == 0 {
            0.0
        } else {
            action_sum[bucket] / action_count[bucket] as f64
        });
    }

    for force in input.context.external_force_world_n {
        features.push((force / config.force_feature_scale_n).clamp(-8.0, 8.0));
    }

    if features.len() != FEATURE_DIM || features.iter().any(|value| !value.is_finite()) {
        return Err(HumanoidContextualCfcErrorV1::InvalidFeatures);
    }
    Ok(features)
}

fn root_velocity_x_address(
    observations: &[SensorimotorObservationV1],
) -> Result<&SensorimotorAddressV1, HumanoidContextualCfcErrorV1> {
    observations
        .iter()
        .find(|observation| is_root_velocity_x(observation.address()))
        .map(|observation| observation.address())
        .ok_or(HumanoidContextualCfcErrorV1::OutputRoleMissing)
}

fn root_velocity_x_value(
    observations: &[SensorimotorObservationV1],
) -> Result<f64, HumanoidContextualCfcErrorV1> {
    observations
        .iter()
        .find_map(|observation| {
            if !is_root_velocity_x(observation.address()) {
                return None;
            }
            match observation {
                SensorimotorObservationV1::Measured(measurement) => Some(measurement.value),
                SensorimotorObservationV1::Missing { .. } => None,
            }
        })
        .ok_or(HumanoidContextualCfcErrorV1::OutputRoleMissing)
}

fn is_root_velocity_x(address: &SensorimotorAddressV1) -> bool {
    address.subject == SensorimotorSubjectV1::BodyRoot
        && address.quantity == SensorimotorQuantityV1::LinearVelocity
        && address.component == SensorimotorComponentV1::X
}

fn observation_contract_set(
    observations: &[SensorimotorObservationV1],
) -> Result<BTreeSet<[u8; 32]>, HumanoidContextualCfcErrorV1> {
    let mut set = BTreeSet::new();
    for observation in observations {
        observation
            .validate()
            .map_err(HumanoidContextualCfcErrorV1::Sensorimotor)?;
        let digest = observation
            .address()
            .semantic_digest()
            .map_err(HumanoidContextualCfcErrorV1::Sensorimotor)?;
        if !set.insert(digest) {
            return Err(HumanoidContextualCfcErrorV1::ContractSetMismatch);
        }
    }
    Ok(set)
}

fn normalize_physical(
    value: f64,
    address: &SensorimotorAddressV1,
) -> Result<f64, HumanoidContextualCfcErrorV1> {
    if !value.is_finite() {
        return Err(HumanoidContextualCfcErrorV1::NonFiniteTrainingValue);
    }
    address
        .validate()
        .map_err(HumanoidContextualCfcErrorV1::Sensorimotor)?;
    let contract = &address.value_contract;
    let span = contract.max - contract.min;
    if !span.is_finite() || span <= 0.0 {
        return Err(HumanoidContextualCfcErrorV1::InvalidValueContract);
    }
    let midpoint = (contract.max + contract.min) * 0.5;
    Ok((value - midpoint) / (span * 0.5))
}

fn denormalize_physical(
    value: f64,
    address: &SensorimotorAddressV1,
) -> Result<f64, HumanoidContextualCfcErrorV1> {
    if !value.is_finite() {
        return Err(HumanoidContextualCfcErrorV1::NonFinitePrediction);
    }
    address
        .validate()
        .map_err(HumanoidContextualCfcErrorV1::Sensorimotor)?;
    let contract = &address.value_contract;
    let span = contract.max - contract.min;
    if !span.is_finite() || span <= 0.0 {
        return Err(HumanoidContextualCfcErrorV1::InvalidValueContract);
    }
    let midpoint = (contract.max + contract.min) * 0.5;
    let physical = midpoint + value * (span * 0.5);
    if !physical.is_finite() {
        return Err(HumanoidContextualCfcErrorV1::NonFinitePrediction);
    }
    Ok(physical)
}

fn training_lineage_digest(
    cases: &[&HumanoidContextualCfcTrainingCaseV1],
    config: &HumanoidContextualCfcTrainingConfigV1,
) -> Result<String, HumanoidContextualCfcErrorV1> {
    config.validate()?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(TRAINING_LINEAGE_DOMAIN_V1);
    feed_str(&mut hasher, TRAINER_ID_V1);
    feed_str(&mut hasher, FEATURE_EXTRACTOR_ID_V1);
    feed_training_config(&mut hasher, config);
    hasher.update(&(cases.len() as u64).to_le_bytes());
    for case in cases {
        case.validate()?;
        feed_str(&mut hasher, &case.case_digest_hex);
    }
    Ok(digest_hex(hasher.finalize().as_bytes()))
}

fn checkpoint_digest(
    parameters: &RootVelocityXCfcParametersV1,
    config: &HumanoidContextualCfcTrainingConfigV1,
    training_lineage_digest_hex: &str,
) -> Result<String, HumanoidContextualCfcErrorV1> {
    config.validate()?;
    if !valid_hex_digest(training_lineage_digest_hex)
        || parameters.weights.len() != FEATURE_DIM
        || parameters.weights.iter().any(|weight| !weight.is_finite())
        || !parameters.log_tau.is_finite()
    {
        return Err(HumanoidContextualCfcErrorV1::NonFiniteCheckpoint);
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(CHECKPOINT_DOMAIN_V1);
    feed_str(&mut hasher, ARCHITECTURE_ID_V1);
    feed_str(&mut hasher, FEATURE_EXTRACTOR_ID_V1);
    feed_str(&mut hasher, training_lineage_digest_hex);
    feed_training_config(&mut hasher, config);
    hasher.update(&parameters.contract_digest);
    hasher.update(&parameters.log_tau.to_bits().to_le_bytes());
    hasher.update(&(parameters.weights.len() as u64).to_le_bytes());
    for weight in &parameters.weights {
        hasher.update(&weight.to_bits().to_le_bytes());
    }
    Ok(digest_hex(hasher.finalize().as_bytes()))
}

fn feed_training_config(
    hasher: &mut blake3::Hasher,
    config: &HumanoidContextualCfcTrainingConfigV1,
) {
    hasher.update(&config.epochs.to_le_bytes());
    hasher.update(&config.learning_rate.to_bits().to_le_bytes());
    hasher.update(&config.initial_tau_seconds.to_bits().to_le_bytes());
    hasher.update(&config.force_feature_scale_n.to_bits().to_le_bytes());
    hasher.update(&config.weight_clip.to_bits().to_le_bytes());
    hasher.update(&(HDC_FEATURES as u64).to_le_bytes());
    hasher.update(&(ACTION_FEATURES as u64).to_le_bytes());
    hasher.update(&(FORCE_FEATURES as u64).to_le_bytes());
}

fn validate_prediction_value(
    prediction: &HumanoidPredictedPhysicalValueV1,
) -> Result<(), HumanoidContextualCfcErrorV1> {
    prediction
        .address
        .validate()
        .map_err(HumanoidContextualCfcErrorV1::Sensorimotor)?;
    if matches!(
        prediction.prediction,
        HumanoidPredictedValueV1::Predicted(value) if !value.is_finite()
    ) {
        return Err(HumanoidContextualCfcErrorV1::NonFinitePrediction);
    }
    Ok(())
}

fn valid_hex_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn parse_hex_digest(value: &str) -> Result<[u8; 32], HumanoidContextualCfcErrorV1> {
    if !valid_hex_digest(value) {
        return Err(HumanoidContextualCfcErrorV1::InvalidDescriptor);
    }
    let mut bytes = [0u8; 32];
    for index in 0..32 {
        bytes[index] = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| HumanoidContextualCfcErrorV1::InvalidDescriptor)?;
    }
    Ok(bytes)
}

fn unavailable_reason_token(reason: HumanoidPredictionUnavailableReasonV1) -> &'static str {
    match reason {
        HumanoidPredictionUnavailableReasonV1::InputMissing => "input_missing",
        HumanoidPredictionUnavailableReasonV1::PredictorAbstained => "predictor_abstained",
        HumanoidPredictionUnavailableReasonV1::UnsupportedRole => "unsupported_role",
    }
}

fn feed_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn digest_hex(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[derive(Debug)]
pub enum HumanoidContextualCfcErrorV1 {
    InvalidTrainingConfig,
    EmptyTrainingSet,
    InvalidTrainingCaseDigest,
    TrainingCaseDigestMismatch,
    TrainingPairMismatch,
    UnsupportedActionMode,
    OutputRoleMissing,
    OutputRoleMismatch,
    ContractSetMismatch,
    MissingStateHdc,
    InvalidFeatures,
    InvalidValueContract,
    NonFiniteTrainingValue,
    NonFiniteCheckpoint,
    InvalidDescriptor,
    InvalidPrediction,
    NonFinitePrediction,
    PredictionDigestMismatch,
    Sensorimotor(&'static str),
    Predictor(crate::continuous_predictor::HumanoidContinuousPredictorErrorV1),
    Context(HumanoidPredictorContextErrorV1),
    Transition(HumanoidTransitionEvidenceErrorV1),
}

impl fmt::Display for HumanoidContextualCfcErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTrainingConfig => write!(f, "invalid contextual CfC training config"),
            Self::EmptyTrainingSet => write!(f, "contextual CfC training set is empty"),
            Self::InvalidTrainingCaseDigest => write!(f, "invalid contextual CfC training-case digest"),
            Self::TrainingCaseDigestMismatch => write!(f, "contextual CfC training-case digest mismatch"),
            Self::TrainingPairMismatch => write!(f, "training target does not match its R4.1 base input"),
            Self::UnsupportedActionMode => write!(f, "contextual CfC action mode is unsupported"),
            Self::OutputRoleMissing => write!(f, "root linear velocity X is not measured"),
            Self::OutputRoleMismatch => write!(f, "training output semantic contract changed"),
            Self::ContractSetMismatch => write!(f, "prediction contract set mismatch"),
            Self::MissingStateHdc => write!(f, "predictor input contains no measured HDC state"),
            Self::InvalidFeatures => write!(f, "contextual CfC features are invalid"),
            Self::InvalidValueContract => write!(f, "sensorimotor value contract has invalid span"),
            Self::NonFiniteTrainingValue => write!(f, "training target contains a non-finite value"),
            Self::NonFiniteCheckpoint => write!(f, "checkpoint contains invalid/non-finite values"),
            Self::InvalidDescriptor => write!(f, "invalid contextual CfC descriptor"),
            Self::InvalidPrediction => write!(f, "invalid contextual CfC prediction"),
            Self::NonFinitePrediction => write!(f, "contextual CfC prediction is non-finite"),
            Self::PredictionDigestMismatch => write!(f, "contextual CfC prediction digest mismatch"),
            Self::Sensorimotor(message) => write!(f, "sensorimotor schema: {message}"),
            Self::Predictor(error) => write!(f, "R4.1 predictor contract: {error}"),
            Self::Context(error) => write!(f, "R4.2 context contract: {error}"),
            Self::Transition(error) => write!(f, "transition evidence: {error}"),
        }
    }
}

impl std::error::Error for HumanoidContextualCfcErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::HumanoidPhysicsSimulator;

    fn zero_command() -> HumanoidCommand {
        HumanoidCommand {
            torques: vec![0.0; HumanoidMorphology::Dmc21.num_actuators()],
        }
    }

    fn case_with_episode(
        force_x: f64,
        dt: f64,
        episode: &str,
    ) -> HumanoidContextualCfcTrainingCaseV1 {
        let mut simulator = ContextInstrumentedSimpleHumanoidSimulator::new();
        simulator.reset_with_perturbation(0.0, 41);
        capture_contextual_cfc_training_case_v1(
            &mut simulator,
            &zero_command(),
            dt,
            [force_x, 0.0, 0.0],
            "r4-cfc-clock",
            episode,
        )
        .unwrap()
    }

    fn training_case(index: usize, force_x: f64, dt: f64) -> HumanoidContextualCfcTrainingCaseV1 {
        case_with_episode(force_x, dt, &format!("r4-cfc-train-{index}"))
    }

    fn predicted_root_velocity_x(prediction: &HumanoidContextualCfcPredictionV1) -> f64 {
        prediction
            .policy_predictions
            .iter()
            .find_map(|predicted| {
                if is_root_velocity_x(&predicted.address) {
                    match predicted.prediction {
                        HumanoidPredictedValueV1::Predicted(value) => Some(value),
                        HumanoidPredictedValueV1::Unavailable(_) => None,
                    }
                } else {
                    None
                }
            })
            .expect("root linear velocity X prediction")
    }

    fn training_set() -> Vec<HumanoidContextualCfcTrainingCaseV1> {
        let dts = [0.015, 0.025, 0.040];
        (-12..=12)
            .enumerate()
            .map(|(index, step)| {
                training_case(index, f64::from(step) * 10.0, dts[index % dts.len()])
            })
            .collect()
    }

    #[test]
    fn training_case_commitment_detects_same_base_target_swap_after_capture() {
        let left = case_with_episode(-50.0, 0.025, "r4-cfc-swap");
        let right = case_with_episode(50.0, 0.025, "r4-cfc-swap");
        assert_eq!(
            left.input.base_input.input_digest_hex,
            right.input.base_input.input_digest_hex
        );
        assert_ne!(
            left.input.contextual_input_digest_hex,
            right.input.contextual_input_digest_hex
        );
        let swapped = HumanoidContextualCfcTrainingCaseV1::from_bound_evidence(
            right.transition.clone(),
            left.input.clone(),
            left.case_digest_hex.clone(),
        );
        assert!(matches!(
            swapped,
            Err(HumanoidContextualCfcErrorV1::TrainingCaseDigestMismatch)
        ));
    }

    #[test]
    fn checkpoint_is_deterministic_and_training_order_invariant() {
        let cases = training_set();
        let config = HumanoidContextualCfcTrainingConfigV1::default();
        let forward = HumanoidRootVelocityXCfcPredictorV1::train(&cases, config).unwrap();
        let mut reversed = cases.clone();
        reversed.reverse();
        let reverse = HumanoidRootVelocityXCfcPredictorV1::train(&reversed, config).unwrap();
        assert_eq!(forward.descriptor(), reverse.descriptor());
        assert_eq!(
            forward.descriptor().descriptor_digest_hex().unwrap(),
            reverse.descriptor().descriptor_digest_hex().unwrap()
        );
        assert_eq!(forward.tau_seconds().to_bits(), reverse.tau_seconds().to_bits());
    }

    #[test]
    fn contextual_prediction_cannot_detach_from_context() {
        let cases = training_set();
        let model = HumanoidRootVelocityXCfcPredictorV1::train(
            &cases,
            HumanoidContextualCfcTrainingConfigV1::default(),
        )
        .unwrap();

        let nominal = case_with_episode(0.0, 0.025, "r4-cfc-context-bind");
        let pushed = case_with_episode(80.0, 0.025, "r4-cfc-context-bind");
        assert_eq!(
            nominal.input.base_input.input_digest_hex,
            pushed.input.base_input.input_digest_hex
        );
        assert_ne!(
            nominal.input.contextual_input_digest_hex,
            pushed.input.contextual_input_digest_hex
        );

        let nominal_prediction = model.predict(&nominal.input).unwrap();
        let pushed_prediction = model.predict(&pushed.input).unwrap();
        assert_ne!(
            nominal_prediction.prediction_digest_hex,
            pushed_prediction.prediction_digest_hex
        );
        nominal_prediction
            .validate_against(model.descriptor(), &nominal.input)
            .unwrap();
        assert!(nominal_prediction
            .validate_against(model.descriptor(), &pushed.input)
            .is_err());
    }

    #[test]
    fn unsupported_roles_are_explicit_abstentions() {
        let cases = training_set();
        let model = HumanoidRootVelocityXCfcPredictorV1::train(
            &cases,
            HumanoidContextualCfcTrainingConfigV1::default(),
        )
        .unwrap();
        let case = training_case(100, 40.0, 0.025);
        let prediction = model.predict(&case.input).unwrap();
        let predicted_count = prediction
            .policy_predictions
            .iter()
            .filter(|value| matches!(value.prediction, HumanoidPredictedValueV1::Predicted(_)))
            .count();
        let unsupported_count = prediction
            .policy_predictions
            .iter()
            .filter(|value| {
                matches!(
                    value.prediction,
                    HumanoidPredictedValueV1::Unavailable(
                        HumanoidPredictionUnavailableReasonV1::UnsupportedRole
                    )
                )
            })
            .count();
        assert_eq!(predicted_count, 1);
        assert!(unsupported_count > 0);
    }

    #[test]
    fn held_out_force_and_dt_response_beats_persistence_for_root_velocity_x() {
        let cases = training_set();
        let model = HumanoidRootVelocityXCfcPredictorV1::train(
            &cases,
            HumanoidContextualCfcTrainingConfigV1::default(),
        )
        .unwrap();

        let held_out = [
            (-87.5, 0.020),
            (-42.5, 0.030),
            (37.5, 0.050),
            (92.5, 0.035),
        ];
        let mut model_error = 0.0;
        let mut persistence_error = 0.0;
        for (index, (force, dt)) in held_out.into_iter().enumerate() {
            let case = training_case(200 + index, force, dt);
            let prediction = model.predict(&case.input).unwrap();
            let predicted = predicted_root_velocity_x(&prediction);
            let pre = root_velocity_x_value(
                &case.input.base_input.pre_state.policy_observations,
            )
            .unwrap();
            let actual = root_velocity_x_value(
                &case.transition.post_state.policy_observations,
            )
            .unwrap();
            model_error += (predicted - actual).abs();
            persistence_error += (pre - actual).abs();
        }

        assert!(persistence_error > 0.0);
        assert!(
            model_error < persistence_error,
            "contextual CfC MAE sum {model_error} must beat persistence {persistence_error}"
        );
    }
}
