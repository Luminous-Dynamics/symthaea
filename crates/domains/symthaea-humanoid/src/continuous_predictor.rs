// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! R4.1 leakage-safe continuous-time predictor contract.
//!
//! The predictor never receives `HumanoidTransitionEvidenceV1` directly because
//! that evidence object contains the post-state target. Instead, R4.1 derives a
//! separate content-addressed input containing only information available at the
//! causal prediction boundary:
//!
//! `pre semantic state + backend-applied physical action + applied dt + provenance`.
//!
//! HDC and physical numerics keep distinct jobs. The pre-state can be encoded in
//! the exact R3.1 HDC geometry, while the backend action remains ordered physical
//! SI values so sub-bin action magnitude is not discarded merely for uniformity.
//!
//! V1 inference is a pure one-step function (`&self`). Hidden mutable recurrent
//! episode state is intentionally excluded. If later LTC/CfC work needs latent
//! memory, that latent state should become explicit, replayable evidence rather
//! than making predictions depend invisibly on evaluation order.
//!
//! Nothing here grants control, safety, capability, or motor authority.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;
use std::fmt::Write as _;

use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::sensorimotor_contingencies::{
    MissingObservationReasonV1, SensorimotorAddressV1, SensorimotorHdcEncoderV1,
    SensorimotorObservationV1,
};

use crate::morphology::HumanoidMorphology;
use crate::semantic_action::{
    HumanoidActuationStageV1, SemanticHumanoidActuationErrorV1,
    SemanticHumanoidActuationFrameV1,
};
use crate::semantic_state::{SemanticHumanoidErrorV1, SemanticHumanoidFrameV1};
use crate::transition_evidence::{
    HumanoidTransitionEvidenceErrorV1, HumanoidTransitionEvidenceV1,
    HumanoidTransitionStateSourceV1,
};
use crate::types::ActuationMode;

const PREDICTOR_DESCRIPTOR_DOMAIN_V1: &[u8] =
    b"symthaea.humanoid.continuous-predictor-descriptor.v1\0";
const PREDICTOR_INPUT_DOMAIN_V1: &[u8] = b"symthaea.humanoid.predictor-input.v1\0";
const PREDICTED_STATE_DOMAIN_V1: &[u8] = b"symthaea.humanoid.predicted-physical-state.v1\0";

/// Model family recorded independently of one concrete checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanoidContinuousPredictorFamilyV1 {
    ClosedFormContinuousTime,
    LiquidTimeConstant,
    Custom(String),
}

/// Typed time contract. V1 deliberately has one valid choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanoidPredictorTimeSemanticsV1 {
    ExplicitBackendAppliedDeltaSeconds,
}

/// Typed causal-action contract. Requested intent is not a valid R4.1 input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanoidPredictorActionSemanticsV1 {
    BackendAppliedPhysicalScalars,
}

/// V1 forbids invisible recurrent episode state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanoidPredictorMemorySemanticsV1 {
    PureOneStep,
}

/// Provenance for the executable predictor implementation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanoidPredictorProvenanceV1 {
    /// Deterministic research implementation with no learned artifact.
    ResearchImplementation { implementation_id: String },
    /// Learned parameters plus the evidence lineage used to produce them.
    LearnedCheckpoint {
        architecture_id: String,
        checkpoint_digest_hex: String,
        training_lineage_digest_hex: String,
    },
}

/// Explicit inference contract for one continuous-time model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanoidContinuousPredictorDescriptorV1 {
    pub schema_id: String,
    pub predictor_id: String,
    pub family: HumanoidContinuousPredictorFamilyV1,
    pub provenance: HumanoidPredictorProvenanceV1,
    pub time_semantics: HumanoidPredictorTimeSemanticsV1,
    pub action_semantics: HumanoidPredictorActionSemanticsV1,
    pub memory_semantics: HumanoidPredictorMemorySemanticsV1,
    pub supported_action_modes: Vec<ActuationMode>,
    pub state_hdc_profile_id: String,
    pub output_representation_id: String,
}

impl HumanoidContinuousPredictorDescriptorV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.humanoid.continuous-predictor-descriptor.v1";
    pub const STATE_HDC_PROFILE_ID: &'static str =
        "symthaea.sensorimotor.hdc.thermometer-prefix-binary.v1";
    pub const OUTPUT_REPRESENTATION_ID: &'static str =
        "symthaea.humanoid.predicted-physical-state.v1";

    pub fn validate(&self) -> Result<(), HumanoidContinuousPredictorErrorV1> {
        if self.schema_id != Self::SCHEMA_ID
            || self.predictor_id.trim().is_empty()
            || self.time_semantics
                != HumanoidPredictorTimeSemanticsV1::ExplicitBackendAppliedDeltaSeconds
            || self.action_semantics
                != HumanoidPredictorActionSemanticsV1::BackendAppliedPhysicalScalars
            || self.memory_semantics != HumanoidPredictorMemorySemanticsV1::PureOneStep
            || self.state_hdc_profile_id != Self::STATE_HDC_PROFILE_ID
            || self.output_representation_id != Self::OUTPUT_REPRESENTATION_ID
            || self.supported_action_modes.is_empty()
        {
            return Err(HumanoidContinuousPredictorErrorV1::InvalidDescriptor);
        }

        if matches!(&self.family, HumanoidContinuousPredictorFamilyV1::Custom(value) if value.trim().is_empty()) {
            return Err(HumanoidContinuousPredictorErrorV1::InvalidDescriptor);
        }

        let mut seen = Vec::new();
        for mode in &self.supported_action_modes {
            if !matches!(
                mode,
                ActuationMode::TorqueNewtonMetres | ActuationMode::PositionTargetRadians
            ) || seen.contains(mode)
            {
                return Err(HumanoidContinuousPredictorErrorV1::InvalidDescriptor);
            }
            seen.push(*mode);
        }

        match &self.provenance {
            HumanoidPredictorProvenanceV1::ResearchImplementation { implementation_id } => {
                if implementation_id.trim().is_empty() {
                    return Err(HumanoidContinuousPredictorErrorV1::InvalidDescriptor);
                }
            }
            HumanoidPredictorProvenanceV1::LearnedCheckpoint {
                architecture_id,
                checkpoint_digest_hex,
                training_lineage_digest_hex,
            } => {
                if architecture_id.trim().is_empty()
                    || !valid_hex_digest(checkpoint_digest_hex)
                    || !valid_hex_digest(training_lineage_digest_hex)
                {
                    return Err(HumanoidContinuousPredictorErrorV1::InvalidDescriptor);
                }
            }
        }
        Ok(())
    }

    pub fn supports_action_mode(&self, mode: ActuationMode) -> bool {
        self.supported_action_modes.contains(&mode)
    }

    /// Content identity of the complete inference contract and model provenance.
    pub fn descriptor_digest_hex(&self) -> Result<String, HumanoidContinuousPredictorErrorV1> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(PREDICTOR_DESCRIPTOR_DOMAIN_V1);
        feed_str(&mut hasher, &self.schema_id);
        feed_str(&mut hasher, &self.predictor_id);
        feed_family(&mut hasher, &self.family);
        feed_provenance(&mut hasher, &self.provenance);
        feed_str(&mut hasher, time_semantics_token(self.time_semantics));
        feed_str(&mut hasher, action_semantics_token(self.action_semantics));
        feed_str(&mut hasher, memory_semantics_token(self.memory_semantics));
        hasher.update(&(self.supported_action_modes.len() as u64).to_le_bytes());
        for mode in &self.supported_action_modes {
            feed_str(&mut hasher, actuation_mode_token(*mode));
        }
        feed_str(&mut hasher, &self.state_hdc_profile_id);
        feed_str(&mut hasher, &self.output_representation_id);
        Ok(digest_hex(hasher.finalize().as_bytes()))
    }
}

/// Leakage-safe causal input to one predictor invocation.
///
/// There is deliberately no transition digest or post-state field. The R3.13
/// transition digest commits to the target and therefore must not be model input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidPredictorInputV1 {
    pub schema_id: String,
    pub morphology: HumanoidMorphology,
    pub clock_domain_id: String,
    pub episode_id: String,
    pub step_sequence: u64,
    pub pre_state_source: HumanoidTransitionStateSourceV1,
    pub pre_state: SemanticHumanoidFrameV1,
    pub backend_applied_action: SemanticHumanoidActuationFrameV1,
    pub applied_dt_seconds: f64,
    /// Target time is causal metadata from the backend action boundary, not the
    /// target state's physical values.
    pub target_timestamp_seconds: f64,
    pub state_hdc_profile_id: String,
    pub input_digest_hex: String,
}

impl HumanoidPredictorInputV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.humanoid.predictor-input.v1";

    /// Derive the leakage-safe model input from a fully validated transition.
    /// Validation may inspect the held-out transition, but only pre/action/time
    /// information is copied into the returned value.
    pub fn from_transition(
        transition: &HumanoidTransitionEvidenceV1,
    ) -> Result<Self, HumanoidContinuousPredictorErrorV1> {
        transition
            .validate()
            .map_err(HumanoidContinuousPredictorErrorV1::Transition)?;
        if !transition.r4_privileged_truth_ready() {
            return Err(HumanoidContinuousPredictorErrorV1::TransitionNotReady);
        }

        let backend_applied_action = transition
            .backend_action
            .to_physical_frame()
            .map_err(HumanoidContinuousPredictorErrorV1::Action)?;

        let mut input = Self {
            schema_id: Self::SCHEMA_ID.to_string(),
            morphology: transition.morphology,
            clock_domain_id: transition.clock_domain_id.clone(),
            episode_id: transition.episode_id.clone(),
            step_sequence: transition.step_sequence(),
            pre_state_source: transition.pre_state_source,
            pre_state: transition.pre_state.clone(),
            backend_applied_action,
            applied_dt_seconds: transition.applied_dt_seconds(),
            target_timestamp_seconds: transition.backend_action.post_step_timestamp_seconds,
            state_hdc_profile_id:
                HumanoidContinuousPredictorDescriptorV1::STATE_HDC_PROFILE_ID.to_string(),
            input_digest_hex: String::new(),
        };
        input.validate_without_digest()?;
        input.input_digest_hex = input.compute_digest_hex()?;
        input.validate()?;
        Ok(input)
    }

    pub fn validate(&self) -> Result<(), HumanoidContinuousPredictorErrorV1> {
        self.validate_without_digest()?;
        if self.input_digest_hex != self.compute_digest_hex()? {
            return Err(HumanoidContinuousPredictorErrorV1::InputDigestMismatch);
        }
        Ok(())
    }

    fn validate_without_digest(&self) -> Result<(), HumanoidContinuousPredictorErrorV1> {
        if self.schema_id != Self::SCHEMA_ID
            || self.clock_domain_id.trim().is_empty()
            || self.episode_id.trim().is_empty()
            || self.step_sequence == 0
            || self.state_hdc_profile_id
                != HumanoidContinuousPredictorDescriptorV1::STATE_HDC_PROFILE_ID
        {
            return Err(HumanoidContinuousPredictorErrorV1::InvalidInput);
        }
        if self.pre_state_source != HumanoidTransitionStateSourceV1::PrivilegedBackendTruth {
            return Err(HumanoidContinuousPredictorErrorV1::TransitionNotReady);
        }
        self.pre_state
            .validate()
            .map_err(HumanoidContinuousPredictorErrorV1::State)?;
        self.backend_applied_action
            .validate()
            .map_err(HumanoidContinuousPredictorErrorV1::Action)?;
        if self.pre_state.morphology != self.morphology
            || self.backend_applied_action.morphology != self.morphology
            || self.backend_applied_action.stage != HumanoidActuationStageV1::BackendAppliedPhysical
        {
            return Err(HumanoidContinuousPredictorErrorV1::InvalidInput);
        }
        if !self.applied_dt_seconds.is_finite()
            || self.applied_dt_seconds <= 0.0
            || !self.target_timestamp_seconds.is_finite()
            || self.target_timestamp_seconds <= self.pre_state.timestamp_seconds
        {
            return Err(HumanoidContinuousPredictorErrorV1::InvalidTime);
        }
        let measured_dt = self.target_timestamp_seconds - self.pre_state.timestamp_seconds;
        if measured_dt.to_bits() != self.applied_dt_seconds.to_bits() {
            return Err(HumanoidContinuousPredictorErrorV1::InvalidTime);
        }
        Ok(())
    }

    /// Exact R3.1 policy-state HDC view. Structured physical state remains the
    /// source of truth; the HDC vector is derived on demand to prevent duplicate
    /// representations from drifting inside the input record.
    pub fn encode_pre_policy_hdc(
        &self,
    ) -> Result<Option<ContinuousHV>, HumanoidContinuousPredictorErrorV1> {
        self.validate()?;
        SensorimotorHdcEncoderV1
            .encode_observations(&self.pre_state.policy_observations)
            .map_err(HumanoidContinuousPredictorErrorV1::Sensorimotor)
    }

    /// Ordered physical SI action values, preserving sub-HDC-bin magnitude.
    pub fn physical_action_values(&self) -> Result<Vec<f64>, HumanoidContinuousPredictorErrorV1> {
        self.validate()?;
        Ok(self
            .backend_applied_action
            .values
            .iter()
            .map(|value| value.value)
            .collect())
    }

    fn compute_digest_hex(&self) -> Result<String, HumanoidContinuousPredictorErrorV1> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(PREDICTOR_INPUT_DOMAIN_V1);
        feed_str(&mut hasher, &self.schema_id);
        feed_str(&mut hasher, self.morphology.schema_id());
        feed_str(&mut hasher, &self.clock_domain_id);
        feed_str(&mut hasher, &self.episode_id);
        hasher.update(&self.step_sequence.to_le_bytes());
        feed_str(&mut hasher, state_source_token(self.pre_state_source));
        feed_semantic_frame(&mut hasher, &self.pre_state)?;
        feed_physical_action(&mut hasher, &self.backend_applied_action)?;
        hasher.update(&self.applied_dt_seconds.to_bits().to_le_bytes());
        hasher.update(&self.target_timestamp_seconds.to_bits().to_le_bytes());
        feed_str(&mut hasher, &self.state_hdc_profile_id);
        Ok(digest_hex(hasher.finalize().as_bytes()))
    }
}

/// Why a predictor did not produce one physical scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanoidPredictionUnavailableReasonV1 {
    InputMissing,
    PredictorAbstained,
    UnsupportedRole,
}

/// Predicted value for one exact sensorimotor contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanoidPredictedValueV1 {
    Predicted(f64),
    Unavailable(HumanoidPredictionUnavailableReasonV1),
}

/// Prediction at one physical semantic address. This is not an observation and
/// therefore deliberately does not reuse `SensorimotorMeasurementV1`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidPredictedPhysicalValueV1 {
    pub address: SensorimotorAddressV1,
    pub prediction: HumanoidPredictedValueV1,
}

impl HumanoidPredictedPhysicalValueV1 {
    fn validate(&self) -> Result<(), HumanoidContinuousPredictorErrorV1> {
        self.address
            .validate()
            .map_err(HumanoidContinuousPredictorErrorV1::Sensorimotor)?;
        if matches!(self.prediction, HumanoidPredictedValueV1::Predicted(value) if !value.is_finite()) {
            return Err(HumanoidContinuousPredictorErrorV1::NonFinitePrediction);
        }
        Ok(())
    }
}

/// Physical prediction produced by one model invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidPredictedStateV1 {
    pub schema_id: String,
    pub predictor_id: String,
    pub predictor_descriptor_digest_hex: String,
    pub input_digest_hex: String,
    pub morphology: HumanoidMorphology,
    pub target_timestamp_seconds: f64,
    /// Canonically ordered by exact contract digest.
    pub policy_predictions: Vec<HumanoidPredictedPhysicalValueV1>,
    /// Optional privileged world-root prediction, intentionally outside policy HDC.
    pub privileged_root_position_world_m: [Option<f64>; 3],
    pub prediction_digest_hex: String,
}

impl HumanoidPredictedStateV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.humanoid.predicted-physical-state.v1";

    pub fn new(
        descriptor: &HumanoidContinuousPredictorDescriptorV1,
        input: &HumanoidPredictorInputV1,
        policy_predictions: Vec<HumanoidPredictedPhysicalValueV1>,
        privileged_root_position_world_m: [Option<f64>; 3],
    ) -> Result<Self, HumanoidContinuousPredictorErrorV1> {
        descriptor.validate()?;
        input.validate()?;
        if !descriptor.supports_action_mode(input.backend_applied_action.physical_mode) {
            return Err(HumanoidContinuousPredictorErrorV1::UnsupportedActionMode);
        }

        let mut keyed = Vec::with_capacity(policy_predictions.len());
        for prediction in policy_predictions {
            prediction.validate()?;
            let digest = prediction
                .address
                .semantic_digest()
                .map_err(HumanoidContinuousPredictorErrorV1::Sensorimotor)?;
            keyed.push((digest, prediction));
        }
        keyed.sort_by_key(|(digest, _)| *digest);
        let policy_predictions = keyed
            .into_iter()
            .map(|(_, prediction)| prediction)
            .collect();

        let mut predicted = Self {
            schema_id: Self::SCHEMA_ID.to_string(),
            predictor_id: descriptor.predictor_id.clone(),
            predictor_descriptor_digest_hex: descriptor.descriptor_digest_hex()?,
            input_digest_hex: input.input_digest_hex.clone(),
            morphology: input.morphology,
            target_timestamp_seconds: input.target_timestamp_seconds,
            policy_predictions,
            privileged_root_position_world_m,
            prediction_digest_hex: String::new(),
        };
        predicted.validate_without_digest(input, descriptor)?;
        predicted.prediction_digest_hex = predicted.compute_digest_hex()?;
        predicted.validate_against(input, descriptor)?;
        Ok(predicted)
    }

    pub fn validate_against(
        &self,
        input: &HumanoidPredictorInputV1,
        descriptor: &HumanoidContinuousPredictorDescriptorV1,
    ) -> Result<(), HumanoidContinuousPredictorErrorV1> {
        descriptor.validate()?;
        self.validate_without_digest(input, descriptor)?;
        if self.prediction_digest_hex != self.compute_digest_hex()? {
            return Err(HumanoidContinuousPredictorErrorV1::PredictionDigestMismatch);
        }
        Ok(())
    }

    fn validate_without_digest(
        &self,
        input: &HumanoidPredictorInputV1,
        descriptor: &HumanoidContinuousPredictorDescriptorV1,
    ) -> Result<(), HumanoidContinuousPredictorErrorV1> {
        input.validate()?;
        if self.schema_id != Self::SCHEMA_ID
            || self.predictor_id != descriptor.predictor_id
            || self.predictor_descriptor_digest_hex != descriptor.descriptor_digest_hex()?
            || self.input_digest_hex != input.input_digest_hex
            || self.morphology != input.morphology
            || self.target_timestamp_seconds.to_bits() != input.target_timestamp_seconds.to_bits()
        {
            return Err(HumanoidContinuousPredictorErrorV1::InvalidPrediction);
        }
        if !descriptor.supports_action_mode(input.backend_applied_action.physical_mode) {
            return Err(HumanoidContinuousPredictorErrorV1::UnsupportedActionMode);
        }
        if self
            .privileged_root_position_world_m
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(HumanoidContinuousPredictorErrorV1::NonFinitePrediction);
        }

        let expected_contracts = contract_set_from_observations(&input.pre_state.policy_observations)?;
        let mut actual_contracts = BTreeSet::new();
        let mut previous: Option<[u8; 32]> = None;
        for prediction in &self.policy_predictions {
            prediction.validate()?;
            let digest = prediction
                .address
                .semantic_digest()
                .map_err(HumanoidContinuousPredictorErrorV1::Sensorimotor)?;
            if previous.is_some_and(|prior| prior >= digest) || !actual_contracts.insert(digest) {
                return Err(HumanoidContinuousPredictorErrorV1::PredictionContractMismatch);
            }
            previous = Some(digest);
        }
        if actual_contracts != expected_contracts {
            return Err(HumanoidContinuousPredictorErrorV1::PredictionContractMismatch);
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> Result<String, HumanoidContinuousPredictorErrorV1> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(PREDICTED_STATE_DOMAIN_V1);
        feed_str(&mut hasher, &self.schema_id);
        feed_str(&mut hasher, &self.predictor_id);
        feed_str(&mut hasher, &self.predictor_descriptor_digest_hex);
        feed_str(&mut hasher, &self.input_digest_hex);
        feed_str(&mut hasher, self.morphology.schema_id());
        hasher.update(&self.target_timestamp_seconds.to_bits().to_le_bytes());
        hasher.update(&(self.policy_predictions.len() as u64).to_le_bytes());
        for prediction in &self.policy_predictions {
            prediction.validate()?;
            let digest = prediction
                .address
                .semantic_digest()
                .map_err(HumanoidContinuousPredictorErrorV1::Sensorimotor)?;
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
        for value in self.privileged_root_position_world_m {
            feed_option_f64(&mut hasher, value);
        }
        Ok(digest_hex(hasher.finalize().as_bytes()))
    }
}

/// Pure one-step continuous-time predictor. `&self` is intentional: v1 does not
/// permit hidden mutable recurrent episode state.
pub trait HumanoidContinuousTimePredictorV1 {
    fn descriptor(&self) -> &HumanoidContinuousPredictorDescriptorV1;

    fn predict(
        &self,
        input: &HumanoidPredictorInputV1,
    ) -> Result<HumanoidPredictedStateV1, HumanoidContinuousPredictorErrorV1>;
}

/// Execute one predictor invocation behind the R4.1 contract gate.
pub fn predict_validated_v1(
    predictor: &dyn HumanoidContinuousTimePredictorV1,
    input: &HumanoidPredictorInputV1,
) -> Result<HumanoidPredictedStateV1, HumanoidContinuousPredictorErrorV1> {
    input.validate()?;
    let descriptor = predictor.descriptor();
    descriptor.validate()?;
    if !descriptor.supports_action_mode(input.backend_applied_action.physical_mode) {
        return Err(HumanoidContinuousPredictorErrorV1::UnsupportedActionMode);
    }
    let prediction = predictor.predict(input)?;
    prediction.validate_against(input, descriptor)?;
    Ok(prediction)
}

fn contract_set_from_observations(
    observations: &[SensorimotorObservationV1],
) -> Result<BTreeSet<[u8; 32]>, HumanoidContinuousPredictorErrorV1> {
    let mut contracts = BTreeSet::new();
    for observation in observations {
        observation
            .validate()
            .map_err(HumanoidContinuousPredictorErrorV1::Sensorimotor)?;
        let digest = observation
            .address()
            .semantic_digest()
            .map_err(HumanoidContinuousPredictorErrorV1::Sensorimotor)?;
        if !contracts.insert(digest) {
            return Err(HumanoidContinuousPredictorErrorV1::PredictionContractMismatch);
        }
    }
    Ok(contracts)
}

fn feed_semantic_frame(
    hasher: &mut blake3::Hasher,
    frame: &SemanticHumanoidFrameV1,
) -> Result<(), HumanoidContinuousPredictorErrorV1> {
    frame
        .validate()
        .map_err(HumanoidContinuousPredictorErrorV1::State)?;
    feed_str(hasher, &frame.schema_id);
    feed_str(hasher, frame.morphology.schema_id());
    feed_str(hasher, &frame.source_observation_schema_id);
    hasher.update(&frame.timestamp_seconds.to_bits().to_le_bytes());
    hasher.update(&(frame.policy_observations.len() as u64).to_le_bytes());
    for observation in &frame.policy_observations {
        let digest = observation
            .address()
            .semantic_digest()
            .map_err(HumanoidContinuousPredictorErrorV1::Sensorimotor)?;
        hasher.update(&digest);
        match observation {
            SensorimotorObservationV1::Measured(measurement) => {
                hasher.update(&[1]);
                hasher.update(&measurement.value.to_bits().to_le_bytes());
            }
            SensorimotorObservationV1::Missing { reason, .. } => {
                hasher.update(&[0]);
                feed_str(hasher, missing_reason_token(reason));
            }
        }
    }
    for value in frame.privileged_root_position_world_m {
        feed_option_f64(hasher, value);
    }
    Ok(())
}

fn feed_physical_action(
    hasher: &mut blake3::Hasher,
    action: &SemanticHumanoidActuationFrameV1,
) -> Result<(), HumanoidContinuousPredictorErrorV1> {
    action
        .validate()
        .map_err(HumanoidContinuousPredictorErrorV1::Action)?;
    feed_str(hasher, &action.schema_id);
    feed_str(hasher, action.morphology.schema_id());
    feed_str(hasher, actuation_stage_token(action.stage));
    feed_str(hasher, actuation_mode_token(action.source_mode));
    feed_str(hasher, actuation_mode_token(action.physical_mode));
    match action.evidence_id.as_deref() {
        Some(value) => {
            hasher.update(&[1]);
            feed_str(hasher, value);
        }
        None => hasher.update(&[0]),
    }
    hasher.update(&(action.clipped_joints as u64).to_le_bytes());
    hasher.update(&(action.values.len() as u64).to_le_bytes());
    for value in &action.values {
        let digest = value
            .address
            .semantic_digest()
            .map_err(HumanoidContinuousPredictorErrorV1::Sensorimotor)?;
        hasher.update(&digest);
        hasher.update(&value.value.to_bits().to_le_bytes());
    }
    Ok(())
}

fn feed_family(hasher: &mut blake3::Hasher, family: &HumanoidContinuousPredictorFamilyV1) {
    match family {
        HumanoidContinuousPredictorFamilyV1::ClosedFormContinuousTime => {
            feed_str(hasher, "closed_form_continuous_time")
        }
        HumanoidContinuousPredictorFamilyV1::LiquidTimeConstant => {
            feed_str(hasher, "liquid_time_constant")
        }
        HumanoidContinuousPredictorFamilyV1::Custom(value) => {
            feed_str(hasher, "custom");
            feed_str(hasher, value);
        }
    }
}

fn feed_provenance(hasher: &mut blake3::Hasher, provenance: &HumanoidPredictorProvenanceV1) {
    match provenance {
        HumanoidPredictorProvenanceV1::ResearchImplementation { implementation_id } => {
            feed_str(hasher, "research_implementation");
            feed_str(hasher, implementation_id);
        }
        HumanoidPredictorProvenanceV1::LearnedCheckpoint {
            architecture_id,
            checkpoint_digest_hex,
            training_lineage_digest_hex,
        } => {
            feed_str(hasher, "learned_checkpoint");
            feed_str(hasher, architecture_id);
            feed_str(hasher, checkpoint_digest_hex);
            feed_str(hasher, training_lineage_digest_hex);
        }
    }
}

fn state_source_token(source: HumanoidTransitionStateSourceV1) -> &'static str {
    match source {
        HumanoidTransitionStateSourceV1::PrivilegedBackendTruth => "privileged_backend_truth",
        HumanoidTransitionStateSourceV1::PolicyObservation => "policy_observation",
    }
}

fn missing_reason_token(reason: &MissingObservationReasonV1) -> &'static str {
    match reason {
        MissingObservationReasonV1::NotPresent => "not_present",
        MissingObservationReasonV1::NotObserved => "not_observed",
        MissingObservationReasonV1::Stale => "stale",
        MissingObservationReasonV1::Invalid => "invalid",
        MissingObservationReasonV1::Disabled => "disabled",
        MissingObservationReasonV1::Unknown => "unknown",
    }
}

fn unavailable_reason_token(reason: HumanoidPredictionUnavailableReasonV1) -> &'static str {
    match reason {
        HumanoidPredictionUnavailableReasonV1::InputMissing => "input_missing",
        HumanoidPredictionUnavailableReasonV1::PredictorAbstained => "predictor_abstained",
        HumanoidPredictionUnavailableReasonV1::UnsupportedRole => "unsupported_role",
    }
}

fn time_semantics_token(semantics: HumanoidPredictorTimeSemanticsV1) -> &'static str {
    match semantics {
        HumanoidPredictorTimeSemanticsV1::ExplicitBackendAppliedDeltaSeconds => {
            "explicit_backend_applied_delta_seconds"
        }
    }
}

fn action_semantics_token(semantics: HumanoidPredictorActionSemanticsV1) -> &'static str {
    match semantics {
        HumanoidPredictorActionSemanticsV1::BackendAppliedPhysicalScalars => {
            "backend_applied_physical_scalars"
        }
    }
}

fn memory_semantics_token(semantics: HumanoidPredictorMemorySemanticsV1) -> &'static str {
    match semantics {
        HumanoidPredictorMemorySemanticsV1::PureOneStep => "pure_one_step",
    }
}

fn actuation_stage_token(stage: HumanoidActuationStageV1) -> &'static str {
    match stage {
        HumanoidActuationStageV1::RequestedPhysicalProjection => "requested_physical_projection",
        HumanoidActuationStageV1::SafetyProjectedPhysical => "safety_projected_physical",
        HumanoidActuationStageV1::BackendAppliedPhysical => "backend_applied_physical",
        HumanoidActuationStageV1::MeasuredExecution => "measured_execution",
    }
}

fn actuation_mode_token(mode: ActuationMode) -> &'static str {
    match mode {
        ActuationMode::NormalizedTorque => "normalized_torque",
        ActuationMode::TorqueNewtonMetres => "torque_newton_metres",
        ActuationMode::PositionTargetRadians => "position_target_radians",
        ActuationMode::NormalizedPosition => "normalized_position",
    }
}

fn valid_hex_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn feed_option_f64(hasher: &mut blake3::Hasher, value: Option<f64>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_bits().to_le_bytes());
        }
        None => hasher.update(&[0]),
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
pub enum HumanoidContinuousPredictorErrorV1 {
    Transition(HumanoidTransitionEvidenceErrorV1),
    TransitionNotReady,
    State(SemanticHumanoidErrorV1),
    Action(SemanticHumanoidActuationErrorV1),
    Sensorimotor(&'static str),
    InvalidDescriptor,
    InvalidInput,
    InvalidTime,
    InputDigestMismatch,
    InvalidPrediction,
    NonFinitePrediction,
    PredictionContractMismatch,
    PredictionDigestMismatch,
    UnsupportedActionMode,
}

impl fmt::Display for HumanoidContinuousPredictorErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transition(error) => write!(f, "invalid transition evidence: {error}"),
            Self::TransitionNotReady => {
                write!(f, "transition is not ready for R4 privileged-truth prediction")
            }
            Self::State(error) => write!(f, "invalid semantic state: {error}"),
            Self::Action(error) => write!(f, "invalid semantic action: {error}"),
            Self::Sensorimotor(message) => write!(f, "sensorimotor schema: {message}"),
            Self::InvalidDescriptor => write!(f, "invalid continuous predictor descriptor"),
            Self::InvalidInput => write!(f, "invalid leakage-safe predictor input"),
            Self::InvalidTime => write!(f, "invalid predictor time boundary"),
            Self::InputDigestMismatch => write!(f, "predictor input content digest mismatch"),
            Self::InvalidPrediction => write!(f, "invalid predicted state"),
            Self::NonFinitePrediction => write!(f, "prediction contains a non-finite value"),
            Self::PredictionContractMismatch => {
                write!(f, "prediction exact-contract set does not match predictor input")
            }
            Self::PredictionDigestMismatch => write!(f, "predicted-state content digest mismatch"),
            Self::UnsupportedActionMode => {
                write!(f, "predictor does not support backend physical action mode")
            }
        }
    }
}

impl std::error::Error for HumanoidContinuousPredictorErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend_action_evidence::InstrumentedSimpleHumanoidSimulator;
    use crate::simulator::HumanoidPhysicsSimulator;
    use crate::transition_evidence::capture_privileged_transition_v1;
    use crate::types::HumanoidCommand;

    fn transition_with_force(gain: f32, force: [f64; 3]) -> HumanoidTransitionEvidenceV1 {
        let morphology = HumanoidMorphology::Dmc21;
        let mut simulator = InstrumentedSimpleHumanoidSimulator::new_for(morphology);
        simulator.reset_with_perturbation(0.0, 77);
        simulator.apply_external_force(force);
        let mut torques = vec![0.0f32; morphology.num_actuators()];
        torques[0] = gain;
        torques[5] = -0.5 * gain;
        capture_privileged_transition_v1(
            &mut simulator,
            &HumanoidCommand { torques },
            0.025,
            "r4-contract-clock",
            "r4-contract-episode",
        )
        .unwrap()
    }

    fn transition(gain: f32) -> HumanoidTransitionEvidenceV1 {
        transition_with_force(gain, [0.0; 3])
    }

    fn descriptor() -> HumanoidContinuousPredictorDescriptorV1 {
        HumanoidContinuousPredictorDescriptorV1 {
            schema_id: HumanoidContinuousPredictorDescriptorV1::SCHEMA_ID.to_string(),
            predictor_id: "test.closed-form.v1".to_string(),
            family: HumanoidContinuousPredictorFamilyV1::ClosedFormContinuousTime,
            provenance: HumanoidPredictorProvenanceV1::ResearchImplementation {
                implementation_id: "test.closed-form.impl.v1".to_string(),
            },
            time_semantics:
                HumanoidPredictorTimeSemanticsV1::ExplicitBackendAppliedDeltaSeconds,
            action_semantics: HumanoidPredictorActionSemanticsV1::BackendAppliedPhysicalScalars,
            memory_semantics: HumanoidPredictorMemorySemanticsV1::PureOneStep,
            supported_action_modes: vec![ActuationMode::TorqueNewtonMetres],
            state_hdc_profile_id:
                HumanoidContinuousPredictorDescriptorV1::STATE_HDC_PROFILE_ID.to_string(),
            output_representation_id:
                HumanoidContinuousPredictorDescriptorV1::OUTPUT_REPRESENTATION_ID.to_string(),
        }
    }

    struct TestPredictor {
        descriptor: HumanoidContinuousPredictorDescriptorV1,
    }

    impl HumanoidContinuousTimePredictorV1 for TestPredictor {
        fn descriptor(&self) -> &HumanoidContinuousPredictorDescriptorV1 {
            &self.descriptor
        }

        fn predict(
            &self,
            input: &HumanoidPredictorInputV1,
        ) -> Result<HumanoidPredictedStateV1, HumanoidContinuousPredictorErrorV1> {
            // Contract test only: copy the pre physical state. R4.0 already owns
            // the persistence baseline; this adapter merely exercises R4.1 type gates.
            let policy_predictions = input
                .pre_state
                .policy_observations
                .iter()
                .map(|observation| HumanoidPredictedPhysicalValueV1 {
                    address: observation.address().clone(),
                    prediction: match observation {
                        SensorimotorObservationV1::Measured(measurement) => {
                            HumanoidPredictedValueV1::Predicted(measurement.value)
                        }
                        SensorimotorObservationV1::Missing { .. } => {
                            HumanoidPredictedValueV1::Unavailable(
                                HumanoidPredictionUnavailableReasonV1::InputMissing,
                            )
                        }
                    },
                })
                .collect();
            HumanoidPredictedStateV1::new(
                &self.descriptor,
                input,
                policy_predictions,
                input.pre_state.privileged_root_position_world_m,
            )
        }
    }

    #[test]
    fn transition_derives_leakage_safe_input_with_hdc_and_physical_action() {
        let transition = transition(0.7);
        let input = HumanoidPredictorInputV1::from_transition(&transition).unwrap();
        input.validate().unwrap();

        assert_eq!(input.pre_state.timestamp_seconds.to_bits(), 0.0f64.to_bits());
        assert_eq!(
            input.target_timestamp_seconds.to_bits(),
            transition.post_state.timestamp_seconds.to_bits()
        );
        assert_eq!(
            input.backend_applied_action.stage,
            HumanoidActuationStageV1::BackendAppliedPhysical
        );
        assert_eq!(
            input.backend_applied_action.physical_mode,
            ActuationMode::TorqueNewtonMetres
        );
        assert_eq!(
            input.physical_action_values().unwrap().len(),
            HumanoidMorphology::Dmc21.num_actuators()
        );
        assert!(input.encode_pre_policy_hdc().unwrap().is_some());
    }

    #[test]
    fn same_pre_state_but_different_action_changes_input_commitment() {
        let quiet = HumanoidPredictorInputV1::from_transition(&transition(0.1)).unwrap();
        let strong = HumanoidPredictorInputV1::from_transition(&transition(0.9)).unwrap();
        assert_eq!(quiet.pre_state, strong.pre_state);
        assert_ne!(quiet.input_digest_hex, strong.input_digest_hex);
    }

    #[test]
    fn different_held_out_outcome_does_not_leak_into_predictor_input() {
        let nominal_transition = transition_with_force(0.6, [0.0; 3]);
        let disturbed_transition = transition_with_force(0.6, [80.0, 0.0, 0.0]);
        assert_eq!(nominal_transition.pre_state, disturbed_transition.pre_state);
        assert_eq!(
            nominal_transition.backend_action.evidence_digest_hex,
            disturbed_transition.backend_action.evidence_digest_hex
        );
        assert_ne!(
            nominal_transition.transition_digest_hex,
            disturbed_transition.transition_digest_hex
        );
        assert_ne!(nominal_transition.post_state, disturbed_transition.post_state);

        let nominal = HumanoidPredictorInputV1::from_transition(&nominal_transition).unwrap();
        let disturbed = HumanoidPredictorInputV1::from_transition(&disturbed_transition).unwrap();
        assert_eq!(nominal.input_digest_hex, disturbed.input_digest_hex);
    }

    #[test]
    fn validated_predictor_gate_binds_full_descriptor_provenance() {
        let input = HumanoidPredictorInputV1::from_transition(&transition(0.6)).unwrap();
        let predictor = TestPredictor {
            descriptor: descriptor(),
        };
        let prediction = predict_validated_v1(&predictor, &input).unwrap();
        prediction
            .validate_against(&input, predictor.descriptor())
            .unwrap();
        assert_eq!(prediction.predictor_id, "test.closed-form.v1");
        assert_eq!(prediction.input_digest_hex, input.input_digest_hex);
        assert_eq!(
            prediction.predictor_descriptor_digest_hex,
            predictor.descriptor().descriptor_digest_hex().unwrap()
        );
    }

    #[test]
    fn predicted_contract_set_must_match_input_exactly() {
        let input = HumanoidPredictorInputV1::from_transition(&transition(0.6)).unwrap();
        let predictor = TestPredictor {
            descriptor: descriptor(),
        };
        let mut prediction = predictor.predict(&input).unwrap();
        prediction.policy_predictions.pop();
        assert!(prediction
            .validate_against(&input, predictor.descriptor())
            .is_err());
    }

    #[test]
    fn learned_checkpoint_changes_descriptor_identity_and_requires_lineage() {
        let research = descriptor();
        let research_digest = research.descriptor_digest_hex().unwrap();
        let mut learned = research.clone();
        learned.provenance = HumanoidPredictorProvenanceV1::LearnedCheckpoint {
            architecture_id: "cfc-vector-v1".to_string(),
            checkpoint_digest_hex: "00".repeat(32),
            training_lineage_digest_hex: "ab".repeat(32),
        };
        learned.validate().unwrap();
        assert_ne!(research_digest, learned.descriptor_digest_hex().unwrap());
        if let HumanoidPredictorProvenanceV1::LearnedCheckpoint {
            checkpoint_digest_hex,
            ..
        } = &mut learned.provenance
        {
            checkpoint_digest_hex.pop();
        }
        assert!(learned.validate().is_err());
    }

    #[test]
    fn input_digest_detects_pre_state_or_action_tampering() {
        let mut state_tamper = HumanoidPredictorInputV1::from_transition(&transition(0.5)).unwrap();
        let measured = state_tamper
            .pre_state
            .policy_observations
            .iter_mut()
            .find_map(|observation| match observation {
                SensorimotorObservationV1::Measured(measurement) => Some(measurement),
                SensorimotorObservationV1::Missing { .. } => None,
            })
            .unwrap();
        measured.value += 0.001;
        assert!(state_tamper.validate().is_err());

        let mut action_tamper = HumanoidPredictorInputV1::from_transition(&transition(0.5)).unwrap();
        action_tamper.backend_applied_action.values[0].value += 0.001;
        assert!(action_tamper.validate().is_err());
    }
}
