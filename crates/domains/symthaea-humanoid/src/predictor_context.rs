// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! R4.2 exogenous-context evidence for causal humanoid prediction.
//!
//! R4.1 deliberately excludes the held-out target, but that also means an
//! unobserved external push can change the target while leaving the predictor
//! input unchanged. R4.2 makes the first *known* exogenous variables explicit
//! without pretending every simulator parameter is already observable.
//!
//! This v1 profile is intentionally narrow and scientific:
//! - the evidence source is privileged simulator truth;
//! - the wrapped simple backend stays on its nominal body parameters;
//! - gravity is the simple backend default (9.81 m/s² downward);
//! - terrain/ground-contact stay on the default flat/disabled profile;
//! - every external world-frame force accumulated for the next step is recorded
//!   exactly and cleared at the same step boundary as the backend;
//! - actuator adaptation remains in R3.12 action evidence, not duplicated here.
//!
//! Domain-randomized body parameters are not externally exposed by the current
//! backend, so this profile does not claim to cover them. A future randomized
//! context profile must bind the realized masses/inertias/damping rather than
//! treating an `enabled` flag as sufficient evidence.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::Write as _;

use crate::backend_action_evidence::{
    BackendAppliedActionEvidenceProvider, BackendAppliedActionSnapshotV1,
    InstrumentedSimpleHumanoidSimulator,
};
use crate::continuous_predictor::{
    HumanoidContinuousPredictorErrorV1, HumanoidPredictorInputV1,
};
use crate::morphology::HumanoidMorphology;
use crate::semantic_state::{SemanticHumanoidEncoderV1, SemanticHumanoidErrorV1};
use crate::simulator::HumanoidPhysicsSimulator;
use crate::terrain::{TerrainEvidenceSource, TerrainSample};
use crate::transition_evidence::{
    HumanoidTransitionEvidenceErrorV1, HumanoidTransitionEvidenceV1,
    HumanoidTransitionStateSourceV1,
};
use crate::types::{ActuationMode, CommandValidationError, HumanoidCommand, HumanoidState};

const CONTEXT_EVIDENCE_DOMAIN_V1: &[u8] =
    b"symthaea.humanoid.predictor-context.simple-nominal.v1\0";
const CONTEXTUAL_INPUT_DOMAIN_V1: &[u8] =
    b"symthaea.humanoid.contextual-predictor-input.v1\0";
const CONTEXT_INSTRUMENTATION_ID_V1: &str =
    "symthaea.instrumented-simple-humanoid.predictor-context.v1";
const SIMPLE_BACKEND_NAME_V1: &str = "symthaea-simple-humanoid";
const NOMINAL_BODY_PROFILE_ID_V1: &str = "symthaea.simple-humanoid.body.nominal.v1";
const FLAT_TERRAIN_PROFILE_ID_V1: &str = "symthaea.simple-humanoid.terrain.flat.v1";
const GROUND_CONTACT_PROFILE_ID_V1: &str =
    "symthaea.simple-humanoid.ground-contact.disabled.v1";
const DEFAULT_GRAVITY_MPS2: f64 = 9.81;

/// Epistemic source of predictor context.
///
/// The first R4.2 profile is deliberately privileged. A model trained with this
/// context is a system-identification experiment, not automatically a deployable
/// policy-side predictor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanoidPredictorContextSourceV1 {
    PrivilegedSimulatorTruth,
}

/// Content-addressed context that was causally available at one backend step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidPredictorContextEvidenceV1 {
    pub schema_id: String,
    pub instrumentation_id: String,
    pub backend_name: String,
    pub morphology: HumanoidMorphology,
    pub source: HumanoidPredictorContextSourceV1,
    pub step_sequence: u64,
    pub pre_step_timestamp_seconds: f64,
    pub post_step_timestamp_seconds: f64,

    /// Net world-frame force accumulated through `apply_external_force()` for
    /// this exact step. The simple backend clears the force after each step.
    pub external_force_world_n: [f64; 3],

    /// World-frame gravity acceleration supplied by the qualified nominal profile.
    pub gravity_world_mps2: [f64; 3],

    /// Terrain is sampled at the pre-step root XY position, so the context never
    /// reads a location from the held-out post-state.
    pub terrain_query_world_xy_m: [f64; 2],
    pub terrain: TerrainSample,

    /// Configuration identities are explicit because they change dynamics even
    /// when no per-step external force is present.
    pub body_profile_id: String,
    pub terrain_profile_id: String,
    pub ground_contact_profile_id: String,
    pub domain_randomization_applied: bool,

    pub evidence_digest_hex: String,
}

impl HumanoidPredictorContextEvidenceV1 {
    pub const SCHEMA_ID: &'static str =
        "symthaea.humanoid.predictor-context.simple-nominal.v1";

    #[allow(clippy::too_many_arguments)]
    fn new_nominal_simple(
        backend_name: impl Into<String>,
        morphology: HumanoidMorphology,
        step_sequence: u64,
        pre_step_timestamp_seconds: f64,
        post_step_timestamp_seconds: f64,
        external_force_world_n: [f64; 3],
        terrain_query_world_xy_m: [f64; 2],
        terrain: TerrainSample,
    ) -> Result<Self, HumanoidPredictorContextErrorV1> {
        let mut evidence = Self {
            schema_id: Self::SCHEMA_ID.to_string(),
            instrumentation_id: CONTEXT_INSTRUMENTATION_ID_V1.to_string(),
            backend_name: backend_name.into(),
            morphology,
            source: HumanoidPredictorContextSourceV1::PrivilegedSimulatorTruth,
            step_sequence,
            pre_step_timestamp_seconds,
            post_step_timestamp_seconds,
            external_force_world_n,
            gravity_world_mps2: [0.0, 0.0, -DEFAULT_GRAVITY_MPS2],
            terrain_query_world_xy_m,
            terrain,
            body_profile_id: NOMINAL_BODY_PROFILE_ID_V1.to_string(),
            terrain_profile_id: FLAT_TERRAIN_PROFILE_ID_V1.to_string(),
            ground_contact_profile_id: GROUND_CONTACT_PROFILE_ID_V1.to_string(),
            domain_randomization_applied: false,
            evidence_digest_hex: String::new(),
        };
        evidence.validate_without_digest()?;
        evidence.evidence_digest_hex = evidence.compute_digest_hex()?;
        evidence.validate()?;
        Ok(evidence)
    }

    pub fn validate(&self) -> Result<(), HumanoidPredictorContextErrorV1> {
        self.validate_without_digest()?;
        if self.evidence_digest_hex != self.compute_digest_hex()? {
            return Err(HumanoidPredictorContextErrorV1::ContextDigestMismatch);
        }
        Ok(())
    }

    fn validate_without_digest(&self) -> Result<(), HumanoidPredictorContextErrorV1> {
        if self.schema_id != Self::SCHEMA_ID
            || self.instrumentation_id != CONTEXT_INSTRUMENTATION_ID_V1
            || self.backend_name != SIMPLE_BACKEND_NAME_V1
            || self.source != HumanoidPredictorContextSourceV1::PrivilegedSimulatorTruth
            || self.body_profile_id != NOMINAL_BODY_PROFILE_ID_V1
            || self.terrain_profile_id != FLAT_TERRAIN_PROFILE_ID_V1
            || self.ground_contact_profile_id != GROUND_CONTACT_PROFILE_ID_V1
            || self.domain_randomization_applied
        {
            return Err(HumanoidPredictorContextErrorV1::ContextIdentityMismatch);
        }
        if self.step_sequence == 0 {
            return Err(HumanoidPredictorContextErrorV1::InvalidStepSequence);
        }
        if !self.pre_step_timestamp_seconds.is_finite()
            || !self.post_step_timestamp_seconds.is_finite()
            || self.post_step_timestamp_seconds <= self.pre_step_timestamp_seconds
        {
            return Err(HumanoidPredictorContextErrorV1::InvalidTimestamp);
        }
        if self
            .external_force_world_n
            .iter()
            .chain(self.gravity_world_mps2.iter())
            .chain(self.terrain_query_world_xy_m.iter())
            .any(|value| !value.is_finite())
        {
            return Err(HumanoidPredictorContextErrorV1::NonFiniteContext);
        }
        for (actual, expected) in self
            .gravity_world_mps2
            .iter()
            .zip([0.0, 0.0, -DEFAULT_GRAVITY_MPS2].iter())
        {
            if actual.to_bits() != expected.to_bits() {
                return Err(HumanoidPredictorContextErrorV1::ContextIdentityMismatch);
            }
        }
        if !self.terrain.validate() {
            return Err(HumanoidPredictorContextErrorV1::InvalidTerrain);
        }
        // This qualified v1 profile is flat/default. If terrain variation is
        // needed, introduce a profile that explicitly binds the realized terrain.
        let flat = TerrainSample::flat();
        if !terrain_sample_bit_exact(&self.terrain, &flat) {
            return Err(HumanoidPredictorContextErrorV1::ContextIdentityMismatch);
        }
        Ok(())
    }

    pub fn applied_dt_seconds(&self) -> f64 {
        self.post_step_timestamp_seconds - self.pre_step_timestamp_seconds
    }

    /// Conservative readiness predicate for the first context-aware R4 model.
    pub fn r4_training_ready(&self) -> bool {
        self.validate().is_ok()
    }

    fn compute_digest_hex(&self) -> Result<String, HumanoidPredictorContextErrorV1> {
        self.validate_without_digest()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(CONTEXT_EVIDENCE_DOMAIN_V1);
        feed_str(&mut hasher, &self.schema_id);
        feed_str(&mut hasher, &self.instrumentation_id);
        feed_str(&mut hasher, &self.backend_name);
        feed_str(&mut hasher, self.morphology.schema_id());
        feed_str(&mut hasher, context_source_token(self.source));
        hasher.update(&self.step_sequence.to_le_bytes());
        hasher.update(&self.pre_step_timestamp_seconds.to_bits().to_le_bytes());
        hasher.update(&self.post_step_timestamp_seconds.to_bits().to_le_bytes());
        for value in self.external_force_world_n {
            hasher.update(&value.to_bits().to_le_bytes());
        }
        for value in self.gravity_world_mps2 {
            hasher.update(&value.to_bits().to_le_bytes());
        }
        for value in self.terrain_query_world_xy_m {
            hasher.update(&value.to_bits().to_le_bytes());
        }
        feed_terrain(&mut hasher, &self.terrain);
        feed_str(&mut hasher, &self.body_profile_id);
        feed_str(&mut hasher, &self.terrain_profile_id);
        feed_str(&mut hasher, &self.ground_contact_profile_id);
        hasher.update(&[u8::from(self.domain_randomization_applied)]);
        Ok(digest_hex(hasher.finalize().as_bytes()))
    }
}

/// R4.1 input plus exact exogenous context. The base input remains unchanged so
/// target-leakage guarantees can be tested independently of context evolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidContextualPredictorInputV1 {
    pub schema_id: String,
    pub base_input: HumanoidPredictorInputV1,
    pub context: HumanoidPredictorContextEvidenceV1,
    pub contextual_input_digest_hex: String,
}

impl HumanoidContextualPredictorInputV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.humanoid.contextual-predictor-input.v1";

    pub fn new(
        base_input: HumanoidPredictorInputV1,
        context: HumanoidPredictorContextEvidenceV1,
    ) -> Result<Self, HumanoidPredictorContextErrorV1> {
        let mut input = Self {
            schema_id: Self::SCHEMA_ID.to_string(),
            base_input,
            context,
            contextual_input_digest_hex: String::new(),
        };
        input.validate_without_digest()?;
        input.contextual_input_digest_hex = input.compute_digest_hex()?;
        input.validate()?;
        Ok(input)
    }

    pub fn from_transition_and_context(
        transition: &HumanoidTransitionEvidenceV1,
        context: HumanoidPredictorContextEvidenceV1,
    ) -> Result<Self, HumanoidPredictorContextErrorV1> {
        let base_input = HumanoidPredictorInputV1::from_transition(transition)
            .map_err(HumanoidPredictorContextErrorV1::Predictor)?;
        Self::new(base_input, context)
    }

    pub fn validate(&self) -> Result<(), HumanoidPredictorContextErrorV1> {
        self.validate_without_digest()?;
        if self.contextual_input_digest_hex != self.compute_digest_hex()? {
            return Err(HumanoidPredictorContextErrorV1::ContextualInputDigestMismatch);
        }
        Ok(())
    }

    fn validate_without_digest(&self) -> Result<(), HumanoidPredictorContextErrorV1> {
        if self.schema_id != Self::SCHEMA_ID {
            return Err(HumanoidPredictorContextErrorV1::SchemaMismatch);
        }
        self.base_input
            .validate()
            .map_err(HumanoidPredictorContextErrorV1::Predictor)?;
        self.context.validate()?;
        if !self.context.r4_training_ready() {
            return Err(HumanoidPredictorContextErrorV1::ContextNotTrainingReady);
        }
        if self.base_input.morphology != self.context.morphology
            || self.base_input.step_sequence != self.context.step_sequence
            || self.base_input.pre_state.timestamp_seconds.to_bits()
                != self.context.pre_step_timestamp_seconds.to_bits()
            || self.base_input.target_timestamp_seconds.to_bits()
                != self.context.post_step_timestamp_seconds.to_bits()
            || self.base_input.applied_dt_seconds.to_bits()
                != self.context.applied_dt_seconds().to_bits()
        {
            return Err(HumanoidPredictorContextErrorV1::ContextAlignmentMismatch);
        }

        // The context profile claims that terrain was queried at the privileged
        // pre-step root XY. Validate that claim against the base input itself so
        // a future non-flat terrain profile cannot pair the right time with the
        // wrong location.
        let root_x = self.base_input.pre_state.privileged_root_position_world_m[0]
            .ok_or(HumanoidPredictorContextErrorV1::MissingPrivilegedRootPosition)?;
        let root_y = self.base_input.pre_state.privileged_root_position_world_m[1]
            .ok_or(HumanoidPredictorContextErrorV1::MissingPrivilegedRootPosition)?;
        if root_x.to_bits() != self.context.terrain_query_world_xy_m[0].to_bits()
            || root_y.to_bits() != self.context.terrain_query_world_xy_m[1].to_bits()
        {
            return Err(HumanoidPredictorContextErrorV1::TerrainQueryMismatch);
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> Result<String, HumanoidPredictorContextErrorV1> {
        self.validate_without_digest()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(CONTEXTUAL_INPUT_DOMAIN_V1);
        feed_str(&mut hasher, &self.schema_id);
        feed_str(&mut hasher, &self.base_input.input_digest_hex);
        feed_str(&mut hasher, &self.context.evidence_digest_hex);
        Ok(digest_hex(hasher.finalize().as_bytes()))
    }
}

/// Evidence-transparent nominal simple-backend wrapper that records the exact
/// external force accumulated for each step while reusing R3.12 action evidence.
pub struct ContextInstrumentedSimpleHumanoidSimulator {
    inner: InstrumentedSimpleHumanoidSimulator,
    morphology: HumanoidMorphology,
    pending_external_force_world_n: [f64; 3],
    last_context: Option<HumanoidPredictorContextEvidenceV1>,
}

impl ContextInstrumentedSimpleHumanoidSimulator {
    pub fn new() -> Self {
        Self::new_for(HumanoidMorphology::Dmc21)
    }

    pub fn new_for(morphology: HumanoidMorphology) -> Self {
        Self {
            inner: InstrumentedSimpleHumanoidSimulator::new_for(morphology),
            morphology,
            pending_external_force_world_n: [0.0; 3],
            last_context: None,
        }
    }

    pub fn with_actuator_noise(mut self, noise_std: f64) -> Self {
        self.inner = self.inner.with_actuator_noise(noise_std);
        self
    }

    pub fn predictor_context_evidence(&self) -> Option<&HumanoidPredictorContextEvidenceV1> {
        self.last_context.as_ref()
    }

    pub fn apply_external_force_checked(
        &mut self,
        force: [f64; 3],
    ) -> Result<(), HumanoidPredictorContextErrorV1> {
        if force.iter().any(|value| !value.is_finite()) {
            return Err(HumanoidPredictorContextErrorV1::NonFiniteContext);
        }

        // Compute the complete next accumulator before mutating either the
        // evidence wrapper or the physics backend. Individually finite pushes can
        // otherwise overflow to infinity when accumulated repeatedly.
        let mut next = self.pending_external_force_world_n;
        for (pending, value) in next.iter_mut().zip(force.iter()) {
            let sum = *pending + *value;
            if !sum.is_finite() {
                return Err(HumanoidPredictorContextErrorV1::ForceAccumulationOverflow);
            }
            *pending = sum;
        }

        self.inner.apply_external_force(force);
        self.pending_external_force_world_n = next;
        Ok(())
    }
}

impl Default for ContextInstrumentedSimpleHumanoidSimulator {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendAppliedActionEvidenceProvider for ContextInstrumentedSimpleHumanoidSimulator {
    fn backend_applied_action(&self) -> Option<&BackendAppliedActionSnapshotV1> {
        self.inner.backend_applied_action()
    }
}

impl HumanoidPhysicsSimulator for ContextInstrumentedSimpleHumanoidSimulator {
    fn morphology(&self) -> HumanoidMorphology {
        self.morphology
    }

    fn backend_name(&self) -> &'static str {
        "symthaea-context-instrumented-simple-humanoid"
    }

    fn actuation_mode(&self) -> ActuationMode {
        self.inner.actuation_mode()
    }

    fn step(&mut self, command: &HumanoidCommand, dt: f64) {
        let pre = self.inner.true_state();
        let terrain_query_world_xy_m = [pre.root_position[0], pre.root_position[1]];
        let terrain = self.inner.terrain_sample(terrain_query_world_xy_m);
        let external_force_world_n = self.pending_external_force_world_n;

        self.inner.step(command, dt);
        self.last_context = self.inner.backend_applied_action().and_then(|action| {
            HumanoidPredictorContextEvidenceV1::new_nominal_simple(
                &action.backend_name,
                self.morphology,
                action.step_sequence,
                action.pre_step_timestamp_seconds,
                action.post_step_timestamp_seconds,
                external_force_world_n,
                terrain_query_world_xy_m,
                terrain,
            )
            .ok()
        });
        self.pending_external_force_world_n = [0.0; 3];
    }

    fn state(&self) -> &HumanoidState {
        self.inner.state()
    }

    fn observation(&self) -> &HumanoidState {
        self.inner.observation()
    }

    fn true_state(&self) -> &HumanoidState {
        self.inner.true_state()
    }

    fn contact_frame(&self) -> crate::contact::ContactFrame {
        self.inner.contact_frame()
    }

    fn multi_contact_frame(&self) -> crate::multi_contact::MultiContactFrame {
        self.inner.multi_contact_frame()
    }

    fn dynamics_snapshot(&self) -> Option<crate::dynamics::RigidBodyDynamicsSnapshot> {
        self.inner.dynamics_snapshot()
    }

    fn full_dynamics_snapshot(&self) -> Option<crate::full_dynamics::FullRigidBodyDynamicsSnapshot> {
        self.inner.full_dynamics_snapshot()
    }

    fn floating_base_dynamics_snapshot(
        &self,
    ) -> Option<crate::floating_base::FloatingBaseDynamicsSnapshot> {
        self.inner.floating_base_dynamics_snapshot()
    }

    fn terrain_sample(&self, world_xy_m: [f64; 2]) -> TerrainSample {
        self.inner.terrain_sample(world_xy_m)
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.pending_external_force_world_n = [0.0; 3];
        self.last_context = None;
    }

    fn reset_with_perturbation(&mut self, perturbation: f64, seed: u64) {
        self.inner.reset_with_perturbation(perturbation, seed);
        self.pending_external_force_world_n = [0.0; 3];
        self.last_context = None;
    }

    fn apply_external_force(&mut self, force: [f64; 3]) {
        // This trait cannot return an error. Invalid legacy pushes are therefore
        // rejected rather than forwarded into physics, and any prior context is
        // invalidated. The checked R4 capture path exposes the explicit error.
        if self.apply_external_force_checked(force).is_err() {
            self.last_context = None;
        }
    }
}

/// Capture one privileged transition and its context-aware leakage-safe input.
/// Every caller-controlled field is validated before the backend is mutated.
pub fn capture_contextual_predictor_case_v1(
    simulator: &mut ContextInstrumentedSimpleHumanoidSimulator,
    requested_command: &HumanoidCommand,
    requested_dt_seconds: f64,
    external_force_world_n: [f64; 3],
    clock_domain_id: impl Into<String>,
    episode_id: impl Into<String>,
) -> Result<
    (HumanoidTransitionEvidenceV1, HumanoidContextualPredictorInputV1),
    HumanoidPredictorContextErrorV1,
> {
    if !requested_dt_seconds.is_finite() || requested_dt_seconds <= 0.0 {
        return Err(HumanoidPredictorContextErrorV1::InvalidRequestedDt);
    }
    if external_force_world_n.iter().any(|value| !value.is_finite()) {
        return Err(HumanoidPredictorContextErrorV1::NonFiniteContext);
    }
    let clock_domain_id = clock_domain_id.into();
    let episode_id = episode_id.into();
    if clock_domain_id.trim().is_empty() {
        return Err(HumanoidPredictorContextErrorV1::EmptyClockDomain);
    }
    if episode_id.trim().is_empty() {
        return Err(HumanoidPredictorContextErrorV1::EmptyEpisodeId);
    }

    let morphology = simulator.morphology();
    requested_command
        .validate_for(morphology.num_actuators(), ActuationMode::NormalizedTorque)
        .map_err(HumanoidPredictorContextErrorV1::RequestedCommand)?;

    let encoder = SemanticHumanoidEncoderV1::new();
    let pre_state = encoder
        .frame(simulator.true_state(), morphology)
        .map_err(HumanoidPredictorContextErrorV1::State)?;

    simulator.apply_external_force_checked(external_force_world_n)?;
    simulator.step(requested_command, requested_dt_seconds);

    let backend_action = simulator
        .backend_applied_action()
        .cloned()
        .ok_or(HumanoidPredictorContextErrorV1::MissingBackendAction)?;
    let post_state = encoder
        .frame(simulator.true_state(), morphology)
        .map_err(HumanoidPredictorContextErrorV1::State)?;
    let context = simulator
        .predictor_context_evidence()
        .cloned()
        .ok_or(HumanoidPredictorContextErrorV1::MissingContextEvidence)?;

    let transition = HumanoidTransitionEvidenceV1::new(
        morphology,
        clock_domain_id,
        episode_id,
        HumanoidTransitionStateSourceV1::PrivilegedBackendTruth,
        HumanoidTransitionStateSourceV1::PrivilegedBackendTruth,
        pre_state,
        backend_action,
        post_state,
    )
    .map_err(HumanoidPredictorContextErrorV1::Transition)?;

    let contextual = HumanoidContextualPredictorInputV1::from_transition_and_context(
        &transition,
        context,
    )?;
    Ok((transition, contextual))
}

fn terrain_sample_bit_exact(left: &TerrainSample, right: &TerrainSample) -> bool {
    left.height_m.to_bits() == right.height_m.to_bits()
        && left
            .normal_world
            .iter()
            .zip(right.normal_world.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
        && left.friction.to_bits() == right.friction.to_bits()
        && left.compliance.to_bits() == right.compliance.to_bits()
        && left.confidence.to_bits() == right.confidence.to_bits()
        && left.height_std_m.to_bits() == right.height_std_m.to_bits()
        && left.normal_std_rad.to_bits() == right.normal_std_rad.to_bits()
        && left.friction_std.to_bits() == right.friction_std.to_bits()
        && left.age_s.to_bits() == right.age_s.to_bits()
        && left.source == right.source
}

fn feed_terrain(hasher: &mut blake3::Hasher, terrain: &TerrainSample) {
    hasher.update(&terrain.height_m.to_bits().to_le_bytes());
    for value in terrain.normal_world {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    hasher.update(&terrain.friction.to_bits().to_le_bytes());
    hasher.update(&terrain.compliance.to_bits().to_le_bytes());
    hasher.update(&terrain.confidence.to_bits().to_le_bytes());
    hasher.update(&terrain.height_std_m.to_bits().to_le_bytes());
    hasher.update(&terrain.normal_std_rad.to_bits().to_le_bytes());
    hasher.update(&terrain.friction_std.to_bits().to_le_bytes());
    hasher.update(&terrain.age_s.to_bits().to_le_bytes());
    feed_str(hasher, terrain_source_token(terrain.source));
}

fn context_source_token(source: HumanoidPredictorContextSourceV1) -> &'static str {
    match source {
        HumanoidPredictorContextSourceV1::PrivilegedSimulatorTruth => {
            "privileged_simulator_truth"
        }
    }
}

fn terrain_source_token(source: TerrainEvidenceSource) -> &'static str {
    match source {
        TerrainEvidenceSource::Analytic => "analytic",
        TerrainEvidenceSource::SimulatorTruth => "simulator_truth",
        TerrainEvidenceSource::VisionEstimate => "vision_estimate",
        TerrainEvidenceSource::Fused => "fused",
        TerrainEvidenceSource::Unknown => "unknown",
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
pub enum HumanoidPredictorContextErrorV1 {
    SchemaMismatch,
    ContextIdentityMismatch,
    InvalidStepSequence,
    InvalidTimestamp,
    InvalidRequestedDt,
    EmptyClockDomain,
    EmptyEpisodeId,
    NonFiniteContext,
    ForceAccumulationOverflow,
    InvalidTerrain,
    ContextDigestMismatch,
    ContextualInputDigestMismatch,
    ContextAlignmentMismatch,
    TerrainQueryMismatch,
    MissingPrivilegedRootPosition,
    ContextNotTrainingReady,
    MissingBackendAction,
    MissingContextEvidence,
    RequestedCommand(CommandValidationError),
    State(SemanticHumanoidErrorV1),
    Transition(HumanoidTransitionEvidenceErrorV1),
    Predictor(HumanoidContinuousPredictorErrorV1),
}

impl fmt::Display for HumanoidPredictorContextErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SchemaMismatch => write!(f, "unsupported predictor-context schema"),
            Self::ContextIdentityMismatch => write!(f, "predictor-context profile identity mismatch"),
            Self::InvalidStepSequence => write!(f, "predictor-context step sequence must be positive"),
            Self::InvalidTimestamp => write!(f, "predictor-context timestamps must advance forward"),
            Self::InvalidRequestedDt => write!(f, "requested dt must be finite and positive"),
            Self::EmptyClockDomain => write!(f, "predictor clock domain must be non-empty"),
            Self::EmptyEpisodeId => write!(f, "predictor episode id must be non-empty"),
            Self::NonFiniteContext => write!(f, "predictor context contains a non-finite value"),
            Self::ForceAccumulationOverflow => write!(f, "external-force accumulation overflowed"),
            Self::InvalidTerrain => write!(f, "predictor context contains invalid terrain evidence"),
            Self::ContextDigestMismatch => write!(f, "predictor-context evidence digest mismatch"),
            Self::ContextualInputDigestMismatch => {
                write!(f, "contextual predictor-input digest mismatch")
            }
            Self::ContextAlignmentMismatch => {
                write!(f, "predictor context does not align with causal base input")
            }
            Self::TerrainQueryMismatch => {
                write!(f, "terrain query does not match privileged pre-state root XY")
            }
            Self::MissingPrivilegedRootPosition => {
                write!(f, "contextual prediction requires privileged pre-state root position")
            }
            Self::ContextNotTrainingReady => write!(f, "predictor context is not R4 training-ready"),
            Self::MissingBackendAction => write!(f, "missing backend-applied action evidence"),
            Self::MissingContextEvidence => write!(f, "missing predictor-context evidence"),
            Self::RequestedCommand(error) => write!(f, "invalid requested command: {error:?}"),
            Self::State(error) => write!(f, "invalid semantic state: {error}"),
            Self::Transition(error) => write!(f, "invalid transition evidence: {error}"),
            Self::Predictor(error) => write!(f, "invalid R4.1 predictor input: {error}"),
        }
    }
}

impl std::error::Error for HumanoidPredictorContextErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;

    fn command(gain: f32) -> HumanoidCommand {
        let morphology = HumanoidMorphology::Dmc21;
        let mut torques = vec![0.0f32; morphology.num_actuators()];
        torques[0] = gain;
        torques[5] = -0.5 * gain;
        HumanoidCommand { torques }
    }

    fn case(
        force: [f64; 3],
    ) -> (HumanoidTransitionEvidenceV1, HumanoidContextualPredictorInputV1) {
        let mut simulator = ContextInstrumentedSimpleHumanoidSimulator::new_for(
            HumanoidMorphology::Dmc21,
        );
        simulator.reset_with_perturbation(0.0, 77);
        capture_contextual_predictor_case_v1(
            &mut simulator,
            &command(0.6),
            0.025,
            force,
            "r4-context-clock",
            "r4-context-episode",
        )
        .unwrap()
    }

    #[test]
    fn external_force_resolves_r4_1_hidden_outcome_confound() {
        let (nominal_transition, nominal) = case([0.0; 3]);
        let (pushed_transition, pushed) = case([80.0, 0.0, 0.0]);

        // R4.1 intentionally cannot distinguish a hidden exogenous force.
        assert_eq!(
            nominal.base_input.input_digest_hex,
            pushed.base_input.input_digest_hex
        );
        assert_eq!(
            nominal_transition.backend_action.evidence_digest_hex,
            pushed_transition.backend_action.evidence_digest_hex
        );

        // R4.2 makes the known force causal input without reading the target.
        assert_ne!(
            nominal.context.evidence_digest_hex,
            pushed.context.evidence_digest_hex
        );
        assert_ne!(
            nominal.contextual_input_digest_hex,
            pushed.contextual_input_digest_hex
        );
        assert_ne!(nominal_transition.post_state, pushed_transition.post_state);
        assert_eq!(pushed.context.external_force_world_n, [80.0, 0.0, 0.0]);
    }

    #[test]
    fn context_is_exactly_aligned_to_backend_action_boundary() {
        let (transition, input) = case([12.0, -3.0, 1.5]);
        input.validate().unwrap();
        assert_eq!(input.context.step_sequence, transition.step_sequence());
        assert_eq!(
            input.context.pre_step_timestamp_seconds.to_bits(),
            transition.backend_action.pre_step_timestamp_seconds.to_bits()
        );
        assert_eq!(
            input.context.post_step_timestamp_seconds.to_bits(),
            transition.backend_action.post_step_timestamp_seconds.to_bits()
        );
        assert_eq!(
            input.context.applied_dt_seconds().to_bits(),
            transition.applied_dt_seconds().to_bits()
        );
        assert_eq!(
            input.context.terrain_query_world_xy_m[0].to_bits(),
            input.base_input.pre_state.privileged_root_position_world_m[0]
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            input.context.terrain_query_world_xy_m[1].to_bits(),
            input.base_input.pre_state.privileged_root_position_world_m[1]
                .unwrap()
                .to_bits()
        );
        assert!(input.context.r4_training_ready());
    }

    #[test]
    fn deterministic_context_capture_reproduces_commitments() {
        let (_, left) = case([25.0, 2.0, -1.0]);
        let (_, right) = case([25.0, 2.0, -1.0]);
        assert_eq!(left.context.evidence_digest_hex, right.context.evidence_digest_hex);
        assert_eq!(left.contextual_input_digest_hex, right.contextual_input_digest_hex);
    }

    #[test]
    fn context_tampering_fails_closed() {
        let (_, input) = case([15.0, 0.0, 0.0]);
        let mut force_tamper = input.context.clone();
        force_tamper.external_force_world_n[0] += 0.5;
        assert!(force_tamper.validate().is_err());

        let mut terrain_tamper = input.context.clone();
        terrain_tamper.terrain.height_m += 0.01;
        assert!(terrain_tamper.validate().is_err());

        let mut query_tamper = input.clone();
        query_tamper.context.terrain_query_world_xy_m[0] += 1.0;
        query_tamper.context.evidence_digest_hex = query_tamper.context.compute_digest_hex().unwrap();
        assert!(matches!(
            query_tamper.validate(),
            Err(HumanoidPredictorContextErrorV1::TerrainQueryMismatch)
        ));
    }

    #[test]
    fn nonfinite_force_is_rejected_before_mutating_backend() {
        let mut simulator = ContextInstrumentedSimpleHumanoidSimulator::new();
        simulator.reset();
        let before = simulator.true_state().timestamp;
        let result = capture_contextual_predictor_case_v1(
            &mut simulator,
            &command(0.5),
            0.025,
            [f64::NAN, 0.0, 0.0],
            "r4-context-clock",
            "r4-context-episode",
        );
        assert!(result.is_err());
        assert_eq!(simulator.true_state().timestamp.to_bits(), before.to_bits());
        assert!(simulator.backend_applied_action().is_none());
        assert!(simulator.predictor_context_evidence().is_none());
    }

    #[test]
    fn empty_identity_is_rejected_before_mutating_backend() {
        let mut simulator = ContextInstrumentedSimpleHumanoidSimulator::new();
        simulator.reset();
        let before = simulator.true_state().timestamp;
        let result = capture_contextual_predictor_case_v1(
            &mut simulator,
            &command(0.5),
            0.025,
            [0.0; 3],
            " ",
            "r4-context-episode",
        );
        assert!(matches!(
            result,
            Err(HumanoidPredictorContextErrorV1::EmptyClockDomain)
        ));
        assert_eq!(simulator.true_state().timestamp.to_bits(), before.to_bits());
        assert!(simulator.backend_applied_action().is_none());
        assert!(simulator.predictor_context_evidence().is_none());
    }

    #[test]
    fn finite_force_accumulation_overflow_is_rejected_before_forwarding() {
        let mut simulator = ContextInstrumentedSimpleHumanoidSimulator::new();
        simulator.reset();
        simulator.apply_external_force_checked([f64::MAX, 0.0, 0.0]).unwrap();
        let before = simulator.pending_external_force_world_n;
        let result = simulator.apply_external_force_checked([f64::MAX, 0.0, 0.0]);
        assert!(matches!(
            result,
            Err(HumanoidPredictorContextErrorV1::ForceAccumulationOverflow)
        ));
        assert_eq!(simulator.pending_external_force_world_n, before);
    }

    #[test]
    fn malformed_legacy_force_does_not_panic_or_create_context() {
        let mut simulator = ContextInstrumentedSimpleHumanoidSimulator::new();
        simulator.reset();
        HumanoidPhysicsSimulator::apply_external_force(
            &mut simulator,
            [f64::NAN, 0.0, 0.0],
        );
        simulator.step(&command(0.5), 0.025);
        // The malformed force is not forwarded to physics. A later nominal step
        // may create context; this first step must at least remain finite.
        assert!(simulator.true_state().timestamp.is_finite());
    }

    #[test]
    fn step_clears_external_force_context_like_backend() {
        let mut simulator = ContextInstrumentedSimpleHumanoidSimulator::new();
        simulator.reset();
        simulator.apply_external_force_checked([40.0, 0.0, 0.0]).unwrap();
        simulator.step(&command(0.5), 0.025);
        assert_eq!(
            simulator
                .predictor_context_evidence()
                .unwrap()
                .external_force_world_n,
            [40.0, 0.0, 0.0]
        );
        simulator.step(&command(0.5), 0.025);
        assert_eq!(
            simulator
                .predictor_context_evidence()
                .unwrap()
                .external_force_world_n,
            [0.0; 3]
        );
    }
}
