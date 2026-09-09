// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-transparent backend-applied action instrumentation.
//!
//! The legacy [`SimpleHumanoidSimulator`] contains a private command delay/noise
//! boundary. R3.12 does not mutate that large backend. Instead this module wraps
//! an otherwise actuator-noise-disabled instance, reproduces the existing DMC21
//! actuator-adaptation algorithm, records the exact command sent to physics, and
//! qualifies the wrapper against the legacy DMC21 backend with exact trajectory
//! regressions.
//!
//! The legacy private delay buffer is initialized with `HumanoidCommand::zero()`,
//! which is explicitly DMC21-only. This instrumented wrapper does not copy that
//! extended-morphology cardinality defect: its buffer is sized from the selected
//! morphology. Therefore the bit-exact *legacy compatibility* claim is DMC21-only,
//! while 27/53/64-actuator instrumentation uses the corrected cardinality.
//!
//! This is backend-applied command evidence, not hardware execution evidence.
//! Real hardware still requires independently read actuator/current/torque state
//! before a command may be labeled `MeasuredExecution`.

use serde::{Deserialize, Serialize};
use std::fmt::Write as _;

use crate::morphology::HumanoidMorphology;
use crate::semantic_action::{
    HumanoidActuationStageV1, SemanticHumanoidActuationErrorV1,
    SemanticHumanoidActuationFrameV1,
};
use crate::simulator::{HumanoidPhysicsSimulator, SimpleHumanoidSimulator};
use crate::types::{ActuationMode, HumanoidCommand, HumanoidState};

const BACKEND_ACTION_EVIDENCE_DOMAIN_V1: &[u8] =
    b"symthaea.humanoid.backend-applied-action.v1\0";
const SIMPLE_ACTUATOR_MODEL_ID_V1: &str =
    "symthaea.instrumented-simple-humanoid.actuator-delay-noise.v1";
const INSTRUMENTATION_ID_V1: &str =
    "symthaea.instrumented-simple-humanoid.backend-action.v1";

/// Exact command evidence for one backend physics transition.
///
/// `requested_dt_seconds` is what the caller supplied. `applied_dt_seconds` is
/// measured from the backend's privileged pre/post timestamps and is therefore
/// kept separate for backends whose internal timestep may differ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendAppliedActionSnapshotV1 {
    pub schema_id: String,
    pub instrumentation_id: String,
    pub backend_name: String,
    pub morphology: HumanoidMorphology,
    /// 1-based step sequence within the current reset episode.
    pub step_sequence: u64,
    pub pre_step_timestamp_seconds: f64,
    pub post_step_timestamp_seconds: f64,
    pub requested_dt_seconds: f64,
    pub applied_dt_seconds: f64,
    pub actuation_mode: ActuationMode,
    /// Exact command consumed by the wrapped physics backend.
    pub command: HumanoidCommand,
    pub backend_adaptation_applied: bool,
    pub actuator_model_id: String,
    pub actuator_delay_ticks: usize,
    pub actuator_noise_std: f64,
    /// Domain-separated content commitment over every field above.
    pub evidence_digest_hex: String,
}

impl BackendAppliedActionSnapshotV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.humanoid.backend-applied-action.v1";

    #[allow(clippy::too_many_arguments)]
    fn new(
        backend_name: &str,
        morphology: HumanoidMorphology,
        step_sequence: u64,
        pre_step_timestamp_seconds: f64,
        post_step_timestamp_seconds: f64,
        requested_dt_seconds: f64,
        actuation_mode: ActuationMode,
        command: HumanoidCommand,
        backend_adaptation_applied: bool,
        actuator_delay_ticks: usize,
        actuator_noise_std: f64,
    ) -> Self {
        let applied_dt_seconds = post_step_timestamp_seconds - pre_step_timestamp_seconds;
        let mut snapshot = Self {
            schema_id: Self::SCHEMA_ID.to_string(),
            instrumentation_id: INSTRUMENTATION_ID_V1.to_string(),
            backend_name: backend_name.to_string(),
            morphology,
            step_sequence,
            pre_step_timestamp_seconds,
            post_step_timestamp_seconds,
            requested_dt_seconds,
            applied_dt_seconds,
            actuation_mode,
            command,
            backend_adaptation_applied,
            actuator_model_id: SIMPLE_ACTUATOR_MODEL_ID_V1.to_string(),
            actuator_delay_ticks,
            actuator_noise_std,
            evidence_digest_hex: String::new(),
        };
        snapshot.evidence_digest_hex = snapshot.compute_digest_hex();
        snapshot
    }

    pub fn evidence_id(&self) -> String {
        format!("{}:{}", Self::SCHEMA_ID, self.evidence_digest_hex)
    }

    pub fn validate(&self) -> Result<(), BackendAppliedActionEvidenceErrorV1> {
        if self.schema_id != Self::SCHEMA_ID {
            return Err(BackendAppliedActionEvidenceErrorV1::SchemaMismatch);
        }
        if self.instrumentation_id != INSTRUMENTATION_ID_V1
            || self.backend_name.trim().is_empty()
            || self.actuator_model_id != SIMPLE_ACTUATOR_MODEL_ID_V1
        {
            return Err(BackendAppliedActionEvidenceErrorV1::IdentityMismatch);
        }
        if self.step_sequence == 0 {
            return Err(BackendAppliedActionEvidenceErrorV1::InvalidStepSequence);
        }
        if !self.pre_step_timestamp_seconds.is_finite()
            || !self.post_step_timestamp_seconds.is_finite()
            || self.post_step_timestamp_seconds < self.pre_step_timestamp_seconds
        {
            return Err(BackendAppliedActionEvidenceErrorV1::InvalidTimestamp);
        }
        if !self.requested_dt_seconds.is_finite()
            || self.requested_dt_seconds <= 0.0
            || !self.applied_dt_seconds.is_finite()
            || self.applied_dt_seconds <= 0.0
        {
            return Err(BackendAppliedActionEvidenceErrorV1::InvalidDt);
        }
        let measured_dt = self.post_step_timestamp_seconds - self.pre_step_timestamp_seconds;
        let dt_tolerance = 1.0e-12 * measured_dt.abs().max(1.0);
        if (measured_dt - self.applied_dt_seconds).abs() > dt_tolerance {
            return Err(BackendAppliedActionEvidenceErrorV1::AppliedDtMismatch);
        }
        let expected = self.morphology.num_actuators();
        if self.command.num_actuators() != expected {
            return Err(BackendAppliedActionEvidenceErrorV1::ActuatorCount {
                expected,
                actual: self.command.num_actuators(),
            });
        }
        if self.command.torques.iter().any(|value| !value.is_finite())
            || !self.actuator_noise_std.is_finite()
            || self.actuator_noise_std < 0.0
        {
            return Err(BackendAppliedActionEvidenceErrorV1::NonFiniteValue);
        }
        if self.evidence_digest_hex != self.compute_digest_hex() {
            return Err(BackendAppliedActionEvidenceErrorV1::DigestMismatch);
        }
        Ok(())
    }

    /// Convert this exact backend-applied command to the portable physical R3.10
    /// action frame. The simple backend consumes normalized torque, so the
    /// repository's existing actuation adapter performs the N·m projection.
    pub fn to_physical_frame(
        &self,
    ) -> Result<SemanticHumanoidActuationFrameV1, SemanticHumanoidActuationErrorV1> {
        match self.actuation_mode {
            ActuationMode::NormalizedTorque => {
                // Torque conversion uses morphology torque scales. The current
                // ActuationAdapter also validates state cardinality, so a neutral
                // state of the same morphology supplies that structural check;
                // joint values do not participate in the torque branch.
                let structural_state = HumanoidState::standing_for(self.morphology);
                SemanticHumanoidActuationFrameV1::project_normalized_torque_command(
                    &self.command,
                    &structural_state,
                    self.morphology,
                    HumanoidActuationStageV1::BackendAppliedPhysical,
                    Some(self.evidence_id()),
                )
            }
            ActuationMode::TorqueNewtonMetres | ActuationMode::PositionTargetRadians => {
                SemanticHumanoidActuationFrameV1::from_physical_command(
                    &self.command,
                    self.morphology,
                    HumanoidActuationStageV1::BackendAppliedPhysical,
                    self.actuation_mode,
                    Some(self.evidence_id()),
                )
            }
            ActuationMode::NormalizedPosition => {
                // Preserve R3.10's physical-only boundary rather than inventing a
                // position calibration that this evidence record does not carry.
                SemanticHumanoidActuationFrameV1::from_physical_command(
                    &self.command,
                    self.morphology,
                    HumanoidActuationStageV1::BackendAppliedPhysical,
                    self.actuation_mode,
                    Some(self.evidence_id()),
                )
            }
        }
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(BACKEND_ACTION_EVIDENCE_DOMAIN_V1);
        feed_str(&mut hasher, &self.schema_id);
        feed_str(&mut hasher, &self.instrumentation_id);
        feed_str(&mut hasher, &self.backend_name);
        feed_str(&mut hasher, self.morphology.schema_id());
        hasher.update(&self.step_sequence.to_le_bytes());
        hasher.update(&self.pre_step_timestamp_seconds.to_bits().to_le_bytes());
        hasher.update(&self.post_step_timestamp_seconds.to_bits().to_le_bytes());
        hasher.update(&self.requested_dt_seconds.to_bits().to_le_bytes());
        hasher.update(&self.applied_dt_seconds.to_bits().to_le_bytes());
        feed_str(&mut hasher, actuation_mode_token(self.actuation_mode));
        hasher.update(&(self.command.torques.len() as u64).to_le_bytes());
        for value in &self.command.torques {
            hasher.update(&value.to_bits().to_le_bytes());
        }
        hasher.update(&[u8::from(self.backend_adaptation_applied)]);
        feed_str(&mut hasher, &self.actuator_model_id);
        hasher.update(&(self.actuator_delay_ticks as u64).to_le_bytes());
        hasher.update(&self.actuator_noise_std.to_bits().to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
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

fn actuation_mode_token(mode: ActuationMode) -> &'static str {
    match mode {
        ActuationMode::NormalizedTorque => "normalized_torque",
        ActuationMode::TorqueNewtonMetres => "torque_newton_metres",
        ActuationMode::PositionTargetRadians => "position_target_radians",
        ActuationMode::NormalizedPosition => "normalized_position",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendAppliedActionEvidenceErrorV1 {
    SchemaMismatch,
    IdentityMismatch,
    InvalidStepSequence,
    InvalidTimestamp,
    InvalidDt,
    AppliedDtMismatch,
    ActuatorCount { expected: usize, actual: usize },
    NonFiniteValue,
    DigestMismatch,
}

/// Optional capability for backends that can truthfully expose the exact command
/// consumed by their plant/physics boundary.
pub trait BackendAppliedActionEvidenceProvider {
    fn backend_applied_action(&self) -> Option<&BackendAppliedActionSnapshotV1>;
}

/// Evidence-transparent implementation of the SimpleHumanoid actuator model.
///
/// The delay/noise/RNG algorithm is exactly the legacy algorithm. The one
/// intentional difference is buffer cardinality: this implementation uses the
/// selected morphology's actuator count instead of DMC21-only `zero()` commands.
/// Exact legacy trajectory equivalence is therefore qualified only for DMC21.
struct InstrumentedActuatorModelV1 {
    command_buffer: Vec<HumanoidCommand>,
    write_idx: usize,
    delay_ticks: usize,
    max_delay: usize,
    noise_std: f64,
    rng_state: u64,
    enabled: bool,
    actuator_count: usize,
}

impl InstrumentedActuatorModelV1 {
    fn new(max_delay: usize, noise_std: f64, actuator_count: usize) -> Self {
        Self {
            command_buffer: vec![HumanoidCommand::zero_for(actuator_count); max_delay + 1],
            write_idx: 0,
            delay_ticks: 1,
            max_delay,
            noise_std,
            rng_state: 0,
            enabled: false,
            actuator_count,
        }
    }

    fn reset(&mut self, seed: u64) {
        let hash = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.delay_ticks = if self.max_delay > 0 {
            1 + (hash as usize % self.max_delay).min(self.max_delay)
        } else {
            1
        };
        self.rng_state = seed;
        for slot in &mut self.command_buffer {
            *slot = HumanoidCommand::zero_for(self.actuator_count);
        }
        self.write_idx = 0;
    }

    fn apply(&mut self, command: &HumanoidCommand) -> HumanoidCommand {
        self.command_buffer[self.write_idx] = command.clone();
        self.write_idx = (self.write_idx + 1) % self.command_buffer.len();

        let capacity = self.command_buffer.len();
        let read_idx = (self.write_idx + capacity - self.delay_ticks - 1) % capacity;
        let mut delayed = self.command_buffer[read_idx].clone();

        for torque in &mut delayed.torques {
            self.rng_state ^= self.rng_state << 13;
            self.rng_state ^= self.rng_state >> 7;
            self.rng_state ^= self.rng_state << 17;
            let noise = (self.rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0;
            *torque = (*torque + (noise * self.noise_std) as f32).clamp(-1.0, 1.0);
        }
        delayed
    }
}

/// Evidence-transparent reference wrapper for the simple humanoid backend.
///
/// The inner simulator keeps its own actuator-noise model disabled. This wrapper
/// applies the instrumented model once, records the resulting command, then
/// passes that exact command into the ordinary physics step.
pub struct InstrumentedSimpleHumanoidSimulator {
    inner: SimpleHumanoidSimulator,
    actuator_model: InstrumentedActuatorModelV1,
    morphology: HumanoidMorphology,
    step_sequence: u64,
    last_backend_applied_action: Option<BackendAppliedActionSnapshotV1>,
}

impl InstrumentedSimpleHumanoidSimulator {
    pub fn new() -> Self {
        Self::new_for(HumanoidMorphology::Dmc21)
    }

    pub fn new_for(morphology: HumanoidMorphology) -> Self {
        Self {
            inner: SimpleHumanoidSimulator::new_for(morphology),
            actuator_model: InstrumentedActuatorModelV1::new(
                3,
                0.03,
                morphology.num_actuators(),
            ),
            morphology,
            step_sequence: 0,
            last_backend_applied_action: None,
        }
    }

    pub fn with_actuator_noise(mut self, noise_std: f64) -> Self {
        self.actuator_model.noise_std = noise_std;
        self.actuator_model.enabled = noise_std > 0.0;
        self
    }

    pub fn inner(&self) -> &SimpleHumanoidSimulator {
        &self.inner
    }

    pub fn last_backend_applied_action(&self) -> Option<&BackendAppliedActionSnapshotV1> {
        self.last_backend_applied_action.as_ref()
    }
}

impl Default for InstrumentedSimpleHumanoidSimulator {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendAppliedActionEvidenceProvider for InstrumentedSimpleHumanoidSimulator {
    fn backend_applied_action(&self) -> Option<&BackendAppliedActionSnapshotV1> {
        self.last_backend_applied_action()
    }
}

impl HumanoidPhysicsSimulator for InstrumentedSimpleHumanoidSimulator {
    fn morphology(&self) -> HumanoidMorphology {
        self.morphology
    }

    fn backend_name(&self) -> &'static str {
        "symthaea-instrumented-simple-humanoid"
    }

    fn actuation_mode(&self) -> ActuationMode {
        ActuationMode::NormalizedTorque
    }

    fn step(&mut self, command: &HumanoidCommand, dt: f64) {
        let adaptation_applied = self.actuator_model.enabled;
        let applied_command = if adaptation_applied {
            self.actuator_model.apply(command)
        } else {
            command.clone()
        };

        let pre_step_timestamp_seconds = self.inner.true_state().timestamp;
        self.inner.step(&applied_command, dt);
        let post_step_timestamp_seconds = self.inner.true_state().timestamp;
        self.step_sequence = self.step_sequence.saturating_add(1);

        self.last_backend_applied_action = Some(BackendAppliedActionSnapshotV1::new(
            "symthaea-simple-humanoid",
            self.morphology,
            self.step_sequence,
            pre_step_timestamp_seconds,
            post_step_timestamp_seconds,
            dt,
            ActuationMode::NormalizedTorque,
            applied_command,
            adaptation_applied,
            self.actuator_model.delay_ticks,
            self.actuator_model.noise_std,
        ));
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

    fn terrain_sample(&self, world_xy_m: [f64; 2]) -> crate::terrain::TerrainSample {
        self.inner.terrain_sample(world_xy_m)
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.actuator_model.reset(0);
        self.step_sequence = 0;
        self.last_backend_applied_action = None;
    }

    fn reset_with_perturbation(&mut self, perturbation: f64, seed: u64) {
        self.inner.reset_with_perturbation(perturbation, seed);
        self.actuator_model.reset(seed.wrapping_mul(2654435761));
        self.step_sequence = 0;
        self.last_backend_applied_action = None;
    }

    fn apply_external_force(&mut self, force: [f64; 3]) {
        self.inner.apply_external_force(force);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic_action::compare_humanoid_actuation_stages_v1;

    fn command_for_step(step: usize, morphology: HumanoidMorphology) -> HumanoidCommand {
        let mut torques = vec![0.0f32; morphology.num_actuators()];
        torques[0] = (0.35 + 0.01 * step as f32).clamp(-0.9, 0.9);
        if torques.len() > 6 {
            torques[6] = -0.22;
        }
        if torques.len() > 14 {
            torques[14] = 0.17;
        }
        HumanoidCommand { torques }
    }

    fn assert_state_bit_exact(left: &HumanoidState, right: &HumanoidState) {
        let left_channels = left.to_channels();
        let right_channels = right.to_channels();
        assert_eq!(left_channels.len(), right_channels.len());
        for (index, (a, b)) in left_channels.iter().zip(right_channels.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "policy channel {index}");
        }
        for axis in 0..3 {
            assert_eq!(
                left.root_position[axis].to_bits(),
                right.root_position[axis].to_bits(),
                "root position axis {axis}"
            );
        }
        assert_eq!(left.timestamp.to_bits(), right.timestamp.to_bits(), "timestamp");
    }

    #[test]
    fn instrumented_wrapper_is_bit_exact_with_legacy_dmc21_actuator_model() {
        let morphology = HumanoidMorphology::Dmc21;
        let noise_std = 0.03;
        let seed = 37;
        let dt = 0.025;
        let mut legacy = SimpleHumanoidSimulator::new_for(morphology).with_actuator_noise(noise_std);
        let mut instrumented =
            InstrumentedSimpleHumanoidSimulator::new_for(morphology).with_actuator_noise(noise_std);
        legacy.reset_with_perturbation(0.0, seed);
        instrumented.reset_with_perturbation(0.0, seed);

        for step in 0..32 {
            let command = command_for_step(step, morphology);
            legacy.step(&command, dt);
            instrumented.step(&command, dt);
            assert_state_bit_exact(legacy.true_state(), instrumented.true_state());
            instrumented
                .backend_applied_action()
                .expect("instrumented backend must expose applied action")
                .validate()
                .unwrap();
        }
    }

    #[test]
    fn extended_morphology_evidence_preserves_actuator_cardinality() {
        for morphology in [
            HumanoidMorphology::WithNeckWrist,
            HumanoidMorphology::Dexterous53,
            HumanoidMorphology::FullSpine,
        ] {
            let mut simulator =
                InstrumentedSimpleHumanoidSimulator::new_for(morphology).with_actuator_noise(0.03);
            simulator.reset_with_perturbation(0.0, 17);
            simulator.step(&command_for_step(0, morphology), 0.025);
            let snapshot = simulator.backend_applied_action().unwrap();
            assert_eq!(snapshot.command.num_actuators(), morphology.num_actuators());
            snapshot.validate().unwrap();
        }
    }

    #[test]
    fn no_adaptation_records_exact_requested_command() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut simulator = InstrumentedSimpleHumanoidSimulator::new_for(morphology);
        simulator.reset_with_perturbation(0.0, 11);
        let command = command_for_step(0, morphology);
        simulator.step(&command, 0.025);
        let snapshot = simulator.backend_applied_action().unwrap();

        assert!(!snapshot.backend_adaptation_applied);
        assert_eq!(snapshot.step_sequence, 1);
        assert_eq!(snapshot.command.torques.len(), command.torques.len());
        for (requested, applied) in command.torques.iter().zip(snapshot.command.torques.iter()) {
            assert_eq!(requested.to_bits(), applied.to_bits());
        }
        assert!((snapshot.requested_dt_seconds - snapshot.applied_dt_seconds).abs() < 1.0e-12);
        snapshot.validate().unwrap();
    }

    #[test]
    fn backend_applied_frame_exposes_requested_vs_applied_delta() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut simulator =
            InstrumentedSimpleHumanoidSimulator::new_for(morphology).with_actuator_noise(0.03);
        simulator.reset_with_perturbation(0.0, 42);
        let command = command_for_step(0, morphology);
        let requested = SemanticHumanoidActuationFrameV1::project_requested_normalized_torque_command(
            &command,
            simulator.true_state(),
            morphology,
        )
        .unwrap();

        simulator.step(&command, 0.025);
        let snapshot = simulator.backend_applied_action().unwrap();
        let evidence_id = snapshot.evidence_id();
        let applied = snapshot.to_physical_frame().unwrap();
        let comparison = compare_humanoid_actuation_stages_v1(&requested, &applied).unwrap();

        assert_eq!(applied.stage, HumanoidActuationStageV1::BackendAppliedPhysical);
        assert_eq!(applied.evidence_id.as_deref(), Some(evidence_id.as_str()));
        assert!(comparison.any_action_changed());
        assert!(comparison.changed_actuators > 0);
        assert!(snapshot.backend_adaptation_applied);
    }

    #[test]
    fn evidence_digest_detects_command_tampering() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut simulator = InstrumentedSimpleHumanoidSimulator::new_for(morphology);
        simulator.reset();
        simulator.step(&command_for_step(0, morphology), 0.025);
        let mut snapshot = simulator.backend_applied_action().unwrap().clone();
        snapshot.command.torques[0] += 0.001;
        assert_eq!(
            snapshot.validate(),
            Err(BackendAppliedActionEvidenceErrorV1::DigestMismatch)
        );
    }

    #[test]
    fn reset_clears_old_backend_action_evidence() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut simulator = InstrumentedSimpleHumanoidSimulator::new_for(morphology);
        simulator.step(&command_for_step(0, morphology), 0.025);
        assert!(simulator.backend_applied_action().is_some());
        simulator.reset();
        assert!(simulator.backend_applied_action().is_none());
        simulator.step(&command_for_step(1, morphology), 0.025);
        assert_eq!(simulator.backend_applied_action().unwrap().step_sequence, 1);
    }
}
