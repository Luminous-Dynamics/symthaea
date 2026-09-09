#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Generate a lean first-party Symthaea robotics platform crate.
#
# Usage: ./scripts/new-platform.sh <name> <num_actuators> <num_state_channels>
# Example: ./scripts/new-platform.sh marine-crawler 6 18
#
# The generated crate is intentionally conservative:
# - it lives under crates/domains/, which is already a workspace glob;
# - its controller emits zero command until platform-specific control is implemented;
# - inter-frame HDC change is named temporal novelty, not prediction error;
# - legacy prediction/confidence fields use restrictive "not qualified" sentinels;
# - its placeholder simulator owns model dt independently of legacy bridge dt;
# - it implements the current EmbodimentBridge + PlatformPlugin shape;
# - it does NOT invent a hardware-safe fallback for an unknown morphology.
#
# Generated code is a development/simulation scaffold, not hardware qualification.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/new-platform.sh <name> <num_actuators> <num_state_channels>
Example: ./scripts/new-platform.sh marine-crawler 6 18

name               lowercase kebab-case, e.g. marine-crawler
num_actuators      positive integer
num_state_channels positive integer
EOF
}

if [ "$#" -ne 3 ]; then
    usage
    exit 1
fi

NAME="$1"
NUM_ACTUATORS="$2"
NUM_STATE_CHANNELS="$3"

if ! [[ "$NAME" =~ ^[a-z][a-z0-9]*(-[a-z0-9]+)*$ ]]; then
    echo "Error: name must be lowercase kebab-case (example: marine-crawler)" >&2
    exit 1
fi
if ! [[ "$NUM_ACTUATORS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: num_actuators must be a positive integer" >&2
    exit 1
fi
if ! [[ "$NUM_STATE_CHANNELS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: num_state_channels must be a positive integer" >&2
    exit 1
fi

CRATE_NAME="symthaea-${NAME}"
MOD_NAME="${NAME//-/_}"
STRUCT_PREFIX=""
IFS='-' read -r -a NAME_PARTS <<< "$NAME"
for part in "${NAME_PARTS[@]}"; do
    STRUCT_PREFIX+="${part^}"
done
CRATE_DIR="crates/domains/${CRATE_NAME}"

if [ -e "$CRATE_DIR" ]; then
    echo "Error: $CRATE_DIR already exists" >&2
    exit 1
fi

cleanup_on_error() {
    local status=$?
    if [ "$status" -ne 0 ] && [ -d "$CRATE_DIR" ]; then
        rm -rf "$CRATE_DIR"
        echo "Generation failed; removed partial directory $CRATE_DIR" >&2
    fi
    exit "$status"
}
trap cleanup_on_error EXIT

printf 'Creating platform crate: %s\n' "$CRATE_NAME"
printf '  Path:           %s\n' "$CRATE_DIR"
printf '  Actuators:      %s\n' "$NUM_ACTUATORS"
printf '  State channels: %s\n' "$NUM_STATE_CHANNELS"
printf '  Rust prefix:    %s\n' "$STRUCT_PREFIX"

mkdir -p "$CRATE_DIR/src"

# ── Cargo.toml ────────────────────────────────────────────────────────────────
cat > "$CRATE_DIR/Cargo.toml" <<TOML
[package]
name = "$CRATE_NAME"
version = "0.1.0"
edition.workspace = true
license.workspace = true
authors = ["Luminous Dynamics <tristan.stoltz@evolvingresonantcocreationism.com>"]
description = "Conservative Symthaea $STRUCT_PREFIX embodiment scaffold"
repository = "https://github.com/Luminous-Dynamics/symthaea"

[dependencies]
serde = { workspace = true, features = ["derive"] }
symthaea-core = { path = "../../core/symthaea-core" }

[features]
default = []
TOML

# ── types.rs ─────────────────────────────────────────────────────────────────
cat > "$CRATE_DIR/src/types.rs" <<RUST
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};

pub const NUM_ACTUATORS: usize = $NUM_ACTUATORS;
pub const NUM_STATE_CHANNELS: usize = $NUM_STATE_CHANNELS;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ${STRUCT_PREFIX}State {
    /// Placeholder state channels. Replace indices with explicit physical semantics.
    pub channels: Vec<f64>,
}

impl ${STRUCT_PREFIX}State {
    pub fn home() -> Self {
        Self {
            channels: vec![0.0; NUM_STATE_CHANNELS],
        }
    }

    pub fn is_valid(&self) -> bool {
        self.channels.len() == NUM_STATE_CHANNELS
            && self.channels.iter().all(|value| value.is_finite())
    }

    pub fn to_f32_channels(&self) -> Vec<f32> {
        self.channels.iter().map(|value| *value as f32).collect()
    }
}

impl Default for ${STRUCT_PREFIX}State {
    fn default() -> Self {
        Self::home()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ${STRUCT_PREFIX}Command {
    /// Placeholder actuator command vector. Define physical units before hardware use.
    pub torques: Vec<f32>,
}

impl ${STRUCT_PREFIX}Command {
    pub fn zero() -> Self {
        Self {
            torques: vec![0.0; NUM_ACTUATORS],
        }
    }

    pub fn is_valid(&self) -> bool {
        self.torques.len() == NUM_ACTUATORS
            && self.torques.iter().all(|value| value.is_finite())
    }

    pub fn control_effort(&self) -> f32 {
        self.torques.iter().map(|value| value.abs()).sum::<f32>() / NUM_ACTUATORS as f32
    }
}

impl Default for ${STRUCT_PREFIX}Command {
    fn default() -> Self {
        Self::zero()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ${STRUCT_PREFIX}Config {
    /// Placeholder fixed simulation rate. This is simulation model time, not cognitive tau.
    pub physics_hz: f64,
}

impl ${STRUCT_PREFIX}Config {
    pub fn physics_dt(&self) -> f64 {
        1.0 / self.physics_hz
    }

    pub fn is_valid(&self) -> bool {
        self.physics_hz.is_finite() && self.physics_hz > 0.0
    }
}

impl Default for ${STRUCT_PREFIX}Config {
    fn default() -> Self {
        Self { physics_hz: 200.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn home_and_zero_have_declared_cardinality() {
        let state = ${STRUCT_PREFIX}State::home();
        let command = ${STRUCT_PREFIX}Command::zero();
        assert!(state.is_valid());
        assert!(command.is_valid());
        assert_eq!(state.channels.len(), NUM_STATE_CHANNELS);
        assert_eq!(command.torques.len(), NUM_ACTUATORS);
        assert_eq!(command.control_effort(), 0.0);
    }

    #[test]
    fn default_config_has_positive_finite_dt() {
        let config = ${STRUCT_PREFIX}Config::default();
        assert!(config.is_valid());
        assert!(config.physics_dt().is_finite());
        assert!(config.physics_dt() > 0.0);
    }
}
RUST

# ── encoder.rs ───────────────────────────────────────────────────────────────
cat > "$CRATE_DIR/src/encoder.rs" <<RUST
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Placeholder state encoder.
//!
//! Before qualification, replace anonymous channel positions/ranges with the
//! Universal Embodiment semantic address + contract profile used by this body.

use crate::types::{${STRUCT_PREFIX}State, NUM_STATE_CHANNELS};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

pub struct ${STRUCT_PREFIX}HdcEncoder {
    role_bases: Vec<ContinuousHV>,
    levels: Vec<ContinuousHV>,
}

impl ${STRUCT_PREFIX}HdcEncoder {
    pub fn new(genesis: &GenesisSeed, requested_levels: usize) -> Self {
        let level_count = requested_levels.max(2);
        Self {
            role_bases: (0..NUM_STATE_CHANNELS)
                .map(|index| genesis.hv(&format!("$MOD_NAME::state_role::{index}"), HDC_DIMENSION))
                .collect(),
            levels: (0..level_count)
                .map(|index| genesis.hv(&format!("$MOD_NAME::state_level::{index}"), HDC_DIMENSION))
                .collect(),
        }
    }

    pub fn encode(&self, state: &${STRUCT_PREFIX}State) -> ContinuousHV {
        assert!(state.is_valid(), "state must satisfy generated scaffold cardinality");
        let channels = state.to_f32_channels();
        let last_level = self.levels.len() - 1;
        let mut result = ContinuousHV::zero(HDC_DIMENSION);

        for (index, value) in channels.iter().copied().enumerate() {
            // DEVELOPMENT PLACEHOLDER ONLY: anonymous channels use [-5, 5].
            // Replace this with per-role physical contracts before evidence claims.
            let normalized = ((value + 5.0) / 10.0).clamp(0.0, 1.0);
            let level_index = (normalized * last_level as f32).round() as usize;
            let bound = self.role_bases[index].bind(&self.levels[level_index.min(last_level)]);
            result.add_in_place(&bound);
        }

        result.normalize()
    }

    pub fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_is_deterministic_and_has_core_dimension() {
        let genesis = GenesisSeed::from_phrase("generated-platform-test");
        let encoder = ${STRUCT_PREFIX}HdcEncoder::new(&genesis, 32);
        let state = ${STRUCT_PREFIX}State::home();
        let a = encoder.encode(&state);
        let b = encoder.encode(&state);
        assert_eq!(a.dim(), HDC_DIMENSION);
        assert_eq!(a.values, b.values);
    }
}
RUST

# ── controller.rs ────────────────────────────────────────────────────────────
cat > "$CRATE_DIR/src/controller.rs" <<RUST
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Conservative generated controller.
//!
//! A newly scaffolded morphology has no qualified action semantics or controller.
//! Emitting random learned weights would turn code generation into implicit actuator
//! authority, so this controller deliberately emits a zero command until replaced.

use crate::types::${STRUCT_PREFIX}Command;
use symthaea_core::hdc::ContinuousHV;

#[derive(Debug, Clone, Default)]
pub struct ${STRUCT_PREFIX}Controller;

impl ${STRUCT_PREFIX}Controller {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&mut self, _thought_hv: &ContinuousHV, _dt: f32) -> ${STRUCT_PREFIX}Command {
        ${STRUCT_PREFIX}Command::zero()
    }

    pub fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_controller_has_no_implicit_actuation() {
        let mut controller = ${STRUCT_PREFIX}Controller::new();
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        assert_eq!(controller.forward(&thought, 0.005), ${STRUCT_PREFIX}Command::zero());
    }
}
RUST

# ── simulator.rs ─────────────────────────────────────────────────────────────
cat > "$CRATE_DIR/src/simulator.rs" <<RUST
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Development-only placeholder plant.
//!
//! Replace this with a validated platform model/backend. The first actuator channels
//! are mapped directly into the first state channels solely so tests can establish
//! that the scaffold's state/action plumbing is live.

use crate::types::{
    ${STRUCT_PREFIX}Command, ${STRUCT_PREFIX}State, NUM_ACTUATORS, NUM_STATE_CHANNELS,
};

pub trait ${STRUCT_PREFIX}PhysicsSimulator: Send + Sync {
    fn step(&mut self, command: &${STRUCT_PREFIX}Command, dt: f64);
    fn state(&self) -> &${STRUCT_PREFIX}State;
    fn reset(&mut self);
}

#[derive(Debug, Clone)]
pub struct Simple${STRUCT_PREFIX}Simulator {
    state: ${STRUCT_PREFIX}State,
    damping: f64,
}

impl Simple${STRUCT_PREFIX}Simulator {
    pub fn new() -> Self {
        Self {
            state: ${STRUCT_PREFIX}State::home(),
            damping: 2.0,
        }
    }
}

impl Default for Simple${STRUCT_PREFIX}Simulator {
    fn default() -> Self {
        Self::new()
    }
}

impl ${STRUCT_PREFIX}PhysicsSimulator for Simple${STRUCT_PREFIX}Simulator {
    fn step(&mut self, command: &${STRUCT_PREFIX}Command, dt: f64) {
        if !command.is_valid() || !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let count = NUM_ACTUATORS.min(NUM_STATE_CHANNELS);
        for index in 0..count {
            self.state.channels[index] += command.torques[index] as f64 * dt;
            self.state.channels[index] *= (-self.damping * dt).exp();
        }
    }

    fn state(&self) -> &${STRUCT_PREFIX}State {
        &self.state
    }

    fn reset(&mut self) {
        self.state = ${STRUCT_PREFIX}State::home();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_command_is_stable() {
        let mut simulator = Simple${STRUCT_PREFIX}Simulator::new();
        for _ in 0..1000 {
            simulator.step(&${STRUCT_PREFIX}Command::zero(), 0.005);
        }
        assert!(simulator.state().is_valid());
        assert_eq!(simulator.state(), &${STRUCT_PREFIX}State::home());
    }

    #[test]
    fn explicit_nonzero_fixture_moves_placeholder_state() {
        let mut simulator = Simple${STRUCT_PREFIX}Simulator::new();
        let mut command = ${STRUCT_PREFIX}Command::zero();
        command.torques[0] = 1.0;
        let before = simulator.state().channels[0];
        simulator.step(&command, 0.01);
        assert_ne!(simulator.state().channels[0], before);
    }
}
RUST

# ── embodiment.rs ────────────────────────────────────────────────────────────
cat > "$CRATE_DIR/src/embodiment.rs" <<RUST
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generated development embodiment.
//!
//! IMPORTANT: this file deliberately does not implement `SafeFallback`. Symthaea
//! cannot know whether zero torque means safe hold, free-fall, loss of braking, or
//! some other dangerous state for a new morphology. Define and test a platform-
//! specific minimum-safe behavior before any hardware qualification.

use crate::controller::${STRUCT_PREFIX}Controller;
use crate::encoder::${STRUCT_PREFIX}HdcEncoder;
use crate::simulator::{${STRUCT_PREFIX}PhysicsSimulator, Simple${STRUCT_PREFIX}Simulator};
use crate::types::{${STRUCT_PREFIX}Config, NUM_ACTUATORS};
use symthaea_core::embodiment::{
    AgentIdentity, EmbodimentBridge, EmbodimentPlatform, EmbodimentResult, EmbodimentTelemetry,
    MotorSafetyLevel, GROUNDING_SENSORIMOTOR,
};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

/// Legacy `EmbodimentResult::prediction_error` sentinel used when no prediction exists.
///
/// This is not a measured 100% model error. New v2 evidence must report prediction
/// availability explicitly; see #1281/#1297.
const LEGACY_NO_PREDICTION_SENTINEL: f32 = 1.0;
/// No observation-quality estimator exists in the generated scaffold.
const LEGACY_NO_OBSERVATION_CONFIDENCE: f32 = 0.0;

pub struct ${STRUCT_PREFIX}Embodiment {
    controller: ${STRUCT_PREFIX}Controller,
    simulator: Simple${STRUCT_PREFIX}Simulator,
    encoder: ${STRUCT_PREFIX}HdcEncoder,
    simulation_dt: f64,
    last_perception: Option<ContinuousHV>,
    last_temporal_novelty: Option<f32>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    agent_identity: AgentIdentity,
}

impl ${STRUCT_PREFIX}Embodiment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = ${STRUCT_PREFIX}Config::default();
        debug_assert!(config.is_valid());
        Self {
            controller: ${STRUCT_PREFIX}Controller::new(),
            simulator: Simple${STRUCT_PREFIX}Simulator::new(),
            encoder: ${STRUCT_PREFIX}HdcEncoder::new(genesis, 32),
            simulation_dt: config.physics_dt(),
            last_perception: None,
            last_temporal_novelty: None,
            total_steps: 0,
            // Before the first evaluated step, expose the most restrictive legacy tier.
            current_safety: MotorSafetyLevel::Red,
            safety_override: None,
            last_control_effort: 0.0,
            // Deterministic local identity only. This is not cryptographic authentication.
            agent_identity: AgentIdentity::new(format!(
                "$MOD_NAME:genesis:{}",
                genesis.timeline_id()
            )),
        }
    }

    pub fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        let phi_level = MotorSafetyLevel::from_phi(phi);
        self.current_safety = self
            .safety_override
            .map_or(phi_level, |override_level| phi_level.max(override_level));

        // `dt` is the legacy bridge/controller integration input. The generated
        // simulator owns independent model time so cognitive/substrate dt cannot
        // silently redefine physical seconds (#1284).
        let valid_controller_dt = dt.is_finite() && dt > 0.0;
        let mut command = self.controller.forward(thought_hv, dt);
        let gain = self.current_safety.motor_gain();
        for torque in &mut command.torques {
            *torque *= gain;
        }
        self.last_control_effort = command.control_effort();

        if valid_controller_dt {
            self.simulator.step(&command, self.simulation_dt);
        }

        let perception = self.encoder.encode(self.simulator.state());
        self.last_temporal_novelty = self.last_perception.as_ref().map(|previous| {
            (1.0 - perception.similarity(previous).max(0.0)).clamp(0.0, 1.0)
        });
        self.last_perception = Some(perception);
        self.total_steps = self.total_steps.saturating_add(1);

        EmbodimentResult {
            num_actuators: NUM_ACTUATORS,
            control_effort: self.last_control_effort,
            success: valid_controller_dt && command.is_valid() && self.simulator.state().is_valid(),
            // No predictive model exists. Never put temporal novelty in this field.
            prediction_error: LEGACY_NO_PREDICTION_SENTINEL,
            safety_level: self.current_safety,
            epistemic_grounding: GROUNDING_SENSORIMOTOR,
            // Motion/novelty is not observation quality. No estimator => conservative legacy value.
            observation_confidence: LEGACY_NO_OBSERVATION_CONFIDENCE,
        }
    }

    pub fn encode_perception(&mut self) -> ContinuousHV {
        let perception = self.encoder.encode(self.simulator.state());
        self.last_perception = Some(perception.clone());
        perception
    }

    /// Fixed placeholder simulator model timestep, independent of bridge/controller dt.
    pub fn simulation_dt(&self) -> f64 {
        self.simulation_dt
    }

    /// Inter-frame HDC change. This is novelty, not predicted-vs-observed residual.
    pub fn temporal_state_novelty(&self) -> Option<f32> {
        self.last_temporal_novelty
    }

    pub fn reset(&mut self) {
        self.simulator.reset();
        self.controller.reset();
        self.encoder.reset();
        self.last_perception = None;
        self.last_temporal_novelty = None;
        self.total_steps = 0;
        self.current_safety = MotorSafetyLevel::Red;
        self.safety_override = None;
        self.last_control_effort = 0.0;
    }

    pub fn telemetry(&self) -> EmbodimentTelemetry {
        EmbodimentTelemetry {
            total_steps: self.total_steps as u64,
            control_effort: self.last_control_effort,
            prediction_error: LEGACY_NO_PREDICTION_SENTINEL,
            safety_level: self.current_safety,
            platform: "$MOD_NAME".to_string(),
            num_actuators: NUM_ACTUATORS,
            epistemic_grounding: "Sensorimotor".to_string(),
            observation_confidence: LEGACY_NO_OBSERVATION_CONFIDENCE,
            platform_specific: Vec::new(),
        }
    }
}

impl EmbodimentBridge for ${STRUCT_PREFIX}Embodiment {
    fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        ${STRUCT_PREFIX}Embodiment::step(self, thought_hv, dt, phi)
    }

    fn encode_perception(&mut self) -> ContinuousHV {
        ${STRUCT_PREFIX}Embodiment::encode_perception(self)
    }

    fn reset(&mut self) {
        ${STRUCT_PREFIX}Embodiment::reset(self)
    }

    fn safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }

    fn set_safety_override(&mut self, level: MotorSafetyLevel) {
        self.safety_override = Some(level);
    }

    fn clear_safety_override(&mut self) {
        self.safety_override = None;
    }

    fn platform(&self) -> EmbodimentPlatform {
        EmbodimentPlatform::${STRUCT_PREFIX}
    }

    fn num_actuators(&self) -> usize {
        NUM_ACTUATORS
    }

    fn total_steps(&self) -> usize {
        self.total_steps
    }

    fn telemetry(&self) -> EmbodimentTelemetry {
        ${STRUCT_PREFIX}Embodiment::telemetry(self)
    }

    fn last_perception_hv(&self) -> Option<ContinuousHV> {
        self.last_perception.clone()
    }

    fn agent_identity(&self) -> AgentIdentity {
        self.agent_identity.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_body_starts_restrictive_and_has_established_local_identity() {
        let body = ${STRUCT_PREFIX}Embodiment::new(&GenesisSeed::from_phrase("test"));
        assert_eq!(body.current_safety, MotorSafetyLevel::Red);
        assert_ne!(body.agent_identity.as_str(), "unset");
    }

    #[test]
    fn scaffold_does_not_claim_prediction_or_observation_confidence() {
        let mut body = ${STRUCT_PREFIX}Embodiment::new(&GenesisSeed::from_phrase("test"));
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let result = body.step(&thought, 0.005, 0.7);
        assert!(result.success);
        assert_eq!(result.prediction_error, LEGACY_NO_PREDICTION_SENTINEL);
        assert_eq!(
            result.observation_confidence,
            LEGACY_NO_OBSERVATION_CONFIDENCE
        );
        assert_eq!(result.epistemic_grounding, GROUNDING_SENSORIMOTOR);
    }

    #[test]
    fn temporal_change_is_kept_out_of_prediction_error() {
        let mut body = ${STRUCT_PREFIX}Embodiment::new(&GenesisSeed::from_phrase("test"));
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        body.step(&thought, 0.005, 0.7);
        body.step(&thought, 0.010, 0.7);
        assert!(body.temporal_state_novelty().is_some());
        assert_eq!(body.telemetry().prediction_error, LEGACY_NO_PREDICTION_SENTINEL);
        assert!((body.simulation_dt() - 0.005).abs() < f64::EPSILON);
    }

    #[test]
    fn invalid_controller_dt_fails_step_without_advancing_placeholder_physics() {
        let mut body = ${STRUCT_PREFIX}Embodiment::new(&GenesisSeed::from_phrase("test"));
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let result = body.step(&thought, f32::NAN, 0.7);
        assert!(!result.success);
        assert!(body.simulator.state().is_valid());
    }
}
RUST

# ── plugin.rs ────────────────────────────────────────────────────────────────
cat > "$CRATE_DIR/src/plugin.rs" <<RUST
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! First-party compile-time platform registration.
//!
//! This plugin proves construction wiring only. It is not hardware qualification.

use crate::embodiment::${STRUCT_PREFIX}Embodiment;
use crate::types::NUM_ACTUATORS;
use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentPlatform, PlatformPlugin};
use symthaea_core::genesis::GenesisSeed;

pub struct ${STRUCT_PREFIX}Plugin;

impl PlatformPlugin for ${STRUCT_PREFIX}Plugin {
    fn platform(&self) -> EmbodimentPlatform {
        EmbodimentPlatform::${STRUCT_PREFIX}
    }

    fn feature_name(&self) -> &'static str {
        "$NAME"
    }

    fn num_actuators(&self) -> usize {
        NUM_ACTUATORS
    }

    fn create_bridge(&self, genesis: &GenesisSeed) -> Box<dyn EmbodimentBridge> {
        Box::new(${STRUCT_PREFIX}Embodiment::new(genesis))
    }
}
RUST

# ── lib.rs ───────────────────────────────────────────────────────────────────
cat > "$CRATE_DIR/src/lib.rs" <<RUST
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # $CRATE_NAME
//!
//! Conservative generated development scaffold for the $STRUCT_PREFIX embodiment.
//!
//! Before hardware qualification, define explicit physical state/action semantics,
//! platform-specific control, a tested `SafeFallback`, calibration, timing, health,
//! and evidence contracts. The generated zero-output controller is intentional.

pub mod controller;
pub mod embodiment;
pub mod encoder;
pub mod plugin;
pub mod simulator;
pub mod types;

pub use types::*;
RUST

trap - EXIT

echo
echo "✓ Created $CRATE_DIR/ with 7 source files"
echo
echo "Required first-party integration steps:"
echo "  1. Add EmbodimentPlatform::$STRUCT_PREFIX in crates/core/symthaea-core/src/embodiment.rs"
echo "  2. Add root optional dependency:"
echo "     $CRATE_NAME = { path = \"crates/domains/$CRATE_NAME\", optional = true }"
echo "  3. Add root feature flag:"
echo "     $NAME = [\"dep:$CRATE_NAME\"]"
echo "  4. Register the plugin in src/cognitive_loop/platform_registry.rs:"
echo "     #[cfg(feature = \"$NAME\")]"
echo "     registry.register(Box::new(${CRATE_NAME//-/_}::plugin::${STRUCT_PREFIX}Plugin));"
echo
echo "No workspace-members edit is needed: crates/domains/* is already a workspace glob."
echo
echo "Before hardware qualification:"
echo "  • replace anonymous state/action channels with typed physical semantics"
echo "  • replace the zero-output controller with a qualified controller"
echo "  • replace the placeholder simulator/backend"
echo "  • implement and test a platform-specific SafeFallback"
echo "  • add calibration, clock/time, health, authority, and evidence profiles"
echo "  • do not reinterpret temporal_state_novelty() as prediction error"
echo
echo "Suggested first gate after integration:"
echo "  cargo test -p $CRATE_NAME --lib -- --test-threads=1"