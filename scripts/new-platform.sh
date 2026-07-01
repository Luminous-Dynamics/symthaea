#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Generate a new Symthaea robotics platform crate.
#
# Usage: ./scripts/new-platform.sh <name> <num_actuators> <num_state_channels>
# Example: ./scripts/new-platform.sh marine-crawler 6 18
#
# This creates a complete, compilable platform crate at:
#   crates/symthaea-<name>/
#
# After generation:
# 1. Edit src/simulator.rs to implement your physics model
# 2. Edit src/encoder.rs CHANNEL_RANGES for your state space
# 3. Add feature flag + match arm to main crate (instructions printed)
# 4. cargo test -p symthaea-<name> --lib

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <name> <num_actuators> <num_state_channels>"
    echo "Example: $0 marine-crawler 6 18"
    exit 1
fi

NAME="$1"
NUM_ACTUATORS="$2"
NUM_STATE_CHANNELS="$3"

# Derive identifiers
CRATE_NAME="symthaea-${NAME}"
MOD_NAME="${NAME//-/_}"
STRUCT_PREFIX="$(echo "$MOD_NAME" | sed -r 's/(^|_)([a-z])/\U\2/g')" # PascalCase
CRATE_DIR="crates/${CRATE_NAME}"

if [ -d "$CRATE_DIR" ]; then
    echo "Error: $CRATE_DIR already exists"
    exit 1
fi

echo "Creating platform crate: ${CRATE_NAME}"
echo "  Actuators: ${NUM_ACTUATORS}"
echo "  State channels: ${NUM_STATE_CHANNELS}"
echo "  Struct prefix: ${STRUCT_PREFIX}"

mkdir -p "${CRATE_DIR}/src"

# ── Cargo.toml ──────────────────────────────────────────────────────────
cat > "${CRATE_DIR}/Cargo.toml" << TOML
[package]
name = "${CRATE_NAME}"
version = "0.1.0"
edition = "2021"
rust-version = "1.95"
authors = ["Luminous Dynamics <tristan.stoltz@evolvingresonantcocreationism.com>"]
description = "HDC-LTC + FEP Active Inference ${STRUCT_PREFIX} platform"
license = "AGPL-3.0-or-later"
repository = "https://github.com/Luminous-Dynamics/symthaea"

[dependencies]
symthaea-core = { path = "../../symthaea-core" }
symthaea-fep = { path = "../symthaea-fep" }
serde = { workspace = true }
serde_json = { workspace = true }
rand = { workspace = true }

[dev-dependencies]
proptest = "1"

[features]
default = []
TOML

# ── types.rs ────────────────────────────────────────────────────────────
cat > "${CRATE_DIR}/src/types.rs" << 'RUST'
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};

RUST

cat >> "${CRATE_DIR}/src/types.rs" << RUST
pub const NUM_ACTUATORS: usize = ${NUM_ACTUATORS};
pub const NUM_STATE_CHANNELS: usize = ${NUM_STATE_CHANNELS};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ${STRUCT_PREFIX}State {
    pub channels: [f64; NUM_STATE_CHANNELS],
}

impl ${STRUCT_PREFIX}State {
    pub fn home() -> Self {
        Self { channels: [0.0; NUM_STATE_CHANNELS] }
    }
    pub fn to_channels(&self) -> [f32; NUM_STATE_CHANNELS] {
        let mut c = [0.0f32; NUM_STATE_CHANNELS];
        for i in 0..NUM_STATE_CHANNELS { c[i] = self.channels[i] as f32; }
        c
    }
    pub fn is_finite(&self) -> bool {
        self.channels.iter().all(|v| v.is_finite())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ${STRUCT_PREFIX}Command {
    pub torques: [f32; NUM_ACTUATORS],
}

impl ${STRUCT_PREFIX}Command {
    pub fn zero() -> Self { Self { torques: [0.0; NUM_ACTUATORS] } }
    pub fn control_effort(&self) -> f32 {
        self.torques.iter().map(|t| t.abs()).sum::<f32>() / NUM_ACTUATORS as f32
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ${STRUCT_PREFIX}Config {
    pub genesis_phrase: String,
    pub learning_rate: f32,
    pub network_layers: usize,
    pub neurons_per_layer: usize,
    pub physics_hz: f64,
    pub cognitive_interval: usize,
    pub steps_per_episode: usize,
}

impl ${STRUCT_PREFIX}Config {
    pub fn physics_dt(&self) -> f64 { 1.0 / self.physics_hz }
}

impl Default for ${STRUCT_PREFIX}Config {
    fn default() -> Self {
        Self {
            genesis_phrase: "${CRATE_NAME}".to_string(),
            learning_rate: 0.001,
            network_layers: 3,
            neurons_per_layer: 8,
            physics_hz: 200.0,
            cognitive_interval: 10,
            steps_per_episode: 2000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_home() { assert!(${STRUCT_PREFIX}State::home().is_finite()); }
    #[test]
    fn test_channels() { assert_eq!(${STRUCT_PREFIX}State::home().to_channels().len(), NUM_STATE_CHANNELS); }
    #[test]
    fn test_zero_cmd() { assert_eq!(${STRUCT_PREFIX}Command::zero().control_effort(), 0.0); }
}
RUST

# ── encoder.rs ──────────────────────────────────────────────────────────
cat > "${CRATE_DIR}/src/encoder.rs" << RUST
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::{${STRUCT_PREFIX}State, NUM_STATE_CHANNELS};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

const DIM: usize = symthaea_core::hdc::HDC_DIMENSION;

pub struct ${STRUCT_PREFIX}HdcEncoder {
    base: Vec<ContinuousHV>,
    levels: Vec<ContinuousHV>,
    nl: usize,
}

impl ${STRUCT_PREFIX}HdcEncoder {
    pub fn new(g: &GenesisSeed, nl: usize) -> Self {
        Self {
            base: (0..NUM_STATE_CHANNELS).map(|i| ContinuousHV::from_genesis(g, &format!("${MOD_NAME}::ch::{i}"), DIM)).collect(),
            levels: (0..nl).map(|l| ContinuousHV::from_genesis(g, &format!("${MOD_NAME}::lv::{l}"), DIM)).collect(),
            nl,
        }
    }

    pub fn encode(&mut self, state: &${STRUCT_PREFIX}State) -> ContinuousHV {
        let ch = state.to_channels();
        let mut res = ContinuousHV::zero(DIM);
        for i in 0..NUM_STATE_CHANNELS {
            // Normalize to [0,1] — adjust ranges for your state space
            let n = ((ch[i] + 5.0) / 10.0).clamp(0.0, 1.0);
            let k = (n * (self.nl - 1) as f32).round() as usize;
            let k = k.min(self.nl - 1);
            let mut lhv = ContinuousHV::zero(DIM);
            for l in 0..=k { lhv.add_in_place(&self.levels[l]); }
            lhv = lhv.normalize();
            res.add_in_place(&lhv.bind(&self.base[i]));
        }
        res.normalize()
    }

    pub fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_encode_dim() {
        let mut enc = ${STRUCT_PREFIX}HdcEncoder::new(&GenesisSeed::from_phrase("test"), 32);
        assert_eq!(enc.encode(&${STRUCT_PREFIX}State::home()).dim(), DIM);
    }
}
RUST

# ── controller.rs ───────────────────────────────────────────────────────
cat > "${CRATE_DIR}/src/controller.rs" << RUST
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::{${STRUCT_PREFIX}Command, ${STRUCT_PREFIX}Config, NUM_ACTUATORS};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig};

const HDC_DIM: usize = symthaea_core::hdc::HDC_DIMENSION;

pub struct ${STRUCT_PREFIX}Controller {
    network: HdcLtcUnifiedNetwork,
    weights: Vec<f32>,
    bias: [f32; NUM_ACTUATORS],
}

impl ${STRUCT_PREFIX}Controller {
    pub fn new(g: &GenesisSeed, c: &${STRUCT_PREFIX}Config) -> Self {
        let nc = UnifiedConfig {
            tau_base: 1.0 / c.physics_hz as f32, backbone_tau: 0.3,
            dimension: HDC_DIM, learning_rate: c.learning_rate,
            ..UnifiedConfig::default()
        };
        let net = UnifiedNetworkConfig {
            layer_sizes: vec![c.neurons_per_layer; c.network_layers],
            neuron_config: nc, use_layer_binding: true, skip_connections: false,
        };
        let network = HdcLtcUnifiedNetwork::from_genesis(net, g);
        let wh = ContinuousHV::from_genesis(g, "${MOD_NAME}::out_w", NUM_ACTUATORS * HDC_DIM);
        let mut w: Vec<f32> = wh.as_slice().to_vec();
        for v in &mut w { *v *= 0.01; }
        Self { network, weights: w, bias: [0.0; NUM_ACTUATORS] }
    }

    pub fn forward(&mut self, hv: &ContinuousHV, dt: f32) -> ${STRUCT_PREFIX}Command {
        self.network.evolve_closed_form(dt, hv);
        let out = self.network.output().normalize();
        let d = out.as_slice();
        let mut t = [0.0f32; NUM_ACTUATORS];
        for o in 0..NUM_ACTUATORS {
            let off = o * HDC_DIM;
            let mut s = self.bias[o];
            for j in 0..HDC_DIM { s += self.weights[off + j] * d[j]; }
            t[o] = s.tanh();
        }
        ${STRUCT_PREFIX}Command { torques: t }
    }

    pub fn reset(&mut self) { self.network.reset(); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_fwd() {
        let mut c = ${STRUCT_PREFIX}Controller::new(&GenesisSeed::from_phrase("t"), &${STRUCT_PREFIX}Config::default());
        let cmd = c.forward(&ContinuousHV::random(HDC_DIM, 42), 0.005);
        assert!(cmd.torques.iter().all(|t| t.is_finite()));
    }
}
RUST

# ── simulator.rs ────────────────────────────────────────────────────────
cat > "${CRATE_DIR}/src/simulator.rs" << RUST
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Physics simulator — CUSTOMIZE THIS for your platform.
use crate::types::{${STRUCT_PREFIX}Command, ${STRUCT_PREFIX}State, NUM_ACTUATORS, NUM_STATE_CHANNELS};

pub trait ${STRUCT_PREFIX}PhysicsSimulator {
    fn step(&mut self, cmd: &${STRUCT_PREFIX}Command, dt: f64);
    fn state(&self) -> &${STRUCT_PREFIX}State;
    fn reset(&mut self);
}

/// Simple integrator: channels += torques * dt (placeholder dynamics).
/// Replace with your actual physics model.
pub struct Simple${STRUCT_PREFIX}Simulator {
    state: ${STRUCT_PREFIX}State,
    damping: f64,
}

impl Simple${STRUCT_PREFIX}Simulator {
    pub fn new() -> Self {
        Self { state: ${STRUCT_PREFIX}State::home(), damping: 2.0 }
    }
}

impl ${STRUCT_PREFIX}PhysicsSimulator for Simple${STRUCT_PREFIX}Simulator {
    fn step(&mut self, cmd: &${STRUCT_PREFIX}Command, dt: f64) {
        // Placeholder: first NUM_ACTUATORS channels are "joint angles"
        // driven by torques, with damping on "velocities" (if state is large enough).
        let n = NUM_ACTUATORS.min(NUM_STATE_CHANNELS);
        for i in 0..n {
            self.state.channels[i] += cmd.torques[i] as f64 * dt;
            self.state.channels[i] *= (-self.damping * dt).exp(); // damping
        }
    }
    fn state(&self) -> &${STRUCT_PREFIX}State { &self.state }
    fn reset(&mut self) { self.state = ${STRUCT_PREFIX}State::home(); }
}

impl Default for Simple${STRUCT_PREFIX}Simulator {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_stable() {
        let mut sim = Simple${STRUCT_PREFIX}Simulator::new();
        for _ in 0..1000 { sim.step(&${STRUCT_PREFIX}Command::zero(), 0.005); }
        assert!(sim.state().is_finite());
    }
    #[test]
    fn test_torque_moves() {
        let mut sim = Simple${STRUCT_PREFIX}Simulator::new();
        let mut cmd = ${STRUCT_PREFIX}Command::zero();
        cmd.torques[0] = 1.0;
        let before = sim.state().channels[0];
        sim.step(&cmd, 0.01);
        assert_ne!(sim.state().channels[0], before);
    }
}
RUST

# ── fep_agent.rs ────────────────────────────────────────────────────────
cat > "${CRATE_DIR}/src/fep_agent.rs" << RUST
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::${STRUCT_PREFIX}State;

pub struct ${STRUCT_PREFIX}FepResult { pub tau_factor: f32, pub free_energy: f64 }

pub struct ActiveInference${STRUCT_PREFIX}Agent {
    agent: symthaea_fep::ActiveInferenceAgent,
}

impl ActiveInference${STRUCT_PREFIX}Agent {
    pub fn new() -> Self {
        let config = symthaea_fep::ActiveInferenceAgentConfig {
            state_dim: ${NUM_ACTUATORS}, obs_dim: ${NUM_ACTUATORS}, num_actions: ${NUM_ACTUATORS},
            belief_learning_rate: 0.1, planning_horizon: 1, action_temperature: 0.5,
            ..symthaea_fep::ActiveInferenceAgentConfig::default()
        };
        Self { agent: symthaea_fep::ActiveInferenceAgent::new(config) }
    }
    pub fn tick(&mut self, state: &${STRUCT_PREFIX}State) -> ${STRUCT_PREFIX}FepResult {
        let deviation: f64 = state.channels.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
        let fe = deviation;
        let tau = if deviation > 2.0 { 0.85 } else { 1.0 };
        ${STRUCT_PREFIX}FepResult { tau_factor: tau, free_energy: fe }
    }
    pub fn reset(&mut self) { *self = Self::new(); }
}
impl Default for ActiveInference${STRUCT_PREFIX}Agent { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_fep_home() {
        let mut a = ActiveInference${STRUCT_PREFIX}Agent::new();
        let r = a.tick(&${STRUCT_PREFIX}State::home());
        assert!(r.free_energy.is_finite());
    }
}
RUST

# ── perturbations.rs ───────────────────────────────────────────────────
cat > "${CRATE_DIR}/src/perturbations.rs" << RUST
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ${STRUCT_PREFIX}Perturbation {
    ExternalForce { magnitude: f64, channel: usize, at_step: usize },
    ActuatorFailure { actuator: usize, at_step: usize },
}

#[derive(Debug, Clone, Default)]
pub struct PerturbationSchedule {
    pub perturbations: Vec<${STRUCT_PREFIX}Perturbation>,
}
RUST

# ── embodiment.rs ───────────────────────────────────────────────────────
cat > "${CRATE_DIR}/src/embodiment.rs" << RUST
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use crate::controller::${STRUCT_PREFIX}Controller;
use crate::encoder::${STRUCT_PREFIX}HdcEncoder;
use crate::simulator::{${STRUCT_PREFIX}PhysicsSimulator, Simple${STRUCT_PREFIX}Simulator};
use crate::types::${STRUCT_PREFIX}Config;

pub use symthaea_core::embodiment::{
    grounding_from_prediction_error, grounding_label, EmbodimentResult, EmbodimentTelemetry,
    MotorSafetyLevel, GROUNDING_SENSORIMOTOR,
};

pub struct ${STRUCT_PREFIX}Embodiment {
    controller: ${STRUCT_PREFIX}Controller,
    simulator: Simple${STRUCT_PREFIX}Simulator,
    encoder: ${STRUCT_PREFIX}HdcEncoder,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
}

impl ${STRUCT_PREFIX}Embodiment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = ${STRUCT_PREFIX}Config::default();
        Self {
            controller: ${STRUCT_PREFIX}Controller::new(genesis, &config),
            simulator: Simple${STRUCT_PREFIX}Simulator::new(),
            encoder: ${STRUCT_PREFIX}HdcEncoder::new(genesis, 32),
            last_perception: None, total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None, last_control_effort: 0.0, last_prediction_error: 0.0,
        }
    }

    pub fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        let phi_level = MotorSafetyLevel::from_phi(phi);
        self.current_safety = match self.safety_override {
            Some(o) => phi_level.max(o), None => phi_level,
        };
        let gain = self.current_safety.motor_gain();
        let mut cmd = self.controller.forward(thought_hv, dt);
        if gain < 1.0 { for t in &mut cmd.torques { *t *= gain; } }
        self.last_control_effort = cmd.control_effort();
        self.simulator.step(&cmd, dt as f64);
        let perception = self.encoder.encode(self.simulator.state());
        let pe = if let Some(ref prev) = self.last_perception {
            (1.0 - perception.similarity(prev).max(0.0)).min(1.0)
        } else { 0.0 };
        self.last_prediction_error = pe;
        self.last_perception = Some(perception);
        self.total_steps += 1;
        EmbodimentResult {
            num_actuators: ${NUM_ACTUATORS}, control_effort: self.last_control_effort,
            success: self.simulator.state().is_finite(), prediction_error: pe,
            safety_level: self.current_safety,
            epistemic_grounding: GROUNDING_SENSORIMOTOR,
            observation_confidence: grounding_from_prediction_error(pe),
        }
    }

    pub fn encode_perception(&mut self) -> ContinuousHV {
        let p = self.encoder.encode(self.simulator.state());
        self.last_perception = Some(p.clone()); p
    }
    pub fn reset(&mut self) {
        self.simulator.reset(); self.controller.reset(); self.encoder.reset();
        self.last_perception = None; self.total_steps = 0;
        self.current_safety = MotorSafetyLevel::Green;
        self.safety_override = None; self.last_control_effort = 0.0; self.last_prediction_error = 0.0;
    }
    pub fn safety_level(&self) -> MotorSafetyLevel { self.current_safety }
    pub fn set_safety_override(&mut self, level: MotorSafetyLevel) { self.safety_override = Some(level); }
    pub fn clear_safety_override(&mut self) { self.safety_override = None; }
    pub fn total_steps(&self) -> usize { self.total_steps }
    pub fn telemetry(&self) -> EmbodimentTelemetry {
        EmbodimentTelemetry {
            total_steps: self.total_steps as u64, control_effort: self.last_control_effort,
            prediction_error: self.last_prediction_error,
            safety_level: format!("{:?}", self.current_safety),
            platform: "${MOD_NAME}".to_string(), num_actuators: ${NUM_ACTUATORS},
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_step() {
        let mut e = ${STRUCT_PREFIX}Embodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.7);
        assert!(r.success);
    }
    #[test]
    fn test_red_halts() {
        let mut e = ${STRUCT_PREFIX}Embodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }
}
RUST

# ── training.rs ─────────────────────────────────────────────────────────
cat > "${CRATE_DIR}/src/training.rs" << RUST
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::controller::${STRUCT_PREFIX}Controller;
use crate::encoder::${STRUCT_PREFIX}HdcEncoder;
use crate::fep_agent::ActiveInference${STRUCT_PREFIX}Agent;
use crate::simulator::{${STRUCT_PREFIX}PhysicsSimulator, Simple${STRUCT_PREFIX}Simulator};
use crate::types::${STRUCT_PREFIX}Config;
use symthaea_core::genesis::GenesisSeed;

#[derive(Debug, Clone)]
pub struct EpisodeMetrics { pub mean_effort: f32, pub steps_survived: usize, pub diverged: bool }

pub struct ${STRUCT_PREFIX}Trainer {
    controller: ${STRUCT_PREFIX}Controller, encoder: ${STRUCT_PREFIX}HdcEncoder,
    fep: ActiveInference${STRUCT_PREFIX}Agent, simulator: Simple${STRUCT_PREFIX}Simulator,
    config: ${STRUCT_PREFIX}Config,
}

impl ${STRUCT_PREFIX}Trainer {
    pub fn new(config: ${STRUCT_PREFIX}Config) -> Self {
        let g = GenesisSeed::from_phrase(&config.genesis_phrase);
        Self {
            controller: ${STRUCT_PREFIX}Controller::new(&g, &config),
            encoder: ${STRUCT_PREFIX}HdcEncoder::new(&g, 32),
            fep: ActiveInference${STRUCT_PREFIX}Agent::new(),
            simulator: Simple${STRUCT_PREFIX}Simulator::new(), config,
        }
    }
    pub fn run_episode(&mut self) -> EpisodeMetrics {
        self.simulator.reset(); self.controller.reset();
        let dt = self.config.physics_dt(); let mut total_e = 0.0f32;
        for step in 0..self.config.steps_per_episode {
            let hv = self.encoder.encode(self.simulator.state());
            if step % self.config.cognitive_interval == 0 { let _ = self.fep.tick(self.simulator.state()); }
            let cmd = self.controller.forward(&hv, dt as f32);
            self.simulator.step(&cmd, dt); total_e += cmd.control_effort();
            if !self.simulator.state().is_finite() {
                return EpisodeMetrics { mean_effort: total_e / (step+1) as f32, steps_survived: step, diverged: true };
            }
        }
        EpisodeMetrics { mean_effort: total_e / self.config.steps_per_episode as f32,
            steps_survived: self.config.steps_per_episode, diverged: false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_episode() {
        let mut c = ${STRUCT_PREFIX}Config::default(); c.steps_per_episode = 200;
        let mut t = ${STRUCT_PREFIX}Trainer::new(c);
        let m = t.run_episode();
        assert!(!m.diverged); assert_eq!(m.steps_survived, 200);
    }
}
RUST

# ── lib.rs ──────────────────────────────────────────────────────────────
cat > "${CRATE_DIR}/src/lib.rs" << RUST
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # ${CRATE_NAME}
//!
//! Consciousness-coupled ${STRUCT_PREFIX} platform.
//! Generated by scripts/new-platform.sh — customize simulator.rs for your physics.

pub mod controller;
pub mod embodiment;
pub mod encoder;
pub mod fep_agent;
pub mod perturbations;
pub mod simulator;
pub mod training;
pub mod types;

pub use types::*;
RUST

echo ""
echo "✓ Created ${CRATE_DIR}/ with 8 source files"
echo ""
echo "Next steps:"
echo "  1. Add to workspace: edit Cargo.toml [workspace] members list"
echo "  2. Add feature flag:"
echo "     ${MOD_NAME} = [\"dep:${CRATE_NAME}\"]"
echo "  3. Add optional dependency:"
echo "     ${CRATE_NAME} = { path = \"crates/${CRATE_NAME}\", optional = true }"
echo "  4. Add EmbodimentPlatform variant in symthaea-core/src/embodiment.rs"
echo "  5. Add match arm in src/cognitive_loop/constructor.rs"
echo "  6. Customize src/simulator.rs with your physics model"
echo "  7. Test: cargo test -p ${CRATE_NAME} --lib"
