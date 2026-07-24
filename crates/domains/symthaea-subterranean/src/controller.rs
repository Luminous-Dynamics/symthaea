// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::{ConfigError, NUM_ACTUATORS, SubterraneanCommand, SubterraneanConfig};
use serde::{Deserialize, Serialize};
use std::fmt;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig};

const HDC_DIM: usize = symthaea_core::hdc::HDC_DIMENSION;

const CHECKPOINT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerCheckpoint {
    pub schema_version: u32,
    pub hdc_dimension: usize,
    pub num_actuators: usize,
    pub weights: Vec<f32>,
    pub bias: [f32; NUM_ACTUATORS],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointError {
    UnsupportedSchema { found: u32, expected: u32 },
    DimensionMismatch { found: usize, expected: usize },
    ActuatorMismatch { found: usize, expected: usize },
    WeightLengthMismatch { found: usize, expected: usize },
    NonFiniteParameter,
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchema { found, expected } => {
                write!(
                    f,
                    "unsupported checkpoint schema {found}; expected {expected}"
                )
            }
            Self::DimensionMismatch { found, expected } => {
                write!(f, "checkpoint HDC dimension {found}; expected {expected}")
            }
            Self::ActuatorMismatch { found, expected } => {
                write!(f, "checkpoint actuator count {found}; expected {expected}")
            }
            Self::WeightLengthMismatch { found, expected } => {
                write!(f, "checkpoint weight length {found}; expected {expected}")
            }
            Self::NonFiniteParameter => write!(f, "checkpoint contains non-finite parameters"),
        }
    }
}

impl std::error::Error for CheckpointError {}

pub struct SubterraneanController {
    network: HdcLtcUnifiedNetwork,
    weights: Vec<f32>,
    bias: [f32; NUM_ACTUATORS],
    learning_rate: f32,
    /// Cached final-layer HV from the last forward() (post-normalize) --
    /// needed by train_step's delta rule.
    last_features: Vec<f32>,
    /// Cached post-tanh outputs from the last forward().
    last_outputs: [f32; NUM_ACTUATORS],
}

impl SubterraneanController {
    pub fn try_new(
        genesis: &GenesisSeed,
        config: &SubterraneanConfig,
    ) -> Result<Self, ConfigError> {
        config.validate()?;
        Ok(Self::new(genesis, config))
    }

    pub fn new(g: &GenesisSeed, c: &SubterraneanConfig) -> Self {
        let nc = UnifiedConfig {
            tau_base: 1.0 / c.physics_hz as f32,
            backbone_tau: 0.3,
            dimension: HDC_DIM,
            learning_rate: c.learning_rate,
            ..UnifiedConfig::default()
        };
        let net = UnifiedNetworkConfig {
            layer_sizes: vec![c.neurons_per_layer; c.network_layers],
            neuron_config: nc,
            use_layer_binding: true,
            skip_connections: false,
        };
        let network = HdcLtcUnifiedNetwork::from_genesis(net, g);
        let wh = ContinuousHV::from_genesis(g, "subterranean::out_w", NUM_ACTUATORS * HDC_DIM);
        let mut w: Vec<f32> = wh.as_slice().to_vec();
        for v in &mut w {
            *v *= 0.01;
        }
        Self {
            network,
            weights: w,
            bias: [0.0; NUM_ACTUATORS],
            learning_rate: c.learning_rate,
            last_features: Vec::new(),
            last_outputs: [0.0; NUM_ACTUATORS],
        }
    }

    pub fn forward(&mut self, hv: &ContinuousHV, dt: f32) -> SubterraneanCommand {
        self.network.evolve_closed_form(dt, hv);
        let out = self.network.output().normalize();
        let d = out.as_slice();
        let mut t = [0.0f32; NUM_ACTUATORS];
        for o in 0..NUM_ACTUATORS {
            let off = o * HDC_DIM;
            let mut s = self.bias[o];
            for j in 0..HDC_DIM {
                s += self.weights[off + j] * d[j];
            }
            t[o] = s.tanh();
        }
        self.last_features = d.to_vec();
        self.last_outputs = t;
        SubterraneanCommand {
            torques: t,
            recovery: crate::types::RecoveryCommand::zero(),
        }
    }

    /// One supervised update of the output projection toward `target`
    /// (delta rule through the tanh), using the features cached by the last
    /// `forward()`. Returns the pre-update mean-squared error. This is what
    /// makes `SubterraneanTrainer` actually train -- previously the trainer
    /// collected metrics and never touched a weight (Tier 2 of
    /// SYMTHAEA_UNAUDITED_PLATFORMS_REVIEW_2026-07-07.md).
    pub fn train_step(&mut self, target: &SubterraneanCommand) -> f32 {
        if self.last_features.is_empty() {
            return 0.0;
        }
        let mut mse = 0.0f32;
        for o in 0..NUM_ACTUATORS {
            let out = self.last_outputs[o];
            let err = target.torques[o] - out;
            mse += err * err;
            // Backprop through tanh: d(out)/d(pre) = 1 - out²
            let delta = self.learning_rate * err * (1.0 - out * out);
            let off = o * HDC_DIM;
            for (j, f) in self.last_features.iter().enumerate() {
                self.weights[off + j] += delta * f;
            }
            self.bias[o] += delta;
        }
        mse / NUM_ACTUATORS as f32
    }

    pub fn checkpoint(&self) -> ControllerCheckpoint {
        ControllerCheckpoint {
            schema_version: CHECKPOINT_SCHEMA_VERSION,
            hdc_dimension: HDC_DIM,
            num_actuators: NUM_ACTUATORS,
            weights: self.weights.clone(),
            bias: self.bias,
        }
    }

    pub fn load_checkpoint(
        &mut self,
        checkpoint: &ControllerCheckpoint,
    ) -> Result<(), CheckpointError> {
        if checkpoint.schema_version != CHECKPOINT_SCHEMA_VERSION {
            return Err(CheckpointError::UnsupportedSchema {
                found: checkpoint.schema_version,
                expected: CHECKPOINT_SCHEMA_VERSION,
            });
        }
        if checkpoint.hdc_dimension != HDC_DIM {
            return Err(CheckpointError::DimensionMismatch {
                found: checkpoint.hdc_dimension,
                expected: HDC_DIM,
            });
        }
        if checkpoint.num_actuators != NUM_ACTUATORS {
            return Err(CheckpointError::ActuatorMismatch {
                found: checkpoint.num_actuators,
                expected: NUM_ACTUATORS,
            });
        }
        let expected_weights = NUM_ACTUATORS * HDC_DIM;
        if checkpoint.weights.len() != expected_weights {
            return Err(CheckpointError::WeightLengthMismatch {
                found: checkpoint.weights.len(),
                expected: expected_weights,
            });
        }
        if checkpoint.weights.iter().any(|value| !value.is_finite())
            || checkpoint.bias.iter().any(|value| !value.is_finite())
        {
            return Err(CheckpointError::NonFiniteParameter);
        }

        self.weights.clone_from(&checkpoint.weights);
        self.bias = checkpoint.bias;
        self.reset();
        Ok(())
    }

    pub fn reset(&mut self) {
        self.network.reset();
        self.last_features.clear();
        self.last_outputs = [0.0; NUM_ACTUATORS];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_fwd() {
        let mut c = SubterraneanController::new(
            &GenesisSeed::from_phrase("t"),
            &SubterraneanConfig::default(),
        );
        let cmd = c.forward(&ContinuousHV::random(HDC_DIM, 42), 0.005);
        assert!(cmd.torques.iter().all(|t| t.is_finite()));
    }
    #[test]
    fn checkpoint_round_trip_preserves_learned_projection() {
        let genesis = GenesisSeed::from_phrase("checkpoint-round-trip");
        let config = SubterraneanConfig::default();
        let input = ContinuousHV::random(HDC_DIM, 9);
        let mut source = SubterraneanController::new(&genesis, &config);
        let mut target = SubterraneanCommand::zero();
        target.set_cutter_head(0.8);
        target.set_left_track(-0.4);
        for _ in 0..8 {
            source.forward(&input, 0.005);
            source.train_step(&target);
        }
        let checkpoint = source.checkpoint();

        let mut restored = SubterraneanController::new(&genesis, &config);
        assert!(restored.load_checkpoint(&checkpoint).is_ok());
        source.reset();
        let source_output = source.forward(&input, 0.005);
        let restored_output = restored.forward(&input, 0.005);
        assert_eq!(source_output.torques, restored_output.torques);
    }

    #[test]
    fn checkpoint_rejects_wrong_dimension() {
        let genesis = GenesisSeed::from_phrase("checkpoint-invalid");
        let config = SubterraneanConfig::default();
        let mut controller = SubterraneanController::new(&genesis, &config);
        let mut checkpoint = controller.checkpoint();
        checkpoint.hdc_dimension += 1;
        assert!(matches!(
            controller.load_checkpoint(&checkpoint),
            Err(CheckpointError::DimensionMismatch { .. })
        ));
    }
}
