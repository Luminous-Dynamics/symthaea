// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Humanoid controller: wraps HdcLtcUnifiedNetwork + output projection (16,384D → 21D).

use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{
    ContinuousHV, HDC_DIMENSION, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig,
};

use crate::morphology::HumanoidMorphology;
use crate::types::{HumanoidCommand, HumanoidConfig, NUM_ACTUATORS};

fn checkpoint_version() -> u32 {
    3
}

/// Versioned output-head checkpoint.
///
/// The underlying HDC-LTC network is deterministically reconstructed from the
/// genesis phrase; learned recurrent parameters/state are not serialized by the
/// current core API. `recurrent_state_included` makes that limitation explicit so
/// evaluation tooling cannot mistake this for a full learner snapshot.
#[derive(Serialize, Deserialize)]
pub struct ControllerCheckpoint {
    #[serde(default = "checkpoint_version")]
    pub version: u32,
    pub output_weights: Vec<f32>,
    pub output_bias: Vec<f32>,
    pub learning_rate: f32,
    pub genesis_phrase: String,
    pub network_layers: usize,
    pub neurons_per_layer: usize,
    #[serde(default)]
    pub morphology: Option<HumanoidMorphology>,
    #[serde(default)]
    pub observation_schema_id: Option<String>,
    #[serde(default)]
    pub recurrent_state_included: bool,
    #[serde(default)]
    pub recurrent_learning_enabled: bool,
}

const BOTTLENECK_DIM: usize = 64;
const DEFAULT_TAU_BASE: f32 = 0.025;

pub struct HumanoidController {
    network: HdcLtcUnifiedNetwork,
    bottleneck_projections: Vec<ContinuousHV>,
    output_weights: Vec<f32>,
    output_bias: Vec<f32>,
    num_outputs: usize,
    learning_rate: f32,
    lr_scale: f32,
    nominal_tau_base: f32,
    tau_factor: f32,
}

impl HumanoidController {
    pub fn new(genesis: &GenesisSeed, config: &HumanoidConfig) -> Self {
        let neuron_config = UnifiedConfig {
            tau_base: DEFAULT_TAU_BASE,
            backbone_tau: 0.3,
            dimension: HDC_DIMENSION,
            learning_rate: config.learning_rate,
            ..UnifiedConfig::default()
        };

        let net_config = UnifiedNetworkConfig {
            layer_sizes: vec![config.neurons_per_layer; config.network_layers],
            neuron_config,
            use_layer_binding: true,
            skip_connections: false,
        };

        let network = HdcLtcUnifiedNetwork::from_genesis(net_config, genesis);

        let bottleneck_projections: Vec<ContinuousHV> = (0..BOTTLENECK_DIM)
            .map(|i| genesis.hv(&format!("humanoid::bottleneck::{i}"), HDC_DIMENSION))
            .collect();

        let num_outputs = config.morphology.num_actuators();
        let total_weights = num_outputs * BOTTLENECK_DIM;
        let weight_hv = genesis.hv("humanoid::output_weights_v2", total_weights);
        let mut output_weights = weight_hv.values;
        for w in &mut output_weights {
            *w *= 0.1;
        }

        let output_bias = vec![0.0f32; num_outputs];

        Self {
            network,
            bottleneck_projections,
            output_weights,
            output_bias,
            num_outputs,
            learning_rate: config.learning_rate,
            lr_scale: 1.0,
            nominal_tau_base: DEFAULT_TAU_BASE,
            tau_factor: 1.0,
        }
    }

    pub fn from_network(network: HdcLtcUnifiedNetwork, genesis: &GenesisSeed) -> Self {
        let num_outputs = NUM_ACTUATORS;
        let bottleneck_projections: Vec<ContinuousHV> = (0..BOTTLENECK_DIM)
            .map(|i| genesis.hv(&format!("humanoid::bottleneck::{i}"), HDC_DIMENSION))
            .collect();
        let total_weights = num_outputs * BOTTLENECK_DIM;
        let weight_hv = genesis.hv("humanoid::output_weights_v2", total_weights);
        let mut output_weights = weight_hv.values;
        for w in &mut output_weights {
            *w *= 0.1;
        }
        let output_bias = vec![0.0f32; num_outputs];

        Self {
            network,
            bottleneck_projections,
            output_weights,
            output_bias,
            num_outputs,
            learning_rate: 0.0005,
            lr_scale: 1.0,
            nominal_tau_base: DEFAULT_TAU_BASE,
            tau_factor: 1.0,
        }
    }

    pub fn forward(&mut self, sensor_hv: &ContinuousHV, dt: f32) -> HumanoidCommand {
        self.network.evolve_closed_form(dt, sensor_hv);
        let output_hv = self.network.output().normalize();

        let mut bottleneck = vec![0.0f32; BOTTLENECK_DIM];
        for (b, proj) in bottleneck
            .iter_mut()
            .zip(self.bottleneck_projections.iter())
        {
            *b = proj.similarity(&output_hv);
        }
        for b in bottleneck.iter_mut() {
            *b = fast_tanh(*b * 3.0);
        }

        let mut raw = vec![0.0f32; self.num_outputs];
        for ((r, &bias), row) in raw
            .iter_mut()
            .zip(self.output_bias.iter())
            .zip(self.output_weights.chunks_exact(BOTTLENECK_DIM))
        {
            *r = bias
                + row
                    .iter()
                    .zip(bottleneck.iter())
                    .map(|(&w, &b)| w * b)
                    .sum::<f32>();
        }

        let mut torques = vec![0.0f32; self.num_outputs];
        for i in 0..self.num_outputs {
            torques[i] = fast_tanh(raw[i]);
        }

        HumanoidCommand { torques }
    }

    /// Fix: Added an isolated head replay optimizer that protects continuous temporal traces from desynchronization pollution
    pub fn train_head_replay(
        &mut self,
        output_hv: &ContinuousHV,
        target: &HumanoidCommand,
        lr: f32,
    ) {
        let output_lr = (lr * (HDC_DIMENSION as f32).sqrt()).min(lr * 10.0);

        let mut bottleneck = vec![0.0f32; BOTTLENECK_DIM];
        for (b, proj) in bottleneck
            .iter_mut()
            .zip(self.bottleneck_projections.iter())
        {
            *b = proj.similarity(output_hv);
        }
        let mut bottleneck_activated = vec![0.0f32; BOTTLENECK_DIM];
        for j in 0..BOTTLENECK_DIM {
            bottleneck_activated[j] = fast_tanh(bottleneck[j] * 3.0);
        }

        let mut raw = vec![0.0f32; self.num_outputs];
        for ((r, &bias), row) in raw
            .iter_mut()
            .zip(self.output_bias.iter())
            .zip(self.output_weights.chunks_exact(BOTTLENECK_DIM))
        {
            *r = bias
                + row
                    .iter()
                    .zip(bottleneck_activated.iter())
                    .map(|(&w, &b)| w * b)
                    .sum::<f32>();
        }

        let mut d_raw = vec![0.0f32; self.num_outputs];
        for i in 0..self.num_outputs {
            let pred = fast_tanh(raw[i]);
            let error = pred - target.torques[i];
            let tanh_deriv = 1.0 - pred * pred;
            d_raw[i] = error * tanh_deriv;
        }

        const GRAD_CLIP: f32 = 0.5;
        for g in &mut d_raw {
            *g = g.clamp(-GRAD_CLIP, GRAD_CLIP);
        }

        const WEIGHT_DECAY: f32 = 1e-5;
        let decay = 1.0 - WEIGHT_DECAY;
        for w in self.output_weights.iter_mut() {
            *w *= decay;
        }

        for ((row, &dr), bias) in self
            .output_weights
            .chunks_exact_mut(BOTTLENECK_DIM)
            .zip(d_raw.iter())
            .zip(self.output_bias.iter_mut())
        {
            for (w, &b) in row.iter_mut().zip(bottleneck_activated.iter()) {
                *w -= output_lr * dr * b;
            }
            *bias -= output_lr * dr;
        }
    }

    pub fn train_step(
        &mut self,
        sensor_hv: &ContinuousHV,
        target: &HumanoidCommand,
        dt: f32,
        lr_override: Option<f32>,
    ) {
        let lr = lr_override.unwrap_or(self.learning_rate);
        let output_lr = (lr * (HDC_DIMENSION as f32).sqrt()).min(lr * 10.0);
        let output_hv = self.network.output().normalize();

        let mut bottleneck = vec![0.0f32; BOTTLENECK_DIM];
        for (b, proj) in bottleneck
            .iter_mut()
            .zip(self.bottleneck_projections.iter())
        {
            *b = proj.similarity(&output_hv);
        }
        let mut bottleneck_activated = vec![0.0f32; BOTTLENECK_DIM];
        for j in 0..BOTTLENECK_DIM {
            bottleneck_activated[j] = fast_tanh(bottleneck[j] * 3.0);
        }

        let mut raw = vec![0.0f32; self.num_outputs];
        for ((r, &bias), row) in raw
            .iter_mut()
            .zip(self.output_bias.iter())
            .zip(self.output_weights.chunks_exact(BOTTLENECK_DIM))
        {
            *r = bias
                + row
                    .iter()
                    .zip(bottleneck_activated.iter())
                    .map(|(&w, &b)| w * b)
                    .sum::<f32>();
        }

        let mut d_raw = vec![0.0f32; self.num_outputs];
        for i in 0..self.num_outputs {
            let pred = fast_tanh(raw[i]);
            let error = pred - target.torques[i];
            let tanh_deriv = 1.0 - pred * pred;
            d_raw[i] = error * tanh_deriv;
        }

        const GRAD_CLIP: f32 = 0.5;
        for g in &mut d_raw {
            *g = g.clamp(-GRAD_CLIP, GRAD_CLIP);
        }

        const WEIGHT_DECAY: f32 = 1e-5;
        let decay = 1.0 - WEIGHT_DECAY;
        for w in self.output_weights.iter_mut() {
            *w *= decay;
        }

        for ((row, &dr), bias) in self
            .output_weights
            .chunks_exact_mut(BOTTLENECK_DIM)
            .zip(d_raw.iter())
            .zip(self.output_bias.iter_mut())
        {
            for (w, &b) in row.iter_mut().zip(bottleneck_activated.iter()) {
                *w -= output_lr * dr * b;
            }
            *bias -= output_lr * dr;
        }

        let dim = HDC_DIMENSION;
        let mut d_bottleneck = vec![0.0f32; BOTTLENECK_DIM];
        for (row, &dr) in self
            .output_weights
            .chunks_exact(BOTTLENECK_DIM)
            .zip(d_raw.iter())
        {
            for (db, &w) in d_bottleneck.iter_mut().zip(row.iter()) {
                *db += dr * w;
            }
        }

        for j in 0..BOTTLENECK_DIM {
            let tanh_deriv = 1.0 - bottleneck_activated[j] * bottleneck_activated[j];
            d_bottleneck[j] *= tanh_deriv * 3.0;
        }

        let mut grad_hv_values = vec![0.0f32; dim];
        for (j, proj) in self.bottleneck_projections.iter().enumerate() {
            let proj_values = proj.as_slice();
            let grad = d_bottleneck[j];
            if grad.abs() > 1e-8 {
                for k in 0..dim {
                    grad_hv_values[k] += grad * proj_values[k];
                }
            }
        }

        let n_layers = self.network.n_layers();
        let mut target_hv = output_hv.add(&ContinuousHV::from_vec(grad_hv_values).scale(-1.0));

        for layer_idx in (0..n_layers).rev() {
            let layer_input = self.network.layer_input(layer_idx, sensor_hv);
            let prev_layer_output = if layer_idx > 0 {
                self.network.output_at_layer(layer_idx - 1).cloned()
            } else {
                None
            };
            let mut avg_d_input = ContinuousHV::zero(dim);
            let mut neuron_count = 0usize;

            if let Some(layer) = self.network.layer_mut(layer_idx) {
                for neuron in layer.iter_mut() {
                    let grads = neuron.backward(&layer_input, &target_hv, dt);
                    avg_d_input = avg_d_input.add(&grads.d_input);
                    neuron_count += 1;
                    neuron.apply_gradients(&grads, lr);
                }
            }

            if let Some(prev_output) = prev_layer_output
                && neuron_count > 0
            {
                let scale = 1.0 / neuron_count as f32;
                target_hv = prev_output.subtract(&avg_d_input.scale(scale));
            }
        }
    }

    /// Set an absolute modulation relative to the nominal time constant.
    ///
    /// Previous behavior multiplied the already-modulated tau on every cognitive
    /// tick, causing exponential drift. Repeated calls with the same factor are now
    /// idempotent.
    pub fn modulate_tau(&mut self, factor: f32) {
        let factor = factor.clamp(0.3, 3.0);
        let target_tau = self.nominal_tau_base * factor;
        for layer_idx in 0..self.network.n_layers() {
            if let Some(layer) = self.network.layer_mut(layer_idx) {
                for neuron in layer.iter_mut() {
                    neuron.set_tau_base(target_tau);
                }
            }
        }
        self.tau_factor = factor;
    }

    pub fn tau_factor(&self) -> f32 {
        self.tau_factor
    }

    pub fn nominal_tau_base(&self) -> f32 {
        self.nominal_tau_base
    }

    pub fn normalize_states(&mut self) {
        for layer_idx in 0..self.network.n_layers() {
            if let Some(layer) = self.network.layer_mut(layer_idx) {
                for neuron in layer.iter_mut() {
                    let norm = neuron.state().norm();
                    if norm > 1.5 {
                        let normalized = neuron.state().normalize();
                        *neuron.state_mut() = normalized;
                    }
                }
            }
        }
    }

    pub fn warmup(&mut self, samples: &[(ContinuousHV, HumanoidCommand)], n_steps: usize, lr: f32) {
        for step in 0..n_steps {
            let (ref hv, ref target) = samples[step % samples.len()];
            self.train_head_replay(hv, target, lr);
        }
        self.network.reset();
    }

    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr.clamp(1e-6, 0.1);
    }
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate * self.lr_scale
    }
    pub fn set_learning_rate_scale(&mut self, scale: f32) {
        self.lr_scale = scale.max(0.0);
    }
    pub fn output_projection(&self) -> (Vec<f32>, Vec<f32>) {
        (self.output_weights.clone(), self.output_bias.to_vec())
    }
    pub fn set_output_projection(&mut self, weights: &[f32], bias: &[f32]) {
        if weights.len() == self.output_weights.len() {
            self.output_weights.copy_from_slice(weights);
        }
        let n = bias.len().min(self.num_outputs);
        self.output_bias[..n].copy_from_slice(&bias[..n]);
    }
    pub fn reset(&mut self) {
        self.network.reset();
    }
    pub fn stats(&self) -> symthaea_core::hdc::UnifiedNetworkStats {
        self.network.stats()
    }
    pub fn network(&self) -> &HdcLtcUnifiedNetwork {
        &self.network
    }
    pub fn mask_actuator(cmd: &mut HumanoidCommand, joint_index: usize) {
        if joint_index < cmd.torques.len() {
            cmd.torques[joint_index] = 0.0;
        }
    }

    pub fn save_checkpoint(&self, path: &str, config: &HumanoidConfig) -> std::io::Result<()> {
        if config.enable_recurrent_learning {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "cannot save a complete learner checkpoint while recurrent learning is enabled; the core network does not yet expose recurrent parameter serialization",
            ));
        }
        let checkpoint = ControllerCheckpoint {
            version: checkpoint_version(),
            output_weights: self.output_weights.clone(),
            output_bias: self.output_bias.to_vec(),
            learning_rate: self.learning_rate,
            genesis_phrase: config.genesis_phrase.clone(),
            network_layers: config.network_layers,
            neurons_per_layer: config.neurons_per_layer,
            morphology: Some(config.morphology),
            observation_schema_id: Some(config.morphology.schema_id().to_string()),
            recurrent_state_included: false,
            recurrent_learning_enabled: config.enable_recurrent_learning,
        };
        let json = serde_json::to_string_pretty(&checkpoint).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    pub fn load_checkpoint(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let checkpoint: ControllerCheckpoint =
            serde_json::from_str(&json).map_err(std::io::Error::other)?;
        let genesis = GenesisSeed::from_phrase(&checkpoint.genesis_phrase);
        if checkpoint.recurrent_state_included {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "checkpoint claims recurrent state that this loader cannot restore",
            ));
        }
        if checkpoint.recurrent_learning_enabled {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "checkpoint was produced with recurrent learning enabled but contains no recurrent parameters",
            ));
        }
        let morphology = checkpoint.morphology.unwrap_or(HumanoidMorphology::Dmc21);
        if let Some(schema_id) = checkpoint.observation_schema_id.as_deref()
            && schema_id != morphology.schema_id()
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "checkpoint observation schema {schema_id} does not match {}",
                    morphology.schema_id()
                ),
            ));
        }
        let config = HumanoidConfig {
            morphology,
            network_layers: checkpoint.network_layers,
            neurons_per_layer: checkpoint.neurons_per_layer,
            learning_rate: checkpoint.learning_rate,
            genesis_phrase: checkpoint.genesis_phrase,
            enable_recurrent_learning: checkpoint.recurrent_learning_enabled,
            ..HumanoidConfig::default()
        };

        let neuron_config = UnifiedConfig {
            tau_base: DEFAULT_TAU_BASE,
            backbone_tau: 0.3,
            dimension: HDC_DIMENSION,
            learning_rate: config.learning_rate,
            ..UnifiedConfig::default()
        };

        let net_config = UnifiedNetworkConfig {
            layer_sizes: vec![config.neurons_per_layer; config.network_layers],
            neuron_config,
            use_layer_binding: true,
            skip_connections: false,
        };

        let network = HdcLtcUnifiedNetwork::from_genesis(net_config, &genesis);
        let num_outputs = config.morphology.num_actuators();
        let expected_weights = num_outputs * BOTTLENECK_DIM;
        if checkpoint.output_weights.len() != expected_weights {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "checkpoint has {} output weights; expected {expected_weights} for {num_outputs} actuators",
                    checkpoint.output_weights.len()
                ),
            ));
        }
        if checkpoint.output_bias.len() != num_outputs {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "checkpoint has {} output biases; expected {num_outputs}",
                    checkpoint.output_bias.len()
                ),
            ));
        }
        let mut output_bias = vec![0.0f32; num_outputs];
        for (i, &v) in checkpoint.output_bias.iter().enumerate().take(num_outputs) {
            output_bias[i] = v;
        }

        let bottleneck_projections: Vec<ContinuousHV> = (0..BOTTLENECK_DIM)
            .map(|i| genesis.hv(&format!("humanoid::bottleneck::{i}"), HDC_DIMENSION))
            .collect();

        Ok(Self {
            network,
            bottleneck_projections,
            output_weights: checkpoint.output_weights,
            output_bias,
            num_outputs,
            learning_rate: checkpoint.learning_rate,
            lr_scale: 1.0,
            nominal_tau_base: DEFAULT_TAU_BASE,
            tau_factor: 1.0,
        })
    }
}

fn fast_tanh(x: f32) -> f32 {
    if x.abs() > 4.97 {
        x.signum()
    } else {
        let x2 = x * x;
        x * (27.0 + x2) / (27.0 + 9.0 * x2)
    }
}
