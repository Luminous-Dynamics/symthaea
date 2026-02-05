//! Temporal dynamics for audio pattern recognition.
//!
//! This module provides:
//! - `CfcCell`: Closed-form Continuous-time cell (stable dynamics)
//! - `HierarchicalCfc`: Multi-timescale CfC network
//! - `LtcNode`: Liquid Time-Constant node (original architecture)
//! - `HierarchicalLtc`: Multi-timescale LTC network
//! - `TemporalWindow`: Rolling buffer for context-aware encoding

use crate::hdc::{HV, HDC_DIM};

/// Number of frames in the temporal window
pub const TEMPORAL_WINDOW_SIZE: usize = 5;

// =============================================================================
// CfC (Closed-form Continuous-time) Cell
// =============================================================================

/// CfC (Closed-form Continuous-time) Cell
/// Provides stable temporal dynamics with guaranteed convergence properties.
pub struct CfcCell {
    state: Vec<f32>,
    prev_state: Vec<f32>,
    tau: f32,
    backbone_weight: Vec<f32>,
    backbone_bias: Vec<f32>,
    dim: usize,
}

impl CfcCell {
    /// Create a new CfC cell with specified time constant
    pub fn new(dim: usize, tau: f32, seed: u64) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut backbone_weight = Vec::with_capacity(dim);
        let mut backbone_bias = Vec::with_capacity(dim);
        let mut initial_state = Vec::with_capacity(dim);

        for i in 0..dim {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            let w = ((hasher.finish() as f32 / u64::MAX as f32) - 0.5) * 0.5;
            backbone_weight.push(w);

            let mut hasher2 = DefaultHasher::new();
            (seed + 100).hash(&mut hasher2);
            i.hash(&mut hasher2);
            let b = ((hasher2.finish() as f32 / u64::MAX as f32) - 0.5) * 0.1;
            backbone_bias.push(b);

            let mut hasher3 = DefaultHasher::new();
            (seed + 200).hash(&mut hasher3);
            i.hash(&mut hasher3);
            let s = ((hasher3.finish() as f32 / u64::MAX as f32) - 0.5) * 0.1;
            initial_state.push(s);
        }

        Self {
            state: initial_state.clone(),
            prev_state: initial_state,
            tau,
            backbone_weight,
            backbone_bias,
            dim,
        }
    }

    /// Step the CfC cell with closed-form solution
    pub fn step(&mut self, dt: f32, input: &[f32]) {
        self.prev_state.copy_from_slice(&self.state);

        let decay = (-dt / self.tau).exp();
        let input_scale = 1.0 - decay;

        for i in 0..self.dim {
            let input_val = if i < input.len() { input[i] } else { 0.0 };
            let x_eq = (self.backbone_weight[i] * input_val + self.backbone_bias[i]).tanh();
            self.state[i] = self.state[i] * decay + x_eq * input_scale;
        }
    }

    pub fn get_state(&self) -> &[f32] {
        &self.state
    }

    pub fn get_velocity(&self) -> Vec<f32> {
        self.state.iter()
            .zip(&self.prev_state)
            .map(|(curr, prev)| curr - prev)
            .collect()
    }

    pub fn get_phase_space(&self) -> Vec<f32> {
        let velocity = self.get_velocity();
        let mut phase_space = Vec::with_capacity(self.dim * 2);
        phase_space.extend_from_slice(&self.state);
        phase_space.extend_from_slice(&velocity);
        phase_space
    }

    pub fn reset(&mut self, seed: u64) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        for i in 0..self.dim {
            let mut hasher = DefaultHasher::new();
            (seed + 200).hash(&mut hasher);
            i.hash(&mut hasher);
            let s = ((hasher.finish() as f32 / u64::MAX as f32) - 0.5) * 0.1;
            self.state[i] = s;
            self.prev_state[i] = s;
        }
    }
}

// =============================================================================
// Hierarchical CfC Network
// =============================================================================

/// Hierarchical CfC Network - Multi-timescale dynamics
pub struct HierarchicalCfc {
    cells: Vec<CfcCell>,
    pub phi: f64,
    #[allow(dead_code)]
    taus: Vec<f32>,
}

impl HierarchicalCfc {
    /// Create with default timescales
    pub fn new(dim: usize) -> Self {
        Self::with_taus(dim, &[500.0, 200.0, 80.0, 40.0, 30.0])
    }

    /// Create with custom timescales
    pub fn with_taus(dim: usize, taus: &[f32]) -> Self {
        let cells: Vec<CfcCell> = taus.iter()
            .enumerate()
            .map(|(i, &tau)| CfcCell::new(dim, tau, 7000 + i as u64))
            .collect();

        Self {
            cells,
            phi: 0.0,
            taus: taus.to_vec(),
        }
    }

    pub fn step(&mut self, dt: f32, input: &[f32]) {
        for cell in &mut self.cells {
            cell.step(dt, input);
        }

        // Compute Phi as variance across levels
        if !self.cells.is_empty() && !self.cells[0].state.is_empty() {
            let dim = self.cells[0].dim;
            let n_cells = self.cells.len();

            let mut total_var = 0.0f64;
            for i in 0..dim {
                let mean: f32 = self.cells.iter().map(|c| c.state[i]).sum::<f32>() / n_cells as f32;
                let var: f32 = self.cells.iter()
                    .map(|c| (c.state[i] - mean).powi(2))
                    .sum::<f32>() / n_cells as f32;
                total_var += var as f64;
            }
            self.phi = total_var / dim as f64;
        }
    }

    pub fn get_multi_scale_phase_space(&self) -> Vec<f32> {
        let mut result = Vec::new();
        for cell in &self.cells {
            result.extend(cell.get_phase_space());
        }
        result
    }

    pub fn get_level_velocities(&self) -> Vec<Vec<f32>> {
        self.cells.iter().map(|c| c.get_velocity()).collect()
    }

    pub fn get_level_states(&self) -> Vec<&[f32]> {
        self.cells.iter().map(|c| c.get_state()).collect()
    }

    pub fn reset(&mut self) {
        for (i, cell) in self.cells.iter_mut().enumerate() {
            cell.reset(7000 + i as u64);
        }
        self.phi = 0.0;
    }
}

// =============================================================================
// Temporal Window Buffer
// =============================================================================

/// Temporal Window Buffer for context-aware encoding
pub struct TemporalWindow {
    buffer: Vec<Vec<f32>>,
    write_pos: usize,
    valid_count: usize,
    feature_dim: usize,
}

impl TemporalWindow {
    pub fn new(feature_dim: usize) -> Self {
        Self {
            buffer: vec![vec![0.0; feature_dim]; TEMPORAL_WINDOW_SIZE],
            write_pos: 0,
            valid_count: 0,
            feature_dim,
        }
    }

    pub fn push(&mut self, features: &[f32]) {
        assert!(features.len() <= self.feature_dim,
            "Feature dimension {} exceeds window dimension {}", features.len(), self.feature_dim);

        self.buffer[self.write_pos][..features.len()].copy_from_slice(features);

        for i in features.len()..self.feature_dim {
            self.buffer[self.write_pos][i] = 0.0;
        }

        self.write_pos = (self.write_pos + 1) % TEMPORAL_WINDOW_SIZE;
        self.valid_count = self.valid_count.saturating_add(1).min(TEMPORAL_WINDOW_SIZE);
    }

    pub fn get_averaged_context(&self) -> Vec<f32> {
        if self.valid_count == 0 {
            return vec![0.0; self.feature_dim];
        }

        let mut avg = vec![0.0; self.feature_dim];
        for i in 0..self.valid_count {
            let idx = (self.write_pos + TEMPORAL_WINDOW_SIZE - 1 - i) % TEMPORAL_WINDOW_SIZE;
            for j in 0..self.feature_dim {
                avg[j] += self.buffer[idx][j];
            }
        }

        let scale = 1.0 / self.valid_count as f32;
        for v in &mut avg {
            *v *= scale;
        }
        avg
    }

    #[allow(dead_code)]
    pub fn get_concatenated(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(TEMPORAL_WINDOW_SIZE * self.feature_dim);
        for i in 0..TEMPORAL_WINDOW_SIZE {
            let idx = (self.write_pos + i) % TEMPORAL_WINDOW_SIZE;
            if i < self.valid_count {
                result.extend_from_slice(&self.buffer[idx]);
            } else {
                result.extend(std::iter::repeat(0.0).take(self.feature_dim));
            }
        }
        result
    }

    pub fn get_weighted_context(&self, decay: f32) -> Vec<f32> {
        if self.valid_count == 0 {
            return vec![0.0; self.feature_dim];
        }

        let mut weighted = vec![0.0; self.feature_dim];
        let mut total_weight = 0.0f32;

        for i in 0..self.valid_count {
            let age = i as f32;
            let weight = (-decay * age).exp();
            total_weight += weight;

            let idx = (self.write_pos + TEMPORAL_WINDOW_SIZE - 1 - i) % TEMPORAL_WINDOW_SIZE;
            for j in 0..self.feature_dim {
                weighted[j] += self.buffer[idx][j] * weight;
            }
        }

        if total_weight > 1e-6 {
            for v in &mut weighted {
                *v /= total_weight;
            }
        }
        weighted
    }

    pub fn reset(&mut self) {
        for frame in &mut self.buffer {
            for v in frame.iter_mut() {
                *v = 0.0;
            }
        }
        self.write_pos = 0;
        self.valid_count = 0;
    }

    #[allow(dead_code)]
    pub fn is_full(&self) -> bool {
        self.valid_count >= TEMPORAL_WINDOW_SIZE
    }
}

// =============================================================================
// LTC Network
// =============================================================================

pub(crate) struct LtcNode {
    pub state: HV,
    prev_state: HV,
    tau: f32,
    weight: HV,
    bias: HV,
}

impl LtcNode {
    fn new(tau: f32, seed: u64) -> Self {
        let initial_state = HV::random_seeded(seed).scale(0.1);
        Self {
            state: initial_state.clone(),
            prev_state: initial_state,
            tau,
            weight: HV::random_seeded(seed + 100).scale(0.5),
            bias: HV::random_seeded(seed + 200).scale(0.1),
        }
    }

    fn step(&mut self, dt: f32, input: &HV) {
        self.prev_state = self.state.clone();

        let mut activation = HV::zero();
        for i in 0..HDC_DIM {
            let wx = self.weight.values[i] * input.values[i];
            activation.values[i] = (wx + self.bias.values[i]).tanh();
        }

        let decay = (-dt / self.tau).exp();
        for i in 0..HDC_DIM {
            self.state.values[i] = self.state.values[i] * decay + activation.values[i] * (1.0 - decay);
        }
    }

    fn get_velocity(&self) -> HV {
        let mut velocity = HV::zero();
        for i in 0..HDC_DIM {
            velocity.values[i] = self.state.values[i] - self.prev_state.values[i];
        }
        velocity
    }
}

/// Preset LTC configurations for different acoustic domains
#[derive(Clone, Copy)]
pub enum LtcPreset {
    /// Default: Speech/Environmental sounds (tau: 500-30ms)
    Standard,
    /// Fast: Bird vocalizations, rapid trills (tau: 100-5ms)
    FastBird,
    /// Slow: Whale calls, low-frequency signals (tau: 2000-200ms)
    SlowWhale,
}

pub struct HierarchicalLtc {
    nodes: Vec<LtcNode>,
    pub phi: f64,
    preset: LtcPreset,
}

impl HierarchicalLtc {
    pub fn new() -> Self {
        Self::with_preset(LtcPreset::Standard)
    }

    pub fn with_preset(preset: LtcPreset) -> Self {
        let taus: [f32; 5] = match preset {
            LtcPreset::Standard => [500.0, 200.0, 80.0, 40.0, 30.0],
            LtcPreset::FastBird => [100.0, 50.0, 20.0, 10.0, 5.0],
            LtcPreset::SlowWhale => [2000.0, 800.0, 300.0, 150.0, 80.0],
        };
        Self {
            nodes: taus.iter().enumerate()
                .map(|(i, &tau)| LtcNode::new(tau, 5000 + i as u64))
                .collect(),
            phi: 0.0,
            preset,
        }
    }

    #[allow(dead_code)]
    pub fn with_taus(taus: &[f32]) -> Self {
        Self {
            nodes: taus.iter().enumerate()
                .map(|(i, &tau)| LtcNode::new(tau, 5000 + i as u64))
                .collect(),
            phi: 0.0,
            preset: LtcPreset::Standard,
        }
    }

    pub fn step(&mut self, dt: f32, input: &HV) {
        for node in &mut self.nodes {
            node.step(dt, input);
        }

        // Compute Phi estimate
        let mean: Vec<f32> = (0..HDC_DIM).map(|i| {
            self.nodes.iter().map(|n| n.state.values[i]).sum::<f32>() / self.nodes.len() as f32
        }).collect();

        let var: f32 = self.nodes.iter().map(|n| {
            n.state.values.iter().zip(&mean).map(|(s, m)| (s - m).powi(2)).sum::<f32>()
        }).sum::<f32>() / (self.nodes.len() * HDC_DIM) as f32;

        self.phi = var as f64;
    }

    #[allow(dead_code)]
    pub fn get_state(&self) -> HV {
        let mut combined = HV::zero();
        for node in &self.nodes {
            combined = combined.add(&node.state);
        }
        combined.scale(1.0 / self.nodes.len() as f32)
    }

    #[allow(dead_code)]
    pub fn get_level_states(&self) -> Vec<HV> {
        self.nodes.iter().map(|n| n.state.clone()).collect()
    }

    pub fn get_level_velocities(&self) -> Vec<HV> {
        self.nodes.iter().map(|n| n.get_velocity()).collect()
    }

    pub fn reset(&mut self) {
        *self = Self::with_preset(self.preset);
    }
}

impl Default for HierarchicalLtc {
    fn default() -> Self {
        Self::new()
    }
}
