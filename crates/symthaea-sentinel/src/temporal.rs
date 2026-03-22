// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal dynamics for audio pattern recognition.
//!
//! This module provides:
//! - `CfcCell`: Closed-form Continuous-time cell (stable dynamics)
//! - `HierarchicalCfc`: Multi-timescale CfC network
//! - `LtcNode`: Liquid Time-Constant node (original architecture)
//! - `HierarchicalLtc`: Multi-timescale LTC network
//! - `TemporalWindow`: Rolling buffer for context-aware encoding

use crate::hdc::{HDC_DIM, HV};

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
        self.state
            .iter()
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
        let cells: Vec<CfcCell> = taus
            .iter()
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
                let var: f32 = self
                    .cells
                    .iter()
                    .map(|c| (c.state[i] - mean).powi(2))
                    .sum::<f32>()
                    / n_cells as f32;
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
        assert!(
            features.len() <= self.feature_dim,
            "Feature dimension {} exceeds window dimension {}",
            features.len(),
            self.feature_dim
        );

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
                result.extend(std::iter::repeat_n(0.0, self.feature_dim));
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
            self.state.values[i] =
                self.state.values[i] * decay + activation.values[i] * (1.0 - decay);
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
            nodes: taus
                .iter()
                .enumerate()
                .map(|(i, &tau)| LtcNode::new(tau, 5000 + i as u64))
                .collect(),
            phi: 0.0,
            preset,
        }
    }

    #[allow(dead_code)]
    pub fn with_taus(taus: &[f32]) -> Self {
        Self {
            nodes: taus
                .iter()
                .enumerate()
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
        let mean: Vec<f32> = (0..HDC_DIM)
            .map(|i| {
                self.nodes.iter().map(|n| n.state.values[i]).sum::<f32>() / self.nodes.len() as f32
            })
            .collect();

        let var: f32 = self
            .nodes
            .iter()
            .map(|n| {
                n.state
                    .values
                    .iter()
                    .zip(&mean)
                    .map(|(s, m)| (s - m).powi(2))
                    .sum::<f32>()
            })
            .sum::<f32>()
            / (self.nodes.len() * HDC_DIM) as f32;

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

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // CfcCell tests
    // =========================================================================

    #[test]
    fn test_cfc_cell_new() {
        let cell = CfcCell::new(8, 100.0, 42);
        assert_eq!(cell.get_state().len(), 8);
        assert!(cell.get_state().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_cfc_cell_step_finite_output() {
        let mut cell = CfcCell::new(4, 50.0, 1);
        let input = vec![0.5, -0.3, 0.8, 0.1];
        cell.step(10.0, &input);
        assert!(cell.get_state().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_cfc_cell_step_changes_state() {
        let mut cell = CfcCell::new(4, 50.0, 1);
        let initial: Vec<f32> = cell.get_state().to_vec();
        cell.step(10.0, &[1.0, 1.0, 1.0, 1.0]);
        let after: Vec<f32> = cell.get_state().to_vec();
        assert_ne!(initial, after, "State should change after step");
    }

    #[test]
    fn test_cfc_cell_velocity_initially_zero() {
        let cell = CfcCell::new(4, 50.0, 1);
        let vel = cell.get_velocity();
        assert_eq!(vel.len(), 4);
        assert!(
            vel.iter().all(|v| v.abs() < 1e-10),
            "Velocity should be zero before any step"
        );
    }

    #[test]
    fn test_cfc_cell_velocity_nonzero_after_step() {
        let mut cell = CfcCell::new(4, 50.0, 1);
        cell.step(10.0, &[1.0, 0.0, -1.0, 0.5]);
        let vel = cell.get_velocity();
        assert!(
            vel.iter().any(|v| v.abs() > 1e-10),
            "Velocity should be nonzero after step with input"
        );
    }

    #[test]
    fn test_cfc_cell_phase_space_dimensions() {
        let cell = CfcCell::new(8, 100.0, 1);
        let ps = cell.get_phase_space();
        assert_eq!(ps.len(), 16, "Phase space = 2 * dim");
    }

    #[test]
    fn test_cfc_cell_phase_space_is_state_plus_velocity() {
        let mut cell = CfcCell::new(4, 50.0, 1);
        cell.step(10.0, &[0.5; 4]);
        let ps = cell.get_phase_space();
        let state = cell.get_state();
        let vel = cell.get_velocity();
        assert_eq!(&ps[..4], state);
        assert_eq!(&ps[4..], vel.as_slice());
    }

    #[test]
    fn test_cfc_cell_reset() {
        let mut cell = CfcCell::new(4, 50.0, 42);
        let initial: Vec<f32> = cell.get_state().to_vec();
        // Evolve
        for _ in 0..20 {
            cell.step(10.0, &[1.0; 4]);
        }
        assert_ne!(cell.get_state(), initial.as_slice());
        // Reset with same seed
        cell.reset(42);
        assert_eq!(cell.get_state(), initial.as_slice());
        let vel = cell.get_velocity();
        assert!(
            vel.iter().all(|v| v.abs() < 1e-10),
            "Velocity zero after reset"
        );
    }

    #[test]
    fn test_cfc_cell_short_input_zero_padded() {
        let mut cell = CfcCell::new(8, 50.0, 1);
        // Input shorter than dim
        cell.step(10.0, &[1.0, 2.0]);
        // Should still produce finite state for all dims
        assert!(cell.get_state().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_cfc_cell_decay_towards_equilibrium() {
        let mut cell = CfcCell::new(4, 50.0, 1);
        // Step with constant input many times
        for _ in 0..100 {
            cell.step(10.0, &[1.0; 4]);
        }
        let state_a: Vec<f32> = cell.get_state().to_vec();
        for _ in 0..100 {
            cell.step(10.0, &[1.0; 4]);
        }
        let state_b: Vec<f32> = cell.get_state().to_vec();
        // Should converge — difference should be small
        let diff: f32 = state_a
            .iter()
            .zip(&state_b)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff < 0.01, "Should converge: diff={diff}");
    }

    // =========================================================================
    // HierarchicalCfc tests
    // =========================================================================

    #[test]
    fn test_hierarchical_cfc_default_taus() {
        let hcfc = HierarchicalCfc::new(4);
        assert_eq!(hcfc.cells.len(), 5);
        assert_eq!(hcfc.phi, 0.0);
    }

    #[test]
    fn test_hierarchical_cfc_custom_taus() {
        let hcfc = HierarchicalCfc::with_taus(4, &[100.0, 50.0]);
        assert_eq!(hcfc.cells.len(), 2);
    }

    #[test]
    fn test_hierarchical_cfc_step_updates_phi() {
        let mut hcfc = HierarchicalCfc::new(8);
        assert_eq!(hcfc.phi, 0.0);
        hcfc.step(10.0, &[1.0; 8]);
        // Phi should be non-negative (variance-based)
        assert!(hcfc.phi >= 0.0, "Phi should be non-negative");
        assert!(hcfc.phi.is_finite(), "Phi should be finite");
    }

    #[test]
    fn test_hierarchical_cfc_multi_scale_phase_space() {
        let hcfc = HierarchicalCfc::new(4);
        let ps = hcfc.get_multi_scale_phase_space();
        // 5 levels x (4 state + 4 velocity) = 40
        assert_eq!(ps.len(), 5 * 4 * 2);
    }

    #[test]
    fn test_hierarchical_cfc_level_velocities() {
        let hcfc = HierarchicalCfc::new(4);
        let vels = hcfc.get_level_velocities();
        assert_eq!(vels.len(), 5);
        for v in &vels {
            assert_eq!(v.len(), 4);
        }
    }

    #[test]
    fn test_hierarchical_cfc_level_states() {
        let hcfc = HierarchicalCfc::new(4);
        let states = hcfc.get_level_states();
        assert_eq!(states.len(), 5);
        for s in &states {
            assert_eq!(s.len(), 4);
        }
    }

    #[test]
    fn test_hierarchical_cfc_different_taus_different_dynamics() {
        let mut hcfc = HierarchicalCfc::new(4);
        for _ in 0..20 {
            hcfc.step(10.0, &[1.0; 4]);
        }
        let states = hcfc.get_level_states();
        // Different time constants should produce different states
        assert_ne!(states[0], states[4], "Fast and slow levels should differ");
    }

    #[test]
    fn test_hierarchical_cfc_reset() {
        let mut hcfc = HierarchicalCfc::new(4);
        for _ in 0..20 {
            hcfc.step(10.0, &[1.0; 4]);
        }
        assert!(hcfc.phi > 0.0);
        hcfc.reset();
        assert_eq!(hcfc.phi, 0.0);
    }

    // =========================================================================
    // TemporalWindow tests
    // =========================================================================

    #[test]
    fn test_temporal_window_new() {
        let tw = TemporalWindow::new(4);
        assert_eq!(tw.get_averaged_context().len(), 4);
        assert!(tw.get_averaged_context().iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_temporal_window_push_and_average() {
        let mut tw = TemporalWindow::new(2);
        tw.push(&[1.0, 2.0]);
        let avg = tw.get_averaged_context();
        assert!((avg[0] - 1.0).abs() < 1e-6);
        assert!((avg[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_temporal_window_averaging_multiple() {
        let mut tw = TemporalWindow::new(1);
        tw.push(&[2.0]);
        tw.push(&[4.0]);
        let avg = tw.get_averaged_context();
        assert!(
            (avg[0] - 3.0).abs() < 1e-6,
            "Average of 2 and 4 should be 3"
        );
    }

    #[test]
    fn test_temporal_window_circular_buffer() {
        let mut tw = TemporalWindow::new(1);
        // Fill past capacity
        for i in 0..10 {
            tw.push(&[i as f32]);
        }
        // Only last TEMPORAL_WINDOW_SIZE values should contribute
        let avg = tw.get_averaged_context();
        let expected: f32 = (5..10).map(|i| i as f32).sum::<f32>() / TEMPORAL_WINDOW_SIZE as f32;
        assert!(
            (avg[0] - expected).abs() < 1e-5,
            "Circular buffer average: got {}, expected {}",
            avg[0],
            expected
        );
    }

    #[test]
    fn test_temporal_window_weighted_context() {
        let mut tw = TemporalWindow::new(1);
        tw.push(&[1.0]);
        tw.push(&[2.0]);
        // Most recent (2.0) should have higher weight than older (1.0)
        let weighted = tw.get_weighted_context(1.0);
        assert!(weighted[0] > 1.0 && weighted[0] < 2.0);
        // High decay should weight recent more
        let weighted_high = tw.get_weighted_context(10.0);
        assert!(
            weighted_high[0] > weighted[0],
            "Higher decay = more recent emphasis"
        );
    }

    #[test]
    fn test_temporal_window_is_full() {
        let mut tw = TemporalWindow::new(2);
        assert!(!tw.is_full());
        for i in 0..TEMPORAL_WINDOW_SIZE {
            tw.push(&[i as f32; 2]);
        }
        assert!(tw.is_full());
    }

    #[test]
    fn test_temporal_window_concatenated() {
        let mut tw = TemporalWindow::new(2);
        tw.push(&[1.0, 2.0]);
        let concat = tw.get_concatenated();
        assert_eq!(concat.len(), TEMPORAL_WINDOW_SIZE * 2);
    }

    #[test]
    fn test_temporal_window_reset() {
        let mut tw = TemporalWindow::new(2);
        tw.push(&[5.0, 10.0]);
        tw.push(&[3.0, 7.0]);
        tw.reset();
        assert!(!tw.is_full());
        let avg = tw.get_averaged_context();
        assert!(avg.iter().all(|v| *v == 0.0), "Should be zero after reset");
    }

    // =========================================================================
    // HierarchicalLtc tests
    // =========================================================================

    #[test]
    fn test_hierarchical_ltc_default() {
        let ltc = HierarchicalLtc::default();
        assert_eq!(ltc.nodes.len(), 5);
        assert_eq!(ltc.phi, 0.0);
    }

    #[test]
    fn test_hierarchical_ltc_presets() {
        let standard = HierarchicalLtc::with_preset(LtcPreset::Standard);
        let fast = HierarchicalLtc::with_preset(LtcPreset::FastBird);
        let slow = HierarchicalLtc::with_preset(LtcPreset::SlowWhale);
        assert_eq!(standard.nodes.len(), 5);
        assert_eq!(fast.nodes.len(), 5);
        assert_eq!(slow.nodes.len(), 5);
    }

    #[test]
    fn test_hierarchical_ltc_step_updates_phi() {
        let mut ltc = HierarchicalLtc::new();
        let input = HV::random_seeded(99);
        ltc.step(10.0, &input);
        assert!(ltc.phi >= 0.0);
        assert!(ltc.phi.is_finite());
    }

    #[test]
    fn test_hierarchical_ltc_level_velocities() {
        let mut ltc = HierarchicalLtc::new();
        let input = HV::random_seeded(99);
        ltc.step(10.0, &input);
        let vels = ltc.get_level_velocities();
        assert_eq!(vels.len(), 5);
        assert!(
            vels.iter()
                .any(|v| v.values.iter().any(|x| x.abs() > 1e-10)),
            "Some velocity should be nonzero after step"
        );
    }

    #[test]
    fn test_hierarchical_ltc_reset() {
        let mut ltc = HierarchicalLtc::new();
        let input = HV::random_seeded(99);
        for _ in 0..10 {
            ltc.step(10.0, &input);
        }
        assert!(ltc.phi > 0.0);
        ltc.reset();
        assert_eq!(ltc.phi, 0.0);
    }
}
