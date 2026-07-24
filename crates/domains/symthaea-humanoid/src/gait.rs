// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gait quality analysis: foot clearance tracking and stride metrics.
//!
//! Provides `GaitAnalyzer` for measuring swing-phase foot clearance,
//! the key metric for detecting shuffling, toe-dragging, and fall risk.
//! Used by the training loop for diagnostics and optional reward shaping.

use crate::contact::ContactFrame;
use crate::types::HumanoidState;

/// Summary of gait quality metrics over an episode.
#[derive(Debug, Clone)]
pub struct GaitSummary {
    /// Mean of per-stride maximum foot clearances (meters).
    pub avg_clearance: f64,
    /// Minimum per-stride maximum clearance (worst-case, meters).
    pub min_clearance: f64,
    /// Number of complete strides analyzed.
    pub stride_count: usize,
    /// Mean stride length across both feet (meters).
    pub avg_stride_length: f64,
    /// Average cadence in steps per second.
    pub avg_cadence: f64,
    /// Gait asymmetry: |R_avg - L_avg| / (R_avg + L_avg), 0 = symmetric.
    pub gait_asymmetry: f64,
    /// Step regularity: exp(-CV) of step intervals (1.0 = perfectly regular).
    pub step_regularity: f64,
    /// Foot strike quality: proper heel-strike dorsiflexion + toe-off plantarflexion.
    pub foot_strike_quality: f64,
}

/// Tracks per-step gait metrics for quality analysis.
///
/// Accumulates foot clearance (z-height during swing), step count,
/// and stride length estimates. Used by training loop for diagnostics
/// and optional gait-quality reward shaping.
pub struct GaitAnalyzer {
    /// Maximum right foot clearance seen in current swing phase.
    max_r_clearance: f64,
    /// Maximum left foot clearance seen in current swing phase.
    max_l_clearance: f64,
    /// Right foot was in swing (off ground) last tick.
    r_in_swing: bool,
    /// Left foot was in swing last tick.
    l_in_swing: bool,
    /// Accumulated per-stride max clearances (for averaging).
    clearance_samples: Vec<f64>,
    /// Total steps analyzed.
    total_steps: usize,
    /// Horizontal position at last right foot touchdown [x, y].
    last_r_touchdown_pos: Option<[f64; 2]>,
    /// Horizontal position at last left foot touchdown [x, y].
    last_l_touchdown_pos: Option<[f64; 2]>,
    /// Timestamp at last right foot touchdown.
    last_r_touchdown_time: Option<f64>,
    /// Timestamp at last left foot touchdown.
    last_l_touchdown_time: Option<f64>,
    /// Per-foot right stride lengths.
    r_stride_lengths: Vec<f64>,
    /// Per-foot left stride lengths.
    l_stride_lengths: Vec<f64>,
    /// Step intervals (time between consecutive touchdowns of same foot).
    step_intervals: Vec<f64>,
    /// Right ankle angles at touchdown (heel-strike dorsiflexion).
    r_touchdown_angles: Vec<f64>,
    /// Left ankle angles at touchdown (heel-strike dorsiflexion).
    l_touchdown_angles: Vec<f64>,
    /// Right ankle angles at liftoff (toe-off plantarflexion).
    r_liftoff_angles: Vec<f64>,
    /// Left ankle angles at liftoff (toe-off plantarflexion).
    l_liftoff_angles: Vec<f64>,
}

/// Ground contact threshold: foot z < this value = stance (on ground).
const CONTACT_THRESHOLD: f64 = 0.03;

impl GaitAnalyzer {
    /// Create a new gait analyzer with empty state.
    pub fn new() -> Self {
        Self {
            max_r_clearance: 0.0,
            max_l_clearance: 0.0,
            r_in_swing: false,
            l_in_swing: false,
            clearance_samples: Vec::new(),
            total_steps: 0,
            last_r_touchdown_pos: None,
            last_l_touchdown_pos: None,
            last_r_touchdown_time: None,
            last_l_touchdown_time: None,
            r_stride_lengths: Vec::new(),
            l_stride_lengths: Vec::new(),
            step_intervals: Vec::new(),
            r_touchdown_angles: Vec::new(),
            l_touchdown_angles: Vec::new(),
            r_liftoff_angles: Vec::new(),
            l_liftoff_angles: Vec::new(),
        }
    }

    /// Update with current humanoid state.
    ///
    /// Detects swing/stance transitions from foot z-heights in `state.extremities`.
    /// Tracks maximum clearance per swing phase and records it on touchdown.
    pub fn update(&mut self, state: &HumanoidState) {
        let r_foot_z = state.extremities[8];
        let l_foot_z = state.extremities[11];

        let r_swing = r_foot_z > CONTACT_THRESHOLD;
        let l_swing = l_foot_z > CONTACT_THRESHOLD;

        // Right foot: track swing clearance
        if r_swing {
            if r_foot_z > self.max_r_clearance {
                self.max_r_clearance = r_foot_z;
            }
        } else if self.r_in_swing {
            // Touchdown: record clearance sample
            if self.max_r_clearance > 0.0 {
                self.clearance_samples.push(self.max_r_clearance);
            }
            self.max_r_clearance = 0.0;
        }

        // Left foot: track swing clearance
        if l_swing {
            if l_foot_z > self.max_l_clearance {
                self.max_l_clearance = l_foot_z;
            }
        } else if self.l_in_swing {
            // Touchdown: record clearance sample
            if self.max_l_clearance > 0.0 {
                self.clearance_samples.push(self.max_l_clearance);
            }
            self.max_l_clearance = 0.0;
        }

        self.r_in_swing = r_swing;
        self.l_in_swing = l_swing;
        self.total_steps += 1;
    }

    /// Update with current state, horizontal position, and timestamp.
    ///
    /// Performs all clearance tracking (same as `update()`), plus computes
    /// stride length (Euclidean distance between consecutive same-foot touchdowns)
    /// and step interval (time between touchdowns).
    pub fn update_with_position(
        &mut self,
        state: &HumanoidState,
        horizontal_pos: [f64; 2],
        time: f64,
    ) {
        let r_foot_z = state.extremities[8];
        let l_foot_z = state.extremities[11];

        let r_swing = r_foot_z > CONTACT_THRESHOLD;
        let l_swing = l_foot_z > CONTACT_THRESHOLD;

        // Right foot: track swing clearance and stride
        if r_swing {
            if r_foot_z > self.max_r_clearance {
                self.max_r_clearance = r_foot_z;
            }
            // Detect liftoff (stance→swing): record ankle angle at toe-off
            if !self.r_in_swing {
                self.r_liftoff_angles.push(state.joint_angles[8]); // right ankle_y
            }
        } else if self.r_in_swing {
            // Touchdown (swing→stance): record clearance + ankle angle
            if self.max_r_clearance > 0.0 {
                self.clearance_samples.push(self.max_r_clearance);
            }
            self.max_r_clearance = 0.0;

            // Record ankle angle at heel-strike
            self.r_touchdown_angles.push(state.joint_angles[8]); // right ankle_y

            // Stride length: distance between consecutive right touchdowns
            if let Some(prev_pos) = self.last_r_touchdown_pos {
                let dx = horizontal_pos[0] - prev_pos[0];
                let dy = horizontal_pos[1] - prev_pos[1];
                let stride = (dx * dx + dy * dy).sqrt();
                if stride > 0.01 {
                    self.r_stride_lengths.push(stride);
                }
            }
            // Step interval: time since last right touchdown
            if let Some(prev_time) = self.last_r_touchdown_time {
                let interval = time - prev_time;
                if interval > 0.01 {
                    self.step_intervals.push(interval);
                }
            }
            self.last_r_touchdown_pos = Some(horizontal_pos);
            self.last_r_touchdown_time = Some(time);
        }

        // Left foot: track swing clearance and stride
        if l_swing {
            if l_foot_z > self.max_l_clearance {
                self.max_l_clearance = l_foot_z;
            }
            // Detect liftoff (stance→swing): record ankle angle at toe-off
            if !self.l_in_swing {
                self.l_liftoff_angles.push(state.joint_angles[14]); // left ankle_y
            }
        } else if self.l_in_swing {
            // Touchdown (swing→stance): record clearance + ankle angle
            if self.max_l_clearance > 0.0 {
                self.clearance_samples.push(self.max_l_clearance);
            }
            self.max_l_clearance = 0.0;

            // Record ankle angle at heel-strike
            self.l_touchdown_angles.push(state.joint_angles[14]); // left ankle_y

            // Stride length
            if let Some(prev_pos) = self.last_l_touchdown_pos {
                let dx = horizontal_pos[0] - prev_pos[0];
                let dy = horizontal_pos[1] - prev_pos[1];
                let stride = (dx * dx + dy * dy).sqrt();
                if stride > 0.01 {
                    self.l_stride_lengths.push(stride);
                }
            }
            // Step interval
            if let Some(prev_time) = self.last_l_touchdown_time {
                let interval = time - prev_time;
                if interval > 0.01 {
                    self.step_intervals.push(interval);
                }
            }
            self.last_l_touchdown_pos = Some(horizontal_pos);
            self.last_l_touchdown_time = Some(time);
        }

        self.r_in_swing = r_swing;
        self.l_in_swing = l_swing;
        self.total_steps += 1;
    }

    /// Reset all accumulated state.
    pub fn reset(&mut self) {
        self.max_r_clearance = 0.0;
        self.max_l_clearance = 0.0;
        self.r_in_swing = false;
        self.l_in_swing = false;
        self.clearance_samples.clear();
        self.total_steps = 0;
        self.last_r_touchdown_pos = None;
        self.last_l_touchdown_pos = None;
        self.last_r_touchdown_time = None;
        self.last_l_touchdown_time = None;
        self.r_stride_lengths.clear();
        self.l_stride_lengths.clear();
        self.step_intervals.clear();
        self.r_touchdown_angles.clear();
        self.l_touchdown_angles.clear();
        self.r_liftoff_angles.clear();
        self.l_liftoff_angles.clear();
    }

    /// Mean of per-stride maximum foot clearances.
    pub fn avg_clearance(&self) -> f64 {
        if self.clearance_samples.is_empty() {
            0.0
        } else {
            self.clearance_samples.iter().sum::<f64>() / self.clearance_samples.len() as f64
        }
    }

    /// Minimum per-stride maximum clearance (worst-case swing).
    pub fn min_clearance(&self) -> f64 {
        self.clearance_samples
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap_or(0.0)
    }

    /// Number of complete strides (touchdown events) recorded.
    pub fn num_strides(&self) -> usize {
        self.clearance_samples.len()
    }

    /// Step regularity: exp(-CV) where CV = std/mean of step intervals.
    ///
    /// Returns 1.0 for perfectly regular steps, decays toward 0 for irregular.
    /// Returns 0.0 if fewer than 2 step intervals recorded.
    pub fn step_regularity(&self) -> f64 {
        if self.step_intervals.len() < 2 {
            return 0.0;
        }
        let mean = self.step_intervals.iter().sum::<f64>() / self.step_intervals.len() as f64;
        if mean < 1e-6 {
            return 0.0;
        }
        let variance = self
            .step_intervals
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / self.step_intervals.len() as f64;
        let cv = variance.sqrt() / mean;
        (-cv).exp()
    }

    /// Foot strike quality: measures proper heel-strike dorsiflexion and toe-off plantarflexion.
    ///
    /// Touchdown: positive ankle_y = dorsiflexion (heel-first) → sigmoid(20 × angle).
    /// Liftoff: negative ankle_y = plantarflexion (toe pushoff) → sigmoid(-20 × angle).
    /// Returns average across all recorded events, or 0.0 if none.
    pub fn foot_strike_quality(&self) -> f64 {
        let sigmoid = |x: f64| -> f64 { 1.0 / (1.0 + (-20.0 * x).exp()) };
        let all: Vec<f64> = self
            .r_touchdown_angles
            .iter()
            .map(|a| sigmoid(*a))
            .chain(self.l_touchdown_angles.iter().map(|a| sigmoid(*a)))
            .chain(self.r_liftoff_angles.iter().map(|a| sigmoid(-*a)))
            .chain(self.l_liftoff_angles.iter().map(|a| sigmoid(-*a)))
            .collect();
        if all.is_empty() {
            0.0
        } else {
            all.iter().sum::<f64>() / all.len() as f64
        }
    }

    /// Full summary of gait quality metrics.
    pub fn summary(&self) -> GaitSummary {
        let all_strides: Vec<f64> = self
            .r_stride_lengths
            .iter()
            .chain(self.l_stride_lengths.iter())
            .copied()
            .collect();
        let avg_stride_length = if all_strides.is_empty() {
            0.0
        } else {
            all_strides.iter().sum::<f64>() / all_strides.len() as f64
        };

        let avg_cadence = if self.step_intervals.is_empty() {
            0.0
        } else {
            let avg_interval =
                self.step_intervals.iter().sum::<f64>() / self.step_intervals.len() as f64;
            if avg_interval > 0.0 {
                1.0 / avg_interval
            } else {
                0.0
            }
        };

        let r_avg = if self.r_stride_lengths.is_empty() {
            0.0
        } else {
            self.r_stride_lengths.iter().sum::<f64>() / self.r_stride_lengths.len() as f64
        };
        let l_avg = if self.l_stride_lengths.is_empty() {
            0.0
        } else {
            self.l_stride_lengths.iter().sum::<f64>() / self.l_stride_lengths.len() as f64
        };
        let gait_asymmetry = if r_avg + l_avg > 0.0 {
            (r_avg - l_avg).abs() / (r_avg + l_avg)
        } else {
            0.0
        };

        GaitSummary {
            avg_clearance: self.avg_clearance(),
            min_clearance: self.min_clearance(),
            stride_count: self.num_strides(),
            avg_stride_length,
            avg_cadence,
            gait_asymmetry,
            step_regularity: self.step_regularity(),
            foot_strike_quality: self.foot_strike_quality(),
        }
    }
}

impl Default for GaitAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gait_analyzer_detects_swing() {
        let mut analyzer = GaitAnalyzer::new();

        // Foot on ground for a few ticks
        for _ in 0..5 {
            let mut state = HumanoidState::standing();
            state.extremities[8] = 0.0; // right foot on ground
            analyzer.update(&state);
        }

        // Right foot lifts (swing phase)
        for _ in 0..10 {
            let mut state = HumanoidState::standing();
            state.extremities[8] = 0.08; // right foot in air
            analyzer.update(&state);
        }

        // Right foot lands (touchdown)
        let mut state = HumanoidState::standing();
        state.extremities[8] = 0.0;
        analyzer.update(&state);

        assert_eq!(
            analyzer.num_strides(),
            1,
            "Should detect 1 stride from swing→stance transition"
        );
    }

    #[test]
    fn test_gait_analyzer_tracks_clearance() {
        let mut analyzer = GaitAnalyzer::new();

        // Ground phase
        let mut state = HumanoidState::standing();
        state.extremities[11] = 0.0;
        analyzer.update(&state);

        // Swing phase with known clearance
        for _ in 0..5 {
            let mut state = HumanoidState::standing();
            state.extremities[11] = 0.08;
            analyzer.update(&state);
        }

        // Touchdown
        let mut state = HumanoidState::standing();
        state.extremities[11] = 0.0;
        analyzer.update(&state);

        assert!(
            (analyzer.avg_clearance() - 0.08).abs() < 1e-6,
            "Average clearance should be ~0.08: got {}",
            analyzer.avg_clearance()
        );
        assert!(
            (analyzer.min_clearance() - 0.08).abs() < 1e-6,
            "Min clearance should be ~0.08: got {}",
            analyzer.min_clearance()
        );
    }

    #[test]
    fn test_stride_length_tracking() {
        let mut analyzer = GaitAnalyzer::new();

        // Right foot: first swing/stance cycle at position [0, 0]
        let mut state = HumanoidState::standing();
        state.extremities[8] = 0.0; // ground
        analyzer.update_with_position(&state, [0.0, 0.0], 0.0);

        // Swing phase
        for i in 1..=5 {
            state.extremities[8] = 0.08;
            analyzer.update_with_position(&state, [0.1 * i as f64, 0.0], i as f64 * 0.025);
        }

        // Touchdown at position [0.6, 0] → first touchdown recorded (no stride yet)
        state.extremities[8] = 0.0;
        analyzer.update_with_position(&state, [0.6, 0.0], 0.15);

        // Second swing/stance cycle
        for i in 7..=11 {
            state.extremities[8] = 0.08;
            analyzer.update_with_position(&state, [0.1 * i as f64, 0.0], i as f64 * 0.025);
        }

        // Touchdown at position [1.3, 0] → stride = dist(0.6, 1.3) = 0.7
        state.extremities[8] = 0.0;
        analyzer.update_with_position(&state, [1.3, 0.0], 0.3);

        let summary = analyzer.summary();
        assert!(
            (summary.avg_stride_length - 0.7).abs() < 0.05,
            "Stride length should be ~0.7m: got {}",
            summary.avg_stride_length
        );
    }

    #[test]
    fn test_cadence_calculation() {
        let mut analyzer = GaitAnalyzer::new();
        let mut state = HumanoidState::standing();

        // Right foot touchdown cycle with 0.5s intervals
        // First touchdown at t=0
        state.extremities[8] = 0.0;
        analyzer.update_with_position(&state, [0.0, 0.0], 0.0);
        state.extremities[8] = 0.08; // swing
        analyzer.update_with_position(&state, [0.1, 0.0], 0.1);
        state.extremities[8] = 0.0; // touchdown at t=0.3 → first recorded
        analyzer.update_with_position(&state, [0.3, 0.0], 0.3);

        state.extremities[8] = 0.08; // swing
        analyzer.update_with_position(&state, [0.5, 0.0], 0.5);
        state.extremities[8] = 0.0; // touchdown at t=0.8 → interval=0.5s
        analyzer.update_with_position(&state, [0.8, 0.0], 0.8);

        let summary = analyzer.summary();
        // cadence = 1.0 / avg_interval = 1.0 / 0.5 = 2.0 steps/s
        assert!(
            (summary.avg_cadence - 2.0).abs() < 0.1,
            "Cadence should be ~2.0 steps/s: got {}",
            summary.avg_cadence
        );
    }

    #[test]
    fn test_gait_asymmetry_symmetric() {
        let mut analyzer = GaitAnalyzer::new();
        let mut state = HumanoidState::standing();

        // Right foot: two stride cycles with stride length ~0.5m
        state.extremities[8] = 0.0;
        analyzer.update_with_position(&state, [0.0, 0.0], 0.0);
        state.extremities[8] = 0.08;
        analyzer.update_with_position(&state, [0.2, 0.0], 0.1);
        state.extremities[8] = 0.0;
        analyzer.update_with_position(&state, [0.5, 0.0], 0.2);
        state.extremities[8] = 0.08;
        analyzer.update_with_position(&state, [0.7, 0.0], 0.3);
        state.extremities[8] = 0.0;
        analyzer.update_with_position(&state, [1.0, 0.0], 0.4);

        // Left foot: same stride length ~0.5m
        state.extremities[8] = 0.0; // keep right grounded
        state.extremities[11] = 0.0;
        analyzer.update_with_position(&state, [1.1, 0.0], 0.5);
        state.extremities[11] = 0.08;
        analyzer.update_with_position(&state, [1.3, 0.0], 0.6);
        state.extremities[11] = 0.0;
        analyzer.update_with_position(&state, [1.6, 0.0], 0.7);
        state.extremities[11] = 0.08;
        analyzer.update_with_position(&state, [1.8, 0.0], 0.8);
        state.extremities[11] = 0.0;
        analyzer.update_with_position(&state, [2.1, 0.0], 0.9);

        let summary = analyzer.summary();
        assert!(
            summary.gait_asymmetry < 0.1,
            "Equal L/R stride lengths should give low asymmetry: got {}",
            summary.gait_asymmetry
        );
    }

    #[test]
    fn test_gait_analyzer_reset() {
        let mut analyzer = GaitAnalyzer::new();

        // Add some data
        let mut state = HumanoidState::standing();
        state.extremities[8] = 0.0;
        analyzer.update(&state);
        state.extremities[8] = 0.1;
        analyzer.update(&state);
        state.extremities[8] = 0.0;
        analyzer.update(&state);

        assert!(analyzer.num_strides() > 0 || analyzer.total_steps > 0);

        analyzer.reset();
        assert_eq!(analyzer.num_strides(), 0);
        assert_eq!(analyzer.total_steps, 0);
        assert!((analyzer.avg_clearance() - 0.0).abs() < 1e-10);
    }

    // ── Step regularity tests ──

    #[test]
    fn test_step_regularity_uniform() {
        let mut analyzer = GaitAnalyzer::new();
        let mut state = HumanoidState::standing();

        // Create 4 right-foot touchdowns with uniform 0.5s intervals
        for i in 0..4 {
            let time = i as f64 * 0.5;
            state.extremities[8] = 0.0; // touchdown
            analyzer.update_with_position(&state, [i as f64 * 0.3, 0.0], time);
            state.extremities[8] = 0.08; // swing
            analyzer.update_with_position(&state, [i as f64 * 0.3 + 0.1, 0.0], time + 0.1);
        }
        // Final touchdown
        state.extremities[8] = 0.0;
        analyzer.update_with_position(&state, [1.5, 0.0], 2.0);

        let regularity = analyzer.step_regularity();
        assert!(
            regularity > 0.9,
            "Uniform step intervals should give high regularity: {regularity}"
        );
    }

    #[test]
    fn test_step_regularity_irregular() {
        let mut analyzer = GaitAnalyzer::new();
        let mut state = HumanoidState::standing();

        // Create touchdowns with irregular intervals: 0.2, 0.8, 0.3, 0.9
        let times = [0.0, 0.2, 1.0, 1.3, 2.2];
        for (i, &time) in times.iter().enumerate() {
            state.extremities[8] = 0.0; // touchdown
            analyzer.update_with_position(&state, [i as f64 * 0.3, 0.0], time);
            if i < times.len() - 1 {
                state.extremities[8] = 0.08; // swing
                analyzer.update_with_position(&state, [i as f64 * 0.3 + 0.1, 0.0], time + 0.05);
            }
        }

        let regularity = analyzer.step_regularity();
        assert!(
            regularity < 0.8,
            "Irregular step intervals should give low regularity: {regularity}"
        );
    }

    // ── Foot strike quality tests ──

    #[test]
    fn test_foot_strike_quality_dorsiflexion() {
        let mut analyzer = GaitAnalyzer::new();
        let mut state = HumanoidState::standing();

        // Ground phase: foot on ground with plantarflexion (good for toe-off)
        state.extremities[8] = 0.0;
        state.joint_angles[8] = -0.1; // plantarflexion
        analyzer.update_with_position(&state, [0.0, 0.0], 0.0);

        // Liftoff: foot lifts (stance→swing) — records liftoff with ankle_y=-0.1
        // Liftoff quality: sigmoid(-20 × (-0.1)) = sigmoid(2) ≈ 0.88 (good plantarflexion)
        state.extremities[8] = 0.08;
        state.joint_angles[8] = -0.1;
        analyzer.update_with_position(&state, [0.1, 0.0], 0.1);

        // Touchdown with dorsiflexion (positive ankle angle = heel-first)
        // Touchdown quality: sigmoid(20 × 0.1) = sigmoid(2) ≈ 0.88 (good dorsiflexion)
        state.extremities[8] = 0.0;
        state.joint_angles[8] = 0.1;
        analyzer.update_with_position(&state, [0.3, 0.0], 0.2);

        let quality = analyzer.foot_strike_quality();
        assert!(
            quality > 0.5,
            "Touchdown with dorsiflexion should give quality > 0.5: {quality}"
        );
    }

    #[test]
    fn test_foot_strike_quality_flat_foot() {
        let mut analyzer = GaitAnalyzer::new();
        let mut state = HumanoidState::standing();

        // Start in swing to avoid recording a liftoff event
        state.extremities[8] = 0.08;
        state.joint_angles[8] = 0.0;
        analyzer.update_with_position(&state, [0.0, 0.0], 0.0);

        state.extremities[8] = 0.08;
        analyzer.update_with_position(&state, [0.1, 0.0], 0.1);

        // Touchdown with flat foot (ankle angle ≈ 0)
        state.extremities[8] = 0.0;
        state.joint_angles[8] = 0.0;
        analyzer.update_with_position(&state, [0.3, 0.0], 0.2);

        let quality = analyzer.foot_strike_quality();
        assert!(
            (quality - 0.5).abs() < 0.1,
            "Flat foot touchdown should give quality ≈ 0.5: {quality}"
        );
    }
}

/// Contact-locked gait phase oscillator.
///
/// Pure wall-clock phase drifts away from actual touchdown under perturbations.
/// This oscillator advances continuously but re-anchors phase to observed
/// foot-strike events: right touchdown at phase 0, left touchdown at phase 0.5.
#[derive(Debug, Clone)]
pub struct ContactLockedGaitClock {
    phase: f64,
    previous_right_contact: bool,
    previous_left_contact: bool,
    contact_threshold_m: f64,
    correction_gain: f64,
}

impl Default for ContactLockedGaitClock {
    fn default() -> Self {
        Self {
            phase: 0.0,
            previous_right_contact: true,
            previous_left_contact: true,
            contact_threshold_m: 0.04,
            correction_gain: 0.65,
        }
    }
}

impl ContactLockedGaitClock {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn phase(&self) -> f64 {
        self.phase
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn advance(&mut self, state: &HumanoidState, dt: f64, nominal_hz: f64) -> f64 {
        let contacts = ContactFrame::estimated_from_state(state, self.contact_threshold_m);
        self.advance_with_contacts(&contacts, dt, nominal_hz)
    }

    pub fn advance_with_contacts(
        &mut self,
        contacts: &ContactFrame,
        dt: f64,
        nominal_hz: f64,
    ) -> f64 {
        if !dt.is_finite() || !nominal_hz.is_finite() || dt <= 0.0 || nominal_hz <= 0.0 {
            self.phase = 0.0;
            return self.phase;
        }
        self.phase = (self.phase + dt * nominal_hz).rem_euclid(1.0);

        let right_contact = contacts.right.in_contact;
        let left_contact = contacts.left.in_contact;
        let right_touchdown = right_contact && !self.previous_right_contact;
        let left_touchdown = left_contact && !self.previous_left_contact;

        if right_touchdown {
            self.phase = circular_blend(self.phase, 0.0, self.correction_gain);
        } else if left_touchdown {
            self.phase = circular_blend(self.phase, 0.5, self.correction_gain);
        }

        self.previous_right_contact = right_contact;
        self.previous_left_contact = left_contact;
        self.phase
    }
}

fn circular_blend(current: f64, target: f64, gain: f64) -> f64 {
    let mut delta = target - current;
    if delta > 0.5 {
        delta -= 1.0;
    } else if delta < -0.5 {
        delta += 1.0;
    }
    (current + gain.clamp(0.0, 1.0) * delta).rem_euclid(1.0)
}

#[cfg(test)]
mod contact_clock_tests {
    use super::*;

    #[test]
    fn advances_at_nominal_frequency() {
        let state = HumanoidState::standing();
        let mut clock = ContactLockedGaitClock::new();
        let phase = clock.advance(&state, 0.1, 1.0);
        assert!((phase - 0.1).abs() < 1.0e-9);
    }

    #[test]
    fn right_touchdown_reanchors_toward_zero() {
        let mut state = HumanoidState::standing();
        let mut clock = ContactLockedGaitClock::new();
        state.extremities[8] = 0.1;
        state.extremities[11] = 0.1;
        clock.advance(&state, 0.25, 1.0);
        state.extremities[8] = 0.0;
        let before = clock.phase();
        let after = clock.advance(&state, 0.01, 1.0);
        assert!(after < before);
    }
}
