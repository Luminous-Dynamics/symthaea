// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared temporal contract for cognitive-state smoothing and SVG animation.

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Nominal cognitive-loop frequency used when only a cycle counter is available.
pub const NOMINAL_CYCLES_PER_SECOND: f64 = 234.0;
/// Breathing period in cognitive cycles.
pub const BREATH_PERIOD_CYCLES: f64 = 120.0;

/// User or platform motion preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotionPreference {
    Full,
    Reduced,
}

impl Default for MotionPreference {
    fn default() -> Self {
        Self::Full
    }
}

/// One explicit visual frame on a monotonic timeline.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrameContext {
    /// Monotonic elapsed time since the visual session began.
    pub elapsed_seconds: f64,
    /// Time since the previous rendered frame.
    pub delta_seconds: f32,
    pub motion: MotionPreference,
    /// Optional stable renderer namespace for multi-instance embedding.
    pub instance_id: Option<u64>,
}

impl FrameContext {
    pub fn new(elapsed_seconds: f64, delta_seconds: f32) -> Self {
        Self {
            elapsed_seconds: finite_nonnegative_f64(elapsed_seconds),
            delta_seconds: finite_nonnegative_f32(delta_seconds),
            motion: MotionPreference::Full,
            instance_id: None,
        }
    }

    /// Compatibility bridge for callers that currently expose only cycle count.
    pub fn from_cycle_count(cycle_count: u64) -> Self {
        let elapsed_seconds = cycle_seconds(cycle_count);
        Self {
            elapsed_seconds,
            delta_seconds: (1.0 / NOMINAL_CYCLES_PER_SECOND) as f32,
            motion: MotionPreference::Full,
            instance_id: None,
        }
    }

    pub fn with_motion(mut self, motion: MotionPreference) -> Self {
        self.motion = motion;
        self
    }

    pub fn with_instance_id(mut self, instance_id: u64) -> Self {
        self.instance_id = Some(instance_id);
        self
    }

    pub fn breathing_phase(&self) -> f64 {
        breathing_phase_at_seconds(self.elapsed_seconds)
    }
}

/// Convert a cognitive cycle counter to nominal elapsed seconds.
pub fn cycle_seconds(cycle_count: u64) -> f64 {
    cycle_count as f64 / NOMINAL_CYCLES_PER_SECOND
}

/// Compute breathing oscillation from cycle count.
pub fn breathing_phase(cycle_count: u64) -> f64 {
    breathing_phase_at_seconds(cycle_seconds(cycle_count))
}

/// Compute breathing oscillation from monotonic elapsed seconds.
pub fn breathing_phase_at_seconds(elapsed_seconds: f64) -> f64 {
    let breath_period_seconds = BREATH_PERIOD_CYCLES / NOMINAL_CYCLES_PER_SECOND;
    let t = finite_nonnegative_f64(elapsed_seconds) * 2.0 * PI / breath_period_seconds;
    (t.sin() + 1.0) / 2.0
}

/// Frame-rate-independent EMA coefficient for time constant `tau_seconds`.
pub fn smoothing_alpha(delta_seconds: f32, tau_seconds: f32) -> f32 {
    let dt = finite_nonnegative_f32(delta_seconds);
    let tau = if tau_seconds.is_finite() && tau_seconds > 0.0 {
        tau_seconds
    } else {
        0.25
    };
    (1.0 - (-dt / tau).exp()).clamp(0.0, 1.0)
}

/// Smooth step easing: 3t² - 2t³
pub fn smooth_step(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Layer fade-in based on consciousness level and threshold.
/// Returns opacity in [0, 1].
pub fn layer_opacity(consciousness: f64, threshold: f64, fade_width: f64) -> f32 {
    let consciousness = if consciousness.is_finite() {
        consciousness
    } else {
        0.0
    };
    if consciousness <= threshold {
        return 0.0;
    }
    let width = if fade_width.is_finite() && fade_width > 0.0 {
        fade_width
    } else {
        1.0
    };
    let t = ((consciousness - threshold) / width).min(1.0);
    smooth_step(t as f32)
}

fn finite_nonnegative_f32(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

fn finite_nonnegative_f64(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breathing_bounded() {
        for c in 0..300 {
            let p = breathing_phase(c);
            assert!((0.0..=1.0).contains(&p), "cycle {c}: {p}");
        }
    }

    #[test]
    fn breathing_periodic() {
        let a = breathing_phase(0);
        let b = breathing_phase(BREATH_PERIOD_CYCLES as u64);
        assert!((a - b).abs() < 0.01);
    }

    #[test]
    fn frame_context_matches_cycle_bridge() {
        let cycle = 120;
        let frame = FrameContext::from_cycle_count(cycle);
        assert!((frame.breathing_phase() - breathing_phase(cycle)).abs() < 1e-9);
    }

    #[test]
    fn time_based_alpha_is_cadence_invariant() {
        let one_step = smoothing_alpha(1.0, 0.5);
        let half_step = smoothing_alpha(0.5, 0.5);
        let two_half_steps = 1.0 - (1.0 - half_step).powi(2);
        assert!((one_step - two_half_steps).abs() < 1e-5);
    }

    #[test]
    fn smooth_step_bounds() {
        assert_eq!(smooth_step(0.0), 0.0);
        assert_eq!(smooth_step(1.0), 1.0);
        assert!((smooth_step(0.5) - 0.5).abs() < 0.01);
    }

    #[test]
    fn layer_opacity_below_threshold() {
        assert_eq!(layer_opacity(0.1, 0.3, 0.2), 0.0);
    }

    #[test]
    fn layer_opacity_above_threshold() {
        let o = layer_opacity(0.6, 0.3, 0.2);
        assert!((o - 1.0).abs() < 0.01);
    }

    #[test]
    fn layer_opacity_mid_fade() {
        let o = layer_opacity(0.4, 0.3, 0.2);
        assert!(o > 0.0 && o < 1.0);
    }
}
