// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Zero-Shot Embodiment Transfer — consciousness migration between bodies.
//!
//! When a consciousness is transferred to a new body, prediction error spikes
//! because the motor model doesn't match. The FEP agent adapts via precision
//! modulation, gradually learning the new body's dynamics. Safety stays at
//! Red until prediction error converges below threshold.
//!
//! The 16,384D HDC encoding is the transfer medium — both source and target
//! platforms encode perception into the same space. The CfC temporal dynamics
//! and genesis-seeded network weights carry over unchanged.
//!
//! Science: Putnam (1967) multiple realizability, Clark (2008) extended mind,
//! Friston (2010) free energy principle for embodied adaptation.

use symthaea_core::embodiment::{EmbodimentPlatform, MotorSafetyLevel};

/// Tracks the adaptation state during embodiment transfer.
#[derive(Debug, Clone)]
pub struct TransferState {
    /// Source platform (where consciousness came from).
    pub source_platform: EmbodimentPlatform,
    /// Target platform (new body).
    pub target_platform: EmbodimentPlatform,
    /// Exponential moving average of prediction error (0.0–1.0).
    pub prediction_error_ema: f32,
    /// Number of cycles since transfer.
    pub cycles_since_transfer: u64,
    /// Whether the transfer has converged (PE_ema < threshold).
    pub converged: bool,
    /// Transfer confidence (0.0–1.0): increases as PE_ema decreases.
    pub transfer_confidence: f64,
    /// Safety override: Red until converged, then release.
    safety_locked: bool,
}

impl TransferState {
    /// Prediction error EMA threshold for convergence.
    const CONVERGENCE_THRESHOLD: f32 = 0.3;
    /// EMA smoothing factor (closer to 0 = smoother, slower).
    const EMA_ALPHA: f32 = 0.1;
    /// Minimum cycles before convergence is possible (allow FEP to adapt).
    const MIN_CONVERGENCE_CYCLES: u64 = 20;

    /// Begin a new transfer.
    pub fn begin(source: EmbodimentPlatform, target: EmbodimentPlatform) -> Self {
        Self {
            source_platform: source,
            target_platform: target,
            prediction_error_ema: 1.0, // Start high
            cycles_since_transfer: 0,
            converged: false,
            transfer_confidence: 0.0,
            safety_locked: true,
        }
    }

    /// Update with the latest prediction error from the embodiment step.
    ///
    /// Returns the safety override level:
    /// - `Some(Red)` while adapting
    /// - `None` once converged (release to normal Phi gating)
    pub fn tick(&mut self, prediction_error: f32) -> Option<MotorSafetyLevel> {
        self.cycles_since_transfer += 1;

        // EMA update
        self.prediction_error_ema = Self::EMA_ALPHA * prediction_error
            + (1.0 - Self::EMA_ALPHA) * self.prediction_error_ema;

        // Confidence = 1 - PE_ema (clamped)
        self.transfer_confidence = (1.0 - self.prediction_error_ema as f64).clamp(0.0, 1.0);

        // Check convergence
        if !self.converged
            && self.cycles_since_transfer >= Self::MIN_CONVERGENCE_CYCLES
            && self.prediction_error_ema < Self::CONVERGENCE_THRESHOLD
        {
            self.converged = true;
            self.safety_locked = false;
        }

        if self.safety_locked {
            Some(MotorSafetyLevel::Red)
        } else {
            None
        }
    }

    /// Whether the consciousness has adapted to the new body.
    pub fn is_converged(&self) -> bool {
        self.converged
    }

    /// Time in cycles to convergence (or current cycle count if not yet converged).
    pub fn adaptation_cycles(&self) -> u64 {
        self.cycles_since_transfer
    }
}

/// Transfer telemetry for monitoring.
#[derive(Debug, Clone, Default)]
pub struct TransferTelemetry {
    pub active: bool,
    pub source_platform: String,
    pub target_platform: String,
    pub prediction_error_ema: f32,
    pub cycles_since_transfer: u64,
    pub converged: bool,
    pub transfer_confidence: f64,
}

impl From<&TransferState> for TransferTelemetry {
    fn from(state: &TransferState) -> Self {
        Self {
            active: true,
            source_platform: format!("{:?}", state.source_platform),
            target_platform: format!("{:?}", state.target_platform),
            prediction_error_ema: state.prediction_error_ema,
            cycles_since_transfer: state.cycles_since_transfer,
            converged: state.converged,
            transfer_confidence: state.transfer_confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_starts_locked() {
        let state = TransferState::begin(EmbodimentPlatform::Auv, EmbodimentPlatform::Quadrotor);
        assert!(!state.converged);
        assert!(state.safety_locked);
        assert_eq!(state.prediction_error_ema, 1.0);
    }

    #[test]
    fn test_transfer_converges_with_low_pe() {
        let mut state =
            TransferState::begin(EmbodimentPlatform::Auv, EmbodimentPlatform::Quadrotor);

        // Feed consistently low prediction errors
        for _ in 0..50 {
            state.tick(0.1);
        }

        assert!(state.converged, "Should converge after 50 cycles of low PE");
        assert!(state.transfer_confidence > 0.5);
        assert_eq!(state.tick(0.1), None, "Safety should be released");
    }

    #[test]
    fn test_transfer_stays_locked_with_high_pe() {
        let mut state =
            TransferState::begin(EmbodimentPlatform::Humanoid, EmbodimentPlatform::Vehicle);

        // Feed high prediction errors
        for _ in 0..100 {
            let override_level = state.tick(0.8);
            assert_eq!(override_level, Some(MotorSafetyLevel::Red));
        }

        assert!(!state.converged);
    }

    #[test]
    fn test_transfer_requires_min_cycles() {
        let mut state =
            TransferState::begin(EmbodimentPlatform::Auv, EmbodimentPlatform::Quadrotor);

        // Even with zero PE, should not converge before MIN_CONVERGENCE_CYCLES
        for i in 0..TransferState::MIN_CONVERGENCE_CYCLES {
            state.tick(0.0);
            if i < TransferState::MIN_CONVERGENCE_CYCLES - 1 {
                assert!(!state.converged, "Should not converge at cycle {i}");
            }
        }

        // Now it should converge
        assert!(state.converged);
    }

    #[test]
    fn test_transfer_telemetry() {
        let state = TransferState::begin(
            EmbodimentPlatform::Manipulator,
            EmbodimentPlatform::Helicopter,
        );
        let tel: TransferTelemetry = (&state).into();
        assert!(tel.active);
        assert_eq!(tel.source_platform, "Manipulator");
        assert_eq!(tel.target_platform, "Helicopter");
    }
}
