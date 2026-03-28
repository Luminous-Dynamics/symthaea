// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Rotor dynamics model for SAR helicopter.
//!
//! Models the key physics that differentiate helicopters from quadrotors:
//! - RPM lag: rotor inertia creates first-order response delay
//! - Gyroscopic precession: applied torque appears 90° from tilt input
//! - Torque reaction: main rotor torque creates yaw (countered by tail rotor)
//! - Autorotation: emergency descent when RPM drops below threshold

use serde::{Deserialize, Serialize};

/// Rotor dynamics parameters for a SAR helicopter (~500kg class).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotorDynamicsConfig {
    /// Main rotor maximum RPM.
    pub max_main_rpm: f64,
    /// Tail rotor maximum RPM.
    pub max_tail_rpm: f64,
    /// Main rotor time constant (seconds). Larger = slower RPM response.
    /// Typical SAR helo: 0.3–0.8s.
    pub main_rotor_tau: f64,
    /// Tail rotor time constant (seconds). Smaller = faster anti-torque response.
    pub tail_rotor_tau: f64,
    /// Main rotor thrust coefficient: thrust_N = coeff × RPM² × collective.
    pub thrust_coefficient: f64,
    /// Torque reaction coefficient: yaw_torque = coeff × RPM².
    /// The main rotor creates a reactive yaw torque countered by the tail rotor.
    pub torque_reaction_coefficient: f64,
    /// Gyroscopic precession gain: applied moment appears 90° ahead of tilt input.
    pub precession_gain: f64,
    /// Autorotation RPM threshold: below this, rotor enters autorotation regime.
    pub autorotation_rpm_threshold: f64,
    /// Autorotation descent rate (m/s, negative = descending).
    pub autorotation_descent_rate: f64,
}

impl Default for RotorDynamicsConfig {
    fn default() -> Self {
        Self {
            max_main_rpm: 5500.0,
            max_tail_rpm: 4000.0,
            main_rotor_tau: 0.5,        // 500ms to reach target RPM
            tail_rotor_tau: 0.2,        // 200ms (faster for yaw control)
            thrust_coefficient: 3.5e-7, // ~5000N at 3500 RPM, coll=0.3
            torque_reaction_coefficient: 1.0e-8,
            precession_gain: 0.15, // 15% of cyclic input appears as precession
            autorotation_rpm_threshold: 1500.0,
            autorotation_descent_rate: -5.0, // ~5 m/s descent in autorotation
        }
    }
}

/// Rotor dynamics state (evolves each physics step).
#[derive(Debug, Clone)]
pub struct RotorDynamicsState {
    /// Current main rotor RPM.
    pub main_rpm: f64,
    /// Current tail rotor RPM.
    pub tail_rpm: f64,
    /// Whether autorotation regime is active.
    pub in_autorotation: bool,
    /// Cumulative main rotor revolutions (for diagnostics).
    pub main_revolutions: f64,
}

impl RotorDynamicsState {
    /// Create initial state at hover RPM.
    pub fn hover() -> Self {
        Self {
            main_rpm: 3500.0,
            tail_rpm: 2000.0,
            in_autorotation: false,
            main_revolutions: 0.0,
        }
    }

    /// Create grounded state (rotors off).
    pub fn grounded() -> Self {
        Self {
            main_rpm: 0.0,
            tail_rpm: 0.0,
            in_autorotation: false,
            main_revolutions: 0.0,
        }
    }
}

/// Rotor dynamics model.
///
/// Computes thrust, torque, and gyroscopic effects from rotor state + commands.
pub struct RotorDynamics {
    config: RotorDynamicsConfig,
    state: RotorDynamicsState,
}

impl RotorDynamics {
    /// Create with default config at hover state.
    pub fn new() -> Self {
        Self {
            config: RotorDynamicsConfig::default(),
            state: RotorDynamicsState::hover(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: RotorDynamicsConfig) -> Self {
        Self {
            config,
            state: RotorDynamicsState::hover(),
        }
    }

    /// Current rotor state.
    pub fn state(&self) -> &RotorDynamicsState {
        &self.state
    }

    /// Config access.
    pub fn config(&self) -> &RotorDynamicsConfig {
        &self.config
    }

    /// Mutable state access (for testing and perturbations).
    pub fn state_mut(&mut self) -> &mut RotorDynamicsState {
        &mut self.state
    }

    /// Step the rotor dynamics forward by dt seconds.
    ///
    /// `thrust_cmd` ∈ [0, 1]: fraction of max main rotor RPM.
    /// `tail_cmd` ∈ [0, 1]: fraction of max tail rotor RPM.
    /// `collective` ∈ [-1, 1]: blade pitch angle.
    ///
    /// Returns (thrust_force_N, torque_reaction_Nm, precession_roll_Nm, precession_pitch_Nm).
    pub fn step(
        &mut self,
        thrust_cmd: f64,
        tail_cmd: f64,
        collective: f64,
        cyclic_lon: f64,
        cyclic_lat: f64,
        dt: f64,
    ) -> RotorOutput {
        // 1. RPM lag: first-order exponential filter
        let target_main = thrust_cmd * self.config.max_main_rpm;
        let target_tail = tail_cmd * self.config.max_tail_rpm;

        let alpha_main = 1.0 - (-dt / self.config.main_rotor_tau).exp();
        let alpha_tail = 1.0 - (-dt / self.config.tail_rotor_tau).exp();

        self.state.main_rpm += alpha_main * (target_main - self.state.main_rpm);
        self.state.tail_rpm += alpha_tail * (target_tail - self.state.tail_rpm);

        // Clamp to non-negative
        self.state.main_rpm = self.state.main_rpm.max(0.0);
        self.state.tail_rpm = self.state.tail_rpm.max(0.0);

        // Track revolutions
        self.state.main_revolutions += self.state.main_rpm / 60.0 * dt;

        // 2. Autorotation check
        self.state.in_autorotation =
            self.state.main_rpm < self.config.autorotation_rpm_threshold && thrust_cmd < 0.1;

        if self.state.in_autorotation {
            return RotorOutput {
                thrust_force: 0.0,
                torque_reaction: 0.0,
                precession_roll: 0.0,
                precession_pitch: 0.0,
                descent_rate_override: Some(self.config.autorotation_descent_rate),
            };
        }

        // 3. Thrust from RPM² × collective
        // thrust = coeff × RPM² × (collective + offset) where offset shifts
        // collective=0 to still produce some lift at hover RPM
        let effective_collective = (collective + 0.5).max(0.0); // Shift so hover collective ~0.3 → 0.8
        let thrust_force =
            self.config.thrust_coefficient * self.state.main_rpm.powi(2) * effective_collective;

        // 4. Torque reaction: main rotor creates yaw torque
        let torque_reaction = self.config.torque_reaction_coefficient * self.state.main_rpm.powi(2);

        // 5. Gyroscopic precession: cyclic input appears 90° ahead
        // Longitudinal cyclic → roll precession, lateral cyclic → pitch precession
        let precession_roll =
            cyclic_lon * self.config.precession_gain * self.state.main_rpm / 1000.0;
        let precession_pitch =
            -cyclic_lat * self.config.precession_gain * self.state.main_rpm / 1000.0;

        RotorOutput {
            thrust_force,
            torque_reaction,
            precession_roll,
            precession_pitch,
            descent_rate_override: None,
        }
    }

    /// Reset to hover state.
    pub fn reset(&mut self) {
        self.state = RotorDynamicsState::hover();
    }

    /// Reset to grounded state.
    pub fn reset_grounded(&mut self) {
        self.state = RotorDynamicsState::grounded();
    }
}

impl Default for RotorDynamics {
    fn default() -> Self {
        Self::new()
    }
}

/// Output from one rotor dynamics step.
#[derive(Debug, Clone, Copy)]
pub struct RotorOutput {
    /// Thrust force in Newtons (along body z-axis).
    pub thrust_force: f64,
    /// Reactive yaw torque from main rotor (Nm).
    pub torque_reaction: f64,
    /// Gyroscopic precession in roll axis (Nm).
    pub precession_roll: f64,
    /// Gyroscopic precession in pitch axis (Nm).
    pub precession_pitch: f64,
    /// If autorotation, overrides descent rate (m/s, negative).
    pub descent_rate_override: Option<f64>,
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotor_hover_produces_thrust() {
        let mut rotor = RotorDynamics::new();
        let output = rotor.step(0.6, 0.5, 0.3, 0.0, 0.0, 0.01);
        assert!(
            output.thrust_force > 0.0,
            "Hover should produce positive thrust"
        );
        assert!(output.descent_rate_override.is_none());
    }

    #[test]
    fn test_rotor_zero_command_no_thrust() {
        let mut rotor = RotorDynamics::new();
        // Run several steps to let RPM decay toward zero
        for _ in 0..1000 {
            rotor.step(0.0, 0.0, -0.5, 0.0, 0.0, 0.01);
        }
        let output = rotor.step(0.0, 0.0, -0.5, 0.0, 0.0, 0.01);
        // Thrust should be very low (RPM decayed toward 0)
        assert!(output.thrust_force < 10.0);
    }

    #[test]
    fn test_rpm_lag() {
        let mut rotor = RotorDynamics::new();
        let initial_rpm = rotor.state().main_rpm;

        // Command max thrust
        rotor.step(1.0, 0.5, 0.3, 0.0, 0.0, 0.001);
        let after_1ms = rotor.state().main_rpm;

        // RPM should increase but not instantly reach max (lag)
        assert!(after_1ms > initial_rpm);
        assert!(after_1ms < rotor.config().max_main_rpm);
    }

    #[test]
    fn test_rpm_convergence() {
        let mut rotor = RotorDynamics::new();
        // Command 80% thrust for 10 seconds
        for _ in 0..10000 {
            rotor.step(0.8, 0.5, 0.3, 0.0, 0.0, 0.001);
        }
        let target = 0.8 * rotor.config().max_main_rpm;
        let actual = rotor.state().main_rpm;
        assert!(
            (actual - target).abs() < 10.0,
            "RPM should converge to target: actual={actual}, target={target}"
        );
    }

    #[test]
    fn test_autorotation_regime() {
        let mut rotor = RotorDynamics::with_config(RotorDynamicsConfig {
            autorotation_rpm_threshold: 1500.0,
            ..Default::default()
        });
        // Set RPM below threshold
        rotor.state.main_rpm = 500.0;
        let output = rotor.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.01);
        assert!(rotor.state().in_autorotation);
        assert!(output.descent_rate_override.is_some());
        assert!(output.thrust_force == 0.0);
    }

    #[test]
    fn test_torque_reaction_increases_with_rpm() {
        let mut rotor = RotorDynamics::new();
        rotor.state.main_rpm = 2000.0;
        let out_low = rotor.step(0.6, 0.5, 0.3, 0.0, 0.0, 0.01);

        rotor.state.main_rpm = 4000.0;
        let out_high = rotor.step(0.6, 0.5, 0.3, 0.0, 0.0, 0.01);

        assert!(
            out_high.torque_reaction > out_low.torque_reaction,
            "Higher RPM should produce more torque reaction"
        );
    }

    #[test]
    fn test_precession_from_cyclic() {
        let mut rotor = RotorDynamics::new();
        // Longitudinal cyclic → roll precession
        let output = rotor.step(0.6, 0.5, 0.3, 0.5, 0.0, 0.01);
        assert!(
            output.precession_roll.abs() > 0.0,
            "Cyclic lon should produce roll precession"
        );

        // Lateral cyclic → pitch precession
        let output = rotor.step(0.6, 0.5, 0.3, 0.0, 0.5, 0.01);
        assert!(
            output.precession_pitch.abs() > 0.0,
            "Cyclic lat should produce pitch precession"
        );
    }

    #[test]
    fn test_reset_to_hover() {
        let mut rotor = RotorDynamics::new();
        rotor.state.main_rpm = 100.0;
        rotor.state.in_autorotation = true;
        rotor.reset();
        assert_eq!(rotor.state().main_rpm, 3500.0);
        assert!(!rotor.state().in_autorotation);
    }

    #[test]
    fn test_revolutions_accumulate() {
        let mut rotor = RotorDynamics::new();
        let before = rotor.state().main_revolutions;
        // At 3500 RPM for 1 second = 3500/60 ≈ 58.3 revolutions
        for _ in 0..1000 {
            rotor.step(0.6, 0.5, 0.3, 0.0, 0.0, 0.001);
        }
        let after = rotor.state().main_revolutions;
        assert!(
            after > before + 50.0,
            "Should accumulate ~58 revolutions in 1s at 3500 RPM"
        );
    }
}
