// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Safety interlock: watchdog, e-stop, current limits, angle bounds.
//!
//! [`SafetyInterlock`] runs between the cognitive loop and [`ServoOutput`](crate::servo::ServoOutput),
//! enforcing hardware safety regardless of what the brain does.
//!
//! ```text
//! CognitiveLoop → HumanoidCommand → SafetyInterlock::filter_command() → ServoOutput::apply()
//! ```

use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Instant;

use symthaea_humanoid::types::{HumanoidCommand, JOINT_NAMES, NUM_ACTUATORS};
use tracing::{debug, warn};

use crate::error::{HalError, HalResult};

// ============================================================================
// SAFETY CONFIG
// ============================================================================

/// Safety interlock configuration.
#[derive(Debug, Clone)]
pub struct SafetyConfig {
    /// Watchdog timeout in milliseconds. If no command arrives within this
    /// window, the interlock trips.
    pub watchdog_timeout_ms: u64,
    /// Per-joint maximum torque magnitude (0.0–1.0). Commands are clamped to this.
    pub max_torque: [f32; NUM_ACTUATORS],
    /// Per-joint maximum angle in degrees (symmetric: ±limit).
    pub max_angle_deg: [f32; NUM_ACTUATORS],
    /// Per-joint current limit in amps (for overcurrent detection).
    pub max_current_a: [f32; NUM_ACTUATORS],
    /// Prediction error gain reduction factor (0.0–1.0). When prediction
    /// error exceeds threshold, torques are scaled by this factor.
    pub prediction_error_gain: f32,
    /// Prediction error threshold (above this, gain reduction kicks in).
    pub prediction_error_threshold: f32,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            watchdog_timeout_ms: 100, // 100ms → must receive command at ≥10Hz
            max_torque: [0.9; NUM_ACTUATORS],
            max_angle_deg: [90.0; NUM_ACTUATORS],
            max_current_a: [2.0; NUM_ACTUATORS],
            prediction_error_gain: 0.3,
            prediction_error_threshold: 0.5,
        }
    }
}

// ============================================================================
// SAFETY INTERLOCK
// ============================================================================

/// Hardware safety layer between cognition and servo output.
pub struct SafetyInterlock {
    config: SafetyConfig,
    /// Emergency stop flag (shared — can be set from any thread).
    estop: Arc<Mutex<bool>>,
    /// Timestamp of last accepted command.
    last_command_time: Instant,
    /// Current prediction error (set externally by cognitive loop).
    prediction_error: f32,
    /// Whether the interlock has tripped (requires explicit reset).
    tripped: bool,
    /// Reason for last trip.
    trip_reason: Option<String>,
}

impl SafetyInterlock {
    /// Create a new safety interlock with default configuration.
    pub fn new() -> Self {
        Self::with_config(SafetyConfig::default())
    }

    /// Create with explicit configuration.
    pub fn with_config(config: SafetyConfig) -> Self {
        Self {
            config,
            estop: Arc::new(Mutex::new(false)),
            last_command_time: Instant::now(),
            prediction_error: 0.0,
            tripped: false,
            trip_reason: None,
        }
    }

    /// Get a clone of the e-stop handle (can be shared across threads).
    pub fn estop_handle(&self) -> Arc<Mutex<bool>> {
        Arc::clone(&self.estop)
    }

    /// Trigger the emergency stop.
    pub fn trigger_estop(&self) {
        *self.estop.lock() = true;
        warn!("E-STOP triggered");
    }

    /// Release the emergency stop (still requires `reset()` to clear trip).
    pub fn release_estop(&self) {
        *self.estop.lock() = false;
        debug!("E-STOP released");
    }

    /// Whether e-stop is currently active.
    pub fn is_estopped(&self) -> bool {
        *self.estop.lock()
    }

    /// Update the prediction error from the cognitive loop.
    pub fn set_prediction_error(&mut self, error: f32) {
        self.prediction_error = error;
    }

    /// Whether the interlock has tripped.
    pub fn is_tripped(&self) -> bool {
        self.tripped
    }

    /// Reason for the last trip.
    pub fn trip_reason(&self) -> Option<&str> {
        self.trip_reason.as_deref()
    }

    /// Reset a tripped interlock (allows commands again).
    ///
    /// Returns an error if e-stop is still active.
    pub fn reset(&mut self) -> HalResult<()> {
        if self.is_estopped() {
            return Err(HalError::EStop);
        }
        self.tripped = false;
        self.trip_reason = None;
        self.last_command_time = Instant::now();
        debug!("safety interlock reset");
        Ok(())
    }

    /// Filter a command through all safety checks.
    ///
    /// On success, returns a (possibly modified) command that's safe to
    /// send to servos. On failure, returns an error and trips the interlock.
    pub fn filter_command(&mut self, command: &HumanoidCommand) -> HalResult<HumanoidCommand> {
        // 1. E-stop check
        if self.is_estopped() {
            self.trip("e-stop active");
            return Err(HalError::EStop);
        }

        // 2. Tripped check
        if self.tripped {
            return Err(HalError::Safety(format!(
                "interlock tripped: {}",
                self.trip_reason.as_deref().unwrap_or("unknown")
            )));
        }

        // 3. Watchdog check
        let elapsed = self.last_command_time.elapsed().as_millis() as u64;
        if elapsed > self.config.watchdog_timeout_ms {
            self.trip("watchdog timeout");
            return Err(HalError::WatchdogTimeout {
                elapsed_ms: elapsed,
                limit_ms: self.config.watchdog_timeout_ms,
            });
        }

        // 4. Apply torque limits and prediction error gain reduction
        let gain = if self.prediction_error > self.config.prediction_error_threshold {
            self.config.prediction_error_gain
        } else {
            1.0
        };

        let mut safe = command.clone();
        for i in 0..NUM_ACTUATORS {
            safe.torques[i] =
                safe.torques[i].clamp(-self.config.max_torque[i], self.config.max_torque[i]);
            safe.torques[i] *= gain;
        }

        // 5. Update watchdog timestamp
        self.last_command_time = Instant::now();

        Ok(safe)
    }

    /// Check a measured current reading against limits.
    ///
    /// Call this with actual current measurements if available.
    pub fn check_current(&mut self, joint: usize, amps: f32) -> HalResult<()> {
        if joint >= NUM_ACTUATORS {
            return Ok(());
        }
        if amps > self.config.max_current_a[joint] {
            self.trip(&format!(
                "overcurrent on joint {} ({})",
                joint, JOINT_NAMES[joint]
            ));
            return Err(HalError::Overcurrent {
                joint,
                name: JOINT_NAMES[joint],
                amps,
                limit: self.config.max_current_a[joint],
            });
        }
        Ok(())
    }

    /// Check a measured joint angle against bounds.
    pub fn check_angle(&mut self, joint: usize, angle_deg: f32) -> HalResult<()> {
        if joint >= NUM_ACTUATORS {
            return Ok(());
        }
        let limit = self.config.max_angle_deg[joint];
        if angle_deg.abs() > limit {
            self.trip(&format!(
                "angle bounds on joint {} ({})",
                joint, JOINT_NAMES[joint]
            ));
            return Err(HalError::AngleBounds {
                joint,
                name: JOINT_NAMES[joint],
                angle: angle_deg,
                min: -limit,
                max: limit,
            });
        }
        Ok(())
    }

    /// Get a reference to the config.
    pub fn config(&self) -> &SafetyConfig {
        &self.config
    }

    /// Get a mutable reference to the config.
    pub fn config_mut(&mut self) -> &mut SafetyConfig {
        &mut self.config
    }

    fn trip(&mut self, reason: &str) {
        if !self.tripped {
            warn!(reason, "safety interlock TRIPPED");
        }
        self.tripped = true;
        self.trip_reason = Some(reason.to_string());
    }
}

impl Default for SafetyInterlock {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_command_passes_normal() {
        let mut interlock = SafetyInterlock::new();
        let cmd = HumanoidCommand::zero();
        let result = interlock.filter_command(&cmd);
        assert!(result.is_ok());
    }

    #[test]
    fn test_filter_command_clamps_torque() {
        let mut interlock = SafetyInterlock::new();
        let mut cmd = HumanoidCommand::zero();
        cmd.torques[0] = 1.0; // max_torque default is 0.9
        let safe = interlock.filter_command(&cmd).unwrap();
        assert!(
            (safe.torques[0] - 0.9).abs() < 0.001,
            "torque should be clamped to 0.9, got {}",
            safe.torques[0]
        );
    }

    #[test]
    fn test_estop_trips_interlock() {
        let mut interlock = SafetyInterlock::new();
        interlock.trigger_estop();

        let cmd = HumanoidCommand::zero();
        let result = interlock.filter_command(&cmd);
        assert!(result.is_err());
        assert!(interlock.is_tripped());
    }

    #[test]
    fn test_estop_handle_shared() {
        let interlock = SafetyInterlock::new();
        let handle = interlock.estop_handle();

        // Trip from another thread
        *handle.lock() = true;
        assert!(interlock.is_estopped());
    }

    #[test]
    fn test_watchdog_timeout() {
        let mut config = SafetyConfig::default();
        config.watchdog_timeout_ms = 0; // instant timeout
        let mut interlock = SafetyInterlock::with_config(config);

        // Force time to elapse
        std::thread::sleep(std::time::Duration::from_millis(1));

        let cmd = HumanoidCommand::zero();
        let result = interlock.filter_command(&cmd);
        assert!(result.is_err());
        assert!(interlock.is_tripped());
    }

    #[test]
    fn test_reset_clears_trip() {
        let mut interlock = SafetyInterlock::new();
        interlock.trigger_estop();
        let _ = interlock.filter_command(&HumanoidCommand::zero()); // trips it

        // Can't reset while e-stop active
        assert!(interlock.reset().is_err());

        // Release e-stop, then reset
        interlock.release_estop();
        interlock.reset().unwrap();
        assert!(!interlock.is_tripped());

        // Commands work again
        let result = interlock.filter_command(&HumanoidCommand::zero());
        assert!(result.is_ok());
    }

    #[test]
    fn test_prediction_error_gain_reduction() {
        let mut interlock = SafetyInterlock::new();
        interlock.set_prediction_error(1.0); // above 0.5 threshold

        let mut cmd = HumanoidCommand::zero();
        cmd.torques[0] = 0.9;
        let safe = interlock.filter_command(&cmd).unwrap();

        // Torque should be reduced by gain factor (0.3)
        assert!(
            (safe.torques[0] - 0.27).abs() < 0.01,
            "expected ~0.27, got {}",
            safe.torques[0]
        );
    }

    #[test]
    fn test_prediction_error_below_threshold_no_reduction() {
        let mut interlock = SafetyInterlock::new();
        interlock.set_prediction_error(0.3); // below 0.5 threshold

        let mut cmd = HumanoidCommand::zero();
        cmd.torques[0] = 0.5;
        let safe = interlock.filter_command(&cmd).unwrap();

        assert!(
            (safe.torques[0] - 0.5).abs() < 0.001,
            "expected 0.5, got {}",
            safe.torques[0]
        );
    }

    #[test]
    fn test_check_current_normal() {
        let mut interlock = SafetyInterlock::new();
        assert!(interlock.check_current(0, 1.0).is_ok());
    }

    #[test]
    fn test_check_current_overcurrent() {
        let mut interlock = SafetyInterlock::new();
        let result = interlock.check_current(0, 3.0); // limit is 2.0
        assert!(result.is_err());
        assert!(interlock.is_tripped());
    }

    #[test]
    fn test_check_angle_normal() {
        let mut interlock = SafetyInterlock::new();
        assert!(interlock.check_angle(0, 45.0).is_ok());
    }

    #[test]
    fn test_check_angle_out_of_bounds() {
        let mut interlock = SafetyInterlock::new();
        let result = interlock.check_angle(0, 95.0); // limit is 90°
        assert!(result.is_err());
        assert!(interlock.is_tripped());
    }

    #[test]
    fn test_tripped_state_persists() {
        let mut interlock = SafetyInterlock::new();
        interlock.trigger_estop();
        let _ = interlock.filter_command(&HumanoidCommand::zero());

        // Even after releasing e-stop, tripped state persists
        interlock.release_estop();
        let result = interlock.filter_command(&HumanoidCommand::zero());
        assert!(result.is_err());

        // Must explicitly reset
        interlock.reset().unwrap();
        let result = interlock.filter_command(&HumanoidCommand::zero());
        assert!(result.is_ok());
    }

    #[test]
    fn test_trip_reason() {
        let mut interlock = SafetyInterlock::new();
        assert!(interlock.trip_reason().is_none());

        interlock.trigger_estop();
        let _ = interlock.filter_command(&HumanoidCommand::zero());
        assert_eq!(interlock.trip_reason(), Some("e-stop active"));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_torques() -> impl Strategy<Value = [f32; NUM_ACTUATORS]> {
        proptest::collection::vec(-2.0f32..2.0, NUM_ACTUATORS..=NUM_ACTUATORS).prop_map(|v| {
            let mut arr = [0.0f32; NUM_ACTUATORS];
            arr.copy_from_slice(&v);
            arr
        })
    }

    proptest! {
        #[test]
        fn filtered_torques_within_max(torques in arb_torques()) {
            let mut interlock = SafetyInterlock::new();
            let mut cmd = HumanoidCommand::zero();
            cmd.torques = torques.to_vec();

            if let Ok(safe) = interlock.filter_command(&cmd) {
                for i in 0..NUM_ACTUATORS {
                    let max = interlock.config().max_torque[i];
                    prop_assert!(
                        safe.torques[i].abs() <= max + f32::EPSILON,
                        "joint {} torque {} exceeds max {}",
                        i, safe.torques[i], max
                    );
                }
            }
        }

        #[test]
        fn filtered_torques_always_finite(torques in arb_torques()) {
            let mut interlock = SafetyInterlock::new();
            let mut cmd = HumanoidCommand::zero();
            cmd.torques = torques.to_vec();

            if let Ok(safe) = interlock.filter_command(&cmd) {
                for (i, &t) in safe.torques.iter().enumerate() {
                    prop_assert!(t.is_finite(), "joint {} torque {} not finite", i, t);
                }
            }
        }
    }
}
