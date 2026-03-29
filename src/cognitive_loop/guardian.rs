// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Physical Guardian — Embodied defensive posture system
//!
//! Maps SafetyLevel to physical robot behavior via HAL (servo control)
//! and CPG (locomotion patterns). Implements consciousness-gated
//! physical actions for infrastructure monitoring.
//!
//! ## Safety-to-Posture Mapping
//!
//! | SafetyLevel | Physical Behavior |
//! |-------------|-------------------|
//! | Green | Normal locomotion, patrol routes |
//! | Yellow | Heightened sensor sweep, cautious gait (walk) |
//! | Orange | Defensive posture, retract extremities, stop patrol |
//! | Red | Safe position (all servos centered), e-stop |
//!
//! ## Consciousness Gating
//!
//! All physical actions require minimum Phi:
//! - Phi < 0.3 → safe-hold (no locomotion)
//! - 0.3 ≤ Phi < 0.5 → observation only (sensors active, actuators locked)
//! - Phi ≥ 0.5 → patrol mode (locomotion + sensor sweep)
//! - Phi ≥ 0.6 → active response (navigate to threat, extend mesh, relay)
//!
//! Basis: NRC defense-in-depth — consciousness degradation limits actuation.

use serde::{Deserialize, Serialize};

use crate::safety::SafetyLevel;

/// Physical posture commands derived from safety level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuardianPosture {
    /// Normal operation — locomotion and patrol permitted.
    Normal,
    /// Cautious — walk gait only, increased sensor sweep.
    Cautious,
    /// Defensive — retract extremities, stop locomotion, sensor-only.
    Defensive,
    /// Emergency — safe servo position, e-stop engaged.
    Emergency,
    /// Hold — insufficient consciousness for any physical action.
    Hold,
}

impl GuardianPosture {
    /// Derive posture from safety level and consciousness (Phi).
    pub fn from_state(safety_level: SafetyLevel, phi: f64) -> Self {
        // Consciousness gating takes priority
        if phi < 0.3 {
            return Self::Hold;
        }

        match safety_level {
            SafetyLevel::Green => {
                if phi >= 0.5 {
                    Self::Normal
                } else {
                    Self::Cautious // Low Phi but safe → cautious
                }
            }
            SafetyLevel::Yellow => Self::Cautious,
            SafetyLevel::Orange => Self::Defensive,
            SafetyLevel::Red => Self::Emergency,
        }
    }

    /// Whether locomotion is permitted in this posture.
    pub fn locomotion_permitted(&self) -> bool {
        matches!(self, Self::Normal | Self::Cautious)
    }

    /// Whether sensor sweep should be active.
    pub fn sensor_sweep_active(&self) -> bool {
        !matches!(self, Self::Hold)
    }

    /// Whether emergency stop is engaged.
    pub fn e_stop_engaged(&self) -> bool {
        matches!(self, Self::Emergency | Self::Hold)
    }

    /// Recommended CPG gait for this posture.
    /// Returns None if locomotion is not permitted.
    pub fn recommended_gait(&self) -> Option<&'static str> {
        match self {
            Self::Normal => Some("walk"),   // default patrol gait
            Self::Cautious => Some("walk"), // cautious = walk only
            _ => None,                      // no locomotion
        }
    }

    /// Sensor sweep interval multiplier (1.0 = normal, <1.0 = faster).
    pub fn sensor_sweep_multiplier(&self) -> f32 {
        match self {
            Self::Normal => 1.0,
            Self::Cautious => 0.5,  // 2× faster sweep
            Self::Defensive => 0.3, // 3× faster sweep (looking for threats)
            Self::Emergency => 1.0, // maintain normal (conserve energy)
            Self::Hold => f32::MAX, // effectively disabled
        }
    }
}

/// Guardian state machine tracking posture transitions and patrol routes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardianState {
    /// Current physical posture.
    pub posture: GuardianPosture,
    /// Previous posture (for transition detection).
    pub previous_posture: GuardianPosture,
    /// Cycle at which current posture was entered.
    pub posture_since: usize,
    /// Whether patrol mode is active.
    pub patrol_active: bool,
    /// Current patrol waypoint index.
    pub patrol_waypoint: usize,
    /// Total patrol waypoints (0 = no route configured).
    pub patrol_waypoint_count: usize,
    /// Cumulative time in emergency posture (cycles).
    pub emergency_cycles: u64,
    /// Number of posture transitions.
    pub transition_count: u64,
}

impl Default for GuardianState {
    fn default() -> Self {
        Self {
            posture: GuardianPosture::Hold,
            previous_posture: GuardianPosture::Hold,
            posture_since: 0,
            patrol_active: false,
            patrol_waypoint: 0,
            patrol_waypoint_count: 0,
            emergency_cycles: 0,
            transition_count: 0,
        }
    }
}

impl GuardianState {
    /// Update posture based on current safety level and Phi.
    ///
    /// Returns true if posture changed.
    pub fn update(&mut self, safety_level: SafetyLevel, phi: f64, cycle: usize) -> bool {
        let new_posture = GuardianPosture::from_state(safety_level, phi);

        if new_posture != self.posture {
            self.previous_posture = self.posture;
            self.posture = new_posture;
            self.posture_since = cycle;
            self.transition_count += 1;

            // Disable patrol on defensive/emergency/hold
            if !new_posture.locomotion_permitted() {
                self.patrol_active = false;
            }

            return true;
        }

        // Track emergency time
        if self.posture == GuardianPosture::Emergency {
            self.emergency_cycles += 1;
        }

        false
    }

    /// Advance to next patrol waypoint (wraps around).
    pub fn advance_waypoint(&mut self) {
        if self.patrol_waypoint_count > 0 {
            self.patrol_waypoint = (self.patrol_waypoint + 1) % self.patrol_waypoint_count;
        }
    }

    /// Start patrol with the given number of waypoints.
    pub fn start_patrol(&mut self, waypoint_count: usize) {
        if self.posture.locomotion_permitted() && waypoint_count > 0 {
            self.patrol_active = true;
            self.patrol_waypoint = 0;
            self.patrol_waypoint_count = waypoint_count;
        }
    }

    /// Stop patrol.
    pub fn stop_patrol(&mut self) {
        self.patrol_active = false;
    }

    /// Generate telemetry snapshot.
    pub fn telemetry(&self, current_cycle: usize) -> GuardianTelemetry {
        GuardianTelemetry {
            posture: format!("{:?}", self.posture),
            locomotion_permitted: self.posture.locomotion_permitted(),
            e_stop_engaged: self.posture.e_stop_engaged(),
            patrol_active: self.patrol_active,
            patrol_waypoint: self.patrol_waypoint,
            sensor_sweep_multiplier: self.posture.sensor_sweep_multiplier(),
            posture_duration: current_cycle.saturating_sub(self.posture_since),
            emergency_cycles: self.emergency_cycles,
            transition_count: self.transition_count,
        }
    }
}

/// Telemetry for the guardian subsystem.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GuardianTelemetry {
    /// Current posture name.
    pub posture: String,
    /// Whether locomotion is permitted.
    pub locomotion_permitted: bool,
    /// Whether e-stop is engaged.
    pub e_stop_engaged: bool,
    /// Whether patrol is active.
    pub patrol_active: bool,
    /// Current patrol waypoint.
    pub patrol_waypoint: usize,
    /// Sensor sweep multiplier.
    pub sensor_sweep_multiplier: f32,
    /// Cycles in current posture.
    pub posture_duration: usize,
    /// Total emergency cycles.
    pub emergency_cycles: u64,
    /// Total posture transitions.
    pub transition_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_posture_from_state_low_phi() {
        assert_eq!(
            GuardianPosture::from_state(SafetyLevel::Green, 0.1),
            GuardianPosture::Hold
        );
        assert_eq!(
            GuardianPosture::from_state(SafetyLevel::Red, 0.1),
            GuardianPosture::Hold
        );
    }

    #[test]
    fn test_posture_from_state_green() {
        assert_eq!(
            GuardianPosture::from_state(SafetyLevel::Green, 0.8),
            GuardianPosture::Normal
        );
        assert_eq!(
            GuardianPosture::from_state(SafetyLevel::Green, 0.35),
            GuardianPosture::Cautious
        );
    }

    #[test]
    fn test_posture_from_state_yellow() {
        assert_eq!(
            GuardianPosture::from_state(SafetyLevel::Yellow, 0.8),
            GuardianPosture::Cautious
        );
    }

    #[test]
    fn test_posture_from_state_orange() {
        assert_eq!(
            GuardianPosture::from_state(SafetyLevel::Orange, 0.8),
            GuardianPosture::Defensive
        );
    }

    #[test]
    fn test_posture_from_state_red() {
        assert_eq!(
            GuardianPosture::from_state(SafetyLevel::Red, 0.8),
            GuardianPosture::Emergency
        );
    }

    #[test]
    fn test_locomotion_permitted() {
        assert!(GuardianPosture::Normal.locomotion_permitted());
        assert!(GuardianPosture::Cautious.locomotion_permitted());
        assert!(!GuardianPosture::Defensive.locomotion_permitted());
        assert!(!GuardianPosture::Emergency.locomotion_permitted());
        assert!(!GuardianPosture::Hold.locomotion_permitted());
    }

    #[test]
    fn test_e_stop_engaged() {
        assert!(!GuardianPosture::Normal.e_stop_engaged());
        assert!(!GuardianPosture::Cautious.e_stop_engaged());
        assert!(!GuardianPosture::Defensive.e_stop_engaged());
        assert!(GuardianPosture::Emergency.e_stop_engaged());
        assert!(GuardianPosture::Hold.e_stop_engaged());
    }

    #[test]
    fn test_guardian_state_update_transition() {
        let mut state = GuardianState::default();

        // Start at Hold (default, Phi too low)
        assert_eq!(state.posture, GuardianPosture::Hold);

        // Transition to Normal (high Phi, Green safety)
        let changed = state.update(SafetyLevel::Green, 0.8, 100);
        assert!(changed);
        assert_eq!(state.posture, GuardianPosture::Normal);
        assert_eq!(state.previous_posture, GuardianPosture::Hold);
        assert_eq!(state.posture_since, 100);

        // No change on same state
        let changed = state.update(SafetyLevel::Green, 0.8, 101);
        assert!(!changed);
    }

    #[test]
    fn test_guardian_state_emergency_tracking() {
        let mut state = GuardianState::default();
        state.update(SafetyLevel::Red, 0.8, 0);

        for i in 1..10 {
            state.update(SafetyLevel::Red, 0.8, i);
        }
        assert_eq!(state.emergency_cycles, 9);
    }

    #[test]
    fn test_patrol_lifecycle() {
        let mut state = GuardianState::default();
        state.update(SafetyLevel::Green, 0.8, 0);

        state.start_patrol(5);
        assert!(state.patrol_active);
        assert_eq!(state.patrol_waypoint, 0);

        state.advance_waypoint();
        assert_eq!(state.patrol_waypoint, 1);

        state.advance_waypoint();
        state.advance_waypoint();
        state.advance_waypoint();
        state.advance_waypoint(); // wraps
        assert_eq!(state.patrol_waypoint, 0);
    }

    #[test]
    fn test_patrol_stops_on_defensive() {
        let mut state = GuardianState::default();
        state.update(SafetyLevel::Green, 0.8, 0);
        state.start_patrol(5);
        assert!(state.patrol_active);

        state.update(SafetyLevel::Orange, 0.8, 10);
        assert!(!state.patrol_active);
    }

    #[test]
    fn test_patrol_not_started_when_holding() {
        let mut state = GuardianState::default();
        // Hold posture (low Phi)
        state.start_patrol(5);
        assert!(!state.patrol_active);
    }

    #[test]
    fn test_sensor_sweep_multiplier() {
        assert!((GuardianPosture::Normal.sensor_sweep_multiplier() - 1.0).abs() < 1e-6);
        assert!(GuardianPosture::Cautious.sensor_sweep_multiplier() < 1.0);
        assert!(
            GuardianPosture::Defensive.sensor_sweep_multiplier()
                < GuardianPosture::Cautious.sensor_sweep_multiplier()
        );
    }

    #[test]
    fn test_telemetry() {
        let mut state = GuardianState::default();
        state.update(SafetyLevel::Green, 0.8, 100);
        state.start_patrol(3);

        let telem = state.telemetry(150);
        assert_eq!(telem.posture, "Normal");
        assert!(telem.locomotion_permitted);
        assert!(!telem.e_stop_engaged);
        assert!(telem.patrol_active);
        assert_eq!(telem.posture_duration, 50);
    }

    #[test]
    fn test_recommended_gait() {
        assert_eq!(GuardianPosture::Normal.recommended_gait(), Some("walk"));
        assert_eq!(GuardianPosture::Cautious.recommended_gait(), Some("walk"));
        assert_eq!(GuardianPosture::Defensive.recommended_gait(), None);
        assert_eq!(GuardianPosture::Emergency.recommended_gait(), None);
    }
}
