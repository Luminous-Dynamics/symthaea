// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! ISO/TS 15066 Speed and Separation Monitoring (SSM) baseline controller.
//!
//! Implements the protective separation distance formula from ISO/TS 15066:2016.
//! Binary stop/go: if human distance < Sp → full stop, else full speed.
//! This is the industry-standard approach that the Phi-gradient must outperform.

/// ISO/TS 15066 SSM controller parameters.
pub struct IsoSsmController {
    /// Human approach speed (m/s). ISO default: 1.6 m/s.
    pub human_speed: f64,
    /// Robot system reaction time (s). Typical: 0.1s.
    pub reaction_time: f64,
    /// Robot controlled stop duration (s). Typical: 0.3s.
    pub stop_time: f64,
    /// Safety margin: intrusion distance C + sensor uncertainty Zd + robot uncertainty Zr (m).
    pub safety_margin: f64,
}

impl Default for IsoSsmController {
    fn default() -> Self {
        Self {
            human_speed: 1.6,
            reaction_time: 0.1,
            stop_time: 0.3,
            safety_margin: 0.15, // C=0.1 + Zd=0.03 + Zr=0.02
        }
    }
}

impl IsoSsmController {
    /// Compute protective separation distance Sp (meters).
    ///
    /// Sp = Sh + Sr + Ss + C + Zd + Zr
    /// where:
    ///   Sh = vH × (TR + TS)     — distance human covers during latency + stopping
    ///   Sr = vR × TR             — distance robot covers during reaction latency
    ///   Ss = vR × TS / 2         — robot stopping distance (linear deceleration)
    pub fn protective_distance(&self, robot_speed: f64) -> f64 {
        let sh = self.human_speed * (self.reaction_time + self.stop_time);
        let sr = robot_speed * self.reaction_time;
        let ss = robot_speed * self.stop_time / 2.0;
        sh + sr + ss + self.safety_margin
    }

    /// Compute motor gain: 1.0 (full speed) or 0.0 (full stop).
    /// This is the binary nature of SSM — no intermediate states.
    pub fn motor_gain(&self, human_distance: f64, robot_speed: f64) -> f64 {
        let sp = self.protective_distance(robot_speed);
        if human_distance < sp { 0.0 } else { 1.0 }
    }

    /// Whether the robot should be stopped given current distances.
    pub fn should_stop(&self, human_distance: f64, robot_speed: f64) -> bool {
        human_distance < self.protective_distance(robot_speed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_parameters_are_conservative() {
        let ssm = IsoSsmController::default();
        assert_eq!(ssm.human_speed, 1.6);
        assert_eq!(ssm.reaction_time, 0.1);
        assert_eq!(ssm.stop_time, 0.3);
    }

    #[test]
    fn protective_distance_increases_with_robot_speed() {
        let ssm = IsoSsmController::default();
        let sp_slow = ssm.protective_distance(0.1);
        let sp_fast = ssm.protective_distance(1.0);
        assert!(sp_fast > sp_slow);
    }

    #[test]
    fn protective_distance_at_zero_robot_speed() {
        let ssm = IsoSsmController::default();
        let sp = ssm.protective_distance(0.0);
        // Should be: Sh + safety_margin = 1.6*(0.1+0.3) + 0.15 = 0.64 + 0.15 = 0.79m
        assert!((sp - 0.79).abs() < 0.01, "Sp = {sp}");
    }

    #[test]
    fn protective_distance_typical_scenario() {
        let ssm = IsoSsmController::default();
        // Robot at 0.5 m/s
        let sp = ssm.protective_distance(0.5);
        // Sh = 1.6*0.4 = 0.64, Sr = 0.5*0.1 = 0.05, Ss = 0.5*0.3/2 = 0.075
        // Total = 0.64 + 0.05 + 0.075 + 0.15 = 0.915
        assert!((sp - 0.915).abs() < 0.01, "Sp = {sp}");
    }

    #[test]
    fn binary_stop_go() {
        let ssm = IsoSsmController::default();
        let sp = ssm.protective_distance(0.5);
        // Just outside Sp → full speed
        assert_eq!(ssm.motor_gain(sp + 0.01, 0.5), 1.0);
        // Just inside Sp → full stop
        assert_eq!(ssm.motor_gain(sp - 0.01, 0.5), 0.0);
    }

    #[test]
    fn no_intermediate_gains() {
        let ssm = IsoSsmController::default();
        // Test many distances — gain is always exactly 0.0 or 1.0
        for d in 0..20 {
            let dist = d as f64 * 0.1;
            let gain = ssm.motor_gain(dist, 0.5);
            assert!(gain == 0.0 || gain == 1.0, "Gain {gain} at distance {dist}");
        }
    }
}
