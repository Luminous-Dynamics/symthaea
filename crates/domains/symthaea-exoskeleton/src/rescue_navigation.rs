// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Relative-navigation and separation detection for SX-014.
//!
//! This module can recommend only bounded rescue modes. It does not contain
//! autonomous return-to-target guidance or collision-avoidance authority.

use serde::{Deserialize, Serialize};

use crate::rescue_propulsion::RescueMode;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RelativeNavigationState {
    /// Target-relative position in a declared local frame, metres.
    pub relative_position_m: [f64; 3],
    /// Target-relative velocity in the same frame, m/s.
    pub relative_velocity_m_s: [f64; 3],
    /// Body angular rate from the independent rescue IMU, rad/s.
    pub angular_rate_rad_s: [f64; 3],
    /// Tether load. Near zero can indicate release/failure, depending on the
    /// declared tether protocol.
    pub tether_tension_n: f64,
    /// Fused navigation quality in [0, 1].
    pub navigation_quality: f64,
    /// Age of the newest state estimate, seconds.
    pub sample_age_s: f64,
}

impl RelativeNavigationState {
    pub fn is_structurally_valid(&self) -> bool {
        finite3(self.relative_position_m)
            && finite3(self.relative_velocity_m_s)
            && finite3(self.angular_rate_rad_s)
            && self.tether_tension_n.is_finite()
            && self.tether_tension_n >= 0.0
            && self.navigation_quality.is_finite()
            && (0.0..=1.0).contains(&self.navigation_quality)
            && self.sample_age_s.is_finite()
            && self.sample_age_s >= 0.0
    }

    pub fn relative_speed_m_s(&self) -> f64 {
        norm3(self.relative_velocity_m_s)
    }

    pub fn angular_speed_rad_s(&self) -> f64 {
        norm3(self.angular_rate_rad_s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SeparationPolicy {
    pub min_navigation_quality: f64,
    pub max_sample_age_s: f64,
    pub tether_attached_threshold_n: f64,
    pub uncommanded_separation_speed_m_s: f64,
    pub detumble_rate_threshold_rad_s: f64,
    /// Maximum automatic drift-arrest delta-v that may be requested by this
    /// trigger layer. The propulsion kernel imposes its own independent total
    /// budget as well.
    pub max_automatic_arrest_delta_v_m_s: f64,
}

impl SeparationPolicy {
    /// Simulation-only thresholds for deterministic software tests.
    pub fn simulation_reference() -> Self {
        Self {
            min_navigation_quality: 0.8,
            max_sample_age_s: 0.25,
            tether_attached_threshold_n: 2.0,
            uncommanded_separation_speed_m_s: 0.05,
            detumble_rate_threshold_rad_s: 0.15,
            max_automatic_arrest_delta_v_m_s: 0.25,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.min_navigation_quality.is_finite()
            && (0.0..=1.0).contains(&self.min_navigation_quality)
            && self.max_sample_age_s.is_finite()
            && self.max_sample_age_s > 0.0
            && self.tether_attached_threshold_n.is_finite()
            && self.tether_attached_threshold_n >= 0.0
            && self.uncommanded_separation_speed_m_s.is_finite()
            && self.uncommanded_separation_speed_m_s >= 0.0
            && self.detumble_rate_threshold_rad_s.is_finite()
            && self.detumble_rate_threshold_rad_s >= 0.0
            && self.max_automatic_arrest_delta_v_m_s.is_finite()
            && self.max_automatic_arrest_delta_v_m_s >= 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeparationTrigger {
    None,
    InvalidNavigation,
    ExcessAngularRate,
    TetherLoss,
    UncommandedSeparation,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SeparationAssessment {
    pub trigger: SeparationTrigger,
    pub recommended_mode: RescueMode,
    pub relative_speed_m_s: f64,
    pub angular_speed_rad_s: f64,
    /// Upper bound on automatic delta-v this trigger layer may ask the
    /// independent propulsion controller to consider.
    pub automatic_delta_v_ceiling_m_s: f64,
    pub automatic_action_permitted: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct SeparationDetector {
    pub policy: SeparationPolicy,
}

impl SeparationDetector {
    pub fn new(policy: SeparationPolicy) -> Self {
        Self { policy }
    }

    pub fn assess(&self, state: &RelativeNavigationState) -> SeparationAssessment {
        let speed = if finite3(state.relative_velocity_m_s) {
            state.relative_speed_m_s()
        } else {
            f64::INFINITY
        };
        let angular_speed = if finite3(state.angular_rate_rad_s) {
            state.angular_speed_rad_s()
        } else {
            f64::INFINITY
        };

        if !self.policy.is_valid()
            || !state.is_structurally_valid()
            || state.navigation_quality < self.policy.min_navigation_quality
            || state.sample_age_s > self.policy.max_sample_age_s
        {
            return SeparationAssessment {
                trigger: SeparationTrigger::InvalidNavigation,
                recommended_mode: RescueMode::Disabled,
                relative_speed_m_s: speed,
                angular_speed_rad_s: angular_speed,
                automatic_delta_v_ceiling_m_s: 0.0,
                automatic_action_permitted: false,
            };
        }

        if angular_speed > self.policy.detumble_rate_threshold_rad_s {
            return SeparationAssessment {
                trigger: SeparationTrigger::ExcessAngularRate,
                recommended_mode: RescueMode::Detumble,
                relative_speed_m_s: speed,
                angular_speed_rad_s: angular_speed,
                automatic_delta_v_ceiling_m_s: 0.0,
                automatic_action_permitted: true,
            };
        }

        let tether_lost = state.tether_tension_n < self.policy.tether_attached_threshold_n;
        if tether_lost && speed >= self.policy.uncommanded_separation_speed_m_s {
            return SeparationAssessment {
                trigger: SeparationTrigger::UncommandedSeparation,
                recommended_mode: RescueMode::ArrestDrift,
                relative_speed_m_s: speed,
                angular_speed_rad_s: angular_speed,
                automatic_delta_v_ceiling_m_s: self.policy.max_automatic_arrest_delta_v_m_s,
                automatic_action_permitted: true,
            };
        }

        if tether_lost {
            return SeparationAssessment {
                trigger: SeparationTrigger::TetherLoss,
                recommended_mode: RescueMode::AttitudeHold,
                relative_speed_m_s: speed,
                angular_speed_rad_s: angular_speed,
                automatic_delta_v_ceiling_m_s: 0.0,
                automatic_action_permitted: false,
            };
        }

        SeparationAssessment {
            trigger: SeparationTrigger::None,
            recommended_mode: RescueMode::Disabled,
            relative_speed_m_s: speed,
            angular_speed_rad_s: angular_speed,
            automatic_delta_v_ceiling_m_s: 0.0,
            automatic_action_permitted: false,
        }
    }
}

fn finite3(v: [f64; 3]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nominal() -> RelativeNavigationState {
        RelativeNavigationState {
            relative_position_m: [3.0, 0.0, 0.0],
            relative_velocity_m_s: [0.0; 3],
            angular_rate_rad_s: [0.0; 3],
            tether_tension_n: 10.0,
            navigation_quality: 0.99,
            sample_age_s: 0.02,
        }
    }

    #[test]
    fn stale_navigation_disables_automatic_action() {
        let mut state = nominal();
        state.sample_age_s = 1.0;
        let result = SeparationDetector::new(SeparationPolicy::simulation_reference()).assess(&state);
        assert_eq!(result.trigger, SeparationTrigger::InvalidNavigation);
        assert_eq!(result.recommended_mode, RescueMode::Disabled);
        assert!(!result.automatic_action_permitted);
    }

    #[test]
    fn high_angular_rate_recommends_detumble() {
        let mut state = nominal();
        state.angular_rate_rad_s = [0.3, 0.0, 0.0];
        let result = SeparationDetector::new(SeparationPolicy::simulation_reference()).assess(&state);
        assert_eq!(result.trigger, SeparationTrigger::ExcessAngularRate);
        assert_eq!(result.recommended_mode, RescueMode::Detumble);
        assert!(result.automatic_action_permitted);
    }

    #[test]
    fn tether_loss_plus_drift_recommends_bounded_arrest() {
        let mut state = nominal();
        state.tether_tension_n = 0.0;
        state.relative_velocity_m_s = [0.10, 0.0, 0.0];
        let result = SeparationDetector::new(SeparationPolicy::simulation_reference()).assess(&state);
        assert_eq!(result.trigger, SeparationTrigger::UncommandedSeparation);
        assert_eq!(result.recommended_mode, RescueMode::ArrestDrift);
        assert!(result.automatic_action_permitted);
        assert!(result.automatic_delta_v_ceiling_m_s > 0.0);
    }

    #[test]
    fn tether_loss_without_drift_does_not_authorize_translation() {
        let mut state = nominal();
        state.tether_tension_n = 0.0;
        let result = SeparationDetector::new(SeparationPolicy::simulation_reference()).assess(&state);
        assert_eq!(result.trigger, SeparationTrigger::TetherLoss);
        assert_eq!(result.recommended_mode, RescueMode::AttitudeHold);
        assert!(!result.automatic_action_permitted);
    }
}
