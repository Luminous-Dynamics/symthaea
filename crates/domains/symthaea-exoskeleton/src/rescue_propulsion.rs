// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded EVA self-rescue propulsion for SX-011 through SX-013.
//!
//! This module is a simulation/reference controller, not a flight-qualified
//! propulsion system. It deliberately keeps rescue propulsion independent from
//! PLSS oxygen and from Symthaea/Phi/learned-model authority.
//!
//! The controller admits only bounded `Detumble`, `AttitudeHold`,
//! `ArrestDrift`, and manual translation requests. Automatic return-to-target
//! guidance is intentionally out of scope for this tranche.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RescueEvidenceLevel {
    Simulation,
    HardwareBench,
    HumanInLoop,
    Qualification,
}

/// Rescue propulsion intentionally excludes breathing oxygen as a propellant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RescuePropellantKind {
    NitrogenColdGas,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RescueMode {
    Disabled,
    Detumble,
    AttitudeHold,
    ArrestDrift,
    ManualTranslation,
}

impl RescueMode {
    pub fn requires_relative_navigation(self) -> bool {
        matches!(self, Self::ArrestDrift)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BodyMassProperties {
    pub mass_kg: f64,
    /// Center of mass in the suit body frame.
    pub center_of_mass_m: [f64; 3],
    /// Principal-axis approximation used by the bounded rescue controller.
    pub inertia_diag_kg_m2: [f64; 3],
}

impl BodyMassProperties {
    pub fn is_valid(&self) -> bool {
        self.mass_kg.is_finite()
            && self.mass_kg > 0.0
            && finite3(self.center_of_mass_m)
            && self
                .inertia_diag_kg_m2
                .iter()
                .all(|v| v.is_finite() && *v > 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThrusterSpec {
    /// Nozzle location in the suit body frame.
    pub position_m: [f64; 3],
    /// Force direction on the suit. Must be approximately unit length.
    pub force_direction: [f64; 3],
    pub max_thrust_n: f64,
    pub enabled: bool,
}

impl ThrusterSpec {
    pub fn is_valid(&self) -> bool {
        if !finite3(self.position_m)
            || !finite3(self.force_direction)
            || !self.max_thrust_n.is_finite()
            || self.max_thrust_n <= 0.0
        {
            return false;
        }
        let n = norm3(self.force_direction);
        (0.99..=1.01).contains(&n)
    }

    /// Exhaust plume direction is opposite the force applied to the suit.
    pub fn plume_direction(&self) -> [f64; 3] {
        scale3(self.force_direction, -1.0)
    }
}

/// Hard no-fire geometry represented as a protected sphere in the suit frame.
///
/// This is intentionally conservative and simple. A flight implementation can
/// replace these with validated meshes/cones while preserving the same hard
/// constraint semantics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProtectedSphere {
    pub center_m: [f64; 3],
    pub radius_m: f64,
}

impl ProtectedSphere {
    pub fn is_valid(&self) -> bool {
        finite3(self.center_m) && self.radius_m.is_finite() && self.radius_m >= 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RescueAuthorityState {
    pub manual_kill: bool,
    pub watchdog_healthy: bool,
    pub isolation_valve_open: bool,
    pub propulsion_power_available: bool,
    pub imu_valid: bool,
    pub relative_navigation_valid: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RescueLimits {
    pub max_command_duration_s: f64,
    pub max_total_delta_v_m_s: f64,
    pub max_translation_accel_m_s2: f64,
    pub max_angular_accel_rad_s2: f64,
    pub min_propellant_kg: f64,
    pub max_thruster_duty_cycle: f64,
    pub evidence: RescueEvidenceLevel,
}

impl RescueLimits {
    /// Software-test values only; not a SAFER or human-rating specification.
    pub fn simulation_reference() -> Self {
        Self {
            max_command_duration_s: 0.5,
            max_total_delta_v_m_s: 3.0,
            max_translation_accel_m_s2: 0.12,
            max_angular_accel_rad_s2: 0.20,
            min_propellant_kg: 0.02,
            max_thruster_duty_cycle: 1.0,
            evidence: RescueEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.max_command_duration_s.is_finite()
            && self.max_command_duration_s > 0.0
            && self.max_total_delta_v_m_s.is_finite()
            && self.max_total_delta_v_m_s > 0.0
            && self.max_translation_accel_m_s2.is_finite()
            && self.max_translation_accel_m_s2 > 0.0
            && self.max_angular_accel_rad_s2.is_finite()
            && self.max_angular_accel_rad_s2 > 0.0
            && self.min_propellant_kg.is_finite()
            && self.min_propellant_kg >= 0.0
            && self.max_thruster_duty_cycle.is_finite()
            && (0.0..=1.0).contains(&self.max_thruster_duty_cycle)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RescueState {
    pub propellant_kg: f64,
    pub cumulative_delta_v_m_s: f64,
    pub angular_rate_rad_s: [f64; 3],
}

impl RescueState {
    pub fn is_valid(&self) -> bool {
        self.propellant_kg.is_finite()
            && self.propellant_kg >= 0.0
            && self.cumulative_delta_v_m_s.is_finite()
            && self.cumulative_delta_v_m_s >= 0.0
            && finite3(self.angular_rate_rad_s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RescueCommand {
    pub mode: RescueMode,
    pub desired_force_n: [f64; 3],
    pub desired_torque_nm: [f64; 3],
    pub duration_s: f64,
}

impl RescueCommand {
    pub fn is_valid(&self) -> bool {
        finite3(self.desired_force_n)
            && finite3(self.desired_torque_nm)
            && self.duration_s.is_finite()
            && self.duration_s > 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThrusterPulse {
    pub thruster_index: usize,
    pub duty_cycle: f64,
    pub duration_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RescueDenialReason {
    Disabled,
    ManualKill,
    WatchdogFault,
    IsolationValveClosed,
    PropulsionPowerUnavailable,
    ImuInvalid,
    RelativeNavigationInvalid,
    InvalidLimits,
    InvalidMassProperties,
    InvalidState,
    InvalidCommand,
    InvalidThrusterGeometry,
    InvalidProtectedGeometry,
    CommandDurationExceeded,
    TranslationAccelerationExceeded,
    AngularAccelerationExceeded,
    DeltaVBudgetExceeded,
    PropellantReserveReached,
    NoSafeThrusterCombination,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RescueDecision {
    pub permitted: bool,
    pub pulses: Vec<ThrusterPulse>,
    pub predicted_force_n: [f64; 3],
    pub predicted_torque_nm: [f64; 3],
    pub predicted_delta_v_m_s: f64,
    pub predicted_propellant_kg: f64,
    pub reasons: Vec<RescueDenialReason>,
}

impl RescueDecision {
    fn denied(reason: RescueDenialReason) -> Self {
        Self {
            permitted: false,
            pulses: Vec::new(),
            predicted_force_n: [0.0; 3],
            predicted_torque_nm: [0.0; 3],
            predicted_delta_v_m_s: 0.0,
            predicted_propellant_kg: 0.0,
            reasons: vec![reason],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RescuePropulsionKernel {
    pub propellant_kind: RescuePropellantKind,
    pub effective_exhaust_velocity_m_s: f64,
    pub thrusters: Vec<ThrusterSpec>,
    pub protected_geometry: Vec<ProtectedSphere>,
    pub limits: RescueLimits,
}

impl RescuePropulsionKernel {
    pub fn new(
        effective_exhaust_velocity_m_s: f64,
        thrusters: Vec<ThrusterSpec>,
        protected_geometry: Vec<ProtectedSphere>,
        limits: RescueLimits,
    ) -> Self {
        Self {
            propellant_kind: RescuePropellantKind::NitrogenColdGas,
            effective_exhaust_velocity_m_s,
            thrusters,
            protected_geometry,
            limits,
        }
    }

    /// Evaluate and allocate a bounded rescue request.
    ///
    /// There is intentionally no Symthaea, Phi, LLM, or PLSS input here.
    pub fn evaluate(
        &self,
        authority: &RescueAuthorityState,
        body: &BodyMassProperties,
        state: &RescueState,
        command: &RescueCommand,
    ) -> RescueDecision {
        if command.mode == RescueMode::Disabled {
            return RescueDecision::denied(RescueDenialReason::Disabled);
        }
        if authority.manual_kill {
            return RescueDecision::denied(RescueDenialReason::ManualKill);
        }
        if !authority.watchdog_healthy {
            return RescueDecision::denied(RescueDenialReason::WatchdogFault);
        }
        if !authority.isolation_valve_open {
            return RescueDecision::denied(RescueDenialReason::IsolationValveClosed);
        }
        if !authority.propulsion_power_available {
            return RescueDecision::denied(RescueDenialReason::PropulsionPowerUnavailable);
        }
        if !authority.imu_valid {
            return RescueDecision::denied(RescueDenialReason::ImuInvalid);
        }
        if command.mode.requires_relative_navigation() && !authority.relative_navigation_valid {
            return RescueDecision::denied(RescueDenialReason::RelativeNavigationInvalid);
        }
        if !self.limits.is_valid()
            || !self.effective_exhaust_velocity_m_s.is_finite()
            || self.effective_exhaust_velocity_m_s <= 0.0
        {
            return RescueDecision::denied(RescueDenialReason::InvalidLimits);
        }
        if !body.is_valid() {
            return RescueDecision::denied(RescueDenialReason::InvalidMassProperties);
        }
        if !state.is_valid() {
            return RescueDecision::denied(RescueDenialReason::InvalidState);
        }
        if !command.is_valid() {
            return RescueDecision::denied(RescueDenialReason::InvalidCommand);
        }
        if self.thrusters.is_empty() || self.thrusters.iter().any(|t| !t.is_valid()) {
            return RescueDecision::denied(RescueDenialReason::InvalidThrusterGeometry);
        }
        if self.protected_geometry.iter().any(|p| !p.is_valid()) {
            return RescueDecision::denied(RescueDenialReason::InvalidProtectedGeometry);
        }
        if command.duration_s > self.limits.max_command_duration_s {
            return RescueDecision::denied(RescueDenialReason::CommandDurationExceeded);
        }
        if state.propellant_kg <= self.limits.min_propellant_kg {
            return RescueDecision::denied(RescueDenialReason::PropellantReserveReached);
        }

        let requested_accel = norm3(command.desired_force_n) / body.mass_kg;
        if requested_accel > self.limits.max_translation_accel_m_s2 {
            return RescueDecision::denied(RescueDenialReason::TranslationAccelerationExceeded);
        }
        let requested_alpha = max_component_ratio(
            command.desired_torque_nm,
            body.inertia_diag_kg_m2,
        );
        if requested_alpha > self.limits.max_angular_accel_rad_s2 {
            return RescueDecision::denied(RescueDenialReason::AngularAccelerationExceeded);
        }

        let force_goal = normalized_or_zero(command.desired_force_n);
        let torque_goal = normalized_or_zero(command.desired_torque_nm);
        let mut pulses = Vec::new();
        let mut resultant_force = [0.0; 3];
        let mut resultant_torque = [0.0; 3];
        let mut scalar_impulse_n_s = 0.0;

        for (index, thruster) in self.thrusters.iter().enumerate() {
            if !thruster.enabled || self.thruster_plume_intersects_protected_geometry(thruster) {
                continue;
            }

            let arm = sub3(thruster.position_m, body.center_of_mass_m);
            let full_force = scale3(thruster.force_direction, thruster.max_thrust_n);
            let full_torque = cross3(arm, full_force);
            let force_alignment = dot3(thruster.force_direction, force_goal).max(0.0);
            let torque_alignment = dot3(normalized_or_zero(full_torque), torque_goal).max(0.0);
            let score = (force_alignment + torque_alignment).min(1.0);
            let duty = score.min(self.limits.max_thruster_duty_cycle);
            if duty <= 1e-9 {
                continue;
            }

            let applied_force = scale3(full_force, duty);
            let applied_torque = scale3(full_torque, duty);
            resultant_force = add3(resultant_force, applied_force);
            resultant_torque = add3(resultant_torque, applied_torque);
            scalar_impulse_n_s += thruster.max_thrust_n * duty * command.duration_s;
            pulses.push(ThrusterPulse {
                thruster_index: index,
                duty_cycle: duty,
                duration_s: command.duration_s,
            });
        }

        if pulses.is_empty() {
            return RescueDecision::denied(RescueDenialReason::NoSafeThrusterCombination);
        }

        let predicted_delta_v = norm3(resultant_force) * command.duration_s / body.mass_kg;
        if state.cumulative_delta_v_m_s + predicted_delta_v > self.limits.max_total_delta_v_m_s {
            return RescueDecision::denied(RescueDenialReason::DeltaVBudgetExceeded);
        }

        let predicted_propellant = scalar_impulse_n_s / self.effective_exhaust_velocity_m_s;
        if state.propellant_kg - predicted_propellant < self.limits.min_propellant_kg {
            return RescueDecision::denied(RescueDenialReason::PropellantReserveReached);
        }

        RescueDecision {
            permitted: true,
            pulses,
            predicted_force_n: resultant_force,
            predicted_torque_nm: resultant_torque,
            predicted_delta_v_m_s: predicted_delta_v,
            predicted_propellant_kg: predicted_propellant,
            reasons: Vec::new(),
        }
    }

    fn thruster_plume_intersects_protected_geometry(&self, thruster: &ThrusterSpec) -> bool {
        let plume = thruster.plume_direction();
        self.protected_geometry
            .iter()
            .any(|sphere| ray_intersects_sphere(thruster.position_m, plume, *sphere))
    }
}

fn ray_intersects_sphere(origin: [f64; 3], direction: [f64; 3], sphere: ProtectedSphere) -> bool {
    let to_center = sub3(sphere.center_m, origin);
    let along = dot3(to_center, direction);
    if along <= 0.0 {
        return false;
    }
    let center_distance_sq = dot3(to_center, to_center);
    let closest_sq = (center_distance_sq - along * along).max(0.0);
    closest_sq <= sphere.radius_m * sphere.radius_m
}

fn max_component_ratio(numerator: [f64; 3], denominator: [f64; 3]) -> f64 {
    (0..3)
        .map(|i| numerator[i].abs() / denominator[i])
        .fold(0.0, f64::max)
}

fn finite3(v: [f64; 3]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

fn normalized_or_zero(a: [f64; 3]) -> [f64; 3] {
    let n = norm3(a);
    if n <= 1e-12 {
        [0.0; 3]
    } else {
        scale3(a, 1.0 / n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel() -> RescuePropulsionKernel {
        let thrusters = vec![
            ThrusterSpec {
                position_m: [0.0, 0.4, 0.0],
                force_direction: [1.0, 0.0, 0.0],
                max_thrust_n: 4.0,
                enabled: true,
            },
            ThrusterSpec {
                position_m: [0.0, -0.4, 0.0],
                force_direction: [1.0, 0.0, 0.0],
                max_thrust_n: 4.0,
                enabled: true,
            },
            ThrusterSpec {
                position_m: [0.4, 0.0, 0.0],
                force_direction: [0.0, 1.0, 0.0],
                max_thrust_n: 4.0,
                enabled: true,
            },
        ];
        RescuePropulsionKernel::new(650.0, thrusters, Vec::new(), RescueLimits::simulation_reference())
    }

    fn authority() -> RescueAuthorityState {
        RescueAuthorityState {
            manual_kill: false,
            watchdog_healthy: true,
            isolation_valve_open: true,
            propulsion_power_available: true,
            imu_valid: true,
            relative_navigation_valid: true,
        }
    }

    fn body() -> BodyMassProperties {
        BodyMassProperties {
            mass_kg: 150.0,
            center_of_mass_m: [0.0; 3],
            inertia_diag_kg_m2: [20.0, 20.0, 10.0],
        }
    }

    fn state() -> RescueState {
        RescueState {
            propellant_kg: 1.0,
            cumulative_delta_v_m_s: 0.0,
            angular_rate_rad_s: [0.0; 3],
        }
    }

    #[test]
    fn manual_kill_is_absolute() {
        let mut a = authority();
        a.manual_kill = true;
        let cmd = RescueCommand {
            mode: RescueMode::ManualTranslation,
            desired_force_n: [4.0, 0.0, 0.0],
            desired_torque_nm: [0.0; 3],
            duration_s: 0.1,
        };
        let decision = kernel().evaluate(&a, &body(), &state(), &cmd);
        assert!(!decision.permitted);
        assert_eq!(decision.reasons, vec![RescueDenialReason::ManualKill]);
    }

    #[test]
    fn bounded_translation_produces_a_finite_allocation() {
        let cmd = RescueCommand {
            mode: RescueMode::ManualTranslation,
            desired_force_n: [4.0, 0.0, 0.0],
            desired_torque_nm: [0.0; 3],
            duration_s: 0.1,
        };
        let decision = kernel().evaluate(&authority(), &body(), &state(), &cmd);
        assert!(decision.permitted);
        assert!(!decision.pulses.is_empty());
        assert!(decision.predicted_delta_v_m_s.is_finite());
        assert!(decision.predicted_propellant_kg > 0.0);
    }

    #[test]
    fn automatic_drift_arrest_requires_relative_navigation() {
        let mut a = authority();
        a.relative_navigation_valid = false;
        let cmd = RescueCommand {
            mode: RescueMode::ArrestDrift,
            desired_force_n: [2.0, 0.0, 0.0],
            desired_torque_nm: [0.0; 3],
            duration_s: 0.1,
        };
        let decision = kernel().evaluate(&a, &body(), &state(), &cmd);
        assert_eq!(
            decision.reasons,
            vec![RescueDenialReason::RelativeNavigationInvalid]
        );
    }

    #[test]
    fn plume_keepout_is_a_hard_constraint() {
        let mut k = kernel();
        // First two +X force thrusters exhaust toward -X. Protect a volume
        // directly behind them so neither may fire.
        k.protected_geometry.push(ProtectedSphere {
            center_m: [-0.5, 0.4, 0.0],
            radius_m: 0.25,
        });
        k.protected_geometry.push(ProtectedSphere {
            center_m: [-0.5, -0.4, 0.0],
            radius_m: 0.25,
        });
        let cmd = RescueCommand {
            mode: RescueMode::ManualTranslation,
            desired_force_n: [4.0, 0.0, 0.0],
            desired_torque_nm: [0.0; 3],
            duration_s: 0.1,
        };
        let decision = k.evaluate(&authority(), &body(), &state(), &cmd);
        assert!(!decision.permitted);
        assert_eq!(
            decision.reasons,
            vec![RescueDenialReason::NoSafeThrusterCombination]
        );
    }

    #[test]
    fn delta_v_budget_fails_closed() {
        let mut s = state();
        s.cumulative_delta_v_m_s = 2.9999;
        let cmd = RescueCommand {
            mode: RescueMode::ManualTranslation,
            desired_force_n: [4.0, 0.0, 0.0],
            desired_torque_nm: [0.0; 3],
            duration_s: 0.5,
        };
        let decision = kernel().evaluate(&authority(), &body(), &s, &cmd);
        assert!(!decision.permitted);
        assert_eq!(decision.reasons, vec![RescueDenialReason::DeltaVBudgetExceeded]);
    }

    #[test]
    fn breathing_oxygen_is_not_a_rescue_propellant_option() {
        assert_eq!(kernel().propellant_kind, RescuePropellantKind::NitrogenColdGas);
    }
}
