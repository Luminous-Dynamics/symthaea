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

use crate::rotor_hub::{RotorHubDynamics, RotorHubOutput};

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
    /// Tail rotor thrust coefficient: thrust_N = coeff × RPM² × pitch_fraction.
    pub tail_thrust_coefficient: f64,
    /// Tail rotor moment arm from the center of mass, meters.
    pub tail_moment_arm: f64,
    /// Gyroscopic precession gain: applied moment appears 90° ahead of tilt input.
    pub precession_gain: f64,
    /// Autorotation RPM threshold: below this, rotor enters autorotation regime.
    pub autorotation_rpm_threshold: f64,
    /// Windmilling target-RPM gain per m/s of downward airflow.
    pub autorotation_inflow_gain: f64,
    /// Upper bound for windmilling RPM in the reduced-order model.
    pub autorotation_max_rpm: f64,
    /// Fraction of powered-flight lift retained for the same RPM/collective.
    pub autorotation_lift_factor: f64,
    /// Effective polar inertia of the main rotor system, kg·m².
    pub main_rotor_inertia_kg_m2: f64,
    /// Collective-dependent RPM penalty during windmilling descent.
    /// Raising collective extracts rotor energy for lift; lowering collective
    /// preserves RPM for a later flare.
    pub autorotation_collective_rpm_penalty: f64,
    /// RPM used as the lower bound for a credible flare-energy margin.
    pub flare_minimum_rpm: f64,
    /// Air density used by the reduced-order induced-flow model, kg/m³.
    pub air_density_kg_m3: f64,
    /// Main-rotor radius used to derive disk area, meters.
    pub rotor_radius_m: f64,
    /// First-order time constant for induced-flow response, seconds.
    pub induced_flow_tau_s: f64,
    /// Horizontal airspeed where effective translational lift begins, m/s.
    pub translational_lift_onset_mps: f64,
    /// Horizontal airspeed where the configured translational gain is reached.
    pub translational_lift_full_mps: f64,
    /// Maximum lift multiplier added by effective translational lift.
    pub translational_lift_gain: f64,
    /// Descent/induced-velocity ratio that enters vortex-ring exposure.
    pub vortex_ring_descent_ratio: f64,
    /// Maximum horizontal airspeed for vortex-ring exposure, m/s.
    pub vortex_ring_max_horizontal_mps: f64,
    /// Lift fraction retained while in the reduced-order vortex-ring regime.
    pub vortex_ring_lift_factor: f64,
}

impl Default for RotorDynamicsConfig {
    fn default() -> Self {
        Self {
            max_main_rpm: 5500.0,
            max_tail_rpm: 4000.0,
            main_rotor_tau: 0.5, // 500ms to reach target RPM
            tail_rotor_tau: 0.2, // 200ms (faster for yaw control)
            // Calibrated trim: 500 kg × 9.81 m/s² at 3300 RPM and
            // effective collective 0.8 (normalized collective 0.3).
            thrust_coefficient: 5.630_165_289e-4,
            torque_reaction_coefficient: 8.0e-5,
            // Neutral pedal at 2000 RPM balances the default main-rotor
            // reaction torque through a 4 m tail moment arm.
            tail_thrust_coefficient: 1.089e-4,
            tail_moment_arm: 4.0,
            precession_gain: 0.15, // 15% of cyclic input appears as precession
            autorotation_rpm_threshold: 1500.0,
            autorotation_inflow_gain: 220.0,
            autorotation_max_rpm: 3000.0,
            autorotation_lift_factor: 0.65,
            main_rotor_inertia_kg_m2: 120.0,
            autorotation_collective_rpm_penalty: 900.0,
            flare_minimum_rpm: 1800.0,
            air_density_kg_m3: 1.225,
            rotor_radius_m: 5.3,
            induced_flow_tau_s: 0.25,
            translational_lift_onset_mps: 5.0,
            translational_lift_full_mps: 18.0,
            translational_lift_gain: 0.15,
            vortex_ring_descent_ratio: 0.7,
            vortex_ring_max_horizontal_mps: 5.0,
            vortex_ring_lift_factor: 0.55,
        }
    }
}

impl RotorDynamicsConfig {
    pub fn validate(&self) -> bool {
        let positive = [
            self.max_main_rpm,
            self.max_tail_rpm,
            self.main_rotor_tau,
            self.tail_rotor_tau,
            self.thrust_coefficient,
            self.tail_moment_arm,
            self.autorotation_rpm_threshold,
            self.autorotation_max_rpm,
            self.main_rotor_inertia_kg_m2,
            self.flare_minimum_rpm,
            self.air_density_kg_m3,
            self.rotor_radius_m,
            self.induced_flow_tau_s,
            self.translational_lift_full_mps,
            self.vortex_ring_max_horizontal_mps,
        ];
        positive
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
            && self.translational_lift_onset_mps.is_finite()
            && self.translational_lift_onset_mps >= 0.0
            && self.translational_lift_full_mps > self.translational_lift_onset_mps
            && self.translational_lift_gain.is_finite()
            && self.translational_lift_gain >= 0.0
            && self.vortex_ring_descent_ratio.is_finite()
            && self.vortex_ring_descent_ratio > 0.0
            && self.vortex_ring_lift_factor.is_finite()
            && (0.0..=1.0).contains(&self.vortex_ring_lift_factor)
    }
}

/// Air-relative flight condition supplied by the body/wind integrator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotorFlightCondition {
    pub horizontal_airspeed_mps: f64,
    pub vertical_airspeed_mps: f64,
    pub height_agl_m: f64,
    /// Local air density. Thrust coefficients are referenced to
    /// `RotorDynamicsConfig::air_density_kg_m3`.
    pub air_density_kg_m3: f64,
}

impl RotorFlightCondition {
    pub fn new(
        horizontal_airspeed_mps: f64,
        vertical_airspeed_mps: f64,
        height_agl_m: f64,
    ) -> Self {
        Self {
            horizontal_airspeed_mps,
            vertical_airspeed_mps,
            height_agl_m,
            air_density_kg_m3: 1.225,
        }
    }

    /// Override the reference sea-level density with a local atmosphere sample.
    pub fn with_air_density(mut self, air_density_kg_m3: f64) -> Self {
        self.air_density_kg_m3 = air_density_kg_m3;
        self
    }

    fn sanitized(self) -> Self {
        Self {
            horizontal_airspeed_mps: if self.horizontal_airspeed_mps.is_finite() {
                self.horizontal_airspeed_mps.max(0.0)
            } else {
                0.0
            },
            vertical_airspeed_mps: if self.vertical_airspeed_mps.is_finite() {
                self.vertical_airspeed_mps
            } else {
                0.0
            },
            height_agl_m: if self.height_agl_m.is_finite() {
                self.height_agl_m.max(0.0)
            } else {
                f64::INFINITY
            },
            air_density_kg_m3: if self.air_density_kg_m3.is_finite() && self.air_density_kg_m3 > 0.0
            {
                self.air_density_kg_m3
            } else {
                // Invalid density must not restore full sea-level authority.
                0.061_25
            },
        }
    }
}

/// Mutually exclusive aerodynamic regime emitted as flight evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotorFlightRegime {
    Normal,
    EffectiveTranslationalLift,
    VortexRingExposure,
    Autorotation,
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
    /// Main-rotor rotational kinetic energy, Joules.
    pub kinetic_energy_j: f64,
    /// Energy margin above the configured minimum flare RPM, Joules.
    pub flare_energy_margin_j: f64,
    /// First-order induced velocity through the rotor disk, m/s.
    pub induced_velocity_mps: f64,
    /// Current aerodynamic regime.
    pub flight_regime: RotorFlightRegime,
}

impl RotorDynamicsState {
    /// Create initial state at hover RPM.
    pub fn hover() -> Self {
        Self {
            main_rpm: 3300.0,
            tail_rpm: 2000.0,
            in_autorotation: false,
            main_revolutions: 0.0,
            kinetic_energy_j: rotor_kinetic_energy_j(120.0, 3300.0),
            flare_energy_margin_j: rotor_kinetic_energy_j(120.0, 3300.0)
                - rotor_kinetic_energy_j(120.0, 1800.0),
            induced_velocity_mps: 4.8,
            flight_regime: RotorFlightRegime::Normal,
        }
    }

    /// Create grounded state (rotors off).
    pub fn grounded() -> Self {
        Self {
            main_rpm: 0.0,
            tail_rpm: 0.0,
            in_autorotation: false,
            main_revolutions: 0.0,
            kinetic_energy_j: 0.0,
            flare_energy_margin_j: -rotor_kinetic_energy_j(120.0, 1800.0),
            induced_velocity_mps: 0.0,
            flight_regime: RotorFlightRegime::Normal,
        }
    }
}

fn rotor_kinetic_energy_j(inertia_kg_m2: f64, rpm: f64) -> f64 {
    let omega_rad_s = rpm.max(0.0) * std::f64::consts::TAU / 60.0;
    0.5 * inertia_kg_m2 * omega_rad_s * omega_rad_s
}

/// Rotor dynamics model.
///
/// Computes thrust, torque, and gyroscopic effects from rotor state + commands.
pub struct RotorDynamics {
    config: RotorDynamicsConfig,
    state: RotorDynamicsState,
    hub: RotorHubDynamics,
}

impl RotorDynamics {
    /// Create with default config at hover state.
    pub fn new() -> Self {
        Self {
            config: RotorDynamicsConfig::default(),
            state: RotorDynamicsState::hover(),
            hub: RotorHubDynamics::default(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: RotorDynamicsConfig) -> Self {
        let mut state = RotorDynamicsState::hover();
        state.kinetic_energy_j =
            rotor_kinetic_energy_j(config.main_rotor_inertia_kg_m2, state.main_rpm);
        state.flare_energy_margin_j = state.kinetic_energy_j
            - rotor_kinetic_energy_j(config.main_rotor_inertia_kg_m2, config.flare_minimum_rpm);
        Self {
            config,
            state,
            hub: RotorHubDynamics::default(),
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
        pedal: f64,
        collective: f64,
        cyclic_lon: f64,
        cyclic_lat: f64,
        vertical_velocity: f64,
        dt: f64,
    ) -> RotorOutput {
        self.step_with_flight_condition(
            thrust_cmd,
            tail_cmd,
            pedal,
            collective,
            cyclic_lon,
            cyclic_lat,
            RotorFlightCondition::new(0.0, vertical_velocity, f64::INFINITY),
            dt,
        )
    }

    /// Step with air-relative flight condition so translational lift and
    /// vortex-ring exposure are explicit, testable regimes.
    pub fn step_with_flight_condition(
        &mut self,
        thrust_cmd: f64,
        tail_cmd: f64,
        pedal: f64,
        collective: f64,
        cyclic_lon: f64,
        cyclic_lat: f64,
        flight_condition: RotorFlightCondition,
        dt: f64,
    ) -> RotorOutput {
        let flight_condition = flight_condition.sanitized();
        // 1. RPM lag: first-order exponential filter
        let powered_target_main = thrust_cmd * self.config.max_main_rpm;
        let descent_speed = (-flight_condition.vertical_airspeed_mps).max(0.0);
        let effective_collective = (collective + 0.5).max(0.0);
        let windmill_target = if thrust_cmd < 0.1 {
            let inflow_target = self.config.autorotation_rpm_threshold
                + self.config.autorotation_inflow_gain * descent_speed;
            let collective_penalty = self.config.autorotation_collective_rpm_penalty
                * effective_collective.clamp(0.0, 1.5);
            (inflow_target - collective_penalty).clamp(0.0, self.config.autorotation_max_rpm)
        } else {
            0.0
        };
        let target_main = powered_target_main.max(windmill_target);
        let target_tail = tail_cmd * self.config.max_tail_rpm;

        let alpha_main = 1.0 - (-dt / self.config.main_rotor_tau).exp();
        let alpha_tail = 1.0 - (-dt / self.config.tail_rotor_tau).exp();

        self.state.main_rpm += alpha_main * (target_main - self.state.main_rpm);
        self.state.tail_rpm += alpha_tail * (target_tail - self.state.tail_rpm);

        // Clamp to non-negative
        self.state.main_rpm = self.state.main_rpm.max(0.0);
        self.state.tail_rpm = self.state.tail_rpm.max(0.0);

        // Track revolutions and explicit rotor-energy state. Energy is the
        // conserved resource an autorotation controller trades between RPM,
        // lift, and flare authority.
        self.state.main_revolutions += self.state.main_rpm / 60.0 * dt;
        self.state.kinetic_energy_j =
            rotor_kinetic_energy_j(self.config.main_rotor_inertia_kg_m2, self.state.main_rpm);
        self.state.flare_energy_margin_j = self.state.kinetic_energy_j
            - rotor_kinetic_energy_j(
                self.config.main_rotor_inertia_kg_m2,
                self.config.flare_minimum_rpm,
            );

        // 2. Autorotation check
        self.state.in_autorotation = thrust_cmd < 0.1
            && (flight_condition.vertical_airspeed_mps < -0.5
                || self.state.main_rpm < self.config.autorotation_rpm_threshold);

        // 3. Thrust from RPM² × collective. Autorotation remains inside the
        // normal force integrator; downward airflow can sustain rotor energy,
        // but the reduced-order model applies a conservative lift factor.
        // thrust = coeff × RPM² × (collective + offset) where offset shifts
        // collective=0 to still produce some lift at hover RPM
        // Shift so hover collective ~0.3 → 0.8.
        let autorotation_factor = if self.state.in_autorotation {
            self.config.autorotation_lift_factor
        } else {
            1.0
        };
        let density_ratio =
            (flight_condition.air_density_kg_m3 / self.config.air_density_kg_m3).clamp(0.05, 2.0);
        let base_thrust_force = self.config.thrust_coefficient
            * self.state.main_rpm.powi(2)
            * effective_collective
            * autorotation_factor
            * density_ratio;

        let disk_area_m2 = std::f64::consts::PI * self.config.rotor_radius_m.powi(2);
        let induced_target_mps = if base_thrust_force > 0.0 {
            (base_thrust_force / (2.0 * flight_condition.air_density_kg_m3 * disk_area_m2)).sqrt()
        } else {
            0.0
        };
        let inflow_alpha = 1.0 - (-dt.max(0.0) / self.config.induced_flow_tau_s).exp();
        self.state.induced_velocity_mps +=
            inflow_alpha * (induced_target_mps - self.state.induced_velocity_mps);
        self.state.induced_velocity_mps = self.state.induced_velocity_mps.max(0.0);

        let translational_blend = ((flight_condition.horizontal_airspeed_mps
            - self.config.translational_lift_onset_mps)
            / (self.config.translational_lift_full_mps - self.config.translational_lift_onset_mps))
            .clamp(0.0, 1.0);
        let translational_factor = 1.0 + self.config.translational_lift_gain * translational_blend;
        let vortex_ring = !self.state.in_autorotation
            && descent_speed
                > self.config.vortex_ring_descent_ratio * self.state.induced_velocity_mps.max(0.1)
            && flight_condition.horizontal_airspeed_mps
                < self.config.vortex_ring_max_horizontal_mps;
        let aerodynamic_efficiency = if vortex_ring {
            self.config.vortex_ring_lift_factor
        } else {
            translational_factor
        };
        self.state.flight_regime = if self.state.in_autorotation {
            RotorFlightRegime::Autorotation
        } else if vortex_ring {
            RotorFlightRegime::VortexRingExposure
        } else if translational_blend > 0.0 {
            RotorFlightRegime::EffectiveTranslationalLift
        } else {
            RotorFlightRegime::Normal
        };
        let thrust_force = base_thrust_force * aerodynamic_efficiency;
        let hub_output = self
            .hub
            .step(
                cyclic_lon,
                cyclic_lat,
                flight_condition.horizontal_airspeed_mps,
                self.state.main_rpm,
                self.config.rotor_radius_m,
                thrust_force,
                disk_area_m2,
                dt.max(1.0e-6),
            )
            .unwrap_or(RotorHubOutput {
                longitudinal_flap_rad: 0.0,
                lateral_flap_rad: 0.0,
                coning_angle_rad: 0.0,
                roll_moment_nm: 0.0,
                pitch_moment_nm: 0.0,
                advance_ratio: 0.0,
                control_authority: 0.0,
            });

        // 4. Torque reaction: powered main rotor creates yaw torque. During
        // windmilling descent only a reduced aerodynamic reaction remains.
        let torque_factor = if self.state.in_autorotation {
            0.25
        } else {
            1.0
        };
        let torque_reaction = self.config.torque_reaction_coefficient
            * self.state.main_rpm.powi(2)
            * torque_factor
            * density_ratio;

        // 5. Tail rotor anti-torque. Pedal changes blade pitch while the
        // tail_rotor command changes available RPM/authority.
        let tail_pitch_fraction = (0.5 + 0.5 * pedal).clamp(0.0, 1.0);
        let tail_thrust = self.config.tail_thrust_coefficient
            * self.state.tail_rpm.powi(2)
            * tail_pitch_fraction
            * density_ratio;
        let tail_yaw_torque = tail_thrust * self.config.tail_moment_arm;

        // 6. Gyroscopic precession: cyclic input appears 90° ahead
        // Longitudinal cyclic → roll precession, lateral cyclic → pitch precession
        let precession_roll =
            cyclic_lon * self.config.precession_gain * self.state.main_rpm / 1000.0;
        let precession_pitch =
            -cyclic_lat * self.config.precession_gain * self.state.main_rpm / 1000.0;

        let omega_rad_s = self.state.main_rpm * std::f64::consts::TAU / 60.0;
        let engine_power_w = if thrust_cmd >= 0.1 {
            torque_reaction * omega_rad_s
        } else {
            0.0
        };
        let aerodynamic_power_w = if self.state.in_autorotation {
            self.config.main_rotor_inertia_kg_m2
                * omega_rad_s
                * (target_main - self.state.main_rpm)
                * std::f64::consts::TAU
                / 60.0
                / self.config.main_rotor_tau.max(1.0e-6)
        } else {
            0.0
        };

        RotorOutput {
            thrust_force,
            torque_reaction,
            tail_yaw_torque,
            precession_roll,
            precession_pitch,
            rotor_kinetic_energy_j: self.state.kinetic_energy_j,
            flare_energy_margin_j: self.state.flare_energy_margin_j,
            engine_power_w,
            aerodynamic_power_w,
            induced_velocity_mps: self.state.induced_velocity_mps,
            aerodynamic_efficiency,
            flight_regime: self.state.flight_regime,
            longitudinal_flap_rad: hub_output.longitudinal_flap_rad,
            lateral_flap_rad: hub_output.lateral_flap_rad,
            coning_angle_rad: hub_output.coning_angle_rad,
            hub_roll_moment_nm: hub_output.roll_moment_nm,
            hub_pitch_moment_nm: hub_output.pitch_moment_nm,
            advance_ratio: hub_output.advance_ratio,
            hub_control_authority: hub_output.control_authority,
            air_density_kg_m3: flight_condition.air_density_kg_m3,
            density_ratio,
        }
    }

    /// Normalized flare-energy margin. Zero means the rotor is at the
    /// configured minimum flare RPM; one means hover kinetic energy.
    pub fn flare_energy_margin_fraction(&self) -> f64 {
        let minimum = rotor_kinetic_energy_j(
            self.config.main_rotor_inertia_kg_m2,
            self.config.flare_minimum_rpm,
        );
        let hover = rotor_kinetic_energy_j(self.config.main_rotor_inertia_kg_m2, 3300.0);
        ((self.state.kinetic_energy_j - minimum) / (hover - minimum).max(1.0)).clamp(-1.0, 1.5)
    }

    /// Reset to hover state.
    pub fn reset(&mut self) {
        let mut state = RotorDynamicsState::hover();
        state.kinetic_energy_j =
            rotor_kinetic_energy_j(self.config.main_rotor_inertia_kg_m2, state.main_rpm);
        state.flare_energy_margin_j = state.kinetic_energy_j
            - rotor_kinetic_energy_j(
                self.config.main_rotor_inertia_kg_m2,
                self.config.flare_minimum_rpm,
            );
        self.state = state;
        self.hub.reset();
    }

    /// Reset to grounded state.
    pub fn reset_grounded(&mut self) {
        let mut state = RotorDynamicsState::grounded();
        state.flare_energy_margin_j = -rotor_kinetic_energy_j(
            self.config.main_rotor_inertia_kg_m2,
            self.config.flare_minimum_rpm,
        );
        self.state = state;
        self.hub.reset();
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
    /// Counter-torque generated by the tail rotor (Nm).
    pub tail_yaw_torque: f64,
    /// Gyroscopic precession in roll axis (Nm).
    pub precession_roll: f64,
    /// Gyroscopic precession in pitch axis (Nm).
    pub precession_pitch: f64,
    /// Main-rotor rotational kinetic energy (J).
    pub rotor_kinetic_energy_j: f64,
    /// Energy above/below the configured minimum flare RPM (J).
    pub flare_energy_margin_j: f64,
    /// Approximate engine power delivered to the rotor (W).
    pub engine_power_w: f64,
    /// Approximate aerodynamic power exchanged in autorotation (W).
    pub aerodynamic_power_w: f64,
    /// Induced flow through the main-rotor disk, m/s.
    pub induced_velocity_mps: f64,
    /// Lift multiplier after translational-lift or vortex-ring effects.
    pub aerodynamic_efficiency: f64,
    /// Explicit aerodynamic regime for safety/evidence logic.
    pub flight_regime: RotorFlightRegime,
    /// First-order longitudinal disk flap, radians.
    pub longitudinal_flap_rad: f64,
    /// First-order lateral disk flap, radians.
    pub lateral_flap_rad: f64,
    /// Thrust-dependent rotor coning angle, radians.
    pub coning_angle_rad: f64,
    /// Roll moment generated by the flapped rotor disk, N·m.
    pub hub_roll_moment_nm: f64,
    /// Pitch moment generated by the flapped rotor disk, N·m.
    pub hub_pitch_moment_nm: f64,
    /// Horizontal speed divided by rotor-tip speed.
    pub advance_ratio: f64,
    /// RPM-dependent cyclic control authority in [0, 1].
    pub hub_control_authority: f64,
    /// Local air density used for this step, kg/m³.
    pub air_density_kg_m3: f64,
    /// Local density divided by the calibrated reference density.
    pub density_ratio: f64,
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotor_hover_produces_thrust() {
        let mut rotor = RotorDynamics::new();
        let output = rotor.step(0.6, 0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.01);
        assert!(
            output.thrust_force > 0.0,
            "Hover should produce positive thrust"
        );
        assert!(!rotor.state().in_autorotation);
    }

    #[test]
    fn test_rotor_zero_command_no_thrust() {
        let mut rotor = RotorDynamics::new();
        // Run several steps to let RPM decay toward zero
        for _ in 0..1000 {
            rotor.step(0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.01);
        }
        let output = rotor.step(0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.01);
        // Thrust should be very low (RPM decayed toward 0)
        assert!(output.thrust_force < 10.0);
    }

    #[test]
    fn test_rpm_lag() {
        let mut rotor = RotorDynamics::new();
        let initial_rpm = rotor.state().main_rpm;

        // Command max thrust
        rotor.step(1.0, 0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.001);
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
            rotor.step(0.8, 0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.001);
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
        let output = rotor.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01);
        assert!(rotor.state().in_autorotation);
        assert!(output.thrust_force >= 0.0);
    }

    #[test]
    fn test_downward_inflow_sustains_autorotation_rpm() {
        let mut still = RotorDynamics::new();
        let mut descending = RotorDynamics::new();
        still.state.main_rpm = 1200.0;
        descending.state.main_rpm = 1200.0;
        for _ in 0..1000 {
            still.step(0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.001);
            descending.step(0.0, 0.0, 0.0, 0.1, 0.0, 0.0, -6.0, 0.001);
        }
        assert!(descending.state().main_rpm > still.state().main_rpm);
        assert!(descending.state().in_autorotation);
    }

    #[test]
    fn test_torque_reaction_increases_with_rpm() {
        let mut rotor = RotorDynamics::new();
        rotor.state.main_rpm = 2000.0;
        let out_low = rotor.step(0.6, 0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.01);

        rotor.state.main_rpm = 4000.0;
        let out_high = rotor.step(0.6, 0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.01);

        assert!(
            out_high.torque_reaction > out_low.torque_reaction,
            "Higher RPM should produce more torque reaction"
        );
    }

    #[test]
    fn test_precession_from_cyclic() {
        let mut rotor = RotorDynamics::new();
        // Longitudinal cyclic → roll precession
        let output = rotor.step(0.6, 0.5, 0.0, 0.3, 0.5, 0.0, 0.0, 0.01);
        assert!(
            output.precession_roll.abs() > 0.0,
            "Cyclic lon should produce roll precession"
        );

        // Lateral cyclic → pitch precession
        let output = rotor.step(0.6, 0.5, 0.0, 0.3, 0.0, 0.5, 0.0, 0.01);
        assert!(
            output.precession_pitch.abs() > 0.0,
            "Cyclic lat should produce pitch precession"
        );
    }

    #[test]
    fn test_hover_trim_matches_500kg_weight() {
        let mut rotor = RotorDynamics::new();
        for _ in 0..5000 {
            rotor.step(0.6, 0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.001);
        }
        let output = rotor.step(0.6, 0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.001);
        let weight = 500.0 * 9.81;
        assert!(
            (output.thrust_force - weight).abs() < 5.0,
            "trim thrust must match weight: thrust={}, weight={weight}",
            output.thrust_force
        );
    }

    #[test]
    fn test_tail_rotor_balances_and_controls_yaw() {
        let mut rotor = RotorDynamics::new();
        for _ in 0..5000 {
            rotor.step(0.6, 0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.001);
        }
        let neutral = rotor.step(0.6, 0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.001);
        let positive = rotor.step(0.6, 0.5, 0.5, 0.3, 0.0, 0.0, 0.0, 0.001);
        assert!((neutral.tail_yaw_torque - neutral.torque_reaction).abs() < 10.0);
        assert!(positive.tail_yaw_torque > neutral.tail_yaw_torque);
    }

    #[test]
    fn low_collective_preserves_autorotation_energy() {
        let mut low = RotorDynamics::new();
        let mut high = RotorDynamics::new();
        low.state.main_rpm = 2200.0;
        high.state.main_rpm = 2200.0;
        for _ in 0..2000 {
            low.step(0.0, 0.0, 0.0, -0.35, 0.0, 0.0, -7.0, 0.001);
            high.step(0.0, 0.0, 0.0, 0.45, 0.0, 0.0, -7.0, 0.001);
        }
        assert!(low.state().main_rpm > high.state().main_rpm);
        assert!(low.state().kinetic_energy_j > high.state().kinetic_energy_j);
    }

    #[test]
    fn rotor_energy_scales_with_rpm_squared() {
        let mut rotor = RotorDynamics::new();
        rotor.state.main_rpm = 1000.0;
        let low = rotor.step(1000.0 / 5500.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        rotor.state.main_rpm = 2000.0;
        let high = rotor.step(2000.0 / 5500.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let ratio = high.rotor_kinetic_energy_j / low.rotor_kinetic_energy_j;
        assert!((ratio - 4.0).abs() < 1.0e-9);
    }

    #[test]
    fn engine_power_is_zero_during_flameout_autorotation() {
        let mut rotor = RotorDynamics::new();
        rotor.state.main_rpm = 2000.0;
        let output = rotor.step(0.0, 0.0, 0.0, -0.2, 0.0, 0.0, -6.0, 0.01);
        assert!(rotor.state().in_autorotation);
        assert_eq!(output.engine_power_w, 0.0);
        assert!(output.rotor_kinetic_energy_j > 0.0);
    }

    #[test]
    fn test_reset_to_hover() {
        let mut rotor = RotorDynamics::new();
        rotor.state.main_rpm = 100.0;
        rotor.state.in_autorotation = true;
        rotor.reset();
        assert_eq!(rotor.state().main_rpm, 3300.0);
        assert!(!rotor.state().in_autorotation);
    }

    #[test]
    fn test_revolutions_accumulate() {
        let mut rotor = RotorDynamics::new();
        let before = rotor.state().main_revolutions;
        // At 3300 RPM for 1 second = 3300/60 = 55 revolutions
        for _ in 0..1000 {
            rotor.step(0.6, 0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.001);
        }
        let after = rotor.state().main_revolutions;
        assert!(
            after > before + 50.0,
            "Should accumulate ~55 revolutions in 1s at hover RPM"
        );
    }

    #[test]
    fn translational_lift_increases_thrust_at_forward_airspeed() {
        let mut slow = RotorDynamics::new();
        let mut fast = RotorDynamics::new();
        let slow_out = slow.step_with_flight_condition(
            0.6,
            0.5,
            0.0,
            0.3,
            0.0,
            0.0,
            RotorFlightCondition::new(0.0, 0.0, 20.0),
            0.01,
        );
        let fast_out = fast.step_with_flight_condition(
            0.6,
            0.5,
            0.0,
            0.3,
            0.0,
            0.0,
            RotorFlightCondition::new(20.0, 0.0, 20.0),
            0.01,
        );
        assert_eq!(
            fast_out.flight_regime,
            RotorFlightRegime::EffectiveTranslationalLift
        );
        assert!(fast_out.thrust_force > slow_out.thrust_force);
        assert!(fast_out.aerodynamic_efficiency > 1.0);
    }

    #[test]
    fn low_speed_descent_exposes_vortex_ring_regime() {
        let mut rotor = RotorDynamics::new();
        let output = rotor.step_with_flight_condition(
            0.6,
            0.5,
            0.0,
            0.3,
            0.0,
            0.0,
            RotorFlightCondition::new(1.0, -6.0, 20.0),
            0.01,
        );
        assert_eq!(output.flight_regime, RotorFlightRegime::VortexRingExposure);
        assert_eq!(
            output.aerodynamic_efficiency,
            rotor.config().vortex_ring_lift_factor
        );
    }

    #[test]
    fn induced_flow_is_finite_and_tracks_thrust() {
        let mut rotor = RotorDynamics::new();
        let before = rotor.state().induced_velocity_mps;
        for _ in 0..100 {
            rotor.step_with_flight_condition(
                0.8,
                0.5,
                0.0,
                0.4,
                0.0,
                0.0,
                RotorFlightCondition::new(0.0, 0.0, 20.0),
                0.01,
            );
        }
        assert!(rotor.state().induced_velocity_mps.is_finite());
        assert!(rotor.state().induced_velocity_mps > before);
    }

    #[test]
    fn lower_density_reduces_thrust_at_equal_controls() {
        let mut sea = RotorDynamics::new();
        let mut high = RotorDynamics::new();
        let sea_out = sea.step_with_flight_condition(
            0.6,
            0.5,
            0.0,
            0.3,
            0.0,
            0.0,
            RotorFlightCondition::new(0.0, 0.0, 20.0).with_air_density(1.225),
            0.01,
        );
        let high_out = high.step_with_flight_condition(
            0.6,
            0.5,
            0.0,
            0.3,
            0.0,
            0.0,
            RotorFlightCondition::new(0.0, 0.0, 2_500.0).with_air_density(0.95),
            0.01,
        );
        assert!(high_out.thrust_force < sea_out.thrust_force);
        assert!(high_out.density_ratio < sea_out.density_ratio);
    }
}
