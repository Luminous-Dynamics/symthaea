// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Physics simulator trait and simple implementation for SAR helicopter.

use crate::actuator_dynamics::ActuatorDynamics;
use crate::atmosphere::{AtmosphereSample, StandardAtmosphere};
use crate::mass_properties::{MassProperties, MassPropertiesModel};
use crate::perturbations::{HelicopterPerturbation, PerturbationEffects, PerturbationError};
use crate::powertrain::{PowertrainModel, PowertrainState};
use crate::rotor_dynamics::{RotorDynamics, RotorFlightCondition, RotorOutput};
use crate::types::{HelicopterCommand, HelicopterState};
use crate::wind_model::{WindConfig, WindModel};
use serde::{Deserialize, Serialize};

/// Classification of the most recent ground contact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LandingOutcome {
    Airborne,
    SafeTouchdown,
    HardLanding,
    Crash,
}

impl Default for LandingOutcome {
    fn default() -> Self {
        Self::Airborne
    }
}

/// Impact evidence retained after ground contact.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LandingContact {
    pub outcome: LandingOutcome,
    pub vertical_impact_speed_mps: f64,
    pub horizontal_impact_speed_mps: f64,
    pub uprightness: f64,
}

impl LandingContact {
    pub const AIRBORNE: Self = Self {
        outcome: LandingOutcome::Airborne,
        vertical_impact_speed_mps: 0.0,
        horizontal_impact_speed_mps: 0.0,
        uprightness: 1.0,
    };
}

/// Trait for helicopter physics simulation backends.
pub trait HelicopterPhysicsSimulator {
    /// Advance one timestep with the given motor command.
    fn step(&mut self, cmd: &HelicopterCommand, dt: f64);
    /// Current flight state.
    fn state(&self) -> &HelicopterState;
    /// Reset to hover at the given altitude.
    fn reset(&mut self, altitude: f64);
    /// Reset to hover with a deterministic perturbation.
    fn reset_with_perturbation(&mut self, altitude: f64, perturbation: f64, seed: u64);
    /// Apply an external force (world-frame, Newtons) for the next step.
    fn apply_external_force(&mut self, force: [f64; 3]);
    /// Replace the current fault state from the active perturbation set.
    /// Backends that do not model faults may retain the default no-op.
    fn set_active_perturbations(
        &mut self,
        _active: &[&HelicopterPerturbation],
    ) -> Result<(), PerturbationError> {
        Ok(())
    }
    /// Most recent ground-contact classification.
    fn landing_contact(&self) -> LandingContact {
        LandingContact::AIRBORNE
    }
}

/// Simple Rust-native physics simulator for SAR helicopter.
///
/// Uses the [`RotorDynamics`] model for thrust/torque computation and
/// semi-implicit Euler integration for body dynamics.
///
/// Atmosphere: owns a [`WindModel`] — each step the gust state evolves,
/// aerodynamic drag acts on *relative* airspeed (vehicle velocity minus
/// wind), and rotor thrust is multiplied by the Cheeseman-Bennett ground
/// effect ratio. The default wind config is calm (zero steady wind, zero
/// gusts), which reproduces still-air drag exactly; enable wind explicitly
/// via [`SimpleHelicopterSimulator::set_wind`] with a seed so runs stay
/// deterministic.
///
/// Mass: 500 kg (light SAR helicopter, e.g., Robinson R44 class).
pub struct SimpleHelicopterSimulator {
    state: HelicopterState,
    rotor: RotorDynamics,
    actuator: ActuatorDynamics,
    powertrain: PowertrainModel,
    wind: WindModel,
    atmosphere: StandardAtmosphere,
    last_atmosphere: AtmosphereSample,
    mass_model: MassPropertiesModel,
    last_mass_properties: MassProperties,
    perturbation_effects: PerturbationEffects,
    external_force: [f64; 3],
    drag_coeff: f64,
    angular_damping: f64,
    landing_contact: LandingContact,
}

impl SimpleHelicopterSimulator {
    /// Create a new simulator hovering at 20m (calm air).
    pub fn new() -> Self {
        Self {
            state: HelicopterState::hover(20.0),
            rotor: RotorDynamics::new(),
            actuator: ActuatorDynamics::new(),
            powertrain: PowertrainModel::new(),
            wind: WindModel::calm(),
            atmosphere: StandardAtmosphere::default(),
            last_atmosphere: StandardAtmosphere::default()
                .sample_bounded(20.0)
                .unwrap_or_else(|_| AtmosphereSample::sea_level()),
            mass_model: MassPropertiesModel::default(),
            last_mass_properties: MassPropertiesModel::default()
                .properties_with_payload_drop(0.0)
                .expect("default mass model must remain valid"),
            perturbation_effects: PerturbationEffects::default(),
            external_force: [0.0; 3],
            drag_coeff: 0.15, // Higher drag than quadrotor (larger body)
            angular_damping: 2.0,
            landing_contact: LandingContact::AIRBORNE,
        }
    }

    /// Enable wind with the given config and gust seed (deterministic).
    pub fn set_wind(&mut self, config: WindConfig, seed: u64) {
        self.wind = WindModel::with_seed(config, seed);
    }

    /// Replace the deterministic atmosphere used by the simulator.
    pub fn set_atmosphere(&mut self, atmosphere: StandardAtmosphere) {
        self.atmosphere = atmosphere;
        self.last_atmosphere = self
            .atmosphere
            .sample_bounded(self.state.altitude())
            .unwrap_or(self.last_atmosphere);
    }

    /// Most recent local atmosphere sample.
    pub fn atmosphere_sample(&self) -> AtmosphereSample {
        self.last_atmosphere
    }

    /// Clone of the current state for evidence recording.
    pub fn state_snapshot(&self) -> HelicopterState {
        self.state.clone()
    }

    /// Current main-rotor kinetic energy.
    pub fn rotor_kinetic_energy_j(&self) -> f64 {
        self.rotor.state().kinetic_energy_j
    }

    /// Last command after servo/governor lag and slew limiting.
    pub fn applied_command(&self) -> HelicopterCommand {
        self.actuator.applied_command()
    }

    /// Current powertrain/fuel state.
    pub fn powertrain_state(&self) -> PowertrainState {
        self.powertrain.state()
    }

    /// Mutable powertrain access for scenario setup and evidence tests.
    pub fn powertrain_mut(&mut self) -> &mut PowertrainModel {
        &mut self.powertrain
    }

    /// Current aggregate fault/perturbation effects.
    pub fn perturbation_effects(&self) -> PerturbationEffects {
        self.perturbation_effects
    }

    /// Replace the rigid-body mass model used by the simulator.
    pub fn set_mass_model(
        &mut self,
        model: MassPropertiesModel,
    ) -> Result<(), crate::mass_properties::MassPropertiesError> {
        let properties =
            model.properties_with_payload_drop(self.perturbation_effects.payload_mass_drop_kg)?;
        self.mass_model = model;
        self.last_mass_properties = properties;
        Ok(())
    }

    /// Current mass, center-of-gravity, and inertia evidence.
    pub fn mass_properties(&self) -> MassProperties {
        self.last_mass_properties
    }

    /// Effective mass after an active payload drop.
    pub fn effective_mass(&self) -> f64 {
        self.last_mass_properties.total_mass_kg
    }

    /// Most recent landing/contact evidence.
    pub fn landing_contact(&self) -> LandingContact {
        self.landing_contact
    }

    /// Access the current wind model.
    pub fn wind_model(&self) -> &WindModel {
        &self.wind
    }
}

impl Default for SimpleHelicopterSimulator {
    fn default() -> Self {
        Self::new()
    }
}

impl HelicopterPhysicsSimulator for SimpleHelicopterSimulator {
    fn step(&mut self, cmd: &HelicopterCommand, dt: f64) {
        let g = 9.81;

        // 1. Evolve wind first so rotor aerodynamics and body drag use the
        // same air-relative condition for this integration step.
        let wind_vel = self.wind.wind_velocity_step(dt);

        self.last_mass_properties = self
            .mass_model
            .properties_with_payload_drop(self.perturbation_effects.payload_mass_drop_kg)
            .unwrap_or_else(|_| {
                self.mass_model
                    .properties_with_payload_drop(0.0)
                    .expect("validated mass model must produce nominal properties")
            });

        // 2. Compile fault-aware actuator request, then step servos/governors.
        let mut fault_aware_cmd = *cmd;
        if !self.perturbation_effects.engine_available {
            fault_aware_cmd.thrust = 0.0;
        } else {
            fault_aware_cmd.thrust *= self.powertrain.available_power_fraction() as f32;
        }
        let applied_cmd = self.actuator.step(&fault_aware_cmd, dt);
        self.last_atmosphere = self
            .atmosphere
            .sample_bounded(self.state.altitude())
            .unwrap_or(self.last_atmosphere);
        let relative_vx = self.state.linear_velocity[0] - wind_vel[0];
        let relative_vy = self.state.linear_velocity[1] - wind_vel[1];
        let relative_vz = self.state.linear_velocity[2] - wind_vel[2];
        let rotor_out: RotorOutput = self.rotor.step_with_flight_condition(
            applied_cmd.thrust as f64 * self.perturbation_effects.main_rotor_efficiency,
            applied_cmd.tail_rotor as f64 * self.perturbation_effects.tail_rotor_efficiency,
            applied_cmd.pedal as f64,
            applied_cmd.collective as f64,
            applied_cmd.cyclic_lon as f64,
            applied_cmd.cyclic_lat as f64,
            RotorFlightCondition::new(
                (relative_vx * relative_vx + relative_vy * relative_vy).sqrt(),
                relative_vz,
                self.state.position[2],
            )
            .with_air_density(self.last_atmosphere.density_kg_m3),
            dt,
        );
        self.powertrain.step(rotor_out.engine_power_w, dt);

        // Update rotor feedback in state
        self.state.main_rotor_rpm = self.rotor.state().main_rpm;
        self.state.tail_rotor_rpm = self.rotor.state().tail_rpm;
        self.state.collective_pitch = applied_cmd.collective as f64 * 0.26; // Normalized → radians
        self.state.cyclic_lon_feedback = applied_cmd.cyclic_lon as f64 * 0.15;
        self.state.cyclic_lat_feedback = applied_cmd.cyclic_lat as f64 * 0.15;

        // 3. Autorotation remains in the same integration path as powered
        // flight. RotorDynamics supplies reduced lift and windmilling RPM;
        // wind, external forces, attitude, and horizontal motion still evolve.

        // 4. Thrust vector in world frame (rotate body-z by quaternion),
        //    augmented by ground effect near the surface (Cheeseman-Bennett).
        let [w, x, y, z] = self.state.quaternion;
        let thrust = rotor_out.thrust_force * self.wind.ground_effect_ratio(self.state.position[2]);

        let fx = 2.0 * (x * z + w * y) * thrust
            + self.external_force[0]
            + self.perturbation_effects.crosswind_force[0];
        let fy = 2.0 * (y * z - w * x) * thrust
            + self.external_force[1]
            + self.perturbation_effects.crosswind_force[1];
        let fz = (1.0 - 2.0 * (x * x + y * y)) * thrust
            + self.external_force[2]
            + self.perturbation_effects.crosswind_force[2];

        // 5. Linear acceleration with quadratic drag: F_drag = -c × |v_rel| × v_rel
        // Quadratic drag better models aerodynamic forces at helicopter speeds
        // (Fossen 2011). The drag coefficient absorbs 0.5 × ρ × Cd × A / mass.
        // Drag acts on airspeed relative to the wind: with the default calm
        // config wind_vel is exactly [0,0,0] and this reduces to still-air drag.
        let mass = self.last_mass_properties.total_mass_kg;
        let drag = self.drag_coeff / mass;
        let vx = self.state.linear_velocity[0] - wind_vel[0];
        let vy = self.state.linear_velocity[1] - wind_vel[1];
        let vz = self.state.linear_velocity[2] - wind_vel[2];
        let ax = fx / mass - drag * vx.abs() * vx;
        let ay = fy / mass - drag * vy.abs() * vy;
        let az = fz / mass - g - drag * vz.abs() * vz;

        // 6. Semi-implicit Euler
        self.state.linear_velocity[0] += ax * dt;
        self.state.linear_velocity[1] += ay * dt;
        self.state.linear_velocity[2] += az * dt;

        self.state.position[0] += self.state.linear_velocity[0] * dt;
        self.state.position[1] += self.state.linear_velocity[1] * dt;
        self.state.position[2] += self.state.linear_velocity[2] * dt;

        // Ground contact classification. A hard clamp alone cannot distinguish
        // a controlled touchdown from a destructive impact.
        if self.state.position[2] <= 0.0 {
            // Latch the first impact evidence until reset. Otherwise a crash
            // would be overwritten by a zero-velocity "safe" contact on the
            // following integration step.
            if matches!(self.landing_contact.outcome, LandingOutcome::Airborne) {
                let vertical_impact_speed_mps = (-self.state.linear_velocity[2]).max(0.0);
                let horizontal_impact_speed_mps = self.state.horizontal_speed();
                let uprightness = self.state.uprightness();
                let outcome = if vertical_impact_speed_mps <= 1.5
                    && horizontal_impact_speed_mps <= 2.0
                    && uprightness >= 0.90
                {
                    LandingOutcome::SafeTouchdown
                } else if vertical_impact_speed_mps <= 4.0
                    && horizontal_impact_speed_mps <= 5.0
                    && uprightness >= 0.70
                {
                    LandingOutcome::HardLanding
                } else {
                    LandingOutcome::Crash
                };
                self.landing_contact = LandingContact {
                    outcome,
                    vertical_impact_speed_mps,
                    horizontal_impact_speed_mps,
                    uprightness,
                };
            }
            self.state.position[2] = 0.0;
            self.state.linear_velocity = [0.0; 3];
            self.state.angular_velocity = [0.0; 3];
        }

        // 7. Angular dynamics
        // Cyclic → body moments + gyroscopic precession from rotor model
        // A displaced center of gravity creates a trim moment because the main
        // rotor thrust line is referenced to the body origin.
        let cg = self.last_mass_properties.center_of_gravity_body_m;
        let cg_roll_moment_nm = -cg[1] * thrust;
        let cg_pitch_moment_nm = cg[0] * thrust;
        let inertia = self.last_mass_properties.diagonal_inertia_about_cg_kg_m2;
        let moment_roll =
            (rotor_out.hub_roll_moment_nm + rotor_out.precession_roll + cg_roll_moment_nm)
                / inertia[0];
        let moment_pitch =
            (rotor_out.hub_pitch_moment_nm + rotor_out.precession_pitch + cg_pitch_moment_nm)
                / inertia[1];
        // Tail rotor anti-torque opposes the main-rotor reaction. Pedal is
        // already represented as tail-blade pitch inside RotorDynamics.
        let moment_yaw = (rotor_out.tail_yaw_torque - rotor_out.torque_reaction) / inertia[2];

        self.state.angular_velocity[0] += moment_roll * dt;
        self.state.angular_velocity[1] += moment_pitch * dt;
        self.state.angular_velocity[2] += moment_yaw * dt;

        // Angular damping
        let ang_decay = (-self.angular_damping * dt).exp();
        for av in &mut self.state.angular_velocity {
            *av *= ang_decay;
        }

        // 8. Quaternion integration (semi-implicit)
        let [qw, qx, qy, qz] = self.state.quaternion;
        let [wx, wy, wz] = self.state.angular_velocity;
        let half_dt = 0.5 * dt;
        let dqw = (-qx * wx - qy * wy - qz * wz) * half_dt;
        let dqx = (qw * wx + qy * wz - qz * wy) * half_dt;
        let dqy = (qw * wy - qx * wz + qz * wx) * half_dt;
        let dqz = (qw * wz + qx * wy - qy * wx) * half_dt;

        self.state.quaternion[0] += dqw;
        self.state.quaternion[1] += dqx;
        self.state.quaternion[2] += dqy;
        self.state.quaternion[3] += dqz;

        // Normalize quaternion
        let norm = (self.state.quaternion[0].powi(2)
            + self.state.quaternion[1].powi(2)
            + self.state.quaternion[2].powi(2)
            + self.state.quaternion[3].powi(2))
        .sqrt();
        if norm > 1e-10 {
            for q in &mut self.state.quaternion {
                *q /= norm;
            }
        }

        // Clear one-shot external force
        self.external_force = [0.0; 3];
    }

    fn state(&self) -> &HelicopterState {
        &self.state
    }

    fn reset(&mut self, altitude: f64) {
        self.state = HelicopterState::hover(altitude);
        self.rotor.reset();
        self.actuator.reset_hover();
        self.powertrain.reset();
        self.wind.reset();
        self.last_atmosphere = self
            .atmosphere
            .sample_bounded(altitude)
            .unwrap_or(self.last_atmosphere);
        self.external_force = [0.0; 3];
        self.perturbation_effects = PerturbationEffects::default();
        self.last_mass_properties = self
            .mass_model
            .properties_with_payload_drop(0.0)
            .expect("validated mass model must produce nominal properties");
        self.landing_contact = LandingContact::AIRBORNE;
    }

    fn reset_with_perturbation(&mut self, altitude: f64, perturbation: f64, _seed: u64) {
        self.reset(altitude);
        // Add lateral velocity as perturbation
        self.state.linear_velocity[0] = perturbation;
    }

    fn apply_external_force(&mut self, force: [f64; 3]) {
        self.external_force = force;
    }

    fn set_active_perturbations(
        &mut self,
        active: &[&HelicopterPerturbation],
    ) -> Result<(), PerturbationError> {
        let effects = PerturbationEffects::from_active(active)?;
        let properties = self
            .mass_model
            .properties_with_payload_drop(effects.payload_mass_drop_kg)
            .map_err(|_| PerturbationError::PayloadDropExceedsAvailable)?;
        self.perturbation_effects = effects;
        self.last_mass_properties = properties;
        Ok(())
    }

    fn landing_contact(&self) -> LandingContact {
        self.landing_contact
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hover_stability() {
        let mut sim = SimpleHelicopterSimulator::new();
        let cmd = HelicopterCommand::hover();
        let initial_alt = sim.state().altitude();

        // Run 5 seconds at 300Hz
        for _ in 0..1500 {
            sim.step(&cmd, 1.0 / 300.0);
        }

        let final_alt = sim.state().altitude();
        // A declared hover trim must remain near its initial altitude.
        assert!(
            (final_alt - initial_alt).abs() < 0.75,
            "Hover should be roughly stable: initial={initial_alt}, final={final_alt}"
        );
        assert!(sim.state().is_finite(), "State must remain finite");
    }

    #[test]
    fn test_zero_command_descends() {
        let mut sim = SimpleHelicopterSimulator::new();
        let cmd = HelicopterCommand::zero();
        let initial_alt = sim.state().altitude();

        for _ in 0..3000 {
            sim.step(&cmd, 1.0 / 300.0);
        }

        let final_alt = sim.state().altitude();
        assert!(
            final_alt < initial_alt,
            "Zero thrust should descend: {final_alt} < {initial_alt}"
        );
    }

    #[test]
    fn test_external_force() {
        let mut sim = SimpleHelicopterSimulator::new();
        sim.apply_external_force([1000.0, 0.0, 0.0]); // 1000N lateral push
        sim.step(&HelicopterCommand::hover(), 0.01);

        assert!(
            sim.state().linear_velocity[0] > 0.0,
            "External force should accelerate in X"
        );
    }

    #[test]
    fn test_ground_constraint() {
        let mut sim = SimpleHelicopterSimulator::new();
        sim.reset(1.0); // Start at 1m
        let cmd = HelicopterCommand::zero();

        // Run until should hit ground
        for _ in 0..10000 {
            sim.step(&cmd, 0.001);
        }

        assert!(
            sim.state().position[2] >= 0.0,
            "Ground constraint should prevent negative altitude"
        );
    }

    #[test]
    fn test_quaternion_normalized() {
        let mut sim = SimpleHelicopterSimulator::new();
        let cmd = HelicopterCommand {
            collective: 0.3,
            cyclic_lon: 0.5,
            cyclic_lat: -0.3,
            pedal: 0.2,
            thrust: 0.6,
            tail_rotor: 0.5,
        };

        for _ in 0..1000 {
            sim.step(&cmd, 1.0 / 300.0);
        }

        let q = sim.state().quaternion;
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Quaternion should remain normalized: norm={norm}"
        );
    }

    #[test]
    fn test_reset() {
        let mut sim = SimpleHelicopterSimulator::new();
        sim.step(&HelicopterCommand::hover(), 0.01);
        sim.reset(50.0);
        assert!((sim.state().altitude() - 50.0).abs() < 1e-10);
        assert_eq!(sim.state().speed(), 0.0);
    }

    #[test]
    fn test_autorotation_descent() {
        let mut sim = SimpleHelicopterSimulator::new();
        // Force low RPM
        sim.rotor.state_mut().main_rpm = 500.0;
        let cmd = HelicopterCommand::zero();
        // The actuator starts at hover trim and slews toward zero thrust
        // with a ~0.18s time constant — one 10ms step isn't enough for the
        // commanded thrust to actually drop below the autorotation
        // threshold, so step long enough for it to settle.
        for _ in 0..50 {
            sim.step(&cmd, 0.01);
        }

        // Should be descending in autorotation
        assert!(sim.rotor.state().in_autorotation);
    }

    #[test]
    fn test_autorotation_does_not_bypass_external_forces() {
        let mut sim = SimpleHelicopterSimulator::new();
        sim.rotor.state_mut().main_rpm = 500.0;
        sim.apply_external_force([1000.0, 0.0, 0.0]);
        for _ in 0..50 {
            sim.step(&HelicopterCommand::zero(), 0.01);
        }
        assert!(sim.rotor.state().in_autorotation);
        assert!(sim.state().linear_velocity[0] > 0.0);
    }

    #[test]
    fn test_rotor_feedback_in_state() {
        let mut sim = SimpleHelicopterSimulator::new();
        sim.step(&HelicopterCommand::hover(), 0.01);
        assert!(sim.state().main_rotor_rpm > 0.0);
        assert!(sim.state().tail_rotor_rpm > 0.0);
    }

    #[test]
    fn test_pedal_changes_yaw_without_direct_moment_shortcut() {
        let mut neutral = SimpleHelicopterSimulator::new();
        let mut pedal = SimpleHelicopterSimulator::new();
        let neutral_cmd = HelicopterCommand::hover();
        let mut pedal_cmd = HelicopterCommand::hover();
        pedal_cmd.pedal = 0.5;
        for _ in 0..300 {
            neutral.step(&neutral_cmd, 1.0 / 300.0);
            pedal.step(&pedal_cmd, 1.0 / 300.0);
        }
        assert!(pedal.state().angular_velocity[2] > neutral.state().angular_velocity[2]);
    }

    #[test]
    fn test_steady_wind_pushes_downwind() {
        let mut calm = SimpleHelicopterSimulator::new();
        let mut windy = SimpleHelicopterSimulator::new();
        windy.set_wind(
            WindConfig {
                steady_wind: [10.0, 0.0, 0.0],
                gust_intensity: 0.0, // deterministic: no gusts
                ..WindConfig::default()
            },
            42,
        );

        let cmd = HelicopterCommand::hover();
        for _ in 0..900 {
            calm.step(&cmd, 1.0 / 300.0);
            windy.step(&cmd, 1.0 / 300.0);
        }

        assert!(
            windy.state().linear_velocity[0] > calm.state().linear_velocity[0],
            "Steady +x wind must push the helicopter downwind: windy vx={}, calm vx={}",
            windy.state().linear_velocity[0],
            calm.state().linear_velocity[0]
        );
        assert!(
            windy.state().linear_velocity[0] > 0.0,
            "Downwind drift must be positive: vx={}",
            windy.state().linear_velocity[0]
        );
    }

    #[test]
    fn test_gusty_wind_deterministic_with_seed() {
        let make = || {
            let mut sim = SimpleHelicopterSimulator::new();
            sim.set_wind(WindConfig::moderate_wind(), 7);
            sim
        };
        let mut a = make();
        let mut b = make();
        let cmd = HelicopterCommand::hover();
        for _ in 0..300 {
            a.step(&cmd, 1.0 / 300.0);
            b.step(&cmd, 1.0 / 300.0);
        }
        assert_eq!(a.state().position, b.state().position);
        assert_eq!(a.state().linear_velocity, b.state().linear_velocity);
    }

    #[test]
    fn test_safe_touchdown_is_not_reported_as_crash() {
        let mut sim = SimpleHelicopterSimulator::new();
        sim.state.position[2] = 0.001;
        sim.state.linear_velocity[2] = -1.0;
        sim.step(&HelicopterCommand::zero(), 0.01);
        assert_eq!(sim.landing_contact().outcome, LandingOutcome::SafeTouchdown);
    }

    #[test]
    fn test_high_sink_rate_is_reported_as_crash() {
        let mut sim = SimpleHelicopterSimulator::new();
        sim.state.position[2] = 0.001;
        sim.state.linear_velocity[2] = -8.0;
        sim.step(&HelicopterCommand::zero(), 0.01);
        assert_eq!(sim.landing_contact().outcome, LandingOutcome::Crash);
        assert!(sim.landing_contact().vertical_impact_speed_mps > 4.0);
    }

    #[test]
    fn test_crash_evidence_is_latched_until_reset() {
        let mut sim = SimpleHelicopterSimulator::new();
        sim.state.position[2] = 0.001;
        sim.state.linear_velocity[2] = -8.0;
        sim.step(&HelicopterCommand::zero(), 0.01);
        let impact = sim.landing_contact();
        sim.step(&HelicopterCommand::zero(), 0.01);
        assert_eq!(sim.landing_contact(), impact);
        sim.reset(20.0);
        assert_eq!(sim.landing_contact().outcome, LandingOutcome::Airborne);
    }

    #[test]
    fn test_actuator_feedback_does_not_jump_instantly() {
        let mut sim = SimpleHelicopterSimulator::new();
        let mut cmd = HelicopterCommand::hover();
        cmd.cyclic_lon = 1.0;
        sim.step(&cmd, 0.01);
        assert!(sim.applied_command().cyclic_lon > 0.0);
        assert!(sim.applied_command().cyclic_lon < 1.0);
        assert!(sim.state().cyclic_lon_feedback < 0.15);
    }

    #[test]
    fn test_engine_flameout_reduces_rotor_command() {
        let mut nominal = SimpleHelicopterSimulator::new();
        let mut failed = SimpleHelicopterSimulator::new();
        let flameout = HelicopterPerturbation::EngineFlameout;
        failed.set_active_perturbations(&[&flameout]).unwrap();
        for _ in 0..300 {
            nominal.step(&HelicopterCommand::hover(), 1.0 / 300.0);
            failed.step(&HelicopterCommand::hover(), 1.0 / 300.0);
        }
        assert!(failed.state().main_rotor_rpm < nominal.state().main_rotor_rpm);
    }

    #[test]
    fn test_tail_rotor_failure_removes_anti_torque() {
        let mut nominal = SimpleHelicopterSimulator::new();
        let mut failed = SimpleHelicopterSimulator::new();
        let tail_failure = HelicopterPerturbation::TailRotorFailure;
        failed.set_active_perturbations(&[&tail_failure]).unwrap();
        for _ in 0..300 {
            nominal.step(&HelicopterCommand::hover(), 1.0 / 300.0);
            failed.step(&HelicopterCommand::hover(), 1.0 / 300.0);
        }
        assert!(failed.state().angular_velocity[2] < nominal.state().angular_velocity[2]);
    }

    #[test]
    fn test_payload_drop_changes_effective_mass_once() {
        let mut sim = SimpleHelicopterSimulator::new();
        let payload = HelicopterPerturbation::PayloadDrop { mass_kg: 75.0 };
        sim.set_active_perturbations(&[&payload]).unwrap();
        assert_eq!(sim.effective_mass(), 425.0);
        sim.set_active_perturbations(&[&payload]).unwrap();
        assert_eq!(sim.effective_mass(), 425.0);
    }

    #[test]
    fn test_hover_consumes_fuel() {
        let mut sim = SimpleHelicopterSimulator::new();
        let initial = sim.powertrain_state().fuel_kg;
        for _ in 0..300 {
            sim.step(&HelicopterCommand::hover(), 1.0 / 300.0);
        }
        assert!(sim.powertrain_state().fuel_kg < initial);
        assert!(sim.powertrain_state().cumulative_energy_j > 0.0);
    }

    #[test]
    fn test_fuel_exhaustion_removes_power() {
        let mut sim = SimpleHelicopterSimulator::new();
        sim.powertrain_mut().set_fuel_fraction(0.0);
        for _ in 0..300 {
            sim.step(&HelicopterCommand::hover(), 1.0 / 300.0);
        }
        assert_eq!(sim.powertrain_state().delivery_fraction, 0.0);
        assert!(sim.state().main_rotor_rpm < 3300.0);
    }

    #[test]
    fn test_ground_effect_boosts_thrust_near_ground() {
        // Same hover command, low vs high altitude: ground effect must
        // produce a higher climb rate (or slower descent) near the ground.
        let mut low = SimpleHelicopterSimulator::new();
        let mut high = SimpleHelicopterSimulator::new();
        low.reset(3.0);
        high.reset(100.0);

        let cmd = HelicopterCommand::hover();
        for _ in 0..300 {
            low.step(&cmd, 1.0 / 300.0);
            high.step(&cmd, 1.0 / 300.0);
        }

        let low_climb = low.state().altitude() - 3.0;
        let high_climb = high.state().altitude() - 100.0;
        assert!(
            low_climb > high_climb,
            "Ground effect should augment thrust near ground: low_climb={low_climb}, high_climb={high_climb}"
        );
    }
    #[test]
    fn hot_high_atmosphere_reduces_available_lift() {
        let mut standard = SimpleHelicopterSimulator::new();
        standard.reset(2_500.0);
        let mut hot = SimpleHelicopterSimulator::new();
        hot.set_atmosphere(
            StandardAtmosphere::with_temperature_offset(
                crate::atmosphere::StandardAtmosphereConfig::default(),
                25.0,
            )
            .unwrap(),
        );
        hot.reset(2_500.0);
        assert!(hot.atmosphere_sample().density_kg_m3 < standard.atmosphere_sample().density_kg_m3);
        let command = HelicopterCommand::hover();
        standard.step(&command, 0.02);
        hot.step(&command, 0.02);
        assert!(hot.state().linear_velocity[2] < standard.state().linear_velocity[2]);
    }

    #[test]
    fn payload_drop_updates_center_of_gravity_and_inertia() {
        let mut sim = SimpleHelicopterSimulator::new();
        let loaded = sim.mass_properties();
        let payload = HelicopterPerturbation::PayloadDrop { mass_kg: 75.0 };
        sim.set_active_perturbations(&[&payload]).unwrap();
        sim.step(&HelicopterCommand::hover(), 0.01);
        let dropped = sim.mass_properties();
        assert_eq!(dropped.total_mass_kg, 425.0);
        assert!(dropped.center_of_gravity_body_m[0] < loaded.center_of_gravity_body_m[0]);
        assert_ne!(
            dropped.diagonal_inertia_about_cg_kg_m2,
            loaded.diagonal_inertia_about_cg_kg_m2
        );
    }
}
