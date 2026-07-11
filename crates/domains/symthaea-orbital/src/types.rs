use orbital_mechanics::coordinates::wgs84::{A as EARTH_RADIUS_KM, MU};
use serde::{Deserialize, Serialize};
pub const NUM_JOINTS: usize = 7;
pub const NUM_STATE_CHANNELS: usize = 33;
pub const NUM_ACTUATORS: usize = 7;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitalState {
    pub joint_angles: [f64; NUM_JOINTS],
    pub joint_velocities: [f64; NUM_JOINTS],
    pub ee_position: [f64; 3],
    pub spacecraft_angular_velocity: [f64; 3],
    pub solar_exposure: f64,
    pub comm_window: f64,
    /// Bus position in a fixed (non-rotating) Earth-centered inertial frame, km.
    /// Real two-body-propagated state — see `simulator.rs`.
    pub position_km: [f64; 3],
    /// Bus velocity in the same frame, km/s.
    pub velocity_km_s: [f64; 3],
    /// Cumulative impulsive delta-v spent so far, m/s. Compare against
    /// `OrbitalConfig::delta_v_budget_m_s` via `delta_v_remaining_m_s()`.
    pub delta_v_used_m_s: f64,
    /// Cumulative reaction-wheel desaturation propellant spent, N·m·s
    /// (angular impulse) — deliberately a separate unit/budget from
    /// `delta_v_used_m_s` (translational burns), not reused for it, since
    /// mixing linear delta-v and angular-impulse propellant into one number
    /// would be dimensionally dishonest. Compare against
    /// `OrbitalConfig::desaturation_budget_nms`.
    pub desaturation_used_nms: f64,
}
impl OrbitalState {
    pub fn stowed() -> Self {
        Self::circular_orbit(400.0)
    }

    /// Initial condition: circular orbit at the given altitude (km), in the
    /// XY plane of the inertial frame (position on +X axis, velocity on +Y).
    pub fn circular_orbit(altitude_km: f64) -> Self {
        let r = EARTH_RADIUS_KM + altitude_km;
        let v_circular = (MU / r).sqrt();
        Self {
            joint_angles: [0.0; NUM_JOINTS],
            joint_velocities: [0.0; NUM_JOINTS],
            ee_position: [0.0, 0.0, 2.0],
            spacecraft_angular_velocity: [0.0; 3],
            solar_exposure: 1.0,
            comm_window: 1.0,
            position_km: [r, 0.0, 0.0],
            velocity_km_s: [0.0, v_circular, 0.0],
            delta_v_used_m_s: 0.0,
            desaturation_used_nms: 0.0,
        }
    }

    /// Altitude above the WGS-84 equatorial radius, km.
    pub fn altitude_km(&self) -> f64 {
        let r = (self.position_km[0].powi(2)
            + self.position_km[1].powi(2)
            + self.position_km[2].powi(2))
        .sqrt();
        r - EARTH_RADIUS_KM
    }

    pub fn delta_v_remaining_m_s(&self, budget_m_s: f64) -> f64 {
        (budget_m_s - self.delta_v_used_m_s).max(0.0)
    }

    pub fn desaturation_remaining_nms(&self, budget_nms: f64) -> f64 {
        (budget_nms - self.desaturation_used_nms).max(0.0)
    }

    pub fn to_channels(&self) -> [f32; NUM_STATE_CHANNELS] {
        let mut c = [0.0f32; NUM_STATE_CHANNELS];
        for i in 0..NUM_JOINTS {
            c[i] = self.joint_angles[i] as f32;
            c[i + 7] = self.joint_velocities[i] as f32;
        }
        c[14] = self.ee_position[0] as f32;
        c[15] = self.ee_position[1] as f32;
        c[16] = self.ee_position[2] as f32;
        c[17] = self.spacecraft_angular_velocity[0] as f32;
        c[18] = self.spacecraft_angular_velocity[1] as f32;
        c[19] = self.spacecraft_angular_velocity[2] as f32;
        c[20] = self.solar_exposure as f32;
        c[21] = self.comm_window as f32;
        c[22] = (self
            .spacecraft_angular_velocity
            .iter()
            .map(|v| v * v)
            .sum::<f64>())
        .sqrt() as f32;
        c[23] = (self.ee_position.iter().map(|p| p * p).sum::<f64>()).sqrt() as f32;
        c[24] = self.joint_velocities.iter().map(|v| v * v).sum::<f64>() as f32;
        c[25] = (self.solar_exposure * self.comm_window) as f32;
        c[26] = self.position_km[0] as f32;
        c[27] = self.position_km[1] as f32;
        c[28] = self.position_km[2] as f32;
        c[29] = self.velocity_km_s[0] as f32;
        c[30] = self.velocity_km_s[1] as f32;
        c[31] = self.velocity_km_s[2] as f32;
        c[32] = self.altitude_km() as f32;
        c
    }
    pub fn is_finite(&self) -> bool {
        self.joint_angles
            .iter()
            .chain(self.joint_velocities.iter())
            .chain(self.ee_position.iter())
            .chain(self.spacecraft_angular_velocity.iter())
            .chain(self.position_km.iter())
            .chain(self.velocity_km_s.iter())
            .chain(std::iter::once(&self.delta_v_used_m_s))
            .chain(std::iter::once(&self.desaturation_used_nms))
            .all(|v| v.is_finite())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OrbitalCommand {
    pub joint_torques: [f32; NUM_ACTUATORS],
    /// Impulsive translational burn commanded this step, m/s per axis (bus
    /// inertial frame). Applied directly to velocity, then charged against
    /// `OrbitalConfig::delta_v_budget_m_s` — see `simulator.rs`.
    pub translational_burn_mps: [f32; 3],
    /// Reaction-wheel desaturation command, N·m per axis: RCS thrusters fire
    /// to unload wheel momentum toward zero while (ideally) leaving bus
    /// attitude undisturbed. Only ever REDUCES |wheel momentum| — it can't
    /// be used to spin a wheel up further (see `simulator.rs`). Charged
    /// against `OrbitalConfig::desaturation_budget_nms`.
    pub desaturation_torque_nm: [f32; 3],
}
impl OrbitalCommand {
    pub fn zero() -> Self {
        Self {
            joint_torques: [0.0; NUM_ACTUATORS],
            translational_burn_mps: [0.0; 3],
            desaturation_torque_nm: [0.0; 3],
        }
    }
    pub fn control_effort(&self) -> f32 {
        self.joint_torques.iter().map(|t| t.abs()).sum::<f32>() / NUM_ACTUATORS as f32
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitalConfig {
    pub max_joint_torques: [f64; NUM_JOINTS],
    pub spacecraft_inertia: [f64; 3],
    pub reaction_wheel_capacity: f64,
    pub physics_hz: f64,
    pub cognitive_interval: usize,
    pub steps_per_episode: usize,
    pub network_layers: usize,
    pub neurons_per_layer: usize,
    pub learning_rate: f32,
    /// Total impulsive delta-v available for the episode, m/s. Illustrative
    /// placeholder order-of-magnitude for a small RCS budget, not a specific
    /// mission's figure — mark as estimated in any downstream use.
    pub delta_v_budget_m_s: f64,
    /// Drag coefficient (dimensionless, typical LEO 2.0-2.5).
    pub drag_cd: f64,
    /// Cross-sectional area for drag, m². Illustrative placeholder.
    pub drag_area_m2: f64,
    /// Spacecraft bus mass, kg. Illustrative placeholder.
    pub drag_mass_kg: f64,
    /// Starting circular-orbit altitude, km. Used by
    /// `SimpleOrbitalSimulator::with_config` (not by the parameterless
    /// `new()`/`OrbitalState::stowed()`, which hardcode 400km for
    /// backward compatibility with existing tests).
    pub initial_altitude_km: f64,
    /// Total reaction-wheel desaturation propellant available for the
    /// episode, N·m·s. Illustrative placeholder (2x `reaction_wheel_capacity`
    /// by default, i.e. enough to fully dump a saturated wheel roughly
    /// twice over) — not a specific mission's RCS propellant figure.
    pub desaturation_budget_nms: f64,
}
impl Default for OrbitalConfig {
    fn default() -> Self {
        Self {
            max_joint_torques: [20.0, 20.0, 15.0, 10.0, 5.0, 3.0, 2.0],
            spacecraft_inertia: [5000.0, 5000.0, 3000.0],
            reaction_wheel_capacity: 50.0,
            physics_hz: 100.0,
            delta_v_budget_m_s: 50.0,
            drag_cd: 2.2,
            drag_area_m2: 2.0,
            drag_mass_kg: 150.0,
            initial_altitude_km: 400.0,
            desaturation_budget_nms: 100.0,
            cognitive_interval: 5,
            steps_per_episode: 3000,
            network_layers: 3,
            neurons_per_layer: 8,
            learning_rate: 0.001,
        }
    }
}
impl OrbitalConfig {
    pub fn physics_dt(&self) -> f64 {
        1.0 / self.physics_hz
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_stowed() {
        let s = OrbitalState::stowed();
        assert!(s.is_finite());
        assert_eq!(s.to_channels().len(), NUM_STATE_CHANNELS);
    }

    #[test]
    fn test_command_zero_effort() {
        let c = OrbitalCommand::zero();
        assert_eq!(c.control_effort(), 0.0);
        assert_eq!(c.joint_torques, [0.0; NUM_ACTUATORS]);
    }

    #[test]
    fn test_command_effort_max() {
        // All torques saturated → effort = 1.0 (mean of absolute values / N_actuators).
        let c = OrbitalCommand {
            joint_torques: [1.0; NUM_ACTUATORS],
            translational_burn_mps: [0.0; 3],
            desaturation_torque_nm: [0.0; 3],
        };
        assert!((c.control_effort() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_command_effort_symmetric_in_sign() {
        // Negative torques contribute equally to effort (magnitude-based).
        let c = OrbitalCommand {
            joint_torques: [-0.5; NUM_ACTUATORS],
            translational_burn_mps: [0.0; 3],
            desaturation_torque_nm: [0.0; 3],
        };
        assert!((c.control_effort() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_config_physics_dt() {
        // Default physics_hz = 100 Hz → dt = 10 ms.
        let cfg = OrbitalConfig::default();
        assert!((cfg.physics_dt() - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_stowed_ee_position_above_origin() {
        // Stowed pose documentation claims the end-effector sits 2 m above
        // the spacecraft origin along z. Verify.
        let s = OrbitalState::stowed();
        assert_eq!(s.ee_position, [0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_stowed_exposure_and_comm_both_nominal() {
        // Stowed initial state should have full solar exposure and open
        // comm window — the "nominal" orbital starting condition.
        let s = OrbitalState::stowed();
        assert_eq!(s.solar_exposure, 1.0);
        assert_eq!(s.comm_window, 1.0);
    }

    #[test]
    fn test_finite_rejects_nan() {
        let mut s = OrbitalState::stowed();
        s.joint_angles[0] = f64::NAN;
        assert!(!s.is_finite());
    }

    #[test]
    fn test_stowed_altitude_matches_default_400km() {
        let s = OrbitalState::stowed();
        assert!(
            (s.altitude_km() - 400.0).abs() < 1e-6,
            "expected 400km altitude, got {}",
            s.altitude_km()
        );
    }

    #[test]
    fn test_circular_orbit_velocity_is_perpendicular_to_position() {
        // A circular orbit's velocity must be perpendicular to its radius
        // vector (dot product zero) — otherwise it isn't actually circular.
        let s = OrbitalState::circular_orbit(550.0);
        let dot: f64 = s
            .position_km
            .iter()
            .zip(s.velocity_km_s.iter())
            .map(|(p, v)| p * v)
            .sum();
        assert!(
            dot.abs() < 1e-9,
            "position.velocity dot = {dot}, expected ~0"
        );
    }

    #[test]
    fn test_higher_orbit_has_lower_circular_velocity() {
        // Kepler: v_circular = sqrt(mu/r) decreases with radius.
        let low = OrbitalState::circular_orbit(400.0);
        let high = OrbitalState::circular_orbit(35786.0); // GEO altitude
        let speed = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!(speed(high.velocity_km_s) < speed(low.velocity_km_s));
    }

    #[test]
    fn test_delta_v_remaining_clamps_at_zero() {
        let mut s = OrbitalState::stowed();
        s.delta_v_used_m_s = 1000.0;
        assert_eq!(s.delta_v_remaining_m_s(50.0), 0.0);
    }

    #[test]
    fn test_desaturation_remaining_clamps_at_zero() {
        let mut s = OrbitalState::stowed();
        s.desaturation_used_nms = 1000.0;
        assert_eq!(s.desaturation_remaining_nms(100.0), 0.0);
    }
}
