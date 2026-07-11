use crate::types::*;
use orbital_mechanics::atmosphere::{DragConfig, drag_acceleration};
use orbital_mechanics::coordinates::wgs84::{A as EARTH_RADIUS_KM, MU};
use orbital_mechanics::state::StateVector;
pub trait OrbitalPhysicsSimulator {
    fn step(&mut self, cmd: &OrbitalCommand, dt: f64);
    fn state(&self) -> &OrbitalState;
    fn reset(&mut self);
}

/// Fixed reference direction (Earth-to-Sun, and — as a deliberate
/// simplification — also the ground station's local vertical at t=0) in the
/// non-rotating inertial frame. Choosing these equal keeps the `stowed()`
/// initial condition self-consistent: the bus starts directly over its home
/// ground station, at local noon, in full sun (matching the hardcoded
/// `solar_exposure`/`comm_window` = 1.0 in `OrbitalState::stowed()`).
///
/// This is NOT a real ephemeris or ground-track model: the sun direction
/// doesn't move (no orbital precession of Earth around the sun) and Earth's
/// rotation isn't modeled (the "ground station" is fixed in inertial space,
/// not on the rotating surface). Both are reasonable within one training
/// episode (tens of simulated seconds, per `OrbitalConfig::steps_per_episode`
/// at the default 100Hz), which is far shorter than either timescale — but
/// would need real ephemeris + Earth rotation to be valid over hours/days.
const REFERENCE_DIRECTION: [f64; 3] = [1.0, 0.0, 0.0];
const MIN_ELEVATION_DEG: f64 = 5.0;

pub struct SimpleOrbitalSimulator {
    state: OrbitalState,
    config: OrbitalConfig,
    ji: [f64; NUM_JOINTS],
    jd: [f64; NUM_JOINTS],
    links: [f64; NUM_JOINTS],
    rwm: [f64; 3],
}
impl SimpleOrbitalSimulator {
    pub fn new() -> Self {
        Self::with_config(OrbitalConfig::default())
    }

    /// Construct with a custom config — e.g. a different starting altitude,
    /// drag properties, or delta-v budget for a scenario/benchmark. `new()`
    /// is just `with_config(OrbitalConfig::default())`.
    pub fn with_config(config: OrbitalConfig) -> Self {
        Self {
            state: OrbitalState::circular_orbit(config.initial_altitude_km),
            config,
            ji: [5.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.3],
            jd: [1.0, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1],
            links: [1.5, 1.5, 1.0, 0.8, 0.5, 0.3, 0.2],
            rwm: [0.0; 3],
        }
    }

    pub fn config(&self) -> &OrbitalConfig {
        &self.config
    }

    /// Current reaction-wheel stored momentum per axis, N·m·s. Magnitude
    /// approaching `config().reaction_wheel_capacity` means the wheel is
    /// near saturation — see `OrbitalCommand::desaturation_torque_nm`.
    pub fn reaction_wheel_momentum(&self) -> [f64; 3] {
        self.rwm
    }

    /// Two-body gravity + atmospheric drag acceleration at the current
    /// state, km/s². Real physics via the shared `orbital-mechanics` crate
    /// (no J2 or third-body perturbations yet — see crate README).
    fn orbital_acceleration(&self) -> [f64; 3] {
        let r_vec = self.state.position_km;
        let r_mag = (r_vec[0].powi(2) + r_vec[1].powi(2) + r_vec[2].powi(2)).sqrt();
        let r3 = r_mag.powi(3);
        let mut accel = [
            -MU * r_vec[0] / r3,
            -MU * r_vec[1] / r3,
            -MU * r_vec[2] / r3,
        ];
        let sv = StateVector::new(
            r_vec[0],
            r_vec[1],
            r_vec[2],
            self.state.velocity_km_s[0],
            self.state.velocity_km_s[1],
            self.state.velocity_km_s[2],
        );
        let drag_cfg = DragConfig::new(
            self.config.drag_cd,
            self.config.drag_area_m2,
            self.config.drag_mass_kg,
        );
        let drag = drag_acceleration(&sv, &drag_cfg);
        for i in 0..3 {
            accel[i] += drag[i];
        }
        accel
    }

    /// Cylindrical-shadow eclipse test against `REFERENCE_DIRECTION`: in
    /// full sun unless on the night side AND within Earth's radius of the
    /// sun-Earth line. No penumbra modeling (hard 0/1 transition).
    fn solar_exposure(&self) -> f64 {
        let r = self.state.position_km;
        let s = REFERENCE_DIRECTION;
        let proj = r[0] * s[0] + r[1] * s[1] + r[2] * s[2];
        if proj > 0.0 {
            return 1.0;
        }
        let perp2 = (0..3).map(|i| (r[i] - proj * s[i]).powi(2)).sum::<f64>();
        if perp2.sqrt() < EARTH_RADIUS_KM {
            0.0
        } else {
            1.0
        }
    }

    /// Horizon/elevation visibility against a single fixed ground point at
    /// `REFERENCE_DIRECTION` on Earth's surface (see doc comment above for
    /// the non-rotating-frame simplification this implies).
    fn comm_window(&self) -> f64 {
        let g = REFERENCE_DIRECTION;
        let ground_pos = [
            EARTH_RADIUS_KM * g[0],
            EARTH_RADIUS_KM * g[1],
            EARTH_RADIUS_KM * g[2],
        ];
        let los: [f64; 3] = std::array::from_fn(|i| self.state.position_km[i] - ground_pos[i]);
        let los_mag = (los[0].powi(2) + los[1].powi(2) + los[2].powi(2)).sqrt();
        if los_mag < 1e-9 {
            return 1.0;
        }
        let vertical_component = los[0] * g[0] + los[1] * g[1] + los[2] * g[2];
        let elevation_deg = (vertical_component / los_mag)
            .clamp(-1.0, 1.0)
            .asin()
            .to_degrees();
        if elevation_deg > MIN_ELEVATION_DEG {
            1.0
        } else {
            0.0
        }
    }
}
impl Default for SimpleOrbitalSimulator {
    fn default() -> Self {
        Self::new()
    }
}
impl OrbitalPhysicsSimulator for SimpleOrbitalSimulator {
    fn step(&mut self, cmd: &OrbitalCommand, dt: f64) {
        let mut react = [0.0f64; 3];
        for i in 0..NUM_JOINTS {
            let torque = cmd.joint_torques[i] as f64 * self.config.max_joint_torques[i];
            let ddq = (torque - self.jd[i] * self.state.joint_velocities[i]) / self.ji[i];
            self.state.joint_velocities[i] += ddq * dt;
            self.state.joint_angles[i] += self.state.joint_velocities[i] * dt;
            self.state.joint_angles[i] = self.state.joint_angles[i].clamp(-2.9, 2.9);
            if self.state.joint_angles[i].abs() >= 2.89 {
                self.state.joint_velocities[i] = 0.0;
            }
            react[i % 3] -= torque;
        }
        for a in 0..3 {
            let alpha = react[a] / self.config.spacecraft_inertia[a];
            self.state.spacecraft_angular_velocity[a] += alpha * dt;
            let wt_desired = (-50.0 * self.state.spacecraft_angular_velocity[a]).clamp(-5.0, 5.0);
            // Clamp the DELTA, not just the stored value: previously this
            // clamped only `self.rwm[a]` after adding wt_desired*dt, but
            // still applied the FULL wt_desired torque to the bus below
            // regardless of whether the wheel had room to absorb it. That
            // silently discarded the excess momentum instead of reflecting
            // a genuinely saturated wheel losing authority -- a saturated
            // wheel physically can't provide more torque in that direction,
            // so the disturbance should reach the bus unopposed. Computing
            // the actually-absorbed torque from the clamped delta fixes
            // both directions correctly: full authority while unsaturated,
            // reduced authority when pushing further into saturation, full
            // authority again when the command reduces |rwm| (desaturating).
            let new_rwm = (self.rwm[a] + wt_desired * dt).clamp(
                -self.config.reaction_wheel_capacity,
                self.config.reaction_wheel_capacity,
            );
            let wt_actual = (new_rwm - self.rwm[a]) / dt;
            self.rwm[a] = new_rwm;
            self.state.spacecraft_angular_velocity[a] +=
                wt_actual * dt / self.config.spacecraft_inertia[a];
        }

        // Reaction-wheel desaturation: RCS thrusters unload wheel momentum
        // toward zero. Only ever reduces |rwm| (can't be used to spin a
        // wheel up further -- see OrbitalCommand::desaturation_torque_nm),
        // and is clamped to whatever remains of the desaturation budget.
        let desat_remaining = self
            .state
            .desaturation_remaining_nms(self.config.desaturation_budget_nms);
        let mut desat_spent = 0.0f64;
        for a in 0..3 {
            // Caller supplies a magnitude (sign of desaturation_torque_nm is
            // ignored) -- the direction toward zero is inferred from rwm's
            // own sign, so firing thrusters to "dump N m/s" doesn't require
            // knowing which way the wheel is currently spinning.
            let requested_magnitude = (cmd.desaturation_torque_nm[a] as f64).abs() * dt;
            let max_reduction = self.rwm[a].abs(); // no overshoot past zero
            let budget_left = (desat_remaining - desat_spent).max(0.0);
            let applied_magnitude = requested_magnitude.min(max_reduction).min(budget_left);
            let applied = applied_magnitude * self.rwm[a].signum();
            self.rwm[a] -= applied;
            desat_spent += applied.abs();
        }
        self.state.desaturation_used_nms += desat_spent;
        // FK
        let (mut x, mut z, mut ca) = (0.0, 0.0, 0.0);
        for i in 0..NUM_JOINTS.min(4) {
            ca += self.state.joint_angles[i];
            x += self.links[i] * ca.sin();
            z += self.links[i] * ca.cos();
        }
        self.state.ee_position = [x, 0.3 * self.state.joint_angles[4].sin(), z];

        // Orbit: impulsive burn (budget-clamped) + two-body gravity + drag,
        // symplectic (semi-implicit) Euler — velocity updates from the
        // acceleration at the CURRENT position, then position updates from
        // the NEW velocity. This conserves orbital energy far better than
        // explicit Euler over many steps, which is what the arm dynamics
        // above use (fine there; short-timescale damped joint dynamics
        // don't accumulate the same secular energy drift an orbit would).
        let remaining_m_s = self
            .state
            .delta_v_remaining_m_s(self.config.delta_v_budget_m_s);
        let requested_mag_m_s = cmd
            .translational_burn_mps
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let burn_scale = if requested_mag_m_s > remaining_m_s && requested_mag_m_s > 1e-12 {
            remaining_m_s / requested_mag_m_s
        } else {
            1.0
        };
        let mut applied_mag_m_s = 0.0f64;
        for i in 0..3 {
            let applied_m_s = cmd.translational_burn_mps[i] as f64 * burn_scale;
            self.state.velocity_km_s[i] += applied_m_s / 1000.0;
            applied_mag_m_s += applied_m_s * applied_m_s;
        }
        self.state.delta_v_used_m_s += applied_mag_m_s.sqrt();

        let accel = self.orbital_acceleration();
        for i in 0..3 {
            self.state.velocity_km_s[i] += accel[i] * dt;
        }
        for i in 0..3 {
            self.state.position_km[i] += self.state.velocity_km_s[i] * dt;
        }

        self.state.solar_exposure = self.solar_exposure();
        self.state.comm_window = self.comm_window();
    }
    fn state(&self) -> &OrbitalState {
        &self.state
    }
    fn reset(&mut self) {
        self.state = OrbitalState::circular_orbit(self.config.initial_altitude_km);
        self.rwm = [0.0; 3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_stable() {
        let mut s = SimpleOrbitalSimulator::new();
        for _ in 0..1000 {
            s.step(&OrbitalCommand::zero(), 0.01);
        }
        assert!(s.state().is_finite());
    }
    #[test]
    fn test_reaction() {
        let mut s = SimpleOrbitalSimulator::new();
        let mut c = OrbitalCommand::zero();
        c.joint_torques[0] = 0.5;
        for _ in 0..100 {
            s.step(&c, 0.01);
        }
        assert!(
            s.state()
                .spacecraft_angular_velocity
                .iter()
                .map(|v| v.abs())
                .sum::<f64>()
                > 0.0
        );
    }

    #[test]
    fn test_saturated_wheel_loses_authority_and_bus_drifts() {
        // Regression test for a real bug (Phase 2, 2026-07-07): the wheel's
        // torque used to be applied to the bus in full regardless of
        // whether the wheel had room to absorb it -- silently discarding
        // excess momentum instead of reflecting that a saturated wheel
        // physically can't provide more torque. Sustained max disturbance
        // on one axis should saturate that axis's wheel, after which
        // continued disturbance should reach the bus (angular velocity
        // grows) instead of being perfectly cancelled forever.
        let mut s = SimpleOrbitalSimulator::new();
        let mut c = OrbitalCommand::zero();
        c.joint_torques[0] = 1.0; // sustained max torque -> sustained max reaction
        let dt = 0.01;
        for _ in 0..20_000 {
            s.step(&c, dt);
        }
        let rwm = s.reaction_wheel_momentum();
        assert!(
            rwm[0].abs() >= s.config().reaction_wheel_capacity - 1e-6,
            "expected axis-0 wheel to be saturated after sustained max disturbance, \
             got rwm={rwm:?}, capacity={}",
            s.config().reaction_wheel_capacity
        );
        let v_before = s.state().spacecraft_angular_velocity[0].abs();
        for _ in 0..2000 {
            s.step(&c, dt);
        }
        let v_after = s.state().spacecraft_angular_velocity[0].abs();
        assert!(
            v_after > v_before,
            "expected angular velocity to keep growing once the wheel is \
             saturated (v_before={v_before}, v_after={v_after}) -- if this \
             fails, the saturation-authority-loss fix regressed"
        );
    }

    #[test]
    fn test_desaturation_reduces_wheel_momentum() {
        let mut s = SimpleOrbitalSimulator::new();
        let mut c = OrbitalCommand::zero();
        c.joint_torques[0] = 1.0;
        for _ in 0..2000 {
            s.step(&c, 0.01);
        }
        let rwm_before = s.reaction_wheel_momentum()[0].abs();
        assert!(
            rwm_before > 0.0,
            "expected nonzero wheel momentum to desaturate"
        );

        let mut desat = OrbitalCommand::zero();
        desat.desaturation_torque_nm[0] = 100.0;
        s.step(&desat, 0.01);
        let rwm_after = s.reaction_wheel_momentum()[0].abs();
        assert!(
            rwm_after < rwm_before,
            "expected desaturation command to reduce |rwm|: before={rwm_before}, after={rwm_after}"
        );
    }

    #[test]
    fn test_desaturation_never_overshoots_past_zero() {
        let mut s = SimpleOrbitalSimulator::new();
        let mut c = OrbitalCommand::zero();
        c.joint_torques[0] = 1.0;
        for _ in 0..100 {
            s.step(&c, 0.01);
        }
        let mut desat = OrbitalCommand::zero();
        desat.desaturation_torque_nm[0] = 1e6; // absurdly large request
        s.step(&desat, 0.01);
        let rwm0 = s.reaction_wheel_momentum()[0];
        assert!(
            rwm0.abs() < 1e-6,
            "expected desaturation to land at ~0 without overshoot, got {rwm0}"
        );
    }

    #[test]
    fn test_desaturation_budget_of_zero_blocks_desaturation() {
        let mut orbital = OrbitalConfig::default();
        orbital.desaturation_budget_nms = 0.0;
        let mut s = SimpleOrbitalSimulator::with_config(orbital);
        let mut c = OrbitalCommand::zero();
        c.joint_torques[0] = 1.0;
        for _ in 0..1000 {
            s.step(&c, 0.01);
        }
        let mut desat = OrbitalCommand::zero();
        desat.desaturation_torque_nm[0] = 1000.0;
        s.step(&desat, 0.01);
        assert_eq!(
            s.state().desaturation_used_nms,
            0.0,
            "zero desaturation budget must block any desaturation spend"
        );
    }

    #[test]
    fn test_reset_returns_to_stowed() {
        // After arbitrary motion, reset() must restore the stowed state.
        let mut s = SimpleOrbitalSimulator::new();
        let stowed_angles = s.state().joint_angles;
        let mut c = OrbitalCommand::zero();
        c.joint_torques[0] = 0.5;
        for _ in 0..200 {
            s.step(&c, 0.01);
        }
        assert_ne!(s.state().joint_angles, stowed_angles, "motion should occur");
        s.reset();
        assert_eq!(s.state().joint_angles, stowed_angles);
        assert_eq!(s.state().joint_velocities, [0.0; NUM_JOINTS]);
        assert_eq!(s.state().spacecraft_angular_velocity, [0.0; 3]);
    }

    #[test]
    fn test_zero_command_no_joint_drift() {
        // Starting at stowed (all angles 0) with zero torque, joints must
        // stay at zero — no spontaneous motion from integration error.
        let mut s = SimpleOrbitalSimulator::new();
        for _ in 0..500 {
            s.step(&OrbitalCommand::zero(), 0.01);
        }
        for i in 0..NUM_JOINTS {
            assert!(
                s.state().joint_angles[i].abs() < 1e-9,
                "joint {} drifted under zero command",
                i
            );
            assert!(s.state().joint_velocities[i].abs() < 1e-9);
        }
    }

    #[test]
    fn test_joint_limits_clamp() {
        // Documented joint range is ±2.9 rad; sustained max torque must
        // saturate at that boundary without going over.
        let mut s = SimpleOrbitalSimulator::new();
        let mut c = OrbitalCommand::zero();
        c.joint_torques[0] = 1.0; // max positive
        for _ in 0..10_000 {
            s.step(&c, 0.01);
        }
        assert!(
            s.state().joint_angles[0] <= 2.9 + 1e-6,
            "joint 0 should saturate at +2.9 rad, got {}",
            s.state().joint_angles[0]
        );
    }

    #[test]
    fn test_deterministic_across_fresh_sims() {
        // Two independent fresh simulators driven by identical commands
        // must end in the same state — RL training precondition.
        let c = OrbitalCommand {
            joint_torques: [0.2, -0.1, 0.3, 0.0, 0.15, -0.05, 0.1],
            translational_burn_mps: [0.0; 3],
            desaturation_torque_nm: [0.0; 3],
        };
        let mut a = SimpleOrbitalSimulator::new();
        let mut b = SimpleOrbitalSimulator::new();
        for _ in 0..300 {
            a.step(&c, 0.01);
            b.step(&c, 0.01);
        }
        assert_eq!(a.state().joint_angles, b.state().joint_angles);
        assert_eq!(
            a.state().spacecraft_angular_velocity,
            b.state().spacecraft_angular_velocity
        );
    }

    #[test]
    fn test_full_orbit_sweeps_solar_exposure() {
        // Integrate over roughly one full ~90-minute LEO orbit (5400 s at
        // dt=1s) under real two-body propagation. The bus starts at the
        // sub-solar point (full sun) and, per Kepler, must pass through
        // Earth's shadow near the antipodal point of its orbit — solar
        // exposure must sweep through both extremes, not stay fixed.
        let mut s = SimpleOrbitalSimulator::new();
        let mut saw_low = false;
        let mut saw_high = false;
        for _ in 0..5400 {
            s.step(&OrbitalCommand::zero(), 1.0);
            if s.state().solar_exposure < 0.1 {
                saw_low = true;
            }
            if s.state().solar_exposure > 0.9 {
                saw_high = true;
            }
        }
        assert!(
            saw_low && saw_high,
            "over a full orbit solar_exposure should sweep through low and high values \
             (saw_low={saw_low}, saw_high={saw_high})"
        );
    }

    #[test]
    fn test_orbit_roughly_closes_after_one_period() {
        // A circular orbit propagated for its own Keplerian period should
        // return close to its starting position (symplectic Euler drifts
        // somewhat at 1s resolution, so allow generous tolerance — this is
        // a sanity check that it's actually orbiting, not flying off).
        let mut s = SimpleOrbitalSimulator::new();
        let r0 = s.state().position_km;
        let period_s = 2.0 * std::f64::consts::PI * ((EARTH_RADIUS_KM + 400.0).powi(3) / MU).sqrt();
        let steps = period_s.round() as usize;
        for _ in 0..steps {
            s.step(&OrbitalCommand::zero(), 1.0);
        }
        let r1 = s.state().position_km;
        let drift: f64 = (0..3).map(|i| (r1[i] - r0[i]).powi(2)).sum::<f64>().sqrt();
        assert!(
            drift < 0.05 * (EARTH_RADIUS_KM + 400.0),
            "position drift after one period was {drift} km, expected roughly closed orbit"
        );
    }

    #[test]
    fn test_delta_v_budget_clamps_applied_burn() {
        // Requesting far more delta-v than remains in the budget must not
        // apply more than the budget allows.
        let mut s = SimpleOrbitalSimulator::new();
        let mut c = OrbitalCommand::zero();
        c.translational_burn_mps = [10_000.0, 0.0, 0.0]; // way over any sane budget
        s.step(&c, 0.01);
        assert!(
            (s.state().delta_v_used_m_s - s.config.delta_v_budget_m_s).abs() < 1e-6,
            "expected delta-v used to clamp at budget {}, got {}",
            s.config.delta_v_budget_m_s,
            s.state().delta_v_used_m_s
        );
    }

    #[test]
    fn test_no_burn_no_delta_v_spent() {
        let mut s = SimpleOrbitalSimulator::new();
        for _ in 0..100 {
            s.step(&OrbitalCommand::zero(), 0.01);
        }
        assert_eq!(s.state().delta_v_used_m_s, 0.0);
    }

    mod proptest_physics {
        use super::*;
        use proptest::prelude::*;
        proptest! { #[test] fn finite(t0 in -1.0f32..1.0, t1 in -1.0f32..1.0, t2 in -1.0f32..1.0, t3 in -1.0f32..1.0, t4 in -1.0f32..1.0, t5 in -1.0f32..1.0, t6 in -1.0f32..1.0, dt in 0.001f64..0.05, steps in 1usize..300) {
            let mut s = SimpleOrbitalSimulator::new(); let c = OrbitalCommand { joint_torques: [t0,t1,t2,t3,t4,t5,t6], translational_burn_mps: [0.0; 3], desaturation_torque_nm: [0.0; 3] };
            for _ in 0..steps { s.step(&c, dt); } prop_assert!(s.state().is_finite());
        }}
    }
}
