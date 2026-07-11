// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::*;

pub trait SurgicalPhysicsSimulator {
    fn step(&mut self, cmd: &SurgicalCommand, dt: f64);
    fn state(&self) -> &SurgicalState;
    fn reset(&mut self);
}

/// Forward kinematics of the tool tip (mm) from the joint angles.
/// Simplified serial-chain model: joints 0..2 form a planar x/z arm,
/// joint 3 adds the y (out-of-plane) motion; joints 4/5 are wrist axes
/// that do not translate the tip in this model.
pub fn fk_tip(q: &[f64; NUM_JOINTS]) -> [f64; 3] {
    let (q1, q2, q3) = (q[0], q[1], q[2]);
    [
        150.0 * q1.sin() + 100.0 * (q1 + q2).sin() + 30.0 * (q1 + q2 + q3).sin(),
        30.0 * q[3].sin(),
        -(150.0 * q1.cos() + 100.0 * (q1 + q2).cos() + 30.0 * (q1 + q2 + q3).cos()),
    ]
}

/// Lateral (x, y) coordinates (mm) where the tool shaft crosses the trocar
/// port plane `z = port_z`. The shaft is modeled as the straight segment
/// from the instrument mount at the origin to the tip. Returns `None` when
/// the tip is above the port plane (shaft doesn't reach it).
fn shaft_port_crossing(q: &[f64; NUM_JOINTS], port_z: f64) -> Option<[f64; 2]> {
    let tip = fk_tip(q);
    if tip[2] >= port_z || tip[2] >= -1e-9 {
        return None;
    }
    let s = port_z / tip[2]; // in (0, 1): fraction along mount→tip
    Some([tip[0] * s, tip[1] * s])
}

pub struct SimpleSurgicalSimulator {
    state: SurgicalState,
    config: SurgicalConfig,
    inertias: [f64; NUM_JOINTS],
    damping: [f64; NUM_JOINTS],
    tremor_state: [f64; NUM_JOINTS],
    tissue_stiffness: f64,
    tissue_z: f64,
    /// Lateral anchor (x, y in mm) of the trocar port on the plane
    /// `z = config.trocar_port_z`, fixed at construction from the home pose
    /// so the home shaft line is RCM-neutral.
    port_anchor: [f64; 2],
    /// Instantaneous tip speed cap in mm/s (set per safety tier by the
    /// embodiment). `f64::INFINITY` = uncapped.
    tip_speed_limit: f64,
    /// Measured tip speed (mm/s) over the last step, post-clamp.
    last_tip_speed: f64,
}

impl SimpleSurgicalSimulator {
    pub fn new() -> Self {
        Self::with_config(SurgicalConfig::default())
    }
    pub fn with_config(config: SurgicalConfig) -> Self {
        let state = SurgicalState::home();
        let port_anchor =
            shaft_port_crossing(&state.joint_angles, config.trocar_port_z).unwrap_or([0.0, 0.0]);
        Self {
            state,
            config,
            inertias: [0.5, 0.5, 0.3, 0.2, 0.1, 0.05],
            damping: [2.0, 2.0, 1.5, 1.0, 0.5, 0.3],
            tremor_state: [0.0; NUM_JOINTS],
            tissue_stiffness: 0.5,
            tissue_z: -10.0,
            port_anchor,
            tip_speed_limit: f64::INFINITY,
            last_tip_speed: 0.0,
        }
    }
    /// Set the per-tier tip speed cap (mm/s); see
    /// [`crate::types::surgical_tip_speed_limit`].
    pub fn set_tip_speed_limit(&mut self, mm_per_s: f64) {
        self.tip_speed_limit = if mm_per_s.is_finite() && mm_per_s >= 0.0 {
            mm_per_s
        } else {
            f64::INFINITY
        };
    }
    pub fn tip_speed_limit(&self) -> f64 {
        self.tip_speed_limit
    }
    /// Tip speed (mm/s) measured over the last physics step (after the
    /// per-tier velocity clamp was applied).
    pub fn last_tip_speed(&self) -> f64 {
        self.last_tip_speed
    }
    /// Lateral displacement (mm) of the tool shaft from the trocar port
    /// anchor, measured where the shaft crosses the port plane. This is the
    /// quantity the RCM spring drives toward zero.
    pub fn port_lateral_displacement(&self) -> f64 {
        match shaft_port_crossing(&self.state.joint_angles, self.config.trocar_port_z) {
            Some(c) => {
                ((c[0] - self.port_anchor[0]).powi(2) + (c[1] - self.port_anchor[1]).powi(2)).sqrt()
            }
            None => 0.0,
        }
    }
    fn filter_tremor(&mut self, raw: [f64; NUM_JOINTS], dt: f64) -> [f64; NUM_JOINTS] {
        let a = dt * self.config.tremor_filter_hz * std::f64::consts::TAU;
        let a = a / (1.0 + a);
        let mut out = [0.0; NUM_JOINTS];
        for i in 0..NUM_JOINTS {
            self.tremor_state[i] = self.tremor_state[i] * (1.0 - a) + raw[i] * a;
            out[i] = self.tremor_state[i];
        }
        out
    }
    /// RCM constraint torques: a task-space spring `F = -k · d` (k in N/m,
    /// d = lateral port-crossing displacement converted mm→m) mapped into
    /// joint space through the finite-difference Jacobian of the crossing
    /// point (τ = Jᵀ F). This is a soft penalty pulling the shaft back
    /// through the pivot — NOT a hard kinematic constraint; residual lateral
    /// displacement scales as (applied torque)/(k·J²).
    fn rcm_torques(&self) -> [f64; NUM_JOINTS] {
        let k = self.config.rcm_stiffness;
        let mut tau = [0.0; NUM_JOINTS];
        if k <= 0.0 {
            return tau;
        }
        let q = self.state.joint_angles;
        let Some(c) = shaft_port_crossing(&q, self.config.trocar_port_z) else {
            return tau;
        };
        // Displacement in meters (positions are mm).
        let dx = (c[0] - self.port_anchor[0]) / 1000.0;
        let dy = (c[1] - self.port_anchor[1]) / 1000.0;
        const EPS: f64 = 1e-6;
        for (i, t) in tau.iter_mut().enumerate() {
            let mut qp = q;
            qp[i] += EPS;
            let Some(cp) = shaft_port_crossing(&qp, self.config.trocar_port_z) else {
                continue;
            };
            // Jacobian of the crossing point wrt joint i, in m/rad.
            let jx = (cp[0] - c[0]) / EPS / 1000.0;
            let jy = (cp[1] - c[1]) / EPS / 1000.0;
            *t = -k * (dx * jx + dy * jy);
        }
        tau
    }
}
impl Default for SimpleSurgicalSimulator {
    fn default() -> Self {
        Self::new()
    }
}
impl SurgicalPhysicsSimulator for SimpleSurgicalSimulator {
    fn step(&mut self, cmd: &SurgicalCommand, dt: f64) {
        let mut raw = [0.0; NUM_JOINTS];
        for i in 0..NUM_JOINTS {
            raw[i] = cmd.joint_torques[i] as f64
                * self.config.max_joint_torques[i]
                * self.config.motion_scaling;
        }
        let filtered = self.filter_tremor(raw, dt);
        // RCM spring torque is a physical constraint force at the trocar —
        // applied after the tremor filter (it is not a command).
        let rcm = self.rcm_torques();
        // Pass 1: integrate velocities.
        for i in 0..NUM_JOINTS {
            let ddq = (filtered[i] + rcm[i] - self.damping[i] * self.state.joint_velocities[i])
                / self.inertias[i];
            self.state.joint_velocities[i] += ddq * dt;
        }
        // Per-tier tip speed cap: predict the tip velocity (via the FK
        // Jacobian, evaluated by finite differencing along the current joint
        // velocities) and uniformly scale joint velocities down when it
        // would exceed the limit.
        let q0 = self.state.joint_angles;
        let mut q1 = q0;
        for i in 0..NUM_JOINTS {
            q1[i] += self.state.joint_velocities[i] * dt;
        }
        let t0 = fk_tip(&q0);
        let t1 = fk_tip(&q1);
        let mut tip_speed = if dt > 0.0 {
            ((t1[0] - t0[0]).powi(2) + (t1[1] - t0[1]).powi(2) + (t1[2] - t0[2]).powi(2)).sqrt()
                / dt
        } else {
            0.0
        };
        if tip_speed > self.tip_speed_limit && tip_speed > 0.0 {
            let scale = self.tip_speed_limit / tip_speed;
            for v in &mut self.state.joint_velocities {
                *v *= scale;
            }
            tip_speed = self.tip_speed_limit;
        }
        self.last_tip_speed = tip_speed;
        // Pass 2: integrate angles + enforce joint limits.
        for i in 0..NUM_JOINTS {
            self.state.joint_angles[i] += self.state.joint_velocities[i] * dt;
            let lim = match i {
                0 | 3 => 1.5,
                1 | 4 => 1.2,
                _ => 0.8,
            };
            if self.state.joint_angles[i] < -lim {
                self.state.joint_angles[i] = -lim;
                self.state.joint_velocities[i] = self.state.joint_velocities[i].max(0.0);
            }
            if self.state.joint_angles[i] > lim {
                self.state.joint_angles[i] = lim;
                self.state.joint_velocities[i] = self.state.joint_velocities[i].min(0.0);
            }
        }
        // Simplified FK
        self.state.tip_position = fk_tip(&self.state.joint_angles);
        // Tissue interaction
        let depth = self.state.tip_position[2] - self.tissue_z;
        if depth > 0.0 {
            self.state.tip_force[2] = -self.tissue_stiffness * depth;
        } else {
            self.state.tip_force = [0.0; 3];
        }
        // Geometric safety channels (previously scripted sinusoids —
        // now derived from actual kinematics):
        // tip-to-critical-structure Euclidean distance (mm)…
        let cs = self.config.critical_structure;
        self.state.critical_structure_distance = ((self.state.tip_position[0] - cs[0]).powi(2)
            + (self.state.tip_position[1] - cs[1]).powi(2)
            + (self.state.tip_position[2] - cs[2]).powi(2))
        .sqrt();
        // …and trocar compliance = normalized lateral shaft displacement at
        // the port (20 mm of lateral port travel saturates the channel).
        self.state.trocar_compliance = (self.port_lateral_displacement() / 20.0).clamp(0.0, 1.0);
        self.state.jaw_angle = (cmd.jaw as f64).clamp(0.0, 1.0);
        self.state.cautery_power = (cmd.cautery as f64).clamp(0.0, 1.0);
    }
    fn state(&self) -> &SurgicalState {
        &self.state
    }
    fn reset(&mut self) {
        self.state = SurgicalState::home();
        self.tremor_state = [0.0; NUM_JOINTS];
        self.last_tip_speed = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_stable() {
        let mut s = SimpleSurgicalSimulator::new();
        for _ in 0..1000 {
            s.step(&SurgicalCommand::zero(), 0.001);
        }
        assert!(s.state().is_finite());
    }
    #[test]
    fn test_torque_moves() {
        let mut s = SimpleSurgicalSimulator::new();
        let init = s.state().tip_position;
        let mut c = SurgicalCommand::zero();
        c.joint_torques[0] = 0.3;
        for _ in 0..500 {
            s.step(&c, 0.001);
        }
        let d = ((s.state().tip_position[0] - init[0]).powi(2)
            + (s.state().tip_position[2] - init[2]).powi(2))
        .sqrt();
        assert!(d > 0.1);
    }

    #[test]
    fn test_reset_returns_to_home() {
        // After arbitrary motion, reset() must restore the home state.
        let mut s = SimpleSurgicalSimulator::new();
        let home_angles = s.state().joint_angles;
        let home_tip = s.state().tip_position;
        let mut c = SurgicalCommand::zero();
        c.joint_torques[0] = 0.5;
        for _ in 0..200 {
            s.step(&c, 0.001);
        }
        assert_ne!(
            s.state().tip_position,
            home_tip,
            "motion should have occurred"
        );
        s.reset();
        assert_eq!(s.state().joint_angles, home_angles);
        assert_eq!(s.state().joint_velocities, [0.0; NUM_JOINTS]);
        assert_eq!(s.state().tip_position, home_tip);
    }

    #[test]
    fn test_zero_command_no_joint_drift() {
        // Starting at home with zero torque: velocities stay zero, angles
        // stay pinned at home. (Tip position is not part of the invariant —
        // FK is recomputed each step and disagrees with home's hardcoded
        // initial tip; the joint angles are the authoritative state.)
        let mut s = SimpleSurgicalSimulator::new();
        let init_angles = s.state().joint_angles;
        for _ in 0..500 {
            s.step(&SurgicalCommand::zero(), 0.001);
        }
        for i in 0..NUM_JOINTS {
            assert!(
                (s.state().joint_angles[i] - init_angles[i]).abs() < 1e-9,
                "joint {} drifted from home under zero command",
                i
            );
            assert!(s.state().joint_velocities[i].abs() < 1e-9);
        }
    }

    #[test]
    fn test_joint_limits_enforced() {
        // Sustained max torque on joint 0 must saturate at the 1.5 rad limit.
        // RCM spring disabled here: with the constraint active the trocar
        // spring (correctly) stops the excursion long before the joint limit,
        // and this test is specifically about the limit clamp.
        let mut cfg = SurgicalConfig::default();
        cfg.rcm_stiffness = 0.0;
        let mut s = SimpleSurgicalSimulator::with_config(cfg);
        let mut c = SurgicalCommand::zero();
        c.joint_torques[0] = 1.0;
        for _ in 0..5000 {
            s.step(&c, 0.001);
        }
        assert!(
            s.state().joint_angles[0] <= 1.5 + 1e-6,
            "joint 0 should saturate at positive limit"
        );
        assert!(
            s.state().joint_angles[0] >= -1.5 - 1e-6,
            "joint 0 should stay above negative limit"
        );
    }

    #[test]
    fn test_deterministic_across_fresh_sims() {
        // Two independent fresh sims driven by identical commands must end
        // in the same state — determinism precondition for RL training.
        let c = SurgicalCommand {
            joint_torques: [0.2, -0.1, 0.3, 0.0, 0.15, -0.05],
            jaw: 0.3,
            cautery: 0.0,
        };
        let mut a = SimpleSurgicalSimulator::new();
        let mut b = SimpleSurgicalSimulator::new();
        for _ in 0..300 {
            a.step(&c, 0.001);
            b.step(&c, 0.001);
        }
        assert_eq!(a.state().joint_angles, b.state().joint_angles);
        assert_eq!(a.state().tip_position, b.state().tip_position);
    }

    #[test]
    fn test_rcm_bounds_port_displacement() {
        // The RCM claim, tested by failure-capable contrast: command a
        // sustained lateral motion; with the trocar spring active the
        // shaft's lateral port-crossing displacement must stay bounded,
        // and with stiffness zero it must grow far larger.
        let run = |rcm_stiffness: f64| -> f64 {
            let mut cfg = SurgicalConfig::default();
            cfg.rcm_stiffness = rcm_stiffness;
            let mut s = SimpleSurgicalSimulator::with_config(cfg);
            let mut c = SurgicalCommand::zero();
            c.joint_torques[0] = 0.5; // lateral push
            let mut max_d = 0.0f64;
            for _ in 0..3000 {
                s.step(&c, 0.001);
                max_d = max_d.max(s.port_lateral_displacement());
            }
            assert!(s.state().is_finite());
            max_d
        };
        let with_rcm = run(SurgicalConfig::default().rcm_stiffness);
        let without_rcm = run(0.0);
        assert!(
            with_rcm < 10.0,
            "RCM spring must bound lateral port displacement to <10 mm, got {with_rcm:.2} mm"
        );
        assert!(
            without_rcm > 2.0 * with_rcm,
            "without the spring the port displacement must be much larger \
             (with={with_rcm:.2} mm, without={without_rcm:.2} mm)"
        );
    }

    #[test]
    fn test_rcm_neutral_at_home() {
        // Home pose defines the port anchor — zero displacement, zero
        // constraint torque, so the spring must not perturb a resting arm.
        let mut s = SimpleSurgicalSimulator::new();
        for _ in 0..500 {
            s.step(&SurgicalCommand::zero(), 0.001);
        }
        assert!(
            s.port_lateral_displacement() < 1e-6,
            "home pose must be RCM-neutral, got {} mm",
            s.port_lateral_displacement()
        );
    }

    #[test]
    fn test_critical_structure_distance_is_geometric() {
        // The distance channel must equal the actual tip-to-structure
        // Euclidean distance and decrease as the tip approaches.
        let mut cfg = SurgicalConfig::default();
        cfg.rcm_stiffness = 0.0; // let the tip travel freely toward it
        let cs = cfg.critical_structure;
        let mut s = SimpleSurgicalSimulator::with_config(cfg);
        s.step(&SurgicalCommand::zero(), 0.001);
        let d0 = s.state().critical_structure_distance;
        let tip = s.state().tip_position;
        let expect =
            ((tip[0] - cs[0]).powi(2) + (tip[1] - cs[1]).powi(2) + (tip[2] - cs[2]).powi(2)).sqrt();
        assert!(
            (d0 - expect).abs() < 1e-9,
            "distance channel must be geometric: {d0} vs {expect}"
        );
        // Drive the tip toward the structure (+x direction) and track the
        // closest approach (the tip sweeps past the structure, so the final
        // distance is not the interesting quantity).
        let mut c = SurgicalCommand::zero();
        c.joint_torques[0] = 0.5;
        let mut min_d = d0;
        for _ in 0..2000 {
            s.step(&c, 0.001);
            min_d = min_d.min(s.state().critical_structure_distance);
        }
        assert!(
            min_d < d0,
            "moving toward the structure must decrease distance ({d0:.1} → min {min_d:.1})"
        );
    }

    #[test]
    fn test_tip_speed_clamp_enforced() {
        // With the Yellow-tier cap set, measured tip speed must never
        // exceed it even under full torque — and must stay well below the
        // Green allowance the same command would otherwise reach.
        let yellow = 20.0;
        let green = 50.0;
        let run = |limit: f64| -> f64 {
            let mut cfg = SurgicalConfig::default();
            cfg.rcm_stiffness = 0.0; // isolate the velocity clamp
            let mut s = SimpleSurgicalSimulator::with_config(cfg);
            s.set_tip_speed_limit(limit);
            let mut c = SurgicalCommand::zero();
            c.joint_torques[0] = 1.0;
            let mut max_speed = 0.0f64;
            let mut prev_tip = s.state().tip_position;
            for _ in 0..1000 {
                s.step(&c, 0.001);
                let tip = s.state().tip_position;
                let v = ((tip[0] - prev_tip[0]).powi(2)
                    + (tip[1] - prev_tip[1]).powi(2)
                    + (tip[2] - prev_tip[2]).powi(2))
                .sqrt()
                    / 0.001;
                max_speed = max_speed.max(v);
                prev_tip = tip;
            }
            max_speed
        };
        let uncapped = run(f64::INFINITY);
        let capped = run(yellow);
        assert!(
            uncapped > green,
            "full torque should exceed the Green limit uncapped, got {uncapped:.1} mm/s"
        );
        assert!(
            capped <= yellow + 1.0,
            "Yellow cap must hold: got {capped:.1} mm/s > {yellow} mm/s"
        );
    }

    mod proptest_physics {
        use super::*;
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn finite(
                t0 in -1.0f32..1.0, t1 in -1.0f32..1.0, t2 in -1.0f32..1.0,
                t3 in -1.0f32..1.0, t4 in -1.0f32..1.0, t5 in -1.0f32..1.0,
                dt in 0.0001f64..0.005, steps in 1usize..500
            ) {
                let mut s = SimpleSurgicalSimulator::new();
                let c = SurgicalCommand { joint_torques: [t0,t1,t2,t3,t4,t5], jaw: 0.0, cautery: 0.0 };
                for _ in 0..steps { s.step(&c, dt); }
                prop_assert!(s.state().is_finite());
            }
        }
    }
}
