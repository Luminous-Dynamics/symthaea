// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MuJoCo physics simulator implementation (behind `mujoco` feature).
//!
//! Wraps the mujoco-rs crate to provide high-fidelity Crazyflie 2 simulation
//! with full contact physics, aerodynamic effects, and sensor models.
//!
//! The MJCF model uses `gear=1` for all actuators, so `QuadrotorCommand::to_ctrl()`
//! maps directly to `mjData.ctrl` with ZERO conversion.

use std::sync::Arc;

use mujoco_rs::prelude::*;

use crate::simulator::PhysicsSimulator;
use crate::types::{FlightState, QuadrotorCommand};

/// Errors that can occur during MuJoCo simulator initialization.
#[derive(Debug)]
pub enum SimulatorError {
    /// Failed to load MJCF model from file path.
    ModelLoadFailed(String),
    /// Failed to parse MJCF XML string.
    ModelParseFailed(String),
}

/// MuJoCo-based physics simulator for the Crazyflie 2 quadrotor.
///
/// Loads an MJCF model and provides step-accurate simulation with full
/// contact physics, rotor dynamics, and aerodynamic effects.
///
/// `MjData` requires `M: Deref<Target = MjModel>`, so we use `Arc<MjModel>`
/// to share ownership between the struct and `MjData`.
pub struct MuJoCoSimulator {
    model: Arc<MjModel>,
    data: MjData<Arc<MjModel>>,
    cached_state: FlightState,
    external_force: [f64; 3],
    body_id: usize,
    sensory_filter: Option<crate::sensory_filter::SensoryFilter>,
}

impl MuJoCoSimulator {
    /// Create a new MuJoCo simulator from an MJCF XML path.
    pub fn new(model_path: &str) -> Result<Self, SimulatorError> {
        let model = Arc::new(
            MjModel::from_xml(model_path)
                .map_err(|e| SimulatorError::ModelLoadFailed(
                    format!("{}: {:?}", model_path, e),
                ))?,
        );
        let body_id = Self::find_body_id(&model, "cf2");
        let data = MjData::new(Arc::clone(&model));

        let mut sim = Self {
            model,
            data,
            cached_state: FlightState::hover(0.1),
            external_force: [0.0; 3],
            body_id,
            sensory_filter: None,
        };
        // Forward kinematics to populate xpos/xquat from qpos
        sim.data.forward();
        sim.extract_state();
        Ok(sim)
    }

    /// Create a MuJoCo simulator from an XML string.
    pub fn from_xml_string(xml: &str) -> Result<Self, SimulatorError> {
        let model = Arc::new(
            MjModel::from_xml_string(xml)
                .map_err(|e| SimulatorError::ModelParseFailed(format!("{:?}", e)))?,
        );
        let body_id = Self::find_body_id(&model, "cf2");
        let data = MjData::new(Arc::clone(&model));

        let mut sim = Self {
            model,
            data,
            cached_state: FlightState::hover(0.1),
            external_force: [0.0; 3],
            body_id,
            sensory_filter: None,
        };
        // Forward kinematics to populate xpos/xquat from qpos
        sim.data.forward();
        sim.extract_state();
        Ok(sim)
    }

    /// Create from the primitive Crazyflie 2 asset (zero mesh deps).
    pub fn from_primitive() -> Result<Self, SimulatorError> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/cf2_simple.xml");
        Self::new(path)
    }

    /// Create from the vendored Crazyflie 2 asset.
    /// Alias for `from_primitive()` — the primitive model is our default.
    pub fn from_vendored() -> Result<Self, SimulatorError> {
        Self::from_primitive()
    }

    /// Get the model timestep.
    pub fn model_timestep(&self) -> f64 {
        self.model.opt().timestep
    }

    /// Get the body ID for named body lookups.
    pub fn body_id(&self) -> usize {
        self.body_id
    }

    /// Get current body mass.
    pub fn body_mass(&self) -> f64 {
        self.model.body_mass()[self.body_id]
    }

    /// Dynamically change body mass (recomputes derived inertial quantities).
    pub fn set_body_mass(&mut self, new_mass: f64) {
        // SAFETY: mujoco-rs doesn't expose model parameter mutation. We cast to mutable
        // and call mj_setConst to re-derive model constants after mass change.
        unsafe {
            let model_ffi = self.model.ffi() as *const _ as *mut mujoco_rs::mujoco_c::mjModel;
            (*model_ffi).body_mass.add(self.body_id).write(new_mass);
            mujoco_rs::mujoco_c::mj_setConst(model_ffi, self.data.ffi_mut());
        }
    }

    /// Dynamically limit thrust actuator range.
    /// Modifies the upper bound of actuator 0 (body_thrust).
    pub fn set_thrust_limit(&mut self, max_thrust: f64) {
        // SAFETY: mujoco-rs doesn't expose actuator parameter mutation.
        // We cast to mutable to modify the control range directly.
        unsafe {
            let model_ffi = self.model.ffi() as *const _ as *mut mujoco_rs::mujoco_c::mjModel;
            (*model_ffi).actuator_ctrlrange.add(1).write(max_thrust);
        }
    }

    /// Get position of a named body.
    pub fn body_position(&self, name: &str) -> [f64; 3] {
        let bid = Self::find_body_id(self.data.model(), name);
        self.data.xpos()[bid]
    }

    /// Get linear velocity of a named body (from subtree linear velocity).
    pub fn body_velocity(&self, name: &str) -> [f64; 3] {
        let bid = Self::find_body_id(self.data.model(), name);
        let cvel = self.data.cvel()[bid];
        // cvel is [angular(3), linear(3)]
        [cvel[3], cvel[4], cvel[5]]
    }

    /// Enable sensory noise filter for sim-to-real transfer validation.
    pub fn enable_sensory_filter(&mut self, config: crate::sensory_filter::SensoryFilterConfig) {
        self.sensory_filter = Some(crate::sensory_filter::SensoryFilter::new(config, 42));
    }

    /// Disable sensory filter.
    pub fn disable_sensory_filter(&mut self) {
        self.sensory_filter = None;
    }

    /// Write a QuadrotorCommand into MuJoCo ctrl array.
    fn write_ctrl(&mut self, cmd: &QuadrotorCommand) {
        let ctrl_vals = cmd.to_ctrl();
        self.data.ctrl_mut()[..4].copy_from_slice(&ctrl_vals);
    }

    /// Apply accumulated external force to the drone body via xfrc_applied.
    fn apply_xfrc(&mut self) {
        if self.external_force[0].abs() > 1e-15
            || self.external_force[1].abs() > 1e-15
            || self.external_force[2].abs() > 1e-15
        {
            let xfrc = self.data.xfrc_applied_mut();
            // xfrc_applied is [nBody x 6]: [force(3), torque(3)]
            xfrc[self.body_id][0] = self.external_force[0];
            xfrc[self.body_id][1] = self.external_force[1];
            xfrc[self.body_id][2] = self.external_force[2];
        }
    }

    /// Clear external forces from xfrc_applied.
    fn clear_xfrc(&mut self) {
        self.data.xfrc_applied_mut()[self.body_id] = [0.0; 6];
        self.external_force = [0.0; 3];
    }

    /// Extract FlightState from MuJoCo qpos/qvel.
    fn extract_state(&mut self) {
        let t = self.data.time();

        // Free joint: qpos[0..7] = pos(3) + quat(4), qvel[0..6] = lin_vel(3) + ang_vel(3)
        let qpos = self.data.qpos();
        let qvel = self.data.qvel();
        let perfect_state = FlightState {
            position: [qpos[0], qpos[1], qpos[2]],
            quaternion: [qpos[3], qpos[4], qpos[5], qpos[6]],
            linear_velocity: [qvel[0], qvel[1], qvel[2]],
            angular_velocity: [qvel[3], qvel[4], qvel[5]],
            timestamp: t,
        };

        // Apply sensory filter if configured
        self.cached_state = match &mut self.sensory_filter {
            Some(filter) => filter.filter(&perfect_state),
            None => perfect_state,
        };
    }

    /// Find body ID by name. Panics if not found.
    fn find_body_id(model: &MjModel, name: &str) -> usize {
        let id = model.name_to_id(MjtObj::mjOBJ_BODY, name);
        assert!(id >= 0, "Body '{}' not found in MJCF model", name);
        id as usize
    }

    /// Get a reference to the shared model Arc (for viewer initialization).
    pub fn model_arc(&self) -> &Arc<MjModel> {
        &self.model
    }

    /// Get mutable access to the simulation data (for viewer sync).
    pub fn data_mut(&mut self) -> &mut MjData<Arc<MjModel>> {
        &mut self.data
    }

    /// Hold a body's velocity to zero (for freezing the beam before release).
    pub fn freeze_body(&mut self, body_name: &str) {
        let bid = Self::find_body_id(self.data.model(), body_name);
        let jnt_adr = self.model.body_jntadr()[bid];
        if jnt_adr >= 0 {
            let dof_adr = self.model.jnt_dofadr()[jnt_adr as usize] as usize;
            let qvel = self.data.qvel_mut();
            // Free joint has 6 DOF
            for i in 0..6 {
                qvel[dof_adr + i] = 0.0;
            }
        }
    }
}

impl PhysicsSimulator for MuJoCoSimulator {
    fn step(&mut self, cmd: &QuadrotorCommand, dt: f64) {
        self.write_ctrl(cmd);
        self.apply_xfrc();

        // Compute number of substeps to match requested dt
        let model_dt = self.model_timestep();
        let substeps = (dt / model_dt).round().max(1.0) as usize;

        for _ in 0..substeps {
            self.data.step();
        }

        self.extract_state();
        self.clear_xfrc();
    }

    fn state(&self) -> &FlightState {
        &self.cached_state
    }

    fn reset(&mut self, altitude: f64) {
        // Reset MuJoCo data to initial state
        self.data.reset();

        // Set initial position via safe qpos/qvel accessors
        let qpos = self.data.qpos_mut();
        qpos[0] = 0.0;
        qpos[1] = 0.0;
        qpos[2] = altitude;
        qpos[3] = 1.0; // w
        qpos[4] = 0.0; // x
        qpos[5] = 0.0; // y
        qpos[6] = 0.0; // z
        let qvel = self.data.qvel_mut();
        for i in 0..6 {
            qvel[i] = 0.0;
        }

        // Forward kinematics to update derived quantities
        self.data.forward();

        self.external_force = [0.0; 3];

        if let Some(ref mut filter) = self.sensory_filter {
            filter.reset();
        }

        self.extract_state();
    }

    fn reset_with_perturbation(&mut self, altitude: f64, perturbation: f64, seed: u64) {
        self.reset(altitude);

        // Apply perturbation to initial state
        let mut rng = seed;
        let mut next_f64 = || -> f64 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64 / u64::MAX as f64) * 2.0 - 1.0
        };

        let qpos = self.data.qpos_mut();
        qpos[0] += perturbation * next_f64() * 0.1;
        qpos[1] += perturbation * next_f64() * 0.1;
        qpos[2] += perturbation * next_f64() * 0.05;

        // Small quaternion perturbation
        let tilt = perturbation * next_f64() * 0.1;
        let q = [1.0, tilt, tilt * 0.5, 0.0];
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        qpos[3] = q[0] / norm;
        qpos[4] = q[1] / norm;
        qpos[5] = q[2] / norm;
        qpos[6] = q[3] / norm;

        self.data.forward();
        self.extract_state();
    }

    fn apply_external_force(&mut self, force: [f64; 3]) {
        self.external_force[0] += force[0];
        self.external_force[1] += force[1];
        self.external_force[2] += force[2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires MuJoCo library
    fn test_mujoco_primitive_loads() {
        let sim = MuJoCoSimulator::from_primitive().unwrap();
        assert!(sim.state().altitude() > 0.0);
        assert!(sim.body_mass() > 0.0);
    }

    #[test]
    #[ignore] // Requires MuJoCo library
    fn test_mujoco_hover_maintains_altitude() {
        let mut sim = MuJoCoSimulator::from_primitive().unwrap();
        let cmd = QuadrotorCommand::hover();
        for _ in 0..500 {
            sim.step(&cmd, 0.002);
        }
        assert!(
            (sim.state().altitude() - 0.1).abs() < 0.1,
            "MuJoCo hover should maintain approximate altitude: got {}",
            sim.state().altitude()
        );
    }

    #[test]
    #[ignore] // Requires MuJoCo library
    fn test_mujoco_external_force() {
        let mut sim = MuJoCoSimulator::from_primitive().unwrap();
        sim.apply_external_force([0.1, 0.0, 0.0]);
        let cmd = QuadrotorCommand::hover();
        sim.step(&cmd, 0.002);
        // Force should have some effect
        assert!(sim.state().position[0].abs() > 0.0 || sim.state().linear_velocity[0].abs() > 0.0);
    }

    #[test]
    #[ignore] // Requires MuJoCo library
    fn test_mujoco_reset() {
        let mut sim = MuJoCoSimulator::from_primitive().unwrap();
        let cmd = QuadrotorCommand::zero();
        for _ in 0..100 {
            sim.step(&cmd, 0.002);
        }
        sim.reset(0.5);
        assert!((sim.state().altitude() - 0.5).abs() < 0.01);
    }

    #[test]
    #[ignore] // Requires MuJoCo library
    fn test_mujoco_mass_change() {
        let mut sim = MuJoCoSimulator::from_primitive().unwrap();
        let original_mass = sim.body_mass();
        sim.set_body_mass(original_mass * 1.5);
        assert!((sim.body_mass() - original_mass * 1.5).abs() < 1e-6);
    }

    #[test]
    #[ignore] // Requires MuJoCo library
    fn test_mujoco_body_position() {
        let sim = MuJoCoSimulator::from_primitive().unwrap();
        let pos = sim.body_position("cf2");
        assert!(pos[2] > 0.0); // Should be above ground
    }

    #[test]
    #[ignore] // Requires MuJoCo library
    fn test_sacrifice_scene_loads() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/cf2_sacrifice.xml");
        let sim = MuJoCoSimulator::new(path).unwrap();
        // Should have all bodies
        let drone_pos = sim.body_position("cf2");
        let human_pos = sim.body_position("human");
        let beam_pos = sim.body_position("beam");
        assert!(drone_pos[2] > 1.0); // Drone at 1.5m
        assert!(human_pos[0] > 1.0); // Human at x=2.0
        assert!(beam_pos[2] > 3.0); // Beam at 4.0m
    }
}
