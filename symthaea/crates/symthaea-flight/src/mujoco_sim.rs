//! MuJoCo physics simulator implementation (behind `mujoco` feature).
//!
//! Wraps the mujoco-rs crate to provide high-fidelity Crazyflie 2 simulation
//! with full contact physics, aerodynamic effects, and sensor models.
//!
//! The MJCF model uses `gear=1` for all actuators, so `QuadrotorCommand::to_ctrl()`
//! maps directly to `mjData.ctrl` with ZERO conversion.

use mujoco_rs::prelude::*;

use crate::simulator::PhysicsSimulator;
use crate::types::{FlightState, QuadrotorCommand};

/// MuJoCo-based physics simulator for the Crazyflie 2 quadrotor.
///
/// Loads an MJCF model and provides step-accurate simulation with full
/// contact physics, rotor dynamics, and aerodynamic effects.
pub struct MuJoCoSimulator {
    model: MjModel,
    data: MjData,
    cached_state: FlightState,
    external_force: [f64; 3],
    body_id: usize,
    #[cfg(feature = "mujoco")]
    sensory_filter: Option<crate::sensory_filter::SensoryFilter>,
}

impl MuJoCoSimulator {
    /// Create a new MuJoCo simulator from an MJCF XML path.
    pub fn new(model_path: &str) -> Self {
        let model = MjModel::from_xml(model_path)
            .unwrap_or_else(|e| panic!("Failed to load MJCF model '{}': {:?}", model_path, e));
        let data = MjData::new(&model);

        // Find the cf2 body index
        let body_id = Self::find_body_id(&model, "cf2");

        let mut sim = Self {
            model,
            data,
            cached_state: FlightState::hover(0.1),
            external_force: [0.0; 3],
            body_id,
            #[cfg(feature = "mujoco")]
            sensory_filter: None,
        };
        sim.extract_state();
        sim
    }

    /// Create a MuJoCo simulator from an XML string.
    pub fn from_xml_string(xml: &str) -> Self {
        let model = MjModel::from_xml_string(xml)
            .unwrap_or_else(|e| panic!("Failed to parse MJCF XML: {:?}", e));
        let data = MjData::new(&model);
        let body_id = Self::find_body_id(&model, "cf2");

        let mut sim = Self {
            model,
            data,
            cached_state: FlightState::hover(0.1),
            external_force: [0.0; 3],
            body_id,
            #[cfg(feature = "mujoco")]
            sensory_filter: None,
        };
        sim.extract_state();
        sim
    }

    /// Create from the primitive Crazyflie 2 asset (zero mesh deps).
    pub fn from_primitive() -> Self {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/cf2_simple.xml");
        Self::new(path)
    }

    /// Create from the vendored Crazyflie 2 asset.
    /// Alias for `from_primitive()` — the primitive model is our default.
    pub fn from_vendored() -> Self {
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
        // Access via FFI: model.body_mass[body_id]
        let ffi = self.model.ffi();
        unsafe { *ffi.body_mass.add(self.body_id) }
    }

    /// Dynamically change body mass (recomputes derived inertial quantities).
    pub fn set_body_mass(&mut self, new_mass: f64) {
        let ffi = self.model.ffi_mut();
        unsafe {
            *ffi.body_mass.add(self.body_id) = new_mass;
        }
        // Recompute derived quantities (inertia, etc.)
        unsafe {
            mujoco_rs::mujoco_c::mj_setConst(
                self.model.ffi_mut() as *mut _,
                self.data.ffi_mut() as *mut _,
            );
        }
    }

    /// Dynamically limit thrust actuator range.
    /// Modifies the upper bound of actuator 0 (body_thrust).
    pub fn set_thrust_limit(&mut self, max_thrust: f64) {
        let ffi = self.model.ffi_mut();
        // actuator_ctrlrange is [nActuator x 2] — [lower, upper] for each
        unsafe {
            // Actuator 0 = body_thrust, index 0*2+1 = upper bound
            *ffi.actuator_ctrlrange.add(1) = max_thrust;
        }
    }

    /// Get position of a named body.
    pub fn body_position(&self, name: &str) -> [f64; 3] {
        let bid = Self::find_body_id(&self.model, name);
        let ffi = self.data.ffi();
        unsafe {
            let ptr = ffi.xpos.add(bid * 3);
            [*ptr, *ptr.add(1), *ptr.add(2)]
        }
    }

    /// Get linear velocity of a named body (from subtree linear velocity).
    pub fn body_velocity(&self, name: &str) -> [f64; 3] {
        let bid = Self::find_body_id(&self.model, name);
        let ffi = self.data.ffi();
        unsafe {
            let ptr = ffi.cvel.add(bid * 6 + 3); // cvel is [nBody x 6]: [angular(3), linear(3)]
            [*ptr, *ptr.add(1), *ptr.add(2)]
        }
    }

    /// Enable sensory noise filter for sim-to-real transfer validation.
    #[cfg(feature = "mujoco")]
    pub fn enable_sensory_filter(&mut self, config: crate::sensory_filter::SensoryFilterConfig) {
        self.sensory_filter = Some(crate::sensory_filter::SensoryFilter::new(config, 42));
    }

    /// Disable sensory filter.
    #[cfg(feature = "mujoco")]
    pub fn disable_sensory_filter(&mut self) {
        self.sensory_filter = None;
    }

    /// Write a QuadrotorCommand into MuJoCo ctrl array.
    fn write_ctrl(&mut self, cmd: &QuadrotorCommand) {
        let ctrl = cmd.to_ctrl();
        let ffi = self.data.ffi_mut();
        for i in 0..4 {
            unsafe {
                *ffi.ctrl.add(i) = ctrl[i];
            }
        }
    }

    /// Apply accumulated external force to the drone body via xfrc_applied.
    fn apply_xfrc(&mut self) {
        if self.external_force[0].abs() > 1e-15
            || self.external_force[1].abs() > 1e-15
            || self.external_force[2].abs() > 1e-15
        {
            let ffi = self.data.ffi_mut();
            let base = self.body_id * 6;
            unsafe {
                // xfrc_applied is [nBody x 6]: [force(3), torque(3)]
                *ffi.xfrc_applied.add(base) = self.external_force[0];
                *ffi.xfrc_applied.add(base + 1) = self.external_force[1];
                *ffi.xfrc_applied.add(base + 2) = self.external_force[2];
            }
        }
    }

    /// Clear external forces from xfrc_applied.
    fn clear_xfrc(&mut self) {
        let ffi = self.data.ffi_mut();
        let base = self.body_id * 6;
        unsafe {
            for i in 0..6 {
                *ffi.xfrc_applied.add(base + i) = 0.0;
            }
        }
        self.external_force = [0.0; 3];
    }

    /// Extract FlightState from MuJoCo qpos/qvel.
    fn extract_state(&mut self) {
        let ffi = self.data.ffi();
        let t = unsafe { ffi.time };

        // Free joint: qpos[0..7] = pos(3) + quat(4), qvel[0..6] = lin_vel(3) + ang_vel(3)
        let joint_info = self
            .data
            .joint("root")
            .expect("MJCF model must have a joint named 'root'");
        let view = joint_info.view(&self.data);

        let qpos = &view.qpos;
        let qvel = &view.qvel;

        let perfect_state = FlightState {
            position: [qpos[0], qpos[1], qpos[2]],
            quaternion: [qpos[3], qpos[4], qpos[5], qpos[6]],
            linear_velocity: [qvel[0], qvel[1], qvel[2]],
            angular_velocity: [qvel[3], qvel[4], qvel[5]],
            timestamp: t,
        };

        // Apply sensory filter if configured
        #[cfg(feature = "mujoco")]
        {
            self.cached_state = match &mut self.sensory_filter {
                Some(filter) => filter.filter(&perfect_state),
                None => perfect_state,
            };
        }
        #[cfg(not(feature = "mujoco"))]
        {
            self.cached_state = perfect_state;
        }
    }

    /// Find body ID by name. Panics if not found.
    fn find_body_id(model: &MjModel, name: &str) -> usize {
        // Use FFI to search for the body by name
        let c_name = std::ffi::CString::new(name).expect("Body name contains null byte");
        let id = unsafe {
            mujoco_rs::mujoco_c::mj_name2id(
                model.ffi() as *const _,
                mujoco_rs::mujoco_c::mjtObj_mjOBJ_BODY as i32,
                c_name.as_ptr(),
            )
        };
        assert!(id >= 0, "Body '{}' not found in MJCF model", name);
        id as usize
    }

    /// Hold a body's velocity to zero (for freezing the beam before release).
    pub fn freeze_body(&mut self, body_name: &str) {
        let bid = Self::find_body_id(&self.model, body_name);
        // Find the joint associated with this body and zero its qvel
        let ffi = self.data.ffi_mut();
        // For a free joint body, the joint dof starts at body_jntadr
        let jnt_adr = unsafe { *self.model.ffi().body_jntadr.add(bid) };
        if jnt_adr >= 0 {
            let dof_adr = unsafe { *self.model.ffi().jnt_dofadr.add(jnt_adr as usize) };
            // Free joint has 6 DOF
            for i in 0..6 {
                unsafe {
                    *ffi.qvel.add((dof_adr as usize) + i) = 0.0;
                }
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
        unsafe {
            mujoco_rs::mujoco_c::mj_resetData(
                self.model.ffi() as *const _ as *mut _,
                self.data.ffi_mut() as *mut _,
            );
        }

        // Set initial position
        let joint_info = self
            .data
            .joint("root")
            .expect("MJCF model must have a joint named 'root'");
        {
            let mut view = joint_info.view_mut(&mut self.data);
            view.qpos[0] = 0.0;
            view.qpos[1] = 0.0;
            view.qpos[2] = altitude;
            view.qpos[3] = 1.0; // w
            view.qpos[4] = 0.0; // x
            view.qpos[5] = 0.0; // y
            view.qpos[6] = 0.0; // z
            for v in view.qvel.iter_mut() {
                *v = 0.0;
            }
        }

        // Forward kinematics to update derived quantities
        unsafe {
            mujoco_rs::mujoco_c::mj_forward(
                self.model.ffi() as *const _ as *mut _,
                self.data.ffi_mut() as *mut _,
            );
        }

        self.external_force = [0.0; 3];

        #[cfg(feature = "mujoco")]
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

        let joint_info = self
            .data
            .joint("root")
            .expect("MJCF model must have a joint named 'root'");
        {
            let mut view = joint_info.view_mut(&mut self.data);
            view.qpos[0] += perturbation * next_f64() * 0.1;
            view.qpos[1] += perturbation * next_f64() * 0.1;
            view.qpos[2] += perturbation * next_f64() * 0.05;

            // Small quaternion perturbation
            let tilt = perturbation * next_f64() * 0.1;
            let q = [1.0, tilt, tilt * 0.5, 0.0];
            let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            view.qpos[3] = q[0] / norm;
            view.qpos[4] = q[1] / norm;
            view.qpos[5] = q[2] / norm;
            view.qpos[6] = q[3] / norm;
        }

        unsafe {
            mujoco_rs::mujoco_c::mj_forward(
                self.model.ffi() as *const _ as *mut _,
                self.data.ffi_mut() as *mut _,
            );
        }

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
        let sim = MuJoCoSimulator::from_primitive();
        assert!(sim.state().altitude() > 0.0);
        assert!(sim.body_mass() > 0.0);
    }

    #[test]
    #[ignore] // Requires MuJoCo library
    fn test_mujoco_hover_maintains_altitude() {
        let mut sim = MuJoCoSimulator::from_primitive();
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
        let mut sim = MuJoCoSimulator::from_primitive();
        sim.apply_external_force([0.1, 0.0, 0.0]);
        let cmd = QuadrotorCommand::hover();
        sim.step(&cmd, 0.002);
        // Force should have some effect
        assert!(sim.state().position[0].abs() > 0.0 || sim.state().linear_velocity[0].abs() > 0.0);
    }

    #[test]
    #[ignore] // Requires MuJoCo library
    fn test_mujoco_reset() {
        let mut sim = MuJoCoSimulator::from_primitive();
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
        let mut sim = MuJoCoSimulator::from_primitive();
        let original_mass = sim.body_mass();
        sim.set_body_mass(original_mass * 1.5);
        assert!((sim.body_mass() - original_mass * 1.5).abs() < 1e-6);
    }

    #[test]
    #[ignore] // Requires MuJoCo library
    fn test_mujoco_body_position() {
        let sim = MuJoCoSimulator::from_primitive();
        let pos = sim.body_position("cf2");
        assert!(pos[2] > 0.0); // Should be above ground
    }

    #[test]
    #[ignore] // Requires MuJoCo library
    fn test_sacrifice_scene_loads() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/cf2_sacrifice.xml");
        let sim = MuJoCoSimulator::new(path);
        // Should have all bodies
        let drone_pos = sim.body_position("cf2");
        let human_pos = sim.body_position("human");
        let beam_pos = sim.body_position("beam");
        assert!(drone_pos[2] > 1.0); // Drone at 1.5m
        assert!(human_pos[0] > 1.0); // Human at x=2.0
        assert!(beam_pos[2] > 3.0); // Beam at 4.0m
    }
}
