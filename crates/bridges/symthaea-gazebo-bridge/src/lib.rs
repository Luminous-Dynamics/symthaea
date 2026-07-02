// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Generic Gazebo (gz-sim) adapter boundary.
#![deny(unsafe_code)]
use std::fs;
use symthaea_sim_bridge::{SimulationBackend, SimulationError, SimulationResult, SolverKind};
#[derive(Debug, Clone)]
pub struct GazeboBridge {
    pub dry_run: bool,
    pub solver_cmd: String,
    pub asset_dir: String,
}
impl Default for GazeboBridge {
    fn default() -> Self {
        Self {
            dry_run: false,
            solver_cmd: "gz".to_string(),
            asset_dir: "assets/gazebo".to_string(),
        }
    }
}
impl GazeboBridge {
    pub fn dry_run() -> Self {
        Self {
            dry_run: true,
            ..Self::default()
        }
    }
    pub fn prepare_asset_dir(&self) -> Result<(), SimulationError> {
        fs::create_dir_all(&self.asset_dir).map_err(|e| SimulationError::Adapter(e.to_string()))?;
        Ok(())
    }
    /// Spawns Gazebo (gz-sim) as a background process.
    pub fn spawn_gazebo(&self) -> Result<std::process::Child, std::io::Error> {
        let sdf_path = "assets/humanoid.sdf";
        std::process::Command::new(&self.solver_cmd)
            .arg("sim")
            .arg("-r")
            .arg("-s")
            .arg(sdf_path)
            .spawn()
    }
}
impl SimulationBackend for GazeboBridge {
    fn name(&self) -> &'static str {
        "gazebo"
    }
    fn supported_solvers(&self) -> &[SolverKind] {
        &[SolverKind::MultibodyDynamics]
    }
    fn run(
        &self,
        request: &symthaea_sim_bridge::SimulationRequest,
    ) -> Result<SimulationResult, SimulationError> {
        if request.solver != SolverKind::MultibodyDynamics {
            return Err(SimulationError::InvalidRequest(format!(
                "Gazebo cannot satisfy {:?}",
                request.solver
            )));
        }
        if self.dry_run {
            return Ok(SimulationResult::converged(&request.id, 0.6)
                .with_metric("trajectory_feasibility", 1.0, "ratio")
                .with_metric("contact_events", 0.0, "count"));
        }
        // TODO(#solver-output-parsing): unlike the batch solvers (ngspice,
        // MuJoCo, OpenFOAM), `gz sim -r` is a continuous, non-terminating
        // simulator -- there is no single exit-code/stdout snapshot to
        // parse, so this adapter cannot yet extract real convergence or
        // trajectory-feasibility metrics from a live run. The previous
        // version of this function spawned a detached Gazebo process via
        // `spawn_gazebo()` (leaking it -- the child handle was dropped
        // without `wait()`/`kill()`) and then returned a hardcoded
        // "converged: 0.98" result that had no relationship to what the
        // simulator actually did. Use `spawn_daemon()` to launch Gazebo and
        // own the process handle instead; `run()` cannot honestly report
        // results until this adapter can query a running instance or parse
        // a results file it writes.
        Err(SimulationError::Adapter(
            "gazebo live-run result extraction is not yet implemented; use \
             spawn_daemon() to launch the simulator and own its process \
             handle, or dry_run() for orchestration testing with placeholder \
             metrics"
                .into(),
        ))
    }
    fn spawn_daemon(
        &self,
        request: &symthaea_sim_bridge::SimulationRequest,
    ) -> Result<Option<std::process::Child>, SimulationError> {
        if request.solver != SolverKind::MultibodyDynamics {
            return Err(SimulationError::InvalidRequest(format!(
                "Gazebo cannot satisfy {:?}",
                request.solver
            )));
        }
        if self.dry_run {
            return Ok(None);
        }
        self.prepare_asset_dir()?;
        let child = self
            .spawn_gazebo()
            .map_err(|e| SimulationError::Adapter(e.to_string()))?;
        Ok(Some(child))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_sim_bridge::{EngineeringDomain, SimulationRequest};
    #[test]
    fn dry_run_returns_multibody_metrics() {
        let backend = GazeboBridge::dry_run();
        let request = SimulationRequest::new(
            "mbd-1",
            EngineeringDomain::Robotics,
            SolverKind::MultibodyDynamics,
            "check trajectory",
        );
        let result = backend.run(&request).unwrap();
        assert!(result.converged);
        assert_eq!(result.metrics.len(), 2);
    }
}
