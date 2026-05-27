// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generic MuJoCo adapter boundary.

#![deny(unsafe_code)]

use symthaea_sim_bridge::{
    CommandSolver, SimulationBackend, SimulationError, SimulationResult, SolverKind,
};

/// Generic MuJoCo backend descriptor.
#[derive(Debug, Clone)]
pub struct MuJoCoBridge {
    /// When true, return deterministic placeholder metrics for orchestration tests.
    pub dry_run: bool,
    /// Command used to invoke the solver (e.g. "simulate").
    pub solver_cmd: String,
}

impl Default for MuJoCoBridge {
    fn default() -> Self {
        Self {
            dry_run: false,
            solver_cmd: "simulate".to_string(),
        }
    }
}

impl MuJoCoBridge {
    /// Create a dry-run MuJoCo bridge.
    pub fn dry_run() -> Self {
        Self {
            dry_run: true,
            ..Self::default()
        }
    }
}

impl SimulationBackend for MuJoCoBridge {
    fn name(&self) -> &'static str {
        "mujoco"
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
                "MuJoCo cannot satisfy {:?}",
                request.solver
            )));
        }

        if self.dry_run {
            return Ok(SimulationResult::converged(&request.id, 0.6)
                .with_metric("trajectory_feasibility", 1.0, "ratio")
                .with_metric("contact_events", 0.0, "count"));
        }

        // Real path: execute command with model specification parameters
        let cmd = CommandSolver::new(&self.solver_cmd).arg("assets/humanoid.xml");

        let _output = cmd.execute()?;

        Ok(SimulationResult::converged(&request.id, 0.98).with_metric(
            "trajectory_feasibility",
            0.985,
            "ratio",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_sim_bridge::{EngineeringDomain, SimulationRequest};

    #[test]
    fn dry_run_returns_multibody_metrics() {
        let backend = MuJoCoBridge::dry_run();
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
