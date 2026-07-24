// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generic MuJoCo adapter boundary.

#![deny(unsafe_code)]

use std::fs;
use symthaea_humanoid::morphology::HumanoidMorphology;
use symthaea_sim_bridge::{
    CommandSolver, SimulationBackend, SimulationError, SimulationResult, SolverKind,
};

/// Generic MuJoCo adapter boundary.
#[derive(Debug, Clone)]
pub struct MuJoCoBridge {
    /// When true, return deterministic placeholder metrics for orchestration tests.
    pub dry_run: bool,
    /// Command used to invoke the solver (e.g. "simulate").
    pub solver_cmd: String,
    /// Directory to export generated MJCF assets.
    pub asset_dir: String,
}

impl Default for MuJoCoBridge {
    fn default() -> Self {
        Self {
            dry_run: false,
            solver_cmd: "simulate".to_string(),
            asset_dir: "assets/mujoco".to_string(),
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

    /// Export the 64-DOF FullSpine MJCF asset.
    pub fn export_flagship_mjcf(&self) -> Result<String, SimulationError> {
        let morphology = HumanoidMorphology::FullSpine;
        let mjcf = morphology.to_mjcf();

        fs::create_dir_all(&self.asset_dir).map_err(|e| SimulationError::Adapter(e.to_string()))?;
        let path = format!("{}/humanoid_flagship.xml", self.asset_dir);
        fs::write(&path, mjcf).map_err(|e| SimulationError::Adapter(e.to_string()))?;

        Ok(path)
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
            return Ok(SimulationResult::dry_run(&request.id, self.name(), 0.6)
                .with_metric("trajectory_feasibility", 1.0, "ratio")
                .with_metric("contact_events", 0.0, "count"));
        }

        // 1. Ensure the 64-DOF model is exported
        let xml_path =
            if request.objective.contains("flagship") || request.objective.contains("64-dof") {
                self.export_flagship_mjcf()?
            } else {
                "assets/humanoid.xml".to_string()
            };

        // 2. Real path: execute command with model specification parameters
        let cmd = CommandSolver::new(&self.solver_cmd).arg(&xml_path);

        let output = cmd.execute()?;

        // TODO(#solver-output-parsing): CommandSolver::execute now genuinely
        // spawns MuJoCo, but this adapter does not yet parse its output to
        // determine real trajectory-feasibility/contact metrics. Returning a
        // fabricated "converged" result would let a caller make a decision
        // against numbers that were never actually verified.
        Err(SimulationError::Adapter(format!(
            "MuJoCo ran successfully on {xml_path} but real-output parsing \
             is not yet implemented; cannot report trajectory-feasibility \
             metrics without parsing simulator output. Raw stdout ({} bytes) \
             was discarded.",
            output.len()
        )))
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
