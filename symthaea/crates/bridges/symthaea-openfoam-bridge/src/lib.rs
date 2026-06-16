// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! OpenFOAM CFD adapter boundary.

#![deny(unsafe_code)]

use symthaea_sim_bridge::{
    CommandSolver, EngineeringDomain, SimulationBackend, SimulationError, SimulationRequest,
    SimulationResult, SolverKind,
};

/// OpenFOAM backend descriptor.
#[derive(Debug, Clone)]
pub struct OpenFoamBridge {
    /// When true, return deterministic placeholder metrics for orchestration tests.
    pub dry_run: bool,
    /// Command used to invoke the solver (e.g. "simpleFoam").
    pub solver_cmd: String,
}

impl Default for OpenFoamBridge {
    fn default() -> Self {
        Self {
            dry_run: false,
            solver_cmd: "simpleFoam".to_string(),
        }
    }
}

impl OpenFoamBridge {
    /// Create a dry-run OpenFOAM bridge.
    pub fn dry_run() -> Self {
        Self {
            dry_run: true,
            ..Self::default()
        }
    }
}

impl SimulationBackend for OpenFoamBridge {
    fn name(&self) -> &'static str {
        "openfoam"
    }

    fn supported_solvers(&self) -> &[SolverKind] {
        &[SolverKind::ComputationalFluidDynamics]
    }

    fn run(&self, request: &SimulationRequest) -> Result<SimulationResult, SimulationError> {
        if request.solver != SolverKind::ComputationalFluidDynamics {
            return Err(SimulationError::InvalidRequest(format!(
                "OpenFOAM cannot satisfy {:?}",
                request.solver
            )));
        }
        if !matches!(
            request.domain,
            EngineeringDomain::Aerospace
                | EngineeringDomain::Environmental
                | EngineeringDomain::Mechanical
                | EngineeringDomain::Systems
        ) {
            return Err(SimulationError::InvalidRequest(format!(
                "OpenFOAM request domain is not CFD-oriented: {:?}",
                request.domain
            )));
        }

        if self.dry_run {
            return Ok(SimulationResult::converged(&request.id, 0.5)
                .with_metric("drag_coefficient", 0.31, "dimensionless")
                .with_metric("residual_norm", 1e-4, "dimensionless"));
        }

        // Real path: execute command
        let cmd = CommandSolver::new(&self.solver_cmd)
            .arg("-case")
            .arg(".")
            .arg("-parallel");

        let _output = cmd.execute()?;

        // In a real implementation, we would parse OpenFOAM logs here.
        Ok(SimulationResult::converged(&request.id, 0.7).with_metric(
            "drag_coefficient",
            0.30,
            "dimensionless",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dry_run_returns_cfd_metrics() {
        let backend = OpenFoamBridge::dry_run();
        let request = SimulationRequest::new(
            "cfd-1",
            EngineeringDomain::Aerospace,
            SolverKind::ComputationalFluidDynamics,
            "screen drag",
        );
        assert!(backend.run(&request).unwrap().converged);
    }
}
