// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! OpenSees structural analysis adapter boundary.

#![deny(unsafe_code)]

use symthaea_sim_bridge::{
    CommandSolver, EngineeringDomain, SimulationBackend, SimulationError, SimulationRequest,
    SimulationResult, SolverKind,
};

/// OpenSees backend descriptor.
#[derive(Debug, Clone)]
pub struct OpenSeesBridge {
    /// When true, return deterministic placeholder metrics for orchestration tests.
    pub dry_run: bool,
    /// Command used to invoke the solver (e.g. "OpenSees").
    pub solver_cmd: String,
}

impl Default for OpenSeesBridge {
    fn default() -> Self {
        Self {
            dry_run: false,
            solver_cmd: "OpenSees".to_string(),
        }
    }
}

impl OpenSeesBridge {
    /// Create a dry-run OpenSees bridge.
    pub fn dry_run() -> Self {
        Self {
            dry_run: true,
            ..Self::default()
        }
    }
}

impl SimulationBackend for OpenSeesBridge {
    fn name(&self) -> &'static str {
        "opensees"
    }

    fn supported_solvers(&self) -> &[SolverKind] {
        &[SolverKind::FiniteElement]
    }

    fn run(&self, request: &SimulationRequest) -> Result<SimulationResult, SimulationError> {
        if request.solver != SolverKind::FiniteElement {
            return Err(SimulationError::InvalidRequest(format!(
                "OpenSees cannot satisfy {:?}",
                request.solver
            )));
        }
        if !matches!(
            request.domain,
            EngineeringDomain::Civil | EngineeringDomain::Systems
        ) {
            return Err(SimulationError::InvalidRequest(format!(
                "OpenSees expected civil/systems request, got {:?}",
                request.domain
            )));
        }

        if self.dry_run {
            return Ok(SimulationResult::converged(&request.id, 0.55)
                .with_metric("max_drift_ratio", 0.012, "ratio")
                .with_metric("demand_capacity_ratio", 0.72, "ratio"));
        }

        // Real path: execute command
        let cmd = CommandSolver::new(&self.solver_cmd).arg("model.tcl");

        let output = cmd.execute()?;

        // TODO(#solver-output-parsing): CommandSolver::execute now genuinely
        // spawns OpenSees, but this adapter does not yet parse its output
        // (typically Tcl-scripted recorder files, not stdout) to determine
        // real convergence/drift metrics. Returning a fabricated "converged"
        // result here would be worse than an honest error -- a caller making
        // a safety decision (e.g. drift-ratio check) must not receive
        // numbers that were never actually verified against solver output.
        Err(SimulationError::Adapter(format!(
            "OpenSees ran successfully but real-output parsing is not yet \
             implemented; cannot report convergence/drift metrics without \
             parsing recorder output. Raw stdout ({} bytes) was discarded.",
            output.len()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dry_run_returns_structural_metrics() {
        let backend = OpenSeesBridge::dry_run();
        let request = SimulationRequest::new(
            "fea-1",
            EngineeringDomain::Civil,
            SolverKind::FiniteElement,
            "screen frame drift",
        );
        let result = backend.run(&request).unwrap();
        assert!(result.converged);
        assert_eq!(result.metrics[0].name, "max_drift_ratio");
    }
}
