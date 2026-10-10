// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! ngspice adapter boundary.

#![deny(unsafe_code)]

/// Strict numeric parsing primitives for single-plot ASCII rawfiles.
pub mod rawfile;

use symthaea_sim_bridge::{
    CommandSolver, EngineeringDomain, SimulationBackend, SimulationError, SimulationRequest,
    SimulationResult, SolverKind,
};

/// ngspice backend descriptor.
#[derive(Debug, Clone)]
pub struct NgspiceBridge {
    /// When true, return deterministic placeholder metrics for orchestration tests.
    pub dry_run: bool,
    /// Command used to invoke the solver (e.g. "ngspice").
    pub solver_cmd: String,
}

impl Default for NgspiceBridge {
    fn default() -> Self {
        Self {
            dry_run: false,
            solver_cmd: "ngspice".to_string(),
        }
    }
}

impl NgspiceBridge {
    /// Create a dry-run ngspice bridge.
    pub fn dry_run() -> Self {
        Self {
            dry_run: true,
            ..Self::default()
        }
    }
}

impl SimulationBackend for NgspiceBridge {
    fn name(&self) -> &'static str {
        "ngspice"
    }

    fn supported_solvers(&self) -> &[SolverKind] {
        &[SolverKind::Circuit]
    }

    fn run(&self, request: &SimulationRequest) -> Result<SimulationResult, SimulationError> {
        if request.solver != SolverKind::Circuit {
            return Err(SimulationError::InvalidRequest(format!(
                "ngspice cannot satisfy {:?}",
                request.solver
            )));
        }
        if !matches!(
            request.domain,
            EngineeringDomain::Electrical | EngineeringDomain::Systems
        ) {
            return Err(SimulationError::InvalidRequest(format!(
                "ngspice expected electrical/systems request, got {:?}",
                request.domain
            )));
        }

        if self.dry_run {
            return Ok(SimulationResult::dry_run(&request.id, self.name(), 0.55)
                .with_metric("peak_voltage", 12.1, "V")
                .with_metric("settling_time", 0.032, "s"));
        }

        // Real path: execute command
        let cmd = CommandSolver::new(&self.solver_cmd)
            .arg("-b") // Batch mode
            .arg("input.sp");

        let output = cmd.execute()?;

        // The strict ASCII parser in the rawfile module is covered by golden
        // fixtures, but this execution path still lacks frozen per-run input
        // identity, raw/log artifact capture, solver/version provenance, and
        // independently justified convergence semantics. Keep failing closed:
        // parsed numbers alone would not be sufficient engineering evidence.
        Err(SimulationError::Adapter(format!(
            "ngspice ran successfully but real-output parsing is not yet \
             implemented; cannot report convergence/voltage metrics without \
             parsing rawfile output. Raw stdout ({} bytes) was discarded.",
            output.len()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dry_run_returns_circuit_metrics() {
        let backend = NgspiceBridge::dry_run();
        let request = SimulationRequest::new(
            "spice-1",
            EngineeringDomain::Electrical,
            SolverKind::Circuit,
            "screen transient response",
        );
        assert!(backend.run(&request).unwrap().converged);
    }
}
