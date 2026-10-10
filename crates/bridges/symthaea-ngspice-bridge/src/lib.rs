// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! ngspice adapter boundary.

#![deny(unsafe_code)]

/// Immutable, request-bound primary netlist identity.
pub mod input;

/// Strict numeric parsing primitives for single-plot ASCII rawfiles.
pub mod rawfile;

use symthaea_sim_bridge::{
    EngineeringDomain, SimulationBackend, SimulationError, SimulationRequest, SimulationResult,
    SolverKind,
};

/// ngspice backend descriptor.
#[derive(Debug, Clone)]
pub struct NgspiceBridge {
    /// When true, return deterministic placeholder metrics for orchestration tests.
    pub dry_run: bool,
    /// Executable reserved for the future artifact-bound solver path (e.g. "ngspice").
    /// The current real path deliberately refuses to spawn any executable until the
    /// SimulationRequest contract carries an immutable netlist artifact.
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

        // Refuse to execute ambient input.sp. SimulationRequest currently
        // carries intent/parameters/metric names, but no immutable netlist
        // artifact, digest, include closure, or working-directory identity.
        // Running a fixed relative path would execute an input not bound to
        // this request and make provenance unverifiable. Add a typed netlist
        // artifact to the execution contract before enabling the real path.
        Err(SimulationError::Adapter(
            "ngspice execution requires an explicit immutable netlist artifact, but \
             SimulationRequest does not carry one; refusing to execute ambient ./input.sp. \
             The bridge remains fail-closed until input identity, raw/log artifacts, \
             solver identity, and separate convergence evidence are wired.".into(),
        ))
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

    #[test]
    fn real_path_refuses_to_execute_unbound_ambient_netlist() {
        let backend = NgspiceBridge {
            dry_run: false,
            // This executable intentionally does not need to exist. The bridge
            // must reject the missing request artifact before attempting spawn.
            solver_cmd: "symthaea-test-must-not-spawn".into(),
        };
        let request = SimulationRequest::new(
            "spice-unbound-input",
            EngineeringDomain::Electrical,
            SolverKind::Circuit,
            "screen transient response",
        );
        let error = backend.run(&request).expect_err("unbound input must fail closed");
        match error {
            SimulationError::Adapter(message) => {
                assert!(message.contains("explicit immutable netlist artifact"));
                assert!(message.contains("refusing to execute ambient ./input.sp"));
            }
            other => panic!("expected an adapter error, got {other:?}"),
        }
    }
}
