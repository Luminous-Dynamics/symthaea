// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_engineering::sim_bridge::{
    CoupledSimulationStage, CouplingMode, EngineeringDomain, MultiPhysicsRequest, SolverKind,
};

fn main() {
    let request = MultiPhysicsRequest::new(
        "wing-coupled-001",
        "aero-thermal-structural concept screening",
        CouplingMode::Iterative,
    )
    .with_stage(
        CoupledSimulationStage::new(
            "flow",
            EngineeringDomain::Aerospace,
            SolverKind::ComputationalFluidDynamics,
        )
        .produces(["pressure_field", "heat_flux"]),
    )
    .with_stage(
        CoupledSimulationStage::new(
            "structure",
            EngineeringDomain::Materials,
            SolverKind::FiniteElement,
        )
        .consumes(["pressure_field", "heat_flux"])
        .produces(["stress_field", "margin"]),
    );

    println!("multi-physics objective: {}", request.objective);
    println!("coupling mode: {:?}", request.coupling);
    println!("stages: {}", request.stages.len());
}
