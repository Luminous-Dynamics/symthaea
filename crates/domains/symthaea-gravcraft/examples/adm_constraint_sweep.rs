// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use std::error::Error;
use symthaea_gravcraft::{
    AmplifierAdmMetric, GravcraftConfig, MetricCommand, adm_constraint_diagnostics,
};

fn main() -> Result<(), Box<dyn Error>> {
    let base = AmplifierAdmMetric::new(
        [0.0; 3],
        MetricCommand {
            amplifiers: [
                (0.05, 10.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
            ],
        },
        GravcraftConfig::default(),
    )?;
    let coordinates = [0.0, 15.0, 0.5, 0.25];
    let steps_m = [
        2.0e-2_f64, 1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4, 3.125e-4,
    ];

    println!("Symthaea GravCraft independent ADM constraint sweep");
    println!("epistemic_status=FINITE_DIFFERENCE_CONSTRAINT_CROSS_CHECK_NOT_SOURCE_FEASIBILITY");
    println!("metric=alpha_1_gamma_identity_static_three_amplifier_shift");
    println!("convention=K_ij=+0.5*(partial_i_beta_j+partial_j_beta_i)");
    println!("coordinates=(ct,x,y,z)=(0,15,0.5,0.25)_m");
    println!(
        "step_m,K_trace_per_m,KijKij_per_m2,H_geometric_per_m2,16pi_rho_4d_per_m2,H_residual_per_m2,Mgeo_x_per_m2,Mgeo_y_per_m2,Mgeo_z_per_m2,8pi_j_x_per_m2,8pi_j_y_per_m2,8pi_j_z_per_m2,Mres_x_per_m2,Mres_y_per_m2,Mres_z_per_m2,M_norm_per_m2"
    );

    for step_m in steps_m {
        let mut field = base.clone();
        field.derivative_step_m = step_m;
        let diagnostics = adm_constraint_diagnostics(&field, coordinates, 96)?;
        println!(
            "{step_m:.9e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
            diagnostics.extrinsic_curvature_trace,
            diagnostics.extrinsic_curvature_squared,
            diagnostics.hamiltonian_geometric_term,
            16.0 * std::f64::consts::PI * diagnostics.eulerian_energy_density,
            diagnostics.hamiltonian_residual,
            diagnostics.momentum_geometric_term[0],
            diagnostics.momentum_geometric_term[1],
            diagnostics.momentum_geometric_term[2],
            8.0 * std::f64::consts::PI * diagnostics.eulerian_momentum_density[0],
            8.0 * std::f64::consts::PI * diagnostics.eulerian_momentum_density[1],
            8.0 * std::f64::consts::PI * diagnostics.eulerian_momentum_density[2],
            diagnostics.momentum_constraint_residual[0],
            diagnostics.momentum_constraint_residual[1],
            diagnostics.momentum_constraint_residual[2],
            diagnostics.momentum_residual_norm,
        );
    }

    println!(
        "The ADM Hamiltonian and momentum residuals are independent projections to compare against the 4D Einstein-implied source."
    );
    println!(
        "Both routes use finite differences in this prototype; agreement under step refinement is necessary but not sufficient for a continuum solution."
    );
    println!(
        "No result here establishes a realizable stress-energy source, global energy conditions, or propulsion."
    );
    Ok(())
}
