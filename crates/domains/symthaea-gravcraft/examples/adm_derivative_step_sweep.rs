// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use std::error::Error;
use symthaea_gravcraft::{
    AmplifierAdmMetric, GravcraftConfig, MetricCommand, implied_source_diagnostics,
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
        1.0e-1_f64, 5.0e-2, 2.5e-2, 1.25e-2, 6.25e-3, 3.125e-3, 1.5625e-3, 7.8125e-4,
        3.90625e-4,
    ];

    println!("Symthaea GravCraft ADM finite-difference convergence sweep");
    println!("epistemic_status=NUMERICAL_STEP_SENSITIVITY_NOT_SOURCE_FEASIBILITY_OR_PROPULSION");
    println!("metric=alpha_1_gamma_identity_static_amplifier_shift");
    println!("coordinates=(ct,x,y,z)=(0,15,0.5,0.25)_m");
    println!("null_direction_samples=96");
    println!(
        "step_m,ricci_scalar_per_m2,kretschmann_per_m4,eulerian_energy_density_per_m2,min_sampled_null_energy_per_m2,max_abs_T_component_per_m2,relative_delta_ricci,relative_delta_kretschmann,relative_delta_rho"
    );

    let mut previous: Option<(f64, f64, f64)> = None;
    for step_m in steps_m {
        let mut field = base.clone();
        field.derivative_step_m = step_m;
        let diagnostics = implied_source_diagnostics(&field, coordinates, 96)?;
        let ricci = diagnostics.curvature.ricci_scalar;
        let kretschmann = diagnostics.curvature.kretschmann_scalar;
        let rho = diagnostics.eulerian_energy_density;
        let max_abs_stress = diagnostics
            .stress_energy
            .iter()
            .flatten()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);

        let relative_delta = |current: f64, previous: f64| {
            (current - previous).abs() / current.abs().max(previous.abs()).max(1.0e-30)
        };
        let (delta_ricci, delta_kretschmann, delta_rho) = previous
            .map(|(previous_ricci, previous_kretschmann, previous_rho)| {
                (
                    relative_delta(ricci, previous_ricci),
                    relative_delta(kretschmann, previous_kretschmann),
                    relative_delta(rho, previous_rho),
                )
            })
            .unwrap_or((f64::NAN, f64::NAN, f64::NAN));

        println!(
            "{step_m:.9e},{ricci:.12e},{kretschmann:.12e},{rho:.12e},{:.12e},{max_abs_stress:.12e},{delta_ricci:.6e},{delta_kretschmann:.6e},{delta_rho:.6e}",
            diagnostics.minimum_sampled_null_energy,
        );
        previous = Some((ricci, kretschmann, rho));
    }

    println!(
        "The sweep reports finite-difference sensitivity; it does not prove convergence to a continuum solution."
    );
    println!(
        "A non-negative sampled null-energy minimum is not a proof of the NEC in all directions."
    );
    println!(
        "No result here establishes a physically realizable stress-energy source or a propulsion mechanism."
    );
    Ok(())
}
