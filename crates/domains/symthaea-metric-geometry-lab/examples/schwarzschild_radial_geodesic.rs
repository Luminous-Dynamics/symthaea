// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use std::f64::consts::PI;
use symthaea_metric_geometry_lab::{GeodesicState, SchwarzschildExterior, geodesic_rk4_step};

fn main() -> Result<(), Box<dyn Error>> {
    let mass: f64 = 1.0;
    let radius: f64 = 10.0;
    let radial_tangent: f64 = -0.05;
    let f: f64 = 1.0 - 2.0 * mass / radius;
    let time_tangent = ((1.0 + radial_tangent * radial_tangent / f) / f).sqrt();
    let field = SchwarzschildExterior::new(mass)?;
    let mut state = GeodesicState {
        coordinates: [0.0, radius, PI / 2.0, 0.0],
        tangent: [time_tangent, radial_tangent, 0.0, 0.0],
    };

    println!("Schwarzschild exterior radial timelike geodesic");
    println!("epistemic_status=ANALYTIC_METRIC_GEODESIC_REFERENCE_NOT_NUMERICAL_RELATIVITY");
    println!("units=G=c=1");
    println!("initial_radius={radius}");
    println!("initial_norm_squared={:.12}", state.norm_squared(&field)?);
    println!("step,coordinate_time,radius,proper_time_norm");

    for step in 0..=1_000 {
        if step % 100 == 0 {
            println!(
                "{step},{:.12},{:.12},{:.12}",
                state.coordinates[0],
                state.coordinates[1],
                state.norm_squared(&field)?,
            );
        }
        if step < 1_000 {
            geodesic_rk4_step(&field, &mut state, 1e-3)?;
        }
    }
    Ok(())
}
