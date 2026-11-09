// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use symthaea_causal_set_lab::{
    FiniteConeLinkProfile, ProductWindowLinkProfile, alexandrov_volume_coefficient,
    double_scaling_product_window,
};

fn main() -> Result<(), Box<dyn Error>> {
    const DENSITY: f64 = 128_000.0;
    const AMBIENT_TIME: f64 = 1.0;
    const MU: f64 = 0.6;
    const ETA: f64 = 0.9;

    // The comparison values are transcribed from the published finite-cone
    // and product-window theory columns in Table I of Boguña & Krioukov
    // (arXiv:2610.03247). They are comparison data, not implementation inputs.
    let published_finite_cone_means = [0.8332_f64, 0.7290, 0.6807];
    let published_product_window_means = [0.8862_f64, 0.8930, 0.9064];

    println!(
        "epistemic_status=ANALYTIC_LINK_SAMPLING_REFERENCE_NOT_SPRINKLING_OR_PHYSICS_VALIDATION"
    );
    println!("source=arXiv:2610.03247_Boguna_Krioukov_2026");
    println!("density={DENSITY}");
    println!("ambient_time_extent={AMBIENT_TIME}");
    println!("mu={MU}");
    println!("eta={ETA}");
    println!(
        "spatial_dimension,volume_coefficient,T_hat,finite_cone_mean,published_finite_cone_mean,finite_delta,product_tauhat_cutoff,product_mean,published_product_mean,product_delta,rapidity_cutoff,root_admissibility"
    );

    for spatial_dimension in 1..=3 {
        let volume_coefficient = alexandrov_volume_coefficient(spatial_dimension)?;
        let spacetime_dimension = (spatial_dimension + 1) as f64;
        let t_hat = AMBIENT_TIME * (DENSITY * volume_coefficient).powf(1.0 / spacetime_dimension);
        let finite_cone = FiniteConeLinkProfile::new(spatial_dimension, t_hat)?;
        let double_scaling = double_scaling_product_window(
            spatial_dimension,
            DENSITY,
            AMBIENT_TIME,
            AMBIENT_TIME,
            MU,
            ETA,
        )?;
        let product_window = ProductWindowLinkProfile::new(
            spatial_dimension,
            double_scaling.dimensionless_proper_time_cutoff,
        )?;

        let finite_mean = finite_cone.mean_dimensionless_proper_time();
        let product_mean = product_window.mean_dimensionless_proper_time()?;
        let finite_delta = finite_mean - published_finite_cone_means[spatial_dimension - 1];
        let product_delta = product_mean - published_product_window_means[spatial_dimension - 1];

        println!(
            "{spatial_dimension},{volume_coefficient:.12},{t_hat:.9},{finite_mean:.9},{:.9},{finite_delta:+.9},{:.9},{product_mean:.9},{:.9},{product_delta:+.9},{:.9},{}",
            published_finite_cone_means[spatial_dimension - 1],
            double_scaling.dimensionless_proper_time_cutoff,
            published_product_window_means[spatial_dimension - 1],
            double_scaling.rapidity_cutoff,
            double_scaling.asymptotically_admissible,
        );
    }

    println!("comparison_status=FINITE_CONE_TABLE_I_DISCREPANCY_REQUIRES_CONVENTION_AUDIT");
    println!(
        "claim_boundary=ANALYTIC_FORMULA_REPRODUCTION_ONLY_NO_CONTINUUM_OR_QUANTUM_GRAVITY_CLAIM"
    );
    Ok(())
}
