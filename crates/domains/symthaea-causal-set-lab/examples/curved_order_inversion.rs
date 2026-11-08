// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use symthaea_causal_set_lab::{
    ConformalFrw1p1, estimate_dimensionless_curvature, qmc_expected_comparable_pair_fraction,
};

fn main() -> Result<(), Box<dyn Error>> {
    const EXTENT: f64 = 2.0;
    const TRUE_Q: f64 = 2.0;
    const EVENT_COUNT: usize = 4096;
    const SEED: u64 = 0xC0A7_2026;
    const QMC_POINTS: usize = 8192;
    const Q_MAX: f64 = 6.0;

    // q=kappa*T^2 is the scale-free quantity inferred from order statistics.
    // This metric has kappa=TRUE_Q/EXTENT^2; without an external T anchor,
    // report TRUE_Q rather than claiming kappa is independently identified.
    let kappa = TRUE_Q / EXTENT.powi(2);
    let metric = ConformalFrw1p1::new(kappa, EXTENT)?;
    let causet = metric.sprinkle_fixed_cardinality(SEED, EVENT_COUNT)?;
    let observed = causet.ordering_fraction().ok_or("empty causal set")?;
    let flat_reference = qmc_expected_comparable_pair_fraction(0.0, QMC_POINTS)?;
    let curved_reference = qmc_expected_comparable_pair_fraction(TRUE_Q, QMC_POINTS)?;
    let estimate = estimate_dimensionless_curvature(observed, Q_MAX, QMC_POINTS, 48)?;

    // Demonstrate the exact scale degeneracy of (kappa,T): (kappa,T) and
    // (16*kappa,T/4) have the same q and therefore the same QMC order reference.
    let rescaled_kappa = kappa * 16.0;
    let rescaled_extent = EXTENT / 4.0;
    let rescaled_q = rescaled_kappa * rescaled_extent * rescaled_extent;

    println!("Symthaea curved causal-set dimensionless curvature inversion");
    println!(
        "epistemic_status=SYNTHETIC_VOLUME_FAITHFUL_1P1D_FRW_INVERSE_NOT_QUANTUM_GRAVITY_EVIDENCE"
    );
    println!("metric=ds2=a(eta)^2*(-deta^2+dx^2),a=exp(kappa*eta^2/2)");
    println!("sampling=FIXED_CARDINALITY_POISSON_CONDITIONED_VOLUME_FAITHFUL");
    println!("extent={EXTENT:.6}");
    println!("true_dimensionless_curvature_q={TRUE_Q:.9}");
    println!("kappa_if_extent_is_externally_anchored={kappa:.9}");
    println!("events={EVENT_COUNT}");
    println!("seed={SEED}");
    println!("qmc_points={QMC_POINTS}");
    println!("observed_order_fraction={observed:.12}");
    println!(
        "flat_q0_reference={:.12}",
        flat_reference.comparable_pair_fraction
    );
    println!(
        "curved_true_q_reference={:.12}",
        curved_reference.comparable_pair_fraction
    );
    println!(
        "flat_reference_abs_error={:.12}",
        (observed - flat_reference.comparable_pair_fraction).abs()
    );
    println!(
        "estimated_dimensionless_curvature_q={:.9}",
        estimate.dimensionless_curvature
    );
    println!(
        "reconstructed_order_fraction={:.12}",
        estimate.reconstructed_ordering_fraction
    );
    println!(
        "inverse_abs_order_fraction_residual={:.12}",
        (estimate.reconstructed_ordering_fraction - observed).abs()
    );
    println!("rescaled_equivalent_q={rescaled_q:.12}");
    println!(
        "Only q=kappa*T^2 is identified by scale-free order statistics; kappa requires an external scale."
    );
    println!(
        "Finite-sample uncertainty across independent sprinklings is separate from QMC-reference error."
    );
    println!(
        "A curved-volume inversion is conditional on the declared metric, region and sampling law."
    );
    Ok(())
}
