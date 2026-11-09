// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use symthaea_causal_set_lab::{
    CausalSetNd, DimensionInferenceStatus, DimensionPrior,
    calibrate_minkowski_ordering_fraction, estimate_finite_size_dimension_posterior,
};

fn main() -> Result<(), Box<dyn Error>> {
    const EVENT_COUNT: usize = 128;
    const CALIBRATION_REPLICATES: usize = 192;
    const PROPER_TIME: f64 = 1.0;
    const BASE_SEED: u64 = 0xB4A9_2026;

    let dimensions = [2_usize, 3, 4];
    let calibration = calibrate_minkowski_ordering_fraction(
        EVENT_COUNT,
        PROPER_TIME,
        &dimensions,
        CALIBRATION_REPLICATES,
        BASE_SEED,
    )?;

    println!("epistemic_status=FINITE_N_MINKOWSKI_MONTE_CARLO_KDE_POSTERIOR_NOT_PHYSICAL_DIMENSION_MEASUREMENT");
    println!("likelihood=whole_causet_empirical_KDE_not_iid_binomial_pairs");
    println!("event_count={EVENT_COUNT}");
    println!("calibration_replicates_per_dimension={CALIBRATION_REPLICATES}");
    println!("proper_time={PROPER_TIME}");
    println!("calibration_base_seed={BASE_SEED}");
    println!("seed_scheme={}", calibration.seed_scheme);
    println!("dimension,calibration_mean_order_fraction,calibration_sd,kde_bandwidth");

    for entry in &calibration.dimensions {
        let mean = entry.ordering_fractions.iter().sum::<f64>()
            / entry.ordering_fractions.len() as f64;
        let variance = entry.ordering_fractions.iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>() / (entry.ordering_fractions.len() - 1) as f64;
        // The posterior API reports the actual robust bandwidth for each
        // target observation; this row is calibration-only descriptive data.
        println!("{},{mean:.9},{:.9},computed-per-query", entry.spacetime_dimension, variance.sqrt());
    }

    let uniform_prior = [
        DimensionPrior {
            spacetime_dimension: 2,
            probability: 1.0 / 3.0,
        },
        DimensionPrior {
            spacetime_dimension: 3,
            probability: 1.0 / 3.0,
        },
        DimensionPrior {
            spacetime_dimension: 4,
            probability: 1.0 / 3.0,
        },
    ];
    let low_dimension_prior = [
        DimensionPrior {
            spacetime_dimension: 2,
            probability: 0.6,
        },
        DimensionPrior {
            spacetime_dimension: 3,
            probability: 0.3,
        },
        DimensionPrior {
            spacetime_dimension: 4,
            probability: 0.1,
        },
    ];

    println!("true_dimension,held_out_seed,observed_ordering_fraction,prior_family,status,posterior_mean,p_dim_le_3,credible_set_95");
    for true_dimension in dimensions {
        // These qualification seeds are deliberately disjoint from the
        // calibration seed schedule. The output remains a synthetic replay.
        for held_out_seed in [0x51A7_u64, 0x2026_1109, 0xC0FF_EE, 0xD1A0_5EED] {
            let causet = CausalSetNd::sprinkle_fixed_cardinality(
                held_out_seed ^ ((true_dimension as u64) << 48),
                EVENT_COUNT,
                true_dimension,
                PROPER_TIME,
            )?;
            let observed = causet.ordering_fraction().ok_or("causet has too few events")?;

            for (prior_name, prior) in [
                ("uniform", &uniform_prior[..]),
                ("low-dimension-favouring", &low_dimension_prior[..]),
            ] {
                let posterior = estimate_finite_size_dimension_posterior(
                    EVENT_COUNT,
                    observed,
                    &calibration,
                    prior,
                )?;
                let status = match posterior.status {
                    DimensionInferenceStatus::CalibratedPosterior => "CALIBRATED_POSTERIOR",
                    DimensionInferenceStatus::OutsideCalibrationSupport => "OUTSIDE_CALIBRATION_SUPPORT",
                };
                let posterior_mean = posterior.posterior_mean_dimension
                    .map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "NA".to_string());
                let p_dim_le_3 = posterior.posterior_probability_dimension_at_most_three
                    .map(|value| format!("{value:.9}"))
                    .unwrap_or_else(|| "NA".to_string());
                let credible_set = posterior.credible_set_95
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(";");
                println!(
                    "{true_dimension},{held_out_seed},{observed:.9},{prior_name},{status},{posterior_mean},{p_dim_le_3},{credible_set}"
                );
            }
        }
    }

    println!("claim_boundary=finite_size_calibration_for_fixed_cardinality_flat_minkowski_sprinklings");
    println!("non_claims=curved_spacetime,non_poisson_sprinklings,non_manifold_causets,quantum_gravity,thermodynamic_continuum_limit");
    Ok(())
}
