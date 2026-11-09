// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use symthaea_causal_set_lab::causal_set_nd::CausalSetNd;
use symthaea_causal_set_lab::{
    DimensionInferenceStatus, DimensionPrior, calibrate_minkowski_ordering_fraction,
    estimate_finite_size_dimension_posterior,
};

const DIMENSIONS: [usize; 3] = [2, 3, 4];
const EVENT_COUNTS: [usize; 5] = [64, 128, 256, 512, 1024];
const CALIBRATION_REPLICATES: [usize; 5] = [192, 192, 128, 64, 24];
const QUALIFICATION_REPLICATES: usize = 24;

const PRIORS: [(&str, [f64; 3]); 3] = [
    ("uniform", [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
    ("low-dimension-favouring", [0.60, 0.30, 0.10]),
    ("high-dimension-favouring", [0.10, 0.30, 0.60]),
];

#[derive(Default)]
struct Aggregate {
    samples: usize,
    calibrated: usize,
    outside_support: usize,
    covered_95: usize,
    correct_map: usize,
    false_dimension_reduction: usize,
    true_probability_sum: f64,
    log_loss_sum: f64,
    brier_sum: f64,
    p_dim_le_3_values: Vec<f64>,
}

impl Aggregate {
    fn record(
        &mut self,
        true_dimension: usize,
        posterior: &symthaea_causal_set_lab::FiniteSizeDimensionPosterior,
    ) {
        self.samples += 1;
        if posterior.status != DimensionInferenceStatus::CalibratedPosterior {
            self.outside_support += 1;
            return;
        }
        self.calibrated += 1;
        let Some(true_probability) = posterior
            .dimensions
            .iter()
            .find(|entry| entry.spacetime_dimension == true_dimension)
            .and_then(|entry| entry.posterior_probability)
        else {
            self.outside_support += 1;
            self.calibrated -= 1;
            return;
        };
        if posterior.credible_set_95.contains(&true_dimension) {
            self.covered_95 += 1;
        }
        self.true_probability_sum += true_probability;
        self.log_loss_sum -= true_probability.max(1.0e-15).ln();

        let predicted_map = posterior
            .dimensions
            .iter()
            .filter_map(|entry| {
                entry
                    .posterior_probability
                    .map(|p| (entry.spacetime_dimension, p))
            })
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .map(|entry| entry.0);
        if predicted_map == Some(true_dimension) {
            self.correct_map += 1;
        }

        for entry in &posterior.dimensions {
            let target = if entry.spacetime_dimension == true_dimension {
                1.0
            } else {
                0.0
            };
            let probability = entry.posterior_probability.unwrap_or(0.0);
            self.brier_sum += (probability - target).powi(2);
        }
        if true_dimension == 4 {
            if let Some(p_low) = posterior.posterior_probability_dimension_at_most_three {
                self.p_dim_le_3_values.push(p_low);
                if p_low >= 0.95 {
                    self.false_dimension_reduction += 1;
                }
            }
        }
    }

    fn line(
        &mut self,
        event_count: usize,
        true_dimension: usize,
        prior_name: &str,
        calibration_replicates: usize,
    ) -> String {
        self.p_dim_le_3_values.sort_by(f64::total_cmp);
        let p95_low_dimension_mass = if self.p_dim_le_3_values.is_empty() {
            "NA".to_string()
        } else {
            let index = ((self.p_dim_le_3_values.len() - 1) as f64 * 0.95).ceil() as usize;
            format!("{:.9}", self.p_dim_le_3_values[index])
        };
        let total = self.samples.max(1) as f64;
        let calibrated = self.calibrated.max(1) as f64;
        let brier_denominator = calibrated * DIMENSIONS.len() as f64;
        let (coverage_low, coverage_high) = wilson_interval(self.covered_95, self.calibrated);
        format!(
            "{event_count},{true_dimension},{prior_name},{calibration_replicates},{},{:.6},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{}",
            self.samples,
            self.calibrated as f64 / total,
            self.outside_support,
            self.covered_95 as f64 / calibrated,
            coverage_low,
            coverage_high,
            self.correct_map as f64 / calibrated,
            self.true_probability_sum / calibrated,
            self.log_loss_sum / calibrated,
            self.brier_sum / brier_denominator,
            self.false_dimension_reduction,
            p95_low_dimension_mass,
        )
    }
}

fn wilson_interval(successes: usize, trials: usize) -> (f64, f64) {
    if trials == 0 {
        return (f64::NAN, f64::NAN);
    }
    let z = 1.959_963_984_540_054;
    let n = trials as f64;
    let observed = successes as f64 / n;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (observed + z2 / (2.0 * n)) / denominator;
    let half_width =
        z * (observed * (1.0 - observed) / n + z2 / (4.0 * n * n)).sqrt() / denominator;
    (
        (center - half_width).max(0.0),
        (center + half_width).min(1.0),
    )
}

fn prior(probabilities: [f64; 3]) -> [DimensionPrior; 3] {
    [
        DimensionPrior {
            spacetime_dimension: 2,
            probability: probabilities[0],
        },
        DimensionPrior {
            spacetime_dimension: 3,
            probability: probabilities[1],
        },
        DimensionPrior {
            spacetime_dimension: 4,
            probability: probabilities[2],
        },
    ]
}

fn main() -> Result<(), Box<dyn Error>> {
    const PROPER_TIME: f64 = 1.0;
    const CALIBRATION_BASE_SEED: u64 = 0xCA1B_2026;
    const QUALIFICATION_BASE_SEED: u64 = 0x51A7_2026;

    println!("Symthaea finite-size Bayesian dimension scale sweep");
    println!(
        "epistemic_status=WHOLE_SPRINKLING_MINKOWSKI_KDE_CALIBRATED_NOT_PHYSICAL_DIMENSION_REDUCTION_EVIDENCE"
    );
    println!("likelihood=empirical_KDE_of_ordering_fraction_not_iid_pair_Bernoulli");
    println!("calibration_unit=one_complete_fixed_cardinality_sprinkling");
    println!("calibration_and_qualification=disjoint_deterministic_seed_manifests");
    println!(
        "event_count,truth_dimension,prior_family,calibration_replicates,qualification_replicates,calibration_fraction,outside_support_count,credible_set_95_coverage_given_support,wilson95_coverage_low,wilson95_coverage_high,map_accuracy_given_support,mean_true_posterior_given_support,mean_log_loss_given_support,mean_brier_given_support,false_p_dim_le_3_count_d4,p95_p_dim_le_3_given_support"
    );

    for (size_index, event_count) in EVENT_COUNTS.iter().copied().enumerate() {
        let calibration_replicates = CALIBRATION_REPLICATES[size_index];
        let calibration_seed = CALIBRATION_BASE_SEED.wrapping_add((event_count as u64) << 20);
        let calibration = calibrate_minkowski_ordering_fraction(
            event_count,
            PROPER_TIME,
            &DIMENSIONS,
            calibration_replicates,
            calibration_seed,
        )?;

        for dimension in DIMENSIONS.iter().copied() {
            let mut aggregates: [Aggregate; 3] = std::array::from_fn(|_| Aggregate::default());
            for replicate in 0..QUALIFICATION_REPLICATES {
                let seed = QUALIFICATION_BASE_SEED
                    .wrapping_add((event_count as u64) << 32)
                    .wrapping_add((dimension as u64) << 24)
                    .wrapping_add(replicate as u64);
                let causet = CausalSetNd::sprinkle_fixed_cardinality(
                    seed,
                    event_count,
                    dimension,
                    PROPER_TIME,
                )?;
                let observed = causet
                    .ordering_fraction()
                    .ok_or("causal set did not contain enough events")?;

                for (prior_index, (_, probabilities)) in PRIORS.iter().enumerate() {
                    let posterior = estimate_finite_size_dimension_posterior(
                        event_count,
                        observed,
                        &calibration,
                        &prior(*probabilities),
                    )?;
                    aggregates[prior_index].record(dimension, &posterior);
                }
            }

            for (prior_index, (prior_name, _)) in PRIORS.iter().enumerate() {
                println!(
                    "{}",
                    aggregates[prior_index].line(
                        event_count,
                        dimension,
                        prior_name,
                        calibration_replicates,
                    )
                );
            }
        }
    }

    println!(
        "interpretation=false_dimension_reduction_count_counts_cases_where_true_D4_has_p_D_le_3_at_least_0.95"
    );
    println!(
        "non_claims=scale_dependent_quantum_dimension,curved_or_non_Poisson_manifolds,thermodynamic_limit,physical_quantum_gravity"
    );
    println!(
        "next_gate=posterior_coverage_and_out_of_family_abstention_across_event_counts_and_window_scales"
    );
    Ok(())
}
