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
        if posterior.status == DimensionInferenceStatus::CalibratedPosterior {
            self.calibrated += 1;
        }
        if posterior.credible_set_95.contains(&true_dimension) {
            self.covered_95 += 1;
        }

        let true_probability = posterior
            .dimensions
            .iter()
            .find(|entry| entry.spacetime_dimension == true_dimension)
            .and_then(|entry| entry.posterior_probability)
            .unwrap_or(0.0);
        self.true_probability_sum += true_probability;
        self.log_loss_sum -= true_probability.max(1.0e-15).ln();

        let predicted_map = posterior
            .dimensions
            .iter()
            .filter_map(|entry| entry.posterior_probability.map(|p| (entry.spacetime_dimension, p)))
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .map(|entry| entry.0);
        if predicted_map == Some(true_dimension) {
            self.correct_map += 1;
        }

        for entry in &posterior.dimensions {
            let target = f64::from(entry.spacetime_dimension == true_dimension);
            let probability = entry.posterior_probability.unwrap_or(0.0);
            self.brier_sum += (probability - target).powi(2);
        }
        if true_dimension == 4 {
            let p_low = posterior
                .posterior_probability_dimension_at_most_three
                .unwrap_or(1.0);
            self.p_dim_le_3_values.push(p_low);
            if p_low >= 0.95 {
                self.false_dimension_reduction += 1;
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
        let n = self.samples.max(1) as f64;
        let brier_denominator = n * DIMENSIONS.len() as f64;
        format!(
            "{event_count},{true_dimension},{prior_name},{calibration_replicates},{},{:.6},{:.9},{:.9},{:.9},{:.9},{:.9},{},{}",
            self.samples,
            self.calibrated as f64 / n,
            self.covered_95 as f64 / n,
            self.correct_map as f64 / n,
            self.true_probability_sum / n,
            self.log_loss_sum / n,
            self.brier_sum / brier_denominator,
            self.false_dimension_reduction,
            p95_low_dimension_mass,
        )
    }
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
    println!("epistemic_status=WHOLE_SPRINKLING_MINKOWSKI_KDE_CALIBRATED_NOT_PHYSICAL_DIMENSION_REDUCTION_EVIDENCE");
    println!("likelihood=empirical_KDE_of_ordering_fraction_not_iid_pair_Bernoulli");
    println!("calibration_unit=one_complete_fixed_cardinality_sprinkling");
    println!("calibration_and_qualification=disjoint_deterministic_seed_manifests");
    println!("event_count,truth_dimension,prior_family,calibration_replicates,qualification_replicates,calibration_fraction,credible_set_95_coverage,map_accuracy,mean_true_posterior,mean_log_loss,mean_brier,false_p_dim_le_3_count,p95_p_dim_le_3");
    
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

        for (dimension_index, dimension) in DIMENSIONS.iter().copied().enumerate() {
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

            // Keep the compiler from optimizing away the deliberate dimension
            // indexing contract; all three named priors are emitted per truth.
            let _ = dimension_index;
        }
    }

    println!("interpretation=false_dimension_reduction_count_counts_cases_where_true_D4_has_p_D_le_3_at_least_0.95");
    println!("non_claims=scale_dependent_quantum_dimension,curved_or_non_Poisson_manifolds,thermodynamic_limit,physical_quantum_gravity");
    println!("next_gate=posterior_coverage_and_out_of_family_abstention_across_event_counts_and_window_scales");
    Ok(())
}
