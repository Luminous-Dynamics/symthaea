// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use symthaea_causal_set_lab::{
    CausalSet2D, CausalSetNd, JointCausalSetSignature, JointDimensionModel,
    JointGaussianCalibration, estimate_dimension_from_ordering_fraction, infer_joint_dimension,
};

const EVENT_COUNT: usize = 128;
const PROFILE_BINS: usize = 4;
const CALIBRATION_REPLICATES: usize = 64;
const QUALIFICATION_REPLICATES: usize = 24;
const DIAGONAL_SHRINKAGE: f64 = 0.15;
const RELATIVE_RIDGE: f64 = 1.0e-5;

fn signature_for(
    seed: u64,
    dimension: usize,
) -> Result<(JointCausalSetSignature, f64), Box<dyn Error>> {
    match dimension {
        2 => {
            let causet = CausalSet2D::sprinkle_fixed_cardinality(seed, EVENT_COUNT, 1.0)?;
            let ordering_fraction = causet.ordering_fraction().ok_or("missing ordering fraction")?;
            let scalar = estimate_dimension_from_ordering_fraction(ordering_fraction, 8.0)
                .unwrap_or(f64::NAN);
            let signature =
                JointCausalSetSignature::from_causal_set_1p1(&causet, PROFILE_BINS)?;
            Ok((signature, scalar))
        }
        3 | 4 => {
            let causet =
                CausalSetNd::sprinkle_fixed_cardinality(seed, EVENT_COUNT, dimension, 1.0)?;
            let ordering_fraction = causet.ordering_fraction().ok_or("missing ordering fraction")?;
            let scalar = estimate_dimension_from_ordering_fraction(ordering_fraction, 8.0)
                .unwrap_or(f64::NAN);
            let signature =
                JointCausalSetSignature::from_causal_set_nd(&causet, PROFILE_BINS)?;
            Ok((signature, scalar))
        }
        _ => Err("pilot supports D=2,3,4 only".into()),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Symthaea whole-sprinkling joint synthetic-likelihood pilot");
    println!("epistemic_status=GAUSSIAN_SYNTHETIC_LIKELIHOOD_PILOT_NOT_COVERAGE_QUALIFIED");
    println!("calibration_unit=one_complete_independent_fixed_cardinality_sprinkling");
    println!("features=ILR(normalized_interval_profile)+dimension_conditioned_longest_chain");
    println!("anti_double_counting=ordering_fraction_and_BDG_bracket_not_added_as_independent_features");
    println!("event_count={EVENT_COUNT},profile_bins={PROFILE_BINS}");
    println!("calibration_replicates={CALIBRATION_REPLICATES},qualification_replicates={QUALIFICATION_REPLICATES}");
    println!("diagonal_shrinkage={DIAGONAL_SHRINKAGE},relative_ridge={RELATIVE_RIDGE}");
    println!("true_dimension,replicate,seed,joint_map,joint_p_d2,joint_p_d3,joint_p_d4,joint_entropy,MM_dimension,MM_abs_error");

    let mut models = Vec::new();
    for dimension in 2..=4 {
        let mut calibration_samples = Vec::with_capacity(CALIBRATION_REPLICATES);
        for replicate in 0..CALIBRATION_REPLICATES {
            let seed = 0xC411_BA7E_u64
                .wrapping_add((dimension as u64) << 32)
                .wrapping_add(replicate as u64);
            let (signature, _) = signature_for(seed, dimension)?;
            calibration_samples.push(signature.ilr_features_for_dimension(dimension)?);
        }

        let calibration = JointGaussianCalibration::fit_whole_sprinkling_features(
            dimension,
            &calibration_samples,
            DIAGONAL_SHRINKAGE,
            RELATIVE_RIDGE,
        )?;
        models.push(JointDimensionModel {
            spacetime_dimension: dimension,
            prior_probability: 1.0 / 3.0,
            calibration,
        });
    }

    let mut confusion = [[0_usize; 3]; 3];
    let mut correct = 0_usize;
    let mut total = 0_usize;
    let mut log_loss = 0.0;
    let mut entropy_sum = 0.0;
    let mut scalar_error_sum = 0.0;
    let mut scalar_error_count = 0_usize;

    for true_dimension in 2..=4 {
        for replicate in 0..QUALIFICATION_REPLICATES {
            let seed = 0x9A11_DA7A_u64
                .wrapping_add((true_dimension as u64) << 32)
                .wrapping_add(replicate as u64);
            let (signature, scalar_dimension) = signature_for(seed, true_dimension)?;
            let posterior = infer_joint_dimension(&signature, &models)?;
            let prediction = posterior.map_dimension;
            confusion[true_dimension - 2][prediction - 2] += 1;
            correct += usize::from(prediction == true_dimension);
            total += 1;
            entropy_sum += posterior.entropy_nats;

            let true_probability = posterior
                .probabilities
                .iter()
                .find(|entry| entry.spacetime_dimension == true_dimension)
                .map(|entry| entry.probability)
                .ok_or("posterior omitted true dimension")?;
            log_loss -= true_probability.max(1.0e-15).ln();

            let scalar_error = if scalar_dimension.is_finite() {
                let error = (scalar_dimension - true_dimension as f64).abs();
                scalar_error_sum += error;
                scalar_error_count += 1;
                format!("{error:.6}")
            } else {
                "out_of_scalar_support".to_string()
            };

            let mut probabilities = [0.0_f64; 3];
            for entry in &posterior.probabilities {
                probabilities[entry.spacetime_dimension - 2] = entry.probability;
            }
            println!(
                "{true_dimension},{replicate},{seed},{prediction},{:.9},{:.9},{:.9},{:.9},{scalar_dimension:.9},{scalar_error}",
                probabilities[0],
                probabilities[1],
                probabilities[2],
                posterior.entropy_nats,
            );
        }
    }

    let accuracy = correct as f64 / total as f64;
    let mean_log_loss = log_loss / total as f64;
    let mean_entropy = entropy_sum / total as f64;
    let mean_scalar_error = if scalar_error_count > 0 {
        scalar_error_sum / scalar_error_count as f64
    } else {
        f64::NAN
    };

    println!("confusion_matrix_rows=true_D2_D3_D4");
    for row in confusion {
        println!("confusion={},{},{}", row[0], row[1], row[2]);
    }
    println!("joint_accuracy={accuracy:.9}");
    println!("joint_mean_log_loss={mean_log_loss:.9}");
    println!("joint_mean_posterior_entropy_nats={mean_entropy:.9}");
    println!("MM_mean_absolute_error_on_supported_cases={mean_scalar_error:.9}");
    println!("calibration_seed_range=0xC411BA7E_by_dimension_and_replicate");
    println!("qualification_seed_range=0x9A11DA7A_by_dimension_and_replicate");
    println!("claim_boundary=finite_simulation_pilot; posterior_coverage_and_out_of_family_abstention_not_yet_qualified");
    Ok(())
}
