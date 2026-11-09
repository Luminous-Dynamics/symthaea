// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use symthaea_causal_set_lab::{
    CausalSet2D, CausalSetNd, JointCausalSetSignature,
};

fn print_signature(
    family: &str,
    actual_dimension: usize,
    signature: &JointCausalSetSignature,
) -> Result<(), Box<dyn Error>> {
    println!(
        "family={family},actual_dimension={actual_dimension},N={},longest_chain={},profile_bins={}",
        signature.event_count,
        signature.longest_chain_length,
        signature.profile_bin_count
    );
    println!("family={family},profile_probabilities={}", signature
        .normalized_interval_profile
        .iter()
        .map(|value| format!("{value:.12}"))
        .collect::<Vec<_>>()
        .join(";"));

    for candidate_dimension in 2..=4 {
        let features = signature.features_for_dimension(candidate_dimension)?;
        let scaled_chain = features[features.len() - 1];
        println!(
            "family={family},candidate_dimension={candidate_dimension},scaled_longest_chain={scaled_chain:.12}"
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    const EVENT_COUNT: usize = 512;
    const PROFILE_BINS: usize = 8;
    const PROPER_TIME: f64 = 1.0;

    println!(
        "epistemic_status=SYNTHETIC_JOINT_CAUSAL_SET_SIGNATURE_NOT_GEOMETRY_INFERENCE"
    );
    println!("event_count={EVENT_COUNT}");
    println!("profile_bins={PROFILE_BINS}");
    println!("features=normalized_interval_profile_plus_dimension_conditioned_longest_chain");
    println!("not_included=ordering_fraction_or_BDG_bracket_as_independent_features");

    let flat_1p1 = CausalSet2D::sprinkle_fixed_cardinality(0x51A7, EVENT_COUNT, PROPER_TIME)?;
    let signature_1p1 =
        JointCausalSetSignature::from_causal_set_1p1(&flat_1p1, PROFILE_BINS)?;
    print_signature("minkowski-1p1", 2, &signature_1p1)?;

    for dimension in [3_usize, 4_usize] {
        let flat_nd = CausalSetNd::sprinkle_fixed_cardinality(
            0xC0FFEE + dimension as u64,
            EVENT_COUNT,
            dimension,
            PROPER_TIME,
        )?;
        let signature =
            JointCausalSetSignature::from_causal_set_nd(&flat_nd, PROFILE_BINS)?;
        print_signature(
            &format!("minkowski-{dimension}d"),
            dimension,
            &signature,
        )?;
    }

    println!(
        "claim_boundary=feature_extraction_only_joint_likelihood_requires_whole_sprinkling_calibration"
    );
    Ok(())
}
