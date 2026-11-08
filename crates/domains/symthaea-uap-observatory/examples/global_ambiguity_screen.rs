// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use symthaea_uap_observatory::{
    CandidateSolution, GlobalIdentifiabilityConfig, analyze_jacobian,
    screen_global_ambiguities,
};

#[derive(Clone, Copy)]
struct Point3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Point3 {
    const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

fn distance(left: Point3, right: Point3) -> f64 {
    (left.x - right.x)
        .hypot(left.y - right.y)
        .hypot(left.z - right.z)
}

fn predict_ranges(target: Point3, anchors: &[Point3]) -> Vec<f64> {
    anchors
        .iter()
        .map(|anchor| distance(target, *anchor))
        .collect()
}

fn candidate(id: &str, target: Point3, anchors: &[Point3]) -> CandidateSolution {
    CandidateSolution {
        solution_id: id.to_string(),
        parameters: vec![target.x, target.y, target.z],
        predicted_observations: predict_ranges(target, anchors),
    }
}

fn config(observation_count: usize) -> GlobalIdentifiabilityConfig {
    GlobalIdentifiabilityConfig {
        parameter_scales: vec![1.0, 1.0, 1.0],
        observation_sigmas: vec![0.01; observation_count],
        max_rms_standardized_residual: 3.0,
        max_abs_standardized_residual: 3.0,
        max_rms_prediction_difference: 3.0,
        max_abs_prediction_difference: 3.0,
        min_scaled_parameter_separation: 1.0,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let target = Point3::new(2.0, 3.0, 4.0);
    let mirror = Point3::new(2.0, 3.0, -4.0);
    let planar_anchors = [
        Point3::new(-1.0, 0.0, 0.0),
        Point3::new(5.0, 0.0, 0.0),
        Point3::new(0.0, 7.0, 0.0),
    ];
    let observations = predict_ranges(target, &planar_anchors);
    let candidates = [
        candidate("target-above-plane", target, &planar_anchors),
        candidate("mirror-below-plane", mirror, &planar_anchors),
    ];

    // The local distance Jacobian has full column rank at the target even
    // though the reflected solution reproduces all three ranges exactly.
    let local_jacobian: Vec<Vec<f64>> = planar_anchors
        .iter()
        .map(|anchor| {
            let dx = target.x - anchor.x;
            let dy = target.y - anchor.y;
            let dz = target.z - anchor.z;
            let range = dx.hypot(dy).hypot(dz);
            vec![dx / range, dy / range, dz / range]
        })
        .collect();
    let local = analyze_jacobian(&local_jacobian, &[1.0, 1.0, 1.0], 1.0e-10)?;
    let global = screen_global_ambiguities(&observations, &candidates, &config(3))?;
    println!("epistemic_status=KNOWN_ANSWER_DISCRETE_AMBIGUITY_REGRESSION_NOT_DEPLOYMENT_RESULT");
    println!("fixture=three_non_collinear_range_anchors_in_z0");
    println!("local_jacobian_rank={}", local.numerical_rank);
    println!("local_jacobian_parameter_count={}", local.parameter_count);
    println!("admissible_candidate_count={}", global.admissible_candidate_count);
    println!("ambiguous_pair_count={}", global.ambiguous_pairs.len());
    for pair in &global.ambiguous_pairs {
        println!(
            "ambiguous_pair={},{};scaled_separation={:.6};prediction_rms_sigma={:.6}",
            pair.left_solution_id,
            pair.right_solution_id,
            pair.scaled_parameter_separation,
            pair.rms_standardized_prediction_difference,
        );
    }

    let spatially_anchored = [
        planar_anchors[0],
        planar_anchors[1],
        planar_anchors[2],
        Point3::new(0.0, 0.0, 3.0),
    ];
    let anchored_observations = predict_ranges(target, &spatially_anchored);
    let anchored_candidates = [
        candidate("target-above-plane", target, &spatially_anchored),
        candidate("mirror-below-plane", mirror, &spatially_anchored),
    ];
    let anchored = screen_global_ambiguities(
        &anchored_observations,
        &anchored_candidates,
        &config(4),
    )?;
    println!("fixture=one_non_coplanar_range_anchor_added");
    println!("admissible_candidate_count={}", anchored.admissible_candidate_count);
    println!("ambiguous_pair_count={}", anchored.ambiguous_pairs.len());

    println!("near_planar_anchor_height_m,admissible_candidates,ambiguous_pairs");
    for height in [0.0_f64, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10] {
        let anchors = [
            planar_anchors[0],
            planar_anchors[1],
            planar_anchors[2],
            Point3::new(0.0, 0.0, height),
        ];
        let measured = predict_ranges(target, &anchors);
        let alternatives = [
            candidate("target", target, &anchors),
            candidate("mirror", mirror, &anchors),
        ];
        let report = screen_global_ambiguities(&measured, &alternatives, &config(4))?;
        println!(
            "{height:.6},{},{}",
            report.admissible_candidate_count,
            report.ambiguous_pairs.len()
        );
    }

    println!("A full-rank local Jacobian does not prove global uniqueness.");
    println!("No supplied ambiguity found is not an exhaustive uniqueness proof.");
    Ok(())
}
