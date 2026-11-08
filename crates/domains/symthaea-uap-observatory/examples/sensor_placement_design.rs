// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use symthaea_uap_observatory::{
    EcefVector3, RayObservation, SensorPlacementCandidate, SensorToEcefQuaternion,
    TargetHypothesis, rank_sensor_placements, rank_sensor_placements_over_target_region,
};

fn bearing_from_target(
    sensor_id: &str,
    origin_ecef_m: EcefVector3,
    target_ecef_m: EcefVector3,
    sigma_rad: f64,
) -> RayObservation {
    let displacement = target_ecef_m.subtract(origin_ecef_m);
    let horizontal = displacement.x.hypot(displacement.y);
    RayObservation {
        sensor_id: sensor_id.to_string(),
        origin_ecef_m,
        sensor_to_ecef: SensorToEcefQuaternion::IDENTITY,
        azimuth_rad: displacement.y.atan2(displacement.x),
        elevation_rad: displacement.z.atan2(horizontal),
        sigma_azimuth_rad: sigma_rad,
        sigma_elevation_rad: sigma_rad,
    }
}

fn candidate(id: &str, origin: EcefVector3, sigma_rad: f64) -> SensorPlacementCandidate {
    SensorPlacementCandidate {
        sensor_id: id.to_string(),
        origin_ecef_m: origin,
        sensor_to_ecef: SensorToEcefQuaternion::IDENTITY,
        sigma_azimuth_rad: sigma_rad,
        sigma_elevation_rad: sigma_rad,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Synthetic geometry, metres in an ECEF-like Cartesian frame.
    // The deliberately broad target hypotheses create a rank-reversal fixture:
    // a nominal one-point D-optimal choice is not the region-robust choice.
    let target = EcefVector3::new(68.9392511821, -41.3562356990, 177.1119045211);
    let sigma = 1.0e-4;
    let existing = vec![
        bearing_from_target(
            "existing-north",
            EcefVector3::new(0.0, 0.0, 0.0),
            target,
            sigma,
        ),
        bearing_from_target(
            "existing-east",
            EcefVector3::new(100.0, 0.0, 0.0),
            target,
            sigma,
        ),
    ];

    let candidates = vec![
        candidate(
            "candidate-north-far",
            EcefVector3::new(-100.0, 0.0, 0.0),
            sigma,
        ),
        candidate(
            "candidate-south",
            EcefVector3::new(0.0, 100.0, 0.0),
            sigma,
        ),
        candidate(
            "candidate-high",
            EcefVector3::new(0.0, 0.0, 100.0),
            sigma,
        ),
        candidate(
            "candidate-west-high",
            EcefVector3::new(-100.0, -100.0, 100.0),
            sigma,
        ),
        candidate(
            "candidate-east-high",
            EcefVector3::new(200.0, 0.0, 100.0),
            sigma,
        ),
        candidate(
            "candidate-far-south",
            EcefVector3::new(0.0, 300.0, 0.0),
            sigma,
        ),
        candidate(
            "candidate-far-north",
            EcefVector3::new(0.0, -300.0, 0.0),
            sigma,
        ),
        candidate(
            "candidate-up-far",
            EcefVector3::new(50.0, 40.0, 500.0),
            sigma,
        ),
    ];

    let nominal = rank_sensor_placements(&existing, target, &candidates)?;
    println!("epistemic_status=SYNTHETIC_LOCAL_FISHER_INFORMATION_DESIGN_NOT_DEPLOYMENT_ADVICE");
    println!("nominal_target_ecef_m=68.939251,-41.356236,177.111905");
    println!("angular_sigma_rad={sigma}");
    println!("nominal_candidate,logdet_information,a_optimal_trace_m2,condition_number");
    for score in &nominal {
        println!(
            "{},{:.9},{:.12e},{:.9}",
            score.sensor_id,
            score.log_determinant_information,
            score.a_optimal_variance_sum_m2,
            score.condition_number,
        );
    }

    // Four weighted target hypotheses. Existing bearings are reconstructed
    // from each sensor pose for every hypothesis, not copied from the nominal
    // target. All weights are positive and normalized by the library.
    let target_hypotheses = vec![
        TargetHypothesis {
            target_ecef_m: target,
            weight: 0.4,
        },
        TargetHypothesis {
            target_ecef_m: EcefVector3::new(36.4468708199, -140.9995554486, 525.5957184517),
            weight: 0.2,
        },
        TargetHypothesis {
            target_ecef_m: EcefVector3::new(-104.8126346103, -92.5351570022, 420.9570591495),
            weight: 0.2,
        },
        TargetHypothesis {
            target_ecef_m: EcefVector3::new(34.7476070368, 102.3108801094, 510.1874258118),
            weight: 0.2,
        },
    ];
    let robust =
        rank_sensor_placements_over_target_region(&existing, &target_hypotheses, &candidates)?;
    println!("epistemic_status=FINITE_SCENARIO_ROBUST_LOCAL_FIM_NOT_CONTINUOUS_REGION_GUARANTEE");
    println!("target_hypotheses=4");
    println!("robust_candidate,expected_logdet,worst_logdet,expected_trace_m2,worst_trace_m2,worst_condition");
    for score in &robust {
        println!(
            "{},{:.9},{:.9},{:.12e},{:.12e},{:.9}",
            score.sensor_id,
            score.expected_log_determinant_information,
            score.worst_case_log_determinant_information,
            score.expected_a_optimal_variance_sum_m2,
            score.worst_case_a_optimal_variance_sum_m2,
            score.worst_case_condition_number,
        );
    }

    let nominal_winner = nominal
        .first()
        .ok_or("nominal placement ranking returned no candidates")?;
    let robust_winner = robust
        .first()
        .ok_or("robust placement ranking returned no candidates")?;
    println!(
        "rank_reversal_fixture=nominal:{},robust:{}",
        nominal_winner.sensor_id, robust_winner.sensor_id
    );
    println!(
        "nominal_winner_remains_robust_winner={}",
        nominal_winner.sensor_id == robust_winner.sensor_id
    );
    Ok(())
}
