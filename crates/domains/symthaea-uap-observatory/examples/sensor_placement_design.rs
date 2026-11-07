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

fn main() -> Result<(), Box<dyn Error>> {
    // Synthetic geometry, metres in an ECEF-like Cartesian frame.
    // This example illustrates local design ranking, not a real deployment.
    let target = EcefVector3::new(50.0, 40.0, 200.0);
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
        SensorPlacementCandidate {
            sensor_id: "candidate-north-far".to_string(),
            origin_ecef_m: EcefVector3::new(-100.0, 0.0, 0.0),
            sensor_to_ecef: SensorToEcefQuaternion::IDENTITY,
            sigma_azimuth_rad: sigma,
            sigma_elevation_rad: sigma,
        },
        SensorPlacementCandidate {
            sensor_id: "candidate-south".to_string(),
            origin_ecef_m: EcefVector3::new(0.0, 100.0, 0.0),
            sensor_to_ecef: SensorToEcefQuaternion::IDENTITY,
            sigma_azimuth_rad: sigma,
            sigma_elevation_rad: sigma,
        },
        SensorPlacementCandidate {
            sensor_id: "candidate-high".to_string(),
            origin_ecef_m: EcefVector3::new(0.0, 0.0, 100.0),
            sensor_to_ecef: SensorToEcefQuaternion::IDENTITY,
            sigma_azimuth_rad: sigma,
            sigma_elevation_rad: sigma,
        },
    ];

    let ranking = rank_sensor_placements(&existing, target, &candidates)?;
    println!("epistemic_status=SYNTHETIC_LOCAL_FISHER_INFORMATION_DESIGN_NOT_DEPLOYMENT_ADVICE");
    println!("target_ecef_m=50,40,200");
    println!("angular_sigma_rad={sigma}");
    println!(
        "candidate,logdet_information,a_optimal_trace_m2,condition_number,cov_xx,cov_yy,cov_zz"
    );
    for score in ranking {
        let covariance = score.covariance_lower_bound_m2;
        println!(
            "{},{:.9},{:.12e},{:.9},{:.12e},{:.12e},{:.12e}",
            score.sensor_id,
            score.log_determinant_information,
            score.a_optimal_variance_sum_m2,
            score.condition_number,
            covariance[0][0],
            covariance[1][1],
            covariance[2][2],
        );
    }

    // A finite weighted region tests whether the nominally best sensor stays
    // useful when target position is uncertain. Existing bearings are
    // re-predicted from each sensor pose at every hypothesis.
    let target_hypotheses = vec![
        TargetHypothesis {
            target_ecef_m: target,
            weight: 0.5,
        },
        TargetHypothesis {
            target_ecef_m: EcefVector3::new(70.0, 30.0, 180.0),
            weight: 0.3,
        },
        TargetHypothesis {
            target_ecef_m: EcefVector3::new(35.0, 55.0, 230.0),
            weight: 0.2,
        },
    ];
    let robust_ranking =
        rank_sensor_placements_over_target_region(&existing, &target_hypotheses, &candidates)?;
    println!("epistemic_status=FINITE_SCENARIO_ROBUST_LOCAL_FIM_NOT_CONTINUOUS_REGION_GUARANTEE");
    println!("target_hypotheses=3");
    println!(
        "candidate,expected_logdet,worst_logdet,expected_trace_m2,worst_trace_m2,worst_condition"
    );
    for score in robust_ranking {
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

    Ok(())
}
