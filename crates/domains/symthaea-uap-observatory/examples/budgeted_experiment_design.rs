// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use symthaea_uap_observatory::{
    BudgetedDesignConfig, DesignScenario, EcefVector3, ExperimentOption, SensorPlacementCandidate,
    SensorToEcefQuaternion, bearing_position_clock_jacobian, exhaustive_budgeted_design,
    greedy_budgeted_design,
};

#[derive(Clone, Copy)]
struct TargetScenario {
    position_ecef_m: EcefVector3,
    velocity_ecef_mps: EcefVector3,
}

fn bearing_option(
    option_id: &str,
    cost: f64,
    exclusive_group: Option<&str>,
    sensor: SensorPlacementCandidate,
    sensor_velocity_by_scenario: &[EcefVector3],
    targets: &[TargetScenario],
) -> Result<ExperimentOption, Box<dyn Error>> {
    if sensor_velocity_by_scenario.len() != targets.len() {
        return Err("sensor and target scenario counts differ".into());
    }

    let mut jacobian_rows_by_scenario = Vec::with_capacity(targets.len());
    for (target, sensor_velocity) in targets.iter().zip(sensor_velocity_by_scenario) {
        let relative_velocity = target.velocity_ecef_mps.subtract(*sensor_velocity);
        let jacobian =
            bearing_position_clock_jacobian(&sensor, target.position_ecef_m, relative_velocity)?;
        jacobian_rows_by_scenario.push(vec![
            jacobian.azimuth_row.to_vec(),
            jacobian.elevation_row.to_vec(),
        ]);
    }

    Ok(ExperimentOption {
        option_id: option_id.to_string(),
        cost,
        exclusive_group: exclusive_group.map(str::to_string),
        jacobian_rows_by_scenario,
    })
}

fn sensor(
    id: &str,
    origin: EcefVector3,
    sigma_azimuth_rad: f64,
    sigma_elevation_rad: f64,
) -> SensorPlacementCandidate {
    SensorPlacementCandidate {
        sensor_id: id.to_string(),
        origin_ecef_m: origin,
        sensor_to_ecef: SensorToEcefQuaternion::IDENTITY,
        sigma_azimuth_rad,
        sigma_elevation_rad,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Linearization variables: target ECEF XYZ plus an additive sensor-clock
    // offset. The observation rows below are generated from the actual bearing
    // geometry, declared noise and relative target/sensor velocities. This is
    // synthetic design data, not a real Observatory deployment.
    let scenarios = vec![
        DesignScenario {
            scenario_id: "nominal-target-region".to_string(),
            target_parameter_count: 3,
            weight: 0.65,
            prior_precision: vec![
                vec![0.01, 0.0, 0.0, 0.0],
                vec![0.0, 0.01, 0.0, 0.0],
                vec![0.0, 0.0, 0.01, 0.0],
                vec![0.0, 0.0, 0.0, 0.001],
            ],
        },
        DesignScenario {
            scenario_id: "high-altitude-target-region".to_string(),
            target_parameter_count: 3,
            weight: 0.35,
            prior_precision: vec![
                vec![0.01, 0.0, 0.0, 0.0],
                vec![0.0, 0.01, 0.0, 0.0],
                vec![0.0, 0.0, 0.01, 0.0],
                vec![0.0, 0.0, 0.0, 0.001],
            ],
        },
    ];
    let targets = [
        TargetScenario {
            position_ecef_m: EcefVector3::new(150.0, 80.0, 220.0),
            velocity_ecef_mps: EcefVector3::new(14.0, -5.0, 1.0),
        },
        TargetScenario {
            position_ecef_m: EcefVector3::new(-80.0, 180.0, 420.0),
            velocity_ecef_mps: EcefVector3::new(8.0, 6.0, -0.5),
        },
    ];

    let station_a_low = sensor(
        "station-a-low-fidelity",
        EcefVector3::new(0.0, 0.0, 0.0),
        0.010,
        0.015,
    );
    let station_a_high = sensor(
        "station-a-high-fidelity",
        EcefVector3::new(0.0, 0.0, 0.0),
        0.003,
        0.004,
    );
    let station_b = sensor(
        "station-b-baseline",
        EcefVector3::new(220.0, -40.0, 15.0),
        0.008,
        0.010,
    );
    let manoeuvre_station = sensor(
        "platform-excitation-manoeuvre",
        EcefVector3::new(-120.0, 260.0, 125.0),
        0.009,
        0.012,
    );

    let options = vec![
        bearing_option(
            "station-a-low-fidelity",
            1.0,
            Some("station-a"),
            station_a_low,
            &[EcefVector3::ZERO, EcefVector3::ZERO],
            &targets,
        )?,
        bearing_option(
            "station-a-high-fidelity",
            2.2,
            Some("station-a"),
            station_a_high,
            &[EcefVector3::ZERO, EcefVector3::ZERO],
            &targets,
        )?,
        bearing_option(
            "station-b-baseline",
            1.25,
            None,
            station_b,
            &[EcefVector3::ZERO, EcefVector3::ZERO],
            &targets,
        )?,
        ExperimentOption {
            option_id: "clock-calibration-pulse".to_string(),
            cost: 0.65,
            exclusive_group: Some("clock-reference".to_string()),
            jacobian_rows_by_scenario: vec![
                vec![vec![0.0, 0.0, 0.0, 1.0 / 0.002]],
                vec![vec![0.0, 0.0, 0.0, 1.0 / 0.002]],
            ],
        },
        bearing_option(
            "platform-excitation-manoeuvre",
            1.1,
            None,
            manoeuvre_station,
            &[
                EcefVector3::new(-24.0, 8.0, 0.5),
                EcefVector3::new(-18.0, 5.0, -1.0),
            ],
            &targets,
        )?,
    ];

    let config = BudgetedDesignConfig {
        budget: 3.0,
        parameter_scales: vec![100.0, 100.0, 100.0, 0.01],
    };

    let greedy = greedy_budgeted_design(&scenarios, &options, &config)?;
    let exact = exhaustive_budgeted_design(&scenarios, &options, &config)?;

    println!("epistemic_status=SYNTHETIC_GEOMETRY_DERIVED_BUDGETED_DESIGN_NOT_DEPLOYMENT_ADVICE");
    println!("state=target_x_m,target_y_m,target_z_m,clock_offset_s");
    println!("jacobian_source=analytic_azimuth_elevation_geometry_whitened_by_sigma");
    println!("budget={:.6}", config.budget);
    println!("candidate_options={}", options.len());
    println!("exclusive_fidelity_groups=station-a,clock-reference");
    println!("greedy_selected={}", greedy.selected_option_ids.join(";"));
    println!("greedy_cost={:.6}", greedy.total_cost);
    println!(
        "greedy_expected_target_eig_nats={:.9}",
        greedy.expected_information_gain_nats
    );
    println!(
        "greedy_expected_joint_eig_nats={:.9}",
        greedy.expected_joint_information_gain_nats
    );
    println!(
        "greedy_worst_case_eig_nats={:.9}",
        greedy.worst_case_information_gain_nats
    );
    println!("exact_selected={}", exact.selected_option_ids.join(";"));
    println!("exact_cost={:.6}", exact.total_cost);
    println!(
        "exact_expected_target_eig_nats={:.9}",
        exact.expected_information_gain_nats
    );
    println!(
        "exact_expected_joint_eig_nats={:.9}",
        exact.expected_joint_information_gain_nats
    );
    println!(
        "greedy_target_optimality_gap_nats={:.9}",
        exact.expected_information_gain_nats - greedy.expected_information_gain_nats
    );
    println!(
        "greedy_joint_optimality_gap_nats={:.9}",
        exact.expected_joint_information_gain_nats - greedy.expected_joint_information_gain_nats
    );
    println!("selection_step,option_id,cost,marginal_expected_eig,marginal_eig_per_cost");
    for (index, step) in greedy.steps.iter().enumerate() {
        println!(
            "{index},{},{:.6},{:.9},{:.9}",
            step.option_id,
            step.cost,
            step.marginal_expected_gain_nats,
            step.marginal_gain_per_cost,
        );
    }
    println!("scenario,information_gain_nats,data_only_rank,data_only_nullity,condition_number");
    for score in &greedy.scenario_scores {
        println!(
            "{},{:.9},{},{},{}",
            score.scenario_id,
            score.information_gain_nats,
            score.data_only_rank,
            score.data_only_nullity,
            score
                .data_only_condition_number
                .map(|value| format!("{value:.9}"))
                .unwrap_or_else(|| "unobservable".to_string()),
        );
    }
    println!(
        "claim_boundary=linearized_geometry_and_relative_velocity; shared_cross_sensor_covariance_not_included"
    );
    Ok(())
}
