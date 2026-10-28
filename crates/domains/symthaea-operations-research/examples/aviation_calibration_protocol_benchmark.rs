// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Synthetic contract-composition example for aviation calibration and robust
//! recovery. This is not a real-data benchmark and does not establish savings.

use std::fmt::Write as _;
use std::process;

use symthaea_operations_research::aviation_calibration_protocol::{
    audit_no_test_set_tuning, audit_split_integrity, EvaluationPartition,
    SplitAuditStatus, SplitIntegrityPolicy, SplitRecord, TuningChoice,
};
use symthaea_operations_research::aviation_distributional_robustness::{
    evaluate_tv_robustness_curve, evaluate_tv_robust_cost_delta, ScenarioCostDelta,
};
use symthaea_operations_research::aviation_probability_calibration::{
    calibrate_tv_radius_from_episodes, CalibrationEpisode, NominalScenarioProbability,
    RegimeNominalDistribution,
};
use symthaea_operations_research::aviation_online_drift::{
    DriftAlternative, DriftMonitorStatus, DriftScenarioProbability, OnlineDriftMonitor,
};

fn main() {
    let mut records = Vec::new();
    for (partition, epoch) in [
        (EvaluationPartition::Training, 1_700_000_000_i64),
        (EvaluationPartition::Validation, 1_700_020_000_i64),
        (EvaluationPartition::Test, 1_700_030_000_i64),
    ] {
        for index in 0..3 {
            records.push(SplitRecord {
                record_id: format!("{partition:?}-row-{index}"),
                event_cluster_id: format!("{partition:?}-episode-{index}"),
                event_time_epoch_seconds: Some(epoch + index),
                partition,
            });
        }
    }

    let regime_counts: [(&str, [(&str, u64); 3]); 2] = [
        ("nominal-weather", [("routine", 60), ("capacity", 30), ("compound-weather", 10)]),
        ("disruption-heavy", [("routine", 40), ("capacity", 30), ("compound-weather", 30)]),
    ];
    let nominal_distributions: Vec<RegimeNominalDistribution> = regime_counts
        .iter()
        .map(|(regime_id, _)| RegimeNominalDistribution {
            regime_id: (*regime_id).into(),
            probabilities: vec![
                NominalScenarioProbability { scenario_id: "routine".into(), probability_ppm: 600_000 },
                NominalScenarioProbability { scenario_id: "capacity".into(), probability_ppm: 300_000 },
                NominalScenarioProbability { scenario_id: "compound-weather".into(), probability_ppm: 100_000 },
            ],
        })
        .collect();

    let mut calibration_episodes = Vec::new();
    let mut calibration_index = 0_u64;
    for (regime_id, outcomes) in regime_counts {
        for (scenario_id, count) in outcomes {
            for _ in 0..count {
                let record_id = format!("calibration-record-{calibration_index}");
                let event_cluster_id = format!("calibration-episode-{calibration_index}");
                let event_time = 1_700_010_000_i64 + calibration_index as i64;
                calibration_episodes.push(CalibrationEpisode {
                    record_id: record_id.clone(),
                    event_cluster_id: event_cluster_id.clone(),
                    regime_id: regime_id.into(),
                    observed_scenario_id: scenario_id.into(),
                    partition: EvaluationPartition::Calibration,
                });
                records.push(SplitRecord {
                    record_id,
                    event_cluster_id,
                    event_time_epoch_seconds: Some(event_time),
                    partition: EvaluationPartition::Calibration,
                });
                calibration_index += 1;
            }
        }
    }

    let split = audit_split_integrity(
        &records,
        SplitIntegrityPolicy {
            minimum_clusters_per_partition: 3,
            require_strict_chronological_order: true,
        },
    );
    let tuning = audit_no_test_set_tuning(&[
        TuningChoice {
            artifact_id: "tv_radius_grid".into(),
            selected_using: EvaluationPartition::Validation,
        },
        TuningChoice {
            artifact_id: "robust_cost_threshold".into(),
            selected_using: EvaluationPartition::Validation,
        },
    ]);

    if split.status != SplitAuditStatus::Accept || tuning.status != SplitAuditStatus::Accept {
        eprintln!(
            "benchmark contract failed: split={:?}, tuning={:?}",
            split.status, tuning.status
        );
        process::exit(1);
    }

    let calibration = match calibrate_tv_radius_from_episodes(
        &calibration_episodes,
        &nominal_distributions,
        50_000,
    ) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("TV-radius episode calibration failed: {error:?}");
            process::exit(1);
        }
    };

    let scenarios = [
        ScenarioCostDelta {
            scenario_id: "routine".into(),
            nominal_probability_ppm: 600_000,
            candidate_minus_baseline_cost_units: -100,
        },
        ScenarioCostDelta {
            scenario_id: "capacity".into(),
            nominal_probability_ppm: 300_000,
            candidate_minus_baseline_cost_units: 0,
        },
        ScenarioCostDelta {
            scenario_id: "compound-weather".into(),
            nominal_probability_ppm: 100_000,
            candidate_minus_baseline_cost_units: 20,
        },
    ];
    let curve = match evaluate_tv_robustness_curve(
        &scenarios,
        &[0, 200_000, 400_000, 500_000, 1_000_000],
        0,
    ) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("robustness evaluation failed: {error:?}");
            process::exit(1);
        }
    };

    let calibrated_report = match evaluate_tv_robust_cost_delta(
        &scenarios,
        calibration.maximum_required_tv_radius_ppm,
    ) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("calibrated-radius evaluation failed: {error:?}");
            process::exit(1);
        }
    };

    let mut drift_monitor = match OnlineDriftMonitor::new(
        vec![
            DriftScenarioProbability { scenario_id: "routine".into(), probability_ppm: 600_000 },
            DriftScenarioProbability { scenario_id: "capacity".into(), probability_ppm: 300_000 },
            DriftScenarioProbability { scenario_id: "compound-weather".into(), probability_ppm: 100_000 },
        ],
        vec![DriftAlternative {
            alternative_id: "disruption-heavy".into(),
            mixture_weight_ppm: 1_000_000,
            probabilities: vec![
                DriftScenarioProbability { scenario_id: "routine".into(), probability_ppm: 400_000 },
                DriftScenarioProbability { scenario_id: "capacity".into(), probability_ppm: 300_000 },
                DriftScenarioProbability { scenario_id: "compound-weather".into(), probability_ppm: 300_000 },
            ],
        }],
        50_000,
    ) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("online drift monitor initialization failed: {error:?}");
            process::exit(1);
        }
    };
    let mut drift_report = drift_monitor.report();
    for (index, scenario_id) in [
        "compound-weather",
        "compound-weather",
        "compound-weather",
    ].into_iter().enumerate() {
        drift_report = match drift_monitor.observe(
            &format!("stream-event-{index}"),
            1_700_040_000_i64 + index as i64,
            scenario_id,
        ) {
            Ok(value) => value,
            Err(error) => {
                eprintln!("online drift monitor update failed: {error:?}");
                process::exit(1);
            }
        };
        if drift_report.status != DriftMonitorStatus::InControl {
            break;
        }
    }

    let mut json = String::new();
    writeln!(
        &mut json,
        "{{\n  \"benchmark\": \"aviation-calibration-robustness-contract\",\n  \"contract_status\": \"ACCEPTED\",\n  \"decision_status\": \"EVALUATED_NOT_SELECTED\",\n  \"claim_ceiling\": \"synthetic fixtures only; policy performance is not certified\",\n  \"split_audit\": {{\n    \"status\": \"{:?}\",\n    \"partitions\": [",
        split.status
    ).expect("write to String");
    for (index, stats) in split.partitions.iter().enumerate() {
        let comma = if index + 1 == split.partitions.len() { "" } else { "," };
        writeln!(
            &mut json,
            "      {{\"partition\": \"{:?}\", \"rows\": {}, \"event_clusters\": {}, \"earliest_epoch_seconds\": {}, \"latest_epoch_seconds\": {}}}{}",
            stats.partition,
            stats.row_count,
            stats.distinct_event_clusters,
            stats.earliest_epoch_seconds.map_or("null".to_string(), |v| v.to_string()),
            stats.latest_epoch_seconds.map_or("null".to_string(), |v| v.to_string()),
            comma
        ).expect("write to String");
    }
    writeln!(
        &mut json,
        "    ]\n  }},\n  \"tuning_audit\": {{\"status\": \"{:?}\", \"tuned_on_test\": false}},\n  \"online_drift_monitor\": {{\"status\": \"{:?}\", \"evidence\": \"{:?}\", \"observations_seen\": {}, \"log_e_value\": {}}},\n  \"radius_calibration\": {{\"regime_count\": {}, \"family_failure_probability_ppm\": {}, \"per_regime_failure_probability_ppm\": {}, \"maximum_required_tv_radius_ppm\": {}, \"any_regime_capped_at_maximum\": {}, \"regimes\": [",
        tuning.status,
        drift_report.status,
        drift_report.evidence,
        drift_report.observations_seen,
        drift_report.log_e_value,
        calibration.regime_count,
        calibration.requested_family_failure_probability_ppm,
        calibration.per_regime_failure_probability_ppm,
        calibration.maximum_required_tv_radius_ppm,
        calibration.any_regime_capped_at_maximum_radius
    ).expect("write to String");
    for (index, regime) in calibration.regimes.iter().enumerate() {
        let comma = if index + 1 == calibration.regimes.len() { "" } else { "," };
        writeln!(
            &mut json,
            "    {{\"regime_id\": \"{}\", \"episodes\": {}, \"empirical_tv_ppm\": {}, \"sampling_allowance_ppm\": {}, \"required_tv_radius_ppm\": {}, \"failure_probability_ppm\": {}, \"capped_at_maximum\": {}}}{}",
            regime.regime_id,
            regime.report.calibration_episode_count,
            regime.report.empirical_tv_distance_ppm,
            regime.report.sampling_allowance_ppm,
            regime.report.required_tv_radius_ppm,
            regime.report.failure_probability_ppm,
            regime.report.capped_at_maximum_radius,
            comma
        ).expect("write to String");
    }
    writeln!(
        &mut json,
        "  ]}},\n  \"calibrated_radius_evaluation\": {{\"worst_case_expected_delta_units\": {}, \"worst_single_scenario_id\": \"{}\"}},\n  \"robustness_curve\": [",
        calibrated_report.worst_case_expected_delta_units(),
        calibrated_report.worst_single_scenario_id()
    ).expect("write to String");
    for (index, point) in curve.points.iter().enumerate() {
        let comma = if index + 1 == curve.points.len() { "" } else { "," };
        writeln!(
            &mut json,
            "    {{\"tv_radius_ppm\": {}, \"nominal_delta_numerator_ppm\": {}, \"worst_case_delta_numerator_ppm\": {}, \"robustness_premium_numerator_ppm\": {}}}{}",
            point.tv_radius_ppm,
            point.nominal_expected_delta_numerator_ppm,
            point.worst_case_expected_delta_numerator_ppm,
            point.robustness_premium_numerator_ppm,
            comma
        ).expect("write to String");
    }
    writeln!(
        &mut json,
        "  ],\n  \"first_ineligible_radius_ppm\": {}\n}}",
        curve.first_ineligible_radius_ppm.map_or("null".to_string(), |v| v.to_string())
    ).expect("write to String");

    println!("{json}");
}
