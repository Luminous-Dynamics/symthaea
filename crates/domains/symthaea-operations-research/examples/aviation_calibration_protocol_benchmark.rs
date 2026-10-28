// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Synthetic contract-composition example for aviation calibration and robust
//! recovery. This is not a real-data benchmark and does not establish savings.

use std::fmt::Write as FmtWrite;
use std::process;

use symthaea_operations_research::aviation_calibration_protocol::{
    audit_no_test_set_tuning, audit_split_integrity, EvaluationPartition,
    SplitAuditStatus, SplitIntegrityPolicy, SplitRecord, TuningChoice,
};
use symthaea_operations_research::aviation_distributional_robustness::{
    evaluate_tv_robustness_curve, ScenarioCostDelta,
};

fn main() {
    let partitions = [
        (EvaluationPartition::Training, 1_700_000_000_i64),
        (EvaluationPartition::Calibration, 1_700_010_000_i64),
        (EvaluationPartition::Validation, 1_700_020_000_i64),
        (EvaluationPartition::Test, 1_700_030_000_i64),
    ];
    let mut records = Vec::new();
    for (partition, epoch) in partitions {
        for index in 0..3 {
            records.push(SplitRecord {
                record_id: format!("{partition:?}-row-{index}"),
                event_cluster_id: format!("{partition:?}-episode-{index}"),
                event_time_epoch_seconds: Some(epoch + index),
                partition,
            });
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

    let mut json = String::new();
    writeln!(
        &mut json,
        "{{\n  \"benchmark\": \"aviation-calibration-robustness-contract\",\n  \"run_status\": \"ACCEPTED\",\n  \"claim_ceiling\": \"synthetic fixtures only\",\n  \"split_audit\": {{\n    \"status\": \"{:?}\",\n    \"partitions\": [",
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
        "    ]\n  }},\n  \"tuning_audit\": {{\"status\": \"{:?}\", \"tuned_on_test\": false}},\n  \"robustness_curve\": [",
        tuning.status
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
