// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Synthetic conformance benchmark for image-to-graph reconstruction scoring.
//!
//! Run:
//! cargo run -p symthaea-canvas --example mycelial_reconstruction_bench -- 100
//!
//! This example uses an explicit synthetic annotation fixture. It does not
//! ingest microscopy images, reconstruct 3D volumes, or report a biological
//! SOTA result. The deliberate false-fusion control demonstrates whether
//! crossing errors propagate into both classification and topology metrics.

use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use symthaea_canvas::{
    AnnotatedHyphaSegment, AnnotatedJunction, AnnotatedJunctionKind, AnnotatedMycelialSkeleton,
    JunctionClassProbabilityPrediction,
    MAX_AUTOMATIC_MATCH_JUNCTIONS_PER_GRAPH, MycelialAnnotationTriageConfig,
    MycelialReconstructionError, MycelialReviewOrderingPolicy, ReconstructionCalibrationClass,
    ReconstructionCalibrationConfig, ReconstructionDimensionality, ReconstructionEvaluationConfig,
    ReconstructionLengthUnit, ReconstructionPoint3, build_mycelial_annotation_triage,
    evaluate_mycelial_reconstruction, evaluate_mycelial_reconstruction_calibration,
    match_junctions_hungarian,
};

const DEFAULT_SAMPLES: usize = 100;
const WARMUP_RUNS: usize = 5;
const MAX_SAMPLES: usize = 10_000;
const MATCH_DISTANCE: f64 = 0.1;

fn junction(id: &str, kind: AnnotatedJunctionKind, x: f64, y: f64) -> AnnotatedJunction {
    AnnotatedJunction {
        id: id.to_string(),
        kind,
        position: ReconstructionPoint3 { x, y, z: 0.0 },
    }
}

fn segment(id: &str, start: &str, end: &str, length: f64) -> AnnotatedHyphaSegment {
    AnnotatedHyphaSegment {
        id: id.to_string(),
        start_junction_id: start.to_string(),
        end_junction_id: end.to_string(),
        length,
    }
}

fn skeleton(
    producer: &str,
    junctions: Vec<AnnotatedJunction>,
    segments: Vec<AnnotatedHyphaSegment>,
) -> AnnotatedMycelialSkeleton {
    AnnotatedMycelialSkeleton {
        schema_version: 1,
        frame_id: "synthetic-frame-0001".to_string(),
        coordinate_frame_id: "synthetic-image-plane-v1".to_string(),
        calibration_digest: "c".repeat(64),
        source_observation_digest: "b".repeat(64),
        dimensionality: ReconstructionDimensionality::TwoD,
        length_unit: ReconstructionLengthUnit::Micrometer,
        producer_id: producer.to_string(),
        producer_version: "synthetic-v1".to_string(),
        junctions,
        segments,
    }
}

fn class_probability_prediction(
    junction: &AnnotatedJunction,
) -> JunctionClassProbabilityPrediction {
    let predicted_class = ReconstructionCalibrationClass::from_junction_kind(junction.kind);
    let mut probabilities = [0.0; 7];
    probabilities[predicted_class.index()] = 0.99;
    probabilities[ReconstructionCalibrationClass::NoObject.index()] = 0.01;
    JunctionClassProbabilityPrediction {
        predicted_junction_id: junction.id.clone(),
        probabilities,
    }
}

fn fixtures() -> (
    AnnotatedMycelialSkeleton,
    AnnotatedMycelialSkeleton,
    AnnotatedMycelialSkeleton,
) {
    let truth = skeleton(
        "ground-truth-annotation",
        vec![
            junction("A", AnnotatedJunctionKind::Branch, 0.0, 0.0),
            junction("B", AnnotatedJunctionKind::Tip, 10.0, 0.0),
            junction("C", AnnotatedJunctionKind::ProjectedCrossing, 0.0, 10.0),
        ],
        vec![
            segment("gt-ab", "A", "B", 10.0),
            segment("gt-ac", "A", "C", 10.0),
        ],
    );

    let prediction = skeleton(
        "candidate-reconstruction",
        vec![
            junction("pA", AnnotatedJunctionKind::Branch, 0.01, 0.0),
            junction("pB", AnnotatedJunctionKind::Tip, 10.01, 0.0),
            junction("pC", AnnotatedJunctionKind::ProjectedCrossing, 0.0, 10.01),
        ],
        vec![
            segment("pred-ab", "pA", "pB", 10.02),
            segment("pred-ac", "pA", "pC", 10.01),
        ],
    );

    let false_fusion = skeleton(
        "candidate-with-projection-fusion-error",
        vec![
            junction("pA", AnnotatedJunctionKind::Branch, 0.01, 0.0),
            junction("pB", AnnotatedJunctionKind::Tip, 10.01, 0.0),
            junction("pC", AnnotatedJunctionKind::Anastomosis, 0.0, 10.01),
        ],
        vec![
            segment("pred-ab", "pA", "pB", 10.02),
            segment("pred-ac", "pA", "pC", 10.01),
            segment("false-crossing-fusion", "pB", "pC", 14.15),
        ],
    );
    (truth, prediction, false_fusion)
}

fn evaluate_case(
    truth: &AnnotatedMycelialSkeleton,
    prediction: &AnnotatedMycelialSkeleton,
) -> Result<(), Box<dyn Error>> {
    let matches = match_junctions_hungarian(truth, prediction, MATCH_DISTANCE)?;
    let score = evaluate_mycelial_reconstruction(
        truth,
        prediction,
        "hungarian-max-cardinality-min-distance-v1",
        &matches,
        ReconstructionEvaluationConfig {
            maximum_junction_match_distance: MATCH_DISTANCE,
            ..ReconstructionEvaluationConfig::default()
        },
    )?;
    score.validate()?;
    println!("matching_policy: {}", score.matching_policy_id);
    println!("matched_junctions: {}", score.identity_matches.len());
    println!(
        "junction_detection_precision: {:?}",
        score.junction_detection.precision()
    );
    println!(
        "junction_detection_recall: {:?}",
        score.junction_detection.recall()
    );
    println!("junction_detection_f1: {:?}", score.junction_detection.f1());
    println!(
        "junction_class_macro_f1: {:?}",
        score.junction_kind_macro_f1
    );
    println!(
        "false_anastomosis_junctions: {}",
        score.false_anastomosis_junction_count
    );
    println!(
        "edge_topology_precision: {:?}",
        score.edge_topology.precision()
    );
    println!("edge_topology_recall: {:?}", score.edge_topology.recall());
    println!("edge_topology_f1: {:?}", score.edge_topology.f1());
    println!(
        "localization_error_p95: {:?}",
        score.localization_error.p95_error
    );
    println!(
        "segment_length_mean_absolute_relative_error: {:?}",
        score.segment_length_error.mean_absolute_relative_error
    );
    println!("cycle_rank_delta: {}", score.topology_delta.cycle_rank);
    println!(
        "component_false_merge_count: {}",
        score.component_error_summary.false_merge_count
    );
    println!(
        "component_false_split_count: {}",
        score.component_error_summary.false_split_count
    );
    println!(
        "unmatched_prediction_component_count: {}",
        score
            .component_error_summary
            .unmatched_prediction_component_count
    );
    println!(
        "unmatched_ground_truth_component_count: {}",
        score
            .component_error_summary
            .unmatched_ground_truth_component_count
    );
    println!(
        "component_identity_attributions: {}",
        score.component_attributions.len()
    );
    println!("score_receipt_digest: {}", score.canonical_digest_hex()?);

    // The class probabilities are synthetic fixture outputs. The false-fusion
    // negative control is deliberately overconfident to expose selective risk.
    let probability_predictions: Vec<_> = prediction
        .junctions
        .iter()
        .map(|junction| {
            let mut item = class_probability_prediction(junction);
            if junction.kind == AnnotatedJunctionKind::Anastomosis {
                item.probabilities[ReconstructionCalibrationClass::Anastomosis.index()] = 0.999;
                item.probabilities[ReconstructionCalibrationClass::NoObject.index()] = 0.001;
            }
            item
        })
        .collect();
    let calibration = evaluate_mycelial_reconstruction_calibration(
        truth,
        prediction,
        &score,
        "synthetic-seven-class-confidence-v1",
        &probability_predictions,
        ReconstructionCalibrationConfig::default(),
    )?;
    println!(
        "top_label_calibration_brier: {:?}",
        calibration.top_label_calibration.brier_score
    );
    println!(
        "top_label_expected_calibration_error: {:?}",
        calibration.top_label_calibration.expected_calibration_error
    );
    println!(
        "top_label_negative_log_likelihood: {:?}",
        calibration.top_label_calibration.negative_log_likelihood
    );
    println!(
        "calibration_false_anastomosis_predictions: {}",
        calibration.counts.false_anastomosis_prediction_count
    );
    println!(
        "calibration_false_anastomosis_rate: {:?}",
        calibration
            .selective_risk_curve
            .last()
            .and_then(|point| point.false_anastomosis_rate)
    );
    println!(
        "selective_risk_points: {}",
        calibration.selective_risk_curve.len()
    );
    println!(
        "calibration_probability_inventory_digest: {}",
        calibration.probability_inventory_digest
    );
    println!(
        "selective_risk_aurc: {:?}",
        calibration.selective_risk_summary.area_under_risk_coverage
    );
    println!(
        "selective_risk_oracle_aurc: {:?}",
        calibration
            .selective_risk_summary
            .oracle_area_under_risk_coverage
    );
    println!(
        "selective_risk_excess_aurc: {:?}",
        calibration
            .selective_risk_summary
            .excess_area_under_risk_coverage
    );
    println!(
        "calibration_receipt_digest: {}",
        calibration.canonical_digest_hex()?
    );

    for policy in [
        MycelialReviewOrderingPolicy::StructuralImpactThenUncertainty,
        MycelialReviewOrderingPolicy::UncertaintyThenStructuralImpact,
        MycelialReviewOrderingPolicy::AnastomosisProbabilityThenImpact,
    ] {
        let triage = build_mycelial_annotation_triage(
            prediction,
            &calibration,
            MycelialAnnotationTriageConfig {
                ordering_policy: policy,
                ..MycelialAnnotationTriageConfig::default()
            },
        )?;
        triage.validate_against(prediction, &calibration)?;
        println!("triage_policy: {policy:?}");
        println!(
            "triage_uncertain_review_count: {}",
            triage.uncertain_review_count
        );
        println!(
            "triage_articulation_critical_count: {}",
            triage.articulation_critical_count
        );
        println!(
            "triage_anastomosis_sensitive_count: {}",
            triage.anastomosis_sensitive_count
        );
        for item in triage.review_items.iter().take(5) {
            println!(
                "triage_item: rank={} id={} class={:?} confidence={:.6} entropy={:.6} degree={} component_delta={} cycle_loss={} anastomosis_p={:.6} crossing_p={:.6}",
                item.review_rank,
                item.junction_id,
                item.predicted_class,
                item.top_label_confidence,
                item.predictive_entropy_nats,
                item.degree,
                item.component_delta,
                item.cycle_rank_loss,
                item.anastomosis_probability,
                item.projected_crossing_probability
            );
        }
        println!("triage_receipt_digest: {}", triage.canonical_digest_hex()?);
    }
    Ok(())
}

fn percentile(sorted: &[u128], percentage: usize) -> u128 {
    sorted[((sorted.len() - 1) * percentage) / 100]
}

fn main() -> Result<(), Box<dyn Error>> {
    let samples = match std::env::args().nth(1) {
        Some(value) => value.parse::<usize>()?,
        None => DEFAULT_SAMPLES,
    };
    if !(1..=MAX_SAMPLES).contains(&samples) {
        return Err(format!("sample count must be in 1..={MAX_SAMPLES}").into());
    }
    if MAX_AUTOMATIC_MATCH_JUNCTIONS_PER_GRAPH < 3 {
        return Err(MycelialReconstructionError::InvalidConfig.into());
    }
    let (truth, prediction, false_fusion) = fixtures();
    truth.validate()?;
    prediction.validate()?;
    false_fusion.validate()?;

    println!("benchmark: mycelial-reconstruction-scorecard-v1");
    println!("scope: synthetic-2d-annotation-conformance-only");
    println!("dimensionality: 2d");
    println!("length_unit: micrometer (synthetic declared unit; not physical measurement)");
    println!(
        "automatic_matcher_max_junctions_per_graph: {}",
        MAX_AUTOMATIC_MATCH_JUNCTIONS_PER_GRAPH
    );
    println!("maximum_match_distance: {MATCH_DISTANCE}");
    println!("case: clean-reconstruction");
    evaluate_case(&truth, &prediction)?;
    println!("case: false-projection-fusion-negative-control");
    evaluate_case(&truth, &false_fusion)?;

    for _ in 0..WARMUP_RUNS {
        let matches = match_junctions_hungarian(&truth, &prediction, MATCH_DISTANCE)?;
        let score = evaluate_mycelial_reconstruction(
            &truth,
            &prediction,
            "hungarian-max-cardinality-min-distance-v1",
            &matches,
            ReconstructionEvaluationConfig {
                maximum_junction_match_distance: MATCH_DISTANCE,
                ..ReconstructionEvaluationConfig::default()
            },
        )?;
        let probabilities: Vec<_> = prediction
            .junctions
            .iter()
            .map(class_probability_prediction)
            .collect();
        black_box(evaluate_mycelial_reconstruction_calibration(
            &truth,
            &prediction,
            &score,
            "synthetic-seven-class-confidence-v1",
            &probabilities,
            ReconstructionCalibrationConfig::default(),
        )?);
    }

    let mut durations_ns = Vec::with_capacity(samples);
    for _ in 0..samples {
        let started = Instant::now();
        let matches = match_junctions_hungarian(&truth, &prediction, MATCH_DISTANCE)?;
        let score = evaluate_mycelial_reconstruction(
            &truth,
            &prediction,
            "hungarian-max-cardinality-min-distance-v1",
            &matches,
            ReconstructionEvaluationConfig {
                maximum_junction_match_distance: MATCH_DISTANCE,
                ..ReconstructionEvaluationConfig::default()
            },
        )?;
        let probabilities: Vec<_> = prediction
            .junctions
            .iter()
            .map(class_probability_prediction)
            .collect();
        let calibration = evaluate_mycelial_reconstruction_calibration(
            &truth,
            &prediction,
            &score,
            "synthetic-seven-class-confidence-v1",
            &probabilities,
            ReconstructionCalibrationConfig::default(),
        )?;
        black_box((
            score.canonical_digest_hex()?,
            calibration.canonical_digest_hex()?,
        ));
        durations_ns.push(started.elapsed().as_nanos());
    }
    durations_ns.sort_unstable();
    println!("warmup_runs: {WARMUP_RUNS}");
    println!("samples: {samples}");
    println!(
        "matching_plus_scoring_time_ns_p50: {}",
        percentile(&durations_ns, 50)
    );
    println!(
        "matching_plus_scoring_time_ns_p95: {}",
        percentile(&durations_ns, 95)
    );
    println!("platform_os: {}", std::env::consts::OS);
    println!("platform_arch: {}", std::env::consts::ARCH);
    println!(
        "benchmark_scope: synthetic conformance; not microscopy reconstruction, biology or SOTA comparison"
    );
    Ok(())
}
