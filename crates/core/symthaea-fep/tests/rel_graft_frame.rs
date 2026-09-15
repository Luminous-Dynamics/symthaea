// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! REL-005A: production-surface qualification of the existing Xenobot graft
//! transform as a diagonal multiplicative frame action.
//!
//! Positive fixtures remain inside `ContinuousHV`'s documented nominal [-1, 1]
//! value range. The test distinguishes raw state transport from fixed-operation
//! invariance and from transported (dressed) algebra/metric covariance.

use std::env;
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};

use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_fep::SwarmCoalition;

const POSITIVE_TOL: f32 = 1.0e-5;
const FIXED_HADAMARD_MIN_DEFECT: f32 = 1.2;
const RAW_COSINE_MIN_DRIFT: f32 = 0.25;
const NEGATIVE_HELD_OUT_MIN_ERROR: f32 = 0.20;
const NEGATIVE_DISPERSION_MIN: f32 = 0.10;
const INVERSE_FLOOR: f32 = 1.0e-7;

fn hv(values: &[f32]) -> ContinuousHV {
    ContinuousHV::from_values(values.to_vec())
}

fn coalition() -> SwarmCoalition {
    SwarmCoalition {
        members: Vec::new(),
        internal_phi: 0.0,
        boundary_phi: 0.0,
        cohesion: 0.0,
        mean_internal_permeability: 0.0,
        mean_external_permeability: 0.0,
    }
}

fn max_abs_error(left: &ContinuousHV, right: &ContinuousHV) -> f32 {
    assert_eq!(left.dim(), right.dim());
    left.values
        .iter()
        .zip(&right.values)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn rms_error(left: &ContinuousHV, right: &ContinuousHV) -> f32 {
    assert_eq!(left.dim(), right.dim());
    (left
        .values
        .iter()
        .zip(&right.values)
        .map(|(a, b)| {
            let delta = a - b;
            delta * delta
        })
        .sum::<f32>()
        / left.dim() as f32)
        .sqrt()
}

fn cosine(left: &ContinuousHV, right: &ContinuousHV) -> f32 {
    assert_eq!(left.dim(), right.dim());
    let mut dot = 0.0_f32;
    let mut left_sq = 0.0_f32;
    let mut right_sq = 0.0_f32;
    for (&a, &b) in left.values.iter().zip(&right.values) {
        dot += a * b;
        left_sq += a * a;
        right_sq += b * b;
    }
    dot / (left_sq * right_sq).sqrt()
}

fn reverse_coordinates(value: &ContinuousHV) -> ContinuousHV {
    let mut values = value.values.clone();
    values.reverse();
    ContinuousHV::from_values(values)
}

fn construction_error(
    frame: &SwarmCoalition,
    source: &[ContinuousHV],
    target: &[ContinuousHV],
    transform: &ContinuousHV,
) -> (f32, f32) {
    let mut max_error = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    let mut count = 0usize;
    for (source_anchor, target_anchor) in source.iter().zip(target) {
        let translated = frame.translate_foreign_hypervector(source_anchor, transform);
        max_error = max_error.max(max_abs_error(&translated, target_anchor));
        for (&actual, &expected) in translated.values.iter().zip(&target_anchor.values) {
            let delta = actual - expected;
            sum_sq += delta * delta;
            count += 1;
        }
    }
    (max_error, (sum_sq / count as f32).sqrt())
}

fn implied_masks(target: &[ContinuousHV], source: &[ContinuousHV]) -> Vec<ContinuousHV> {
    target
        .iter()
        .zip(source)
        .map(|(target_anchor, source_anchor)| target_anchor.bind(&source_anchor.inverse()))
        .collect()
}

fn mask_dispersion(masks: &[ContinuousHV], center: &ContinuousHV) -> (f32, f32) {
    let mut max_error = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    let mut count = 0usize;
    for mask in masks {
        max_error = max_error.max(max_abs_error(mask, center));
        for (&value, &center_value) in mask.values.iter().zip(&center.values) {
            let delta = value - center_value;
            sum_sq += delta * delta;
            count += 1;
        }
    }
    (max_error, (sum_sq / count as f32).sqrt())
}

#[test]
fn rel_005a_measurement_contract() {
    let frame = coalition();
    let mask_ab = hv(&[0.5, -0.75, 0.9, -0.85, 0.8, -0.7, 0.95, -0.6]);
    let mask_bc = hv(&[-0.8, 0.7, -0.9, 0.65, 0.6, 0.75, -0.85, 0.9]);

    let anchors_a = vec![
        hv(&[0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, -0.95]),
        hv(&[-0.9, 0.8, -0.7, 0.6, -0.5, 0.4, -0.85, 0.75]),
        hv(&[0.95, -1.0, 0.9, -0.8, 0.7, -0.6, 0.5, -0.4]),
        hv(&[-0.35, 0.45, -0.55, 0.65, -0.75, 0.85, -0.95, 1.0]),
    ];
    let anchors_b = anchors_a
        .iter()
        .map(|anchor| anchor.bind(&mask_ab))
        .collect::<Vec<_>>();
    let anchors_c = anchors_b
        .iter()
        .map(|anchor| anchor.bind(&mask_bc))
        .collect::<Vec<_>>();

    assert!(
        anchors_a
            .iter()
            .chain(&anchors_b)
            .chain(&anchors_c)
            .all(|hv| {
                hv.values
                    .iter()
                    .all(|value| value.is_finite() && value.abs() <= 1.0)
            })
    );
    let positive_inverse_floor_affected_count = anchors_a
        .iter()
        .flat_map(|anchor| anchor.values.iter())
        .filter(|value| value.abs() < INVERSE_FLOOR)
        .count();
    assert_eq!(positive_inverse_floor_affected_count, 0);

    let estimated_ab = frame
        .compute_xenobot_graft_transform(&anchors_b, &anchors_a)
        .expect("shared-mask A->B fixture must produce a transform");
    let mask_recovery_max = max_abs_error(&estimated_ab, &mask_ab);
    let mask_recovery_rms = rms_error(&estimated_ab, &mask_ab);
    assert!(mask_recovery_max <= POSITIVE_TOL);

    let shared_implied_masks = implied_masks(&anchors_b, &anchors_a);
    let (shared_mask_dispersion_max, shared_mask_dispersion_rms) =
        mask_dispersion(&shared_implied_masks, &estimated_ab);
    assert!(shared_mask_dispersion_max <= POSITIVE_TOL);

    let (construction_max, construction_rms) =
        construction_error(&frame, &anchors_a, &anchors_b, &estimated_ab);
    assert!(construction_max <= POSITIVE_TOL);

    let held_out_a = hv(&[0.55, -0.65, 0.75, -0.85, 0.95, -1.0, 0.9, -0.8]);
    let expected_held_out_b = held_out_a.bind(&mask_ab);
    let actual_held_out_b = frame.translate_foreign_hypervector(&held_out_a, &estimated_ab);
    let held_out_max = max_abs_error(&actual_held_out_b, &expected_held_out_b);
    let held_out_rms = rms_error(&actual_held_out_b, &expected_held_out_b);
    assert!(held_out_max <= POSITIVE_TOL);

    let estimated_bc = frame
        .compute_xenobot_graft_transform(&anchors_c, &anchors_b)
        .expect("shared-mask B->C fixture must produce a transform");
    let estimated_ac = frame
        .compute_xenobot_graft_transform(&anchors_c, &anchors_a)
        .expect("shared-mask A->C fixture must produce a transform");
    let expected_ac = mask_ab.bind(&mask_bc);
    assert!(max_abs_error(&estimated_ac, &expected_ac) <= POSITIVE_TOL);

    let composed_ab_bc = estimated_ab.bind(&estimated_bc);
    let direct_vs_composed_mask_error = max_abs_error(&estimated_ac, &composed_ab_bc);
    assert!(direct_vs_composed_mask_error <= POSITIVE_TOL);

    let held_out_b = frame.translate_foreign_hypervector(&held_out_a, &estimated_ab);
    let held_out_c_sequential = frame.translate_foreign_hypervector(&held_out_b, &estimated_bc);
    let held_out_c_direct = frame.translate_foreign_hypervector(&held_out_a, &estimated_ac);
    let direct_vs_sequential_transport_error =
        max_abs_error(&held_out_c_direct, &held_out_c_sequential);
    assert!(direct_vs_sequential_transport_error <= POSITIVE_TOL);

    let estimated_ca = frame
        .compute_xenobot_graft_transform(&anchors_a, &anchors_c)
        .expect("shared-mask C->A fixture must produce a transform");
    let loop_b = frame.translate_foreign_hypervector(&held_out_a, &estimated_ab);
    let loop_c = frame.translate_foreign_hypervector(&loop_b, &estimated_bc);
    let loop_a = frame.translate_foreign_hypervector(&loop_c, &estimated_ca);
    let loop_closure_max = max_abs_error(&loop_a, &held_out_a);
    let loop_closure_rms = rms_error(&loop_a, &held_out_a);
    assert!(loop_closure_max <= POSITIVE_TOL);

    let x = hv(&[1.0, -0.7, -0.7, 0.8, -0.5, 0.4, 0.9, -0.7]);
    let y = hv(&[-0.6, -0.8, -0.4, 1.0, 0.5, -0.7, 0.4, 1.0]);
    let fx = frame.translate_foreign_hypervector(&x, &estimated_ab);
    let fy = frame.translate_foreign_hypervector(&y, &estimated_ab);

    let source_bundle = ContinuousHV::bundle(&[&x, &y]);
    let transported_source_bundle =
        frame.translate_foreign_hypervector(&source_bundle, &estimated_ab);
    let target_bundle = ContinuousHV::bundle(&[&fx, &fy]);
    let bundle_covariance_defect = max_abs_error(&transported_source_bundle, &target_bundle);
    assert!(bundle_covariance_defect <= POSITIVE_TOL);

    let transported_unit =
        frame.translate_foreign_hypervector(&ContinuousHV::ones(8), &estimated_ab);
    let transported_unit_defect = max_abs_error(&transported_unit, &estimated_ab);
    assert!(transported_unit_defect <= POSITIVE_TOL);

    let transported_source_bind = frame.translate_foreign_hypervector(&x.bind(&y), &estimated_ab);
    let fixed_target_bind = fx.bind(&fy);
    let fixed_hadamard_defect = max_abs_error(&transported_source_bind, &fixed_target_bind);
    assert!(fixed_hadamard_defect > FIXED_HADAMARD_MIN_DEFECT);

    let estimated_ab_inverse = estimated_ab.inverse();
    let dressed_target_bind = fixed_target_bind.bind(&estimated_ab_inverse);
    let dressed_hadamard_defect = max_abs_error(&transported_source_bind, &dressed_target_bind);
    assert!(dressed_hadamard_defect <= POSITIVE_TOL);
    let dressed_unit_action = transported_unit.bind(&fx).bind(&estimated_ab_inverse);
    let dressed_unit_action_defect = max_abs_error(&dressed_unit_action, &fx);
    assert!(dressed_unit_action_defect <= POSITIVE_TOL);

    let source_cosine = cosine(&x, &y);
    let raw_target_cosine = cosine(&fx, &fy);
    let raw_cosine_drift = (source_cosine - raw_target_cosine).abs();
    assert!(raw_cosine_drift > RAW_COSINE_MIN_DRIFT);
    let pulled_back_x = fx.bind(&estimated_ab_inverse);
    let pulled_back_y = fy.bind(&estimated_ab_inverse);
    let pulled_back_cosine_drift = (source_cosine - cosine(&pulled_back_x, &pulled_back_y)).abs();
    assert!(pulled_back_cosine_drift <= POSITIVE_TOL);

    assert!(frame.compute_xenobot_graft_transform(&[], &[]).is_none());
    assert!(
        frame
            .compute_xenobot_graft_transform(&anchors_b[..2], &anchors_a[..1])
            .is_none()
    );

    let dimension_mismatch_panics = catch_unwind(AssertUnwindSafe(|| {
        let native = [hv(&[0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, -0.95])];
        let foreign = [hv(&[0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0])];
        let _ = frame.compute_xenobot_graft_transform(&native, &foreign);
    }))
    .is_err();
    assert!(dimension_mismatch_panics);

    let shuffled_a = vec![
        anchors_a[1].clone(),
        anchors_a[2].clone(),
        anchors_a[3].clone(),
        anchors_a[0].clone(),
    ];
    let shuffled_transform = frame
        .compute_xenobot_graft_transform(&anchors_b, &shuffled_a)
        .expect("shuffled equal-size anchors still produce the current API estimate");
    let shuffled_held_out = frame.translate_foreign_hypervector(&held_out_a, &shuffled_transform);
    let shuffled_held_out_max = max_abs_error(&shuffled_held_out, &expected_held_out_b);
    let (shuffled_mask_dispersion_max, _) =
        mask_dispersion(&implied_masks(&anchors_b, &shuffled_a), &shuffled_transform);
    assert!(shuffled_held_out_max > NEGATIVE_HELD_OUT_MIN_ERROR);
    assert!(shuffled_mask_dispersion_max > NEGATIVE_DISPERSION_MIN);

    let mixed_targets = anchors_a
        .iter()
        .enumerate()
        .map(|(index, anchor)| {
            if index < 2 {
                anchor.bind(&mask_ab)
            } else {
                anchor.bind(&mask_bc)
            }
        })
        .collect::<Vec<_>>();
    let mixed_transform = frame
        .compute_xenobot_graft_transform(&mixed_targets, &anchors_a)
        .expect("mixed masks are currently averaged rather than rejected");
    let mixed_held_out = frame.translate_foreign_hypervector(&held_out_a, &mixed_transform);
    let mixed_mask_held_out_max = max_abs_error(&mixed_held_out, &expected_held_out_b);
    let (mixed_mask_dispersion_max, _) =
        mask_dispersion(&implied_masks(&mixed_targets, &anchors_a), &mixed_transform);
    assert!(mixed_mask_held_out_max > NEGATIVE_HELD_OUT_MIN_ERROR);
    assert!(mixed_mask_dispersion_max > NEGATIVE_DISPERSION_MIN);

    let mut corrupted_targets = anchors_b.clone();
    corrupted_targets[0].values[0] += 1.5;
    let corrupted_transform = frame
        .compute_xenobot_graft_transform(&corrupted_targets, &anchors_a)
        .expect("one corrupted anchor is currently averaged rather than rejected");
    let corrupted_held_out = frame.translate_foreign_hypervector(&held_out_a, &corrupted_transform);
    let corrupted_anchor_held_out_max = max_abs_error(&corrupted_held_out, &expected_held_out_b);
    let (corrupted_mask_dispersion_max, _) = mask_dispersion(
        &implied_masks(&corrupted_targets, &anchors_a),
        &corrupted_transform,
    );
    assert!(corrupted_anchor_held_out_max > NEGATIVE_HELD_OUT_MIN_ERROR);
    assert!(corrupted_mask_dispersion_max > NEGATIVE_DISPERSION_MIN);

    let near_floor_source = hv(&[5.0e-8, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.0]);
    let near_floor_target = near_floor_source.bind(&mask_ab);
    let near_floor_transform = frame
        .compute_xenobot_graft_transform(
            &[near_floor_target],
            std::slice::from_ref(&near_floor_source),
        )
        .expect("near-floor input still produces the current API estimate");
    let near_floor_held_out =
        frame.translate_foreign_hypervector(&held_out_a, &near_floor_transform);
    let near_floor_held_out_max = max_abs_error(&near_floor_held_out, &expected_held_out_b);
    let inverse_floor_affected_count = near_floor_source
        .values
        .iter()
        .filter(|value| value.abs() < INVERSE_FLOOR)
        .count();
    let inverse_floor_affected_rate =
        inverse_floor_affected_count as f32 / near_floor_source.dim() as f32;
    assert_eq!(inverse_floor_affected_count, 1);
    assert!(near_floor_held_out_max > NEGATIVE_HELD_OUT_MIN_ERROR);

    let permuted_targets = anchors_a
        .iter()
        .map(reverse_coordinates)
        .collect::<Vec<_>>();
    let permutation_estimate = frame
        .compute_xenobot_graft_transform(&permuted_targets, &anchors_a)
        .expect("coordinate reversal currently yields a best-fit mask estimate");
    let permutation_predicted =
        frame.translate_foreign_hypervector(&held_out_a, &permutation_estimate);
    let permutation_expected = reverse_coordinates(&held_out_a);
    let permutation_held_out_max = max_abs_error(&permutation_predicted, &permutation_expected);
    assert!(permutation_held_out_max > NEGATIVE_HELD_OUT_MIN_ERROR);

    let nonfinite_source = hv(&[f32::NAN, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.0]);
    let nonfinite_target = nonfinite_source.bind(&mask_ab);
    let nonfinite_transform = frame
        .compute_xenobot_graft_transform(
            &[nonfinite_target],
            std::slice::from_ref(&nonfinite_source),
        )
        .expect("current API does not reject non-finite anchor content");
    let nonfinite_input_returns_nonfinite_transform = nonfinite_transform
        .values
        .iter()
        .any(|value| !value.is_finite());
    assert!(nonfinite_input_returns_nonfinite_transform);

    let metrics = format!(concat!(
        "{{\n",
        "  \"schema\": \"symthaea.rel.graft-frame-measurement.v2\",\n",
        "  \"authority\": \"MeasurementOnly\",\n",
        "  \"positive_fixture_within_nominal_range\": true,\n",
        "  \"positive_inverse_floor_affected_count\": {positive_inverse_floor_affected_count},\n",
        "  \"mask_recovery_max\": {mask_recovery_max:.9e},\n",
        "  \"mask_recovery_rms\": {mask_recovery_rms:.9e},\n",
        "  \"shared_mask_dispersion_max\": {shared_mask_dispersion_max:.9e},\n",
        "  \"shared_mask_dispersion_rms\": {shared_mask_dispersion_rms:.9e},\n",
        "  \"construction_anchor_max\": {construction_max:.9e},\n",
        "  \"construction_anchor_rms\": {construction_rms:.9e},\n",
        "  \"held_out_max\": {held_out_max:.9e},\n",
        "  \"held_out_rms\": {held_out_rms:.9e},\n",
        "  \"direct_vs_composed_mask_max\": {direct_vs_composed_mask_error:.9e},\n",
        "  \"direct_vs_sequential_transport_max\": {direct_vs_sequential_transport_error:.9e},\n",
        "  \"loop_closure_max\": {loop_closure_max:.9e},\n",
        "  \"loop_closure_rms\": {loop_closure_rms:.9e},\n",
        "  \"bundle_covariance_defect\": {bundle_covariance_defect:.9e},\n",
        "  \"transported_unit_defect\": {transported_unit_defect:.9e},\n",
        "  \"dressed_unit_action_defect\": {dressed_unit_action_defect:.9e},\n",
        "  \"fixed_hadamard_covariance_defect\": {fixed_hadamard_defect:.9e},\n",
        "  \"dressed_hadamard_covariance_defect\": {dressed_hadamard_defect:.9e},\n",
        "  \"raw_cosine_drift\": {raw_cosine_drift:.9e},\n",
        "  \"pulled_back_cosine_drift\": {pulled_back_cosine_drift:.9e},\n",
        "  \"shuffled_anchor_held_out_max\": {shuffled_held_out_max:.9e},\n",
        "  \"shuffled_mask_dispersion_max\": {shuffled_mask_dispersion_max:.9e},\n",
        "  \"mixed_mask_held_out_max\": {mixed_mask_held_out_max:.9e},\n",
        "  \"mixed_mask_dispersion_max\": {mixed_mask_dispersion_max:.9e},\n",
        "  \"corrupted_anchor_held_out_max\": {corrupted_anchor_held_out_max:.9e},\n",
        "  \"corrupted_mask_dispersion_max\": {corrupted_mask_dispersion_max:.9e},\n",
        "  \"near_floor_held_out_max\": {near_floor_held_out_max:.9e},\n",
        "  \"inverse_floor_affected_count\": {inverse_floor_affected_count},\n",
        "  \"inverse_floor_affected_rate\": {inverse_floor_affected_rate:.9e},\n",
        "  \"permutation_held_out_max\": {permutation_held_out_max:.9e},\n",
        "  \"dimension_mismatch_panics\": {dimension_mismatch_panics},\n",
        "  \"nonfinite_input_returns_nonfinite_transform\": {nonfinite_input_returns_nonfinite_transform},\n",
        "  \"claims\": {{\n",
        "    \"fixed_hadamard_invariance_established\": false,\n",
        "    \"fixed_cosine_invariance_established\": false,\n",
        "    \"general_coordinate_transform_established\": false,\n",
        "    \"physical_gauge_symmetry_established\": false,\n",
        "    \"consciousness_claim_established\": false\n",
        "  }}\n",
        "}}\n"
    ));

    println!("REL005A_METRICS={metrics}");
    if let Ok(path) = env::var("REL005A_METRICS_PATH") {
        fs::write(path, metrics).expect("write REL-005A metrics artifact");
    }
}
