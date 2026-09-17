// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! REL-005A ExecutionOnly v3.
//!
//! Replays the frozen #3142 fixtures and production surfaces while emitting
//! observations without applying scientific pass/fail thresholds. The only
//! retained numeric boundary is INVERSE_FLOOR because it defines the frozen
//! inverse-exposure measurement itself rather than scientific qualification.

use std::env;
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};

use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_fep::SwarmCoalition;

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

fn max_abs_error(left: &ContinuousHV, right: &ContinuousHV) -> Option<f32> {
    if left.dim() != right.dim() {
        return None;
    }
    Some(
        left.values
            .iter()
            .zip(&right.values)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max),
    )
}

fn rms_error(left: &ContinuousHV, right: &ContinuousHV) -> Option<f32> {
    if left.dim() != right.dim() || left.dim() == 0 {
        return None;
    }
    Some(
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
            .sqrt(),
    )
}

fn cosine(left: &ContinuousHV, right: &ContinuousHV) -> Option<f32> {
    if left.dim() != right.dim() || left.dim() == 0 {
        return None;
    }
    let mut dot = 0.0_f32;
    let mut left_sq = 0.0_f32;
    let mut right_sq = 0.0_f32;
    for (&a, &b) in left.values.iter().zip(&right.values) {
        dot += a * b;
        left_sq += a * a;
        right_sq += b * b;
    }
    let denom = (left_sq * right_sq).sqrt();
    if denom == 0.0 || !denom.is_finite() {
        None
    } else {
        Some(dot / denom)
    }
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
) -> Option<(f32, f32)> {
    if source.len() != target.len() || source.is_empty() {
        return None;
    }
    let mut max_error = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    let mut count = 0usize;
    for (source_anchor, target_anchor) in source.iter().zip(target) {
        let translated = frame.translate_foreign_hypervector(source_anchor, transform);
        max_error = max_error.max(max_abs_error(&translated, target_anchor)?);
        if translated.dim() != target_anchor.dim() {
            return None;
        }
        for (&actual, &expected) in translated.values.iter().zip(&target_anchor.values) {
            let delta = actual - expected;
            sum_sq += delta * delta;
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some((max_error, (sum_sq / count as f32).sqrt()))
    }
}

fn implied_masks(target: &[ContinuousHV], source: &[ContinuousHV]) -> Vec<ContinuousHV> {
    target
        .iter()
        .zip(source)
        .map(|(target_anchor, source_anchor)| target_anchor.bind(&source_anchor.inverse()))
        .collect()
}

fn mask_dispersion(masks: &[ContinuousHV], center: &ContinuousHV) -> Option<(f32, f32)> {
    if masks.is_empty() {
        return None;
    }
    let mut max_error = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    let mut count = 0usize;
    for mask in masks {
        max_error = max_error.max(max_abs_error(mask, center)?);
        if mask.dim() != center.dim() {
            return None;
        }
        for (&value, &center_value) in mask.values.iter().zip(&center.values) {
            let delta = value - center_value;
            sum_sq += delta * delta;
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some((max_error, (sum_sq / count as f32).sqrt()))
    }
}

fn json_number(value: Option<f32>) -> String {
    match value {
        Some(value) if value.is_finite() => format!("{value:.9e}"),
        _ => "null".to_string(),
    }
}

fn json_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn json_bool(value: bool) -> String {
    value.to_string()
}

fn json_hv(value: Option<&ContinuousHV>) -> String {
    match value {
        Some(value) => {
            let rendered = value
                .values
                .iter()
                .map(|component| {
                    if component.is_finite() {
                        format!("{component:.9e}")
                    } else if component.is_nan() {
                        "\"NaN\"".to_string()
                    } else if component.is_sign_positive() {
                        "\"Infinity\"".to_string()
                    } else {
                        "\"-Infinity\"".to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{rendered}]")
        }
        None => "null".to_string(),
    }
}

fn json_f32_slice(values: &[f32]) -> String {
    let rendered = values
        .iter()
        .map(|value| {
            if value.is_finite() {
                format!("{value:.9e}")
            } else if value.is_nan() {
                "\"NaN\"".to_string()
            } else if value.is_sign_positive() {
                "\"Infinity\"".to_string()
            } else {
                "\"-Infinity\"".to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{rendered}]")
}

fn emit_json(fields: Vec<(&str, String)>) -> String {
    let body = fields
        .into_iter()
        .map(|(key, value)| format!("  \"{key}\": {value}"))
        .collect::<Vec<_>>()
        .join(",\n");
    format!("{{\n{body}\n}}\n")
}

#[test]
fn rel_005a_execution_only_v3() {
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

    let positive_fixture_nonfinite_count = anchors_a
        .iter()
        .chain(&anchors_b)
        .chain(&anchors_c)
        .flat_map(|value| value.values.iter())
        .filter(|value| !value.is_finite())
        .count();
    let positive_fixture_max_abs = anchors_a
        .iter()
        .chain(&anchors_b)
        .chain(&anchors_c)
        .flat_map(|value| value.values.iter())
        .filter(|value| value.is_finite())
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    let positive_source_abs_values = anchors_a
        .iter()
        .flat_map(|anchor| anchor.values.iter())
        .map(|value| value.abs())
        .collect::<Vec<_>>();
    let positive_inverse_floor_affected_count = positive_source_abs_values
        .iter()
        .filter(|value| **value < INVERSE_FLOOR)
        .count();

    let estimated_ab = frame.compute_xenobot_graft_transform(&anchors_b, &anchors_a);
    let mask_recovery_max = estimated_ab
        .as_ref()
        .and_then(|transform| max_abs_error(transform, &mask_ab));
    let mask_recovery_rms = estimated_ab
        .as_ref()
        .and_then(|transform| rms_error(transform, &mask_ab));
    let shared_implied_masks = implied_masks(&anchors_b, &anchors_a);
    let shared_mask_dispersion = estimated_ab
        .as_ref()
        .and_then(|transform| mask_dispersion(&shared_implied_masks, transform));
    let construction = estimated_ab
        .as_ref()
        .and_then(|transform| construction_error(&frame, &anchors_a, &anchors_b, transform));

    let held_out_a = hv(&[0.55, -0.65, 0.75, -0.85, 0.95, -1.0, 0.9, -0.8]);
    let expected_held_out_b = held_out_a.bind(&mask_ab);
    let actual_held_out_b = estimated_ab
        .as_ref()
        .map(|transform| frame.translate_foreign_hypervector(&held_out_a, transform));
    let held_out_max = actual_held_out_b
        .as_ref()
        .and_then(|actual| max_abs_error(actual, &expected_held_out_b));
    let held_out_rms = actual_held_out_b
        .as_ref()
        .and_then(|actual| rms_error(actual, &expected_held_out_b));

    let estimated_bc = frame.compute_xenobot_graft_transform(&anchors_c, &anchors_b);
    let estimated_ac = frame.compute_xenobot_graft_transform(&anchors_c, &anchors_a);
    let expected_ac = mask_ab.bind(&mask_bc);
    let direct_expected_mask_error = estimated_ac
        .as_ref()
        .and_then(|transform| max_abs_error(transform, &expected_ac));
    let composed_ab_bc = match (&estimated_ab, &estimated_bc) {
        (Some(ab), Some(bc)) => Some(ab.bind(bc)),
        _ => None,
    };
    let direct_vs_composed_mask_error = match (&estimated_ac, &composed_ab_bc) {
        (Some(ac), Some(composed)) => max_abs_error(ac, composed),
        _ => None,
    };

    let held_out_b = estimated_ab
        .as_ref()
        .map(|transform| frame.translate_foreign_hypervector(&held_out_a, transform));
    let held_out_c_sequential = match (&held_out_b, &estimated_bc) {
        (Some(value), Some(transform)) => {
            Some(frame.translate_foreign_hypervector(value, transform))
        }
        _ => None,
    };
    let held_out_c_direct = estimated_ac
        .as_ref()
        .map(|transform| frame.translate_foreign_hypervector(&held_out_a, transform));
    let direct_vs_sequential_transport_error =
        match (&held_out_c_direct, &held_out_c_sequential) {
            (Some(direct), Some(sequential)) => max_abs_error(direct, sequential),
            _ => None,
        };

    let estimated_ca = frame.compute_xenobot_graft_transform(&anchors_a, &anchors_c);
    let loop_b = estimated_ab
        .as_ref()
        .map(|transform| frame.translate_foreign_hypervector(&held_out_a, transform));
    let loop_c = match (&loop_b, &estimated_bc) {
        (Some(value), Some(transform)) => {
            Some(frame.translate_foreign_hypervector(value, transform))
        }
        _ => None,
    };
    let loop_a = match (&loop_c, &estimated_ca) {
        (Some(value), Some(transform)) => {
            Some(frame.translate_foreign_hypervector(value, transform))
        }
        _ => None,
    };
    let loop_closure_max = loop_a
        .as_ref()
        .and_then(|value| max_abs_error(value, &held_out_a));
    let loop_closure_rms = loop_a
        .as_ref()
        .and_then(|value| rms_error(value, &held_out_a));

    let x = hv(&[1.0, -0.7, -0.7, 0.8, -0.5, 0.4, 0.9, -0.7]);
    let y = hv(&[-0.6, -0.8, -0.4, 1.0, 0.5, -0.7, 0.4, 1.0]);
    let fx = estimated_ab
        .as_ref()
        .map(|transform| frame.translate_foreign_hypervector(&x, transform));
    let fy = estimated_ab
        .as_ref()
        .map(|transform| frame.translate_foreign_hypervector(&y, transform));

    let source_bundle = ContinuousHV::bundle(&[&x, &y]);
    let transported_source_bundle = estimated_ab
        .as_ref()
        .map(|transform| frame.translate_foreign_hypervector(&source_bundle, transform));
    let target_bundle = match (&fx, &fy) {
        (Some(fx), Some(fy)) => Some(ContinuousHV::bundle(&[fx, fy])),
        _ => None,
    };
    let bundle_covariance_defect = match (&transported_source_bundle, &target_bundle) {
        (Some(actual), Some(expected)) => max_abs_error(actual, expected),
        _ => None,
    };

    let transported_unit = estimated_ab.as_ref().map(|transform| {
        frame.translate_foreign_hypervector(&ContinuousHV::ones(8), transform)
    });
    let transported_unit_defect = match (&transported_unit, &estimated_ab) {
        (Some(actual), Some(expected)) => max_abs_error(actual, expected),
        _ => None,
    };

    let transported_source_bind = estimated_ab
        .as_ref()
        .map(|transform| frame.translate_foreign_hypervector(&x.bind(&y), transform));
    let fixed_target_bind = match (&fx, &fy) {
        (Some(fx), Some(fy)) => Some(fx.bind(fy)),
        _ => None,
    };
    let fixed_hadamard_defect = match (&transported_source_bind, &fixed_target_bind) {
        (Some(actual), Some(expected)) => max_abs_error(actual, expected),
        _ => None,
    };

    let estimated_ab_inverse = estimated_ab.as_ref().map(ContinuousHV::inverse);
    let dressed_target_bind = match (&fixed_target_bind, &estimated_ab_inverse) {
        (Some(fixed), Some(inverse)) => Some(fixed.bind(inverse)),
        _ => None,
    };
    let dressed_hadamard_defect = match (&transported_source_bind, &dressed_target_bind) {
        (Some(actual), Some(expected)) => max_abs_error(actual, expected),
        _ => None,
    };
    let dressed_unit_action = match (&transported_unit, &fx, &estimated_ab_inverse) {
        (Some(unit), Some(fx), Some(inverse)) => Some(unit.bind(fx).bind(inverse)),
        _ => None,
    };
    let dressed_unit_action_defect = match (&dressed_unit_action, &fx) {
        (Some(actual), Some(expected)) => max_abs_error(actual, expected),
        _ => None,
    };

    let source_cosine = cosine(&x, &y);
    let raw_target_cosine = match (&fx, &fy) {
        (Some(fx), Some(fy)) => cosine(fx, fy),
        _ => None,
    };
    let raw_cosine_drift = match (source_cosine, raw_target_cosine) {
        (Some(source), Some(target)) => Some((source - target).abs()),
        _ => None,
    };
    let pulled_back_x = match (&fx, &estimated_ab_inverse) {
        (Some(fx), Some(inverse)) => Some(fx.bind(inverse)),
        _ => None,
    };
    let pulled_back_y = match (&fy, &estimated_ab_inverse) {
        (Some(fy), Some(inverse)) => Some(fy.bind(inverse)),
        _ => None,
    };
    let pulled_back_cosine = match (&pulled_back_x, &pulled_back_y) {
        (Some(x), Some(y)) => cosine(x, y),
        _ => None,
    };
    let pulled_back_cosine_drift = match (source_cosine, pulled_back_cosine) {
        (Some(source), Some(pulled)) => Some((source - pulled).abs()),
        _ => None,
    };

    let empty_anchor_result_is_none =
        frame.compute_xenobot_graft_transform(&[], &[]).is_none();
    let unequal_anchor_count_result_is_none = frame
        .compute_xenobot_graft_transform(&anchors_b[..2], &anchors_a[..1])
        .is_none();

    let dimension_mismatch_panics = catch_unwind(AssertUnwindSafe(|| {
        let native = [hv(&[0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, -0.95])];
        let foreign = [hv(&[0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0])];
        let _ = frame.compute_xenobot_graft_transform(&native, &foreign);
    }))
    .is_err();

    let shuffled_a = vec![
        anchors_a[1].clone(),
        anchors_a[2].clone(),
        anchors_a[3].clone(),
        anchors_a[0].clone(),
    ];
    let shuffled_transform = frame.compute_xenobot_graft_transform(&anchors_b, &shuffled_a);
    let shuffled_held_out = shuffled_transform
        .as_ref()
        .map(|transform| frame.translate_foreign_hypervector(&held_out_a, transform));
    let shuffled_held_out_max = shuffled_held_out
        .as_ref()
        .and_then(|actual| max_abs_error(actual, &expected_held_out_b));
    let shuffled_mask_dispersion = shuffled_transform.as_ref().and_then(|transform| {
        mask_dispersion(&implied_masks(&anchors_b, &shuffled_a), transform)
    });

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
    let mixed_transform = frame.compute_xenobot_graft_transform(&mixed_targets, &anchors_a);
    let mixed_held_out = mixed_transform
        .as_ref()
        .map(|transform| frame.translate_foreign_hypervector(&held_out_a, transform));
    let mixed_mask_held_out_max = mixed_held_out
        .as_ref()
        .and_then(|actual| max_abs_error(actual, &expected_held_out_b));
    let mixed_mask_dispersion = mixed_transform.as_ref().and_then(|transform| {
        mask_dispersion(&implied_masks(&mixed_targets, &anchors_a), transform)
    });

    let mut corrupted_targets = anchors_b.clone();
    corrupted_targets[0].values[0] += 1.5;
    let corrupted_transform =
        frame.compute_xenobot_graft_transform(&corrupted_targets, &anchors_a);
    let corrupted_held_out = corrupted_transform
        .as_ref()
        .map(|transform| frame.translate_foreign_hypervector(&held_out_a, transform));
    let corrupted_anchor_held_out_max = corrupted_held_out
        .as_ref()
        .and_then(|actual| max_abs_error(actual, &expected_held_out_b));
    let corrupted_mask_dispersion = corrupted_transform.as_ref().and_then(|transform| {
        mask_dispersion(
            &implied_masks(&corrupted_targets, &anchors_a),
            transform,
        )
    });

    let near_floor_source = hv(&[5.0e-8, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.0]);
    let near_floor_target = near_floor_source.bind(&mask_ab);
    let near_floor_transform = frame.compute_xenobot_graft_transform(
        &[near_floor_target],
        std::slice::from_ref(&near_floor_source),
    );
    let near_floor_held_out = near_floor_transform
        .as_ref()
        .map(|transform| frame.translate_foreign_hypervector(&held_out_a, transform));
    let near_floor_held_out_max = near_floor_held_out
        .as_ref()
        .and_then(|actual| max_abs_error(actual, &expected_held_out_b));
    let near_floor_source_abs_values = near_floor_source
        .values
        .iter()
        .map(|value| value.abs())
        .collect::<Vec<_>>();
    let inverse_floor_affected_count = near_floor_source_abs_values
        .iter()
        .filter(|value| **value < INVERSE_FLOOR)
        .count();
    let inverse_floor_affected_rate =
        inverse_floor_affected_count as f32 / near_floor_source.dim() as f32;

    let permuted_targets = anchors_a
        .iter()
        .map(reverse_coordinates)
        .collect::<Vec<_>>();
    let permutation_estimate =
        frame.compute_xenobot_graft_transform(&permuted_targets, &anchors_a);
    let permutation_predicted = permutation_estimate
        .as_ref()
        .map(|transform| frame.translate_foreign_hypervector(&held_out_a, transform));
    let permutation_expected = reverse_coordinates(&held_out_a);
    let permutation_held_out_max = permutation_predicted
        .as_ref()
        .and_then(|actual| max_abs_error(actual, &permutation_expected));

    let nonfinite_source = hv(&[f32::NAN, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.0]);
    let nonfinite_target = nonfinite_source.bind(&mask_ab);
    let nonfinite_transform = frame.compute_xenobot_graft_transform(
        &[nonfinite_target],
        std::slice::from_ref(&nonfinite_source),
    );
    let nonfinite_output_nonfinite_count = nonfinite_transform.as_ref().map(|transform| {
        transform
            .values
            .iter()
            .filter(|value| !value.is_finite())
            .count()
    });

    let fields = vec![
        ("schema", "\"symthaea.rel.graft-frame-execution-observation.v3\"".to_string()),
        ("authority", "\"ExecutionOnly\"".to_string()),
        ("frozen_scientific_subject", "\"7f5826675b44dd0d3f702f62bc9818f7990e4a01\"".to_string()),
        ("frozen_test_blob", "\"787ea051ae0bf8e5667b6925481afd46c15d0bc4\"".to_string()),
        ("predicate_contract_head", "\"43bf1d588447f602ce0f1549986bb558a839762c\"".to_string()),
        ("positive_fixture_nonfinite_count", positive_fixture_nonfinite_count.to_string()),
        ("positive_fixture_max_abs", json_number(Some(positive_fixture_max_abs))),
        ("positive_source_abs_values", json_f32_slice(&positive_source_abs_values)),
        ("positive_inverse_floor_affected_count", positive_inverse_floor_affected_count.to_string()),
        ("transform_ab_present", json_bool(estimated_ab.is_some())),
        ("transform_ab_values", json_hv(estimated_ab.as_ref())),
        ("mask_recovery_max", json_number(mask_recovery_max)),
        ("mask_recovery_rms", json_number(mask_recovery_rms)),
        ("shared_mask_dispersion_max", json_number(shared_mask_dispersion.map(|value| value.0))),
        ("shared_mask_dispersion_rms", json_number(shared_mask_dispersion.map(|value| value.1))),
        ("construction_anchor_max", json_number(construction.map(|value| value.0))),
        ("construction_anchor_rms", json_number(construction.map(|value| value.1))),
        ("held_out_actual_b", json_hv(actual_held_out_b.as_ref())),
        ("held_out_max", json_number(held_out_max)),
        ("held_out_rms", json_number(held_out_rms)),
        ("transform_bc_present", json_bool(estimated_bc.is_some())),
        ("transform_bc_values", json_hv(estimated_bc.as_ref())),
        ("transform_ac_present", json_bool(estimated_ac.is_some())),
        ("transform_ac_values", json_hv(estimated_ac.as_ref())),
        ("direct_expected_mask_error", json_number(direct_expected_mask_error)),
        ("direct_vs_composed_mask_max", json_number(direct_vs_composed_mask_error)),
        ("direct_vs_sequential_transport_max", json_number(direct_vs_sequential_transport_error)),
        ("transform_ca_present", json_bool(estimated_ca.is_some())),
        ("transform_ca_values", json_hv(estimated_ca.as_ref())),
        ("loop_closure_max", json_number(loop_closure_max)),
        ("loop_closure_rms", json_number(loop_closure_rms)),
        ("bundle_covariance_defect", json_number(bundle_covariance_defect)),
        ("transported_unit_defect", json_number(transported_unit_defect)),
        ("fixed_hadamard_covariance_defect", json_number(fixed_hadamard_defect)),
        ("dressed_hadamard_covariance_defect", json_number(dressed_hadamard_defect)),
        ("dressed_unit_action_defect", json_number(dressed_unit_action_defect)),
        ("raw_cosine_drift", json_number(raw_cosine_drift)),
        ("pulled_back_cosine_drift", json_number(pulled_back_cosine_drift)),
        ("empty_anchor_result_is_none", json_bool(empty_anchor_result_is_none)),
        ("unequal_anchor_count_result_is_none", json_bool(unequal_anchor_count_result_is_none)),
        ("dimension_mismatch_panics", json_bool(dimension_mismatch_panics)),
        ("shuffled_transform_present", json_bool(shuffled_transform.is_some())),
        ("shuffled_transform_values", json_hv(shuffled_transform.as_ref())),
        ("shuffled_anchor_held_out_max", json_number(shuffled_held_out_max)),
        ("shuffled_mask_dispersion_max", json_number(shuffled_mask_dispersion.map(|value| value.0))),
        ("mixed_transform_present", json_bool(mixed_transform.is_some())),
        ("mixed_transform_values", json_hv(mixed_transform.as_ref())),
        ("mixed_mask_held_out_max", json_number(mixed_mask_held_out_max)),
        ("mixed_mask_dispersion_max", json_number(mixed_mask_dispersion.map(|value| value.0))),
        ("corrupted_transform_present", json_bool(corrupted_transform.is_some())),
        ("corrupted_transform_values", json_hv(corrupted_transform.as_ref())),
        ("corrupted_anchor_held_out_max", json_number(corrupted_anchor_held_out_max)),
        ("corrupted_mask_dispersion_max", json_number(corrupted_mask_dispersion.map(|value| value.0))),
        ("near_floor_transform_present", json_bool(near_floor_transform.is_some())),
        ("near_floor_transform_values", json_hv(near_floor_transform.as_ref())),
        ("near_floor_source_abs_values", json_f32_slice(&near_floor_source_abs_values)),
        ("near_floor_held_out_max", json_number(near_floor_held_out_max)),
        ("inverse_floor_affected_count", inverse_floor_affected_count.to_string()),
        ("inverse_floor_affected_rate", json_number(Some(inverse_floor_affected_rate))),
        ("permutation_transform_present", json_bool(permutation_estimate.is_some())),
        ("permutation_transform_values", json_hv(permutation_estimate.as_ref())),
        ("permutation_held_out_max", json_number(permutation_held_out_max)),
        ("nonfinite_transform_present", json_bool(nonfinite_transform.is_some())),
        ("nonfinite_transform_values", json_hv(nonfinite_transform.as_ref())),
        ("nonfinite_output_nonfinite_count", json_usize(nonfinite_output_nonfinite_count)),
        (
            "claims",
            "{\"observation_sealed\": false, \"comparison_only_adjudicated\": false, \"rel_005a_qualified\": false, \"scientific_pass\": false, \"scientific_fail\": false}".to_string(),
        ),
    ];

    let observation = emit_json(fields);
    println!("REL005A_EXECUTION_V3={observation}");
    if let Ok(path) = env::var("REL005A_EXECUTION_OBSERVATION_PATH") {
        fs::write(path, &observation).expect("write REL-005A ExecutionOnly v3 observation");
    }
}
