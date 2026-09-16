// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! REL-005A red-run diagnostic.
//!
//! MeasurementOnly reproduction of the corrupted-anchor case that first failed
//! in exact-subject run 34919245190. This file deliberately carries no
//! effect-size qualification threshold: its job is to preserve the production
//! response before any interpretation or contract revision.

use std::env;
use std::fs;

use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_fep::SwarmCoalition;

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

#[test]
fn rel_005a_corrupted_anchor_measurement_only() {
    let frame = coalition();
    let mask_ab = hv(&[0.5, -0.75, 0.9, -0.85, 0.8, -0.7, 0.95, -0.6]);
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

    let held_out_a = hv(&[0.55, -0.65, 0.75, -0.85, 0.95, -1.0, 0.9, -0.8]);
    let expected_held_out_b = held_out_a.bind(&mask_ab);

    let mut corrupted_targets = anchors_b.clone();
    corrupted_targets[0].values[0] += 1.5;

    let implied_masks = corrupted_targets
        .iter()
        .zip(&anchors_a)
        .map(|(native, foreign)| native.bind(&foreign.inverse()))
        .collect::<Vec<_>>();
    let implied_mask_first_coordinates = implied_masks
        .iter()
        .map(|mask| mask.values[0])
        .collect::<Vec<_>>();

    let corrupted_transform = frame
        .compute_xenobot_graft_transform(&corrupted_targets, &anchors_a)
        .expect("corrupted equal-size anchors must produce the current best-fit transform");
    let corrupted_held_out = frame.translate_foreign_hypervector(&held_out_a, &corrupted_transform);
    let corrupted_anchor_held_out_max = max_abs_error(&corrupted_held_out, &expected_held_out_b);

    let mut corrupted_mask_dispersion_max = 0.0_f32;
    for implied in &implied_masks {
        corrupted_mask_dispersion_max =
            corrupted_mask_dispersion_max.max(max_abs_error(implied, &corrupted_transform));
    }

    let scalar_prediction_transform_first = 1.4375_f32;
    let scalar_prediction_held_out_max = 0.515625_f32;

    let metrics = format!(
        concat!(
            "{{\n",
            "  \"schema\": \"symthaea.rel.graft-frame-red-diagnostic.v1\",\n",
            "  \"authority\": \"MeasurementOnly\",\n",
            "  \"source_run\": 34919245190,\n",
            "  \"corrupted_target_first\": {corrupted_target_first:.9e},\n",
            "  \"source_anchor_first\": {source_anchor_first:.9e},\n",
            "  \"implied_mask_first_coordinates\": {implied_mask_first_coordinates:?},\n",
            "  \"fitted_transform_first\": {fitted_transform_first:.9e},\n",
            "  \"scalar_prediction_transform_first\": {scalar_prediction_transform_first:.9e},\n",
            "  \"held_out_source_first\": {held_out_source_first:.9e},\n",
            "  \"held_out_expected_first\": {held_out_expected_first:.9e},\n",
            "  \"held_out_actual_first\": {held_out_actual_first:.9e},\n",
            "  \"corrupted_anchor_held_out_max\": {corrupted_anchor_held_out_max:.9e},\n",
            "  \"scalar_prediction_held_out_max\": {scalar_prediction_held_out_max:.9e},\n",
            "  \"corrupted_mask_dispersion_max\": {corrupted_mask_dispersion_max:.9e},\n",
            "  \"transform_values\": {transform_values:?}\n",
            "}}\n"
        ),
        corrupted_target_first = corrupted_targets[0].values[0],
        source_anchor_first = anchors_a[0].values[0],
        implied_mask_first_coordinates = implied_mask_first_coordinates,
        fitted_transform_first = corrupted_transform.values[0],
        scalar_prediction_transform_first = scalar_prediction_transform_first,
        held_out_source_first = held_out_a.values[0],
        held_out_expected_first = expected_held_out_b.values[0],
        held_out_actual_first = corrupted_held_out.values[0],
        corrupted_anchor_held_out_max = corrupted_anchor_held_out_max,
        scalar_prediction_held_out_max = scalar_prediction_held_out_max,
        corrupted_mask_dispersion_max = corrupted_mask_dispersion_max,
        transform_values = corrupted_transform.values,
    );

    println!("REL005A_RED_DIAGNOSTIC={metrics}");
    if let Ok(path) = env::var("REL005A_DIAGNOSTIC_PATH") {
        fs::write(path, &metrics).expect("write REL-005A red diagnostic artifact");
    }
}
