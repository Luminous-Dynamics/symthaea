// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_hdc_ltc::{TemporalAxis, TemporalPhasor, UnitaryRole};

const TOL: f64 = 1e-12;

#[test]
fn public_temporal_axis_is_deterministic_and_continuous() {
    let a = TemporalAxis::new(256, 0x5449_4D45).unwrap();
    let b = TemporalAxis::new(256, 0x5449_4D45).unwrap();
    assert_eq!(a, b);

    let t1 = a.at(1.125).unwrap();
    let t2 = a.at(2.375).unwrap();
    let composed = t1.bind(&t2).unwrap();
    let direct = a.at(3.5).unwrap();
    assert!(composed.max_abs_difference(&direct).unwrap() < TOL);
}

#[test]
fn public_temporal_inverse_recovers_identity() {
    let axis = TemporalAxis::new(256, 0x494E_5652).unwrap();
    let point = axis.at(-91.75).unwrap();
    let identity = TemporalPhasor::identity(axis.dim()).unwrap();
    let recovered = point.bind(&point.inverse()).unwrap();
    assert!(recovered.max_abs_difference(&identity).unwrap() < TOL);
}

#[test]
fn public_bipolar_role_commutes_with_temporal_binding() {
    let axis = TemporalAxis::new(256, 0x524F_4C45).unwrap();
    let role = UnitaryRole::new(axis.dim(), 0x4844_4352);
    let state_time = axis.at(4.25).unwrap();
    let delta_time = axis.at(0.375).unwrap();

    let bind_role_first = state_time
        .bind_role(&role)
        .unwrap()
        .bind(&delta_time)
        .unwrap();
    let bind_time_first = state_time
        .bind(&delta_time)
        .unwrap()
        .bind_role(&role)
        .unwrap();

    assert!(
        bind_role_first
            .max_abs_difference(&bind_time_first)
            .unwrap()
            < TOL
    );
}

#[test]
fn public_validity_interval_translates_under_same_group_action() {
    let axis = TemporalAxis::new(256, 0x494E_5456).unwrap();
    let validity = axis.interval(2.0, 5.5).unwrap();
    let shift = axis.at(11.25).unwrap();
    let translated = shift.bind_interval(&validity).unwrap();
    let expected = axis.interval(13.25, 16.75).unwrap();
    assert!(translated.max_abs_difference(&expected).unwrap() < TOL);
}

#[test]
fn public_interval_query_is_finite_without_claiming_historical_memory() {
    let axis = TemporalAxis::new(512, 0x5343_4F52).unwrap();
    let validity = axis.interval(10.0, 20.0).unwrap();
    for time in [0.0, 10.0, 12.5, 19.999, 20.0, 30.0] {
        let score = validity.score_at(&axis.at(time).unwrap()).unwrap();
        assert!(score.is_finite());
    }

    // Deliberately no assertion that a point inside the interval must always
    // outrank every point outside it. That retrieval theorem belongs to the next
    // temporal-memory tranche, not to this algebra substrate.
}
