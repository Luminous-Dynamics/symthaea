// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Public-contract tests for the real Hadamard-unitary HLS role algebra.
//!
//! These tests intentionally use only the crate's exported API. They establish
//! representation-level invariants before any claim is made that temporal
//! evolution itself is binding- or permutation-equivariant.

use symthaea_hdc_ltc::{ContinuousHV, UnitaryRole};

#[test]
fn role_binding_is_an_isometry() {
    let role = UnitaryRole::new(16_384, 0xA11CE);
    let x = ContinuousHV::new_random(16_384, 1);
    let y = ContinuousHV::new_random(16_384, 2);

    let bx = role.bind(&x);
    let by = role.bind(&y);

    assert_eq!(bx.norm(), x.norm());
    assert_eq!(by.norm(), y.norm());
    assert_eq!(bx.dot(&by), x.dot(&y));
    assert!((bx.similarity(&by) - x.similarity(&y)).abs() < 1e-7);
}

#[test]
fn role_binding_is_exactly_self_inverse() {
    let role = UnitaryRole::new(16_384, 0xB1D);
    let value = ContinuousHV::new_random(16_384, 99);
    let recovered = role.unbind(&role.bind(&value));
    assert_eq!(recovered, value);
}

#[test]
fn composed_roles_match_sequential_binding() {
    let r1 = UnitaryRole::new(4096, 11);
    let r2 = UnitaryRole::new(4096, 12);
    let value = ContinuousHV::new_random(4096, 13);

    let sequential = r2.bind(&r1.bind(&value));
    let composed = r1.compose(&r2).bind(&value);
    assert_eq!(sequential, composed);
}

#[test]
fn permutation_preserves_role_invariant_and_binding_norm() {
    let role = UnitaryRole::new(4096, 21).permute(137);
    let value = ContinuousHV::new_random(4096, 22);

    assert!(role
        .as_slice()
        .iter()
        .all(|component| *component == -1.0 || *component == 1.0));
    assert_eq!(role.bind(&value).norm(), value.norm());
}

#[test]
fn malformed_serialized_roles_fail_closed() {
    let malformed = "[1.0,-1.0,0.25,1.0]";
    assert!(serde_json::from_str::<UnitaryRole>(malformed).is_err());
}
