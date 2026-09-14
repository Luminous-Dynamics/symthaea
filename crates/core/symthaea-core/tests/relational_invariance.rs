// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! REL-001/002/003: independent relational-invariance qualification surface.
//!
//! This integration test intentionally keeps transformation/oracle logic outside
//! production HDC methods. It tests existing public operations without adding a
//! new production representation or changing cognition paths.
//!
//! Claim boundary:
//! - coordinate permutations are tested as representation-specific candidate
//!   automorphisms/isometries;
//! - metric preservation alone is NOT treated as algebra preservation;
//! - binding preservation alone is NOT treated as metric preservation;
//! - covariance under a transformed sequence operator is kept distinct from
//!   commutation with the fixed production sequence operator.

use symthaea_core::hdc::{binary_hv::BinaryHV, unified_hv::ContinuousHV};

/// Reverse the 2,048 storage bytes. This is a deterministic permutation of
/// BinaryHV coordinates in 8-bit blocks; no production permutation helper is
/// used by the oracle transformation itself.
fn reverse_binary_coordinates(hv: &BinaryHV) -> BinaryHV {
    let mut bytes = hv.0;
    bytes.reverse();
    BinaryHV(bytes)
}

/// Reverse continuous coordinates without using the production `permute`
/// operation. This is an involution, so it is its own inverse.
fn reverse_continuous_coordinates(hv: &ContinuousHV) -> ContinuousHV {
    let mut values = hv.values.clone();
    values.reverse();
    ContinuousHV::from_vec(values)
}

/// Invertible XOR-linear shear over the first two binary coordinates:
///
///     y0 = x0 XOR x1
///     y1 = x1
///
/// with all remaining bits unchanged. Applying the map twice recovers the
/// input. It preserves XOR composition exactly, but it is not a Hamming
/// isometry in general.
fn binary_xor_linear_shear(hv: &BinaryHV) -> BinaryHV {
    let mut bytes = hv.0;
    let x0 = bytes[0] & 1;
    let x1 = (bytes[0] >> 1) & 1;
    let y0 = x0 ^ x1;
    bytes[0] = (bytes[0] & !1) | y0;
    BinaryHV(bytes)
}

/// Pairwise 45-degree orthogonal mixing. The same 2x2 orthogonal block is
/// applied to every adjacent coordinate pair. This preserves Euclidean inner
/// products/cosine up to floating-point roundoff but does not generally
/// preserve Hadamard binding.
fn orthogonal_pair_mix(hv: &ContinuousHV) -> ContinuousHV {
    assert_eq!(hv.values.len() % 2, 0, "fixture dimension must be even");
    let scale = std::f32::consts::FRAC_1_SQRT_2;
    let mut values = vec![0.0; hv.values.len()];

    for i in (0..hv.values.len()).step_by(2) {
        let a = hv.values[i];
        let b = hv.values[i + 1];
        values[i] = (a + b) * scale;
        values[i + 1] = (a - b) * scale;
    }

    ContinuousHV::from_vec(values)
}

fn max_abs_component_error(a: &ContinuousHV, b: &ContinuousHV) -> f32 {
    assert_eq!(a.values.len(), b.values.len());
    a.values
        .iter()
        .zip(&b.values)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn rel_001_binary_coordinate_permutation_preserves_binding_and_hamming_geometry() {
    let a = BinaryHV::random(0x5245_4c01);
    let b = BinaryHV::random(0x5245_4c02);

    // Algebra automorphism: g(a XOR b) = g(a) XOR g(b).
    let transformed_bound = reverse_binary_coordinates(&a.bind(&b));
    let bound_transformed = reverse_binary_coordinates(&a).bind(&reverse_binary_coordinates(&b));
    assert!(transformed_bound == bound_transformed);

    // Metric isometry: a common coordinate permutation preserves every
    // Hamming match/mismatch exactly.
    assert_eq!(
        a.hamming_distance(&b),
        reverse_binary_coordinates(&a).hamming_distance(&reverse_binary_coordinates(&b))
    );
    assert_eq!(
        a.similarity(&b),
        reverse_binary_coordinates(&a).similarity(&reverse_binary_coordinates(&b))
    );
}

#[test]
fn rel_001_continuous_coordinate_permutation_preserves_binding_and_cosine_geometry() {
    let a = ContinuousHV::random(256, 0x5245_4c11);
    let b = ContinuousHV::random(256, 0x5245_4c12);

    // Hadamard multiplication is coordinatewise, so common coordinate
    // permutations are exact algebra automorphisms.
    let transformed_bound = reverse_continuous_coordinates(&a.bind(&b));
    let bound_transformed =
        reverse_continuous_coordinates(&a).bind(&reverse_continuous_coordinates(&b));
    assert_eq!(transformed_bound.values, bound_transformed.values);

    // The implementation accumulates floating-point dot products/norms in
    // coordinate order, so reversing that order can change only roundoff.
    let before = a.similarity(&b);
    let after = reverse_continuous_coordinates(&a).similarity(&reverse_continuous_coordinates(&b));
    assert!(
        (before - after).abs() <= 2.0e-6,
        "coordinate permutation changed cosine beyond roundoff: before={before}, after={after}"
    );
}

#[test]
fn rel_001_sequence_fixed_operator_and_covariant_operator_are_distinct_claims() {
    let x = ContinuousHV::from_vec(vec![0.11, -0.23, 0.37, -0.41, 0.59, -0.61, 0.73, -0.89]);

    // For the chosen frame change g (coordinate reversal), g and the fixed
    // cyclic production shift rho do not commute on this non-degenerate fixture.
    let g_rho_x = reverse_continuous_coordinates(&x.permute(1));
    let rho_g_x = reverse_continuous_coordinates(&x).permute(1);
    assert_ne!(g_rho_x.values, rho_g_x.values);

    // Covariance is a different statement. Define rho' = g rho g^-1.
    // Reversal is involutive, so g^-1 = g. Then rho'(g x) must equal g(rho x).
    let gx = reverse_continuous_coordinates(&x);
    let rho_prime_gx =
        reverse_continuous_coordinates(&reverse_continuous_coordinates(&gx).permute(1));
    assert_eq!(g_rho_x.values, rho_prime_gx.values);
}

#[test]
fn rel_003_continuous_metric_isometry_does_not_imply_hadamard_automorphism() {
    let a = ContinuousHV::from_vec(vec![0.21, -0.74, 0.43, 0.88, -0.52, 0.17, 0.69, -0.31]);
    let b = ContinuousHV::from_vec(vec![-0.63, 0.28, 0.91, -0.36, 0.44, -0.82, 0.13, 0.57]);

    // Positive control for the partial claim: the pairwise transform is
    // orthogonal, so cosine is preserved up to narrow f32 roundoff.
    let before = a.similarity(&b);
    let after = orthogonal_pair_mix(&a).similarity(&orthogonal_pair_mix(&b));
    assert!(
        (before - after).abs() <= 2.0e-6,
        "orthogonal control failed cosine isometry: before={before}, after={after}"
    );

    // Negative control for the stronger invalid claim: generic orthogonal
    // mixing does not distribute over element-wise/Hadamard multiplication.
    let transformed_bound = orthogonal_pair_mix(&a.bind(&b));
    let bound_transformed = orthogonal_pair_mix(&a).bind(&orthogonal_pair_mix(&b));
    let error = max_abs_component_error(&transformed_bound, &bound_transformed);
    assert!(
        error > 5.0e-2,
        "fixture failed to falsify Hadamard automorphism; max component error={error}"
    );
}

#[test]
fn rel_003_binary_xor_automorphism_does_not_imply_hamming_isometry() {
    let a = BinaryHV::random(0x5245_4c31);
    let b = BinaryHV::random(0x5245_4c32);

    // Positive control for the partial claim: the shear is F2-linear, so it
    // distributes over XOR binding exactly.
    let transformed_bound = binary_xor_linear_shear(&a.bind(&b));
    let bound_transformed = binary_xor_linear_shear(&a).bind(&binary_xor_linear_shear(&b));
    assert!(transformed_bound == bound_transformed);

    // Negative control for the stronger claim. A one-bit difference at x1
    // becomes a two-bit difference at (y0, y1).
    let zero = BinaryHV::zero();
    let mut one_bit = BinaryHV::zero();
    one_bit.0[0] = 0b0000_0010;

    assert_eq!(zero.hamming_distance(&one_bit), 1);
    assert_eq!(
        binary_xor_linear_shear(&zero).hamming_distance(&binary_xor_linear_shear(&one_bit)),
        2
    );
    assert_ne!(
        zero.similarity(&one_bit),
        binary_xor_linear_shear(&zero).similarity(&binary_xor_linear_shear(&one_bit))
    );
}
