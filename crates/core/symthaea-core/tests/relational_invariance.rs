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

fn deterministic_permutation(dim: usize, seed: u64) -> Vec<usize> {
    assert!(dim > 0);
    let mut permutation: Vec<usize> = (0..dim).collect();
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;

    for i in (1..dim).rev() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let target = (state % (i as u64 + 1)) as usize;
        permutation.swap(i, target);
    }

    permutation
}

fn inverse_permutation(permutation: &[usize]) -> Vec<usize> {
    let mut inverse = vec![usize::MAX; permutation.len()];
    for (source, &target) in permutation.iter().enumerate() {
        assert!(target < permutation.len());
        assert_eq!(inverse[target], usize::MAX, "permutation is not injective");
        inverse[target] = source;
    }
    assert!(inverse.iter().all(|&index| index != usize::MAX));
    inverse
}

fn compose_permutations(first: &[usize], second: &[usize]) -> Vec<usize> {
    assert_eq!(first.len(), second.len());
    first.iter().map(|&target| second[target]).collect()
}

fn permute_continuous(hv: &ContinuousHV, permutation: &[usize]) -> ContinuousHV {
    assert_eq!(hv.values.len(), permutation.len());
    let mut values = vec![0.0; permutation.len()];
    for (source, &target) in permutation.iter().enumerate() {
        values[target] = hv.values[source];
    }
    ContinuousHV::from_vec(values)
}

fn binary_bit(hv: &BinaryHV, index: usize) -> u8 {
    (hv.0[index / 8] >> (index % 8)) & 1
}

fn permute_binary(hv: &BinaryHV, permutation: &[usize]) -> BinaryHV {
    assert_eq!(permutation.len(), BinaryHV::DIM);
    let mut result = BinaryHV::zero();
    for (source, &target) in permutation.iter().enumerate() {
        assert!(target < BinaryHV::DIM);
        if binary_bit(hv, source) == 1 {
            result.0[target / 8] |= 1 << (target % 8);
        }
    }
    result
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
fn rel_001_binary_permutation_family_preserves_core_structure() {
    let a = BinaryHV::random(0x5245_4c01);
    let b = BinaryHV::random(0x5245_4c02);
    let c = BinaryHV::random(0x5245_4c03);
    let original_bundle = BinaryHV::bundle(&[a, b, c]);

    for seed in [0x5245_4c10, 0x5245_4c11, 0x5245_4c12] {
        let permutation = deterministic_permutation(BinaryHV::DIM, seed);
        let inverse = inverse_permutation(&permutation);
        let pa = permute_binary(&a, &permutation);
        let pb = permute_binary(&b, &permutation);
        let pc = permute_binary(&c, &permutation);

        let transformed_bound = permute_binary(&a.bind(&b), &permutation);
        let bound_transformed = pa.bind(&pb);
        assert!(transformed_bound == bound_transformed);

        assert_eq!(a.hamming_distance(&b), pa.hamming_distance(&pb));
        assert_eq!(a.similarity(&b), pa.similarity(&pb));

        let transformed_bundle = permute_binary(&original_bundle, &permutation);
        let bundle_transformed = BinaryHV::bundle(&[pa, pb, pc]);
        assert!(transformed_bundle == bundle_transformed);

        assert!(permute_binary(&pa, &inverse) == a);
    }
}

#[test]
fn rel_001_continuous_permutation_family_preserves_core_structure() {
    let a = ContinuousHV::random(256, 0x5245_4c21);
    let b = ContinuousHV::random(256, 0x5245_4c22);
    let c = ContinuousHV::random(256, 0x5245_4c23);
    let original_bundle = ContinuousHV::bundle(&[&a, &b, &c]);

    for seed in [0x5245_4c30, 0x5245_4c31, 0x5245_4c32] {
        let permutation = deterministic_permutation(a.dim(), seed);
        let inverse = inverse_permutation(&permutation);
        let pa = permute_continuous(&a, &permutation);
        let pb = permute_continuous(&b, &permutation);
        let pc = permute_continuous(&c, &permutation);

        let transformed_bound = permute_continuous(&a.bind(&b), &permutation);
        let bound_transformed = pa.bind(&pb);
        assert_eq!(transformed_bound.values, bound_transformed.values);

        let before = a.similarity(&b);
        let after = pa.similarity(&pb);
        assert!(
            (before - after).abs() <= 1.0e-5,
            "coordinate permutation changed cosine beyond narrow roundoff: before={before}, after={after}"
        );

        let transformed_bundle = permute_continuous(&original_bundle, &permutation);
        let bundle_transformed = ContinuousHV::bundle(&[&pa, &pb, &pc]);
        assert_eq!(transformed_bundle.values, bundle_transformed.values);

        assert_eq!(permute_continuous(&pa, &inverse).values, a.values);
    }
}

#[test]
fn rel_001_permutation_composition_is_a_consistent_group_action() {
    let continuous = ContinuousHV::random(256, 0x5245_4c41);
    let binary = BinaryHV::random(0x5245_4c42);

    let continuous_p = deterministic_permutation(continuous.dim(), 0x5245_4c43);
    let continuous_q = deterministic_permutation(continuous.dim(), 0x5245_4c44);
    let continuous_composed = compose_permutations(&continuous_p, &continuous_q);
    assert_eq!(
        permute_continuous(
            &permute_continuous(&continuous, &continuous_p),
            &continuous_q,
        )
        .values,
        permute_continuous(&continuous, &continuous_composed).values
    );

    let binary_p = deterministic_permutation(BinaryHV::DIM, 0x5245_4c45);
    let binary_q = deterministic_permutation(BinaryHV::DIM, 0x5245_4c46);
    let binary_composed = compose_permutations(&binary_p, &binary_q);
    assert!(
        permute_binary(&permute_binary(&binary, &binary_p), &binary_q)
            == permute_binary(&binary, &binary_composed)
    );
}

#[test]
fn rel_001_sequence_fixed_symmetry_and_covariance_are_distinct_claims() {
    let x = ContinuousHV::from_vec(vec![0.11, -0.23, 0.37, -0.41, 0.59, -0.61, 0.73, -0.89]);

    // Powers of the fixed cyclic shift commute exactly with the one-step shift.
    for shift in [0, 1, 3, 7] {
        assert_eq!(
            x.permute(1).permute(shift).values,
            x.permute(shift).permute(1).values
        );
    }

    // A general coordinate permutation need not commute with the fixed rho.
    let permutation = deterministic_permutation(x.dim(), 0x5245_4c51);
    let inverse = inverse_permutation(&permutation);
    let g_rho_x = permute_continuous(&x.permute(1), &permutation);
    let gx = permute_continuous(&x, &permutation);
    let rho_g_x = gx.permute(1);
    assert_ne!(g_rho_x.values, rho_g_x.values);

    // Covariance is a different statement: rho' = g rho g^-1.
    let rho_prime_gx = permute_continuous(
        &permute_continuous(&gx, &inverse).permute(1),
        &permutation,
    );
    assert_eq!(g_rho_x.values, rho_prime_gx.values);
}

#[test]
fn rel_003_continuous_metric_isometry_does_not_imply_hadamard_automorphism() {
    let a = ContinuousHV::from_vec(vec![0.21, -0.74, 0.43, 0.88, -0.52, 0.17, 0.69, -0.31]);
    let b = ContinuousHV::from_vec(vec![-0.63, 0.28, 0.91, -0.36, 0.44, -0.82, 0.13, 0.57]);

    let before = a.similarity(&b);
    let after = orthogonal_pair_mix(&a).similarity(&orthogonal_pair_mix(&b));
    assert!(
        (before - after).abs() <= 2.0e-6,
        "orthogonal control failed cosine isometry: before={before}, after={after}"
    );

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
    let a = BinaryHV::random(0x5245_4c61);
    let b = BinaryHV::random(0x5245_4c62);

    let transformed_bound = binary_xor_linear_shear(&a.bind(&b));
    let bound_transformed = binary_xor_linear_shear(&a).bind(&binary_xor_linear_shear(&b));
    assert!(transformed_bound == bound_transformed);

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
