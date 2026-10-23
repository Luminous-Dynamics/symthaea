// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent public-surface conformance tests for HDC-ENC-001.
//!
//! The golden fixture below was preregistered in PR #3629 before any executable
//! Rust result for the linear-v1 implementation was observed. Its expected
//! values were derived independently from the public Genesis SHAKE-256 rule and
//! the specified Fisher-Yates/rejection algorithm.

use symthaea_core::{
    genesis::GenesisSeed,
    hdc::unified_hv::ContinuousHV,
    hdc_encoding::LinearLevelCodebook,
};

fn cosine(a: &ContinuousHV, b: &ContinuousHV) -> f32 {
    assert_eq!(a.dim(), b.dim());
    let mut dot = 0.0f32;
    let mut aa = 0.0f32;
    let mut bb = 0.0f32;
    for (&x, &y) in a.values.iter().zip(&b.values) {
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    dot / (aa * bb).sqrt()
}

fn norm(v: &ContinuousHV) -> f32 {
    v.values.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[test]
fn independent_linear_v1_known_answer_matches_public_api() {
    let genesis = GenesisSeed::from_phrase("hdc-enc-001-independent-linear-golden");
    let codebook = LinearLevelCodebook::from_genesis(&genesis, "golden::ordered", 5, 16)
        .expect("preregistered fixture must construct");
    let encoded = codebook
        .encode_normalized(0.5)
        .expect("preregistered fixture input must encode");

    assert_eq!(encoded.selected_level, 2);
    assert_eq!(encoded.flipped_components, 4);

    let expected_bits = [
        0xbe80_0000,
        0xbe80_0000,
        0xbe80_0000,
        0x3e80_0000,
        0xbe80_0000,
        0xbe80_0000,
        0xbe80_0000,
        0x3e80_0000,
        0x3e80_0000,
        0xbe80_0000,
        0xbe80_0000,
        0x3e80_0000,
        0x3e80_0000,
        0xbe80_0000,
        0xbe80_0000,
        0x3e80_0000,
    ];
    let actual_bits: Vec<u32> = encoded
        .representation
        .values
        .iter()
        .map(|value| value.to_bits())
        .collect();

    assert_eq!(actual_bits.as_slice(), expected_bits.as_slice());
}

#[test]
fn shared_binding_by_level_code_is_scaled_cosine_isometry() {
    const DIM: usize = 1024;
    let genesis = GenesisSeed::from_phrase("hdc-enc-001-binding-conformance");
    let codebook = LinearLevelCodebook::from_genesis(&genesis, "binding::ordered", 17, DIM)
        .expect("binding fixture must construct");
    let level = codebook
        .encode_normalized(0.375)
        .expect("binding fixture input must encode");

    let a = ContinuousHV::random(DIM, 11);
    let b = ContinuousHV::random(DIM, 29);
    let bound_a = a.bind(&level.representation);
    let bound_b = b.bind(&level.representation);

    let before = cosine(&a, &b);
    let after = cosine(&bound_a, &bound_b);
    assert!(
        (before - after).abs() < 2.0e-5,
        "shared level binding changed cosine: before={before}, after={after}"
    );

    let expected_scale = 1.0f32 / (DIM as f32).sqrt();
    let observed_scale_a = norm(&bound_a) / norm(&a);
    let observed_scale_b = norm(&bound_b) / norm(&b);
    assert!((observed_scale_a - expected_scale).abs() < 2.0e-6);
    assert!((observed_scale_b - expected_scale).abs() < 2.0e-6);
}
