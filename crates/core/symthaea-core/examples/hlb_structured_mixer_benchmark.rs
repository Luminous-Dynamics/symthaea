// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Characterization of fixed and parameterized orthogonal HDC mixing operators.
//!
//! This measures operator fidelity, basis-coordinate spread, parameter count,
//! and end-to-end CPU latency (including output allocation). It does not measure
//! downstream task quality and must not be treated as an HDC-LTC performance
//! claim. Hardware and compiler identity belong in the CI/source receipt.

use std::hint::black_box;
use std::time::Instant;

use symthaea_core::hdc::structured_mixer::{
    HdcButterflyMixer, HdcStructuredMixer, HdcStructuredMixerKind,
};
use symthaea_core::hdc::unified_hv::ContinuousHV;

const DIMENSIONS: &[usize] = &[256, 1_024, 4_096, 16_384];
const WARMUP_ITERATIONS: usize = 4;

fn main() {
    println!(
        "dimension,operator,trainable_angle_count,butterfly_pair_count,dense_matrix_parameter_count,basis_spread,round_trip_max_abs_error,relative_norm_error,latency_ns_per_apply,timed_iterations"
    );

    for &dimension in DIMENSIONS {
        let input = ContinuousHV::random(dimension, 0x51A7 + dimension as u64);
        let basis = ContinuousHV::basis(7, dimension);
        let butterfly_pairs = (dimension / 2) * (dimension.trailing_zeros() as usize);
        let dense_parameters = dimension * dimension;
        let iterations = if dimension >= 16_384 { 16 } else { 48 };

        for (name, kind) in [
            ("identity", HdcStructuredMixerKind::Identity),
            (
                "cyclic_permutation",
                HdcStructuredMixerKind::CyclicPermutation { shift: 17 },
            ),
            ("walsh_hadamard", HdcStructuredMixerKind::WalshHadamard),
        ] {
            let mixer = HdcStructuredMixer::new(kind, dimension).expect("valid fixed mixer");
            let run = |value: &ContinuousHV| mixer.apply(value).expect("fixed mixer apply");
            let undo = |value: &ContinuousHV| mixer.inverse(value).expect("fixed mixer inverse");
            report_case(
                dimension,
                name,
                0,
                if matches!(kind, HdcStructuredMixerKind::WalshHadamard) {
                    butterfly_pairs
                } else {
                    0
                },
                dense_parameters,
                &input,
                &basis,
                &run,
                &undo,
                iterations,
            );
        }

        let angle_count = HdcButterflyMixer::parameter_count_for_dimension(dimension);
        let angles = (0..angle_count)
            .map(|index| {
                let phase = (index as f32 * 0.017 + dimension as f32 * 0.000_13).sin();
                0.55 * phase
            })
            .collect::<Vec<_>>();
        let butterfly = HdcButterflyMixer::new(dimension, angles).expect("valid butterfly");
        let run = |value: &ContinuousHV| butterfly.apply(value).expect("butterfly apply");
        let undo = |value: &ContinuousHV| butterfly.inverse(value).expect("butterfly inverse");
        report_case(
            dimension,
            "parameterized_orthogonal_butterfly",
            butterfly.parameter_count(),
            butterfly_pairs,
            dense_parameters,
            &input,
            &basis,
            &run,
            &undo,
            iterations,
        );
    }
}

fn report_case(
    dimension: usize,
    name: &str,
    trainable_angles: usize,
    butterfly_pairs: usize,
    dense_parameters: usize,
    input: &ContinuousHV,
    basis: &ContinuousHV,
    apply: &impl Fn(&ContinuousHV) -> ContinuousHV,
    inverse: &impl Fn(&ContinuousHV) -> ContinuousHV,
    iterations: usize,
) {
    for _ in 0..WARMUP_ITERATIONS {
        black_box(apply(black_box(input)));
    }

    let started = Instant::now();
    for _ in 0..iterations {
        black_box(apply(black_box(input)));
    }
    let latency_ns = started.elapsed().as_nanos() / iterations as u128;

    let transformed = apply(input);
    let restored = inverse(&transformed);
    let round_trip_max_abs_error = input
        .values
        .iter()
        .zip(&restored.values)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    let input_norm = input.norm();
    let relative_norm_error = if input_norm > f32::EPSILON {
        (transformed.norm() - input_norm).abs() / input_norm
    } else {
        0.0
    };
    let spread = apply(basis)
        .values
        .iter()
        .filter(|value| value.abs() > 1e-6)
        .count();

    println!(
        "{dimension},{name},{trainable_angles},{butterfly_pairs},{dense_parameters},{spread},{round_trip_max_abs_error:.9},{relative_norm_error:.9},{latency_ns},{iterations}"
    );
}
