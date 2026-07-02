#![cfg(feature = "bench-fixtures")]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use vsv_core::{
    calibration,
    loss::{loss_delta_mse, mse_single},
    CanaryCNN, Fixed,
};

fn build_logits() -> (Vec<Fixed>, Vec<Fixed>, usize) {
    let model = CanaryCNN::pretrained();
    let input = calibration::CALIBRATION_INPUTS[0];
    let label = calibration::CALIBRATION_LABELS[0];

    let logits = model.forward(&input);
    let logits_vec = logits.to_vec();
    let mut improved = logits_vec.clone();

    for val in &mut improved {
        *val = *val - Fixed::from_f32(0.01);
    }

    (logits_vec, improved, label)
}

fn bench_mse_single(c: &mut Criterion) {
    let (logits, _, label) = build_logits();
    c.bench_function("mse_single", |b| {
        b.iter(|| mse_single(black_box(&logits), black_box(label)))
    });
}

fn bench_loss_delta(c: &mut Criterion) {
    let (before, after, label) = build_logits();
    c.bench_function("loss_delta_mse", |b| {
        b.iter(|| loss_delta_mse(black_box(&before), black_box(&after), black_box(label)))
    });
}

criterion_group!(loss_benches, bench_mse_single, bench_loss_delta);
criterion_main!(loss_benches);
