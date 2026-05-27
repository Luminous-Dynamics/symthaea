// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Embodied Cognitive Cycle Benchmark
//!
//! Measures the performance cost of closing the proprioceptive loop:
//! - Baseline: cognitive cycle without embodiment
//! - Humanoid: cognitive cycle + 21-DOF bipedal motor bridge
//! - Overhead: isolation of embodiment cost
//!
//! ## Usage
//! ```bash
//! cargo bench --bench embodied_cycle --features humanoid
//! cargo bench --bench embodied_cycle --features humanoid -- --save-baseline main
//! ```

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use symthaea::cognitive_loop::motor_bridge::EmbodimentPlatform;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

fn warm_up(service: &mut CognitiveLoopService, cycles: usize) {
    for _ in 0..cycles {
        let _ = service.cycle("warm up");
    }
}

fn bench_embodied_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("embodied_cycle");
    group.sample_size(20);

    // Baseline: no embodiment
    group.bench_function("disembodied", |b| {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            async_training: false,
            learning_threshold: 0.0,
            ..Default::default()
        })
        .unwrap();
        warm_up(&mut service, 5);

        b.iter(|| black_box(service.cycle("cognitive cycle baseline")))
    });

    // Humanoid: 21-DOF motor bridge
    group.bench_function("humanoid_21dof", |b| {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            embodiment_platform: EmbodimentPlatform::Humanoid,
            embodiment_blend_weight: 0.2,
            embodiment_step_interval: 1,
            async_training: false,
            learning_threshold: 0.0,
            ..Default::default()
        })
        .unwrap();
        warm_up(&mut service, 5);

        b.iter(|| black_box(service.cycle("embodied humanoid cycle")))
    });

    group.finish();
}

criterion_group!(benches, bench_embodied_cycle);
criterion_main!(benches);
