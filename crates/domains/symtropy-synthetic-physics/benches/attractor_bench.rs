// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use symtropy_synthetic_physics::{
    circuit_breakers::GraphSafetyGuards, run_experiment, update_rules::UpdateRule,
};

fn bench_triangulation_100_ticks(c: &mut Criterion) {
    c.bench_function("triangulation_pressure_100_ticks", |b| {
        b.iter(|| {
            run_experiment(
                black_box(UpdateRule::TriangulationPressure { probability: 0.1 }),
                black_box(100),
                black_box(42),
                black_box(GraphSafetyGuards::default()),
            )
        })
    });
}

fn bench_free_energy_100_ticks(c: &mut Criterion) {
    c.bench_function("free_energy_minimization_100_ticks", |b| {
        b.iter(|| {
            run_experiment(
                black_box(UpdateRule::FreeEnergyMinimization {
                    candidates_per_tick: 10,
                }),
                black_box(100),
                black_box(42),
                black_box(GraphSafetyGuards::default()),
            )
        })
    });
}

criterion_group!(
    benches,
    bench_triangulation_100_ticks,
    bench_free_energy_100_ticks
);
criterion_main!(benches);
