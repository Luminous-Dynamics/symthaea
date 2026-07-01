// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Bridge Integration Benchmarks
//!
//! Run with: `cargo bench --bench bridge_integration -p symthaea-core`
//!
//! Benchmarks for the three bridge modules:
//! - math_bridge: UnifiedMathEngine operations across all 5 domains
//! - simulation_bridge: PhysicsSimulator for each system type
//! - Cross-bridge: full pipeline (simulate → encode → process)

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::dynamical_system::HarmonicOscillator;
use symthaea_core::hdc::math_bridge::{MathValue, UnifiedMathEngine};
use symthaea_core::physics::simulation_bridge::{
    PhysicsSimulator, SimulationAnalysis, state_to_binary_hv,
};

// =============================================================================
// MATH BRIDGE BENCHMARKS
// =============================================================================

fn bench_math_engine_creation(c: &mut Criterion) {
    c.bench_function("math_engine_new", |b| {
        b.iter(|| {
            let engine = UnifiedMathEngine::new();
            black_box(engine)
        });
    });
}

fn bench_math_add_naturals(c: &mut Criterion) {
    let engine = UnifiedMathEngine::new();
    let a = MathValue::Natural(42);
    let b = MathValue::Natural(58);
    c.bench_function("math_add_naturals", |bench| {
        bench.iter(|| {
            let result = engine.add(black_box(&a), black_box(&b));
            black_box(result)
        });
    });
}

fn bench_math_multiply_complex(c: &mut Criterion) {
    let engine = UnifiedMathEngine::new();
    let a = MathValue::Complex { re: 3.0, im: 4.0 };
    let b = MathValue::Complex { re: 1.0, im: -2.0 };
    c.bench_function("math_multiply_complex", |bench| {
        bench.iter(|| {
            let result = engine.multiply(black_box(&a), black_box(&b));
            black_box(result)
        });
    });
}

fn bench_math_sqrt_promotion(c: &mut Criterion) {
    let engine = UnifiedMathEngine::new();
    let a = MathValue::Integer(-4);
    c.bench_function("math_sqrt_negative_promotion", |bench| {
        bench.iter(|| {
            let result = engine.sqrt(black_box(&a));
            black_box(result)
        });
    });
}

fn bench_math_domain_chain(c: &mut Criterion) {
    let engine = UnifiedMathEngine::new();
    c.bench_function("math_full_domain_chain", |bench| {
        bench.iter(|| {
            // N → Z → Q → R → C in sequence
            let r1 = engine.add(&MathValue::Natural(10), &MathValue::Natural(5));
            let r2 = engine.subtract(&MathValue::Natural(3), &MathValue::Natural(7));
            let r3 = engine.divide(&MathValue::Integer(7), &MathValue::Integer(3));
            let r4 = engine.sqrt(&MathValue::Natural(2));
            let r5 = engine.sqrt(&MathValue::Integer(-4));
            black_box((r1, r2, r3, r4, r5))
        });
    });
}

// =============================================================================
// SIMULATION BRIDGE BENCHMARKS
// =============================================================================

fn bench_simulate_harmonic(c: &mut Criterion) {
    let sim = PhysicsSimulator::harmonic(2.0, 1.0, 0.0);
    c.bench_function("simulate_harmonic_5s", |bench| {
        bench.iter(|| {
            let result = sim.simulate(black_box(5.0), black_box(0.01));
            black_box(result.steps)
        });
    });
}

fn bench_simulate_lorenz(c: &mut Criterion) {
    let sim = PhysicsSimulator::lorenz();
    c.bench_function("simulate_lorenz_2s", |bench| {
        bench.iter(|| {
            let result = sim.simulate(black_box(2.0), black_box(0.005));
            black_box(result.steps)
        });
    });
}

fn bench_simulate_with_hdc(c: &mut Criterion) {
    let sim = PhysicsSimulator::harmonic(2.0, 1.0, 0.0);
    c.bench_function("simulate_harmonic_with_hdc", |bench| {
        bench.iter(|| {
            let (result, hvs) = sim.simulate_with_hdc(black_box(2.0), black_box(0.01));
            black_box((result.steps, hvs.len()))
        });
    });
}

fn bench_analysis_from_result(c: &mut Criterion) {
    let sim = PhysicsSimulator::harmonic(2.0, 1.0, 0.0);
    let result = sim.simulate(5.0, 0.01);
    c.bench_function("analysis_from_result", |bench| {
        bench.iter(|| {
            let analysis = SimulationAnalysis::from_result(black_box(&result));
            black_box(analysis.num_steps)
        });
    });
}

// =============================================================================
// HDC ENCODING BENCHMARKS
// =============================================================================

fn bench_state_to_binary_hv(c: &mut Criterion) {
    let state = vec![1.0, 0.5, -0.3, 2.1, -1.7, 0.0];
    c.bench_function("state_to_binary_hv", |bench| {
        bench.iter(|| {
            let hv = state_to_binary_hv(black_box(&state));
            black_box(hv)
        });
    });
}

fn bench_trajectory_encoding(c: &mut Criterion) {
    let sim = PhysicsSimulator::harmonic(2.0, 1.0, 0.0);
    let result = sim.simulate(5.0, 0.01);

    let mut group = c.benchmark_group("trajectory_encoding");
    for count in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &count,
            |bench, &count| {
                bench.iter(|| {
                    let hvs: Vec<BinaryHV> = result
                        .states
                        .iter()
                        .take(count)
                        .map(|s| state_to_binary_hv(s))
                        .collect();
                    black_box(hvs.len())
                });
            },
        );
    }
    group.finish();
}

// =============================================================================
// CROSS-BRIDGE PIPELINE BENCHMARKS
// =============================================================================

fn bench_full_pipeline_harmonic(c: &mut Criterion) {
    c.bench_function("full_pipeline_harmonic_to_hdc", |bench| {
        bench.iter(|| {
            // Simulate → Analyze → Encode
            let sim = PhysicsSimulator::harmonic(2.0, 1.0, 0.0);
            let osc = HarmonicOscillator::simple(2.0);
            let result = sim.simulate(2.0, 0.01);
            let analysis =
                SimulationAnalysis::from_result(&result).with_harmonic_energy(&osc, &result);

            let hvs: Vec<BinaryHV> = result
                .states
                .iter()
                .step_by(20)
                .take(10)
                .map(|s| state_to_binary_hv(s))
                .collect();

            black_box((analysis.energy_drift, hvs.len()))
        });
    });
}

fn bench_full_pipeline_math_to_hdc(c: &mut Criterion) {
    let engine = UnifiedMathEngine::new();
    c.bench_function("full_pipeline_math_5ops", |bench| {
        bench.iter(|| {
            let results = vec![
                engine.add(&MathValue::Natural(42), &MathValue::Natural(58)),
                engine.multiply(&MathValue::Integer(-3), &MathValue::Real(2.7)),
                engine.sqrt(&MathValue::Real(144.0)),
                engine.subtract(&MathValue::Real(10.0), &MathValue::Real(3.5)),
                engine.sqrt(&MathValue::Integer(-1)),
            ];
            let hvs: Vec<BinaryHV> = results.iter().map(|r| r.encoding).collect();
            black_box(hvs.len())
        });
    });
}

// =============================================================================
// CRITERION GROUPS
// =============================================================================

criterion_group!(
    math_benches,
    bench_math_engine_creation,
    bench_math_add_naturals,
    bench_math_multiply_complex,
    bench_math_sqrt_promotion,
    bench_math_domain_chain,
);

criterion_group!(
    simulation_benches,
    bench_simulate_harmonic,
    bench_simulate_lorenz,
    bench_simulate_with_hdc,
    bench_analysis_from_result,
);

criterion_group!(
    encoding_benches,
    bench_state_to_binary_hv,
    bench_trajectory_encoding,
);

criterion_group!(
    pipeline_benches,
    bench_full_pipeline_harmonic,
    bench_full_pipeline_math_to_hdc,
);

criterion_main!(
    math_benches,
    simulation_benches,
    encoding_benches,
    pipeline_benches
);
