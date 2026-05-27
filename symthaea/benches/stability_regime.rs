// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use criterion::{Criterion, criterion_group, criterion_main};

use symthaea::consciousness::primitive_consciousness::ConsciousnessPrimitiveProcessor;
use symthaea::consciousness::stability_regime::{
    CfCPrimitive, StabilityRegimeConfig, StabilityRegimeProcessor, StabilityRegimeType,
};
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::primitive_system::PrimitiveTier;
use symthaea_core::hdc::unified_hv::ContinuousHV;

fn make_primitive(
    name: &str,
    tier: PrimitiveTier,
) -> symthaea_core::hdc::primitive_system::Primitive {
    symthaea_core::hdc::primitive_system::Primitive {
        name: name.to_string(),
        tier,
        domain: "bench".to_string(),
        encoding: BinaryHV::random(name.len() as u64 * 31),
        definition: name.to_string(),
        is_base: true,
        derivation: None,
    }
}

fn bench_evolve_crystallized(c: &mut Criterion) {
    let config = StabilityRegimeConfig::default();
    let prim = make_primitive("FORCE", PrimitiveTier::Physical);
    let mut cfc = CfCPrimitive::new(prim, &config, 42);
    let params = config.params(StabilityRegimeType::Crystallized).clone();
    let input = ContinuousHV::random(16_384, 99);

    c.bench_function("evolve_crystallized", |b| {
        b.iter(|| cfc.evolve(0.1, &input, &params))
    });
}

fn bench_evolve_plastic(c: &mut Criterion) {
    let config = StabilityRegimeConfig::default();
    let prim = make_primitive("PLAN", PrimitiveTier::Strategic);
    let mut cfc = CfCPrimitive::new(prim, &config, 42);
    let params = config.params(StabilityRegimeType::Plastic).clone();
    let input = ContinuousHV::random(16_384, 99);

    c.bench_function("evolve_plastic", |b| {
        b.iter(|| cfc.evolve(0.1, &input, &params))
    });
}

fn bench_evolve_fluid(c: &mut Criterion) {
    let config = StabilityRegimeConfig::default();
    let prim = make_primitive("COMPOSE", PrimitiveTier::Compositional);
    let mut cfc = CfCPrimitive::new(prim, &config, 42);
    let params = config.params(StabilityRegimeType::Fluid).clone();
    let input = ContinuousHV::random(16_384, 99);

    c.bench_function("evolve_fluid", |b| {
        b.iter(|| cfc.evolve(0.1, &input, &params))
    });
}

fn bench_process_input(c: &mut Criterion) {
    let mut processor = StabilityRegimeProcessor::new();
    let input = BinaryHV::random(42);

    c.bench_function("process_input_stability_regime", |b| {
        b.iter(|| processor.process_input(&input, 0.1, 0.0))
    });
}

fn bench_static_baseline(c: &mut Criterion) {
    let mut processor = ConsciousnessPrimitiveProcessor::new();
    let input = BinaryHV::random(42);

    c.bench_function("process_input_static_baseline", |b| {
        b.iter(|| processor.process_input(&input, 0.0))
    });
}

criterion_group!(
    benches,
    bench_evolve_crystallized,
    bench_evolve_plastic,
    bench_evolve_fluid,
    bench_process_input,
    bench_static_baseline,
);
criterion_main!(benches);
