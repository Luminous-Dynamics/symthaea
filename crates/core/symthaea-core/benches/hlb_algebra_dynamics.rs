// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Holographic Liquid Brain microbenchmark suite.
//!
//! Run: cargo bench -p symthaea-core --bench hlb_algebra_dynamics
//!
//! Reports operator cost across dimensions. It intentionally does not claim
//! task-level superiority; HLB-HDC/HLB-DYN accuracy and error-rate experiments
//! are specified in docs/research/SYMTHAEA_HOLOGRAPHIC_LIQUID_BRAIN_SWI_002.md.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use symthaea_core::hdc::{
    ContinuousHV, HDC_DIMENSION, HdcHrrCleanupPolicy, HdcHrrCodebook, HdcHrrRole,
    HdcLtcUnifiedNeuron, HdcStructuredMixer, HdcStructuredMixerKind, UnifiedConfig,
    circular_convolution, circular_correlation,
};

fn bench_hdc_algebra(c: &mut Criterion) {
    let mut group = c.benchmark_group("HLB_algebra");
    for dimension in [256usize, 1_024, 4_096, 16_384] {
        let left = ContinuousHV::random(dimension, 10);
        let right = ContinuousHV::random(dimension, 20);
        let hrr_role = HdcHrrRole::from_seed(dimension, 30).expect("valid radix-2 dimension");
        let bound = hrr_role.bind(&right).expect("matching HRR dimensions");
        let hadamard = HdcStructuredMixer::new(HdcStructuredMixerKind::Identity, dimension)
            .expect("non-zero dimension");
        let permutation = HdcStructuredMixer::new(
            HdcStructuredMixerKind::CyclicPermutation { shift: 17 },
            dimension,
        )
        .expect("non-zero dimension");

        group.bench_with_input(
            BenchmarkId::new("hadamard_bind", dimension),
            &dimension,
            |b, _| b.iter(|| black_box(left.bind(black_box(&right)))),
        );
        group.bench_with_input(
            BenchmarkId::new("cosine_similarity", dimension),
            &dimension,
            |b, _| b.iter(|| black_box(left.similarity(black_box(&right)))),
        );
        group.bench_with_input(
            BenchmarkId::new("cyclic_permutation", dimension),
            &dimension,
            |b, _| b.iter(|| black_box(permutation.apply(black_box(&left)).unwrap())),
        );
        group.bench_with_input(
            BenchmarkId::new("walsh_hadamard", dimension),
            &dimension,
            |b, _| {
                let mixer = HdcStructuredMixer::new(
                    HdcStructuredMixerKind::WalshHadamard,
                    dimension,
                )
                .expect("power-of-two dimension");
                b.iter(|| black_box(mixer.apply(black_box(&left)).unwrap()))
            },
        );
        group.bench_with_input(
            BenchmarkId::new("identity_mixer", dimension),
            &dimension,
            |b, _| b.iter(|| black_box(hadamard.apply(black_box(&left)).unwrap())),
        );
        group.bench_with_input(
            BenchmarkId::new("hrr_circular_convolution", dimension),
            &dimension,
            |b, _| {
                b.iter(|| black_box(circular_convolution(black_box(&left), black_box(&right)).unwrap()))
            },
        );
        group.bench_with_input(
            BenchmarkId::new("hrr_unitary_bind", dimension),
            &dimension,
            |b, _| b.iter(|| black_box(hrr_role.bind(black_box(&right)).unwrap())),
        );
        group.bench_with_input(
            BenchmarkId::new("hrr_unitary_unbind", dimension),
            &dimension,
            |b, _| b.iter(|| black_box(hrr_role.unbind(black_box(&bound)).unwrap())),
        );
        group.bench_with_input(
            BenchmarkId::new("hrr_general_correlation", dimension),
            &dimension,
            |b, _| {
                b.iter(|| black_box(circular_correlation(black_box(&bound), black_box(hrr_role.vector())).unwrap()))
            },
        );
    }
    group.finish();
}

fn bench_liquid_integrators(c: &mut Criterion) {
    let mut group = c.benchmark_group("HLB_liquid_integrators");
    group.sample_size(20);

    for dimension in [256usize, 1_024, 4_096, HDC_DIMENSION] {
        let config = UnifiedConfig {
            dimension,
            tau_base: 0.05,
            fourier_frequencies: Vec::new(),
            ..UnifiedConfig::default()
        };
        let input = ContinuousHV::random(dimension, 40);
        let mut closed = HdcLtcUnifiedNeuron::new(config.clone(), 50);
        let mut fused = HdcLtcUnifiedNeuron::new(config.clone(), 51);
        let mut iterative = HdcLtcUnifiedNeuron::new(config.clone(), 52);
        let mut rk4 = HdcLtcUnifiedNeuron::new(config, 53);
        let dt = 0.05;

        group.bench_with_input(
            BenchmarkId::new("adaptive_closed_form", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    closed.reset();
                    closed.evolve_closed_form(dt, black_box(&input));
                    black_box(closed.state().norm())
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("fused_closed_form", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    fused.reset();
                    fused.evolve_closed_form_fused(dt, black_box(&input));
                    black_box(fused.state().norm())
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("iterative_substeps", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    iterative.reset();
                    iterative.evolve_closed_form_iterative(dt, black_box(&input));
                    black_box(iterative.state().norm())
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("rk4_reference", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    rk4.reset();
                    rk4.evolve_rk4(dt, black_box(&input));
                    black_box(rk4.state().norm())
                })
            },
        );
    }
    group.finish();
}

fn bench_hrr_record_retrieval(c: &mut Criterion) {
    let mut group = c.benchmark_group("HLB_HRR_record_retrieval");
    let dimension = 4_096usize;

    for load in [2usize, 8, 16, 32, 64] {
        let roles: Vec<HdcHrrRole> = (0..load)
            .map(|i| HdcHrrRole::from_seed(dimension, 1_000 + i as u64).unwrap())
            .collect();
        let fillers: Vec<ContinuousHV> = (0..load)
            .map(|i| ContinuousHV::random(dimension, 2_000 + i as u64))
            .collect();
        let mut record = ContinuousHV::zero(dimension);
        for (role, filler) in roles.iter().zip(fillers.iter()) {
            record = record.add(&role.bind(filler).unwrap());
        }
        let codebook = HdcHrrCodebook::new(fillers.clone()).unwrap();
        let policy = HdcHrrCleanupPolicy::new(0.0, 0.0);

        group.bench_with_input(
            BenchmarkId::new("unbind_and_confidence_cleanup", load),
            &load,
            |b, _| {
                b.iter(|| {
                    let decoded = roles[0].unbind(black_box(&record)).unwrap();
                    let result = codebook.cleanup(black_box(&decoded), policy).unwrap();
                    black_box((result.index, result.similarity, result.margin, result.accepted))
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_hdc_algebra,
    bench_liquid_integrators,
    bench_hrr_record_retrieval,
);
criterion_main!(benches);
