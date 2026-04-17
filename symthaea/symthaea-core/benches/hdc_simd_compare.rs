// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Focused HDC SIMD comparison benchmark.
//!
//! This is the audit harness for deciding whether external HDC kernels are worth
//! porting. It keeps the workload fixed at Symthaea's canonical 16,384-bit
//! `BinaryHV` representation and compares current runtime-dispatched SIMD paths
//! against scalar or packed baselines.
//!
//! Run:
//!
//! ```bash
//! cargo bench -p symthaea-core --bench hdc_simd_compare
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::native_similarity::PackedBipolar;
use symthaea_core::hdc::simd_detect::SimdCapabilities;

fn seeded_vectors(count: usize) -> Vec<BinaryHV> {
    (0..count)
        .map(|idx| BinaryHV::random(0x5EED_0000 + idx as u64))
        .collect()
}

fn hv_to_bipolar(hv: &BinaryHV) -> Vec<i8> {
    let mut bipolar = Vec::with_capacity(BinaryHV::DIM);
    for byte in hv.0 {
        for bit_idx in 0..8 {
            bipolar.push(if (byte >> bit_idx) & 1 == 1 { 1 } else { -1 });
        }
    }
    bipolar
}

fn packed_vectors(vectors: &[BinaryHV]) -> Vec<PackedBipolar> {
    vectors
        .iter()
        .map(|hv| PackedBipolar::from_bipolar(&hv_to_bipolar(hv)))
        .collect()
}

fn bind_u64_words(a: &BinaryHV, b: &BinaryHV) -> BinaryHV {
    let mut result = [0u8; BinaryHV::BYTES];

    for (out, (a_word, b_word)) in result
        .chunks_exact_mut(8)
        .zip(a.0.chunks_exact(8).zip(b.0.chunks_exact(8)))
    {
        let a_word = u64::from_ne_bytes(a_word.try_into().expect("8-byte chunk"));
        let b_word = u64::from_ne_bytes(b_word.try_into().expect("8-byte chunk"));
        out.copy_from_slice(&(a_word ^ b_word).to_ne_bytes());
    }

    BinaryHV(result)
}

fn copy_hv(a: &BinaryHV) -> BinaryHV {
    BinaryHV(a.0)
}

fn bind_u64_words_into(a: &BinaryHV, b: &BinaryHV, out: &mut [u8; BinaryHV::BYTES]) {
    for (out, (a_word, b_word)) in out
        .chunks_exact_mut(8)
        .zip(a.0.chunks_exact(8).zip(b.0.chunks_exact(8)))
    {
        let a_word = u64::from_ne_bytes(a_word.try_into().expect("8-byte chunk"));
        let b_word = u64::from_ne_bytes(b_word.try_into().expect("8-byte chunk"));
        out.copy_from_slice(&(a_word ^ b_word).to_ne_bytes());
    }
}

fn copy_hv_into(a: &BinaryHV, out: &mut [u8; BinaryHV::BYTES]) {
    out.copy_from_slice(&a.0);
}

fn report_simd_capabilities(c: &mut Criterion) {
    let capabilities = SimdCapabilities::detect();
    let mut group = c.benchmark_group("hdc_simd_capabilities");
    group.bench_function(format!("{capabilities:?}"), |b| b.iter(|| black_box(())));
    group.finish();
}

fn bench_bind_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc_simd_bind_16k");
    group.throughput(Throughput::Bytes(BinaryHV::BYTES as u64));

    let a = BinaryHV::random(1);
    let b = BinaryHV::random(2);

    group.bench_function("current_simd_dispatch", |bench| {
        bench.iter(|| black_box(a.bind(black_box(&b))))
    });

    group.bench_function("scalar_byte_xor", |bench| {
        bench.iter(|| black_box(a.bind_scalar(black_box(&b))))
    });

    group.bench_function("u64_word_xor_baseline", |bench| {
        bench.iter(|| black_box(bind_u64_words(black_box(&a), black_box(&b))))
    });

    group.bench_function("copy_only_baseline", |bench| {
        bench.iter(|| black_box(copy_hv(black_box(&a))))
    });

    group.bench_function("u64_word_xor_into_reused_buffer", |bench| {
        let mut out = [0u8; BinaryHV::BYTES];
        bench.iter(|| {
            bind_u64_words_into(black_box(&a), black_box(&b), black_box(&mut out));
            black_box(out[0])
        })
    });

    group.bench_function("copy_into_reused_buffer", |bench| {
        let mut out = [0u8; BinaryHV::BYTES];
        bench.iter(|| {
            copy_hv_into(black_box(&a), black_box(&mut out));
            black_box(out[0])
        })
    });

    group.bench_function("bind_assign_inplace", |bench| {
        let mut acc = a;
        bench.iter(|| {
            acc.bind_assign(black_box(&b));
            black_box(acc.0[0])
        })
    });

    group.finish();
}

fn bench_bind_chain_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc_simd_bind_chain_16k");

    for count in [2usize, 3, 5, 11] {
        let vectors = seeded_vectors(count);
        let refs: Vec<&BinaryHV> = vectors.iter().collect();
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::new("legacy_chained_bind_byval", count),
            &vectors,
            |bench, vectors| {
                bench.iter(|| {
                    let mut acc = vectors[0];
                    for v in &vectors[1..] {
                        acc = acc.bind(v);
                    }
                    black_box(acc)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bind_chain_inplace", count),
            &refs,
            |bench, refs| bench.iter(|| black_box(BinaryHV::bind_chain(black_box(refs)))),
        );
    }

    group.finish();
}

fn bench_similarity_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc_simd_similarity_16k");
    group.throughput(Throughput::Bytes(BinaryHV::BYTES as u64));

    let a = BinaryHV::random(3);
    let b = BinaryHV::random(4);
    let packed_a = PackedBipolar::from_bipolar(&hv_to_bipolar(&a));
    let packed_b = PackedBipolar::from_bipolar(&hv_to_bipolar(&b));

    group.bench_function("current_simd_matching_bits", |bench| {
        bench.iter(|| black_box(a.similarity(black_box(&b))))
    });

    group.bench_function("scalar_byte_popcount", |bench| {
        bench.iter(|| black_box(a.similarity_scalar(black_box(&b))))
    });

    group.bench_function("packed_u64_xor_popcount", |bench| {
        bench.iter(|| black_box(packed_a.xor_similarity(black_box(&packed_b))))
    });

    group.finish();
}

fn bench_hamming_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc_simd_hamming_16k");
    group.throughput(Throughput::Bytes(BinaryHV::BYTES as u64));

    let a = BinaryHV::random(5);
    let b = BinaryHV::random(6);
    let packed_a = PackedBipolar::from_bipolar(&hv_to_bipolar(&a));
    let packed_b = PackedBipolar::from_bipolar(&hv_to_bipolar(&b));

    group.bench_function("current_simd_distance", |bench| {
        bench.iter(|| black_box(a.hamming_distance(black_box(&b))))
    });

    group.bench_function("scalar_byte_popcount", |bench| {
        bench.iter(|| black_box(a.hamming_distance_scalar(black_box(&b))))
    });

    group.bench_function("packed_u64_xor_popcount", |bench| {
        bench.iter(|| black_box(packed_a.hamming_distance(black_box(&packed_b))))
    });

    group.finish();
}

fn bench_bundle_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc_simd_bundle_16k");

    for count in [3usize, 5, 11, 33, 101] {
        let vectors = seeded_vectors(count);
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::new("current_simd_majority", count),
            &vectors,
            |bench, vectors| bench.iter(|| black_box(BinaryHV::bundle(black_box(vectors)))),
        );

        group.bench_with_input(
            BenchmarkId::new("thread_local_scalar_safe", count),
            &vectors,
            |bench, vectors| bench.iter(|| black_box(BinaryHV::bundle_safe(black_box(vectors)))),
        );
    }

    group.finish();
}

fn bench_search_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc_simd_search_16k");

    for count in [64usize, 256, 1024] {
        let query = BinaryHV::random(0xABCD);
        let candidates = seeded_vectors(count);
        let packed_query = PackedBipolar::from_bipolar(&hv_to_bipolar(&query));
        let packed_candidates = packed_vectors(&candidates);

        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::new("binaryhv_simd_argmax", count),
            &candidates,
            |bench, candidates| {
                bench.iter(|| {
                    let best = candidates
                        .iter()
                        .map(|candidate| query.similarity(candidate))
                        .fold(f32::NEG_INFINITY, f32::max);
                    black_box(best)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("packed_u64_argmax", count),
            &packed_candidates,
            |bench, candidates| {
                bench.iter(|| {
                    let best = candidates
                        .iter()
                        .map(|candidate| packed_query.xor_similarity(candidate))
                        .fold(f32::NEG_INFINITY, f32::max);
                    black_box(best)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    report_simd_capabilities,
    bench_bind_paths,
    bench_bind_chain_paths,
    bench_similarity_paths,
    bench_hamming_paths,
    bench_bundle_paths,
    bench_search_paths,
);

criterion_main!(benches);
