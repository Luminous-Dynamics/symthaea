// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmarks for the Prism epistemic search engine.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use prism_search::SearchEngine;
use prism_search::binary_hv::BinaryHV;

fn bench_binary_hv_similarity(c: &mut Criterion) {
    let a = BinaryHV::random(42);
    let b = BinaryHV::random(43);
    c.bench_function("binary_hv_similarity", |bencher| {
        bencher.iter(|| black_box(a.similarity(&b)))
    });
}

fn bench_binary_hv_bind(c: &mut Criterion) {
    let a = BinaryHV::random(42);
    let b = BinaryHV::random(43);
    c.bench_function("binary_hv_bind", |bencher| {
        bencher.iter(|| black_box(a.bind(&b)))
    });
}

fn bench_binary_hv_bundle_10(c: &mut Criterion) {
    let vecs: Vec<BinaryHV> = (0..10).map(|i| BinaryHV::random(i)).collect();
    c.bench_function("binary_hv_bundle_10", |bencher| {
        bencher.iter(|| black_box(BinaryHV::bundle(&vecs)))
    });
}

fn bench_encode_text(c: &mut Criterion) {
    c.bench_function("encode_text_short", |bencher| {
        bencher.iter(|| black_box(prism_search::encode_text("ocean acidification causes")))
    });
}

fn bench_search(c: &mut Criterion) {
    let engine = SearchEngine::with_seed_claims();
    c.bench_function("search_top_10", |bencher| {
        bencher.iter(|| black_box(engine.search("consciousness neural correlates", 10)))
    });
}

criterion_group!(
    benches,
    bench_binary_hv_similarity,
    bench_binary_hv_bind,
    bench_binary_hv_bundle_10,
    bench_encode_text,
    bench_search,
);
criterion_main!(benches);
