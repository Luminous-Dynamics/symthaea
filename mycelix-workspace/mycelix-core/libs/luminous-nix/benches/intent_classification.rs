// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Benchmarks for intent classification performance

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use luminous_nix::IntentClassifier;

fn classification_benchmark(c: &mut Criterion) {
    let mut classifier = IntentClassifier::new();
    classifier.bootstrap();

    // Benchmark simple queries
    c.bench_function("classify_search_simple", |b| {
        b.iter(|| {
            classifier.classify(black_box("search firefox"))
        })
    });

    c.bench_function("classify_install_simple", |b| {
        b.iter(|| {
            classifier.classify(black_box("install vim"))
        })
    });

    // Benchmark complex queries
    c.bench_function("classify_search_complex", |b| {
        b.iter(|| {
            classifier.classify(black_box("find me a good markdown editor for writing documentation"))
        })
    });

    // Benchmark novel queries (zero-shot)
    c.bench_function("classify_novel_query", |b| {
        b.iter(|| {
            classifier.classify(black_box("i want to grab the latest version of rustup"))
        })
    });
}

fn bootstrap_benchmark(c: &mut Criterion) {
    c.bench_function("bootstrap_classifier", |b| {
        b.iter(|| {
            let mut classifier = IntentClassifier::new();
            classifier.bootstrap();
            black_box(classifier)
        })
    });
}

criterion_group!(benches, classification_benchmark, bootstrap_benchmark);
criterion_main!(benches);
