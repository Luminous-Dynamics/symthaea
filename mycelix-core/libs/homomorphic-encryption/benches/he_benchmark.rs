// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Benchmarks for homomorphic encryption operations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use mycelix_homomorphic::{
    PaillierKeyPair,
    encoding::{FixedPointEncoder, EncodingParams},
    encrypted_vector::{EncryptedVector, EncryptedGradient, GradientMetadata},
};

fn keygen_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Key Generation");
    
    for bits in [1024, 2048].iter() {
        group.bench_with_input(BenchmarkId::new("Paillier", bits), bits, |b, &bits| {
            b.iter(|| {
                black_box(PaillierKeyPair::generate(bits).unwrap())
            });
        });
    }
    
    group.finish();
}

fn encryption_benchmark(c: &mut Criterion) {
    let keypair = PaillierKeyPair::generate(1024).unwrap();
    let pk = keypair.public_key();
    
    let mut group = c.benchmark_group("Encryption");
    
    group.bench_function("encrypt_i64", |b| {
        b.iter(|| {
            black_box(pk.encrypt(black_box(42i64)))
        });
    });
    
    group.bench_function("encrypt_f64", |b| {
        b.iter(|| {
            black_box(pk.encrypt_f64(black_box(0.123456), 6))
        });
    });
    
    group.finish();
}

fn homomorphic_ops_benchmark(c: &mut Criterion) {
    let keypair = PaillierKeyPair::generate(1024).unwrap();
    let pk = keypair.public_key();
    
    let a = pk.encrypt(100i64);
    let b = pk.encrypt(50i64);
    
    let mut group = c.benchmark_group("Homomorphic Operations");
    
    group.bench_function("add", |b| {
        b.iter(|| {
            black_box(black_box(&a).add(black_box(&b)))
        });
    });
    
    group.bench_function("scalar_mul", |b| {
        b.iter(|| {
            black_box(black_box(&a).scalar_mul(black_box(10)))
        });
    });
    
    group.bench_function("rerandomize", |b| {
        b.iter(|| {
            black_box(a.rerandomize(pk))
        });
    });
    
    group.finish();
}

fn decryption_benchmark(c: &mut Criterion) {
    let keypair = PaillierKeyPair::generate(1024).unwrap();
    let pk = keypair.public_key();
    let ciphertext = pk.encrypt(42i64);
    
    c.bench_function("decrypt", |b| {
        b.iter(|| {
            black_box(keypair.decrypt_i64(black_box(&ciphertext)).unwrap())
        });
    });
}

fn vector_ops_benchmark(c: &mut Criterion) {
    let keypair = PaillierKeyPair::generate(1024).unwrap();
    let pk = keypair.public_key();
    
    let mut group = c.benchmark_group("Vector Operations");
    
    for size in [10, 100, 1000].iter() {
        let elements: Vec<_> = (0..*size).map(|i| pk.encrypt(i as i64)).collect();
        let vec1 = EncryptedVector::new(elements.clone());
        let vec2 = EncryptedVector::new(elements);
        
        group.bench_with_input(BenchmarkId::new("vector_add", size), size, |b, _| {
            b.iter(|| {
                black_box(vec1.add(black_box(&vec2)).unwrap())
            });
        });
    }
    
    group.finish();
}

fn gradient_aggregation_benchmark(c: &mut Criterion) {
    let keypair = PaillierKeyPair::generate(1024).unwrap();
    let pk = keypair.public_key();
    let encoder = FixedPointEncoder::new(EncodingParams::new(6));
    
    let create_gradient = |id: usize| {
        let values: Vec<f64> = (0..100).map(|i| (i as f64) * 0.01).collect();
        let metadata = GradientMetadata {
            participant_id: format!("p{}", id),
            round: 1,
            sample_count: 100,
            dimension: values.len(),
            precision: 6,
            timestamp: 0,
        };
        EncryptedGradient::from_gradient(&values, pk, &encoder, metadata).unwrap()
    };
    
    let mut group = c.benchmark_group("Gradient Aggregation");
    
    for count in [3, 10, 50].iter() {
        let gradients: Vec<_> = (0..*count).map(create_gradient).collect();
        
        group.bench_with_input(BenchmarkId::new("aggregate", count), count, |b, _| {
            b.iter(|| {
                black_box(EncryptedGradient::aggregate(black_box(&gradients)).unwrap())
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    keygen_benchmark,
    encryption_benchmark,
    homomorphic_ops_benchmark,
    decryption_benchmark,
    vector_ops_benchmark,
    gradient_aggregation_benchmark,
);

criterion_main!(benches);
