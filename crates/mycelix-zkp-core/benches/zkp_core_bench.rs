// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmarks for mycelix-zkp-core core operations.
//!
//! Run: cargo bench -p mycelix-zkp-core --features full

#![allow(unused_imports)]

use std::hint::black_box;
use std::time::Instant;

use mycelix_zkp_core::domain::DomainTag;
use mycelix_zkp_core::fixed_point::FixedPoint;
use mycelix_zkp_core::pogq::{simulate_pogq, PoGQPublicInputs, PoGQWitness};
use mycelix_zkp_core::types::AuthenticatedProof;

fn bench_fixed_point_ops(iterations: u64) {
    let a = FixedPoint::from_f32(0.85);
    let b = FixedPoint::from_f32(0.75);

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = black_box(a.mul(b));
    }
    let elapsed = start.elapsed();
    println!(
        "  FixedPoint mul:  {:>8.1} ns/op ({} ops in {:?})",
        elapsed.as_nanos() as f64 / iterations as f64,
        iterations,
        elapsed
    );

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = black_box(a + b);
    }
    let elapsed = start.elapsed();
    println!(
        "  FixedPoint add:  {:>8.1} ns/op ({} ops in {:?})",
        elapsed.as_nanos() as f64 / iterations as f64,
        iterations,
        elapsed
    );
}

fn bench_pogq_simulation(iterations: u64) {
    let inputs = PoGQPublicInputs {
        threshold: FixedPoint::from_f32(0.7),
        ..Default::default()
    };
    // 8 rounds (matches Winterfell trace length)
    let witness = PoGQWitness {
        scores: vec![
            FixedPoint::from_f32(0.8),
            FixedPoint::from_f32(0.6),
            FixedPoint::from_f32(0.5),
            FixedPoint::from_f32(0.9),
            FixedPoint::from_f32(0.3),
            FixedPoint::from_f32(0.4),
            FixedPoint::from_f32(0.95),
            FixedPoint::from_f32(0.85),
        ],
    };

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = black_box(simulate_pogq(&inputs, &witness));
    }
    let elapsed = start.elapsed();
    println!(
        "  PoGQ sim (8 rounds): {:>8.1} ns/op ({} ops in {:?})",
        elapsed.as_nanos() as f64 / iterations as f64,
        iterations,
        elapsed
    );
}

fn bench_domain_tag(iterations: u64) {
    let start = Instant::now();
    for _ in 0..iterations {
        let tag = black_box(DomainTag::new("Governance", "AnonVote", 1));
        let _ = black_box(tag.matches("Governance", "AnonVote"));
    }
    let elapsed = start.elapsed();
    println!(
        "  DomainTag create+match: {:>8.1} ns/op ({} ops in {:?})",
        elapsed.as_nanos() as f64 / iterations as f64,
        iterations,
        elapsed
    );
}

fn bench_signed_message(iterations: u64) {
    use mycelix_zkp_core::types::{BackendId, ProofMetadata};

    let meta = ProofMetadata {
        domain_tag: DomainTag::new("Bench", "SignedMsg", 1),
        protocol_version: 1,
        client_id: [0xAA; 32],
        timestamp: 1700000000,
        nonce: [0xBB; 32],
        backend: BackendId::Winterfell,
    };
    let proof = AuthenticatedProof {
        proof: vec![0u8; 200_000], // ~200KB proof
        signature: vec![],
        metadata: meta,
        public_inputs_hash: [0xCC; 32],
    };

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = black_box(proof.construct_signed_message());
    }
    let elapsed = start.elapsed();
    println!(
        "  SignedMessage (200KB proof): {:>8.1} us/op ({} ops in {:?})",
        elapsed.as_micros() as f64 / iterations as f64,
        iterations,
        elapsed
    );
}

#[cfg(feature = "dilithium")]
fn bench_dilithium(iterations: u64) {
    use mycelix_zkp_core::dilithium::{verify_signature, DilithiumKeypair};

    let kp = DilithiumKeypair::generate();

    // Keygen
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = black_box(DilithiumKeypair::generate());
    }
    let elapsed = start.elapsed();
    println!(
        "  Dilithium5 keygen: {:>8.1} us/op ({} ops in {:?})",
        elapsed.as_micros() as f64 / iterations as f64,
        iterations,
        elapsed
    );

    // Sign
    let message = b"benchmark message for DASTARK signing";
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = black_box(kp.sign(message));
    }
    let elapsed = start.elapsed();
    println!(
        "  Dilithium5 sign:   {:>8.1} us/op ({} ops in {:?})",
        elapsed.as_micros() as f64 / iterations as f64,
        iterations,
        elapsed
    );

    // Verify
    let signature = kp.sign(message).unwrap();
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = black_box(verify_signature(message, &signature, kp.public_key()));
    }
    let elapsed = start.elapsed();
    println!(
        "  Dilithium5 verify: {:>8.1} us/op ({} ops in {:?})",
        elapsed.as_micros() as f64 / iterations as f64,
        iterations,
        elapsed
    );
}

fn main() {
    println!("=== mycelix-zkp-core benchmarks ===\n");

    println!("Fixed-point arithmetic:");
    bench_fixed_point_ops(1_000_000);

    println!("\nPoGQ simulation:");
    bench_pogq_simulation(100_000);

    println!("\nDomain tags:");
    bench_domain_tag(1_000_000);

    println!("\nSigned message construction:");
    bench_signed_message(1_000);

    #[cfg(feature = "dilithium")]
    {
        println!("\nDilithium5 PQ signatures:");
        bench_dilithium(100);
    }

    println!("\n=== done ===");
}
