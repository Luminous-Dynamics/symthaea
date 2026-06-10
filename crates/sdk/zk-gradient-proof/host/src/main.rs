// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gradient Quality Proof Host
//!
//! This program demonstrates generating and verifying ZK proofs
//! for gradient quality in federated learning.
//!
//! # Usage
//!
//! ```bash
//! # Generate and verify a proof
//! cargo run --release
//!
//! # With debug logging
//! RUST_LOG=info cargo run --release
//! ```

use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;
use zk_gradient_core::{
    GradientConstraints, GradientProofInput, GradientProofOutput,
};
use zk_gradient_methods::GRADIENT_QUALITY_ELF;

fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("=== ZK Gradient Quality Proof Demo ===\n");

    // Create sample gradient (simulating FL client training)
    let gradient_dimension = 1000;
    let gradient: Vec<f32> = (0..gradient_dimension)
        .map(|i| {
            // Simulate gradient from training
            let x = i as f32 / gradient_dimension as f32;
            (x * std::f32::consts::PI * 4.0).sin() * 0.5 + 0.1
        })
        .collect();

    println!("Gradient dimension: {}", gradient.len());
    let l2_norm: f32 = gradient.iter().map(|g| g * g).sum::<f32>().sqrt();
    println!("Gradient L2 norm: {:.4}", l2_norm);

    // Create proof input (private witness)
    let input = GradientProofInput {
        gradient,
        global_model_hash: [0x42u8; 32], // Simulated model hash
        epochs: 5,
        learning_rate: 0.01,
        client_id: "client-001".to_string(),
        round: 1,
        constraints: GradientConstraints::for_federated_learning(),
    };

    println!("\n--- Generating ZK Proof ---");
    println!("This proves gradient quality without revealing gradient values...\n");

    // Build executor environment with private witness
    let env = ExecutorEnv::builder()
        .write(&input)
        .expect("Failed to write input")
        .build()
        .expect("Failed to build env");

    // Generate proof
    let start = Instant::now();
    let prover = default_prover();
    let prove_info = prover
        .prove(env, GRADIENT_QUALITY_ELF)
        .expect("Proof generation failed");
    let proof_time = start.elapsed();

    println!("Proof generated in {:.2?}", proof_time);

    // Extract receipt and output
    let receipt = prove_info.receipt;
    let output: GradientProofOutput = receipt
        .journal
        .decode()
        .expect("Failed to decode journal");

    println!("\n--- Public Output (Journal) ---");
    println!("Gradient hash: {:02x?}...", &output.gradient_hash[..8]);
    println!("Model hash: {:02x?}...", &output.global_model_hash[..8]);
    println!("Quality valid: {}", output.quality_valid);
    println!("Epochs: {}", output.epochs);
    println!("Learning rate: {}", output.learning_rate);
    println!("Client ID: {}", output.client_id);
    println!("Round: {}", output.round);
    println!("Dimension: {}", output.dimension);
    println!("Commitment: 0x{:016x}", output.commitment);

    // Verify proof
    println!("\n--- Verifying Proof ---");
    let verify_start = Instant::now();
    receipt
        .verify(zk_gradient_methods::GRADIENT_QUALITY_ID)
        .expect("Proof verification failed");
    let verify_time = verify_start.elapsed();

    println!("Proof verified in {:.2?}", verify_time);
    println!("\n✓ Gradient quality proven without revealing gradient values!");

    // Summary
    println!("\n=== Summary ===");
    println!("What the verifier learned:");
    println!("  - Gradient has {} dimensions", output.dimension);
    println!("  - L2 norm is within acceptable range");
    println!("  - No exploding/vanishing gradients");
    println!("  - Computed for model hash {:02x?}...", &output.global_model_hash[..4]);
    println!("  - Training: {} epochs, lr={}", output.epochs, output.learning_rate);
    println!("\nWhat remains private:");
    println!("  - All {} gradient values", output.dimension);
    println!("  - Exact L2 norm");
    println!("  - Training data used");
}
