// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Example: Generate a REAL cryptographic ZK proof.
//!
//! This example generates a real Risc0 STARK proof that can be
//! cryptographically verified. Takes ~40 seconds but produces
//! a proof that is mathematically sound.
//!
//! Run with: cargo run --example real_proof --features prover --release

use mycelix_zk_tax::{FilingStatus, Jurisdiction, TaxBracketProver};
use std::time::Instant;

fn main() {
    println!("=== Mycelix REAL ZK Proof Generation ===\n");
    println!("This will generate a cryptographically valid STARK proof.");
    println!("Expected time: ~30-60 seconds\n");

    let income: u64 = 125_000;
    let jurisdiction = Jurisdiction::US;
    let filing_status = FilingStatus::MarriedFilingJointly;
    let tax_year = 2024;

    println!("Private Input:");
    println!("  Income: ${}", income);
    println!("  Jurisdiction: {:?}", jurisdiction);
    println!("  Filing Status: {:?}", filing_status);
    println!("  Tax Year: {}\n", tax_year);

    // Real prover (not dev mode)
    let prover = TaxBracketProver::new();
    assert!(!prover.is_dev_mode(), "Should be using real prover!");

    println!("Starting Risc0 zkVM proof generation...");
    let start = Instant::now();

    let proof = prover
        .prove(income, jurisdiction, filing_status, tax_year)
        .expect("Real proof generation failed");

    let prove_time = start.elapsed();

    println!("\n=== REAL Proof Generated! ===");
    println!("Proof Time: {:?}", prove_time);
    println!("\nPublic Output:");
    println!("  Bracket Index: {}", proof.bracket_index);
    println!("  Marginal Rate: {}%", proof.rate_bps as f64 / 100.0);
    println!("  Bracket Range: ${} - ${}", proof.bracket_lower, proof.bracket_upper);
    println!("  Receipt Size: {} bytes", proof.receipt_bytes.len());

    // Verify the proof
    println!("\nVerifying proof...");
    let verify_start = Instant::now();
    proof.verify().expect("Proof verification failed!");
    let verify_time = verify_start.elapsed();
    println!("Verification Time: {:?}", verify_time);

    // Export to JSON
    let json = proof.to_json().expect("JSON serialization failed");
    println!("\nProof JSON size: {} bytes", json.len());

    // Export to binary
    let bytes = proof.to_bytes().expect("Binary serialization failed");
    println!("Proof binary size: {} bytes", bytes.len());

    println!("\n=== Summary ===");
    println!("{}", proof.summary());
    println!("\n✓ Real ZK proof complete!");
}
