// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Example: Generate a real ZK proof of tax bracket membership.
//!
//! This demonstrates the full Mycelix ZK Tax workflow:
//! 1. User provides income (kept private)
//! 2. SDK generates ZK proof of bracket membership
//! 3. Proof can be verified without revealing income
//!
//! Run with: cargo run --example generate_proof --features prover

use mycelix_zk_tax::{FilingStatus, Jurisdiction, TaxBracketProver, TaxBracketReceipt};
use std::time::Instant;

fn main() {
    println!("=== Mycelix ZK Tax Bracket Proof Demo ===\n");

    // Simulated user income (this stays PRIVATE!)
    let income: u64 = 85_000;
    let jurisdiction = Jurisdiction::US;
    let filing_status = FilingStatus::Single;
    let tax_year = 2024;

    println!("Private Input (never revealed in proof):");
    println!("  Income: ${}", income);
    println!("  Jurisdiction: {:?}", jurisdiction);
    println!("  Filing Status: {:?}", filing_status);
    println!("  Tax Year: {}\n", tax_year);

    // Use dev mode for fast testing (real mode takes ~40 seconds)
    // To use real ZK proofs, use TaxBracketProver::new() instead
    let prover = TaxBracketProver::dev_mode();
    println!("Using: {} mode", if prover.is_dev_mode() { "DEV" } else { "REAL ZK" });

    println!("\nGenerating proof...");
    let start = Instant::now();

    let proof = prover
        .prove(income, jurisdiction, filing_status, tax_year)
        .expect("Proof generation failed");

    let elapsed = start.elapsed();

    println!("\n=== Proof Generated! ===");
    println!("Time: {:?}", elapsed);
    println!("\nPublic Output (safe to share):");
    println!("  Bracket Index: {}", proof.bracket_index);
    println!("  Marginal Rate: {}%", proof.rate_bps as f64 / 100.0);
    println!("  Bracket Range: ${} - ${}", proof.bracket_lower, proof.bracket_upper);
    println!("  Commitment: {}", proof.commitment);
    println!("\nWhat this proves:");
    println!("  'I have income in the {}% tax bracket'", proof.rate_bps as f64 / 100.0);
    println!("  WITHOUT revealing the actual income!");

    // Create a receipt for verification (lightweight DHT storage)
    let receipt = TaxBracketReceipt::from(&proof);
    println!("\n=== Receipt for DHT Storage ===");
    println!("  Proof ID: {}", receipt.proof_id);
    println!("  Jurisdiction: {}", receipt.jurisdiction.name());
    println!("  Filing Status: {}", receipt.filing_status.name());
    println!("  Tax Year: {}", receipt.tax_year);
    println!("  Timestamp: {}", receipt.timestamp);

    // Full proof details
    println!("\n=== Full Proof Summary ===");
    println!("{}", proof.summary());
    println!("  Receipt bytes: {} bytes", proof.receipt_bytes.len());

    println!("\n=== Use Cases ===");
    println!("1. KYC/AML: Prove income tier without exact amount");
    println!("2. Loans: Qualify for income requirements privately");
    println!("3. Benefits: Prove bracket eligibility without disclosure");
    println!("4. Tax Prep: Share bracket info with advisors safely");

    println!("\n✓ Demo complete!");
}
