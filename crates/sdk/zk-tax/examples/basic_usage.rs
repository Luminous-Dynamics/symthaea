// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Basic usage example for the ZK-Tax SDK.
//!
//! Run with: `cargo run --example basic_usage --features prover`

use mycelix_zk_tax::{
    FilingStatus, Jurisdiction, TaxBracketProver,
    brackets::get_brackets,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Mycelix ZK-Tax Basic Usage Example ===\n");

    // Create a prover in dev mode (fast, for testing)
    let prover = TaxBracketProver::dev_mode();

    // ==========================================================================
    // Example 1: Generate a simple tax bracket proof
    // ==========================================================================
    println!("1. Generating tax bracket proof for $85,000 income (US, Single, 2024)");

    let proof = prover.prove(
        85_000,                     // Income (private - never revealed)
        Jurisdiction::US,           // Country
        FilingStatus::Single,       // Filing status
        2024,                       // Tax year
    )?;

    println!("   Bracket Index: {}", proof.bracket_index);
    println!("   Tax Rate: {}%", proof.rate_bps as f64 / 100.0);
    println!("   Bracket Range: ${} - ${}", proof.bracket_lower, proof.bracket_upper);
    println!("   Commitment: {}\n", proof.commitment.to_hex());

    // ==========================================================================
    // Example 2: Verify the proof
    // ==========================================================================
    println!("2. Verifying the proof");

    proof.verify()?;
    println!("   ✓ Proof verified successfully!\n");

    // ==========================================================================
    // Example 3: Query tax brackets
    // ==========================================================================
    println!("3. US 2024 Tax Brackets (Single Filing)");

    let brackets = get_brackets(Jurisdiction::US, 2024, FilingStatus::Single)?;
    for (i, bracket) in brackets.iter().enumerate() {
        let upper = if bracket.upper == u64::MAX {
            "∞".to_string()
        } else {
            format!("${}", bracket.upper)
        };
        println!(
            "   Bracket {}: ${} - {} @ {}%",
            i,
            bracket.lower,
            upper,
            bracket.rate_bps as f64 / 100.0
        );
    }

    // ==========================================================================
    // Example 4: Different jurisdictions
    // ==========================================================================
    println!("\n4. Tax brackets in different countries for $100,000 income");

    let jurisdictions = [
        Jurisdiction::US,
        Jurisdiction::UK,
        Jurisdiction::DE,
        Jurisdiction::FR,
        Jurisdiction::JP,
    ];

    for j in jurisdictions {
        let proof = prover.prove(100_000, j, FilingStatus::Single, 2024)?;
        println!(
            "   {}: Bracket {} @ {}%",
            j.name(),
            proof.bracket_index,
            proof.rate_bps as f64 / 100.0
        );
    }

    // ==========================================================================
    // Example 5: Different filing statuses
    // ==========================================================================
    println!("\n5. Same income ($150,000), different filing statuses (US 2024)");

    let statuses = [
        FilingStatus::Single,
        FilingStatus::MarriedFilingJointly,
        FilingStatus::MarriedFilingSeparately,
        FilingStatus::HeadOfHousehold,
    ];

    for status in statuses {
        let proof = prover.prove(150_000, Jurisdiction::US, status, 2024)?;
        println!(
            "   {}: Bracket {} @ {}%",
            status.name(),
            proof.bracket_index,
            proof.rate_bps as f64 / 100.0
        );
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
