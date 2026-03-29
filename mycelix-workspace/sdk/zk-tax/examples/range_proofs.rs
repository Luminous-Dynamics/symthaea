// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Example: Income Range Proofs
//!
//! Demonstrates how to prove income is within a specified range
//! without revealing the exact amount. Useful for:
//! - Rental applications (prove income >= 3x rent)
//! - Loan qualification (prove income above minimum)
//! - Benefits eligibility (prove income below maximum)
//!
//! Run with: cargo run --example range_proofs

use mycelix_zk_tax::proof::RangeProofBuilder;

fn main() {
    println!("=== Income Range Proof Examples ===\n");

    // =========================================================================
    // Example 1: Rental Application - Prove income >= 3x annual rent
    // =========================================================================
    println!("--- Example 1: Rental Application ---");

    let monthly_rent = 2_500;
    let annual_rent = monthly_rent * 12; // $30,000
    let my_income = 95_000; // My actual income (PRIVATE!)

    println!("Scenario: Landlord requires income >= 3x annual rent");
    println!("  Monthly rent: ${}", monthly_rent);
    println!("  3x requirement: ${}/year", annual_rent * 3);
    println!("  My income: ${}  (NEVER REVEALED)", my_income);

    let rental_proof = RangeProofBuilder::new(my_income, 2024)
        .prove_multiple_of(annual_rent, 3)
        .expect("Should qualify!");

    println!("\nProof generated!");
    println!("  {}", rental_proof.summary());
    println!("  Landlord sees: Income >= ${}", rental_proof.range_lower);
    println!("  Landlord does NOT see: Exact income");
    println!("  Verification: {}", if rental_proof.verify().is_ok() { "VALID" } else { "INVALID" });

    // =========================================================================
    // Example 2: Loan Qualification - Prove income above minimum
    // =========================================================================
    println!("\n--- Example 2: Loan Qualification ---");

    let minimum_for_loan = 50_000;
    let my_income = 78_000;

    println!("Scenario: Loan requires minimum income of ${}", minimum_for_loan);
    println!("  My income: ${}  (NEVER REVEALED)", my_income);

    let loan_proof = RangeProofBuilder::new(my_income, 2024)
        .prove_above(minimum_for_loan)
        .expect("Should qualify!");

    println!("\nProof generated!");
    println!("  {}", loan_proof.summary());
    println!("  Bank sees: Income >= ${}", loan_proof.range_lower);
    println!("  Verification: {}", if loan_proof.verify().is_ok() { "VALID" } else { "INVALID" });

    // =========================================================================
    // Example 3: Benefits Eligibility - Prove income below threshold
    // =========================================================================
    println!("\n--- Example 3: Benefits Eligibility ---");

    let maximum_for_benefits = 75_000;
    let my_income = 62_000;

    println!("Scenario: Benefit program requires income <= ${}", maximum_for_benefits);
    println!("  My income: ${}  (NEVER REVEALED)", my_income);

    let benefits_proof = RangeProofBuilder::new(my_income, 2024)
        .prove_below(maximum_for_benefits)
        .expect("Should qualify!");

    println!("\nProof generated!");
    println!("  {}", benefits_proof.summary());
    println!("  Agency sees: Income <= ${}", benefits_proof.range_upper);
    println!("  Verification: {}", if benefits_proof.verify().is_ok() { "VALID" } else { "INVALID" });

    // =========================================================================
    // Example 4: Custom Range - Prove income within specific bounds
    // =========================================================================
    println!("\n--- Example 4: Custom Income Band ---");

    let my_income = 85_000;
    let range_lower = 70_000;
    let range_upper = 100_000;

    println!("Scenario: Prove income is between ${} and ${}", range_lower, range_upper);
    println!("  My income: ${}  (NEVER REVEALED)", my_income);

    let band_proof = RangeProofBuilder::new(my_income, 2024)
        .prove_between(range_lower, range_upper)
        .expect("Should fit in range!");

    println!("\nProof generated!");
    println!("  {}", band_proof.summary());
    println!("  Range width: ${}", band_proof.range_width());
    println!("  Contains $80K: {}", band_proof.contains(80_000));
    println!("  Contains $50K: {}", band_proof.contains(50_000));
    println!("  Verification: {}", if band_proof.verify().is_ok() { "VALID" } else { "INVALID" });

    // =========================================================================
    // Example 5: Failed Qualification
    // =========================================================================
    println!("\n--- Example 5: Insufficient Income (Failure Case) ---");

    let my_income = 45_000;
    let minimum_required = 60_000;

    println!("Scenario: Trying to prove income >= ${} with actual ${}", minimum_required, my_income);

    match RangeProofBuilder::new(my_income, 2024).prove_above(minimum_required) {
        Ok(_) => println!("Unexpectedly succeeded!"),
        Err(e) => println!("  Correctly failed: {}", e),
    }

    println!("\n=== Key Takeaways ===");
    println!("- Range proofs verify income is within bounds");
    println!("- Exact income is NEVER revealed to verifier");
    println!("- Cryptographic commitment ensures integrity");
    println!("- Failed proofs indicate ineligibility (no false claims)");

    println!("\n Done!");
}
