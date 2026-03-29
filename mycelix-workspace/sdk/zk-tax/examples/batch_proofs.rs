// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Example: Multi-Year Batch Proofs
//!
//! Demonstrates proving consistent income bracket across multiple years
//! without revealing exact incomes. Useful for:
//! - Mortgage applications (3-year income history)
//! - Visa applications (proof of financial stability)
//! - Employment verification (career progression)
//!
//! Run with: cargo run --example batch_proofs

use mycelix_zk_tax::{FilingStatus, Jurisdiction};
use mycelix_zk_tax::proof::{BatchProofBuilder, prove_three_year_history};

fn main() {
    println!("=== Multi-Year Batch Proof Examples ===\n");

    // =========================================================================
    // Example 1: Mortgage Application - 3-Year Income History
    // =========================================================================
    println!("--- Example 1: Mortgage Application (3-Year History) ---");

    let incomes = [
        (2022, 78_000u64),
        (2023, 82_000),
        (2024, 85_000),
    ];

    println!("Scenario: Lender requires 3-year income consistency");
    for (year, income) in &incomes {
        println!("  {}: ${} (NEVER REVEALED)", year, income);
    }

    let mortgage_proof = prove_three_year_history(
        incomes,
        Jurisdiction::US,
        FilingStatus::Single,
    ).expect("Should generate proof");

    println!("\nProof generated!");
    println!("  {}", mortgage_proof.summary());
    println!("  Year count: {}", mortgage_proof.year_count());
    println!("  Year range: {:?}", mortgage_proof.year_range());
    println!("  Consistent bracket: {:?}", mortgage_proof.consistent_bracket());
    println!("  Shows growth: {}", mortgage_proof.shows_growth());
    println!("  Verification: {}", if mortgage_proof.verify().is_ok() { "VALID" } else { "INVALID" });

    // =========================================================================
    // Example 2: Career Growth Demonstration
    // =========================================================================
    println!("\n--- Example 2: Career Progression Proof ---");

    let career_history = BatchProofBuilder::new(Jurisdiction::US, FilingStatus::Single)
        .add_year(2020, 45_000)   // Entry level
        .add_year(2021, 55_000)   // Promotion
        .add_year(2022, 75_000)   // Senior role
        .add_year(2023, 95_000)   // Lead position
        .add_year(2024, 120_000)  // Director
        .build_dev()
        .expect("Should generate proof");

    println!("Scenario: Demonstrating 5-year career growth");
    println!("\nProof generated!");
    println!("  {}", career_history.summary());
    println!("  Min bracket: {:?}", career_history.min_bracket());
    println!("  Max bracket: {:?}", career_history.max_bracket());
    println!("  Shows growth: {}", career_history.shows_growth());
    println!("  Verification: {}", if career_history.verify().is_ok() { "VALID" } else { "INVALID" });

    // =========================================================================
    // Example 3: Visa Application - Consistent Income
    // =========================================================================
    println!("\n--- Example 3: Visa Application (Income Consistency) ---");

    let visa_proof = BatchProofBuilder::new(Jurisdiction::UK, FilingStatus::Single)
        .add_years_uniform(&[2022, 2023, 2024], 55_000)
        .build_dev()
        .expect("Should generate proof");

    println!("Scenario: UK visa requires consistent income over 3 years");
    println!("\nProof generated!");
    println!("  {}", visa_proof.summary());
    println!("  All in same bracket: {:?}", visa_proof.consistent_bracket());
    println!("  Verification: {}", if visa_proof.verify().is_ok() { "VALID" } else { "INVALID" });

    // =========================================================================
    // Example 4: Full History with Builder Pattern
    // =========================================================================
    println!("\n--- Example 4: Complete 6-Year History (All Supported Years) ---");

    let full_history = BatchProofBuilder::new(Jurisdiction::US, FilingStatus::Single)
        .add_year_range(2020, 2025, 85_000)
        .build_dev()
        .expect("Should generate proof");

    println!("Scenario: Full 6-year income history (2020-2025)");
    println!("\nProof generated!");
    println!("  {}", full_history.summary());
    println!("  Year count: {}", full_history.year_count());
    println!("  Verification: {}", if full_history.verify().is_ok() { "VALID" } else { "INVALID" });

    // =========================================================================
    // Example 5: Declining Income Pattern
    // =========================================================================
    println!("\n--- Example 5: Declining Income Pattern ---");

    let declining = BatchProofBuilder::new(Jurisdiction::US, FilingStatus::Single)
        .add_year(2022, 150_000)
        .add_year(2023, 100_000)
        .add_year(2024, 75_000)
        .build_dev()
        .expect("Should generate proof");

    println!("Scenario: Income decline (e.g., career change)");
    println!("\nProof generated!");
    println!("  {}", declining.summary());
    println!("  Shows growth: {} (expected: false)", declining.shows_growth());
    println!("  Min bracket: {:?}", declining.min_bracket());
    println!("  Max bracket: {:?}", declining.max_bracket());

    // =========================================================================
    // Example 6: Married Filing Jointly
    // =========================================================================
    println!("\n--- Example 6: Married Filing Jointly ---");

    let mfj_proof = BatchProofBuilder::new(Jurisdiction::US, FilingStatus::MarriedFilingJointly)
        .add_year(2022, 180_000)  // Combined household income
        .add_year(2023, 195_000)
        .add_year(2024, 210_000)
        .build_dev()
        .expect("Should generate proof");

    println!("Scenario: Joint filing household income history");
    println!("\nProof generated!");
    println!("  {}", mfj_proof.summary());
    println!("  Shows growth: {}", mfj_proof.shows_growth());
    println!("  Verification: {}", if mfj_proof.verify().is_ok() { "VALID" } else { "INVALID" });

    println!("\n=== Key Takeaways ===");
    println!("- Batch proofs combine multiple years into one verifiable package");
    println!("- Can prove consistency, growth, or specific patterns");
    println!("- Individual year incomes are NEVER revealed");
    println!("- Useful for mortgage, visa, and employment verification");

    println!("\n Done!");
}
