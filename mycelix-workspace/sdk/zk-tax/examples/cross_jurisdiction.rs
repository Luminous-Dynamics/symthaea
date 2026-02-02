//! Example: Cross-Jurisdiction Tax Bracket Proofs
//!
//! Demonstrates proving tax bracket status across multiple countries
//! simultaneously. Useful for:
//! - International tax treaty verification
//! - Expatriate tax planning
//! - Cross-border employment eligibility
//! - Multi-national tax optimization
//!
//! Run with: cargo run --example cross_jurisdiction

use mycelix_zk_tax::{FilingStatus, Jurisdiction};
use mycelix_zk_tax::proof::CrossJurisdictionProofBuilder;

fn main() {
    println!("=== Cross-Jurisdiction Tax Proof Examples ===\n");

    // =========================================================================
    // Example 1: US-UK Tax Treaty Verification
    // =========================================================================
    println!("--- Example 1: US-UK Tax Treaty Verification ---");

    let usd_income = 100_000;
    println!("Income: ${} USD equivalent (NEVER REVEALED)", usd_income);

    let proof = CrossJurisdictionProofBuilder::new(usd_income, 2024)
        .add_jurisdiction(Jurisdiction::US, FilingStatus::Single)
        .add_jurisdiction(Jurisdiction::UK, FilingStatus::Single)
        .build_dev()
        .expect("Should generate proof");

    println!("\nProof generated!");
    println!("  {}", proof.summary());
    println!("  Jurisdictions: {}", proof.jurisdiction_count());

    for info in &proof.jurisdiction_brackets {
        println!("\n  {} ({}):",
                 info.jurisdiction.name(),
                 info.jurisdiction.currency());
        println!("    Bracket: {} ({}%)",
                 info.bracket_index,
                 info.rate_bps as f64 / 100.0);
        println!("    Range: {} - {}",
                 info.bracket_lower,
                 if info.bracket_upper == u64::MAX { "+".to_string() } else { info.bracket_upper.to_string() });
    }

    println!("\n  Verification: {}", if proof.verify().is_ok() { "VALID" } else { "INVALID" });

    // =========================================================================
    // Example 2: OECD Countries Comparison
    // =========================================================================
    println!("\n--- Example 2: Major OECD Economies ---");

    let usd_income = 150_000;
    println!("Income: ${} USD equivalent (NEVER REVEALED)", usd_income);

    let oecd_proof = CrossJurisdictionProofBuilder::new(usd_income, 2024)
        .add_oecd_common(FilingStatus::Single)
        .build_dev()
        .expect("Should generate proof");

    println!("\nProof generated for {} countries!", oecd_proof.jurisdiction_count());
    println!("  {}", oecd_proof.summary());

    let (min_rate, max_rate) = oecd_proof.rate_range_bps().unwrap();
    println!("\n  Rate analysis:");
    println!("    Min marginal: {:.0}%", min_rate as f64 / 100.0);
    println!("    Max marginal: {:.0}%", max_rate as f64 / 100.0);
    println!("    Average: {:.1}%", oecd_proof.average_rate_bps() as f64 / 100.0);
    println!("    Rates within 5%: {}", oecd_proof.rates_within_tolerance(500));

    // =========================================================================
    // Example 3: Americas Regional Comparison
    // =========================================================================
    println!("\n--- Example 3: Americas (North & South) ---");

    let usd_income = 75_000;

    let americas_proof = CrossJurisdictionProofBuilder::new(usd_income, 2024)
        .add_jurisdictions(&[
            Jurisdiction::US,
            Jurisdiction::CA,
            Jurisdiction::MX,
            Jurisdiction::BR,
            Jurisdiction::AR,
        ], FilingStatus::Single)
        .build_dev()
        .expect("Should generate proof");

    println!("Income: ${} USD equivalent", usd_income);
    println!("\nProof generated for {} countries!", americas_proof.jurisdiction_count());

    println!("\n  Bracket breakdown:");
    for info in &americas_proof.jurisdiction_brackets {
        println!("    {}: {}% (bracket {})",
                 info.jurisdiction.name(),
                 info.rate_bps as f64 / 100.0,
                 info.bracket_index);
    }

    // =========================================================================
    // Example 4: High Income Expat Planning
    // =========================================================================
    println!("\n--- Example 4: High Income ($500K) Tax Comparison ---");

    let usd_income = 500_000;

    let high_income_proof = CrossJurisdictionProofBuilder::new(usd_income, 2024)
        .add_jurisdiction(Jurisdiction::US, FilingStatus::Single)
        .add_jurisdiction(Jurisdiction::UK, FilingStatus::Single)
        .add_jurisdiction(Jurisdiction::DE, FilingStatus::Single)
        .add_jurisdiction(Jurisdiction::FR, FilingStatus::Single)
        .add_jurisdiction(Jurisdiction::SA, FilingStatus::Single) // No income tax
        .build_dev()
        .expect("Should generate proof");

    println!("Income: ${} USD equivalent (NEVER REVEALED)", usd_income);
    println!("\nProof generated for {} jurisdictions!", high_income_proof.jurisdiction_count());

    println!("\n  High earner rates:");
    for info in &high_income_proof.jurisdiction_brackets {
        let rate = info.rate_bps as f64 / 100.0;
        let indicator = if rate == 0.0 { " (no income tax!)" }
                       else if rate >= 40.0 { " (high)" }
                       else if rate >= 30.0 { " (moderate)" }
                       else { " (low)" };
        println!("    {}: {:.0}%{}", info.jurisdiction.name(), rate, indicator);
    }

    // =========================================================================
    // Example 5: Consistent Bracket Analysis
    // =========================================================================
    println!("\n--- Example 5: Bracket Consistency Analysis ---");

    let usd_income = 40_000;  // Lower income for similar brackets

    let consistent_proof = CrossJurisdictionProofBuilder::new(usd_income, 2024)
        .add_jurisdiction(Jurisdiction::US, FilingStatus::Single)
        .add_jurisdiction(Jurisdiction::UK, FilingStatus::Single)
        .add_jurisdiction(Jurisdiction::CA, FilingStatus::Single)
        .build_dev()
        .expect("Should generate proof");

    println!("Income: ${} USD equivalent", usd_income);
    println!("\nConsistency check:");
    match consistent_proof.consistent_bracket() {
        Some(bracket) => println!("  All in bracket {} - CONSISTENT!", bracket),
        None => println!("  Different brackets across jurisdictions"),
    }

    println!("  Rates within 10%: {}", consistent_proof.rates_within_tolerance(1000));

    // =========================================================================
    // Example 6: Tamper Detection
    // =========================================================================
    println!("\n--- Example 6: Proof Integrity ---");

    let proof = CrossJurisdictionProofBuilder::new(100_000, 2024)
        .add_jurisdiction(Jurisdiction::US, FilingStatus::Single)
        .add_jurisdiction(Jurisdiction::UK, FilingStatus::Single)
        .build_dev()
        .expect("Should generate proof");

    println!("Original verification: {}", if proof.verify().is_ok() { "VALID" } else { "INVALID" });

    // Simulate tampering
    let mut tampered = proof.clone();
    tampered.jurisdiction_brackets[0].rate_bps = 9999; // Fake rate

    println!("After tampering: {}", if tampered.verify().is_ok() { "VALID" } else { "DETECTED - INVALID" });

    println!("\n=== Key Takeaways ===");
    println!("- Compare tax brackets across multiple countries in one proof");
    println!("- Currency conversion handled automatically");
    println!("- Useful for expat planning, tax treaty verification");
    println!("- Cryptographic commitment prevents tampering");
    println!("- Income is NEVER revealed to any jurisdiction");

    println!("\n Done!");
}
