//! Example: Effective Tax Rate Proofs
//!
//! Demonstrates proving your ACTUAL tax burden (effective rate) rather than
//! just your marginal bracket. The effective rate is total tax / total income.
//!
//! This is more accurate for:
//! - Cross-jurisdiction tax comparison
//! - Financial planning
//! - Tax optimization verification
//!
//! Run with: cargo run --example effective_rate

use mycelix_zk_tax::{FilingStatus, Jurisdiction};
use mycelix_zk_tax::proof::{EffectiveTaxRateProof, EffectiveTaxRateBuilder};

fn main() {
    println!("=== Effective Tax Rate Proof Examples ===\n");

    println!("Understanding the difference:");
    println!("  - MARGINAL rate: Tax rate on your LAST dollar of income");
    println!("  - EFFECTIVE rate: Total tax / Total income (your REAL burden)");
    println!();

    // =========================================================================
    // Example 1: Middle Income - Progressive Tax Benefit
    // =========================================================================
    println!("--- Example 1: Middle Income ($85K) ---");

    let income = 85_000;
    println!("Income: ${} (NEVER REVEALED)", income);

    let proof = EffectiveTaxRateProof::prove_dev(
        income,
        Jurisdiction::US,
        FilingStatus::Single,
        2024,
    ).expect("Should generate proof");

    println!("\nProof generated!");
    println!("  {}", proof.summary());
    println!("\nKey metrics:");
    println!("  Marginal bracket: {} ({}%)", proof.marginal_bracket, proof.marginal_rate_percent());
    println!("  Effective rate: {:.1}%", proof.effective_rate_percent());
    println!("  Rate range: {:.1}% - {:.1}%",
             proof.effective_rate_lower_bps as f64 / 100.0,
             proof.effective_rate_upper_bps as f64 / 100.0);
    println!("  Progressive savings: {:.1}% lower than marginal!",
             proof.progressive_tax_savings_bps() as f64 / 100.0);
    println!("  Verification: {}", if proof.verify().is_ok() { "VALID" } else { "INVALID" });

    // =========================================================================
    // Example 2: Low Income - Maximum Progressive Benefit
    // =========================================================================
    println!("\n--- Example 2: Low Income ($25K) ---");

    let income = 25_000;
    let proof = EffectiveTaxRateProof::prove_dev(
        income,
        Jurisdiction::US,
        FilingStatus::Single,
        2024,
    ).expect("Should generate proof");

    println!("Income: ${} (NEVER REVEALED)", income);
    println!("\nProof generated!");
    println!("  Marginal rate: {:.0}%", proof.marginal_rate_percent());
    println!("  Effective rate: {:.1}%", proof.effective_rate_percent());
    println!("  Progressive savings: {:.1}%", proof.progressive_tax_savings_bps() as f64 / 100.0);

    // =========================================================================
    // Example 3: High Income - Tax Burden Analysis
    // =========================================================================
    println!("\n--- Example 3: High Income ($500K) ---");

    let income = 500_000;
    let proof = EffectiveTaxRateProof::prove_dev(
        income,
        Jurisdiction::US,
        FilingStatus::Single,
        2024,
    ).expect("Should generate proof");

    println!("Income: ${} (NEVER REVEALED)", income);
    println!("\nProof generated!");
    println!("  Marginal rate: {:.0}%", proof.marginal_rate_percent());
    println!("  Effective rate: {:.1}%", proof.effective_rate_percent());
    println!("  Progressive savings: {:.1}%", proof.progressive_tax_savings_bps() as f64 / 100.0);

    // Even at 35% marginal, effective rate is lower due to progressive brackets

    // =========================================================================
    // Example 4: Married Filing Jointly Comparison
    // =========================================================================
    println!("\n--- Example 4: Single vs Married (Same Income) ---");

    let income = 150_000;

    let single_proof = EffectiveTaxRateProof::prove_dev(
        income,
        Jurisdiction::US,
        FilingStatus::Single,
        2024,
    ).expect("Should generate proof");

    let mfj_proof = EffectiveTaxRateProof::prove_dev(
        income,
        Jurisdiction::US,
        FilingStatus::MarriedFilingJointly,
        2024,
    ).expect("Should generate proof");

    println!("Income: ${} (NEVER REVEALED)", income);
    println!("\nSingle filer:");
    println!("  Marginal: {:.0}%, Effective: {:.1}%",
             single_proof.marginal_rate_percent(),
             single_proof.effective_rate_percent());
    println!("\nMarried Filing Jointly:");
    println!("  Marginal: {:.0}%, Effective: {:.1}%",
             mfj_proof.marginal_rate_percent(),
             mfj_proof.effective_rate_percent());
    println!("\nTax savings from MFJ: {:.1}% lower effective rate",
             single_proof.effective_rate_percent() - mfj_proof.effective_rate_percent());

    // =========================================================================
    // Example 5: Using the Builder Pattern
    // =========================================================================
    println!("\n--- Example 5: Builder Pattern ---");

    let proof = EffectiveTaxRateBuilder::new(
        100_000,
        Jurisdiction::US,
        FilingStatus::Single,
        2024,
    ).build().expect("Should generate proof");

    println!("Using EffectiveTaxRateBuilder for cleaner API:");
    println!("  {}", proof.summary());

    // =========================================================================
    // Example 6: Different Jurisdictions
    // =========================================================================
    println!("\n--- Example 6: UK Effective Rate (Approximation) ---");

    let uk_proof = EffectiveTaxRateProof::prove_dev(
        50_000,  // 50K GBP equivalent
        Jurisdiction::UK,
        FilingStatus::Single,
        2024,
    ).expect("Should generate proof");

    println!("UK income: 50,000 GBP (NEVER REVEALED)");
    println!("\nProof generated!");
    println!("  {}", uk_proof.summary());
    println!("  Note: Non-US jurisdictions use approximation (70% of marginal)");

    println!("\n=== Key Takeaways ===");
    println!("- Effective rate is always lower than marginal (progressive tax)");
    println!("- Higher incomes see bigger spread between marginal/effective");
    println!("- Filing status significantly impacts effective rate");
    println!("- Exact income is NEVER revealed - only rate ranges");

    println!("\n Done!");
}
