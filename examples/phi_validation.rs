//! Minimal Φ Validation Example
//!
//! This example runs the minimal Φ validation comparing Random vs Star topologies.
//!
//! **Usage**:
//! ```bash
//! cargo run --example phi_validation --release
//! ```

use symthaea::hdc::phi_topology_validation::MinimalPhiValidation;

fn main() {
    println!("\n🔬 MINIMAL Φ VALIDATION: Random vs Star Topologies");
    println!("=====================================================\n");

    // Run quick validation (10 samples each)
    println!("⚙️  Initializing validation with 10 samples per topology...\n");

    let mut validation = MinimalPhiValidation::quick();
    let result = validation.run();

    println!("\n{}", "=".repeat(60));
    println!("📊 FINAL RESULTS");
    println!("{}", "=".repeat(60));
    println!("{}", result.summary());
    println!("{}", "=".repeat(60));
    println!();

    // Detailed analysis
    println!("📈 DETAILED ANALYSIS:");
    println!();
    println!("   Hypothesis: Star topology should have higher Φ than Random");
    println!();
    println!("   ✓ Direction Test:");
    if result.mean_phi_star > result.mean_phi_random {
        println!("     ✅ PASS: Φ_star ({:.4}) > Φ_random ({:.4})",
            result.mean_phi_star, result.mean_phi_random);
        let diff = result.mean_phi_star - result.mean_phi_random;
        let pct = (diff / result.mean_phi_random) * 100.0;
        println!("     Φ difference: {:.4} ({:.1}% higher)", diff, pct);
    } else {
        println!("     ❌ FAIL: Star did not have higher Φ");
        println!("     Φ_star={:.4}, Φ_random={:.4}", result.mean_phi_star, result.mean_phi_random);
    }
    println!();

    println!("   ✓ Statistical Significance:");
    if result.is_significant() {
        println!("     ✅ PASS: p = {:.4} < 0.05", result.p_value);
        println!("     The difference is statistically significant!");
    } else {
        println!("     ⚠️  Not significant: p = {:.4} ≥ 0.05", result.p_value);
        println!("     This may be due to small sample size (n={}).", result.n_random);
    }
    println!();

    println!("   ✓ Effect Size:");
    if result.has_large_effect() {
        println!("     ✅ PASS: Cohen's d = {:.3} > 0.5", result.effect_size);
        println!("     This is a large effect size!");
    } else {
        println!("     ⚠️  Effect size: Cohen's d = {:.3} ≤ 0.5", result.effect_size);
    }
    println!();

    println!("{}", "=".repeat(60));
    println!("🎯 OVERALL VALIDATION");
    println!("{}", "=".repeat(60));
    if result.validation_succeeded() {
        println!("✅ SUCCESS! All validation criteria met:");
        println!("   • Direction correct (Star > Random)");
        println!("   • Statistically significant (p < 0.05)");
        println!("   • Large effect size (d > 0.5)");
        println!();
        println!("🎉 HYPOTHESIS CONFIRMED:");
        println!("   Network topology DOES affect integrated information (Φ)!");
        println!("   Star topology has significantly higher Φ than random topology.");
    } else if result.mean_phi_star > result.mean_phi_random {
        println!("⚠️  PARTIAL SUCCESS:");
        println!("   • Direction correct (Star > Random) ✓");
        if !result.is_significant() {
            println!("   • Not statistically significant (p={:.4}) ✗", result.p_value);
        }
        if !result.has_large_effect() {
            println!("   • Effect size not large (d={:.3}) ✗", result.effect_size);
        }
        println!();
        println!("💡 RECOMMENDATION:");
        println!("   The trend is in the right direction. Consider:");
        println!("   • Increasing sample size (try 20 or 50 samples)");
        println!("   • Using more nodes per topology (try 12-15 nodes)");
    } else {
        println!("❌ VALIDATION FAILED:");
        println!("   Direction incorrect: Star does not have higher Φ");
        println!("   This suggests an implementation issue.");
    }
    println!("{}", "=".repeat(60));
    println!();

    println!("⏱️  Performance: {}ms total, {:.2}ms per Φ calculation",
        result.total_time_ms, result.avg_time_per_phi_ms);
    println!();

    // Exit with appropriate code
    if result.validation_succeeded() {
        println!("✨ Validation complete: Hypothesis confirmed! ✨");
        std::process::exit(0);
    } else if result.mean_phi_star > result.mean_phi_random {
        println!("⚠️  Validation complete: Trend correct but not statistically robust.");
        std::process::exit(1);
    } else {
        println!("❌ Validation complete: Hypothesis not supported.");
        std::process::exit(2);
    }
}
