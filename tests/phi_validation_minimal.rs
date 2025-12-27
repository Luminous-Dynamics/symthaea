//! Minimal Φ Validation Test: Random vs Star Topologies
//!
//! This test validates that network topology affects integrated information (Φ).
//!
//! **Hypothesis**: Star topology should have significantly higher Φ than random topology
//!
//! **Success Criteria**:
//! - p < 0.05 (statistically significant)
//! - Cohen's d > 0.5 (large effect size)
//! - Φ_star > Φ_random (correct direction)

use symthaea::hdc::phi_topology_validation::MinimalPhiValidation;

#[test]
fn test_minimal_phi_validation_quick() {
    println!("\n🔬 QUICK Φ VALIDATION TEST (10 samples each)");
    println!("================================================\n");

    // Run quick validation (10 samples each for speed)
    let mut validation = MinimalPhiValidation::quick();
    let result = validation.run();

    println!("\n📊 RESULTS:");
    println!("{}", result.summary());
    println!();

    // Assertions
    println!("✅ VALIDATION CHECKS:");
    println!("   Random < Star: {} (Φ_random={:.4} < Φ_star={:.4})",
        result.mean_phi_random < result.mean_phi_star,
        result.mean_phi_random,
        result.mean_phi_star
    );
    println!("   Significant: {} (p={:.4} < 0.05)", result.is_significant(), result.p_value);
    println!("   Large Effect: {} (d={:.3} > 0.5)", result.has_large_effect(), result.effect_size);
    println!("   Overall: {}", if result.validation_succeeded() { "✅ PASSED" } else { "❌ FAILED" });
    println!();

    // Report regardless of pass/fail
    assert!(
        result.mean_phi_star > result.mean_phi_random,
        "Star topology should have higher Φ than random (got random={:.4}, star={:.4})",
        result.mean_phi_random,
        result.mean_phi_star
    );
}

#[test]
#[ignore] // Run with `cargo test --ignored` for full validation
fn test_minimal_phi_validation_standard() {
    println!("\n🔬 STANDARD Φ VALIDATION TEST (20 samples each)");
    println!("==================================================\n");

    // Run standard validation (20 samples each)
    let mut validation = MinimalPhiValidation::standard();
    let result = validation.run();

    println!("\n📊 RESULTS:");
    println!("{}", result.summary());
    println!();

    // Statistical assertions
    println!("✅ VALIDATION CHECKS:");
    println!("   Random < Star: {}", result.mean_phi_random < result.mean_phi_star);
    println!("   Significant (p<0.05): {}", result.is_significant());
    println!("   Large Effect (d>0.5): {}", result.has_large_effect());
    println!("   Overall Success: {}", result.validation_succeeded());
    println!();

    // Main assertion
    assert!(
        result.validation_succeeded(),
        "Validation should show Star > Random with p<0.05 and d>0.5\n\
         Got: Random={:.4}±{:.4}, Star={:.4}±{:.4}, p={:.4}, d={:.3}",
        result.mean_phi_random, result.std_phi_random,
        result.mean_phi_star, result.std_phi_star,
        result.p_value, result.effect_size
    );
}

#[test]
#[ignore] // Run with `cargo test --ignored` for publication-ready validation
fn test_minimal_phi_validation_publication() {
    println!("\n🔬 PUBLICATION Φ VALIDATION TEST (50 samples each)");
    println!("=====================================================\n");

    // Run publication-quality validation (50 samples each)
    let mut validation = MinimalPhiValidation::publication();
    let result = validation.run();

    println!("\n📊 RESULTS:");
    println!("{}", result.summary());
    println!();

    // Detailed output for publication
    println!("📈 DETAILED STATISTICS:");
    println!("   Sample Sizes: n_random={}, n_star={}", result.n_random, result.n_star);
    println!("   Random: M={:.4}, SD={:.4}", result.mean_phi_random, result.std_phi_random);
    println!("   Star:   M={:.4}, SD={:.4}", result.mean_phi_star, result.std_phi_star);
    println!("   t({:.1}) = {:.3}, p = {:.6}", result.degrees_of_freedom, result.t_statistic, result.p_value);
    println!("   Cohen's d = {:.3}", result.effect_size);
    println!("   Performance: {}ms total, {:.2}ms per Φ", result.total_time_ms, result.avg_time_per_phi_ms);
    println!();

    // Publication criteria
    assert!(
        result.validation_succeeded(),
        "Publication-quality validation failed"
    );

    assert!(
        result.p_value < 0.01,
        "Publication requires p < 0.01, got p = {:.6}",
        result.p_value
    );

    assert!(
        result.effect_size > 0.8,
        "Publication requires large effect (d > 0.8), got d = {:.3}",
        result.effect_size
    );
}
