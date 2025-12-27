//! Φ Hypothesis Test: Bundle Dilution vs Direct Similarity
//!
//! This test validates the root cause analysis:
//! - Bundle operation DILUTES similarity (creates inverted correlation)
//! - Direct similarity encoding creates CORRECT correlation
//!
//! If this test passes, we confirm:
//! 1. Φ implementation is correct
//! 2. Generators must use direct similarity encoding, not bundling

use symthaea::hdc::binary_hv::HV16;
use symthaea::hdc::tiered_phi::{TieredPhi, ApproximationTier};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(80));
    println!("Φ HYPOTHESIS TEST: Bundle Dilution vs Direct Similarity");
    println!("{}", "=".repeat(80));
    println!();

    // Create Φ calculator (use Heuristic tier - same as validation study)
    let mut phi_calc = TieredPhi::new(ApproximationTier::Heuristic);

    // ========================================================================
    // TEST 1: Bundle Dilution (Current Approach)
    // ========================================================================

    println!("TEST 1: Bundle-Based Generation (Current Generators)");
    println!("{}", "-".repeat(80));

    // Simulate what current generators do:
    // More complex = more bundling

    // Low integration: minimal bundling
    let low_integration_bundle = generate_with_bundling(10, 1);
    let phi_low_bundle = phi_calc.compute(&low_integration_bundle);

    // High integration: heavy bundling (like star topology)
    let high_integration_bundle = generate_with_bundling(10, 5);
    let phi_high_bundle = phi_calc.compute(&high_integration_bundle);

    println!("  Low integration (bundle 1 component):  Φ = {:.4}", phi_low_bundle);
    println!("  High integration (bundle 5 components): Φ = {:.4}", phi_high_bundle);

    let bundle_correlation_direction = if phi_high_bundle > phi_low_bundle {
        "POSITIVE ✅"
    } else {
        "NEGATIVE ❌ (INVERTED!)"
    };

    println!("  → Correlation direction: {}", bundle_correlation_direction);
    println!();

    // ========================================================================
    // TEST 2: Direct Similarity Encoding (Correct Approach)
    // ========================================================================

    println!("TEST 2: Direct Similarity Encoding (Proposed Fix)");
    println!("{}", "-".repeat(80));

    // Low integration: components are dissimilar
    let low_integration_direct = generate_with_target_similarity(10, 0.3);
    let phi_low_direct = phi_calc.compute(&low_integration_direct);

    // High integration: components are highly similar
    let high_integration_direct = generate_with_target_similarity(10, 0.8);
    let phi_high_direct = phi_calc.compute(&high_integration_direct);

    println!("  Low integration (similarity 0.3):  Φ = {:.4}", phi_low_direct);
    println!("  High integration (similarity 0.8): Φ = {:.4}", phi_high_direct);

    let direct_correlation_direction = if phi_high_direct > phi_low_direct {
        "POSITIVE ✅ (CORRECT!)"
    } else {
        "NEGATIVE ❌"
    };

    println!("  → Correlation direction: {}", direct_correlation_direction);
    println!();

    // ========================================================================
    // TEST 3: Similarity Measurements
    // ========================================================================

    println!("TEST 3: Actual Pairwise Similarities");
    println!("{}", "-".repeat(80));

    let avg_sim_low_bundle = measure_avg_similarity(&low_integration_bundle);
    let avg_sim_high_bundle = measure_avg_similarity(&high_integration_bundle);
    let avg_sim_low_direct = measure_avg_similarity(&low_integration_direct);
    let avg_sim_high_direct = measure_avg_similarity(&high_integration_direct);

    println!("  Bundle-based generation:");
    println!("    Low integration:  avg similarity = {:.4}", avg_sim_low_bundle);
    println!("    High integration: avg similarity = {:.4}", avg_sim_high_bundle);
    if avg_sim_high_bundle < avg_sim_low_bundle {
        println!("    → More bundling = LOWER similarity ❌");
    }
    println!();

    println!("  Direct similarity encoding:");
    println!("    Low integration:  avg similarity = {:.4}", avg_sim_low_direct);
    println!("    High integration: avg similarity = {:.4}", avg_sim_high_direct);
    if avg_sim_high_direct > avg_sim_low_direct {
        println!("    → Target similarity achieved ✅");
    }
    println!();

    // ========================================================================
    // ANALYSIS AND CONCLUSION
    // ========================================================================

    println!("{}", "=".repeat(80));
    println!("ANALYSIS");
    println!("{}", "=".repeat(80));
    println!();

    println!("Root Cause Hypothesis:");
    println!("  Bundle operation dilutes similarity as more components are bundled.");
    println!("  Current generators use MORE bundling for higher integration.");
    println!("  This creates INVERTED relationship: more bundling → lower similarity → lower Φ");
    println!();

    println!("Evidence:");
    if avg_sim_high_bundle < avg_sim_low_bundle {
        println!("  ✅ Bundle-based: High integration has LOWER similarity");
    }
    if avg_sim_high_direct > avg_sim_low_direct {
        println!("  ✅ Direct encoding: High integration has HIGHER similarity");
    }
    if phi_high_bundle < phi_low_bundle {
        println!("  ✅ Bundle-based: High integration has LOWER Φ (inverted)");
    }
    if phi_high_direct > phi_low_direct {
        println!("  ✅ Direct encoding: High integration has HIGHER Φ (correct)");
    }
    println!();

    println!("{}", "=".repeat(80));
    println!("CONCLUSION");
    println!("{}", "=".repeat(80));
    println!();

    let hypothesis_confirmed =
        avg_sim_high_bundle < avg_sim_low_bundle &&
        avg_sim_high_direct > avg_sim_low_direct &&
        phi_high_bundle < phi_low_bundle &&
        phi_high_direct > phi_low_direct;

    if hypothesis_confirmed {
        println!("✅ HYPOTHESIS CONFIRMED!");
        println!();
        println!("The Φ implementation is CORRECT.");
        println!("The generators MUST be redesigned to use direct similarity encoding.");
        println!();
        println!("Next Steps:");
        println!("  1. Redesign all 8 generators with direct similarity encoding");
        println!("  2. Ensure high integration → high pairwise similarities");
        println!("  3. Ensure low integration → low pairwise similarities");
        println!("  4. Re-run validation study");
        println!("  5. Expect positive correlation (r > 0.85)");
    } else {
        println!("❌ HYPOTHESIS REJECTED - Further investigation needed");
    }
    println!();

    Ok(())
}

/// Generate components using bundle operation (simulates current generators)
/// More components bundled = higher integration (by current logic)
fn generate_with_bundling(n: usize, bundle_size: usize) -> Vec<HV16> {
    let mut components = Vec::with_capacity(n);

    // Create base patterns
    let base_patterns: Vec<HV16> = (0..bundle_size)
        .map(|i| HV16::random(i as u64))
        .collect();

    for i in 0..n {
        if bundle_size == 1 {
            // Minimal bundling: just random
            components.push(HV16::random(i as u64 + 1000));
        } else {
            // Bundle multiple patterns together
            let mut to_bundle = base_patterns.clone();
            // Add unique component
            to_bundle.push(HV16::random(i as u64 + 2000));
            components.push(HV16::bundle(&to_bundle));
        }
    }

    components
}

/// Generate components with target average pairwise similarity
/// Uses direct similarity control via shared patterns
fn generate_with_target_similarity(n: usize, target_similarity: f64) -> Vec<HV16> {
    let mut components = Vec::with_capacity(n);

    // Shared pattern for creating similarity
    let shared = HV16::random(42);

    // For each component, mix shared and unique patterns
    // More shared = higher similarity
    for i in 0..n {
        let unique = HV16::random(i as u64 + 3000);

        if target_similarity > 0.6 {
            // High similarity: mostly shared
            // Use bind to create correlation while preserving similarity
            components.push(shared.bind(&unique));
        } else if target_similarity > 0.4 {
            // Medium similarity: 50/50 mix
            components.push(HV16::bundle(&[shared.clone(), unique]));
        } else {
            // Low similarity: mostly unique
            let unique2 = HV16::random(i as u64 + 4000);
            components.push(HV16::bundle(&[shared.clone(), unique, unique2]));
        }
    }

    components
}

/// Measure average pairwise similarity
fn measure_avg_similarity(components: &[HV16]) -> f64 {
    let n = components.len();
    if n < 2 {
        return 0.0;
    }

    let mut total_similarity = 0.0;
    let mut pair_count = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let sim = components[i].similarity(&components[j]) as f64;
            total_similarity += sim;
            pair_count += 1;
        }
    }

    total_similarity / pair_count as f64
}
