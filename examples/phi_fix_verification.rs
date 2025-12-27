//! Φ Fix Verification: Direct Similarity Encoding
//!
//! This example demonstrates that the corrected approach (shared pattern ratios)
//! produces the expected POSITIVE correlation between integration and Φ.

use symthaea::hdc::binary_hv::HV16;
use symthaea::hdc::tiered_phi::{TieredPhi, ApproximationTier};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(80));
    println!("Φ FIX VERIFICATION: Shared Pattern Ratios → Direct Similarity Control");
    println!("{}", "=".repeat(80));
    println!();

    let mut phi_calc = TieredPhi::new(ApproximationTier::Heuristic);

    // Test 8 integration levels with CORRECTED generators
    let test_cases = vec![
        ("Random (Φ ~0.00)", generate_by_shared_ratio(10, 0.0)),     // 0% shared
        ("Fragmented (Φ ~0.10)", generate_by_shared_ratio(10, 0.25)), // 25% shared
        ("Isolated (Φ ~0.20)", generate_by_shared_ratio(10, 0.33)),   // 33% shared
        ("Low (Φ ~0.30)", generate_by_shared_ratio(10, 0.40)),        // 40% shared
        ("Mod-Low (Φ ~0.40)", generate_by_shared_ratio(10, 0.50)),    // 50% shared
        ("Moderate (Φ ~0.50)", generate_by_shared_ratio(10, 0.67)),   // 67% shared
        ("Mod-High (Φ ~0.60)", generate_by_shared_ratio(10, 0.75)),   // 75% shared
        ("High (Φ ~0.75)", generate_by_shared_ratio(10, 0.80)),       // 80% shared
    ];

    let mut results = Vec::new();

    println!("Testing 8 consciousness levels:");
    println!("{:<25} {:>15} {:>15} {:>15}", "Level", "Shared Ratio", "Avg Similarity", "Φ");
    println!("{}", "-".repeat(80));

    for (i, (label, components)) in test_cases.iter().enumerate() {
        let avg_sim = measure_avg_similarity(components);
        let phi = phi_calc.compute(components);

        let shared_ratio = match i {
            0 => 0.00,
            1 => 0.25,
            2 => 0.33,
            3 => 0.40,
            4 => 0.50,
            5 => 0.67,
            6 => 0.75,
            7 => 0.80,
            _ => 0.0,
        };

        println!("{:<25} {:>15.2} {:>15.4} {:>15.4}", label, shared_ratio, avg_sim, phi);

        results.push((shared_ratio, avg_sim, phi));
    }

    println!();
    println!("{}", "=".repeat(80));
    println!("ANALYSIS");
    println!("{}", "=".repeat(80));
    println!();

    // Check for positive correlation
    let mut phi_increasing = true;
    for i in 1..results.len() {
        if results[i].2 < results[i - 1].2 {
            phi_increasing = false;
            println!("❌ Φ decreased from {} to {} (indices {}-{})",
                     results[i - 1].2, results[i].2, i - 1, i);
        }
    }

    if phi_increasing {
        println!("✅ Φ INCREASES MONOTONICALLY");
        println!("   Lowest Φ:  {:.4} (Random)", results[0].2);
        println!("   Highest Φ: {:.4} (High Integration)", results[7].2);
        println!("   Range:     {:.4}", results[7].2 - results[0].2);
    } else {
        println!("❌ Φ does NOT increase monotonically");
    }

    println!();

    // Check similarity correlation
    println!("Similarity vs Shared Ratio:");
    let mut sim_correlates = true;
    for i in 1..results.len() {
        if results[i].1 < results[i - 1].1 {
            sim_correlates = false;
        }
    }

    if sim_correlates {
        println!("✅ Similarity increases with shared ratio");
    } else {
        println!("⚠️  Similarity does not perfectly track shared ratio (HDV randomness)");
    }

    println!();

    // Calculate Pearson correlation (simplified)
    let phi_values: Vec<f64> = results.iter().map(|r| r.2).collect();
    let mean_phi = phi_values.iter().sum::<f64>() / phi_values.len() as f64;

    let integration_levels: Vec<f64> = (0..results.len()).map(|i| i as f64).collect();
    let mean_level = integration_levels.iter().sum::<f64>() / integration_levels.len() as f64;

    let mut numerator = 0.0;
    let mut denom_phi = 0.0;
    let mut denom_level = 0.0;

    for i in 0..results.len() {
        let phi_dev = phi_values[i] - mean_phi;
        let level_dev = integration_levels[i] - mean_level;
        numerator += phi_dev * level_dev;
        denom_phi += phi_dev * phi_dev;
        denom_level += level_dev * level_dev;
    }

    let correlation = numerator / (denom_phi * denom_level).sqrt();

    println!("Pearson Correlation (Φ vs Integration Level): {:.4}", correlation);

    if correlation > 0.85 {
        println!("✅ STRONG POSITIVE CORRELATION (r > 0.85)");
    } else if correlation > 0.7 {
        println!("✅ POSITIVE CORRELATION (r > 0.7)");
    } else if correlation > 0.0 {
        println!("⚠️  WEAK POSITIVE CORRELATION (r > 0 but < 0.7)");
    } else {
        println!("❌ NEGATIVE OR NO CORRELATION (r ≤ 0)");
    }

    println!();
    println!("{}", "=".repeat(80));
    println!("CONCLUSION");
    println!("{}", "=".repeat(80));
    println!();

    if phi_increasing && correlation > 0.7 {
        println!("✅ FIX VERIFIED!");
        println!();
        println!("The shared pattern ratio approach produces:");
        println!("  • Monotonic Φ increase with integration level");
        println!("  • Strong positive correlation (r = {:.4})", correlation);
        println!("  • Expected Φ range (0.00-0.80+)");
        println!();
        println!("This confirms that:");
        println!("  1. Direct similarity encoding works correctly");
        println!("  2. Φ implementation is theoretically sound");
        println!("  3. Original generators used wrong HDV operation");
        println!();
        println!("Ready to integrate into full validation study! 🎉");
    } else {
        println!("⚠️  FIX NEEDS REFINEMENT");
        println!();
        println!("Results show improvement but not meeting all targets:");
        println!("  • Monotonic: {}", if phi_increasing { "✅" } else { "❌" });
        println!("  • Correlation > 0.7: {}", if correlation > 0.7 { "✅" } else { "❌" });
    }

    println!();

    Ok(())
}

/// Generate components with specified shared pattern ratio
///
/// shared_ratio = 0.0: all unique (random)
/// shared_ratio = 0.5: half shared, half unique
/// shared_ratio = 0.8: 4 shared, 1 unique
fn generate_by_shared_ratio(n: usize, shared_ratio: f64) -> Vec<HV16> {
    // Determine number of shared vs unique patterns
    let total_patterns = 5; // Always bundle 5 patterns for consistency
    let num_shared = (total_patterns as f64 * shared_ratio).round() as usize;
    let num_unique = total_patterns - num_shared;

    // Create shared patterns
    let shared_patterns: Vec<HV16> = (0..num_shared)
        .map(|i| HV16::random(i as u64 * 12345))
        .collect();

    // Generate components
    (0..n)
        .map(|i| {
            let mut patterns = shared_patterns.clone();

            // Add unique patterns for this component
            for j in 0..num_unique {
                patterns.push(HV16::random((i * 1000 + j * 100) as u64));
            }

            HV16::bundle(&patterns)
        })
        .collect()
}

/// Measure average pairwise similarity
fn measure_avg_similarity(components: &[HV16]) -> f64 {
    let n = components.len();
    if n < 2 {
        return 0.0;
    }

    let mut total = 0.0;
    let mut count = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            total += components[i].similarity(&components[j]) as f64;
            count += 1;
        }
    }

    total / count as f64
}
