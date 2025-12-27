//! Quick test to verify Φ calculation fix
//!
//! This creates two scenarios with different integration levels
//! and verifies that Φ values are meaningfully different.

use symthaea::hdc::binary_hv::HV16;
use symthaea::hdc::tiered_phi::{TieredPhi, ApproximationTier};

fn main() {
    println!("Testing Φ Calculation Fix");
    println!("=" .repeat(60));
    println!();

    // Test 1: Low integration (random uncorrelated components)
    println!("Test 1: Low Integration (Random Components)");
    let mut phi_calc = TieredPhi::new(ApproximationTier::Heuristic);
    let low_integration: Vec<HV16> = (0..10)
        .map(|i| HV16::random((i * 1000) as u64))
        .collect();

    let phi_low = phi_calc.compute(&low_integration);
    println!("  Φ (low integration): {:.4}", phi_low);
    println!();

    // Test 2: High integration (similar components with strong correlations)
    println!("Test 2: High Integration (Similar Components)");
    let base = HV16::random(42);
    let high_integration: Vec<HV16> = (0..10)
        .map(|i| {
            let mut variant = base.clone();
            // Flip just a few bits to create highly similar variants
            variant.0[i % 256] ^= 0x01;
            variant
        })
        .collect();

    let phi_high = phi_calc.compute(&high_integration);
    println!("  Φ (high integration): {:.4}", phi_high);
    println!();

    // Test 3: Medium integration
    println!("Test 3: Medium Integration (Moderately Similar)");
    let medium_integration: Vec<HV16> = (0..10)
        .map(|i| {
            let mut variant = base.clone();
            // Flip more bits for medium similarity
            for j in 0..10 {
                variant.0[(i * 10 + j) % 256] ^= 0xFF;
            }
            variant
        })
        .collect();

    let phi_medium = phi_calc.compute(&medium_integration);
    println!("  Φ (medium integration): {:.4}", phi_medium);
    println!();

    // Verification
    println!("Verification:");
    println!("=" .repeat(60));

    let all_different = phi_low != phi_medium && phi_medium != phi_high && phi_low != phi_high;
    println!("✓ All Φ values are different: {}", all_different);

    let high_is_highest = phi_high > phi_medium && phi_high > phi_low;
    println!("✓ High integration has highest Φ: {} ({:.4} > {:.4} and {:.4})",
        high_is_highest, phi_high, phi_medium, phi_low);

    let values_in_range = phi_low >= 0.0 && phi_low <= 1.0
        && phi_medium >= 0.0 && phi_medium <= 1.0
        && phi_high >= 0.0 && phi_high <= 1.0;
    println!("✓ All values in [0,1]: {}", values_in_range);

    let not_all_zero_point_zero_eight =
        (phi_low - 0.08).abs() > 0.02
        || (phi_medium - 0.08).abs() > 0.02
        || (phi_high - 0.08).abs() > 0.02;
    println!("✓ NOT all converging to ~0.08: {}", not_all_zero_point_zero_eight);
    println!();

    if all_different && high_is_highest && values_in_range && not_all_zero_point_zero_eight {
        println!("🎉 FIX SUCCESSFUL! Φ calculation now produces meaningful values");
    } else {
        println!("⚠️  FIX INCOMPLETE - still has issues");
        println!("   Values: low={:.4}, medium={:.4}, high={:.4}", phi_low, phi_medium, phi_high);
    }
}
