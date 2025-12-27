// RealHV Φ vs Binary Binarization Comparison
//
// Tests the hypothesis using:
// 1. RealHV Φ (continuous, no binarization) - ULTIMATE TEST
// 2. Probabilistic binarization (best binary method from previous tests)
// 3. Mean threshold (original method for baseline)
//
// Hypothesis: Star topology has significantly higher Φ than Random topology

use symthaea::hdc::{
    consciousness_topology_generators::ConsciousnessTopology,
    tiered_phi::{TieredPhi, ApproximationTier},
    phi_topology_validation::{
        real_hv_to_hv16,
        real_hv_to_hv16_probabilistic,
    },
    phi_real::RealPhiCalculator,
    binary_hv::HV16,
};

fn main() {
    println!("\n🔬 REAL PHI COMPARISON - ULTIMATE TEST");
    println!("============================================================");
    println!("Testing hypothesis without binarization artifacts\n");

    // Parameters
    let n_samples = 10;
    let n_nodes = 8;
    let dim = symthaea::hdc::HDC_DIMENSION;  // 16,384 (2^14)
    let seed_base = 42;

    // Create calculators
    let mut binary_phi_calculator = TieredPhi::new(ApproximationTier::Spectral);
    let real_phi_calculator = RealPhiCalculator::new();

    println!("📊 Testing {} samples per topology\n", n_samples);

    // =====================================================================
    // METHOD 1: RealHV Φ (Continuous - No Binarization) ⭐ ULTIMATE
    // =====================================================================
    println!("🔹 Method 1: RealHV Φ (Continuous - No Binarization)");
    println!("{}", "─".repeat(60));

    let mut phi_random_real = Vec::new();
    let mut phi_star_real = Vec::new();

    for i in 0..n_samples {
        let seed = seed_base + (i as u64 * 1000);

        // Random topology
        let topology_random = ConsciousnessTopology::random(n_nodes, dim, seed);
        let phi_r = real_phi_calculator.compute(&topology_random.node_representations);
        phi_random_real.push(phi_r);

        // Star topology
        let topology_star = ConsciousnessTopology::star(n_nodes, dim, seed);
        let phi_s = real_phi_calculator.compute(&topology_star.node_representations);
        phi_star_real.push(phi_s);
    }

    let mean_random_real = mean(&phi_random_real);
    let std_random_real = std_dev(&phi_random_real, mean_random_real);
    let mean_star_real = mean(&phi_star_real);
    let std_star_real = std_dev(&phi_star_real, mean_star_real);
    let diff_real = mean_star_real - mean_random_real;
    let diff_percent_real = (diff_real / mean_random_real) * 100.0;

    println!("  Random: Φ = {:.4} ± {:.4}", mean_random_real, std_random_real);
    println!("  Star:   Φ = {:.4} ± {:.4}", mean_star_real, std_star_real);
    println!("  Δ(Star-Random): {:.4} ({:+.2}%)", diff_real, diff_percent_real);

    if diff_real > 0.0 {
        println!("  ✅ Star > Random (hypothesis supported!)\n");
    } else {
        println!("  ❌ Random > Star (hypothesis not supported)\n");
    }

    // =====================================================================
    // METHOD 2: Probabilistic Binarization (Best Binary Method)
    // =====================================================================
    println!("🔹 Method 2: Probabilistic Binarization (Best Binary)");
    println!("{}", "─".repeat(60));

    let mut phi_random_prob = Vec::new();
    let mut phi_star_prob = Vec::new();

    for i in 0..n_samples {
        let seed = seed_base + (i as u64 * 1000);

        // Random topology
        let topology_random = ConsciousnessTopology::random(n_nodes, dim, seed);
        let components_r: Vec<HV16> = topology_random.node_representations
            .iter()
            .map(|hv| real_hv_to_hv16_probabilistic(hv, seed))
            .collect();
        let phi_r = binary_phi_calculator.compute(&components_r);
        phi_random_prob.push(phi_r);

        // Star topology
        let topology_star = ConsciousnessTopology::star(n_nodes, dim, seed);
        let components_s: Vec<HV16> = topology_star.node_representations
            .iter()
            .map(|hv| real_hv_to_hv16_probabilistic(hv, seed))
            .collect();
        let phi_s = binary_phi_calculator.compute(&components_s);
        phi_star_prob.push(phi_s);
    }

    let mean_random_prob = mean(&phi_random_prob);
    let std_random_prob = std_dev(&phi_random_prob, mean_random_prob);
    let mean_star_prob = mean(&phi_star_prob);
    let std_star_prob = std_dev(&phi_star_prob, mean_star_prob);
    let diff_prob = mean_star_prob - mean_random_prob;
    let diff_percent_prob = (diff_prob / mean_random_prob) * 100.0;

    println!("  Random: Φ = {:.4} ± {:.4}", mean_random_prob, std_random_prob);
    println!("  Star:   Φ = {:.4} ± {:.4}", mean_star_prob, std_star_prob);
    println!("  Δ(Star-Random): {:.4} ({:+.2}%)", diff_prob, diff_percent_prob);

    if diff_prob > 0.0 {
        println!("  ✅ Star > Random (hypothesis supported!)\n");
    } else {
        println!("  ❌ Random > Star (hypothesis not supported)\n");
    }

    // =====================================================================
    // METHOD 3: Mean Threshold (Baseline Comparison)
    // =====================================================================
    println!("🔹 Method 3: Mean Threshold (Baseline)");
    println!("{}", "─".repeat(60));

    let mut phi_random_mean = Vec::new();
    let mut phi_star_mean = Vec::new();

    for i in 0..n_samples {
        let seed = seed_base + (i as u64 * 1000);

        // Random topology
        let topology_random = ConsciousnessTopology::random(n_nodes, dim, seed);
        let components_r: Vec<HV16> = topology_random.node_representations
            .iter()
            .map(|hv| real_hv_to_hv16(hv))
            .collect();
        let phi_r = binary_phi_calculator.compute(&components_r);
        phi_random_mean.push(phi_r);

        // Star topology
        let topology_star = ConsciousnessTopology::star(n_nodes, dim, seed);
        let components_s: Vec<HV16> = topology_star.node_representations
            .iter()
            .map(|hv| real_hv_to_hv16(hv))
            .collect();
        let phi_s = binary_phi_calculator.compute(&components_s);
        phi_star_mean.push(phi_s);
    }

    let mean_random_mean = mean(&phi_random_mean);
    let std_random_mean = std_dev(&phi_random_mean, mean_random_mean);
    let mean_star_mean = mean(&phi_star_mean);
    let std_star_mean = std_dev(&phi_star_mean, mean_star_mean);
    let diff_mean = mean_star_mean - mean_random_mean;
    let diff_percent_mean = (diff_mean / mean_random_mean) * 100.0;

    println!("  Random: Φ = {:.4} ± {:.4}", mean_random_mean, std_random_mean);
    println!("  Star:   Φ = {:.4} ± {:.4}", mean_star_mean, std_star_mean);
    println!("  Δ(Star-Random): {:.4} ({:+.2}%)", diff_mean, diff_percent_mean);

    if diff_mean > 0.0 {
        println!("  ✅ Star > Random (hypothesis supported!)\n");
    } else {
        println!("  ❌ Random > Star (hypothesis not supported)\n");
    }

    // =====================================================================
    // FINAL COMPARISON
    // =====================================================================
    println!("============================================================");
    println!("🎯 FINAL COMPARISON");
    println!("============================================================\n");

    println!("| Method | Random Φ | Star Φ | Δ (%) | Hypothesis |");
    println!("|--------|----------|--------|-------|------------|");

    println!("| Mean Threshold | {:.4} ± {:.4} | {:.4} ± {:.4} | {:+.2}% | {} |",
        mean_random_mean, std_random_mean,
        mean_star_mean, std_star_mean,
        diff_percent_mean,
        if diff_mean > 0.0 { "✅ Supported" } else { "❌ Not supported" }
    );

    println!("| Probabilistic | {:.4} ± {:.4} | {:.4} ± {:.4} | {:+.2}% | {} |",
        mean_random_prob, std_random_prob,
        mean_star_prob, std_star_prob,
        diff_percent_prob,
        if diff_prob > 0.0 { "✅ Supported" } else { "❌ Not supported" }
    );

    println!("| **RealHV (No Binary)** | **{:.4} ± {:.4}** | **{:.4} ± {:.4}** | **{:+.2}%** | {} |",
        mean_random_real, std_random_real,
        mean_star_real, std_star_real,
        diff_percent_real,
        if diff_real > 0.0 { "✅ **Supported**" } else { "❌ Not supported" }
    );

    println!("\n📈 KEY INSIGHTS:\n");

    // Compare RealHV to Probabilistic
    if diff_percent_real > diff_percent_prob {
        let improvement = diff_percent_real - diff_percent_prob;
        println!("  • RealHV Φ shows STRONGER effect than probabilistic (+{:.2}% vs +{:.2}%)",
            diff_percent_real, diff_percent_prob);
        println!("    Δ improvement: {:.2} percentage points", improvement);
        println!("    ⭐ Continuous calculation preserves MORE heterogeneity information!");
    } else if diff_percent_real > 0.0 && diff_percent_prob > 0.0 {
        println!("  • Both RealHV and Probabilistic support hypothesis");
        println!("    RealHV: {:+.2}%, Probabilistic: {:+.2}%", diff_percent_real, diff_percent_prob);
    }

    // Compare to baseline
    if diff_percent_real > 0.0 && diff_percent_mean < 0.0 {
        println!("\n  • Mean threshold REVERSES the effect (Δ = {:.2}%)", diff_percent_mean);
        println!("    Demonstrates binarization artifacts compress heterogeneity");
    }

    println!("\n🏆 CONCLUSION:");
    if diff_percent_real > 0.0 {
        println!("  ✅ HYPOTHESIS CONFIRMED with continuous RealHV Φ!");
        println!("     Star topology has {:.2}% higher Φ than Random", diff_percent_real);
        println!("     This validates that heterogeneity → higher integrated information");
    } else {
        println!("  ⚠️  Unexpected result - needs investigation");
    }

    println!("\n============================================================");
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn std_dev(values: &[f64], mean: f64) -> f64 {
    let variance = values.iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    variance.sqrt()
}
