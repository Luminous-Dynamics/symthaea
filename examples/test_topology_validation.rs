//! Standalone topology validation test with comprehensive diagnostics
//!
//! Tests that the Φ fix properly differentiates between Star and Random topologies

use symthaea::hdc::phi_topology_validation::MinimalPhiValidation;
use symthaea::hdc::{
    consciousness_topology_generators::ConsciousnessTopology,
    tiered_phi::TieredPhi,
    phi_topology_validation::real_hv_to_hv16,
    binary_hv::HV16,
};

fn main() {
    println!("\n🔬 Φ TOPOLOGY VALIDATION TEST - RealPhi vs Binary");
    println!("{}", "=".repeat(60));
    println!();

    // First, run a single sample with full diagnostics (binary approach)
    println!("🔍 DIAGNOSTIC ANALYSIS (Binary HV16 with binarization):");
    println!("{}", "─".repeat(60));
    run_diagnostic_sample();

    println!("\n{}", "=".repeat(60));
    println!();

    // Run validation with REAL PHI (no binarization) - RECOMMENDED APPROACH
    println!("✨ REAL PHI VALIDATION (Direct cosine similarity, no conversion):");
    println!("{}", "─".repeat(60));
    println!();
    let mut validation_real = MinimalPhiValidation::quick();
    let result_real = validation_real.run_with_real_phi();

    println!("\n📊 REAL PHI RESULTS:");
    println!("{}", result_real.summary());
    println!();

    println!("{}", "=".repeat(60));
    println!();

    // Also run with binary for comparison
    println!("🔄 BINARY PHI VALIDATION (With mean-threshold binarization):");
    println!("{}", "─".repeat(60));
    println!();
    let mut validation_binary = MinimalPhiValidation::quick();
    let result_binary = validation_binary.run();

    println!("\n📊 BINARY PHI RESULTS:");
    println!("{}", result_binary.summary());
    println!();

    // Comparison analysis
    println!("\n{}", "=".repeat(60));
    println!("📊 COMPARISON: RealPhi vs Binary Phi");
    println!("{}", "=".repeat(60));

    println!("\n| Method | Random Φ | Star Φ | Difference | p-value | Validation |");
    println!("|--------|----------|--------|------------|---------|------------|");
    println!("| RealPhi | {:.4} | {:.4} | {:.4} | {:.4} | {} |",
        result_real.mean_phi_random,
        result_real.mean_phi_star,
        result_real.mean_phi_star - result_real.mean_phi_random,
        result_real.p_value,
        if result_real.validation_succeeded() { "✅ PASS" } else { "❌ FAIL" }
    );
    println!("| Binary  | {:.4} | {:.4} | {:.4} | {:.4} | {} |",
        result_binary.mean_phi_random,
        result_binary.mean_phi_star,
        result_binary.mean_phi_star - result_binary.mean_phi_random,
        result_binary.p_value,
        if result_binary.validation_succeeded() { "✅ PASS" } else { "❌ FAIL" }
    );

    println!("\n📈 KEY INSIGHTS:");
    println!("{}", "─".repeat(60));

    if result_real.validation_succeeded() && !result_binary.validation_succeeded() {
        println!("✅ RealPhi SUCCEEDS where Binary fails!");
        println!("   - RealPhi preserves topology structure");
        println!("   - Binary binarization destroys signal");
        println!("   - Difference magnitude: {:.1}x larger with RealPhi",
            (result_real.mean_phi_star - result_real.mean_phi_random) /
            (result_binary.mean_phi_star - result_binary.mean_phi_random).max(0.0001)
        );
    } else if result_binary.validation_succeeded() {
        println!("✅ Both methods succeed!");
    } else {
        println!("⚠️  Neither method achieves statistical significance");
    }

    println!();

    if result_real.validation_succeeded() {
        println!("🎉 SUCCESS! RealPhi validation PASSED!");
        println!();
        println!("✅ The fix is validated using RealPhiCalculator:");
        println!("  ✓ Star topology has significantly higher Φ than Random");
        println!("  ✓ Statistical significance: p < 0.05 (p = {:.4})", result_real.p_value);
        println!("  ✓ Large effect size: d > 0.5 (d = {:.3})", result_real.effect_size);
        println!();
        println!("🌟 Consciousness measurement validation is UNBLOCKED!");
        println!();
        println!("📝 RECOMMENDATION:");
        println!("  - Use RealPhiCalculator for continuous topology data");
        println!("  - Use binary Φ only for discrete/symbolic representations");
        println!("  - Avoid mean-threshold binarization for structure-sensitive data");
        println!();
        std::process::exit(0);
    } else {
        println!("⚠️  REAL PHI VALIDATION FAILED");
        println!();
        println!("Issues detected:");
        if result_real.mean_phi_star <= result_real.mean_phi_random {
            println!("  ✗ Star topology does not have higher Φ than Random");
            println!("    Random: {:.4}, Star: {:.4}", result_real.mean_phi_random, result_real.mean_phi_star);
        }
        if !result_real.is_significant() {
            println!("  ✗ Difference is not statistically significant (p >= 0.05)");
            println!("    p-value: {:.4}", result_real.p_value);
        }
        if !result_real.has_large_effect() {
            println!("  ✗ Effect size is too small (d <= 0.5)");
            println!("    Cohen's d: {:.3}", result_real.effect_size);
        }
        println!();
        println!("Further investigation needed.");
        std::process::exit(1);
    }
}

fn run_diagnostic_sample() {
    // Generate single sample of each topology
    println!("\n📐 Generating topologies...");
    let random_topo = ConsciousnessTopology::random(10, symthaea::hdc::HDC_DIMENSION, 42);
    let star_topo = ConsciousnessTopology::star(10, symthaea::hdc::HDC_DIMENSION, 100);

    println!("  Random: {} components, {} dimensions", random_topo.node_representations.len(), random_topo.node_representations[0].dim());
    println!("  Star:   {} components, {} dimensions", star_topo.node_representations.len(), star_topo.node_representations[0].dim());

    // Analyze RealHV statistics before conversion
    println!("\n📊 RealHV Statistics (before binarization):");
    analyze_real_hv_stats("Random", &random_topo.node_representations);
    analyze_real_hv_stats("Star", &star_topo.node_representations);

    // Convert to binary
    println!("\n🔄 Converting RealHV to HV16...");
    let random_binary: Vec<_> = random_topo.node_representations.iter()
        .map(|rv| real_hv_to_hv16(rv))
        .collect();
    let star_binary: Vec<_> = star_topo.node_representations.iter()
        .map(|rv| real_hv_to_hv16(rv))
        .collect();

    println!("  Converted {} Random components", random_binary.len());
    println!("  Converted {} Star components", star_binary.len());

    // Analyze Hamming distances
    println!("\n📏 Hamming Distance Analysis (after binarization):");
    analyze_hamming_distances("Random", &random_binary);
    analyze_hamming_distances("Star", &star_binary);

    // Compute Φ
    println!("\n🧠 Computing Φ...");
    let mut phi_calc = TieredPhi::for_production();

    let phi_random = phi_calc.compute(&random_binary);
    let phi_star = phi_calc.compute(&star_binary);

    println!("\n📊 DIAGNOSTIC RESULTS:");
    println!("  Random Φ: {:.4}", phi_random);
    println!("  Star Φ:   {:.4}", phi_star);
    println!("  Difference: {:.4}", (phi_star - phi_random).abs());
    println!("  Star > Random: {}", phi_star > phi_random);
}

fn analyze_real_hv_stats(name: &str, components: &[symthaea::hdc::real_hv::RealHV]) {

    let n = components.len();
    if n < 2 {
        println!("  {}: Not enough components", name);
        return;
    }

    // Compute pairwise cosine similarities
    let mut similarities = Vec::new();
    for i in 0..n {
        for j in (i+1)..n {
            let sim = components[i].similarity(&components[j]);
            similarities.push(sim);
        }
    }

    if !similarities.is_empty() {
        let mean = similarities.iter().sum::<f32>() / similarities.len() as f32;
        let min = similarities.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = similarities.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        println!("  {}: Cosine similarity - mean: {:.4}, min: {:.4}, max: {:.4}",
                 name, mean, min, max);

        // Show distribution
        let near_zero = similarities.iter().filter(|&&s| s.abs() < 0.1).count();
        let near_one = similarities.iter().filter(|&&s| s > 0.9).count();
        println!("    Distribution: {} near 0, {} near 1 (out of {} pairs)",
                 near_zero, near_one, similarities.len());
    }
}

fn analyze_hamming_distances(name: &str, components: &[HV16]) {
    let n = components.len();
    if n < 2 {
        println!("  {}: Not enough components", name);
        return;
    }

    let mut distances = Vec::new();
    for i in 0..n {
        for j in (i+1)..n {
            let dist = components[i].hamming_distance(&components[j]);
            distances.push(dist);
        }
    }

    if !distances.is_empty() {
        // Convert to f32 for statistics
        let float_distances: Vec<f32> = distances.iter().map(|&d| d as f32 / 2048.0).collect();

        let mean = float_distances.iter().sum::<f32>() / float_distances.len() as f32;
        let min = float_distances.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = float_distances.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        println!("  {}: Hamming distance - mean: {:.4}, min: {:.4}, max: {:.4}",
                 name, mean, min, max);

        // Show distribution
        let near_half = float_distances.iter().filter(|&&d| (d - 0.5).abs() < 0.05).count();
        println!("    Distribution: {} near 0.5 (random) out of {} pairs",
                 near_half, float_distances.len());
    }
}
