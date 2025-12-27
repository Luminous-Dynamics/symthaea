// Binarization Method Comparison
//
// Compares four binarization methods:
// 1. Mean threshold (original)
// 2. Median threshold (robust to outliers)
// 3. Probabilistic (sigmoid-based)
// 4. Quantile (percentile-based)

use symthaea::hdc::{
    consciousness_topology_generators::ConsciousnessTopology,
    tiered_phi::{TieredPhi, ApproximationTier},
    phi_topology_validation::{
        real_hv_to_hv16,
        real_hv_to_hv16_median,
        real_hv_to_hv16_probabilistic,
        real_hv_to_hv16_quantile,
    },
};

fn main() {
    println!("\n🔬 BINARIZATION METHOD COMPARISON");
    println!("============================================================");
    println!("Comparing 4 binarization methods on Random vs Star topologies\n");

    // Parameters
    let n_samples = 10;
    let n_nodes = 8;
    let dim = symthaea::hdc::HDC_DIMENSION;  // 16,384 (2^14)
    let seed_base = 42;

    // Create Φ calculator
    let mut phi_calculator = TieredPhi::new(ApproximationTier::Spectral);

    // Test each binarization method
    let methods = vec![
        ("Mean Threshold", BinarizationMethod::Mean),
        ("Median Threshold", BinarizationMethod::Median),
        ("Probabilistic", BinarizationMethod::Probabilistic),
        ("Quantile (50th)", BinarizationMethod::Quantile),
    ];

    println!("📊 Testing {} samples per topology\n", n_samples);

    for (name, method) in methods {
        println!("🔹 Method: {}", name);
        println!("{}", "─".repeat(60));

        // Generate Random topologies
        let mut phi_random_values = Vec::new();
        for i in 0..n_samples {
            let seed = seed_base + (i as u64 * 1000);
            let topology = ConsciousnessTopology::random(n_nodes, dim, seed);
            let components = convert_topology(&topology, method.clone(), seed);
            let phi = phi_calculator.compute(&components);
            phi_random_values.push(phi);
        }

        // Generate Star topologies
        let mut phi_star_values = Vec::new();
        for i in 0..n_samples {
            let seed = seed_base + (i as u64 * 1000);
            let topology = ConsciousnessTopology::star(n_nodes, dim, seed);
            let components = convert_topology(&topology, method.clone(), seed);
            let phi = phi_calculator.compute(&components);
            phi_star_values.push(phi);
        }

        // Compute statistics
        let mean_random = mean(&phi_random_values);
        let std_random = std_dev(&phi_random_values, mean_random);
        let mean_star = mean(&phi_star_values);
        let std_star = std_dev(&phi_star_values, mean_star);

        let diff = mean_star - mean_random;
        let diff_percent = (diff / mean_random) * 100.0;

        println!("  Random: Φ = {:.4} ± {:.4}", mean_random, std_random);
        println!("  Star:   Φ = {:.4} ± {:.4}", mean_star, std_star);
        println!("  Δ(Star-Random): {:.4} ({:+.2}%)", diff, diff_percent);

        if diff > 0.0 {
            println!("  ✅ Star > Random (hypothesis supported!)");
        } else {
            println!("  ❌ Random > Star (hypothesis not supported)");
        }
        println!();
    }

    println!("============================================================");
    println!("🎯 SUMMARY");
    println!("============================================================");
    println!("Test if alternative binarization methods preserve Star's");
    println!("heterogeneity advantage and support the hypothesis:");
    println!("  H₁: Star topology has significantly higher Φ than Random");
    println!();
}

#[derive(Clone)]
enum BinarizationMethod {
    Mean,
    Median,
    Probabilistic,
    Quantile,
}

fn convert_topology(
    topology: &ConsciousnessTopology,
    method: BinarizationMethod,
    seed: u64,
) -> Vec<symthaea::hdc::binary_hv::HV16> {
    topology.node_representations
        .iter()
        .map(|real_hv| match method {
            BinarizationMethod::Mean => real_hv_to_hv16(real_hv),
            BinarizationMethod::Median => real_hv_to_hv16_median(real_hv),
            BinarizationMethod::Probabilistic => real_hv_to_hv16_probabilistic(real_hv, seed),
            BinarizationMethod::Quantile => real_hv_to_hv16_quantile(real_hv, 50.0),
        })
        .collect()
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
