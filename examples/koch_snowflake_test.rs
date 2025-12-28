/// Koch Snowflake Fractal Topology Test
///
/// Tests the Koch Snowflake implementation with different depths
/// and measures integrated information (Φ) to validate the fractal
/// consciousness hypothesis.

use symthaea::hdc::{
    consciousness_topology_generators::ConsciousnessTopology,
    phi_real::RealPhiCalculator,
    HDC_DIMENSION,
};

fn main() {
    println!("\n🌨️  KOCH SNOWFLAKE FRACTAL TOPOLOGY TEST");
    println!("{}", "=".repeat(60));
    println!("Testing: Koch Snowflake at depths 0-4");
    println!("HDC Dimension: {} (2^14)", HDC_DIMENSION);
    println!("Fractal dimension: log(4)/log(3) ≈ 1.262");
    println!("{}", "=".repeat(60));

    let calc = RealPhiCalculator::new();
    let n_samples = 5;

    // Test depths 0 through 4
    for depth in 0..=4 {
        let n_nodes = 3 * 4_usize.pow(depth as u32);

        println!("\n📊 Depth {} ({} nodes)...", depth, n_nodes);

        let mut phi_values = Vec::new();

        for sample in 0..n_samples {
            let seed = 42 + sample as u64 * 1000;

            // Generate Koch Snowflake
            let topology = ConsciousnessTopology::koch_snowflake(depth, HDC_DIMENSION, seed);

            // Compute Φ
            let phi = calc.compute(&topology.node_representations);
            phi_values.push(phi);

            print!(".");
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
        println!();

        // Statistics
        let mean = phi_values.iter().sum::<f64>() / n_samples as f64;
        let variance = phi_values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / n_samples as f64;
        let std_dev = variance.sqrt();

        println!("   Mean Φ: {:.6} ± {:.6}", mean, std_dev);
        println!("   Range:  [{:.6}, {:.6}]",
                 phi_values.iter().cloned().fold(f64::INFINITY, f64::min),
                 phi_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    }

    println!("\n{}", "=".repeat(60));
    println!("🧠 THEORETICAL IMPLICATIONS");
    println!("{}", "=".repeat(60));
    println!();
    println!("Koch Snowflake Properties:");
    println!("  • Infinite perimeter in finite area");
    println!("  • Fractal dimension ≈ 1.262 (between 1D curve and 2D area)");
    println!("  • Self-similar at all scales");
    println!();
    println!("Consciousness Hypothesis:");
    println!("  • Fractal structure may create unique integration patterns");
    println!("  • Self-similarity could enable hierarchical information flow");
    println!("  • Lower dimension than Sierpinski (1.585) may yield different Φ");

    println!("\n{}", "=".repeat(60));
    println!("✅ Koch Snowflake Test Complete!");
    println!("{}", "=".repeat(60));
}
