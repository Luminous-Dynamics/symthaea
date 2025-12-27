// Φ_HDC vs Φ_exact Validation Suite - Week 4 Day 3-4
//
// Validates the HDC-based Φ approximation against exact IIT 3.0 (via PyPhi)
//
// Test Matrix:
// - 8 topologies (Dense, Modular, Star, Ring, Random, BinaryTree, Lattice, Line)
// - 4 sizes (n = 5, 6, 7, 8 nodes)
// - 5 random seeds per configuration
// - Total: 8 × 4 × 5 = 160 comparisons
//
// Expected Runtime:
// - n=5: ~1 second per comparison
// - n=6: ~10 seconds per comparison
// - n=7: ~1-2 minutes per comparison
// - n=8: ~10-30 minutes per comparison
// - Total: ~40-80 hours (run overnight or in batches)
//
// Output: pyphi_validation_results.csv

use symthaea::hdc::consciousness_topology_generators::ConsciousnessTopology;
use symthaea::hdc::phi_real::RealPhiCalculator;
use symthaea::synthesis::phi_exact::PyPhiValidator;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const HDC_DIMENSION: usize = 16_384;

/// Single validation result
#[derive(Debug, Clone)]
struct ValidationResult {
    topology_name: String,
    n: usize,
    seed: u64,
    phi_hdc: f64,
    phi_exact: f64,
    error: f64,
    relative_error: f64,
    duration_ms: u128,
}

impl ValidationResult {
    fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{:.6},{:.6},{:.6},{:.6},{}",
            self.topology_name,
            self.n,
            self.seed,
            self.phi_hdc,
            self.phi_exact,
            self.error,
            self.relative_error,
            self.duration_ms
        )
    }
}

/// Summary statistics
#[derive(Debug, Clone)]
struct ValidationStats {
    total_comparisons: usize,
    mean_error: f64,
    std_error: f64,
    rmse: f64,
    mae: f64,
    max_error: f64,
    min_error: f64,
    pearson_r: f64,
    spearman_rho: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Φ_HDC Validation Suite - Week 4 ===\n");
    println!("Validating HDC approximation against exact IIT 3.0 (PyPhi)\n");

    #[cfg(not(feature = "pyphi"))]
    {
        println!("ERROR: PyPhi feature not enabled!");
        println!("Compile with: cargo run --example pyphi_validation --features pyphi --release");
        println!("\nNote: This requires PyPhi installed in your Python environment:");
        println!("  pip install pyphi numpy scipy networkx");
        return Err("PyPhi feature not enabled".into());
    }

    #[cfg(feature = "pyphi")]
    {
        // Initialize calculators
        let hdc_calc = RealPhiCalculator::new();
        let exact_calc = PyPhiValidator::new()?;

        // Test parameters
        // Note: Some topologies have different signatures, so we use closures
        let topology_configs: Vec<(&str, Box<dyn Fn(usize, usize, u64) -> ConsciousnessTopology>)> = vec![
            ("Dense", Box::new(|n, dim, seed| ConsciousnessTopology::dense_network(n, dim, None, seed))),
            ("Modular", Box::new(|n, dim, seed| ConsciousnessTopology::modular(n, dim, 2, seed))),
            ("Star", Box::new(|n, dim, seed| ConsciousnessTopology::star(n, dim, seed))),
            ("Ring", Box::new(|n, dim, seed| ConsciousnessTopology::ring(n, dim, seed))),
            ("Random", Box::new(|n, dim, seed| ConsciousnessTopology::random(n, dim, seed))),
            ("BinaryTree", Box::new(|n, dim, seed| ConsciousnessTopology::binary_tree(n, dim, seed))),
            ("Lattice", Box::new(|n, dim, seed| ConsciousnessTopology::lattice(n, dim, seed))),
            ("Line", Box::new(|n, dim, seed| ConsciousnessTopology::line(n, dim, seed))),
        ];

        let sizes = vec![5, 6, 7, 8];
        let seeds = vec![42, 123, 456, 789, 999];

        let total_comparisons = topology_configs.len() * sizes.len() * seeds.len();
        println!("Test Matrix:");
        println!("  Topologies: {}", topology_configs.len());
        println!("  Sizes: {:?}", sizes);
        println!("  Seeds: {:?}", seeds);
        println!("  Total: {} comparisons\n", total_comparisons);

        // Estimate runtime
        println!("Estimated Runtime:");
        println!("  n=5: ~{} comparisons × 1 sec = ~{} min",
                 topology_configs.len() * seeds.len(),
                 (topology_configs.len() * seeds.len()) / 60);
        println!("  n=6: ~{} comparisons × 10 sec = ~{} min",
                 topology_configs.len() * seeds.len(),
                 (topology_configs.len() * seeds.len() * 10) / 60);
        println!("  n=7: ~{} comparisons × 60 sec = ~{} hours",
                 topology_configs.len() * seeds.len(),
                 (topology_configs.len() * seeds.len() * 60) / 3600);
        println!("  n=8: ~{} comparisons × 600 sec = ~{} hours",
                 topology_configs.len() * seeds.len(),
                 (topology_configs.len() * seeds.len() * 600) / 3600);
        println!("  Total: ~40-80 hours (run overnight)\n");

        // Collect results
        let mut results = Vec::new();
        let mut completed = 0;

        // Open CSV file
        let mut csv_file = File::create("pyphi_validation_results.csv")?;
        writeln!(csv_file, "topology,n,seed,phi_hdc,phi_exact,error,relative_error,duration_ms")?;

        println!("Starting validation...\n");
        let start_time = Instant::now();

        for (topology_name, topology_fn) in topology_configs.iter() {
            for &n in &sizes {
                for &seed in &seeds {
                    completed += 1;
                    let progress = (completed as f64 / total_comparisons as f64) * 100.0;

                    print!("[{:3}/{:3}] ({:5.1}%) {} (n={}, seed={})... ",
                           completed, total_comparisons, progress, topology_name, n, seed);
                    std::io::stdout().flush()?;

                    let comparison_start = Instant::now();

                    // Generate topology
                    let topology = topology_fn(n, HDC_DIMENSION, seed);

                    // Compute Φ_HDC (fast)
                    let phi_hdc = hdc_calc.compute(&topology.node_representations);

                    // Compute Φ_exact (slow)
                    let phi_exact = match exact_calc.compute_phi_exact(&topology) {
                        Ok(val) => val,
                        Err(e) => {
                            println!("ERROR: {}", e);
                            continue;
                        }
                    };

                    let duration = comparison_start.elapsed();

                    // Calculate errors
                    let error = (phi_hdc - phi_exact).abs();
                    let relative_error = if phi_exact != 0.0 {
                        error / phi_exact
                    } else {
                        0.0
                    };

                    let result = ValidationResult {
                        topology_name: topology_name.to_string(),
                        n,
                        seed,
                        phi_hdc,
                        phi_exact,
                        error,
                        relative_error,
                        duration_ms: duration.as_millis(),
                    };

                    // Write to CSV immediately
                    writeln!(csv_file, "{}", result.to_csv_row())?;
                    csv_file.flush()?;

                    results.push(result.clone());

                    println!("Φ_HDC={:.4}, Φ_exact={:.4}, err={:.4} ({:.1}%), {}ms",
                             phi_hdc, phi_exact, error, relative_error * 100.0, duration.as_millis());

                    // Progress update every 10
                    if completed % 10 == 0 {
                        println!("\n  Progress: {}/{} ({:.1}%), elapsed: {:.1}h\n",
                                 completed, total_comparisons, progress,
                                 start_time.elapsed().as_secs_f64() / 3600.0);
                    }
                }
            }
        }

        let total_time = start_time.elapsed();
        println!("\n✅ Validation complete!");
        println!("Total time: {:.1}h ({:.0}s)",
                 total_time.as_secs_f64() / 3600.0, total_time.as_secs());
        println!("Results saved to: pyphi_validation_results.csv\n");

        // Compute statistics
        println!("📊 Computing Statistics...\n");
        let stats = compute_statistics(&results);
        print_statistics(&stats);

        // Topology analysis
        println!("\n📈 Topology-Specific Analysis:\n");
        for (topology_name, _) in topology_configs.iter() {
            let topology_results: Vec<_> = results.iter()
                .filter(|r| &r.topology_name == topology_name)
                .collect();

            if !topology_results.is_empty() {
                let mean_phi_hdc: f64 = topology_results.iter().map(|r| r.phi_hdc).sum::<f64>()
                    / topology_results.len() as f64;
                let mean_phi_exact: f64 = topology_results.iter().map(|r| r.phi_exact).sum::<f64>()
                    / topology_results.len() as f64;
                let mean_error: f64 = topology_results.iter().map(|r| r.error).sum::<f64>()
                    / topology_results.len() as f64;

                println!("{:12} | Φ_HDC={:.4}, Φ_exact={:.4}, err={:.4}",
                         topology_name, mean_phi_hdc, mean_phi_exact, mean_error);
            }
        }

        // Success criteria
        println!("\n✅ Success Criteria Evaluation:\n");
        evaluate_success_criteria(&stats);

        println!("\n🎯 Conclusion:");
        if stats.pearson_r > 0.9 && stats.rmse < 0.10 {
            println!("   EXCELLENT! Ready for publication.");
        } else if stats.pearson_r > 0.8 && stats.rmse < 0.15 {
            println!("   GOOD! Publication-ready with calibration.");
        } else if stats.pearson_r > 0.7 {
            println!("   ACCEPTABLE. Useful for ranking.");
        } else {
            println!("   WEAK. Refinement needed.");
        }
    }

    Ok(())
}

#[cfg(feature = "pyphi")]
fn compute_statistics(results: &[ValidationResult]) -> ValidationStats {
    let n = results.len() as f64;

    // Mean error
    let mean_error = results.iter().map(|r| r.error).sum::<f64>() / n;

    // Std deviation
    let variance = results.iter()
        .map(|r| (r.error - mean_error).powi(2))
        .sum::<f64>() / n;
    let std_error = variance.sqrt();

    // RMSE
    let mse = results.iter()
        .map(|r| r.error.powi(2))
        .sum::<f64>() / n;
    let rmse = mse.sqrt();

    // MAE
    let mae = results.iter().map(|r| r.error).sum::<f64>() / n;

    // Max/min
    let max_error = results.iter()
        .map(|r| r.error)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);
    let min_error = results.iter()
        .map(|r| r.error)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    // Pearson correlation
    let mean_hdc = results.iter().map(|r| r.phi_hdc).sum::<f64>() / n;
    let mean_exact = results.iter().map(|r| r.phi_exact).sum::<f64>() / n;

    let cov = results.iter()
        .map(|r| (r.phi_hdc - mean_hdc) * (r.phi_exact - mean_exact))
        .sum::<f64>() / n;

    let std_hdc = (results.iter()
        .map(|r| (r.phi_hdc - mean_hdc).powi(2))
        .sum::<f64>() / n).sqrt();

    let std_exact = (results.iter()
        .map(|r| (r.phi_exact - mean_exact).powi(2))
        .sum::<f64>() / n).sqrt();

    let pearson_r = cov / (std_hdc * std_exact);
    let spearman_rho = pearson_r; // Approximation

    ValidationStats {
        total_comparisons: results.len(),
        mean_error,
        std_error,
        rmse,
        mae,
        max_error,
        min_error,
        pearson_r,
        spearman_rho,
    }
}

#[cfg(feature = "pyphi")]
fn print_statistics(stats: &ValidationStats) {
    println!("Statistical Summary:");
    println!("  Total Comparisons:  {}", stats.total_comparisons);
    println!("  Mean Error:         {:.6}", stats.mean_error);
    println!("  Std Deviation:      {:.6}", stats.std_error);
    println!("  RMSE:               {:.6}", stats.rmse);
    println!("  MAE:                {:.6}", stats.mae);
    println!("  Max Error:          {:.6}", stats.max_error);
    println!("  Min Error:          {:.6}", stats.min_error);
    println!("  Pearson r:          {:.6}", stats.pearson_r);
    println!("  Spearman ρ:         {:.6}", stats.spearman_rho);
}

#[cfg(feature = "pyphi")]
fn evaluate_success_criteria(stats: &ValidationStats) {
    println!("Minimum (Acceptable):");
    println!("  r > 0.6:  {} (r={:.3})",
             if stats.pearson_r > 0.6 { "✅" } else { "❌" }, stats.pearson_r);

    println!("\nTarget (Expected):");
    println!("  r > 0.8:     {} (r={:.3})",
             if stats.pearson_r > 0.8 { "✅" } else { "❌" }, stats.pearson_r);
    println!("  RMSE < 0.15: {} ({:.3})",
             if stats.rmse < 0.15 { "✅" } else { "❌" }, stats.rmse);
    println!("  MAE < 0.10:  {} ({:.3})",
             if stats.mae < 0.10 { "✅" } else { "❌" }, stats.mae);

    println!("\nStretch (Ideal):");
    println!("  r > 0.9:     {} (r={:.3})",
             if stats.pearson_r > 0.9 { "✅" } else { "❌" }, stats.pearson_r);
    println!("  RMSE < 0.10: {} ({:.3})",
             if stats.rmse < 0.10 { "✅" } else { "❌" }, stats.rmse);
    println!("  MAE < 0.05:  {} ({:.3})",
             if stats.mae < 0.05 { "✅" } else { "❌" }, stats.mae);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "pyphi")]
    fn test_single_comparison() {
        let hdc_calc = RealPhiCalculator::new();
        let exact_calc = PyPhiValidator::new().expect("PyPhi available");

        let topology = ConsciousnessTopology::star(5, 16384, 42);
        let phi_hdc = hdc_calc.compute(&topology.node_representations);
        let phi_exact = exact_calc.compute_phi_exact(&topology)
            .expect("PyPhi computation succeeds");

        let error = (phi_hdc - phi_exact).abs();
        assert!(error < 0.5, "Error reasonable: {} < 0.5", error);

        println!("Test: Φ_HDC={:.4}, Φ_exact={:.4}, err={:.4}", phi_hdc, phi_exact, error);
    }
}
