//! Fractal Dimension ↔ Φ Relationship Study
//!
//! Systematic investigation of how fractal dimension correlates with
//! integrated information (Φ) across self-similar network topologies.
//!
//! # Hypothesis
//!
//! There exists a functional relationship Φ = f(d) where:
//! - d is the fractal (Hausdorff) dimension of the topology
//! - Φ is the integrated information of the system
//!
//! # Methodology
//!
//! Test topologies ordered by fractal dimension:
//! | Topology         | Fractal Dim | Description                    |
//! |------------------|-------------|--------------------------------|
//! | Cantor Set       | 0.631       | Disconnected dust              |
//! | Line/Chain       | 1.0         | 1D minimal connectivity        |
//! | Koch Snowflake   | 1.262       | Infinite perimeter, finite area|
//! | Sierpinski       | 1.585       | Self-similar triangles         |
//! | Torus            | 2.0         | Closed 2D surface              |
//! | Menger Sponge    | 2.727       | 3D fractal, infinite surface   |
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example fractal_dimension_phi_study
//! ```

use symthaea::hdc::{
    consciousness_topology_generators::ConsciousnessTopology,
    phi_real::RealPhiCalculator,
};

/// Fractal topology with its theoretical dimension
struct FractalTopology {
    name: &'static str,
    dimension: f64,
    topology: ConsciousnessTopology,
}

/// Results for a single fractal topology
#[derive(Debug)]
struct FractalResult {
    name: String,
    fractal_dimension: f64,
    phi: f64,
    node_count: usize,
    edge_count: usize,
    density: f64,
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     FRACTAL DIMENSION ↔ Φ RELATIONSHIP STUDY                   ║");
    println!("║     Testing Φ = f(d) hypothesis across fractal topologies      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Configuration
    let dim = 2048;  // HDC dimension (smaller for faster computation)
    let seed = 42u64;

    println!("Configuration:");
    println!("  HDC Dimension: {}", dim);
    println!("  Random seed: {}", seed);
    println!();

    // Build fractal topologies ordered by dimension
    let topologies = build_fractal_topologies(dim, seed);

    // Compute Φ for each topology
    let calculator = RealPhiCalculator::new();
    let mut results: Vec<FractalResult> = Vec::new();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("COMPUTING Φ FOR EACH FRACTAL TOPOLOGY");
    println!("═══════════════════════════════════════════════════════════════════\n");

    for fractal in topologies {
        print!("Testing {}... ", fractal.name);

        let node_count = fractal.topology.n_nodes;
        let edge_count = fractal.topology.edges.len();
        let density = if node_count > 1 {
            2.0 * edge_count as f64 / (node_count * (node_count - 1)) as f64
        } else {
            0.0
        };

        // Get node representations
        let nodes = &fractal.topology.node_representations;

        // Compute Φ
        let phi = calculator.compute(nodes);

        println!("done.");
        println!("  d = {:.3}, Φ = {:.6}, nodes = {}, edges = {}, density = {:.4}",
                 fractal.dimension, phi, node_count, edge_count, density);
        println!();

        results.push(FractalResult {
            name: fractal.name.to_string(),
            fractal_dimension: fractal.dimension,
            phi,
            node_count,
            edge_count,
            density,
        });
    }

    // Analyze results
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("RESULTS SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("┌─────────────────┬────────────┬────────────┬───────┬───────┬──────────┐");
    println!("│ Topology        │ Fractal d  │     Φ      │ Nodes │ Edges │ Density  │");
    println!("├─────────────────┼────────────┼────────────┼───────┼───────┼──────────┤");

    for r in &results {
        println!("│ {:15} │ {:10.4} │ {:10.6} │ {:5} │ {:5} │ {:8.4} │",
                 r.name, r.fractal_dimension, r.phi, r.node_count, r.edge_count, r.density);
    }
    println!("└─────────────────┴────────────┴────────────┴───────┴───────┴──────────┘");

    // Correlation analysis
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("CORRELATION ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Calculate Pearson correlation between fractal dimension and Φ
    let (r_phi, r_density) = compute_correlations(&results);

    println!("Pearson correlations:");
    println!("  r(d, Φ)       = {:+.4}  {}", r_phi, interpret_correlation(r_phi));
    println!("  r(d, density) = {:+.4}  {}", r_density, interpret_correlation(r_density));

    // Calculate correlation between density and Φ
    let r_density_phi = compute_correlation(
        &results.iter().map(|r| r.density).collect::<Vec<_>>(),
        &results.iter().map(|r| r.phi).collect::<Vec<_>>(),
    );
    println!("  r(density, Φ) = {:+.4}  {}", r_density_phi, interpret_correlation(r_density_phi));

    // Regression analysis
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("REGRESSION ANALYSIS: Φ = a + b·d");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let (slope, intercept, r_squared) = linear_regression(
        &results.iter().map(|r| r.fractal_dimension).collect::<Vec<_>>(),
        &results.iter().map(|r| r.phi).collect::<Vec<_>>(),
    );

    println!("Linear model: Φ = {:.6} + {:.6}·d", intercept, slope);
    println!("R² = {:.4}", r_squared);
    println!();

    // Print predicted vs actual
    println!("┌─────────────────┬────────────┬────────────┬────────────┐");
    println!("│ Topology        │ Actual Φ   │ Predicted  │ Residual   │");
    println!("├─────────────────┼────────────┼────────────┼────────────┤");

    for r in &results {
        let predicted = intercept + slope * r.fractal_dimension;
        let residual = r.phi - predicted;
        println!("│ {:15} │ {:10.6} │ {:10.6} │ {:+10.6} │",
                 r.name, r.phi, predicted, residual);
    }
    println!("└─────────────────┴────────────┴────────────┴────────────┘");

    // ASCII scatter plot
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("Φ vs FRACTAL DIMENSION (ASCII Scatter Plot)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    print_ascii_scatter(&results);

    // Conclusions
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("CONCLUSIONS");
    println!("═══════════════════════════════════════════════════════════════════\n");

    if r_phi.abs() > 0.7 {
        println!("STRONG correlation detected between fractal dimension and Φ!");
        if r_phi > 0.0 {
            println!("Higher fractal dimension → Higher integrated information");
        } else {
            println!("Higher fractal dimension → Lower integrated information");
        }
    } else if r_phi.abs() > 0.4 {
        println!("MODERATE correlation detected between fractal dimension and Φ.");
        println!("The relationship exists but other factors also influence Φ.");
    } else {
        println!("WEAK correlation between fractal dimension and Φ.");
        println!("Fractal dimension alone does not determine integrated information.");
    }

    println!();
    println!("Key insight: Network density shows r(density, Φ) = {:+.4}", r_density_phi);
    if r_density_phi.abs() > r_phi.abs() {
        println!("→ Density is a stronger predictor of Φ than fractal dimension!");
    }

    println!("\n✦ Study complete.");
}

fn build_fractal_topologies(dim: usize, seed: u64) -> Vec<FractalTopology> {
    println!("Building fractal topologies...\n");

    vec![
        // d ≈ 1.0 (linear chain)
        FractalTopology {
            name: "Line/Chain",
            dimension: 1.0,
            topology: ConsciousnessTopology::line(32, dim, seed),
        },
        // d ≈ 1.262 (log(4)/log(3))
        FractalTopology {
            name: "Koch Snowflake",
            dimension: (4.0_f64).ln() / (3.0_f64).ln(),  // 1.2618...
            topology: ConsciousnessTopology::koch_snowflake(3, dim, seed),
        },
        // d ≈ 1.585 (log(3)/log(2))
        FractalTopology {
            name: "Sierpinski",
            dimension: (3.0_f64).ln() / (2.0_f64).ln(),  // 1.5849...
            topology: ConsciousnessTopology::sierpinski_gasket(4, dim, seed),
        },
        // d ≈ 1.7 (tree-like fractal)
        FractalTopology {
            name: "Fractal Tree",
            dimension: 1.7,  // Approximate for typical fractal trees
            topology: ConsciousnessTopology::fractal_tree(3, 3, dim, seed),
        },
        // d = 2.0
        FractalTopology {
            name: "Torus",
            dimension: 2.0,
            topology: ConsciousnessTopology::torus(6, 6, dim, seed),
        },
        // d = 3.0 (3D hypercube)
        FractalTopology {
            name: "Hypercube 3D",
            dimension: 3.0,
            topology: ConsciousnessTopology::hypercube(3, dim, seed),
        },
    ]
}

fn compute_correlations(results: &[FractalResult]) -> (f64, f64) {
    let dims: Vec<f64> = results.iter().map(|r| r.fractal_dimension).collect();
    let phis: Vec<f64> = results.iter().map(|r| r.phi).collect();
    let densities: Vec<f64> = results.iter().map(|r| r.density).collect();

    (compute_correlation(&dims, &phis), compute_correlation(&dims, &densities))
}

fn compute_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    let mut ss_yy = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        ss_xy += dx * dy;
        ss_xx += dx * dx;
        ss_yy += dy * dy;
    }

    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;
    let r_squared = (ss_xy * ss_xy) / (ss_xx * ss_yy);

    (slope, intercept, r_squared)
}

fn interpret_correlation(r: f64) -> &'static str {
    let abs_r = r.abs();
    if abs_r > 0.9 { "very strong" }
    else if abs_r > 0.7 { "strong" }
    else if abs_r > 0.5 { "moderate" }
    else if abs_r > 0.3 { "weak" }
    else { "very weak/none" }
}

fn print_ascii_scatter(results: &[FractalResult]) {
    const WIDTH: usize = 60;
    const HEIGHT: usize = 20;

    // Find ranges
    let min_d = results.iter().map(|r| r.fractal_dimension).fold(f64::INFINITY, f64::min);
    let max_d = results.iter().map(|r| r.fractal_dimension).fold(f64::NEG_INFINITY, f64::max);
    let min_phi = results.iter().map(|r| r.phi).fold(f64::INFINITY, f64::min);
    let max_phi = results.iter().map(|r| r.phi).fold(f64::NEG_INFINITY, f64::max);

    // Add margins
    let d_range = max_d - min_d;
    let phi_range = max_phi - min_phi;
    let min_d = min_d - d_range * 0.1;
    let max_d = max_d + d_range * 0.1;
    let min_phi = min_phi - phi_range * 0.1;
    let max_phi = max_phi + phi_range * 0.1;

    // Create grid
    let mut grid = vec![vec![' '; WIDTH]; HEIGHT];

    // Plot points
    let symbols = ['C', 'L', 'K', 'S', 'T', 'M'];  // Cantor, Line, Koch, Sierpinski, Torus, Menger

    for (i, r) in results.iter().enumerate() {
        let x = ((r.fractal_dimension - min_d) / (max_d - min_d) * (WIDTH - 1) as f64) as usize;
        let y = HEIGHT - 1 - ((r.phi - min_phi) / (max_phi - min_phi) * (HEIGHT - 1) as f64) as usize;
        let x = x.min(WIDTH - 1);
        let y = y.min(HEIGHT - 1);
        grid[y][x] = symbols[i];
    }

    // Print grid with axes
    println!("Φ");
    println!("│");
    for (i, row) in grid.iter().enumerate() {
        if i == 0 {
            print!("{:.3}│", max_phi);
        } else if i == HEIGHT - 1 {
            print!("{:.3}│", min_phi);
        } else {
            print!("     │");
        }
        println!("{}", row.iter().collect::<String>());
    }
    println!("     └{}", "─".repeat(WIDTH));
    println!("      {:.2}{:width$}{:.2}  → Fractal Dimension",
             min_d, "", max_d, width = WIDTH - 10);

    // Legend
    println!();
    println!("Legend: C=Cantor  L=Line  K=Koch  S=Sierpinski  T=Torus  M=Menger");
}
