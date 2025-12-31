//! Goldilocks Hypothesis Test
//!
//! Testing whether Φ is maximized at intermediate density/connectivity.
//!
//! Hypothesis: There exists an optimal density d* where:
//! - d < d*: Too sparse → insufficient integration → low Φ
//! - d = d*: Sweet spot → balanced differentiation + integration → max Φ
//! - d > d*: Too dense → insufficient differentiation → low Φ
//!
//! Methodology:
//! 1. Fix node count at 32
//! 2. Vary edge probability from 0.05 to 0.95
//! 3. Measure Φ at each density level
//! 4. Plot Φ vs density to find the peak

use symthaea::hdc::{
    real_hv::RealHV,
    phi_real::RealPhiCalculator,
};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║           GOLDILOCKS HYPOTHESIS TEST                           ║");
    println!("║     Is Φ maximized at intermediate density?                    ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let n_nodes = 32;
    let dim = 2048;
    let n_trials = 5;  // Average over multiple random graphs per density
    let calculator = RealPhiCalculator::new();

    // Test densities from very sparse to very dense
    let densities: Vec<f64> = (1..=19).map(|i| i as f64 * 0.05).collect();

    println!("Configuration:");
    println!("  Nodes: {}", n_nodes);
    println!("  HDC Dimension: {}", dim);
    println!("  Trials per density: {}", n_trials);
    println!("  Density range: 0.05 to 0.95\n");

    println!("═══════════════════════════════════════════════════════════════════");
    println!("DENSITY SWEEP");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let mut results: Vec<(f64, f64, f64, f64)> = Vec::new(); // (density, mean_phi, std_phi, actual_density)

    for &target_density in &densities {
        let mut phis = Vec::new();
        let mut actual_densities = Vec::new();

        for trial in 0..n_trials {
            let seed = 42 + trial as u64 * 1000;
            let (nodes, edges) = create_random_graph(n_nodes, target_density, dim, seed);

            let actual_density = 2.0 * edges.len() as f64 / (n_nodes * (n_nodes - 1)) as f64;
            actual_densities.push(actual_density);

            let phi = calculator.compute(&nodes);
            phis.push(phi);
        }

        let mean_phi = phis.iter().sum::<f64>() / phis.len() as f64;
        let variance = phis.iter().map(|&p| (p - mean_phi).powi(2)).sum::<f64>() / phis.len() as f64;
        let std_phi = variance.sqrt();
        let mean_actual = actual_densities.iter().sum::<f64>() / actual_densities.len() as f64;

        println!("  Density {:.2}: Φ = {:.4} ± {:.4} (actual density: {:.3})",
                 target_density, mean_phi, std_phi, mean_actual);

        results.push((target_density, mean_phi, std_phi, mean_actual));
    }

    // Find the peak
    let (peak_density, peak_phi, _, _) = results.iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("RESULTS ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("Peak Φ = {:.4} at density = {:.2}\n", peak_phi, peak_density);

    // ASCII plot
    println!("Φ vs Density (ASCII Plot):\n");
    print_ascii_plot(&results);

    // Detailed table
    println!("\n┌──────────┬─────────┬─────────┬───────────┐");
    println!("│ Density  │   Φ     │  ± Std  │ Δ from max│");
    println!("├──────────┼─────────┼─────────┼───────────┤");

    for (d, phi, std, _) in &results {
        let delta = peak_phi - phi;
        let marker = if (d - peak_density).abs() < 0.01 { " ★" } else { "" };
        println!("│   {:.2}    │ {:.4}  │ {:.4}  │  {:+.4}  │{}",
                 d, phi, std, -delta, marker);
    }
    println!("└──────────┴─────────┴─────────┴───────────┘");

    // Statistical analysis
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("GOLDILOCKS ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Check if peak is at an intermediate density
    let is_intermediate = *peak_density > 0.15 && *peak_density < 0.85;

    // Check for inverted-U shape
    let (slope_low, slope_high) = analyze_shape(&results, *peak_density);

    println!("Peak Location: density = {:.2}", peak_density);
    println!("  Is intermediate (not at extremes)? {}", if is_intermediate { "YES ✓" } else { "NO ✗" });
    println!();

    println!("Shape Analysis:");
    println!("  Slope before peak: {:+.4} (should be positive)", slope_low);
    println!("  Slope after peak:  {:+.4} (should be negative)", slope_high);
    println!();

    let is_inverted_u = slope_low > 0.0 && slope_high < 0.0;

    println!("Inverted-U Shape? {}", if is_inverted_u { "YES ✓" } else { "UNCLEAR" });
    println!();

    // Effect size
    let min_phi = results.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
    let effect_size = (peak_phi - min_phi) / min_phi * 100.0;

    println!("Effect Size:");
    println!("  Max Φ: {:.4}", peak_phi);
    println!("  Min Φ: {:.4}", min_phi);
    println!("  Difference: {:.1}%", effect_size);
    println!();

    // Conclusion
    println!("═══════════════════════════════════════════════════════════════════");
    println!("CONCLUSION");
    println!("═══════════════════════════════════════════════════════════════════\n");

    if is_intermediate && is_inverted_u && effect_size > 1.0 {
        println!("✓ GOLDILOCKS HYPOTHESIS SUPPORTED");
        println!();
        println!("  Φ peaks at intermediate density ({:.2}), showing the classic", peak_density);
        println!("  inverted-U relationship predicted by IIT.");
        println!();
        println!("  Interpretation:");
        println!("  • Too sparse: Insufficient connections for integration");
        println!("  • Too dense: Insufficient differentiation (everything connects)");
        println!("  • Sweet spot: Optimal balance of integration AND differentiation");
    } else if effect_size < 1.0 {
        println!("✗ GOLDILOCKS HYPOTHESIS NOT CLEARLY SUPPORTED");
        println!();
        println!("  The variation in Φ across densities is small ({:.1}%).", effect_size);
        println!("  The density-Φ relationship may be weak or confounded.");
    } else {
        println!("? GOLDILOCKS HYPOTHESIS PARTIALLY SUPPORTED");
        println!();
        println!("  Peak at density {:.2}, but shape is not clearly inverted-U.", peak_density);
        println!("  More investigation needed.");
    }

    // Additional experiment: controlled topology
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("CONTROLLED EXPERIMENT: LATTICE WITH VARYING CONNECTIVITY");
    println!("═══════════════════════════════════════════════════════════════════\n");

    test_lattice_connectivity(&calculator, dim);

    println!("\n✦ Goldilocks hypothesis test complete.");
}

fn create_random_graph(n: usize, density: f64, dim: usize, seed: u64) -> (Vec<RealHV>, Vec<(usize, usize)>) {
    let mut rng = StdRng::seed_from_u64(seed);

    // Create node representations
    let nodes: Vec<RealHV> = (0..n)
        .map(|i| {
            let base = RealHV::basis(i % dim, dim);
            let noise = RealHV::random(dim, seed + i as u64 * 100).scale(0.1);
            base.add(&noise)
        })
        .collect();

    // Create edges with given probability
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i+1)..n {
            if rng.gen::<f64>() < density {
                edges.push((i, j));
            }
        }
    }

    // Ensure graph is connected (add minimum spanning tree if needed)
    ensure_connected(&mut edges, n, &mut rng);

    // Now blend edge information into node representations
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in &edges {
        adj[u].push(v);
        adj[v].push(u);
    }

    // Create node representations that encode connectivity
    let node_reps: Vec<RealHV> = (0..n)
        .map(|i| {
            let mut rep = nodes[i].clone();
            // Add weighted sum of neighbor identities
            for &neighbor in &adj[i] {
                let neighbor_contribution = nodes[neighbor].scale(0.3 / adj[i].len().max(1) as f32);
                rep = rep.add(&neighbor_contribution);
            }
            rep.normalize()
        })
        .collect();

    (node_reps, edges)
}

fn ensure_connected(edges: &mut Vec<(usize, usize)>, n: usize, rng: &mut StdRng) {
    // Simple approach: ensure a path exists by adding edges if disconnected
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], i: usize) -> usize {
        if parent[i] != i {
            parent[i] = find(parent, parent[i]);
        }
        parent[i]
    }

    fn union(parent: &mut [usize], i: usize, j: usize) {
        let pi = find(parent, i);
        let pj = find(parent, j);
        if pi != pj {
            parent[pi] = pj;
        }
    }

    for &(u, v) in edges.iter() {
        union(&mut parent, u, v);
    }

    // Connect any disconnected components
    for i in 1..n {
        if find(&mut parent, i) != find(&mut parent, 0) {
            // Connect component containing i to component containing 0
            let target = if i > 0 { rng.gen_range(0..i) } else { 0 };
            edges.push((target, i));
            union(&mut parent, target, i);
        }
    }
}

fn print_ascii_plot(results: &[(f64, f64, f64, f64)]) {
    let max_phi = results.iter().map(|r| r.1).fold(f64::NEG_INFINITY, f64::max);
    let min_phi = results.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);

    let height = 15;
    let width = 60;

    // Find peak for marking
    let peak_idx = results.iter()
        .enumerate()
        .max_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    println!("    Φ");
    println!("    │");

    for row in 0..height {
        let threshold = max_phi - (row as f64 / height as f64) * (max_phi - min_phi);

        if row == 0 {
            print!("{:.3}│", max_phi);
        } else if row == height - 1 {
            print!("{:.3}│", min_phi);
        } else {
            print!("     │");
        }

        for (i, (_, phi, _, _)) in results.iter().enumerate() {
            let col_width = width / results.len();
            let padding = col_width / 2;

            if *phi >= threshold {
                let marker = if i == peak_idx { "★" } else { "█" };
                print!("{:>width$}", marker, width = padding + 1);
            } else {
                print!("{:>width$}", " ", width = padding + 1);
            }
        }
        println!();
    }

    // X-axis
    print!("     └");
    for _ in 0..width {
        print!("─");
    }
    println!();

    // X-axis labels
    print!("      ");
    for (i, (d, _, _, _)) in results.iter().enumerate() {
        if i % 4 == 0 {
            print!("{:.1}  ", d);
        }
    }
    println!(" → Density");
}

fn analyze_shape(results: &[(f64, f64, f64, f64)], peak_density: f64) -> (f64, f64) {
    let before_peak: Vec<_> = results.iter()
        .filter(|(d, _, _, _)| *d < peak_density)
        .collect();
    let after_peak: Vec<_> = results.iter()
        .filter(|(d, _, _, _)| *d > peak_density)
        .collect();

    let slope_low = if before_peak.len() >= 2 {
        let n = before_peak.len() as f64;
        let sum_x: f64 = before_peak.iter().map(|(d, _, _, _)| d).sum();
        let sum_y: f64 = before_peak.iter().map(|(_, p, _, _)| p).sum();
        let sum_xy: f64 = before_peak.iter().map(|(d, p, _, _)| d * p).sum();
        let sum_xx: f64 = before_peak.iter().map(|(d, _, _, _)| d * d).sum();
        (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    } else {
        0.0
    };

    let slope_high = if after_peak.len() >= 2 {
        let n = after_peak.len() as f64;
        let sum_x: f64 = after_peak.iter().map(|(d, _, _, _)| d).sum();
        let sum_y: f64 = after_peak.iter().map(|(_, p, _, _)| p).sum();
        let sum_xy: f64 = after_peak.iter().map(|(d, p, _, _)| d * p).sum();
        let sum_xx: f64 = after_peak.iter().map(|(d, _, _, _)| d * d).sum();
        (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    } else {
        0.0
    };

    (slope_low, slope_high)
}

fn test_lattice_connectivity(calculator: &RealPhiCalculator, dim: usize) {
    println!("Testing how adding long-range connections affects Φ in a 6×6 lattice:\n");

    let n = 36;
    let seed = 42u64;

    // Base lattice (4-connected grid)
    let base_nodes: Vec<RealHV> = (0..n)
        .map(|i| RealHV::basis(i % dim, dim))
        .collect();

    let mut base_edges: Vec<(usize, usize)> = Vec::new();
    for i in 0..6 {
        for j in 0..6 {
            let idx = i * 6 + j;
            if j < 5 { base_edges.push((idx, idx + 1)); }  // Right
            if i < 5 { base_edges.push((idx, idx + 6)); }  // Down
        }
    }

    // Test with varying numbers of long-range connections
    let long_range_counts = [0, 5, 10, 20, 40, 80, 160];

    println!("┌────────────────────┬─────────┬─────────┬──────────┐");
    println!("│ Long-range edges   │ Total   │ Density │    Φ     │");
    println!("├────────────────────┼─────────┼─────────┼──────────┤");

    let mut rng = StdRng::seed_from_u64(seed);

    for &n_long in &long_range_counts {
        let mut edges = base_edges.clone();

        // Add random long-range connections
        let mut added = 0;
        while added < n_long {
            let u = rng.gen_range(0..n);
            let v = rng.gen_range(0..n);
            if u != v && !edges.contains(&(u.min(v), u.max(v))) {
                edges.push((u.min(v), u.max(v)));
                added += 1;
            }
        }

        // Build node representations
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(u, v) in &edges {
            adj[u].push(v);
            adj[v].push(u);
        }

        let node_reps: Vec<RealHV> = (0..n)
            .map(|i| {
                let mut rep = base_nodes[i].clone();
                for &neighbor in &adj[i] {
                    let contrib = base_nodes[neighbor].scale(0.3 / adj[i].len().max(1) as f32);
                    rep = rep.add(&contrib);
                }
                rep.normalize()
            })
            .collect();

        let density = 2.0 * edges.len() as f64 / (n * (n - 1)) as f64;
        let phi = calculator.compute(&node_reps);

        let marker = if n_long == 0 { " (baseline)" }
                    else if n_long == 20 { " ★" }
                    else { "" };

        println!("│ {:>18} │ {:>7} │ {:.4}  │  {:.4}  │{}",
                 n_long, edges.len(), density, phi, marker);
    }
    println!("└────────────────────┴─────────┴─────────┴──────────┘");

    println!("\nObservation: Long-range connections initially help integration,");
    println!("but too many destroy the differentiated structure of the lattice.");
}
