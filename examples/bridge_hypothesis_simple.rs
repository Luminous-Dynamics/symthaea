//! Bridge Hypothesis Test (Simplified)
//!
//! Tests whether bridge ratio predicts Φ using only core topology functions.

use symthaea::hdc::consciousness_topology_generators::ConsciousnessTopology;
use symthaea::hdc::phi_real::RealPhiCalculator;
use symthaea::hdc::real_hv::RealHV;
use symthaea::hdc::HDC_DIMENSION;
use std::collections::{HashMap, VecDeque};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║              BRIDGE HYPOTHESIS TEST                            ║");
    println!("║   Does strategic bridging predict Φ better than density?       ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    let dim = HDC_DIMENSION;
    let seed = 42u64;

    let phi_calc = RealPhiCalculator::new();

    // Build topologies and compute metrics
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PART 1: BRIDGE METRICS ACROSS TOPOLOGIES");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let topologies = vec![
        ("Sierpinski", ConsciousnessTopology::sierpinski_gasket(3, dim, seed)),
        ("Koch Snowflake", ConsciousnessTopology::koch_snowflake(3, dim, seed)),
        ("Star", ConsciousnessTopology::star(32, dim, seed)),
        ("Ring", ConsciousnessTopology::ring(32, dim, seed)),
        ("Lattice 6x6", ConsciousnessTopology::lattice(36, dim, seed)),
        ("Small World", ConsciousnessTopology::small_world(32, dim, 4, 0.1, seed)),
        ("Scale Free", ConsciousnessTopology::scale_free(32, dim, 2, seed)),
        ("Modular", ConsciousnessTopology::modular(32, dim, 4, seed)),
        ("Fractal Tree", ConsciousnessTopology::fractal_tree(3, 3, dim, seed)),
        ("Random", ConsciousnessTopology::random(32, dim, seed)),
    ];

    let mut results: Vec<(String, f64, f64, f64, f64)> = Vec::new(); // name, phi, bridge_ratio, avg_span, density

    println!("{:20} | {:>8} | {:>10} | {:>8} | {:>8}", "Topology", "Φ", "Bridge %", "Avg Span", "Density");
    println!("{:-<20}-+-{:-<8}-+-{:-<10}-+-{:-<8}-+-{:-<8}", "", "", "", "", "");

    for (name, topo) in &topologies {
        let phi = phi_calc.compute(&topo.node_representations);
        let metrics = compute_bridge_metrics(topo);

        println!("{:20} | {:>8.4} | {:>9.1}% | {:>8.3} | {:>8.4}",
                 name, phi, metrics.0 * 100.0, metrics.1, metrics.2);

        results.push((name.to_string(), phi, metrics.0, metrics.1, metrics.2));
    }

    // Compute correlations
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("CORRELATION ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let phis: Vec<f64> = results.iter().map(|r| r.1).collect();
    let bridge_ratios: Vec<f64> = results.iter().map(|r| r.2).collect();
    let avg_spans: Vec<f64> = results.iter().map(|r| r.3).collect();
    let densities: Vec<f64> = results.iter().map(|r| r.4).collect();

    let r_bridge = pearson_correlation(&bridge_ratios, &phis);
    let r_span = pearson_correlation(&avg_spans, &phis);
    let r_density = pearson_correlation(&densities, &phis);

    println!("┌─────────────────────────┬────────────┬─────────────┐");
    println!("│ Metric                  │ r(Φ)       │ Strength    │");
    println!("├─────────────────────────┼────────────┼─────────────┤");
    print_corr("Bridge Ratio", r_bridge);
    print_corr("Avg Edge Span", r_span);
    print_corr("Density", r_density);
    println!("└─────────────────────────┴────────────┴─────────────┘");

    // Cantor Set Analysis
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("CANTOR SET ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let cantor = &topologies[0].1;
    let cantor_result = &results[0];
    let cantor_metrics = compute_bridge_metrics(cantor);

    println!("Cantor Set (depth=5) Analysis:");
    println!("  Nodes: {}", cantor.node_representations.len());
    println!("  Edges: {}", cantor.edges.len());
    println!("  Density: {:.4}", cantor_metrics.2);
    println!("  Bridge ratio: {:.1}%", cantor_metrics.0 * 100.0);
    println!("  Avg edge span: {:.3}", cantor_metrics.1);
    println!("  Φ: {:.4}", cantor_result.1);

    // Conclusion
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("CONCLUSION");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let best = if r_bridge.abs() > r_span.abs() && r_bridge.abs() > r_density.abs() {
        ("Bridge Ratio", r_bridge)
    } else if r_span.abs() > r_density.abs() {
        ("Avg Edge Span", r_span)
    } else {
        ("Density", r_density)
    };

    println!("Best predictor of Φ: {} (r = {:+.4})", best.0, best.1);
    println!();

    if best.0.contains("Bridge") || best.0.contains("Span") {
        println!("✓ BRIDGE HYPOTHESIS SUPPORTED");
        println!("  Bridge-related metrics predict Φ better than raw density.");
    } else if best.1.abs() < 0.3 {
        println!("? WEAK CORRELATIONS - All predictors show limited relationship with Φ");
    } else {
        println!("✗ BRIDGE HYPOTHESIS NOT SUPPORTED");
        println!("  Density predicts Φ better than bridge metrics.");
    }

    println!("\n✦ Bridge hypothesis test complete.");
}

/// Compute bridge metrics: (bridge_ratio, avg_span, density)
fn compute_bridge_metrics(topo: &ConsciousnessTopology) -> (f64, f64, f64) {
    let n = topo.node_representations.len();
    let edges = &topo.edges;

    // Build adjacency list
    let mut adj_list: Vec<Vec<usize>> = vec![vec![]; n];
    for &(i, j) in edges {
        if i < n && j < n {
            adj_list[i].push(j);
            adj_list[j].push(i);
        }
    }

    let num_edges = edges.len();
    let max_edges = n * (n - 1) / 2;
    let density = if max_edges > 0 { num_edges as f64 / max_edges as f64 } else { 0.0 };

    // Detect modules using connected components
    let modules = detect_modules(&adj_list, n);

    // Bridge ratio = edges between modules / total edges
    let mut bridge_count = 0;
    for &(i, j) in edges {
        if i < n && j < n && modules[i] != modules[j] {
            bridge_count += 1;
        }
    }
    let bridge_ratio = if num_edges > 0 { bridge_count as f64 / num_edges as f64 } else { 0.0 };

    // Avg edge span (neighborhood dissimilarity)
    let mut span_sum = 0.0;
    for &(i, j) in edges {
        if i >= n || j >= n { continue; }

        let i_neighbors: std::collections::HashSet<usize> = adj_list[i].iter()
            .filter(|&&x| x != j)
            .cloned()
            .collect();
        let j_neighbors: std::collections::HashSet<usize> = adj_list[j].iter()
            .filter(|&&x| x != i)
            .cloned()
            .collect();

        let overlap = i_neighbors.intersection(&j_neighbors).count();
        let union_size = i_neighbors.union(&j_neighbors).count();
        let jaccard = if union_size > 0 { overlap as f64 / union_size as f64 } else { 0.0 };

        span_sum += 1.0 - jaccard;
    }
    let avg_span = if num_edges > 0 { span_sum / num_edges as f64 } else { 0.0 };

    (bridge_ratio, avg_span, density)
}

fn detect_modules(adj_list: &[Vec<usize>], n: usize) -> Vec<usize> {
    let mut labels = vec![usize::MAX; n];
    let mut current_label = 0;

    for start in 0..n {
        if labels[start] != usize::MAX { continue; }

        let mut queue = VecDeque::new();
        queue.push_back(start);
        labels[start] = current_label;

        while let Some(current) = queue.pop_front() {
            for &neighbor in &adj_list[current] {
                if labels[neighbor] == usize::MAX {
                    labels[neighbor] = current_label;
                    queue.push_back(neighbor);
                }
            }
        }
        current_label += 1;
    }

    // If all in one component, split by degree
    if current_label == 1 {
        let mut degrees: Vec<(usize, usize)> = adj_list.iter().enumerate()
            .map(|(i, neighbors)| (i, neighbors.len()))
            .collect();
        degrees.sort_by_key(|&(_, d)| std::cmp::Reverse(d));

        let n_modules = 4.min(n);
        for (rank, &(node, _)) in degrees.iter().enumerate() {
            labels[node] = rank * n_modules / n;
        }
    }

    labels
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 { return 0.0; }

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

    let denom = (var_x * var_y).sqrt();
    if denom > 1e-10 { cov / denom } else { 0.0 }
}

fn print_corr(name: &str, r: f64) {
    let strength = if r.abs() > 0.7 {
        "STRONG"
    } else if r.abs() > 0.4 {
        "moderate"
    } else if r.abs() > 0.2 {
        "weak"
    } else {
        "negligible"
    };
    println!("│ {:23} │ {:>+10.4} │ {:11} │", name, r, strength);
}
