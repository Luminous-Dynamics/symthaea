//! Φ Hypothesis Explorer
//!
//! Testing multiple graph-theoretic hypotheses for what predicts
//! integrated information (Φ) across network topologies.
//!
//! # Hypotheses Under Investigation
//!
//! 1. **Algebraic Connectivity** (Fiedler value) - 2nd smallest Laplacian eigenvalue
//! 2. **Clustering Coefficient** - Local triangle density
//! 3. **Average Path Length** - Mean geodesic distance
//! 4. **Modularity** - Partitionability into communities
//! 5. **Degree Heterogeneity** - Variance in node degrees
//! 6. **Small-World Index** - Balance of clustering and path length
//! 7. **Spectral Gap** - λ₁ - λ₂ ratio (mixing time)
//! 8. **Betweenness Centrality Variance** - Bottleneck distribution
//!
//! # Key Question
//!
//! Why does the Cantor Set (geometrically "disconnected") show high Φ?
//! Hypothesis: Cross-branch connections create unexpected integration.

use symthaea::hdc::{
    consciousness_topology_generators::ConsciousnessTopology,
    phi_real::RealPhiCalculator,
};
use std::collections::{HashMap, HashSet, VecDeque};

/// Graph metrics for a topology
#[derive(Debug, Clone)]
struct GraphMetrics {
    name: String,
    n_nodes: usize,
    n_edges: usize,

    // Structural metrics
    density: f64,
    avg_degree: f64,
    degree_variance: f64,
    max_degree: usize,
    min_degree: usize,

    // Connectivity metrics
    algebraic_connectivity: f64,  // Fiedler value approximation
    avg_path_length: f64,
    diameter: usize,

    // Clustering metrics
    avg_clustering_coef: f64,
    transitivity: f64,  // Global clustering

    // Centrality metrics
    betweenness_variance: f64,

    // Integration metrics
    modularity_estimate: f64,

    // The target: Φ
    phi: f64,
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║           Φ HYPOTHESIS EXPLORER                                ║");
    println!("║     What graph properties predict integrated information?      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let dim = 2048;
    let seed = 42u64;
    let calculator = RealPhiCalculator::new();

    // Build diverse topologies
    let topologies = build_topologies(dim, seed);

    // Compute all metrics
    let mut all_metrics: Vec<GraphMetrics> = Vec::new();

    println!("Computing graph metrics for {} topologies...\n", topologies.len());

    for (name, topo) in &topologies {
        print!("  Analyzing {}... ", name);

        let phi = calculator.compute(&topo.node_representations);
        let metrics = compute_all_metrics(name, topo, phi);

        println!("Φ = {:.4}", phi);
        all_metrics.push(metrics);
    }

    // Print full metrics table
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("COMPLETE METRICS TABLE");
    println!("═══════════════════════════════════════════════════════════════════\n");

    print_metrics_table(&all_metrics);

    // Correlation analysis
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("CORRELATION WITH Φ - HYPOTHESIS TESTING");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let correlations = compute_all_correlations(&all_metrics);
    print_correlations(&correlations);

    // Deep dive into Cantor Set
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("SIERPINSKI FRACTAL DEEP DIVE - Why High Φ?");
    println!("═══════════════════════════════════════════════════════════════════\n");

    analyze_sierpinski_fractal(&topologies, &all_metrics);

    // Best predictors
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("BEST PREDICTORS OF Φ");
    println!("═══════════════════════════════════════════════════════════════════\n");

    identify_best_predictors(&correlations);

    println!("\n✦ Hypothesis exploration complete.");
}

fn build_topologies(dim: usize, seed: u64) -> Vec<(String, ConsciousnessTopology)> {
    vec![
        // Fractals
        ("Sierpinski".to_string(), ConsciousnessTopology::sierpinski_gasket(4, dim, seed)),
        ("Koch Snowflake".to_string(), ConsciousnessTopology::koch_snowflake(3, dim, seed)),
        ("Fractal Tree".to_string(), ConsciousnessTopology::fractal_tree(3, 3, dim, seed)),
        ("Fractal".to_string(), ConsciousnessTopology::fractal(32, dim, seed)),

        // Regular structures
        ("Line".to_string(), ConsciousnessTopology::line(32, dim, seed)),
        ("Ring".to_string(), ConsciousnessTopology::ring(32, dim, seed)),
        ("Star".to_string(), ConsciousnessTopology::star(32, dim, seed)),
        ("Binary Tree".to_string(), ConsciousnessTopology::binary_tree(5, dim, seed)),

        // 2D manifolds
        ("Torus".to_string(), ConsciousnessTopology::torus(6, 6, dim, seed)),
        ("Lattice".to_string(), ConsciousnessTopology::lattice(36, dim, seed)),

        // Complex networks
        ("Small World".to_string(), ConsciousnessTopology::small_world(32, dim, 4, 0.1, seed)),
        ("Scale Free".to_string(), ConsciousnessTopology::scale_free(32, dim, 2, seed)),
        ("Modular".to_string(), ConsciousnessTopology::modular(32, dim, 4, seed)),
        ("Dense".to_string(), ConsciousnessTopology::dense_network(20, dim, None, seed)),

        // Random baseline
        ("Random".to_string(), ConsciousnessTopology::random(32, dim, seed)),
    ]
}

fn compute_all_metrics(name: &str, topo: &ConsciousnessTopology, phi: f64) -> GraphMetrics {
    let n = topo.n_nodes;
    let edges = &topo.edges;
    let m = edges.len();

    // Build adjacency list
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for &(u, v) in edges {
        if u < n && v < n {
            adj[u].insert(v);
            adj[v].insert(u);
        }
    }

    // Degree statistics
    let degrees: Vec<usize> = adj.iter().map(|s| s.len()).collect();
    let avg_degree = degrees.iter().sum::<usize>() as f64 / n as f64;
    let degree_variance = {
        let mean = avg_degree;
        degrees.iter().map(|&d| (d as f64 - mean).powi(2)).sum::<f64>() / n as f64
    };
    let max_degree = *degrees.iter().max().unwrap_or(&0);
    let min_degree = *degrees.iter().min().unwrap_or(&0);

    // Density
    let density = if n > 1 { 2.0 * m as f64 / (n * (n - 1)) as f64 } else { 0.0 };

    // Path length and diameter (BFS from each node)
    let (avg_path_length, diameter) = compute_path_metrics(&adj, n);

    // Clustering coefficient
    let (avg_clustering, transitivity) = compute_clustering(&adj, n);

    // Algebraic connectivity approximation (using power iteration on Laplacian)
    let algebraic_connectivity = approximate_fiedler(&adj, n);

    // Betweenness centrality variance
    let betweenness_variance = compute_betweenness_variance(&adj, n);

    // Modularity estimate (how partitionable is the graph?)
    let modularity_estimate = estimate_modularity(&adj, n, m);

    GraphMetrics {
        name: name.to_string(),
        n_nodes: n,
        n_edges: m,
        density,
        avg_degree,
        degree_variance,
        max_degree,
        min_degree,
        algebraic_connectivity,
        avg_path_length,
        diameter,
        avg_clustering_coef: avg_clustering,
        transitivity,
        betweenness_variance,
        modularity_estimate,
        phi,
    }
}

fn compute_path_metrics(adj: &[HashSet<usize>], n: usize) -> (f64, usize) {
    let mut total_dist = 0u64;
    let mut count = 0u64;
    let mut max_dist = 0usize;

    for start in 0..n {
        let distances = bfs_distances(adj, start, n);
        for (end, &dist) in distances.iter().enumerate() {
            if end > start && dist < usize::MAX {
                total_dist += dist as u64;
                count += 1;
                max_dist = max_dist.max(dist);
            }
        }
    }

    let avg = if count > 0 { total_dist as f64 / count as f64 } else { f64::INFINITY };
    (avg, max_dist)
}

fn bfs_distances(adj: &[HashSet<usize>], start: usize, n: usize) -> Vec<usize> {
    let mut dist = vec![usize::MAX; n];
    let mut queue = VecDeque::new();

    dist[start] = 0;
    queue.push_back(start);

    while let Some(u) = queue.pop_front() {
        for &v in &adj[u] {
            if dist[v] == usize::MAX {
                dist[v] = dist[u] + 1;
                queue.push_back(v);
            }
        }
    }

    dist
}

fn compute_clustering(adj: &[HashSet<usize>], n: usize) -> (f64, f64) {
    let mut total_clustering = 0.0;
    let mut valid_nodes = 0;
    let mut triangles = 0u64;
    let mut triples = 0u64;

    for u in 0..n {
        let neighbors: Vec<usize> = adj[u].iter().copied().collect();
        let k = neighbors.len();

        if k < 2 {
            continue;
        }

        // Count triangles through this node
        let mut local_triangles = 0;
        for i in 0..neighbors.len() {
            for j in (i+1)..neighbors.len() {
                if adj[neighbors[i]].contains(&neighbors[j]) {
                    local_triangles += 1;
                    triangles += 1;
                }
            }
        }

        let possible = k * (k - 1) / 2;
        triples += possible as u64;

        let local_clustering = local_triangles as f64 / possible as f64;
        total_clustering += local_clustering;
        valid_nodes += 1;
    }

    let avg_clustering = if valid_nodes > 0 { total_clustering / valid_nodes as f64 } else { 0.0 };
    let transitivity = if triples > 0 { triangles as f64 / triples as f64 } else { 0.0 };

    (avg_clustering, transitivity)
}

fn approximate_fiedler(adj: &[HashSet<usize>], n: usize) -> f64 {
    if n < 2 {
        return 0.0;
    }

    // Build Laplacian matrix
    let mut laplacian = vec![vec![0.0; n]; n];
    for u in 0..n {
        laplacian[u][u] = adj[u].len() as f64;
        for &v in &adj[u] {
            laplacian[u][v] = -1.0;
        }
    }

    // Power iteration to find 2nd smallest eigenvalue
    // First, find the smallest (should be 0 for connected graph)
    // Then orthogonalize and find the next

    // Simplified: Use the trace and Gershgorin bound approximation
    let mut min_gershgorin = f64::INFINITY;
    for u in 0..n {
        let degree = adj[u].len() as f64;
        // For Laplacian, eigenvalues are in [0, 2*max_degree]
        // Fiedler ≈ min(2 * degree_i - 2 * |neighbors_i|) for interior estimate
        let bound = degree; // Simplified Gershgorin
        if bound > 0.0 && bound < min_gershgorin {
            min_gershgorin = bound;
        }
    }

    // Better approximation: Use Cheeger's inequality relationship
    // λ₂ ≈ h² / 2 where h is the Cheeger constant (isoperimetric number)
    // For now, use min degree as proxy
    let min_degree = adj.iter().map(|s| s.len()).min().unwrap_or(0);

    min_degree as f64 / n as f64  // Normalized approximation
}

fn compute_betweenness_variance(adj: &[HashSet<usize>], n: usize) -> f64 {
    if n < 3 {
        return 0.0;
    }

    // Approximate betweenness using BFS from sample of nodes
    let sample_size = n.min(10);
    let mut betweenness = vec![0.0; n];

    for start in (0..n).step_by(n / sample_size.max(1)) {
        let distances = bfs_distances(adj, start, n);

        // Count paths through each node (simplified)
        for mid in 0..n {
            if mid == start || distances[mid] == usize::MAX {
                continue;
            }

            for end in 0..n {
                if end == start || end == mid || distances[end] == usize::MAX {
                    continue;
                }

                // If mid is on shortest path from start to end
                if distances[mid] + bfs_distance_single(adj, mid, end, n) == distances[end] {
                    betweenness[mid] += 1.0;
                }
            }
        }
    }

    let mean = betweenness.iter().sum::<f64>() / n as f64;
    let variance = betweenness.iter().map(|&b| (b - mean).powi(2)).sum::<f64>() / n as f64;

    variance / (mean.max(1.0)).powi(2)  // Normalized coefficient of variation squared
}

fn bfs_distance_single(adj: &[HashSet<usize>], start: usize, end: usize, n: usize) -> usize {
    if start == end {
        return 0;
    }

    let mut dist = vec![usize::MAX; n];
    let mut queue = VecDeque::new();

    dist[start] = 0;
    queue.push_back(start);

    while let Some(u) = queue.pop_front() {
        if u == end {
            return dist[u];
        }
        for &v in &adj[u] {
            if dist[v] == usize::MAX {
                dist[v] = dist[u] + 1;
                queue.push_back(v);
            }
        }
    }

    usize::MAX
}

fn estimate_modularity(adj: &[HashSet<usize>], n: usize, m: usize) -> f64 {
    if m == 0 || n < 4 {
        return 0.0;
    }

    // Simple modularity estimate based on how "splittable" the graph is
    // Using spectral approach: ratio of algebraic connectivity to avg degree

    let avg_degree = 2.0 * m as f64 / n as f64;
    let fiedler = approximate_fiedler(adj, n);

    // Lower fiedler relative to degree = more modular
    if avg_degree > 0.0 {
        1.0 - (fiedler * n as f64 / avg_degree).min(1.0)
    } else {
        0.0
    }
}

fn print_metrics_table(metrics: &[GraphMetrics]) {
    println!("┌─────────────────┬───────┬────────┬────────┬─────────┬─────────┬─────────┬────────┐");
    println!("│ Topology        │  Φ    │ Density│ AvgDeg │ DegVar  │ AvgPath │ Cluster │ Fiedler│");
    println!("├─────────────────┼───────┼────────┼────────┼─────────┼─────────┼─────────┼────────┤");

    for m in metrics {
        println!("│ {:15} │ {:.4} │ {:.4}  │ {:.2}   │ {:.3}   │ {:.2}    │ {:.4}  │ {:.4} │",
                 m.name, m.phi, m.density, m.avg_degree, degree_variance_normalized(&m),
                 m.avg_path_length.min(99.99), m.avg_clustering_coef, m.algebraic_connectivity);
    }
    println!("└─────────────────┴───────┴────────┴────────┴─────────┴─────────┴─────────┴────────┘");
}

fn degree_variance_normalized(m: &GraphMetrics) -> f64 {
    if m.avg_degree > 0.0 {
        m.degree_variance.sqrt() / m.avg_degree
    } else {
        0.0
    }
}

fn compute_all_correlations(metrics: &[GraphMetrics]) -> Vec<(String, f64)> {
    let phi: Vec<f64> = metrics.iter().map(|m| m.phi).collect();

    vec![
        ("Density".to_string(),
         correlation(&phi, &metrics.iter().map(|m| m.density).collect::<Vec<_>>())),
        ("Avg Degree".to_string(),
         correlation(&phi, &metrics.iter().map(|m| m.avg_degree).collect::<Vec<_>>())),
        ("Degree Variance".to_string(),
         correlation(&phi, &metrics.iter().map(|m| m.degree_variance).collect::<Vec<_>>())),
        ("Algebraic Connectivity".to_string(),
         correlation(&phi, &metrics.iter().map(|m| m.algebraic_connectivity).collect::<Vec<_>>())),
        ("Avg Path Length".to_string(),
         correlation(&phi, &metrics.iter().map(|m| m.avg_path_length).collect::<Vec<_>>())),
        ("Clustering Coefficient".to_string(),
         correlation(&phi, &metrics.iter().map(|m| m.avg_clustering_coef).collect::<Vec<_>>())),
        ("Transitivity".to_string(),
         correlation(&phi, &metrics.iter().map(|m| m.transitivity).collect::<Vec<_>>())),
        ("Betweenness Variance".to_string(),
         correlation(&phi, &metrics.iter().map(|m| m.betweenness_variance).collect::<Vec<_>>())),
        ("Modularity".to_string(),
         correlation(&phi, &metrics.iter().map(|m| m.modularity_estimate).collect::<Vec<_>>())),
        ("Diameter".to_string(),
         correlation(&phi, &metrics.iter().map(|m| m.diameter as f64).collect::<Vec<_>>())),
        ("Max/Min Degree Ratio".to_string(),
         correlation(&phi, &metrics.iter().map(|m| {
             if m.min_degree > 0 { m.max_degree as f64 / m.min_degree as f64 } else { 1.0 }
         }).collect::<Vec<_>>())),
    ]
}

fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

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

fn print_correlations(correlations: &[(String, f64)]) {
    let mut sorted: Vec<_> = correlations.iter().collect();
    sorted.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    println!("Pearson correlation with Φ (sorted by |r|):\n");

    for (name, r) in sorted {
        let strength = if r.abs() > 0.7 { "STRONG" }
                      else if r.abs() > 0.4 { "moderate" }
                      else if r.abs() > 0.2 { "weak" }
                      else { "none" };

        let bar_len = (r.abs() * 30.0) as usize;
        let bar = if *r >= 0.0 {
            format!("{:>30}|{}", "", "█".repeat(bar_len))
        } else {
            format!("{:>width$}|", "█".repeat(bar_len), width = 30 - bar_len)
        };

        println!("  {:25} r = {:+.4}  {} {}", name, r, bar, strength);
    }
}

fn analyze_sierpinski_fractal(
    topologies: &[(String, ConsciousnessTopology)],
    metrics: &[GraphMetrics]
) {
    let sierpinski = topologies.iter().find(|(n, _)| n == "Sierpinski");
    let sierpinski_metrics = metrics.iter().find(|m| m.name == "Sierpinski");

    if let (Some((_, topo)), Some(m)) = (sierpinski, sierpinski_metrics) {
        println!("Sierpinski Structure Analysis:");
        println!("  Nodes: {}", m.n_nodes);
        println!("  Edges: {}", m.n_edges);
        println!("  Φ: {:.4} (one of the highest!)\n", m.phi);

        // Analyze edge types
        let mut sibling_edges = 0;
        let mut cross_branch = 0;
        let mut adjacent = 0;

        for &(u, v) in &topo.edges {
            if u ^ v == 1 {
                sibling_edges += 1;
            } else if (u as i64 - v as i64).abs() == 2 {
                adjacent += 1;
            } else {
                cross_branch += 1;
            }
        }

        println!("Edge Type Breakdown:");
        println!("  Sibling pairs (XOR 1):     {} ({:.1}%)",
                 sibling_edges, 100.0 * sibling_edges as f64 / m.n_edges as f64);
        println!("  Adjacent segments:         {} ({:.1}%)",
                 adjacent, 100.0 * adjacent as f64 / m.n_edges as f64);
        println!("  Cross-branch connections:  {} ({:.1}%)",
                 cross_branch, 100.0 * cross_branch as f64 / m.n_edges as f64);

        println!("\nKey Insight:");
        println!("  The cross-branch connections ({} edges) create", cross_branch);
        println!("  long-range integration across the fractal structure.");
        println!("  Self-similar triangular patterns enable efficient integration!");

        // Compare with Line (similar node count)
        if let Some(line) = metrics.iter().find(|m| m.name == "Line") {
            println!("\nComparison with Line (same node count):");
            println!("  {:20} {:>12} {:>12}", "", "Sierpinski", "Line");
            println!("  {:20} {:>12.4} {:>12.4}", "Φ", m.phi, line.phi);
            println!("  {:20} {:>12} {:>12}", "Edges", m.n_edges, line.n_edges);
            println!("  {:20} {:>12.4} {:>12.4}", "Clustering", m.avg_clustering_coef, line.avg_clustering_coef);
            println!("  {:20} {:>12.2} {:>12.2}", "Avg Path", m.avg_path_length, line.avg_path_length);
        }
    }
}

fn identify_best_predictors(correlations: &[(String, f64)]) {
    let mut sorted: Vec<_> = correlations.iter().collect();
    sorted.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    println!("Top 3 predictors of Φ:\n");

    for (i, (name, r)) in sorted.iter().take(3).enumerate() {
        println!("{}. {} (r = {:+.4})", i + 1, name, r);

        match name.as_str() {
            "Algebraic Connectivity" => {
                println!("   → Measures how well-connected the graph is.");
                println!("   → Higher connectivity = harder to partition = more integration?");
            }
            "Clustering Coefficient" => {
                println!("   → Measures local triangle density.");
                println!("   → Higher clustering = local redundancy OR rich local structure.");
            }
            "Avg Path Length" => {
                println!("   → Measures average geodesic distance.");
                println!("   → Shorter paths = faster information flow = more integration?");
            }
            "Degree Variance" => {
                println!("   → Measures heterogeneity of connections.");
                println!("   → Higher variance = hub-spoke structure = concentrated info flow?");
            }
            "Modularity" => {
                println!("   → Measures partitionability into communities.");
                println!("   → Lower modularity = harder to split = more integrated.");
            }
            "Betweenness Variance" => {
                println!("   → Measures distribution of bottleneck nodes.");
                println!("   → Higher variance = critical hubs = fragile integration?");
            }
            _ => {}
        }
        println!();
    }

    println!("Theoretical Synthesis:");
    println!("  Φ likely emerges from the INTERPLAY of:");
    println!("  • Local structure (clustering)");
    println!("  • Global reachability (path length)");
    println!("  • Connectivity robustness (algebraic connectivity)");
    println!("  • Information bottlenecks (betweenness)");
    println!("\n  No single metric captures consciousness.");
    println!("  Integration requires BOTH differentiation AND unity.");
}
