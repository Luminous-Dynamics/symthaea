//! Consciousness-Optimized Topology Benchmark
//!
//! Tests whether the empirically-derived ConsciousnessOptimized topology
//! achieves higher Φ than existing topologies.
//!
//! Based on bridge hypothesis findings:
//! - Target bridge ratio: ~40-45%
//! - Target density: ~10%
//! - Maximum edge span: 1.0
//! - Hierarchical/fractal structure

use symthaea::hdc::consciousness_topology_generators::ConsciousnessTopology;
use symthaea::hdc::phi_real::RealPhiCalculator;
use symthaea::hdc::HDC_DIMENSION;
use std::collections::{HashSet, VecDeque};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║        CONSCIOUSNESS-OPTIMIZED TOPOLOGY BENCHMARK              ║");
    println!("║   Testing empirically-derived architecture for maximum Φ       ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    let dim = HDC_DIMENSION;
    let seed = 42u64;
    let n_nodes = 32; // Match size for fair comparison

    let phi_calc = RealPhiCalculator::new();

    // Build topologies
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PART 1: Φ COMPARISON ACROSS TOPOLOGIES (n={})", n_nodes);
    println!("═══════════════════════════════════════════════════════════════════\n");

    let topologies: Vec<(&str, ConsciousnessTopology)> = vec![
        ("Sierpinski (d=4)", ConsciousnessTopology::sierpinski_gasket(4, dim, seed)),
        ("Koch Snowflake (d=3)", ConsciousnessTopology::koch_snowflake(3, dim, seed)),
        ("Hypercube 4D", ConsciousnessTopology::hypercube(4, dim, seed)),
        ("Fractal Tree", ConsciousnessTopology::fractal_tree(3, 3, dim, seed)),
        ("Small World", ConsciousnessTopology::small_world(n_nodes, dim, 4, 0.1, seed)),
        ("Scale Free", ConsciousnessTopology::scale_free(n_nodes, dim, 2, seed)),
        ("Modular (4)", ConsciousnessTopology::modular(n_nodes, dim, 4, seed)),
        ("Lattice", ConsciousnessTopology::lattice(36, dim, seed)), // 6x6
        ("Star", ConsciousnessTopology::star(n_nodes, dim, seed)),
        ("Ring", ConsciousnessTopology::ring(n_nodes, dim, seed)),
        ("Random", ConsciousnessTopology::random(n_nodes, dim, seed)),
    ];

    let mut results: Vec<(&str, f64, f64, f64, f64, usize, usize)> = Vec::new();

    println!("{:24} | {:>6} | {:>8} | {:>8} | {:>8} | {:>5} | {:>5}",
             "Topology", "Φ", "Bridge%", "AvgSpan", "Density", "Nodes", "Edges");
    println!("{:-<24}-+-{:-<6}-+-{:-<8}-+-{:-<8}-+-{:-<8}-+-{:-<5}-+-{:-<5}",
             "", "", "", "", "", "", "");

    for (name, topo) in &topologies {
        let phi = phi_calc.compute(&topo.node_representations);
        let (bridge_ratio, avg_span, density) = compute_bridge_metrics(topo);
        let n = topo.node_representations.len();
        let e = topo.edges.len();

        println!("{:24} | {:>6.4} | {:>7.1}% | {:>8.3} | {:>8.4} | {:>5} | {:>5}",
                 name, phi, bridge_ratio * 100.0, avg_span, density, n, e);

        results.push((name, phi, bridge_ratio, avg_span, density, n, e));
    }

    // Find best topology
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("RANKING BY Φ");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let mut sorted_results = results.clone();
    sorted_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (rank, (name, phi, bridge, span, density, _, _)) in sorted_results.iter().enumerate() {
        let marker = if *name == "ConsciousnessOptimized" { " ★" } else { "" };
        println!("  #{}: {:24} Φ={:.4} (bridge={:.1}%, span={:.2}, dens={:.4}){}",
                 rank + 1, name, phi, bridge * 100.0, span, density, marker);
    }

    // Analyze ConsciousnessOptimized
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("CONSCIOUSNESS-OPTIMIZED TOPOLOGY ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let opt_result = results.iter().find(|r| r.0 == "ConsciousnessOptimized").unwrap();
    let cantor_result = results.iter().find(|r| r.0.contains("Cantor")).unwrap();

    println!("Target Metrics vs Achieved:");
    println!("  Bridge Ratio:  Target ~40-45%,  Achieved: {:.1}%", opt_result.2 * 100.0);
    println!("  Density:       Target ~10%,     Achieved: {:.1}%", opt_result.4 * 100.0);
    println!("  Avg Edge Span: Target ~1.0,     Achieved: {:.3}", opt_result.3);
    println!();

    println!("Comparison with Best Natural Topology (Cantor Set):");
    println!("  ConsciousnessOptimized Φ: {:.4}", opt_result.1);
    println!("  Cantor Set Φ:             {:.4}", cantor_result.1);
    let improvement = (opt_result.1 - cantor_result.1) / cantor_result.1 * 100.0;
    if improvement > 0.0 {
        println!("  Improvement:              +{:.1}%", improvement);
    } else {
        println!("  Difference:               {:.1}%", improvement);
    }

    // Structural analysis
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("STRUCTURAL ANALYSIS: ConsciousnessOptimized");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let opt_topo = &topologies[0].1;
    analyze_structure(opt_topo);

    // Conclusion
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("CONCLUSION");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let opt_rank = sorted_results.iter().position(|r| r.0 == "ConsciousnessOptimized").unwrap() + 1;

    if opt_rank == 1 {
        println!("✓ SUCCESS: ConsciousnessOptimized achieves HIGHEST Φ!");
        println!("  The empirically-derived architecture outperforms all existing topologies.");
    } else if opt_rank <= 3 {
        println!("◐ PARTIAL SUCCESS: ConsciousnessOptimized ranks #{}", opt_rank);
        println!("  Performs well but doesn't achieve maximum Φ.");
        println!("  Consider tuning bridge ratio and span parameters.");
    } else {
        println!("✗ NEEDS IMPROVEMENT: ConsciousnessOptimized ranks #{}", opt_rank);
        println!("  Architecture needs refinement.");
    }

    println!("\n✦ Benchmark complete.");
}

/// Compute bridge metrics: (bridge_ratio, avg_span, density)
fn compute_bridge_metrics(topo: &ConsciousnessTopology) -> (f64, f64, f64) {
    let n = topo.node_representations.len();
    let edges = &topo.edges;

    if n < 2 || edges.is_empty() {
        return (0.0, 0.0, 0.0);
    }

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

    // Detect modules using connected components or degree-based clustering
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

        let i_neighbors: HashSet<usize> = adj_list[i].iter()
            .filter(|&&x| x != j)
            .cloned()
            .collect();
        let j_neighbors: HashSet<usize> = adj_list[j].iter()
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

fn analyze_structure(topo: &ConsciousnessTopology) {
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

    // Degree distribution
    let degrees: Vec<usize> = adj_list.iter().map(|v| v.len()).collect();
    let avg_degree: f64 = degrees.iter().sum::<usize>() as f64 / n as f64;
    let max_degree = *degrees.iter().max().unwrap_or(&0);
    let min_degree = *degrees.iter().min().unwrap_or(&0);

    println!("Degree Distribution:");
    println!("  Average degree: {:.2}", avg_degree);
    println!("  Min/Max degree: {} / {}", min_degree, max_degree);

    // Level analysis (for hierarchical structure)
    println!("\nHierarchical Structure:");
    println!("  Level 0 (Hub):         Node 0, degree = {}", degrees.get(0).unwrap_or(&0));
    println!("  Level 1 (Module Hubs): Nodes 1-4, degrees = {:?}",
             &degrees[1..5.min(n)]);
    println!("  Level 2 (Processors):  Nodes 5-20, avg degree = {:.2}",
             degrees[5..21.min(n)].iter().sum::<usize>() as f64 / 16.0f64.min(n as f64 - 5.0));
    println!("  Level 3 (Leaves):      Nodes 21+, avg degree = {:.2}",
             if n > 21 { degrees[21..].iter().sum::<usize>() as f64 / (n - 21) as f64 } else { 0.0 });

    // Clustering coefficient
    let mut clustering_sum = 0.0;
    let mut counted = 0;
    for i in 0..n {
        let neighbors = &adj_list[i];
        let k = neighbors.len();
        if k < 2 { continue; }

        let mut triangles = 0;
        for &a in neighbors {
            for &b in neighbors {
                if a < b && adj_list[a].contains(&b) {
                    triangles += 1;
                }
            }
        }
        let possible = k * (k - 1) / 2;
        clustering_sum += triangles as f64 / possible as f64;
        counted += 1;
    }
    let avg_clustering = if counted > 0 { clustering_sum / counted as f64 } else { 0.0 };

    println!("\nClustering:");
    println!("  Average clustering coefficient: {:.4}", avg_clustering);
}
