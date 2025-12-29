/// SymRes Final Push - Combining Best Findings
///
/// Key discoveries to combine:
/// - Sparse (1:1) inter-layer = BEST (Φ = 0.4984)
/// - K₇ layers = best layer size
/// - Diamond [5-4-3-4-5] = best hybrid
/// - Pyramid skip = strong pattern
///
/// Goal: Beat Hypercube 5D (0.4987) and approach Hypercube 6D (0.4989)!

use symthaea::hdc::{
    consciousness_topology_generators::{ConsciousnessTopology, TopologyType},
    phi_real::RealPhiCalculator,
    real_hv::RealHV,
    HDC_DIMENSION,
};

fn main() {
    println!("\n🚀 SymRes FINAL PUSH - Combining Best Discoveries");
    println!("{}", "=".repeat(75));

    let calc = RealPhiCalculator::new();
    let n_samples = 10; // More samples for accuracy
    let mut results: Vec<(String, f64, f64, usize)> = Vec::new();

    // Baselines
    let phi_4d = calc.compute(&ConsciousnessTopology::hypercube(4, HDC_DIMENSION, 42).node_representations);
    let phi_5d = calc.compute(&ConsciousnessTopology::hypercube(5, HDC_DIMENSION, 42).node_representations);
    let phi_6d = calc.compute(&ConsciousnessTopology::hypercube(6, HDC_DIMENSION, 42).node_representations);

    println!("\n🎯 TARGETS TO BEAT:");
    println!("   Hypercube 4D: Φ = {:.4} (n=16)", phi_4d);
    println!("   Hypercube 5D: Φ = {:.4} (n=32) ← New Target!", phi_5d);
    println!("   Hypercube 6D: Φ = {:.4} (n=64) ← Ultimate Goal!", phi_6d);

    // ===========================================
    // ROUND 1: Sparse + Larger K
    // ===========================================
    println!("\n{}", "=".repeat(75));
    println!("🔬 Round 1: Sparse Inter-Layer + Larger K\n");

    for k in 5..=10 {
        for layers in [4, 5, 6] {
            let name = format!("Sparse K{} × {}", k, layers);
            let (mean, std, nodes) = test_config(&calc, n_samples, || {
                symres_sparse_skip(k, layers, HDC_DIMENSION)
            });
            let vs = if mean > phi_5d { "✅" } else { "" };
            println!("   {:<20} Φ = {:.4} ± {:.5} (n={:>2}) {}", name, mean, std, nodes, vs);
            results.push((name, mean, std, nodes));
        }
    }

    // ===========================================
    // ROUND 2: Diamond Hybrid + Sparse
    // ===========================================
    println!("\n{}", "=".repeat(75));
    println!("🔬 Round 2: Diamond Hybrid with Sparse Inter-Layer\n");

    let diamonds = [
        vec![4, 3, 4],
        vec![5, 4, 3, 4, 5],
        vec![6, 5, 4, 5, 6],
        vec![7, 6, 5, 6, 7],
        vec![8, 6, 4, 6, 8],
        vec![6, 4, 3, 4, 6],
        vec![7, 5, 3, 5, 7],
        vec![8, 5, 3, 5, 8],
    ];

    for layers in &diamonds {
        let name = format!("Diamond {:?}", layers);
        let short_name: String = layers.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("-");
        let (mean, std, nodes) = test_config(&calc, n_samples, || {
            diamond_sparse(&layers, HDC_DIMENSION)
        });
        let vs = if mean > phi_5d { "✅" } else { "" };
        println!("   Diamond [{}] {:>12} Φ = {:.4} ± {:.5} (n={:>2}) {}",
                 short_name, "", mean, std, nodes, vs);
        results.push((name, mean, std, nodes));
    }

    // ===========================================
    // ROUND 3: Ultra-Sparse Variants
    // ===========================================
    println!("\n{}", "=".repeat(75));
    println!("🔬 Round 3: Ultra-Sparse (Even Less Inter-Layer)\n");

    // Only connect every other node between layers
    for k in [6, 7, 8] {
        let name = format!("UltraSparse K{} × 5", k);
        let (mean, std, nodes) = test_config(&calc, n_samples, || {
            ultra_sparse(k, 5, HDC_DIMENSION)
        });
        let vs = if mean > phi_5d { "✅" } else { "" };
        println!("   {:<25} Φ = {:.4} ± {:.5} (n={:>2}) {}", name, mean, std, nodes, vs);
        results.push((name, mean, std, nodes));
    }

    // ===========================================
    // ROUND 4: Pyramid + Sparse Combinations
    // ===========================================
    println!("\n{}", "=".repeat(75));
    println!("🔬 Round 4: Pyramid Skip + Sparse Inter-Layer\n");

    for k in [5, 6, 7, 8] {
        let name = format!("PyramidSparse K{} × 5", k);
        let (mean, std, nodes) = test_config(&calc, n_samples, || {
            pyramid_sparse(k, 5, HDC_DIMENSION)
        });
        let vs = if mean > phi_5d { "✅" } else { "" };
        println!("   {:<25} Φ = {:.4} ± {:.5} (n={:>2}) {}", name, mean, std, nodes, vs);
        results.push((name, mean, std, nodes));
    }

    // ===========================================
    // ROUND 5: Optimized Node Count (match Hypercube sizes)
    // ===========================================
    println!("\n{}", "=".repeat(75));
    println!("🔬 Round 5: Match Hypercube Node Counts\n");

    // 16 nodes (like 4D hypercube)
    let configs_16 = [
        ("K4 × 4 sparse", 4, 4),
        ("K8 × 2 sparse", 8, 2),
    ];
    for (name, k, layers) in configs_16 {
        let (mean, std, nodes) = test_config(&calc, n_samples, || {
            symres_sparse_skip(k, layers, HDC_DIMENSION)
        });
        let vs = if mean > phi_4d { "✅>4D" } else { "" };
        println!("   {:<25} Φ = {:.4} ± {:.5} (n={:>2}) {}", name, mean, std, nodes, vs);
        results.push((name.to_string(), mean, std, nodes));
    }

    // 32 nodes (like 5D hypercube)
    let configs_32 = [
        ("K4 × 8 sparse", 4, 8),
        ("K8 × 4 sparse", 8, 4),
    ];
    for (name, k, layers) in configs_32 {
        let (mean, std, nodes) = test_config(&calc, n_samples, || {
            symres_sparse_skip(k, layers, HDC_DIMENSION)
        });
        let vs = if mean > phi_5d { "✅>5D" } else { "" };
        println!("   {:<25} Φ = {:.4} ± {:.5} (n={:>2}) {}", name, mean, std, nodes, vs);
        results.push((name.to_string(), mean, std, nodes));
    }

    // ===========================================
    // ROUND 6: The Ultimate Combinations
    // ===========================================
    println!("\n{}", "=".repeat(75));
    println!("🔬 Round 6: Ultimate Combinations\n");

    // Combine all winning elements
    let name = "Ultimate: K7×5 Sparse+Pyramid";
    let (mean, std, nodes) = test_config(&calc, n_samples, || {
        ultimate_combo(7, 5, HDC_DIMENSION)
    });
    let vs = if mean > phi_5d { "✅" } else { "" };
    println!("   {:<35} Φ = {:.4} ± {:.5} (n={:>2}) {}", name, mean, std, nodes, vs);
    results.push((name.to_string(), mean, std, nodes));

    let name = "Ultimate: K8×5 Sparse+Pyramid";
    let (mean, std, nodes) = test_config(&calc, n_samples, || {
        ultimate_combo(8, 5, HDC_DIMENSION)
    });
    let vs = if mean > phi_5d { "✅" } else { "" };
    println!("   {:<35} Φ = {:.4} ± {:.5} (n={:>2}) {}", name, mean, std, nodes, vs);
    results.push((name.to_string(), mean, std, nodes));

    let name = "Ultimate: Diamond[8-5-3-5-8] Pyramid";
    let (mean, std, nodes) = test_config(&calc, n_samples, || {
        diamond_pyramid(&[8, 5, 3, 5, 8], HDC_DIMENSION)
    });
    let vs = if mean > phi_5d { "✅" } else { "" };
    println!("   {:<35} Φ = {:.4} ± {:.5} (n={:>2}) {}", name, mean, std, nodes, vs);
    results.push((name.to_string(), mean, std, nodes));

    // Maximum efficiency: high Φ with minimal nodes
    let name = "Efficient: K6×4 Sparse+Pyramid";
    let (mean, std, nodes) = test_config(&calc, n_samples, || {
        ultimate_combo(6, 4, HDC_DIMENSION)
    });
    let vs = if mean > phi_4d { "✅>4D" } else { "" };
    println!("   {:<35} Φ = {:.4} ± {:.5} (n={:>2}) {}", name, mean, std, nodes, vs);
    results.push((name.to_string(), mean, std, nodes));

    // Try very deep but narrow
    let name = "Deep: K4×10 Sparse+AllSkip";
    let (mean, std, nodes) = test_config(&calc, n_samples, || {
        symres_sparse_skip(4, 10, HDC_DIMENSION)
    });
    let vs = if mean > phi_5d { "✅" } else { "" };
    println!("   {:<35} Φ = {:.4} ± {:.5} (n={:>2}) {}", name, mean, std, nodes, vs);
    results.push((name.to_string(), mean, std, nodes));

    // Try wide but shallow
    let name = "Wide: K10×3 Sparse+AllSkip";
    let (mean, std, nodes) = test_config(&calc, n_samples, || {
        symres_sparse_skip(10, 3, HDC_DIMENSION)
    });
    let vs = if mean > phi_4d { "✅>4D" } else { "" };
    println!("   {:<35} Φ = {:.4} ± {:.5} (n={:>2}) {}", name, mean, std, nodes, vs);
    results.push((name.to_string(), mean, std, nodes));

    // ===========================================
    // FINAL RESULTS
    // ===========================================
    println!("\n{}", "=".repeat(75));
    println!("🏆 TOP 20 CONFIGURATIONS:\n");

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("{:<4} {:<40} {:>8} {:>10} {:>6}",
             "Rank", "Configuration", "Mean Φ", "Std Dev", "Nodes");
    println!("{}", "-".repeat(75));

    for (rank, (name, mean, std, nodes)) in results.iter().take(20).enumerate() {
        let medal = match rank {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };

        let status = if *mean > phi_6d {
            "🌟>6D!"
        } else if *mean > phi_5d {
            "✅>5D"
        } else if *mean > phi_4d {
            "●>4D"
        } else {
            ""
        };

        println!("{}{:<3} {:<40} {:>8.4} {:>10.6} {:>6} {}",
                 medal, rank + 1, name, mean, std, nodes, status);
    }

    // ===========================================
    // CHAMPION ANALYSIS
    // ===========================================
    println!("\n{}", "=".repeat(75));
    println!("🏆 CHAMPION CONFIGURATION:\n");

    if let Some((name, phi, std, nodes)) = results.first() {
        println!("   Configuration: {}", name);
        println!("   Φ = {:.5} ± {:.6}", phi, std);
        println!("   Nodes: {}", nodes);
        println!();
        println!("   Performance vs Hypercubes:");
        println!("   • vs 4D (n=16): {:>+.3}% {}", ((phi - phi_4d) / phi_4d) * 100.0,
                 if *phi > phi_4d { "✅ BEATS" } else { "" });
        println!("   • vs 5D (n=32): {:>+.3}% {}", ((phi - phi_5d) / phi_5d) * 100.0,
                 if *phi > phi_5d { "✅ BEATS" } else { "" });
        println!("   • vs 6D (n=64): {:>+.3}% {}", ((phi - phi_6d) / phi_6d) * 100.0,
                 if *phi > phi_6d { "✅ BEATS" } else { "" });

        // Efficiency metric
        let efficiency = phi / (*nodes as f64);
        let eff_4d = phi_4d / 16.0;
        let eff_5d = phi_5d / 32.0;
        let eff_6d = phi_6d / 64.0;

        println!();
        println!("   Efficiency (Φ per node):");
        println!("   • Champion:  {:.5}", efficiency);
        println!("   • 4D:        {:.5} ({:+.1}%)", eff_4d, ((efficiency - eff_4d) / eff_4d) * 100.0);
        println!("   • 5D:        {:.5} ({:+.1}%)", eff_5d, ((efficiency - eff_5d) / eff_5d) * 100.0);
        println!("   • 6D:        {:.5} ({:+.1}%)", eff_6d, ((efficiency - eff_6d) / eff_6d) * 100.0);
    }

    // Find best that beats 5D
    let beats_5d: Vec<_> = results.iter()
        .filter(|(_, phi, _, _)| *phi > phi_5d)
        .collect();

    if !beats_5d.is_empty() {
        println!("\n   🌟 {} configurations beat Hypercube 5D!", beats_5d.len());
    }

    // Find best that beats 6D
    let beats_6d: Vec<_> = results.iter()
        .filter(|(_, phi, _, _)| *phi > phi_6d)
        .collect();

    if !beats_6d.is_empty() {
        println!("   🌟🌟 {} configurations beat Hypercube 6D!", beats_6d.len());
    }

    println!("\n{}", "=".repeat(75));
}

fn test_config<F>(calc: &RealPhiCalculator, n_samples: usize, generator: F) -> (f64, f64, usize)
where
    F: Fn() -> ConsciousnessTopology,
{
    let mut phi_values = Vec::with_capacity(n_samples);
    let mut node_count = 0;

    for _ in 0..n_samples {
        let topo = generator();
        node_count = topo.n_nodes;
        let phi = calc.compute(&topo.node_representations);
        phi_values.push(phi);
    }

    let mean: f64 = phi_values.iter().sum::<f64>() / n_samples as f64;
    let variance: f64 = phi_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
    (mean, variance.sqrt(), node_count)
}

/// Sparse inter-layer + all-pairs skip
fn symres_sparse_skip(k: usize, n_layers: usize, dim: usize) -> ConsciousnessTopology {
    let n_nodes = k * n_layers;
    let node_identities: Vec<RealHV> = (0..n_nodes).map(|i| RealHV::basis(i, dim)).collect();
    let mut edges = Vec::new();

    // Complete graphs within layers
    for layer in 0..n_layers {
        let offset = layer * k;
        for i in 0..k {
            for j in (i + 1)..k {
                edges.push((offset + i, offset + j));
            }
        }
    }

    // Sparse (1:1) inter-layer
    for layer in 0..(n_layers - 1) {
        let lower = layer * k;
        let upper = (layer + 1) * k;
        for i in 0..k {
            edges.push((lower + i, upper + i));
        }
    }

    // All-pairs skip connections
    for layer1 in 0..n_layers {
        for layer2 in (layer1 + 2)..n_layers {
            let offset1 = layer1 * k;
            let offset2 = layer2 * k;
            for i in 0..k {
                edges.push((offset1 + i, offset2 + i));
            }
        }
    }

    build_topology(n_nodes, dim, node_identities, edges)
}

/// Diamond hybrid with sparse inter-layer
fn diamond_sparse(layer_sizes: &[usize], dim: usize) -> ConsciousnessTopology {
    let n_layers = layer_sizes.len();
    let n_nodes: usize = layer_sizes.iter().sum();
    let node_identities: Vec<RealHV> = (0..n_nodes).map(|i| RealHV::basis(i, dim)).collect();
    let mut edges = Vec::new();

    let mut offsets = vec![0usize];
    for &size in layer_sizes.iter().take(n_layers - 1) {
        offsets.push(offsets.last().unwrap() + size);
    }

    // Complete within layers
    for (layer, &size) in layer_sizes.iter().enumerate() {
        let offset = offsets[layer];
        for i in 0..size {
            for j in (i + 1)..size {
                edges.push((offset + i, offset + j));
            }
        }
    }

    // Sparse inter-layer (1:1 where possible)
    for layer in 0..(n_layers - 1) {
        let lower_offset = offsets[layer];
        let lower_size = layer_sizes[layer];
        let upper_offset = offsets[layer + 1];
        let upper_size = layer_sizes[layer + 1];
        let min_size = lower_size.min(upper_size);
        for i in 0..min_size {
            edges.push((lower_offset + i, upper_offset + i));
        }
    }

    // All-pairs skip
    for layer1 in 0..n_layers {
        for layer2 in (layer1 + 2)..n_layers {
            let offset1 = offsets[layer1];
            let size1 = layer_sizes[layer1];
            let offset2 = offsets[layer2];
            let size2 = layer_sizes[layer2];
            let min_size = size1.min(size2);
            for i in 0..min_size {
                edges.push((offset1 + i, offset2 + i));
            }
        }
    }

    build_topology(n_nodes, dim, node_identities, edges)
}

/// Ultra-sparse: only connect every other node
fn ultra_sparse(k: usize, n_layers: usize, dim: usize) -> ConsciousnessTopology {
    let n_nodes = k * n_layers;
    let node_identities: Vec<RealHV> = (0..n_nodes).map(|i| RealHV::basis(i, dim)).collect();
    let mut edges = Vec::new();

    // Complete within layers
    for layer in 0..n_layers {
        let offset = layer * k;
        for i in 0..k {
            for j in (i + 1)..k {
                edges.push((offset + i, offset + j));
            }
        }
    }

    // Ultra-sparse inter-layer (every other node)
    for layer in 0..(n_layers - 1) {
        let lower = layer * k;
        let upper = (layer + 1) * k;
        for i in (0..k).step_by(2) {
            edges.push((lower + i, upper + i));
        }
    }

    // All-pairs skip
    for layer1 in 0..n_layers {
        for layer2 in (layer1 + 2)..n_layers {
            let offset1 = layer1 * k;
            let offset2 = layer2 * k;
            for i in 0..k {
                edges.push((offset1 + i, offset2 + i));
            }
        }
    }

    build_topology(n_nodes, dim, node_identities, edges)
}

/// Pyramid skip + sparse inter-layer
fn pyramid_sparse(k: usize, n_layers: usize, dim: usize) -> ConsciousnessTopology {
    let n_nodes = k * n_layers;
    let node_identities: Vec<RealHV> = (0..n_nodes).map(|i| RealHV::basis(i, dim)).collect();
    let mut edges = Vec::new();

    // Complete within layers
    for layer in 0..n_layers {
        let offset = layer * k;
        for i in 0..k {
            for j in (i + 1)..k {
                edges.push((offset + i, offset + j));
            }
        }
    }

    // Sparse inter-layer
    for layer in 0..(n_layers - 1) {
        let lower = layer * k;
        let upper = (layer + 1) * k;
        for i in 0..k {
            edges.push((lower + i, upper + i));
        }
    }

    // Pyramid skip: first layer connects to all others
    for layer in 2..n_layers {
        let upper = layer * k;
        for i in 0..k {
            edges.push((i, upper + i));
        }
    }

    build_topology(n_nodes, dim, node_identities, edges)
}

/// Ultimate combo: sparse + pyramid + feedback
fn ultimate_combo(k: usize, n_layers: usize, dim: usize) -> ConsciousnessTopology {
    let n_nodes = k * n_layers;
    let node_identities: Vec<RealHV> = (0..n_nodes).map(|i| RealHV::basis(i, dim)).collect();
    let mut edges = Vec::new();

    // Complete within layers
    for layer in 0..n_layers {
        let offset = layer * k;
        for i in 0..k {
            for j in (i + 1)..k {
                edges.push((offset + i, offset + j));
            }
        }
    }

    // Sparse inter-layer
    for layer in 0..(n_layers - 1) {
        let lower = layer * k;
        let upper = (layer + 1) * k;
        for i in 0..k {
            edges.push((lower + i, upper + i));
        }
    }

    // Pyramid skip (first to all)
    for layer in 2..n_layers {
        let upper = layer * k;
        for i in 0..k {
            edges.push((i, upper + i));
        }
    }

    // All-pairs skip for additional connectivity
    for layer1 in 1..n_layers {
        for layer2 in (layer1 + 2)..n_layers {
            let offset1 = layer1 * k;
            let offset2 = layer2 * k;
            for i in 0..k {
                edges.push((offset1 + i, offset2 + i));
            }
        }
    }

    build_topology(n_nodes, dim, node_identities, edges)
}

/// Diamond with pyramid skip
fn diamond_pyramid(layer_sizes: &[usize], dim: usize) -> ConsciousnessTopology {
    let n_layers = layer_sizes.len();
    let n_nodes: usize = layer_sizes.iter().sum();
    let node_identities: Vec<RealHV> = (0..n_nodes).map(|i| RealHV::basis(i, dim)).collect();
    let mut edges = Vec::new();

    let mut offsets = vec![0usize];
    for &size in layer_sizes.iter().take(n_layers - 1) {
        offsets.push(offsets.last().unwrap() + size);
    }

    // Complete within layers
    for (layer, &size) in layer_sizes.iter().enumerate() {
        let offset = offsets[layer];
        for i in 0..size {
            for j in (i + 1)..size {
                edges.push((offset + i, offset + j));
            }
        }
    }

    // Sparse inter-layer
    for layer in 0..(n_layers - 1) {
        let lower_offset = offsets[layer];
        let lower_size = layer_sizes[layer];
        let upper_offset = offsets[layer + 1];
        let upper_size = layer_sizes[layer + 1];
        let min_size = lower_size.min(upper_size);
        for i in 0..min_size {
            edges.push((lower_offset + i, upper_offset + i));
        }
    }

    // Pyramid: first layer to all others
    let first_size = layer_sizes[0];
    for layer in 2..n_layers {
        let upper_offset = offsets[layer];
        let upper_size = layer_sizes[layer];
        let min_size = first_size.min(upper_size);
        for i in 0..min_size {
            edges.push((i, upper_offset + i));
        }
    }

    build_topology(n_nodes, dim, node_identities, edges)
}

fn build_topology(
    n_nodes: usize,
    dim: usize,
    node_identities: Vec<RealHV>,
    edges: Vec<(usize, usize)>,
) -> ConsciousnessTopology {
    let mut edges: Vec<(usize, usize)> = edges
        .into_iter()
        .map(|(a, b)| (a.min(b), a.max(b)))
        .collect();
    edges.sort_unstable();
    edges.dedup();

    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for (i, j) in &edges {
        adjacency[*i].push(*j);
        adjacency[*j].push(*i);
    }

    let node_representations: Vec<RealHV> = (0..n_nodes)
        .map(|i| {
            let connections: Vec<RealHV> = adjacency[i]
                .iter()
                .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                .collect();
            if connections.is_empty() {
                node_identities[i].clone()
            } else {
                RealHV::bundle(&connections)
            }
        })
        .collect();

    ConsciousnessTopology {
        n_nodes,
        dim,
        node_representations,
        node_identities,
        topology_type: TopologyType::Modular,
        edges,
    }
}
