/// Optimize SymRes Architecture
///
/// Systematic exploration of Symmetric Residual parameters:
/// - Layer size (K₃, K₄, K₅, K₆)
/// - Number of layers (3-8)
/// - Skip connection patterns
/// - Inter-layer connection density
/// - Feedback loops
///
/// Goal: Find the optimal configuration that maximizes Φ

use symthaea::hdc::{
    consciousness_topology_generators::{ConsciousnessTopology, TopologyType},
    phi_real::RealPhiCalculator,
    real_hv::RealHV,
    HDC_DIMENSION,
};

fn main() {
    println!("\n⚡ SymRes Architecture Optimization");
    println!("{}", "=".repeat(75));

    let calc = RealPhiCalculator::new();
    let n_samples = 5;
    let mut all_results: Vec<(String, f64, f64, usize, String)> = Vec::new();

    // ===========================================
    // BASELINE
    // ===========================================
    println!("\n🎯 BASELINES:\n");

    let hyper4d = ConsciousnessTopology::hypercube(4, HDC_DIMENSION, 42);
    let phi_4d = calc.compute(&hyper4d.node_representations);
    println!("   Hypercube 4D (target):     Φ = {:.4} (n=16)", phi_4d);

    let hyper5d = ConsciousnessTopology::hypercube(5, HDC_DIMENSION, 42);
    let phi_5d = calc.compute(&hyper5d.node_representations);
    println!("   Hypercube 5D:              Φ = {:.4} (n=32)", phi_5d);

    let hyper6d = ConsciousnessTopology::hypercube(6, HDC_DIMENSION, 42);
    let phi_6d = calc.compute(&hyper6d.node_representations);
    println!("   Hypercube 6D:              Φ = {:.4} (n=64)", phi_6d);

    // ===========================================
    // EXPERIMENT 1: Layer Size Sweep
    // ===========================================
    println!("\n{}", "=".repeat(75));
    println!("📊 Experiment 1: Layer Size (K_n) with 5 layers, dense skip\n");

    for k in 3..=7 {
        let name = format!("K{} × 5 layers", k);
        let (mean, std, nodes) = test_config(&calc, n_samples, || {
            symres_configurable(k, 5, SkipPattern::AllPairs, InterLayer::Full, false, HDC_DIMENSION)
        });
        println!("   {:<25} Φ = {:.4} ± {:.4} (n={})", name, mean, std, nodes);
        all_results.push((name, mean, std, nodes, "LayerSize".to_string()));
    }

    // ===========================================
    // EXPERIMENT 2: Layer Count Sweep
    // ===========================================
    println!("\n{}", "=".repeat(75));
    println!("📊 Experiment 2: Layer Count with K₄, dense skip\n");

    for layers in 3..=8 {
        let name = format!("K4 × {} layers", layers);
        let (mean, std, nodes) = test_config(&calc, n_samples, || {
            symres_configurable(4, layers, SkipPattern::AllPairs, InterLayer::Full, false, HDC_DIMENSION)
        });
        println!("   {:<25} Φ = {:.4} ± {:.4} (n={})", name, mean, std, nodes);
        all_results.push((name, mean, std, nodes, "LayerCount".to_string()));
    }

    // ===========================================
    // EXPERIMENT 3: Skip Connection Patterns
    // ===========================================
    println!("\n{}", "=".repeat(75));
    println!("📊 Experiment 3: Skip Patterns (K₄ × 5 layers)\n");

    let skip_patterns = [
        (SkipPattern::None, "No skip"),
        (SkipPattern::Every2, "Skip every 2"),
        (SkipPattern::Every3, "Skip every 3"),
        (SkipPattern::AllPairs, "All pairs (dense)"),
        (SkipPattern::FirstLast, "First↔Last only"),
        (SkipPattern::Pyramid, "Pyramid (1→all)"),
    ];

    for (pattern, pattern_name) in skip_patterns {
        let name = format!("{}", pattern_name);
        let (mean, std, nodes) = test_config(&calc, n_samples, || {
            symres_configurable(4, 5, pattern, InterLayer::Full, false, HDC_DIMENSION)
        });
        println!("   {:<25} Φ = {:.4} ± {:.4} (n={})", name, mean, std, nodes);
        all_results.push((name, mean, std, nodes, "SkipPattern".to_string()));
    }

    // ===========================================
    // EXPERIMENT 4: Inter-Layer Connection Density
    // ===========================================
    println!("\n{}", "=".repeat(75));
    println!("📊 Experiment 4: Inter-Layer Density (K₄ × 5, all-pairs skip)\n");

    let inter_patterns = [
        (InterLayer::Sparse, "Sparse (1:1)"),
        (InterLayer::Medium, "Medium (1:2)"),
        (InterLayer::Full, "Full (all:all)"),
        (InterLayer::Ring, "Ring pattern"),
    ];

    for (inter, inter_name) in inter_patterns {
        let name = format!("{}", inter_name);
        let (mean, std, nodes) = test_config(&calc, n_samples, || {
            symres_configurable(4, 5, SkipPattern::AllPairs, inter, false, HDC_DIMENSION)
        });
        println!("   {:<25} Φ = {:.4} ± {:.4} (n={})", name, mean, std, nodes);
        all_results.push((name, mean, std, nodes, "InterLayer".to_string()));
    }

    // ===========================================
    // EXPERIMENT 5: Feedback Loops
    // ===========================================
    println!("\n{}", "=".repeat(75));
    println!("📊 Experiment 5: Feedback Loops (K₄ × 5, all-pairs skip)\n");

    let name = "Without feedback";
    let (mean, std, nodes) = test_config(&calc, n_samples, || {
        symres_configurable(4, 5, SkipPattern::AllPairs, InterLayer::Full, false, HDC_DIMENSION)
    });
    println!("   {:<25} Φ = {:.4} ± {:.4} (n={})", name, mean, std, nodes);
    all_results.push((name.to_string(), mean, std, nodes, "Feedback".to_string()));

    let name = "With feedback (last→first)";
    let (mean, std, nodes) = test_config(&calc, n_samples, || {
        symres_configurable(4, 5, SkipPattern::AllPairs, InterLayer::Full, true, HDC_DIMENSION)
    });
    println!("   {:<25} Φ = {:.4} ± {:.4} (n={})", name, mean, std, nodes);
    all_results.push((name.to_string(), mean, std, nodes, "Feedback".to_string()));

    // ===========================================
    // EXPERIMENT 6: Hybrid Layer Sizes
    // ===========================================
    println!("\n{}", "=".repeat(75));
    println!("📊 Experiment 6: Hybrid Layer Sizes\n");

    let hybrid_configs = [
        (vec![3, 4, 5, 4, 3], "Hourglass [3-4-5-4-3]"),
        (vec![5, 4, 3, 4, 5], "Diamond [5-4-3-4-5]"),
        (vec![3, 3, 4, 4, 5, 5], "Growing [3-3-4-4-5-5]"),
        (vec![5, 5, 4, 4, 3, 3], "Shrinking [5-5-4-4-3-3]"),
        (vec![4, 3, 4, 3, 4], "Alternating [4-3-4-3-4]"),
        (vec![3, 5, 3, 5, 3], "Pulse [3-5-3-5-3]"),
    ];

    for (layer_sizes, config_name) in hybrid_configs {
        let name = format!("{}", config_name);
        let (mean, std, nodes) = test_config(&calc, n_samples, || {
            symres_hybrid(&layer_sizes, SkipPattern::AllPairs, HDC_DIMENSION)
        });
        println!("   {:<25} Φ = {:.4} ± {:.4} (n={})", name, mean, std, nodes);
        all_results.push((name, mean, std, nodes, "Hybrid".to_string()));
    }

    // ===========================================
    // EXPERIMENT 7: Extreme Configurations
    // ===========================================
    println!("\n{}", "=".repeat(75));
    println!("📊 Experiment 7: Extreme Configurations\n");

    // Many small layers
    let name = "K3 × 10 layers (30 nodes)";
    let (mean, std, nodes) = test_config(&calc, n_samples, || {
        symres_configurable(3, 10, SkipPattern::AllPairs, InterLayer::Full, true, HDC_DIMENSION)
    });
    println!("   {:<25} Φ = {:.4} ± {:.4} (n={})", name, mean, std, nodes);
    all_results.push((name.to_string(), mean, std, nodes, "Extreme".to_string()));

    // Few large layers
    let name = "K8 × 3 layers (24 nodes)";
    let (mean, std, nodes) = test_config(&calc, n_samples, || {
        symres_configurable(8, 3, SkipPattern::AllPairs, InterLayer::Full, true, HDC_DIMENSION)
    });
    println!("   {:<25} Φ = {:.4} ± {:.4} (n={})", name, mean, std, nodes);
    all_results.push((name.to_string(), mean, std, nodes, "Extreme".to_string()));

    // Maximum density at small scale
    let name = "K5 × 4 layers maximally connected";
    let (mean, std, nodes) = test_config(&calc, n_samples, || {
        symres_maximum_density(5, 4, HDC_DIMENSION)
    });
    println!("   {:<25} Φ = {:.4} ± {:.4} (n={})", name, mean, std, nodes);
    all_results.push((name.to_string(), mean, std, nodes, "Extreme".to_string()));

    // ===========================================
    // FINAL RANKINGS
    // ===========================================
    println!("\n{}", "=".repeat(75));
    println!("🏆 TOP 15 CONFIGURATIONS:\n");

    all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("{:<4} {:<30} {:>8} {:>8} {:>6} {:<12}",
             "Rank", "Configuration", "Mean Φ", "Std Dev", "Nodes", "Category");
    println!("{}", "-".repeat(75));

    for (rank, (name, mean, std, nodes, category)) in all_results.iter().take(15).enumerate() {
        let medal = match rank {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };

        let vs_4d = ((*mean - phi_4d) / phi_4d) * 100.0;
        let beats = if *mean > phi_4d { "✅" } else { "" };

        println!("{}{:<3} {:<30} {:>8.4} {:>8.4} {:>6} {:<12} {:>+.2}% {}",
                 medal, rank + 1, name, mean, std, nodes, category, vs_4d, beats);
    }

    // ===========================================
    // OPTIMAL CONFIGURATION
    // ===========================================
    println!("\n{}", "=".repeat(75));
    println!("🎯 OPTIMAL SYMRES CONFIGURATION:\n");

    if let Some((name, phi, _, nodes, _)) = all_results.first() {
        println!("   Winner: {}", name);
        println!("   Φ = {:.4} (n={})", phi, nodes);
        println!("   vs Hypercube 4D: {:+.3}%", ((phi - phi_4d) / phi_4d) * 100.0);
        println!("   vs Hypercube 5D: {:+.3}%", ((phi - phi_5d) / phi_5d) * 100.0);
        println!("   vs Hypercube 6D: {:+.3}%", ((phi - phi_6d) / phi_6d) * 100.0);
    }

    // Key findings
    println!("\n📈 Key Findings:");

    // Best layer size
    let best_layer_size = all_results.iter()
        .filter(|(_, _, _, _, cat)| cat == "LayerSize")
        .max_by(|(_, a, _, _, _), (_, b, _, _, _)| a.partial_cmp(b).unwrap());
    if let Some((name, phi, _, _, _)) = best_layer_size {
        println!("   • Best layer size: {} (Φ = {:.4})", name, phi);
    }

    // Best layer count
    let best_layer_count = all_results.iter()
        .filter(|(_, _, _, _, cat)| cat == "LayerCount")
        .max_by(|(_, a, _, _, _), (_, b, _, _, _)| a.partial_cmp(b).unwrap());
    if let Some((name, phi, _, _, _)) = best_layer_count {
        println!("   • Best layer count: {} (Φ = {:.4})", name, phi);
    }

    // Best skip pattern
    let best_skip = all_results.iter()
        .filter(|(_, _, _, _, cat)| cat == "SkipPattern")
        .max_by(|(_, a, _, _, _), (_, b, _, _, _)| a.partial_cmp(b).unwrap());
    if let Some((name, phi, _, _, _)) = best_skip {
        println!("   • Best skip pattern: {} (Φ = {:.4})", name, phi);
    }

    println!("\n{}", "=".repeat(75));
}

// ===========================================
// CONFIGURATION TYPES
// ===========================================

#[derive(Clone, Copy)]
enum SkipPattern {
    None,
    Every2,
    Every3,
    AllPairs,
    FirstLast,
    Pyramid,
}

#[derive(Clone, Copy)]
enum InterLayer {
    Sparse,  // 1:1 connections
    Medium,  // 1:2 connections
    Full,    // all:all connections
    Ring,    // ring pattern between layers
}

// ===========================================
// TOPOLOGY BUILDERS
// ===========================================

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

fn symres_configurable(
    k: usize,
    n_layers: usize,
    skip: SkipPattern,
    inter: InterLayer,
    feedback: bool,
    dim: usize,
) -> ConsciousnessTopology {
    let n_nodes = k * n_layers;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    let mut edges = Vec::new();

    // Intra-layer: each layer is K_k
    for layer in 0..n_layers {
        let offset = layer * k;
        for i in 0..k {
            for j in (i + 1)..k {
                edges.push((offset + i, offset + j));
            }
        }
    }

    // Inter-layer connections
    for layer in 0..(n_layers - 1) {
        let lower = layer * k;
        let upper = (layer + 1) * k;

        match inter {
            InterLayer::Sparse => {
                for i in 0..k {
                    edges.push((lower + i, upper + i));
                }
            }
            InterLayer::Medium => {
                for i in 0..k {
                    edges.push((lower + i, upper + i));
                    edges.push((lower + i, upper + (i + 1) % k));
                }
            }
            InterLayer::Full => {
                for i in 0..k {
                    for j in 0..k {
                        edges.push((lower + i, upper + j));
                    }
                }
            }
            InterLayer::Ring => {
                for i in 0..k {
                    edges.push((lower + i, upper + i));
                    edges.push((lower + i, upper + (i + 1) % k));
                    edges.push((lower + (i + 1) % k, upper + i));
                }
            }
        }
    }

    // Skip connections
    match skip {
        SkipPattern::None => {}
        SkipPattern::Every2 => {
            for layer in 0..n_layers.saturating_sub(2) {
                let lower = layer * k;
                let upper = (layer + 2) * k;
                for i in 0..k {
                    edges.push((lower + i, upper + i));
                }
            }
        }
        SkipPattern::Every3 => {
            for layer in 0..n_layers.saturating_sub(3) {
                let lower = layer * k;
                let upper = (layer + 3) * k;
                for i in 0..k {
                    edges.push((lower + i, upper + i));
                }
            }
        }
        SkipPattern::AllPairs => {
            for layer1 in 0..n_layers {
                for layer2 in (layer1 + 2)..n_layers {
                    let offset1 = layer1 * k;
                    let offset2 = layer2 * k;
                    for i in 0..k {
                        edges.push((offset1 + i, offset2 + i));
                    }
                }
            }
        }
        SkipPattern::FirstLast => {
            if n_layers >= 2 {
                let first = 0;
                let last = (n_layers - 1) * k;
                for i in 0..k {
                    edges.push((first + i, last + i));
                }
            }
        }
        SkipPattern::Pyramid => {
            // First layer connects to all others
            for layer in 2..n_layers {
                let upper = layer * k;
                for i in 0..k {
                    edges.push((i, upper + i));
                }
            }
        }
    }

    // Feedback (last → first)
    if feedback && n_layers >= 2 {
        let last = (n_layers - 1) * k;
        for i in 0..k {
            edges.push((last + i, i));
        }
    }

    build_topology(n_nodes, dim, node_identities, edges)
}

fn symres_hybrid(layer_sizes: &[usize], skip: SkipPattern, dim: usize) -> ConsciousnessTopology {
    let n_layers = layer_sizes.len();
    let n_nodes: usize = layer_sizes.iter().sum();

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    let mut edges = Vec::new();

    // Calculate layer offsets
    let mut offsets = vec![0usize];
    for &size in layer_sizes.iter().take(n_layers - 1) {
        offsets.push(offsets.last().unwrap() + size);
    }

    // Intra-layer: each layer is complete graph
    for (layer, &size) in layer_sizes.iter().enumerate() {
        let offset = offsets[layer];
        for i in 0..size {
            for j in (i + 1)..size {
                edges.push((offset + i, offset + j));
            }
        }
    }

    // Inter-layer connections (full)
    for layer in 0..(n_layers - 1) {
        let lower_offset = offsets[layer];
        let lower_size = layer_sizes[layer];
        let upper_offset = offsets[layer + 1];
        let upper_size = layer_sizes[layer + 1];

        for i in 0..lower_size {
            for j in 0..upper_size {
                edges.push((lower_offset + i, upper_offset + j));
            }
        }
    }

    // Skip connections
    if matches!(skip, SkipPattern::AllPairs) {
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
    }

    build_topology(n_nodes, dim, node_identities, edges)
}

fn symres_maximum_density(k: usize, n_layers: usize, dim: usize) -> ConsciousnessTopology {
    let n_nodes = k * n_layers;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    let mut edges = Vec::new();

    // Intra-layer: complete graphs
    for layer in 0..n_layers {
        let offset = layer * k;
        for i in 0..k {
            for j in (i + 1)..k {
                edges.push((offset + i, offset + j));
            }
        }
    }

    // ALL possible inter-layer connections (maximum density)
    for layer1 in 0..n_layers {
        for layer2 in (layer1 + 1)..n_layers {
            let offset1 = layer1 * k;
            let offset2 = layer2 * k;
            for i in 0..k {
                for j in 0..k {
                    edges.push((offset1 + i, offset2 + j));
                }
            }
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
