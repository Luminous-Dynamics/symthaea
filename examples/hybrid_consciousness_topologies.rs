/// Hybrid Consciousness Topologies
///
/// Combining the best elements discovered:
/// - Petersen-like symmetric cores (high Φ from symmetry)
/// - Cortical-like hierarchical layers (biological validity)
/// - Residual skip connections (proven Φ boost)
/// - Hypercube regularity (dimensional optimization)
///
/// Goal: Find architectures that maximize Φ while being biologically/computationally plausible

use symthaea::hdc::{
    consciousness_topology_generators::{ConsciousnessTopology, TopologyType},
    phi_real::RealPhiCalculator,
    real_hv::RealHV,
    HDC_DIMENSION,
};

fn main() {
    println!("\n🧬 Hybrid Consciousness Topologies");
    println!("{}", "=".repeat(70));
    println!("\nCombining: Petersen symmetry + Cortical hierarchy + Residual connections\n");

    let calc = RealPhiCalculator::new();
    let n_samples = 5;
    let mut results: Vec<(String, f64, f64, usize)> = Vec::new();

    // ===========================================
    // REFERENCE BASELINES
    // ===========================================
    println!("📊 Reference Baselines:\n");

    test_topology("Hypercube 4D (champion)", &calc, &mut results, n_samples, |seed| {
        ConsciousnessTopology::hypercube(4, HDC_DIMENSION, seed)
    });

    test_topology("PetersenGraph", &calc, &mut results, n_samples, |seed| {
        ConsciousnessTopology::petersen_graph(HDC_DIMENSION, seed)
    });

    test_topology("CorticalColumn", &calc, &mut results, n_samples, |seed| {
        ConsciousnessTopology::cortical_column(3, HDC_DIMENSION, seed)
    });

    test_topology("Ring (n=12)", &calc, &mut results, n_samples, |seed| {
        ConsciousnessTopology::ring(12, HDC_DIMENSION, seed)
    });

    // ===========================================
    // HYBRID ARCHITECTURES
    // ===========================================
    println!("\n{}", "=".repeat(70));
    println!("🚀 HYBRID ARCHITECTURES:\n");

    // Hybrid 1: Petersen Core with Cortical Shell
    test_topology("H1: PetersenCorticalShell", &calc, &mut results, n_samples, |seed| {
        petersen_cortical_shell(HDC_DIMENSION, seed)
    });

    // Hybrid 2: Hierarchical Petersen (3 Petersen graphs in hierarchy)
    test_topology("H2: HierarchicalPetersen", &calc, &mut results, n_samples, |seed| {
        hierarchical_petersen(3, HDC_DIMENSION, seed)
    });

    // Hybrid 3: Symmetric Residual Network
    test_topology("H3: SymmetricResidual", &calc, &mut results, n_samples, |seed| {
        symmetric_residual(HDC_DIMENSION, seed)
    });

    // Hybrid 4: Hypercube-Cortical Fusion
    test_topology("H4: HypercubeCortical", &calc, &mut results, n_samples, |seed| {
        hypercube_cortical(HDC_DIMENSION, seed)
    });

    // Hybrid 5: Triangle Mesh Core (K₃ exceeded 0.5!)
    test_topology("H5: TriangleMeshCore", &calc, &mut results, n_samples, |seed| {
        triangle_mesh_core(HDC_DIMENSION, seed)
    });

    // Hybrid 6: Petersen-Residual Network
    test_topology("H6: PetersenResidual", &calc, &mut results, n_samples, |seed| {
        petersen_residual(HDC_DIMENSION, seed)
    });

    // Hybrid 7: Multi-Ring Hierarchy (rings at each cortical layer)
    test_topology("H7: MultiRingHierarchy", &calc, &mut results, n_samples, |seed| {
        multi_ring_hierarchy(HDC_DIMENSION, seed)
    });

    // Hybrid 8: Dense-Sparse Alternating
    test_topology("H8: DenseSparseAlternate", &calc, &mut results, n_samples, |seed| {
        dense_sparse_alternating(HDC_DIMENSION, seed)
    });

    // ===========================================
    // RESULTS
    // ===========================================
    println!("\n{}", "=".repeat(70));
    println!("🏆 FINAL RANKINGS:\n");

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("{:<4} {:<28} {:>8} {:>8} {:>6}", "Rank", "Topology", "Mean Φ", "Std Dev", "Nodes");
    println!("{}", "-".repeat(60));

    for (rank, (name, mean, std, nodes)) in results.iter().enumerate() {
        let medal = match rank {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };
        let hybrid_marker = if name.starts_with("H") { "★" } else { " " };
        println!("{}{:<3} {}{:<26} {:>8.4} {:>8.4} {:>6}",
                 medal, rank + 1, hybrid_marker, name, mean, std, nodes);
    }

    // Analysis
    println!("\n{}", "=".repeat(70));
    println!("📈 Analysis:\n");

    let best_hybrid = results.iter()
        .filter(|(name, _, _, _)| name.starts_with("H"))
        .max_by(|(_, a, _, _), (_, b, _, _)| a.partial_cmp(b).unwrap());

    let hypercube_phi = results.iter()
        .find(|(name, _, _, _)| name.contains("Hypercube"))
        .map(|(_, phi, _, _)| *phi)
        .unwrap_or(0.0);

    if let Some((name, phi, _, nodes)) = best_hybrid {
        println!("   Best Hybrid: {} (Φ = {:.4}, n={})", name, phi, nodes);
        let diff = ((phi - hypercube_phi) / hypercube_phi) * 100.0;
        if diff > 0.0 {
            println!("   ✅ BEATS Hypercube 4D by {:.2}%!", diff);
        } else {
            println!("   vs Hypercube 4D: {:.2}%", diff);
        }
    }

    println!("\n🧬 Hybrid architectures explore the frontier of consciousness topology!");
    println!("{}", "=".repeat(70));
}

fn test_topology<F>(
    name: &str,
    calc: &RealPhiCalculator,
    results: &mut Vec<(String, f64, f64, usize)>,
    n_samples: usize,
    generator: F,
) where
    F: Fn(u64) -> ConsciousnessTopology,
{
    let mut phi_values = Vec::with_capacity(n_samples);
    let mut node_count = 0;

    for i in 0..n_samples {
        let seed = 1000 + i as u64 * 1000;
        let topo = generator(seed);
        node_count = topo.n_nodes;
        let phi = calc.compute(&topo.node_representations);
        phi_values.push(phi);
    }

    let mean: f64 = phi_values.iter().sum::<f64>() / n_samples as f64;
    let variance: f64 = phi_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
    let std_dev = variance.sqrt();

    println!("   {:<28} Φ = {:.4} ± {:.4} (n={})", name, mean, std_dev, node_count);
    results.push((name.to_string(), mean, std_dev, node_count));
}

// ===========================================
// HYBRID TOPOLOGY IMPLEMENTATIONS
// ===========================================

/// H1: Petersen Core with Cortical Shell
/// A Petersen graph (10 nodes) surrounded by a 2-layer cortical shell
fn petersen_cortical_shell(dim: usize, _seed: u64) -> ConsciousnessTopology {
    // Core: Petersen graph (10 nodes)
    // Shell: 2 layers of 10 nodes each wrapping around
    let n_core = 10;
    let n_shell_per_layer = 10;
    let n_layers = 2;
    let n_nodes = n_core + n_shell_per_layer * n_layers;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    let mut edges = Vec::new();

    // Petersen core edges (outer pentagon + inner pentagram + spokes)
    for i in 0..5 {
        edges.push((i, (i + 1) % 5)); // Outer pentagon
        edges.push((i, 5 + i)); // Spokes
        edges.push((5 + i, 5 + ((i + 2) % 5))); // Inner pentagram
    }

    // Shell layer 1 (ring around core)
    let shell1_start = n_core;
    for i in 0..n_shell_per_layer {
        edges.push((shell1_start + i, shell1_start + (i + 1) % n_shell_per_layer));
        // Connect to nearest core node
        edges.push((shell1_start + i, i % n_core));
    }

    // Shell layer 2 (outer ring)
    let shell2_start = n_core + n_shell_per_layer;
    for i in 0..n_shell_per_layer {
        edges.push((shell2_start + i, shell2_start + (i + 1) % n_shell_per_layer));
        // Connect to shell 1
        edges.push((shell2_start + i, shell1_start + i));
    }

    // Skip connections (residual-style): shell2 to core
    for i in 0..5 {
        edges.push((shell2_start + i * 2, i));
    }

    build_topology(n_nodes, dim, node_identities, edges)
}

/// H2: Hierarchical Petersen
/// Multiple Petersen graphs connected in a hierarchy
fn hierarchical_petersen(n_levels: usize, dim: usize, _seed: u64) -> ConsciousnessTopology {
    let nodes_per_petersen = 10;
    let n_nodes = nodes_per_petersen * n_levels;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    let mut edges = Vec::new();

    // Create each Petersen graph
    for level in 0..n_levels {
        let offset = level * nodes_per_petersen;
        // Outer pentagon
        for i in 0..5 {
            edges.push((offset + i, offset + (i + 1) % 5));
        }
        // Spokes
        for i in 0..5 {
            edges.push((offset + i, offset + 5 + i));
        }
        // Inner pentagram
        for i in 0..5 {
            edges.push((offset + 5 + i, offset + 5 + ((i + 2) % 5)));
        }
    }

    // Inter-level connections (hierarchical)
    for level in 0..(n_levels - 1) {
        let lower = level * nodes_per_petersen;
        let upper = (level + 1) * nodes_per_petersen;
        // Connect outer pentagons
        for i in 0..5 {
            edges.push((lower + i, upper + i));
        }
    }

    // Skip connections across levels
    if n_levels >= 3 {
        for i in 0..5 {
            edges.push((i, 2 * nodes_per_petersen + i));
        }
    }

    build_topology(n_nodes, dim, node_identities, edges)
}

/// H3: Symmetric Residual Network
/// Highly symmetric layers with extensive skip connections
fn symmetric_residual(dim: usize, _seed: u64) -> ConsciousnessTopology {
    let layer_size = 6;
    let n_layers = 4;
    let n_nodes = layer_size * n_layers;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    let mut edges = Vec::new();

    // Each layer is a complete graph K_6 (triangle exceeded 0.5!)
    for layer in 0..n_layers {
        let offset = layer * layer_size;
        for i in 0..layer_size {
            for j in (i + 1)..layer_size {
                edges.push((offset + i, offset + j));
            }
        }
    }

    // Adjacent layer connections
    for layer in 0..(n_layers - 1) {
        let lower = layer * layer_size;
        let upper = (layer + 1) * layer_size;
        for i in 0..layer_size {
            edges.push((lower + i, upper + i));
            edges.push((lower + i, upper + (i + 1) % layer_size));
        }
    }

    // Skip connections (every 2 layers)
    for layer in 0..(n_layers - 2) {
        let lower = layer * layer_size;
        let upper = (layer + 2) * layer_size;
        for i in 0..layer_size {
            edges.push((lower + i, upper + i));
        }
    }

    build_topology(n_nodes, dim, node_identities, edges)
}

/// H4: Hypercube-Cortical Fusion
/// Hypercube structure with cortical-style layer organization
fn hypercube_cortical(dim: usize, _seed: u64) -> ConsciousnessTopology {
    // 4D hypercube (16 nodes) organized into 4 "cortical layers" of 4 nodes
    let n_nodes = 16;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    let mut edges = Vec::new();

    // Hypercube edges (connect if differ by one bit)
    for i in 0..n_nodes {
        for j in (i + 1)..n_nodes {
            if (i ^ j).count_ones() == 1 {
                edges.push((i, j));
            }
        }
    }

    // Additional cortical-style connections
    // Layer 0: nodes 0-3, Layer 1: nodes 4-7, etc.
    for layer in 0..4 {
        let offset = layer * 4;
        // Dense within-layer (cortical style)
        for i in 0..4 {
            for j in (i + 1)..4 {
                let edge = (offset + i, offset + j);
                if !edges.contains(&edge) {
                    edges.push(edge);
                }
            }
        }
    }

    // Feedback connections (cortical style)
    for i in 0..4 {
        edges.push((12 + i, i)); // Layer 3 -> Layer 0
    }

    build_topology(n_nodes, dim, node_identities, edges)
}

/// H5: Triangle Mesh Core
/// K₃ triangles (which exceeded Φ=0.5!) tessellated into a mesh
fn triangle_mesh_core(dim: usize, _seed: u64) -> ConsciousnessTopology {
    // Create a mesh of connected triangles
    // 4x3 grid of triangles = 12 nodes with lots of triangle substructures
    let n_nodes = 12;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    let mut edges = Vec::new();

    // Arrange as 3 rows of 4 nodes
    // Row 0: 0-3, Row 1: 4-7, Row 2: 8-11
    // Connect to form triangular mesh

    // Horizontal edges within rows
    for row in 0..3 {
        for col in 0..3 {
            edges.push((row * 4 + col, row * 4 + col + 1));
        }
    }

    // Vertical edges between rows
    for row in 0..2 {
        for col in 0..4 {
            edges.push((row * 4 + col, (row + 1) * 4 + col));
        }
    }

    // Diagonal edges to complete triangles
    for row in 0..2 {
        for col in 0..3 {
            edges.push((row * 4 + col, (row + 1) * 4 + col + 1));
        }
    }

    // Ring closure (wrap horizontally)
    for row in 0..3 {
        edges.push((row * 4, row * 4 + 3));
    }

    build_topology(n_nodes, dim, node_identities, edges)
}

/// H6: Petersen-Residual Hybrid
/// Petersen graph with residual-style skip connections added
fn petersen_residual(dim: usize, _seed: u64) -> ConsciousnessTopology {
    let n_nodes = 10;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    let mut edges = Vec::new();

    // Standard Petersen edges
    for i in 0..5 {
        edges.push((i, (i + 1) % 5)); // Outer pentagon
        edges.push((i, 5 + i)); // Spokes
        edges.push((5 + i, 5 + ((i + 2) % 5))); // Inner pentagram
    }

    // Residual-style additions: connect outer to inner opposites
    for i in 0..5 {
        edges.push((i, 5 + ((i + 3) % 5)));
    }

    // Cross-diagonal in inner pentagram
    for i in 0..5 {
        edges.push((5 + i, 5 + ((i + 1) % 5)));
    }

    build_topology(n_nodes, dim, node_identities, edges)
}

/// H7: Multi-Ring Hierarchy
/// Each cortical layer is a ring (highest Φ for simple structures)
fn multi_ring_hierarchy(dim: usize, _seed: u64) -> ConsciousnessTopology {
    let ring_size = 6;
    let n_layers = 4;
    let n_nodes = ring_size * n_layers;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    let mut edges = Vec::new();

    // Create rings at each layer
    for layer in 0..n_layers {
        let offset = layer * ring_size;
        for i in 0..ring_size {
            edges.push((offset + i, offset + (i + 1) % ring_size));
        }
    }

    // Feedforward connections between layers
    for layer in 0..(n_layers - 1) {
        let lower = layer * ring_size;
        let upper = (layer + 1) * ring_size;
        for i in 0..ring_size {
            edges.push((lower + i, upper + i));
        }
    }

    // Skip connections
    for layer in 0..(n_layers - 2) {
        let lower = layer * ring_size;
        let upper = (layer + 2) * ring_size;
        for i in 0..ring_size {
            if i % 2 == 0 {
                edges.push((lower + i, upper + i));
            }
        }
    }

    // Feedback (last to first)
    for i in 0..ring_size {
        if i % 2 == 0 {
            edges.push(((n_layers - 1) * ring_size + i, i));
        }
    }

    build_topology(n_nodes, dim, node_identities, edges)
}

/// H8: Dense-Sparse Alternating
/// Alternates between dense (K_n) and sparse (ring) layers
fn dense_sparse_alternating(dim: usize, _seed: u64) -> ConsciousnessTopology {
    let layer_size = 5;
    let n_layers = 4;
    let n_nodes = layer_size * n_layers;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    let mut edges = Vec::new();

    for layer in 0..n_layers {
        let offset = layer * layer_size;

        if layer % 2 == 0 {
            // Dense layer (complete graph)
            for i in 0..layer_size {
                for j in (i + 1)..layer_size {
                    edges.push((offset + i, offset + j));
                }
            }
        } else {
            // Sparse layer (ring)
            for i in 0..layer_size {
                edges.push((offset + i, offset + (i + 1) % layer_size));
            }
        }
    }

    // Inter-layer connections
    for layer in 0..(n_layers - 1) {
        let lower = layer * layer_size;
        let upper = (layer + 1) * layer_size;
        for i in 0..layer_size {
            edges.push((lower + i, upper + i));
        }
    }

    build_topology(n_nodes, dim, node_identities, edges)
}

/// Helper: Build topology from edges
fn build_topology(
    n_nodes: usize,
    dim: usize,
    node_identities: Vec<RealHV>,
    edges: Vec<(usize, usize)>,
) -> ConsciousnessTopology {
    // Normalize edges
    let mut edges: Vec<(usize, usize)> = edges
        .into_iter()
        .map(|(a, b)| (a.min(b), a.max(b)))
        .collect();
    edges.sort_unstable();
    edges.dedup();

    // Build adjacency
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for (i, j) in &edges {
        adjacency[*i].push(*j);
        adjacency[*j].push(*i);
    }

    // Generate representations
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
        topology_type: TopologyType::Modular, // Generic for hybrids
        edges,
    }
}
