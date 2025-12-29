/// Hypercube-SymRes Hybrid: Combining Two Pathways to Φ = 0.5
///
/// Goal: Beat Hypercube 6D (Φ = 0.4989) by combining:
/// 1. Hypercube topology (dimensional regularity)
/// 2. SymRes principles (dense local + sparse long-range)
///
/// Approach: Arrange complete subgraphs in hypercube patterns

use symthaea::hdc::{
    consciousness_topology_generators::ConsciousnessTopology,
    phi_real::RealPhiCalculator,
    real_hv::RealHV,
    HDC_DIMENSION,
};
use std::collections::HashMap;

fn main() {
    let calc = RealPhiCalculator::new();
    let n_samples = 5;

    println!("\n🧬 HYPERCUBE-SYMRES HYBRID EXPLORATION");
    println!("{}", "=".repeat(75));
    println!("\n🎯 TARGETS:");
    println!("   Hypercube 5D: Φ = 0.4987 (n=32)");
    println!("   Hypercube 6D: Φ = 0.4989 (n=64) ← GOAL TO BEAT");
    println!("   Best SymRes:  Φ = 0.4987 (K7×6, n=42)");

    let mut all_results: Vec<(String, f64, f64, usize)> = Vec::new();

    // ==========================================================
    // PART 1: Hypercube of Complete Subgraphs
    // Replace each vertex of a hypercube with a Kn complete graph
    // ==========================================================
    println!("\n{}", "=".repeat(75));
    println!("🔬 Part 1: Hypercube of Complete Subgraphs\n");
    println!("   Each hypercube vertex → complete graph Kn");
    println!("   Hypercube edges → sparse inter-cluster connections\n");

    let configs_p1 = [
        (3, 3, "3D-Cube of K3"),
        (3, 4, "3D-Cube of K4"),
        (3, 5, "3D-Cube of K5"),
        (4, 3, "4D-Tesseract of K3"),
        (4, 4, "4D-Tesseract of K4"),
        (4, 5, "4D-Tesseract of K5"),
        (5, 3, "5D-Penteract of K3"),
        (5, 4, "5D-Penteract of K4"),
    ];

    for (dim, k, name) in configs_p1 {
        let (mean, std, n_nodes) = run_samples(|| {
            hypercube_of_complete_graphs(dim, k, HDC_DIMENSION, rand::random())
        }, &calc, n_samples);

        let indicator = if mean > 0.4989 { " ✅ BEATS 6D!" }
                       else if mean > 0.4987 { " ●>5D" }
                       else if mean > 0.4977 { " ●>4D" }
                       else { "" };

        println!("   {:25} Φ = {:.4} ± {:.5} (n={}){}",
                 name, mean, std, n_nodes, indicator);
        all_results.push((name.to_string(), mean, std, n_nodes));
    }

    // ==========================================================
    // PART 2: SymRes Layers in Hypercube Arrangement
    // Instead of linear layer stack, arrange layers as hypercube vertices
    // ==========================================================
    println!("\n{}", "=".repeat(75));
    println!("🔬 Part 2: SymRes Layers as Hypercube Vertices\n");
    println!("   Each layer is Kn, layers connected in hypercube pattern\n");

    let configs_p2 = [
        (3, 5, "8 K5-layers (3D cube)"),
        (3, 6, "8 K6-layers (3D cube)"),
        (3, 7, "8 K7-layers (3D cube)"),
        (4, 4, "16 K4-layers (4D tesseract)"),
        (4, 5, "16 K5-layers (4D tesseract)"),
        (4, 6, "16 K6-layers (4D tesseract)"),
        (5, 4, "32 K4-layers (5D penteract)"),
        (5, 5, "32 K5-layers (5D penteract)"),
    ];

    for (hyper_dim, k, name) in configs_p2 {
        let (mean, std, n_nodes) = run_samples(|| {
            symres_hypercube_arrangement(hyper_dim, k, HDC_DIMENSION, rand::random())
        }, &calc, n_samples);

        let indicator = if mean > 0.4989 { " ✅ BEATS 6D!" }
                       else if mean > 0.4987 { " ●>5D" }
                       else if mean > 0.4977 { " ●>4D" }
                       else { "" };

        println!("   {:30} Φ = {:.4} ± {:.5} (n={}){}",
                 name, mean, std, n_nodes, indicator);
        all_results.push((name.to_string(), mean, std, n_nodes));
    }

    // ==========================================================
    // PART 3: Enhanced Hypercube with Skip Connections
    // Regular hypercube + additional diagonal/skip connections
    // ==========================================================
    println!("\n{}", "=".repeat(75));
    println!("🔬 Part 3: Enhanced Hypercube with Skip Connections\n");

    let configs_p3 = [
        (4, 1, "4D + 1-skip diagonals"),
        (4, 2, "4D + 2-skip diagonals"),
        (5, 1, "5D + 1-skip diagonals"),
        (5, 2, "5D + 2-skip diagonals"),
        (6, 1, "6D + 1-skip diagonals"),
        (6, 2, "6D + 2-skip diagonals"),
    ];

    for (dim, skip_level, name) in configs_p3 {
        let (mean, std, n_nodes) = run_samples(|| {
            hypercube_with_skips(dim, skip_level, HDC_DIMENSION, rand::random())
        }, &calc, n_samples);

        let indicator = if mean > 0.4989 { " ✅ BEATS 6D!" }
                       else if mean > 0.4987 { " ●>5D" }
                       else if mean > 0.4977 { " ●>4D" }
                       else { "" };

        println!("   {:25} Φ = {:.4} ± {:.5} (n={}){}",
                 name, mean, std, n_nodes, indicator);
        all_results.push((name.to_string(), mean, std, n_nodes));
    }

    // ==========================================================
    // PART 4: Recursive Hypercube (Hypercube of Hypercubes)
    // ==========================================================
    println!("\n{}", "=".repeat(75));
    println!("🔬 Part 4: Recursive Hypercube Structures\n");

    let configs_p4 = [
        (2, 2, "2D of 2D (4×4 = 16 nodes)"),
        (2, 3, "2D of 3D (4×8 = 32 nodes)"),
        (3, 2, "3D of 2D (8×4 = 32 nodes)"),
        (2, 4, "2D of 4D (4×16 = 64 nodes)"),
        (3, 3, "3D of 3D (8×8 = 64 nodes)"),
    ];

    for (outer_dim, inner_dim, name) in configs_p4 {
        let (mean, std, n_nodes) = run_samples(|| {
            recursive_hypercube(outer_dim, inner_dim, HDC_DIMENSION, rand::random())
        }, &calc, n_samples);

        let indicator = if mean > 0.4989 { " ✅ BEATS 6D!" }
                       else if mean > 0.4987 { " ●>5D" }
                       else if mean > 0.4977 { " ●>4D" }
                       else { "" };

        println!("   {:30} Φ = {:.4} ± {:.5} (n={}){}",
                 name, mean, std, n_nodes, indicator);
        all_results.push((name.to_string(), mean, std, n_nodes));
    }

    // ==========================================================
    // PART 5: Mixed-Dimension Structures
    // ==========================================================
    println!("\n{}", "=".repeat(75));
    println!("🔬 Part 5: Mixed-Dimension Structures\n");

    let configs_p5 = [
        "Ring of K7 (7 clusters)",
        "Torus of K5 (3×3 = 9 clusters)",
        "Petersen of K4 (10 clusters)",
        "K7 layers + feedback loops",
        "Hypercube 4D + K3 decorations",
    ];

    for name in configs_p5 {
        let (mean, std, n_nodes) = match name {
            "Ring of K7 (7 clusters)" => run_samples(|| {
                ring_of_complete_graphs(7, 7, HDC_DIMENSION, rand::random())
            }, &calc, n_samples),
            "Torus of K5 (3×3 = 9 clusters)" => run_samples(|| {
                torus_of_complete_graphs(3, 5, HDC_DIMENSION, rand::random())
            }, &calc, n_samples),
            "Petersen of K4 (10 clusters)" => run_samples(|| {
                petersen_of_complete_graphs(4, HDC_DIMENSION, rand::random())
            }, &calc, n_samples),
            "K7 layers + feedback loops" => run_samples(|| {
                symres_with_feedback(7, 6, HDC_DIMENSION, rand::random())
            }, &calc, n_samples),
            "Hypercube 4D + K3 decorations" => run_samples(|| {
                decorated_hypercube(4, 3, HDC_DIMENSION, rand::random())
            }, &calc, n_samples),
            _ => (0.0, 0.0, 0),
        };

        let indicator = if mean > 0.4989 { " ✅ BEATS 6D!" }
                       else if mean > 0.4987 { " ●>5D" }
                       else if mean > 0.4977 { " ●>4D" }
                       else { "" };

        println!("   {:30} Φ = {:.4} ± {:.5} (n={}){}",
                 name, mean, std, n_nodes, indicator);
        all_results.push((name.to_string(), mean, std, n_nodes));
    }

    // ==========================================================
    // RESULTS SUMMARY
    // ==========================================================
    println!("\n{}", "=".repeat(75));
    println!("🏆 TOP 15 CONFIGURATIONS:\n");

    all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("{:4} {:35} {:>8} {:>10} {:>6}",
             "Rank", "Configuration", "Mean Φ", "Std Dev", "Nodes");
    println!("{}", "-".repeat(75));

    for (i, (name, mean, std, nodes)) in all_results.iter().take(15).enumerate() {
        let medal = match i {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };
        let indicator = if *mean > 0.4989 { "✅>6D" }
                       else if *mean > 0.4987 { "●>5D" }
                       else if *mean > 0.4977 { "●>4D" }
                       else { "" };

        println!("{}{:2}  {:35} {:>8.4} {:>10.6} {:>6} {}",
                 medal, i+1, name, mean, std, nodes, indicator);
    }

    // ==========================================================
    // CHAMPION ANALYSIS
    // ==========================================================
    if let Some((name, mean, _, nodes)) = all_results.first() {
        println!("\n{}", "=".repeat(75));
        println!("🏆 CHAMPION HYBRID:\n");
        println!("   Configuration: {}", name);
        println!("   Φ = {:.5}", mean);
        println!("   Nodes: {}", nodes);
        println!("\n   Performance vs Pure Hypercubes:");
        println!("   • vs 4D (n=16): {:+.3}%", (mean - 0.4977) / 0.4977 * 100.0);
        println!("   • vs 5D (n=32): {:+.3}%", (mean - 0.4987) / 0.4987 * 100.0);
        println!("   • vs 6D (n=64): {:+.3}%", (mean - 0.4989) / 0.4989 * 100.0);

        if *mean > 0.4989 {
            println!("\n   🎉 NEW Φ CHAMPION! Beats Hypercube 6D!");
        }
    }

    println!("\n{}", "=".repeat(75));
}

/// Run multiple samples and compute statistics
fn run_samples<F>(mut generator: F, calc: &RealPhiCalculator, n: usize) -> (f64, f64, usize)
where
    F: FnMut() -> ConsciousnessTopology,
{
    let mut phis: Vec<f64> = Vec::new();
    let mut n_nodes = 0;

    for _ in 0..n {
        let topo = generator();
        n_nodes = topo.node_representations.len();
        let phi = calc.compute(&topo.node_representations);
        phis.push(phi);
    }

    let mean = phis.iter().sum::<f64>() / phis.len() as f64;
    let variance = phis.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / phis.len() as f64;
    let std = variance.sqrt();

    (mean, std, n_nodes)
}

// ==========================================================
// HYBRID TOPOLOGY GENERATORS
// ==========================================================

/// Create a hypercube where each vertex is a complete graph Kk
fn hypercube_of_complete_graphs(dim: usize, k: usize, hd_dim: usize, seed: u64) -> ConsciousnessTopology {
    use symthaea::hdc::consciousness_topology_generators::TopologyType;

    let n_hypercube_vertices = 1 << dim; // 2^dim
    let n_nodes = n_hypercube_vertices * k;

    // Create node identities
    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| {
            let base = RealHV::basis(i, hd_dim);
            let noise = RealHV::random(hd_dim, seed + i as u64 * 1000).scale(0.05);
            base.add(&noise)
        })
        .collect();

    let mut edges = Vec::new();

    // Create complete graphs within each hypercube vertex
    for v in 0..n_hypercube_vertices {
        let start = v * k;
        for i in 0..k {
            for j in (i+1)..k {
                edges.push((start + i, start + j));
            }
        }
    }

    // Connect hypercube neighbors (sparse inter-cluster)
    for v in 0..n_hypercube_vertices {
        for d in 0..dim {
            let neighbor = v ^ (1 << d); // Flip bit d
            if neighbor > v { // Avoid duplicate edges
                // Connect one node from each cluster (sparse connection)
                let from = v * k;
                let to = neighbor * k;
                edges.push((from, to));
            }
        }
    }

    // Build adjacency and representations
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
        dim: hd_dim,
        node_representations,
        node_identities,
        topology_type: TopologyType::Hypercube,
        edges,
    }
}

/// Arrange SymRes layers in a hypercube pattern
fn symres_hypercube_arrangement(hyper_dim: usize, k: usize, hd_dim: usize, seed: u64) -> ConsciousnessTopology {
    use symthaea::hdc::consciousness_topology_generators::TopologyType;

    let n_layers = 1 << hyper_dim; // 2^hyper_dim layers
    let n_nodes = n_layers * k;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| {
            let base = RealHV::basis(i, hd_dim);
            let noise = RealHV::random(hd_dim, seed + i as u64 * 1000).scale(0.05);
            base.add(&noise)
        })
        .collect();

    let mut edges = Vec::new();

    // Complete graphs within each layer
    for layer in 0..n_layers {
        let start = layer * k;
        for i in 0..k {
            for j in (i+1)..k {
                edges.push((start + i, start + j));
            }
        }
    }

    // Hypercube connections between layers (sparse: 1 connection per edge)
    for layer in 0..n_layers {
        for d in 0..hyper_dim {
            let neighbor_layer = layer ^ (1 << d);
            if neighbor_layer > layer {
                // Sparse connection: first node of each layer
                edges.push((layer * k, neighbor_layer * k));
            }
        }
    }

    // Skip connections (all-pairs between non-adjacent layers)
    for layer1 in 0..n_layers {
        for layer2 in (layer1 + 2)..n_layers {
            // Check if they differ by more than 1 bit (non-adjacent in hypercube)
            let diff = layer1 ^ layer2;
            if diff.count_ones() > 1 {
                edges.push((layer1 * k, layer2 * k));
            }
        }
    }

    // Build adjacency and representations
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
        dim: hd_dim,
        node_representations,
        node_identities,
        topology_type: TopologyType::Hypercube,
        edges,
    }
}

/// Hypercube with additional skip (diagonal) connections
fn hypercube_with_skips(dim: usize, skip_level: usize, hd_dim: usize, seed: u64) -> ConsciousnessTopology {
    use symthaea::hdc::consciousness_topology_generators::TopologyType;

    let n_nodes = 1 << dim;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| {
            let base = RealHV::basis(i, hd_dim);
            let noise = RealHV::random(hd_dim, seed + i as u64 * 1000).scale(0.05);
            base.add(&noise)
        })
        .collect();

    let mut edges = Vec::new();

    // Standard hypercube edges
    for v in 0..n_nodes {
        for d in 0..dim {
            let neighbor = v ^ (1 << d);
            if neighbor > v {
                edges.push((v, neighbor));
            }
        }
    }

    // Skip connections (nodes that differ by exactly skip_level+1 bits)
    for v1 in 0..n_nodes {
        for v2 in (v1+1)..n_nodes {
            let diff = (v1 ^ v2).count_ones() as usize;
            if diff == skip_level + 1 {
                edges.push((v1, v2));
            }
        }
    }

    // Build adjacency and representations
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
        dim: hd_dim,
        node_representations,
        node_identities,
        topology_type: TopologyType::Hypercube,
        edges,
    }
}

/// Recursive hypercube: hypercube of hypercubes
fn recursive_hypercube(outer_dim: usize, inner_dim: usize, hd_dim: usize, seed: u64) -> ConsciousnessTopology {
    use symthaea::hdc::consciousness_topology_generators::TopologyType;

    let n_outer = 1 << outer_dim;
    let n_inner = 1 << inner_dim;
    let n_nodes = n_outer * n_inner;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| {
            let base = RealHV::basis(i, hd_dim);
            let noise = RealHV::random(hd_dim, seed + i as u64 * 1000).scale(0.05);
            base.add(&noise)
        })
        .collect();

    let mut edges = Vec::new();

    // Inner hypercube edges within each outer vertex
    for outer in 0..n_outer {
        let start = outer * n_inner;
        for v in 0..n_inner {
            for d in 0..inner_dim {
                let neighbor = v ^ (1 << d);
                if neighbor > v {
                    edges.push((start + v, start + neighbor));
                }
            }
        }
    }

    // Outer hypercube edges (connecting corresponding nodes)
    for outer in 0..n_outer {
        for d in 0..outer_dim {
            let neighbor_outer = outer ^ (1 << d);
            if neighbor_outer > outer {
                // Connect corresponding nodes in inner hypercubes
                for inner in 0..n_inner {
                    edges.push((outer * n_inner + inner, neighbor_outer * n_inner + inner));
                }
            }
        }
    }

    // Build adjacency and representations
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
        dim: hd_dim,
        node_representations,
        node_identities,
        topology_type: TopologyType::Hypercube,
        edges,
    }
}

/// Ring of complete graphs
fn ring_of_complete_graphs(n_clusters: usize, k: usize, hd_dim: usize, seed: u64) -> ConsciousnessTopology {
    use symthaea::hdc::consciousness_topology_generators::TopologyType;

    let n_nodes = n_clusters * k;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| {
            let base = RealHV::basis(i, hd_dim);
            let noise = RealHV::random(hd_dim, seed + i as u64 * 1000).scale(0.05);
            base.add(&noise)
        })
        .collect();

    let mut edges = Vec::new();

    // Complete graphs within each cluster
    for c in 0..n_clusters {
        let start = c * k;
        for i in 0..k {
            for j in (i+1)..k {
                edges.push((start + i, start + j));
            }
        }
    }

    // Ring connections between clusters
    for c in 0..n_clusters {
        let next = (c + 1) % n_clusters;
        edges.push((c * k, next * k)); // Sparse: one connection per pair
    }

    // Build adjacency and representations
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
        dim: hd_dim,
        node_representations,
        node_identities,
        topology_type: TopologyType::Hypercube,
        edges,
    }
}

/// Torus of complete graphs (grid × grid clusters)
fn torus_of_complete_graphs(grid_size: usize, k: usize, hd_dim: usize, seed: u64) -> ConsciousnessTopology {
    use symthaea::hdc::consciousness_topology_generators::TopologyType;

    let n_clusters = grid_size * grid_size;
    let n_nodes = n_clusters * k;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| {
            let base = RealHV::basis(i, hd_dim);
            let noise = RealHV::random(hd_dim, seed + i as u64 * 1000).scale(0.05);
            base.add(&noise)
        })
        .collect();

    let mut edges = Vec::new();

    // Complete graphs within each cluster
    for c in 0..n_clusters {
        let start = c * k;
        for i in 0..k {
            for j in (i+1)..k {
                edges.push((start + i, start + j));
            }
        }
    }

    // Torus connections between clusters
    for row in 0..grid_size {
        for col in 0..grid_size {
            let c = row * grid_size + col;
            let right = row * grid_size + (col + 1) % grid_size;
            let down = ((row + 1) % grid_size) * grid_size + col;

            if right > c || col == grid_size - 1 {
                edges.push((c * k, right * k));
            }
            if down > c || row == grid_size - 1 {
                edges.push((c * k, down * k));
            }
        }
    }

    // Build adjacency and representations
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
        dim: hd_dim,
        node_representations,
        node_identities,
        topology_type: TopologyType::Hypercube,
        edges,
    }
}

/// Petersen graph of complete graphs (10 clusters)
fn petersen_of_complete_graphs(k: usize, hd_dim: usize, seed: u64) -> ConsciousnessTopology {
    use symthaea::hdc::consciousness_topology_generators::TopologyType;

    let n_clusters = 10;
    let n_nodes = n_clusters * k;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| {
            let base = RealHV::basis(i, hd_dim);
            let noise = RealHV::random(hd_dim, seed + i as u64 * 1000).scale(0.05);
            base.add(&noise)
        })
        .collect();

    let mut edges = Vec::new();

    // Complete graphs within each cluster
    for c in 0..n_clusters {
        let start = c * k;
        for i in 0..k {
            for j in (i+1)..k {
                edges.push((start + i, start + j));
            }
        }
    }

    // Petersen graph edges between clusters
    // Outer pentagon: 0-1-2-3-4-0
    let outer_edges = [(0,1), (1,2), (2,3), (3,4), (4,0)];
    // Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    let spoke_edges = [(0,5), (1,6), (2,7), (3,8), (4,9)];
    // Inner pentagram: 5-7-9-6-8-5
    let inner_edges = [(5,7), (7,9), (9,6), (6,8), (8,5)];

    for (a, b) in outer_edges.iter().chain(spoke_edges.iter()).chain(inner_edges.iter()) {
        edges.push((a * k, b * k));
    }

    // Build adjacency and representations
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
        dim: hd_dim,
        node_representations,
        node_identities,
        topology_type: TopologyType::Hypercube,
        edges,
    }
}

/// SymRes with feedback loops
fn symres_with_feedback(k: usize, n_layers: usize, hd_dim: usize, seed: u64) -> ConsciousnessTopology {
    use symthaea::hdc::consciousness_topology_generators::TopologyType;

    let n_nodes = n_layers * k;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| {
            let base = RealHV::basis(i, hd_dim);
            let noise = RealHV::random(hd_dim, seed + i as u64 * 1000).scale(0.05);
            base.add(&noise)
        })
        .collect();

    let mut edges = Vec::new();

    // Complete graphs within each layer
    for layer in 0..n_layers {
        let start = layer * k;
        for i in 0..k {
            for j in (i+1)..k {
                edges.push((start + i, start + j));
            }
        }
    }

    // Forward inter-layer (sparse)
    for layer in 0..(n_layers - 1) {
        edges.push((layer * k, (layer + 1) * k));
    }

    // Feedback loops (last to first, and middle to earlier)
    edges.push(((n_layers - 1) * k, 0)); // Last to first
    if n_layers > 3 {
        edges.push(((n_layers - 1) * k, (n_layers / 3) * k)); // Last to 1/3
        edges.push(((n_layers * 2 / 3) * k, (n_layers / 3) * k)); // 2/3 to 1/3
    }

    // Skip connections
    for l1 in 0..n_layers {
        for l2 in (l1 + 2)..n_layers {
            edges.push((l1 * k, l2 * k));
        }
    }

    // Build adjacency and representations
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
        dim: hd_dim,
        node_representations,
        node_identities,
        topology_type: TopologyType::Hypercube,
        edges,
    }
}

/// Hypercube with K3 decorations on each vertex
fn decorated_hypercube(dim: usize, decoration_k: usize, hd_dim: usize, seed: u64) -> ConsciousnessTopology {
    use symthaea::hdc::consciousness_topology_generators::TopologyType;

    let n_hypercube = 1 << dim;
    let n_decoration = decoration_k - 1; // Additional nodes per hypercube vertex
    let n_nodes = n_hypercube + n_hypercube * n_decoration;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| {
            let base = RealHV::basis(i, hd_dim);
            let noise = RealHV::random(hd_dim, seed + i as u64 * 1000).scale(0.05);
            base.add(&noise)
        })
        .collect();

    let mut edges = Vec::new();

    // Standard hypercube edges
    for v in 0..n_hypercube {
        for d in 0..dim {
            let neighbor = v ^ (1 << d);
            if neighbor > v {
                edges.push((v, neighbor));
            }
        }
    }

    // K_decoration_k decorations at each vertex
    for v in 0..n_hypercube {
        let decoration_start = n_hypercube + v * n_decoration;
        // Connect hypercube vertex to all decoration nodes
        for d in 0..n_decoration {
            edges.push((v, decoration_start + d));
        }
        // Connect decoration nodes to each other
        for i in 0..n_decoration {
            for j in (i+1)..n_decoration {
                edges.push((decoration_start + i, decoration_start + j));
            }
        }
    }

    // Build adjacency and representations
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
        dim: hd_dim,
        node_representations,
        node_identities,
        topology_type: TopologyType::Hypercube,
        edges,
    }
}
