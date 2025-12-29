/// Mission: Beat Hypercube 4D (Φ = 0.4976)
///
/// Strategies:
/// 1. Higher-dimensional hypercubes (5D, 6D)
/// 2. K₃-centric designs (triangles exceeded 0.5!)
/// 3. Optimized symmetric residual with smaller complete graphs
/// 4. Maximally symmetric small structures

use symthaea::hdc::{
    consciousness_topology_generators::{ConsciousnessTopology, TopologyType},
    phi_real::RealPhiCalculator,
    real_hv::RealHV,
    HDC_DIMENSION,
};

fn main() {
    println!("\n🎯 MISSION: Beat Hypercube 4D (Φ = 0.4976)");
    println!("{}", "=".repeat(70));

    let calc = RealPhiCalculator::new();
    let n_samples = 10;
    let mut results: Vec<(String, f64, f64, usize)> = Vec::new();

    // ===========================================
    // BASELINE: The Target to Beat
    // ===========================================
    println!("\n🎯 TARGET TO BEAT:\n");

    test_topology("Hypercube 4D (TARGET)", &calc, &mut results, n_samples, |seed| {
        ConsciousnessTopology::hypercube(4, HDC_DIMENSION, seed)
    });

    // ===========================================
    // STRATEGY 1: Higher Dimensional Hypercubes
    // ===========================================
    println!("\n📐 Strategy 1: Higher Dimensions\n");

    test_topology("Hypercube 5D", &calc, &mut results, n_samples, |seed| {
        ConsciousnessTopology::hypercube(5, HDC_DIMENSION, seed)
    });

    test_topology("Hypercube 6D", &calc, &mut results, n_samples, |seed| {
        ConsciousnessTopology::hypercube(6, HDC_DIMENSION, seed)
    });

    // ===========================================
    // STRATEGY 2: K₃-Centric Designs
    // ===========================================
    println!("\n🔺 Strategy 2: K₃-Centric (triangles exceeded 0.5!)\n");

    // Pure K₃ for reference
    test_topology("K₃ (reference)", &calc, &mut results, n_samples, |_seed| {
        k_n(3, HDC_DIMENSION)
    });

    // Connected triangles
    test_topology("TriangleChain (4 K₃)", &calc, &mut results, n_samples, |_seed| {
        triangle_chain(4, HDC_DIMENSION)
    });

    test_topology("TriangleRing (6 K₃)", &calc, &mut results, n_samples, |_seed| {
        triangle_ring(6, HDC_DIMENSION)
    });

    // Tetrahedra (3D analog of triangles)
    test_topology("TetrahedralMesh", &calc, &mut results, n_samples, |_seed| {
        tetrahedral_mesh(HDC_DIMENSION)
    });

    // ===========================================
    // STRATEGY 3: Optimized Symmetric Residual
    // ===========================================
    println!("\n⚡ Strategy 3: Optimized Symmetric Residual\n");

    // Try smaller complete graphs (K₃ was best!)
    test_topology("SymRes K₃ layers", &calc, &mut results, n_samples, |_seed| {
        symmetric_residual_kn(3, 6, HDC_DIMENSION) // 6 layers of K₃
    });

    test_topology("SymRes K₄ layers", &calc, &mut results, n_samples, |_seed| {
        symmetric_residual_kn(4, 5, HDC_DIMENSION) // 5 layers of K₄
    });

    test_topology("SymRes K₅ layers", &calc, &mut results, n_samples, |_seed| {
        symmetric_residual_kn(5, 4, HDC_DIMENSION) // 4 layers of K₅
    });

    // More aggressive skip connections
    test_topology("SymRes K₄ + Dense Skip", &calc, &mut results, n_samples, |_seed| {
        symmetric_residual_dense_skip(4, 5, HDC_DIMENSION)
    });

    // ===========================================
    // STRATEGY 4: Maximally Symmetric Small Structures
    // ===========================================
    println!("\n🔮 Strategy 4: Maximally Symmetric\n");

    // Icosahedron (12 vertices, 30 edges, very symmetric)
    test_topology("Icosahedron", &calc, &mut results, n_samples, |_seed| {
        icosahedron(HDC_DIMENSION)
    });

    // Dodecahedron (20 vertices, very symmetric)
    test_topology("Dodecahedron", &calc, &mut results, n_samples, |_seed| {
        dodecahedron(HDC_DIMENSION)
    });

    // Cuboctahedron (12 vertices, Archimedean solid)
    test_topology("Cuboctahedron", &calc, &mut results, n_samples, |_seed| {
        cuboctahedron(HDC_DIMENSION)
    });

    // ===========================================
    // RESULTS
    // ===========================================
    println!("\n{}", "=".repeat(70));
    println!("🏆 RESULTS - Did We Beat Hypercube 4D?\n");

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let target_phi = results.iter()
        .find(|(name, _, _, _)| name.contains("TARGET"))
        .map(|(_, phi, _, _)| *phi)
        .unwrap_or(0.4976);

    println!("{:<4} {:<25} {:>10} {:>10} {:>6} {:>10}",
             "Rank", "Topology", "Mean Φ", "Std Dev", "Nodes", "vs Target");
    println!("{}", "-".repeat(70));

    for (rank, (name, mean, std, nodes)) in results.iter().enumerate() {
        let diff = ((*mean - target_phi) / target_phi) * 100.0;
        let status = if *mean > target_phi && !name.contains("TARGET") {
            "✅ BEATS!"
        } else if name.contains("TARGET") {
            "🎯 TARGET"
        } else {
            ""
        };

        let medal = match rank {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };

        println!("{}{:<3} {:<25} {:>10.4} {:>10.4} {:>6} {:>+9.2}% {}",
                 medal, rank + 1, name, mean, std, nodes, diff, status);
    }

    // Summary
    println!("\n{}", "=".repeat(70));

    let winners: Vec<_> = results.iter()
        .filter(|(name, phi, _, _)| *phi > target_phi && !name.contains("TARGET"))
        .collect();

    if winners.is_empty() {
        println!("❌ No structure beat Hypercube 4D this round.");
        println!("\n💡 Insight: Hypercube 4D is remarkably hard to beat!");
        println!("   The 4D tesseract sits at a sweet spot of symmetry and connectivity.");
    } else {
        println!("✅ VICTORY! {} structure(s) beat Hypercube 4D:\n", winners.len());
        for (name, phi, _, nodes) in winners {
            let improvement = ((phi - target_phi) / target_phi) * 100.0;
            println!("   🏆 {} (Φ = {:.4}, n={}) - {:.3}% improvement!",
                     name, phi, nodes, improvement);
        }
    }

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

    println!("   {:<25} Φ = {:.4} ± {:.4} (n={})", name, mean, std_dev, node_count);
    results.push((name.to_string(), mean, std_dev, node_count));
}

// ===========================================
// TOPOLOGY IMPLEMENTATIONS
// ===========================================

/// Complete graph K_n
fn k_n(n: usize, dim: usize) -> ConsciousnessTopology {
    let node_identities: Vec<RealHV> = (0..n)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            edges.push((i, j));
        }
    }

    build_topology(n, dim, node_identities, edges)
}

/// Chain of triangles sharing vertices
fn triangle_chain(n_triangles: usize, dim: usize) -> ConsciousnessTopology {
    // Each triangle shares one vertex with next
    // n triangles = 2n + 1 vertices
    let n_nodes = 2 * n_triangles + 1;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    let mut edges = Vec::new();

    for t in 0..n_triangles {
        let base = 2 * t;
        // Triangle vertices: base, base+1, base+2
        edges.push((base, base + 1));
        edges.push((base + 1, base + 2));
        edges.push((base, base + 2));
    }

    build_topology(n_nodes, dim, node_identities, edges)
}

/// Ring of triangles
fn triangle_ring(n_triangles: usize, dim: usize) -> ConsciousnessTopology {
    // Ring of n triangles, each sharing an edge with neighbors
    // n triangles in a ring = n vertices (one per triangle apex) + n vertices (shared base)
    let n_nodes = 2 * n_triangles;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    let mut edges = Vec::new();

    // Inner ring (apexes)
    for i in 0..n_triangles {
        edges.push((i, (i + 1) % n_triangles));
    }

    // Outer ring (bases)
    for i in 0..n_triangles {
        let outer = n_triangles + i;
        edges.push((outer, n_triangles + (i + 1) % n_triangles));
    }

    // Spokes (connecting inner to outer, forming triangles)
    for i in 0..n_triangles {
        let inner = i;
        let outer1 = n_triangles + i;
        let outer2 = n_triangles + (i + 1) % n_triangles;
        edges.push((inner, outer1));
        edges.push((inner, outer2));
    }

    build_topology(n_nodes, dim, node_identities, edges)
}

/// Tetrahedral mesh (connected tetrahedra)
fn tetrahedral_mesh(dim: usize) -> ConsciousnessTopology {
    // 2 tetrahedra sharing a face = 5 vertices
    let n_nodes = 5;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    let mut edges = Vec::new();

    // First tetrahedron: 0-1-2-3
    for i in 0..4 {
        for j in (i + 1)..4 {
            edges.push((i, j));
        }
    }

    // Second tetrahedron shares face 1-2-3, adds vertex 4
    edges.push((4, 1));
    edges.push((4, 2));
    edges.push((4, 3));

    build_topology(n_nodes, dim, node_identities, edges)
}

/// Symmetric residual with K_n layers
fn symmetric_residual_kn(k: usize, n_layers: usize, dim: usize) -> ConsciousnessTopology {
    let n_nodes = k * n_layers;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    let mut edges = Vec::new();

    // Each layer is K_k
    for layer in 0..n_layers {
        let offset = layer * k;
        for i in 0..k {
            for j in (i + 1)..k {
                edges.push((offset + i, offset + j));
            }
        }
    }

    // Adjacent layer connections
    for layer in 0..(n_layers - 1) {
        let lower = layer * k;
        let upper = (layer + 1) * k;
        for i in 0..k {
            edges.push((lower + i, upper + i));
            edges.push((lower + i, upper + (i + 1) % k));
        }
    }

    // Skip connections (every 2 layers)
    for layer in 0..(n_layers.saturating_sub(2)) {
        let lower = layer * k;
        let upper = (layer + 2) * k;
        for i in 0..k {
            edges.push((lower + i, upper + i));
        }
    }

    build_topology(n_nodes, dim, node_identities, edges)
}

/// Symmetric residual with dense skip connections
fn symmetric_residual_dense_skip(k: usize, n_layers: usize, dim: usize) -> ConsciousnessTopology {
    let n_nodes = k * n_layers;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    let mut edges = Vec::new();

    // Each layer is K_k
    for layer in 0..n_layers {
        let offset = layer * k;
        for i in 0..k {
            for j in (i + 1)..k {
                edges.push((offset + i, offset + j));
            }
        }
    }

    // ALL pairs of layers connected (dense skip)
    for layer1 in 0..n_layers {
        for layer2 in (layer1 + 1)..n_layers {
            let offset1 = layer1 * k;
            let offset2 = layer2 * k;
            for i in 0..k {
                edges.push((offset1 + i, offset2 + i));
            }
        }
    }

    build_topology(n_nodes, dim, node_identities, edges)
}

/// Icosahedron (12 vertices, 30 edges)
fn icosahedron(dim: usize) -> ConsciousnessTopology {
    let n_nodes = 12;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    // Icosahedron edges (standard embedding)
    let edges = vec![
        // Top cap
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
        // Top ring
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 1),
        // Middle connections
        (1, 6), (2, 6), (2, 7), (3, 7), (3, 8),
        (4, 8), (4, 9), (5, 9), (5, 10), (1, 10),
        // Bottom ring
        (6, 7), (7, 8), (8, 9), (9, 10), (10, 6),
        // Bottom cap
        (11, 6), (11, 7), (11, 8), (11, 9), (11, 10),
    ];

    build_topology(n_nodes, dim, node_identities, edges)
}

/// Dodecahedron (20 vertices, 30 edges)
fn dodecahedron(dim: usize) -> ConsciousnessTopology {
    let n_nodes = 20;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    // Dodecahedron edges (standard)
    let edges = vec![
        // Top pentagon
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        // Upper middle ring
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        // Lower middle ring
        (5, 10), (6, 11), (7, 12), (8, 13), (9, 14),
        (10, 11), (11, 12), (12, 13), (13, 14), (14, 10),
        // Bottom pentagon
        (10, 15), (11, 16), (12, 17), (13, 18), (14, 19),
        (15, 16), (16, 17), (17, 18), (18, 19), (19, 15),
    ];

    build_topology(n_nodes, dim, node_identities, edges)
}

/// Cuboctahedron (12 vertices, 24 edges)
fn cuboctahedron(dim: usize) -> ConsciousnessTopology {
    let n_nodes = 12;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    // Cuboctahedron: square and triangular faces
    let edges = vec![
        // Top square
        (0, 1), (1, 2), (2, 3), (3, 0),
        // Middle ring
        (0, 4), (1, 4), (1, 5), (2, 5),
        (2, 6), (3, 6), (3, 7), (0, 7),
        // Middle ring connections
        (4, 5), (5, 6), (6, 7), (7, 4),
        // Bottom square
        (4, 8), (5, 9), (6, 10), (7, 11),
        (8, 9), (9, 10), (10, 11), (11, 8),
    ];

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
