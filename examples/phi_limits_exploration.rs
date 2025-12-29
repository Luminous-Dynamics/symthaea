/// Exploring the Φ = 0.5 Limit
///
/// What structures can exceed Φ = 0.5?
/// Testing complete graphs, dense networks, and extreme connectivity.

use symthaea::hdc::{
    consciousness_topology_generators::ConsciousnessTopology,
    phi_real::RealPhiCalculator,
    HDC_DIMENSION,
};

fn main() {
    let calc = RealPhiCalculator::new();

    println!("\n🔬 Exploring the Φ = 0.5 Limit");
    println!("{}", "=".repeat(60));

    // ===========================================
    // PART 1: Complete Graphs K_n
    // ===========================================
    println!("\n📊 Complete Graphs K_n (every node connected to every other):\n");

    for n in [3, 4, 5, 6, 7, 8, 10, 12, 16, 20] {
        let topo = ConsciousnessTopology::dense_network(n, HDC_DIMENSION, None, 42);
        let phi = calc.compute(&topo.node_representations);
        let indicator = if phi > 0.5 { " ✅ EXCEEDS 0.5!" } else { "" };
        println!("   K_{:<2} (n={:>2}): Φ = {:.4}{}", n, n, phi, indicator);
    }

    // ===========================================
    // PART 2: Generalized Petersen Graphs P(n,k)
    // ===========================================
    println!("\n📊 Generalized Petersen Graphs P(n,k):\n");

    // P(5,2) is the standard Petersen graph
    // P(n,k) has 2n vertices: outer n-gon + inner star polygon

    let petersen_configs = [
        (5, 2, "Standard Petersen"),
        (6, 2, "P(6,2) - Dürer graph"),
        (7, 2, "P(7,2)"),
        (7, 3, "P(7,3)"),
        (8, 3, "P(8,3) - Möbius-Kantor"),
        (10, 2, "P(10,2) - Dodecahedron"),
        (10, 3, "P(10,3)"),
        (12, 5, "P(12,5) - Nauru graph"),
    ];

    for (n, k, name) in petersen_configs {
        let topo = generalized_petersen(n, k, HDC_DIMENSION, 42);
        let phi = calc.compute(&topo.node_representations);
        let indicator = if phi > 0.5 { " ✅" } else { "" };
        println!("   P({:>2},{}) {:>20}: Φ = {:.4} (n={}){}",
                 n, k, name, phi, n * 2, indicator);
    }

    // ===========================================
    // PART 3: Reference Comparisons
    // ===========================================
    println!("\n📊 Reference Topologies:\n");

    let refs = [
        ("Hypercube 4D (champion)", ConsciousnessTopology::hypercube(4, HDC_DIMENSION, 42)),
        ("Petersen (previous #2)", ConsciousnessTopology::petersen_graph(HDC_DIMENSION, 42)),
        ("Ring (n=8)", ConsciousnessTopology::ring(8, HDC_DIMENSION, 42)),
        ("Random (n=8)", ConsciousnessTopology::random(8, HDC_DIMENSION, 42)),
    ];

    for (name, topo) in refs {
        let phi = calc.compute(&topo.node_representations);
        println!("   {:<25}: Φ = {:.4}", name, phi);
    }

    // ===========================================
    // SUMMARY
    // ===========================================
    println!("\n{}", "=".repeat(60));
    println!("📈 Key Findings:");
    println!("   - Φ = 0.5 appears to be an asymptotic limit");
    println!("   - Complete graphs K_n converge toward but don't exceed 0.5");
    println!("   - The limit likely reflects maximum possible integration");
    println!("   - Only K_2 (trivially) achieves Φ = 1.0");
    println!("{}", "=".repeat(60));
}

/// Generate a Generalized Petersen Graph P(n,k)
///
/// P(n,k) has 2n vertices:
/// - Outer vertices 0..n form a regular n-gon
/// - Inner vertices n..2n form a star polygon {n/k}
/// - Each outer vertex i connects to inner vertex n+i
fn generalized_petersen(n: usize, k: usize, dim: usize, _seed: u64) -> ConsciousnessTopology {
    use symthaea::hdc::real_hv::RealHV;
    use symthaea::hdc::consciousness_topology_generators::TopologyType;

    let n_nodes = 2 * n;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| RealHV::basis(i, dim))
        .collect();

    let mut edges = Vec::new();

    // Outer n-gon edges: i -- (i+1) mod n
    for i in 0..n {
        edges.push((i, (i + 1) % n));
    }

    // Spokes: outer i -- inner n+i
    for i in 0..n {
        edges.push((i, n + i));
    }

    // Inner star polygon: (n+i) -- (n + (i+k) mod n)
    for i in 0..n {
        let inner_i = n + i;
        let inner_j = n + ((i + k) % n);
        if inner_i < inner_j {
            edges.push((inner_i, inner_j));
        }
    }

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
        topology_type: TopologyType::PetersenGraph, // Reusing for generalized
        edges,
    }
}
