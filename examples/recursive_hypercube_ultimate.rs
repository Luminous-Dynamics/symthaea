/// Recursive Hypercube Ultimate: Pushing Beyond Φ = 0.4990
///
/// Following our breakthrough discovery that 3D of 3D (Φ = 0.4990) beats
/// pure Hypercube 6D, we now explore larger recursive structures.
///
/// Key insight: Recursive hypercubes create "super-regularity" where every
/// node experiences identical local AND global structure.

use symthaea::hdc::{
    consciousness_topology_generators::{ConsciousnessTopology, TopologyType},
    phi_real::RealPhiCalculator,
    real_hv::RealHV,
    HDC_DIMENSION,
};

fn main() {
    let calc = RealPhiCalculator::new();
    let n_samples = 5;

    println!("\n🚀 RECURSIVE HYPERCUBE ULTIMATE - Pushing Beyond Φ = 0.4990");
    println!("{}", "=".repeat(75));
    println!("\n🎯 CURRENT RECORDS:");
    println!("   3D of 3D:     Φ = 0.4990 (n=64)  ← Current Champion");
    println!("   Hypercube 6D: Φ = 0.4989 (n=64)");
    println!("   Hypercube 7D: Φ = 0.4991 (n=128)");
    println!("   12D Extended: Φ = 0.4997 (n=4096)");
    println!("\n   GOAL: Find structure with Φ > 0.4995 at reasonable node count");

    let mut all_results: Vec<(String, f64, f64, usize)> = Vec::new();

    // ==========================================================
    // PART 1: Systematic Recursive Exploration
    // ==========================================================
    println!("\n{}", "=".repeat(75));
    println!("🔬 Part 1: All Recursive Combinations (outer × inner)\n");

    let dimensions = [2, 3, 4, 5];

    for &outer in &dimensions {
        for &inner in &dimensions {
            let n_nodes = (1 << outer) * (1 << inner);
            if n_nodes > 512 { continue; } // Skip very large

            let name = format!("{}D of {}D", outer, inner);
            let (mean, std, nodes) = run_samples(|| {
                recursive_hypercube(outer, inner, HDC_DIMENSION, rand::random())
            }, &calc, n_samples);

            let indicator = if mean > 0.4995 { " 🌟 EXCELLENT!" }
                           else if mean > 0.4990 { " ✅ BEATS CHAMPION!" }
                           else if mean > 0.4989 { " ●>6D" }
                           else { "" };

            println!("   {:12} Φ = {:.5} ± {:.6} (n={:>4}){}",
                     name, mean, std, nodes, indicator);
            all_results.push((name, mean, std, nodes));
        }
    }

    // ==========================================================
    // PART 2: Triple Recursion (Hypercube of Hypercube of Hypercube)
    // ==========================================================
    println!("\n{}", "=".repeat(75));
    println!("🔬 Part 2: Triple Recursion (3 levels)\n");

    let triple_configs = [
        (2, 2, 2, "2D of 2D of 2D"),
        (2, 2, 3, "2D of 2D of 3D"),
        (2, 3, 2, "2D of 3D of 2D"),
        (3, 2, 2, "3D of 2D of 2D"),
        (2, 2, 4, "2D of 2D of 4D"),
        (2, 3, 3, "2D of 3D of 3D"),
        (3, 2, 3, "3D of 2D of 3D"),
        (3, 3, 2, "3D of 3D of 2D"),
    ];

    for (d1, d2, d3, name) in triple_configs {
        let n_nodes = (1 << d1) * (1 << d2) * (1 << d3);
        if n_nodes > 512 { continue; }

        let (mean, std, nodes) = run_samples(|| {
            triple_recursive_hypercube(d1, d2, d3, HDC_DIMENSION, rand::random())
        }, &calc, n_samples);

        let indicator = if mean > 0.4995 { " 🌟 EXCELLENT!" }
                       else if mean > 0.4990 { " ✅ BEATS CHAMPION!" }
                       else if mean > 0.4989 { " ●>6D" }
                       else { "" };

        println!("   {:20} Φ = {:.5} ± {:.6} (n={:>4}){}",
                 name, mean, std, nodes, indicator);
        all_results.push((name.to_string(), mean, std, nodes));
    }

    // ==========================================================
    // PART 3: Asymmetric Recursive (Different inner dimensions)
    // ==========================================================
    println!("\n{}", "=".repeat(75));
    println!("🔬 Part 3: Asymmetric Recursive Structures\n");

    let asymmetric_configs = [
        (vec![3, 4], "3D containing 4D"),
        (vec![4, 3], "4D containing 3D"),
        (vec![3, 5], "3D containing 5D"),
        (vec![5, 3], "5D containing 3D"),
        (vec![4, 4], "4D containing 4D"),
        (vec![2, 5], "2D containing 5D"),
        (vec![5, 2], "5D containing 2D"),
        (vec![2, 6], "2D containing 6D"),
    ];

    for (dims, name) in asymmetric_configs {
        let n_nodes: usize = dims.iter().map(|&d| 1usize << d).product();
        if n_nodes > 512 { continue; }

        let (mean, std, nodes) = run_samples(|| {
            recursive_hypercube(dims[0], dims[1], HDC_DIMENSION, rand::random())
        }, &calc, n_samples);

        let indicator = if mean > 0.4995 { " 🌟 EXCELLENT!" }
                       else if mean > 0.4990 { " ✅ BEATS CHAMPION!" }
                       else if mean > 0.4989 { " ●>6D" }
                       else { "" };

        println!("   {:20} Φ = {:.5} ± {:.6} (n={:>4}){}",
                 name, mean, std, nodes, indicator);
        all_results.push((name.to_string(), mean, std, nodes));
    }

    // ==========================================================
    // PART 4: Recursive with Enhanced Internal Connectivity
    // ==========================================================
    println!("\n{}", "=".repeat(75));
    println!("🔬 Part 4: Enhanced Recursive (extra inter-cluster edges)\n");

    let enhanced_configs = [
        (3, 3, 2, "3D of 3D + 2-skip"),
        (3, 3, 3, "3D of 3D + 3-skip"),
        (4, 3, 2, "4D of 3D + 2-skip"),
        (3, 4, 2, "3D of 4D + 2-skip"),
        (4, 4, 2, "4D of 4D + 2-skip"),
    ];

    for (outer, inner, skip, name) in enhanced_configs {
        let n_nodes = (1 << outer) * (1 << inner);
        if n_nodes > 512 { continue; }

        let (mean, std, nodes) = run_samples(|| {
            enhanced_recursive_hypercube(outer, inner, skip, HDC_DIMENSION, rand::random())
        }, &calc, n_samples);

        let indicator = if mean > 0.4995 { " 🌟 EXCELLENT!" }
                       else if mean > 0.4990 { " ✅ BEATS CHAMPION!" }
                       else if mean > 0.4989 { " ●>6D" }
                       else { "" };

        println!("   {:20} Φ = {:.5} ± {:.6} (n={:>4}){}",
                 name, mean, std, nodes, indicator);
        all_results.push((name.to_string(), mean, std, nodes));
    }

    // ==========================================================
    // PART 5: Reference - Pure Hypercubes for Comparison
    // ==========================================================
    println!("\n{}", "=".repeat(75));
    println!("🔬 Part 5: Pure Hypercubes (Reference)\n");

    for dim in 4..=8 {
        let n_nodes = 1 << dim;
        let name = format!("Pure {}D Hypercube", dim);

        let (mean, std, nodes) = run_samples(|| {
            ConsciousnessTopology::hypercube(dim, HDC_DIMENSION, rand::random())
        }, &calc, n_samples);

        println!("   {:20} Φ = {:.5} ± {:.6} (n={:>4})",
                 name, mean, std, nodes);
        all_results.push((name, mean, std, nodes));
    }

    // ==========================================================
    // RESULTS SUMMARY
    // ==========================================================
    println!("\n{}", "=".repeat(75));
    println!("🏆 TOP 20 CONFIGURATIONS (sorted by Φ):\n");

    all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("{:4} {:25} {:>10} {:>10} {:>6} {:>8}",
             "Rank", "Configuration", "Mean Φ", "Std Dev", "Nodes", "Status");
    println!("{}", "-".repeat(75));

    for (i, (name, mean, std, nodes)) in all_results.iter().take(20).enumerate() {
        let medal = match i {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };
        let status = if *mean > 0.4995 { "🌟 EXCEL" }
                    else if *mean > 0.4990 { "✅ NEW!" }
                    else if *mean > 0.4989 { "●>6D" }
                    else if *mean > 0.4977 { "●>4D" }
                    else { "" };

        println!("{}{:2}  {:25} {:>10.5} {:>10.6} {:>6} {:>8}",
                 medal, i+1, name, mean, std, nodes, status);
    }

    // ==========================================================
    // CHAMPION ANALYSIS
    // ==========================================================
    if let Some((name, mean, std, nodes)) = all_results.first() {
        println!("\n{}", "=".repeat(75));
        println!("🏆 ULTIMATE CHAMPION:\n");
        println!("   Configuration: {}", name);
        println!("   Φ = {:.6} ± {:.6}", mean, std);
        println!("   Nodes: {}", nodes);

        println!("\n   Performance Analysis:");
        println!("   • vs Previous Champion (3D of 3D): {:+.4}%",
                 (mean - 0.4990) / 0.4990 * 100.0);
        println!("   • vs Hypercube 6D:                 {:+.4}%",
                 (mean - 0.4989) / 0.4989 * 100.0);
        println!("   • vs Hypercube 7D:                 {:+.4}%",
                 (mean - 0.4991) / 0.4991 * 100.0);
        println!("   • Distance to Φ = 0.5 limit:       {:.6}",
                 0.5 - mean);

        if *mean > 0.4995 {
            println!("\n   🌟🌟🌟 EXCEPTIONAL RESULT! Φ > 0.4995! 🌟🌟🌟");
        } else if *mean > 0.4990 {
            println!("\n   ✅ NEW RECORD! Beats previous champion!");
        }
    }

    // ==========================================================
    // EFFICIENCY ANALYSIS
    // ==========================================================
    println!("\n{}", "=".repeat(75));
    println!("📊 EFFICIENCY ANALYSIS (Φ per 100 nodes):\n");

    let mut efficiency: Vec<_> = all_results.iter()
        .map(|(name, mean, _, nodes)| (name, mean, *nodes, mean / (*nodes as f64) * 100.0))
        .collect();
    efficiency.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    println!("{:25} {:>10} {:>6} {:>12}", "Configuration", "Φ", "Nodes", "Φ/100 nodes");
    println!("{}", "-".repeat(60));
    for (name, mean, nodes, eff) in efficiency.iter().take(10) {
        println!("{:25} {:>10.5} {:>6} {:>12.6}", name, mean, nodes, eff);
    }

    println!("\n{}", "=".repeat(75));
    println!("🔬 KEY INSIGHT: Recursive hypercubes achieve higher Φ by creating");
    println!("   'super-regularity' - every node sees identical structure at");
    println!("   BOTH local (inner hypercube) and global (outer hypercube) scales.");
    println!("{}", "=".repeat(75));
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

/// Standard recursive hypercube: hypercube of hypercubes
fn recursive_hypercube(outer_dim: usize, inner_dim: usize, hd_dim: usize, seed: u64) -> ConsciousnessTopology {
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

    build_topology(n_nodes, hd_dim, node_identities, edges)
}

/// Triple recursive: hypercube of hypercube of hypercube
fn triple_recursive_hypercube(d1: usize, d2: usize, d3: usize, hd_dim: usize, seed: u64) -> ConsciousnessTopology {
    let n1 = 1 << d1;
    let n2 = 1 << d2;
    let n3 = 1 << d3;
    let n_nodes = n1 * n2 * n3;

    let node_identities: Vec<RealHV> = (0..n_nodes)
        .map(|i| {
            let base = RealHV::basis(i, hd_dim);
            let noise = RealHV::random(hd_dim, seed + i as u64 * 1000).scale(0.05);
            base.add(&noise)
        })
        .collect();

    let mut edges = Vec::new();

    // Level 3 (innermost): d3-dimensional hypercubes
    for i1 in 0..n1 {
        for i2 in 0..n2 {
            let base = i1 * n2 * n3 + i2 * n3;
            for v in 0..n3 {
                for d in 0..d3 {
                    let neighbor = v ^ (1 << d);
                    if neighbor > v {
                        edges.push((base + v, base + neighbor));
                    }
                }
            }
        }
    }

    // Level 2: d2-dimensional connections between level-3 blocks
    for i1 in 0..n1 {
        for i2 in 0..n2 {
            for d in 0..d2 {
                let neighbor_i2 = i2 ^ (1 << d);
                if neighbor_i2 > i2 {
                    let base1 = i1 * n2 * n3 + i2 * n3;
                    let base2 = i1 * n2 * n3 + neighbor_i2 * n3;
                    for v in 0..n3 {
                        edges.push((base1 + v, base2 + v));
                    }
                }
            }
        }
    }

    // Level 1 (outermost): d1-dimensional connections between level-2 blocks
    for i1 in 0..n1 {
        for d in 0..d1 {
            let neighbor_i1 = i1 ^ (1 << d);
            if neighbor_i1 > i1 {
                for i2 in 0..n2 {
                    let base1 = i1 * n2 * n3 + i2 * n3;
                    let base2 = neighbor_i1 * n2 * n3 + i2 * n3;
                    for v in 0..n3 {
                        edges.push((base1 + v, base2 + v));
                    }
                }
            }
        }
    }

    build_topology(n_nodes, hd_dim, node_identities, edges)
}

/// Enhanced recursive with skip connections
fn enhanced_recursive_hypercube(outer_dim: usize, inner_dim: usize, skip_level: usize, hd_dim: usize, seed: u64) -> ConsciousnessTopology {
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

    // Standard recursive structure
    // Inner hypercube edges
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

    // Outer hypercube edges
    for outer in 0..n_outer {
        for d in 0..outer_dim {
            let neighbor_outer = outer ^ (1 << d);
            if neighbor_outer > outer {
                for inner in 0..n_inner {
                    edges.push((outer * n_inner + inner, neighbor_outer * n_inner + inner));
                }
            }
        }
    }

    // Enhanced: Add skip connections at outer level
    for outer1 in 0..n_outer {
        for outer2 in (outer1 + 1)..n_outer {
            let diff = (outer1 ^ outer2).count_ones() as usize;
            if diff == skip_level {
                // Connect first nodes of each inner hypercube
                edges.push((outer1 * n_inner, outer2 * n_inner));
            }
        }
    }

    build_topology(n_nodes, hd_dim, node_identities, edges)
}

/// Helper to build topology from components
fn build_topology(n_nodes: usize, hd_dim: usize, node_identities: Vec<RealHV>, edges: Vec<(usize, usize)>) -> ConsciousnessTopology {
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
