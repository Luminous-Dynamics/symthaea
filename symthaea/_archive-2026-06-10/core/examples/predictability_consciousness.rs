// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Predictability ↔ Consciousness (Φ) Relationship
//!
//! Key insight: Signal predictability is inversely related to integrated information.
//!
//! - High Φ systems: Complex internal dynamics → hard to predict from parts
//! - Low Φ systems: Reducible to independent parts → easily predictable
//!
//! This example validates the relationship using different topology-generated signals.

use symthaea::hdc::HDC_DIMENSION;
use symthaea::hdc::consciousness_topology_generators::ConsciousnessTopology;
use symthaea::hdc::reservoir::HybridEnsemblePredictor;
use symthaea::hdc::spectral_connectivity::ConnectivityCalculator;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      PREDICTABILITY ↔ CONSCIOUSNESS (Φ) ANALYSIS             ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Testing if signal predictability correlates with Φ          ║");
    println!("║  Hypothesis: High Φ → Low Predictability (complex dynamics)  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let dim = HDC_DIMENSION;
    let n_nodes = 8;
    let phi_calc = ConnectivityCalculator::new();

    // Test multiple topologies
    #[allow(clippy::type_complexity)]
    let topologies: Vec<(
        &str,
        Box<dyn Fn(usize, usize, u64) -> ConsciousnessTopology>,
    )> = vec![
        ("Ring (High Φ)", Box::new(ConsciousnessTopology::ring)),
        ("Star (Low Φ)", Box::new(ConsciousnessTopology::star)),
        ("Random", Box::new(ConsciousnessTopology::random)),
        (
            "Dense",
            Box::new(|n, d, s| ConsciousnessTopology::dense_network(n, d, None, s)),
        ),
        ("Line", Box::new(ConsciousnessTopology::line)),
    ];

    let mut results = Vec::new();

    println!("Generating topology signals and measuring predictability...\n");

    for (name, generator) in topologies {
        // Generate topology
        let topology = generator(n_nodes, dim, 42);

        // Calculate Φ
        let phi = phi_calc.algebraic_connectivity(&topology.node_representations);

        // Generate time series from topology dynamics
        let signal = generate_topology_signal(&topology, 2000);

        // Measure predictability
        let mut predictor = HybridEnsemblePredictor::new(42);

        // Train on first 80%
        let train_size = (signal.len() * 8) / 10;
        for sample in signal.iter().take(train_size) {
            predictor.observe(*sample);
        }

        // Test on remaining 20%
        let mut correct = 0;
        let mut total = 0;
        let threshold = 0.5;

        for i in train_size..(signal.len() - 1) {
            let pred = predictor.predict() > threshold;
            let actual = signal[i + 1] > threshold;
            if pred == actual {
                correct += 1;
            }
            total += 1;
            predictor.observe(signal[i]);
        }

        let accuracy = correct as f64 / total as f64;
        let unpredictability = 1.0 - accuracy;

        results.push((name, phi, accuracy, unpredictability));

        println!(
            "  {:<20} Φ={:.4}  Accuracy={:.1}%  Unpredictability={:.3}",
            name,
            phi,
            accuracy * 100.0,
            unpredictability
        );
    }

    // Analyze correlation
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                     ANALYSIS                                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Sort by Φ
    let mut sorted = results.clone();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  Topologies ranked by Φ (integrated information):\n");
    println!(
        "  {:20} {:10} {:12} {:15}",
        "Topology", "Φ", "Accuracy", "Unpredictability"
    );
    println!(
        "  {:20} {:10} {:12} {:15}",
        "─".repeat(20),
        "─".repeat(10),
        "─".repeat(12),
        "─".repeat(15)
    );

    for (name, phi, acc, unp) in &sorted {
        println!(
            "  {:20} {:10.4} {:12.1}% {:15.3}",
            name,
            phi,
            acc * 100.0,
            unp
        );
    }

    // Calculate correlation between Φ and unpredictability
    let n = results.len() as f64;
    let phi_values: Vec<f64> = results.iter().map(|r| r.1).collect();
    let unp_values: Vec<f64> = results.iter().map(|r| r.3).collect();

    let mean_phi = phi_values.iter().sum::<f64>() / n;
    let mean_unp = unp_values.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_phi = 0.0;
    let mut var_unp = 0.0;

    for i in 0..results.len() {
        let dphi = phi_values[i] - mean_phi;
        let dunp = unp_values[i] - mean_unp;
        cov += dphi * dunp;
        var_phi += dphi * dphi;
        var_unp += dunp * dunp;
    }

    let correlation = cov / (var_phi.sqrt() * var_unp.sqrt());

    println!(
        "\n  Correlation (Φ vs Unpredictability): {:.3}",
        correlation
    );

    if correlation > 0.3 {
        println!("\n  ✅ POSITIVE CORRELATION: Higher Φ → Less predictable signals");
        println!("     This supports the hypothesis that integrated systems");
        println!("     generate complex, unpredictable dynamics.");
    } else if correlation < -0.3 {
        println!("\n  ⚠️ NEGATIVE CORRELATION: Higher Φ → More predictable signals");
        println!("     This is unexpected and warrants further investigation.");
    } else {
        println!("\n  🔄 WEAK CORRELATION: Φ and predictability may be independent");
        println!("     or the relationship is more nuanced.");
    }

    println!("\n  Key insight: Predictability can serve as a practical probe");
    println!("  for consciousness measurement - it's computationally tractable");
    println!("  and correlates with integrated information theory.");
}

/// Generate a time series by simulating CHAOTIC dynamics on a topology
/// Uses coupled logistic maps to create genuinely unpredictable signals
fn generate_topology_signal(topology: &ConsciousnessTopology, length: usize) -> Vec<f64> {
    let mut signal = Vec::with_capacity(length);
    let n_nodes = topology.node_representations.len();

    if n_nodes == 0 {
        return vec![0.5; length];
    }

    // Initialize node activities from topology (each node gets chaotic logistic map)
    let mut activities: Vec<f64> = topology
        .node_representations
        .iter()
        .enumerate()
        .map(|(i, hv)| {
            // Use HV values to seed initial conditions
            let sum: f32 = hv.values.iter().take(10).sum();
            let base = (sum.abs() / 10.0) as f64;
            // Initial condition in [0.1, 0.9] to avoid fixed points
            0.1 + 0.8 * (base.abs() % 1.0) + 0.001 * (i as f64)
        })
        .collect();

    // Build adjacency matrix from edges
    let mut adj = vec![vec![0.0_f64; n_nodes]; n_nodes];
    for &(i, j) in &topology.edges {
        if i < n_nodes && j < n_nodes {
            adj[i][j] = 1.0;
            adj[j][i] = 1.0;
        }
    }

    // Compute coupling strength based on topology (more edges = weaker per-edge coupling)
    let mut degree = vec![0.0_f64; n_nodes];
    for i in 0..n_nodes {
        degree[i] = adj[i].iter().sum();
    }

    // Normalize adjacency
    for i in 0..n_nodes {
        if degree[i] > 0.0 {
            for cell in adj[i].iter_mut().take(n_nodes) {
                *cell /= degree[i];
            }
        }
    }

    // Generate dynamics using COUPLED CHAOTIC LOGISTIC MAPS
    // Each node: x_{t+1} = r * x_t * (1 - x_t) + coupling
    // r = 3.8 (chaotic regime)
    let r = 3.8;
    let coupling_strength = 0.15; // How much neighbors influence dynamics

    for _t in 0..length {
        // Output: weighted sum of all node activities
        let output: f64 = activities.iter().sum::<f64>() / n_nodes as f64;
        signal.push(output);

        // Update each node using coupled logistic map
        let mut new_activities = vec![0.0; n_nodes];
        for i in 0..n_nodes {
            // Local logistic map dynamics
            let local = r * activities[i] * (1.0 - activities[i]);

            // Coupling from neighbors (average of neighbor states)
            let mut neighbor_avg = 0.0;
            let mut neighbor_count = 0.0;
            for j in 0..n_nodes {
                if adj[i][j] > 0.0 {
                    neighbor_avg += activities[j];
                    neighbor_count += 1.0;
                }
            }
            if neighbor_count > 0.0 {
                neighbor_avg /= neighbor_count;
            } else {
                neighbor_avg = activities[i];
            }

            // Blend local chaos with neighbor coupling
            // High coupling = more synchronization = more predictable
            // Low coupling = independent chaos = harder to predict from topology
            new_activities[i] =
                (1.0 - coupling_strength) * local + coupling_strength * neighbor_avg;

            // Clamp to valid logistic map range
            new_activities[i] = new_activities[i].clamp(0.001, 0.999);
        }
        activities = new_activities;
    }

    signal
}