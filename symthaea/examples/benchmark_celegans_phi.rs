// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # C. elegans Connectome Φ Benchmark
//!
//! Validates Symthaea's Integrated Information (Φ) computation on the
//! C. elegans nervous system - the only fully mapped connectome of any organism.
//!
//! ## Significance
//! The C. elegans nematode has exactly 302 neurons with ~7,000 chemical synapses
//! and ~600 gap junctions. This provides a ground-truth biological network for
//! testing consciousness metrics at a scale between toy models and human brains.
//!
//! ## Method
//! 1. Parse the connectome edge list (neurons + synaptic connections)
//! 2. Encode each neuron as an HDC vector incorporating connectivity pattern
//! 3. Compute Φ for subcircuits of varying size
//! 4. Compare Φ values against known neuroscience:
//!    - Motor circuits should show moderate integration
//!    - Sensory circuits should show lower integration
//!    - Hub neurons (e.g., AVAL, AVAR) should increase Φ when included
//!    - Random subsets should show lower Φ than functional circuits
//!
//! ## Data Source
//! OpenWorm project: https://github.com/openworm/OpenWorm
//! White et al. (1986) "The structure of the nervous system of C. elegans"
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_celegans_phi --release
//! ```

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use symthaea::hdc::unified_hv::ContinuousHV;
use symthaea::phi_engine::PhiEngine;

const DATA_DIR: &str = "data/benchmarks/celegans";

/// Neuron connection
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Connection {
    source: String,
    target: String,
    weight: f64,
    conn_type: String, // "chemical" or "electrical"
}

/// Parsed connectome
struct Connectome {
    neurons: Vec<String>,
    connections: Vec<Connection>,
    adjacency: HashMap<String, Vec<(String, f64)>>,
}

impl Connectome {
    fn from_edgelist(path: &Path) -> Self {
        let file = File::open(path).unwrap_or_else(|_| panic!("Cannot open {:?}", path));
        let reader = BufReader::new(file);

        let mut connections = Vec::new();
        let mut neuron_set = HashSet::new();
        let mut adjacency: HashMap<String, Vec<(String, f64)>> = HashMap::new();

        for (i, line) in reader.lines().enumerate() {
            let line = line.unwrap();
            if i == 0 || line.trim().is_empty() {
                continue; // skip header
            }

            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() < 4 {
                continue;
            }

            let source = fields[0].trim().to_string();
            let target = fields[1].trim().to_string();
            let weight: f64 = fields[2].trim().parse().unwrap_or(1.0);
            let conn_type = fields[3].trim().to_string();

            neuron_set.insert(source.clone());
            neuron_set.insert(target.clone());

            adjacency
                .entry(source.clone())
                .or_default()
                .push((target.clone(), weight));

            connections.push(Connection {
                source,
                target,
                weight,
                conn_type,
            });
        }

        let mut neurons: Vec<String> = neuron_set.into_iter().collect();
        neurons.sort();

        println!("  Neurons: {}", neurons.len());
        println!("  Connections: {}", connections.len());

        // Connection type breakdown
        let chemical = connections
            .iter()
            .filter(|c| c.conn_type == "chemical")
            .count();
        let electrical = connections
            .iter()
            .filter(|c| c.conn_type.contains("electr") || c.conn_type.contains("gap"))
            .count();
        println!(
            "  Chemical synapses: {}, Gap junctions: {}, Other: {}",
            chemical,
            electrical,
            connections.len() - chemical - electrical
        );

        Self {
            neurons,
            connections,
            adjacency,
        }
    }

    /// Get degree (number of connections) for a neuron
    fn degree(&self, neuron: &str) -> usize {
        self.adjacency.get(neuron).map(|v| v.len()).unwrap_or(0)
    }

    /// Get top hub neurons by degree
    fn hub_neurons(&self, n: usize) -> Vec<(String, usize)> {
        let mut degrees: Vec<(String, usize)> = self
            .neurons
            .iter()
            .map(|n| (n.clone(), self.degree(n)))
            .collect();
        degrees.sort_by(|a, b| b.1.cmp(&a.1));
        degrees.into_iter().take(n).collect()
    }

    /// Encode a subset of neurons into HDC vectors based on connectivity
    fn encode_subset(&self, subset: &[String], dim: usize) -> Vec<ContinuousHV> {
        let neuron_to_idx: HashMap<&str, usize> = subset
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();

        // Create random base vectors for each neuron
        let base_hvs: Vec<ContinuousHV> = subset
            .iter()
            .enumerate()
            .map(|(i, _)| ContinuousHV::random(dim, 42 + i as u64))
            .collect();

        // Encode each neuron as: base + weighted sum of connected neurons' bases
        subset
            .iter()
            .enumerate()
            .map(|(i, neuron)| {
                let mut values = base_hvs[i].values.clone();

                // Add connectivity information
                if let Some(neighbors) = self.adjacency.get(neuron) {
                    for (target, weight) in neighbors {
                        if let Some(&target_idx) = neuron_to_idx.get(target.as_str()) {
                            let bound = base_hvs[i].bind(&base_hvs[target_idx]);
                            let w = (*weight as f32).min(10.0) / 10.0;
                            for (v, &b) in values.iter_mut().zip(bound.values.iter()) {
                                *v += w * b;
                            }
                        }
                    }
                }

                // Normalize
                let norm: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in &mut values {
                        *v /= norm;
                    }
                }

                ContinuousHV::from_vec(values)
            })
            .collect()
    }
}

/// Known functional circuits in C. elegans
fn known_circuits() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        // Locomotion command circuit (high integration expected)
        (
            "Locomotion Command",
            vec![
                "AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER",
            ],
        ),
        // Touch receptor circuit
        (
            "Touch Sensory",
            vec!["ALML", "ALMR", "AVM", "PLML", "PLMR", "PVM"],
        ),
        // Pharyngeal circuit (feeding)
        (
            "Pharyngeal",
            vec!["M1", "M2L", "M2R", "M3L", "M3R", "M4", "M5"],
        ),
        // Thermotaxis circuit
        (
            "Thermotaxis",
            vec!["AFDL", "AFDR", "AIYL", "AIYR", "AIZL", "AIZR"],
        ),
        // Hub interneurons (highest integration expected)
        (
            "Hub Interneurons",
            vec![
                "AVAL", "AVAR", "AVBL", "AVBR", "RIBL", "RIBR", "AIBL", "AIBR",
            ],
        ),
    ]
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     C. elegans Connectome Φ Benchmark                      ║");
    println!("║     Integrated Information on a Real Biological Network    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let data_path = Path::new(DATA_DIR);
    if !data_path.exists() {
        eprintln!("ERROR: C. elegans data not found at {}", DATA_DIR);
        return;
    }

    // Load connectome
    println!("Loading C. elegans connectome...");
    let edgelist_path = data_path.join("herm_full_edgelist.csv");
    let connectome = Connectome::from_edgelist(&edgelist_path);

    // Show hub neurons
    println!("\nTop 10 hub neurons by out-degree:");
    let hubs = connectome.hub_neurons(10);
    for (i, (name, degree)) in hubs.iter().enumerate() {
        println!("  {}. {} (degree: {})", i + 1, name, degree);
    }

    let dim = 4096; // HDC dimension for Φ computation
    let phi_engine = PhiEngine::auto();

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Computing Φ for known functional circuits");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let circuits = known_circuits();
    let mut circuit_results: Vec<(&str, f64, usize, f64)> = Vec::new();

    for (name, neurons) in &circuits {
        // Filter to neurons that exist in our dataset
        let available: Vec<String> = neurons
            .iter()
            .filter(|n| connectome.neurons.contains(&n.to_string()))
            .map(|n| n.to_string())
            .collect();

        if available.len() < 3 {
            println!(
                "  {}: Skipped (only {} neurons found in data)",
                name,
                available.len()
            );
            continue;
        }

        let t = Instant::now();
        let encoded = connectome.encode_subset(&available, dim);
        let result = phi_engine.compute(&encoded);
        let elapsed = t.elapsed().as_secs_f64();

        println!(
            "  {:20} │ neurons: {:2}/{:2} │ Φ = {:.6} │ method: {:15} │ {:.1}ms",
            name,
            available.len(),
            neurons.len(),
            result.phi,
            result.method,
            elapsed * 1000.0
        );

        circuit_results.push((name, result.phi, available.len(), elapsed));
    }

    // Random subsets for comparison
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Computing Φ for random neuron subsets (baseline)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut random_phis: Vec<f64> = Vec::new();

    for trial in 0..10 {
        // Pick random subset of 8 neurons
        let n = connectome.neurons.len();
        let seed = 100 + trial as u64;
        let mut indices: Vec<usize> = (0..n).collect();

        // Simple shuffle using seed
        let mut rng_state = seed;
        for i in (1..indices.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state as usize) % (i + 1);
            indices.swap(i, j);
        }

        let subset: Vec<String> = indices[..8]
            .iter()
            .map(|&i| connectome.neurons[i].clone())
            .collect();

        let encoded = connectome.encode_subset(&subset, dim);
        let result = phi_engine.compute(&encoded);
        random_phis.push(result.phi);

        println!(
            "  Random-{:02} │ Φ = {:.6} │ neurons: {:?}",
            trial + 1,
            result.phi,
            &subset[..3.min(subset.len())]
        );
    }

    let random_mean = random_phis.iter().sum::<f64>() / random_phis.len() as f64;
    let random_std = (random_phis
        .iter()
        .map(|x| (x - random_mean).powi(2))
        .sum::<f64>()
        / (random_phis.len() - 1) as f64)
        .sqrt();

    println!(
        "\n  Random baseline: Φ = {:.6} ± {:.6}",
        random_mean, random_std
    );

    // Scale analysis
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Φ scaling with subset size (using top hub neurons)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let hub_neurons: Vec<String> = connectome
        .hub_neurons(32)
        .into_iter()
        .map(|(name, _)| name)
        .collect();

    let sizes = vec![4, 8, 12, 16, 20, 24, 28, 32];
    let mut scale_results: Vec<(usize, f64, f64)> = Vec::new();

    for &size in &sizes {
        if size > hub_neurons.len() {
            break;
        }

        let subset: Vec<String> = hub_neurons[..size].to_vec();
        let t = Instant::now();
        let encoded = connectome.encode_subset(&subset, dim);
        let result = phi_engine.compute(&encoded);
        let elapsed = t.elapsed().as_secs_f64();

        println!(
            "  n={:2} │ Φ = {:.6} │ method: {:15} │ {:.1}ms",
            size,
            result.phi,
            result.method,
            elapsed * 1000.0
        );

        scale_results.push((size, result.phi, elapsed));
    }

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    RESULTS SUMMARY                         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    let functional_mean = circuit_results
        .iter()
        .map(|(_, phi, _, _)| *phi)
        .sum::<f64>()
        / circuit_results.len().max(1) as f64;

    println!(
        "║  Functional circuits Φ (mean): {:.6}                    ║",
        functional_mean
    );
    println!(
        "║  Random subsets Φ (mean):      {:.6} ± {:.6}          ║",
        random_mean, random_std
    );

    let functional_over_random = if random_mean > 0.0 {
        functional_mean / random_mean
    } else {
        f64::INFINITY
    };
    println!(
        "║  Functional/Random ratio:      {:.2}x                       ║",
        functional_over_random
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Validation
    println!("VALIDATION");
    println!("═══════════════════════════════════════════════════════════════");
    let phi_computed = !circuit_results.is_empty();
    let scale_monotonic = scale_results.windows(2).all(|w| w[1].1 >= w[0].1 * 0.5);
    println!(
        "  Φ computable on biological connectome:   {}",
        if phi_computed { "PASS" } else { "FAIL" }
    );
    println!(
        "  Φ scales reasonably with system size:    {}",
        if scale_monotonic { "PASS" } else { "FAIL" }
    );
    println!(
        "  Functional > Random (ratio > 1.0):       {} ({:.2}x)",
        if functional_over_random > 1.0 {
            "PASS"
        } else {
            "FAIL"
        },
        functional_over_random
    );

    // Save results
    let result_json = serde_json::json!({
        "benchmark": "C. elegans Connectome Phi",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "connectome": {
            "neurons": connectome.neurons.len(),
            "connections": connectome.connections.len(),
        },
        "circuit_results": circuit_results.iter().map(|(name, phi, n, t)| {
            serde_json::json!({
                "circuit": name,
                "phi": phi,
                "neurons_used": n,
                "time_s": t,
            })
        }).collect::<Vec<_>>(),
        "random_baseline": {
            "mean": random_mean,
            "std": random_std,
            "trials": random_phis.len(),
        },
        "scale_analysis": scale_results.iter().map(|(n, phi, t)| {
            serde_json::json!({ "n": n, "phi": phi, "time_s": t })
        }).collect::<Vec<_>>(),
    });

    let result_path = "data/benchmarks/celegans/results.json";
    if let Ok(f) = File::create(result_path) {
        serde_json::to_writer_pretty(f, &result_json).ok();
        println!("\nResults saved to {}", result_path);
    }
}